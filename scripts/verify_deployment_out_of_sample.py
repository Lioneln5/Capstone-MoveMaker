"""Genuine out-of-sample verification of the deployment scoring pipeline.

Every other check built this session (scripts/verify_deployment_models.py)
was in-sample: every event tested was part of deployment training
(signed_year <= 2023). This script is different in kind, not just degree --
it scores the 2024-2025 extension cohort that arrived via the Codex merge,
which DEPLOYMENT_CUTOFF_SIGNED_YEAR = 2023 has excluded from training since
before that cohort existed. These rows the model has genuinely never seen.

Eligibility per phase reuses each diagnostic's own already-validated cohort
filter (imported, not re-derived) plus an explicit signed_year > 2023 cutoff
plus (for Phase 2) a per-row per-horizon observability flag, so a row only
counts toward a metric if its outcome has actually had time to happen and be
recorded -- a June-2025 signing does not get scored as a "no 24-month
outbound event" just because none has been recorded by August 2026, when
the horizon itself is not yet observable_by that date.

Every event is scored exactly once through the live retriever + scorer (not
by reading precomputed features from extension_modeling_master.csv), so
this is also the largest live-pipeline run yet -- it re-exercises the
retriever's real reconstruction, not the training-time batch computation,
on data outside anything build_contract_extension_integration.py's own
parity check ever covered.

Honesty rule for this script: report exactly what comes out. Do not tune,
retry with different splits, or cherry-pick a rosier cohort if the numbers
disappoint. A worse-than-training result here is itself the information the
project needs before calling this deployment-ready.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from deployment_scorer import DeploymentScorer  # noqa: E402
import run_extension_opportunity_diagnostic as phase1  # noqa: E402
import run_extension_survival_diagnostic as phase2  # noqa: E402
import run_extension_value_preservation_diagnostic as phase3  # noqa: E402
import run_contract_financial_exposure_benchmark as phase4  # noqa: E402

OUTPUT = ROOT / "Data" / "processed" / "deployment_models"
DEPLOYMENT_CUTOFF_SIGNED_YEAR = 2023

# What each already-published diagnostic reported as its own genuine
# out-of-sample number (pooled across 4 rolling origins, 2020-2023). This
# holdout is a single, smaller, more recent cohort -- it is not expected to
# match exactly, and this script does not require it to beat these numbers,
# only to land in a plausible neighborhood. A published citation, not a
# pass/fail bar this script invented.
PUBLISHED_REFERENCE = {
    "sustained_contribution_probability_24m": ("AUC", 0.788),
    "year2_opportunity_share": ("Spearman", 0.512),
    "any_outbound_24m": ("AUC", 0.710),
    "any_outbound_36m": ("AUC", 0.699),
    "permanent_outbound_24m": ("AUC", 0.696),
    "downside_25pct_12m": ("AUC", 0.781),
    "downside_25pct_24m": ("AUC", 0.821),
    "downside_50pct_24m": ("AUC", 0.805),
    "value_log_ratio_12m": ("Spearman", 0.607),
    "value_log_ratio_24m": ("Spearman", 0.663),
    "annual_wage": ("Spearman", 0.889),
    "contract_duration": ("Spearman", 0.659),
    "fixed_wage_commitment": ("Spearman", 0.832),
    "wage_to_market_value": ("Spearman", 0.835),
}


def add(checks: list[dict], name: str, observed: object, expected: object, passed: bool, notes: str) -> None:
    checks.append({"check": name, "observed": observed, "expected": expected, "passed": bool(passed), "notes": notes})


def build_eligibility() -> dict[str, pd.DataFrame]:
    """Returns, per endpoint, the held-out (signed_year>2023) rows eligible
    for that endpoint's specific metric -- each frame keyed by
    capology_extension_event_id with an 'actual' column attached."""
    eligible: dict[str, pd.DataFrame] = {}

    f1 = phase1.load_frame()
    held1 = f1.loc[f1["signed_year"].gt(DEPLOYMENT_CUTOFF_SIGNED_YEAR) & f1["cohort_primary_y2"]]
    eligible["year2_opportunity_share"] = held1[["capology_extension_event_id", "canonical_player_id", "canonical_club_id_x", "competition_id", "signed_date", "annual_gross_eur", "exact_duration_years", "at_signing_market_value_eur", "player_name_normalized_x", "target_year2_opportunity_share"]].rename(columns={"target_year2_opportunity_share": "actual"})
    eligible["sustained_contribution_probability_24m"] = held1[["capology_extension_event_id", "canonical_player_id", "canonical_club_id_x", "competition_id", "signed_date", "annual_gross_eur", "exact_duration_years", "at_signing_market_value_eur", "player_name_normalized_x", "target_sustained_meaningful_contribution"]].rename(columns={"target_sustained_meaningful_contribution": "actual"})

    f2 = phase2.load_frame()
    held2 = f2.loc[f2["signed_year"].gt(DEPLOYMENT_CUTOFF_SIGNED_YEAR) & f2["survival_primary_cohort"]]
    for endpoint, horizon in [("any_outbound", "24m"), ("any_outbound", "36m"), ("permanent_outbound", "24m")]:
        observable = held2[f"{endpoint}_horizon_observable_{horizon}"].astype(bool)
        subset = held2.loc[observable]
        key = f"{endpoint}_{horizon}"
        eligible[key] = subset[["capology_extension_event_id", "canonical_player_id", "canonical_club_id_x", "competition_id", "signed_date", "annual_gross_eur", "exact_duration_years", "at_signing_market_value_eur", "player_name_normalized_x", f"{endpoint}_event_by_{horizon}"]].rename(columns={f"{endpoint}_event_by_{horizon}": "actual"})

    f3 = phase3.load_frame()
    f3 = f3.assign(signed_year=pd.to_datetime(f3["signed_date"], errors="coerce").dt.year)
    for endpoint in ["value_log_ratio_12m", "value_log_ratio_24m", "downside_25pct_12m", "downside_25pct_24m", "downside_50pct_24m"]:
        spec = phase3.ENDPOINTS[endpoint]
        data = phase3.endpoint_frame(f3, endpoint)
        held = data.loc[data["signed_year"].gt(DEPLOYMENT_CUTOFF_SIGNED_YEAR)]
        eligible[endpoint] = held[["capology_extension_event_id", "canonical_player_id", "canonical_club_id_x", "competition_id", "signed_date", "annual_gross_eur", "exact_duration_years", "at_signing_market_value_eur", "player_name_normalized_x", spec["target"]]].rename(columns={spec["target"]: "actual"})

    f4 = phase4.load_frame()
    for endpoint in phase4.SELECTED_MODELS:
        spec = phase4.ENDPOINTS[endpoint]
        data = f4.loc[f4[spec["cohort"]] & f4["signed_year"].gt(DEPLOYMENT_CUTOFF_SIGNED_YEAR)]
        eligible[endpoint] = data[["capology_extension_event_id", "canonical_player_id", "canonical_club_id_x", "competition_id", "signed_date", "annual_gross_eur", "exact_duration_years", "at_signing_market_value_eur", "player_name_normalized_x", spec["target"]]].rename(columns={spec["target"]: "actual"})

    return eligible


def score_union(eligible: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    """Score every unique held-out event exactly once; return {event_id: scored_output}."""
    all_rows = pd.concat(eligible.values(), ignore_index=True).drop_duplicates("capology_extension_event_id")
    scorer = DeploymentScorer()
    results: dict[str, dict[str, Any]] = {}
    for i, row in enumerate(all_rows.itertuples(index=False), start=1):
        if i % 50 == 0:
            print(f"  scored {i}/{len(all_rows)}...", flush=True)
        out = scorer.score_extension(
            canonical_player_id=int(row.canonical_player_id), canonical_club_id=int(row.canonical_club_id_x),
            competition_id=row.competition_id, decision_date=row.signed_date,
            proposed_annual_fixed_wage_eur=float(row.annual_gross_eur) if pd.notna(row.annual_gross_eur) else 1.0,
            proposed_contract_years=float(row.exact_duration_years) if pd.notna(row.exact_duration_years) else 1.0,
            current_public_market_value_eur=float(row.at_signing_market_value_eur) if pd.notna(row.at_signing_market_value_eur) else None,
            player_name_normalized=row.player_name_normalized_x,
        )
        results[row.capology_extension_event_id] = out
    return results


MODULE_LOOKUP = {
    "year2_opportunity_share": ("opportunity", "year2_opportunity_reference_range"),
    "sustained_contribution_probability_24m": ("opportunity", "sustained_contribution_probability_24m"),
    "any_outbound_24m": ("continuity", "continuous_stay_probability_24m"),  # predicted STAY; invert for event prob
    "any_outbound_36m": ("continuity", "continuous_stay_probability_36m"),
    "permanent_outbound_24m": ("continuity", "permanent_relationship_probability_24m"),
    "value_log_ratio_12m": ("asset_value", "value_multiplier_reference_range_12m"),
    "value_log_ratio_24m": ("asset_value", "value_multiplier_reference_range_24m"),
    "downside_25pct_12m": ("asset_value", "value_downside_25pct_probability_12m"),
    "downside_25pct_24m": ("asset_value", "value_downside_25pct_probability_24m"),
    "downside_50pct_24m": ("asset_value", "value_downside_50pct_probability_24m"),
    "annual_wage": ("peer_context", "annual_wage_peer"),
    "contract_duration": ("peer_context", "contract_duration_peer"),
    "fixed_wage_commitment": ("peer_context", "fixed_wage_commitment_peer"),
    "wage_to_market_value": ("peer_context", "wage_to_market_value_peer"),
}
INVERT_TO_EVENT_PROBABILITY = {"any_outbound_24m", "any_outbound_36m", "permanent_outbound_24m"}


def evaluate(checks: list[dict], eligible: dict[str, pd.DataFrame], scored: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for endpoint, frame in eligible.items():
        card_name, metric_key = MODULE_LOOKUP[endpoint]
        rows = []
        for _, row in frame.iterrows():
            out = scored.get(row["capology_extension_event_id"])
            if out is None or out["status"] != "scored":
                continue
            entry = out["modules"][card_name][metric_key]
            value = entry.get("value") if entry.get("status", "").startswith("supported") else None
            if value is None or pd.isna(row["actual"]):
                continue
            pred = 1.0 - value if endpoint in INVERT_TO_EVENT_PROBABILITY else value
            rows.append({"actual": row["actual"], "pred": pred})
        n = len(rows)
        if n < 15:
            add(checks, f"oos_coverage_{endpoint}", n, ">=15", False, "Too few genuinely mature, scoreable held-out rows to report a metric honestly; reporting coverage only.")
            summary[endpoint] = {"n": n, "metric": None, "value": None}
            continue
        data = pd.DataFrame(rows)
        metric_name, published = PUBLISHED_REFERENCE[endpoint]
        if metric_name == "AUC":
            if data["actual"].nunique() < 2:
                add(checks, f"oos_coverage_{endpoint}", n, ">=15", True, f"n={n} but the held-out cohort has only one observed outcome class; AUC is undefined, reporting Brier only.")
                observed_value = float(brier_score_loss(data["actual"], data["pred"]))
                summary[endpoint] = {"n": n, "metric": "Brier", "value": observed_value, "published_auc_reference": published}
                continue
            observed_value = float(roc_auc_score(data["actual"], data["pred"]))
        else:
            observed_value = float(data["actual"].corr(data["pred"], method="spearman"))
        summary[endpoint] = {"n": n, "metric": metric_name, "value": observed_value, "published_reference": published}
        # Plausibility, not a beat-the-benchmark bar: a single small out-of-time
        # cohort can reasonably swing +/-0.15 around a 4-origin pooled figure.
        # Only fail if the sign of the signal reverses (near/below chance).
        floor = 0.55 if metric_name == "AUC" else 0.20
        add(checks, f"oos_signal_{endpoint}", round(observed_value, 4), f">={floor} (published pooled reference: {published})", observed_value >= floor,
            f"n={n} genuinely unseen held-out extensions (signed_year>2023).")
    return summary


def main() -> None:
    checks: list[dict] = []
    print("Building per-endpoint eligibility from the held-out (signed_year>2023) cohort...")
    eligible = build_eligibility()
    for endpoint, frame in eligible.items():
        add(checks, f"oos_eligible_rows_{endpoint}", len(frame), ">0", len(frame) > 0, "Rows with signed_year>2023, the diagnostic's own cohort filter, and (Phase 2) a matured horizon.")

    total_unique = pd.concat(eligible.values(), ignore_index=True)["capology_extension_event_id"].nunique()
    print(f"Scoring {total_unique} unique held-out events through the live retriever + scorer (this will take a while -- unindexed table scans)...")
    scored = score_union(eligible)
    scored_ok = sum(1 for v in scored.values() if v["status"] == "scored")
    refused = sum(1 for v in scored.values() if v["status"] == "refused")
    add(checks, "oos_scoring_completed_without_exception", len(scored), total_unique, len(scored) == total_unique, f"scored={scored_ok} refused={refused} of {total_unique}.")

    print("Evaluating predictions against real (matured) outcomes...")
    summary = evaluate(checks, eligible, scored)

    result = pd.DataFrame(checks)
    result.to_csv(OUTPUT / "out_of_sample_verification.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    payload = {
        "checks_passed": int(result["passed"].sum()),
        "checks_total": len(result),
        "all_checks_passed": bool(result["passed"].all()),
        "held_out_definition": f"signed_year > {DEPLOYMENT_CUTOFF_SIGNED_YEAR}, never part of deployment training",
        "total_unique_held_out_events_scored": int(total_unique),
        "endpoint_results": summary,
    }
    (OUTPUT / "out_of_sample_verification.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

    print(result.to_string(index=False))
    print(f"\n{payload['checks_passed']}/{payload['checks_total']} checks passed.")
    print("\nPer-endpoint out-of-sample results (n, metric, observed vs. published pooled reference):")
    for endpoint, s in summary.items():
        print(f"  {endpoint}: n={s['n']} {s.get('metric')}={s.get('value')} (published: {s.get('published_reference') or s.get('published_auc_reference')})")


if __name__ == "__main__":
    main()

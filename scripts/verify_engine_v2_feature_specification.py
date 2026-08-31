"""Independent verification for Engine V2 Phase 2."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.engine_v2.feature_contract import (  # noqa: E402
    ADVANCED_PERFORMANCE,
    COMMERCIAL_SCENARIO,
    HEADLINE_ENDPOINTS,
    RAW_SCORING_RATES,
)

OUTPUT = ROOT / "Data" / "processed" / "engine_v2_feature_specification"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    results: list[dict] = []

    def add(name: str, passed: bool, observed: object, expected: object, notes: str) -> None:
        results.append({
            "check": name, "passed": bool(passed), "observed": observed,
            "expected": expected, "notes": notes,
        })

    contract = pd.read_csv(OUTPUT / "endpoint_contract.csv")
    features = pd.read_csv(OUTPUT / "candidate_feature_manifest.csv")
    inventory = pd.read_csv(OUTPUT / "v1_feature_policy_audit.csv")
    metrics = pd.read_csv(OUTPUT / "origin_metrics.csv")
    predictions = pd.read_csv(OUTPUT / "out_of_time_predictions.csv")
    comparisons = pd.read_csv(OUTPUT / "candidate_comparisons.csv")
    subgroup = pd.read_csv(OUTPUT / "subgroup_audit.csv")
    decisions = pd.read_csv(OUTPUT / "phase2_decisions.csv")
    sources = pd.read_csv(OUTPUT / "source_manifest.csv")
    summary = json.loads((OUTPUT / "run_summary.json").read_text(encoding="utf-8"))

    add("four_modules_exact", set(contract["module"]) == {item.module for item in HEADLINE_ENDPOINTS}, sorted(contract["module"]), "four declared modules", "No combined score.")
    add("one_contract_per_module", contract["module"].is_unique and len(contract) == 4, len(contract), 4, "Endpoint topology is unambiguous.")
    add("three_predictive_endpoint_decisions", set(decisions["endpoint"]) == {"sustained_meaningful_contribution", "any_outbound_24m", "downside_25pct_24m"}, sorted(decisions["endpoint"]), "three predictive endpoints", "Wage benchmark remains a later validation task.")
    add("all_candidates_not_deployed", set(decisions["deployment_status"]) == {"not_deployed"}, sorted(decisions["deployment_status"].unique()), ["not_deployed"], "Phase 2 cannot promote artifacts.")
    add("all_candidates_advance_only_to_phase3", set(decisions["phase2_decision"]) == {"advance_to_phase3_candidate"}, sorted(decisions["phase2_decision"].unique()), ["advance_to_phase3_candidate"], "Advance means more testing, not deployment.")

    safe = features[features["variant"].isin(["V2_core_involvement", "V2_plus_prior_volume"])]
    safe_features = set(safe["feature"])
    add("no_raw_scoring_in_v2", not safe_features.intersection(RAW_SCORING_RATES), sorted(safe_features.intersection(RAW_SCORING_RATES)), [], "Goals/assists rates are removed.")
    add("no_advanced_stats_in_v2", not safe_features.intersection(ADVANCED_PERFORMANCE), sorted(safe_features.intersection(ADVANCED_PERFORMANCE)), [], "Unvalidated event block is removed.")
    add("no_commercial_scenario_in_v2_predictions", not safe_features.intersection(COMMERCIAL_SCENARIO), sorted(safe_features.intersection(COMMERCIAL_SCENARIO)), [], "User proposal cannot change sporting/value risk.")
    add("historical_reference_never_selected", not decisions["selected_phase2_candidate"].eq("historical_M5_reference").any(), decisions["selected_phase2_candidate"].tolist(), "only contract-valid candidates", "Accuracy cannot override feature semantics.")

    key = ["endpoint", "origin_id", "variant", "capology_extension_event_id"]
    add("prediction_keys_unique", not predictions.duplicated(key).any(), int(predictions.duplicated(key).sum()), 0, "One prediction per event/endpoint/origin/variant.")
    add("rolling_years_exact", set(predictions["evaluation_year"]) == {2020, 2021, 2022, 2023}, sorted(predictions["evaluation_year"].unique()), [2020, 2021, 2022, 2023], "Chronological origins only.")
    add("prediction_probabilities_bounded", predictions["prediction"].between(0, 1).all(), [float(predictions["prediction"].min()), float(predictions["prediction"].max())], "[0,1]", "Valid probability outputs.")
    add("prediction_actuals_binary", set(predictions["actual"]) == {0, 1}, sorted(predictions["actual"].unique()), [0, 1], "All selected endpoints are binary.")

    recomputed = predictions.groupby(["endpoint", "origin_id", "variant"], as_index=False)["brier_loss"].mean().rename(columns={"brier_loss": "recomputed"})
    merged = metrics.merge(recomputed, on=["endpoint", "origin_id", "variant"], validate="one_to_one")
    max_metric_diff = float((merged["brier"] - merged["recomputed"]).abs().max())
    add("origin_brier_recomputes", max_metric_diff < 1e-12, max_metric_diff, "<1e-12", "Metrics derive from saved predictions.")

    comparison_diffs = []
    for row in comparisons.itertuples(index=False):
        data = predictions.loc[predictions["endpoint"].eq(row.endpoint)]
        left = data.loc[data["variant"].eq(row.baseline), ["origin_id", "capology_extension_event_id", "brier_loss"]].rename(columns={"brier_loss": "baseline"})
        right = data.loc[data["variant"].eq(row.candidate), ["origin_id", "capology_extension_event_id", "brier_loss"]].rename(columns={"brier_loss": "candidate"})
        pair = left.merge(right, on=["origin_id", "capology_extension_event_id"], validate="one_to_one")
        observed = float((pair["baseline"] - pair["candidate"]).mean())
        comparison_diffs.append(abs(observed - row.pooled_brier_improvement))
    add("all_pooled_comparisons_recompute", max(comparison_diffs) < 1e-12, max(comparison_diffs), "<1e-12", "Pairwise results are independently reproducible.")

    for row in decisions.itertuples(index=False):
        reference = comparisons.loc[
            comparisons["endpoint"].eq(row.endpoint)
            & comparisons["baseline"].eq("B0_intercept_only")
            & comparisons["candidate"].eq(row.selected_phase2_candidate)
        ].iloc[0]
        add(
            f"{row.endpoint}_validated_signal",
            reference.bootstrap_ci_lower_95 > 0 and reference.origin_wins >= 3,
            {"improvement": reference.pooled_brier_improvement, "ci_lower": reference.bootstrap_ci_lower_95, "wins": int(reference.origin_wins)},
            "positive lower CI and >=3 origin wins",
            "Candidate beats chronology-only history without prohibited inputs.",
        )
        market = comparisons.loc[
            comparisons["endpoint"].eq(row.endpoint)
            & comparisons["baseline"].eq("B1_market_profile")
            & comparisons["candidate"].eq(row.selected_phase2_candidate)
        ].iloc[0]
        add(
            f"{row.endpoint}_recent_involvement_adds_value",
            market.pooled_brier_improvement > 0 and market.origin_wins >= 2,
            {"improvement": market.pooled_brier_improvement, "wins": int(market.origin_wins)},
            "positive pooled improvement and >=2 wins",
            "The selected specification adds more than age/position/league/value alone.",
        )

    add("subgroup_rows_present", len(subgroup) > 0 and subgroup["n"].gt(0).all(), len(subgroup), ">0", "Phase 3 has saved subgroup inputs to audit.")
    add("twelve_v1_artifacts_used_raw_scoring", inventory.loc[inventory["feature_group"].eq("raw_scoring_rates"), "artifact"].nunique() == 12, int(inventory.loc[inventory["feature_group"].eq("raw_scoring_rates"), "artifact"].nunique()), 12, "Reproduces the prior deployment audit.")
    add("thirteen_v1_artifacts_violate_phase2_policy", inventory.loc[inventory["v2_violation"], "artifact"].nunique() == 13, int(inventory.loc[inventory["v2_violation"], "artifact"].nunique()), 13, "Violations remain audit evidence, not modified artifacts.")

    bad_sources = []
    for row in sources.itertuples(index=False):
        path = ROOT / row.source
        if not path.exists() or sha256(path) != row.sha256:
            bad_sources.append(row.source)
    add("source_manifest_integrity", not bad_sources, bad_sources, [], "Every source hash still matches.")

    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    bad_frozen = [item["path"] for item in freeze["files"] if sha256(ROOT / item["path"]) != item["sha256"]]
    add("v1_hashes_unchanged", not bad_frozen, bad_frozen, [], "All Phase-0 protected files remain frozen.")
    add("no_phase2_joblib", not list(OUTPUT.rglob("*.joblib")), [str(x) for x in OUTPUT.rglob("*.joblib")], [], "No candidate can be loaded accidentally as production.")
    add("summary_reconciles", summary["evaluated_endpoints"] == len(decisions) and summary["out_of_time_prediction_rows"] == len(predictions), {"endpoints": summary["evaluated_endpoints"], "predictions": summary["out_of_time_prediction_rows"]}, {"endpoints": len(decisions), "predictions": len(predictions)}, "Top-level counts agree.")

    manifest = pd.read_csv(OUTPUT / "output_manifest.csv")
    bad_outputs = []
    for row in manifest.itertuples(index=False):
        path = OUTPUT / row.file
        if not path.exists() or path.stat().st_size != row.bytes or sha256(path) != row.sha256:
            bad_outputs.append(row.file)
    add("output_manifest_integrity", not bad_outputs, bad_outputs, [], "Every declared Phase-2 output matches its size and hash.")

    checks = pd.DataFrame(results)
    checks.to_csv(OUTPUT / "independent_verification.csv", index=False, lineterminator="\n")
    payload = {
        "checks_passed": int(checks["passed"].sum()),
        "checks_total": len(checks),
        "all_checks_passed": bool(checks["passed"].all()),
        "verified_scope": "phase2_feature_contract_chronological_candidate_signal_and_v1_isolation",
    }
    (OUTPUT / "independent_verification.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest_rows = []
    for path in sorted(OUTPUT.iterdir()):
        if path.is_file() and path.name != "output_manifest.csv":
            manifest_rows.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest_rows).to_csv(OUTPUT / "output_manifest.csv", index=False, lineterminator="\n")
    print(json.dumps(payload, indent=2))
    if not payload["all_checks_passed"]:
        failed = checks.loc[~checks["passed"], "check"].tolist()
        raise SystemExit(f"Failed checks: {failed}")


if __name__ == "__main__":
    main()

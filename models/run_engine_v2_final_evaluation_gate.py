"""Engine V2 Phase 5: final-evaluation readiness and sealed wage holdout.

This phase does not deploy or serialize a model.  It audits whether later
extension cohorts satisfy the frozen maturity and identity-separation rules.
Only the annual-wage benchmark has a sufficiently mature, player-disjoint 2024
cohort, so it is the only sealed holdout opened here.  Predictive target values
for the other modules are never summarized or scored.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "models") not in sys.path:
    sys.path.insert(0, str(ROOT / "models"))

import run_contract_financial_exposure_benchmark as wage_source  # noqa: E402
import run_engine_v2_feature_specification as phase2  # noqa: E402
import run_engine_v2_candidate_contract as phase4  # noqa: E402


OUTPUT = ROOT / "Data" / "processed" / "engine_v2_final_evaluation_gate"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"
MASTER = ROOT / "Data" / "processed" / "contract_extension_integration" / "extension_modeling_master.csv"
PHASE4 = ROOT / "Data" / "processed" / "engine_v2_candidate_contract"

HOLDOUT_YEAR = 2024
DEVELOPMENT_CUTOFF = 2023
MIN_FINAL_POOLED_ROWS = 200
MIN_FINAL_GROUP_ROWS = 30
WAGE_FEATURES = list(wage_source.FEATURE_SETS["B2_plus_prior_wage"])
WAGE_TARGET = wage_source.ENDPOINTS["annual_wage"]["target"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def readiness_row(endpoint: str, module: str, frame: pd.DataFrame, prior_exposure: str) -> dict:
    development = frame.loc[frame["signed_year"].le(DEVELOPMENT_CUTOFF)].copy()
    later = frame.loc[frame["signed_year"].eq(HOLDOUT_YEAR)].copy()
    development_players = set(development["canonical_player_id"].dropna())
    repeated = later["canonical_player_id"].isin(development_players)
    disjoint = later.loc[~repeated].copy()
    league_counts = disjoint.groupby("league", dropna=False).size()
    position_counts = disjoint.groupby("canonical_position", dropna=False).size()
    minimum_league = int(league_counts.min()) if len(league_counts) else 0
    minimum_position = int(position_counts.min()) if len(position_counts) else 0
    count_ready = len(disjoint) >= MIN_FINAL_POOLED_ROWS and minimum_league >= MIN_FINAL_GROUP_ROWS
    pristine = prior_exposure == "no_target_or_performance_exposure_found"
    eligible_to_open = bool(count_ready and pristine and endpoint == "annual_wage_peer_benchmark")
    if eligible_to_open:
        decision = "open_sealed_2024_holdout"
    elif not pristine:
        decision = "do_not_open_not_pristine"
    else:
        decision = "do_not_open_underpowered"
    return {
        "module": module,
        "endpoint": endpoint,
        "development_cutoff": DEVELOPMENT_CUTOFF,
        "candidate_holdout_year": HOLDOUT_YEAR,
        "development_eligible_rows": int(len(development)),
        "later_eligible_rows": int(len(later)),
        "repeat_player_rows_removed": int(repeated.sum()),
        "player_disjoint_rows": int(len(disjoint)),
        "player_disjoint_unique_players": int(disjoint["canonical_player_id"].nunique()),
        "minimum_player_disjoint_league_rows": minimum_league,
        "minimum_player_disjoint_position_rows": minimum_position,
        "prior_label_or_performance_exposure": prior_exposure,
        "count_gate_passed": bool(count_ready),
        "eligible_to_open": eligible_to_open,
        "phase5_decision": decision,
    }


def readiness_audit() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    frames = phase2.load_endpoint_frames()
    wage = wage_source.load_frame()
    wage = wage.loc[wage["benchmark_wage_cohort"]].copy()
    frames["annual_wage_peer_benchmark"] = wage

    exposure = {
        "sustained_meaningful_contribution": "aggregate_2024_labels_exposed_in_phase4_target_relationship_audit",
        "any_outbound_24m": "aggregate_2024_labels_exposed_in_phase4_target_relationship_audit",
        "downside_25pct_24m": "no_target_or_performance_exposure_found",
        "annual_wage_peer_benchmark": "no_target_or_performance_exposure_found",
    }
    modules = {
        "sustained_meaningful_contribution": "future_role",
        "any_outbound_24m": "club_continuity",
        "downside_25pct_24m": "public_value_downside",
        "annual_wage_peer_benchmark": "wage_benchmark",
    }
    rows = [readiness_row(endpoint, modules[endpoint], frame, exposure[endpoint]) for endpoint, frame in frames.items()]
    return pd.DataFrame(rows), frames


def wage_holdout(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    train = frame.loc[frame["signed_year"].le(2022)].copy()
    validation = frame.loc[frame["signed_year"].eq(2023)].copy()
    development_players = set(pd.concat([train, validation])["canonical_player_id"].dropna())
    evaluation = frame.loc[
        frame["signed_year"].eq(HOLDOUT_YEAR)
        & ~frame["canonical_player_id"].isin(development_players)
    ].sort_values("capology_extension_event_id").copy()

    support = phase4.wage_support(pd.concat([train, validation], ignore_index=True), evaluation)
    candidate, tuning, encoded, half_width = wage_source.tune_predict(
        train, validation, evaluation, "annual_wage", "B2_plus_prior_wage", WAGE_FEATURES,
    )
    baseline, _, _, _ = wage_source.tune_predict(
        train, validation, evaluation, "annual_wage", "B0_chronological_median", [],
    )
    actual = pd.to_numeric(evaluation[WAGE_TARGET]).to_numpy(float)
    lower, upper = candidate - half_width, candidate + half_width

    predictions = pd.DataFrame({
        "capology_extension_event_id": evaluation["capology_extension_event_id"].to_numpy(),
        "canonical_player_id": evaluation["canonical_player_id"].to_numpy(),
        "league": evaluation["league"].to_numpy(),
        "canonical_position": evaluation["canonical_position"].to_numpy(),
        "age_band": evaluation["age_band"].astype("string").to_numpy(),
        "prior_wage_available": evaluation["prior_wage_available"].to_numpy(),
        "actual_log_eur": actual,
        "prediction_log_eur": candidate,
        "baseline_prediction_log_eur": baseline,
        "prediction_eur": np.exp(candidate),
        "interval_lower_eur": np.exp(lower),
        "interval_upper_eur": np.exp(upper),
        "within_80_interval": (actual >= lower) & (actual <= upper),
        "absolute_log_error": np.abs(actual - candidate),
        "baseline_absolute_log_error": np.abs(actual - baseline),
        "support_status": support["support_status"].to_numpy(),
        "support_reasons": support["support_reasons"].to_numpy(),
    })

    metric_rows = []
    for scope, sample in [("all_player_disjoint", predictions), ("scorable_only", predictions.loc[~predictions.support_status.eq("refused")])]:
        metric_rows.append({
            "scope": scope,
            "rows": int(len(sample)),
            "log_mae": float(sample.absolute_log_error.mean()),
            "baseline_log_mae": float(sample.baseline_absolute_log_error.mean()),
            "mae_improvement_vs_chronology": float((sample.baseline_absolute_log_error - sample.absolute_log_error).mean()),
            "interval_80_coverage": float(sample.within_80_interval.mean()),
            "refused_rate": float(predictions.support_status.eq("refused").mean()),
            "r2": float(r2_score(actual, candidate)) if scope == "all_player_disjoint" else np.nan,
            "selected_parameter": tuning["selected_parameter"],
            "encoded_features": int(encoded),
            "validation_interval_abs_residual_q80": float(half_width),
        })
    metrics = pd.DataFrame(metric_rows)

    subgroup_rows = []
    for group_type, column in [("league", "league"), ("position", "canonical_position"), ("prior_wage", "prior_wage_available")]:
        for group, sample in predictions.groupby(column, dropna=False):
            rows = len(sample)
            improvement = float((sample.baseline_absolute_log_error - sample.absolute_log_error).mean())
            coverage = float(sample.within_80_interval.mean())
            count_sufficient = rows >= MIN_FINAL_GROUP_ROWS
            passed = bool(count_sufficient and improvement > 0 and .70 <= coverage <= .90)
            subgroup_rows.append({
                "group_type": group_type,
                "group": str(group),
                "rows": int(rows),
                "log_mae": float(sample.absolute_log_error.mean()),
                "baseline_log_mae": float(sample.baseline_absolute_log_error.mean()),
                "mae_improvement_vs_chronology": improvement,
                "interval_80_coverage": coverage,
                "refused_rate": float(sample.support_status.eq("refused").mean()),
                "count_sufficient": bool(count_sufficient),
                "final_gate_status": "passed" if passed else "failed" if count_sufficient else "insufficient_evidence",
            })
    subgroup = pd.DataFrame(subgroup_rows)

    pooled = metrics.loc[metrics.scope.eq("all_player_disjoint")].iloc[0]
    broad = subgroup.loc[subgroup.group_type.isin(["league", "position"])]
    release_passed = bool(
        pooled.mae_improvement_vs_chronology > 0
        and .70 <= pooled.interval_80_coverage <= .90
        and pooled.refused_rate <= .15
        and broad.final_gate_status.eq("passed").all()
    )
    decision = {
        "candidate": "B2_plus_prior_wage",
        "holdout_year": HOLDOUT_YEAR,
        "holdout_kind": "sealed_player_disjoint_temporal",
        "holdout_rows": int(len(predictions)),
        "release_gate_passed": release_passed,
        "decision": "authorize_artifact_build" if release_passed else "block_artifact_build",
        "deployment_status": "not_deployed",
        "artifact_serialized": False,
        "notes": "Final holdout results are immutable. A failed gate may not be repaired by tuning on 2024.",
    }
    return predictions, metrics, subgroup, decision


def source_manifest() -> pd.DataFrame:
    sources = [
        (MASTER, "integrated extension events and outcome-maturity flags"),
        (PHASE4 / "phase4_decisions.json", "frozen candidate decisions"),
        (PHASE4 / "final_holdout_plan.json", "pre-existing holdout requirements"),
        (PHASE4 / "target_relationship_audit.csv", "prior aggregate label exposure evidence"),
        (FREEZE, "frozen Engine V1 hashes"),
    ]
    return pd.DataFrame([
        {"path": str(path.relative_to(ROOT)), "purpose": purpose, "sha256": sha256(path), "bytes": path.stat().st_size}
        for path, purpose in sources
    ])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT.iterdir():
        if path.is_file():
            path.unlink()

    phase4_verify = json.loads((PHASE4 / "independent_verification.json").read_text(encoding="utf-8"))
    if not phase4_verify.get("all_checks_passed"):
        raise RuntimeError("Phase 4 independent verification must pass before Phase 5.")

    readiness, frames = readiness_audit()
    wage_ready = readiness.loc[readiness.endpoint.eq("annual_wage_peer_benchmark"), "eligible_to_open"].item()
    if not wage_ready:
        raise RuntimeError("The wage holdout is not eligible to open under the frozen readiness gates.")
    predictions, metrics, subgroup, wage_decision = wage_holdout(frames["annual_wage_peer_benchmark"])

    exposure = pd.DataFrame([
        {
            "module": "future_role_and_continuity",
            "phase": "phase4_target_relationship_audit",
            "exposure": "aggregate_joint_label_counts",
            "affected_holdout_year": 2024,
            "affected_eligible_rows": int(len(frames["sustained_meaningful_contribution"].loc[lambda x: x.signed_year.eq(2024)])),
            "performance_predictions_seen": False,
            "holdout_classification": "not_pristine_restricted_confirmatory_only",
        },
        {
            "module": "public_value_downside",
            "phase": "engine_v2_phases1_to4",
            "exposure": "none_found",
            "affected_holdout_year": 2024,
            "affected_eligible_rows": int(len(frames["downside_25pct_24m"].loc[lambda x: x.signed_year.eq(2024)])),
            "performance_predictions_seen": False,
            "holdout_classification": "sealed_but_underpowered",
        },
        {
            "module": "wage_benchmark",
            "phase": "phase5",
            "exposure": "sealed_holdout_opened_once",
            "affected_holdout_year": 2024,
            "affected_eligible_rows": int(len(predictions)),
            "performance_predictions_seen": True,
            "holdout_classification": "opened_immutable_final_evaluation",
        },
    ])

    overall = {
        "phase": 5,
        "engine_version": "2.0.0-candidate",
        "decision": "not_ready_for_engine_deployment",
        "wage_benchmark": wage_decision,
        "future_role": "holdout_not_opened_underpowered_and_not_pristine",
        "club_continuity": "release_blocked_E2-CRIT-001_and_holdout_not_pristine",
        "public_value_downside": "holdout_not_opened_underpowered",
        "artifact_policy": "No predictive or benchmark artifact may be serialized unless its immutable final gate passes; Phase 5 itself serializes none.",
    }

    readiness.to_csv(OUTPUT / "holdout_readiness.csv", index=False, lineterminator="\n")
    exposure.to_csv(OUTPUT / "prior_exposure_audit.csv", index=False, lineterminator="\n")
    predictions.to_csv(OUTPUT / "wage_final_holdout_predictions.csv", index=False, lineterminator="\n")
    metrics.to_csv(OUTPUT / "wage_final_holdout_metrics.csv", index=False, lineterminator="\n")
    subgroup.to_csv(OUTPUT / "wage_final_holdout_subgroups.csv", index=False, lineterminator="\n")
    write_json(OUTPUT / "phase5_decisions.json", overall)
    source_manifest().to_csv(OUTPUT / "source_manifest.csv", index=False, lineterminator="\n")

    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    freeze_ok = all(sha256(ROOT / item["path"]) == item["sha256"] for item in freeze["files"])
    checks = pd.DataFrame([
        ("phase4_verified", True, "Phase 4 verifier passed before execution."),
        ("only_wage_holdout_opened", readiness.eligible_to_open.sum() == 1 and wage_ready, "Risk holdouts remain sealed."),
        ("future_role_underpowered", readiness.loc[readiness.endpoint.eq("sustained_meaningful_contribution"), "phase5_decision"].item() != "open_sealed_2024_holdout", "No role scoring."),
        ("value_downside_underpowered", readiness.loc[readiness.endpoint.eq("downside_25pct_24m"), "phase5_decision"].item() == "do_not_open_underpowered", "No downside scoring."),
        ("prior_2024_exposure_not_hidden", exposure.holdout_classification.str.contains("not_pristine").any(), "Phase 4 aggregate exposure is explicit."),
        ("wage_player_identity_disjoint", not predictions.canonical_player_id.isin(set(frames["annual_wage_peer_benchmark"].loc[lambda x: x.signed_year.le(2023), "canonical_player_id"].dropna())).any(), "No player crosses development/final partitions."),
        ("wage_holdout_rows_frozen", len(predictions) == readiness.loc[readiness.endpoint.eq("annual_wage_peer_benchmark"), "player_disjoint_rows"].item(), "Readiness and evaluation cohorts agree."),
        ("wage_intervals_ordered", (predictions.interval_lower_eur <= predictions.prediction_eur).all() and (predictions.prediction_eur <= predictions.interval_upper_eur).all(), "Every estimate lies within its reference interval."),
        ("wage_refusals_explained", predictions.loc[predictions.support_status.eq("refused"), "support_reasons"].fillna("").str.len().gt(0).all(), "No silent refusal."),
        ("no_artifact_serialized", not list(OUTPUT.rglob("*.joblib")), "Phase 5 creates evidence only."),
        ("v1_hashes_unchanged", freeze_ok, "Frozen V1 remains byte-identical."),
    ], columns=["check", "passed", "notes"])
    checks.to_csv(OUTPUT / "build_checks.csv", index=False, lineterminator="\n")
    if not checks.passed.all():
        raise RuntimeError(checks.loc[~checks.passed].to_string(index=False))

    summary = {
        "phase": 5,
        "all_build_checks_passed": True,
        "holdout_year": HOLDOUT_YEAR,
        "wage_holdout_rows": int(len(predictions)),
        "wage_release_gate_passed": bool(wage_decision["release_gate_passed"]),
        "overall_decision": overall["decision"],
    }
    write_json(OUTPUT / "run_summary.json", summary)

    readme = [
        "# Engine V2 Phase 5: final-evaluation gate", "",
        "Phase 5 audits whether 2024 can honestly serve as a final temporal test. It opens only the annual-wage benchmark holdout; no other 2024 target is scored.", "",
        "## Readiness decision", "",
    ]
    for row in readiness.itertuples(index=False):
        readme.append(f"- **{row.module}:** `{row.phase5_decision}` — {row.player_disjoint_rows} player-disjoint eligible rows; minimum league count {row.minimum_player_disjoint_league_rows}.")
    pooled = metrics.loc[metrics.scope.eq("all_player_disjoint")].iloc[0]
    readme += [
        "", "## Sealed wage result", "",
        f"The frozen B2 prior-wage candidate was evaluated once on {len(predictions)} player-disjoint 2024 extensions. Log-MAE was {pooled.log_mae:.4f} versus {pooled.baseline_log_mae:.4f} for chronology-only median; the nominal 80% range covered {pooled.interval_80_coverage:.1%}; refusal was {pooled.refused_rate:.1%}.",
        f"The immutable release decision is `{wage_decision['decision']}`. The candidate was not serialized or deployed.", "",
        "## Why the remaining holdouts stayed closed", "",
        "Future-role and continuity labels for 81 eligible 2024 events were already included in Phase 4's aggregate joint-target table. No 2024 predictions or performance metrics were inspected, but these labels are no longer pristine. After also removing players seen through 2023, only 56 future-role rows remain. Public-value downside remains unexposed, but only 26 player-disjoint rows are mature. Neither can support the frozen pooled and Big-Five subgroup gates.", "",
        "Phase 5 therefore records readiness and refusal; it does not manufacture a final result from an underpowered sample.", "",
    ]
    (OUTPUT / "README.md").write_text("\n".join(readme), encoding="utf-8")

    manifest_rows = []
    for path in sorted(OUTPUT.iterdir()):
        if path.is_file() and path.name != "output_manifest.csv":
            manifest_rows.append({"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size})
    pd.DataFrame(manifest_rows).to_csv(OUTPUT / "output_manifest.csv", index=False, lineterminator="\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

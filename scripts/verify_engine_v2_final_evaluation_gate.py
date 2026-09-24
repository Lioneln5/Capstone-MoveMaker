"""Independent verification for Engine V2 Phase 5."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "models") not in sys.path:
    sys.path.insert(0, str(ROOT / "models"))

import run_contract_financial_exposure_benchmark as wage_source  # noqa: E402
from mixed_type_preprocessor import fit_preprocessor  # noqa: E402


OUTPUT = ROOT / "Data" / "processed" / "engine_v2_final_evaluation_gate"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"
FEATURES = list(wage_source.FEATURE_SETS["B2_plus_prior_wage"])
TARGET = wage_source.ENDPOINTS["annual_wage"]["target"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    checks: list[dict] = []

    def add(name: str, passed: bool, observed: object, expected: object, notes: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "observed": observed, "expected": expected, "notes": notes})

    readiness = pd.read_csv(OUTPUT / "holdout_readiness.csv")
    exposure = pd.read_csv(OUTPUT / "prior_exposure_audit.csv")
    predictions = pd.read_csv(OUTPUT / "wage_final_holdout_predictions.csv")
    metrics = pd.read_csv(OUTPUT / "wage_final_holdout_metrics.csv")
    subgroups = pd.read_csv(OUTPUT / "wage_final_holdout_subgroups.csv")
    decisions = json.loads((OUTPUT / "phase5_decisions.json").read_text(encoding="utf-8"))
    summary = json.loads((OUTPUT / "run_summary.json").read_text(encoding="utf-8"))

    add("four_modules_audited", len(readiness) == 4 and readiness.endpoint.nunique() == 4, len(readiness), 4, "Every candidate module receives a readiness decision.")
    add("only_wage_opened", readiness.loc[readiness.eligible_to_open, "endpoint"].tolist() == ["annual_wage_peer_benchmark"], readiness.loc[readiness.eligible_to_open, "endpoint"].tolist(), ["annual_wage_peer_benchmark"], "No predictive risk target was opened.")
    add("risk_holdouts_have_no_prediction_files", not any(path.name.startswith(("future_role", "continuity", "value_downside")) and "prediction" in path.name for path in OUTPUT.iterdir()), sorted(path.name for path in OUTPUT.iterdir()), "no risk prediction files", "Underpowered/non-pristine targets remain unscored.")
    add("phase4_exposure_explicit", (readiness.prior_label_or_performance_exposure.str.contains("aggregate_2024").sum() == 2), int(readiness.prior_label_or_performance_exposure.str.contains("aggregate_2024").sum()), 2, "Role and continuity are not mislabeled pristine.")
    value_row = readiness.loc[readiness.endpoint.eq("downside_25pct_24m")].iloc[0]
    add("value_holdout_underpowered", value_row.player_disjoint_rows == 26, int(value_row.player_disjoint_rows), 26, "Value target is underpowered.")
    add("value_exposure_explicit", "v1_oos" in value_row.prior_label_or_performance_exposure and value_row.phase5_decision == "do_not_open_not_pristine", [value_row.prior_label_or_performance_exposure, value_row.phase5_decision], ["contains v1_oos", "do_not_open_not_pristine"], "The former sealed claim is corrected.")
    add("role_holdout_underpowered", readiness.loc[readiness.endpoint.eq("sustained_meaningful_contribution"), "player_disjoint_rows"].item() == 56, int(readiness.loc[readiness.endpoint.eq("sustained_meaningful_contribution"), "player_disjoint_rows"].item()), 56, "Role target remains unscored.")

    frame = wage_source.load_frame()
    frame = frame.loc[frame.benchmark_wage_cohort].copy()
    train = frame.loc[frame.signed_year.le(2022)].copy()
    validation = frame.loc[frame.signed_year.eq(2023)].copy()
    development = pd.concat([train, validation], ignore_index=True)
    development_players = set(development.canonical_player_id.dropna())
    evaluation = frame.loc[frame.signed_year.eq(2024) & ~frame.canonical_player_id.isin(development_players)].sort_values("capology_extension_event_id").copy()
    add("wage_holdout_count_reconstructed", len(evaluation) == len(predictions) == 235, [len(evaluation), len(predictions)], [235, 235], "Cohort derives from raw eligibility plus player-disjoint rule.")
    add("wage_ids_exact", evaluation.capology_extension_event_id.astype(str).tolist() == predictions.capology_extension_event_id.astype(str).tolist(), len(evaluation), len(predictions), "No holdout row was added or removed after scoring.")
    add("player_identity_disjoint", not evaluation.canonical_player_id.isin(development_players).any(), int(evaluation.canonical_player_id.isin(development_players).sum()), 0, "No player crosses development and final partitions.")

    pre = fit_preprocessor(train, FEATURES)
    x_train, x_validation = pre.transform(train), pre.transform(validation)
    y_train = pd.to_numeric(train[TARGET]).to_numpy(float)
    y_validation = pd.to_numeric(validation[TARGET]).to_numpy(float)
    candidates = []
    for alpha in wage_source.RIDGE_ALPHAS:
        estimate = Ridge(alpha=alpha).fit(x_train, y_train).predict(x_validation)
        candidates.append((mean_absolute_error(y_validation, estimate), float(np.sqrt(np.mean((y_validation - estimate) ** 2))), alpha, estimate))
    _, _, alpha, validation_estimate = min(candidates, key=lambda item: item[:2])
    half_width = float(np.quantile(np.abs(y_validation - validation_estimate), .80))
    fit_pre = fit_preprocessor(development, FEATURES)
    expected = Ridge(alpha=alpha).fit(fit_pre.transform(development), pd.to_numeric(development[TARGET]).to_numpy(float)).predict(fit_pre.transform(evaluation))
    prediction_diff = float(np.max(np.abs(expected - predictions.prediction_log_eur.to_numpy(float))))
    add("selected_alpha_reconstructed", alpha == 10.0, alpha, 10.0, "Final fit uses the frozen validation recipe.")
    add("wage_predictions_reconstructed", prediction_diff < 1e-12, prediction_diff, "<1e-12", "Independent Ridge refit reproduces every final prediction.")
    add("interval_width_reconstructed", abs(half_width - metrics.validation_interval_abs_residual_q80.dropna().iloc[0]) < 1e-12, half_width, float(metrics.validation_interval_abs_residual_q80.dropna().iloc[0]), "Interval is fixed from 2023 validation residuals.")

    actual = pd.to_numeric(evaluation[TARGET]).to_numpy(float)
    baseline = np.repeat(float(pd.to_numeric(development[TARGET]).median()), len(evaluation))
    diffs = {
        "actual": float(np.max(np.abs(actual - predictions.actual_log_eur.to_numpy(float)))),
        "baseline": float(np.max(np.abs(baseline - predictions.baseline_prediction_log_eur.to_numpy(float)))),
        "candidate_loss": abs(mean_absolute_error(actual, expected) - metrics.loc[metrics.scope.eq("all_player_disjoint"), "log_mae"].item()),
        "baseline_loss": abs(mean_absolute_error(actual, baseline) - metrics.loc[metrics.scope.eq("all_player_disjoint"), "baseline_log_mae"].item()),
    }
    add("wage_targets_and_losses_reconstructed", max(diffs.values()) < 1e-12, diffs, "all <1e-12", "Saved final metrics derive from row-level evidence.")
    coverage = float(np.mean((actual >= expected - half_width) & (actual <= expected + half_width)))
    add("pooled_interval_coverage_reconstructed", abs(coverage - metrics.loc[metrics.scope.eq("all_player_disjoint"), "interval_80_coverage"].item()) < 1e-12, coverage, float(metrics.loc[metrics.scope.eq("all_player_disjoint"), "interval_80_coverage"].item()), "Final coverage uses the frozen interval.")
    add("pooled_wage_improves_baseline", metrics.loc[metrics.scope.eq("all_player_disjoint"), "mae_improvement_vs_chronology"].item() > 0, float(metrics.loc[metrics.scope.eq("all_player_disjoint"), "mae_improvement_vs_chronology"].item()), ">0", "Pooled performance is positive but not sufficient for release.")
    add("pooled_coverage_in_gate", .70 <= coverage <= .90, coverage, "[0.70,0.90]", "Pooled interval coverage passes.")
    refused_rate = float(predictions.support_status.eq("refused").mean())
    add("refusal_rate_reconstructed", abs(refused_rate - metrics.loc[metrics.scope.eq("all_player_disjoint"), "refused_rate"].item()) < 1e-12 and refused_rate <= .15, refused_rate, "saved and <=0.15", "Coverage is operationally acceptable.")
    add("refusals_explained", predictions.loc[predictions.support_status.eq("refused"), "support_reasons"].fillna("").str.len().gt(0).all(), int(predictions.support_status.eq("refused").sum()), "all explained", "No silent refusal.")

    subgroup_diffs = []
    for row in subgroups.itertuples(index=False):
        column = {"league": "league", "position": "canonical_position", "prior_wage": "prior_wage_available"}[row.group_type]
        sample = predictions.loc[predictions[column].astype(str).eq(str(row.group))]
        subgroup_diffs.extend([
            abs(len(sample) - row.rows),
            abs(sample.absolute_log_error.mean() - row.log_mae),
            abs(sample.baseline_absolute_log_error.mean() - row.baseline_log_mae),
            abs(sample.within_80_interval.mean() - row.interval_80_coverage),
        ])
    add("subgroup_metrics_reconstructed", max(subgroup_diffs) < 1e-12, max(subgroup_diffs), "<1e-12", "Every subgroup row derives from final predictions.")
    ligue = subgroups.loc[(subgroups.group_type.eq("league")) & subgroups.group.eq("Ligue 1")].iloc[0]
    add("ligue1_failure_explicit", ligue.final_gate_status == "failed" and ligue.interval_80_coverage < .70, [ligue.final_gate_status, float(ligue.interval_80_coverage)], ["failed", "<0.70"], "A pooled pass cannot hide failed league coverage.")
    add("wage_artifact_build_blocked", decisions["wage_benchmark"]["decision"] == "block_artifact_build" and not decisions["wage_benchmark"]["release_gate_passed"], decisions["wage_benchmark"], "blocked", "No post-holdout tuning or promotion.")
    add("overall_engine_blocked", decisions["decision"] == "not_ready_for_engine_deployment", decisions["decision"], "not_ready_for_engine_deployment", "Phase 5 does not authorize deployment.")
    add("no_phase5_model_artifacts", not list(OUTPUT.rglob("*.joblib")), len(list(OUTPUT.rglob("*.joblib"))), 0, "Evidence only.")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    mutable_verification_reports = {
        "Data/processed/deployment_models/live_request_integrity.csv",
        "Data/processed/deployment_models/live_request_integrity.json",
    }
    changed = [
        item["path"] for item in freeze["files"]
        if item["path"] not in mutable_verification_reports
        and sha256(ROOT / item["path"]) != item["sha256"]
    ]
    add("v1_hashes_unchanged", not changed, changed, [], "Frozen V1 remains byte-identical.")
    add("summary_matches", summary["wage_holdout_rows"] == len(predictions) and not summary["wage_release_gate_passed"], summary, "235 rows and gate false", "Run summary reflects evidence.")
    value_exposure = exposure.loc[exposure.module.eq("public_value_downside")].iloc[0]
    add("prior_exposure_rows_present", len(exposure) == 3 and exposure.holdout_classification.str.contains("not_pristine").sum() == 2, exposure.holdout_classification.tolist(), "two not_pristine rows", "Holdout contamination is permanent and visible.")
    add("value_performance_exposure_recorded", bool(value_exposure.performance_predictions_seen) and "v1" in value_exposure.holdout_classification, value_exposure.to_dict(), "V1 performance exposed", "Value exposure predates Phase 5 and cannot be undone.")

    results = pd.DataFrame(checks)
    results.to_csv(OUTPUT / "independent_verification.csv", index=False, lineterminator="\n")
    payload = {
        "all_checks_passed": bool(results.passed.all()),
        "checks_passed": int(results.passed.sum()),
        "checks_total": int(len(results)),
        "failed_checks": results.loc[~results.passed, "check"].tolist(),
    }
    (OUTPUT / "independent_verification.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not payload["all_checks_passed"]:
        raise RuntimeError(results.loc[~results.passed].to_string(index=False))

    manifest_rows = []
    for path in sorted(OUTPUT.iterdir()):
        if path.is_file() and path.name != "output_manifest.csv":
            manifest_rows.append({"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size})
    pd.DataFrame(manifest_rows).to_csv(OUTPUT / "output_manifest.csv", index=False, lineterminator="\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

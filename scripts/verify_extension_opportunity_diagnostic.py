"""Independently verify the Phase-1 extension opportunity diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Data" / "processed" / "extension_opportunity_diagnostic"
FROZEN_SOURCE = ROOT / "Data" / "processed" / "frozen_contract_sources" / "phase1_5_2026-08-11" / "contract_extension_integration"
SOURCE = FROZEN_SOURCE / "extension_modeling_master.csv"
UPSTREAM_VERIFY = FROZEN_SOURCE / "independent_verification.json"

FEATURE_VARIANTS = {
    "M0_chronological_baseline",
    "M1_player_market_profile",
    "M2_plus_prior_opportunity",
    "M3_plus_performance_history",
    "M4_plus_contract_terms",
    "M5_plus_relative_financial_context",
    "M6_compact_nonlinear",
}
ENDPOINT_TARGETS = {
    "year2_opportunity_share": "target_year2_opportunity_share",
    "major_role_decline": "target_major_role_decline",
    "sustained_meaningful_contribution": "target_sustained_meaningful_contribution",
    "contract_covered_minimal_involvement": "target_contract_covered_minimal_involvement",
}
EXPECTED_DECISIONS = {
    "year2_opportunity_share": "advance_predictive_candidate",
    "major_role_decline": "descriptive_only_no_validated_incremental_signal",
    "sustained_meaningful_contribution": "advance_predictive_candidate",
    "contract_covered_minimal_involvement": "exploratory_signal_calibration_or_ranking_failed",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").fillna("").str.strip().str.casefold().isin({"true", "1", "yes"})


def add(checks: list[dict], name: str, observed: object, expected: object, passed: bool, notes: str) -> None:
    checks.append({"check": name, "observed": observed, "expected": expected, "passed": bool(passed), "notes": notes})


def continuous_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    correlation = pd.Series(actual).corr(pd.Series(prediction), method="spearman")
    return {
        "evaluation_mae": mean_absolute_error(actual, prediction),
        "evaluation_rmse": np.sqrt(mean_squared_error(actual, prediction)),
        "evaluation_r2": r2_score(actual, prediction),
        "evaluation_spearman": correlation,
    }


def binary_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "evaluation_brier": brier_score_loss(actual, prediction),
        "evaluation_log_loss": log_loss(actual, prediction, labels=[0, 1]),
        "evaluation_actual_rate": actual.mean(),
        "evaluation_predicted_rate": prediction.mean(),
        "evaluation_roc_auc": roc_auc_score(actual, prediction),
        "evaluation_average_precision": average_precision_score(actual, prediction),
    }


def close_or_both_nan(left: object, right: object, tolerance: float = 1e-10) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return bool(np.isclose(float(left), float(right), atol=tolerance, rtol=tolerance))


def main() -> None:
    required = {
        "opportunity_model_matrix.csv", "cohort_funnel.csv", "target_distributions.csv",
        "role_band_transitions.csv", "threshold_sensitivity.csv", "feature_manifest.csv",
        "model_origin_metrics.csv", "model_predictions.csv", "model_comparison_results.csv",
        "calibration_risk_bands.csv", "subgroup_stability.csv", "opportunity_decision_summary.csv",
        "build_checks.csv", "source_manifest.csv", "output_manifest.csv", "run_summary.json", "README.md",
    }
    missing = sorted(name for name in required if not (OUTPUT / name).exists())
    if missing:
        raise RuntimeError(f"Missing diagnostic artifacts: {missing}")

    checks: list[dict] = []
    add(checks, "required_outputs", len(missing), 0, not missing, "All declared diagnostic artifacts exist.")

    upstream = json.loads(UPSTREAM_VERIFY.read_text(encoding="utf-8"))
    add(checks, "upstream_verification", upstream.get("all_checks_passed"), True, bool(upstream.get("all_checks_passed")), "The contract integration passed its independent verifier.")

    master = pd.read_csv(SOURCE, low_memory=False)
    matrix = pd.read_csv(OUTPUT / "opportunity_model_matrix.csv", low_memory=False)
    predictions = pd.read_csv(OUTPUT / "model_predictions.csv", low_memory=False)
    origin_metrics = pd.read_csv(OUTPUT / "model_origin_metrics.csv", low_memory=False)
    comparisons = pd.read_csv(OUTPUT / "model_comparison_results.csv", low_memory=False)
    bands = pd.read_csv(OUTPUT / "calibration_risk_bands.csv", low_memory=False)
    decisions = pd.read_csv(OUTPUT / "opportunity_decision_summary.csv", low_memory=False)
    features = pd.read_csv(OUTPUT / "feature_manifest.csv", low_memory=False)
    build_checks = pd.read_csv(OUTPUT / "build_checks.csv", low_memory=False)

    add(checks, "source_row_count", len(master), 2425, len(master) == 2425, "All clean extension events remain available to cohort construction.")
    add(checks, "matrix_event_grain", int(matrix["capology_extension_event_id"].duplicated().sum()), 0, not matrix["capology_extension_event_id"].duplicated().any(), "One matrix row per extension event.")
    add(checks, "matrix_source_id_set", len(set(matrix["capology_extension_event_id"])), len(set(master["capology_extension_event_id"])), set(matrix["capology_extension_event_id"]) == set(master["capology_extension_event_id"]), "The model matrix preserves the exact source ID set.")

    signed = pd.to_datetime(master["signed_date"], errors="coerce")
    expiration = pd.to_datetime(master["expiration_date"], errors="coerce")
    pre = pd.to_numeric(master["pre365_same_club_all_competition_opportunity_share"], errors="coerce")
    y1 = pd.to_numeric(master["post_y1_same_club_all_competition_opportunity_share"], errors="coerce")
    y2 = pd.to_numeric(master["post_y2_same_club_all_competition_opportunity_share"], errors="coerce")
    linked = master["canonical_player_id"].notna()
    evidence = as_bool(master["pre365_window_evidence_eligible_10_club_games"]) & as_bool(master["post_y2_window_evidence_eligible_10_club_games"])
    covers_y2 = expiration.ge(signed + pd.Timedelta(days=730))
    primary = linked & evidence & covers_y2 & pre.notna() & y2.notna()
    initial = primary & pre.ge(0.25)
    uninterrupted = pd.to_numeric(master["uninterrupted_same_club_spell_730d"], errors="coerce").eq(1)
    expected_targets = pd.DataFrame({
        "capology_extension_event_id": master["capology_extension_event_id"],
        "cohort_primary_y2": primary,
        "cohort_initial_role_y2": initial,
        "target_year2_opportunity_share": y2,
        "target_opportunity_change": y2 - pre,
        "target_major_role_decline": ((pre - y2).ge(0.20) & y2.le(0.50 * pre)).astype(int),
        "target_sustained_meaningful_contribution": (y1.ge(0.25) & y2.ge(0.25)).astype(int),
        "target_contract_covered_minimal_involvement": (y2.lt(0.10) & uninterrupted).astype(int),
    })
    observed_targets = matrix[list(expected_targets.columns)].copy()
    merged = expected_targets.merge(observed_targets, on="capology_extension_event_id", suffixes=("_expected", "_observed"), validate="one_to_one")
    target_mismatches = 0
    for column in expected_targets.columns[1:]:
        expected = merged[f"{column}_expected"]
        observed = merged[f"{column}_observed"]
        if column.startswith("cohort_"):
            target_mismatches += int((as_bool(expected) != as_bool(observed)).sum())
        else:
            target_mismatches += int((~np.isclose(pd.to_numeric(expected, errors="coerce"), pd.to_numeric(observed, errors="coerce"), equal_nan=True)).sum())
    add(checks, "target_recomputation", target_mismatches, 0, target_mismatches == 0, "All cohorts and outcomes reproduce directly from independently loaded source fields.")
    add(checks, "primary_cohort_size", int(primary.sum()), 1385, int(primary.sum()) == 1385, "Eligible Year-2 contract-covered cohort is fixed.")
    add(checks, "initial_role_cohort_size", int(initial.sum()), 922, int(initial.sum()) == 922, "Major-decline cohort requires at least 25% pre-extension opportunity.")
    major_rate = float(expected_targets.loc[initial, "target_major_role_decline"].mean())
    sustained_rate = float(expected_targets.loc[primary, "target_sustained_meaningful_contribution"].mean())
    minimal_rate = float(expected_targets.loc[primary, "target_contract_covered_minimal_involvement"].mean())
    add(checks, "major_decline_prevalence", round(major_rate, 9), round(327 / 922, 9), np.isclose(major_rate, 327 / 922), "The pre-registered decline event count is independently reproduced.")
    add(checks, "sustained_contribution_prevalence", round(sustained_rate, 9), round(694 / 1385, 9), np.isclose(sustained_rate, 694 / 1385), "The pre-registered sustained-contribution count is independently reproduced.")
    add(checks, "minimal_involvement_prevalence", round(minimal_rate, 9), round(120 / 1385, 9), np.isclose(minimal_rate, 120 / 1385), "The contract-covered minimal-involvement count is independently reproduced.")

    timing_safe = features["timing"].eq("known_at_extension_signing").all() and not features["feature"].str.startswith("post_").any()
    add(checks, "feature_timing", int(timing_safe), 1, timing_safe, "All modeled features are declared available at signing and no post-signing feature appears.")
    declared_variants = set(features["model_variant"])
    expected_feature_variants = FEATURE_VARIANTS - {"M0_chronological_baseline"}
    add(checks, "feature_variants", len(declared_variants), len(expected_feature_variants), declared_variants == expected_feature_variants, "Feature manifest covers every non-empty feature set; M0 is intentionally featureless.")

    prediction_key = ["endpoint", "origin_id", "model_variant", "capology_extension_event_id"]
    add(checks, "prediction_grain", int(predictions.duplicated(prediction_key).sum()), 0, not predictions.duplicated(prediction_key).any(), "One prediction per endpoint/origin/model/event.")
    add(checks, "prediction_row_count", len(predictions), 23730, len(predictions) == 23730, "Prediction count matches the locked evaluation cohorts and model ladder.")
    add(checks, "evaluation_years", " | ".join(map(str, sorted(predictions["evaluation_year"].unique()))), "2020 | 2021 | 2022 | 2023", sorted(predictions["evaluation_year"].unique()) == [2020, 2021, 2022, 2023], "Only declared out-of-time years are evaluated.")
    add(checks, "prediction_variants", len(set(predictions["model_variant"])), 7, set(predictions["model_variant"]) == FEATURE_VARIANTS, "Every endpoint/origin uses the fixed feature ladder.")
    binary = predictions["kind"].eq("binary")
    bounds_ok = predictions.loc[binary, "prediction"].between(0, 1).all() and predictions.loc[~binary, "prediction"].between(0, 1.05).all()
    add(checks, "prediction_bounds", int(bounds_ok), 1, bounds_ok, "Probabilities and opportunity-share estimates obey declared bounds.")

    matrix_targets = matrix[["capology_extension_event_id", *ENDPOINT_TARGETS.values()]].copy()
    actual_check = predictions.merge(matrix_targets, on="capology_extension_event_id", how="left", validate="many_to_one")
    actual_mismatches = 0
    for endpoint, target in ENDPOINT_TARGETS.items():
        rows = actual_check["endpoint"].eq(endpoint)
        actual_mismatches += int((~np.isclose(actual_check.loc[rows, "actual"], actual_check.loc[rows, target], equal_nan=True)).sum())
    add(checks, "prediction_actual_values", actual_mismatches, 0, actual_mismatches == 0, "Saved actuals match the independently reconstructed event targets.")

    metric_key = ["endpoint", "origin_id", "model_variant"]
    add(checks, "origin_metric_grain", int(origin_metrics.duplicated(metric_key).sum()), 0, not origin_metrics.duplicated(metric_key).any(), "One metric record per endpoint/origin/model.")
    add(checks, "origin_metric_row_count", len(origin_metrics), 112, len(origin_metrics) == 4 * 4 * 7, "Four endpoints by four origins by seven model variants.")
    metric_mismatches = 0
    for key, group in predictions.groupby(metric_key, observed=True):
        saved = origin_metrics.loc[
            origin_metrics["endpoint"].eq(key[0]) & origin_metrics["origin_id"].eq(key[1]) & origin_metrics["model_variant"].eq(key[2])
        ].iloc[0]
        actual = group["actual"].to_numpy()
        prediction = group["prediction"].to_numpy()
        recalculated = continuous_metrics(actual, prediction) if group["kind"].iat[0] == "continuous" else binary_metrics(actual.astype(int), prediction)
        metric_mismatches += sum(not close_or_both_nan(saved[name], value) for name, value in recalculated.items())
    add(checks, "origin_metric_recomputation", metric_mismatches, 0, metric_mismatches == 0, "All evaluation metrics reproduce from row-level predictions.")

    add(checks, "comparison_row_count", len(comparisons), 96, len(comparisons) == 4 * 3 * 8, "All registered endpoint/window/comparison combinations exist.")
    comparison_mismatches = 0
    for row in comparisons.itertuples(index=False):
        years = [int(value.strip()) for value in str(row.evaluation_years).split("|")]
        subset = predictions[predictions["endpoint"].eq(row.endpoint) & predictions["evaluation_year"].isin(years)]
        keys = ["origin_id", "evaluation_year", "capology_extension_event_id", "canonical_player_id"]
        base = subset[subset["model_variant"].eq(row.baseline)][keys + ["primary_loss"]].rename(columns={"primary_loss": "base"})
        candidate = subset[subset["model_variant"].eq(row.candidate)][keys + ["primary_loss"]].rename(columns={"primary_loss": "candidate"})
        pair = base.merge(candidate, on=keys, validate="one_to_one")
        improvement = float((pair["base"] - pair["candidate"]).mean())
        wins = int(pair.assign(delta=pair["base"] - pair["candidate"]).groupby("evaluation_year")["delta"].mean().gt(0).sum())
        comparison_mismatches += int(not np.isclose(improvement, row.mean_loss_improvement, atol=1e-12, rtol=1e-12))
        comparison_mismatches += int(wins != row.origin_wins)
        comparison_mismatches += int(len(pair) != row.evaluation_rows)
    add(checks, "comparison_recomputation", comparison_mismatches, 0, comparison_mismatches == 0, "Paired loss improvements, origin wins, and row counts reproduce from predictions.")

    band_mismatches = 0
    add(checks, "risk_band_row_count", len(bands), 20, len(bands) == 4 * 5, "Five equal-frequency M5 bands are reported for every endpoint.")
    for endpoint, group in bands.groupby("endpoint", observed=True):
        observed_monotone = bool(group.sort_values("risk_band")["observed_outcome"].is_monotonic_increasing)
        band_mismatches += int(as_bool(group["observed_monotone"]).nunique() != 1)
        band_mismatches += int(as_bool(group["observed_monotone"]).iat[0] != observed_monotone)
    add(checks, "risk_band_monotonicity_flags", band_mismatches, 0, band_mismatches == 0, "Reported monotonicity flags agree with observed band outcomes.")

    decision_map = dict(zip(decisions["endpoint"], decisions["status"]))
    add(checks, "decision_endpoint_set", len(decision_map), 4, set(decision_map) == set(EXPECTED_DECISIONS), "One decision exists for every pre-registered endpoint.")
    add(checks, "decision_rule_results", decision_map, EXPECTED_DECISIONS, decision_map == EXPECTED_DECISIONS, "Statuses follow the documented evidence gates for this run.")
    total = comparisons[(comparisons["window"].eq("pooled_four")) & comparisons["comparison"].eq("M5_vs_M0_total")].set_index("endpoint")
    contract = comparisons[(comparisons["window"].eq("pooled_four")) & comparisons["comparison"].eq("M5_vs_M3_all_contract_economics")].set_index("endpoint")
    decision_metric_mismatches = 0
    for row in decisions.itertuples(index=False):
        decision_metric_mismatches += int(not np.isclose(row.total_loss_improvement_vs_baseline, total.loc[row.endpoint, "mean_loss_improvement"]))
        decision_metric_mismatches += int(not np.isclose(row.contract_incremental_improvement, contract.loc[row.endpoint, "mean_loss_improvement"]))
    add(checks, "decision_metric_traceability", decision_metric_mismatches, 0, decision_metric_mismatches == 0, "Decision metrics trace exactly to pooled comparison results.")

    source_manifest = pd.read_csv(OUTPUT / "source_manifest.csv")
    source_failures = []
    for row in source_manifest.itertuples(index=False):
        path = ROOT / row.source_file
        if not path.exists() or sha256(path) != row.sha256:
            source_failures.append(row.source_file)
    add(checks, "source_hashes", len(source_failures), 0, not source_failures, f"Failures: {source_failures or 'none'}")

    output_manifest = pd.read_csv(OUTPUT / "output_manifest.csv")
    output_failures = []
    for row in output_manifest.itertuples(index=False):
        if row.output_file in {"independent_verification.csv", "independent_verification.json"}:
            continue
        path = OUTPUT / row.output_file
        if not path.exists() or sha256(path) != row.sha256:
            output_failures.append(row.output_file)
    add(checks, "builder_output_hashes", len(output_failures), 0, not output_failures, f"Failures: {output_failures or 'none'}")
    build_ok = as_bool(build_checks["passed"]).all()
    add(checks, "builder_checks", int(build_ok), 1, build_ok, "All builder checks passed.")

    result = pd.DataFrame(checks)
    result.to_csv(OUTPUT / "independent_verification.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    payload = {
        "checks_passed": int(result["passed"].sum()),
        "checks_total": len(result),
        "all_checks_passed": bool(result["passed"].all()),
        "source_extension_events": len(master),
        "primary_y2_cohort": int(primary.sum()),
        "initial_role_y2_cohort": int(initial.sum()),
        "prediction_rows_verified": len(predictions),
        "decisions_verified": decision_map,
    }
    (OUTPUT / "independent_verification.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    files = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name != "output_manifest.csv")
    manifest = pd.DataFrame([{"output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files])
    manifest.to_csv(OUTPUT / "output_manifest.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    if not result["passed"].all():
        failed = result.loc[~result["passed"], "check"].tolist()
        raise RuntimeError(f"Independent opportunity-diagnostic verification failed: {failed}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

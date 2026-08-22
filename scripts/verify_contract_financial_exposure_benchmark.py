"""Independently verify the Phase-4 financial exposure benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Data" / "processed" / "contract_financial_exposure_benchmark"
FROZEN_SOURCE = ROOT / "Data" / "processed" / "frozen_contract_sources" / "phase1_5_2026-08-11" / "contract_extension_integration"
SOURCE = FROZEN_SOURCE / "extension_modeling_master.csv"
UPSTREAM_VERIFY = FROZEN_SOURCE / "independent_verification.json"
ENDPOINTS = {
    "annual_wage": ("benchmark_wage_cohort", "target_log_annual_wage_eur"),
    "contract_duration": ("benchmark_duration_cohort", "target_contract_duration_years"),
    "fixed_wage_commitment": ("benchmark_commitment_cohort", "target_log_fixed_wage_commitment_eur"),
    "wage_to_market_value": ("benchmark_wage_value_cohort", "target_log_wage_to_market_value"),
}
SELECTED_MODELS = {
    "annual_wage": "B2_plus_prior_wage",
    "contract_duration": "B5_compact_nonlinear",
    "fixed_wage_commitment": "B2_plus_prior_wage",
    "wage_to_market_value": "B2_plus_prior_wage",
}
EXPECTED_DECISIONS = {endpoint: "advance_historical_peer_benchmark" for endpoint in ENDPOINTS}
WINDOWS = {
    "pooled_four": [2020, 2021, 2022, 2023],
    "mature_three": [2021, 2022, 2023],
    "terminal_two": [2022, 2023],
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


def metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    rho = np.nan if np.ptp(actual) == 0 or np.ptp(prediction) == 0 else pd.Series(actual).corr(pd.Series(prediction), method="spearman")
    return {
        "evaluation_mae": float(mean_absolute_error(actual, prediction)),
        "evaluation_rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
        "evaluation_r2": float(r2_score(actual, prediction)),
        "evaluation_spearman": float(rho) if pd.notna(rho) else np.nan,
        "evaluation_median_absolute_error": float(np.median(np.abs(actual - prediction))),
    }


def close(left: object, right: object, tolerance: float = 1e-10) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return bool(np.isclose(float(left), float(right), atol=tolerance, rtol=tolerance))


def main() -> None:
    required = {
        "financial_exposure_model_matrix.csv", "benchmark_cohort_audit.csv", "bonus_coverage_by_year.csv",
        "deterministic_metric_catalog.csv", "financial_scope_limitations.csv", "historical_peer_summary.csv",
        "feature_manifest.csv", "model_origin_metrics.csv", "benchmark_predictions.csv",
        "model_comparison_results.csv", "benchmark_bands.csv", "subgroup_stability.csv",
        "benchmark_decision_summary.csv", "build_checks.csv", "source_manifest.csv",
        "output_manifest.csv", "run_summary.json", "README.md",
    }
    missing = sorted(name for name in required if not (OUTPUT / name).exists())
    if missing:
        raise RuntimeError(f"Missing Phase-4 artifacts: {missing}")
    checks: list[dict] = []
    add(checks, "required_outputs", len(missing), 0, not missing, "All declared Phase-4 artifacts exist.")
    upstream = json.loads(UPSTREAM_VERIFY.read_text(encoding="utf-8"))
    add(checks, "upstream_verification", upstream.get("all_checks_passed"), True, bool(upstream.get("all_checks_passed")), "Contract integration passed independent verification.")

    source = pd.read_csv(SOURCE, low_memory=False)
    matrix = pd.read_csv(OUTPUT / "financial_exposure_model_matrix.csv", low_memory=False)
    predictions = pd.read_csv(OUTPUT / "benchmark_predictions.csv", low_memory=False)
    origin_metrics = pd.read_csv(OUTPUT / "model_origin_metrics.csv", low_memory=False)
    comparisons = pd.read_csv(OUTPUT / "model_comparison_results.csv", low_memory=False)
    bands = pd.read_csv(OUTPUT / "benchmark_bands.csv", low_memory=False)
    decisions = pd.read_csv(OUTPUT / "benchmark_decision_summary.csv", low_memory=False)
    features = pd.read_csv(OUTPUT / "feature_manifest.csv", low_memory=False)
    catalog = pd.read_csv(OUTPUT / "deterministic_metric_catalog.csv", low_memory=False)
    limitations = pd.read_csv(OUTPUT / "financial_scope_limitations.csv", low_memory=False)
    bonus = pd.read_csv(OUTPUT / "bonus_coverage_by_year.csv", low_memory=False)
    build_checks = pd.read_csv(OUTPUT / "build_checks.csv", low_memory=False)

    add(checks, "source_row_count", len(source), 2425, len(source) == 2425, "Extension source row count is unchanged.")
    add(checks, "matrix_event_grain", int(matrix["capology_extension_event_id"].duplicated().sum()), 0, not matrix["capology_extension_event_id"].duplicated().any(), "One matrix row per extension event.")
    add(checks, "matrix_source_ids", len(set(matrix["capology_extension_event_id"])), len(set(source["capology_extension_event_id"])), set(matrix["capology_extension_event_id"]) == set(source["capology_extension_event_id"]), "Matrix preserves the exact source ID set.")

    source_lookup = source.set_index("capology_extension_event_id").loc[matrix["capology_extension_event_id"]]
    wage = pd.to_numeric(source_lookup["annual_gross_eur"], errors="coerce")
    duration = pd.to_numeric(source_lookup["exact_duration_years"], errors="coerce")
    market = pd.to_numeric(source_lookup["at_signing_market_value_eur"], errors="coerce")
    linked = source_lookup["canonical_player_id"].notna() & pd.to_datetime(source_lookup["signed_date"], errors="coerce").notna()
    commitment = wage * duration
    expected_cohorts = {
        "benchmark_wage_cohort": linked & wage.gt(0),
        "benchmark_duration_cohort": linked & duration.gt(0),
        "benchmark_commitment_cohort": linked & commitment.gt(0),
        "benchmark_wage_value_cohort": linked & wage.gt(0) & market.gt(0),
    }
    expected_counts = {"benchmark_wage_cohort": 2265, "benchmark_duration_cohort": 2358, "benchmark_commitment_cohort": 2265, "benchmark_wage_value_cohort": 2222}
    cohort_mismatches = 0
    for column, expected in expected_cohorts.items():
        observed = as_bool(matrix[column]).to_numpy()
        cohort_mismatches += int((observed != expected.to_numpy()).sum())
        add(checks, f"{column}_count", int(expected.sum()), expected_counts[column], int(expected.sum()) == expected_counts[column], "Eligible cohort is independently reproduced.")
    add(checks, "cohort_flags", cohort_mismatches, 0, cohort_mismatches == 0, "Every saved cohort flag matches the source-derived flag.")

    expected_targets = {
        "target_log_annual_wage_eur": np.log(wage.where(wage.gt(0))),
        "target_contract_duration_years": duration,
        "target_log_fixed_wage_commitment_eur": np.log(commitment.where(commitment.gt(0))),
        "target_log_wage_to_market_value": np.log(wage.div(market.where(market.gt(0))).where(wage.gt(0))),
        "target_fixed_wage_commitment_eur": commitment,
        "target_wage_to_market_value": wage.div(market.where(market.gt(0))),
    }
    target_mismatches = 0
    for column, expected in expected_targets.items():
        observed = pd.to_numeric(matrix[column], errors="coerce")
        target_mismatches += int((~np.isclose(observed, expected, equal_nan=True)).sum())
    add(checks, "target_and_formula_recomputation", target_mismatches, 0, target_mismatches == 0, "Wage, duration, commitment, and wage/value targets reproduce exactly.")
    age_expected = pd.to_numeric(source_lookup["age_at_signing"], errors="coerce") + duration
    age_mismatches = int((~np.isclose(pd.to_numeric(matrix["age_at_expiration"], errors="coerce"), age_expected, equal_nan=True)).sum())
    add(checks, "age_at_expiration_formula", age_mismatches, 0, age_mismatches == 0, "Age at expiration equals signing age plus exact duration.")

    expected_bonus = as_bool(source_lookup["bonus_available"])
    bonus_map = source_lookup.assign(_year=pd.to_datetime(source_lookup["signed_date"], errors="coerce").dt.year, _available=expected_bonus).groupby("_year").agg(records=("canonical_player_id", "size"), bonus_available=("_available", "sum")).reset_index()
    bonus_merged = bonus.merge(bonus_map, left_on="signed_year", right_on="_year", suffixes=("_saved", "_expected"))
    bonus_ok = len(bonus_merged) == len(bonus_map) and (bonus_merged["records_saved"] == bonus_merged["records_expected"]).all() and (bonus_merged["bonus_available_saved"] == bonus_merged["bonus_available_expected"]).all()
    add(checks, "bonus_coverage_recomputation", int(bonus_ok), 1, bonus_ok, "Bonus availability by signing year reproduces; missing bonuses remain missing.")

    predictors = set(features.loc[features["target_leakage_role"].eq("predictor"), "feature"])
    prohibited = set(features.loc[features["target_leakage_role"].eq("prohibited"), "feature"])
    feature_ok = not predictors.intersection(prohibited) and features.loc[features["target_leakage_role"].eq("predictor"), "timing"].eq("known_before_current_extension_terms").all()
    add(checks, "feature_timing_and_leakage", int(feature_ok), 1, feature_ok, "Current extension wage, term, and their derivatives are excluded from predictors.")

    prediction_key = ["endpoint", "origin_id", "model_variant", "capology_extension_event_id"]
    add(checks, "prediction_grain", int(predictions.duplicated(prediction_key).sum()), 0, not predictions.duplicated(prediction_key).any(), "One prediction per endpoint/origin/model/event.")
    add(checks, "prediction_row_count", len(predictions), 36264, len(predictions) == 36264, "Prediction rows match four endpoints, four origins, and six variants.")
    interval_ok = predictions["interval_lower"].le(predictions["prediction"]).all() and predictions["prediction"].le(predictions["interval_upper"]).all()
    interval_flag = ((predictions["actual"] >= predictions["interval_lower"]) & (predictions["actual"] <= predictions["interval_upper"])) == as_bool(predictions["within_80_interval"])
    add(checks, "interval_recomputation", int(interval_ok and interval_flag.all()), 1, interval_ok and interval_flag.all(), "Interval ordering and row-level coverage flags reproduce.")

    matrix_lookup = matrix.set_index("capology_extension_event_id")
    actual_mismatches = 0
    for endpoint, (_, target) in ENDPOINTS.items():
        subset = predictions[predictions["endpoint"].eq(endpoint)]
        expected = pd.to_numeric(matrix_lookup.loc[subset["capology_extension_event_id"], target], errors="coerce").to_numpy()
        actual_mismatches += int((~np.isclose(subset["actual"], expected, equal_nan=True)).sum())
    add(checks, "prediction_actual_values", actual_mismatches, 0, actual_mismatches == 0, "Saved actuals match independently reconstructed targets.")

    metric_key = ["endpoint", "origin_id", "model_variant"]
    add(checks, "origin_metric_grain", int(origin_metrics.duplicated(metric_key).sum()), 0, not origin_metrics.duplicated(metric_key).any(), "One metric row per endpoint/origin/model.")
    add(checks, "origin_metric_row_count", len(origin_metrics), 96, len(origin_metrics) == 4 * 4 * 6, "Four endpoints by four origins by six variants.")
    metric_mismatches = 0
    for key, group in predictions.groupby(metric_key, observed=True):
        saved = origin_metrics.loc[origin_metrics["endpoint"].eq(key[0]) & origin_metrics["origin_id"].eq(key[1]) & origin_metrics["model_variant"].eq(key[2])].iloc[0]
        recalculated = metrics(group["actual"].to_numpy(), group["prediction"].to_numpy())
        recalculated["evaluation_interval_80_coverage"] = float(as_bool(group["within_80_interval"]).mean())
        recalculated["evaluation_interval_80_mean_width"] = float((group["interval_upper"] - group["interval_lower"]).mean())
        metric_mismatches += sum(not close(saved[name], value) for name, value in recalculated.items())
    add(checks, "origin_metric_recomputation", metric_mismatches, 0, metric_mismatches == 0, "Evaluation losses, ranks, and interval metrics reproduce from row-level predictions.")

    add(checks, "comparison_row_count", len(comparisons), 108, len(comparisons) == 4 * 3 * 9, "All endpoint/window/comparison combinations exist.")
    comparison_mismatches = 0
    for row in comparisons.itertuples(index=False):
        years = WINDOWS[row.window]
        subset = predictions[predictions["endpoint"].eq(row.endpoint) & predictions["evaluation_year"].isin(years)]
        keys = ["origin_id", "evaluation_year", "capology_extension_event_id", "canonical_player_id"]
        left = subset[subset["model_variant"].eq(row.baseline)][keys + ["primary_loss"]].rename(columns={"primary_loss": "baseline_loss"})
        right = subset[subset["model_variant"].eq(row.candidate)][keys + ["primary_loss"]].rename(columns={"primary_loss": "candidate_loss"})
        pair = left.merge(right, on=keys, validate="one_to_one")
        delta = pair["baseline_loss"] - pair["candidate_loss"]
        wins = int(pair.assign(delta=delta).groupby("evaluation_year")["delta"].mean().gt(0).sum())
        comparison_mismatches += int(len(pair) != row.evaluation_rows) + int(not close(delta.mean(), row.mean_mae_improvement)) + int(wins != row.origin_wins)
    add(checks, "comparison_recomputation", comparison_mismatches, 0, comparison_mismatches == 0, "Paired MAE improvements, row counts, and origin wins reproduce.")

    add(checks, "benchmark_band_row_count", len(bands), 20, len(bands) == 4 * 5, "Five selected-model bands exist per endpoint.")
    band_mismatches = 0
    for endpoint, group in bands.groupby("endpoint", observed=True):
        selected = SELECTED_MODELS[endpoint]
        source_rows = predictions[predictions["endpoint"].eq(endpoint) & predictions["model_variant"].eq(selected)].copy()
        source_rows["benchmark_band"] = pd.qcut(source_rows["prediction"].rank(method="first"), 5, labels=False) + 1
        rebuilt = source_rows.groupby("benchmark_band").agg(records=("actual", "size"), observed_mean=("actual", "mean")).reset_index()
        saved = group.sort_values("benchmark_band")
        monotone = bool(rebuilt["observed_mean"].is_monotonic_increasing)
        band_mismatches += int(not saved["selected_model"].eq(selected).all())
        band_mismatches += int(not np.array_equal(saved["records"].to_numpy(), rebuilt["records"].to_numpy()))
        band_mismatches += int(not np.allclose(saved["observed_mean"], rebuilt["observed_mean"]))
        band_mismatches += int(as_bool(saved["observed_monotone"]).nunique() != 1 or as_bool(saved["observed_monotone"]).iat[0] != monotone)
    add(checks, "benchmark_band_recomputation", band_mismatches, 0, band_mismatches == 0, "Selected-model quintiles and monotonicity reproduce.")

    decision_map = dict(zip(decisions["endpoint"], decisions["status"]))
    selected_map = dict(zip(decisions["endpoint"], decisions["selected_model"]))
    add(checks, "decision_results", decision_map, EXPECTED_DECISIONS, decision_map == EXPECTED_DECISIONS, "Every benchmark satisfies the declared support gate.")
    add(checks, "selected_models", selected_map, SELECTED_MODELS, selected_map == SELECTED_MODELS, "Endpoint-specific parsimonious/nonlinear selections match the evidence.")
    trace_mismatches = 0
    for row in decisions.itertuples(index=False):
        total_name = "B5_vs_B0_nonlinear_total" if row.selected_model == "B5_compact_nonlinear" else "B2_vs_B0_parsimonious_total"
        total = comparisons[comparisons["endpoint"].eq(row.endpoint) & comparisons["window"].eq("pooled_four") & comparisons["comparison"].eq(total_name)].iloc[0]
        selected_metrics = origin_metrics[origin_metrics["endpoint"].eq(row.endpoint) & origin_metrics["model_variant"].eq(row.selected_model)]
        weight = selected_metrics["evaluation_rows"]
        coverage = np.average(selected_metrics["evaluation_interval_80_coverage"], weights=weight)
        trace_mismatches += int(not close(row.total_mae_improvement_vs_median, total.mean_mae_improvement))
        trace_mismatches += int(not close(row.total_ci_lower_95, total.cluster_ci_lower_95))
        trace_mismatches += int(not close(row.empirical_80_interval_coverage, coverage))
        supported = total.cluster_ci_lower_95 > 0 and total.origin_wins >= 3 and as_bool(bands[bands["endpoint"].eq(row.endpoint)]["observed_monotone"]).iat[0] and .70 <= coverage <= .90
        trace_mismatches += int(supported != (row.status == "advance_historical_peer_benchmark"))
    add(checks, "decision_traceability", trace_mismatches, 0, trace_mismatches == 0, "Selected metrics, uncertainty, stability, and decision gates trace to saved evidence.")

    required_metrics = {"age_at_expiration", "fixed_wage_commitment", "wage_to_market_value", "estimated_acquisition_commitment", "transfer_premium", "annual_wage_peer_premium", "contract_duration_peer_delta", "fixed_commitment_peer_premium"}
    add(checks, "metric_catalog_scope", len(required_metrics - set(catalog["metric"])), 0, required_metrics.issubset(set(catalog["metric"])), "Required deterministic and peer-comparison outputs are documented.")
    required_limits = {"employer_taxes", "agent_and_intermediary_fees", "signing_on_fees", "release_clauses", "club_extension_options", "performance_based_pay"}
    add(checks, "limitation_catalog_scope", len(required_limits - set(limitations["item"])), 0, required_limits.issubset(set(limitations["item"])), "Unobserved acquisition-cost and contract-term fields are explicit.")

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
        "checks_passed": int(result["passed"].sum()), "checks_total": len(result),
        "all_checks_passed": bool(result["passed"].all()), "source_extension_events": len(source),
        "wage_cohort": 2265, "duration_cohort": 2358, "commitment_cohort": 2265,
        "wage_value_cohort": 2222, "prediction_rows_verified": len(predictions),
        "decisions_verified": decision_map, "selected_models_verified": selected_map,
    }
    (OUTPUT / "independent_verification.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    files = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name != "output_manifest.csv")
    manifest = pd.DataFrame([{"output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files])
    manifest.to_csv(OUTPUT / "output_manifest.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    if not result["passed"].all():
        failed = result.loc[~result["passed"], "check"].tolist()
        raise RuntimeError(f"Independent Phase-4 verification failed: {failed}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

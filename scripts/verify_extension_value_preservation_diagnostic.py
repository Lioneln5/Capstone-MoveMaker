"""Independently verify the Phase-3 extension value diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, mean_absolute_error,
    mean_squared_error, r2_score, roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Data" / "processed" / "extension_value_preservation_diagnostic"
FROZEN_SOURCE = ROOT / "Data" / "processed" / "frozen_contract_sources" / "phase1_5_2026-08-11" / "contract_extension_integration"
SOURCE = FROZEN_SOURCE / "extension_modeling_master.csv"
UPSTREAM_VERIFY = FROZEN_SOURCE / "independent_verification.json"
ENDPOINTS = {
    "value_log_ratio_12m": ("continuous", "12m", "target_log_value_ratio_12m"),
    "downside_10pct_12m": ("binary", "12m", "target_downside_10pct_12m"),
    "downside_25pct_12m": ("binary", "12m", "target_downside_25pct_12m"),
    "downside_50pct_12m": ("binary", "12m", "target_downside_50pct_12m"),
    "value_log_ratio_24m": ("continuous", "24m", "target_log_value_ratio_24m"),
    "downside_10pct_24m": ("binary", "24m", "target_downside_10pct_24m"),
    "downside_25pct_24m": ("binary", "24m", "target_downside_25pct_24m"),
    "downside_50pct_24m": ("binary", "24m", "target_downside_50pct_24m"),
}
EXPECTED_DECISIONS = {
    "value_log_ratio_12m": "advance_broad_value_range_candidate",
    "downside_10pct_12m": "advance_downside_probability_candidate",
    "downside_25pct_12m": "advance_downside_probability_candidate",
    "downside_50pct_12m": "descriptive_or_exploratory_only",
    "value_log_ratio_24m": "advance_broad_value_range_candidate",
    "downside_10pct_24m": "advance_downside_probability_candidate",
    "downside_25pct_24m": "advance_downside_probability_candidate",
    "downside_50pct_24m": "advance_downside_probability_candidate",
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
        "evaluation_median_absolute_error": np.median(np.abs(actual - prediction)),
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


def close(left: object, right: object, tolerance: float = 1e-10) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return bool(np.isclose(float(left), float(right), atol=tolerance, rtol=tolerance))


def main() -> None:
    required = {
        "value_preservation_model_matrix.csv", "value_target_audit.csv", "valuation_timing_audit.csv",
        "feature_manifest.csv", "model_origin_metrics.csv", "value_predictions.csv",
        "model_comparison_results.csv", "value_risk_bands.csv", "probability_ordering_audit.csv",
        "subgroup_stability.csv", "value_decision_summary.csv", "build_checks.csv",
        "source_manifest.csv", "output_manifest.csv", "run_summary.json", "README.md",
    }
    missing = sorted(name for name in required if not (OUTPUT / name).exists())
    if missing:
        raise RuntimeError(f"Missing value diagnostic artifacts: {missing}")
    checks: list[dict] = []
    add(checks, "required_outputs", len(missing), 0, not missing, "All declared artifacts exist.")
    upstream = json.loads(UPSTREAM_VERIFY.read_text(encoding="utf-8"))
    add(checks, "upstream_verification", upstream.get("all_checks_passed"), True, bool(upstream.get("all_checks_passed")), "The contract integration passed independent verification.")

    master = pd.read_csv(SOURCE, low_memory=False)
    matrix = pd.read_csv(OUTPUT / "value_preservation_model_matrix.csv", low_memory=False)
    predictions = pd.read_csv(OUTPUT / "value_predictions.csv", low_memory=False)
    origin_metrics = pd.read_csv(OUTPUT / "model_origin_metrics.csv", low_memory=False)
    comparisons = pd.read_csv(OUTPUT / "model_comparison_results.csv", low_memory=False)
    bands = pd.read_csv(OUTPUT / "value_risk_bands.csv", low_memory=False)
    ordering = pd.read_csv(OUTPUT / "probability_ordering_audit.csv", low_memory=False)
    decisions = pd.read_csv(OUTPUT / "value_decision_summary.csv", low_memory=False)
    timing = pd.read_csv(OUTPUT / "valuation_timing_audit.csv", low_memory=False)
    features = pd.read_csv(OUTPUT / "feature_manifest.csv", low_memory=False)
    build_checks = pd.read_csv(OUTPUT / "build_checks.csv", low_memory=False)

    add(checks, "source_row_count", len(master), 2425, len(master) == 2425, "Clean extension source is unchanged.")
    add(checks, "matrix_event_grain", int(matrix["capology_extension_event_id"].duplicated().sum()), 0, not matrix["capology_extension_event_id"].duplicated().any(), "One matrix row per extension event.")
    add(checks, "matrix_source_ids", len(set(matrix["capology_extension_event_id"])), len(set(master["capology_extension_event_id"])), set(matrix["capology_extension_event_id"]) == set(master["capology_extension_event_id"]), "Matrix preserves the exact source ID set.")

    at_value = pd.to_numeric(master["at_signing_market_value_eur"], errors="coerce")
    linked = master["canonical_player_id"].notna()
    base = linked & as_bool(master["at_signing_strict_timing_eligible"]) & at_value.gt(0)
    add(checks, "base_cohort_recomputation", int(base.sum()), 2285, int(base.sum()) == 2285, "Linked strict positive signing valuations are reproduced.")
    lookup = matrix.set_index("capology_extension_event_id")
    source_ids = master["capology_extension_event_id"]
    target_mismatches = 0
    expected_counts = {12: 2162, 24: 1944}
    for horizon in [12, 24]:
        post = pd.to_numeric(master[f"post{horizon}_market_value_eur"], errors="coerce")
        cohort = base & as_bool(master[f"post{horizon}_strict_timing_eligible"]) & post.gt(0)
        add(checks, f"cohort_{horizon}m_recomputation", int(cohort.sum()), expected_counts[horizon], int(cohort.sum()) == expected_counts[horizon], "Strict positive post-extension valuation cohort is reproduced.")
        ratio = post.div(at_value.where(at_value.gt(0)))
        observed = lookup.loc[source_ids]
        target_mismatches += int((as_bool(observed[f"value_cohort_{horizon}m"]).to_numpy() != cohort.to_numpy()).sum())
        target_mismatches += int((~np.isclose(pd.to_numeric(observed[f"target_value_ratio_{horizon}m"], errors="coerce"), ratio, equal_nan=True)).sum())
        target_mismatches += int((~np.isclose(pd.to_numeric(observed[f"target_log_value_ratio_{horizon}m"], errors="coerce"), np.log(ratio.where(ratio.gt(0))), equal_nan=True)).sum())
        for threshold in [10, 25, 50]:
            boundary = np.nextafter(at_value * (1 - threshold / 100), np.inf)
            target = post.le(boundary).astype(int)
            target_mismatches += int((pd.to_numeric(observed[f"target_downside_{threshold}pct_{horizon}m"], errors="coerce").to_numpy() != target.to_numpy()).sum())
    add(checks, "target_recomputation", target_mismatches, 0, target_mismatches == 0, "Cohorts, ratios, log ratios, and ordered downside targets reproduce from source values.")

    nested = True
    for horizon in [12, 24]:
        eligible = as_bool(matrix[f"value_cohort_{horizon}m"])
        nested &= matrix.loc[eligible, f"target_downside_50pct_{horizon}m"].le(matrix.loc[eligible, f"target_downside_25pct_{horizon}m"]).all()
        nested &= matrix.loc[eligible, f"target_downside_25pct_{horizon}m"].le(matrix.loc[eligible, f"target_downside_10pct_{horizon}m"]).all()
    add(checks, "target_ordering", int(nested), 1, nested, "Observed downside targets are correctly nested.")
    timing_ok = timing["at_signing_max_abs_gap_days"].le(timing["at_signing_tolerance_days"]).all() and timing["post_max_abs_gap_days"].le(timing["post_tolerance_days"]).all()
    add(checks, "valuation_timing", int(timing_ok), 1, timing_ok, "Every eligible landmark satisfies the declared timing tolerance.")
    feature_safe = features["timing"].eq("known_at_extension_signing").all() and not features["feature"].str.startswith("post").any()
    add(checks, "feature_timing", int(feature_safe), 1, feature_safe, "No future valuation or post-signing performance feature is modeled.")

    prediction_key = ["endpoint", "origin_id", "model_variant", "capology_extension_event_id"]
    add(checks, "prediction_grain", int(predictions.duplicated(prediction_key).sum()), 0, not predictions.duplicated(prediction_key).any(), "One prediction per endpoint/origin/model/event.")
    add(checks, "prediction_row_count", len(predictions), 78372, len(predictions) == 78372, "Prediction count matches fixed cohorts, origins, endpoints, and models.")
    binary = predictions["kind"].eq("binary")
    bounds = predictions.loc[binary, "prediction"].between(0, 1).all()
    add(checks, "probability_bounds", int(bounds), 1, bounds, "All downside forecasts are probabilities.")
    preservation = predictions["endpoint"].str.startswith("downside_10pct")
    complements = np.allclose(predictions.loc[preservation, "value_preservation_probability"] + predictions.loc[preservation, "prediction"], 1.0)
    add(checks, "preservation_probability_complements", int(complements), 1, complements, "Value-preservation probability is exactly one minus 10% downside probability.")

    target_lookup = lookup
    actual_mismatches = 0
    for endpoint, (_, _, target) in ENDPOINTS.items():
        subset = predictions[predictions["endpoint"].eq(endpoint)]
        expected = pd.to_numeric(target_lookup.loc[subset["capology_extension_event_id"], target], errors="coerce").to_numpy()
        actual_mismatches += int((~np.isclose(subset["actual"].to_numpy(), expected, equal_nan=True)).sum())
    add(checks, "prediction_actual_values", actual_mismatches, 0, actual_mismatches == 0, "Saved actuals match independently reconstructed event targets.")

    metric_key = ["endpoint", "origin_id", "model_variant"]
    add(checks, "origin_metric_grain", int(origin_metrics.duplicated(metric_key).sum()), 0, not origin_metrics.duplicated(metric_key).any(), "One metric row per endpoint/origin/model.")
    add(checks, "origin_metric_row_count", len(origin_metrics), 224, len(origin_metrics) == 8 * 4 * 7, "Eight endpoints by four origins by seven model variants.")
    metric_mismatches = 0
    for key, group in predictions.groupby(metric_key, observed=True):
        saved = origin_metrics.loc[
            origin_metrics["endpoint"].eq(key[0]) & origin_metrics["origin_id"].eq(key[1]) & origin_metrics["model_variant"].eq(key[2])
        ].iloc[0]
        actual, estimate = group["actual"].to_numpy(), group["prediction"].to_numpy()
        recalculated = continuous_metrics(actual, estimate) if group["kind"].iat[0] == "continuous" else binary_metrics(actual.astype(int), estimate)
        metric_mismatches += sum(not close(saved[name], value) for name, value in recalculated.items())
    add(checks, "origin_metric_recomputation", metric_mismatches, 0, metric_mismatches == 0, "All evaluation metrics reproduce from row-level predictions.")

    add(checks, "comparison_row_count", len(comparisons), 192, len(comparisons) == 8 * 3 * 8, "All endpoint/window/comparison combinations exist.")
    comparison_mismatches = 0
    for row in comparisons.itertuples(index=False):
        years = [int(value.strip()) for value in str(row.evaluation_years).split("|")]
        subset = predictions[predictions["endpoint"].eq(row.endpoint) & predictions["evaluation_year"].isin(years)]
        keys = ["origin_id", "evaluation_year", "capology_extension_event_id", "canonical_player_id"]
        base_rows = subset[subset["model_variant"].eq(row.baseline)][keys + ["primary_loss"]].rename(columns={"primary_loss": "base"})
        candidate_rows = subset[subset["model_variant"].eq(row.candidate)][keys + ["primary_loss"]].rename(columns={"primary_loss": "candidate"})
        pair = base_rows.merge(candidate_rows, on=keys, validate="one_to_one")
        delta = pair["base"] - pair["candidate"]
        wins = int(pair.assign(delta=delta).groupby("evaluation_year")["delta"].mean().gt(0).sum())
        comparison_mismatches += int(len(pair) != row.evaluation_rows)
        comparison_mismatches += int(not np.isclose(delta.mean(), row.mean_loss_improvement, atol=1e-12, rtol=1e-12))
        comparison_mismatches += int(wins != row.origin_wins)
    add(checks, "comparison_recomputation", comparison_mismatches, 0, comparison_mismatches == 0, "Paired improvements, row counts, and origin wins reproduce.")

    add(checks, "risk_band_row_count", len(bands), 40, len(bands) == 8 * 5, "Five M5 bands exist for every endpoint.")
    band_mismatches = 0
    for _, group in bands.groupby("endpoint", observed=True):
        monotone = bool(group.sort_values("risk_band")["observed_outcome"].is_monotonic_increasing)
        band_mismatches += int(as_bool(group["observed_monotone"]).nunique() != 1)
        band_mismatches += int(as_bool(group["observed_monotone"]).iat[0] != monotone)
    add(checks, "risk_band_flags", band_mismatches, 0, band_mismatches == 0, "Monotonicity flags match observed band outcomes.")

    add(checks, "ordering_audit_rows", len(ordering), 4, len(ordering) == 4, "M5 and M6 ordering are audited at both horizons.")
    ordering_mismatches = 0
    for row in ordering.itertuples(index=False):
        subset = predictions[predictions["horizon"].eq(row.horizon) & predictions["kind"].eq("binary") & predictions["model_variant"].eq(row.model_variant)]
        wide = subset.pivot(index=["origin_id", "evaluation_year", "capology_extension_event_id"], columns="endpoint", values="prediction")
        p10, p25, p50 = wide[f"downside_10pct_{row.horizon}"], wide[f"downside_25pct_{row.horizon}"], wide[f"downside_50pct_{row.horizon}"]
        ordering_mismatches += int(int((p25 > p10).sum()) != row.down25_gt_down10_violations)
        ordering_mismatches += int(int((p50 > p25).sum()) != row.down50_gt_down25_violations)
        ordering_mismatches += int(int(((p25 > p10) | (p50 > p25)).sum()) != row.any_order_violation_rows)
    add(checks, "ordering_recomputation", ordering_mismatches, 0, ordering_mismatches == 0, "Every raw threshold-order violation is reproduced from predictions.")

    decision_map = dict(zip(decisions["endpoint"], decisions["status"]))
    add(checks, "decision_set", len(decision_map), 8, set(decision_map) == set(EXPECTED_DECISIONS), "All four endpoints are adjudicated at 12 and 24 months.")
    add(checks, "decision_results", decision_map, EXPECTED_DECISIONS, decision_map == EXPECTED_DECISIONS, "Statuses match the pre-registered evidence gates.")
    trace_mismatches = 0
    for row in decisions.itertuples(index=False):
        total = comparisons[comparisons["endpoint"].eq(row.endpoint) & comparisons["window"].eq("pooled_four") & comparisons["comparison"].eq("M5_vs_M0_total")].iloc[0]
        contract = comparisons[comparisons["endpoint"].eq(row.endpoint) & comparisons["window"].eq("pooled_four") & comparisons["comparison"].eq("M5_vs_M3_all_contract_economics")].iloc[0]
        trace_mismatches += int(not np.isclose(row.total_loss_improvement_vs_baseline, total.mean_loss_improvement))
        trace_mismatches += int(not np.isclose(row.contract_incremental_improvement, contract.mean_loss_improvement))
    add(checks, "decision_traceability", trace_mismatches, 0, trace_mismatches == 0, "Decision metrics trace to pooled comparisons.")

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
        "all_checks_passed": bool(result["passed"].all()),
        "source_extension_events": len(master), "value_12m_cohort": 2162, "value_24m_cohort": 1944,
        "prediction_rows_verified": len(predictions), "decisions_verified": decision_map,
    }
    (OUTPUT / "independent_verification.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    files = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name != "output_manifest.csv")
    manifest = pd.DataFrame([{"output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files])
    manifest.to_csv(OUTPUT / "output_manifest.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    if not result["passed"].all():
        failed = result.loc[~result["passed"], "check"].tolist()
        raise RuntimeError(f"Independent value verification failed: {failed}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

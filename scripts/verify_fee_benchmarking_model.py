"""Independently verify the reported-fee benchmarking model artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "Data" / "processed" / "financial_module_feasibility" / "financial_event_audit.csv"
OUTPUT_DIR = ROOT / "Data" / "processed" / "fee_benchmarking_model"
M1 = "M1_market_profile"
POLICY = "M_policy_validation_selected"
ANCHOR = "A1_pre_transfer_value_anchor"
MEDIAN = "M0_chronological_median"
MODEL_ORDER = [
    M1, "M2_plus_valuation_trajectory", "M3_plus_prior_opportunity",
    "M4_plus_player_performance", "M5_plus_compact_club_context",
]
EXPECTED_EVALUATION_COUNTS = {2020: 152, 2021: 176, 2022: 205, 2023: 205}


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().str.lower().isin({"true", "1", "yes"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left: Any, right: Any, atol: float = 1e-10) -> bool:
    return bool(np.isclose(float(left), float(right), atol=atol, rtol=0, equal_nan=True))


def add(
    checks: list[dict[str, Any]], check: str, passed: bool, observed: Any,
    expected: Any, note: str,
) -> None:
    checks.append({
        "check": check, "passed": bool(passed), "observed": observed,
        "expected": expected, "note": note,
    })


def verify() -> None:
    checks: list[dict[str, Any]] = []
    required = [
        "fee_benchmark_targets.csv", "fee_benchmark_feature_manifest.csv",
        "raw_model_tuning_results.csv", "raw_model_selections.csv",
        "policy_model_selections.csv", "fee_benchmark_predictions.csv",
        "fee_benchmark_origin_results.csv", "fee_benchmark_comparison_results.csv",
        "fee_interval_coverage.csv", "historical_fee_benchmark_cards.csv",
        "fee_benchmark_subgroup_stability.csv", "fee_benchmark_decision_summary.csv",
        "build_checks.csv", "source_manifest.csv", "README.md", "run_summary.json",
        "output_manifest.csv",
    ]
    missing = [name for name in required if not (OUTPUT_DIR / name).exists()]
    add(checks, "required_outputs_exist", not missing, ";".join(missing), "none", "All promised artifacts exist.")
    if missing:
        raise RuntimeError(f"Missing outputs: {missing}")

    audit = pd.read_csv(AUDIT_PATH, low_memory=False)
    targets = pd.read_csv(OUTPUT_DIR / "fee_benchmark_targets.csv", low_memory=False)
    features = pd.read_csv(OUTPUT_DIR / "fee_benchmark_feature_manifest.csv")
    tuning = pd.read_csv(OUTPUT_DIR / "raw_model_tuning_results.csv")
    raw_selections = pd.read_csv(OUTPUT_DIR / "raw_model_selections.csv")
    policy_selections = pd.read_csv(OUTPUT_DIR / "policy_model_selections.csv")
    predictions = pd.read_csv(OUTPUT_DIR / "fee_benchmark_predictions.csv", low_memory=False)
    origins = pd.read_csv(OUTPUT_DIR / "fee_benchmark_origin_results.csv")
    comparisons = pd.read_csv(OUTPUT_DIR / "fee_benchmark_comparison_results.csv")
    intervals = pd.read_csv(OUTPUT_DIR / "fee_interval_coverage.csv")
    cards = pd.read_csv(OUTPUT_DIR / "historical_fee_benchmark_cards.csv")
    subgroups = pd.read_csv(OUTPUT_DIR / "fee_benchmark_subgroup_stability.csv")
    decision = pd.read_csv(OUTPUT_DIR / "fee_benchmark_decision_summary.csv")
    build_checks = pd.read_csv(OUTPUT_DIR / "build_checks.csv")
    sources = pd.read_csv(OUTPUT_DIR / "source_manifest.csv")
    manifest = pd.read_csv(OUTPUT_DIR / "output_manifest.csv")
    summary = json.loads((OUTPUT_DIR / "run_summary.json").read_text(encoding="utf-8"))

    eligible = audit.loc[as_bool(audit["eligible_fee_benchmarking"])].copy()
    eligible["transfer_event_id"] = eligible["transfer_event_id"].astype(str)
    targets["transfer_event_id"] = targets["transfer_event_id"].astype(str)
    add(checks, "audit_cohort_count", len(eligible) == 1486, len(eligible), 1486, "Frozen audit cohort independently reproduced.")
    add(checks, "target_cohort_count", len(targets) == 1486, len(targets), 1486, "Every audited deal retained.")
    add(checks, "target_key_unique", targets["transfer_event_id"].is_unique, targets["transfer_event_id"].nunique(), len(targets), "One target row per deal.")
    add(checks, "cohort_ids_exact", set(targets["transfer_event_id"]) == set(eligible["transfer_event_id"]), len(set(targets["transfer_event_id"]) ^ set(eligible["transfer_event_id"])), 0, "No target IDs added or lost.")
    matched = targets.merge(
        eligible[["transfer_event_id", "canonical_transfer_fee_eur", "valuation_pre_eur"]],
        on="transfer_event_id", suffixes=("_stored", "_audit"), validate="one_to_one",
    )
    fee_exact = np.allclose(
        matched["canonical_transfer_fee_eur_stored"], matched["canonical_transfer_fee_eur_audit"],
        atol=1e-12, rtol=0,
    )
    log_exact = np.allclose(
        matched["target_log_fee_eur"], np.log(matched["canonical_transfer_fee_eur_audit"]),
        atol=1e-12, rtol=0,
    )
    add(checks, "fee_target_exact", fee_exact and log_exact, fee_exact and log_exact, True, "Reported fee and log target exactly recomputed.")
    add(checks, "positive_fee_and_value", matched["canonical_transfer_fee_eur_audit"].gt(0).all() and matched["valuation_pre_eur_audit"].gt(0).all(), f"{matched['canonical_transfer_fee_eur_audit'].min()};{matched['valuation_pre_eur_audit'].min()}", ">0;>0", "Cohort semantics preserved.")

    feature_counts = features.groupby("model_variant")["raw_feature_count"].max().astype(int).to_dict()
    expected_counts = dict(zip(MODEL_ORDER, [10, 22, 30, 46, 72]))
    add(checks, "feature_counts", feature_counts == expected_counts, json.dumps(feature_counts, sort_keys=True), json.dumps(expected_counts, sort_keys=True), "Frozen feature counts exact.")
    feature_sets = {
        model: features.loc[features["model_variant"].eq(model)].sort_values("feature_order")["feature"].tolist()
        for model in MODEL_ORDER
    }
    nested = all(set(feature_sets[MODEL_ORDER[index - 1]]).issubset(feature_sets[MODEL_ORDER[index]]) for index in range(1, len(MODEL_ORDER)))
    add(checks, "feature_sets_nested", nested, nested, True, "Every block only adds predictors.")
    forbidden = [feature for feature in features["feature"].astype(str) if any(token in feature.lower() for token in ["canonical_transfer_fee", "log_fee", "fee_band", "post12", "post24", "target_", "eligible_"])]
    add(checks, "no_fee_or_outcome_predictors", not forbidden, ";".join(forbidden), "none", "Target and downstream fields excluded.")
    add(checks, "feature_timing_complete", features["timing_classification"].notna().all(), features["timing_classification"].notna().sum(), len(features), "Every feature timing classified.")

    add(checks, "tuning_rows", len(tuning) == 200, len(tuning), 200, "Ten candidates per model/origin.")
    add(checks, "raw_selection_rows", len(raw_selections) == 20, len(raw_selections), 20, "Five learned blocks by four origins.")
    temporal = (
        (raw_selections["internal_train_end_year"] < raw_selections["internal_selector_year"]).all()
        and (raw_selections["internal_selector_year"] < raw_selections["validation_year"]).all()
        and (raw_selections["validation_year"] < raw_selections["evaluation_year"]).all()
    )
    add(checks, "nested_temporal_order", temporal, temporal, True, "Model selection, outer validation, evaluation ordered.")
    selected_tuning = tuning.sort_values([
        "origin_id", "model_variant", "internal_selector_mae_log",
        "internal_selector_rmse_log", "model_family", "hyperparameters",
    ]).groupby(["origin_id", "model_variant"], as_index=False).first()
    raw_match = raw_selections.merge(
        selected_tuning[["origin_id", "model_variant", "model_family", "hyperparameters"]],
        on=["origin_id", "model_variant"], validate="one_to_one",
    )
    selection_exact = raw_match["selected_family"].eq(raw_match["model_family"]).all() and raw_match["selected_hyperparameters"].eq(raw_match["hyperparameters"]).all()
    add(checks, "raw_model_selection_exact", selection_exact, selection_exact, True, "Internal minimum family/hyperparameters reproduced.")
    add(checks, "policy_selection_rows", len(policy_selections) == 24, len(policy_selections), 24, "Six block/anchor candidates by four origins.")
    selected_policy = policy_selections.loc[as_bool(policy_selections["selected_for_policy"])]
    add(checks, "one_policy_per_origin", len(selected_policy) == 4 and selected_policy.groupby("origin_id").size().eq(1).all(), len(selected_policy), 4, "Outer validation selection unique.")
    policy_min = policy_selections.sort_values([
        "origin_id", "validation_mae_log", "validation_rmse_log", "candidate_model",
    ]).groupby("origin_id", as_index=False).first()
    policy_match = selected_policy.merge(
        policy_min[["origin_id", "candidate_model"]], on="origin_id",
        suffixes=("_stored", "_minimum"), validate="one_to_one",
    )
    add(checks, "policy_selection_exact", policy_match["candidate_model_stored"].eq(policy_match["candidate_model_minimum"]).all(), policy_match[["candidate_model_stored", "candidate_model_minimum"]].to_dict("records"), "equal", "Validation MAE minimum reproduced.")

    prediction_key = ["split", "origin_id", "model_variant", "transfer_event_id"]
    add(checks, "prediction_key_unique", not predictions.duplicated(prediction_key).any(), predictions.duplicated(prediction_key).sum(), 0, "Prediction grain unique.")
    add(checks, "prediction_splits", set(predictions["split"]) == {"validation", "evaluation"}, ";".join(sorted(predictions["split"].unique())), "evaluation;validation", "Both forecast layers retained.")
    m1_eval = predictions.loc[predictions["split"].eq("evaluation") & predictions["model_variant"].eq(M1)]
    counts = m1_eval.groupby("evaluation_year").size().to_dict()
    add(checks, "evaluation_counts", counts == EXPECTED_EVALUATION_COUNTS, json.dumps(counts, sort_keys=True), json.dumps(EXPECTED_EVALUATION_COUNTS, sort_keys=True), "Four origin counts exact.")
    add(checks, "evaluation_rows", len(m1_eval) == 738, len(m1_eval), 738, "Complete benchmark evaluation.")
    add(checks, "positive_point_estimates", m1_eval["predicted_fee_eur"].gt(0).all(), m1_eval["predicted_fee_eur"].min(), ">0", "Log predictions back-transform positively.")
    actual_exact = np.allclose(m1_eval["actual_fee_eur"], np.exp(m1_eval["actual_log_fee_eur"]), atol=1e-6, rtol=1e-12)
    predicted_exact = np.allclose(m1_eval["predicted_fee_eur"], np.exp(m1_eval["predicted_log_fee_eur"]), atol=1e-6, rtol=1e-12)
    add(checks, "euro_back_transforms", actual_exact and predicted_exact, actual_exact and predicted_exact, True, "Log/euro fields reconcile.")

    origin_metrics_ok = True
    for row in origins.itertuples(index=False):
        group = predictions.loc[
            predictions["split"].eq("evaluation")
            & predictions["origin_id"].eq(row.origin_id)
            & predictions["model_variant"].eq(row.model_variant)
        ]
        mae = np.abs(group["actual_log_fee_eur"] - group["predicted_log_fee_eur"]).mean()
        origin_metrics_ok &= close(mae, row.mae_log)
    add(checks, "origin_mae_recomputed", origin_metrics_ok, origin_metrics_ok, True, "Every origin/model MAE-log exact.")

    comparison_ok = True
    for row in comparisons.loc[comparisons["window"].eq("pooled_four")].itertuples(index=False):
        keys = ["origin_id", "evaluation_year", "transfer_event_id", "player_id"]
        left = predictions.loc[
            predictions["split"].eq("evaluation") & predictions["model_variant"].eq(row.baseline_model),
            keys + ["actual_log_fee_eur", "predicted_log_fee_eur"],
        ].rename(columns={"predicted_log_fee_eur": "baseline"})
        right = predictions.loc[
            predictions["split"].eq("evaluation") & predictions["model_variant"].eq(row.candidate_model),
            keys + ["predicted_log_fee_eur"],
        ].rename(columns={"predicted_log_fee_eur": "candidate"})
        pair = left.merge(right, on=keys, validate="one_to_one")
        baseline_mae = np.abs(pair["actual_log_fee_eur"] - pair["baseline"]).mean()
        candidate_mae = np.abs(pair["actual_log_fee_eur"] - pair["candidate"]).mean()
        comparison_ok &= close(baseline_mae, row.baseline_mae_log) and close(candidate_mae, row.candidate_mae_log) and close(baseline_mae - candidate_mae, row.mae_log_improvement)
    add(checks, "pooled_comparisons_recomputed", comparison_ok, comparison_ok, True, "Every pooled comparison exact.")

    pooled = comparisons.loc[comparisons["window"].eq("pooled_four")].set_index("comparison")
    profile_anchor = pooled.loc["market_profile_vs_value_anchor"]
    add(checks, "compact_model_beats_value_anchor", profile_anchor["cluster_ci_lower_95"] > 0 and profile_anchor["origin_wins"] >= 3, f"{profile_anchor['mae_log_improvement']};{profile_anchor['cluster_ci_lower_95']};{profile_anchor['origin_wins']}", "positive CI; >=3 wins", "Commercial hurdle passed.")
    policy_profile = pooled.loc["policy_vs_market_profile"]
    add(checks, "richer_policy_not_overclaimed", policy_profile["cluster_ci_lower_95"] <= 0, policy_profile["cluster_ci_lower_95"], "<=0", "Richer increment unsupported.")

    interval_ok = True
    for row in intervals.itertuples(index=False):
        if row.window not in {"pooled_four", "mature_three", "terminal_two"}:
            group = m1_eval.loc[m1_eval["origin_id"].eq(row.window)]
        else:
            years = [int(value) for value in str(row.evaluation_years).split(";")]
            group = m1_eval.loc[m1_eval["evaluation_year"].isin(years)]
        lower = "interval80_lower_eur" if row.interval == "80pct" else "interval90_lower_eur"
        upper = "interval80_upper_eur" if row.interval == "80pct" else "interval90_upper_eur"
        coverage = group["actual_fee_eur"].between(group[lower], group[upper]).mean()
        ratio = (group[upper] / group[lower]).median()
        interval_ok &= close(coverage, row.observed_coverage) and close(ratio, row.median_upper_to_lower_ratio)
    add(checks, "interval_metrics_recomputed", interval_ok, interval_ok, True, "Coverage and multiplicative widths exact.")
    nested_intervals = (
        (m1_eval["interval90_lower_eur"] <= m1_eval["interval80_lower_eur"]).all()
        and (m1_eval["interval80_lower_eur"] <= m1_eval["predicted_fee_eur"]).all()
        and (m1_eval["predicted_fee_eur"] <= m1_eval["interval80_upper_eur"]).all()
        and (m1_eval["interval80_upper_eur"] <= m1_eval["interval90_upper_eur"]).all()
    )
    add(checks, "intervals_nested", nested_intervals, nested_intervals, True, "90% contains 80% contains point estimate.")
    pooled_intervals = intervals.loc[intervals["window"].eq("pooled_four")].set_index("interval")
    add(checks, "interval_coverage_valid_but_wide", pooled_intervals.loc["80pct", "observed_coverage"] >= 0.70 and pooled_intervals.loc["90pct", "observed_coverage"] >= 0.80 and pooled_intervals.loc["80pct", "median_upper_to_lower_ratio"] > 5 and pooled_intervals.loc["90pct", "median_upper_to_lower_ratio"] > 10, f"{pooled_intervals.loc['80pct','observed_coverage']};{pooled_intervals.loc['90pct','observed_coverage']};{pooled_intervals.loc['80pct','median_upper_to_lower_ratio']};{pooled_intervals.loc['90pct','median_upper_to_lower_ratio']}", "coverage adequate; widths broad", "Uncertainty limitation preserved.")

    add(checks, "card_rows", len(cards) == 738 and cards["transfer_event_id"].nunique() == 738, f"{len(cards)};{cards['transfer_event_id'].nunique()}", "738;738", "One historical card per evaluation deal.")
    card_ratio = cards["actual_fee_eur"] / cards["predicted_fee_eur"]
    add(checks, "card_premium_ratio_exact", np.allclose(card_ratio, cards["actual_to_benchmark_ratio"], atol=1e-12, rtol=0), True, True, "Historical premium diagnostic exact.")
    negative = subgroups.loc[
        as_bool(subgroups["adequate_for_inference"])
        & subgroups["stability_label"].eq("supported_negative")
    ]
    add(checks, "no_supported_negative_subgroups", negative.empty, len(negative), 0, "Compact benchmark not reliably worse in any adequate group.")

    row = decision.iloc[0]
    add(checks, "decision_status", row["status"] == "validated_fee_benchmark" and row["recommended_model_scope"] == M1, f"{row['status']};{row['recommended_model_scope']}", f"validated_fee_benchmark;{M1}", "Decision matches supported compact scope.")
    add(checks, "decision_range_guardrail", as_bool(pd.Series([row["interval_coverage_valid"]])).iloc[0] and not as_bool(pd.Series([row["interval_ranges_precise"]])).iloc[0], f"{row['interval_coverage_valid']};{row['interval_ranges_precise']}", "true;false", "Coverage and precision separated.")
    add(checks, "builder_checks_pass", as_bool(build_checks["passed"]).all(), build_checks["passed"].sum(), len(build_checks), "All builder checks pass.")

    source_ok = True
    for item in sources.itertuples(index=False):
        path = ROOT / item.source
        source_ok &= path.exists() and path.stat().st_size == item.size_bytes and sha256_file(path) == item.sha256
    add(checks, "source_manifest_hashes", source_ok, source_ok, True, "Inputs unchanged and hash-matched.")
    output_ok = True
    for item in manifest.itertuples(index=False):
        path = OUTPUT_DIR / item.file
        output_ok &= path.exists() and path.stat().st_size == item.size_bytes and sha256_file(path) == item.sha256
    add(checks, "output_manifest_hashes", output_ok, output_ok, True, "All modeled artifacts hash-match.")
    add(checks, "summary_counts", summary["cohort_rows"] == 1486 and summary["evaluation_rows"] == 738 and summary["decision"] == "validated_fee_benchmark", json.dumps({key: summary[key] for key in ["cohort_rows", "evaluation_rows", "decision"]}, sort_keys=True), "1486;738;validated", "Run summary reconciles.")
    add(checks, "protected_outputs_unchanged", summary["protected_outputs_unchanged"] is True, summary["protected_outputs_unchanged"], True, "Runner reports protected output identity.")

    result = pd.DataFrame(checks)
    result.to_csv(OUTPUT_DIR / "independent_verification.csv", index=False)
    payload = {
        "status": "pass" if result["passed"].all() else "fail",
        "checks": len(result), "passed": int(result["passed"].sum()),
        "failed": result.loc[~result["passed"], "check"].tolist(),
    }
    (OUTPUT_DIR / "independent_verification.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise RuntimeError(f"Independent verification failed: {payload['failed']}")


if __name__ == "__main__":
    verify()

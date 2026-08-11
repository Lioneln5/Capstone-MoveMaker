"""Independently verify under-utilization calibration and compact refinement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "Data" / "processed" / "paid_transfer_underutilization_model"
OUTPUT_DIR = ROOT / "Data" / "processed" / "paid_transfer_underutilization_refinement"
COMPACT_MODELS = [
    "C1_paid_deal_market_profile", "C2_plus_prior_opportunity",
    "C3_plus_compact_destination_context",
]
ENDPOINTS = ["opportunity_share_24m", "underutilization_10pct", "underutilization_25pct"]
CLASS_ENDPOINTS = ["underutilization_10pct", "underutilization_25pct"]


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
        "compact_feature_manifest.csv", "raw_model_tuning_results.csv",
        "raw_model_selections.csv", "raw_validation_evaluation_predictions.csv",
        "calibration_policy_selection.csv", "classification_predictions.csv",
        "classification_origin_metrics.csv", "classification_comparison_results.csv",
        "continuous_predictions.csv", "continuous_comparison_results.csv",
        "probability_consistency_audit.csv", "calibrated_risk_bands.csv",
        "subgroup_stability.csv", "refinement_decision_summary.csv", "build_checks.csv",
        "source_manifest.csv", "README.md", "run_summary.json", "output_manifest.csv",
    ]
    missing = [name for name in required if not (OUTPUT_DIR / name).exists()]
    add(checks, "required_outputs_exist", not missing, ";".join(missing), "none", "All promised outputs exist.")
    if missing:
        raise RuntimeError(f"Missing outputs: {missing}")

    upstream = json.loads((MODEL_DIR / "independent_verification.json").read_text(encoding="utf-8"))
    add(checks, "upstream_model_verified", upstream.get("status") == "pass", upstream.get("status"), "pass", "Frozen source model independently verified.")
    features = pd.read_csv(OUTPUT_DIR / "compact_feature_manifest.csv")
    tuning = pd.read_csv(OUTPUT_DIR / "raw_model_tuning_results.csv")
    raw_selections = pd.read_csv(OUTPUT_DIR / "raw_model_selections.csv")
    raw = pd.read_csv(OUTPUT_DIR / "raw_validation_evaluation_predictions.csv", low_memory=False)
    selection = pd.read_csv(OUTPUT_DIR / "calibration_policy_selection.csv")
    classification = pd.read_csv(OUTPUT_DIR / "classification_predictions.csv", low_memory=False)
    metrics = pd.read_csv(OUTPUT_DIR / "classification_origin_metrics.csv")
    comparisons = pd.read_csv(OUTPUT_DIR / "classification_comparison_results.csv")
    continuous_predictions = pd.read_csv(OUTPUT_DIR / "continuous_predictions.csv", low_memory=False)
    continuous = pd.read_csv(OUTPUT_DIR / "continuous_comparison_results.csv")
    consistency = pd.read_csv(OUTPUT_DIR / "probability_consistency_audit.csv")
    bands = pd.read_csv(OUTPUT_DIR / "calibrated_risk_bands.csv")
    subgroups = pd.read_csv(OUTPUT_DIR / "subgroup_stability.csv")
    decisions = pd.read_csv(OUTPUT_DIR / "refinement_decision_summary.csv")
    build_checks = pd.read_csv(OUTPUT_DIR / "build_checks.csv")
    sources = pd.read_csv(OUTPUT_DIR / "source_manifest.csv")
    manifest = pd.read_csv(OUTPUT_DIR / "output_manifest.csv")
    summary = json.loads((OUTPUT_DIR / "run_summary.json").read_text(encoding="utf-8"))

    feature_counts = features.groupby("compact_model")["raw_feature_count"].max().astype(int).to_dict()
    expected_counts = dict(zip(COMPACT_MODELS, [11, 19, 28]))
    add(checks, "compact_feature_counts", feature_counts == expected_counts, json.dumps(feature_counts, sort_keys=True), json.dumps(expected_counts, sort_keys=True), "Compact dimensions exact.")
    feature_sets = {
        model: features.loc[features["compact_model"].eq(model)].sort_values("feature_order")["feature"].tolist()
        for model in COMPACT_MODELS
    }
    add(checks, "feature_sets_nested", set(feature_sets[COMPACT_MODELS[0]]).issubset(feature_sets[COMPACT_MODELS[1]]) and set(feature_sets[COMPACT_MODELS[1]]).issubset(feature_sets[COMPACT_MODELS[2]]), True, True, "Each compact block only adds fields.")
    added_context = set(feature_sets[COMPACT_MODELS[2]]) - set(feature_sets[COMPACT_MODELS[1]])
    add(checks, "compact_context_nine_fields", len(added_context) == 9 and all("destination_pre365" in feature for feature in added_context), ";".join(sorted(added_context)), "nine destination pre365 fields", "No two-year/origin context reintroduced.")
    forbidden = [feature for feature in features["feature"].astype(str) if any(token in feature.lower() for token in ["post12", "post24", "target_", "eligible_"])]
    add(checks, "no_outcome_predictors", not forbidden, ";".join(forbidden), "none", "All refined predictors are pre-transfer.")
    add(checks, "feature_timing_complete", features["timing_classification"].notna().all(), features["timing_classification"].notna().sum(), len(features), "Timing declared for every field.")

    add(checks, "tuning_rows", len(tuning) == 288, len(tuning), 288, "Ten regression and seven classification candidates across compact blocks and origins.")
    add(checks, "raw_selection_rows", len(raw_selections) == 36, len(raw_selections), 36, "Three endpoints by four origins by three blocks.")
    temporal = (
        (raw_selections["internal_model_train_end_year"] < raw_selections["internal_model_selector_year"]).all()
        and (raw_selections["internal_model_selector_year"] < raw_selections["calibration_validation_year"]).all()
        and (raw_selections["calibration_validation_year"] < raw_selections["evaluation_year"]).all()
    )
    add(checks, "nested_temporal_order", temporal, temporal, True, "Raw tuning, calibration selection, and evaluation strictly ordered.")
    tuning_groups = tuning.groupby(["endpoint", "origin_id", "compact_model"])
    chosen = tuning_groups.apply(
        lambda group: group.sort_values([
            "internal_selector_primary_loss", "internal_selector_secondary_loss",
            "model_family", "hyperparameters",
        ]).iloc[0][["model_family", "hyperparameters"]],
        include_groups=False,
    ).reset_index()
    matched_selection = raw_selections.merge(chosen, on=["endpoint", "origin_id", "compact_model"], validate="one_to_one")
    raw_selection_exact = matched_selection["selected_family"].eq(matched_selection["model_family"]).all() and matched_selection["selected_hyperparameters"].eq(matched_selection["hyperparameters"]).all()
    add(checks, "raw_candidate_selection_exact", raw_selection_exact, raw_selection_exact, True, "Internal selector minimum reproduced.")

    raw_key = ["split", "endpoint", "origin_id", "compact_model", "transfer_event_id"]
    add(checks, "raw_prediction_key_unique", not raw.duplicated(raw_key).any(), raw.duplicated(raw_key).sum(), 0, "Raw prediction grain unique.")
    add(checks, "raw_split_set", set(raw["split"]) == {"validation", "evaluation"}, ";".join(sorted(raw["split"].unique())), "evaluation;validation", "Both temporal forecast layers retained.")
    add(checks, "raw_endpoint_set", set(raw["endpoint"]) == set(ENDPOINTS), ";".join(sorted(raw["endpoint"].unique())), ";".join(ENDPOINTS), "All endpoints refined.")

    add(checks, "calibration_candidates", len(selection) == 60, len(selection), 60, "Three blocks by five strategies by four origins.")
    selected = selection.loc[selection["selected_for_calibrated_policy"].astype(str).str.lower().eq("true")]
    add(checks, "one_joint_policy_per_origin", len(selected) == 4 and selected.groupby("origin_id").size().eq(1).all(), len(selected), 4, "One joint block/calibrator selected per origin.")
    add(checks, "selected_compact_context_all_origins", selected["compact_model"].eq(COMPACT_MODELS[2]).all(), ";".join(selected["compact_model"]), COMPACT_MODELS[2], "Compact destination block selected four times.")
    minimum = selection.sort_values([
        "origin_id", "selector_mean_brier", "selector_mean_log_loss", "compact_model",
        "calibration_strategy",
    ]).groupby("origin_id", as_index=False).first()
    selected_check = selected.merge(minimum[["origin_id", "compact_model", "calibration_strategy"]], on="origin_id", suffixes=("_stored", "_minimum"), validate="one_to_one")
    selection_exact = selected_check["compact_model_stored"].eq(selected_check["compact_model_minimum"]).all() and selected_check["calibration_strategy_stored"].eq(selected_check["calibration_strategy_minimum"]).all()
    add(checks, "calibration_policy_selection_exact", selection_exact, selection_exact, True, "Joint selector minimum reproduced.")

    class_key = ["strategy", "endpoint", "origin_id", "transfer_event_id"]
    add(checks, "classification_key_unique", not classification.duplicated(class_key).any(), classification.duplicated(class_key).sum(), 0, "Final probability grain unique.")
    add(checks, "probability_bounds", classification["probability"].between(0, 1).all(), f"{classification['probability'].min()}..{classification['probability'].max()}", "0..1", "All probabilities valid.")
    calibrated = classification.loc[classification["strategy"].eq("selected_compact_calibrated")]
    add(checks, "calibrated_evaluation_rows", calibrated["transfer_event_id"].nunique() == 584 and len(calibrated) == 1168, f"{calibrated['transfer_event_id'].nunique()};{len(calibrated)}", "584;1168", "Two thresholds for every event.")
    wide = calibrated.pivot(index=["origin_id", "transfer_event_id"], columns="endpoint", values="probability")
    violations = int((wide["underutilization_10pct"] > wide["underutilization_25pct"]).sum())
    add(checks, "calibrated_probability_order", violations == 0, violations, 0, "Threshold probabilities logically nested.")
    consistency_violations = int(consistency.loc[consistency["strategy"].eq("selected_compact_calibrated"), "order_violation_rows"].sum())
    add(checks, "consistency_audit_exact", consistency_violations == violations, consistency_violations, violations, "Stored order audit recomputed.")

    metric_ok = True
    for row in metrics.itertuples(index=False):
        group = classification.loc[
            classification["strategy"].eq(row.strategy)
            & classification["endpoint"].eq(row.endpoint)
            & classification["origin_id"].eq(row.origin_id)
        ]
        metric_ok &= close(brier_score_loss(group["actual"], group["probability"]), row.brier)
    add(checks, "origin_brier_recomputed", metric_ok, metric_ok, True, "Every origin Brier exact.")

    comparison_ok = True
    for row in comparisons.loc[comparisons["window"].eq("pooled_four")].itertuples(index=False):
        subset = classification.loc[classification["endpoint"].eq(row.endpoint)]
        keys = ["origin_id", "evaluation_year", "transfer_event_id", "player_id"]
        left = subset.loc[subset["strategy"].eq(row.baseline_strategy), keys + ["actual", "probability"]].rename(columns={"probability": "baseline"})
        right = subset.loc[subset["strategy"].eq(row.candidate_strategy), keys + ["probability"]].rename(columns={"probability": "candidate"})
        pair = left.merge(right, on=keys, validate="one_to_one")
        baseline_brier = brier_score_loss(pair["actual"], pair["baseline"])
        candidate_brier = brier_score_loss(pair["actual"], pair["candidate"])
        comparison_ok &= close(baseline_brier, row.baseline_brier) and close(candidate_brier, row.candidate_brier) and close(baseline_brier - candidate_brier, row.brier_improvement)
    add(checks, "classification_comparisons_recomputed", comparison_ok, comparison_ok, True, "All pooled Brier comparisons exact.")

    continuous_key = ["strategy", "origin_id", "transfer_event_id"]
    add(checks, "continuous_key_unique", not continuous_predictions.duplicated(continuous_key).any(), continuous_predictions.duplicated(continuous_key).sum(), 0, "Continuous forecast grain unique.")
    continuous_ok = True
    for row in continuous.loc[continuous["window"].eq("pooled_four")].itertuples(index=False):
        keys = ["origin_id", "evaluation_year", "transfer_event_id", "player_id"]
        left = continuous_predictions.loc[continuous_predictions["strategy"].eq(row.baseline_strategy), keys + ["actual", "prediction"]].rename(columns={"prediction": "baseline"})
        right = continuous_predictions.loc[continuous_predictions["strategy"].eq(row.candidate_strategy), keys + ["prediction"]].rename(columns={"prediction": "candidate"})
        if row.candidate_strategy == "selected_compact_policy":
            right = continuous_predictions.loc[
                continuous_predictions["strategy"].str.endswith("_raw")
                & continuous_predictions["selected_for_continuous_policy"].astype(str).str.lower().eq("true"),
                keys + ["prediction"],
            ].rename(columns={"prediction": "candidate"})
        pair = left.merge(right, on=keys, validate="one_to_one")
        baseline_mae = np.abs(pair["actual"] - pair["baseline"]).mean()
        candidate_mae = np.abs(pair["actual"] - pair["candidate"]).mean()
        continuous_ok &= close(baseline_mae, row.baseline_mae) and close(candidate_mae, row.candidate_mae) and close(baseline_mae - candidate_mae, row.mae_improvement)
    add(checks, "continuous_comparisons_recomputed", continuous_ok, continuous_ok, True, "All pooled MAE comparisons exact.")

    compact_context = continuous.loc[
        continuous["comparison"].eq("fixed_compact_context_vs_existing_policy")
        & continuous["window"].eq("pooled_four")
    ].iloc[0]
    add(checks, "continuous_replacement_supported", compact_context["cluster_ci_lower_95"] > 0 and compact_context["origin_wins"] >= 3, f"{compact_context['mae_improvement']};{compact_context['cluster_ci_lower_95']};{compact_context['origin_wins']}", "positive improvement/CI; >=3 wins", "Fixed compact context beats existing continuous policy.")
    context_binary = comparisons.loc[
        comparisons["comparison"].eq("compact_context_vs_prior_opportunity")
        & comparisons["window"].eq("pooled_four")
    ].set_index("endpoint")
    add(checks, "context_under25_supported", context_binary.loc["underutilization_25pct", "cluster_ci_lower_95"] > 0 and context_binary.loc["underutilization_25pct", "origin_wins"] == 4, context_binary.loc["underutilization_25pct", "cluster_ci_lower_95"], ">0 with 4 wins", "Compact destination context adds supported under-25 signal.")
    add(checks, "context_under10_not_overclaimed", context_binary.loc["underutilization_10pct", "cluster_ci_lower_95"] <= 0, context_binary.loc["underutilization_10pct", "cluster_ci_lower_95"], "<=0", "Severe-risk context gain remains uncertain.")

    decision_map = decisions.set_index("endpoint")["recommendation"].to_dict()
    expected_decisions = {
        "opportunity_share_24m": "replace_existing_continuous_policy_with_fixed_compact_context",
        "underutilization_10pct": "retain_existing_ordered_ranking_and_use_risk_bands_not_absolute_probabilities",
        "underutilization_25pct": "use_compact_calibrated_policy_for_probabilities_retain_existing_as_ranking_reference",
    }
    add(checks, "decision_recommendations", decision_map == expected_decisions, json.dumps(decision_map, sort_keys=True), json.dumps(expected_decisions, sort_keys=True), "Asymmetric replacement policy preserved.")
    negative_counts = subgroups.loc[
        subgroups["adequate_for_inference"].astype(str).str.lower().eq("true")
        & subgroups["stability_label"].eq("supported_negative")
    ].groupby("endpoint").size().to_dict()
    add(checks, "subgroup_negative_counts", negative_counts == {"underutilization_10pct": 3}, json.dumps(negative_counts, sort_keys=True), '{"underutilization_10pct": 3}', "Under-10 calibration subgroup failures block replacement.")
    band_counts = bands.groupby(["strategy", "endpoint"])["rows"].sum()
    add(checks, "risk_band_counts", band_counts.eq(584).all(), band_counts.to_dict(), "584 each", "Every displayed policy/event assigned a band.")
    add(checks, "builder_checks_pass", build_checks["passed"].astype(str).str.lower().eq("true").all(), build_checks["passed"].sum(), len(build_checks), "All builder checks pass.")

    source_ok = True
    for row in sources.itertuples(index=False):
        path = ROOT / row.source
        source_ok &= path.exists() and path.stat().st_size == row.size_bytes and sha256_file(path) == row.sha256
    add(checks, "source_manifest_hashes", source_ok, source_ok, True, "All source artifacts unchanged.")
    output_ok = True
    for row in manifest.itertuples(index=False):
        path = OUTPUT_DIR / row.file
        output_ok &= path.exists() and path.stat().st_size == row.size_bytes and sha256_file(path) == row.sha256
    add(checks, "output_manifest_hashes", output_ok, output_ok, True, "All modeled artifacts hash-match.")
    add(checks, "summary_counts", summary["cohort_rows"] == 1074 and summary["evaluation_rows"] == 584 and summary["policy_order_violations"] == 0, json.dumps({key: summary[key] for key in ["cohort_rows", "evaluation_rows", "policy_order_violations"]}, sort_keys=True), "1074;584;0", "Run summary reconciles.")
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

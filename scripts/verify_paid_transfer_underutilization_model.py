"""Independently verify the paid-transfer under-utilization model artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "Data" / "processed" / "financial_module_feasibility" / "financial_event_audit.csv"
OUTPUT_DIR = ROOT / "Data" / "processed" / "paid_transfer_underutilization_model"
EXPECTED_EVALUATION_COUNTS = {2020: 105, 2021: 136, 2022: 174, 2023: 169}
ENDPOINT_KIND = {
    "opportunity_share_24m": "regression",
    "underutilization_10pct": "classification",
    "underutilization_25pct": "classification",
}
FIXED_MODELS = [
    "M0_chronological_constant", "M1_paid_deal_market_profile",
    "M2_plus_prior_opportunity", "M3_plus_player_performance",
    "M4_plus_destination_context", "M5_plus_origin_transition_context",
]


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
        "underutilization_targets.csv", "underutilization_feature_manifest.csv",
        "validation_candidate_results.csv", "feature_block_selections.csv",
        "underutilization_predictions.csv", "underutilization_origin_results.csv",
        "underutilization_comparison_results.csv", "ordered_policy_predictions.csv",
        "underutilization_risk_bands.csv", "underutilization_subgroup_performance.csv",
        "underutilization_decision_summary.csv", "build_checks.csv", "source_manifest.csv",
        "README.md", "run_summary.json", "output_manifest.csv",
    ]
    missing = [name for name in required if not (OUTPUT_DIR / name).exists()]
    add(checks, "required_outputs_exist", not missing, ";".join(missing), "none", "All promised artifacts exist.")
    if missing:
        raise RuntimeError(f"Missing outputs: {missing}")

    audit = pd.read_csv(AUDIT_PATH, low_memory=False)
    targets = pd.read_csv(OUTPUT_DIR / "underutilization_targets.csv", low_memory=False)
    features = pd.read_csv(OUTPUT_DIR / "underutilization_feature_manifest.csv")
    tuning = pd.read_csv(OUTPUT_DIR / "validation_candidate_results.csv")
    selections = pd.read_csv(OUTPUT_DIR / "feature_block_selections.csv")
    predictions = pd.read_csv(OUTPUT_DIR / "underutilization_predictions.csv", low_memory=False)
    origins = pd.read_csv(OUTPUT_DIR / "underutilization_origin_results.csv")
    comparisons = pd.read_csv(OUTPUT_DIR / "underutilization_comparison_results.csv")
    ordered = pd.read_csv(OUTPUT_DIR / "ordered_policy_predictions.csv")
    bands = pd.read_csv(OUTPUT_DIR / "underutilization_risk_bands.csv")
    subgroups = pd.read_csv(OUTPUT_DIR / "underutilization_subgroup_performance.csv")
    decisions = pd.read_csv(OUTPUT_DIR / "underutilization_decision_summary.csv")
    build_checks = pd.read_csv(OUTPUT_DIR / "build_checks.csv")
    sources = pd.read_csv(OUTPUT_DIR / "source_manifest.csv")
    output_manifest = pd.read_csv(OUTPUT_DIR / "output_manifest.csv")
    summary = json.loads((OUTPUT_DIR / "run_summary.json").read_text(encoding="utf-8"))

    eligible = audit.loc[as_bool(audit["eligible_paid_transfer_underutilization_24m"])].copy()
    eligible["transfer_event_id"] = eligible["transfer_event_id"].astype(str)
    targets["transfer_event_id"] = targets["transfer_event_id"].astype(str)
    add(checks, "audit_cohort_count", len(eligible) == 1074, len(eligible), 1074, "Audit independently yields frozen cohort.")
    add(checks, "target_cohort_count", len(targets) == 1074, len(targets), 1074, "Target artifact retains every eligible row.")
    add(checks, "target_key_unique", targets["transfer_event_id"].is_unique, targets["transfer_event_id"].nunique(), len(targets), "One target row per event.")
    add(checks, "cohort_ids_exact", set(targets["transfer_event_id"]) == set(eligible["transfer_event_id"]), len(set(targets["transfer_event_id"]) ^ set(eligible["transfer_event_id"])), 0, "No eligible IDs lost or added.")

    audit_target = eligible[["transfer_event_id", "destination_opportunity_share_24m"]]
    merged = targets.merge(audit_target, on="transfer_event_id", how="inner", validate="one_to_one")
    exact_target = np.allclose(
        merged["target_opportunity_share_24m"], merged["destination_opportunity_share_24m"],
        atol=1e-12, rtol=0,
    )
    add(checks, "continuous_target_exact", exact_target, exact_target, True, "Target equals audited opportunity share.")
    recomputed10 = merged["destination_opportunity_share_24m"].lt(0.10).astype(int)
    recomputed25 = merged["destination_opportunity_share_24m"].lt(0.25).astype(int)
    add(checks, "under10_target_exact", np.array_equal(recomputed10, merged["target_underutilization_10pct"]), int(recomputed10.sum()), 355, "Strict under-10 threshold recomputed.")
    add(checks, "under25_target_exact", np.array_equal(recomputed25, merged["target_underutilization_25pct"]), int(recomputed25.sum()), 553, "Strict under-25 threshold recomputed.")
    add(checks, "target_nesting", (merged["target_underutilization_10pct"] <= merged["target_underutilization_25pct"]).all(), True, True, "Severe target nested in meaningful target.")

    feature_counts = features.groupby("feature_set")["raw_feature_count"].max().fillna(0).astype(int).to_dict()
    expected_counts = dict(zip(FIXED_MODELS, [0, 11, 19, 35, 79, 139]))
    add(checks, "feature_counts", feature_counts == expected_counts, json.dumps(feature_counts, sort_keys=True), json.dumps(expected_counts, sort_keys=True), "Frozen block sizes exact.")
    feature_names = features["feature"].dropna().astype(str).str.lower()
    forbidden = feature_names[feature_names.str.contains("post12|post24|target_|eligible_", regex=True)].tolist()
    add(checks, "feature_timing_no_outcomes", not forbidden, ";".join(forbidden), "none", "No post-transfer or target fields in manifest.")
    add(checks, "feature_timing_declared", features["timing_classification"].notna().all(), features["timing_classification"].notna().sum(), len(features), "Every feature has timing classification.")

    prediction_key = ["endpoint", "origin_id", "model_variant", "transfer_event_id"]
    add(checks, "prediction_key_unique", not predictions.duplicated(prediction_key).any(), predictions.duplicated(prediction_key).sum(), 0, "Prediction grain unique.")
    add(checks, "endpoint_set", set(predictions["endpoint"]) == set(ENDPOINT_KIND), ";".join(sorted(predictions["endpoint"].unique())), ";".join(ENDPOINT_KIND), "Exactly three endpoints.")
    add(checks, "evaluation_years", set(predictions["evaluation_year"]) == set(EXPECTED_EVALUATION_COUNTS), sorted(predictions["evaluation_year"].unique()), sorted(EXPECTED_EVALUATION_COUNTS), "Four frozen seasons.")
    policy = predictions.loc[predictions["model_variant"].eq("M_policy_validation_selected")]
    observed_counts = policy.loc[policy["endpoint"].eq("opportunity_share_24m")].groupby("evaluation_year").size().to_dict()
    add(checks, "evaluation_counts", observed_counts == EXPECTED_EVALUATION_COUNTS, json.dumps(observed_counts, sort_keys=True), json.dumps(EXPECTED_EVALUATION_COUNTS, sort_keys=True), "Rolling evaluation row counts exact.")
    add(checks, "policy_rows", len(policy) == 3 * 584, len(policy), 1752, "One selected forecast per endpoint and event.")
    ordered_rows = predictions.loc[predictions["model_variant"].eq("M_policy_ordered")]
    add(checks, "ordered_prediction_rows", len(ordered_rows) == 2 * 584, len(ordered_rows), 1168, "Ordered rows exist for both binary endpoints.")
    class_values = predictions.loc[predictions["endpoint_kind"].eq("classification"), "prediction"]
    add(checks, "probability_bounds", class_values.between(0, 1).all(), f"{class_values.min()}..{class_values.max()}", "0..1", "All classification predictions bounded.")

    add(checks, "ordered_table_rows", len(ordered) == 584, len(ordered), 584, "One paired threshold row per evaluation event.")
    violations = int((ordered["ordered_probability_under10"] > ordered["ordered_probability_under25"]).sum())
    add(checks, "ordered_table_nesting", violations == 0, violations, 0, "Ordered probabilities logically nested.")
    raw_violations = int((ordered["raw_probability_under10"] > ordered["raw_probability_under25"]).sum())
    add(checks, "raw_violation_count", raw_violations == 48, raw_violations, 48, "Pre-projection violations preserved for audit.")
    midpoint10 = np.where(
        ordered["raw_probability_under10"] > ordered["raw_probability_under25"],
        (ordered["raw_probability_under10"] + ordered["raw_probability_under25"]) / 2,
        ordered["raw_probability_under10"],
    )
    midpoint25 = np.where(
        ordered["raw_probability_under10"] > ordered["raw_probability_under25"],
        (ordered["raw_probability_under10"] + ordered["raw_probability_under25"]) / 2,
        ordered["raw_probability_under25"],
    )
    projection_exact = np.allclose(midpoint10, ordered["ordered_probability_under10"], atol=1e-12, rtol=0) and np.allclose(midpoint25, ordered["ordered_probability_under25"], atol=1e-12, rtol=0)
    add(checks, "ordering_projection_exact", projection_exact, projection_exact, True, "Violations use Euclidean midpoint projection.")

    fixed_origins = origins.loc[origins["model_variant"].isin(FIXED_MODELS)].copy()
    expected_selections = fixed_origins.sort_values(
        ["endpoint", "origin_id", "validation_primary_loss", "validation_secondary_loss"]
    ).groupby(["endpoint", "origin_id"], as_index=False).first()[["endpoint", "origin_id", "model_variant"]]
    selection_check = selections.merge(expected_selections, on=["endpoint", "origin_id"], validate="one_to_one")
    selected_exact = selection_check["selected_feature_set"].eq(selection_check["model_variant"]).all()
    add(checks, "feature_block_selection_exact", selected_exact, int(selected_exact), 1, "Lowest validation loss selects block.")
    add(checks, "selection_rows", len(selections) == 12, len(selections), 12, "Three endpoints by four origins.")
    add(checks, "tuning_rows_present", len(tuning) == 480, len(tuning), 480, "Ten regression and seven classification candidates per nonconstant block/origin.")

    metric_ok = True
    for row in origins.itertuples(index=False):
        subset = predictions.loc[
            predictions["endpoint"].eq(row.endpoint)
            & predictions["origin_id"].eq(row.origin_id)
            & predictions["model_variant"].eq(row.model_variant)
        ]
        if row.endpoint_kind == "regression":
            value = np.abs(subset["actual"] - subset["prediction"]).mean()
            metric_ok &= close(value, row.mae)
        else:
            value = brier_score_loss(subset["actual"], subset["prediction"])
            metric_ok &= close(value, row.brier)
    add(checks, "origin_primary_metrics_recomputed", metric_ok, metric_ok, True, "MAE/Brier independently recomputed.")

    comparison_ok = True
    for row in comparisons.itertuples(index=False):
        if row.window != "pooled_four":
            continue
        subset = predictions.loc[predictions["endpoint"].eq(row.endpoint)]
        keys = ["origin_id", "evaluation_year", "transfer_event_id", "player_id"]
        left = subset.loc[subset["model_variant"].eq(row.baseline_model), keys + ["actual", "prediction"]].rename(columns={"prediction": "baseline"})
        right = subset.loc[subset["model_variant"].eq(row.candidate_model), keys + ["prediction"]].rename(columns={"prediction": "candidate"})
        pair = left.merge(right, on=keys, validate="one_to_one")
        if row.endpoint_kind == "regression":
            base = np.abs(pair["actual"] - pair["baseline"]).mean()
            candidate = np.abs(pair["actual"] - pair["candidate"]).mean()
        else:
            base = np.square(pair["actual"] - pair["baseline"]).mean()
            candidate = np.square(pair["actual"] - pair["candidate"]).mean()
        comparison_ok &= close(base, row.baseline_loss) and close(candidate, row.candidate_loss) and close(base - candidate, row.loss_improvement)
    add(checks, "pooled_comparisons_recomputed", comparison_ok, comparison_ok, True, "Primary pooled losses exact.")

    primary = comparisons.loc[
        comparisons["window"].eq("pooled_four")
        & comparisons["comparison"].isin(["policy_vs_M0_total", "ordered_policy_vs_M0_total"])
    ]
    primary = primary.loc[
        ((primary["endpoint_kind"].eq("regression")) & primary["comparison"].eq("policy_vs_M0_total"))
        | ((primary["endpoint_kind"].eq("classification")) & primary["comparison"].eq("ordered_policy_vs_M0_total"))
    ]
    add(checks, "all_primary_signals_supported", (primary["cluster_ci_lower_95"] > 0).all() and (primary["origin_wins"] == 4).all(), f"min_ci={primary['cluster_ci_lower_95'].min()};wins={primary['origin_wins'].tolist()}", "positive CI; 4 wins", "All primary endpoints repeat out of season.")
    add(checks, "decision_status_consistent", decisions["status"].eq("validated_predictive_signal").all(), ";".join(decisions["status"]), "all validated_predictive_signal", "Decision labels follow primary evidence.")

    band_counts = bands.groupby("endpoint")["rows"].sum().to_dict()
    add(checks, "risk_band_row_totals", band_counts == {endpoint: 584 for endpoint in ENDPOINT_KIND}, json.dumps(band_counts, sort_keys=True), "584 each", "Every recommended forecast assigned once.")
    band_endpoints = bands.set_index(["endpoint", "risk_band"])
    separation_ok = (
        band_endpoints.loc[("underutilization_10pct", 5), "mean_actual"] > band_endpoints.loc[("underutilization_10pct", 1), "mean_actual"]
        and band_endpoints.loc[("underutilization_25pct", 5), "mean_actual"] > band_endpoints.loc[("underutilization_25pct", 1), "mean_actual"]
    )
    add(checks, "binary_risk_band_separation", separation_ok, separation_ok, True, "Highest-risk quintiles have more under-use.")
    add(checks, "subgroup_dimensions", set(subgroups["dimension"]) == {"historical_position_group", "age_band", "fee_band", "destination_competition_id", "evaluation_year"}, ";".join(sorted(subgroups["dimension"].unique())), "five frozen dimensions", "Descriptive audit coverage complete.")
    add(checks, "build_checks_pass", build_checks["passed"].astype(str).str.lower().eq("true").all(), build_checks["passed"].sum(), len(build_checks), "Builder checks all pass.")

    source_hash_ok = True
    for row in sources.itertuples(index=False):
        path = ROOT / row.source
        source_hash_ok &= path.exists() and path.stat().st_size == row.size_bytes and sha256_file(path) == row.sha256
    add(checks, "source_manifest_hashes", source_hash_ok, source_hash_ok, True, "Input files unchanged and hash-matched.")
    output_hash_ok = True
    for row in output_manifest.itertuples(index=False):
        path = OUTPUT_DIR / row.file
        output_hash_ok &= path.exists() and path.stat().st_size == row.size_bytes and sha256_file(path) == row.sha256
    add(checks, "output_manifest_hashes", output_hash_ok, output_hash_ok, True, "Every listed artifact hash-matched.")
    add(checks, "summary_counts", summary["cohort_rows"] == 1074 and summary["evaluation_rows"] == 584 and summary["target_counts"] == {"under10": 355, "under25": 553}, json.dumps({"cohort_rows": summary["cohort_rows"], "evaluation_rows": summary["evaluation_rows"], "target_counts": summary["target_counts"]}, sort_keys=True), "1074;584;355;553", "Run summary reconciles.")
    add(checks, "protected_outputs_unchanged", summary["protected_outputs_unchanged"] is True, summary["protected_outputs_unchanged"], True, "Runner reports protected-tree identity.")

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

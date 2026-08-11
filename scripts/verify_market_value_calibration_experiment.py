"""Independently verify the market-value calibration and consistency experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "Data" / "processed" / "market_value_downside_model"
OUTPUT_DIR = ROOT / "Data" / "processed" / "market_value_calibration_experiment"
M1 = "M1_market_profile"
THRESHOLDS = ["downside_10pct", "downside_25pct", "downside_50pct"]
STRATEGIES = {
    "raw_independent", "raw_projected", "platt_independent", "platt_projected",
    "isotonic_independent", "isotonic_projected", "gaussian_distributional",
    "empirical_distributional", "policy_selected",
}
POLICY_ELIGIBLE = {
    "raw_projected", "platt_projected", "isotonic_projected",
    "gaussian_distributional", "empirical_distributional",
}


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").fillna("").str.strip().str.lower().isin({"true", "1", "yes"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add(checks: list[dict[str, Any]], check: str, passed: bool, observed: Any, expected: Any, note: str) -> None:
    checks.append({"check": check, "passed": bool(passed), "observed": observed, "expected": expected, "note": note})


def ece(actual: pd.Series, probability: pd.Series) -> float:
    work = pd.DataFrame({"actual": actual.astype(float), "probability": probability.astype(float)})
    work["bin"] = pd.qcut(work["probability"].rank(method="first"), 10, labels=False, duplicates="drop")
    grouped = work.groupby("bin").agg(n=("actual", "size"), actual=("actual", "mean"), probability=("probability", "mean"))
    return float((grouped["n"] / len(work) * (grouped["actual"] - grouped["probability"]).abs()).sum())


def calibration_parameters(actual: pd.Series, probability: pd.Series) -> tuple[float, float]:
    p = np.clip(probability.to_numpy(float), 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, max_iter=3000, random_state=20260810).fit(x, actual.to_numpy(int))
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def verify() -> None:
    required = [
        "README.md", "run_summary.json", "calibration_validation_predictions.csv",
        "calibration_strategy_selection.csv", "calibrated_predictions.csv",
        "calibration_origin_metrics.csv", "calibration_comparison_results.csv",
        "probability_consistency_audit.csv", "calibrated_risk_bands.csv",
        "calibration_subgroup_stability.csv",
        "calibration_decision_summary.csv", "build_checks.csv", "source_manifest.csv",
        "output_manifest.csv",
    ]
    checks: list[dict[str, Any]] = []
    missing = [name for name in required if not (OUTPUT_DIR / name).exists()]
    add(checks, "required_outputs_exist", not missing, ";".join(missing), "", "All runner outputs exist.")
    if missing:
        raise RuntimeError(f"Missing outputs: {missing}")

    validation = pd.read_csv(OUTPUT_DIR / "calibration_validation_predictions.csv", low_memory=False)
    selection = pd.read_csv(OUTPUT_DIR / "calibration_strategy_selection.csv", low_memory=False)
    predictions = pd.read_csv(OUTPUT_DIR / "calibrated_predictions.csv", low_memory=False)
    metrics = pd.read_csv(OUTPUT_DIR / "calibration_origin_metrics.csv", low_memory=False)
    comparisons = pd.read_csv(OUTPUT_DIR / "calibration_comparison_results.csv", low_memory=False)
    consistency = pd.read_csv(OUTPUT_DIR / "probability_consistency_audit.csv", low_memory=False)
    risk_bands = pd.read_csv(OUTPUT_DIR / "calibrated_risk_bands.csv", low_memory=False)
    subgroups = pd.read_csv(OUTPUT_DIR / "calibration_subgroup_stability.csv", low_memory=False)
    decisions = pd.read_csv(OUTPUT_DIR / "calibration_decision_summary.csv", low_memory=False)
    build_checks = pd.read_csv(OUTPUT_DIR / "build_checks.csv")
    sources = pd.read_csv(OUTPUT_DIR / "source_manifest.csv")
    manifest = pd.read_csv(OUTPUT_DIR / "output_manifest.csv")
    run = json.loads((OUTPUT_DIR / "run_summary.json").read_text(encoding="utf-8"))

    add(checks, "validation_key_unique", not validation.duplicated(["endpoint", "origin_id", "transfer_event_id"]).any(), int(validation.duplicated(["endpoint", "origin_id", "transfer_event_id"]).sum()), 0, "One reconstructed validation forecast per target/event.")
    add(checks, "validation_endpoint_set", set(validation["endpoint"]) == set(THRESHOLDS + ["value_log_ratio_24m"]), ";".join(sorted(set(validation["endpoint"]))), ";".join(sorted(THRESHOLDS + ["value_log_ratio_24m"])), "Continuous and threshold validation forecasts exist.")
    origin_results = pd.read_csv(MODEL_DIR / "market_value_origin_results.csv", low_memory=False)
    expected_validation = origin_results.loc[origin_results["model_variant"].eq(M1)].groupby(["endpoint", "origin_id"])["validation_n"].first()
    actual_validation = validation.groupby(["endpoint", "origin_id"]).size()
    add(checks, "validation_counts", expected_validation.equals(actual_validation), actual_validation.to_dict(), expected_validation.to_dict(), "Reconstructed rows match the frozen model split.")

    key = ["strategy", "endpoint", "origin_id", "transfer_event_id"]
    add(checks, "prediction_key_unique", not predictions.duplicated(key).any(), int(predictions.duplicated(key).sum()), 0, "One probability per strategy/target/origin/event.")
    add(checks, "strategy_set", set(predictions["strategy"]) == STRATEGIES, ";".join(sorted(set(predictions["strategy"]))), ";".join(sorted(STRATEGIES)), "All predeclared strategies are present.")
    add(checks, "threshold_set", set(predictions["endpoint"]) == set(THRESHOLDS), ";".join(sorted(set(predictions["endpoint"]))), ";".join(THRESHOLDS), "Only binary downside targets are calibrated.")
    add(checks, "probability_bounds", predictions["probability"].between(0, 1).all(), f"{predictions['probability'].min()}..{predictions['probability'].max()}", "0..1", "Every challenger output is a probability.")
    aligned = predictions.groupby(["strategy", "endpoint", "origin_id"]).size().groupby(level=[1, 2]).nunique().eq(1).all()
    add(checks, "paired_rows", aligned, aligned, True, "All strategies score identical evaluation rows.")

    frozen = pd.read_csv(MODEL_DIR / "market_value_predictions.csv", low_memory=False)
    frozen = frozen.loc[frozen["model_variant"].eq(M1) & frozen["endpoint"].isin(THRESHOLDS)]
    raw = predictions.loc[predictions["strategy"].eq("raw_independent")]
    raw_check = raw.merge(
        frozen[["endpoint", "origin_id", "transfer_event_id", "prediction"]],
        on=["endpoint", "origin_id", "transfer_event_id"], validate="one_to_one",
    )
    add(checks, "raw_benchmark_frozen", np.allclose(raw_check["probability"], raw_check["prediction"], atol=1e-12), float(np.abs(raw_check["probability"] - raw_check["prediction"]).max()), 0, "Calibration does not refit or alter the benchmark.")
    add(checks, "raw_probability_column", np.allclose(predictions["raw_probability"], predictions.merge(raw[["endpoint", "origin_id", "transfer_event_id", "probability"]], on=["endpoint", "origin_id", "transfer_event_id"], suffixes=("", "_expected"), validate="many_to_one")["probability_expected"], atol=1e-12), True, True, "Every challenger retains its paired raw probability.")

    selected = selection.loc[as_bool(selection["selected_for_policy"])]
    add(checks, "one_selected_per_origin", len(selected) == 4 and selected.groupby("origin_id").size().eq(1).all(), selected.groupby("origin_id").size().to_dict(), "one each", "Policy choice is unique.")
    add(checks, "selected_policy_eligible", set(selected["strategy"]).issubset(POLICY_ELIGIBLE), ";".join(sorted(set(selected["strategy"]))), ";".join(sorted(POLICY_ELIGIBLE)), "Selected methods guarantee ordering.")
    selection_exact = True
    for origin_id, group in selection.loc[as_bool(selection["policy_eligible"])].groupby("origin_id"):
        expected = group.sort_values(["selector_mean_brier", "selector_mean_log_loss", "strategy"]).iloc[0]["strategy"]
        observed = selected.loc[selected["origin_id"].eq(origin_id), "strategy"].iloc[0]
        selection_exact &= expected == observed
    add(checks, "selection_rule_exact", selection_exact, selection_exact, True, "Policy minimizes selector Brier with declared tie-breaks.")
    split_ok = (selection["calibration_rows"] > 0).all() and (selection["selector_rows"] > 0).all()
    add(checks, "calibration_selector_split_nonempty", split_ok, split_ok, True, "Each validation season has separate calibration and selection portions.")

    policy = predictions.loc[predictions["strategy"].eq("policy_selected")]
    chosen_match = True
    for origin_id, selected_strategy in selected.set_index("origin_id")["strategy"].items():
        left = policy.loc[policy["origin_id"].eq(origin_id), ["endpoint", "transfer_event_id", "probability"]]
        right = predictions.loc[predictions["origin_id"].eq(origin_id) & predictions["strategy"].eq(selected_strategy), ["endpoint", "transfer_event_id", "probability"]]
        pair = left.merge(right, on=["endpoint", "transfer_event_id"], suffixes=("_policy", "_selected"), validate="one_to_one")
        chosen_match &= np.allclose(pair["probability_policy"], pair["probability_selected"], atol=1e-12)
    add(checks, "policy_matches_selected_strategy", chosen_match, chosen_match, True, "Evaluation policy copies only prior-season-selected strategy.")

    consistency_ok = True
    recomputed_consistency = {}
    for (strategy, origin_id), group in predictions.groupby(["strategy", "origin_id"]):
        wide = group.pivot(index="transfer_event_id", columns="endpoint", values="probability")
        violations = int(((wide["downside_25pct"] > wide["downside_10pct"] + 1e-12) | (wide["downside_50pct"] > wide["downside_25pct"] + 1e-12)).sum())
        reported = int(consistency.loc[consistency["strategy"].eq(strategy) & consistency["origin_id"].eq(origin_id), "any_violation_rows"].iloc[0])
        consistency_ok &= violations == reported
        recomputed_consistency[(strategy, origin_id)] = violations
    add(checks, "consistency_recomputed", consistency_ok, consistency_ok, True, "Every threshold-order count recomputes.")
    raw_violations = sum(value for (strategy, _), value in recomputed_consistency.items() if strategy == "raw_independent")
    policy_violations = sum(value for (strategy, _), value in recomputed_consistency.items() if strategy == "policy_selected")
    add(checks, "raw_violation_count", raw_violations == 157, raw_violations, 157, "Frozen independent classifiers contain the audited contradictions.")
    add(checks, "policy_zero_violations", policy_violations == 0, policy_violations, 0, "Selected policy is logically nested.")
    ordered_strategies = {"raw_projected", "platt_projected", "isotonic_projected", "gaussian_distributional", "empirical_distributional", "policy_selected"}
    ordered_ok = all(value == 0 for (strategy, _), value in recomputed_consistency.items() if strategy in ordered_strategies)
    add(checks, "ordered_strategies_zero_violations", ordered_ok, ordered_ok, True, "Every constrained strategy is nested.")
    band_counts = risk_bands.groupby(["strategy", "endpoint"])["rows"].sum()
    expected_band_counts = predictions.loc[
        predictions["strategy"].isin(["raw_independent", "policy_selected"])
    ].groupby(["strategy", "endpoint"]).size()
    add(checks, "risk_band_counts", band_counts.equals(expected_band_counts), band_counts.to_dict(), expected_band_counts.to_dict(), "Risk bands cover each raw and policy forecast once.")
    add(checks, "risk_band_levels", set(risk_bands["risk_band"]) == {1, 2, 3, 4, 5}, ";".join(map(str, sorted(set(risk_bands["risk_band"])))), "1;2;3;4;5", "Five within-origin risk bands are present.")

    metrics_ok = True
    for _, row in metrics.iterrows():
        group = predictions.loc[predictions["strategy"].eq(row["strategy"]) & predictions["endpoint"].eq(row["endpoint"]) & predictions["origin_id"].eq(row["origin_id"])]
        intercept, slope = calibration_parameters(group["actual"], group["probability"])
        observed = {
            "brier": brier_score_loss(group["actual"], group["probability"]),
            "log_loss": log_loss(group["actual"], group["probability"], labels=[0, 1]),
            "roc_auc": roc_auc_score(group["actual"], group["probability"]),
            "average_precision": average_precision_score(group["actual"], group["probability"]),
            "ece_10bin": ece(group["actual"], group["probability"]),
            "calibration_intercept": intercept, "calibration_slope": slope,
        }
        metrics_ok &= all(np.isclose(float(row[name]), float(value), atol=1e-12) for name, value in observed.items())
    add(checks, "origin_metrics_recomputed", metrics_ok, metrics_ok, True, "Brier, log loss, discrimination and calibration metrics are exact.")

    comparison_ok = True
    for _, row in comparisons.iterrows():
        years = [int(value) for value in str(row["evaluation_years"]).split(";")]
        raw_rows = predictions.loc[predictions["strategy"].eq("raw_independent") & predictions["endpoint"].eq(row["endpoint"]) & predictions["evaluation_year"].isin(years)]
        candidate_rows = predictions.loc[predictions["strategy"].eq(row["strategy"]) & predictions["endpoint"].eq(row["endpoint"]) & predictions["evaluation_year"].isin(years)]
        pair = raw_rows[["origin_id", "transfer_event_id", "actual", "probability"]].merge(candidate_rows[["origin_id", "transfer_event_id", "probability"]], on=["origin_id", "transfer_event_id"], suffixes=("_raw", "_candidate"), validate="one_to_one")
        raw_brier = brier_score_loss(pair["actual"], pair["probability_raw"])
        candidate_brier = brier_score_loss(pair["actual"], pair["probability_candidate"])
        comparison_ok &= np.isclose(row["raw_brier"], raw_brier, atol=1e-12)
        comparison_ok &= np.isclose(row["candidate_brier"], candidate_brier, atol=1e-12)
        comparison_ok &= np.isclose(row["brier_improvement"], raw_brier - candidate_brier, atol=1e-12)
    add(checks, "comparison_brier_recomputed", comparison_ok, comparison_ok, True, "Every paired Brier improvement is exact.")
    interval_ok = (comparisons["cluster_ci_lower_95"] <= comparisons["cluster_ci_upper_95"]).all() and comparisons["cluster_probability_candidate_better"].between(0, 1).all()
    add(checks, "cluster_intervals_valid", interval_ok, interval_ok, True, "Bootstrap intervals and probabilities are valid.")

    decision_ok = True
    pooled = comparisons.loc[comparisons["strategy"].eq("policy_selected") & comparisons["window"].eq("pooled_four")].set_index("endpoint")
    for row in decisions.itertuples(index=False):
        result = pooled.loc[row.endpoint]
        expected = result.cluster_ci_lower_95 > 0 and result.origin_wins >= 3 and result.auc_change >= -0.005 and policy_violations == 0 and row.policy_weighted_ece < row.raw_weighted_ece and (row.endpoint != "downside_25pct" or row.adequate_supported_negative_subgroups == 0)
        decision_ok &= bool(row.replace_raw_probability_model) == bool(expected)
    add(checks, "replacement_decisions_recomputed", decision_ok, decision_ok, True, "Replacement gate follows Brier, origins, AUC, ECE, consistency and subgroup rules.")
    add(checks, "primary_no_supported_negative_subgroup", not ((as_bool(subgroups["adequate_for_inference"])) & subgroups["stability_label"].eq("supported_negative")).any(), int(((as_bool(subgroups["adequate_for_inference"])) & subgroups["stability_label"].eq("supported_negative")).sum()), 0, "No adequate subgroup is reliably harmed.")
    subgroup_bounds = (subgroups["policy_brier"].between(0, 1)).all() and (subgroups["raw_brier"].between(0, 1)).all()
    add(checks, "subgroup_metric_bounds", subgroup_bounds, subgroup_bounds, True, "Subgroup Brier scores are valid.")
    add(checks, "runner_checks_pass", as_bool(build_checks["passed"]).all(), int(as_bool(build_checks["passed"]).sum()), len(build_checks), "All runner invariants passed.")

    source_map = sources.set_index("source")
    source_ok = True
    for key, row in source_map.iterrows():
        path = ROOT / key
        source_ok &= path.exists() and path.stat().st_size == int(row["size_bytes"])
        source_ok &= sha256_file(path) == row["sha256"]
    add(checks, "source_manifest_hashes", source_ok, source_ok, True, "All frozen inputs match.")
    manifest_ok = True
    for _, row in manifest.iterrows():
        path = OUTPUT_DIR / row["file"]
        manifest_ok &= path.exists() and path.stat().st_size == int(row["size_bytes"])
        manifest_ok &= sha256_file(path) == row["sha256"]
    add(checks, "output_manifest_hashes", manifest_ok, manifest_ok, True, "All runner outputs match.")
    add(checks, "protected_outputs_unchanged", run["protected_outputs_unchanged"] is True, run["protected_outputs_unchanged"], True, "Frozen upstream outputs were not modified.")

    result = pd.DataFrame(checks)
    result.to_csv(OUTPUT_DIR / "independent_verification.csv", index=False)
    failures = result.loc[~result["passed"]]
    payload = {
        "status": "pass" if failures.empty else "fail", "checks": len(result),
        "passed": int(result["passed"].sum()), "failed": len(failures),
        "failed_checks": failures["check"].tolist(),
    }
    (OUTPUT_DIR / "independent_verification.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    if not failures.empty:
        raise RuntimeError(f"Verification failed: {failures['check'].tolist()}")


if __name__ == "__main__":
    verify()

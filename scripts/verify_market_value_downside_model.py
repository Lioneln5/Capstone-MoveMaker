"""Independently verify the 24-month market-value downside model outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "Data" / "processed" / "market_value_downside_model"
MASTER_PATH = ROOT / "Data" / "processed" / "transfermarkt_merged" / "transfermarkt_comprehensive_master.csv"
AUDIT_PATH = ROOT / "Data" / "processed" / "financial_module_feasibility" / "financial_event_audit.csv"
AUDIT_VERIFICATION_PATH = ROOT / "Data" / "processed" / "financial_module_feasibility" / "independent_verification.json"
SOURCE_PATHS = [MASTER_PATH, AUDIT_PATH, AUDIT_VERIFICATION_PATH]
MODEL_ORDER = [
    "M0_chronological_constant", "M1_market_profile", "M2_plus_valuation_trajectory",
    "M3_plus_sporting_history", "M4_plus_destination_context",
    "M5_plus_origin_transition_context",
]
ENDPOINT_KINDS = {
    "value_log_ratio_24m": "regression", "downside_10pct": "classification",
    "downside_25pct": "classification", "downside_50pct": "classification",
}
ORIGIN_EVAL_YEARS = {
    "origin_2019": 2020, "origin_2020": 2021,
    "origin_2021": 2022, "origin_2022": 2023,
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


def regression_metrics(actual: pd.Series, prediction: pd.Series) -> dict[str, float]:
    actual_array = actual.to_numpy(float)
    prediction_array = prediction.to_numpy(float)
    residual = actual_array - prediction_array
    denominator = np.square(actual_array - actual_array.mean()).sum()
    return {
        "mae": np.abs(residual).mean(), "rmse": np.sqrt(np.square(residual).mean()),
        "r2": 1 - np.square(residual).sum() / denominator if denominator > 1e-12 else np.nan,
        "actual_mean": actual_array.mean(), "predicted_mean": prediction_array.mean(),
    }


def classification_metrics(actual: pd.Series, prediction: pd.Series) -> dict[str, float]:
    y = actual.to_numpy(int)
    p = np.clip(prediction.to_numpy(float), 1e-6, 1 - 1e-6)
    return {
        "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p, labels=[0, 1]),
        "actual_rate": y.mean(), "predicted_rate": p.mean(),
        "roc_auc": roc_auc_score(y, p), "average_precision": average_precision_score(y, p),
    }


def verify() -> None:
    required = [
        "README.md", "run_summary.json", "market_value_downside_targets.csv",
        "market_value_feature_manifest.csv", "validation_candidate_results.csv",
        "market_value_predictions.csv", "market_value_origin_results.csv",
        "market_value_comparison_results.csv", "market_value_risk_bands.csv",
        "market_value_decision_summary.csv", "build_checks.csv", "source_manifest.csv",
        "output_manifest.csv",
    ]
    checks: list[dict[str, Any]] = []
    missing = [name for name in required if not (OUTPUT_DIR / name).exists()]
    add(checks, "required_outputs_exist", not missing, ";".join(missing), "", "All runner outputs exist.")
    if missing:
        raise RuntimeError(f"Missing outputs: {missing}")

    targets = pd.read_csv(OUTPUT_DIR / "market_value_downside_targets.csv", low_memory=False)
    features = pd.read_csv(OUTPUT_DIR / "market_value_feature_manifest.csv", low_memory=False)
    tuning = pd.read_csv(OUTPUT_DIR / "validation_candidate_results.csv", low_memory=False)
    predictions = pd.read_csv(OUTPUT_DIR / "market_value_predictions.csv", low_memory=False)
    origins = pd.read_csv(OUTPUT_DIR / "market_value_origin_results.csv", low_memory=False)
    comparisons = pd.read_csv(OUTPUT_DIR / "market_value_comparison_results.csv", low_memory=False)
    bands = pd.read_csv(OUTPUT_DIR / "market_value_risk_bands.csv", low_memory=False)
    decisions = pd.read_csv(OUTPUT_DIR / "market_value_decision_summary.csv", low_memory=False)
    build_checks = pd.read_csv(OUTPUT_DIR / "build_checks.csv")
    sources = pd.read_csv(OUTPUT_DIR / "source_manifest.csv")
    manifest = pd.read_csv(OUTPUT_DIR / "output_manifest.csv")
    run = json.loads((OUTPUT_DIR / "run_summary.json").read_text(encoding="utf-8"))

    add(checks, "target_rows", len(targets) == 4371, len(targets), 4371, "Strict audited cohort is frozen.")
    add(checks, "target_key_unique", targets["transfer_event_id"].is_unique, targets["transfer_event_id"].nunique(), len(targets), "One row per transfer.")
    pre = pd.to_numeric(targets["valuation_pre_eur"], errors="coerce")
    post = pd.to_numeric(targets["valuation_post24_nearest_eur"], errors="coerce")
    log_ratio = np.log(post / pre)
    add(checks, "continuous_target_recomputed", np.allclose(log_ratio, targets["target_log_value_ratio_24m"], atol=1e-12), float(np.abs(log_ratio - targets["target_log_value_ratio_24m"]).max()), 0, "Continuous target is exact log value ratio.")
    pct = post / pre - 1
    add(checks, "percentage_target_recomputed", np.allclose(pct, targets["target_value_change_24m_pct"], atol=1e-12), float(np.abs(pct - targets["target_value_change_24m_pct"]).max()), 0, "Interpretive percentage target is exact.")
    for threshold in [10, 25, 50]:
        boundary = np.nextafter(pre * (1 - threshold / 100), np.inf)
        recomputed = post.le(boundary).astype(int)
        stored = targets[f"target_downside_{threshold}pct"].astype(int)
        add(checks, f"downside_{threshold}pct_recomputed", recomputed.equals(stored), int(recomputed.sum()), int(stored.sum()), "Binary threshold is exact.")
    add(checks, "thresholds_nested", (targets["target_downside_50pct"] <= targets["target_downside_25pct"]).all() and (targets["target_downside_25pct"] <= targets["target_downside_10pct"]).all(), True, True, "Severe downside is nested.")

    audit = pd.read_csv(AUDIT_PATH, usecols=["transfer_event_id", "eligible_market_value_downside_24m"])
    audit_ids = set(audit.loc[as_bool(audit["eligible_market_value_downside_24m"]), "transfer_event_id"].astype(str))
    add(checks, "audit_membership_exact", set(targets["transfer_event_id"].astype(str)) == audit_ids, len(targets), len(audit_ids), "Model cohort exactly matches feasibility gate.")

    counts = features.groupby("feature_set")["raw_feature_count"].first().to_dict()
    add(checks, "feature_sets_complete", set(counts) == set(MODEL_ORDER), ";".join(sorted(counts)), ";".join(sorted(MODEL_ORDER)), "All nested blocks are documented.")
    nested = True
    prior: set[str] = set()
    for model in MODEL_ORDER:
        current = set(features.loc[(features["feature_set"].eq(model)) & features["feature"].notna() & features["feature"].ne(""), "feature"])
        nested &= prior.issubset(current)
        prior = current
    add(checks, "feature_sets_nested", nested, nested, True, "Each block only adds fields.")
    forbidden = features["feature"].fillna("").str.lower().str.contains("post12|post24|target_|eligible_", regex=True)
    add(checks, "feature_manifest_leakage_safe", not forbidden.any(), ";".join(features.loc[forbidden, "feature"].astype(str)), "none", "No post-transfer or target predictors.")
    timing_safe = ~features["timing_classification"].fillna("").str.lower().str.contains("after transfer|post-transfer")
    add(checks, "feature_timing_safe", timing_safe.all(), int((~timing_safe).sum()), 0, "Every modeled feature is transfer-time safe.")

    prediction_key = ["endpoint", "origin_id", "model_variant", "transfer_event_id"]
    add(checks, "prediction_key_unique", not predictions.duplicated(prediction_key).any(), int(predictions.duplicated(prediction_key).sum()), 0, "No duplicate forecast.")
    add(checks, "endpoint_set", set(predictions["endpoint"]) == set(ENDPOINT_KINDS), ";".join(sorted(set(predictions["endpoint"]))), ";".join(sorted(ENDPOINT_KINDS)), "All declared targets are modeled.")
    add(checks, "model_set", set(predictions["model_variant"]) == set(MODEL_ORDER), ";".join(sorted(set(predictions["model_variant"]))), ";".join(sorted(MODEL_ORDER)), "All feature blocks are evaluated.")
    eval_year_ok = all((predictions.loc[predictions["origin_id"].eq(origin), "evaluation_year"] == year).all() for origin, year in ORIGIN_EVAL_YEARS.items())
    add(checks, "origin_evaluation_years", eval_year_ok, eval_year_ok, True, "Rolling origins map to the next season.")
    alignment = predictions.groupby(["endpoint", "origin_id", "model_variant"]).size().groupby(level=[0, 1]).nunique().eq(1).all()
    add(checks, "paired_rows", alignment, alignment, True, "Every model sees identical evaluation transfers.")
    class_rows = predictions["endpoint_kind"].eq("classification")
    add(checks, "probability_bounds", predictions.loc[class_rows, "prediction"].between(0, 1).all(), f"{predictions.loc[class_rows, 'prediction'].min()}..{predictions.loc[class_rows, 'prediction'].max()}", "0..1", "All classifier outputs are probabilities.")
    baseline = predictions.loc[predictions["model_variant"].eq(MODEL_ORDER[0])]
    constant_ok = baseline.groupby(["endpoint", "origin_id"])["prediction"].nunique().eq(1).all()
    add(checks, "chronological_baselines_constant", constant_ok, constant_ok, True, "Baseline uses only fit-window outcome distribution.")

    selected_ok = True
    for (endpoint, origin_id, model), group in tuning.groupby(["endpoint", "origin_id", "model_variant"]):
        ordered = group.sort_values(["validation_primary_loss", "validation_secondary_loss", "model_family", "hyperparameters"])
        best = ordered.iloc[0]
        reported = origins.loc[
            origins["endpoint"].eq(endpoint) & origins["origin_id"].eq(origin_id)
            & origins["model_variant"].eq(model)
        ].iloc[0]
        selected_ok &= reported["selected_family"] == best["model_family"]
        selected_ok &= reported["selected_hyperparameters"] == best["hyperparameters"]
    add(checks, "validation_selection_exact", selected_ok, selected_ok, True, "Family and hyperparameters minimize prior-season loss with deterministic tie-breaks.")

    metrics_ok = True
    for _, row in origins.iterrows():
        group = predictions.loc[
            predictions["endpoint"].eq(row["endpoint"]) & predictions["origin_id"].eq(row["origin_id"])
            & predictions["model_variant"].eq(row["model_variant"])
        ]
        observed = regression_metrics(group["actual"], group["prediction"]) if row["endpoint_kind"] == "regression" else classification_metrics(group["actual"], group["prediction"])
        for metric, value in observed.items():
            metrics_ok &= np.isclose(float(row[metric]), float(value), atol=1e-12, equal_nan=True)
    add(checks, "origin_metrics_recomputed", metrics_ok, metrics_ok, True, "Every origin metric recomputes from predictions.")

    comparison_ok = True
    for _, row in comparisons.iterrows():
        years = [int(value) for value in str(row["evaluation_years"]).split(";")]
        subset = predictions.loc[predictions["endpoint"].eq(row["endpoint"]) & predictions["evaluation_year"].isin(years)]
        keys = ["origin_id", "evaluation_year", "transfer_event_id", "player_id"]
        left = subset.loc[subset["model_variant"].eq(row["baseline_model"]), keys + ["actual", "prediction"]]
        right = subset.loc[subset["model_variant"].eq(row["candidate_model"]), keys + ["prediction"]]
        pair = left.merge(right, on=keys, suffixes=("_base", "_candidate"), validate="one_to_one")
        if row["endpoint_kind"] == "regression":
            base_loss = np.abs(pair["actual"] - pair["prediction_base"]).mean()
            candidate_loss = np.abs(pair["actual"] - pair["prediction_candidate"]).mean()
        else:
            base_loss = np.square(pair["actual"] - pair["prediction_base"]).mean()
            candidate_loss = np.square(pair["actual"] - pair["prediction_candidate"]).mean()
        comparison_ok &= np.isclose(row["baseline_loss"], base_loss, atol=1e-12)
        comparison_ok &= np.isclose(row["candidate_loss"], candidate_loss, atol=1e-12)
        comparison_ok &= np.isclose(row["loss_improvement"], base_loss - candidate_loss, atol=1e-12)
    add(checks, "comparison_losses_recomputed", comparison_ok, comparison_ok, True, "All paired improvements are exact.")
    ci_valid = (comparisons["cluster_ci_lower_95"] <= comparisons["cluster_ci_upper_95"]).all() and comparisons["cluster_probability_candidate_better"].between(0, 1).all()
    add(checks, "cluster_interval_validity", ci_valid, ci_valid, True, "Cluster intervals and probabilities are valid.")

    decision_ok = True
    pooled_total = comparisons.loc[comparisons["window"].eq("pooled_four") & comparisons["comparison"].eq("M5_vs_M0_total")].set_index("endpoint")
    for row in decisions.itertuples(index=False):
        total = pooled_total.loc[row.endpoint]
        expected = "validated_predictive_signal" if total.cluster_ci_lower_95 > 0 and total.origin_wins >= 3 else ("promising_but_uncertain" if total.loss_improvement > 0 else "not_validated")
        decision_ok &= row.status == expected
    add(checks, "decision_rules_recomputed", decision_ok, decision_ok, True, "Endpoint status follows frozen CI/origin rule.")
    band_counts = bands.groupby("endpoint")["rows"].sum()
    full_counts = predictions.loc[predictions["model_variant"].eq(MODEL_ORDER[-1])].groupby("endpoint").size()
    add(checks, "risk_band_counts", band_counts.equals(full_counts), band_counts.to_dict(), full_counts.to_dict(), "Risk bands cover every full-model forecast once.")
    add(checks, "risk_band_levels", set(bands["risk_band"]) == {1, 2, 3, 4, 5}, ";".join(map(str, sorted(set(bands["risk_band"])))), "1;2;3;4;5", "Five within-origin bands are present.")
    add(checks, "runner_checks_pass", as_bool(build_checks["passed"]).all(), int(as_bool(build_checks["passed"]).sum()), len(build_checks), "All runner invariants passed.")

    source_map = sources.set_index("source")
    source_ok = True
    for path in SOURCE_PATHS:
        key = path.relative_to(ROOT).as_posix()
        source_ok &= key in source_map.index
        source_ok &= int(source_map.loc[key, "size_bytes"]) == path.stat().st_size
        source_ok &= source_map.loc[key, "sha256"] == sha256_file(path)
    add(checks, "source_manifest_hashes", source_ok, source_ok, True, "Inputs match the model run.")
    manifest_ok = True
    for _, row in manifest.iterrows():
        path = OUTPUT_DIR / row["file"]
        manifest_ok &= path.exists() and path.stat().st_size == int(row["size_bytes"])
        manifest_ok &= sha256_file(path) == row["sha256"]
    add(checks, "output_manifest_hashes", manifest_ok, manifest_ok, True, "Runner outputs match their hashes.")
    add(checks, "protected_outputs_unchanged", run["protected_outputs_unchanged"] is True, run["protected_outputs_unchanged"], True, "Existing verified outputs were not altered.")

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

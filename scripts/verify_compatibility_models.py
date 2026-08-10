"""Independently verify the first MoveMaker compatibility model benchmark."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT / "Data" / "processed" / "compatibility_targets"
MODEL_DIR = ROOT / "Data" / "processed" / "compatibility_models"
MATRIX_PATH = TARGET_DIR / "transfer_compatibility_strict_model_matrix.csv"

ALPHA_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
RANDOM_SEED = 20260809
BOOTSTRAP_REPETITIONS = 5000

TARGETS = {
    "opportunity": (
        "target_destination_minutes_share",
        "eligible_opportunity_model_primary",
    ),
    "performance": (
        "target_role_performance_index",
        "eligible_performance_model_primary",
    ),
    "adaptation": (
        "target_performance_change_rolling3",
        "eligible_adaptation_model_primary",
    ),
}
MODELS = ["mean_only", "player_history_baseline", "strict_compatibility"]
SPLITS = ["train", "validation", "test"]
FORBIDDEN_PREFIXES = (
    "target_",
    "eligible_",
    "opportunity_model_exclusion",
    "performance_model_exclusion",
    "adaptation_model_exclusion",
    "roster_",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    if len(actual) < 2 or np.std(actual) < 1e-12 or np.std(predicted) < 1e-12:
        return float("nan")
    return float(np.corrcoef(actual, predicted)[0, 1])


def recompute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    residual = actual - predicted
    denominator = float(np.sum(np.square(actual - np.mean(actual))))
    ranks_actual = pd.Series(actual).rank(method="average").to_numpy()
    ranks_predicted = pd.Series(predicted).rank(method="average").to_numpy()
    return {
        "n": int(len(actual)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "r2": float(1.0 - np.sum(np.square(residual)) / denominator) if denominator > 1e-12 else float("nan"),
        "pearson": correlation(actual, predicted),
        "spearman": correlation(ranks_actual, ranks_predicted),
        "actual_mean": float(np.mean(actual)),
        "predicted_mean": float(np.mean(predicted)),
        "actual_std": float(np.std(actual, ddof=1)) if len(actual) > 1 else float("nan"),
    }


def close(left: Any, right: Any, tolerance: float = 1e-11) -> bool:
    try:
        left_float = float(left)
        right_float = float(right)
    except (TypeError, ValueError):
        return left == right
    if math.isnan(left_float) and math.isnan(right_float):
        return True
    return abs(left_float - right_float) <= tolerance


def paired_bootstrap(actual: np.ndarray, baseline: np.ndarray, compatibility: np.ndarray, seed: int) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    n = len(actual)
    indices = rng.integers(0, n, size=(BOOTSTRAP_REPETITIONS, n))
    baseline_abs = np.abs(actual - baseline)
    compatibility_abs = np.abs(actual - compatibility)
    mae_improvements = np.mean(baseline_abs[indices] - compatibility_abs[indices], axis=1)
    baseline_sq = np.square(actual - baseline)
    compatibility_sq = np.square(actual - compatibility)
    rmse_improvements = np.sqrt(np.mean(baseline_sq[indices], axis=1)) - np.sqrt(
        np.mean(compatibility_sq[indices], axis=1)
    )
    return {
        "test_n": n,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "mae_improvement": float(np.mean(baseline_abs - compatibility_abs)),
        "mae_improvement_ci_lower_95": float(np.quantile(mae_improvements, 0.025)),
        "mae_improvement_ci_upper_95": float(np.quantile(mae_improvements, 0.975)),
        "mae_probability_compatibility_better": float(np.mean(mae_improvements > 0)),
        "rmse_improvement": float(np.sqrt(np.mean(baseline_sq)) - np.sqrt(np.mean(compatibility_sq))),
        "rmse_improvement_ci_lower_95": float(np.quantile(rmse_improvements, 0.025)),
        "rmse_improvement_ci_upper_95": float(np.quantile(rmse_improvements, 0.975)),
        "rmse_probability_compatibility_better": float(np.mean(rmse_improvements > 0)),
    }


def run() -> None:
    required = [
        "model_feature_dictionary.csv",
        "model_target_summary.csv",
        "model_metrics.csv",
        "model_test_comparison.csv",
        "model_test_bootstrap.csv",
        "model_validation_tuning.csv",
        "model_predictions.csv",
        "model_test_predictions_wide.csv",
        "model_coefficients.csv",
        "model_top_compatibility_features.csv",
        "model_build_checks.csv",
        "model_run_summary.json",
        "source_manifest.csv",
        "README.md",
    ]
    checks: list[dict[str, Any]] = []

    def add(check: str, passed: bool, observed: Any, expected: Any, detail: str) -> None:
        checks.append(
            {
                "check": check,
                "severity": "blocking",
                "passed": bool(passed),
                "observed": observed,
                "expected": expected,
                "detail": detail,
            }
        )

    missing = [name for name in required if not (MODEL_DIR / name).exists()]
    add(
        "required_model_outputs_exist",
        not missing,
        ";".join(missing) if missing else "all present",
        "all present",
        "All auditable model artifacts must exist before result verification.",
    )
    if missing:
        raise RuntimeError(f"Missing model outputs: {missing}")

    matrix = pd.read_csv(MATRIX_PATH, low_memory=False)
    features = pd.read_csv(MODEL_DIR / "model_feature_dictionary.csv", low_memory=False)
    summary = pd.read_csv(MODEL_DIR / "model_target_summary.csv", low_memory=False)
    model_metrics = pd.read_csv(MODEL_DIR / "model_metrics.csv", low_memory=False)
    comparison = pd.read_csv(MODEL_DIR / "model_test_comparison.csv", low_memory=False)
    bootstrap = pd.read_csv(MODEL_DIR / "model_test_bootstrap.csv", low_memory=False)
    tuning = pd.read_csv(MODEL_DIR / "model_validation_tuning.csv", low_memory=False)
    predictions = pd.read_csv(MODEL_DIR / "model_predictions.csv", low_memory=False)
    test_prediction_wide = pd.read_csv(MODEL_DIR / "model_test_predictions_wide.csv", low_memory=False)
    coefficients = pd.read_csv(MODEL_DIR / "model_coefficients.csv", low_memory=False)
    top_features = pd.read_csv(MODEL_DIR / "model_top_compatibility_features.csv", low_memory=False)
    build_checks = pd.read_csv(MODEL_DIR / "model_build_checks.csv", low_memory=False)
    source_manifest = pd.read_csv(MODEL_DIR / "source_manifest.csv", low_memory=False)

    add(
        "build_checks_all_pass",
        bool_series(build_checks["passed"]).all(),
        int((~bool_series(build_checks["passed"])).sum()),
        0,
        "The training-time assertions all passed.",
    )

    source_hash_failures = []
    for _, row in source_manifest.iterrows():
        path = ROOT / row["source_file"]
        if not path.exists() or sha256_file(path) != row["sha256"]:
            source_hash_failures.append(row["source_file"])
    add(
        "source_manifest_hashes_match",
        not source_hash_failures,
        ";".join(source_hash_failures) if source_hash_failures else "all match",
        "all match",
        "The verifier is reading the exact target matrix and dictionaries used by training.",
    )

    baseline_features = features.loc[bool_series(features["in_player_history_baseline"]), "raw_feature"].tolist()
    compatibility_features = features.loc[bool_series(features["in_strict_compatibility_model"]), "raw_feature"].tolist()
    add(
        "baseline_features_subset_compatibility",
        set(baseline_features).issubset(compatibility_features),
        len(set(baseline_features) - set(compatibility_features)),
        0,
        "The compatibility model is a nested incremental comparison.",
    )
    incremental = sorted(set(compatibility_features) - set(baseline_features))
    add(
        "incremental_features_are_compatibility_only",
        all(name.startswith(("preference_", "strict_")) for name in incremental),
        ";".join(name for name in incremental if not name.startswith(("preference_", "strict_"))),
        "preference_ or strict_ only",
        "Only teammate-preference and destination-context features distinguish the full model.",
    )
    forbidden = [name for name in compatibility_features if name.startswith(FORBIDDEN_PREFIXES)]
    add(
        "no_forbidden_features",
        not forbidden,
        ";".join(forbidden),
        "none",
        "No outcome, eligibility, exclusion, or target-roster feature is used.",
    )
    add(
        "declared_timing_is_pretransfer",
        features["timing_policy"].isin(["available_at_transfer", "strict_preseason_safe"]).all(),
        ";".join(sorted(features["timing_policy"].dropna().unique())),
        "available_at_transfer;strict_preseason_safe",
        "Every declared raw predictor is available by the transfer decision point.",
    )

    metric_failures: list[str] = []
    actual_failures: list[str] = []
    key_failures: list[str] = []
    fit_scope_failures: list[str] = []
    for target_index, (target_key, (target_column, eligibility_column)) in enumerate(TARGETS.items()):
        eligible = bool_series(matrix[eligibility_column]) & matrix[target_column].notna()
        source = matrix.loc[
            eligible,
            ["canonical_performance_id", "recommended_time_split", target_column],
        ].copy()
        source[target_column] = pd.to_numeric(source[target_column], errors="coerce")
        expected_summary = summary.loc[summary["target_key"].eq(target_key)].iloc[0]
        for split in SPLITS:
            expected_count = int(source["recommended_time_split"].eq(split).sum())
            summary_count = int(expected_summary[f"{split}_rows"])
            add(
                f"{target_key}_{split}_cohort_count_reconciles",
                expected_count == summary_count,
                summary_count,
                expected_count,
                "Model summary rows reconcile to the independently filtered primary target cohort.",
            )
            split_predictions = predictions.loc[
                predictions["target_key"].eq(target_key)
                & predictions["recommended_time_split"].eq(split)
            ]
            keys = {
                model: set(
                    split_predictions.loc[
                        split_predictions["model_variant"].eq(model), "canonical_performance_id"
                    ].astype(str)
                )
                for model in MODELS
            }
            if len({frozenset(value) for value in keys.values()}) != 1:
                key_failures.append(f"{target_key}:{split}")
            source_values = source.loc[
                source["recommended_time_split"].eq(split)
            ].set_index(source.loc[source["recommended_time_split"].eq(split), "canonical_performance_id"].astype(str))[target_column]
            for model in MODELS:
                rows = split_predictions.loc[split_predictions["model_variant"].eq(model)].copy()
                rows["key"] = rows["canonical_performance_id"].astype(str)
                expected_actual = rows["key"].map(source_values)
                if not np.allclose(rows["actual"], expected_actual, atol=1e-12, rtol=0, equal_nan=False):
                    actual_failures.append(f"{target_key}:{split}:{model}")
                expected_scope = "train_plus_validation" if split == "test" else "train"
                if not rows["fit_scope"].eq(expected_scope).all():
                    fit_scope_failures.append(f"{target_key}:{split}:{model}")
                recalculated = recompute_metrics(
                    rows["actual"].to_numpy(dtype=float), rows["prediction"].to_numpy(dtype=float)
                )
                reported = model_metrics.loc[
                    model_metrics["target_key"].eq(target_key)
                    & model_metrics["split"].eq(split)
                    & model_metrics["model_variant"].eq(model)
                ]
                if len(reported) != 1:
                    metric_failures.append(f"{target_key}:{split}:{model}:row_count")
                    continue
                reported_row = reported.iloc[0]
                for metric_name, value in recalculated.items():
                    if not close(value, reported_row[metric_name]):
                        metric_failures.append(f"{target_key}:{split}:{model}:{metric_name}")

        test_rows = predictions.loc[
            predictions["target_key"].eq(target_key)
            & predictions["recommended_time_split"].eq("test")
        ]
        baseline_rows = test_rows.loc[test_rows["model_variant"].eq("player_history_baseline")].sort_values(
            "canonical_performance_id"
        )
        compatibility_rows = test_rows.loc[test_rows["model_variant"].eq("strict_compatibility")].sort_values(
            "canonical_performance_id"
        )
        recomputed_bootstrap = paired_bootstrap(
            baseline_rows["actual"].to_numpy(dtype=float),
            baseline_rows["prediction"].to_numpy(dtype=float),
            compatibility_rows["prediction"].to_numpy(dtype=float),
            RANDOM_SEED + target_index,
        )
        reported_bootstrap = bootstrap.loc[bootstrap["target_key"].eq(target_key)].iloc[0]
        bootstrap_matches = all(
            close(value, reported_bootstrap[name]) for name, value in recomputed_bootstrap.items()
        )
        add(
            f"{target_key}_paired_bootstrap_exact",
            bootstrap_matches,
            "match" if bootstrap_matches else "mismatch",
            "match",
            "The paired test-error uncertainty interval is independently regenerated from predictions.",
        )

    add(
        "prediction_keys_match_across_models",
        not key_failures,
        ";".join(key_failures),
        "none",
        "Mean, baseline, and compatibility variants score identical rows in every split.",
    )
    add(
        "prediction_actuals_match_source_targets",
        not actual_failures,
        ";".join(actual_failures),
        "none",
        "Every exported actual outcome matches the verified target matrix by canonical key.",
    )
    add(
        "prediction_fit_scopes_are_chronological",
        not fit_scope_failures,
        ";".join(fit_scope_failures),
        "none",
        "Train/validation predictions use train fits; test predictions use train+validation fits.",
    )
    add(
        "all_reported_metrics_recompute",
        not metric_failures,
        ";".join(metric_failures),
        "none",
        "All sample sizes, errors, R-squared values, and correlations are rebuilt from row predictions.",
    )

    tuning_failures = []
    for target_key in TARGETS:
        for model in ["player_history_baseline", "strict_compatibility"]:
            rows = tuning.loc[
                tuning["target_key"].eq(target_key) & tuning["model_variant"].eq(model)
            ].copy()
            if set(rows["alpha"].astype(float)) != set(ALPHA_GRID):
                tuning_failures.append(f"{target_key}:{model}:grid")
                continue
            rows = rows.sort_values(["mae", "rmse", "alpha"], ascending=[True, True, False])
            selected = float(rows.iloc[0]["alpha"])
            reported = model_metrics.loc[
                model_metrics["target_key"].eq(target_key)
                & model_metrics["model_variant"].eq(model)
                & model_metrics["split"].eq("test"),
                "selected_alpha",
            ]
            if len(reported) != 1 or not close(selected, reported.iloc[0]):
                tuning_failures.append(f"{target_key}:{model}:selection")
    add(
        "validation_alpha_selection_recomputes",
        not tuning_failures,
        ";".join(tuning_failures),
        "none",
        "Selected shrinkage minimizes 2022 validation MAE with declared tie-breaking.",
    )
    add(
        "tuning_contains_no_test_fields",
        all("test" not in column.lower() for column in tuning.columns),
        ";".join(tuning.columns),
        "no test columns",
        "The hyperparameter-selection table has no test-set metrics or sample fields.",
    )

    comparison_failures = []
    for _, row in comparison.iterrows():
        target_key = row["target_key"]
        metric_name = row["metric"]
        reported_metrics = model_metrics.loc[
            model_metrics["target_key"].eq(target_key) & model_metrics["split"].eq("test")
        ].set_index("model_variant")
        baseline_value = reported_metrics.loc["player_history_baseline", metric_name]
        compatibility_value = reported_metrics.loc["strict_compatibility", metric_name]
        preferred = (
            baseline_value - compatibility_value
            if metric_name in {"mae", "rmse"}
            else compatibility_value - baseline_value
        )
        if not (
            close(baseline_value, row["player_history_baseline"])
            and close(compatibility_value, row["strict_compatibility"])
            and close(compatibility_value - baseline_value, row["compatibility_minus_baseline"])
            and close(preferred, row["improvement_in_preferred_direction"])
        ):
            comparison_failures.append(f"{target_key}:{metric_name}")
    add(
        "test_comparison_reconciles",
        not comparison_failures,
        ";".join(comparison_failures),
        "none",
        "The compact held-out comparison exactly reconciles to reported test metrics.",
    )

    coefficient_features = set(coefficients["raw_feature"])
    add(
        "coefficient_features_are_declared",
        coefficient_features.issubset(compatibility_features),
        ";".join(sorted(coefficient_features - set(compatibility_features))),
        "none",
        "Every encoded coefficient traces back to a declared raw feature.",
    )
    add(
        "coefficients_are_final_fit_only",
        coefficients["fit_scope"].eq("train_plus_validation").all(),
        ";".join(sorted(coefficients["fit_scope"].unique())),
        "train_plus_validation",
        "Attribution values correspond to the models used for the 2023 test predictions.",
    )
    wide_failures = []
    for _, row in test_prediction_wide.iterrows():
        subset = predictions.loc[
            predictions["canonical_performance_id"].astype(str).eq(str(row["canonical_performance_id"]))
            & predictions["target_key"].eq(row["target_key"])
            & predictions["recommended_time_split"].eq("test")
        ].set_index("model_variant")
        if any(
            model not in subset.index or not close(row[model], subset.loc[model, "prediction"])
            for model in MODELS
        ):
            wide_failures.append(f"{row['canonical_performance_id']}:{row['target_key']}")
            continue
        expected_improvement = abs(row["actual"] - row["player_history_baseline"]) - abs(
            row["actual"] - row["strict_compatibility"]
        )
        if not close(expected_improvement, row["compatibility_error_improvement"]):
            wide_failures.append(f"{row['canonical_performance_id']}:{row['target_key']}:improvement")
    add(
        "wide_test_predictions_reconcile",
        not wide_failures,
        ";".join(wide_failures[:20]),
        "none",
        "The analyst-friendly paired test table reconciles to long-form predictions and errors.",
    )
    add(
        "top_features_trace_to_coefficients",
        set(top_features["encoded_feature"]).issubset(set(coefficients["encoded_feature"])),
        len(set(top_features["encoded_feature"]) - set(coefficients["encoded_feature"])),
        0,
        "Every published top compatibility signal is a final-fit standardized coefficient.",
    )
    add(
        "opportunity_predictions_bounded",
        predictions.loc[predictions["target_key"].eq("opportunity"), "prediction"].between(0, 1).all(),
        "within bounds",
        "within [0,1]",
        "The bounded opportunity target is scored on its valid support.",
    )

    verification = pd.DataFrame(checks)
    failures = verification.loc[~verification["passed"], "check"].tolist()
    status = "pass" if not failures else "fail"
    verification.to_csv(MODEL_DIR / "independent_verification.csv", index=False)
    verification_summary = {
        "status": status,
        "checks": int(len(verification)),
        "blocking_checks": int(len(verification)),
        "blocking_failures": int(len(failures)),
        "verified_prediction_rows": int(len(predictions)),
        "verified_targets": int(len(TARGETS)),
    }
    with (MODEL_DIR / "independent_verification.json").open("w", encoding="utf-8") as handle:
        json.dump(verification_summary, handle, indent=2)

    output_paths = sorted(
        path for path in MODEL_DIR.iterdir() if path.is_file() and path.name != "output_manifest.csv"
    )
    pd.DataFrame(
        [
            {
                "output_file": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in output_paths
        ]
    ).to_csv(MODEL_DIR / "output_manifest.csv", index=False)

    print(json.dumps(verification_summary, indent=2))
    if failures:
        raise RuntimeError(f"Independent verification failures: {failures}")


if __name__ == "__main__":
    run()

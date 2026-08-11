"""Train first-version MoveMaker player-history and compatibility benchmarks.

The experiment is deliberately conservative:
* one model per prediction target;
* train through 2021, tune on 2022, and evaluate once on 2023;
* a market-aware player-history baseline versus the same model plus strict,
  preseason-safe compatibility features;
* regularized linear regression implemented with NumPy so the run has no
  external machine-learning dependency.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "Data" / "processed" / "compatibility_targets"
OUTPUT_DIR = ROOT / "Data" / "processed" / "compatibility_models"

MATRIX_PATH = SOURCE_DIR / "transfer_compatibility_strict_model_matrix.csv"
FIELD_DICTIONARY_PATH = SOURCE_DIR / "target_field_dictionary.csv"
TARGET_VERIFICATION_PATH = SOURCE_DIR / "independent_verification.json"

RANDOM_SEED = 20260809
BOOTSTRAP_REPETITIONS = 5000
ALPHA_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

TARGETS = [
    {
        "target_key": "opportunity",
        "target_column": "target_destination_minutes_share",
        "eligibility_column": "eligible_opportunity_model_primary",
        "clip_lower": 0.0,
        "clip_upper": 1.0,
        "interpretation": "Share of full destination club-season league minutes earned.",
    },
    {
        "target_key": "performance",
        "target_column": "target_role_performance_index",
        "eligibility_column": "eligible_performance_model_primary",
        "clip_lower": None,
        "clip_upper": None,
        "interpretation": "Role- and season-standardized destination performance level.",
    },
    {
        "target_key": "adaptation",
        "target_column": "target_performance_change_rolling3",
        "eligibility_column": "eligible_adaptation_model_primary",
        "clip_lower": None,
        "clip_upper": None,
        "interpretation": "Destination performance minus weighted prior-three-season form.",
    },
]

TRANSFER_BASELINE_FEATURES = [
    "transfer_days_from_target_season_start",
    "canonical_transfer_type",
    "canonical_transfer_fee_status",
    "canonical_transfer_fee_eur",
    "listed_market_value_at_transfer_eur",
    "age_at_transfer",
    "position",
    "sub_position",
    "foot",
    "height_in_cm",
]

FORBIDDEN_PREDICTOR_PREFIXES = (
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


def clean_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    if len(actual) < 2 or np.std(actual) < 1e-12 or np.std(predicted) < 1e-12:
        return float("nan")
    return float(np.corrcoef(actual, predicted)[0, 1])


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = actual - predicted
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    denominator = float(np.sum(np.square(actual - np.mean(actual))))
    r2 = float(1.0 - np.sum(np.square(residual)) / denominator) if denominator > 1e-12 else float("nan")
    rank_actual = pd.Series(actual).rank(method="average").to_numpy()
    rank_predicted = pd.Series(predicted).rank(method="average").to_numpy()
    return {
        "n": int(len(actual)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson": correlation(actual, predicted),
        "spearman": correlation(rank_actual, rank_predicted),
        "actual_mean": float(np.mean(actual)),
        "predicted_mean": float(np.mean(predicted)),
        "actual_std": float(np.std(actual, ddof=1)) if len(actual) > 1 else float("nan"),
    }


def apply_bounds(predicted: np.ndarray, lower: float | None, upper: float | None) -> np.ndarray:
    result = np.asarray(predicted, dtype=float)
    if lower is not None:
        result = np.maximum(result, lower)
    if upper is not None:
        result = np.minimum(result, upper)
    return result


@dataclass
class Preprocessor:
    raw_features: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    medians: dict[str, float]
    numeric_missing_indicators: set[str]
    categories: dict[str, list[str]]
    encoded_names_all: list[str]
    encoded_raw_all: list[str]
    means_all: np.ndarray
    scales_all: np.ndarray
    keep_mask: np.ndarray

    @property
    def encoded_names(self) -> list[str]:
        return [name for name, keep in zip(self.encoded_names_all, self.keep_mask) if keep]

    @property
    def encoded_raw_features(self) -> list[str]:
        return [name for name, keep in zip(self.encoded_raw_all, self.keep_mask) if keep]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        blocks: list[np.ndarray] = []
        numeric = set(self.numeric_features)
        # Preserve raw feature order. fit_preprocessor builds its means, scales,
        # encoded names, and keep mask in this order; regrouping numeric and
        # categorical fields here would apply those statistics to the wrong
        # transformed columns whenever a feature set mixes types.
        for feature in self.raw_features:
            if feature in numeric:
                values = pd.to_numeric(frame[feature], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                )
                missing = values.isna().to_numpy(dtype=float)
                blocks.append(values.fillna(self.medians[feature]).to_numpy(dtype=float).reshape(-1, 1))
                if feature in self.numeric_missing_indicators:
                    blocks.append(missing.reshape(-1, 1))
            else:
                values = frame[feature].astype("string").fillna("__MISSING__").astype(str)
                known = set(self.categories[feature]) - {"__OTHER__"}
                values = values.where(values.isin(known), "__OTHER__")
                for category in self.categories[feature]:
                    blocks.append(values.eq(category).to_numpy(dtype=float).reshape(-1, 1))
        if not blocks:
            return np.zeros((len(frame), 0), dtype=float)
        encoded = np.concatenate(blocks, axis=1)
        standardized = (encoded - self.means_all) / self.scales_all
        return standardized[:, self.keep_mask]


def fit_preprocessor(frame: pd.DataFrame, raw_features: list[str]) -> Preprocessor:
    numeric_features: list[str] = []
    categorical_features: list[str] = []
    medians: dict[str, float] = {}
    missing_indicators: set[str] = set()
    categories: dict[str, list[str]] = {}
    encoded_names: list[str] = []
    encoded_raw: list[str] = []
    blocks: list[np.ndarray] = []

    for feature in raw_features:
        series = frame[feature]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            numeric_features.append(feature)
            values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
            finite = values.dropna()
            median = float(finite.median()) if len(finite) else 0.0
            medians[feature] = median
            missing = values.isna().to_numpy(dtype=float)
            blocks.append(values.fillna(median).to_numpy(dtype=float).reshape(-1, 1))
            encoded_names.append(feature)
            encoded_raw.append(feature)
            if missing.any():
                missing_indicators.add(feature)
                blocks.append(missing.reshape(-1, 1))
                encoded_names.append(f"{feature}__missing")
                encoded_raw.append(feature)
        else:
            categorical_features.append(feature)
            values = series.astype("string").fillna("__MISSING__").astype(str)
            feature_categories = sorted(values.unique().tolist())
            if "__OTHER__" not in feature_categories:
                feature_categories.append("__OTHER__")
            categories[feature] = feature_categories
            for category in feature_categories:
                blocks.append(values.eq(category).to_numpy(dtype=float).reshape(-1, 1))
                encoded_names.append(f"{feature}=={category}")
                encoded_raw.append(feature)

    encoded = np.concatenate(blocks, axis=1) if blocks else np.zeros((len(frame), 0), dtype=float)
    means = np.mean(encoded, axis=0) if encoded.shape[1] else np.array([], dtype=float)
    scales = np.std(encoded, axis=0, ddof=0) if encoded.shape[1] else np.array([], dtype=float)
    keep = np.isfinite(scales) & (scales > 1e-12)
    safe_scales = np.where(keep, scales, 1.0)
    return Preprocessor(
        raw_features=raw_features,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        medians=medians,
        numeric_missing_indicators=missing_indicators,
        categories=categories,
        encoded_names_all=encoded_names,
        encoded_raw_all=encoded_raw,
        means_all=means,
        scales_all=safe_scales,
        keep_mask=keep,
    )


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[float, np.ndarray]:
    y = np.asarray(y, dtype=float)
    intercept = float(np.mean(y))
    if x.shape[1] == 0:
        return intercept, np.zeros(0, dtype=float)
    gram = (x.T @ x) / len(x)
    rhs = (x.T @ (y - intercept)) / len(x)
    beta = np.linalg.solve(gram + alpha * np.eye(x.shape[1]), rhs)
    return intercept, beta


def predict_ridge(x: np.ndarray, intercept: float, beta: np.ndarray) -> np.ndarray:
    return intercept + x @ beta


def feature_group(feature: str) -> str:
    if feature in TRANSFER_BASELINE_FEATURES:
        return "transfer_and_market"
    if feature.startswith("advanced_"):
        return "lagged_player_history"
    if feature.startswith("preference_"):
        return "historical_teammate_preference"
    if feature.startswith("strict_"):
        return "destination_club_compatibility"
    return "other"


def paired_bootstrap(actual: np.ndarray, baseline: np.ndarray, compatibility: np.ndarray, seed: int) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    n = len(actual)
    indices = rng.integers(0, n, size=(BOOTSTRAP_REPETITIONS, n))
    base_abs = np.abs(actual - baseline)
    comp_abs = np.abs(actual - compatibility)
    mae_improvements = np.mean(base_abs[indices] - comp_abs[indices], axis=1)
    base_sq = np.square(actual - baseline)
    comp_sq = np.square(actual - compatibility)
    rmse_improvements = np.sqrt(np.mean(base_sq[indices], axis=1)) - np.sqrt(np.mean(comp_sq[indices], axis=1))
    return {
        "test_n": int(n),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "mae_improvement": float(np.mean(base_abs - comp_abs)),
        "mae_improvement_ci_lower_95": float(np.quantile(mae_improvements, 0.025)),
        "mae_improvement_ci_upper_95": float(np.quantile(mae_improvements, 0.975)),
        "mae_probability_compatibility_better": float(np.mean(mae_improvements > 0)),
        "rmse_improvement": float(np.sqrt(np.mean(base_sq)) - np.sqrt(np.mean(comp_sq))),
        "rmse_improvement_ci_lower_95": float(np.quantile(rmse_improvements, 0.025)),
        "rmse_improvement_ci_upper_95": float(np.quantile(rmse_improvements, 0.975)),
        "rmse_probability_compatibility_better": float(np.mean(rmse_improvements > 0)),
    }


def build_feature_sets(field_dictionary: pd.DataFrame, matrix_columns: list[str]) -> tuple[list[str], list[str], pd.DataFrame]:
    fields = field_dictionary.loc[
        field_dictionary["table"].eq("transfer_compatibility_strict_model_matrix")
    ].copy()
    strict_safe = fields.loc[
        fields["usage"].eq("strict_compatibility_predictor")
        & fields["timing_policy"].eq("strict_preseason_safe"),
        "column",
    ].tolist()
    safe_advanced = [column for column in strict_safe if column.startswith("advanced_") and "source_season" not in column]
    safe_compatibility = [
        column
        for column in strict_safe
        if (column.startswith("preference_") or column.startswith("strict_")) and "source_season" not in column
    ]
    baseline = [column for column in TRANSFER_BASELINE_FEATURES + safe_advanced if column in matrix_columns]
    compatibility = baseline + [column for column in safe_compatibility if column in matrix_columns]
    baseline = list(dict.fromkeys(baseline))
    compatibility = list(dict.fromkeys(compatibility))

    records = []
    for column in compatibility:
        records.append(
            {
                "raw_feature": column,
                "feature_group": feature_group(column),
                "in_player_history_baseline": column in baseline,
                "in_strict_compatibility_model": True,
                "timing_policy": "available_at_transfer" if column in TRANSFER_BASELINE_FEATURES else "strict_preseason_safe",
                "inclusion_reason": (
                    "Common transfer-time or player-history control included in both models."
                    if column in baseline
                    else "Incremental historical preference or destination prior-season compatibility signal."
                ),
            }
        )
    feature_dictionary = pd.DataFrame(records)
    return baseline, compatibility, feature_dictionary


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(MATRIX_PATH, low_memory=False)
    field_dictionary = pd.read_csv(FIELD_DICTIONARY_PATH, low_memory=False)
    with TARGET_VERIFICATION_PATH.open("r", encoding="utf-8") as handle:
        upstream_verification = json.load(handle)
    if upstream_verification.get("status") != "pass":
        raise RuntimeError("Upstream target verification is not passing.")

    baseline_features, compatibility_features, feature_dictionary = build_feature_sets(
        field_dictionary, matrix.columns.tolist()
    )
    model_features = {
        "player_history_baseline": baseline_features,
        "strict_compatibility": compatibility_features,
    }

    checks: list[dict[str, Any]] = []

    def add_check(check: str, passed: bool, observed: Any, expected: Any, detail: str) -> None:
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

    add_check(
        "upstream_target_verification_passes",
        upstream_verification.get("status") == "pass",
        upstream_verification.get("status"),
        "pass",
        "The benchmark only runs from an independently verified target layer.",
    )
    add_check(
        "baseline_is_subset_of_compatibility",
        set(baseline_features).issubset(compatibility_features),
        len(set(baseline_features) - set(compatibility_features)),
        0,
        "The compatibility comparison changes only the added compatibility feature block.",
    )
    add_check(
        "compatibility_has_incremental_features",
        len(compatibility_features) > len(baseline_features),
        len(compatibility_features) - len(baseline_features),
        ">0",
        "Historical teammate preferences and destination prior-season style must add information.",
    )
    forbidden = [
        column
        for column in compatibility_features
        if column.startswith(FORBIDDEN_PREDICTOR_PREFIXES)
    ]
    add_check(
        "no_forbidden_predictor_prefixes",
        not forbidden,
        ";".join(forbidden),
        "none",
        "Targets, eligibility controls, exclusion reasons, and roster scenarios are not predictors.",
    )

    metrics_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    target_summary_rows: list[dict[str, Any]] = []

    for target_index, config in enumerate(TARGETS):
        target_key = config["target_key"]
        target_column = config["target_column"]
        eligibility_column = config["eligibility_column"]
        eligible = bool_series(matrix[eligibility_column]) & matrix[target_column].notna()
        cohort = matrix.loc[eligible].copy()
        cohort[target_column] = pd.to_numeric(cohort[target_column], errors="coerce")
        cohort = cohort.loc[cohort[target_column].notna()].copy()
        cohort = cohort.sort_values(["season_start_year", "canonical_performance_id"]).reset_index(drop=True)
        split_frames = {
            split: cohort.loc[cohort["recommended_time_split"].eq(split)].copy()
            for split in ["train", "validation", "test"]
        }
        for split, frame in split_frames.items():
            add_check(
                f"{target_key}_{split}_nonempty",
                len(frame) > 0,
                len(frame),
                ">0",
                "Every target needs a nonempty chronological split.",
            )
        train = split_frames["train"]
        validation = split_frames["validation"]
        test = split_frames["test"]
        train_validation = pd.concat([train, validation], ignore_index=True)

        ids_by_split = {
            split: set(frame["canonical_performance_id"].astype(str)) for split, frame in split_frames.items()
        }
        overlap = sum(
            len(ids_by_split[left] & ids_by_split[right])
            for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]
        )
        add_check(
            f"{target_key}_split_ids_disjoint",
            overlap == 0,
            overlap,
            0,
            "A destination player-season appears in exactly one chronological split.",
        )

        y_train = train[target_column].to_numpy(dtype=float)
        y_validation = validation[target_column].to_numpy(dtype=float)
        y_test = test[target_column].to_numpy(dtype=float)
        y_train_validation = train_validation[target_column].to_numpy(dtype=float)

        # Mean-only diagnostic establishes the minimum useful benchmark.
        mean_train = float(np.mean(y_train))
        mean_train_validation = float(np.mean(y_train_validation))
        mean_predictions = {
            "train": np.full(len(train), mean_train),
            "validation": np.full(len(validation), mean_train),
            "test": np.full(len(test), mean_train_validation),
        }
        for split, frame in split_frames.items():
            actual = frame[target_column].to_numpy(dtype=float)
            predicted = apply_bounds(
                mean_predictions[split], config["clip_lower"], config["clip_upper"]
            )
            result = metrics(actual, predicted)
            metrics_rows.append(
                {
                    "target_key": target_key,
                    "target_column": target_column,
                    "model_variant": "mean_only",
                    "split": split,
                    "fit_scope": "train" if split != "test" else "train_plus_validation",
                    "selected_alpha": None,
                    "raw_feature_count": 0,
                    "encoded_feature_count": 0,
                    **result,
                }
            )
            for row_index, (_, row) in enumerate(frame.iterrows()):
                prediction_rows.append(
                    {
                        "canonical_performance_id": row["canonical_performance_id"],
                        "canonical_player_name": row["canonical_player_name"],
                        "canonical_club_name": row["canonical_club_name"],
                        "season_start_year": int(row["season_start_year"]),
                        "role_group": row["role_group"],
                        "recommended_time_split": split,
                        "target_key": target_key,
                        "target_column": target_column,
                        "model_variant": "mean_only",
                        "fit_scope": "train" if split != "test" else "train_plus_validation",
                        "actual": actual[row_index],
                        "prediction_raw": mean_predictions[split][row_index],
                        "prediction": predicted[row_index],
                        "residual": actual[row_index] - predicted[row_index],
                        "absolute_error": abs(actual[row_index] - predicted[row_index]),
                    }
                )

        selected_alpha: dict[str, float] = {}
        selected_validation_predictions: dict[str, np.ndarray] = {}
        selected_test_predictions: dict[str, np.ndarray] = {}

        for variant, raw_features in model_features.items():
            preprocessor = fit_preprocessor(train, raw_features)
            x_train = preprocessor.transform(train)
            x_validation = preprocessor.transform(validation)
            candidates: list[tuple[float, float, float]] = []
            for alpha in ALPHA_GRID:
                intercept, beta = fit_ridge(x_train, y_train, alpha)
                validation_raw = predict_ridge(x_validation, intercept, beta)
                validation_prediction = apply_bounds(
                    validation_raw, config["clip_lower"], config["clip_upper"]
                )
                result = metrics(y_validation, validation_prediction)
                tuning_rows.append(
                    {
                        "target_key": target_key,
                        "target_column": target_column,
                        "model_variant": variant,
                        "alpha": alpha,
                        "train_rows": len(train),
                        "validation_rows": len(validation),
                        "raw_feature_count": len(raw_features),
                        "encoded_feature_count": x_train.shape[1],
                        **result,
                    }
                )
                candidates.append((result["mae"], result["rmse"], -alpha))
            best_index = min(range(len(ALPHA_GRID)), key=lambda index: candidates[index])
            alpha = ALPHA_GRID[best_index]
            selected_alpha[variant] = alpha

            train_intercept, train_beta = fit_ridge(x_train, y_train, alpha)
            train_raw = predict_ridge(x_train, train_intercept, train_beta)
            validation_raw = predict_ridge(x_validation, train_intercept, train_beta)
            train_prediction = apply_bounds(train_raw, config["clip_lower"], config["clip_upper"])
            validation_prediction = apply_bounds(
                validation_raw, config["clip_lower"], config["clip_upper"]
            )
            selected_validation_predictions[variant] = validation_prediction

            final_preprocessor = fit_preprocessor(train_validation, raw_features)
            x_train_validation = final_preprocessor.transform(train_validation)
            x_test = final_preprocessor.transform(test)
            final_intercept, final_beta = fit_ridge(x_train_validation, y_train_validation, alpha)
            test_raw = predict_ridge(x_test, final_intercept, final_beta)
            test_prediction = apply_bounds(test_raw, config["clip_lower"], config["clip_upper"])
            selected_test_predictions[variant] = test_prediction

            predictions_by_split = {
                "train": (train_raw, train_prediction),
                "validation": (validation_raw, validation_prediction),
                "test": (test_raw, test_prediction),
            }
            for split, frame in split_frames.items():
                actual = frame[target_column].to_numpy(dtype=float)
                raw_prediction, prediction = predictions_by_split[split]
                result = metrics(actual, prediction)
                metrics_rows.append(
                    {
                        "target_key": target_key,
                        "target_column": target_column,
                        "model_variant": variant,
                        "split": split,
                        "fit_scope": "train" if split != "test" else "train_plus_validation",
                        "selected_alpha": alpha,
                        "raw_feature_count": len(raw_features),
                        "encoded_feature_count": (
                            x_train.shape[1] if split != "test" else x_train_validation.shape[1]
                        ),
                        **result,
                    }
                )
                for row_index, (_, row) in enumerate(frame.iterrows()):
                    prediction_rows.append(
                        {
                            "canonical_performance_id": row["canonical_performance_id"],
                            "canonical_player_name": row["canonical_player_name"],
                            "canonical_club_name": row["canonical_club_name"],
                            "season_start_year": int(row["season_start_year"]),
                            "role_group": row["role_group"],
                            "recommended_time_split": split,
                            "target_key": target_key,
                            "target_column": target_column,
                            "model_variant": variant,
                            "fit_scope": "train" if split != "test" else "train_plus_validation",
                            "actual": actual[row_index],
                            "prediction_raw": raw_prediction[row_index],
                            "prediction": prediction[row_index],
                            "residual": actual[row_index] - prediction[row_index],
                            "absolute_error": abs(actual[row_index] - prediction[row_index]),
                        }
                    )

            for encoded_name, raw_feature, coefficient in zip(
                final_preprocessor.encoded_names,
                final_preprocessor.encoded_raw_features,
                final_beta,
            ):
                coefficient_rows.append(
                    {
                        "target_key": target_key,
                        "target_column": target_column,
                        "model_variant": variant,
                        "selected_alpha": alpha,
                        "encoded_feature": encoded_name,
                        "raw_feature": raw_feature,
                        "feature_group": feature_group(raw_feature),
                        "standardized_coefficient": float(coefficient),
                        "absolute_standardized_coefficient": abs(float(coefficient)),
                        "fit_scope": "train_plus_validation",
                    }
                )

        bootstrap = paired_bootstrap(
            y_test,
            selected_test_predictions["player_history_baseline"],
            selected_test_predictions["strict_compatibility"],
            RANDOM_SEED + target_index,
        )
        bootstrap_rows.append(
            {
                "target_key": target_key,
                "target_column": target_column,
                **bootstrap,
            }
        )

        target_summary_rows.append(
            {
                "target_key": target_key,
                "target_column": target_column,
                "interpretation": config["interpretation"],
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
                "baseline_raw_features": len(baseline_features),
                "compatibility_raw_features": len(compatibility_features),
                "baseline_selected_alpha": selected_alpha["player_history_baseline"],
                "compatibility_selected_alpha": selected_alpha["strict_compatibility"],
                **bootstrap,
            }
        )

    metrics_frame = pd.DataFrame(metrics_rows)
    tuning_frame = pd.DataFrame(tuning_rows)
    predictions_frame = pd.DataFrame(prediction_rows)
    coefficients_frame = pd.DataFrame(coefficient_rows)
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    target_summary_frame = pd.DataFrame(target_summary_rows)

    coefficients_frame["absolute_rank_within_model"] = coefficients_frame.groupby(
        ["target_key", "model_variant"]
    )["absolute_standardized_coefficient"].rank(method="first", ascending=False).astype(int)
    coefficients_frame = coefficients_frame.sort_values(
        ["target_key", "model_variant", "absolute_rank_within_model"]
    ).reset_index(drop=True)
    top_compatibility_features = (
        coefficients_frame.loc[
            coefficients_frame["model_variant"].eq("strict_compatibility")
            & coefficients_frame["feature_group"].isin(
                ["historical_teammate_preference", "destination_club_compatibility"]
            )
        ]
        .sort_values(["target_key", "absolute_standardized_coefficient"], ascending=[True, False])
        .groupby("target_key", as_index=False)
        .head(15)
        .reset_index(drop=True)
    )

    test_prediction_wide = (
        predictions_frame.loc[predictions_frame["recommended_time_split"].eq("test")]
        .pivot(
            index=[
                "canonical_performance_id",
                "canonical_player_name",
                "canonical_club_name",
                "season_start_year",
                "role_group",
                "target_key",
                "target_column",
                "actual",
            ],
            columns="model_variant",
            values="prediction",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    test_prediction_wide["baseline_absolute_error"] = (
        test_prediction_wide["actual"] - test_prediction_wide["player_history_baseline"]
    ).abs()
    test_prediction_wide["compatibility_absolute_error"] = (
        test_prediction_wide["actual"] - test_prediction_wide["strict_compatibility"]
    ).abs()
    test_prediction_wide["compatibility_error_improvement"] = (
        test_prediction_wide["baseline_absolute_error"]
        - test_prediction_wide["compatibility_absolute_error"]
    )

    comparison_rows: list[dict[str, Any]] = []
    for config in TARGETS:
        target_key = config["target_key"]
        test_metrics = metrics_frame.loc[
            metrics_frame["target_key"].eq(target_key) & metrics_frame["split"].eq("test")
        ].set_index("model_variant")
        for metric_name in ["mae", "rmse", "r2", "pearson", "spearman"]:
            baseline_value = float(test_metrics.loc["player_history_baseline", metric_name])
            compatibility_value = float(test_metrics.loc["strict_compatibility", metric_name])
            lower_is_better = metric_name in {"mae", "rmse"}
            improvement = (
                baseline_value - compatibility_value
                if lower_is_better
                else compatibility_value - baseline_value
            )
            comparison_rows.append(
                {
                    "target_key": target_key,
                    "target_column": config["target_column"],
                    "split": "test",
                    "metric": metric_name,
                    "direction": "lower_is_better" if lower_is_better else "higher_is_better",
                    "player_history_baseline": baseline_value,
                    "strict_compatibility": compatibility_value,
                    "compatibility_minus_baseline": compatibility_value - baseline_value,
                    "improvement_in_preferred_direction": improvement,
                    "relative_error_improvement": (
                        improvement / baseline_value
                        if lower_is_better and abs(baseline_value) > 1e-12
                        else None
                    ),
                }
            )
    comparison_frame = pd.DataFrame(comparison_rows)

    # Post-build checks on exact model products.
    for config in TARGETS:
        target_key = config["target_key"]
        target_predictions = predictions_frame.loc[predictions_frame["target_key"].eq(target_key)]
        for split in ["train", "validation", "test"]:
            counts = target_predictions.loc[
                target_predictions["recommended_time_split"].eq(split)
            ].groupby("model_variant")["canonical_performance_id"].nunique()
            add_check(
                f"{target_key}_{split}_prediction_keys_match",
                counts.nunique() == 1 and len(counts) == 3,
                ";".join(f"{key}:{value}" for key, value in counts.items()),
                "identical across mean, baseline, compatibility",
                "Every model is scored on the identical destination player-season cohort.",
            )
        target_values = target_predictions[["canonical_performance_id", "actual"]].drop_duplicates()
        add_check(
            f"{target_key}_actual_is_unique_by_key",
            not target_values["canonical_performance_id"].duplicated().any(),
            int(target_values["canonical_performance_id"].duplicated().sum()),
            0,
            "All variants share one canonical actual outcome per destination player-season.",
        )

    add_check(
        "all_predictions_finite",
        np.isfinite(predictions_frame[["actual", "prediction_raw", "prediction"]].to_numpy(dtype=float)).all(),
        int((~np.isfinite(predictions_frame[["actual", "prediction_raw", "prediction"]].to_numpy(dtype=float))).sum()),
        0,
        "The exported scoring rows contain no NaN or infinite values.",
    )
    opportunity_predictions = predictions_frame.loc[
        predictions_frame["target_key"].eq("opportunity"), "prediction"
    ]
    add_check(
        "opportunity_predictions_bounded",
        opportunity_predictions.between(0.0, 1.0).all(),
        f"min={opportunity_predictions.min():.6f};max={opportunity_predictions.max():.6f}",
        "within [0,1]",
        "Opportunity predictions are clipped only after fitting to respect the target support.",
    )
    add_check(
        "selected_alphas_from_declared_grid",
        set(metrics_frame["selected_alpha"].dropna().astype(float)).issubset(set(ALPHA_GRID)),
        ";".join(map(str, sorted(metrics_frame["selected_alpha"].dropna().unique()))),
        ";".join(map(str, ALPHA_GRID)),
        "Hyperparameters are selected only from the declared grid using 2022 validation MAE.",
    )
    add_check(
        "test_rows_not_in_tuning_table",
        "test_rows" not in tuning_frame.columns,
        ";".join(tuning_frame.columns),
        "no test fields",
        "The tuning artifact contains only train and validation sample sizes and metrics.",
    )

    check_frame = pd.DataFrame(checks)
    if not check_frame["passed"].all():
        failed = check_frame.loc[~check_frame["passed"], "check"].tolist()
        raise RuntimeError(f"Blocking model checks failed: {failed}")

    feature_dictionary.to_csv(OUTPUT_DIR / "model_feature_dictionary.csv", index=False)
    target_summary_frame.to_csv(OUTPUT_DIR / "model_target_summary.csv", index=False)
    metrics_frame.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    comparison_frame.to_csv(OUTPUT_DIR / "model_test_comparison.csv", index=False)
    bootstrap_frame.to_csv(OUTPUT_DIR / "model_test_bootstrap.csv", index=False)
    tuning_frame.to_csv(OUTPUT_DIR / "model_validation_tuning.csv", index=False)
    predictions_frame.to_csv(OUTPUT_DIR / "model_predictions.csv", index=False)
    test_prediction_wide.to_csv(OUTPUT_DIR / "model_test_predictions_wide.csv", index=False)
    coefficients_frame.to_csv(OUTPUT_DIR / "model_coefficients.csv", index=False)
    top_compatibility_features.to_csv(OUTPUT_DIR / "model_top_compatibility_features.csv", index=False)
    check_frame.to_csv(OUTPUT_DIR / "model_build_checks.csv", index=False)

    run_summary = {
        "status": "pass",
        "model_family": "standardized_ridge_regression",
        "selection_metric": "validation_mae",
        "chronological_policy": "train through 2021; validation 2022; test 2023",
        "random_seed": RANDOM_SEED,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "alpha_grid": ALPHA_GRID,
        "baseline_raw_features": len(baseline_features),
        "compatibility_raw_features": len(compatibility_features),
        "targets": target_summary_frame.to_dict(orient="records"),
        "blocking_checks": int(len(check_frame)),
        "blocking_failures": int((~check_frame["passed"]).sum()),
    }
    with (OUTPUT_DIR / "model_run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2, allow_nan=False)

    source_manifest = pd.DataFrame(
        [
            {
                "source_file": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in [MATRIX_PATH, FIELD_DICTIONARY_PATH, TARGET_VERIFICATION_PATH]
        ]
    )
    source_manifest.to_csv(OUTPUT_DIR / "source_manifest.csv", index=False)

    readme = f"""# MoveMaker compatibility model benchmark — first version

## Experiment

Three independent continuous targets are modeled with a mean-only diagnostic, a market-aware player-history baseline, and the same baseline plus strict compatibility features. Models use standardized ridge regression, select alpha by 2022 validation MAE, then refit train+validation before a single 2023 test evaluation.

## Feature comparison

- Player-history baseline: {len(baseline_features)} raw transfer-time, market, and lagged player-history features.
- Strict compatibility model: {len(compatibility_features)} raw features, adding historical teammate-style preferences and destination-club prior-season context/fit.
- Player IDs, club IDs, names, season labels, targets, target diagnostics, eligibility flags, exclusion reasons, and target-roster scenario fields are excluded.

## Interpretation

`model_test_comparison.csv` is the main held-out comparison. Positive `improvement_in_preferred_direction` means compatibility improved the metric. `model_test_bootstrap.csv` reports paired 95% bootstrap intervals and the probability that compatibility reduced test error. With only 37–58 test transfers per target, uncertainty must be emphasized over point estimates.

## Reproduction

Run `scripts/train_compatibility_models.py`, followed by `scripts/verify_compatibility_models.py`. The training script requires only the bundled pandas and NumPy runtime.
"""
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    output_paths = sorted(
        path
        for path in OUTPUT_DIR.iterdir()
        if path.is_file() and path.name not in {"output_manifest.csv"}
    )
    output_manifest = pd.DataFrame(
        [
            {
                "output_file": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in output_paths
        ]
    )
    output_manifest.to_csv(OUTPUT_DIR / "output_manifest.csv", index=False)

    print(
        json.dumps(
            {
                "status": "pass",
                "output_directory": str(OUTPUT_DIR),
                "baseline_raw_features": len(baseline_features),
                "compatibility_raw_features": len(compatibility_features),
                "checks": len(check_frame),
                "targets": target_summary_frame.to_dict(orient="records"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()

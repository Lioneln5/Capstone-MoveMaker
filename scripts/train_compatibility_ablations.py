"""Run stronger-model and compatibility feature-block ablations for MoveMaker."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_compatibility_models import (
    ALPHA_GRID as RIDGE_ALPHA_GRID,
    FORBIDDEN_PREDICTOR_PREFIXES,
    TARGETS,
    apply_bounds,
    bool_series,
    fit_preprocessor,
    fit_ridge,
    metrics,
    paired_bootstrap,
    predict_ridge,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "Data" / "processed" / "compatibility_models"
TARGET_DIR = ROOT / "Data" / "processed" / "compatibility_targets"
OUTPUT_DIR = ROOT / "Data" / "processed" / "compatibility_model_ablations"

MATRIX_PATH = TARGET_DIR / "transfer_compatibility_strict_model_matrix.csv"
FEATURE_PATH = MODEL_DIR / "model_feature_dictionary.csv"
UPSTREAM_VERIFICATION_PATH = MODEL_DIR / "independent_verification.json"

RANDOM_SEED = 20260809
ELASTIC_ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
ELASTIC_L1_GRID = [0.1, 0.5, 0.9]
GBT_CONFIGS = [
    {"learning_rate": 0.05, "max_depth": 1, "min_samples_leaf": 12, "max_features": 48},
    {"learning_rate": 0.03, "max_depth": 2, "min_samples_leaf": 15, "max_features": 48},
    {"learning_rate": 0.05, "max_depth": 2, "min_samples_leaf": 10, "max_features": 48},
]
GBT_ESTIMATORS = [50, 100]
MODEL_FAMILIES = ["ridge", "elastic_net", "gradient_boosted_trees"]
FEATURE_SET_ORDER = [
    "player_history_baseline",
    "plus_club_context",
    "plus_player_preferences",
    "context_and_preferences",
    "full_explicit_fit",
]


def stable_seed(*parts: str) -> int:
    payload = "|".join(parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def soft_threshold(value: float, threshold: float) -> float:
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


def fit_elastic_net(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    l1_ratio: float,
    max_iter: int = 1000,
    tolerance: float = 1e-7,
) -> tuple[float, np.ndarray, int]:
    y = np.asarray(y, dtype=float)
    intercept = float(np.mean(y))
    centered = y - intercept
    beta = np.zeros(x.shape[1], dtype=float)
    residual = centered.copy()
    if x.shape[1] == 0:
        return intercept, beta, 0
    column_norm = np.sum(np.square(x), axis=0) / len(x)
    l1_penalty = alpha * l1_ratio
    l2_penalty = alpha * (1.0 - l1_ratio)
    for iteration in range(1, max_iter + 1):
        max_change = 0.0
        for feature_index in range(x.shape[1]):
            old = beta[feature_index]
            column = x[:, feature_index]
            rho = float(column @ (residual + column * old) / len(x))
            denominator = column_norm[feature_index] + l2_penalty
            new = soft_threshold(rho, l1_penalty) / denominator if denominator > 1e-15 else 0.0
            if new != old:
                residual += column * (old - new)
                beta[feature_index] = new
                max_change = max(max_change, abs(new - old))
        if max_change < tolerance:
            return intercept, beta, iteration
    return intercept, beta, max_iter


def predict_linear(x: np.ndarray, intercept: float, beta: np.ndarray) -> np.ndarray:
    return intercept + x @ beta


def best_tree_split(
    x: np.ndarray,
    residual: np.ndarray,
    row_indices: np.ndarray,
    feature_indices: np.ndarray,
    min_samples_leaf: int,
) -> tuple[int | None, float | None, float]:
    n_rows = len(row_indices)
    if n_rows < 2 * min_samples_leaf:
        return None, None, 0.0
    node_values = residual[row_indices]
    parent_sse = float(np.sum(np.square(node_values - np.mean(node_values))))
    best_feature: int | None = None
    best_threshold: float | None = None
    best_loss = parent_sse
    for feature_index in feature_indices:
        values = x[row_indices, feature_index]
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        sorted_residual = node_values[order]
        change_positions = np.flatnonzero(np.diff(sorted_values) > 1e-12)
        valid_positions = change_positions[
            (change_positions + 1 >= min_samples_leaf)
            & (n_rows - change_positions - 1 >= min_samples_leaf)
        ]
        if not len(valid_positions):
            continue
        cumulative = np.cumsum(sorted_residual)
        cumulative_sq = np.cumsum(np.square(sorted_residual))
        total = cumulative[-1]
        total_sq = cumulative_sq[-1]
        left_n = valid_positions + 1
        right_n = n_rows - left_n
        left_sum = cumulative[valid_positions]
        left_sq = cumulative_sq[valid_positions]
        right_sum = total - left_sum
        right_sq = total_sq - left_sq
        losses = left_sq - np.square(left_sum) / left_n + right_sq - np.square(right_sum) / right_n
        candidate_index = int(np.argmin(losses))
        candidate_loss = float(losses[candidate_index])
        if candidate_loss < best_loss - 1e-12:
            position = int(valid_positions[candidate_index])
            best_loss = candidate_loss
            best_feature = int(feature_index)
            best_threshold = float((sorted_values[position] + sorted_values[position + 1]) / 2.0)
    return best_feature, best_threshold, max(0.0, parent_sse - best_loss)


def fit_regression_tree(
    x: np.ndarray,
    residual: np.ndarray,
    row_indices: np.ndarray,
    depth_remaining: int,
    min_samples_leaf: int,
    max_features: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    value = float(np.mean(residual[row_indices])) if len(row_indices) else 0.0
    if depth_remaining <= 0 or len(row_indices) < 2 * min_samples_leaf or x.shape[1] == 0:
        return {"value": value}
    feature_count = min(max_features, x.shape[1])
    candidate_features = rng.choice(x.shape[1], size=feature_count, replace=False)
    feature, threshold, gain = best_tree_split(
        x, residual, row_indices, candidate_features, min_samples_leaf
    )
    if feature is None or threshold is None or gain <= 1e-12:
        return {"value": value}
    left_mask = x[row_indices, feature] <= threshold
    left_rows = row_indices[left_mask]
    right_rows = row_indices[~left_mask]
    if len(left_rows) < min_samples_leaf or len(right_rows) < min_samples_leaf:
        return {"value": value}
    return {
        "value": value,
        "feature": feature,
        "threshold": threshold,
        "gain": gain,
        "left": fit_regression_tree(
            x,
            residual,
            left_rows,
            depth_remaining - 1,
            min_samples_leaf,
            max_features,
            rng,
        ),
        "right": fit_regression_tree(
            x,
            residual,
            right_rows,
            depth_remaining - 1,
            min_samples_leaf,
            max_features,
            rng,
        ),
    }


def predict_tree(x: np.ndarray, tree: dict[str, Any]) -> np.ndarray:
    output = np.empty(len(x), dtype=float)

    def assign(rows: np.ndarray, node: dict[str, Any]) -> None:
        if "feature" not in node:
            output[rows] = node["value"]
            return
        feature = node["feature"]
        mask = x[rows, feature] <= node["threshold"]
        assign(rows[mask], node["left"])
        assign(rows[~mask], node["right"])

    assign(np.arange(len(x)), tree)
    return output


def fit_gradient_boosting(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    max_depth: int,
    min_samples_leaf: int,
    max_features: int,
    n_estimators: int,
    seed: int,
) -> tuple[float, list[dict[str, Any]]]:
    intercept = float(np.mean(y))
    prediction = np.full(len(y), intercept, dtype=float)
    trees: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    row_indices = np.arange(len(y))
    for _ in range(n_estimators):
        residual = y - prediction
        tree = fit_regression_tree(
            x,
            residual,
            row_indices,
            max_depth,
            min_samples_leaf,
            max_features,
            rng,
        )
        update = predict_tree(x, tree)
        prediction += learning_rate * update
        trees.append(tree)
    return intercept, trees


def predict_gradient_boosting(
    x: np.ndarray,
    intercept: float,
    trees: list[dict[str, Any]],
    learning_rate: float,
) -> np.ndarray:
    prediction = np.full(len(x), intercept, dtype=float)
    for tree in trees:
        prediction += learning_rate * predict_tree(x, tree)
    return prediction


def collect_tree_gains(tree: dict[str, Any], gains: defaultdict[int, float], counts: defaultdict[int, int]) -> None:
    if "feature" not in tree:
        return
    feature = int(tree["feature"])
    gains[feature] += float(tree["gain"])
    counts[feature] += 1
    collect_tree_gains(tree["left"], gains, counts)
    collect_tree_gains(tree["right"], gains, counts)


def feature_sets(feature_dictionary: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, str]]:
    baseline = feature_dictionary.loc[
        bool_series(feature_dictionary["in_player_history_baseline"]), "raw_feature"
    ].tolist()
    full = feature_dictionary.loc[
        bool_series(feature_dictionary["in_strict_compatibility_model"]), "raw_feature"
    ].tolist()
    incremental = [feature for feature in full if feature not in baseline]
    preferences = [feature for feature in incremental if feature.startswith("preference_")]
    fit_tokens = ("familiarity", "preference", "strict_player_", "cosine", "fit_score", "slope_signal")
    context = [
        feature
        for feature in incremental
        if feature.startswith("strict_") and not any(token in feature for token in fit_tokens)
    ]
    explicit_fit = [
        feature for feature in incremental if feature not in preferences and feature not in context
    ]
    sets = {
        "player_history_baseline": baseline,
        "plus_club_context": baseline + context,
        "plus_player_preferences": baseline + preferences,
        "context_and_preferences": baseline + context + preferences,
        "full_explicit_fit": baseline + context + preferences + explicit_fit,
    }
    block = {}
    for feature in baseline:
        source_group = feature_dictionary.loc[
            feature_dictionary["raw_feature"].eq(feature), "feature_group"
        ].iloc[0]
        block[feature] = source_group
    block.update({feature: "destination_club_context" for feature in context})
    block.update({feature: "historical_teammate_preference" for feature in preferences})
    block.update({feature: "explicit_player_club_fit" for feature in explicit_fit})
    return sets, block


def hyperparameter_string(row: dict[str, Any]) -> str:
    keys = [
        "alpha",
        "l1_ratio",
        "learning_rate",
        "max_depth",
        "min_samples_leaf",
        "max_features",
        "n_estimators",
    ]
    return json.dumps({key: row[key] for key in keys if row.get(key) is not None}, sort_keys=True)


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(MATRIX_PATH, low_memory=False)
    feature_dictionary = pd.read_csv(FEATURE_PATH, low_memory=False)
    upstream_verification = json.loads(UPSTREAM_VERIFICATION_PATH.read_text(encoding="utf-8"))
    if upstream_verification.get("status") != "pass":
        raise RuntimeError("Upstream model benchmark verification is not passing.")

    sets, feature_blocks = feature_sets(feature_dictionary)
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
        "upstream_model_verification_passes",
        upstream_verification.get("status") == "pass",
        upstream_verification.get("status"),
        "pass",
        "Ablations start from a fully verified first benchmark.",
    )
    add_check(
        "feature_set_order_declared",
        list(sets) == FEATURE_SET_ORDER,
        ";".join(sets),
        ";".join(FEATURE_SET_ORDER),
        "Feature sets have one declared comparison order.",
    )
    add_check(
        "baseline_nested_in_all_sets",
        all(set(sets["player_history_baseline"]).issubset(values) for values in sets.values()),
        "nested",
        "nested",
        "Every ablation holds the common player-history controls fixed.",
    )
    add_check(
        "full_set_has_expected_features",
        set(sets["full_explicit_fit"])
        == set(
            feature_dictionary.loc[
                bool_series(feature_dictionary["in_strict_compatibility_model"]), "raw_feature"
            ]
        ),
        len(sets["full_explicit_fit"]),
        int(bool_series(feature_dictionary["in_strict_compatibility_model"]).sum()),
        "The terminal ablation exactly equals the prior strict compatibility model.",
    )
    forbidden = [
        feature
        for values in sets.values()
        for feature in values
        if feature.startswith(FORBIDDEN_PREDICTOR_PREFIXES)
    ]
    add_check(
        "no_forbidden_features",
        not forbidden,
        ";".join(sorted(set(forbidden))),
        "none",
        "Targets, target diagnostics, eligibility fields, and roster scenarios remain excluded.",
    )

    feature_set_rows = []
    for feature_set_name, values in sets.items():
        for order, feature in enumerate(values, start=1):
            feature_set_rows.append(
                {
                    "feature_set": feature_set_name,
                    "feature_order": order,
                    "raw_feature": feature,
                    "introduced_block": feature_blocks[feature],
                    "raw_feature_count": len(values),
                }
            )
    feature_set_frame = pd.DataFrame(feature_set_rows)

    tuning_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []

    for target_index, config in enumerate(TARGETS):
        target_key = config["target_key"]
        target_column = config["target_column"]
        eligible = bool_series(matrix[config["eligibility_column"]]) & matrix[target_column].notna()
        cohort = matrix.loc[eligible].copy()
        cohort[target_column] = pd.to_numeric(cohort[target_column], errors="coerce")
        cohort = cohort.loc[cohort[target_column].notna()].copy()
        cohort = cohort.sort_values(["season_start_year", "canonical_performance_id"]).reset_index(drop=True)
        splits = {
            split: cohort.loc[cohort["recommended_time_split"].eq(split)].copy()
            for split in ["train", "validation", "test"]
        }
        train = splits["train"]
        validation = splits["validation"]
        test = splits["test"]
        train_validation = pd.concat([train, validation], ignore_index=True)
        y_train = train[target_column].to_numpy(dtype=float)
        y_validation = validation[target_column].to_numpy(dtype=float)
        y_test = test[target_column].to_numpy(dtype=float)
        y_train_validation = train_validation[target_column].to_numpy(dtype=float)
        for split, frame in splits.items():
            add_check(
                f"{target_key}_{split}_nonempty",
                len(frame) > 0,
                len(frame),
                ">0",
                "Every target retains all three chronological partitions.",
            )

        for feature_set_name in FEATURE_SET_ORDER:
            print(f"training target={target_key} feature_set={feature_set_name}", flush=True)
            raw_features = sets[feature_set_name]
            train_preprocessor = fit_preprocessor(train, raw_features)
            x_train = train_preprocessor.transform(train)
            x_validation = train_preprocessor.transform(validation)
            candidate_rows: dict[str, list[dict[str, Any]]] = {family: [] for family in MODEL_FAMILIES}

            for alpha in RIDGE_ALPHA_GRID:
                intercept, beta = fit_ridge(x_train, y_train, alpha)
                prediction = apply_bounds(
                    predict_ridge(x_validation, intercept, beta),
                    config["clip_lower"],
                    config["clip_upper"],
                )
                result = metrics(y_validation, prediction)
                row = {
                    "target_key": target_key,
                    "target_column": target_column,
                    "feature_set": feature_set_name,
                    "model_family": "ridge",
                    "alpha": alpha,
                    "l1_ratio": None,
                    "learning_rate": None,
                    "max_depth": None,
                    "min_samples_leaf": None,
                    "max_features": None,
                    "n_estimators": None,
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "raw_feature_count": len(raw_features),
                    "encoded_feature_count": x_train.shape[1],
                    **result,
                }
                row["hyperparameters"] = hyperparameter_string(row)
                tuning_rows.append(row)
                candidate_rows["ridge"].append(row)

            for alpha in ELASTIC_ALPHA_GRID:
                for l1_ratio in ELASTIC_L1_GRID:
                    intercept, beta, iterations = fit_elastic_net(
                        x_train, y_train, alpha, l1_ratio
                    )
                    prediction = apply_bounds(
                        predict_linear(x_validation, intercept, beta),
                        config["clip_lower"],
                        config["clip_upper"],
                    )
                    result = metrics(y_validation, prediction)
                    row = {
                        "target_key": target_key,
                        "target_column": target_column,
                        "feature_set": feature_set_name,
                        "model_family": "elastic_net",
                        "alpha": alpha,
                        "l1_ratio": l1_ratio,
                        "learning_rate": None,
                        "max_depth": None,
                        "min_samples_leaf": None,
                        "max_features": None,
                        "n_estimators": None,
                        "iterations": iterations,
                        "nonzero_coefficients": int(np.count_nonzero(np.abs(beta) > 1e-12)),
                        "train_rows": len(train),
                        "validation_rows": len(validation),
                        "raw_feature_count": len(raw_features),
                        "encoded_feature_count": x_train.shape[1],
                        **result,
                    }
                    row["hyperparameters"] = hyperparameter_string(row)
                    tuning_rows.append(row)
                    candidate_rows["elastic_net"].append(row)

            for gbt_index, gbt_config in enumerate(GBT_CONFIGS):
                for n_estimators in GBT_ESTIMATORS:
                    seed = stable_seed(
                        str(RANDOM_SEED),
                        target_key,
                        feature_set_name,
                        "gbt",
                        str(gbt_config["learning_rate"]),
                        str(gbt_config["max_depth"]),
                        str(gbt_config["min_samples_leaf"]),
                        str(gbt_config["max_features"]),
                        str(n_estimators),
                    )
                    intercept, trees = fit_gradient_boosting(
                        x_train,
                        y_train,
                        n_estimators=n_estimators,
                        seed=seed,
                        **gbt_config,
                    )
                    prediction = apply_bounds(
                        predict_gradient_boosting(
                            x_validation, intercept, trees, gbt_config["learning_rate"]
                        ),
                        config["clip_lower"],
                        config["clip_upper"],
                    )
                    result = metrics(y_validation, prediction)
                    row = {
                        "target_key": target_key,
                        "target_column": target_column,
                        "feature_set": feature_set_name,
                        "model_family": "gradient_boosted_trees",
                        "alpha": None,
                        "l1_ratio": None,
                        **gbt_config,
                        "n_estimators": n_estimators,
                        "train_rows": len(train),
                        "validation_rows": len(validation),
                        "raw_feature_count": len(raw_features),
                        "encoded_feature_count": x_train.shape[1],
                        **result,
                    }
                    row["hyperparameters"] = hyperparameter_string(row)
                    tuning_rows.append(row)
                    candidate_rows["gradient_boosted_trees"].append(row)

            for model_family in MODEL_FAMILIES:
                selected = min(
                    candidate_rows[model_family],
                    key=lambda row: (row["mae"], row["rmse"], row["hyperparameters"]),
                )
                final_preprocessor = fit_preprocessor(train_validation, raw_features)
                x_train_validation = final_preprocessor.transform(train_validation)
                x_test = final_preprocessor.transform(test)

                if model_family == "ridge":
                    train_intercept, train_beta = fit_ridge(x_train, y_train, float(selected["alpha"]))
                    train_raw = predict_ridge(x_train, train_intercept, train_beta)
                    validation_raw = predict_ridge(x_validation, train_intercept, train_beta)
                    final_intercept, final_beta = fit_ridge(
                        x_train_validation, y_train_validation, float(selected["alpha"])
                    )
                    test_raw = predict_ridge(x_test, final_intercept, final_beta)
                    for encoded, raw, coefficient in zip(
                        final_preprocessor.encoded_names,
                        final_preprocessor.encoded_raw_features,
                        final_beta,
                    ):
                        importance_rows.append(
                            {
                                "target_key": target_key,
                                "feature_set": feature_set_name,
                                "model_family": model_family,
                                "encoded_feature": encoded,
                                "raw_feature": raw,
                                "introduced_block": feature_blocks[raw],
                                "importance_type": "absolute_standardized_coefficient",
                                "importance": abs(float(coefficient)),
                                "signed_coefficient": float(coefficient),
                                "split_count": None,
                                "fit_scope": "train_plus_validation",
                            }
                        )
                elif model_family == "elastic_net":
                    train_intercept, train_beta, _ = fit_elastic_net(
                        x_train,
                        y_train,
                        float(selected["alpha"]),
                        float(selected["l1_ratio"]),
                    )
                    train_raw = predict_linear(x_train, train_intercept, train_beta)
                    validation_raw = predict_linear(x_validation, train_intercept, train_beta)
                    final_intercept, final_beta, _ = fit_elastic_net(
                        x_train_validation,
                        y_train_validation,
                        float(selected["alpha"]),
                        float(selected["l1_ratio"]),
                    )
                    test_raw = predict_linear(x_test, final_intercept, final_beta)
                    for encoded, raw, coefficient in zip(
                        final_preprocessor.encoded_names,
                        final_preprocessor.encoded_raw_features,
                        final_beta,
                    ):
                        if abs(float(coefficient)) <= 1e-12:
                            continue
                        importance_rows.append(
                            {
                                "target_key": target_key,
                                "feature_set": feature_set_name,
                                "model_family": model_family,
                                "encoded_feature": encoded,
                                "raw_feature": raw,
                                "introduced_block": feature_blocks[raw],
                                "importance_type": "absolute_standardized_coefficient",
                                "importance": abs(float(coefficient)),
                                "signed_coefficient": float(coefficient),
                                "split_count": None,
                                "fit_scope": "train_plus_validation",
                            }
                        )
                else:
                    gbt_params = {
                        "learning_rate": float(selected["learning_rate"]),
                        "max_depth": int(selected["max_depth"]),
                        "min_samples_leaf": int(selected["min_samples_leaf"]),
                        "max_features": int(selected["max_features"]),
                        "n_estimators": int(selected["n_estimators"]),
                    }
                    train_seed = stable_seed(
                        str(RANDOM_SEED),
                        target_key,
                        feature_set_name,
                        "gbt",
                        str(gbt_params["learning_rate"]),
                        str(gbt_params["max_depth"]),
                        str(gbt_params["min_samples_leaf"]),
                        str(gbt_params["max_features"]),
                        str(gbt_params["n_estimators"]),
                    )
                    train_intercept, train_trees = fit_gradient_boosting(
                        x_train, y_train, seed=train_seed, **gbt_params
                    )
                    train_raw = predict_gradient_boosting(
                        x_train, train_intercept, train_trees, gbt_params["learning_rate"]
                    )
                    validation_raw = predict_gradient_boosting(
                        x_validation, train_intercept, train_trees, gbt_params["learning_rate"]
                    )
                    final_seed = stable_seed(
                        str(RANDOM_SEED), target_key, feature_set_name, "gbt", "selected_final"
                    )
                    final_intercept, final_trees = fit_gradient_boosting(
                        x_train_validation,
                        y_train_validation,
                        seed=final_seed,
                        **gbt_params,
                    )
                    test_raw = predict_gradient_boosting(
                        x_test, final_intercept, final_trees, gbt_params["learning_rate"]
                    )
                    gains: defaultdict[int, float] = defaultdict(float)
                    counts: defaultdict[int, int] = defaultdict(int)
                    for tree in final_trees:
                        collect_tree_gains(tree, gains, counts)
                    total_gain = sum(gains.values())
                    for feature_index, gain in gains.items():
                        raw = final_preprocessor.encoded_raw_features[feature_index]
                        importance_rows.append(
                            {
                                "target_key": target_key,
                                "feature_set": feature_set_name,
                                "model_family": model_family,
                                "encoded_feature": final_preprocessor.encoded_names[feature_index],
                                "raw_feature": raw,
                                "introduced_block": feature_blocks[raw],
                                "importance_type": "normalized_split_gain",
                                "importance": gain / total_gain if total_gain > 0 else 0.0,
                                "signed_coefficient": None,
                                "split_count": counts[feature_index],
                                "fit_scope": "train_plus_validation",
                            }
                        )

                predictions_by_split = {
                    "train": apply_bounds(train_raw, config["clip_lower"], config["clip_upper"]),
                    "validation": apply_bounds(
                        validation_raw, config["clip_lower"], config["clip_upper"]
                    ),
                    "test": apply_bounds(test_raw, config["clip_lower"], config["clip_upper"]),
                }
                raw_by_split = {"train": train_raw, "validation": validation_raw, "test": test_raw}
                for split, frame in splits.items():
                    actual = frame[target_column].to_numpy(dtype=float)
                    prediction = predictions_by_split[split]
                    result = metrics(actual, prediction)
                    metric_rows.append(
                        {
                            "target_key": target_key,
                            "target_column": target_column,
                            "feature_set": feature_set_name,
                            "model_family": model_family,
                            "split": split,
                            "fit_scope": "train" if split != "test" else "train_plus_validation",
                            "selected_hyperparameters": selected["hyperparameters"],
                            "raw_feature_count": len(raw_features),
                            "encoded_feature_count": (
                                x_train.shape[1] if split != "test" else x_train_validation.shape[1]
                            ),
                            **result,
                        }
                    )
                    for index, (_, row) in enumerate(frame.iterrows()):
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
                                "feature_set": feature_set_name,
                                "model_family": model_family,
                                "fit_scope": "train" if split != "test" else "train_plus_validation",
                                "actual": actual[index],
                                "prediction_raw": raw_by_split[split][index],
                                "prediction": prediction[index],
                                "residual": actual[index] - prediction[index],
                                "absolute_error": abs(actual[index] - prediction[index]),
                            }
                        )

            print(f"completed target={target_key} feature_set={feature_set_name}", flush=True)

    tuning_frame = pd.DataFrame(tuning_rows)
    metrics_frame = pd.DataFrame(metric_rows)
    predictions_frame = pd.DataFrame(prediction_rows)
    importance_frame = pd.DataFrame(importance_rows)
    importance_frame["importance_rank"] = importance_frame.groupby(
        ["target_key", "feature_set", "model_family"]
    )["importance"].rank(method="first", ascending=False).astype(int)
    importance_frame = importance_frame.sort_values(
        ["target_key", "feature_set", "model_family", "importance_rank"]
    ).reset_index(drop=True)
    top_importance_frame = importance_frame.loc[importance_frame["importance_rank"].le(15)].copy()

    result_rows = []
    for _, row in metrics_frame.loc[metrics_frame["split"].eq("test")].iterrows():
        validation_row = metrics_frame.loc[
            metrics_frame["target_key"].eq(row["target_key"])
            & metrics_frame["feature_set"].eq(row["feature_set"])
            & metrics_frame["model_family"].eq(row["model_family"])
            & metrics_frame["split"].eq("validation")
        ].iloc[0]
        result_rows.append(
            {
                "target_key": row["target_key"],
                "feature_set": row["feature_set"],
                "model_family": row["model_family"],
                "selected_hyperparameters": row["selected_hyperparameters"],
                "raw_feature_count": row["raw_feature_count"],
                "validation_mae": validation_row["mae"],
                "validation_rmse": validation_row["rmse"],
                "test_n": row["n"],
                "test_mae": row["mae"],
                "test_rmse": row["rmse"],
                "test_r2": row["r2"],
                "test_pearson": row["pearson"],
                "test_spearman": row["spearman"],
            }
        )
    results_frame = pd.DataFrame(result_rows)

    block_effect_rows = []
    for target_key in results_frame["target_key"].unique():
        for model_family in MODEL_FAMILIES:
            group = results_frame.loc[
                results_frame["target_key"].eq(target_key)
                & results_frame["model_family"].eq(model_family)
            ]
            baseline = group.loc[group["feature_set"].eq("player_history_baseline")].iloc[0]
            for _, row in group.iterrows():
                block_effect_rows.append(
                    {
                        **row.to_dict(),
                        "validation_mae_improvement_vs_baseline": baseline["validation_mae"] - row["validation_mae"],
                        "test_mae_improvement_vs_baseline": baseline["test_mae"] - row["test_mae"],
                        "test_relative_mae_improvement_vs_baseline": (
                            (baseline["test_mae"] - row["test_mae"]) / baseline["test_mae"]
                        ),
                    }
                )
    block_effect_frame = pd.DataFrame(block_effect_rows)

    family_bootstrap_rows = []
    for target_index, config in enumerate(TARGETS):
        target_key = config["target_key"]
        for family_index, model_family in enumerate(MODEL_FAMILIES):
            rows = predictions_frame.loc[
                predictions_frame["target_key"].eq(target_key)
                & predictions_frame["model_family"].eq(model_family)
                & predictions_frame["recommended_time_split"].eq("test")
                & predictions_frame["feature_set"].isin(
                    ["player_history_baseline", "full_explicit_fit"]
                )
            ]
            baseline = rows.loc[rows["feature_set"].eq("player_history_baseline")].sort_values(
                "canonical_performance_id"
            )
            full = rows.loc[rows["feature_set"].eq("full_explicit_fit")].sort_values(
                "canonical_performance_id"
            )
            result = paired_bootstrap(
                baseline["actual"].to_numpy(dtype=float),
                baseline["prediction"].to_numpy(dtype=float),
                full["prediction"].to_numpy(dtype=float),
                RANDOM_SEED + 100 + target_index * 10 + family_index,
            )
            family_bootstrap_rows.append(
                {
                    "target_key": target_key,
                    "model_family": model_family,
                    "comparison": "full_explicit_fit_minus_player_history_baseline",
                    **result,
                }
            )
    family_bootstrap_frame = pd.DataFrame(family_bootstrap_rows)

    selection_rows = []
    selected_pair_predictions = []
    selected_bootstrap_rows = []
    for target_index, config in enumerate(TARGETS):
        target_key = config["target_key"]
        target_results = results_frame.loc[results_frame["target_key"].eq(target_key)]
        baseline_candidates = target_results.loc[
            target_results["feature_set"].eq("player_history_baseline")
        ]
        full_candidates = target_results.loc[target_results["feature_set"].eq("full_explicit_fit")]
        best_baseline = baseline_candidates.sort_values(
            ["validation_mae", "validation_rmse", "model_family"]
        ).iloc[0]
        best_full = full_candidates.sort_values(
            ["validation_mae", "validation_rmse", "model_family"]
        ).iloc[0]
        best_overall = target_results.sort_values(
            ["validation_mae", "validation_rmse", "feature_set", "model_family"]
        ).iloc[0]
        for selection_type, row in [
            ("validation_selected_baseline", best_baseline),
            ("validation_selected_full", best_full),
            ("validation_selected_overall", best_overall),
        ]:
            selection_rows.append({"selection_type": selection_type, **row.to_dict()})

        target_predictions = predictions_frame.loc[
            predictions_frame["target_key"].eq(target_key)
            & predictions_frame["recommended_time_split"].eq("test")
        ]
        baseline_predictions = target_predictions.loc[
            target_predictions["feature_set"].eq(best_baseline["feature_set"])
            & target_predictions["model_family"].eq(best_baseline["model_family"])
        ].sort_values("canonical_performance_id")
        full_predictions = target_predictions.loc[
            target_predictions["feature_set"].eq(best_full["feature_set"])
            & target_predictions["model_family"].eq(best_full["model_family"])
        ].sort_values("canonical_performance_id")
        bootstrap_result = paired_bootstrap(
            baseline_predictions["actual"].to_numpy(dtype=float),
            baseline_predictions["prediction"].to_numpy(dtype=float),
            full_predictions["prediction"].to_numpy(dtype=float),
            RANDOM_SEED + 500 + target_index,
        )
        selected_bootstrap_rows.append(
            {
                "target_key": target_key,
                "baseline_model_family": best_baseline["model_family"],
                "full_model_family": best_full["model_family"],
                **bootstrap_result,
            }
        )
        for baseline_row, full_row in zip(
            baseline_predictions.itertuples(index=False), full_predictions.itertuples(index=False)
        ):
            selected_pair_predictions.append(
                {
                    "canonical_performance_id": baseline_row.canonical_performance_id,
                    "canonical_player_name": baseline_row.canonical_player_name,
                    "canonical_club_name": baseline_row.canonical_club_name,
                    "season_start_year": baseline_row.season_start_year,
                    "role_group": baseline_row.role_group,
                    "target_key": target_key,
                    "actual": baseline_row.actual,
                    "baseline_model_family": best_baseline["model_family"],
                    "baseline_prediction": baseline_row.prediction,
                    "full_model_family": best_full["model_family"],
                    "full_prediction": full_row.prediction,
                    "baseline_absolute_error": abs(baseline_row.actual - baseline_row.prediction),
                    "full_absolute_error": abs(full_row.actual - full_row.prediction),
                    "full_error_improvement": abs(baseline_row.actual - baseline_row.prediction)
                    - abs(full_row.actual - full_row.prediction),
                }
            )

    selection_frame = pd.DataFrame(selection_rows)
    selected_bootstrap_frame = pd.DataFrame(selected_bootstrap_rows)
    selected_pair_frame = pd.DataFrame(selected_pair_predictions)

    # Final invariants across the complete run.
    add_check(
        "selected_specs_cover_all_target_feature_family_combinations",
        len(results_frame) == len(TARGETS) * len(FEATURE_SET_ORDER) * len(MODEL_FAMILIES),
        len(results_frame),
        len(TARGETS) * len(FEATURE_SET_ORDER) * len(MODEL_FAMILIES),
        "Every declared target, feature set, and model family has one validation-selected specification.",
    )
    add_check(
        "all_predictions_finite",
        np.isfinite(predictions_frame[["actual", "prediction_raw", "prediction"]].to_numpy(dtype=float)).all(),
        int((~np.isfinite(predictions_frame[["actual", "prediction_raw", "prediction"]].to_numpy(dtype=float))).sum()),
        0,
        "Every scored row is finite.",
    )
    opportunity = predictions_frame.loc[predictions_frame["target_key"].eq("opportunity"), "prediction"]
    add_check(
        "opportunity_predictions_bounded",
        opportunity.between(0, 1).all(),
        f"min={opportunity.min():.6f};max={opportunity.max():.6f}",
        "within [0,1]",
        "All opportunity predictions respect the target support.",
    )
    add_check(
        "tuning_contains_no_test_fields",
        all("test" not in column.lower() for column in tuning_frame.columns),
        ";".join(tuning_frame.columns),
        "no test columns",
        "Model and hyperparameter selection uses train and validation data only.",
    )
    prediction_key_counts = predictions_frame.groupby(
        ["target_key", "recommended_time_split", "feature_set", "model_family"]
    )["canonical_performance_id"].nunique()
    key_variation = prediction_key_counts.groupby(level=[0, 1]).nunique()
    add_check(
        "prediction_keys_match_all_specs",
        key_variation.eq(1).all(),
        int((~key_variation.eq(1)).sum()),
        0,
        "All model and feature variants score identical rows within each target and split.",
    )
    add_check(
        "selected_pair_keys_match",
        len(selected_pair_frame)
        == sum(
            int(
                bool_series(matrix[config["eligibility_column"]])
                .loc[matrix["recommended_time_split"].eq("test")]
                .sum()
            )
            for config in TARGETS
        ),
        len(selected_pair_frame),
        "all eligible test rows",
        "Validation-selected baseline and full models are paired on identical held-out rows.",
    )

    checks_frame = pd.DataFrame(checks)
    if not checks_frame["passed"].all():
        raise RuntimeError(
            f"Blocking ablation checks failed: {checks_frame.loc[~checks_frame['passed'], 'check'].tolist()}"
        )

    feature_set_frame.to_csv(OUTPUT_DIR / "ablation_feature_sets.csv", index=False)
    tuning_frame.to_csv(OUTPUT_DIR / "ablation_validation_tuning.csv", index=False)
    metrics_frame.to_csv(OUTPUT_DIR / "ablation_model_metrics.csv", index=False)
    results_frame.to_csv(OUTPUT_DIR / "ablation_test_results.csv", index=False)
    block_effect_frame.to_csv(OUTPUT_DIR / "ablation_feature_block_effects.csv", index=False)
    predictions_frame.to_csv(OUTPUT_DIR / "ablation_predictions.csv", index=False)
    importance_frame.to_csv(OUTPUT_DIR / "ablation_feature_importance.csv", index=False)
    top_importance_frame.to_csv(OUTPUT_DIR / "ablation_top_features.csv", index=False)
    family_bootstrap_frame.to_csv(OUTPUT_DIR / "ablation_family_bootstrap.csv", index=False)
    selection_frame.to_csv(OUTPUT_DIR / "ablation_model_selection.csv", index=False)
    selected_bootstrap_frame.to_csv(OUTPUT_DIR / "ablation_selected_bootstrap.csv", index=False)
    selected_pair_frame.to_csv(OUTPUT_DIR / "ablation_selected_test_predictions.csv", index=False)
    checks_frame.to_csv(OUTPUT_DIR / "ablation_build_checks.csv", index=False)

    run_summary = {
        "status": "pass",
        "targets": len(TARGETS),
        "feature_sets": {name: len(values) for name, values in sets.items()},
        "model_families": MODEL_FAMILIES,
        "selection_policy": "minimum 2022 validation MAE; 2023 scored after selection",
        "candidate_tuning_rows": len(tuning_frame),
        "selected_specifications": len(results_frame),
        "prediction_rows": len(predictions_frame),
        "blocking_checks": len(checks_frame),
        "blocking_failures": int((~checks_frame["passed"]).sum()),
        "random_seed": RANDOM_SEED,
    }
    (OUTPUT_DIR / "ablation_run_summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )

    pd.DataFrame(
        [
            {
                "source_file": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in [MATRIX_PATH, FEATURE_PATH, UPSTREAM_VERIFICATION_PATH]
        ]
    ).to_csv(OUTPUT_DIR / "source_manifest.csv", index=False)

    readme = """# MoveMaker stronger-model compatibility ablations

This phase compares ridge regression, sparse elastic net, and a deterministic nonlinear gradient-boosted regression-tree implementation. Each learner is evaluated on five declared feature sets: player-history baseline, baseline plus raw club context, baseline plus historical player preferences, context plus preferences, and the full model including explicit fit interactions.

All hyperparameters and validation-selected specifications are chosen on 2022 MAE. Models are then refit on train+validation and scored once on 2023. The principal causal-safe reading is within-family full-versus-baseline error change; cross-family selected comparisons are predictive benchmarks, not causal estimates.

Use `ablation_feature_block_effects.csv` for the full model/feature grid, `ablation_family_bootstrap.csv` for paired within-family uncertainty, and `ablation_model_selection.csv` plus `ablation_selected_bootstrap.csv` for the validation-selected baseline/full comparison. Test cohorts remain small, so repeated effects across validation, test, role, and league matter more than a single point estimate.
"""
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    output_paths = sorted(
        path for path in OUTPUT_DIR.iterdir() if path.is_file() and path.name != "output_manifest.csv"
    )
    pd.DataFrame(
        [
            {"output_file": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in output_paths
        ]
    ).to_csv(OUTPUT_DIR / "output_manifest.csv", index=False)

    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    run()

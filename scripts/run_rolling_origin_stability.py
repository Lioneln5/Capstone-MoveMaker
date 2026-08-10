"""Run expanding-window rolling-origin stability tests for MoveMaker compatibility models."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_compatibility_ablations import (
    ELASTIC_ALPHA_GRID,
    ELASTIC_L1_GRID,
    FEATURE_SET_ORDER,
    GBT_CONFIGS,
    GBT_ESTIMATORS,
    MODEL_FAMILIES,
    fit_elastic_net,
    fit_gradient_boosting,
    feature_sets,
    hyperparameter_string,
    predict_gradient_boosting,
    predict_linear,
    stable_seed,
)
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
TARGET_DIR = ROOT / "Data" / "processed" / "compatibility_targets"
MODEL_DIR = ROOT / "Data" / "processed" / "compatibility_models"
ABLATION_DIR = ROOT / "Data" / "processed" / "compatibility_model_ablations"
OUTPUT_DIR = ROOT / "Data" / "processed" / "rolling_origin_stability"

MATRIX_PATH = TARGET_DIR / "transfer_compatibility_strict_model_matrix.csv"
FEATURE_PATH = MODEL_DIR / "model_feature_dictionary.csv"
PRIOR_SELECTION_PATH = ABLATION_DIR / "ablation_model_selection.csv"
PRIOR_RESULTS_PATH = ABLATION_DIR / "ablation_test_results.csv"
UPSTREAM_VERIFICATION_PATH = ABLATION_DIR / "independent_verification.json"

RANDOM_SEED = 20260809
BOOTSTRAP_REPETITIONS = 5000
FIRST_TRAIN_YEAR = 2018
VALIDATION_YEARS = [2019, 2020, 2021, 2022]
LOCKED_TARGETS = ["opportunity", "performance"]
POLICIES = [
    "rolling_selected_baseline",
    "rolling_selected_full",
    "rolling_selected_overall",
]


def origin_id(validation_year: int) -> str:
    return f"origin_{validation_year}"


def clean_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def selected_params(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    return json.loads(str(row["selected_hyperparameters"]))


def candidate_key(row: dict[str, Any] | pd.Series) -> tuple[float, float, str]:
    return (
        float(row["mae"] if "mae" in row else row["validation_mae"]),
        float(row["rmse"] if "rmse" in row else row["validation_rmse"]),
        str(row.get("hyperparameters", row.get("selected_hyperparameters", ""))),
    )


def stratified_paired_bootstrap(
    pairs: pd.DataFrame,
    seed: int,
    baseline_column: str = "baseline_prediction",
    candidate_column: str = "candidate_prediction",
) -> dict[str, float | int]:
    """Bootstrap paired errors within each evaluation season and pool by row count."""
    rng = np.random.default_rng(seed)
    total_n = len(pairs)
    mae_draw_sum = np.zeros(BOOTSTRAP_REPETITIONS, dtype=float)
    baseline_sq_sum = np.zeros(BOOTSTRAP_REPETITIONS, dtype=float)
    candidate_sq_sum = np.zeros(BOOTSTRAP_REPETITIONS, dtype=float)
    for _, group in pairs.groupby("origin_id", sort=True):
        actual = group["actual"].to_numpy(dtype=float)
        baseline = group[baseline_column].to_numpy(dtype=float)
        candidate = group[candidate_column].to_numpy(dtype=float)
        n = len(group)
        draw = rng.integers(0, n, size=(BOOTSTRAP_REPETITIONS, n))
        absolute_difference = np.abs(actual - baseline) - np.abs(actual - candidate)
        baseline_squared = np.square(actual - baseline)
        candidate_squared = np.square(actual - candidate)
        mae_draw_sum += absolute_difference[draw].sum(axis=1)
        baseline_sq_sum += baseline_squared[draw].sum(axis=1)
        candidate_sq_sum += candidate_squared[draw].sum(axis=1)

    actual = pairs["actual"].to_numpy(dtype=float)
    baseline = pairs[baseline_column].to_numpy(dtype=float)
    candidate = pairs[candidate_column].to_numpy(dtype=float)
    baseline_abs = np.abs(actual - baseline)
    candidate_abs = np.abs(actual - candidate)
    mae_draws = mae_draw_sum / total_n
    rmse_draws = np.sqrt(baseline_sq_sum / total_n) - np.sqrt(candidate_sq_sum / total_n)
    return {
        "pooled_n": int(total_n),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "pooled_baseline_mae": float(np.mean(baseline_abs)),
        "pooled_candidate_mae": float(np.mean(candidate_abs)),
        "pooled_mae_improvement": float(np.mean(baseline_abs - candidate_abs)),
        "pooled_relative_mae_improvement": float(
            np.mean(baseline_abs - candidate_abs) / np.mean(baseline_abs)
        ),
        "mae_improvement_ci_lower_95": float(np.quantile(mae_draws, 0.025)),
        "mae_improvement_ci_upper_95": float(np.quantile(mae_draws, 0.975)),
        "mae_probability_candidate_better": float(np.mean(mae_draws > 0)),
        "pooled_baseline_rmse": float(np.sqrt(np.mean(np.square(actual - baseline)))),
        "pooled_candidate_rmse": float(np.sqrt(np.mean(np.square(actual - candidate)))),
        "pooled_rmse_improvement": float(
            np.sqrt(np.mean(np.square(actual - baseline)))
            - np.sqrt(np.mean(np.square(actual - candidate)))
        ),
        "rmse_improvement_ci_lower_95": float(np.quantile(rmse_draws, 0.025)),
        "rmse_improvement_ci_upper_95": float(np.quantile(rmse_draws, 0.975)),
        "rmse_probability_candidate_better": float(np.mean(rmse_draws > 0)),
    }


def stability_label(summary: dict[str, Any]) -> str:
    origins_better = int(summary["origins_candidate_better"])
    origin_count = int(summary["origin_count"])
    pooled_improvement = float(summary["pooled_mae_improvement"])
    probability = float(summary["mae_probability_candidate_better"])
    if pooled_improvement > 0 and origins_better >= max(3, origin_count - 1) and probability >= 0.90:
        return "stable_positive"
    if pooled_improvement > 0 and origins_better >= math.ceil(origin_count / 2):
        return "mixed_positive"
    if pooled_improvement <= 0 and origins_better <= 1:
        return "unstable_negative"
    return "mixed"


def summarize_comparison(
    pairs: pd.DataFrame,
    origin_bootstrap: pd.DataFrame,
    seed: int,
) -> dict[str, Any]:
    pooled = stratified_paired_bootstrap(pairs, seed)
    improvements = origin_bootstrap["mae_improvement"].to_numpy(dtype=float)
    summary: dict[str, Any] = {
        **pooled,
        "origin_count": int(len(origin_bootstrap)),
        "origins_candidate_better": int(np.sum(improvements > 0)),
        "origin_win_rate": float(np.mean(improvements > 0)),
        "mean_origin_mae_improvement": float(np.mean(improvements)),
        "median_origin_mae_improvement": float(np.median(improvements)),
        "origin_improvement_std": float(np.std(improvements, ddof=1)) if len(improvements) > 1 else 0.0,
        "worst_origin_mae_improvement": float(np.min(improvements)),
        "best_origin_mae_improvement": float(np.max(improvements)),
    }
    summary["stability_label"] = stability_label(summary)
    return summary


def tune_candidates(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    target_config: dict[str, Any],
    target_key: str,
    feature_set_name: str,
    common: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    family_rows: dict[str, list[dict[str, Any]]] = {family: [] for family in MODEL_FAMILIES}

    for alpha in RIDGE_ALPHA_GRID:
        intercept, beta = fit_ridge(x_train, y_train, alpha)
        prediction = apply_bounds(
            predict_ridge(x_validation, intercept, beta),
            target_config["clip_lower"],
            target_config["clip_upper"],
        )
        row = {
            **common,
            "feature_set": feature_set_name,
            "model_family": "ridge",
            "alpha": alpha,
            "l1_ratio": None,
            "learning_rate": None,
            "max_depth": None,
            "min_samples_leaf": None,
            "max_features": None,
            "n_estimators": None,
            "iterations": None,
            "nonzero_coefficients": None,
            **metrics(y_validation, prediction),
        }
        row["hyperparameters"] = hyperparameter_string(row)
        rows.append(row)
        family_rows["ridge"].append(row)

    for alpha in ELASTIC_ALPHA_GRID:
        for l1_ratio in ELASTIC_L1_GRID:
            intercept, beta, iterations = fit_elastic_net(x_train, y_train, alpha, l1_ratio)
            prediction = apply_bounds(
                predict_linear(x_validation, intercept, beta),
                target_config["clip_lower"],
                target_config["clip_upper"],
            )
            row = {
                **common,
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
                **metrics(y_validation, prediction),
            }
            row["hyperparameters"] = hyperparameter_string(row)
            rows.append(row)
            family_rows["elastic_net"].append(row)

    for gbt_config in GBT_CONFIGS:
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
                target_config["clip_lower"],
                target_config["clip_upper"],
            )
            row = {
                **common,
                "feature_set": feature_set_name,
                "model_family": "gradient_boosted_trees",
                "alpha": None,
                "l1_ratio": None,
                **gbt_config,
                "n_estimators": n_estimators,
                "iterations": None,
                "nonzero_coefficients": None,
                **metrics(y_validation, prediction),
            }
            row["hyperparameters"] = hyperparameter_string(row)
            rows.append(row)
            family_rows["gradient_boosted_trees"].append(row)

    selected = {
        family: min(candidates, key=candidate_key) for family, candidates in family_rows.items()
    }
    return rows, selected


def fit_selected_model(
    family: str,
    selected: dict[str, Any],
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_evaluation: np.ndarray,
    target_config: dict[str, Any],
    target_key: str,
    feature_set_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if family == "ridge":
        intercept, beta = fit_ridge(x_fit, y_fit, float(selected["alpha"]))
        raw = predict_ridge(x_evaluation, intercept, beta)
    elif family == "elastic_net":
        intercept, beta, _ = fit_elastic_net(
            x_fit,
            y_fit,
            float(selected["alpha"]),
            float(selected["l1_ratio"]),
        )
        raw = predict_linear(x_evaluation, intercept, beta)
    else:
        params = {
            "learning_rate": float(selected["learning_rate"]),
            "max_depth": int(selected["max_depth"]),
            "min_samples_leaf": int(selected["min_samples_leaf"]),
            "max_features": int(selected["max_features"]),
            "n_estimators": int(selected["n_estimators"]),
        }
        seed = stable_seed(
            str(RANDOM_SEED), target_key, feature_set_name, "gbt", "selected_final"
        )
        intercept, trees = fit_gradient_boosting(x_fit, y_fit, seed=seed, **params)
        raw = predict_gradient_boosting(
            x_evaluation, intercept, trees, params["learning_rate"]
        )
    bounded = apply_bounds(raw, target_config["clip_lower"], target_config["clip_upper"])
    return raw, bounded


def paired_rows(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    candidate_label: str,
) -> pd.DataFrame:
    keys = ["origin_id", "target_key", "canonical_performance_id"]
    left = baseline[
        keys
        + [
            "canonical_player_name",
            "canonical_club_name",
            "season_start_year",
            "role_group",
            "actual",
            "prediction",
            "absolute_error",
        ]
    ].rename(
        columns={
            "prediction": "baseline_prediction",
            "absolute_error": "baseline_absolute_error",
        }
    )
    right = candidate[keys + ["actual", "prediction", "absolute_error"]].rename(
        columns={
            "actual": "candidate_actual",
            "prediction": "candidate_prediction",
            "absolute_error": "candidate_absolute_error",
        }
    )
    merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if not np.allclose(
        merged["actual"].to_numpy(dtype=float),
        merged["candidate_actual"].to_numpy(dtype=float),
        atol=1e-12,
        rtol=0,
    ):
        raise RuntimeError(f"Actual values do not align for {candidate_label}.")
    merged = merged.drop(columns="candidate_actual")
    merged["candidate_label"] = candidate_label
    merged["mae_improvement"] = (
        merged["baseline_absolute_error"] - merged["candidate_absolute_error"]
    )
    return merged


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(MATRIX_PATH, low_memory=False)
    feature_dictionary = pd.read_csv(FEATURE_PATH, low_memory=False)
    prior_selection = pd.read_csv(PRIOR_SELECTION_PATH, low_memory=False)
    prior_results = pd.read_csv(PRIOR_RESULTS_PATH, low_memory=False)
    upstream = json.loads(UPSTREAM_VERIFICATION_PATH.read_text(encoding="utf-8"))
    if upstream.get("status") != "pass":
        raise RuntimeError("Upstream compatibility ablations are not independently verified.")

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

    origin_rows = [
        {
            "origin_id": origin_id(validation_year),
            "train_start_year": FIRST_TRAIN_YEAR,
            "train_end_year": validation_year - 1,
            "validation_year": validation_year,
            "evaluation_year": validation_year + 1,
            "training_window": f"{FIRST_TRAIN_YEAR}-{validation_year - 1}",
            "selection_rule": "minimum validation MAE, then RMSE, then hyperparameter JSON",
        }
        for validation_year in VALIDATION_YEARS
    ]
    origins = pd.DataFrame(origin_rows)

    add_check(
        "upstream_ablation_verification_passes",
        upstream.get("status") == "pass",
        upstream.get("status"),
        "pass",
        "Rolling tests begin from the independently verified ablation phase.",
    )
    add_check(
        "four_declared_expanding_origins",
        origins[["validation_year", "evaluation_year"]].values.tolist()
        == [[2019, 2020], [2020, 2021], [2021, 2022], [2022, 2023]],
        json.dumps(origins[["validation_year", "evaluation_year"]].values.tolist()),
        "[[2019,2020],[2020,2021],[2021,2022],[2022,2023]]",
        "Every origin validates on one season and evaluates on the immediately following season.",
    )
    add_check(
        "feature_sets_match_prior_ablation",
        list(sets) == FEATURE_SET_ORDER,
        ";".join(sets),
        ";".join(FEATURE_SET_ORDER),
        "The rolling grid holds the previously declared five feature designs fixed.",
    )
    forbidden = sorted(
        {
            feature
            for values in sets.values()
            for feature in values
            if feature.startswith(FORBIDDEN_PREDICTOR_PREFIXES)
        }
    )
    add_check(
        "no_forbidden_predictors",
        not forbidden,
        ";".join(forbidden),
        "none",
        "Targets, eligibility flags, target diagnostics, and roster scenarios are excluded.",
    )

    feature_rows = []
    for set_name, values in sets.items():
        for feature_order, feature in enumerate(values, start=1):
            feature_rows.append(
                {
                    "feature_set": set_name,
                    "feature_order": feature_order,
                    "raw_feature": feature,
                    "introduced_block": feature_blocks[feature],
                    "raw_feature_count": len(values),
                }
            )
    feature_set_frame = pd.DataFrame(feature_rows)

    tuning_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []

    for target_index, target_config in enumerate(TARGETS):
        target_key = target_config["target_key"]
        target_column = target_config["target_column"]
        eligible = bool_series(matrix[target_config["eligibility_column"]]) & matrix[
            target_column
        ].notna()
        cohort = matrix.loc[eligible].copy()
        cohort[target_column] = pd.to_numeric(cohort[target_column], errors="coerce")
        cohort = cohort.loc[cohort[target_column].notna()].copy()
        cohort = cohort.sort_values(
            ["season_start_year", "canonical_performance_id"]
        ).reset_index(drop=True)

        for origin_index, origin in origins.iterrows():
            validation_year = int(origin["validation_year"])
            evaluation_year = int(origin["evaluation_year"])
            train = cohort.loc[
                cohort["season_start_year"].between(FIRST_TRAIN_YEAR, validation_year - 1)
            ].copy()
            validation = cohort.loc[cohort["season_start_year"].eq(validation_year)].copy()
            evaluation = cohort.loc[cohort["season_start_year"].eq(evaluation_year)].copy()
            fit_frame = pd.concat([train, validation], ignore_index=True)
            y_train = train[target_column].to_numpy(dtype=float)
            y_validation = validation[target_column].to_numpy(dtype=float)
            y_fit = fit_frame[target_column].to_numpy(dtype=float)
            y_evaluation = evaluation[target_column].to_numpy(dtype=float)
            cohort_rows.append(
                {
                    "origin_id": origin["origin_id"],
                    "target_key": target_key,
                    "target_column": target_column,
                    "train_start_year": FIRST_TRAIN_YEAR,
                    "train_end_year": validation_year - 1,
                    "validation_year": validation_year,
                    "evaluation_year": evaluation_year,
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "evaluation_rows": len(evaluation),
                }
            )
            add_check(
                f"{target_key}_{origin['origin_id']}_cohorts_nonempty",
                min(len(train), len(validation), len(evaluation)) > 0,
                f"train={len(train)};validation={len(validation)};evaluation={len(evaluation)}",
                "all >0",
                "Every rolling origin has train, validation, and evaluation observations.",
            )

            for feature_set_name in FEATURE_SET_ORDER:
                print(
                    f"rolling target={target_key} origin={origin['origin_id']} feature_set={feature_set_name}",
                    flush=True,
                )
                raw_features = sets[feature_set_name]
                train_preprocessor = fit_preprocessor(train, raw_features)
                x_train = train_preprocessor.transform(train)
                x_validation = train_preprocessor.transform(validation)
                common = {
                    "target_key": target_key,
                    "target_column": target_column,
                    "origin_id": origin["origin_id"],
                    "train_start_year": FIRST_TRAIN_YEAR,
                    "train_end_year": validation_year - 1,
                    "validation_year": validation_year,
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "raw_feature_count": len(raw_features),
                    "encoded_feature_count": x_train.shape[1],
                }
                candidates, selected_by_family = tune_candidates(
                    x_train,
                    y_train,
                    x_validation,
                    y_validation,
                    target_config,
                    target_key,
                    feature_set_name,
                    common,
                )
                tuning_rows.extend(candidates)

                final_preprocessor = fit_preprocessor(fit_frame, raw_features)
                x_fit = final_preprocessor.transform(fit_frame)
                x_evaluation = final_preprocessor.transform(evaluation)
                for family_index, model_family in enumerate(MODEL_FAMILIES):
                    selected = selected_by_family[model_family]
                    raw_prediction, prediction = fit_selected_model(
                        model_family,
                        selected,
                        x_fit,
                        y_fit,
                        x_evaluation,
                        target_config,
                        target_key,
                        feature_set_name,
                    )
                    evaluation_metrics = metrics(y_evaluation, prediction)
                    selected_rows.append(
                        {
                            "target_key": target_key,
                            "target_column": target_column,
                            "origin_id": origin["origin_id"],
                            "train_start_year": FIRST_TRAIN_YEAR,
                            "train_end_year": validation_year - 1,
                            "validation_year": validation_year,
                            "evaluation_year": evaluation_year,
                            "feature_set": feature_set_name,
                            "model_family": model_family,
                            "selected_hyperparameters": selected["hyperparameters"],
                            "raw_feature_count": len(raw_features),
                            "train_rows": len(train),
                            "validation_rows": len(validation),
                            "evaluation_rows": len(evaluation),
                            "encoded_feature_count_train": x_train.shape[1],
                            "encoded_feature_count_fit": x_fit.shape[1],
                            "validation_mae": selected["mae"],
                            "validation_rmse": selected["rmse"],
                            "validation_r2": selected["r2"],
                            "validation_pearson": selected["pearson"],
                            "validation_spearman": selected["spearman"],
                            "evaluation_n": evaluation_metrics["n"],
                            "evaluation_mae": evaluation_metrics["mae"],
                            "evaluation_rmse": evaluation_metrics["rmse"],
                            "evaluation_r2": evaluation_metrics["r2"],
                            "evaluation_pearson": evaluation_metrics["pearson"],
                            "evaluation_spearman": evaluation_metrics["spearman"],
                            "evaluation_actual_mean": evaluation_metrics["actual_mean"],
                            "evaluation_predicted_mean": evaluation_metrics["predicted_mean"],
                            "evaluation_actual_std": evaluation_metrics["actual_std"],
                        }
                    )
                    for row_number, (_, scored_row) in enumerate(evaluation.iterrows()):
                        prediction_rows.append(
                            {
                                "origin_id": origin["origin_id"],
                                "train_end_year": validation_year - 1,
                                "validation_year": validation_year,
                                "evaluation_year": evaluation_year,
                                "canonical_performance_id": scored_row[
                                    "canonical_performance_id"
                                ],
                                "canonical_player_name": scored_row["canonical_player_name"],
                                "canonical_club_name": scored_row["canonical_club_name"],
                                "season_start_year": int(scored_row["season_start_year"]),
                                "role_group": scored_row["role_group"],
                                "canonical_competition_id": scored_row.get(
                                    "canonical_competition_id"
                                ),
                                "target_key": target_key,
                                "target_column": target_column,
                                "feature_set": feature_set_name,
                                "model_family": model_family,
                                "selected_hyperparameters": selected["hyperparameters"],
                                "actual": y_evaluation[row_number],
                                "prediction_raw": raw_prediction[row_number],
                                "prediction": prediction[row_number],
                                "residual": y_evaluation[row_number] - prediction[row_number],
                                "absolute_error": abs(
                                    y_evaluation[row_number] - prediction[row_number]
                                ),
                            }
                        )

    tuning = pd.DataFrame(tuning_rows)
    selected_specs = pd.DataFrame(selected_rows)
    predictions = pd.DataFrame(prediction_rows)
    cohorts = pd.DataFrame(cohort_rows)

    feature_effect_rows = []
    for (target_key, rolling_origin, model_family), group in selected_specs.groupby(
        ["target_key", "origin_id", "model_family"], sort=True
    ):
        baseline = group.loc[group["feature_set"].eq("player_history_baseline")].iloc[0]
        for _, row in group.iterrows():
            feature_effect_rows.append(
                {
                    **row.to_dict(),
                    "validation_mae_improvement_vs_baseline": float(
                        baseline["validation_mae"] - row["validation_mae"]
                    ),
                    "evaluation_mae_improvement_vs_baseline": float(
                        baseline["evaluation_mae"] - row["evaluation_mae"]
                    ),
                    "evaluation_relative_mae_improvement_vs_baseline": float(
                        (baseline["evaluation_mae"] - row["evaluation_mae"])
                        / baseline["evaluation_mae"]
                    ),
                }
            )
    feature_effects = pd.DataFrame(feature_effect_rows)

    family_bootstrap_rows = []
    family_pair_frames = []
    for target_index, target_config in enumerate(TARGETS):
        target_key = target_config["target_key"]
        for family_index, model_family in enumerate(MODEL_FAMILIES):
            for origin_index, rolling_origin in enumerate(origins["origin_id"]):
                spec = predictions.loc[
                    predictions["target_key"].eq(target_key)
                    & predictions["origin_id"].eq(rolling_origin)
                    & predictions["model_family"].eq(model_family)
                ]
                baseline = spec.loc[
                    spec["feature_set"].eq("player_history_baseline")
                ].sort_values("canonical_performance_id")
                full = spec.loc[spec["feature_set"].eq("full_explicit_fit")].sort_values(
                    "canonical_performance_id"
                )
                result = paired_bootstrap(
                    baseline["actual"].to_numpy(dtype=float),
                    baseline["prediction"].to_numpy(dtype=float),
                    full["prediction"].to_numpy(dtype=float),
                    RANDOM_SEED + 1000 + target_index * 100 + family_index * 10 + origin_index,
                )
                result["evaluation_n"] = result.pop("test_n")
                family_bootstrap_rows.append(
                    {
                        "target_key": target_key,
                        "origin_id": rolling_origin,
                        "validation_year": int(
                            origins.loc[origins["origin_id"].eq(rolling_origin), "validation_year"].iloc[0]
                        ),
                        "evaluation_year": int(
                            origins.loc[origins["origin_id"].eq(rolling_origin), "evaluation_year"].iloc[0]
                        ),
                        "model_family": model_family,
                        "comparison": "full_explicit_fit_vs_player_history_baseline",
                        **result,
                    }
                )
                pair = paired_rows(baseline, full, "full_explicit_fit")
                pair["model_family"] = model_family
                family_pair_frames.append(pair)
    family_bootstrap = pd.DataFrame(family_bootstrap_rows)
    family_pairs = pd.concat(family_pair_frames, ignore_index=True)

    family_stability_rows = []
    for target_index, target_key in enumerate([config["target_key"] for config in TARGETS]):
        for family_index, model_family in enumerate(MODEL_FAMILIES):
            pairs = family_pairs.loc[
                family_pairs["target_key"].eq(target_key)
                & family_pairs["model_family"].eq(model_family)
            ]
            origin_results = family_bootstrap.loc[
                family_bootstrap["target_key"].eq(target_key)
                & family_bootstrap["model_family"].eq(model_family)
            ]
            family_stability_rows.append(
                {
                    "target_key": target_key,
                    "model_family": model_family,
                    "comparison": "full_explicit_fit_vs_player_history_baseline",
                    **summarize_comparison(
                        pairs,
                        origin_results,
                        RANDOM_SEED + 2000 + target_index * 10 + family_index,
                    ),
                }
            )
    family_stability = pd.DataFrame(family_stability_rows)

    policy_selection_rows = []
    policy_prediction_frames = []
    for (target_key, rolling_origin), group in selected_specs.groupby(
        ["target_key", "origin_id"], sort=True
    ):
        filters = {
            "rolling_selected_baseline": group["feature_set"].eq(
                "player_history_baseline"
            ),
            "rolling_selected_full": group["feature_set"].eq("full_explicit_fit"),
            "rolling_selected_overall": pd.Series(True, index=group.index),
        }
        for policy in POLICIES:
            pool = group.loc[filters[policy]].copy()
            chosen = pool.sort_values(
                [
                    "validation_mae",
                    "validation_rmse",
                    "model_family",
                    "feature_set",
                    "selected_hyperparameters",
                ],
                kind="mergesort",
            ).iloc[0]
            policy_selection_rows.append(
                {
                    "selection_type": policy,
                    **chosen.to_dict(),
                    "candidate_specifications_considered": len(pool),
                }
            )
            selected_predictions = predictions.loc[
                predictions["target_key"].eq(target_key)
                & predictions["origin_id"].eq(rolling_origin)
                & predictions["feature_set"].eq(chosen["feature_set"])
                & predictions["model_family"].eq(chosen["model_family"])
            ].copy()
            selected_predictions.insert(0, "selection_type", policy)
            policy_prediction_frames.append(selected_predictions)
    policy_selections = pd.DataFrame(policy_selection_rows)
    policy_predictions = pd.concat(policy_prediction_frames, ignore_index=True)

    policy_bootstrap_rows = []
    policy_pair_frames = []
    for target_index, target_key in enumerate([config["target_key"] for config in TARGETS]):
        for origin_index, rolling_origin in enumerate(origins["origin_id"]):
            origin_predictions = policy_predictions.loc[
                policy_predictions["target_key"].eq(target_key)
                & policy_predictions["origin_id"].eq(rolling_origin)
            ]
            baseline = origin_predictions.loc[
                origin_predictions["selection_type"].eq("rolling_selected_baseline")
            ].sort_values("canonical_performance_id")
            full = origin_predictions.loc[
                origin_predictions["selection_type"].eq("rolling_selected_full")
            ].sort_values("canonical_performance_id")
            result = paired_bootstrap(
                baseline["actual"].to_numpy(dtype=float),
                baseline["prediction"].to_numpy(dtype=float),
                full["prediction"].to_numpy(dtype=float),
                RANDOM_SEED + 3000 + target_index * 10 + origin_index,
            )
            result["evaluation_n"] = result.pop("test_n")
            policy_bootstrap_rows.append(
                {
                    "target_key": target_key,
                    "origin_id": rolling_origin,
                    "comparison": "rolling_selected_full_vs_rolling_selected_baseline",
                    **result,
                }
            )
            policy_pair_frames.append(paired_rows(baseline, full, "rolling_selected_full"))
    policy_bootstrap = pd.DataFrame(policy_bootstrap_rows)
    policy_pairs = pd.concat(policy_pair_frames, ignore_index=True)

    policy_comparison_stability_rows = []
    for target_index, target_key in enumerate([config["target_key"] for config in TARGETS]):
        pairs = policy_pairs.loc[policy_pairs["target_key"].eq(target_key)]
        origin_results = policy_bootstrap.loc[policy_bootstrap["target_key"].eq(target_key)]
        policy_comparison_stability_rows.append(
            {
                "target_key": target_key,
                "comparison": "rolling_selected_full_vs_rolling_selected_baseline",
                **summarize_comparison(
                    pairs, origin_results, RANDOM_SEED + 4000 + target_index
                ),
            }
        )
    policy_comparison_stability = pd.DataFrame(policy_comparison_stability_rows)

    policy_stability_rows = []
    for (target_key, policy), group in policy_predictions.groupby(
        ["target_key", "selection_type"], sort=True
    ):
        result = metrics(
            group["actual"].to_numpy(dtype=float), group["prediction"].to_numpy(dtype=float)
        )
        origin_mae = group.groupby("origin_id")["absolute_error"].mean()
        selections = policy_selections.loc[
            policy_selections["target_key"].eq(target_key)
            & policy_selections["selection_type"].eq(policy)
        ].copy()
        selections["spec"] = selections["feature_set"] + " | " + selections["model_family"]
        frequencies = selections["spec"].value_counts()
        policy_stability_rows.append(
            {
                "target_key": target_key,
                "selection_type": policy,
                "origin_count": int(origin_mae.size),
                "pooled_n": result["n"],
                "pooled_mae": result["mae"],
                "pooled_rmse": result["rmse"],
                "pooled_r2": result["r2"],
                "pooled_pearson": result["pearson"],
                "pooled_spearman": result["spearman"],
                "mean_origin_mae": float(origin_mae.mean()),
                "origin_mae_std": float(origin_mae.std(ddof=1)),
                "worst_origin_mae": float(origin_mae.max()),
                "best_origin_mae": float(origin_mae.min()),
                "distinct_selected_specs": int(len(frequencies)),
                "most_frequent_spec": frequencies.index[0],
                "most_frequent_spec_origins": int(frequencies.iloc[0]),
            }
        )
    policy_stability = pd.DataFrame(policy_stability_rows)

    selection_frequency = (
        policy_selections.assign(
            selected_spec=lambda frame: frame["feature_set"] + " | " + frame["model_family"]
        )
        .groupby(
            ["target_key", "selection_type", "feature_set", "model_family", "selected_spec"],
            dropna=False,
            sort=True,
        )
        .agg(
            origin_selections=("origin_id", "size"),
            mean_validation_mae=("validation_mae", "mean"),
            mean_evaluation_mae=("evaluation_mae", "mean"),
        )
        .reset_index()
    )
    selection_frequency["origin_selection_rate"] = (
        selection_frequency["origin_selections"] / len(origins)
    )

    locked_metric_rows = []
    locked_prediction_frames = []
    locked_design_rows = []
    for target_key in LOCKED_TARGETS:
        for design_role, source_type in [
            ("baseline", "validation_selected_baseline"),
            ("candidate", "validation_selected_overall"),
        ]:
            design = prior_selection.loc[
                prior_selection["target_key"].eq(target_key)
                & prior_selection["selection_type"].eq(source_type)
            ].iloc[0]
            locked_design_rows.append(
                {
                    "target_key": target_key,
                    "design_role": design_role,
                    "feature_set": design["feature_set"],
                    "model_family": design["model_family"],
                    "design_source": source_type,
                    "design_selection_year": 2022,
                    "design_evaluation_year_used_in_choice": False,
                    "interpretation": "retrospective design stress test; hyperparameters retuned within each origin",
                }
            )
            for rolling_origin in origins["origin_id"]:
                selected = selected_specs.loc[
                    selected_specs["target_key"].eq(target_key)
                    & selected_specs["origin_id"].eq(rolling_origin)
                    & selected_specs["feature_set"].eq(design["feature_set"])
                    & selected_specs["model_family"].eq(design["model_family"])
                ].iloc[0]
                locked_metric_rows.append(
                    {
                        "design_role": design_role,
                        "design_source": source_type,
                        **selected.to_dict(),
                    }
                )
                selected_predictions = predictions.loc[
                    predictions["target_key"].eq(target_key)
                    & predictions["origin_id"].eq(rolling_origin)
                    & predictions["feature_set"].eq(design["feature_set"])
                    & predictions["model_family"].eq(design["model_family"])
                ].copy()
                selected_predictions.insert(0, "design_role", design_role)
                locked_prediction_frames.append(selected_predictions)
    locked_designs = pd.DataFrame(locked_design_rows)
    locked_metrics = pd.DataFrame(locked_metric_rows)
    locked_predictions = pd.concat(locked_prediction_frames, ignore_index=True)

    locked_bootstrap_rows = []
    locked_pair_frames = []
    for target_index, target_key in enumerate(LOCKED_TARGETS):
        for origin_index, rolling_origin in enumerate(origins["origin_id"]):
            origin_predictions = locked_predictions.loc[
                locked_predictions["target_key"].eq(target_key)
                & locked_predictions["origin_id"].eq(rolling_origin)
            ]
            baseline = origin_predictions.loc[
                origin_predictions["design_role"].eq("baseline")
            ].sort_values("canonical_performance_id")
            candidate = origin_predictions.loc[
                origin_predictions["design_role"].eq("candidate")
            ].sort_values("canonical_performance_id")
            result = paired_bootstrap(
                baseline["actual"].to_numpy(dtype=float),
                baseline["prediction"].to_numpy(dtype=float),
                candidate["prediction"].to_numpy(dtype=float),
                RANDOM_SEED + 5000 + target_index * 10 + origin_index,
            )
            result["evaluation_n"] = result.pop("test_n")
            locked_bootstrap_rows.append(
                {
                    "target_key": target_key,
                    "origin_id": rolling_origin,
                    "comparison": "locked_current_candidate_vs_locked_current_baseline",
                    **result,
                }
            )
            locked_pair_frames.append(
                paired_rows(baseline, candidate, "locked_current_candidate")
            )
    locked_bootstrap = pd.DataFrame(locked_bootstrap_rows)
    locked_pairs = pd.concat(locked_pair_frames, ignore_index=True)

    locked_stability_rows = []
    for target_index, target_key in enumerate(LOCKED_TARGETS):
        target_pairs = locked_pairs.loc[locked_pairs["target_key"].eq(target_key)]
        origin_results = locked_bootstrap.loc[locked_bootstrap["target_key"].eq(target_key)]
        baseline_design = locked_designs.loc[
            locked_designs["target_key"].eq(target_key)
            & locked_designs["design_role"].eq("baseline")
        ].iloc[0]
        candidate_design = locked_designs.loc[
            locked_designs["target_key"].eq(target_key)
            & locked_designs["design_role"].eq("candidate")
        ].iloc[0]
        locked_stability_rows.append(
            {
                "target_key": target_key,
                "baseline_feature_set": baseline_design["feature_set"],
                "baseline_model_family": baseline_design["model_family"],
                "candidate_feature_set": candidate_design["feature_set"],
                "candidate_model_family": candidate_design["model_family"],
                "comparison": "locked_current_candidate_vs_locked_current_baseline",
                **summarize_comparison(
                    target_pairs,
                    origin_results,
                    RANDOM_SEED + 6000 + target_index,
                ),
            }
        )
    locked_stability = pd.DataFrame(locked_stability_rows)

    # Complete-run invariants.
    expected_tuning_rows = (
        len(TARGETS)
        * len(origins)
        * len(FEATURE_SET_ORDER)
        * (len(RIDGE_ALPHA_GRID) + len(ELASTIC_ALPHA_GRID) * len(ELASTIC_L1_GRID) + len(GBT_CONFIGS) * len(GBT_ESTIMATORS))
    )
    add_check(
        "tuning_grid_complete",
        len(tuning) == expected_tuning_rows,
        len(tuning),
        expected_tuning_rows,
        "Every origin evaluates all declared hyperparameter candidates using validation data only.",
    )
    add_check(
        "selected_spec_grid_complete",
        len(selected_specs)
        == len(TARGETS) * len(origins) * len(FEATURE_SET_ORDER) * len(MODEL_FAMILIES),
        len(selected_specs),
        len(TARGETS) * len(origins) * len(FEATURE_SET_ORDER) * len(MODEL_FAMILIES),
        "Every target, origin, feature set, and model family has one past-only selected specification.",
    )
    selection_failures = []
    for keys, group in tuning.groupby(
        ["target_key", "origin_id", "feature_set", "model_family"], sort=True
    ):
        expected = group.sort_values(
            ["mae", "rmse", "hyperparameters"], kind="mergesort"
        ).iloc[0]
        reported = selected_specs.loc[
            selected_specs["target_key"].eq(keys[0])
            & selected_specs["origin_id"].eq(keys[1])
            & selected_specs["feature_set"].eq(keys[2])
            & selected_specs["model_family"].eq(keys[3])
        ].iloc[0]
        if expected["hyperparameters"] != reported["selected_hyperparameters"]:
            selection_failures.append(":".join(keys))
    add_check(
        "all_hyperparameters_selected_from_origin_validation",
        not selection_failures,
        ";".join(selection_failures),
        "none",
        "No evaluation metric participates in hyperparameter selection.",
    )
    add_check(
        "tuning_has_no_evaluation_metrics",
        not any(column.startswith("evaluation_") for column in tuning.columns),
        ";".join(column for column in tuning.columns if column.startswith("evaluation_")),
        "none",
        "Tuning artifacts contain validation diagnostics only.",
    )
    expected_prediction_rows = int(
        sum(row["evaluation_rows"] for row in cohort_rows)
        * len(FEATURE_SET_ORDER)
        * len(MODEL_FAMILIES)
    )
    add_check(
        "prediction_row_count_complete",
        len(predictions) == expected_prediction_rows,
        len(predictions),
        expected_prediction_rows,
        "Every selected specification scores every eligible evaluation row.",
    )
    add_check(
        "all_predictions_finite",
        np.isfinite(
            predictions[["actual", "prediction_raw", "prediction", "absolute_error"]].to_numpy(
                dtype=float
            )
        ).all(),
        int(
            (~np.isfinite(
                predictions[["actual", "prediction_raw", "prediction", "absolute_error"]].to_numpy(
                    dtype=float
                )
            )).sum()
        ),
        0,
        "Every rolling evaluation prediction is finite.",
    )
    opportunity_predictions = predictions.loc[
        predictions["target_key"].eq("opportunity"), "prediction"
    ]
    add_check(
        "opportunity_predictions_bounded",
        opportunity_predictions.between(0, 1).all(),
        f"min={opportunity_predictions.min():.6f};max={opportunity_predictions.max():.6f}",
        "within [0,1]",
        "Opportunity predictions respect their natural support at every origin.",
    )
    key_variation = (
        predictions.groupby(["target_key", "origin_id", "feature_set", "model_family"])[
            "canonical_performance_id"
        ]
        .apply(lambda values: frozenset(values.astype(str)))
        .groupby(level=[0, 1])
        .nunique()
    )
    add_check(
        "evaluation_keys_match_all_specs",
        key_variation.eq(1).all(),
        int((~key_variation.eq(1)).sum()),
        0,
        "Model and feature choices never change a target-origin evaluation cohort.",
    )
    prior_comparison = selected_specs.loc[selected_specs["origin_id"].eq("origin_2022")].merge(
        prior_results,
        on=["target_key", "feature_set", "model_family"],
        suffixes=("_rolling", "_prior"),
        validate="one_to_one",
    )
    reproduced = (
        np.allclose(
            prior_comparison["validation_mae_rolling"],
            prior_comparison["validation_mae_prior"],
            atol=1e-10,
            rtol=0,
        )
        and np.allclose(
            prior_comparison["evaluation_mae"],
            prior_comparison["test_mae"],
            atol=1e-10,
            rtol=0,
        )
        and (
            prior_comparison["selected_hyperparameters_rolling"]
            == prior_comparison["selected_hyperparameters_prior"]
        ).all()
    )
    add_check(
        "origin_2022_exactly_reproduces_prior_ablation",
        reproduced,
        f"matched_rows={len(prior_comparison)}",
        "45 exact specifications",
        "The terminal rolling origin is an exact extension of the prior one-shot experiment.",
    )
    add_check(
        "policy_selection_grid_complete",
        len(policy_selections) == len(TARGETS) * len(origins) * len(POLICIES),
        len(policy_selections),
        len(TARGETS) * len(origins) * len(POLICIES),
        "Every origin has baseline-only, full-only, and all-grid validation policies.",
    )
    add_check(
        "locked_designs_exclude_paused_adaptation",
        set(locked_designs["target_key"]) == set(LOCKED_TARGETS),
        ";".join(sorted(locked_designs["target_key"].unique())),
        ";".join(LOCKED_TARGETS),
        "Only opportunity and performance receive current-candidate stability decisions.",
    )
    add_check(
        "locked_design_metric_grid_complete",
        len(locked_metrics) == len(LOCKED_TARGETS) * len(origins) * 2,
        len(locked_metrics),
        len(LOCKED_TARGETS) * len(origins) * 2,
        "Each locked baseline and candidate design is evaluated at every origin.",
    )
    add_check(
        "comparison_bootstrap_grid_complete",
        len(family_bootstrap) == len(TARGETS) * len(origins) * len(MODEL_FAMILIES)
        and len(policy_bootstrap) == len(TARGETS) * len(origins)
        and len(locked_bootstrap) == len(LOCKED_TARGETS) * len(origins),
        f"family={len(family_bootstrap)};policy={len(policy_bootstrap)};locked={len(locked_bootstrap)}",
        "family=36;policy=12;locked=8",
        "All declared paired comparisons have origin-level uncertainty estimates.",
    )

    checks_frame = pd.DataFrame(checks)
    if not checks_frame["passed"].all():
        raise RuntimeError(
            f"Blocking rolling-origin checks failed: {checks_frame.loc[~checks_frame['passed'], 'check'].tolist()}"
        )

    outputs = {
        "rolling_origin_definitions.csv": origins,
        "rolling_origin_cohorts.csv": cohorts,
        "rolling_feature_sets.csv": feature_set_frame,
        "rolling_validation_tuning.csv": tuning,
        "rolling_selected_specs.csv": selected_specs,
        "rolling_predictions.csv": predictions,
        "rolling_origin_feature_effects.csv": feature_effects,
        "rolling_family_bootstrap.csv": family_bootstrap,
        "rolling_family_stability_summary.csv": family_stability,
        "rolling_policy_selections.csv": policy_selections,
        "rolling_policy_predictions.csv": policy_predictions,
        "rolling_policy_bootstrap.csv": policy_bootstrap,
        "rolling_policy_stability_summary.csv": policy_stability,
        "rolling_policy_comparison_stability.csv": policy_comparison_stability,
        "rolling_selection_frequency.csv": selection_frequency,
        "rolling_locked_designs.csv": locked_designs,
        "rolling_locked_metrics.csv": locked_metrics,
        "rolling_locked_predictions.csv": locked_predictions,
        "rolling_locked_bootstrap.csv": locked_bootstrap,
        "rolling_locked_stability_summary.csv": locked_stability,
        "rolling_build_checks.csv": checks_frame,
    }
    for filename, frame in outputs.items():
        frame.to_csv(OUTPUT_DIR / filename, index=False)

    summary = {
        "status": "pass",
        "origins": len(origins),
        "targets": len(TARGETS),
        "locked_targets": LOCKED_TARGETS,
        "feature_sets": {name: len(values) for name, values in sets.items()},
        "model_families": MODEL_FAMILIES,
        "tuning_candidates": len(tuning),
        "selected_specifications": len(selected_specs),
        "evaluation_prediction_rows": len(predictions),
        "selection_policy": "expanding training window; minimum origin validation MAE; next season evaluated after selection",
        "blocking_checks": len(checks_frame),
        "blocking_failures": int((~checks_frame["passed"]).sum()),
        "random_seed": RANDOM_SEED,
    }
    (OUTPUT_DIR / "rolling_run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    source_paths = [
        MATRIX_PATH,
        FEATURE_PATH,
        PRIOR_SELECTION_PATH,
        PRIOR_RESULTS_PATH,
        UPSTREAM_VERIFICATION_PATH,
    ]
    pd.DataFrame(
        [
            {
                "source_file": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in source_paths
        ]
    ).to_csv(OUTPUT_DIR / "source_manifest.csv", index=False)

    readme = """# MoveMaker rolling-origin stability testing

This phase runs four expanding-window origins: train through 2018 / validate 2019 / evaluate 2020, then advances one season at a time through the terminal train-through-2021 / validate-2022 / evaluate-2023 origin. At every origin, all ridge, elastic-net, and deterministic boosted-tree hyperparameters are selected using that origin's validation season only. The following season is scored after selection.

`rolling_policy_comparison_stability.csv` is the primary leakage-safe policy result: at each origin it separately selects the best baseline family and best full-compatibility family on validation data, then compares them on the next season. `rolling_family_stability_summary.csv` holds model family fixed while comparing full compatibility with the player-history baseline.

`rolling_locked_stability_summary.csv` is a retrospective design stress test for the current opportunity and performance candidates chosen in the preceding 2022-validation phase. It locks feature set and model family but retunes hyperparameters inside each historical origin. Because the design itself was chosen later, use this as a stability diagnostic rather than an unbiased prospective performance estimate. Adaptation remains diagnostic-only and is excluded from the locked-candidate decision.

`rolling_sensitivity_summary.csv` repeats policy, locked-design, and within-family comparisons across the mature three origins and terminal two origins. This prevents the smallest initial training window from dominating the interpretation.

Positive MAE improvement means the compatibility candidate made lower absolute error than its paired baseline. The declared stability label is `stable_positive` when the pooled improvement is positive, at least three of four origins improve, and the stratified bootstrap probability of improvement is at least 90%; `mixed_positive` when pooled improvement is positive and at least half the origins improve; `unstable_negative` when pooled improvement is non-positive and at most one origin improves; otherwise it is `mixed`.
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
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()

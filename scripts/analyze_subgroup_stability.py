"""Analyze frozen compatibility candidates by role and competition."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_compatibility_models import sha256_file


ROOT = Path(__file__).resolve().parents[1]
ROLLING_DIR = ROOT / "Data" / "processed" / "rolling_origin_stability"
CROSSWALK_DIR = ROOT / "Data" / "processed" / "integration_crosswalks"
OUTPUT_DIR = ROOT / "Data" / "processed" / "subgroup_stability"

PREDICTIONS_PATH = ROLLING_DIR / "rolling_predictions.csv"
POLICY_PREDICTIONS_PATH = ROLLING_DIR / "rolling_policy_predictions.csv"
ROLLING_SENSITIVITY_PATH = ROLLING_DIR / "rolling_sensitivity_summary.csv"
UPSTREAM_VERIFICATION_PATH = ROLLING_DIR / "independent_verification.json"
COMPETITION_CROSSWALK_PATH = CROSSWALK_DIR / "competition_crosswalk.csv"

RANDOM_SEED = 20260810
BOOTSTRAP_REPETITIONS = 5000
PERMUTATION_REPETITIONS = 5000
MIN_POOLED_N = 15
MIN_ORIGIN_N = 5
FDR_ALPHA = 0.10

CANDIDATES = {
    "opportunity": {
        "feature_set": "full_explicit_fit",
        "model_family": "gradient_boosted_trees",
        "target_label": "Opportunity",
    },
    "performance": {
        "feature_set": "full_explicit_fit",
        "model_family": "gradient_boosted_trees",
        "target_label": "Performance",
    },
}
REFERENCES = {
    "same_family_baseline": "Frozen learner using player-history baseline features",
    "rolling_selected_baseline": "Best baseline learner selected on each origin validation season",
}
SUBSETS = {
    "all_four_origins": ["origin_2019", "origin_2020", "origin_2021", "origin_2022"],
    "mature_three_origins": ["origin_2020", "origin_2021", "origin_2022"],
    "terminal_two_origins": ["origin_2021", "origin_2022"],
}
DIMENSIONS = {
    "role": ("role_group", "role_name"),
    "competition": ("canonical_competition_id", "competition_name"),
}
ROLE_NAMES = {"DF": "Defender", "FW": "Forward", "MF": "Midfielder"}


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def paired_bootstrap(
    actual: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    seed: int,
) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(actual)
    indices = rng.integers(0, n, size=(BOOTSTRAP_REPETITIONS, n))
    baseline_abs = np.abs(actual - baseline)
    candidate_abs = np.abs(actual - candidate)
    improvements = np.mean(baseline_abs[indices] - candidate_abs[indices], axis=1)
    baseline_sq = np.square(actual - baseline)
    candidate_sq = np.square(actual - candidate)
    rmse_improvements = np.sqrt(np.mean(baseline_sq[indices], axis=1)) - np.sqrt(
        np.mean(candidate_sq[indices], axis=1)
    )
    lower_tail = (np.sum(improvements <= 0) + 1) / (BOOTSTRAP_REPETITIONS + 1)
    upper_tail = (np.sum(improvements >= 0) + 1) / (BOOTSTRAP_REPETITIONS + 1)
    return {
        "n": int(n),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "baseline_mae": float(np.mean(baseline_abs)),
        "candidate_mae": float(np.mean(candidate_abs)),
        "mae_improvement": float(np.mean(baseline_abs - candidate_abs)),
        "relative_mae_improvement": float(
            np.mean(baseline_abs - candidate_abs) / np.mean(baseline_abs)
        ),
        "mae_improvement_ci_lower_95": float(np.quantile(improvements, 0.025)),
        "mae_improvement_ci_upper_95": float(np.quantile(improvements, 0.975)),
        "mae_probability_candidate_better": float(np.mean(improvements > 0)),
        "bootstrap_p_two_sided": float(min(1.0, 2.0 * min(lower_tail, upper_tail))),
        "baseline_rmse": float(np.sqrt(np.mean(baseline_sq))),
        "candidate_rmse": float(np.sqrt(np.mean(candidate_sq))),
        "rmse_improvement": float(
            np.sqrt(np.mean(baseline_sq)) - np.sqrt(np.mean(candidate_sq))
        ),
        "rmse_improvement_ci_lower_95": float(np.quantile(rmse_improvements, 0.025)),
        "rmse_improvement_ci_upper_95": float(np.quantile(rmse_improvements, 0.975)),
    }


def stratified_bootstrap(pairs: pd.DataFrame, seed: int) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    total_n = len(pairs)
    improvement_sum = np.zeros(BOOTSTRAP_REPETITIONS, dtype=float)
    baseline_sq_sum = np.zeros(BOOTSTRAP_REPETITIONS, dtype=float)
    candidate_sq_sum = np.zeros(BOOTSTRAP_REPETITIONS, dtype=float)
    for _, group in pairs.groupby("origin_id", sort=True):
        actual = group["actual"].to_numpy(dtype=float)
        baseline = group["baseline_prediction"].to_numpy(dtype=float)
        candidate = group["candidate_prediction"].to_numpy(dtype=float)
        n = len(group)
        indices = rng.integers(0, n, size=(BOOTSTRAP_REPETITIONS, n))
        improvement_sum += (
            np.abs(actual - baseline) - np.abs(actual - candidate)
        )[indices].sum(axis=1)
        baseline_sq_sum += np.square(actual - baseline)[indices].sum(axis=1)
        candidate_sq_sum += np.square(actual - candidate)[indices].sum(axis=1)
    actual = pairs["actual"].to_numpy(dtype=float)
    baseline = pairs["baseline_prediction"].to_numpy(dtype=float)
    candidate = pairs["candidate_prediction"].to_numpy(dtype=float)
    baseline_abs = np.abs(actual - baseline)
    candidate_abs = np.abs(actual - candidate)
    draws = improvement_sum / total_n
    rmse_draws = np.sqrt(baseline_sq_sum / total_n) - np.sqrt(
        candidate_sq_sum / total_n
    )
    lower_tail = (np.sum(draws <= 0) + 1) / (BOOTSTRAP_REPETITIONS + 1)
    upper_tail = (np.sum(draws >= 0) + 1) / (BOOTSTRAP_REPETITIONS + 1)
    return {
        "pooled_n": int(total_n),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "pooled_baseline_mae": float(np.mean(baseline_abs)),
        "pooled_candidate_mae": float(np.mean(candidate_abs)),
        "pooled_mae_improvement": float(np.mean(baseline_abs - candidate_abs)),
        "pooled_relative_mae_improvement": float(
            np.mean(baseline_abs - candidate_abs) / np.mean(baseline_abs)
        ),
        "mae_improvement_ci_lower_95": float(np.quantile(draws, 0.025)),
        "mae_improvement_ci_upper_95": float(np.quantile(draws, 0.975)),
        "mae_probability_candidate_better": float(np.mean(draws > 0)),
        "bootstrap_p_two_sided": float(min(1.0, 2.0 * min(lower_tail, upper_tail))),
        "pooled_baseline_rmse": float(np.sqrt(np.mean(np.square(actual - baseline)))),
        "pooled_candidate_rmse": float(
            np.sqrt(np.mean(np.square(actual - candidate)))
        ),
        "pooled_rmse_improvement": float(
            np.sqrt(np.mean(np.square(actual - baseline)))
            - np.sqrt(np.mean(np.square(actual - candidate)))
        ),
        "rmse_improvement_ci_lower_95": float(np.quantile(rmse_draws, 0.025)),
        "rmse_improvement_ci_upper_95": float(np.quantile(rmse_draws, 0.975)),
    }


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().astype(float).sort_values()
    if valid.empty:
        return result
    m = len(valid)
    ranked = valid.to_numpy() * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    result.loc[valid.index] = np.minimum(adjusted, 1.0)
    return result


def evidence_label(row: pd.Series) -> str:
    improvement = float(row["pooled_mae_improvement"])
    lower = float(row["mae_improvement_ci_lower_95"])
    upper = float(row["mae_improvement_ci_upper_95"])
    probability = float(row["mae_probability_candidate_better"])
    quality = row["sample_quality"]
    q_value = row.get("fdr_q_value")
    if quality != "adequate":
        if improvement > 0 and probability >= 0.80:
            return "small_cell_directional_positive"
        if improvement < 0 and probability <= 0.20:
            return "small_cell_directional_negative"
        return "small_cell_mixed"
    if pd.notna(q_value) and float(q_value) <= FDR_ALPHA and lower > 0:
        return "fdr_supported_positive"
    if pd.notna(q_value) and float(q_value) <= FDR_ALPHA and upper < 0:
        return "fdr_supported_negative"
    if lower > 0:
        return "nominal_supported_positive"
    if upper < 0:
        return "nominal_supported_negative"
    if improvement > 0 and probability >= 0.80:
        return "directional_positive"
    if improvement < 0 and probability <= 0.20:
        return "directional_negative"
    return "mixed"


def pair_frames(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    reference_type: str,
) -> pd.DataFrame:
    keys = ["origin_id", "target_key", "canonical_performance_id"]
    candidate_columns = keys + [
        "evaluation_year",
        "canonical_player_name",
        "canonical_club_name",
        "role_group",
        "canonical_competition_id",
        "actual",
        "prediction",
        "feature_set",
        "model_family",
        "selected_hyperparameters",
    ]
    baseline_columns = keys + [
        "actual",
        "prediction",
        "feature_set",
        "model_family",
        "selected_hyperparameters",
    ]
    left = candidate[candidate_columns].rename(
        columns={
            "prediction": "candidate_prediction",
            "feature_set": "candidate_feature_set",
            "model_family": "candidate_model_family",
            "selected_hyperparameters": "candidate_hyperparameters",
        }
    )
    right = baseline[baseline_columns].rename(
        columns={
            "actual": "baseline_actual",
            "prediction": "baseline_prediction",
            "feature_set": "baseline_feature_set",
            "model_family": "baseline_model_family",
            "selected_hyperparameters": "baseline_hyperparameters",
        }
    )
    paired = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if not np.allclose(
        paired["actual"].to_numpy(dtype=float),
        paired["baseline_actual"].to_numpy(dtype=float),
        atol=1e-12,
        rtol=0,
    ):
        raise RuntimeError(f"Actual values do not align for {reference_type}.")
    paired = paired.drop(columns="baseline_actual")
    paired["reference_type"] = reference_type
    paired["candidate_absolute_error"] = np.abs(
        paired["actual"] - paired["candidate_prediction"]
    )
    paired["baseline_absolute_error"] = np.abs(
        paired["actual"] - paired["baseline_prediction"]
    )
    paired["mae_improvement"] = (
        paired["baseline_absolute_error"] - paired["candidate_absolute_error"]
    )
    return paired


def heterogeneity_test(
    pairs: pd.DataFrame,
    group_column: str,
    seed: int,
) -> dict[str, Any]:
    data = pairs.loc[pairs[group_column].notna()].copy()
    values = data["mae_improvement"].to_numpy(dtype=float)
    labels_text = data[group_column].astype(str)
    categories = sorted(labels_text.unique())
    category_index = {label: index for index, label in enumerate(categories)}
    labels = labels_text.map(category_index).to_numpy(dtype=int)
    origin_text = data["origin_id"].astype(str)
    origin_rows = [
        np.flatnonzero(origin_text.eq(origin).to_numpy())
        for origin in sorted(origin_text.unique())
    ]
    counts = np.bincount(labels, minlength=len(categories))
    sums = np.bincount(labels, weights=values, minlength=len(categories))
    means = sums / counts
    overall = float(np.mean(values))
    observed = float(np.sum(counts * np.square(means - overall)))
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(PERMUTATION_REPETITIONS):
        permuted = labels.copy()
        for rows in origin_rows:
            permuted[rows] = rng.permutation(permuted[rows])
        permuted_means = (
            np.bincount(permuted, weights=values, minlength=len(categories)) / counts
        )
        statistic = float(np.sum(counts * np.square(permuted_means - overall)))
        extreme += statistic >= observed - 1e-15
    best_index = int(np.argmax(means))
    worst_index = int(np.argmin(means))
    return {
        "pooled_n": len(data),
        "subgroup_count": len(categories),
        "heterogeneity_statistic": observed,
        "permutation_repetitions": PERMUTATION_REPETITIONS,
        "heterogeneity_p_value": (extreme + 1) / (PERMUTATION_REPETITIONS + 1),
        "best_subgroup": categories[best_index],
        "best_subgroup_mae_improvement": float(means[best_index]),
        "worst_subgroup": categories[worst_index],
        "worst_subgroup_mae_improvement": float(means[worst_index]),
        "subgroup_improvement_range": float(means[best_index] - means[worst_index]),
    }


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(PREDICTIONS_PATH, low_memory=False)
    policy_predictions = pd.read_csv(POLICY_PREDICTIONS_PATH, low_memory=False)
    rolling_sensitivity = pd.read_csv(ROLLING_SENSITIVITY_PATH, low_memory=False)
    competition_crosswalk = pd.read_csv(COMPETITION_CROSSWALK_PATH, low_memory=False)
    upstream = json.loads(UPSTREAM_VERIFICATION_PATH.read_text(encoding="utf-8"))
    if upstream.get("status") != "pass":
        raise RuntimeError("Rolling-origin stability verification is not passing.")

    competition_names = (
        competition_crosswalk.drop_duplicates("canonical_competition_id")
        .set_index("canonical_competition_id")["canonical_name"]
        .to_dict()
    )
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
        "upstream_rolling_verification_passes",
        upstream.get("status") == "pass",
        upstream.get("status"),
        "pass",
        "Subgroup tests start from independently verified rolling predictions.",
    )

    paired_frames = []
    candidate_definition_rows = []
    for target_key, definition in CANDIDATES.items():
        candidate = predictions.loc[
            predictions["target_key"].eq(target_key)
            & predictions["feature_set"].eq(definition["feature_set"])
            & predictions["model_family"].eq(definition["model_family"])
        ].copy()
        same_family_baseline = predictions.loc[
            predictions["target_key"].eq(target_key)
            & predictions["feature_set"].eq("player_history_baseline")
            & predictions["model_family"].eq(definition["model_family"])
        ].copy()
        selected_baseline = policy_predictions.loc[
            policy_predictions["target_key"].eq(target_key)
            & policy_predictions["selection_type"].eq("rolling_selected_baseline")
        ].copy()
        candidate_definition_rows.append(
            {
                "target_key": target_key,
                "target_label": definition["target_label"],
                "candidate_feature_set": definition["feature_set"],
                "candidate_model_family": definition["model_family"],
                "same_family_baseline_feature_set": "player_history_baseline",
                "same_family_baseline_model_family": definition["model_family"],
                "secondary_reference": "rolling_selected_baseline",
                "subgroup_retuning_permitted": False,
                "candidate_scope": "frozen from rolling-origin phase",
            }
        )
        paired_frames.append(
            pair_frames(candidate, same_family_baseline, "same_family_baseline")
        )
        paired_frames.append(
            pair_frames(candidate, selected_baseline, "rolling_selected_baseline")
        )

    pairs = pd.concat(paired_frames, ignore_index=True)
    pairs["role_name"] = pairs["role_group"].map(ROLE_NAMES)
    pairs["competition_name"] = pairs["canonical_competition_id"].map(competition_names)
    pairs = pairs.sort_values(
        ["target_key", "reference_type", "origin_id", "canonical_performance_id"]
    ).reset_index(drop=True)
    candidate_definitions = pd.DataFrame(candidate_definition_rows)

    coverage_rows = []
    for target_key in CANDIDATES:
        target_pairs = pairs.loc[
            pairs["target_key"].eq(target_key)
            & pairs["reference_type"].eq("same_family_baseline")
        ]
        for dimension, (value_column, name_column) in DIMENSIONS.items():
            for subgroup_value, group in target_pairs.groupby(value_column, sort=True):
                origin_counts = group.groupby("origin_id").size()
                coverage_rows.append(
                    {
                        "target_key": target_key,
                        "subgroup_dimension": dimension,
                        "subgroup_value": subgroup_value,
                        "subgroup_name": group[name_column].iloc[0],
                        "all_origin_n": len(group),
                        "mature_origin_n": int(
                            group["origin_id"].isin(SUBSETS["mature_three_origins"]).sum()
                        ),
                        "terminal_origin_n": int(
                            group["origin_id"].isin(SUBSETS["terminal_two_origins"]).sum()
                        ),
                        "origins_present": int(origin_counts.size),
                        "minimum_origin_n": int(origin_counts.min()),
                        "maximum_origin_n": int(origin_counts.max()),
                    }
                )
    coverage = pd.DataFrame(coverage_rows)

    origin_result_rows = []
    for (target_key, reference_type), target_reference_pairs in pairs.groupby(
        ["target_key", "reference_type"], sort=True
    ):
        for dimension, (value_column, name_column) in DIMENSIONS.items():
            for (subgroup_value, rolling_origin), group in target_reference_pairs.groupby(
                [value_column, "origin_id"], sort=True
            ):
                result = paired_bootstrap(
                    group["actual"].to_numpy(dtype=float),
                    group["baseline_prediction"].to_numpy(dtype=float),
                    group["candidate_prediction"].to_numpy(dtype=float),
                    stable_seed(
                        RANDOM_SEED,
                        "origin",
                        target_key,
                        reference_type,
                        dimension,
                        subgroup_value,
                        rolling_origin,
                    ),
                )
                origin_result_rows.append(
                    {
                        "target_key": target_key,
                        "reference_type": reference_type,
                        "subgroup_dimension": dimension,
                        "subgroup_value": subgroup_value,
                        "subgroup_name": group[name_column].iloc[0],
                        "origin_id": rolling_origin,
                        "evaluation_year": int(group["evaluation_year"].iloc[0]),
                        "sample_quality": "adequate"
                        if len(group) >= MIN_ORIGIN_N
                        else "very_small",
                        **result,
                    }
                )
    origin_results = pd.DataFrame(origin_result_rows)

    aggregate_rows = []
    for (target_key, reference_type), target_reference_pairs in pairs.groupby(
        ["target_key", "reference_type"], sort=True
    ):
        for subset_name, included_origins in SUBSETS.items():
            subset = target_reference_pairs.loc[
                target_reference_pairs["origin_id"].isin(included_origins)
            ]
            for dimension, (value_column, name_column) in DIMENSIONS.items():
                for subgroup_value, group in subset.groupby(value_column, sort=True):
                    origin_sizes = group.groupby("origin_id").size()
                    result = stratified_bootstrap(
                        group,
                        stable_seed(
                            RANDOM_SEED,
                            "aggregate",
                            target_key,
                            reference_type,
                            subset_name,
                            dimension,
                            subgroup_value,
                        ),
                    )
                    origin_comparisons = origin_results.loc[
                        origin_results["target_key"].eq(target_key)
                        & origin_results["reference_type"].eq(reference_type)
                        & origin_results["subgroup_dimension"].eq(dimension)
                        & origin_results["subgroup_value"].eq(subgroup_value)
                        & origin_results["origin_id"].isin(included_origins)
                    ]
                    adequate = (
                        len(group) >= MIN_POOLED_N
                        and int(origin_sizes.min()) >= MIN_ORIGIN_N
                        and origin_sizes.size == len(included_origins)
                    )
                    aggregate_rows.append(
                        {
                            "target_key": target_key,
                            "candidate_feature_set": CANDIDATES[target_key]["feature_set"],
                            "candidate_model_family": CANDIDATES[target_key]["model_family"],
                            "reference_type": reference_type,
                            "reference_description": REFERENCES[reference_type],
                            "origin_subset": subset_name,
                            "included_origins": ";".join(included_origins),
                            "subgroup_dimension": dimension,
                            "subgroup_value": subgroup_value,
                            "subgroup_name": group[name_column].iloc[0],
                            "origin_count": int(origin_sizes.size),
                            "minimum_origin_n": int(origin_sizes.min()),
                            "maximum_origin_n": int(origin_sizes.max()),
                            "sample_quality": "adequate" if adequate else "small_cell",
                            "origins_candidate_better": int(
                                (origin_comparisons["mae_improvement"] > 0).sum()
                            ),
                            "origin_win_rate": float(
                                (origin_comparisons["mae_improvement"] > 0).mean()
                            ),
                            **result,
                        }
                    )
    aggregate_results = pd.DataFrame(aggregate_rows)
    aggregate_results["fdr_family"] = None
    aggregate_results["fdr_q_value"] = np.nan
    primary_mask = (
        aggregate_results["reference_type"].eq("same_family_baseline")
        & aggregate_results["origin_subset"].eq("mature_three_origins")
        & aggregate_results["sample_quality"].eq("adequate")
    )
    aggregate_results.loc[primary_mask, "fdr_family"] = (
        "mature_same_family_adequate_role_and_competition_cells"
    )
    aggregate_results.loc[primary_mask, "fdr_q_value"] = benjamini_hochberg(
        aggregate_results.loc[primary_mask, "bootstrap_p_two_sided"]
    )
    aggregate_results["evidence_label"] = aggregate_results.apply(evidence_label, axis=1)
    aggregate_results = aggregate_results.sort_values(
        [
            "origin_subset",
            "reference_type",
            "target_key",
            "subgroup_dimension",
            "subgroup_value",
        ]
    ).reset_index(drop=True)
    primary_results = aggregate_results.loc[
        aggregate_results["reference_type"].eq("same_family_baseline")
        & aggregate_results["origin_subset"].eq("mature_three_origins")
    ].copy()

    heterogeneity_rows = []
    for (target_key, reference_type), target_reference_pairs in pairs.groupby(
        ["target_key", "reference_type"], sort=True
    ):
        for subset_name, included_origins in SUBSETS.items():
            subset = target_reference_pairs.loc[
                target_reference_pairs["origin_id"].isin(included_origins)
            ]
            for dimension, (value_column, name_column) in DIMENSIONS.items():
                result = heterogeneity_test(
                    subset,
                    value_column,
                    stable_seed(
                        RANDOM_SEED,
                        "heterogeneity",
                        target_key,
                        reference_type,
                        subset_name,
                        dimension,
                    ),
                )
                result["best_subgroup_name"] = subset.loc[
                    subset[value_column].astype(str).eq(str(result["best_subgroup"])),
                    name_column,
                ].iloc[0]
                result["worst_subgroup_name"] = subset.loc[
                    subset[value_column].astype(str).eq(str(result["worst_subgroup"])),
                    name_column,
                ].iloc[0]
                heterogeneity_rows.append(
                    {
                        "target_key": target_key,
                        "reference_type": reference_type,
                        "origin_subset": subset_name,
                        "subgroup_dimension": dimension,
                        **result,
                    }
                )
    heterogeneity = pd.DataFrame(heterogeneity_rows)
    heterogeneity["fdr_family"] = None
    heterogeneity["heterogeneity_fdr_q_value"] = np.nan
    heterogeneity_primary = (
        heterogeneity["reference_type"].eq("same_family_baseline")
        & heterogeneity["origin_subset"].eq("mature_three_origins")
    )
    heterogeneity.loc[heterogeneity_primary, "fdr_family"] = (
        "mature_same_family_role_and_competition_heterogeneity"
    )
    heterogeneity.loc[
        heterogeneity_primary, "heterogeneity_fdr_q_value"
    ] = benjamini_hochberg(
        heterogeneity.loc[heterogeneity_primary, "heterogeneity_p_value"]
    )
    heterogeneity["heterogeneity_evidence"] = np.where(
        heterogeneity["heterogeneity_fdr_q_value"].le(FDR_ALPHA),
        "fdr_supported_heterogeneity",
        np.where(
            heterogeneity["heterogeneity_p_value"].le(0.05),
            "nominal_heterogeneity",
            "no_detected_heterogeneity",
        ),
    )

    decision_rows = []
    for target_key, definition in CANDIDATES.items():
        primary = primary_results.loc[primary_results["target_key"].eq(target_key)]
        adequate = primary.loc[primary["sample_quality"].eq("adequate")]
        role_rows = adequate.loc[adequate["subgroup_dimension"].eq("role")]
        competition_rows = adequate.loc[
            adequate["subgroup_dimension"].eq("competition")
        ]
        strongest = adequate.sort_values(
            "pooled_relative_mae_improvement", ascending=False
        ).iloc[0]
        weakest = adequate.sort_values("pooled_relative_mae_improvement").iloc[0]
        role_heterogeneity = heterogeneity.loc[
            heterogeneity["target_key"].eq(target_key)
            & heterogeneity["reference_type"].eq("same_family_baseline")
            & heterogeneity["origin_subset"].eq("mature_three_origins")
            & heterogeneity["subgroup_dimension"].eq("role")
        ].iloc[0]
        competition_heterogeneity = heterogeneity.loc[
            heterogeneity["target_key"].eq(target_key)
            & heterogeneity["reference_type"].eq("same_family_baseline")
            & heterogeneity["origin_subset"].eq("mature_three_origins")
            & heterogeneity["subgroup_dimension"].eq("competition")
        ].iloc[0]
        supported_positive = adequate["evidence_label"].isin(
            ["fdr_supported_positive", "nominal_supported_positive"]
        )
        supported_negative = adequate["evidence_label"].isin(
            ["fdr_supported_negative", "nominal_supported_negative"]
        )
        if supported_negative.any():
            recommendation = "retain_with_subgroup_guardrails"
        elif supported_positive.sum() >= 2:
            recommendation = "advance_frozen_candidate_to_case_studies"
        else:
            recommendation = "continue_research_without_operational_subgroup_claims"
        decision_rows.append(
            {
                "target_key": target_key,
                "candidate_feature_set": definition["feature_set"],
                "candidate_model_family": definition["model_family"],
                "adequate_primary_subgroups": len(adequate),
                "adequate_roles": len(role_rows),
                "adequate_competitions": len(competition_rows),
                "positive_primary_subgroups": int(
                    (adequate["pooled_mae_improvement"] > 0).sum()
                ),
                "negative_primary_subgroups": int(
                    (adequate["pooled_mae_improvement"] < 0).sum()
                ),
                "supported_positive_subgroups": int(supported_positive.sum()),
                "supported_negative_subgroups": int(supported_negative.sum()),
                "strongest_subgroup_dimension": strongest["subgroup_dimension"],
                "strongest_subgroup_name": strongest["subgroup_name"],
                "strongest_relative_mae_improvement": strongest[
                    "pooled_relative_mae_improvement"
                ],
                "weakest_subgroup_dimension": weakest["subgroup_dimension"],
                "weakest_subgroup_name": weakest["subgroup_name"],
                "weakest_relative_mae_improvement": weakest[
                    "pooled_relative_mae_improvement"
                ],
                "role_heterogeneity_p_value": role_heterogeneity[
                    "heterogeneity_p_value"
                ],
                "role_heterogeneity_fdr_q_value": role_heterogeneity[
                    "heterogeneity_fdr_q_value"
                ],
                "competition_heterogeneity_p_value": competition_heterogeneity[
                    "heterogeneity_p_value"
                ],
                "competition_heterogeneity_fdr_q_value": competition_heterogeneity[
                    "heterogeneity_fdr_q_value"
                ],
                "recommendation": recommendation,
            }
        )
    decision_summary = pd.DataFrame(decision_rows)

    add_check(
        "candidate_definitions_are_frozen",
        len(candidate_definitions) == 2
        and not candidate_definitions["subgroup_retuning_permitted"].any()
        and set(candidate_definitions["candidate_model_family"])
        == {"gradient_boosted_trees"},
        candidate_definitions[
            ["target_key", "candidate_feature_set", "candidate_model_family"]
        ].to_dict("records"),
        "opportunity=gradient_boosted_trees/full;performance=gradient_boosted_trees/full",
        "Subgroup outcomes never select a feature set, family, or hyperparameter.",
    )
    expected_pair_rows = 2 * (207 + 172)
    add_check(
        "paired_prediction_rows_complete",
        len(pairs) == expected_pair_rows,
        len(pairs),
        expected_pair_rows,
        "Each frozen candidate is paired with both references on every rolling evaluation row.",
    )
    add_check(
        "paired_values_finite",
        np.isfinite(
            pairs[
                [
                    "actual",
                    "candidate_prediction",
                    "baseline_prediction",
                    "candidate_absolute_error",
                    "baseline_absolute_error",
                    "mae_improvement",
                ]
            ].to_numpy(dtype=float)
        ).all(),
        int(
            (
                ~np.isfinite(
                    pairs[
                        [
                            "actual",
                            "candidate_prediction",
                            "baseline_prediction",
                            "candidate_absolute_error",
                            "baseline_absolute_error",
                            "mae_improvement",
                        ]
                    ].to_numpy(dtype=float)
                )
            ).sum()
        ),
        0,
        "All subgroup pair values are finite.",
    )
    add_check(
        "subgroup_fields_complete",
        pairs[
            ["role_group", "role_name", "canonical_competition_id", "competition_name"]
        ]
        .notna()
        .all()
        .all(),
        int(
            pairs[
                [
                    "role_group",
                    "role_name",
                    "canonical_competition_id",
                    "competition_name",
                ]
            ]
            .isna()
            .sum()
            .sum()
        ),
        0,
        "Every paired row has a mapped role and competition.",
    )
    add_check(
        "expected_roles_and_competitions_only",
        set(pairs["role_group"]) == {"DF", "FW", "MF"}
        and set(pairs["canonical_competition_id"])
        == {"ES1", "FR1", "GB1", "IT1", "L1"},
        f"roles={sorted(pairs['role_group'].unique())};competitions={sorted(pairs['canonical_competition_id'].unique())}",
        "roles=DF,FW,MF;competitions=ES1,FR1,GB1,IT1,L1",
        "Subgroup scope is the three outfield roles and five modeled Big Five leagues.",
    )
    add_check(
        "coverage_grid_complete",
        len(coverage) == len(CANDIDATES) * (3 + 5),
        len(coverage),
        len(CANDIDATES) * (3 + 5),
        "Every target has one coverage row for each role and competition.",
    )
    add_check(
        "origin_result_grid_complete",
        len(origin_results) == len(CANDIDATES) * len(REFERENCES) * (3 + 5) * 4,
        len(origin_results),
        len(CANDIDATES) * len(REFERENCES) * (3 + 5) * 4,
        "Every target, reference, subgroup, and origin has a paired result.",
    )
    add_check(
        "aggregate_result_grid_complete",
        len(aggregate_results)
        == len(CANDIDATES) * len(REFERENCES) * (3 + 5) * len(SUBSETS),
        len(aggregate_results),
        len(CANDIDATES) * len(REFERENCES) * (3 + 5) * len(SUBSETS),
        "All, mature, and terminal summaries cover every declared subgroup comparison.",
    )
    add_check(
        "primary_result_grid_complete",
        len(primary_results) == len(CANDIDATES) * (3 + 5),
        len(primary_results),
        len(CANDIDATES) * (3 + 5),
        "Primary reporting contains all mature same-family role and competition cells.",
    )
    add_check(
        "heterogeneity_grid_complete",
        len(heterogeneity)
        == len(CANDIDATES) * len(REFERENCES) * len(DIMENSIONS) * len(SUBSETS),
        len(heterogeneity),
        len(CANDIDATES) * len(REFERENCES) * len(DIMENSIONS) * len(SUBSETS),
        "Role and competition heterogeneity is tested for every target, reference, and origin subset.",
    )
    confirmatory_subgroup_mask = (
        aggregate_results["reference_type"].eq("same_family_baseline")
        & aggregate_results["origin_subset"].eq("mature_three_origins")
        & aggregate_results["sample_quality"].eq("adequate")
    )
    subgroup_fdr = aggregate_results.loc[
        confirmatory_subgroup_mask, "fdr_q_value"
    ]
    heterogeneity_fdr = heterogeneity.loc[
        heterogeneity_primary, "heterogeneity_fdr_q_value"
    ]
    fdr_values_valid = bool(
        subgroup_fdr.notna().all()
        and subgroup_fdr.between(0, 1).all()
        and heterogeneity_fdr.notna().all()
        and heterogeneity_fdr.between(0, 1).all()
    )
    add_check(
        "fdr_values_valid",
        fdr_values_valid,
        json.dumps(
            {
                "subgroup_count": int(len(subgroup_fdr)),
                "subgroup_missing": int(subgroup_fdr.isna().sum()),
                "subgroup_min": (
                    float(subgroup_fdr.min()) if subgroup_fdr.notna().any() else None
                ),
                "subgroup_max": (
                    float(subgroup_fdr.max()) if subgroup_fdr.notna().any() else None
                ),
                "heterogeneity_count": int(len(heterogeneity_fdr)),
                "heterogeneity_missing": int(heterogeneity_fdr.isna().sum()),
                "heterogeneity_min": (
                    float(heterogeneity_fdr.min())
                    if heterogeneity_fdr.notna().any()
                    else None
                ),
                "heterogeneity_max": (
                    float(heterogeneity_fdr.max())
                    if heterogeneity_fdr.notna().any()
                    else None
                ),
            },
            sort_keys=True,
        ),
        "all confirmatory q-values present and within [0,1]",
        "Confirmatory subgroup and heterogeneity q-values are bounded.",
    )
    candidate_reference_consistency = (
        pairs.groupby(["target_key", "origin_id", "canonical_performance_id"])[
            "candidate_prediction"
        ].nunique()
        <= 1
    ).all()
    add_check(
        "candidate_predictions_identical_across_references",
        candidate_reference_consistency,
        candidate_reference_consistency,
        True,
        "Changing the benchmark never changes the frozen candidate prediction.",
    )
    partition_failures = []
    for target_key in CANDIDATES:
        for reference_type in REFERENCES:
            target_reference = pairs.loc[
                pairs["target_key"].eq(target_key)
                & pairs["reference_type"].eq(reference_type)
            ]
            for subset_name, included_origins in SUBSETS.items():
                subset = target_reference.loc[
                    target_reference["origin_id"].isin(included_origins)
                ]
                overall = float(subset["mae_improvement"].mean())
                for dimension, (value_column, _) in DIMENSIONS.items():
                    statistics = subset.groupby(value_column)["mae_improvement"].agg(
                        ["mean", "size"]
                    )
                    weighted = float(
                        np.average(statistics["mean"], weights=statistics["size"])
                    )
                    if not math.isclose(overall, weighted, abs_tol=1e-12):
                        partition_failures.append(
                            f"{target_key}:{reference_type}:{subset_name}:{dimension}"
                        )
    add_check(
        "subgroup_partitions_reconcile_to_overall",
        not partition_failures,
        ";".join(partition_failures),
        "none",
        "Role and competition weighted improvements each reconcile to the full paired cohort.",
    )
    continuity_failures = []
    for target_key, definition in CANDIDATES.items():
        pairs_mature = pairs.loc[
            pairs["target_key"].eq(target_key)
            & pairs["reference_type"].eq("same_family_baseline")
            & pairs["origin_id"].isin(SUBSETS["mature_three_origins"])
        ]
        improvement = float(pairs_mature["mae_improvement"].mean())
        prior = rolling_sensitivity.loc[
            rolling_sensitivity["analysis_type"].eq(
                "within_family_full_vs_baseline"
            )
            & rolling_sensitivity["target_key"].eq(target_key)
            & rolling_sensitivity["model_family"].eq(definition["model_family"])
            & rolling_sensitivity["origin_subset"].eq("mature_three_origins")
        ].iloc[0]
        if not math.isclose(
            improvement, float(prior["pooled_mae_improvement"]), abs_tol=1e-12
        ):
            continuity_failures.append(target_key)
    add_check(
        "mature_overall_effects_reconcile_to_rolling_phase",
        not continuity_failures,
        ";".join(continuity_failures),
        "none",
        "Frozen candidate mature effects exactly match the prior rolling sensitivity analysis.",
    )

    checks_frame = pd.DataFrame(checks)
    if not checks_frame["passed"].all():
        failed_checks = checks_frame.loc[~checks_frame["passed"]]
        raise RuntimeError(
            "Blocking subgroup checks failed: "
            + failed_checks[["check", "observed", "expected"]].to_json(
                orient="records"
            )
        )

    outputs = {
        "subgroup_candidate_definitions.csv": candidate_definitions,
        "subgroup_pair_predictions.csv": pairs,
        "subgroup_coverage.csv": coverage,
        "subgroup_origin_results.csv": origin_results,
        "subgroup_aggregate_results.csv": aggregate_results,
        "subgroup_primary_results.csv": primary_results,
        "subgroup_heterogeneity.csv": heterogeneity,
        "subgroup_decision_summary.csv": decision_summary,
        "subgroup_build_checks.csv": checks_frame,
    }
    for filename, frame in outputs.items():
        frame.to_csv(OUTPUT_DIR / filename, index=False)

    run_summary = {
        "status": "pass",
        "targets": list(CANDIDATES),
        "references": list(REFERENCES),
        "subgroup_dimensions": list(DIMENSIONS),
        "origin_subsets": list(SUBSETS),
        "paired_prediction_rows": len(pairs),
        "origin_result_rows": len(origin_results),
        "aggregate_result_rows": len(aggregate_results),
        "primary_result_rows": len(primary_results),
        "heterogeneity_rows": len(heterogeneity),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "permutation_repetitions": PERMUTATION_REPETITIONS,
        "minimum_pooled_n": MIN_POOLED_N,
        "minimum_origin_n": MIN_ORIGIN_N,
        "fdr_alpha": FDR_ALPHA,
        "blocking_checks": len(checks_frame),
        "blocking_failures": int((~checks_frame["passed"]).sum()),
        "random_seed": RANDOM_SEED,
    }
    (OUTPUT_DIR / "subgroup_run_summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )
    source_paths = [
        PREDICTIONS_PATH,
        POLICY_PREDICTIONS_PATH,
        ROLLING_SENSITIVITY_PATH,
        UPSTREAM_VERIFICATION_PATH,
        COMPETITION_CROSSWALK_PATH,
    ]
    pd.DataFrame(
        [
            {
                "source_file": str(source.relative_to(ROOT)),
                "sha256": sha256_file(source),
                "bytes": source.stat().st_size,
            }
            for source in source_paths
        ]
    ).to_csv(OUTPUT_DIR / "source_manifest.csv", index=False)

    readme = """# MoveMaker frozen-candidate subgroup stability

This phase uses the already-scored rolling-origin predictions and does not retrain from subgroup outcomes. After correcting the shared encoder and rerunning the 2022 validation selection, both opportunity and performance are frozen as full compatibility plus gradient-boosted trees.

The primary comparison is the candidate versus the same model family using only player-history baseline features. This isolates compatibility information from learner choice. A secondary comparison uses the best baseline family selected on each origin validation season.

Primary subgroup reporting uses the mature three origins. A cell is marked adequate when pooled n is at least 15, every included origin is present, and the minimum origin cell is at least 5. All subgroup cells remain in the outputs, but small cells are explicitly labeled and excluded from confirmatory FDR adjustment.

Paired uncertainty uses 5,000 bootstrap resamples stratified by origin. The primary mature, same-family, adequately sized role and competition cells share one Benjamini-Hochberg FDR family at q=10%. Heterogeneity is tested by permuting subgroup labels within each origin 5,000 times, with separate FDR adjustment across the four primary target-dimension tests.

Use subgroup_primary_results.csv for the main mature role and competition readout, subgroup_origin_results.csv for season-level behavior, subgroup_heterogeneity.csv for evidence that effects differ between roles or leagues, and subgroup_pair_predictions.csv for the auditable player-level paired errors.
"""
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    output_paths = sorted(
        path
        for path in OUTPUT_DIR.iterdir()
        if path.is_file() and path.name != "output_manifest.csv"
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
    ).to_csv(OUTPUT_DIR / "output_manifest.csv", index=False)
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    run()

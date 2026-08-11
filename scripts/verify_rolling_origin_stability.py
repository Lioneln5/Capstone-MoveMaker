"""Independently verify MoveMaker rolling-origin stability outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_compatibility_models import bool_series, metrics, paired_bootstrap, sha256_file


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "Data" / "processed" / "rolling_origin_stability"
TARGET_DIR = ROOT / "Data" / "processed" / "compatibility_targets"
ABLATION_DIR = ROOT / "Data" / "processed" / "compatibility_model_ablations"
MATRIX_PATH = TARGET_DIR / "transfer_compatibility_strict_model_matrix.csv"
PRIOR_SELECTION_PATH = ABLATION_DIR / "ablation_model_selection.csv"
PRIOR_RESULTS_PATH = ABLATION_DIR / "ablation_test_results.csv"

RANDOM_SEED = 20260809
BOOTSTRAP_REPETITIONS = 5000
ORIGINS = {
    "origin_2019": (2018, 2018, 2019, 2020),
    "origin_2020": (2018, 2019, 2020, 2021),
    "origin_2021": (2018, 2020, 2021, 2022),
    "origin_2022": (2018, 2021, 2022, 2023),
}
TARGETS = {
    "opportunity": ("target_destination_minutes_share", "eligible_opportunity_model_primary"),
    "performance": ("target_role_performance_index", "eligible_performance_model_primary"),
    "adaptation": ("target_performance_change_rolling3", "eligible_adaptation_model_primary"),
}
FEATURE_SETS = [
    "player_history_baseline",
    "plus_club_context",
    "plus_player_preferences",
    "context_and_preferences",
    "full_explicit_fit",
]
FEATURE_SET_COUNTS = {
    "player_history_baseline": 81,
    "plus_club_context": 133,
    "plus_player_preferences": 121,
    "context_and_preferences": 173,
    "full_explicit_fit": 202,
}
MODEL_FAMILIES = ["ridge", "elastic_net", "gradient_boosted_trees"]
POLICIES = [
    "rolling_selected_baseline",
    "rolling_selected_full",
    "rolling_selected_overall",
]
LOCKED_TARGETS = ["opportunity", "performance"]


def close(left: Any, right: Any, tolerance: float = 1e-10) -> bool:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return left == right
    if math.isnan(left_value) and math.isnan(right_value):
        return True
    return abs(left_value - right_value) <= tolerance


def stratified_bootstrap(
    pairs: pd.DataFrame,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    total_n = len(pairs)
    mae_sum = np.zeros(BOOTSTRAP_REPETITIONS, dtype=float)
    baseline_sq_sum = np.zeros(BOOTSTRAP_REPETITIONS, dtype=float)
    candidate_sq_sum = np.zeros(BOOTSTRAP_REPETITIONS, dtype=float)
    for _, group in pairs.groupby("origin_id", sort=True):
        actual = group["actual"].to_numpy(dtype=float)
        baseline = group["baseline_prediction"].to_numpy(dtype=float)
        candidate = group["candidate_prediction"].to_numpy(dtype=float)
        n = len(group)
        indices = rng.integers(0, n, size=(BOOTSTRAP_REPETITIONS, n))
        mae_sum += (
            np.abs(actual - baseline) - np.abs(actual - candidate)
        )[indices].sum(axis=1)
        baseline_sq_sum += np.square(actual - baseline)[indices].sum(axis=1)
        candidate_sq_sum += np.square(actual - candidate)[indices].sum(axis=1)
    actual = pairs["actual"].to_numpy(dtype=float)
    baseline = pairs["baseline_prediction"].to_numpy(dtype=float)
    candidate = pairs["candidate_prediction"].to_numpy(dtype=float)
    baseline_abs = np.abs(actual - baseline)
    candidate_abs = np.abs(actual - candidate)
    mae_draws = mae_sum / total_n
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


def pair_predictions(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["origin_id", "target_key", "canonical_performance_id"]
    left = baseline[keys + ["actual", "prediction"]].rename(
        columns={"prediction": "baseline_prediction"}
    )
    right = candidate[keys + ["actual", "prediction"]].rename(
        columns={"actual": "candidate_actual", "prediction": "candidate_prediction"}
    )
    merged = left.merge(right, on=keys, validate="one_to_one")
    if not np.allclose(merged["actual"], merged["candidate_actual"], atol=1e-12, rtol=0):
        raise RuntimeError("Paired actual values do not align.")
    return merged.drop(columns="candidate_actual")


def verify_summary(
    reported: pd.Series,
    pairs: pd.DataFrame,
    seed: int,
    origin_bootstrap: pd.DataFrame,
) -> list[str]:
    failures = []
    recalculated = stratified_bootstrap(pairs, seed)
    improvements = origin_bootstrap["mae_improvement"].to_numpy(dtype=float)
    extra = {
        "origin_count": len(origin_bootstrap),
        "origins_candidate_better": int(np.sum(improvements > 0)),
        "origin_win_rate": float(np.mean(improvements > 0)),
        "mean_origin_mae_improvement": float(np.mean(improvements)),
        "median_origin_mae_improvement": float(np.median(improvements)),
        "origin_improvement_std": float(np.std(improvements, ddof=1)) if len(improvements) > 1 else 0.0,
        "worst_origin_mae_improvement": float(np.min(improvements)),
        "best_origin_mae_improvement": float(np.max(improvements)),
    }
    for name, value in {**recalculated, **extra}.items():
        if not close(value, reported[name]):
            failures.append(name)
    return failures


def run() -> None:
    required = [
        "rolling_origin_definitions.csv",
        "rolling_origin_cohorts.csv",
        "rolling_feature_sets.csv",
        "rolling_validation_tuning.csv",
        "rolling_selected_specs.csv",
        "rolling_predictions.csv",
        "rolling_origin_feature_effects.csv",
        "rolling_family_bootstrap.csv",
        "rolling_family_stability_summary.csv",
        "rolling_policy_selections.csv",
        "rolling_policy_predictions.csv",
        "rolling_policy_bootstrap.csv",
        "rolling_policy_stability_summary.csv",
        "rolling_policy_comparison_stability.csv",
        "rolling_selection_frequency.csv",
        "rolling_locked_designs.csv",
        "rolling_locked_metrics.csv",
        "rolling_locked_predictions.csv",
        "rolling_locked_bootstrap.csv",
        "rolling_locked_stability_summary.csv",
        "rolling_sensitivity_summary.csv",
        "rolling_build_checks.csv",
        "rolling_run_summary.json",
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

    missing = [name for name in required if not (OUTPUT_DIR / name).exists()]
    add(
        "required_outputs_exist",
        not missing,
        ";".join(missing) if missing else "all present",
        "all present",
        "Every declared rolling-origin artifact exists.",
    )
    if missing:
        raise RuntimeError(f"Missing rolling-origin outputs: {missing}")

    matrix = pd.read_csv(MATRIX_PATH, low_memory=False)
    prior_selection = pd.read_csv(PRIOR_SELECTION_PATH, low_memory=False)
    prior_results = pd.read_csv(PRIOR_RESULTS_PATH, low_memory=False)
    origins = pd.read_csv(OUTPUT_DIR / "rolling_origin_definitions.csv")
    cohorts = pd.read_csv(OUTPUT_DIR / "rolling_origin_cohorts.csv")
    feature_sets = pd.read_csv(OUTPUT_DIR / "rolling_feature_sets.csv", low_memory=False)
    tuning = pd.read_csv(OUTPUT_DIR / "rolling_validation_tuning.csv", low_memory=False)
    selected = pd.read_csv(OUTPUT_DIR / "rolling_selected_specs.csv", low_memory=False)
    predictions = pd.read_csv(OUTPUT_DIR / "rolling_predictions.csv", low_memory=False)
    effects = pd.read_csv(OUTPUT_DIR / "rolling_origin_feature_effects.csv", low_memory=False)
    family_bootstrap = pd.read_csv(OUTPUT_DIR / "rolling_family_bootstrap.csv")
    family_stability = pd.read_csv(OUTPUT_DIR / "rolling_family_stability_summary.csv")
    policy_selections = pd.read_csv(OUTPUT_DIR / "rolling_policy_selections.csv", low_memory=False)
    policy_predictions = pd.read_csv(OUTPUT_DIR / "rolling_policy_predictions.csv", low_memory=False)
    policy_bootstrap = pd.read_csv(OUTPUT_DIR / "rolling_policy_bootstrap.csv")
    policy_stability = pd.read_csv(OUTPUT_DIR / "rolling_policy_stability_summary.csv")
    policy_comparison = pd.read_csv(OUTPUT_DIR / "rolling_policy_comparison_stability.csv")
    selection_frequency = pd.read_csv(OUTPUT_DIR / "rolling_selection_frequency.csv")
    locked_designs = pd.read_csv(OUTPUT_DIR / "rolling_locked_designs.csv")
    locked_metrics = pd.read_csv(OUTPUT_DIR / "rolling_locked_metrics.csv", low_memory=False)
    locked_predictions = pd.read_csv(OUTPUT_DIR / "rolling_locked_predictions.csv", low_memory=False)
    locked_bootstrap = pd.read_csv(OUTPUT_DIR / "rolling_locked_bootstrap.csv")
    locked_stability = pd.read_csv(OUTPUT_DIR / "rolling_locked_stability_summary.csv")
    sensitivity = pd.read_csv(OUTPUT_DIR / "rolling_sensitivity_summary.csv")
    build_checks = pd.read_csv(OUTPUT_DIR / "rolling_build_checks.csv")
    source_manifest = pd.read_csv(OUTPUT_DIR / "source_manifest.csv")

    add(
        "build_checks_all_pass",
        bool_series(build_checks["passed"]).all(),
        int((~bool_series(build_checks["passed"])).sum()),
        0,
        "All construction-time rolling invariants passed.",
    )
    source_failures = []
    for _, row in source_manifest.iterrows():
        path = ROOT / row["source_file"]
        if not path.exists() or sha256_file(path) != row["sha256"]:
            source_failures.append(row["source_file"])
    add(
        "source_manifest_hashes_match",
        not source_failures,
        ";".join(source_failures) if source_failures else "all match",
        "all match",
        "Verification uses the exact source artifacts recorded by the rolling run.",
    )

    reported_origins = {
        row.origin_id: (
            int(row.train_start_year),
            int(row.train_end_year),
            int(row.validation_year),
            int(row.evaluation_year),
        )
        for row in origins.itertuples(index=False)
    }
    add(
        "origin_definitions_exact",
        reported_origins == ORIGINS,
        json.dumps(reported_origins, sort_keys=True),
        json.dumps(ORIGINS, sort_keys=True),
        "The four expanding windows are exact and chronological.",
    )
    add(
        "no_origin_time_overlap",
        all(train_end < validation < evaluation for _, train_end, validation, evaluation in ORIGINS.values()),
        "strict chronology",
        "train_end < validation < evaluation",
        "No origin trains or validates on its evaluation season.",
    )

    feature_members = {
        name: set(feature_sets.loc[feature_sets["feature_set"].eq(name), "raw_feature"])
        for name in FEATURE_SETS
    }
    add(
        "feature_set_counts_exact",
        {name: len(values) for name, values in feature_members.items()} == FEATURE_SET_COUNTS,
        json.dumps({name: len(values) for name, values in feature_members.items()}, sort_keys=True),
        json.dumps(FEATURE_SET_COUNTS, sort_keys=True),
        "The rolling run uses the same five fixed feature designs as the prior phase.",
    )
    baseline_features = feature_members["player_history_baseline"]
    add(
        "baseline_nested_in_all_feature_sets",
        all(baseline_features.issubset(values) for values in feature_members.values()),
        "nested",
        "nested",
        "Every compatibility comparison retains the common player-history baseline.",
    )

    cohort_failures = []
    actual_failures = []
    key_failures = []
    metric_failures = []
    for target_key, (target_column, eligibility_column) in TARGETS.items():
        eligible = bool_series(matrix[eligibility_column]) & matrix[target_column].notna()
        source = matrix.loc[
            eligible, ["canonical_performance_id", "season_start_year", target_column]
        ].copy()
        source["key"] = source["canonical_performance_id"].astype(str)
        for rolling_origin, (_, train_end, validation_year, evaluation_year) in ORIGINS.items():
            expected_counts = {
                "train_rows": int(
                    source["season_start_year"].between(2018, train_end).sum()
                ),
                "validation_rows": int(source["season_start_year"].eq(validation_year).sum()),
                "evaluation_rows": int(source["season_start_year"].eq(evaluation_year).sum()),
            }
            reported = cohorts.loc[
                cohorts["target_key"].eq(target_key)
                & cohorts["origin_id"].eq(rolling_origin)
            ]
            if len(reported) != 1 or any(
                int(reported.iloc[0][column]) != value
                for column, value in expected_counts.items()
            ):
                cohort_failures.append(f"{target_key}:{rolling_origin}")

            expected_evaluation = source.loc[source["season_start_year"].eq(evaluation_year)].copy()
            expected_actual = expected_evaluation.set_index("key")[target_column]
            spec_rows = predictions.loc[
                predictions["target_key"].eq(target_key)
                & predictions["origin_id"].eq(rolling_origin)
            ]
            sets_by_spec = spec_rows.groupby(["feature_set", "model_family"])[
                "canonical_performance_id"
            ].apply(lambda values: frozenset(values.astype(str)))
            if len(sets_by_spec) != 15 or len(set(sets_by_spec)) != 1:
                key_failures.append(f"{target_key}:{rolling_origin}")
            for feature_set in FEATURE_SETS:
                for family in MODEL_FAMILIES:
                    rows = spec_rows.loc[
                        spec_rows["feature_set"].eq(feature_set)
                        & spec_rows["model_family"].eq(family)
                    ].copy()
                    rows["key"] = rows["canonical_performance_id"].astype(str)
                    if len(rows) != len(expected_evaluation) or not np.allclose(
                        rows["actual"].to_numpy(dtype=float),
                        rows["key"].map(expected_actual).to_numpy(dtype=float),
                        atol=1e-12,
                        rtol=0,
                    ):
                        actual_failures.append(
                            f"{target_key}:{rolling_origin}:{feature_set}:{family}"
                        )
                    recalculated = metrics(
                        rows["actual"].to_numpy(dtype=float),
                        rows["prediction"].to_numpy(dtype=float),
                    )
                    reported_metric = selected.loc[
                        selected["target_key"].eq(target_key)
                        & selected["origin_id"].eq(rolling_origin)
                        & selected["feature_set"].eq(feature_set)
                        & selected["model_family"].eq(family)
                    ]
                    if len(reported_metric) != 1:
                        metric_failures.append(
                            f"{target_key}:{rolling_origin}:{feature_set}:{family}:row"
                        )
                    else:
                        for metric_name, value in recalculated.items():
                            output_name = (
                                "evaluation_n"
                                if metric_name == "n"
                                else f"evaluation_{metric_name}"
                            )
                            if not close(value, reported_metric.iloc[0][output_name]):
                                metric_failures.append(
                                    f"{target_key}:{rolling_origin}:{feature_set}:{family}:{metric_name}"
                                )
    add(
        "origin_cohort_counts_recomputed",
        not cohort_failures,
        ";".join(cohort_failures),
        "none",
        "Every target-origin cohort count matches the source eligibility rules.",
    )
    add(
        "evaluation_actuals_match_source",
        not actual_failures,
        str(len(actual_failures)),
        0,
        "Every held-out actual value maps back to the target matrix.",
    )
    add(
        "evaluation_keys_match_all_specs",
        not key_failures,
        ";".join(key_failures),
        "none",
        "Every specification scores identical rows within target and origin.",
    )
    add(
        "selected_spec_metrics_recomputed",
        not metric_failures,
        str(len(metric_failures)),
        0,
        "All 180 reported evaluation metric rows recompute from predictions.",
    )

    selection_failures = []
    for keys, group in tuning.groupby(
        ["target_key", "origin_id", "feature_set", "model_family"], sort=True
    ):
        best = group.sort_values(["mae", "rmse", "hyperparameters"], kind="mergesort").iloc[0]
        reported = selected.loc[
            selected["target_key"].eq(keys[0])
            & selected["origin_id"].eq(keys[1])
            & selected["feature_set"].eq(keys[2])
            & selected["model_family"].eq(keys[3])
        ].iloc[0]
        if (
            best["hyperparameters"] != reported["selected_hyperparameters"]
            or not close(best["mae"], reported["validation_mae"])
            or not close(best["rmse"], reported["validation_rmse"])
        ):
            selection_failures.append(":".join(keys))
    add(
        "hyperparameter_selection_recomputed",
        not selection_failures,
        ";".join(selection_failures),
        "none",
        "Every selected hyperparameter set is the declared validation-only minimum.",
    )
    add(
        "candidate_and_spec_counts_exact",
        len(tuning) == 1680 and len(selected) == 180 and len(predictions) == 8115,
        f"tuning={len(tuning)};selected={len(selected)};predictions={len(predictions)}",
        "tuning=1680;selected=180;predictions=8115",
        "The full rolling model grid and evaluation predictions are complete.",
    )
    add(
        "opportunity_predictions_bounded",
        predictions.loc[predictions["target_key"].eq("opportunity"), "prediction"].between(0, 1).all(),
        "within bounds",
        "within [0,1]",
        "All rolling opportunity predictions respect their natural support.",
    )

    terminal = selected.loc[selected["origin_id"].eq("origin_2022")].merge(
        prior_results,
        on=["target_key", "feature_set", "model_family"],
        suffixes=("_rolling", "_prior"),
        validate="one_to_one",
    )
    terminal_match = (
        len(terminal) == 45
        and np.allclose(
            terminal["validation_mae_rolling"], terminal["validation_mae_prior"], atol=1e-10, rtol=0
        )
        and np.allclose(terminal["evaluation_mae"], terminal["test_mae"], atol=1e-10, rtol=0)
        and (
            terminal["selected_hyperparameters_rolling"]
            == terminal["selected_hyperparameters_prior"]
        ).all()
    )
    add(
        "terminal_origin_reproduces_prior_ablation",
        terminal_match,
        len(terminal),
        45,
        "The 2022-validation/2023-evaluation origin exactly reproduces the preceding phase.",
    )

    policy_failures = []
    policy_prediction_failures = []
    for (target_key, rolling_origin, policy), reported_group in policy_selections.groupby(
        ["target_key", "origin_id", "selection_type"], sort=True
    ):
        pool = selected.loc[
            selected["target_key"].eq(target_key)
            & selected["origin_id"].eq(rolling_origin)
        ].copy()
        if policy == "rolling_selected_baseline":
            pool = pool.loc[pool["feature_set"].eq("player_history_baseline")]
        elif policy == "rolling_selected_full":
            pool = pool.loc[pool["feature_set"].eq("full_explicit_fit")]
        expected = pool.sort_values(
            [
                "validation_mae",
                "validation_rmse",
                "model_family",
                "feature_set",
                "selected_hyperparameters",
            ],
            kind="mergesort",
        ).iloc[0]
        reported = reported_group.iloc[0]
        if expected["feature_set"] != reported["feature_set"] or expected["model_family"] != reported[
            "model_family"
        ]:
            policy_failures.append(f"{target_key}:{rolling_origin}:{policy}")
        source_keys = set(
            predictions.loc[
                predictions["target_key"].eq(target_key)
                & predictions["origin_id"].eq(rolling_origin)
                & predictions["feature_set"].eq(reported["feature_set"])
                & predictions["model_family"].eq(reported["model_family"]),
                "canonical_performance_id",
            ].astype(str)
        )
        policy_keys = set(
            policy_predictions.loc[
                policy_predictions["target_key"].eq(target_key)
                & policy_predictions["origin_id"].eq(rolling_origin)
                & policy_predictions["selection_type"].eq(policy),
                "canonical_performance_id",
            ].astype(str)
        )
        if source_keys != policy_keys:
            policy_prediction_failures.append(f"{target_key}:{rolling_origin}:{policy}")
    add(
        "rolling_policy_selections_recomputed",
        not policy_failures and len(policy_selections) == 36,
        ";".join(policy_failures),
        "none across 36 selections",
        "Baseline, full, and overall policies select only from each origin's validation results.",
    )
    add(
        "policy_predictions_map_to_selected_specs",
        not policy_prediction_failures,
        ";".join(policy_prediction_failures),
        "none",
        "Every policy prediction row comes from its selected target-origin specification.",
    )

    locked_failures = []
    for target_key in LOCKED_TARGETS:
        for role, source_type in [
            ("baseline", "validation_selected_baseline"),
            ("candidate", "validation_selected_overall"),
        ]:
            expected = prior_selection.loc[
                prior_selection["target_key"].eq(target_key)
                & prior_selection["selection_type"].eq(source_type)
            ].iloc[0]
            reported = locked_designs.loc[
                locked_designs["target_key"].eq(target_key)
                & locked_designs["design_role"].eq(role)
            ]
            if (
                len(reported) != 1
                or reported.iloc[0]["feature_set"] != expected["feature_set"]
                or reported.iloc[0]["model_family"] != expected["model_family"]
            ):
                locked_failures.append(f"{target_key}:{role}")
    add(
        "locked_designs_match_prior_validation_choices",
        not locked_failures and set(locked_designs["target_key"]) == set(LOCKED_TARGETS),
        ";".join(locked_failures),
        "none",
        "Locked opportunity and performance designs match the preceding validation-selected choices.",
    )
    add(
        "locked_design_outputs_complete",
        len(locked_metrics) == 16
        and len(locked_bootstrap) == 8
        and set(locked_predictions["target_key"]) == set(LOCKED_TARGETS),
        f"metrics={len(locked_metrics)};bootstraps={len(locked_bootstrap)}",
        "metrics=16;bootstraps=8",
        "Both locked designs are evaluated at all four origins; paused adaptation remains excluded.",
    )

    origin_bootstrap_failures = []
    family_pairs: dict[tuple[str, str], pd.DataFrame] = {}
    for target_index, target_key in enumerate(TARGETS):
        for family_index, family in enumerate(MODEL_FAMILIES):
            target_family_pairs = []
            for origin_index, rolling_origin in enumerate(ORIGINS):
                group = predictions.loc[
                    predictions["target_key"].eq(target_key)
                    & predictions["origin_id"].eq(rolling_origin)
                    & predictions["model_family"].eq(family)
                ]
                baseline = group.loc[group["feature_set"].eq("player_history_baseline")].sort_values(
                    "canonical_performance_id"
                )
                candidate = group.loc[group["feature_set"].eq("full_explicit_fit")].sort_values(
                    "canonical_performance_id"
                )
                result = paired_bootstrap(
                    baseline["actual"].to_numpy(dtype=float),
                    baseline["prediction"].to_numpy(dtype=float),
                    candidate["prediction"].to_numpy(dtype=float),
                    RANDOM_SEED + 1000 + target_index * 100 + family_index * 10 + origin_index,
                )
                reported = family_bootstrap.loc[
                    family_bootstrap["target_key"].eq(target_key)
                    & family_bootstrap["model_family"].eq(family)
                    & family_bootstrap["origin_id"].eq(rolling_origin)
                ].iloc[0]
                for name, value in result.items():
                    output_name = "evaluation_n" if name == "test_n" else name
                    if not close(value, reported[output_name]):
                        origin_bootstrap_failures.append(
                            f"family:{target_key}:{family}:{rolling_origin}:{name}"
                        )
                target_family_pairs.append(pair_predictions(baseline, candidate))
            family_pairs[(target_key, family)] = pd.concat(target_family_pairs, ignore_index=True)

    policy_pairs: dict[str, pd.DataFrame] = {}
    for target_index, target_key in enumerate(TARGETS):
        target_pairs = []
        for origin_index, rolling_origin in enumerate(ORIGINS):
            group = policy_predictions.loc[
                policy_predictions["target_key"].eq(target_key)
                & policy_predictions["origin_id"].eq(rolling_origin)
            ]
            baseline = group.loc[
                group["selection_type"].eq("rolling_selected_baseline")
            ].sort_values("canonical_performance_id")
            candidate = group.loc[
                group["selection_type"].eq("rolling_selected_full")
            ].sort_values("canonical_performance_id")
            result = paired_bootstrap(
                baseline["actual"].to_numpy(dtype=float),
                baseline["prediction"].to_numpy(dtype=float),
                candidate["prediction"].to_numpy(dtype=float),
                RANDOM_SEED + 3000 + target_index * 10 + origin_index,
            )
            reported = policy_bootstrap.loc[
                policy_bootstrap["target_key"].eq(target_key)
                & policy_bootstrap["origin_id"].eq(rolling_origin)
            ].iloc[0]
            for name, value in result.items():
                output_name = "evaluation_n" if name == "test_n" else name
                if not close(value, reported[output_name]):
                    origin_bootstrap_failures.append(
                        f"policy:{target_key}:{rolling_origin}:{name}"
                    )
            target_pairs.append(pair_predictions(baseline, candidate))
        policy_pairs[target_key] = pd.concat(target_pairs, ignore_index=True)

    locked_pairs: dict[str, pd.DataFrame] = {}
    for target_index, target_key in enumerate(LOCKED_TARGETS):
        target_pairs = []
        for origin_index, rolling_origin in enumerate(ORIGINS):
            group = locked_predictions.loc[
                locked_predictions["target_key"].eq(target_key)
                & locked_predictions["origin_id"].eq(rolling_origin)
            ]
            baseline = group.loc[group["design_role"].eq("baseline")].sort_values(
                "canonical_performance_id"
            )
            candidate = group.loc[group["design_role"].eq("candidate")].sort_values(
                "canonical_performance_id"
            )
            result = paired_bootstrap(
                baseline["actual"].to_numpy(dtype=float),
                baseline["prediction"].to_numpy(dtype=float),
                candidate["prediction"].to_numpy(dtype=float),
                RANDOM_SEED + 5000 + target_index * 10 + origin_index,
            )
            reported = locked_bootstrap.loc[
                locked_bootstrap["target_key"].eq(target_key)
                & locked_bootstrap["origin_id"].eq(rolling_origin)
            ].iloc[0]
            for name, value in result.items():
                output_name = "evaluation_n" if name == "test_n" else name
                if not close(value, reported[output_name]):
                    origin_bootstrap_failures.append(
                        f"locked:{target_key}:{rolling_origin}:{name}"
                    )
            target_pairs.append(pair_predictions(baseline, candidate))
        locked_pairs[target_key] = pd.concat(target_pairs, ignore_index=True)
    add(
        "origin_bootstraps_recomputed",
        not origin_bootstrap_failures,
        str(len(origin_bootstrap_failures)),
        0,
        "All 56 family, policy, and locked-design paired origin bootstraps reproduce.",
    )

    summary_failures = []
    for target_index, target_key in enumerate(TARGETS):
        for family_index, family in enumerate(MODEL_FAMILIES):
            reported = family_stability.loc[
                family_stability["target_key"].eq(target_key)
                & family_stability["model_family"].eq(family)
            ].iloc[0]
            origin_results = family_bootstrap.loc[
                family_bootstrap["target_key"].eq(target_key)
                & family_bootstrap["model_family"].eq(family)
            ]
            failures = verify_summary(
                reported,
                family_pairs[(target_key, family)],
                RANDOM_SEED + 2000 + target_index * 10 + family_index,
                origin_results,
            )
            summary_failures.extend(
                f"family:{target_key}:{family}:{name}" for name in failures
            )
        reported_policy = policy_comparison.loc[
            policy_comparison["target_key"].eq(target_key)
        ].iloc[0]
        policy_failures_for_target = verify_summary(
            reported_policy,
            policy_pairs[target_key],
            RANDOM_SEED + 4000 + target_index,
            policy_bootstrap.loc[policy_bootstrap["target_key"].eq(target_key)],
        )
        summary_failures.extend(
            f"policy:{target_key}:{name}" for name in policy_failures_for_target
        )
    for target_index, target_key in enumerate(LOCKED_TARGETS):
        reported = locked_stability.loc[locked_stability["target_key"].eq(target_key)].iloc[0]
        failures = verify_summary(
            reported,
            locked_pairs[target_key],
            RANDOM_SEED + 6000 + target_index,
            locked_bootstrap.loc[locked_bootstrap["target_key"].eq(target_key)],
        )
        summary_failures.extend(f"locked:{target_key}:{name}" for name in failures)
    add(
        "aggregate_stability_summaries_recomputed",
        not summary_failures,
        str(len(summary_failures)),
        0,
        "All family, rolling-policy, and locked-design pooled summaries reproduce from paired predictions.",
    )

    effect_failures = []
    for _, row in effects.iterrows():
        baseline = selected.loc[
            selected["target_key"].eq(row["target_key"])
            & selected["origin_id"].eq(row["origin_id"])
            & selected["model_family"].eq(row["model_family"])
            & selected["feature_set"].eq("player_history_baseline")
        ].iloc[0]
        if not close(
            baseline["evaluation_mae"] - row["evaluation_mae"],
            row["evaluation_mae_improvement_vs_baseline"],
        ):
            effect_failures.append(
                f"{row['target_key']}:{row['origin_id']}:{row['model_family']}:{row['feature_set']}"
            )
    add(
        "feature_effects_recomputed",
        not effect_failures and len(effects) == 180,
        str(len(effect_failures)),
        0,
        "Every origin-specific feature-block MAE effect reconciles to selected metrics.",
    )

    frequency_total = selection_frequency.groupby(["target_key", "selection_type"])[
        "origin_selections"
    ].sum()
    add(
        "selection_frequencies_reconcile",
        len(frequency_total) == 9 and frequency_total.eq(4).all(),
        frequency_total.to_dict(),
        "4 selections for each of 9 target-policy groups",
        "Selection-frequency reporting reconciles to the four rolling origins.",
    )

    sensitivity_failures = []
    expected_subsets = {
        "mature_three_origins": {"origin_2020", "origin_2021", "origin_2022"},
        "terminal_two_origins": {"origin_2021", "origin_2022"},
    }
    for _, row in sensitivity.iterrows():
        if set(str(row["included_origins"]).split(";")) != expected_subsets[row["origin_subset"]]:
            sensitivity_failures.append(f"{row['analysis_type']}:{row['target_key']}:{row['origin_subset']}:origins")
            continue
        if row["analysis_type"] == "rolling_selection_policy":
            pairs = policy_pairs[row["target_key"]]
        elif row["analysis_type"] == "retrospective_locked_design":
            pairs = locked_pairs[row["target_key"]]
        else:
            pairs = family_pairs[(row["target_key"], row["model_family"])]
        subset_pairs = pairs.loc[pairs["origin_id"].isin(expected_subsets[row["origin_subset"]])]
        baseline_mae = float(np.mean(np.abs(subset_pairs["actual"] - subset_pairs["baseline_prediction"])))
        candidate_mae = float(np.mean(np.abs(subset_pairs["actual"] - subset_pairs["candidate_prediction"])))
        if (
            not close(baseline_mae, row["pooled_baseline_mae"])
            or not close(candidate_mae, row["pooled_candidate_mae"])
            or int(row["pooled_n"]) != len(subset_pairs)
        ):
            sensitivity_failures.append(
                f"{row['analysis_type']}:{row['target_key']}:{row['origin_subset']}:metrics"
            )
    add(
        "mature_origin_sensitivity_recomputed",
        not sensitivity_failures and len(sensitivity) == 28,
        str(len(sensitivity_failures)),
        0,
        "All mature-three and terminal-two sensitivity cohorts and pooled MAEs reconcile.",
    )

    add(
        "policy_stability_and_frequency_outputs_complete",
        len(policy_stability) == 9 and len(selection_frequency) > 0,
        f"policy_stability={len(policy_stability)};frequency_rows={len(selection_frequency)}",
        "policy_stability=9;frequency_rows>0",
        "Every target-policy combination has a pooled performance and selection-stability record.",
    )

    verification = pd.DataFrame(checks)
    failures = verification.loc[
        verification["severity"].eq("blocking") & ~verification["passed"], "check"
    ].tolist()
    summary = {
        "status": "pass" if not failures else "fail",
        "checks": len(verification),
        "blocking_checks": int(verification["severity"].eq("blocking").sum()),
        "blocking_failures": len(failures),
        "verified_origins": len(origins),
        "verified_tuning_candidates": len(tuning),
        "verified_selected_specs": len(selected),
        "verified_prediction_rows": len(predictions),
        "verified_sensitivity_rows": len(sensitivity),
    }
    verification.to_csv(OUTPUT_DIR / "independent_verification.csv", index=False)
    (OUTPUT_DIR / "independent_verification.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
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
    if failures:
        raise RuntimeError(f"Independent rolling-origin verification failures: {failures}")


if __name__ == "__main__":
    run()

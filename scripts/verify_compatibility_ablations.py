"""Independently verify stronger-model compatibility ablations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_compatibility_models import bool_series, metrics, paired_bootstrap, sha256_file


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "Data" / "processed" / "compatibility_model_ablations"
TARGET_DIR = ROOT / "Data" / "processed" / "compatibility_targets"
MATRIX_PATH = TARGET_DIR / "transfer_compatibility_strict_model_matrix.csv"

RANDOM_SEED = 20260809
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
MODEL_FAMILIES = ["ridge", "elastic_net", "gradient_boosted_trees"]
SPLITS = ["train", "validation", "test"]


def close(left: Any, right: Any, tolerance: float = 1e-10) -> bool:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return left == right
    if math.isnan(left_value) and math.isnan(right_value):
        return True
    return abs(left_value - right_value) <= tolerance


def run() -> None:
    required = [
        "ablation_feature_sets.csv",
        "ablation_validation_tuning.csv",
        "ablation_model_metrics.csv",
        "ablation_test_results.csv",
        "ablation_feature_block_effects.csv",
        "ablation_predictions.csv",
        "ablation_feature_importance.csv",
        "ablation_top_features.csv",
        "ablation_family_bootstrap.csv",
        "ablation_model_selection.csv",
        "ablation_selected_bootstrap.csv",
        "ablation_selected_test_predictions.csv",
        "ablation_build_checks.csv",
        "ablation_run_summary.json",
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
        "Every auditable ablation artifact exists.",
    )
    if missing:
        raise RuntimeError(f"Missing required outputs: {missing}")

    matrix = pd.read_csv(MATRIX_PATH, low_memory=False)
    feature_sets = pd.read_csv(OUTPUT_DIR / "ablation_feature_sets.csv", low_memory=False)
    tuning = pd.read_csv(OUTPUT_DIR / "ablation_validation_tuning.csv", low_memory=False)
    model_metrics = pd.read_csv(OUTPUT_DIR / "ablation_model_metrics.csv", low_memory=False)
    results = pd.read_csv(OUTPUT_DIR / "ablation_test_results.csv", low_memory=False)
    effects = pd.read_csv(OUTPUT_DIR / "ablation_feature_block_effects.csv", low_memory=False)
    predictions = pd.read_csv(OUTPUT_DIR / "ablation_predictions.csv", low_memory=False)
    importance = pd.read_csv(OUTPUT_DIR / "ablation_feature_importance.csv", low_memory=False)
    top_features = pd.read_csv(OUTPUT_DIR / "ablation_top_features.csv", low_memory=False)
    family_bootstrap = pd.read_csv(OUTPUT_DIR / "ablation_family_bootstrap.csv", low_memory=False)
    selections = pd.read_csv(OUTPUT_DIR / "ablation_model_selection.csv", low_memory=False)
    selected_bootstrap = pd.read_csv(OUTPUT_DIR / "ablation_selected_bootstrap.csv", low_memory=False)
    selected_pairs = pd.read_csv(OUTPUT_DIR / "ablation_selected_test_predictions.csv", low_memory=False)
    build_checks = pd.read_csv(OUTPUT_DIR / "ablation_build_checks.csv", low_memory=False)
    source_manifest = pd.read_csv(OUTPUT_DIR / "source_manifest.csv", low_memory=False)

    add(
        "build_checks_all_pass",
        bool_series(build_checks["passed"]).all(),
        int((~bool_series(build_checks["passed"])).sum()),
        0,
        "All training-time invariants passed.",
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
        "Verification uses the exact source artifacts used for training.",
    )

    set_members = {
        name: set(feature_sets.loc[feature_sets["feature_set"].eq(name), "raw_feature"])
        for name in FEATURE_SETS
    }
    baseline = set_members["player_history_baseline"]
    add(
        "baseline_nested_in_all_feature_sets",
        all(baseline.issubset(values) for values in set_members.values()),
        "nested",
        "nested",
        "Every comparison retains the same player/market-history controls.",
    )
    add(
        "feature_set_counts_exact",
        {name: len(values) for name, values in set_members.items()}
        == {
            "player_history_baseline": 81,
            "plus_club_context": 133,
            "plus_player_preferences": 121,
            "context_and_preferences": 173,
            "full_explicit_fit": 202,
        },
        json.dumps({name: len(values) for name, values in set_members.items()}, sort_keys=True),
        "81;133;121;173;202",
        "The declared feature blocks have exact reproducible sizes.",
    )
    add(
        "feature_blocks_are_declared",
        set(feature_sets["introduced_block"]).issubset(
            {
                "transfer_and_market",
                "lagged_player_history",
                "destination_club_context",
                "historical_teammate_preference",
                "explicit_player_club_fit",
            }
        ),
        ";".join(sorted(feature_sets["introduced_block"].unique())),
        "declared blocks only",
        "Every raw feature belongs to one documented ablation block.",
    )

    add(
        "tuning_candidate_count_exact",
        len(tuning) == 420,
        len(tuning),
        420,
        "The validation grid contains all declared ridge, elastic-net, and boosted-tree candidates.",
    )
    add(
        "tuning_contains_no_test_fields",
        all("test" not in column.lower() for column in tuning.columns),
        ";".join(tuning.columns),
        "no test fields",
        "2023 never appears in model or hyperparameter selection artifacts.",
    )

    actual_failures: list[str] = []
    metric_failures: list[str] = []
    key_failures: list[str] = []
    cohort_failures: list[str] = []
    for target_key, (target_column, eligibility_column) in TARGETS.items():
        eligible = bool_series(matrix[eligibility_column]) & matrix[target_column].notna()
        source = matrix.loc[
            eligible,
            ["canonical_performance_id", "recommended_time_split", target_column],
        ].copy()
        source["key"] = source["canonical_performance_id"].astype(str)
        for split in SPLITS:
            expected_rows = source.loc[source["recommended_time_split"].eq(split)]
            spec_counts = predictions.loc[
                predictions["target_key"].eq(target_key)
                & predictions["recommended_time_split"].eq(split)
            ].groupby(["feature_set", "model_family"])["canonical_performance_id"].nunique()
            if len(spec_counts) != 15 or not spec_counts.eq(len(expected_rows)).all():
                cohort_failures.append(f"{target_key}:{split}")
            spec_keys = predictions.loc[
                predictions["target_key"].eq(target_key)
                & predictions["recommended_time_split"].eq(split)
            ].groupby(["feature_set", "model_family"])["canonical_performance_id"].apply(
                lambda values: frozenset(values.astype(str))
            )
            if len(set(spec_keys)) != 1:
                key_failures.append(f"{target_key}:{split}")
            expected_actual = expected_rows.set_index("key")[target_column]
            for feature_set in FEATURE_SETS:
                for model_family in MODEL_FAMILIES:
                    rows = predictions.loc[
                        predictions["target_key"].eq(target_key)
                        & predictions["recommended_time_split"].eq(split)
                        & predictions["feature_set"].eq(feature_set)
                        & predictions["model_family"].eq(model_family)
                    ].copy()
                    rows["key"] = rows["canonical_performance_id"].astype(str)
                    if not np.allclose(
                        rows["actual"].to_numpy(dtype=float),
                        rows["key"].map(expected_actual).to_numpy(dtype=float),
                        atol=1e-12,
                        rtol=0,
                    ):
                        actual_failures.append(f"{target_key}:{split}:{feature_set}:{model_family}")
                    recalculated = metrics(
                        rows["actual"].to_numpy(dtype=float),
                        rows["prediction"].to_numpy(dtype=float),
                    )
                    reported = model_metrics.loc[
                        model_metrics["target_key"].eq(target_key)
                        & model_metrics["split"].eq(split)
                        & model_metrics["feature_set"].eq(feature_set)
                        & model_metrics["model_family"].eq(model_family)
                    ]
                    if len(reported) != 1:
                        metric_failures.append(f"{target_key}:{split}:{feature_set}:{model_family}:rows")
                    else:
                        for metric_name, value in recalculated.items():
                            if not close(value, reported.iloc[0][metric_name]):
                                metric_failures.append(
                                    f"{target_key}:{split}:{feature_set}:{model_family}:{metric_name}"
                                )
    add(
        "cohort_counts_match_all_specs",
        not cohort_failures,
        ";".join(cohort_failures),
        "none",
        "Every specification scores the full eligible chronological cohort.",
    )
    add(
        "prediction_keys_match_all_specs",
        not key_failures,
        ";".join(key_failures),
        "none",
        "Feature and model changes never alter scoring rows.",
    )
    add(
        "actual_targets_match_source",
        not actual_failures,
        ";".join(actual_failures[:20]),
        "none",
        "Every prediction row maps to the verified canonical target by key.",
    )
    add(
        "all_metrics_recompute",
        not metric_failures,
        ";".join(metric_failures[:20]),
        "none",
        "All errors, correlations, and R-squared values are rebuilt from row predictions.",
    )

    selection_failures = []
    tuning_metric_failures = []
    for target_key in TARGETS:
        for feature_set in FEATURE_SETS:
            for model_family in MODEL_FAMILIES:
                candidates = tuning.loc[
                    tuning["target_key"].eq(target_key)
                    & tuning["feature_set"].eq(feature_set)
                    & tuning["model_family"].eq(model_family)
                ].sort_values(["mae", "rmse", "hyperparameters"])
                best = candidates.iloc[0]
                reported = model_metrics.loc[
                    model_metrics["target_key"].eq(target_key)
                    & model_metrics["feature_set"].eq(feature_set)
                    & model_metrics["model_family"].eq(model_family)
                    & model_metrics["split"].eq("validation")
                ].iloc[0]
                if best["hyperparameters"] != reported["selected_hyperparameters"]:
                    selection_failures.append(f"{target_key}:{feature_set}:{model_family}")
                if not close(best["mae"], reported["mae"]) or not close(best["rmse"], reported["rmse"]):
                    tuning_metric_failures.append(f"{target_key}:{feature_set}:{model_family}")
    add(
        "validation_hyperparameter_selection_exact",
        not selection_failures,
        ";".join(selection_failures),
        "none",
        "Every selected model minimizes 2022 MAE with declared tie-breaking.",
    )
    add(
        "selected_validation_metrics_match_tuning",
        not tuning_metric_failures,
        ";".join(tuning_metric_failures),
        "none",
        "The reported validation metrics correspond to the exact selected candidate.",
    )

    result_failures = []
    for _, row in results.iterrows():
        validation = model_metrics.loc[
            model_metrics["target_key"].eq(row["target_key"])
            & model_metrics["feature_set"].eq(row["feature_set"])
            & model_metrics["model_family"].eq(row["model_family"])
            & model_metrics["split"].eq("validation")
        ].iloc[0]
        test = model_metrics.loc[
            model_metrics["target_key"].eq(row["target_key"])
            & model_metrics["feature_set"].eq(row["feature_set"])
            & model_metrics["model_family"].eq(row["model_family"])
            & model_metrics["split"].eq("test")
        ].iloc[0]
        if not (
            close(row["validation_mae"], validation["mae"])
            and close(row["validation_rmse"], validation["rmse"])
            and close(row["test_mae"], test["mae"])
            and close(row["test_rmse"], test["rmse"])
            and close(row["test_r2"], test["r2"])
        ):
            result_failures.append(f"{row['target_key']}:{row['feature_set']}:{row['model_family']}")
    add(
        "test_results_reconcile_to_metrics",
        not result_failures,
        ";".join(result_failures),
        "none",
        "The compact result grid reconciles to full validation and test metrics.",
    )

    effect_failures = []
    for _, row in effects.iterrows():
        baseline_row = results.loc[
            results["target_key"].eq(row["target_key"])
            & results["model_family"].eq(row["model_family"])
            & results["feature_set"].eq("player_history_baseline")
        ].iloc[0]
        if not (
            close(
                baseline_row["validation_mae"] - row["validation_mae"],
                row["validation_mae_improvement_vs_baseline"],
            )
            and close(
                baseline_row["test_mae"] - row["test_mae"],
                row["test_mae_improvement_vs_baseline"],
            )
        ):
            effect_failures.append(f"{row['target_key']}:{row['feature_set']}:{row['model_family']}")
    add(
        "feature_block_effects_recompute",
        not effect_failures,
        ";".join(effect_failures),
        "none",
        "Every block improvement is an exact same-family difference from the baseline.",
    )

    family_bootstrap_failures = []
    for target_index, target_key in enumerate(TARGETS):
        for family_index, model_family in enumerate(MODEL_FAMILIES):
            rows = predictions.loc[
                predictions["target_key"].eq(target_key)
                & predictions["model_family"].eq(model_family)
                & predictions["recommended_time_split"].eq("test")
            ]
            baseline_rows = rows.loc[
                rows["feature_set"].eq("player_history_baseline")
            ].sort_values("canonical_performance_id")
            full_rows = rows.loc[rows["feature_set"].eq("full_explicit_fit")].sort_values(
                "canonical_performance_id"
            )
            recomputed = paired_bootstrap(
                baseline_rows["actual"].to_numpy(dtype=float),
                baseline_rows["prediction"].to_numpy(dtype=float),
                full_rows["prediction"].to_numpy(dtype=float),
                RANDOM_SEED + 100 + target_index * 10 + family_index,
            )
            reported = family_bootstrap.loc[
                family_bootstrap["target_key"].eq(target_key)
                & family_bootstrap["model_family"].eq(model_family)
            ].iloc[0]
            if not all(close(value, reported[name]) for name, value in recomputed.items()):
                family_bootstrap_failures.append(f"{target_key}:{model_family}")
    add(
        "within_family_bootstrap_exact",
        not family_bootstrap_failures,
        ";".join(family_bootstrap_failures),
        "none",
        "All paired full-versus-baseline uncertainty estimates regenerate exactly.",
    )

    selection_policy_failures = []
    selected_bootstrap_failures = []
    pair_failures = []
    for target_index, target_key in enumerate(TARGETS):
        target_results = results.loc[results["target_key"].eq(target_key)]
        expected = {
            "validation_selected_baseline": target_results.loc[
                target_results["feature_set"].eq("player_history_baseline")
            ].sort_values(["validation_mae", "validation_rmse", "model_family"]).iloc[0],
            "validation_selected_full": target_results.loc[
                target_results["feature_set"].eq("full_explicit_fit")
            ].sort_values(["validation_mae", "validation_rmse", "model_family"]).iloc[0],
            "validation_selected_overall": target_results.sort_values(
                ["validation_mae", "validation_rmse", "feature_set", "model_family"]
            ).iloc[0],
        }
        for selection_type, expected_row in expected.items():
            reported = selections.loc[
                selections["target_key"].eq(target_key)
                & selections["selection_type"].eq(selection_type)
            ]
            if len(reported) != 1 or not (
                reported.iloc[0]["feature_set"] == expected_row["feature_set"]
                and reported.iloc[0]["model_family"] == expected_row["model_family"]
                and close(reported.iloc[0]["validation_mae"], expected_row["validation_mae"])
            ):
                selection_policy_failures.append(f"{target_key}:{selection_type}")

        baseline_spec = expected["validation_selected_baseline"]
        full_spec = expected["validation_selected_full"]
        prediction_subset = predictions.loc[
            predictions["target_key"].eq(target_key)
            & predictions["recommended_time_split"].eq("test")
        ]
        baseline_rows = prediction_subset.loc[
            prediction_subset["feature_set"].eq(baseline_spec["feature_set"])
            & prediction_subset["model_family"].eq(baseline_spec["model_family"])
        ].sort_values("canonical_performance_id")
        full_rows = prediction_subset.loc[
            prediction_subset["feature_set"].eq(full_spec["feature_set"])
            & prediction_subset["model_family"].eq(full_spec["model_family"])
        ].sort_values("canonical_performance_id")
        recomputed = paired_bootstrap(
            baseline_rows["actual"].to_numpy(dtype=float),
            baseline_rows["prediction"].to_numpy(dtype=float),
            full_rows["prediction"].to_numpy(dtype=float),
            RANDOM_SEED + 500 + target_index,
        )
        reported_bootstrap = selected_bootstrap.loc[
            selected_bootstrap["target_key"].eq(target_key)
        ].iloc[0]
        if not all(close(value, reported_bootstrap[name]) for name, value in recomputed.items()):
            selected_bootstrap_failures.append(target_key)

        reported_pairs = selected_pairs.loc[selected_pairs["target_key"].eq(target_key)].sort_values(
            "canonical_performance_id"
        )
        if not (
            reported_pairs["canonical_performance_id"].astype(str).tolist()
            == baseline_rows["canonical_performance_id"].astype(str).tolist()
            == full_rows["canonical_performance_id"].astype(str).tolist()
            and np.allclose(reported_pairs["baseline_prediction"], baseline_rows["prediction"])
            and np.allclose(reported_pairs["full_prediction"], full_rows["prediction"])
        ):
            pair_failures.append(target_key)
    add(
        "validation_model_selection_recomputes",
        not selection_policy_failures,
        ";".join(selection_policy_failures),
        "none",
        "Baseline, full, and overall specifications are selected strictly by 2022 validation metrics.",
    )
    add(
        "selected_bootstrap_exact",
        not selected_bootstrap_failures,
        ";".join(selected_bootstrap_failures),
        "none",
        "Validation-selected baseline/full uncertainty regenerates from their 2023 predictions.",
    )
    add(
        "selected_pair_predictions_reconcile",
        not pair_failures,
        ";".join(pair_failures),
        "none",
        "Analyst-friendly paired predictions exactly match selected model rows.",
    )

    add(
        "importance_features_are_declared",
        set(importance["raw_feature"]).issubset(set_members["full_explicit_fit"]),
        len(set(importance["raw_feature"]) - set_members["full_explicit_fit"]),
        0,
        "Every attribution traces to a declared strict feature.",
    )
    add(
        "top_features_trace_to_importance",
        set(top_features["encoded_feature"]).issubset(set(importance["encoded_feature"])),
        len(set(top_features["encoded_feature"]) - set(importance["encoded_feature"])),
        0,
        "Published top signals are a subset of final-fit model importance rows.",
    )
    add(
        "all_predictions_finite",
        np.isfinite(predictions[["actual", "prediction_raw", "prediction"]].to_numpy(dtype=float)).all(),
        int((~np.isfinite(predictions[["actual", "prediction_raw", "prediction"]].to_numpy(dtype=float))).sum()),
        0,
        "No scored row contains a missing or infinite value.",
    )
    add(
        "opportunity_predictions_bounded",
        predictions.loc[predictions["target_key"].eq("opportunity"), "prediction"].between(0, 1).all(),
        "within bounds",
        "within [0,1]",
        "All opportunity models respect the valid target range.",
    )

    verification = pd.DataFrame(checks)
    failures = verification.loc[~verification["passed"], "check"].tolist()
    summary = {
        "status": "pass" if not failures else "fail",
        "checks": len(verification),
        "blocking_checks": len(verification),
        "blocking_failures": len(failures),
        "verified_prediction_rows": len(predictions),
        "verified_selected_specs": len(results),
        "verified_tuning_candidates": len(tuning),
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
        raise RuntimeError(f"Independent verification failures: {failures}")


if __name__ == "__main__":
    run()

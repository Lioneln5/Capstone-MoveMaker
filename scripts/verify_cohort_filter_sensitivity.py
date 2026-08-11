"""Independent verification for the MoveMaker cohort-filter sensitivity diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_rolling_origin_stability as rolling


ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT / "Data" / "processed" / "compatibility_targets"
SUPPORT_DIR = TARGET_DIR / "cohort_filter_sensitivity"
MODEL_DIR = ROOT / "Data" / "processed" / "compatibility_models"
ROLLING_DIR = ROOT / "Data" / "processed" / "rolling_origin_stability"

MATRIX_PATH = TARGET_DIR / "transfer_compatibility_strict_model_matrix.csv"
COMPARISON_PATH = TARGET_DIR / "cohort_filter_sensitivity.csv"
FUNNEL_PATH = SUPPORT_DIR / "cohort_filter_sensitivity_funnel.csv"
MEMBERSHIP_PATH = SUPPORT_DIR / "cohort_filter_sensitivity_membership.csv"
ORIGIN_RESULTS_PATH = SUPPORT_DIR / "cohort_filter_sensitivity_origin_results.csv"
SELECTED_SPECS_PATH = SUPPORT_DIR / "cohort_filter_sensitivity_selected_specs.csv"
PAIR_PREDICTIONS_PATH = SUPPORT_DIR / "cohort_filter_sensitivity_pair_predictions.csv"
SOURCE_MANIFEST_PATH = SUPPORT_DIR / "source_manifest.csv"
BUILD_CHECKS_PATH = SUPPORT_DIR / "build_checks.csv"
OUTPUT_CSV = SUPPORT_DIR / "independent_verification.csv"
OUTPUT_JSON = SUPPORT_DIR / "independent_verification.json"

CONDITIONS = {
    "primary_policy": (False, False, False),
    "all_season_arrivals": (True, False, False),
    "no_destination_minutes_threshold": (False, True, False),
    "all_roles": (False, False, True),
    "fully_relaxed": (True, True, True),
}
TARGET_COLUMNS = {
    "opportunity": (
        "target_destination_minutes_share",
        "target_destination_minutes_share_available",
        "eligible_opportunity_model_primary",
    ),
    "performance": (
        "target_role_performance_index",
        "target_role_performance_index_available",
        "eligible_performance_model_primary",
    ),
    "adaptation": (
        "target_performance_change_rolling3",
        "target_performance_change_rolling3_available",
        "eligible_adaptation_model_primary",
    ),
}
STAGES = [
    "all_target_rows",
    "target_available",
    "transfer_type_transfer_or_loan",
    "strict_compatibility_features",
    "june_october_arrival_window",
    "outfield_role",
    "destination_minutes_at_least_450",
    "prior3_performance_baseline_available",
    "prior3_baseline_minutes_at_least_450",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.casefold().isin({"true", "1", "yes"})


def stage_sets(
    matrix: pd.DataFrame, target_key: str, condition_name: str
) -> tuple[dict[str, set[str]], list[dict[str, Any]]]:
    relax_window, relax_minutes, relax_role = CONDITIONS[condition_name]
    target_column, availability_column, _ = TARGET_COLUMNS[target_key]
    all_rows = pd.Series(True, index=matrix.index)
    masks = {
        "all_target_rows": all_rows,
        "target_available": bool_series(matrix[availability_column])
        & pd.to_numeric(matrix[target_column], errors="coerce").notna(),
        "transfer_type_transfer_or_loan": matrix["canonical_transfer_type"].isin(
            {"transfer", "loan"}
        ),
        "strict_compatibility_features": bool_series(
            matrix["strict_model_feature_eligible"]
        ),
        "june_october_arrival_window": matrix["transfer_window"].eq(
            "summer_primary"
        ),
        "outfield_role": matrix["role_group"].isin({"DF", "MF", "FW"}),
        "destination_minutes_at_least_450": pd.to_numeric(
            matrix["target_destination_minutes"], errors="coerce"
        ).ge(450.0),
        "prior3_performance_baseline_available": bool_series(
            matrix["prior3_role_performance_baseline_available"]
        ),
        "prior3_baseline_minutes_at_least_450": pd.to_numeric(
            matrix["prior3_role_performance_baseline_minutes"], errors="coerce"
        ).ge(450.0),
    }
    applicable = {
        stage: True for stage in STAGES
    }
    applicable["destination_minutes_at_least_450"] = target_key in {
        "performance",
        "adaptation",
    }
    applicable["prior3_performance_baseline_available"] = target_key == "adaptation"
    applicable["prior3_baseline_minutes_at_least_450"] = target_key == "adaptation"
    active = dict(applicable)
    active["june_october_arrival_window"] = not relax_window
    active["outfield_role"] = not relax_role
    active["destination_minutes_at_least_450"] = (
        applicable["destination_minutes_at_least_450"] and not relax_minutes
    )

    current = pd.Series(True, index=matrix.index)
    sets: dict[str, set[str]] = {}
    rows: list[dict[str, Any]] = []
    for order, stage in enumerate(STAGES, start=1):
        before = int(current.sum())
        if applicable[stage] and active[stage]:
            current = current & masks[stage]
        after = int(current.sum())
        sets[stage] = set(matrix.loc[current, "canonical_performance_id"].astype(str))
        rows.append(
            {
                "funnel_order": order,
                "funnel_stage": stage,
                "stage_applicable": bool(applicable[stage]),
                "filter_active": bool(applicable[stage] and active[stage]),
                "rows_entering_stage": before,
                "rows_removed_at_stage": before - after,
                "rows_remaining": after,
            }
        )
    return sets, rows


def main() -> None:
    required = [
        MATRIX_PATH,
        COMPARISON_PATH,
        FUNNEL_PATH,
        MEMBERSHIP_PATH,
        ORIGIN_RESULTS_PATH,
        SELECTED_SPECS_PATH,
        PAIR_PREDICTIONS_PATH,
        SOURCE_MANIFEST_PATH,
        BUILD_CHECKS_PATH,
    ]
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
        "required_outputs_exist",
        all(path.exists() for path in required),
        ";".join(path.name for path in required if not path.exists()),
        "all present",
        "Every declared diagnostic and audit artifact must exist.",
    )
    if not all(path.exists() for path in required):
        pd.DataFrame(checks).to_csv(OUTPUT_CSV, index=False)
        raise SystemExit(1)

    matrix = pd.read_csv(MATRIX_PATH, low_memory=False)
    matrix["canonical_performance_id"] = matrix["canonical_performance_id"].astype(str)
    comparison = pd.read_csv(COMPARISON_PATH, low_memory=False)
    funnel = pd.read_csv(FUNNEL_PATH, low_memory=False)
    membership = pd.read_csv(MEMBERSHIP_PATH, low_memory=False)
    origins = pd.read_csv(ORIGIN_RESULTS_PATH, low_memory=False)
    selected = pd.read_csv(SELECTED_SPECS_PATH, low_memory=False)
    pairs = pd.read_csv(PAIR_PREDICTIONS_PATH, low_memory=False)
    manifest = pd.read_csv(SOURCE_MANIFEST_PATH, low_memory=False)
    build_checks = pd.read_csv(BUILD_CHECKS_PATH, low_memory=False)

    add_check(
        "matrix_grain_unique",
        not matrix["canonical_performance_id"].duplicated().any(),
        int(matrix["canonical_performance_id"].duplicated().sum()),
        0,
        "The frozen strict matrix remains one row per destination performance record.",
    )
    add_check(
        "declared_condition_target_grid_complete",
        len(comparison) == len(CONDITIONS) * len(TARGET_COLUMNS)
        and set(comparison["filter_condition"]) == set(CONDITIONS)
        and set(comparison["target_key"]) == set(TARGET_COLUMNS),
        len(comparison),
        len(CONDITIONS) * len(TARGET_COLUMNS),
        "Every condition-target combination must appear exactly once.",
    )
    add_check(
        "all_build_checks_pass",
        bool(bool_series(build_checks["passed"]).all()),
        int((~bool_series(build_checks["passed"])).sum()),
        0,
        "Independent verification starts only from a passing diagnostic build.",
    )

    source_mismatches = []
    for row in manifest.itertuples(index=False):
        path = ROOT / str(row.source_file)
        current = sha256(path)
        if current != str(row.before_sha256) or current != str(row.after_sha256):
            source_mismatches.append(str(row.source_file))
    add_check(
        "source_hashes_unchanged",
        not source_mismatches,
        ";".join(source_mismatches),
        "none",
        "Every source file must still match its recorded pre-run and post-run SHA-256 hash.",
    )

    reconstructed: dict[tuple[str, str, str], set[str]] = {}
    funnel_mismatches = []
    membership_mismatches = []
    primary_flag_mismatches = []
    for condition_name in CONDITIONS:
        for target_key, (_, _, eligibility_column) in TARGET_COLUMNS.items():
            sets, rows = stage_sets(matrix, target_key, condition_name)
            for stage, members in sets.items():
                reconstructed[(condition_name, target_key, stage)] = members
            reported_funnel = funnel.loc[
                funnel["filter_condition"].eq(condition_name)
                & funnel["target_key"].eq(target_key)
            ].sort_values("funnel_order")
            expected_funnel = pd.DataFrame(rows)
            compare_columns = [
                "funnel_order",
                "funnel_stage",
                "stage_applicable",
                "filter_active",
                "rows_entering_stage",
                "rows_removed_at_stage",
                "rows_remaining",
            ]
            actual = reported_funnel[compare_columns].reset_index(drop=True).copy()
            actual["stage_applicable"] = bool_series(actual["stage_applicable"])
            actual["filter_active"] = bool_series(actual["filter_active"])
            if not actual.equals(expected_funnel[compare_columns]):
                funnel_mismatches.append(f"{condition_name}:{target_key}")

            expected_members = sets[STAGES[-1]]
            reported_members = set(
                membership.loc[
                    membership["filter_condition"].eq(condition_name)
                    & membership["target_key"].eq(target_key),
                    "canonical_performance_id",
                ].astype(str)
            )
            if expected_members != reported_members:
                membership_mismatches.append(f"{condition_name}:{target_key}")
            if condition_name == "primary_policy":
                frozen_members = set(
                    matrix.loc[
                        bool_series(matrix[eligibility_column]),
                        "canonical_performance_id",
                    ].astype(str)
                )
                if expected_members != frozen_members:
                    primary_flag_mismatches.append(target_key)

    add_check(
        "funnel_counts_independently_recomputed",
        not funnel_mismatches,
        ";".join(funnel_mismatches),
        "none",
        "Every funnel stage is independently reconstructed from the frozen matrix.",
    )
    add_check(
        "final_membership_independently_recomputed",
        not membership_mismatches,
        ";".join(membership_mismatches),
        "none",
        "Every reported final cohort exactly matches independent policy reconstruction.",
    )
    add_check(
        "primary_flags_exactly_reproduced",
        not primary_flag_mismatches,
        ";".join(primary_flag_mismatches),
        "none",
        "Primary reconstructed membership must equal the frozen eligibility flags.",
    )

    stage_superset_failures = []
    final_superset_failures = []
    for condition_name in set(CONDITIONS) - {"primary_policy"}:
        for target_key in TARGET_COLUMNS:
            for stage in STAGES:
                primary = reconstructed[("primary_policy", target_key, stage)]
                relaxed = reconstructed[(condition_name, target_key, stage)]
                if not primary.issubset(relaxed):
                    stage_superset_failures.append(
                        f"{condition_name}:{target_key}:{stage}"
                    )
            primary_final = reconstructed[("primary_policy", target_key, STAGES[-1])]
            relaxed_final = reconstructed[(condition_name, target_key, STAGES[-1])]
            if not primary_final.issubset(relaxed_final):
                final_superset_failures.append(f"{condition_name}:{target_key}")
    add_check(
        "relaxed_funnel_stages_are_primary_supersets",
        not stage_superset_failures,
        ";".join(stage_superset_failures),
        "none",
        "Every relaxed stage must retain all IDs surviving the corresponding primary stage.",
    )
    add_check(
        "relaxed_final_cohorts_are_primary_supersets",
        not final_superset_failures,
        ";".join(final_superset_failures),
        "none",
        "Every relaxed final cohort must retain every frozen primary row.",
    )

    add_check(
        "only_ridge_selected",
        set(selected["model_family"]) == {"ridge"}
        and set(pairs["model_family"]) == {"ridge"},
        ";".join(sorted(set(selected["model_family"]) | set(pairs["model_family"]))),
        "ridge",
        "The diagnostic cannot contain elastic-net or boosted-tree fits.",
    )
    expected_feature_sets = {"player_history_baseline", "full_explicit_fit"}
    add_check(
        "only_declared_feature_sets_selected",
        set(selected["feature_set"]) == expected_feature_sets,
        ";".join(sorted(selected["feature_set"].unique())),
        ";".join(sorted(expected_feature_sets)),
        "The diagnostic must score exactly two declared feature sets.",
    )
    add_check(
        "selected_spec_grid_complete",
        len(selected)
        == len(CONDITIONS) * len(TARGET_COLUMNS) * len(rolling.VALIDATION_YEARS) * 2,
        len(selected),
        len(CONDITIONS) * len(TARGET_COLUMNS) * len(rolling.VALIDATION_YEARS) * 2,
        "Each condition, target, origin, and feature set needs one selected ridge specification.",
    )

    timing_failures = []
    for column in [
        "advanced_max_source_season_used",
        "preference_max_source_season",
        "strict_context_source_season",
    ]:
        values = pd.to_numeric(pairs[column], errors="coerce")
        bad = values.notna() & values.ge(pd.to_numeric(pairs["season_start_year"]))
        if bad.any():
            timing_failures.append(f"{column}:{int(bad.sum())}")
    if not bool_series(pairs["source_timing_safe"]).all():
        timing_failures.append(
            f"source_timing_safe:{int((~bool_series(pairs['source_timing_safe'])).sum())}"
        )
    add_check(
        "all_scored_rows_use_preseason_sources",
        not timing_failures,
        ";".join(timing_failures),
        "none",
        "All player, preference, and destination-context source seasons precede outcomes.",
    )

    feature_dictionary = pd.read_csv(
        MODEL_DIR / "model_feature_dictionary.csv", low_memory=False
    )
    feature_sets, _ = rolling.feature_sets(feature_dictionary)
    used_features = feature_sets["player_history_baseline"] + feature_sets[
        "full_explicit_fit"
    ]
    forbidden = sorted(
        {
            feature
            for feature in used_features
            if feature.startswith(rolling.FORBIDDEN_PREDICTOR_PREFIXES)
        }
    )
    add_check(
        "no_post_outcome_or_eligibility_predictors",
        not forbidden,
        ";".join(forbidden),
        "none",
        "No target, eligibility, exclusion, or roster field enters scoring.",
    )

    metric_mismatches = []
    for target_index, target_key in enumerate(TARGET_COLUMNS):
        for condition_name in CONDITIONS:
            group = pairs.loc[
                pairs["filter_condition"].eq(condition_name)
                & pairs["target_key"].eq(target_key)
            ].copy()
            origin_group = origins.loc[
                origins["filter_condition"].eq(condition_name)
                & origins["target_key"].eq(target_key)
            ].copy()
            rebuilt = rolling.summarize_comparison(
                group,
                origin_group,
                rolling.RANDOM_SEED + 2000 + target_index * 10,
            )
            reported = comparison.loc[
                comparison["filter_condition"].eq(condition_name)
                & comparison["target_key"].eq(target_key)
            ].iloc[0]
            numeric_fields = [
                "pooled_n",
                "pooled_baseline_mae",
                "pooled_candidate_mae",
                "pooled_mae_improvement",
                "mae_improvement_ci_lower_95",
                "mae_improvement_ci_upper_95",
                "origin_count",
                "origins_candidate_better",
                "origin_win_rate",
                "origin_improvement_std",
            ]
            for field in numeric_fields:
                if not np.isclose(
                    float(rebuilt[field]), float(reported[field]), atol=1e-12, rtol=0
                ):
                    metric_mismatches.append(f"{condition_name}:{target_key}:{field}")
            if rebuilt["stability_label"] != reported["stability_label"]:
                metric_mismatches.append(
                    f"{condition_name}:{target_key}:stability_label"
                )
    add_check(
        "pooled_metrics_and_labels_recomputed",
        not metric_mismatches,
        ";".join(metric_mismatches),
        "none",
        "All pooled effects, intervals, win rates, variance, and labels are recomputed from pairs.",
    )

    published = pd.read_csv(
        ROLLING_DIR / "rolling_family_stability_summary.csv", low_memory=False
    )
    published = published.loc[published["model_family"].eq("ridge")].set_index(
        "target_key"
    )
    primary = comparison.loc[
        comparison["filter_condition"].eq("primary_policy")
    ].set_index("target_key")
    published_fields = [
        "pooled_n",
        "pooled_baseline_mae",
        "pooled_candidate_mae",
        "pooled_mae_improvement",
        "mae_improvement_ci_lower_95",
        "mae_improvement_ci_upper_95",
        "origin_count",
        "origins_candidate_better",
        "origin_win_rate",
        "origin_improvement_std",
    ]
    published_mismatches = []
    for target in TARGET_COLUMNS:
        for field in published_fields:
            if not np.isclose(
                float(primary.loc[target, field]),
                float(published.loc[target, field]),
                atol=1e-12,
                rtol=0,
            ):
                published_mismatches.append(f"{target}:{field}")
        if primary.loc[target, "stability_label"] != published.loc[
            target, "stability_label"
        ]:
            published_mismatches.append(f"{target}:stability_label")
    add_check(
        "published_primary_ridge_effects_match",
        not published_mismatches,
        ";".join(published_mismatches),
        "none",
        "The diagnostic primary row must reproduce all published ridge point, interval, origin, and stability fields.",
    )

    checks_frame = pd.DataFrame(checks)
    failures = int((~checks_frame["passed"]).sum())
    checks_frame.to_csv(OUTPUT_CSV, index=False, encoding="utf-8", lineterminator="\n")
    summary = {
        "status": "pass" if failures == 0 else "fail",
        "blocking_checks": len(checks_frame),
        "blocking_failures": failures,
        "conditions": len(CONDITIONS),
        "targets": len(TARGET_COLUMNS),
        "pair_rows": len(pairs),
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

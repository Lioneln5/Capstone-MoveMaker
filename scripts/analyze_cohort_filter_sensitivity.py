"""Diagnostic cohort-filter sensitivity for MoveMaker compatibility models.

This script does not rebuild or overwrite the frozen compatibility feature, target,
model, ablation, rolling-origin, subgroup, or case-study stages. It reads the frozen
strict model matrix, relaxes declared cohort filters in memory, and evaluates only
ridge player-history baseline versus ridge full explicit compatibility features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import build_compatibility_features as feature_logic
import build_compatibility_targets as target_logic
import run_rolling_origin_stability as rolling


ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT / "Data" / "processed" / "compatibility_targets"
MODEL_DIR = ROOT / "Data" / "processed" / "compatibility_models"
ROLLING_DIR = ROOT / "Data" / "processed" / "rolling_origin_stability"
SUPPORT_DIR = TARGET_DIR / "cohort_filter_sensitivity"

MATRIX_PATH = TARGET_DIR / "transfer_compatibility_strict_model_matrix.csv"
TARGETS_PATH = TARGET_DIR / "transfer_compatibility_targets.csv"
EXCLUSION_PATH = TARGET_DIR / "target_exclusion_summary.csv"
FEATURE_DICTIONARY_PATH = MODEL_DIR / "model_feature_dictionary.csv"
PRIMARY_ROLLING_PATH = ROLLING_DIR / "rolling_family_stability_summary.csv"

COMPARISON_PATH = TARGET_DIR / "cohort_filter_sensitivity.csv"
FUNNEL_PATH = SUPPORT_DIR / "cohort_filter_sensitivity_funnel.csv"
MEMBERSHIP_PATH = SUPPORT_DIR / "cohort_filter_sensitivity_membership.csv"
ORIGIN_RESULTS_PATH = SUPPORT_DIR / "cohort_filter_sensitivity_origin_results.csv"
SELECTED_SPECS_PATH = SUPPORT_DIR / "cohort_filter_sensitivity_selected_specs.csv"
PAIR_PREDICTIONS_PATH = SUPPORT_DIR / "cohort_filter_sensitivity_pair_predictions.csv"
SOURCE_MANIFEST_PATH = SUPPORT_DIR / "source_manifest.csv"
BUILD_CHECKS_PATH = SUPPORT_DIR / "build_checks.csv"
README_PATH = SUPPORT_DIR / "README.md"
RUN_SUMMARY_PATH = SUPPORT_DIR / "run_summary.json"

EFFECT_UNCHANGED_TOLERANCE = 0.001
MEANINGFUL_CI_WIDTH_CHANGE = 0.10

CONDITIONS = [
    {
        "condition_order": 1,
        "filter_condition": "primary_policy",
        "description": "Frozen primary cohort policy.",
        "relax_arrival_window": False,
        "relax_destination_minutes": False,
        "relax_outfield_role": False,
    },
    {
        "condition_order": 2,
        "filter_condition": "all_season_arrivals",
        "description": "Drops the June-October arrival-window restriction only.",
        "relax_arrival_window": True,
        "relax_destination_minutes": False,
        "relax_outfield_role": False,
    },
    {
        "condition_order": 3,
        "filter_condition": "no_destination_minutes_threshold",
        "description": "Drops the 450 destination-minute threshold only.",
        "relax_arrival_window": False,
        "relax_destination_minutes": True,
        "relax_outfield_role": False,
    },
    {
        "condition_order": 4,
        "filter_condition": "all_roles",
        "description": "Drops the outfield-only restriction only.",
        "relax_arrival_window": False,
        "relax_destination_minutes": False,
        "relax_outfield_role": True,
    },
    {
        "condition_order": 5,
        "filter_condition": "fully_relaxed",
        "description": "Drops arrival-window, destination-minute, and outfield-only restrictions together.",
        "relax_arrival_window": True,
        "relax_destination_minutes": True,
        "relax_outfield_role": True,
    },
]

TARGET_AVAILABILITY_COLUMNS = {
    "opportunity": "target_destination_minutes_share_available",
    "performance": "target_role_performance_index_available",
    "adaptation": "target_performance_change_rolling3_available",
}


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    target_logic.write_csv(frame, path)


def prepare_matrix() -> pd.DataFrame:
    matrix = pd.read_csv(MATRIX_PATH, low_memory=False)
    matrix["canonical_performance_id"] = matrix["canonical_performance_id"].map(
        feature_logic.clean_id
    )
    matrix["season_start_year"] = pd.to_numeric(
        matrix["season_start_year"], errors="raise"
    ).astype("int64")
    matrix["transfer_date"] = pd.to_datetime(matrix["transfer_date"], errors="coerce")
    numeric_columns = [
        "target_destination_minutes",
        "target_destination_minutes_share",
        "target_role_performance_index",
        "target_performance_change_rolling3",
        "prior3_role_performance_baseline_minutes",
        "advanced_max_source_season_used",
        "preference_max_source_season",
        "strict_context_source_season",
    ]
    for column in numeric_columns:
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
    boolean_columns = [
        "strict_model_feature_eligible",
        "target_destination_minutes_share_available",
        "target_role_performance_index_available",
        "prior3_role_performance_baseline_available",
        "target_performance_change_rolling3_available",
        "eligible_opportunity_model_primary",
        "eligible_performance_model_primary",
        "eligible_adaptation_model_primary",
    ]
    for column in boolean_columns:
        matrix[column] = target_logic.bool_series(matrix[column])
    return matrix


def policy_eligibility(
    matrix: pd.DataFrame,
    target_config: dict[str, Any],
    condition: dict[str, Any],
) -> tuple[pd.Series, pd.Series]:
    """Call the frozen target exclusion logic after neutralizing only relaxed filters."""
    policy_frame = matrix.copy()
    if condition["relax_arrival_window"]:
        policy_frame["transfer_window"] = "summer_primary"
    if condition["relax_outfield_role"]:
        policy_frame["role_group"] = "DF"
    if condition["relax_destination_minutes"]:
        policy_frame["target_destination_minutes"] = float(target_logic.MIN_TARGET_MINUTES)

    target_key = target_config["target_key"]
    reasons = target_logic.exclusion_reasons(policy_frame, target_key)
    target_available = pd.to_numeric(
        matrix[target_config["target_column"]], errors="coerce"
    ).notna()
    return reasons.eq("") & target_available, reasons


def funnel_stages(
    matrix: pd.DataFrame,
    target_config: dict[str, Any],
    condition: dict[str, Any],
) -> list[tuple[str, bool, bool, pd.Series]]:
    target_key = target_config["target_key"]
    all_rows = pd.Series(True, index=matrix.index)
    target_available = (
        target_logic.bool_series(matrix[TARGET_AVAILABILITY_COLUMNS[target_key]])
        & pd.to_numeric(matrix[target_config["target_column"]], errors="coerce").notna()
    )
    transfer_type = matrix["canonical_transfer_type"].isin(target_logic.PRIMARY_TRANSFER_TYPES)
    strict_features = target_logic.bool_series(matrix["strict_model_feature_eligible"])
    summer = matrix["transfer_window"].eq("summer_primary")
    outfield = matrix["role_group"].isin(target_logic.OUTFIELD_ROLES)
    destination_minutes = pd.to_numeric(
        matrix["target_destination_minutes"], errors="coerce"
    ).ge(target_logic.MIN_TARGET_MINUTES)
    prior_available = target_logic.bool_series(
        matrix["prior3_role_performance_baseline_available"]
    )
    prior_minutes = pd.to_numeric(
        matrix["prior3_role_performance_baseline_minutes"], errors="coerce"
    ).ge(target_logic.MIN_BASELINE_MINUTES)

    return [
        ("all_target_rows", True, True, all_rows),
        ("target_available", True, True, target_available),
        ("transfer_type_transfer_or_loan", True, True, transfer_type),
        ("strict_compatibility_features", True, True, strict_features),
        (
            "june_october_arrival_window",
            True,
            not condition["relax_arrival_window"],
            summer,
        ),
        (
            "outfield_role",
            True,
            not condition["relax_outfield_role"],
            outfield,
        ),
        (
            "destination_minutes_at_least_450",
            target_key in {"performance", "adaptation"},
            target_key in {"performance", "adaptation"}
            and not condition["relax_destination_minutes"],
            destination_minutes,
        ),
        (
            "prior3_performance_baseline_available",
            target_key == "adaptation",
            target_key == "adaptation",
            prior_available,
        ),
        (
            "prior3_baseline_minutes_at_least_450",
            target_key == "adaptation",
            target_key == "adaptation",
            prior_minutes,
        ),
    ]


def build_funnel(
    matrix: pd.DataFrame,
    target_config: dict[str, Any],
    condition: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    current = pd.Series(True, index=matrix.index)
    rows: list[dict[str, Any]] = []
    memberships: dict[str, set[str]] = {}
    for order, (stage, applicable, active, mask) in enumerate(
        funnel_stages(matrix, target_config, condition), start=1
    ):
        before = int(current.sum())
        if applicable and active:
            current = current & mask
        after = int(current.sum())
        memberships[stage] = set(
            matrix.loc[current, "canonical_performance_id"].astype(str)
        )
        rows.append(
            {
                "condition_order": condition["condition_order"],
                "filter_condition": condition["filter_condition"],
                "target_key": target_config["target_key"],
                "funnel_order": order,
                "funnel_stage": stage,
                "stage_applicable": bool(applicable),
                "filter_active": bool(applicable and active),
                "rows_entering_stage": before,
                "rows_removed_at_stage": before - after,
                "rows_remaining": after,
            }
        )
    return rows, memberships


def select_ridge(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    raw_features: list[str],
    target_config: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """Ridge-only wrapper around the existing rolling-origin preprocessing/model APIs."""
    preprocessor = rolling.fit_preprocessor(train, raw_features)
    x_train = preprocessor.transform(train)
    x_validation = preprocessor.transform(validation)
    y_train = train[target_config["target_column"]].to_numpy(dtype=float)
    y_validation = validation[target_config["target_column"]].to_numpy(dtype=float)
    candidates: list[dict[str, Any]] = []
    for alpha in rolling.RIDGE_ALPHA_GRID:
        intercept, beta = rolling.fit_ridge(x_train, y_train, float(alpha))
        prediction = rolling.apply_bounds(
            rolling.predict_ridge(x_validation, intercept, beta),
            target_config["clip_lower"],
            target_config["clip_upper"],
        )
        candidates.append(
            {
                "alpha": float(alpha),
                **rolling.metrics(y_validation, prediction),
            }
        )
    selected = min(
        candidates,
        key=lambda row: (float(row["mae"]), float(row["rmse"]), float(row["alpha"])),
    )
    return selected, x_train.shape[1]


def source_timing_safe(frame: pd.DataFrame) -> pd.Series:
    safe = pd.Series(True, index=frame.index)
    for column in [
        "advanced_max_source_season_used",
        "preference_max_source_season",
        "strict_context_source_season",
    ]:
        values = pd.to_numeric(frame[column], errors="coerce")
        safe = safe & (values.isna() | values.lt(frame["season_start_year"]))
    return safe


def score_condition_target(
    matrix: pd.DataFrame,
    target_config: dict[str, Any],
    condition: dict[str, Any],
    eligibility: pd.Series,
    feature_sets: dict[str, list[str]],
    target_index: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    target_key = target_config["target_key"]
    target_column = target_config["target_column"]
    cohort = matrix.loc[eligibility].copy()
    cohort[target_column] = pd.to_numeric(cohort[target_column], errors="coerce")
    cohort = cohort.loc[cohort[target_column].notna()].sort_values(
        ["season_start_year", "canonical_performance_id"], kind="stable"
    )

    prediction_frames: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    origin_rows: list[dict[str, Any]] = []
    pair_frames: list[pd.DataFrame] = []

    for origin_index, validation_year in enumerate(rolling.VALIDATION_YEARS):
        evaluation_year = validation_year + 1
        rolling_origin = rolling.origin_id(validation_year)
        train = cohort.loc[
            cohort["season_start_year"].between(
                rolling.FIRST_TRAIN_YEAR, validation_year - 1
            )
        ].copy()
        validation = cohort.loc[cohort["season_start_year"].eq(validation_year)].copy()
        evaluation = cohort.loc[cohort["season_start_year"].eq(evaluation_year)].copy()
        if min(len(train), len(validation), len(evaluation)) == 0:
            raise RuntimeError(
                f"Empty rolling split: {condition['filter_condition']} {target_key} {rolling_origin}"
            )
        fit_frame = pd.concat([train, validation], ignore_index=True)
        y_fit = fit_frame[target_column].to_numpy(dtype=float)
        y_evaluation = evaluation[target_column].to_numpy(dtype=float)
        evaluation_safe = source_timing_safe(evaluation)

        per_origin_predictions: dict[str, pd.DataFrame] = {}
        for feature_set_name in ["player_history_baseline", "full_explicit_fit"]:
            raw_features = feature_sets[feature_set_name]
            selected, encoded_train = select_ridge(
                train, validation, raw_features, target_config
            )
            final_preprocessor = rolling.fit_preprocessor(fit_frame, raw_features)
            x_fit = final_preprocessor.transform(fit_frame)
            x_evaluation = final_preprocessor.transform(evaluation)
            raw_prediction, prediction = rolling.fit_selected_model(
                "ridge",
                selected,
                x_fit,
                y_fit,
                x_evaluation,
                target_config,
                target_key,
                feature_set_name,
            )
            evaluation_metrics = rolling.metrics(y_evaluation, prediction)
            selected_rows.append(
                {
                    "condition_order": condition["condition_order"],
                    "filter_condition": condition["filter_condition"],
                    "target_key": target_key,
                    "target_column": target_column,
                    "origin_id": rolling_origin,
                    "train_end_year": validation_year - 1,
                    "validation_year": validation_year,
                    "evaluation_year": evaluation_year,
                    "feature_set": feature_set_name,
                    "model_family": "ridge",
                    "selected_alpha": selected["alpha"],
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "evaluation_rows": len(evaluation),
                    "raw_feature_count": len(raw_features),
                    "encoded_feature_count_train": encoded_train,
                    "encoded_feature_count_fit": x_fit.shape[1],
                    "validation_mae": selected["mae"],
                    "validation_rmse": selected["rmse"],
                    "evaluation_mae": evaluation_metrics["mae"],
                    "evaluation_rmse": evaluation_metrics["rmse"],
                }
            )
            prediction_frame = pd.DataFrame(
                {
                    "condition_order": condition["condition_order"],
                    "filter_condition": condition["filter_condition"],
                    "origin_id": rolling_origin,
                    "train_end_year": validation_year - 1,
                    "validation_year": validation_year,
                    "evaluation_year": evaluation_year,
                    "canonical_performance_id": evaluation[
                        "canonical_performance_id"
                    ].astype(str).to_numpy(),
                    "canonical_player_name": evaluation[
                        "canonical_player_name"
                    ].astype(str).to_numpy(),
                    "canonical_club_name": evaluation[
                        "canonical_club_name"
                    ].astype(str).to_numpy(),
                    "season_start_year": evaluation["season_start_year"].to_numpy(),
                    "role_group": evaluation["role_group"].astype(str).to_numpy(),
                    "target_key": target_key,
                    "target_column": target_column,
                    "feature_set": feature_set_name,
                    "model_family": "ridge",
                    "selected_alpha": selected["alpha"],
                    "actual": y_evaluation,
                    "prediction_raw": raw_prediction,
                    "prediction": prediction,
                    "absolute_error": np.abs(y_evaluation - prediction),
                    "advanced_max_source_season_used": evaluation[
                        "advanced_max_source_season_used"
                    ].to_numpy(),
                    "preference_max_source_season": evaluation[
                        "preference_max_source_season"
                    ].to_numpy(),
                    "strict_context_source_season": evaluation[
                        "strict_context_source_season"
                    ].to_numpy(),
                    "source_timing_safe": evaluation_safe.to_numpy(),
                }
            )
            prediction_frames.append(prediction_frame)
            per_origin_predictions[feature_set_name] = prediction_frame

        baseline = per_origin_predictions["player_history_baseline"].sort_values(
            "canonical_performance_id"
        )
        candidate = per_origin_predictions["full_explicit_fit"].sort_values(
            "canonical_performance_id"
        )
        pair = rolling.paired_rows(baseline, candidate, "full_explicit_fit")
        audit_columns = baseline[
            [
                "origin_id",
                "target_key",
                "canonical_performance_id",
                "advanced_max_source_season_used",
                "preference_max_source_season",
                "strict_context_source_season",
                "source_timing_safe",
            ]
        ]
        pair = pair.merge(
            audit_columns,
            on=["origin_id", "target_key", "canonical_performance_id"],
            how="left",
            validate="one_to_one",
        )
        pair.insert(0, "filter_condition", condition["filter_condition"])
        pair.insert(0, "condition_order", condition["condition_order"])
        pair["model_family"] = "ridge"
        pair_frames.append(pair)

        bootstrap = rolling.paired_bootstrap(
            baseline["actual"].to_numpy(dtype=float),
            baseline["prediction"].to_numpy(dtype=float),
            candidate["prediction"].to_numpy(dtype=float),
            rolling.RANDOM_SEED
            + 1000
            + target_index * 100
            + origin_index,
        )
        origin_rows.append(
            {
                "condition_order": condition["condition_order"],
                "filter_condition": condition["filter_condition"],
                "target_key": target_key,
                "origin_id": rolling_origin,
                "validation_year": validation_year,
                "evaluation_year": evaluation_year,
                "model_family": "ridge",
                "comparison": "full_explicit_fit_vs_player_history_baseline",
                "evaluation_n": bootstrap.pop("test_n"),
                "baseline_mae": float(baseline["absolute_error"].mean()),
                "candidate_mae": float(candidate["absolute_error"].mean()),
                **bootstrap,
            }
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    pairs = pd.concat(pair_frames, ignore_index=True)
    origins = pd.DataFrame(origin_rows)
    summary = rolling.summarize_comparison(
        pairs,
        origins,
        rolling.RANDOM_SEED
        + 2000
        + target_index * 10,
    )
    return predictions, pd.DataFrame(selected_rows), origins, {**summary, "pairs": pairs}


def effect_change(delta: float) -> str:
    if delta > EFFECT_UNCHANGED_TOLERANCE:
        return "increased"
    if delta < -EFFECT_UNCHANGED_TOLERANCE:
        return "decreased"
    return "left approximately unchanged"


def ci_change(current_width: float, primary_width: float) -> str:
    if current_width <= primary_width * (1 - MEANINGFUL_CI_WIDTH_CHANGE):
        return "meaningfully tighter"
    if current_width >= primary_width * (1 + MEANINGFUL_CI_WIDTH_CHANGE):
        return "meaningfully wider"
    return "not meaningfully changed"


def build_readme(comparison: pd.DataFrame, origins: pd.DataFrame) -> str:
    lines = [
        "# MoveMaker cohort-filter sensitivity diagnostic",
        "",
        "This diagnostic leaves the frozen primary cohort and every upstream model stage unchanged. It relaxes the arrival window, 450 destination-minute threshold, and outfield-only restriction one at a time, plus all three together. Every comparison uses ridge with `player_history_baseline` versus ridge with `full_explicit_fit` across the same four expanding rolling origins.",
        "",
        "The opportunity target has no primary destination-minute threshold, so `no_destination_minutes_threshold` is expected to reproduce its primary cohort exactly. All-season opportunity results include partial-season arrivals and are diagnostic rather than directly comparable full-season opportunity estimates.",
        "",
        "## Results",
        "",
        "Effect changes within +/-0.001 MAE are described as approximately unchanged. A confidence interval is described as meaningfully tighter or wider when its width changes by at least 10% relative to the primary row.",
        "",
    ]
    for target_key in ["opportunity", "performance", "adaptation"]:
        target_rows = comparison.loc[comparison["target_key"].eq(target_key)].sort_values(
            "condition_order"
        )
        primary = target_rows.loc[
            target_rows["filter_condition"].eq("primary_policy")
        ].iloc[0]
        lines.extend(
            [
                f"### {target_key.title()}",
                "",
                f"Primary ridge compatibility improvement: {primary['pooled_mae_improvement']:.6f}, 95% CI [{primary['mae_improvement_ci_lower_95']:.6f}, {primary['mae_improvement_ci_upper_95']:.6f}], pooled evaluation n={int(primary['pooled_n'])}.",
                "",
            ]
        )
        for row in target_rows.loc[
            ~target_rows["filter_condition"].eq("primary_policy")
        ].itertuples(index=False):
            lines.append(
                f"- `{row.filter_condition}`: eligible cohort {int(row.eligible_cohort_n):,} ({int(row.delta_eligible_cohort_n_vs_primary):+,}); pooled n {int(row.pooled_n):,} ({int(row.delta_pooled_n_vs_primary):+,}); improvement {row.pooled_mae_improvement:.6f}, which {row.effect_change_vs_primary} the point effect; CI [{row.mae_improvement_ci_lower_95:.6f}, {row.mae_improvement_ci_upper_95:.6f}] was {row.ci_width_change_vs_primary}."
            )
        lines.append("")

    def result(condition: str, target: str) -> pd.Series:
        return comparison.loc[
            comparison["filter_condition"].eq(condition)
            & comparison["target_key"].eq(target)
        ].iloc[0]

    opportunity_primary = result("primary_policy", "opportunity")
    opportunity_all_season = result("all_season_arrivals", "opportunity")
    opportunity_all_roles = result("all_roles", "opportunity")
    opportunity_full = result("fully_relaxed", "opportunity")
    performance_primary = result("primary_policy", "performance")
    performance_minutes = result("no_destination_minutes_threshold", "performance")
    performance_full = result("fully_relaxed", "performance")
    adaptation_primary = result("primary_policy", "adaptation")
    adaptation_minutes = result("no_destination_minutes_threshold", "adaptation")
    adaptation_full = result("fully_relaxed", "adaptation")
    performance_all_season_first = origins.loc[
        origins["filter_condition"].eq("all_season_arrivals")
        & origins["target_key"].eq("performance")
        & origins["origin_id"].eq("origin_2019"),
        "mae_improvement",
    ].iloc[0]
    adaptation_all_season_first = origins.loc[
        origins["filter_condition"].eq("all_season_arrivals")
        & origins["target_key"].eq("adaptation")
        & origins["origin_id"].eq("origin_2019"),
        "mae_improvement",
    ].iloc[0]
    adaptation_full_first = origins.loc[
        origins["filter_condition"].eq("fully_relaxed")
        & origins["target_key"].eq("adaptation")
        & origins["origin_id"].eq("origin_2019"),
        "mae_improvement",
    ].iloc[0]
    lines.extend(
        [
            "## Plain pattern",
            "",
            f"- Opportunity does not strengthen with more rows. The primary improvement is {opportunity_primary['pooled_mae_improvement']:.6f}; it falls to {opportunity_all_season['pooled_mae_improvement']:.6f} with all-season arrivals, {opportunity_all_roles['pooled_mae_improvement']:.6f} with all roles, and {opportunity_full['pooled_mae_improvement']:.6f} when fully relaxed. Those broader intervals are tighter because n grows, but they center much closer to zero.",
            f"- Removing only the destination-minute threshold changes performance from {performance_primary['pooled_mae_improvement']:.6f} to {performance_minutes['pooled_mae_improvement']:.6f} and adaptation from {adaptation_primary['pooled_mae_improvement']:.6f} to {adaptation_minutes['pooled_mae_improvement']:.6f}; neither confidence interval tightens meaningfully. That filter is not the main sample-size limiter.",
            f"- The all-season performance and adaptation jumps are instability warnings, not clean strengthening. Their first-origin improvements are {performance_all_season_first:.6f} and {adaptation_all_season_first:.6f}, respectively, while later-origin changes are near zero and the pooled intervals widen. Fully relaxed adaptation reverses sharply because its first-origin improvement is {adaptation_full_first:.6f}.",
            f"- The fully relaxed performance cohort is the largest performance comparison and has a much smaller effect ({performance_full['pooled_mae_improvement']:.6f}) with a tighter interval. Fully relaxed adaptation becomes highly unstable ({adaptation_full['pooled_mae_improvement']:.6f}). Across targets, cohort composition and early-origin ridge extrapolation matter more than raw row count alone.",
            "",
        ]
    )

    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "This is a cohort-sensitivity pattern check, not a replacement cohort and not a proof or disproof of compatibility. Larger samples mix in substantively different transfer timing, playing-time reliability, and goalkeeper populations. Point-effect direction, origin consistency, and confidence-interval width should therefore be read together.",
            "",
            "## Verification",
            "",
            "The separate verifier checks that every relaxed final cohort and every relaxed funnel stage contains its primary counterpart, all declared source hashes remain unchanged, only ridge and the two declared feature sets were scored, and every scored feature source season precedes its destination outcome season.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    SUPPORT_DIR.mkdir(parents=True, exist_ok=True)
    source_paths = [
        MATRIX_PATH,
        TARGETS_PATH,
        EXCLUSION_PATH,
        FEATURE_DICTIONARY_PATH,
        PRIMARY_ROLLING_PATH,
        ROOT / "scripts" / "build_compatibility_targets.py",
        ROOT / "scripts" / "build_compatibility_features.py",
        ROOT / "scripts" / "run_rolling_origin_stability.py",
    ]
    before_hashes = {path: feature_logic.sha256(path) for path in source_paths}

    matrix = prepare_matrix()
    feature_dictionary = pd.read_csv(FEATURE_DICTIONARY_PATH, low_memory=False)
    feature_sets, _ = rolling.feature_sets(feature_dictionary)
    selected_feature_sets = {
        key: feature_sets[key]
        for key in ["player_history_baseline", "full_explicit_fit"]
    }
    forbidden = sorted(
        {
            feature
            for values in selected_feature_sets.values()
            for feature in values
            if feature.startswith(rolling.FORBIDDEN_PREDICTOR_PREFIXES)
        }
    )

    target_configs = {config["target_key"]: config for config in rolling.TARGETS}
    funnel_rows: list[dict[str, Any]] = []
    funnel_memberships: dict[tuple[str, str, str], set[str]] = {}
    membership_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    origin_frames: list[pd.DataFrame] = []
    pair_frames: list[pd.DataFrame] = []
    build_checks: list[dict[str, Any]] = []

    def add_check(check: str, passed: bool, observed: Any, expected: Any, detail: str) -> None:
        build_checks.append(
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
        "no_forbidden_predictors",
        not forbidden,
        ";".join(forbidden),
        "none",
        "Target, eligibility, exclusion, and roster fields cannot enter either model.",
    )
    add_check(
        "only_two_declared_feature_sets",
        set(selected_feature_sets) == {"player_history_baseline", "full_explicit_fit"},
        ";".join(selected_feature_sets),
        "player_history_baseline;full_explicit_fit",
        "The diagnostic does not run the unnecessary five-feature-set grid.",
    )

    for condition in CONDITIONS:
        for target_index, target_key in enumerate(
            ["opportunity", "performance", "adaptation"]
        ):
            target_config = target_configs[target_key]
            eligibility, reasons = policy_eligibility(matrix, target_config, condition)
            funnel, stage_memberships = build_funnel(matrix, target_config, condition)
            funnel_rows.extend(funnel)
            for stage, members in stage_memberships.items():
                funnel_memberships[(condition["filter_condition"], target_key, stage)] = members
            final_funnel_n = funnel[-1]["rows_remaining"]
            add_check(
                f"{condition['filter_condition']}_{target_key}_funnel_matches_eligibility",
                int(eligibility.sum()) == final_funnel_n,
                f"eligibility={int(eligibility.sum())};funnel={final_funnel_n}",
                "equal",
                "Sequential funnel and imported exclusion logic must select identical rows.",
            )
            if condition["filter_condition"] == "primary_policy":
                frozen = target_logic.bool_series(matrix[target_config["eligibility_column"]])
                add_check(
                    f"{target_key}_primary_membership_reproduced",
                    eligibility.equals(frozen),
                    int((eligibility != frozen).sum()),
                    0,
                    "The diagnostic must exactly reproduce the frozen primary eligibility flag.",
                )

            for row in matrix.loc[eligibility, ["canonical_performance_id"]].itertuples(
                index=False
            ):
                membership_rows.append(
                    {
                        "condition_order": condition["condition_order"],
                        "filter_condition": condition["filter_condition"],
                        "target_key": target_key,
                        "canonical_performance_id": row.canonical_performance_id,
                    }
                )

            predictions, selected, origins, summary_bundle = score_condition_target(
                matrix,
                target_config,
                condition,
                eligibility,
                selected_feature_sets,
                target_index,
            )
            pairs = summary_bundle.pop("pairs")
            prediction_frames.append(predictions)
            selected_frames.append(selected)
            origin_frames.append(origins)
            pair_frames.append(pairs)
            comparison_rows.append(
                {
                    "condition_order": condition["condition_order"],
                    "filter_condition": condition["filter_condition"],
                    "condition_description": condition["description"],
                    "target_key": target_key,
                    "model_family": "ridge",
                    "comparison": "full_explicit_fit_vs_player_history_baseline",
                    "relax_arrival_window": condition["relax_arrival_window"],
                    "relax_destination_minutes": condition[
                        "relax_destination_minutes"
                    ],
                    "relax_outfield_role": condition["relax_outfield_role"],
                    "eligible_cohort_n": int(eligibility.sum()),
                    **summary_bundle,
                }
            )

    funnel_frame = pd.DataFrame(funnel_rows)
    membership_frame = pd.DataFrame(membership_rows)
    predictions_frame = pd.concat(prediction_frames, ignore_index=True)
    selected_frame = pd.concat(selected_frames, ignore_index=True)
    origins_frame = pd.concat(origin_frames, ignore_index=True)
    pairs_frame = pd.concat(pair_frames, ignore_index=True)
    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["condition_order", "target_key"], kind="stable"
    )

    for target_key in ["opportunity", "performance", "adaptation"]:
        primary = comparison.loc[
            comparison["target_key"].eq(target_key)
            & comparison["filter_condition"].eq("primary_policy")
        ].iloc[0]
        mask = comparison["target_key"].eq(target_key)
        comparison.loc[mask, "delta_eligible_cohort_n_vs_primary"] = (
            comparison.loc[mask, "eligible_cohort_n"] - primary["eligible_cohort_n"]
        )
        comparison.loc[mask, "delta_pooled_n_vs_primary"] = (
            comparison.loc[mask, "pooled_n"] - primary["pooled_n"]
        )
        comparison.loc[mask, "delta_mae_improvement_vs_primary"] = (
            comparison.loc[mask, "pooled_mae_improvement"]
            - primary["pooled_mae_improvement"]
        )
        comparison.loc[mask, "ci_width"] = (
            comparison.loc[mask, "mae_improvement_ci_upper_95"]
            - comparison.loc[mask, "mae_improvement_ci_lower_95"]
        )
        primary_width = float(
            primary["mae_improvement_ci_upper_95"]
            - primary["mae_improvement_ci_lower_95"]
        )
        comparison.loc[mask, "delta_ci_width_vs_primary"] = (
            comparison.loc[mask, "ci_width"] - primary_width
        )
        comparison.loc[mask, "effect_change_vs_primary"] = comparison.loc[
            mask, "delta_mae_improvement_vs_primary"
        ].map(effect_change)
        comparison.loc[mask, "ci_width_change_vs_primary"] = comparison.loc[
            mask, "ci_width"
        ].map(lambda value: ci_change(float(value), primary_width))

    primary_published = pd.read_csv(PRIMARY_ROLLING_PATH, low_memory=False)
    published_ridge = primary_published.loc[
        primary_published["model_family"].eq("ridge")
        & primary_published["comparison"].eq(
            "full_explicit_fit_vs_player_history_baseline"
        )
    ].set_index("target_key")
    reproduced_primary = comparison.loc[
        comparison["filter_condition"].eq("primary_policy")
    ].set_index("target_key")
    primary_differences = []
    for target_key in ["opportunity", "performance", "adaptation"]:
        primary_differences.append(
            abs(
                float(reproduced_primary.loc[target_key, "pooled_mae_improvement"])
                - float(published_ridge.loc[target_key, "pooled_mae_improvement"])
            )
        )
    add_check(
        "primary_ridge_rolling_effect_reproduced",
        max(primary_differences) < 1e-12,
        max(primary_differences),
        "<1e-12",
        "The diagnostic primary row must reproduce the published rolling ridge result.",
    )

    for condition in CONDITIONS[1:]:
        condition_name = condition["filter_condition"]
        for target_key in ["opportunity", "performance", "adaptation"]:
            primary_members = set(
                membership_frame.loc[
                    membership_frame["filter_condition"].eq("primary_policy")
                    & membership_frame["target_key"].eq(target_key),
                    "canonical_performance_id",
                ]
            )
            relaxed_members = set(
                membership_frame.loc[
                    membership_frame["filter_condition"].eq(condition_name)
                    & membership_frame["target_key"].eq(target_key),
                    "canonical_performance_id",
                ]
            )
            add_check(
                f"{condition_name}_{target_key}_final_superset",
                primary_members.issubset(relaxed_members),
                len(primary_members - relaxed_members),
                0,
                "Every relaxed final cohort must retain every primary-cohort row.",
            )
            for stage in [row[0] for row in funnel_stages(matrix, target_configs[target_key], condition)]:
                primary_stage = funnel_memberships[("primary_policy", target_key, stage)]
                relaxed_stage = funnel_memberships[(condition_name, target_key, stage)]
                add_check(
                    f"{condition_name}_{target_key}_{stage}_stage_superset",
                    primary_stage.issubset(relaxed_stage),
                    len(primary_stage - relaxed_stage),
                    0,
                    "The relaxed funnel must contain the primary funnel at the same stage.",
                )

    add_check(
        "only_ridge_scored",
        set(predictions_frame["model_family"]) == {"ridge"},
        ";".join(sorted(predictions_frame["model_family"].unique())),
        "ridge",
        "No elastic-net or boosted-tree models are run in this diagnostic.",
    )
    add_check(
        "only_declared_feature_sets_scored",
        set(predictions_frame["feature_set"])
        == {"player_history_baseline", "full_explicit_fit"},
        ";".join(sorted(predictions_frame["feature_set"].unique())),
        "full_explicit_fit;player_history_baseline",
        "Only the requested baseline and full compatibility designs are evaluated.",
    )
    add_check(
        "all_scored_rows_preseason_safe",
        bool(target_logic.bool_series(predictions_frame["source_timing_safe"]).all()),
        int((~target_logic.bool_series(predictions_frame["source_timing_safe"])).sum()),
        0,
        "Every recorded source season must precede the destination outcome season.",
    )

    after_hashes = {path: feature_logic.sha256(path) for path in source_paths}
    source_manifest = pd.DataFrame(
        [
            {
                "source_file": path.relative_to(ROOT).as_posix(),
                "file_bytes": path.stat().st_size,
                "before_sha256": before_hashes[path],
                "after_sha256": after_hashes[path],
                "unchanged": before_hashes[path] == after_hashes[path],
            }
            for path in source_paths
        ]
    )
    add_check(
        "all_source_files_unchanged_during_run",
        bool(source_manifest["unchanged"].all()),
        int((~source_manifest["unchanged"]).sum()),
        0,
        "Hash equality proves the diagnostic did not modify any declared source file.",
    )

    checks_frame = pd.DataFrame(build_checks)
    failures = int((~checks_frame["passed"]).sum())
    write_csv(comparison, COMPARISON_PATH)
    write_csv(funnel_frame, FUNNEL_PATH)
    write_csv(membership_frame, MEMBERSHIP_PATH)
    write_csv(origins_frame, ORIGIN_RESULTS_PATH)
    write_csv(selected_frame, SELECTED_SPECS_PATH)
    write_csv(pairs_frame, PAIR_PREDICTIONS_PATH)
    write_csv(source_manifest, SOURCE_MANIFEST_PATH)
    write_csv(checks_frame, BUILD_CHECKS_PATH)
    README_PATH.write_text(build_readme(comparison, origins_frame), encoding="utf-8")
    run_summary = {
        "status": "pass" if failures == 0 else "fail",
        "conditions": len(CONDITIONS),
        "targets": len(rolling.TARGETS),
        "comparison_rows": len(comparison),
        "blocking_checks": len(checks_frame),
        "blocking_failures": failures,
    }
    RUN_SUMMARY_PATH.write_text(
        json.dumps(run_summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

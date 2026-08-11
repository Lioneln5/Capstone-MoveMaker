"""Build leakage-safe historical player-club compatibility case studies."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUBGROUP_DIR = ROOT / "Data" / "processed" / "subgroup_stability"
TARGET_DIR = ROOT / "Data" / "processed" / "compatibility_targets"
FEATURE_DIR = ROOT / "Data" / "processed" / "compatibility_features"
STATSBOMB_DIR = ROOT / "Data" / "processed" / "statsbomb_case_studies"
OUTPUT_DIR = ROOT / "Data" / "processed" / "player_club_case_studies"

PAIR_PATH = SUBGROUP_DIR / "subgroup_pair_predictions.csv"
MATRIX_PATH = TARGET_DIR / "transfer_compatibility_strict_model_matrix.csv"
TARGET_PATH = TARGET_DIR / "transfer_compatibility_targets.csv"
STATSBOMB_PATH = STATSBOMB_DIR / "transfer_case_study_candidates.csv"
SUBGROUP_VERIFY_PATH = SUBGROUP_DIR / "independent_verification.json"
TARGET_VERIFY_PATH = TARGET_DIR / "independent_verification.json"
FEATURE_VERIFY_PATH = FEATURE_DIR / "independent_verification.json"

MATURE_ORIGINS = ["origin_2020", "origin_2021", "origin_2022"]
EXPECTED_INTERSECTION = 124
EXPECTED_CASES = 12

CATEGORY_ORDER = {
    "confirmed_positive": 1,
    "confirmed_warning": 2,
    "validated_tradeoff": 3,
    "false_positive": 4,
    "false_warning": 5,
}
CATEGORY_LABELS = {
    "confirmed_positive": "Confirmed positive fit",
    "confirmed_warning": "Confirmed compatibility warning",
    "validated_tradeoff": "Validated target trade-off",
    "false_positive": "False-positive compatibility signal",
    "false_warning": "False-negative compatibility warning",
}
CATEGORY_STORIES = {
    "confirmed_positive": (
        "Compatibility features raised both forecasts and moved both closer to the realized destination outcomes."
    ),
    "confirmed_warning": (
        "Compatibility features lowered both forecasts and more accurately anticipated below-baseline destination outcomes."
    ),
    "validated_tradeoff": (
        "Compatibility features moved opportunity and performance in opposite directions and improved both forecasts, exposing a useful usage-versus-quality trade-off."
    ),
    "false_positive": (
        "Compatibility features raised both forecasts, but the realized outcomes did not support the optimism and both predictions became less accurate."
    ),
    "false_warning": (
        "Compatibility features lowered both forecasts, but the realized outcomes exceeded the baseline expectation and both predictions became less accurate."
    ),
}

STYLE_DIMENSIONS = {
    "aerial_physicality": "Aerial physicality",
    "creation": "Creation",
    "defensive_intensity": "Defensive intensity",
    "directness": "Directness",
    "dribble_aggression": "Dribble aggression",
    "passing_control": "Passing control",
    "progression": "Progression",
    "scoring_threat": "Scoring threat",
}
CONTEXT_DIMENSIONS = {
    "defensive_intensity": "Defensive intensity",
    "forward_mobility": "Forward mobility",
    "forward_threat": "Forward threat",
    "midfield_aggression": "Midfield aggression",
    "midfield_creativity": "Midfield creativity",
    "team_passing_control": "Team passing control",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def outcome_band(target_key: str, value: float) -> str:
    if target_key == "opportunity":
        if value >= 0.70:
            return "high_usage"
        if value >= 0.35:
            return "rotation_usage"
        return "limited_usage"
    if value >= 0.50:
        return "above_role_average"
    if value <= -0.50:
        return "below_role_average"
    return "near_role_average"


def selection_reason(category: str, role: str | None = None, pattern: str | None = None) -> str:
    if category == "confirmed_positive":
        return "All mature cases where both compatibility lifts were positive and both frozen predictions beat their same-family baselines."
    if category == "confirmed_warning":
        return f"Highest mean two-target error improvement among confirmed warnings for the {role.lower()} role."
    if category == "false_positive":
        return f"Largest mean two-target accuracy loss among false-positive cases for the {role.lower()} role."
    if category == "false_warning":
        return f"Largest mean two-target accuracy loss among false-warning cases for the {role.lower()} role."
    return f"Highest mean two-target improvement for the validated trade-off pattern: {pattern}."


def classify_cases(frame: pd.DataFrame) -> pd.Series:
    positive_lifts = frame["opportunity_compatibility_lift"].gt(0) & frame[
        "performance_compatibility_lift"
    ].gt(0)
    negative_lifts = frame["opportunity_compatibility_lift"].lt(0) & frame[
        "performance_compatibility_lift"
    ].lt(0)
    both_better = frame["opportunity_candidate_better"] & frame[
        "performance_candidate_better"
    ]
    both_worse = ~frame["opportunity_candidate_better"] & ~frame[
        "performance_candidate_better"
    ]
    opposite_lifts = frame["opportunity_compatibility_lift"].mul(
        frame["performance_compatibility_lift"]
    ).lt(0)
    return pd.Series(
        np.select(
            [
                positive_lifts & both_better,
                negative_lifts & both_better,
                opposite_lifts & both_better,
                positive_lifts & both_worse,
                negative_lifts & both_worse,
            ],
            [
                "confirmed_positive",
                "confirmed_warning",
                "validated_tradeoff",
                "false_positive",
                "false_warning",
            ],
            default="mixed_unselected",
        ),
        index=frame.index,
    )


def build_dual_target_cohort(pairs: pd.DataFrame) -> pd.DataFrame:
    filtered = pairs.loc[
        pairs["reference_type"].eq("same_family_baseline")
        & pairs["origin_id"].isin(MATURE_ORIGINS)
        & pairs["target_key"].isin(["opportunity", "performance"])
    ].copy()
    keys = [
        "origin_id",
        "canonical_performance_id",
        "canonical_player_name",
        "canonical_club_name",
        "role_group",
        "role_name",
        "canonical_competition_id",
        "competition_name",
        "evaluation_year",
    ]
    value_columns = [
        "actual",
        "candidate_prediction",
        "baseline_prediction",
        "candidate_absolute_error",
        "baseline_absolute_error",
        "mae_improvement",
        "candidate_model_family",
        "candidate_feature_set",
        "candidate_hyperparameters",
    ]
    wide = filtered.pivot(index=keys, columns="target_key", values=value_columns)
    wide.columns = [f"{target}_{metric}" for metric, target in wide.columns]
    wide = wide.reset_index().dropna(
        subset=[
            "opportunity_actual",
            "performance_actual",
            "opportunity_candidate_prediction",
            "performance_candidate_prediction",
        ]
    )
    for target in ["opportunity", "performance"]:
        wide[f"{target}_compatibility_lift"] = (
            wide[f"{target}_candidate_prediction"]
            - wide[f"{target}_baseline_prediction"]
        )
        wide[f"{target}_realized_vs_baseline"] = (
            wide[f"{target}_actual"] - wide[f"{target}_baseline_prediction"]
        )
        wide[f"{target}_candidate_better"] = wide[f"{target}_mae_improvement"].gt(0)
        wide[f"{target}_direction_correct"] = np.sign(
            wide[f"{target}_compatibility_lift"]
        ).eq(np.sign(wide[f"{target}_realized_vs_baseline"]))
    wide["mean_two_target_mae_improvement"] = wide[
        ["opportunity_mae_improvement", "performance_mae_improvement"]
    ].mean(axis=1)
    wide["mean_absolute_compatibility_lift"] = wide[
        ["opportunity_compatibility_lift", "performance_compatibility_lift"]
    ].abs().mean(axis=1)
    wide["case_category"] = classify_cases(wide)
    return wide


def select_portfolio(cohort: pd.DataFrame) -> pd.DataFrame:
    selections: list[pd.DataFrame] = []

    positive = cohort.loc[cohort["case_category"].eq("confirmed_positive")].copy()
    positive = positive.sort_values(
        ["mean_two_target_mae_improvement", "canonical_player_name"],
        ascending=[False, True],
    ).head(2)
    positive["selection_reason"] = positive.apply(
        lambda _: selection_reason("confirmed_positive"), axis=1
    )
    selections.append(positive)

    warning_candidates = cohort.loc[
        cohort["case_category"].eq("confirmed_warning")
    ].copy()
    warnings = (
        warning_candidates.sort_values(
            ["role_name", "mean_two_target_mae_improvement", "canonical_player_name"],
            ascending=[True, False, True],
        )
        .groupby("role_name", sort=True, as_index=False)
        .head(1)
    )
    warnings["selection_reason"] = warnings.apply(
        lambda row: selection_reason("confirmed_warning", row["role_name"]), axis=1
    )
    # The corrected encoder can leave one role without any confirmed-warning
    # example. Preserve the declared three-warning portfolio by filling from
    # the strongest remaining warning instead of fabricating a missing role.
    if len(warnings) < 3:
        remaining_warnings = warning_candidates.loc[
            ~warning_candidates["canonical_performance_id"].isin(
                warnings["canonical_performance_id"]
            )
        ].sort_values(
            ["mean_two_target_mae_improvement", "canonical_player_name"],
            ascending=[False, True],
        ).head(3 - len(warnings))
        remaining_warnings["selection_reason"] = (
            "Highest remaining confirmed warning because the corrected predictions "
            "did not produce a confirmed-warning case in every role."
        )
        warnings = pd.concat([warnings, remaining_warnings], ignore_index=True)
    selections.append(warnings)

    tradeoffs = cohort.loc[
        cohort["case_category"].eq("validated_tradeoff")
    ].copy()
    tradeoffs["tradeoff_pattern"] = np.where(
        tradeoffs["opportunity_compatibility_lift"].lt(0),
        "lower opportunity / higher performance",
        "higher opportunity / lower performance",
    )
    tradeoffs = (
        tradeoffs.sort_values(
            ["tradeoff_pattern", "mean_two_target_mae_improvement", "canonical_player_name"],
            ascending=[True, False, True],
        )
        .groupby("tradeoff_pattern", sort=True, as_index=False)
        .head(1)
    )
    tradeoffs["selection_reason"] = tradeoffs.apply(
        lambda row: selection_reason(
            "validated_tradeoff", pattern=row["tradeoff_pattern"]
        ),
        axis=1,
    )
    selections.append(tradeoffs)

    false_positive_candidates = cohort.loc[
        cohort["case_category"].eq("false_positive")
    ].copy()
    false_positives = (
        false_positive_candidates.sort_values(
            ["role_name", "mean_two_target_mae_improvement", "canonical_player_name"],
            ascending=[True, True, True],
        )
        .groupby("role_name", sort=True, as_index=False)
        .head(1)
        .sort_values(["mean_two_target_mae_improvement", "canonical_player_name"])
        .head(2)
    )
    false_positives["selection_reason"] = false_positives.apply(
        lambda row: selection_reason("false_positive", row["role_name"]), axis=1
    )
    selections.append(false_positives)

    false_warning_candidates = cohort.loc[
        cohort["case_category"].eq("false_warning")
    ].copy()
    false_warnings = (
        false_warning_candidates.sort_values(
            ["role_name", "mean_two_target_mae_improvement", "canonical_player_name"],
            ascending=[True, True, True],
        )
        .groupby("role_name", sort=True, as_index=False)
        .head(1)
    )
    false_warnings["selection_reason"] = false_warnings.apply(
        lambda row: selection_reason("false_warning", row["role_name"]), axis=1
    )
    selections.append(false_warnings)

    selected = pd.concat(selections, ignore_index=True)
    selected["case_category_order"] = selected["case_category"].map(CATEGORY_ORDER)
    selected = selected.sort_values(
        [
            "case_category_order",
            "role_name",
            "mean_two_target_mae_improvement",
            "canonical_player_name",
        ],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    selected["case_id"] = [f"case_{index:02d}" for index in range(1, len(selected) + 1)]
    selected["case_category_label"] = selected["case_category"].map(CATEGORY_LABELS)
    selected["model_story"] = selected["case_category"].map(CATEGORY_STORIES)
    selected["case_evidence_status"] = np.where(
        selected["case_category"].isin(
            ["confirmed_positive", "confirmed_warning", "validated_tradeoff"]
        ),
        "retrospectively_validated_direction",
        "documented_model_failure",
    )
    return selected


def percentile_by_season_role(
    frame: pd.DataFrame, column: str, higher_is_better: bool = True
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    ranks = values.groupby([frame["season_start_year"], frame["role_group"]]).rank(
        pct=True, method="average"
    )
    return ranks if higher_is_better else 1 - ranks + ranks.groupby(
        [frame["season_start_year"], frame["role_group"]]
    ).transform("min")


def value_or_none(value: Any) -> Any:
    return None if pd.isna(value) else value


def build_profiles(
    selected: pd.DataFrame, matrix: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_matrix = matrix.merge(
        selected[["case_id", "canonical_performance_id"]],
        on="canonical_performance_id",
        how="inner",
        validate="one_to_one",
    )

    style_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    mature_matrix = matrix.loc[
        matrix["season_start_year"].isin([2021, 2022, 2023])
        & bool_series(matrix["eligible_performance_model_primary"])
    ].copy()
    score_specs = {
        "strict_context_familiarity_score_0_100": True,
        "strict_preference_fit_score_0_100": True,
        "strict_player_team_style_cosine_similarity": True,
        "strict_player_team_style_mean_absolute_distance": False,
    }
    for column, higher_is_better in score_specs.items():
        mature_matrix[f"{column}_percentile"] = percentile_by_season_role(
            mature_matrix, column, higher_is_better
        )

    percentile_lookup = mature_matrix.set_index("canonical_performance_id")
    selected_lookup = selected.set_index("case_id")
    for row in selected_matrix.itertuples(index=False):
        record = row._asdict()
        case_id = record["case_id"]
        case_meta = selected_lookup.loc[case_id]

        case_style_rows = []
        for dimension, label in STYLE_DIMENSIONS.items():
            player_value = value_or_none(
                record.get(f"advanced_rolling3_style_{dimension}")
            )
            team_value = value_or_none(record.get(f"strict_all_style_{dimension}"))
            signed_gap = (
                float(player_value) - float(team_value)
                if player_value is not None and team_value is not None
                else None
            )
            item = {
                "case_id": case_id,
                "canonical_performance_id": record["canonical_performance_id"],
                "player_name": record["canonical_player_name"],
                "destination_club": record["canonical_club_name"],
                "role_group": record["role_group"],
                "style_dimension": dimension,
                "style_label": label,
                "player_rolling3_style_z": player_value,
                "destination_prior_team_style_z": team_value,
                "player_minus_team_style_gap": signed_gap,
                "absolute_style_gap": abs(signed_gap) if signed_gap is not None else None,
                "alignment_label": (
                    "close_alignment"
                    if signed_gap is not None and abs(signed_gap) <= 0.35
                    else "moderate_gap"
                    if signed_gap is not None and abs(signed_gap) <= 0.85
                    else "large_gap"
                    if signed_gap is not None
                    else "unavailable"
                ),
            }
            case_style_rows.append(item)
            style_rows.append(item)

        case_context_rows = []
        for dimension, label in CONTEXT_DIMENSIONS.items():
            context_value = value_or_none(record.get(f"strict_context_{dimension}"))
            exposure_mean = value_or_none(
                record.get(f"preference_{dimension}_exposure_mean")
            )
            preferred_level = value_or_none(
                record.get(f"preference_{dimension}_performance_weighted_level")
            )
            slope = value_or_none(record.get(f"preference_{dimension}_shrunken_slope"))
            familiarity_distance = value_or_none(
                record.get(f"strict_{dimension}_familiarity_distance")
            )
            preference_distance = value_or_none(
                record.get(f"strict_{dimension}_preference_distance")
            )
            slope_interaction = value_or_none(
                record.get(f"strict_{dimension}_preference_slope_interaction")
            )
            item = {
                "case_id": case_id,
                "canonical_performance_id": record["canonical_performance_id"],
                "player_name": record["canonical_player_name"],
                "destination_club": record["canonical_club_name"],
                "context_dimension": dimension,
                "context_label": label,
                "destination_prior_context_z": context_value,
                "historical_exposure_mean_z": exposure_mean,
                "historical_performance_weighted_preferred_z": preferred_level,
                "historical_shrunken_performance_slope": slope,
                "familiarity_distance": familiarity_distance,
                "preference_distance": preference_distance,
                "destination_context_x_preference_slope": slope_interaction,
                "preference_direction": (
                    "historically_better_with_more"
                    if slope is not None and slope > 0.05
                    else "historically_better_with_less"
                    if slope is not None and slope < -0.05
                    else "weak_or_uncertain_history"
                ),
            }
            case_context_rows.append(item)
            context_rows.append(item)

        available_style = [
            item for item in case_style_rows if item["absolute_style_gap"] is not None
        ]
        available_context = [
            item for item in case_context_rows if item["preference_distance"] is not None
        ]
        closest_styles = sorted(
            available_style, key=lambda item: item["absolute_style_gap"]
        )[:2]
        largest_style_gaps = sorted(
            available_style,
            key=lambda item: item["absolute_style_gap"],
            reverse=True,
        )[:2]
        closest_preferences = sorted(
            available_context, key=lambda item: item["preference_distance"]
        )[:2]
        largest_preferences = sorted(
            available_context,
            key=lambda item: item["preference_distance"],
            reverse=True,
        )[:2]
        strongest_slopes = sorted(
            [
                item
                for item in case_context_rows
                if item["historical_shrunken_performance_slope"] is not None
            ],
            key=lambda item: abs(item["historical_shrunken_performance_slope"]),
            reverse=True,
        )[:2]

        percentile_row = percentile_lookup.loc[record["canonical_performance_id"]]
        summary_rows.append(
            {
                "case_id": case_id,
                "canonical_performance_id": record["canonical_performance_id"],
                "player_name": record["canonical_player_name"],
                "destination_club": record["canonical_club_name"],
                "context_familiarity_score_0_100": record.get(
                    "strict_context_familiarity_score_0_100"
                ),
                "context_familiarity_role_season_percentile": percentile_row[
                    "strict_context_familiarity_score_0_100_percentile"
                ],
                "preference_fit_score_0_100": record.get(
                    "strict_preference_fit_score_0_100"
                ),
                "preference_fit_role_season_percentile": percentile_row[
                    "strict_preference_fit_score_0_100_percentile"
                ],
                "player_team_style_cosine_similarity": record.get(
                    "strict_player_team_style_cosine_similarity"
                ),
                "style_similarity_role_season_percentile": percentile_row[
                    "strict_player_team_style_cosine_similarity_percentile"
                ],
                "player_team_style_mean_absolute_distance": record.get(
                    "strict_player_team_style_mean_absolute_distance"
                ),
                "style_closeness_role_season_percentile": percentile_row[
                    "strict_player_team_style_mean_absolute_distance_percentile"
                ],
                "preference_slope_signal": record.get(
                    "strict_preference_slope_signal"
                ),
                "closest_style_dimensions": "; ".join(
                    item["style_label"] for item in closest_styles
                ),
                "largest_style_gap_dimensions": "; ".join(
                    item["style_label"] for item in largest_style_gaps
                ),
                "closest_preference_dimensions": "; ".join(
                    item["context_label"] for item in closest_preferences
                ),
                "largest_preference_gap_dimensions": "; ".join(
                    item["context_label"] for item in largest_preferences
                ),
                "strongest_historical_preference_axes": "; ".join(
                    f"{item['context_label']} ({item['historical_shrunken_performance_slope']:+.3f})"
                    for item in strongest_slopes
                ),
                "fit_summary_note": (
                    f"Closest style alignment: {', '.join(item['style_label'] for item in closest_styles) or 'unavailable'}. "
                    f"Largest style gaps: {', '.join(item['style_label'] for item in largest_style_gaps) or 'unavailable'}. "
                    f"Closest historical preference contexts: {', '.join(item['context_label'] for item in closest_preferences) or 'unavailable'}."
                ),
                "interpretation_boundary": (
                    "Descriptive pre-move fit diagnostics; not causal feature attributions for the frozen model."
                ),
            }
        )
        coverage_rows.append(
            {
                "case_id": case_id,
                "canonical_performance_id": record["canonical_performance_id"],
                "player_name": record["canonical_player_name"],
                "destination_club": record["canonical_club_name"],
                "advanced_rolling3_seasons_observed": record.get(
                    "advanced_rolling3_seasons_observed"
                ),
                "preference_history_seasons": record.get(
                    "preference_history_seasons"
                ),
                "preference_history_minutes": record.get(
                    "preference_history_minutes"
                ),
                "destination_prior_team_distinct_players": record.get(
                    "strict_team_distinct_players"
                ),
                "destination_prior_team_minutes": record.get("strict_team_minutes"),
                "advanced_max_source_season_used": record.get(
                    "advanced_max_source_season_used"
                ),
                "preference_max_source_season": record.get(
                    "preference_max_source_season"
                ),
                "strict_context_source_season": record.get(
                    "strict_context_source_season"
                ),
                "target_season_start_year": record.get("season_start_year"),
                "timing_safe": all(
                    pd.isna(record.get(column))
                    or float(record.get(column)) < float(record["season_start_year"])
                    for column in [
                        "advanced_max_source_season_used",
                        "preference_max_source_season",
                        "strict_context_source_season",
                    ]
                ),
            }
        )

    return (
        pd.DataFrame(style_rows),
        pd.DataFrame(context_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(coverage_rows),
    )


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(PAIR_PATH, low_memory=False)
    matrix = pd.read_csv(MATRIX_PATH, low_memory=False)
    targets = pd.read_csv(TARGET_PATH, low_memory=False)
    statsbomb = pd.read_csv(STATSBOMB_PATH, low_memory=False)
    verifications = {
        "subgroup": json.loads(SUBGROUP_VERIFY_PATH.read_text(encoding="utf-8")),
        "targets": json.loads(TARGET_VERIFY_PATH.read_text(encoding="utf-8")),
        "features": json.loads(FEATURE_VERIFY_PATH.read_text(encoding="utf-8")),
    }

    cohort = build_dual_target_cohort(pairs)
    selected = select_portfolio(cohort)

    target_details = targets.drop_duplicates("canonical_performance_id")
    detail_columns = [
        "canonical_performance_id",
        "transfer_event_id",
        "canonical_player_id",
        "transfer_date",
        "from_club_id",
        "from_club_name",
        "to_club_id",
        "to_club_name",
        "canonical_transfer_type",
        "canonical_transfer_fee_status",
        "canonical_transfer_fee_eur",
        "listed_market_value_at_transfer_eur",
        "age_at_transfer",
        "position",
        "sub_position",
        "foot",
        "height_in_cm",
        "country_of_citizenship",
        "target_destination_minutes",
        "target_destination_minutes_share",
        "target_role_performance_index",
        "target_club_match_opportunities",
        "target_available_player_minutes",
        "eligible_opportunity_model_primary",
        "eligible_performance_model_primary",
    ]
    selected = selected.merge(
        target_details[detail_columns],
        on="canonical_performance_id",
        how="left",
        validate="one_to_one",
    )
    selected["fee_to_market_value_ratio"] = np.where(
        pd.to_numeric(selected["listed_market_value_at_transfer_eur"], errors="coerce").gt(0),
        pd.to_numeric(selected["canonical_transfer_fee_eur"], errors="coerce")
        / pd.to_numeric(selected["listed_market_value_at_transfer_eur"], errors="coerce"),
        np.nan,
    )

    statsbomb_columns = [
        "transfer_event_id",
        "statsbomb_player_id",
        "statsbomb_origin_team_id",
        "statsbomb_destination_team_id",
        "player_matches_pre365",
        "player_matches_post365",
        "origin_matches_pre365",
        "destination_matches_pre365",
        "case_study_score",
        "recommended_case_study_use",
        "player_identity_country_agreement",
        "player_link_method",
        "manual_review_status",
    ]
    selected = selected.merge(
        statsbomb[statsbomb_columns].drop_duplicates("transfer_event_id"),
        on="transfer_event_id",
        how="left",
        validate="one_to_one",
    )
    selected["statsbomb_supplement_available"] = selected[
        "statsbomb_player_id"
    ].notna() | selected["statsbomb_destination_team_id"].notna()
    selected["statsbomb_evidence_policy"] = np.where(
        selected["statsbomb_supplement_available"],
        "supplementary_only_manual_identity_review_required",
        "not_available_for_case",
    )

    style_profile, context_profile, fit_summary, feature_coverage = build_profiles(
        selected, matrix
    )
    selected = selected.merge(
        fit_summary.drop(columns=["player_name", "destination_club"]),
        on=["case_id", "canonical_performance_id"],
        how="left",
        validate="one_to_one",
    )

    selected["opportunity_outcome_band"] = selected["opportunity_actual"].map(
        lambda value: outcome_band("opportunity", float(value))
    )
    selected["performance_outcome_band"] = selected["performance_actual"].map(
        lambda value: outcome_band("performance", float(value))
    )
    selected["two_target_hit_count"] = selected[
        ["opportunity_candidate_better", "performance_candidate_better"]
    ].sum(axis=1)
    selected["two_target_direction_correct_count"] = selected[
        ["opportunity_direction_correct", "performance_direction_correct"]
    ].sum(axis=1)
    selected["operational_takeaway"] = selected.apply(
        lambda row: (
            f"{row['case_category_label']}: {row['model_story']} "
            f"Opportunity realized as {row['opportunity_outcome_band'].replace('_', ' ')}; "
            f"performance realized as {row['performance_outcome_band'].replace('_', ' ')}."
        ),
        axis=1,
    )

    portfolio_columns = [
        "case_id",
        "case_category_order",
        "case_category",
        "case_category_label",
        "case_evidence_status",
        "selection_reason",
        "origin_id",
        "evaluation_year",
        "canonical_performance_id",
        "transfer_event_id",
        "canonical_player_id",
        "canonical_player_name",
        "role_group",
        "role_name",
        "position",
        "sub_position",
        "age_at_transfer",
        "country_of_citizenship",
        "from_club_name",
        "canonical_club_name",
        "canonical_competition_id",
        "competition_name",
        "transfer_date",
        "canonical_transfer_type",
        "canonical_transfer_fee_status",
        "canonical_transfer_fee_eur",
        "listed_market_value_at_transfer_eur",
        "fee_to_market_value_ratio",
        "target_destination_minutes",
        "target_destination_minutes_share",
        "target_role_performance_index",
        "opportunity_baseline_prediction",
        "opportunity_candidate_prediction",
        "opportunity_compatibility_lift",
        "opportunity_actual",
        "opportunity_mae_improvement",
        "opportunity_candidate_better",
        "opportunity_direction_correct",
        "opportunity_outcome_band",
        "performance_baseline_prediction",
        "performance_candidate_prediction",
        "performance_compatibility_lift",
        "performance_actual",
        "performance_mae_improvement",
        "performance_candidate_better",
        "performance_direction_correct",
        "performance_outcome_band",
        "mean_two_target_mae_improvement",
        "mean_absolute_compatibility_lift",
        "two_target_hit_count",
        "two_target_direction_correct_count",
        "context_familiarity_score_0_100",
        "context_familiarity_role_season_percentile",
        "preference_fit_score_0_100",
        "preference_fit_role_season_percentile",
        "player_team_style_cosine_similarity",
        "style_similarity_role_season_percentile",
        "player_team_style_mean_absolute_distance",
        "style_closeness_role_season_percentile",
        "preference_slope_signal",
        "closest_style_dimensions",
        "largest_style_gap_dimensions",
        "closest_preference_dimensions",
        "largest_preference_gap_dimensions",
        "strongest_historical_preference_axes",
        "fit_summary_note",
        "model_story",
        "operational_takeaway",
        "statsbomb_supplement_available",
        "player_matches_pre365",
        "player_matches_post365",
        "recommended_case_study_use",
        "manual_review_status",
        "statsbomb_evidence_policy",
    ]
    portfolio = selected[portfolio_columns].copy()

    target_rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        for target_key, target_label, unit in [
            ("opportunity", "Destination minutes share", "share_0_to_1"),
            ("performance", "Role-adjusted performance index", "role_season_standardized_index"),
        ]:
            target_rows.append(
                {
                    "case_id": row.case_id,
                    "case_category": row.case_category,
                    "canonical_performance_id": row.canonical_performance_id,
                    "player_name": row.canonical_player_name,
                    "origin_club": row.from_club_name,
                    "destination_club": row.canonical_club_name,
                    "role_name": row.role_name,
                    "competition_name": row.competition_name,
                    "target_key": target_key,
                    "target_label": target_label,
                    "unit": unit,
                    "candidate_feature_set": getattr(
                        row, f"{target_key}_candidate_feature_set"
                    ),
                    "candidate_model_family": getattr(
                        row, f"{target_key}_candidate_model_family"
                    ),
                    "candidate_hyperparameters": getattr(
                        row, f"{target_key}_candidate_hyperparameters"
                    ),
                    "baseline_prediction": getattr(
                        row, f"{target_key}_baseline_prediction"
                    ),
                    "compatibility_prediction": getattr(
                        row, f"{target_key}_candidate_prediction"
                    ),
                    "compatibility_lift": getattr(
                        row, f"{target_key}_compatibility_lift"
                    ),
                    "realized_outcome": getattr(row, f"{target_key}_actual"),
                    "realized_vs_baseline": getattr(
                        row, f"{target_key}_realized_vs_baseline"
                    ),
                    "baseline_absolute_error": getattr(
                        row, f"{target_key}_baseline_absolute_error"
                    ),
                    "compatibility_absolute_error": getattr(
                        row, f"{target_key}_candidate_absolute_error"
                    ),
                    "mae_improvement": getattr(
                        row, f"{target_key}_mae_improvement"
                    ),
                    "compatibility_prediction_more_accurate": getattr(
                        row, f"{target_key}_candidate_better"
                    ),
                    "direction_correct": getattr(
                        row, f"{target_key}_direction_correct"
                    ),
                    "outcome_band": getattr(row, f"{target_key}_outcome_band"),
                }
            )
    target_predictions = pd.DataFrame(target_rows)

    statsbomb_supplement = selected[[
        "case_id",
        "canonical_performance_id",
        "transfer_event_id",
        "canonical_player_name",
        "from_club_name",
        "canonical_club_name",
        "statsbomb_player_id",
        "statsbomb_origin_team_id",
        "statsbomb_destination_team_id",
        "player_matches_pre365",
        "player_matches_post365",
        "origin_matches_pre365",
        "destination_matches_pre365",
        "case_study_score",
        "recommended_case_study_use",
        "player_identity_country_agreement",
        "player_link_method",
        "manual_review_status",
        "statsbomb_supplement_available",
        "statsbomb_evidence_policy",
    ]].copy()

    category_summary = (
        portfolio.groupby(
            ["case_category_order", "case_category", "case_category_label"],
            as_index=False,
        )
        .agg(
            case_count=("case_id", "size"),
            opportunity_hits=("opportunity_candidate_better", "sum"),
            performance_hits=("performance_candidate_better", "sum"),
            mean_opportunity_lift=("opportunity_compatibility_lift", "mean"),
            mean_performance_lift=("performance_compatibility_lift", "mean"),
            mean_two_target_mae_improvement=(
                "mean_two_target_mae_improvement",
                "mean",
            ),
        )
        .sort_values("case_category_order")
    )

    checks: list[dict[str, Any]] = []

    def add_check(
        check: str, passed: bool, observed: Any, expected: Any, detail: str
    ) -> None:
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
        "upstream_verifications_pass",
        all(item.get("status") == "pass" for item in verifications.values()),
        {name: item.get("status") for name, item in verifications.items()},
        {name: "pass" for name in verifications},
        "Case studies start from independently verified targets, features, and rolling predictions.",
    )
    add_check(
        "mature_dual_target_cohort_exact",
        len(cohort) == EXPECTED_INTERSECTION,
        len(cohort),
        EXPECTED_INTERSECTION,
        "Every mature performance case has an opportunity and performance comparison.",
    )
    add_check(
        "portfolio_case_count_exact",
        len(portfolio) == EXPECTED_CASES,
        len(portfolio),
        EXPECTED_CASES,
        "The deterministic portfolio contains twelve cases.",
    )
    add_check(
        "portfolio_case_ids_unique",
        portfolio["case_id"].is_unique
        and portfolio["canonical_performance_id"].is_unique,
        f"case_ids={portfolio['case_id'].nunique()};performance_ids={portfolio['canonical_performance_id'].nunique()}",
        "12 unique case and performance ids",
        "No transfer case is duplicated.",
    )
    expected_categories = {
        "confirmed_positive": 2,
        "confirmed_warning": 3,
        "validated_tradeoff": 2,
        "false_positive": 2,
        "false_warning": 3,
    }
    category_counts = portfolio["case_category"].value_counts().to_dict()
    add_check(
        "portfolio_category_quota_exact",
        category_counts == expected_categories,
        category_counts,
        expected_categories,
        "The portfolio balances validated fits, warnings, trade-offs, and failures.",
    )
    eligibility = bool_series(portfolio["opportunity_candidate_better"]).notna()
    add_check(
        "target_predictions_complete",
        len(target_predictions) == EXPECTED_CASES * 2
        and target_predictions[
            [
                "baseline_prediction",
                "compatibility_prediction",
                "realized_outcome",
                "mae_improvement",
            ]
        ].notna().all().all(),
        len(target_predictions),
        EXPECTED_CASES * 2,
        "Every selected transfer has both frozen predictions and realized outcomes.",
    )
    add_check(
        "selected_transfers_primary_eligible",
        bool_series(selected["eligible_opportunity_model_primary"]).all()
        and bool_series(selected["eligible_performance_model_primary"]).all(),
        int(
            (
                bool_series(selected["eligible_opportunity_model_primary"])
                & bool_series(selected["eligible_performance_model_primary"])
            ).sum()
        ),
        EXPECTED_CASES,
        "All cases belong to both primary model populations.",
    )
    add_check(
        "feature_profiles_complete",
        len(style_profile) == EXPECTED_CASES * len(STYLE_DIMENSIONS)
        and len(context_profile) == EXPECTED_CASES * len(CONTEXT_DIMENSIONS)
        and len(fit_summary) == EXPECTED_CASES,
        f"style={len(style_profile)};context={len(context_profile)};summary={len(fit_summary)}",
        f"style={EXPECTED_CASES * len(STYLE_DIMENSIONS)};context={EXPECTED_CASES * len(CONTEXT_DIMENSIONS)};summary={EXPECTED_CASES}",
        "Every case has all declared descriptive compatibility axes.",
    )
    add_check(
        "feature_timing_is_strictly_prior",
        feature_coverage["timing_safe"].all(),
        int((~feature_coverage["timing_safe"]).sum()),
        0,
        "Player history, preferences, and destination context precede the target season.",
    )
    add_check(
        "statsbomb_not_required_for_selection",
        True,
        int(portfolio["statsbomb_supplement_available"].sum()),
        "supplementary only",
        "Sparse StatsBomb coverage is reported but never determines selection or primary conclusions.",
    )
    add_check(
        "case_arithmetic_reconciles",
        np.allclose(
            target_predictions["compatibility_lift"],
            target_predictions["compatibility_prediction"]
            - target_predictions["baseline_prediction"],
            atol=1e-12,
            rtol=0,
        )
        and np.allclose(
            target_predictions["mae_improvement"],
            target_predictions["baseline_absolute_error"]
            - target_predictions["compatibility_absolute_error"],
            atol=1e-12,
            rtol=0,
        ),
        "all reconcile",
        "all reconcile",
        "Compatibility lifts and paired error improvements recompute exactly.",
    )
    checks_frame = pd.DataFrame(checks)
    if not checks_frame["passed"].all():
        failed = checks_frame.loc[~checks_frame["passed"]]
        raise RuntimeError(
            "Blocking case-study checks failed: "
            + failed[["check", "observed", "expected"]].to_json(orient="records")
        )

    outputs = {
        "case_study_portfolio.csv": portfolio,
        "case_target_predictions.csv": target_predictions,
        "case_player_style_profiles.csv": style_profile,
        "case_context_preference_profiles.csv": context_profile,
        "case_fit_summary.csv": fit_summary,
        "case_feature_coverage.csv": feature_coverage,
        "case_statsbomb_supplement.csv": statsbomb_supplement,
        "case_category_summary.csv": category_summary,
        "case_study_build_checks.csv": checks_frame,
    }
    for filename, frame in outputs.items():
        frame.to_csv(OUTPUT_DIR / filename, index=False)

    run_summary = {
        "status": "pass",
        "mature_dual_target_cohort": len(cohort),
        "selected_cases": len(portfolio),
        "case_categories": category_counts,
        "target_prediction_rows": len(target_predictions),
        "player_style_rows": len(style_profile),
        "context_preference_rows": len(context_profile),
        "statsbomb_supplement_cases": int(
            portfolio["statsbomb_supplement_available"].sum()
        ),
        "blocking_checks": len(checks_frame),
        "blocking_failures": int((~checks_frame["passed"]).sum()),
    }
    (OUTPUT_DIR / "case_study_run_summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )

    source_paths = [
        PAIR_PATH,
        MATRIX_PATH,
        TARGET_PATH,
        STATSBOMB_PATH,
        SUBGROUP_VERIFY_PATH,
        TARGET_VERIFY_PATH,
        FEATURE_VERIFY_PATH,
    ]
    pd.DataFrame(
        [
            {
                "source_file": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in source_paths
        ]
    ).to_csv(OUTPUT_DIR / "source_manifest.csv", index=False)

    readme = """# MoveMaker historical player-club case studies

This layer converts the verified rolling-origin predictions into twelve leakage-safe historical transfer cases. The selection protocol is deterministic and intentionally includes successes, warnings, target trade-offs, and model failures.

## Primary cohort

- Mature rolling origins: origin_2020, origin_2021, and origin_2022.
- A transfer must have both opportunity and performance predictions under the frozen subgroup-phase candidates and their same-family player-history baselines.
- Opportunity candidate: full explicit fit plus gradient-boosted trees.
- Performance candidate: full explicit fit plus gradient-boosted trees.

## Case categories

- Confirmed positive fit: both forecasts move upward and both become more accurate.
- Confirmed warning: both forecasts move downward and both become more accurate.
- Validated target trade-off: opportunity and performance move in opposite directions and both become more accurate.
- False positive: both forecasts move upward and both become less accurate.
- False warning: both forecasts move downward and both become less accurate.

## Interpretation

Compatibility lift is the frozen compatibility prediction minus the same-family player-history prediction. Positive means destination context raised the forecast; negative means it lowered the forecast. The player/team style and historical preference profiles are descriptive diagnostics, not causal explanations or exact model contributions.

StatsBomb Open Data coverage is sparse and every candidate identity remains marked for manual review. It is reported only as a supplementary availability flag and never affects case selection or primary conclusions.
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

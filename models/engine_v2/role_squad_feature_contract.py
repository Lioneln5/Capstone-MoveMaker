"""Decision-time feature contract for player role and squad competition."""

from __future__ import annotations

from typing import Final


CONTRACT_VERSION: Final[str] = "engine_v2_role_squad_v1_2026-09-01"
LOOKBACK_DAYS: Final[int] = 365
RECENT_DAYS: Final[int] = 180
PRIOR_DAYS: Final[int] = LOOKBACK_DAYS - RECENT_DAYS
MIN_CLUB_GAMES: Final[int] = 10
MIN_SPLIT_GAMES: Final[int] = 5
MIN_SOURCE_GAME_COVERAGE: Final[float] = 0.80
MIN_ACTIVE_SELECTIONS: Final[int] = 3
MIN_KNOWN_POSITION_VALUES: Final[int] = 3
MAX_VALUATION_AGE_DAYS: Final[int] = 365

ROLE_LEVEL_FEATURES: Final[list[str]] = [
    "role_lineup_selection_rate_365d",
    "role_start_rate_365d",
    "role_substitute_selection_rate_365d",
    "role_captain_rate_365d",
    "role_position_entropy_365d",
    "role_days_since_last_selection_scaled_365d",
]

ROLE_TRAJECTORY_FEATURES: Final[list[str]] = [
    "role_recent_start_rate_180d",
    "role_start_rate_change_recent_vs_prior",
    "role_selection_rate_change_recent_vs_prior",
    "role_opportunity_share_change_recent_vs_prior",
]

SQUAD_ONFIELD_FEATURES: Final[list[str]] = [
    "squad_same_position_competitor_count_log1p",
    "squad_same_position_strong_competitor_count_log1p",
    "player_same_position_start_percentile",
    "player_same_position_minutes_percentile",
    "player_start_rate_gap_to_position_leader",
    "player_opportunity_gap_to_position_leader",
    "player_share_of_position_minutes",
    "squad_position_minutes_hhi",
]

SQUAD_VALUE_INCOMING_FEATURES: Final[list[str]] = [
    "squad_position_known_value_players_log1p",
    "player_same_position_value_percentile",
    "player_value_to_position_median_log",
    "incoming_same_position_players_180d_log1p",
    "incoming_same_position_higher_value_players_180d_log1p",
    "incoming_same_position_max_value_ratio_log",
]

MODEL_FEATURES: Final[list[str]] = (
    ROLE_LEVEL_FEATURES
    + ROLE_TRAJECTORY_FEATURES
    + SQUAD_ONFIELD_FEATURES
    + SQUAD_VALUE_INCOMING_FEATURES
)

FEATURE_BLOCKS: Final[dict[str, list[str]]] = {
    "R1_current_role_structure": ROLE_LEVEL_FEATURES,
    "R2_role_trajectory": ROLE_TRAJECTORY_FEATURES,
    "S1_onfield_position_competition": SQUAD_ONFIELD_FEATURES,
    "S2_value_and_incoming_competition": SQUAD_VALUE_INCOMING_FEATURES,
    "C1_combined_role_and_competition": MODEL_FEATURES,
}


def contract_payload() -> dict[str, object]:
    return {
        "contract_version": CONTRACT_VERSION,
        "windows": {
            "full": "[signed_date - 365 days, signed_date)",
            "prior": "[signed_date - 365 days, signed_date - 180 days)",
            "recent": "[signed_date - 180 days, signed_date)",
            "incoming": "[signed_date - 180 days, signed_date)",
        },
        "availability": {
            "minimum_club_games_full_window": MIN_CLUB_GAMES,
            "minimum_club_games_each_trajectory_split": MIN_SPLIT_GAMES,
            "minimum_lineup_or_appearance_game_coverage": MIN_SOURCE_GAME_COVERAGE,
            "minimum_selections_for_active_competitor": MIN_ACTIVE_SELECTIONS,
            "minimum_known_values_in_position_group": MIN_KNOWN_POSITION_VALUES,
            "maximum_predecision_valuation_age_days": MAX_VALUATION_AGE_DAYS,
        },
        "blocks": FEATURE_BLOCKS,
        "zero_semantics": {
            "player_not_selected": "Zero role rate is allowed only when club lineup coverage passes the source gate.",
            "no_competitors": "Zero competitors is allowed only after the same club/source gate passes.",
            "no_incoming_competitor": "Zero incoming players is a valid count from the canonical transfer history.",
            "no_known_incoming_value": "Incoming maximum-value ratio is zero when no incoming player has a known value; the companion known-value count preserves that distinction.",
            "position_entropy": "Set to zero for fewer than two observed positions; selection rate distinguishes no role from a stable role.",
        },
        "timing": {
            "lineups_appearances_games": "strictly before signing",
            "incoming_transfers": "strictly before signing",
            "valuations": "on or before signing and no more than 365 days old",
        },
        "final_holdout_policy": "2024+ outcomes remain sealed during development.",
        "deployment_status": "research_only_not_deployed",
    }

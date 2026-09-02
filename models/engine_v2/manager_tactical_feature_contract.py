"""Decision-time feature contract for manager and tactical stability."""

from __future__ import annotations

from typing import Final


CONTRACT_VERSION: Final[str] = "engine_v2_manager_tactical_v1_2026-09-01"
LOOKBACK_DAYS: Final[int] = 365
RECENT_DAYS: Final[int] = 180
MIN_CLUB_GAMES: Final[int] = 10
MIN_SPLIT_GAMES: Final[int] = 5
MIN_CURRENT_MANAGER_GAMES: Final[int] = 5
MIN_TRANSITION_SIDE_GAMES: Final[int] = 5
MIN_SOURCE_GAME_COVERAGE: Final[float] = 0.80

MANAGER_STABILITY_FEATURES: Final[list[str]] = [
    "manager_unique_count_365d_log1p",
    "manager_change_count_365d_log1p",
    "current_manager_observed_match_share_365d",
    "current_manager_spell_games_365d_log1p",
    "current_manager_spell_days_scaled_365d",
    "manager_change_within_90d",
]

FORMATION_STABILITY_FEATURES: Final[list[str]] = [
    "formation_unique_shape_count_365d_log1p",
    "formation_shape_entropy_365d",
    "formation_shape_change_rate_365d",
    "dominant_formation_shape_share_365d",
    "current_formation_shape_spell_share_365d",
    "formation_distribution_shift_recent_vs_prior",
]

CURRENT_MANAGER_PLAYER_ROLE_FEATURES: Final[list[str]] = [
    "current_manager_player_selection_rate",
    "current_manager_player_start_rate",
    "current_manager_player_opportunity_share",
    "current_manager_player_captain_rate",
]

MANAGER_TRANSITION_ROLE_CHANGE_FEATURES: Final[list[str]] = [
    "manager_transition_observed_365d",
    "days_since_manager_transition_scaled_365d",
    "player_selection_rate_change_post_vs_pre_manager",
    "player_start_rate_change_post_vs_pre_manager",
    "player_opportunity_share_change_post_vs_pre_manager",
]

MODEL_FEATURES: Final[list[str]] = (
    MANAGER_STABILITY_FEATURES
    + FORMATION_STABILITY_FEATURES
    + CURRENT_MANAGER_PLAYER_ROLE_FEATURES
    + MANAGER_TRANSITION_ROLE_CHANGE_FEATURES
)

FEATURE_BLOCKS: Final[dict[str, list[str]]] = {
    "M1_manager_stability": MANAGER_STABILITY_FEATURES,
    "T1_formation_tactical_stability": FORMATION_STABILITY_FEATURES,
    "M2_player_under_current_manager": CURRENT_MANAGER_PLAYER_ROLE_FEATURES,
    "M3_role_change_at_manager_transition": MANAGER_TRANSITION_ROLE_CHANGE_FEATURES,
    "C1_combined_manager_tactical": MODEL_FEATURES,
}


def contract_payload() -> dict[str, object]:
    return {
        "contract_version": CONTRACT_VERSION,
        "windows": {
            "full": "[signed_date - 365 days, signed_date)",
            "prior": "[signed_date - 365 days, signed_date - 180 days)",
            "recent": "[signed_date - 180 days, signed_date)",
            "recent_manager_change": "[signed_date - 90 days, signed_date)",
        },
        "availability": {
            "minimum_club_games_full_window": MIN_CLUB_GAMES,
            "minimum_games_each_formation_split": MIN_SPLIT_GAMES,
            "minimum_current_manager_games_for_player_rates": MIN_CURRENT_MANAGER_GAMES,
            "minimum_games_each_side_of_manager_transition": MIN_TRANSITION_SIDE_GAMES,
            "minimum_manager_lineup_appearance_or_formation_game_coverage": MIN_SOURCE_GAME_COVERAGE,
        },
        "blocks": FEATURE_BLOCKS,
        "definitions": {
            "manager_identity": "Unicode-normalized manager name recorded for each club-game.",
            "current_manager": "Manager in the latest manager-coded club game strictly before signing.",
            "observed_manager_spell": "Consecutive manager-coded games at the end of the prior-year window with the current manager; not an official employment tenure.",
            "manager_transition": "Latest change in adjacent manager-coded club games during the prior-year window.",
            "formation_shape": "Leading numeric formation shape; textual modifiers such as Attacking, Defending, flat, Diamond, or double 6 are removed.",
            "no_manager_transition": "Transition indicator is zero, days-since is one, and role-change fields are structural zero when manager coverage passes and no change is observed.",
        },
        "zero_semantics": {
            "player_not_selected": "Zero player rate is accepted only when current-manager lineup and appearance game coverage passes the source gate.",
            "no_manager_change": "Zero role-change fields mean no observed transition, distinguished by manager_transition_observed_365d.",
            "stable_formation": "Zero formation change or distribution shift is valid only after formation coverage passes.",
        },
        "timing": {
            "games_lineups_appearances": "strictly before signing",
            "future_outcomes": "never read by profile construction",
        },
        "interpretation_boundary": {
            "manager_features": "Observed match-record stability, not appointment-contract tenure or causal manager quality.",
            "formation_features": "Reported base-shape stability, not full tactical intent or in-match tactical changes.",
            "player_manager_features": "Observed role under a manager, not a causal manager preference effect.",
        },
        "final_holdout_policy": "2024+ outcomes remain sealed during development.",
        "deployment_status": "research_only_not_deployed",
    }

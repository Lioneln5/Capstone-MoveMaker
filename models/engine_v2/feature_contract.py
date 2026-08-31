"""Endpoint and feature policy for the Engine V2 research candidates.

The contract distinguishes three concepts that V1 sometimes mixed together:

* a historical predictor observed before the extension;
* a user-controlled proposal used to describe a financial scenario; and
* a post-extension outcome used only as a target.

Passing a predictive ablation is not enough to override this contract.  A
feature must also have the same meaning at historical training time and at
scenario-scoring time.
"""

from __future__ import annotations

from dataclasses import dataclass


PLAYER_CONTEXT = (
    "age_at_signing",
    "canonical_position",
    "league",
)
PUBLIC_VALUE_CONTEXT = ("log_market_value_at_signing",)
RECENT_INVOLVEMENT = (
    "pre365_same_club_all_competition_opportunity_share",
    "pre365_same_club_all_competition_appearance_rate",
)
PRIOR_VOLUME = (
    "pre1_canonical_minutes_played",
    "pre2_canonical_minutes_played",
)
RAW_SCORING_RATES = (
    "pre365_goals_per90",
    "pre365_assists_per90",
    "pre1_canonical_goals_per90",
    "pre1_canonical_assists_per90",
    "pre2_canonical_goals_per90",
    "pre2_canonical_assists_per90",
)
ADVANCED_PERFORMANCE = (
    "pre1_fbref_expected_goals",
    "pre1_fbref_progressive_carries",
    "pre1_fbref_progressive_passes",
    "pre1_fbref_pass_completion_pct",
)
COMMERCIAL_SCENARIO = (
    "exact_duration_years",
    "age_at_expiration",
    "log_annual_gross_eur",
    "log_fixed_wage_commitment_eur",
    "club_salary_percentile",
    "club_salary_share_known",
    "salary_change_from_prior_pct",
    "annual_wage_to_market_value",
    "commitment_to_market_value",
)
HISTORICAL_SALARY_CONTEXT = (
    "log_prior_season_annual_gross_eur",
    "prior_wage_available",
)

FEATURE_GROUPS = {
    "player_context": PLAYER_CONTEXT,
    "public_value_context": PUBLIC_VALUE_CONTEXT,
    "recent_involvement": RECENT_INVOLVEMENT,
    "prior_volume": PRIOR_VOLUME,
    "raw_scoring_rates": RAW_SCORING_RATES,
    "advanced_performance": ADVANCED_PERFORMANCE,
    "commercial_scenario": COMMERCIAL_SCENARIO,
    "historical_salary_context": HISTORICAL_SALARY_CONTEXT,
}

CORE_PROFILE_FEATURES = list(PLAYER_CONTEXT + PUBLIC_VALUE_CONTEXT)
V2_CORE_FEATURES = list(PLAYER_CONTEXT + PUBLIC_VALUE_CONTEXT + RECENT_INVOLVEMENT)
V2_HISTORY_FEATURES = list(
    PLAYER_CONTEXT + PUBLIC_VALUE_CONTEXT + RECENT_INVOLVEMENT + PRIOR_VOLUME
)


@dataclass(frozen=True)
class EndpointContract:
    module: str
    endpoint: str
    horizon: str
    target: str
    public_status: str
    model_form: str
    allowed_feature_groups: tuple[str, ...]
    prohibited_feature_groups: tuple[str, ...]
    interpretation: str


HEADLINE_ENDPOINTS = (
    EndpointContract(
        module="future_role",
        endpoint="sustained_meaningful_contribution",
        horizon="year_1_and_year_2",
        target="at_least_25pct_same_club_opportunity_share_in_each_year",
        public_status="phase2_candidate_only",
        model_form="binary_probability",
        allowed_feature_groups=("player_context", "public_value_context", "recent_involvement", "prior_volume"),
        prohibited_feature_groups=("raw_scoring_rates", "advanced_performance", "commercial_scenario", "historical_salary_context"),
        interpretation="Historical probability of sustained same-club involvement conditional on an extension scenario.",
    ),
    EndpointContract(
        module="club_continuity",
        endpoint="any_outbound_24m",
        horizon="24m",
        target="any_recorded_loan_or_permanent_outbound_move",
        public_status="phase2_candidate_only",
        model_form="direct_fixed_horizon_binary_probability",
        allowed_feature_groups=("player_context", "public_value_context", "recent_involvement", "prior_volume"),
        prohibited_feature_groups=("raw_scoring_rates", "advanced_performance", "commercial_scenario", "historical_salary_context"),
        interpretation="Historical probability of uninterrupted club involvement for 24 months; outbound includes loans.",
    ),
    EndpointContract(
        module="public_value_downside",
        endpoint="downside_25pct_24m",
        horizon="24m",
        target="public_market_value_decline_of_at_least_25pct",
        public_status="phase2_candidate_only",
        model_form="binary_probability",
        allowed_feature_groups=("player_context", "public_value_context", "recent_involvement", "prior_volume"),
        prohibited_feature_groups=("raw_scoring_rates", "advanced_performance", "commercial_scenario", "historical_salary_context"),
        interpretation="Historical probability of a public-value estimate falling by at least 25%; not realized loss.",
    ),
    EndpointContract(
        module="wage_benchmark",
        endpoint="annual_wage_peer_benchmark",
        horizon="proposal",
        target="historical_annual_fixed_wage",
        public_status="retained_for_later_v2_validation",
        model_form="peer_estimate_plus_proposal_comparison",
        allowed_feature_groups=("player_context", "public_value_context", "historical_salary_context"),
        prohibited_feature_groups=("raw_scoring_rates", "advanced_performance", "commercial_scenario"),
        interpretation="Compare a proposal with a historical peer estimate; never label it optimal or fair value.",
    ),
)


def feature_group(feature: str) -> str:
    """Return the single declared group for a known feature."""

    matches = [group for group, members in FEATURE_GROUPS.items() if feature in members]
    if len(matches) != 1:
        raise ValueError(f"Feature must map to exactly one group: {feature!r} -> {matches}")
    return matches[0]


def contract_for(module: str) -> EndpointContract:
    matches = [item for item in HEADLINE_ENDPOINTS if item.module == module]
    if len(matches) != 1:
        raise KeyError(module)
    return matches[0]


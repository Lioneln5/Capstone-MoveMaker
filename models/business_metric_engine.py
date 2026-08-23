"""Pure business-metric calculator for MoveMaker extension profiles.

The engine does not fit or invoke predictive models. It consumes a proposed
incumbent-club extension, optional backend context, and already-produced model
outputs, then returns calculations, benchmarks, qualified proxies, and paired
scenarios with explicit evidence classes and warnings.
"""

from __future__ import annotations

import bisect
import math
from typing import Any


SCOPE = "incumbent_club_extension"
METRIC_VERSION = "business_metrics_v1_2026-08-12"


def _finite_number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    if positive and number <= 0:
        raise ValueError(f"{field} must be positive")
    return number


def _optional_number(value: Any, field: str, *, positive: bool = False) -> float | None:
    if value is None:
        return None
    return _finite_number(value, field, positive=positive)


def _probability(value: Any, field: str) -> float | None:
    number = _optional_number(value, field)
    if number is not None and not 0 <= number <= 1:
        raise ValueError(f"{field} must be between 0 and 1")
    return number


def _metric(
    key: str,
    label: str,
    value: Any,
    unit: str,
    evidence_class: str,
    status: str,
    interpretation: str,
    warning: str,
    components: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "metric_key": key,
        "display_label": label,
        "value": value,
        "unit": unit,
        "evidence_class": evidence_class,
        "status": status,
        "interpretation": interpretation,
        "warning": warning,
        "components": components or {},
    }


def _unavailable(key: str, label: str, reason: str) -> dict[str, Any]:
    return _metric(
        key, label, None, "unavailable", "unavailable", "unavailable",
        "Metric not calculated.", reason,
    )


def offer_percentile(residual_log: float, sorted_reference_residuals: list[float]) -> float:
    """Empirical CDF percentile using the frozen out-of-time reference."""
    if not sorted_reference_residuals:
        raise ValueError("offer-aggressiveness reference distribution is empty")
    if any(not math.isfinite(float(value)) for value in sorted_reference_residuals):
        raise ValueError("offer-aggressiveness reference contains non-finite values")
    if any(left > right for left, right in zip(sorted_reference_residuals, sorted_reference_residuals[1:])):
        raise ValueError("offer-aggressiveness reference must be sorted ascending")
    # Tiny tolerance makes the CDF stable when the same commitment is rebuilt
    # from rounded wage and date-derived duration fields after CSV round-trip.
    return bisect.bisect_right(sorted_reference_residuals, residual_log + 1e-12) / len(sorted_reference_residuals)


def build_extension_business_profile(
    payload: dict[str, Any],
    sorted_offer_reference_residuals: list[float],
) -> dict[str, Any]:
    """Create the supported MoveMaker incumbent-extension business profile."""
    scope = payload.get("profile_scope")
    if scope != SCOPE:
        raise ValueError(f"profile_scope must be {SCOPE!r}; acquisition and betting modes are unsupported")

    wage = _finite_number(payload.get("proposed_annual_fixed_wage_eur"), "proposed_annual_fixed_wage_eur", positive=True)
    years = _finite_number(payload.get("proposed_contract_years"), "proposed_contract_years", positive=True)
    age = _finite_number(payload.get("age_at_decision"), "age_at_decision")
    market = _finite_number(payload.get("current_public_market_value_eur"), "current_public_market_value_eur", positive=True)
    if not 15 <= age <= 50:
        raise ValueError("age_at_decision must be between 15 and 50 years")
    # Historical exact durations are date-derived and can exceed a nominal
    # ten-year term by a few weeks, so retain a narrow tolerance.
    if years > 10.5:
        raise ValueError("proposed_contract_years must not exceed 10.5 date-derived years")

    prior_wage = _optional_number(payload.get("prior_annual_fixed_wage_eur"), "prior_annual_fixed_wage_eur", positive=True)
    peer_estimate = _optional_number(payload.get("peer_fixed_commitment_estimate_eur"), "peer_fixed_commitment_estimate_eur", positive=True)
    peer_lower = _optional_number(payload.get("peer_fixed_commitment_lower_eur"), "peer_fixed_commitment_lower_eur", positive=True)
    peer_upper = _optional_number(payload.get("peer_fixed_commitment_upper_eur"), "peer_fixed_commitment_upper_eur", positive=True)
    contribution_probability = _probability(payload.get("sustained_contribution_probability_24m"), "sustained_contribution_probability_24m")
    stay_24 = _probability(payload.get("continuous_stay_probability_24m"), "continuous_stay_probability_24m")
    stay_36 = _probability(payload.get("continuous_stay_probability_36m"), "continuous_stay_probability_36m")
    permanent_relationship = _probability(payload.get("permanent_relationship_probability_24m"), "permanent_relationship_probability_24m")
    value_10_12m = _probability(payload.get("public_value_downside_10pct_probability_12m"), "public_value_downside_10pct_probability_12m")
    value_25_12m = _probability(payload.get("public_value_downside_25pct_probability_12m"), "public_value_downside_25pct_probability_12m")
    value_10_24m = _probability(payload.get("public_value_downside_10pct_probability_24m"), "public_value_downside_10pct_probability_24m")
    value_25 = _probability(payload.get("public_value_downside_25pct_probability_24m"), "public_value_downside_25pct_probability_24m")
    value_50 = _probability(payload.get("public_value_downside_50pct_probability_24m"), "public_value_downside_50pct_probability_24m")
    peer_duration_estimate = _optional_number(payload.get("peer_contract_duration_estimate_years"), "peer_contract_duration_estimate_years", positive=True)
    peer_duration_lower = _optional_number(payload.get("peer_contract_duration_lower_years"), "peer_contract_duration_lower_years", positive=True)
    peer_duration_upper = _optional_number(payload.get("peer_contract_duration_upper_years"), "peer_contract_duration_upper_years", positive=True)
    peer_wage_value_estimate = _optional_number(payload.get("peer_wage_to_market_value_estimate"), "peer_wage_to_market_value_estimate", positive=True)
    peer_wage_value_lower = _optional_number(payload.get("peer_wage_to_market_value_lower"), "peer_wage_to_market_value_lower", positive=True)
    peer_wage_value_upper = _optional_number(payload.get("peer_wage_to_market_value_upper"), "peer_wage_to_market_value_upper", positive=True)
    salary_percentile = _probability(payload.get("club_salary_percentile"), "club_salary_percentile")
    payroll_share = _probability(payload.get("club_known_payroll_share"), "club_known_payroll_share")
    salary_panel = payload.get("club_known_annual_fixed_wages_excluding_player_eur")
    salary_context_source = "precomputed_backend_context"
    salary_panel_rows: int | None = None
    if salary_panel is not None:
        if not isinstance(salary_panel, list) or not salary_panel:
            raise ValueError("club_known_annual_fixed_wages_excluding_player_eur must be a non-empty list when provided")
        known_other_wages = [
            _finite_number(value, "club_known_annual_fixed_wages_excluding_player_eur[]", positive=True)
            for value in salary_panel
        ]
        less = sum(value < wage for value in known_other_wages)
        equal = sum(math.isclose(value, wage, rel_tol=1e-12, abs_tol=1e-8) for value in known_other_wages)
        salary_panel_rows = len(known_other_wages) + 1
        salary_percentile = (less + (equal + 2) / 2) / salary_panel_rows
        payroll_share = wage / (sum(known_other_wages) + wage)
        salary_context_source = "calculated_from_matched_panel_excluding_player"

    if peer_lower is not None and peer_upper is not None and peer_lower > peer_upper:
        raise ValueError("peer_fixed_commitment_lower_eur cannot exceed peer_fixed_commitment_upper_eur")
    if value_25 is not None and value_50 is not None and value_50 > value_25 + 1e-12:
        raise ValueError("50% public-value downside probability cannot exceed the 25% probability")
    if value_10_24m is not None and value_25 is not None and value_25 > value_10_24m + 1e-12:
        raise ValueError("25% public-value downside probability cannot exceed the 10% probability")
    if value_10_12m is not None and value_25_12m is not None and value_25_12m > value_10_12m + 1e-12:
        raise ValueError("12-month 25% public-value downside probability cannot exceed the 10% probability")
    if stay_24 is not None and stay_36 is not None and stay_36 > stay_24 + 1e-12:
        raise ValueError("36-month continuous-stay probability cannot exceed the 24-month probability")
    if permanent_relationship is not None and stay_24 is not None and permanent_relationship < stay_24 - 1e-12:
        # A loan interrupts continuous stay but not the permanent relationship,
        # so "permanent relationship intact" is the weaker (>=) condition.
        raise ValueError("permanent-relationship probability cannot be lower than the continuous-stay probability")
    if peer_duration_lower is not None and peer_duration_upper is not None and peer_duration_lower > peer_duration_upper:
        raise ValueError("peer_contract_duration_lower_years cannot exceed peer_contract_duration_upper_years")
    if peer_wage_value_lower is not None and peer_wage_value_upper is not None and peer_wage_value_lower > peer_wage_value_upper:
        raise ValueError("peer_wage_to_market_value_lower cannot exceed peer_wage_to_market_value_upper")

    fixed_commitment = wage * years
    commitment_to_value = fixed_commitment / market
    age_at_expiration = age + years
    first_24m_commitment = wage * min(years, 2)
    post_24m_commitment = wage * max(years - 2, 0)
    post_36m_commitment = wage * max(years - 3, 0)

    cards: dict[str, list[dict[str, Any]]] = {
        "capital_commitment": [],
        "offer_context": [],
        "contribution_exposure": [],
        "asset_value": [],
        "continuity": [],
        "salary_structure": [],
    }

    cards["capital_commitment"].extend([
        _metric(
            "fixed_wage_commitment_eur", "Fixed-wage commitment", fixed_commitment, "EUR",
            "exact_calculation", "supported",
            "Total fixed wages scheduled by the proposed term.",
            "Partial commitment only; excludes bonuses, taxes, agent fees, options, clauses, and other costs.",
            {"annual_fixed_wage_eur": wage, "contract_years": years},
        ),
        _metric(
            "fixed_commitment_to_public_value", "Commitment / public value", commitment_to_value, "ratio",
            "exact_ratio_proxy", "supported_proxy",
            "Scale of proposed fixed wages relative to the dated public market-value estimate.",
            "Not ROI, accounting exposure, book value, or expected loss.",
            {"fixed_wage_commitment_eur": fixed_commitment, "current_public_market_value_eur": market},
        ),
        _metric(
            "age_at_expiration", "Age at expiration", age_at_expiration, "years",
            "exact_calculation", "supported",
            "Player age when the proposed fixed term ends.",
            "Does not incorporate an unentered unilateral or conditional option.",
            {"age_at_decision": age, "contract_years": years},
        ),
    ])
    if prior_wage is None:
        cards["capital_commitment"].append(_unavailable(
            "proposed_wage_change_pct", "Proposed wage increase", "Prior annual fixed wage is unavailable; it was not imputed as zero.",
        ))
    else:
        cards["capital_commitment"].append(_metric(
            "proposed_wage_change_pct", "Proposed wage increase", wage / prior_wage - 1, "ratio",
            "exact_calculation", "supported_when_prior_wage_known",
            "Change from the known prior annual fixed wage.",
            "Public wage figures may be estimates and do not include unreported compensation.",
            {"proposed_annual_fixed_wage_eur": wage, "prior_annual_fixed_wage_eur": prior_wage},
        ))

    if peer_estimate is None:
        cards["offer_context"].extend([
            _unavailable("fixed_commitment_peer_difference_eur", "Commitment versus historical peers", "Serialized Phase-4 peer estimate is unavailable."),
            _unavailable("offer_aggressiveness_percentile", "Offer aggressiveness percentile", "Serialized Phase-4 peer estimate is unavailable."),
        ])
    else:
        residual = math.log(fixed_commitment / peer_estimate)
        percentile = offer_percentile(residual, sorted_offer_reference_residuals)
        if peer_lower is None or peer_upper is None:
            range_position = "unavailable"
        elif fixed_commitment < peer_lower:
            range_position = "below_historical_80pct_range"
        elif fixed_commitment > peer_upper:
            range_position = "above_historical_80pct_range"
        else:
            range_position = "within_historical_80pct_range"
        cards["offer_context"].extend([
            _metric(
                "fixed_commitment_peer_difference_eur", "Commitment versus historical peers",
                fixed_commitment - peer_estimate, "EUR", "historical_benchmark_translation", "supported",
                "Euro difference from the modeled historical peer commitment.",
                "Historical deviation is not proof of overpayment or an optimal offer.",
                {
                    "proposed_fixed_commitment_eur": fixed_commitment,
                    "peer_estimate_eur": peer_estimate,
                    "peer_lower_eur": peer_lower,
                    "peer_upper_eur": peer_upper,
                    "range_position": range_position,
                },
            ),
            _metric(
                "offer_aggressiveness_percentile", "Offer aggressiveness percentile", percentile, "percentile",
                "historical_oos_reference", "supported_with_frozen_reference",
                "Position of the proposal's peer-adjusted commitment in the frozen out-of-time extension distribution.",
                "Not fairness, optimality, expected return, or a recommendation.",
                {"peer_adjusted_log_residual": residual, "reference_rows": len(sorted_offer_reference_residuals)},
            ),
        ])

    if peer_duration_estimate is None:
        cards["offer_context"].append(_unavailable(
            "contract_duration_peer_delta_years", "Term versus historical peers", "Serialized Phase-4 duration peer estimate is unavailable.",
        ))
    else:
        if peer_duration_lower is None or peer_duration_upper is None:
            duration_range_position = "unavailable"
        elif years < peer_duration_lower:
            duration_range_position = "below_historical_80pct_range"
        elif years > peer_duration_upper:
            duration_range_position = "above_historical_80pct_range"
        else:
            duration_range_position = "within_historical_80pct_range"
        cards["offer_context"].append(_metric(
            "contract_duration_peer_delta_years", "Term versus historical peers", years - peer_duration_estimate, "years",
            "historical_benchmark_translation", "supported",
            "Difference in years from the modeled historical peer contract length for similar players.",
            "Historical deviation is not an optimal-term recommendation.",
            {
                "proposed_contract_years": years, "peer_estimate_years": peer_duration_estimate,
                "peer_lower_years": peer_duration_lower, "peer_upper_years": peer_duration_upper,
                "range_position": duration_range_position,
            },
        ))

    wage_to_value = wage / market
    if peer_wage_value_estimate is None:
        cards["offer_context"].append(_unavailable(
            "wage_to_market_value_peer_comparison", "Wage/value versus historical peers", "Serialized Phase-4 wage-to-value peer estimate is unavailable.",
        ))
    else:
        if peer_wage_value_lower is None or peer_wage_value_upper is None:
            wage_value_range_position = "unavailable"
        elif wage_to_value < peer_wage_value_lower:
            wage_value_range_position = "below_historical_80pct_range"
        elif wage_to_value > peer_wage_value_upper:
            wage_value_range_position = "above_historical_80pct_range"
        else:
            wage_value_range_position = "within_historical_80pct_range"
        cards["offer_context"].append(_metric(
            "wage_to_market_value_peer_comparison", "Wage/value versus historical peers", wage_to_value / peer_wage_value_estimate - 1, "ratio",
            "historical_benchmark_translation", "supported",
            "Proposed annual wage relative to public market value, compared with the modeled historical peer ratio for similar players.",
            "Historical deviation is not proof of overpayment or underpayment.",
            {
                "proposed_wage_to_market_value": wage_to_value, "peer_estimate_ratio": peer_wage_value_estimate,
                "peer_lower_ratio": peer_wage_value_lower, "peer_upper_ratio": peer_wage_value_upper,
                "range_position": wage_value_range_position,
            },
        ))

    if contribution_probability is None:
        cards["contribution_exposure"].append(_unavailable(
            "low_contribution_wage_exposure_eur", "Low-contribution wage exposure", "The approved Phase-1 contribution probability is unavailable.",
        ))
    else:
        low_probability = 1 - contribution_probability
        exposure = first_24m_commitment * low_probability
        cards["contribution_exposure"].append(_metric(
            "low_contribution_wage_exposure_eur", "Low-contribution wage exposure", exposure, "EUR",
            "probability_weighted_proxy", "supported_proxy",
            "Two-year fixed commitment weighted by modeled risk of not remaining a meaningful contributor through Year 2.",
            "Not expected accounting loss, cash loss, or wasted wages.",
            {
                "fixed_wages_through_24m_eur": first_24m_commitment,
                "low_contribution_probability_24m": low_probability,
                "sustained_contribution_probability_24m": contribution_probability,
            },
        ))

    # Value-preservation probability (Tier 2, 2026-08-12): the complement of
    # the 10%-downside model, per Phase 5's own definition ("Value
    # preservation is the complement of 10% downside: retaining at least
    # 90% of signing value"). Listed first in this card deliberately -- it's
    # the same evidence as the downside-risk metrics below, just framed as
    # the probability of a GOOD outcome instead of a bad one, which reads
    # more clearly to a non-specialist audience without changing the claim.
    if value_10_12m is None:
        cards["asset_value"].append(_unavailable(
            "value_preservation_probability_12m", "12-month value preservation probability", "The approved Phase-3 12-month 10% downside probability is unavailable.",
        ))
    else:
        cards["asset_value"].append(_metric(
            "value_preservation_probability_12m", "12-month value preservation probability", 1 - value_10_12m, "probability",
            "model_probability", "supported_candidate",
            "Probability the player retains at least 90% of current public market value through 12 months.",
            "Public-value preservation, not accounting value, sale proceeds, or a guarantee.",
            {"public_value_downside_10pct_probability_12m": value_10_12m},
        ))
    if value_10_24m is None:
        cards["asset_value"].append(_unavailable(
            "value_preservation_probability_24m", "24-month value preservation probability", "The approved Phase-3 24-month 10% downside probability is unavailable.",
        ))
    else:
        cards["asset_value"].append(_metric(
            "value_preservation_probability_24m", "24-month value preservation probability", 1 - value_10_24m, "probability",
            "model_probability", "supported_candidate",
            "Probability the player retains at least 90% of current public market value through 24 months.",
            "Public-value preservation, not accounting value, sale proceeds, or a guarantee.",
            {"public_value_downside_10pct_probability_24m": value_10_24m},
        ))

    if value_25 is None:
        cards["asset_value"].append(_unavailable(
            "public_value_downside_25pct_scenario", "24-month public-value downside scenario", "The approved Phase-3 25% downside probability is unavailable.",
        ))
    else:
        cards["asset_value"].append(_metric(
            "public_value_downside_25pct_scenario", "24-month public-value downside scenario",
            {"probability": value_25, "minimum_threshold_eur": market * .25}, "probability_and_EUR",
            "paired_probability_scenario", "supported_paired_display",
            "Probability and minimum euro threshold defining a 25% public-value decline.",
            "Do not multiply into expected loss; public market value is not sale proceeds, impairment, or book value.",
            {"current_public_market_value_eur": market, "downside_threshold_pct": .25},
        ))
    if value_50 is not None:
        cards["asset_value"].append(_metric(
            "public_value_downside_50pct_scenario", "Severe 24-month public-value downside scenario",
            {"probability": value_50, "minimum_threshold_eur": market * .50}, "probability_and_EUR",
            "paired_probability_scenario", "supported_paired_display_detail",
            "Probability and minimum euro threshold defining a 50% public-value decline.",
            "Do not multiply into expected loss; this is a public-value threshold scenario.",
            {"current_public_market_value_eur": market, "downside_threshold_pct": .50},
        ))

    cards["continuity"].append(_metric(
        "scheduled_fixed_wages_after_36m_eur", "Fixed wages scheduled after month 36", post_36m_commitment, "EUR",
        "exact_calculation", "supported",
        "Fixed wages scheduled beyond the model's 36-month continuity horizon.",
        "This amount is scheduled commitment, not expected loss.",
        {"annual_fixed_wage_eur": wage, "contract_years_beyond_36m": max(years - 3, 0)},
    ))
    if stay_36 is None:
        cards["continuity"].append(_unavailable(
            "contract_continuity_horizon", "Contract-continuity horizon", "The approved Phase-2 36-month stay probability is unavailable.",
        ))
    else:
        cards["continuity"].append(_metric(
            "contract_continuity_horizon", "Contract-continuity horizon",
            {"continuous_stay_probability_36m": stay_36, "scheduled_fixed_wages_after_36m_eur": post_36m_commitment},
            "probability_and_EUR", "paired_probability_scenario", "supported_paired_display",
            "Thirty-six-month continuous-stay probability shown beside scheduled fixed wages beyond that horizon.",
            "Do not multiply into expected wage loss; outbound movement can stop wages or produce a fee.",
            {"continuous_stay_probability_24m": stay_24, "scheduled_fixed_wages_after_24m_eur": post_24m_commitment},
        ))

    if permanent_relationship is None:
        cards["continuity"].append(_unavailable(
            "permanent_relationship_probability_24m", "Permanent-relationship probability (24m)",
            "The approved Phase-2 permanent-outbound model output is unavailable.",
        ))
    else:
        cards["continuity"].append(_metric(
            "permanent_relationship_probability_24m", "Permanent-relationship probability (24m)", permanent_relationship, "probability",
            "model_probability", "supported_candidate",
            "Probability the player's permanent relationship with the club is intact through 24 months -- a loan move does not end this, only a permanent transfer does.",
            "Not an exact departure date or a guarantee; an intervening loan can still occur.",
            {"continuous_stay_probability_24m": stay_24},
        ))

    if salary_percentile is None:
        cards["salary_structure"].append(_unavailable(
            "club_salary_percentile", "Club salary position", "A matched same-club public salary panel is unavailable.",
        ))
    else:
        cards["salary_structure"].append(_metric(
            "club_salary_percentile", "Club salary position", salary_percentile, "percentile",
            "observed_context", "supported_when_matched",
            "Position within the known same-club salary panel.",
            "The public salary panel may be incomplete and excludes some compensation.",
            {"context_source": salary_context_source, "known_panel_rows_including_proposal": salary_panel_rows},
        ))
    if payroll_share is None:
        cards["salary_structure"].append(_unavailable(
            "club_known_payroll_share", "Share of known club payroll", "A matched same-club public salary panel is unavailable.",
        ))
    else:
        cards["salary_structure"].append(_metric(
            "club_known_payroll_share", "Share of known club payroll", payroll_share, "ratio",
            "observed_context", "supported_when_matched",
            "Player annual fixed wage as a share of known public club fixed wages.",
            "The denominator is not audited total payroll.",
            {"context_source": salary_context_source, "known_panel_rows_including_proposal": salary_panel_rows},
        ))

    all_metrics = [metric for card in cards.values() for metric in card]
    return {
        "profile_scope": SCOPE,
        "metric_version": METRIC_VERSION,
        "player": payload.get("player"),
        "current_club": payload.get("current_club"),
        "decision_date": payload.get("decision_date"),
        "cards": cards,
        "availability": {
            "available_metrics": sum(metric["status"] != "unavailable" for metric in all_metrics),
            "unavailable_metrics": sum(metric["status"] == "unavailable" for metric in all_metrics),
        },
        "overall_contract_score": None,
        "overall_contract_score_status": "rejected_show_separate_business_metrics",
        "global_warnings": [
            "Validated for incumbent-club extension decisions in the modeled Big-Five scope only.",
            "No displayed metric is an approve/reject recommendation.",
            "Public market value is not realized sale proceeds or accounting value.",
            "Missing inputs remain unavailable and are never silently replaced with zero.",
        ],
    }

"""Frozen Engine V2 product statuses for movement and retention outputs.

This contract separates target semantics from deployment authorization.  A
development result is not a live product output, and retired/research-only
targets must never be reintroduced through a proxy or composite score.
"""

from __future__ import annotations

from typing import Final


CONTRACT_VERSION: Final[str] = "engine_v2_movement_product_scope_v1_2026-09-02"

RETIRED: Final[str] = "retired"
DEVELOPMENT_CANDIDATE: Final[str] = "development_candidate_not_deployed"
RESEARCH_ONLY: Final[str] = "research_only_unavailable"
HISTORICAL_ONLY: Final[str] = "frozen_v1_historical_only"

PRODUCT_OUTPUTS: Final[tuple[dict[str, object], ...]] = (
    {
        "output_id": "generic_continuity_24m",
        "public_label": "Generic club continuity",
        "target_field": "any_outbound_event_by_24m",
        "status": RETIRED,
        "definition": "Any recorded loan or permanent outbound movement within 24 months.",
        "numeric_output_allowed": False,
        "public_probability_claim_allowed": False,
        "final_validation_required": False,
        "reason_code": "ENDPOINT_RETIRED:MIXED_MOVEMENT_MECHANISMS",
        "product_note": "The former Contract-Horizon Stability output mixed temporary and permanent movement and is removed from the candidate product.",
    },
    {
        "output_id": "meaningful_retention_24m",
        "public_label": "Meaningful Retention",
        "target_field": "target_strict_meaningful_stay_24m",
        "status": DEVELOPMENT_CANDIDATE,
        "definition": "No outbound movement within 24 months and at least 25% same-club opportunity share in both Year 1 and Year 2.",
        "numeric_output_allowed": False,
        "public_probability_claim_allowed": False,
        "final_validation_required": True,
        "reason_code": "FINAL_TEMPORAL_VALIDATION_REQUIRED",
        "product_note": "Primary movement/retention candidate. It is not a rename of the contribution-only V1 endpoint.",
    },
    {
        "output_id": "temporary_displacement_risk_24m",
        "public_label": "Temporary Displacement Risk",
        "target_field": "target_temporary_first_24m",
        "status": DEVELOPMENT_CANDIDATE,
        "definition": "The first resolved outbound interruption within 24 months is a loan or loan return rather than a permanent transfer.",
        "numeric_output_allowed": False,
        "public_probability_claim_allowed": False,
        "final_validation_required": True,
        "reason_code": "FINAL_TEMPORAL_VALIDATION_REQUIRED",
        "product_note": "Secondary candidate representing a different football mechanism from permanent separation.",
    },
    {
        "output_id": "permanent_separation_risk_24m",
        "public_label": "Permanent Separation",
        "target_field": "target_permanent_relationship_ended_24m",
        "status": RESEARCH_ONLY,
        "definition": "The permanent relationship with the extension club ends within 24 months, whether or not a loan happened first.",
        "numeric_output_allowed": False,
        "public_probability_claim_allowed": False,
        "final_validation_required": False,
        "reason_code": "RESEARCH_ONLY:SUBGROUP_RELIABILITY_LIMITED",
        "product_note": "Unavailable as a personalized product output until a new frozen candidate repairs cross-league reliability.",
    },
    {
        "output_id": "sustained_meaningful_contribution_24m",
        "public_label": "Sustained Meaningful Contribution",
        "target_field": "target_sustained_meaningful_contribution",
        "status": HISTORICAL_ONLY,
        "definition": "At least 25% same-club opportunity share in both Year 1 and Year 2, without the no-outbound condition required by Meaningful Retention.",
        "numeric_output_allowed": False,
        "public_probability_claim_allowed": False,
        "final_validation_required": False,
        "reason_code": "FROZEN_V1_HISTORICAL_EVIDENCE_ONLY",
        "product_note": "Retained for V1 audit and contribution research; it must not be relabeled as Meaningful Retention.",
    },
)

PROHIBITED_COMPOSITES: Final[tuple[str, ...]] = (
    "overall_continuity_score",
    "contract_horizon_stability",
    "combined_retention_departure_score",
)


def contract_payload() -> dict[str, object]:
    return {
        "contract_version": CONTRACT_VERSION,
        "decision_date": "2026-09-02",
        "outputs": [dict(item) for item in PRODUCT_OUTPUTS],
        "prohibited_composites": list(PROHIBITED_COMPOSITES),
        "product_invariants": {
            "generic_continuity_retired": True,
            "movement_mechanisms_remain_separate": True,
            "development_candidate_is_not_deployment_authorization": True,
            "unavailable_outputs_return_no_personalized_probability": True,
            "no_proxy_may_reintroduce_retired_continuity": True,
        },
        "final_holdout_policy": "No 2024+ outcome was opened for this scope decision; later untouched evidence is required before candidate promotion.",
        "deployment_status": "no_engine_v2_movement_output_deployed",
    }


def validate_contract(payload: dict[str, object] | None = None) -> list[str]:
    """Return invariant violations; an empty list means the contract is valid."""

    candidate = contract_payload() if payload is None else payload
    violations: list[str] = []
    outputs = {str(item["output_id"]): item for item in candidate.get("outputs", [])}
    required = {
        "generic_continuity_24m", "meaningful_retention_24m",
        "temporary_displacement_risk_24m", "permanent_separation_risk_24m",
        "sustained_meaningful_contribution_24m",
    }
    if set(outputs) != required:
        violations.append("output_set_mismatch")
    for output_id, item in outputs.items():
        if bool(item.get("numeric_output_allowed")):
            violations.append(f"numeric_output_allowed:{output_id}")
        if bool(item.get("public_probability_claim_allowed")):
            violations.append(f"public_claim_allowed:{output_id}")
    if outputs.get("generic_continuity_24m", {}).get("status") != RETIRED:
        violations.append("generic_continuity_not_retired")
    if outputs.get("meaningful_retention_24m", {}).get("target_field") != "target_strict_meaningful_stay_24m":
        violations.append("meaningful_retention_target_mismatch")
    if outputs.get("temporary_displacement_risk_24m", {}).get("status") != DEVELOPMENT_CANDIDATE:
        violations.append("temporary_displacement_status_mismatch")
    if outputs.get("permanent_separation_risk_24m", {}).get("status") != RESEARCH_ONLY:
        violations.append("permanent_separation_not_research_only")
    return violations

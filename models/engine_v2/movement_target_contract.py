"""Frozen development semantics for Engine V2 movement outcomes.

This module defines targets only.  It does not fit, promote, or deploy a model.
The contract intentionally separates first movement state, eventual permanent
separation, and meaningful sporting retention because they answer different
football questions.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd


CONTRACT_VERSION: Final[str] = "engine_v2_movement_targets_v1_2026-09-01"
HORIZON_DAYS: Final[int] = 730
RAW_STATE_ORDER: Final[tuple[str, ...]] = (
    "no_outbound_24m",
    "permanent_first_24m",
    "loan_first_24m",
    "loan_return_first_24m",
    "unresolved_first_type_24m",
)
MODELED_STATE_ORDER: Final[tuple[str, ...]] = (
    "no_outbound_24m",
    "temporary_first_24m",
    "permanent_first_24m",
)

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "any_outbound_event_by_24m",
    "first_any_outbound_type",
    "permanent_outbound_event_by_24m",
    "target_sustained_meaningful_contribution",
)

TARGET_DEFINITIONS: Final[tuple[dict[str, object], ...]] = (
    {
        "target": "movement_state_raw",
        "kind": "five_state_audit",
        "horizon_days": HORIZON_DAYS,
        "rule": "No outbound, or first outbound typed as transfer, loan, loan_return, or unresolved.",
        "football_question": "What was the first recorded interruption in the player's relationship with the extension club?",
        "unresolved_policy": "Retain as unresolved; never assign to a modeled class.",
    },
    {
        "target": "movement_state_3",
        "kind": "three_state_competing_first_movement",
        "horizon_days": HORIZON_DAYS,
        "rule": "No outbound vs temporary first (loan or loan_return) vs permanent transfer first.",
        "football_question": "Did the club retain, temporarily displace, or permanently move the player first?",
        "unresolved_policy": "Exclude unresolved first types from model fitting and evaluation.",
    },
    {
        "target": "target_temporary_first_24m",
        "kind": "binary_first_movement",
        "horizon_days": HORIZON_DAYS,
        "rule": "movement_state_3 == temporary_first_24m among rows with resolved first-state semantics.",
        "football_question": "Was temporary displacement the first interruption?",
        "unresolved_policy": "Outcome unavailable for unresolved first types.",
    },
    {
        "target": "target_permanent_first_24m",
        "kind": "binary_first_movement",
        "horizon_days": HORIZON_DAYS,
        "rule": "movement_state_3 == permanent_first_24m among rows with resolved first-state semantics.",
        "football_question": "Was permanent separation the first interruption?",
        "unresolved_policy": "Outcome unavailable for unresolved first types.",
    },
    {
        "target": "target_permanent_relationship_ended_24m",
        "kind": "binary_event_within_horizon",
        "horizon_days": HORIZON_DAYS,
        "rule": "permanent_outbound_event_by_24m == 1, regardless of whether a loan happened first.",
        "football_question": "Did the player's permanent relationship with the club end within two years?",
        "unresolved_policy": "Use only rows whose permanent-outbound horizon is observable.",
    },
    {
        "target": "target_strict_meaningful_stay_24m",
        "kind": "binary_joint_sporting_retention",
        "horizon_days": HORIZON_DAYS,
        "rule": "No outbound within 24 months AND at least 25% same-club opportunity share in both Year 1 and Year 2.",
        "football_question": "Did the club both keep and meaningfully use the player for two years?",
        "unresolved_policy": "Requires observable movement and contribution outcomes.",
    },
)


def add_movement_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with all contract targets added deterministically."""

    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise KeyError(f"Movement target source columns missing: {missing}")

    result = frame.copy()
    any_outbound = pd.to_numeric(result["any_outbound_event_by_24m"], errors="coerce")
    first_type = result["first_any_outbound_type"].astype("string")
    observable = any_outbound.notna()

    conditions = [
        observable & any_outbound.eq(0),
        observable & any_outbound.eq(1) & first_type.eq("transfer").fillna(False),
        observable & any_outbound.eq(1) & first_type.eq("loan").fillna(False),
        observable & any_outbound.eq(1) & first_type.eq("loan_return").fillna(False),
    ]
    result["movement_state_raw"] = np.select(
        [condition.to_numpy(bool) for condition in conditions],
        list(RAW_STATE_ORDER[:4]),
        default="unresolved_first_type_24m",
    )
    result.loc[~observable, "movement_state_raw"] = pd.NA
    result["movement_state_3"] = result["movement_state_raw"].replace(
        {
            "loan_first_24m": "temporary_first_24m",
            "loan_return_first_24m": "temporary_first_24m",
            "unresolved_first_type_24m": pd.NA,
        }
    ).astype("string")

    resolved = result["movement_state_3"].notna()
    result["target_temporary_first_24m"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["target_permanent_first_24m"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result.loc[resolved, "target_temporary_first_24m"] = (
        result.loc[resolved, "movement_state_3"].eq("temporary_first_24m").astype(int)
    )
    result.loc[resolved, "target_permanent_first_24m"] = (
        result.loc[resolved, "movement_state_3"].eq("permanent_first_24m").astype(int)
    )

    permanent = pd.to_numeric(result["permanent_outbound_event_by_24m"], errors="coerce")
    result["target_permanent_relationship_ended_24m"] = permanent.astype("Int64")

    sustained = pd.to_numeric(result["target_sustained_meaningful_contribution"], errors="coerce")
    strict_observable = any_outbound.notna() & sustained.notna()
    result["target_strict_meaningful_stay_24m"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result.loc[strict_observable, "target_strict_meaningful_stay_24m"] = (
        any_outbound.loc[strict_observable].eq(0)
        & sustained.loc[strict_observable].eq(1)
    ).astype(int)
    return result


def contract_payload() -> dict[str, object]:
    """Serializable representation used by diagnostics and independent checks."""

    return {
        "contract_version": CONTRACT_VERSION,
        "horizon_days": HORIZON_DAYS,
        "raw_state_order": list(RAW_STATE_ORDER),
        "modeled_state_order": list(MODELED_STATE_ORDER),
        "required_source_columns": list(REQUIRED_COLUMNS),
        "definitions": list(TARGET_DEFINITIONS),
        "final_holdout_policy": "2024+ rows remain sealed during development diagnostics.",
        "deployment_status": "research_only_not_deployed",
    }

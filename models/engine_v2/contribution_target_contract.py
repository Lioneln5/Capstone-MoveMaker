"""Engine V2 contract for sustained realized club contribution.

This module repairs only the asymmetric Year-1/Year-2 evidence defect found in
the future-role target audit.  It does not repair incomplete relegation-season
schedules, normalize extra-time minutes, fit a model, or authorize deployment.

The repaired target is deliberately nullable.  A row receives a binary label
only when both exact post-signing years are fully observable, contain at least
10 captured extension-club matches, and have a nonmissing opportunity share.
"""

from __future__ import annotations

from typing import Final

import pandas as pd


CONTRACT_VERSION: Final[str] = "engine_v2_sustained_realized_contribution_v1_2026-09-04"
TARGET_FIELD: Final[str] = "target_sustained_realized_contribution_25pct"
AVAILABLE_FIELD: Final[str] = "sustained_realized_contribution_outcome_available"
REASON_FIELD: Final[str] = "sustained_realized_contribution_unavailability_reason"
POLICY_THRESHOLD: Final[float] = 0.25
MINIMUM_CAPTURED_CLUB_GAMES_PER_YEAR: Final[int] = 10

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "post_y1_window_fully_observable",
    "post_y2_window_fully_observable",
    "post_y1_same_club_all_competition_games",
    "post_y2_same_club_all_competition_games",
    "post_y1_window_evidence_eligible_10_club_games",
    "post_y2_window_evidence_eligible_10_club_games",
    "post_y1_same_club_all_competition_opportunity_share",
    "post_y2_same_club_all_competition_opportunity_share",
)


def _as_bool(series: pd.Series) -> pd.Series:
    if str(series.dtype) in {"bool", "boolean"}:
        return series.fillna(False).astype(bool)
    return series.astype("string").fillna("").str.strip().str.casefold().isin(
        {"true", "1", "yes"}
    )


def _year_evidence(frame: pd.DataFrame, year: int) -> tuple[pd.Series, pd.Series]:
    """Return primitive evidence eligibility and the numeric opportunity share.

    The stored builder flag is checked against its primitive definition.  A
    disagreement fails loudly so stale or malformed evidence cannot silently
    receive a target.
    """

    observable = _as_bool(frame[f"post_y{year}_window_fully_observable"])
    games = pd.to_numeric(
        frame[f"post_y{year}_same_club_all_competition_games"], errors="coerce"
    )
    recomputed = observable & games.ge(MINIMUM_CAPTURED_CLUB_GAMES_PER_YEAR)
    stored = _as_bool(frame[f"post_y{year}_window_evidence_eligible_10_club_games"])
    mismatch = recomputed.ne(stored)
    if mismatch.any():
        raise ValueError(
            f"Year-{year} stored evidence flag disagrees with full-observability "
            f"and >= {MINIMUM_CAPTURED_CLUB_GAMES_PER_YEAR} captured-game rule "
            f"on {int(mismatch.sum())} rows."
        )
    share = pd.to_numeric(
        frame[f"post_y{year}_same_club_all_competition_opportunity_share"],
        errors="coerce",
    )
    return recomputed, share


def add_symmetric_contribution_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the nullable repaired contribution target.

    No row is labeled unless Year 1 and Year 2 independently pass the same
    evidence rule and both opportunity shares are observed.  Ineligible rows
    remain ``pd.NA`` rather than becoming negative outcomes.
    """

    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise KeyError(f"Contribution target source columns missing: {missing}")

    result = frame.copy()
    y1_gate, y1_share = _year_evidence(result, 1)
    y2_gate, y2_share = _year_evidence(result, 2)
    y1_valid = y1_gate & y1_share.notna()
    y2_valid = y2_gate & y2_share.notna()
    available = y1_valid & y2_valid

    result[AVAILABLE_FIELD] = available
    result[TARGET_FIELD] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result.loc[available, TARGET_FIELD] = (
        y1_share.loc[available].ge(POLICY_THRESHOLD)
        & y2_share.loc[available].ge(POLICY_THRESHOLD)
    ).astype(int)

    reasons = pd.Series("AVAILABLE", index=result.index, dtype="string")
    reason_parts = (
        (~y1_gate, "YEAR1_EVIDENCE_INELIGIBLE"),
        (y1_share.isna(), "YEAR1_SHARE_MISSING"),
        (~y2_gate, "YEAR2_EVIDENCE_INELIGIBLE"),
        (y2_share.isna(), "YEAR2_SHARE_MISSING"),
    )
    for mask, code in reason_parts:
        affected = ~available & mask
        prior = reasons.loc[affected]
        reasons.loc[affected] = prior.where(prior.ne("AVAILABLE"), "") + (
            ";" + code
        )
        reasons.loc[affected] = reasons.loc[affected].str.lstrip(";")
    result[REASON_FIELD] = reasons
    return result


def contract_payload() -> dict[str, object]:
    """Return the machine-readable target contract."""

    return {
        "contract_version": CONTRACT_VERSION,
        "target_field": TARGET_FIELD,
        "available_field": AVAILABLE_FIELD,
        "reason_field": REASON_FIELD,
        "target_name": "sustained_realized_extension_club_contribution",
        "policy_threshold_each_year": POLICY_THRESHOLD,
        "minimum_captured_club_games_each_year": MINIMUM_CAPTURED_CLUB_GAMES_PER_YEAR,
        "year_1_rule": "fully observable AND >=10 captured extension-club matches AND nonmissing opportunity share",
        "year_2_rule": "fully observable AND >=10 captured extension-club matches AND nonmissing opportunity share",
        "positive_rule": "Year-1 share >=25% AND Year-2 share >=25%",
        "ineligible_policy": "target is unavailable; never coerce missing or ineligible evidence to zero",
        "semantic_scope": "realized same-club contribution, not role when available",
        "remaining_blockers": [
            "complete relegation/lower-tier club schedules or freeze a completeness refusal rule",
            "cap normalized opportunity at 100% or use exact match-minute capacity",
            "rerun affected development evidence after the complete target repair",
            "later untouched player-disjoint temporal evaluation",
        ],
        "final_holdout_policy": "2024+ outcomes are not evaluated by this repair stage",
        "deployment_status": "target_repair_step_1_only_not_deployed",
    }

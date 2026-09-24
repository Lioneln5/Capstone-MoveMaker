"""Complete Engine V2 contract for sustained realized club contribution.

This contract repairs the outcome definition only. It does not fit a model,
open a 2024+ final cohort, or authorize deployment.

The target is deliberately nullable. Both post-extension years must have:

* a fully observable anniversary window;
* at least ten captured extension-club matches and a nonmissing minutes total;
* original-Big-Five league match coverage of at least 80% of the median club
  observed over the same competition and exact dates; and
* a bounded opportunity share recomputed from minutes and captured games.

The peer-relative coverage rule distinguishes an individual club coverage
break from a league-wide schedule interruption. This matters for COVID-era
Ligue 1 windows, where a flat 30-match rule would reject clubs whose coverage
was complete relative to every peer. It also fails closed when relegation or
another top-flight exit leaves the lower-tier schedule unobserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd


CONTRACT_VERSION: Final[str] = "engine_v2_sustained_realized_contribution_v2_2026-09-07"
TARGET_FIELD: Final[str] = "target_sustained_realized_contribution_25pct_v2"
AVAILABLE_FIELD: Final[str] = "sustained_realized_contribution_outcome_available_v2"
REASON_FIELD: Final[str] = "sustained_realized_contribution_unavailability_reason_v2"
POLICY_THRESHOLD: Final[float] = 0.25
MINIMUM_CAPTURED_CLUB_GAMES_PER_YEAR: Final[int] = 10
MINIMUM_ORIGINAL_LEAGUE_PEER_COVERAGE: Final[float] = 0.80
MINIMUM_PEER_CLUBS: Final[int] = 10


@dataclass(frozen=True)
class YearFields:
    window_observable: str
    club_games: str
    club_minutes: str
    stored_evidence: str
    original_league_games: str
    peer_median_games: str
    peer_clubs: str
    coverage_ratio: str
    coverage_eligible: str
    raw_share: str
    normalized_share: str
    share_capped: str


def year_fields(year: int) -> YearFields:
    prefix = f"post_y{year}"
    return YearFields(
        window_observable=f"{prefix}_window_fully_observable",
        club_games=f"{prefix}_same_club_all_competition_games",
        club_minutes=f"{prefix}_same_club_all_competition_minutes",
        stored_evidence=f"{prefix}_window_evidence_eligible_10_club_games",
        original_league_games=f"{prefix}_same_club_same_big5_competition_games",
        peer_median_games=f"{prefix}_original_big5_peer_median_games_exact_window",
        peer_clubs=f"{prefix}_original_big5_peer_clubs_exact_window",
        coverage_ratio=f"{prefix}_original_big5_schedule_coverage_ratio",
        coverage_eligible=f"{prefix}_original_big5_schedule_coverage_eligible_80pct_peer",
        raw_share=f"{prefix}_realized_contribution_opportunity_share_raw",
        normalized_share=f"{prefix}_realized_contribution_opportunity_share_normalized",
        share_capped=f"{prefix}_realized_contribution_share_capped_at_100pct",
    )


BASE_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "capology_extension_event_id",
    "canonical_club_id_x",
    "competition_id",
    "post_y1_window_start_exclusive_or_inclusive",
    "post_y1_window_end_exclusive_or_inclusive",
    "post_y2_window_start_exclusive_or_inclusive",
    "post_y2_window_end_exclusive_or_inclusive",
)

GAME_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "competition_id",
    "date",
    "home_club_id",
    "away_club_id",
)


def _as_bool(series: pd.Series) -> pd.Series:
    if str(series.dtype) in {"bool", "boolean"}:
        return series.fillna(False).astype(bool)
    return series.astype("string").fillna("").str.strip().str.casefold().isin(
        {"true", "1", "yes"}
    )


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise KeyError(f"{label} source columns missing: {missing}")


def _club_game_long(games: pd.DataFrame) -> pd.DataFrame:
    _require_columns(games, GAME_REQUIRED_COLUMNS, "Game")
    base = games.loc[:, list(GAME_REQUIRED_COLUMNS)].copy()
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    home = base[["game_id", "competition_id", "date", "home_club_id"]].rename(
        columns={"home_club_id": "club_id"}
    )
    away = base[["game_id", "competition_id", "date", "away_club_id"]].rename(
        columns={"away_club_id": "club_id"}
    )
    long = pd.concat([home, away], ignore_index=True)
    long["club_id"] = pd.to_numeric(long["club_id"], errors="coerce").astype("Int64")
    return long.dropna(subset=["game_id", "competition_id", "date", "club_id"]).drop_duplicates(
        ["game_id", "club_id"]
    )


def add_original_league_schedule_coverage(
    frame: pd.DataFrame, games: pd.DataFrame
) -> pd.DataFrame:
    """Add exact-window peer coverage fields without inspecting 2024+ labels.

    ``frame`` must already be restricted to the intended development period.
    The function compares each extension club's original-league game count
    with the median count across all clubs observed in the same competition
    and exact anniversary window. Stored target-club counts are checked against
    the raw games table and disagreement fails loudly.
    """

    _require_columns(frame, BASE_REQUIRED_COLUMNS, "Contribution")
    result = frame.copy()
    if result["capology_extension_event_id"].duplicated().any():
        raise ValueError("Contribution frame contains duplicate extension event IDs.")

    club_games = _club_game_long(games)
    competition_groups = {
        str(competition): group.sort_values("date")
        for competition, group in club_games.groupby("competition_id", sort=False)
    }

    for year in (1, 2):
        fields = year_fields(year)
        _require_columns(result, (fields.original_league_games,), "Contribution")
        peer_medians: list[float] = []
        peer_counts: list[int] = []
        raw_target_counts: list[float] = []

        start_column = f"post_y{year}_window_start_exclusive_or_inclusive"
        end_column = f"post_y{year}_window_end_exclusive_or_inclusive"
        for row in result.itertuples(index=False):
            competition = getattr(row, "competition_id")
            club_id = pd.to_numeric(
                pd.Series([getattr(row, "canonical_club_id_x")]), errors="coerce"
            ).iat[0]
            start = pd.to_datetime(getattr(row, start_column), errors="coerce")
            end = pd.to_datetime(getattr(row, end_column), errors="coerce")
            group = competition_groups.get(str(competition))
            if group is None or pd.isna(club_id) or pd.isna(start) or pd.isna(end):
                peer_medians.append(np.nan)
                peer_counts.append(0)
                raw_target_counts.append(np.nan)
                continue

            window = group.loc[group["date"].gt(start) & group["date"].le(end)]
            counts = window.groupby("club_id", observed=True)["game_id"].nunique()
            peer_medians.append(float(counts.median()) if len(counts) else np.nan)
            peer_counts.append(int(len(counts)))
            raw_target_counts.append(float(counts.get(int(club_id), 0)))

        stored_counts = pd.to_numeric(result[fields.original_league_games], errors="coerce")
        raw_counts = pd.Series(raw_target_counts, index=result.index, dtype="float64")
        comparable = stored_counts.notna() & raw_counts.notna()
        counts_match = pd.Series(
            np.isclose(stored_counts, raw_counts, atol=0.0, rtol=0.0),
            index=result.index,
        )
        mismatch = comparable & ~counts_match
        if mismatch.any():
            raise ValueError(
                f"Year-{year} stored original-league game count disagrees with "
                f"the raw games table on {int(mismatch.sum())} rows."
            )

        peer_median = pd.Series(peer_medians, index=result.index, dtype="float64")
        peers = pd.Series(peer_counts, index=result.index, dtype="Int64")
        ratio = stored_counts.div(peer_median.where(peer_median.gt(0)))
        coverage = (
            peer_median.notna()
            & peers.ge(MINIMUM_PEER_CLUBS)
            & ratio.ge(MINIMUM_ORIGINAL_LEAGUE_PEER_COVERAGE)
        )
        result[fields.peer_median_games] = peer_median
        result[fields.peer_clubs] = peers
        result[fields.coverage_ratio] = ratio
        result[fields.coverage_eligible] = coverage

    return result


def _year_target_inputs(
    frame: pd.DataFrame, year: int
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    fields = year_fields(year)
    required = (
        fields.window_observable,
        fields.club_games,
        fields.club_minutes,
        fields.stored_evidence,
        fields.original_league_games,
        fields.peer_median_games,
        fields.peer_clubs,
        fields.coverage_ratio,
        fields.coverage_eligible,
    )
    _require_columns(frame, required, "Contribution")

    observable = _as_bool(frame[fields.window_observable])
    games = pd.to_numeric(frame[fields.club_games], errors="coerce")
    minutes = pd.to_numeric(frame[fields.club_minutes], errors="coerce")
    base_evidence = observable & games.ge(MINIMUM_CAPTURED_CLUB_GAMES_PER_YEAR)
    stored_evidence = _as_bool(frame[fields.stored_evidence])
    stored_mismatch = base_evidence.ne(stored_evidence)
    if stored_mismatch.any():
        raise ValueError(
            f"Year-{year} stored evidence flag disagrees with full-observability "
            f"and >= {MINIMUM_CAPTURED_CLUB_GAMES_PER_YEAR} captured-game rule "
            f"on {int(stored_mismatch.sum())} rows."
        )

    peer_median = pd.to_numeric(frame[fields.peer_median_games], errors="coerce")
    peers = pd.to_numeric(frame[fields.peer_clubs], errors="coerce")
    ratio = pd.to_numeric(frame[fields.coverage_ratio], errors="coerce")
    recomputed_coverage = (
        peer_median.notna()
        & peer_median.gt(0)
        & peers.ge(MINIMUM_PEER_CLUBS)
        & ratio.ge(MINIMUM_ORIGINAL_LEAGUE_PEER_COVERAGE)
    )
    stored_coverage = _as_bool(frame[fields.coverage_eligible])
    coverage_mismatch = recomputed_coverage.ne(stored_coverage)
    if coverage_mismatch.any():
        raise ValueError(
            f"Year-{year} stored peer-coverage flag disagrees with the declared "
            f"80% rule on {int(coverage_mismatch.sum())} rows."
        )

    raw_share = minutes.div(90.0 * games.where(games.gt(0)))
    normalized_share = raw_share.clip(lower=0.0, upper=1.0)
    share_capped = raw_share.gt(1.0)
    valid = base_evidence & recomputed_coverage & minutes.notna() & minutes.ge(0)
    return valid, raw_share, normalized_share.where(valid), share_capped


def add_complete_contribution_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the complete nullable contribution target."""

    result = frame.copy()
    year_results: dict[int, tuple[pd.Series, pd.Series, pd.Series, pd.Series]] = {}
    for year in (1, 2):
        year_results[year] = _year_target_inputs(result, year)
        _, raw_share, normalized_share, share_capped = year_results[year]
        fields = year_fields(year)
        result[fields.raw_share] = raw_share
        result[fields.normalized_share] = normalized_share
        result[fields.share_capped] = share_capped

    y1_valid, _, y1_share, _ = year_results[1]
    y2_valid, _, y2_share, _ = year_results[2]
    available = y1_valid & y2_valid
    result[AVAILABLE_FIELD] = available
    result[TARGET_FIELD] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result.loc[available, TARGET_FIELD] = (
        y1_share.loc[available].ge(POLICY_THRESHOLD)
        & y2_share.loc[available].ge(POLICY_THRESHOLD)
    ).astype(int)

    reason_lists: list[list[str]] = [[] for _ in range(len(result))]
    for year in (1, 2):
        fields = year_fields(year)
        observable = _as_bool(result[fields.window_observable]).to_numpy()
        games = pd.to_numeric(result[fields.club_games], errors="coerce").to_numpy()
        minutes = pd.to_numeric(result[fields.club_minutes], errors="coerce").to_numpy()
        peer_median = pd.to_numeric(result[fields.peer_median_games], errors="coerce").to_numpy()
        peers = pd.to_numeric(result[fields.peer_clubs], errors="coerce").to_numpy()
        ratio = pd.to_numeric(result[fields.coverage_ratio], errors="coerce").to_numpy()
        for position in range(len(result)):
            prefix = f"YEAR{year}"
            if not observable[position]:
                reason_lists[position].append(f"{prefix}_WINDOW_NOT_FULLY_OBSERVABLE")
            if pd.isna(games[position]) or games[position] < MINIMUM_CAPTURED_CLUB_GAMES_PER_YEAR:
                reason_lists[position].append(f"{prefix}_BASE_GAME_EVIDENCE_INSUFFICIENT")
            if pd.isna(minutes[position]) or minutes[position] < 0:
                reason_lists[position].append(f"{prefix}_MINUTES_INVALID_OR_MISSING")
            if pd.isna(peer_median[position]) or peer_median[position] <= 0 or pd.isna(peers[position]) or peers[position] < MINIMUM_PEER_CLUBS:
                reason_lists[position].append(f"{prefix}_PEER_SCHEDULE_REFERENCE_UNAVAILABLE")
            elif pd.isna(ratio[position]) or ratio[position] < MINIMUM_ORIGINAL_LEAGUE_PEER_COVERAGE:
                reason_lists[position].append(
                    f"{prefix}_ORIGINAL_BIG5_SCHEDULE_BELOW_80PCT_PEER"
                )

    result[REASON_FIELD] = pd.Series(
        ["AVAILABLE" if not reasons else ";".join(reasons) for reasons in reason_lists],
        index=result.index,
        dtype="string",
    )
    return result


def contract_payload() -> dict[str, object]:
    """Return the machine-readable complete target contract."""

    return {
        "contract_version": CONTRACT_VERSION,
        "target_field": TARGET_FIELD,
        "available_field": AVAILABLE_FIELD,
        "reason_field": REASON_FIELD,
        "target_name": "sustained_realized_extension_club_contribution",
        "policy_threshold_each_year": POLICY_THRESHOLD,
        "minimum_base_captured_club_games_each_year": MINIMUM_CAPTURED_CLUB_GAMES_PER_YEAR,
        "original_league_schedule_coverage_rule": "target club original-league games / contemporaneous exact-window peer median >= 0.80",
        "minimum_original_league_peer_coverage": MINIMUM_ORIGINAL_LEAGUE_PEER_COVERAGE,
        "minimum_peer_clubs": MINIMUM_PEER_CLUBS,
        "opportunity_share_formula": "captured extension-club minutes / (90 * captured extension-club games), clipped to [0,1]",
        "positive_rule": "normalized Year-1 share >=25% AND normalized Year-2 share >=25%",
        "ineligible_policy": "target is unavailable; never coerce missing or incomplete evidence to zero",
        "schedule_scope": "all captured club competitions for opportunity; original Big Five league used to verify schedule continuity",
        "semantic_scope": "sustained realized same-club contribution, not tactical role or role when available",
        "availability_scope": "injury, suspension, international duty, loans, transfers and non-selection are not removed from realized contribution",
        "threshold_status": "declared policy boundary with 20% and 30% sensitivity reported",
        "final_holdout_policy": "2024+ extension outcomes are excluded from this repair and remain unopened",
        "models_fitted": 0,
        "deployment_status": "complete_target_repair_development_only_not_deployed",
        "remaining_blockers": [
            "rerun rolling-origin development modeling under this frozen target",
            "freeze any surviving candidate before later untouched temporal evaluation",
            "pass later player-disjoint pooled and advertised-subgroup gates",
        ],
    }

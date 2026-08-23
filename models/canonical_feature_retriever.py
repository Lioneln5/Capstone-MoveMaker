"""Canonical, temporally-safe feature retrieval for live extension scoring.

This module is the missing link between the offline-validated Phase 1-4
diagnostics (models/run_extension_*_diagnostic.py,
models/run_contract_financial_exposure_benchmark.py) and a live web-app
scorer. It does NOT fit or contain a predictive model. It reconstructs, for
an arbitrary (canonical_player_id, canonical_club_id, competition_id,
decision_date), the exact feature vector those four diagnostics consume --
using the same source tables and the same windowing/derivation logic as
scripts/build_contract_extension_integration.py -- so that a refit,
serialized model can be scored on a proposed extension that has not
happened yet, not only on the 2,425 already-signed Capology events.

Non-negotiable temporal-integrity rule: nothing dated after
`decision_date` may enter any feature. Every field that cannot be computed
as of `decision_date` is returned as None with an explicit reason in
`unavailable`, never imputed or zero-filled.

Known deliberate scope limits (see module docstring end):
- `league` (competition_id) is caller-supplied, not inferred, because a
  club's competition can change season to season (promotion/relegation)
  and the training data always took it directly from the Capology event
  row rather than inferring it.
- Season-performance lookups require the SAME club in the prior season,
  matching the training semantics exactly (this is an incumbent-club
  extension feature set: "how did he play here" is the intended question,
  not "how did he play anywhere").
- Club salary panel matching approximates the season containing
  `decision_date`; the original build script joined the extension to
  whichever Capology page-season it was scraped from, which is not
  independently reconstructible here.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("MOVEMAKER_DATA_ROOT", ROOT / "Data")).expanduser().resolve()
PROCESSED = DATA_ROOT / "processed"
PERFORMANCE_DIR = PROCESSED / "canonical_performance"
INTEGRATION_DIR = PROCESSED / "canonical_integration"
TM_TABLES = PROCESSED / "transfermarkt_clean" / "tables"
CONTRACTS_DIR = PROCESSED / "capology_contracts"

PLAYER_DIM_PATH = INTEGRATION_DIR / "canonical_player_dimension.csv"
VALUATION_PATH = INTEGRATION_DIR / "canonical_valuation_history.csv"
APPEARANCE_PATH = TM_TABLES / "appearances_clean.csv"
GAMES_PATH = TM_TABLES / "games_clean.csv"
PERFORMANCE_PATH = PERFORMANCE_DIR / "canonical_player_team_season_performance.csv"
ADVANCED_PATH = PERFORMANCE_DIR / "fbref_advanced_performance_features.csv"
SALARY_PANEL_PATH = CONTRACTS_DIR / "canonical_salary_panel.csv"
LIVE_REFRESH_DIR = PROCESSED / "live_input_refresh"
LIVE_PERFORMANCE_PATH = LIVE_REFRESH_DIR / "canonical_performance_overlay.csv"
LIVE_SALARY_PATH = LIVE_REFRESH_DIR / "canonical_salary_overlay.csv"

BIG5 = {"GB1", "ES1", "IT1", "L1", "FR1"}

# Every retained model's categorical "league" feature was fit on the Capology
# source's own `league` column, which is the full display name ("Premier
# League"), NOT the `competition_id` code ("GB1") -- confirmed directly from
# the frozen/live extension_modeling_master.csv (`league` vs `competition_id`
# columns) and from the fitted preprocessor's own `categories["league"]`
# (['Bundesliga', 'La Liga', 'Ligue 1', 'Premier League', 'Serie A',
# '__OTHER__']). This mapping mirrors scripts/build_capology_contracts.py's
# LEAGUE_META and scripts/build_live_input_refresh.py's COMPETITIONS -- the
# same (code -> display name) pairs already used elsewhere to build the
# training data itself, not a new source of truth.
COMPETITION_TO_LEAGUE_NAME = {
    "GB1": "Premier League", "ES1": "La Liga", "IT1": "Serie A",
    "L1": "Bundesliga", "FR1": "Ligue 1",
}

PERFORMANCE_COLUMNS = [
    "canonical_performance_id", "canonical_player_id", "canonical_club_id",
    "canonical_competition_id", "season_start_year", "data_coverage_tier",
    "advanced_features_available", "basic_model_eligible_450_minutes",
    "canonical_matches_played", "canonical_minutes_played", "canonical_goals",
    "canonical_assists", "canonical_yellow_cards", "canonical_red_cards",
    "canonical_goals_per90", "canonical_assists_per90", "current_starts",
    "current_start_pct",
]
ADVANCED_COLUMNS = [
    "canonical_performance_id", "fbref_season_end_date", "fbref_expected_goals",
    "fbref_non_penalty_expected_goals", "fbref_progressive_carries",
    "fbref_progressive_passes", "fbref_tackles_won", "fbref_interceptions",
    "fbref_clearances", "fbref_pass_completion_pct", "fbref_key_passes",
    "fbref_total_shots", "fbref_shot_creating_actions_per90",
    "fbref_goal_creating_actions_per90", "fbref_aerial_duels_won_pct",
]

# Every feature the selected Phase 1-4 model variants need (union of M5 for
# opportunity/survival/value-preservation, B2 for wage/commitment/wage-value,
# B5 for duration), and where it comes from. Kept here so the retriever's
# contract can be audited against models/*/feature_manifest.csv by eye.
FEATURE_SOURCE = {
    "age_at_signing": "computed: canonical_player_dimension.canonical_date_of_birth vs decision_date",
    "age_at_expiration": "computed: age_at_signing + proposed_contract_years",
    "canonical_position": "canonical_player_dimension.canonical_position",
    "league": "caller-supplied competition_id (not inferred)",
    "log_market_value_at_signing": "canonical_valuation_history, nearest valuation_date <= decision_date",
    "pre365_same_club_all_competition_opportunity_share": "appearances_clean + games_clean, window [decision_date-365, decision_date)",
    "pre365_same_club_all_competition_appearance_rate": "appearances_clean + games_clean, same window",
    "pre365_goals_per90": "90 * pre365 goals / pre365 minutes (minutes>0)",
    "pre365_assists_per90": "90 * pre365 assists / pre365 minutes (minutes>0)",
    "pre1_canonical_minutes_played": "canonical_player_team_season_performance, most recently completed season, same club",
    "pre1_canonical_goals_per90": "canonical_player_team_season_performance, same row",
    "pre1_canonical_assists_per90": "canonical_player_team_season_performance, same row",
    "pre2_canonical_minutes_played": "canonical_player_team_season_performance, season before pre1, same club",
    "pre2_canonical_goals_per90": "canonical_player_team_season_performance, same row",
    "pre2_canonical_assists_per90": "canonical_player_team_season_performance, same row",
    "pre1_fbref_expected_goals": "fbref_advanced_performance_features, joined to pre1 performance row",
    "pre1_fbref_progressive_carries": "fbref_advanced_performance_features, same join",
    "pre1_fbref_progressive_passes": "fbref_advanced_performance_features, same join",
    "pre1_fbref_pass_completion_pct": "fbref_advanced_performance_features, same join",
    "exact_duration_years": "caller-supplied proposed_contract_years",
    "log_annual_gross_eur": "caller-supplied proposed_annual_fixed_wage_eur",
    "log_fixed_wage_commitment_eur": "computed: proposed_annual_fixed_wage_eur * proposed_contract_years",
    "annual_wage_to_market_value": "computed from proposed wage and market value",
    "commitment_to_market_value": "computed from fixed commitment and market value",
    "log_prior_season_annual_gross_eur": "canonical_salary_panel, same player + club, season before decision season",
    "prior_wage_available": "computed: whether log_prior_season_annual_gross_eur resolved",
    "club_salary_percentile": "canonical_salary_panel, club-season panel with proposed wage inserted",
    "club_salary_share_known": "canonical_salary_panel, same panel",
    "salary_change_from_prior_pct": "computed from proposed wage and prior season wage",
}


def _clean_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").fillna("").str.strip().str.casefold().isin({"true", "1", "yes"})


def completed_season_start_year(as_of: pd.Timestamp) -> int:
    """Largest season-start-year S such that season S (Aug 1 S - Jun 30 S+1)
    is fully complete as of `as_of`. Mirrors the "season is complete on
    June 30; June 30 signings may use that completed season" rule documented
    in scripts/build_contract_extension_integration.py's README output."""
    boundary = pd.Timestamp(year=as_of.year, month=6, day=30)
    return as_of.year - 1 if as_of >= boundary else as_of.year - 2


def current_season_start_year(as_of: pd.Timestamp) -> int:
    """Season-start-year of the season `as_of` falls inside (Aug 1 - Jun 30),
    used for club-salary-panel matching against the season a decision is
    made within, not the last fully completed one."""
    return as_of.year if as_of.month >= 7 else as_of.year - 1


@dataclass
class FeatureResult:
    features: dict[str, Any] = field(default_factory=dict)
    unavailable: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    decision_date: pd.Timestamp | None = None
    in_scope: bool = True
    refusal_reason: str | None = None
    incumbency_verified: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision_date": None if self.decision_date is None else self.decision_date.date().isoformat(),
            "in_scope": self.in_scope,
            "refusal_reason": self.refusal_reason,
            "incumbency_verified": self.incumbency_verified,
            "features": self.features,
            "unavailable": self.unavailable,
            "warnings": self.warnings,
        }


class CanonicalFeatureRetriever:
    """Lazy-loads the canonical source tables once, then answers many
    feature-vector requests against them. Construct one instance per
    process/service; do not re-instantiate per request."""

    def __init__(self, warm: bool = False) -> None:
        self._players: pd.DataFrame | None = None
        self._valuations: pd.DataFrame | None = None
        self._appearances: pd.DataFrame | None = None
        self._games: pd.DataFrame | None = None
        self._performance: pd.DataFrame | None = None
        self._salary_panel: pd.DataFrame | None = None
        self._sporting_data_cutoff: pd.Timestamp | None = None  # set on first games/appearances load
        if warm:
            self.warm_up()

    def warm_up(self) -> None:
        """Eagerly parse every lazy-loaded source table once. Measured cost
        is ~44s (dominated by the ~1.9M-row appearances table and the
        ~500K-row valuation history) -- entirely CSV-parsing, not retriever
        logic: a warm-cache get_extension_features() call takes ~0.08-0.15s.
        Call this once at process startup (see api/main.py's startup hook)
        so that cost is paid before the server accepts traffic, not on
        whichever request happens to arrive first."""
        self._load_players()
        self._load_valuations()
        self._load_appearances()
        self._load_games()
        self._load_performance()
        self._load_salary_panel()

    # -- lazy loaders -----------------------------------------------------
    def _load_players(self) -> pd.DataFrame:
        if self._players is None:
            frame = pd.read_csv(
                PLAYER_DIM_PATH,
                usecols=["canonical_player_id", "canonical_date_of_birth", "canonical_position", "canonical_sub_position"],
                low_memory=False,
            )
            frame["canonical_player_id"] = _clean_id(frame["canonical_player_id"])
            frame["canonical_date_of_birth"] = pd.to_datetime(frame["canonical_date_of_birth"], errors="coerce")
            self._players = frame.set_index("canonical_player_id", drop=False)
        return self._players

    def _load_valuations(self) -> pd.DataFrame:
        if self._valuations is None:
            frame = pd.read_csv(
                VALUATION_PATH,
                usecols=["canonical_player_id", "valuation_date", "canonical_market_value_eur"],
                low_memory=False,
            )
            frame["canonical_player_id"] = _clean_id(frame["canonical_player_id"])
            frame["valuation_date"] = pd.to_datetime(frame["valuation_date"], errors="coerce")
            frame["canonical_market_value_eur"] = pd.to_numeric(frame["canonical_market_value_eur"], errors="coerce")
            frame = frame.loc[
                frame["canonical_player_id"].notna()
                & frame["valuation_date"].notna()
                & frame["canonical_market_value_eur"].gt(0)
            ].sort_values(["canonical_player_id", "valuation_date"])
            self._valuations = frame
        return self._valuations

    def _load_appearances(self) -> pd.DataFrame:
        if self._appearances is None:
            frame = pd.read_csv(
                APPEARANCE_PATH,
                usecols=["game_id", "player_id", "player_club_id", "date", "competition_id", "goals", "assists", "minutes_played"],
                low_memory=False,
            )
            frame["player_id"] = _clean_id(frame["player_id"])
            frame["player_club_id"] = _clean_id(frame["player_club_id"])
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            self._appearances = frame.loc[frame["date"].notna()]
        return self._appearances

    def _load_games(self) -> pd.DataFrame:
        if self._games is None:
            frame = pd.read_csv(
                GAMES_PATH,
                usecols=["game_id", "competition_id", "date", "home_club_id", "away_club_id", "is_national_team_game"],
                low_memory=False,
            )
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame = frame.loc[frame["date"].notna() & ~_as_bool(frame["is_national_team_game"])]
            home = frame[["game_id", "competition_id", "date", "home_club_id"]].rename(columns={"home_club_id": "club_id"})
            away = frame[["game_id", "competition_id", "date", "away_club_id"]].rename(columns={"away_club_id": "club_id"})
            club_games = pd.concat([home, away], ignore_index=True)
            club_games["club_id"] = _clean_id(club_games["club_id"])
            self._games = club_games.drop_duplicates(["game_id", "club_id"])
            # Recorded once, used by _pre365_window to detect when as_of runs
            # past the sporting data's own known coverage -- otherwise a live
            # "today" decision_date silently undercounts recent involvement
            # as missing games, not as missing data. Discovered via a real
            # incident, not written defensively in advance: see the Jude
            # Bellingham case (2026-08-12 session) where the analogous salary-
            # panel exact-match bug produced a materially wrong continuity
            # score with no warning at all.
            self._sporting_data_cutoff = frame["date"].max()
        return self._games

    def _load_performance(self) -> pd.DataFrame:
        if self._performance is None:
            perf = pd.read_csv(PERFORMANCE_PATH, usecols=PERFORMANCE_COLUMNS, low_memory=False)
            perf["canonical_player_id"] = _clean_id(perf["canonical_player_id"])
            perf["canonical_club_id"] = _clean_id(perf["canonical_club_id"])
            perf["season_start_year"] = pd.to_numeric(perf["season_start_year"], errors="coerce").astype("Int64")
            perf = perf.loc[perf["canonical_competition_id"].isin(BIG5)].copy()
            advanced = pd.read_csv(ADVANCED_PATH, usecols=ADVANCED_COLUMNS, low_memory=False)
            perf = perf.merge(advanced, on="canonical_performance_id", how="left", validate="one_to_one")
            join_key = ["canonical_player_id", "canonical_club_id", "canonical_competition_id", "season_start_year"]
            if LIVE_PERFORMANCE_PATH.exists():
                live = pd.read_csv(LIVE_PERFORMANCE_PATH, low_memory=False)
                live["canonical_player_id"] = _clean_id(live["canonical_player_id"])
                live["canonical_club_id"] = _clean_id(live["canonical_club_id"])
                live["season_start_year"] = pd.to_numeric(live["season_start_year"], errors="coerce").astype("Int64")
                # Align to the canonical table's complete schema. Fields not
                # supplied by the supplemental snapshot remain explicitly
                # missing; the fitted preprocessors retain their validated
                # missing-indicator behavior.
                live = live.reindex(columns=perf.columns)
                perf = pd.concat([perf, live], ignore_index=True)
                perf = perf.sort_values(join_key).drop_duplicates(join_key, keep="last")
            if perf.duplicated(join_key).any():
                raise RuntimeError("Canonical performance join key is not unique; source data changed shape.")
            self._performance = perf.set_index(join_key, drop=False)
        return self._performance

    def _load_salary_panel(self) -> pd.DataFrame:
        if self._salary_panel is None:
            frame = pd.read_csv(SALARY_PANEL_PATH, low_memory=False)
            frame["canonical_club_id"] = _clean_id(frame["canonical_club_id"])
            frame["annual_gross_eur"] = pd.to_numeric(frame["annual_gross_eur"], errors="coerce")
            frame["source_page_season_start_year"] = pd.to_numeric(frame["source_page_season_start_year"], errors="coerce").astype("Int64")
            if LIVE_SALARY_PATH.exists():
                live = pd.read_csv(LIVE_SALARY_PATH, low_memory=False)
                live["canonical_club_id"] = _clean_id(live["canonical_club_id"])
                live["annual_gross_eur"] = pd.to_numeric(live["annual_gross_eur"], errors="coerce")
                live["source_page_season_start_year"] = pd.to_numeric(
                    live["source_page_season_start_year"], errors="coerce"
                ).astype("Int64")
                frame = pd.concat([frame, live.reindex(columns=frame.columns)], ignore_index=True)
                salary_key = [
                    "player_name_normalized", "canonical_club_id",
                    "source_page_season_start_year",
                ]
                frame = frame.sort_values(salary_key).drop_duplicates(salary_key, keep="last")
            self._salary_panel = frame
        return self._salary_panel

    # -- component lookups --------------------------------------------------
    def _player_row(self, player_id: int) -> pd.Series | None:
        players = self._load_players()
        if player_id not in players.index:
            return None
        row = players.loc[player_id]
        return row.iloc[0] if isinstance(row, pd.DataFrame) else row

    def _market_value_as_of(self, player_id: int, as_of: pd.Timestamp) -> tuple[float | None, int | None]:
        values = self._load_valuations()
        subset = values.loc[values["canonical_player_id"].eq(player_id) & values["valuation_date"].le(as_of)]
        if subset.empty:
            return None, None
        last = subset.iloc[-1]
        gap_days = int((as_of - last["valuation_date"]).days)
        return float(last["canonical_market_value_eur"]), gap_days

    def get_player_input_defaults(
        self,
        canonical_player_id: int,
        canonical_club_id: int,
        decision_date: str | pd.Timestamp,
        player_name_normalized: str | None,
    ) -> dict[str, Any]:
        """Return dated public values suitable for prefilling the live form.

        These values are starting points, not hidden scoring substitutions:
        the browser shows their dates and keeps both fields editable.  In
        particular, the salary is the latest known same-club public estimate,
        while the actual request field remains the user's *proposed* wage.
        """
        as_of = pd.Timestamp(decision_date)
        player_id = int(canonical_player_id)
        club_id = int(canonical_club_id)

        valuation_rows = self._load_valuations()
        valuation_rows = valuation_rows.loc[
            valuation_rows["canonical_player_id"].eq(player_id)
            & valuation_rows["valuation_date"].le(as_of)
        ]
        if valuation_rows.empty:
            market_value, valuation_date, valuation_gap_days = None, None, None
        else:
            valuation = valuation_rows.iloc[-1]
            valuation_date = valuation["valuation_date"]
            market_value = float(valuation["canonical_market_value_eur"])
            valuation_gap_days = int((as_of - valuation_date).days)

        decision_season = current_season_start_year(as_of)
        wage, wage_season = self._prior_wage(
            player_name_normalized, club_id, decision_season
        )
        return {
            "current_public_market_value_eur": market_value,
            "market_value_date": None if valuation_date is None else valuation_date.date().isoformat(),
            "market_value_age_days": valuation_gap_days,
            "latest_known_same_club_annual_wage_eur": wage,
            "wage_source_season_start_year": wage_season,
            "wage_source_season": None if wage_season is None else f"{wage_season}-{wage_season + 1}",
            "current_decision_season": f"{decision_season}-{decision_season + 1}",
            "wage_is_current_season": wage_season == decision_season if wage_season is not None else False,
        }

    def _pre365_window(self, player_id: int, club_id: int, competition_id: str, as_of: pd.Timestamp) -> dict[str, Any]:
        start = as_of - pd.Timedelta(days=365)
        apps = self._load_appearances()
        apps = apps.loc[
            apps["player_id"].eq(player_id)
            & apps["player_club_id"].eq(club_id)
            & apps["date"].ge(start)
            & apps["date"].lt(as_of)
        ]
        games = self._load_games()
        games = games.loc[games["club_id"].eq(club_id) & games["date"].ge(start) & games["date"].lt(as_of)]
        club_game_count = int(len(games))
        minutes = float(pd.to_numeric(apps["minutes_played"], errors="coerce").sum())
        goals = float(pd.to_numeric(apps["goals"], errors="coerce").sum())
        assists = float(pd.to_numeric(apps["assists"], errors="coerce").sum())
        result: dict[str, Any] = {
            "pre365_same_club_all_competition_games": club_game_count,
            "pre365_same_club_all_competition_appearances": int(len(apps)),
            "pre365_same_club_all_competition_minutes": minutes,
            "pre365_same_club_all_competition_goals": goals,
            "pre365_same_club_all_competition_assists": assists,
            "pre365_window_evidence_eligible_10_club_games": club_game_count >= 10,
        }
        if club_game_count > 0:
            result["pre365_same_club_all_competition_opportunity_share"] = min(minutes / (90 * club_game_count), 1.05)
            result["pre365_same_club_all_competition_appearance_rate"] = len(apps) / club_game_count
        else:
            result["pre365_same_club_all_competition_opportunity_share"] = None
            result["pre365_same_club_all_competition_appearance_rate"] = None
        result["pre365_goals_per90"] = 90 * goals / minutes if minutes > 0 else None
        result["pre365_assists_per90"] = 90 * assists / minutes if minutes > 0 else None
        # How much of the requested [start, as_of) window falls after the
        # sporting data's own last recorded date -- that portion is
        # necessarily zero-filled above, indistinguishable from "no games
        # happened" unless surfaced explicitly.
        if self._sporting_data_cutoff is not None and as_of > self._sporting_data_cutoff:
            result["pre365_data_coverage_gap_days"] = int((as_of - self._sporting_data_cutoff).days)
        else:
            result["pre365_data_coverage_gap_days"] = 0
        return result

    def _season_row(self, player_id: int, club_id: int, competition_id: str, season_start_year: int) -> pd.Series | None:
        perf = self._load_performance()
        key = (player_id, club_id, competition_id, season_start_year)
        if key not in perf.index:
            return None
        row = perf.loc[key]
        return row.iloc[0] if isinstance(row, pd.DataFrame) else row

    def _prior_wage(self, player_name_normalized: str | None, club_id: int, season_start_year: int) -> tuple[float | None, int | None]:
        """Most recent known annual_gross_eur for this player at this club,
        at or before season_start_year -- NOT an exact-season match. The
        salary panel currently stops at the 2023-24 season (source_page_
        season_start_year <= 2023); a live decision_date of "today" computes
        a target season year well past that. Requiring an exact match would
        silently return "unavailable" for this feature on every live scoring
        call past 2023, for every player, misreporting a coverage gap as a
        genuine missing-data case. Returns (wage, season actually used) so
        the caller can attach a staleness warning when they differ."""
        if player_name_normalized is None:
            return None, None
        panel = self._load_salary_panel()
        subset = panel.loc[
            panel["player_name_normalized"].eq(player_name_normalized)
            & panel["canonical_club_id"].eq(club_id)
            & panel["source_page_season_start_year"].le(season_start_year)
            & panel["annual_gross_eur"].notna()
        ]
        if subset.empty:
            return None, None
        row = subset.sort_values("source_page_season_start_year").iloc[-1]
        return float(row["annual_gross_eur"]), int(row["source_page_season_start_year"])

    def _club_salary_context(
        self, club_id: int, season_start_year: int, proposed_wage: float, player_name_normalized: str | None,
    ) -> tuple[float | None, float | None, int, int | None]:
        """Same fallback rule as _prior_wage: use the most recent available
        club-season salary panel at or before season_start_year, not an
        exact match, for the same reason. Returns (percentile, share, rows,
        season actually used)."""
        panel = self._load_salary_panel()
        candidate_seasons = panel.loc[panel["canonical_club_id"].eq(club_id) & panel["source_page_season_start_year"].le(season_start_year)]
        if candidate_seasons.empty:
            return None, None, 0, None
        actual_season = int(candidate_seasons["source_page_season_start_year"].max())
        others = panel.loc[
            panel["canonical_club_id"].eq(club_id)
            & panel["source_page_season_start_year"].eq(actual_season)
            & panel["annual_gross_eur"].notna()
        ]
        if player_name_normalized is not None:
            # Exclude the player's own existing row so the proposed wage is
            # inserted exactly once, matching business_metric_engine.py's
            # documented double-counting guard.
            others = others.loc[others["player_name_normalized"].ne(player_name_normalized)]
        known = others["annual_gross_eur"].to_numpy()
        if len(known) == 0:
            return None, None, 0, actual_season
        less = int((known < proposed_wage).sum())
        equal = int(np.isclose(known, proposed_wage, rtol=1e-12, atol=1e-8).sum())
        rows = len(known) + 1
        percentile = (less + (equal + 2) / 2) / rows
        share = proposed_wage / (float(known.sum()) + proposed_wage)
        return percentile, share, rows, actual_season

    # -- public entry point --------------------------------------------------
    def get_extension_features(
        self,
        canonical_player_id: int,
        canonical_club_id: int,
        competition_id: str,
        decision_date: str | pd.Timestamp,
        proposed_annual_fixed_wage_eur: float,
        proposed_contract_years: float,
        current_public_market_value_eur: float | None = None,
        player_name_normalized: str | None = None,
    ) -> FeatureResult:
        result = FeatureResult(decision_date=pd.Timestamp(decision_date))
        as_of = result.decision_date
        player_id = int(canonical_player_id)
        club_id = int(canonical_club_id)

        # -- player reference (stable attributes) --
        player_row = self._player_row(player_id)
        if player_row is None:
            result.unavailable["canonical_position"] = "canonical_player_id not found in canonical_player_dimension"
            result.unavailable["age_at_signing"] = "canonical_player_id not found in canonical_player_dimension"
        else:
            dob = player_row["canonical_date_of_birth"]
            if pd.isna(dob):
                result.unavailable["age_at_signing"] = "date_of_birth unavailable for this player"
            elif dob > as_of:
                raise ValueError("date_of_birth is after decision_date; refusing to compute a negative age")
            else:
                result.features["age_at_signing"] = (as_of - dob).days / 365.25
                result.features["age_at_expiration"] = result.features["age_at_signing"] + proposed_contract_years
            position = player_row["canonical_position"]
            if pd.isna(position):
                result.unavailable["canonical_position"] = "position unavailable in canonical_player_dimension"
            else:
                result.features["canonical_position"] = position

        if competition_id not in BIG5:
            # Hard refusal, not a warning: every retained model's categorical
            # "league" feature was trained on rows that are *exclusively*
            # the five Big-Five leagues (the Capology source itself only
            # covers the Big Five), so an out-of-scope value has zero
            # training support behind the preprocessor's "__OTHER__" bucket
            # -- it would not degrade gracefully, it would be meaningless.
            # The product scope requires this be an explicit out-of-scope-league
            # refusal case.
            result.in_scope = False
            result.refusal_reason = (
                f"competition_id {competition_id!r} is outside the modeled Big-Five scope (GB1/ES1/IT1/L1/FR1); "
                "Phase 1-4 models have no training support outside it."
            )
            return result
        # Store the TRAINING-TIME string, not the raw competition_id code --
        # every fitted preprocessor's "league" categories are display names
        # (see COMPETITION_TO_LEAGUE_NAME above). Passing the code straight
        # through here previously meant every live "league" value fell into
        # the fitted preprocessor's dropped-for-zero-variance "__OTHER__"
        # bucket (no training row was ever "__OTHER__", so keep_mask masks
        # that whole column out at fit time) -- i.e. live scoring silently
        # contributed ZERO league signal for every request, on every
        # Phase 1-4 endpoint that uses "league", regardless of which of the
        # five leagues the player was actually in. Found via a 2026-08-16
        # audit that compared a live-scored feature vector's "league" value
        # against the deployed artifact's own fitted `categories["league"]`.
        result.features["league"] = COMPETITION_TO_LEAGUE_NAME[competition_id]

        # -- incumbency signal (warning, not a hard refusal): this tool is
        # validated for an existing player-club relationship, never a
        # hypothetical new-club transfer, so a caller misusing it for
        # acquisition scouting should be flagged. But zero prior senior
        # appearances is NOT reliable proof of "wrong club" -- a real parity
        # run against 250 genuine historical Capology extension events found
        # ~1% of them are young/academy players or injury-return contracts
        # signing an extension before ever making a senior appearance for
        # that same club. Hard-refusing on appearance count alone produced
        # false refusals on real training data, so this stays a strong,
        # explicit warning the caller/app can choose to gate on, rather than
        # blocking scoring outright.
        apps = self._load_appearances()
        ever_appeared = apps.loc[
            apps["player_id"].eq(player_id) & apps["player_club_id"].eq(club_id) & apps["date"].lt(as_of)
        ]
        result.incumbency_verified = not ever_appeared.empty
        if not result.incumbency_verified:
            result.warnings.append(
                f"no recorded senior appearance for canonical_player_id={player_id} at canonical_club_id={club_id} "
                "before decision_date -- could be a genuine pre-debut/academy/injury-return extension, or a caller "
                "misusing this incumbent-club tool for new-club acquisition scoring. Confirm the club relationship "
                "independently before trusting this profile."
            )

        # -- market value at decision date --
        if current_public_market_value_eur is not None:
            market_value = float(current_public_market_value_eur)
            result.features["at_signing_market_value_eur"] = market_value
        else:
            market_value, gap_days = self._market_value_as_of(player_id, as_of)
            if market_value is None:
                result.unavailable["log_market_value_at_signing"] = "no valuation on or before decision_date in canonical_valuation_history"
            else:
                result.features["at_signing_market_value_eur"] = market_value
                if gap_days is not None and gap_days > 365:
                    result.warnings.append(
                        f"nearest market valuation is {gap_days} days before decision_date (>365); treat market-value-derived features as stale."
                    )
        if "at_signing_market_value_eur" in result.features:
            result.features["log_market_value_at_signing"] = math.log1p(max(result.features["at_signing_market_value_eur"], 0.0))

        # -- pre-365-day exact sporting window (same club) --
        window = self._pre365_window(player_id, club_id, competition_id, as_of)
        coverage_gap_days = window.pop("pre365_data_coverage_gap_days", 0)
        for key, value in window.items():
            if value is None:
                result.unavailable[key] = "fewer than 10 same-club club games in the pre-365-day window"
            else:
                result.features[key] = value
        if coverage_gap_days > 0:
            result.warnings.append(
                f"the pre-365-day sporting window extends {coverage_gap_days} day(s) past the last date our sporting data "
                "actually covers -- that portion is not observed, not confirmed empty, so recent involvement may be "
                "undercounted here. This gap grows every day the sporting-data snapshot isn't refreshed."
            )

        # -- pre1 / pre2 completed-season performance (same club) --
        pre1_year = completed_season_start_year(as_of)
        for label, season_start_year in [("pre1", pre1_year), ("pre2", pre1_year - 1)]:
            row = self._season_row(player_id, club_id, competition_id, season_start_year)
            if row is None:
                result.unavailable[f"{label}_canonical_minutes_played"] = (
                    f"no {label} same-club season performance row for season_start_year={season_start_year}"
                )
                result.unavailable[f"{label}_canonical_goals_per90"] = result.unavailable[f"{label}_canonical_minutes_played"]
                result.unavailable[f"{label}_canonical_assists_per90"] = result.unavailable[f"{label}_canonical_minutes_played"]
                if label == "pre1":
                    for advanced_key in ["expected_goals", "progressive_carries", "progressive_passes", "pass_completion_pct"]:
                        result.unavailable[f"pre1_fbref_{advanced_key}"] = result.unavailable[f"{label}_canonical_minutes_played"]
                continue
            for source_col, out_key in [
                ("canonical_minutes_played", f"{label}_canonical_minutes_played"),
                ("canonical_goals_per90", f"{label}_canonical_goals_per90"),
                ("canonical_assists_per90", f"{label}_canonical_assists_per90"),
            ]:
                value = row[source_col]
                if pd.isna(value):
                    result.unavailable[out_key] = f"{source_col} missing on the matched {label} performance row"
                else:
                    result.features[out_key] = float(value)
            if label == "pre1":
                for source_col, out_key in [
                    ("fbref_expected_goals", "pre1_fbref_expected_goals"),
                    ("fbref_progressive_carries", "pre1_fbref_progressive_carries"),
                    ("fbref_progressive_passes", "pre1_fbref_progressive_passes"),
                    ("fbref_pass_completion_pct", "pre1_fbref_pass_completion_pct"),
                ]:
                    value = row[source_col] if source_col in row.index else None
                    if value is None or pd.isna(value):
                        result.unavailable[out_key] = "advanced FBref features unavailable in the matched completed-season snapshot"
                    else:
                        result.features[out_key] = float(value)

        # -- proposed-contract arithmetic (caller-supplied, no lookup) --
        wage = float(proposed_annual_fixed_wage_eur)
        years = float(proposed_contract_years)
        fixed_commitment = wage * years
        result.features["exact_duration_years"] = years
        result.features["log_annual_gross_eur"] = math.log1p(max(wage, 0.0))
        result.features["log_fixed_wage_commitment_eur"] = math.log1p(max(fixed_commitment, 0.0))
        if "at_signing_market_value_eur" in result.features and result.features["at_signing_market_value_eur"] > 0:
            mv = result.features["at_signing_market_value_eur"]
            result.features["annual_wage_to_market_value"] = wage / mv
            result.features["commitment_to_market_value"] = fixed_commitment / mv
        else:
            result.unavailable["annual_wage_to_market_value"] = "market value unavailable"
            result.unavailable["commitment_to_market_value"] = "market value unavailable"

        # -- prior wage and club salary context --
        # Both fall back to the most recent AVAILABLE season at or before the
        # target, not an exact match -- the salary panel currently stops at
        # the 2023-24 season, well behind "today" for a live decision_date.
        decision_season = current_season_start_year(as_of)
        prior_wage, prior_wage_season = self._prior_wage(player_name_normalized, club_id, decision_season - 1)
        if prior_wage is None:
            result.unavailable["log_prior_season_annual_gross_eur"] = "no salary-panel row for this player at this club in or before the prior season"
            result.features["prior_wage_available"] = False
        else:
            result.features["log_prior_season_annual_gross_eur"] = math.log1p(max(prior_wage, 0.0))
            result.features["prior_wage_available"] = True
            result.features["salary_change_from_prior_pct"] = (wage - prior_wage) / prior_wage if prior_wage > 0 else None
            if prior_wage_season != decision_season - 1:
                result.warnings.append(
                    f"prior-season wage is from the {prior_wage_season}-{prior_wage_season + 1} salary panel (the most recent available), "
                    f"not the immediate {decision_season - 1}-{decision_season} season -- the salary panel does not yet cover that season for this club."
                )

        percentile, share, panel_rows, panel_season = self._club_salary_context(club_id, decision_season, wage, player_name_normalized)
        if percentile is None:
            result.unavailable["club_salary_percentile"] = "no club-season salary panel available at or before the decision season"
            result.unavailable["club_salary_share_known"] = "no club-season salary panel available at or before the decision season"
        else:
            result.features["club_salary_percentile"] = percentile
            result.features["club_salary_share_known"] = share
            season_note = "" if panel_season == decision_season else (
                f" Panel is from the {panel_season}-{panel_season + 1} season (the most recent available), "
                f"not the current {decision_season}-{decision_season + 1} season."
            )
            result.warnings.append(f"club salary panel context computed from {panel_rows} known rows (proposal included); panel is not audited total payroll.{season_note}")

        return result


def _run_parity_check(sample_size: int = 15, seed: int = 7) -> None:
    """Self-test: reconstruct features for N real historical extension events
    and diff against the values scripts/build_contract_extension_integration.py
    already computed and independently verified. Not a pytest suite -- a
    concrete, runnable sanity check that this reimplementation agrees with
    the training-time computation before it is trusted for live scoring."""
    master_path = PROCESSED / "contract_extension_integration" / "extension_modeling_master.csv"
    master = pd.read_csv(master_path, low_memory=False)
    master["signed_date"] = pd.to_datetime(master["signed_date"], errors="coerce")
    eligible = master.loc[
        master["canonical_player_id"].notna()
        & master["canonical_club_id_x"].notna()
        & master["signed_date"].notna()
        & master["pre365_window_evidence_eligible_10_club_games"].astype("string").str.casefold().eq("true")
    ]
    sample = eligible.sample(n=min(sample_size, len(eligible)), random_state=seed)

    retriever = CanonicalFeatureRetriever()
    compare_fields = [
        "pre365_same_club_all_competition_opportunity_share",
        "pre365_same_club_all_competition_appearance_rate",
        "pre365_goals_per90",
        "pre365_assists_per90",
        "pre1_canonical_minutes_played",
        "pre1_canonical_goals_per90",
        "pre1_canonical_assists_per90",
        "pre2_canonical_minutes_played",
        "pre2_canonical_goals_per90",
        "pre2_canonical_assists_per90",
        "pre1_fbref_expected_goals",
        "pre1_fbref_progressive_carries",
        "pre1_fbref_progressive_passes",
        "pre1_fbref_pass_completion_pct",
        "age_at_signing",
        "at_signing_market_value_eur",
        "club_salary_percentile",
        "club_salary_share_known",
    ]
    # Compared separately from compare_fields (string categories, not numeric
    # -- pd.to_numeric would silently turn them into NaN-vs-NaN "matches").
    # "league" was NOT covered here before 2026-08-16, which is exactly how a
    # real code/display-name mismatch (see COMPETITION_TO_LEAGUE_NAME above)
    # went undetected: every previous run of this self-test reported 100%
    # while every live "league" value was actually being silently dropped by
    # the fitted preprocessor's category matching.
    string_compare_fields = ["league", "canonical_position"]
    mismatches = 0
    checked = 0
    for _, row in sample.iterrows():
        result = retriever.get_extension_features(
            canonical_player_id=int(row["canonical_player_id"]),
            canonical_club_id=int(row["canonical_club_id_x"]),
            competition_id=row["competition_id"],
            decision_date=row["signed_date"],
            proposed_annual_fixed_wage_eur=float(row["annual_gross_eur"]) if pd.notna(row["annual_gross_eur"]) else 1.0,
            proposed_contract_years=float(row["exact_duration_years"]) if pd.notna(row["exact_duration_years"]) else 1.0,
            current_public_market_value_eur=None,
            player_name_normalized=row.get("player_name_normalized_x"),
        )
        # pre365_goals_per90/assists_per90 are not stored columns in the
        # master; the diagnostic scripts derive them from raw goals/assists/
        # minutes. Reproduce that same derivation here as the expected value.
        # club_salary_percentile/share_known in the master were computed from
        # the club salary-PANEL's own wage figure for this player-season,
        # which in ~15% of matched events differs from the extension event's
        # own negotiated annual_gross_eur (see extension_vs_panel_annual_
        # difference_eur -- a real, already-documented data nuance, not a
        # retriever bug). Only trust this comparison where the two figures
        # agree exactly, so it isolates the retriever's percentile formula.
        wage_matches_panel = math.isclose(
            pd.to_numeric(row.get("annual_gross_eur"), errors="coerce"),
            pd.to_numeric(row.get("salary_panel_annual_gross_eur"), errors="coerce"),
            rel_tol=1e-9, abs_tol=1e-6,
        ) if pd.notna(row.get("salary_panel_annual_gross_eur")) else False

        pre_minutes = pd.to_numeric(row.get("pre365_same_club_all_competition_minutes"), errors="coerce")
        expected_overrides = {
            "pre365_goals_per90": (90 * pd.to_numeric(row.get("pre365_same_club_all_competition_goals"), errors="coerce") / pre_minutes) if pre_minutes and pre_minutes > 0 else np.nan,
            "pre365_assists_per90": (90 * pd.to_numeric(row.get("pre365_same_club_all_competition_assists"), errors="coerce") / pre_minutes) if pre_minutes and pre_minutes > 0 else np.nan,
        }
        for field_name in compare_fields:
            if field_name in ("club_salary_percentile", "club_salary_share_known") and not wage_matches_panel:
                continue  # not comparable: extension wage differs from the panel wage master used (documented, not a bug)
            checked += 1
            expected = expected_overrides[field_name] if field_name in expected_overrides else row.get(field_name)
            actual = result.features.get(field_name)
            expected_val = pd.to_numeric(pd.Series([expected]), errors="coerce").iloc[0]
            actual_val = pd.to_numeric(pd.Series([actual]), errors="coerce").iloc[0]
            both_nan = pd.isna(expected_val) and pd.isna(actual_val)
            close = both_nan or (pd.notna(expected_val) and pd.notna(actual_val) and math.isclose(expected_val, actual_val, rel_tol=1e-6, abs_tol=1e-6))
            if not close:
                mismatches += 1
                print(f"MISMATCH event={row['capology_extension_event_id']} field={field_name} expected={expected_val} actual={actual_val}")
        for field_name in string_compare_fields:
            checked += 1
            expected = str(row.get(field_name)) if pd.notna(row.get(field_name)) else None
            actual = result.features.get(field_name)
            close = expected == actual
            if not close:
                mismatches += 1
                print(f"MISMATCH event={row['capology_extension_event_id']} field={field_name} expected={expected!r} actual={actual!r}")
    print(f"\nParity check: {checked - mismatches}/{checked} field comparisons matched across {len(sample)} sampled historical events.")


if __name__ == "__main__":
    _run_parity_check(sample_size=60)

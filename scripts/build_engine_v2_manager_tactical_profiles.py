"""Build decision-time manager, formation, and player-manager profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.engine_v2.manager_tactical_feature_contract import (  # noqa: E402
    CONTRACT_VERSION,
    CURRENT_MANAGER_PLAYER_ROLE_FEATURES,
    FORMATION_STABILITY_FEATURES,
    LOOKBACK_DAYS,
    MANAGER_STABILITY_FEATURES,
    MANAGER_TRANSITION_ROLE_CHANGE_FEATURES,
    MIN_CLUB_GAMES,
    MIN_CURRENT_MANAGER_GAMES,
    MIN_SOURCE_GAME_COVERAGE,
    MIN_SPLIT_GAMES,
    MIN_TRANSITION_SIDE_GAMES,
    MODEL_FEATURES,
    RECENT_DAYS,
    contract_payload,
)


DEFAULT_OUTPUT = ROOT / "Data" / "processed" / "engine_v2_manager_tactical_profiles"
P = ROOT / "Data" / "processed"
EXTENSIONS = P / "capology_contracts" / "canonical_extension_events.csv"
GAMES = P / "transfermarkt_clean" / "tables" / "games_clean.csv"
LINEUPS = P / "transfermarkt_clean" / "tables" / "game_lineups_clean.csv"
APPEARANCES = P / "transfermarkt_clean" / "tables" / "appearances_clean.csv"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def normalize_manager(value: object) -> str | None:
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    ascii_value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    normalized = re.sub(r"[^a-z0-9]+", "", ascii_value.lower())
    return normalized or None


def formation_shape(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    match = re.match(r"^(\d(?:-\d)+)", str(value).strip())
    return match.group(1) if match else None


def normalized_entropy(values: pd.Series) -> float:
    counts = values.dropna().astype(str).value_counts()
    if len(counts) < 2:
        return 0.0
    probability = counts.to_numpy(float) / counts.sum()
    return float(-(probability * np.log(probability)).sum() / np.log(len(counts)))


def distribution_shift(left: pd.Series, right: pd.Series) -> float:
    left_counts = left.dropna().astype(str).value_counts(normalize=True)
    right_counts = right.dropna().astype(str).value_counts(normalize=True)
    keys = left_counts.index.union(right_counts.index)
    return float(0.5 * sum(abs(left_counts.get(key, 0.0) - right_counts.get(key, 0.0)) for key in keys))


def source_coverage(game_ids: set[int], evidence: pd.DataFrame) -> float:
    if not game_ids:
        return np.nan
    return float(evidence.loc[evidence["game_id"].isin(game_ids), "game_id"].nunique() / len(game_ids))


def player_rates(
    lineups: pd.DataFrame,
    appearances: pd.DataFrame,
    player_id: int,
    game_count: int,
) -> dict[str, float]:
    selected = lineups.loc[lineups["player_id"].eq(player_id)]
    played = appearances.loc[appearances["player_id"].eq(player_id)]
    denominator = max(game_count, 1)
    return {
        "selection": float(selected["game_id"].nunique() / denominator),
        "start": float(selected["is_starting_lineup"].sum() / denominator),
        "captain": float(selected["team_captain"].sum() / denominator),
        "opportunity": float(played["minutes_played"].sum() / (90 * denominator)),
    }


def apply_availability(record: dict[str, object], features: list[str], available: bool) -> None:
    if not available:
        for feature in features:
            record[feature] = np.nan


def load_anchors() -> pd.DataFrame:
    frame = pd.read_csv(
        EXTENSIONS,
        usecols=[
            "capology_extension_event_id", "canonical_player_id", "canonical_club_id",
            "signed_date", "league", "canonical_position",
        ],
        low_memory=False,
    )
    frame["signed_date"] = pd.to_datetime(frame["signed_date"], errors="raise")
    frame["canonical_player_id"] = pd.to_numeric(frame["canonical_player_id"], errors="coerce")
    frame["canonical_club_id"] = pd.to_numeric(frame["canonical_club_id"], errors="coerce")
    return frame.sort_values(["signed_date", "capology_extension_event_id"]).reset_index(drop=True)


def load_club_games(anchors: pd.DataFrame) -> pd.DataFrame:
    clubs = set(anchors["canonical_club_id"].dropna().astype(int))
    earliest = anchors["signed_date"].min() - pd.Timedelta(days=LOOKBACK_DAYS)
    latest = anchors["signed_date"].max()
    games = pd.read_csv(
        GAMES,
        usecols=[
            "game_id", "date", "home_club_id", "away_club_id",
            "home_club_manager_name", "away_club_manager_name",
            "home_club_formation", "away_club_formation", "is_national_team_game",
        ],
        low_memory=False,
    )
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    games = games.loc[
        games["date"].between(earliest, latest, inclusive="left")
        & ~as_bool(games["is_national_team_game"])
    ].copy()
    home = games[["game_id", "date", "home_club_id", "home_club_manager_name", "home_club_formation"]].copy()
    away = games[["game_id", "date", "away_club_id", "away_club_manager_name", "away_club_formation"]].copy()
    home.columns = away.columns = ["game_id", "date", "club_id", "manager_name", "formation_reported"]
    frame = pd.concat([home, away], ignore_index=True)
    frame["club_id"] = pd.to_numeric(frame["club_id"], errors="coerce")
    frame = frame.loc[frame["club_id"].isin(clubs)].drop_duplicates(["game_id", "club_id"])
    frame["manager_key"] = frame["manager_name"].map(normalize_manager)
    frame["formation_shape"] = frame["formation_reported"].map(formation_shape)
    return frame.sort_values(["club_id", "date", "game_id"]).reset_index(drop=True)


def load_lineups(anchors: pd.DataFrame) -> pd.DataFrame:
    clubs = set(anchors["canonical_club_id"].dropna().astype(int))
    earliest = anchors["signed_date"].min() - pd.Timedelta(days=LOOKBACK_DAYS)
    latest = anchors["signed_date"].max()
    frame = pd.read_csv(
        LINEUPS,
        usecols=["game_id", "date", "player_id", "club_id", "team_captain", "is_starting_lineup"],
        low_memory=False,
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["club_id"] = pd.to_numeric(frame["club_id"], errors="coerce")
    frame["player_id"] = pd.to_numeric(frame["player_id"], errors="coerce")
    frame["is_starting_lineup"] = as_bool(frame["is_starting_lineup"])
    frame["team_captain"] = pd.to_numeric(frame["team_captain"], errors="coerce").fillna(0).gt(0)
    return frame.loc[
        frame["club_id"].isin(clubs)
        & frame["date"].between(earliest, latest, inclusive="left")
    ].sort_values(["club_id", "date", "game_id", "player_id"]).reset_index(drop=True)


def load_appearances(anchors: pd.DataFrame) -> pd.DataFrame:
    clubs = set(anchors["canonical_club_id"].dropna().astype(int))
    earliest = anchors["signed_date"].min() - pd.Timedelta(days=LOOKBACK_DAYS)
    latest = anchors["signed_date"].max()
    frame = pd.read_csv(
        APPEARANCES,
        usecols=["game_id", "date", "player_id", "player_club_id", "minutes_played"],
        low_memory=False,
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["player_club_id"] = pd.to_numeric(frame["player_club_id"], errors="coerce")
    frame["player_id"] = pd.to_numeric(frame["player_id"], errors="coerce")
    frame["minutes_played"] = pd.to_numeric(frame["minutes_played"], errors="coerce").fillna(0).clip(0, 130)
    return frame.loc[
        frame["player_club_id"].isin(clubs)
        & frame["date"].between(earliest, latest, inclusive="left")
    ].sort_values(["player_club_id", "date", "game_id", "player_id"]).reset_index(drop=True)


def profile_anchor(
    anchor: pd.Series,
    club_games: pd.DataFrame,
    club_lineups: pd.DataFrame,
    club_appearances: pd.DataFrame,
) -> tuple[dict[str, object], dict[str, object]]:
    decision = anchor["signed_date"]
    start = decision - pd.Timedelta(days=LOOKBACK_DAYS)
    recent_start = decision - pd.Timedelta(days=RECENT_DAYS)
    player_id = int(anchor["canonical_player_id"]) if pd.notna(anchor["canonical_player_id"]) else None
    games = club_games.loc[club_games["date"].ge(start) & club_games["date"].lt(decision)].sort_values(["date", "game_id"])
    manager_games = games.dropna(subset=["manager_key"]).copy()
    formation_games = games.dropna(subset=["formation_shape"]).copy()
    recent_games = games.loc[games["date"].ge(recent_start)]
    prior_games = games.loc[games["date"].lt(recent_start)]
    recent_formations = recent_games.dropna(subset=["formation_shape"])
    prior_formations = prior_games.dropna(subset=["formation_shape"])

    manager_coverage = len(manager_games) / len(games) if len(games) else np.nan
    formation_coverage = len(formation_games) / len(games) if len(games) else np.nan
    recent_formation_coverage = len(recent_formations) / len(recent_games) if len(recent_games) else np.nan
    prior_formation_coverage = len(prior_formations) / len(prior_games) if len(prior_games) else np.nan
    manager_available = bool(
        len(games) >= MIN_CLUB_GAMES
        and pd.notna(manager_coverage)
        and manager_coverage >= MIN_SOURCE_GAME_COVERAGE
    )
    formation_available = bool(
        len(games) >= MIN_CLUB_GAMES
        and len(recent_games) >= MIN_SPLIT_GAMES
        and len(prior_games) >= MIN_SPLIT_GAMES
        and pd.notna(formation_coverage)
        and formation_coverage >= MIN_SOURCE_GAME_COVERAGE
        and recent_formation_coverage >= MIN_SOURCE_GAME_COVERAGE
        and prior_formation_coverage >= MIN_SOURCE_GAME_COVERAGE
    )

    manager_keys = manager_games["manager_key"].to_numpy(object)
    change_locations = np.where(manager_keys[1:] != manager_keys[:-1])[0] + 1 if len(manager_keys) > 1 else np.array([], dtype=int)
    last_change = int(change_locations[-1]) if len(change_locations) else None
    current_manager_spell = manager_games.iloc[last_change:] if last_change is not None else manager_games
    previous_manager_start = int(change_locations[-2]) if len(change_locations) > 1 else 0
    previous_manager_spell = manager_games.iloc[previous_manager_start:last_change] if last_change is not None else manager_games.iloc[0:0]
    current_manager_name = current_manager_spell.iloc[-1]["manager_name"] if len(current_manager_spell) else None
    manager_changes = len(change_locations)
    manager_change_days = (
        (decision - current_manager_spell["date"].min()).days if last_change is not None and len(current_manager_spell) else None
    )

    formation_keys = formation_games["formation_shape"].to_numpy(object)
    formation_change_count = int(np.sum(formation_keys[1:] != formation_keys[:-1])) if len(formation_keys) > 1 else 0
    formation_counts = formation_games["formation_shape"].value_counts()
    dominant_formation_share = float(formation_counts.iloc[0] / len(formation_games)) if len(formation_games) else np.nan
    if len(formation_games):
        current_formation = formation_games.iloc[-1]["formation_shape"]
        reversed_shapes = formation_games.iloc[::-1]["formation_shape"].to_numpy(object)
        formation_break = np.where(reversed_shapes != current_formation)[0]
        current_formation_spell_games = int(formation_break[0]) if len(formation_break) else len(formation_games)
    else:
        current_formation = None
        current_formation_spell_games = 0

    current_ids = set(current_manager_spell["game_id"].astype(int))
    current_lineups = club_lineups.loc[club_lineups["game_id"].isin(current_ids)]
    current_apps = club_appearances.loc[club_appearances["game_id"].isin(current_ids)]
    current_lineup_coverage = source_coverage(current_ids, current_lineups)
    current_appearance_coverage = source_coverage(current_ids, current_apps)
    current_role_available = bool(
        manager_available
        and player_id is not None
        and len(current_manager_spell) >= MIN_CURRENT_MANAGER_GAMES
        and current_lineup_coverage >= MIN_SOURCE_GAME_COVERAGE
        and current_appearance_coverage >= MIN_SOURCE_GAME_COVERAGE
    )
    current_rates = (
        player_rates(current_lineups, current_apps, player_id, len(current_manager_spell))
        if player_id is not None and len(current_manager_spell)
        else {name: np.nan for name in ["selection", "start", "captain", "opportunity"]}
    )

    transition_observed = last_change is not None
    if transition_observed:
        previous_ids = set(previous_manager_spell["game_id"].astype(int))
        previous_lineups = club_lineups.loc[club_lineups["game_id"].isin(previous_ids)]
        previous_apps = club_appearances.loc[club_appearances["game_id"].isin(previous_ids)]
        previous_lineup_coverage = source_coverage(previous_ids, previous_lineups)
        previous_appearance_coverage = source_coverage(previous_ids, previous_apps)
        transition_available = bool(
            manager_available
            and player_id is not None
            and len(current_manager_spell) >= MIN_TRANSITION_SIDE_GAMES
            and len(previous_manager_spell) >= MIN_TRANSITION_SIDE_GAMES
            and current_lineup_coverage >= MIN_SOURCE_GAME_COVERAGE
            and current_appearance_coverage >= MIN_SOURCE_GAME_COVERAGE
            and previous_lineup_coverage >= MIN_SOURCE_GAME_COVERAGE
            and previous_appearance_coverage >= MIN_SOURCE_GAME_COVERAGE
        )
        previous_rates = (
            player_rates(previous_lineups, previous_apps, player_id, len(previous_manager_spell))
            if player_id is not None and len(previous_manager_spell)
            else {name: np.nan for name in ["selection", "start", "captain", "opportunity"]}
        )
        transition_days_scaled = min(max(manager_change_days or 0, 0), LOOKBACK_DAYS) / LOOKBACK_DAYS
        selection_change = current_rates["selection"] - previous_rates["selection"]
        start_change = current_rates["start"] - previous_rates["start"]
        opportunity_change = current_rates["opportunity"] - previous_rates["opportunity"]
    else:
        previous_lineup_coverage = np.nan
        previous_appearance_coverage = np.nan
        transition_available = bool(manager_available and player_id is not None)
        transition_days_scaled = 1.0
        selection_change = start_change = opportunity_change = 0.0

    manager_spell_days = (
        min(max((decision - current_manager_spell["date"].min()).days, 0), LOOKBACK_DAYS)
        if len(current_manager_spell)
        else np.nan
    )
    recent_change_90 = bool(
        transition_observed and manager_change_days is not None and manager_change_days <= 90
    )
    record: dict[str, object] = {
        "capology_extension_event_id": anchor["capology_extension_event_id"],
        "canonical_player_id": player_id if player_id is not None else np.nan,
        "canonical_club_id": int(anchor["canonical_club_id"]) if pd.notna(anchor["canonical_club_id"]) else np.nan,
        "signed_date": decision.date().isoformat(),
        "signed_year": int(decision.year),
        "league": anchor["league"],
        "canonical_position": anchor["canonical_position"],
        "club_games_365d": len(games),
        "manager_coded_games_365d": len(manager_games),
        "formation_coded_games_365d": len(formation_games),
        "manager_game_coverage_365d": manager_coverage,
        "formation_game_coverage_365d": formation_coverage,
        "current_manager_name": current_manager_name,
        "current_manager_spell_games_365d": len(current_manager_spell),
        "previous_manager_spell_games_365d": len(previous_manager_spell),
        "current_manager_lineup_game_coverage": current_lineup_coverage,
        "current_manager_appearance_game_coverage": current_appearance_coverage,
        "previous_manager_lineup_game_coverage": previous_lineup_coverage,
        "previous_manager_appearance_game_coverage": previous_appearance_coverage,
        "current_formation_shape": current_formation,
        "manager_stability_available": manager_available,
        "formation_stability_available": formation_available,
        "current_manager_player_role_available": current_role_available,
        "manager_transition_role_change_available": transition_available,
        "manager_unique_count_365d": int(manager_games["manager_key"].nunique()),
        "manager_unique_count_365d_log1p": math.log1p(manager_games["manager_key"].nunique()),
        "manager_change_count_365d": manager_changes,
        "manager_change_count_365d_log1p": math.log1p(manager_changes),
        "current_manager_observed_match_share_365d": len(current_manager_spell) / len(manager_games) if len(manager_games) else np.nan,
        "current_manager_spell_games_365d_log1p": math.log1p(len(current_manager_spell)),
        "current_manager_spell_days_scaled_365d": manager_spell_days / LOOKBACK_DAYS if pd.notna(manager_spell_days) else np.nan,
        "manager_change_within_90d": float(recent_change_90),
        "formation_unique_shape_count_365d": int(formation_games["formation_shape"].nunique()),
        "formation_unique_shape_count_365d_log1p": math.log1p(formation_games["formation_shape"].nunique()),
        "formation_shape_entropy_365d": normalized_entropy(formation_games["formation_shape"]),
        "formation_shape_change_rate_365d": formation_change_count / max(len(formation_games) - 1, 1),
        "dominant_formation_shape_share_365d": dominant_formation_share,
        "current_formation_shape_spell_share_365d": current_formation_spell_games / len(formation_games) if len(formation_games) else np.nan,
        "formation_distribution_shift_recent_vs_prior": distribution_shift(recent_formations["formation_shape"], prior_formations["formation_shape"]),
        "current_manager_player_selection_rate": current_rates["selection"],
        "current_manager_player_start_rate": current_rates["start"],
        "current_manager_player_opportunity_share": current_rates["opportunity"],
        "current_manager_player_captain_rate": current_rates["captain"],
        "manager_transition_observed_365d": float(transition_observed),
        "days_since_manager_transition_scaled_365d": transition_days_scaled,
        "player_selection_rate_change_post_vs_pre_manager": selection_change,
        "player_start_rate_change_post_vs_pre_manager": start_change,
        "player_opportunity_share_change_post_vs_pre_manager": opportunity_change,
    }
    apply_availability(record, MANAGER_STABILITY_FEATURES, manager_available)
    apply_availability(record, FORMATION_STABILITY_FEATURES, formation_available)
    apply_availability(record, CURRENT_MANAGER_PLAYER_ROLE_FEATURES, current_role_available)
    apply_availability(record, MANAGER_TRANSITION_ROLE_CHANGE_FEATURES, transition_available)

    audit = {
        "capology_extension_event_id": anchor["capology_extension_event_id"],
        "signed_date": decision.date().isoformat(),
        "latest_game_date_used": games["date"].max().date().isoformat() if len(games) else None,
        "latest_manager_game_date_used": manager_games["date"].max().date().isoformat() if len(manager_games) else None,
        "latest_formation_game_date_used": formation_games["date"].max().date().isoformat() if len(formation_games) else None,
        "latest_current_manager_lineup_date_used": current_lineups["date"].max().date().isoformat() if len(current_lineups) else None,
        "latest_current_manager_appearance_date_used": current_apps["date"].max().date().isoformat() if len(current_apps) else None,
        "manager_stability_available": manager_available,
        "formation_stability_available": formation_available,
        "current_manager_player_role_available": current_role_available,
        "manager_transition_role_change_available": transition_available,
    }
    return record, audit


def build_profiles() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    anchors = load_anchors()
    games = load_club_games(anchors)
    lineups = load_lineups(anchors)
    appearances = load_appearances(anchors)
    game_groups = {int(key): value for key, value in games.groupby("club_id")}
    lineup_groups = {int(key): value for key, value in lineups.groupby("club_id")}
    appearance_groups = {int(key): value for key, value in appearances.groupby("player_club_id")}
    empty_games, empty_lineups, empty_appearances = games.iloc[0:0], lineups.iloc[0:0], appearances.iloc[0:0]
    profiles, audits = [], []
    for _, anchor in anchors.iterrows():
        club = int(anchor["canonical_club_id"]) if pd.notna(anchor["canonical_club_id"]) else None
        record, audit = profile_anchor(
            anchor,
            game_groups.get(club, empty_games),
            lineup_groups.get(club, empty_lineups),
            appearance_groups.get(club, empty_appearances),
        )
        profiles.append(record)
        audits.append(audit)
    counts = {
        "anchor_rows": len(anchors),
        "relevant_club_games": len(games),
        "relevant_lineup_rows": len(lineups),
        "relevant_appearance_rows": len(appearances),
    }
    return pd.DataFrame(profiles), pd.DataFrame(audits), counts


def feature_dictionary() -> pd.DataFrame:
    definitions = {
        "manager_unique_count_365d_log1p": "log(1 + distinct observed managers in the prior-year club games).",
        "manager_change_count_365d_log1p": "log(1 + adjacent changes in manager-coded club games).",
        "current_manager_observed_match_share_365d": "Current manager's trailing consecutive games / manager-coded games.",
        "current_manager_spell_games_365d_log1p": "log(1 + trailing consecutive games under the current observed manager).",
        "current_manager_spell_days_scaled_365d": "Days since the first match of the observed current-manager spell, capped and divided by 365.",
        "manager_change_within_90d": "One when the latest observed manager transition is within 90 days.",
        "formation_unique_shape_count_365d_log1p": "log(1 + distinct normalized base formation shapes).",
        "formation_shape_entropy_365d": "Normalized entropy of reported base formation shapes.",
        "formation_shape_change_rate_365d": "Adjacent base-shape changes / possible observed formation transitions.",
        "dominant_formation_shape_share_365d": "Share of formation-coded games using the most common base shape.",
        "current_formation_shape_spell_share_365d": "Trailing consecutive current-shape games / formation-coded games.",
        "formation_distribution_shift_recent_vs_prior": "Total-variation distance between recent-180-day and prior-185-day formation distributions.",
        "current_manager_player_selection_rate": "Player matchday selections / games in the current observed manager spell.",
        "current_manager_player_start_rate": "Player starts / games in the current observed manager spell.",
        "current_manager_player_opportunity_share": "Player minutes / 90-minute opportunities in the current observed manager spell.",
        "current_manager_player_captain_rate": "Player captain selections / games in the current observed manager spell.",
        "manager_transition_observed_365d": "One when at least one observed manager transition occurred in the prior year.",
        "days_since_manager_transition_scaled_365d": "Days since latest transition / 365; one when no transition was observed.",
        "player_selection_rate_change_post_vs_pre_manager": "Current-manager selection rate minus immediately previous-manager rate.",
        "player_start_rate_change_post_vs_pre_manager": "Current-manager start rate minus immediately previous-manager rate.",
        "player_opportunity_share_change_post_vs_pre_manager": "Current-manager opportunity share minus immediately previous-manager share.",
    }
    block = {}
    for name in MANAGER_STABILITY_FEATURES:
        block[name] = "manager_stability"
    for name in FORMATION_STABILITY_FEATURES:
        block[name] = "formation_tactical_stability"
    for name in CURRENT_MANAGER_PLAYER_ROLE_FEATURES:
        block[name] = "player_under_current_manager"
    for name in MANAGER_TRANSITION_ROLE_CHANGE_FEATURES:
        block[name] = "role_change_at_manager_transition"
    return pd.DataFrame(
        [
            {
                "feature": name,
                "block": block[name],
                "definition": definitions[name],
                "timing": "strictly pre-signing; source-gated",
            }
            for name in MODEL_FEATURES
        ]
    )


def build_checks(profiles: pd.DataFrame, audits: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(check: str, passed: bool, observed: object, expected: object, note: str) -> None:
        rows.append({"check": check, "passed": bool(passed), "observed": observed, "expected": expected, "note": note})

    add(
        "event_grain",
        len(profiles) == 2805 and profiles["capology_extension_event_id"].is_unique,
        {"rows": len(profiles), "duplicates": int(profiles["capology_extension_event_id"].duplicated().sum())},
        {"rows": 2805, "duplicates": 0},
        "Canonical extension grain is preserved.",
    )
    signing = pd.to_datetime(audits["signed_date"], errors="raise")
    for field in [
        "latest_game_date_used", "latest_manager_game_date_used", "latest_formation_game_date_used",
        "latest_current_manager_lineup_date_used", "latest_current_manager_appearance_date_used",
    ]:
        dates = pd.to_datetime(audits[field], errors="coerce")
        add(
            f"{field}_predecision",
            bool((dates.dropna() < signing.loc[dates.notna()]).all()),
            int((dates >= signing).fillna(False).sum()),
            0,
            "All operational evidence must be strictly before signing.",
        )
    availability = {
        "manager_stability_available": MANAGER_STABILITY_FEATURES,
        "formation_stability_available": FORMATION_STABILITY_FEATURES,
        "current_manager_player_role_available": CURRENT_MANAGER_PLAYER_ROLE_FEATURES,
        "manager_transition_role_change_available": MANAGER_TRANSITION_ROLE_CHANGE_FEATURES,
    }
    for flag, features in availability.items():
        expected = profiles[flag].astype(bool)
        observed = profiles[features].notna().all(axis=1)
        add(f"{flag}_matches_features", expected.equals(observed), int((expected != observed).sum()), 0, "Availability is explicit and reconstructible.")
    unit_features = [
        "current_manager_observed_match_share_365d", "current_manager_spell_days_scaled_365d",
        "manager_change_within_90d", "formation_shape_entropy_365d",
        "formation_shape_change_rate_365d", "dominant_formation_shape_share_365d",
        "current_formation_shape_spell_share_365d", "formation_distribution_shift_recent_vs_prior",
        "current_manager_player_selection_rate", "current_manager_player_start_rate",
        "current_manager_player_captain_rate", "manager_transition_observed_365d",
        "days_since_manager_transition_scaled_365d",
    ]
    add("unit_features_bounded", profiles[unit_features].stack().between(0, 1.05).all(), True, True, "Unit-scaled values remain bounded.")
    delta_features = [
        "player_selection_rate_change_post_vs_pre_manager",
        "player_start_rate_change_post_vs_pre_manager",
        "player_opportunity_share_change_post_vs_pre_manager",
    ]
    add("transition_deltas_bounded", profiles[delta_features].stack().between(-1.5, 1.5).all(), True, True, "Role changes remain within plausible bounds.")
    add("no_targets", not any(column.startswith("target_") or "movement_state" in column for column in profiles), True, True, "Feature build contains no outcomes.")
    finite = profiles[MODEL_FEATURES].to_numpy(float)
    add("finite_model_features", bool(np.isfinite(finite[~np.isnan(finite)]).all()), True, True, "No infinite model input.")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    unchanged = all(sha256(ROOT / item["path"]) == item["sha256"] for item in freeze["files"])
    add("v1_frozen", unchanged, unchanged, True, "All frozen V1 files remain byte-identical.")
    return pd.DataFrame(rows)


def write_readme(output: Path, profiles: pd.DataFrame, counts: dict[str, int]) -> None:
    development = profiles.loc[profiles["signed_year"].between(2020, 2023)]
    flags = [
        "manager_stability_available", "formation_stability_available",
        "current_manager_player_role_available", "manager_transition_role_change_available",
    ]
    coverage = {flag: float(development[flag].mean()) for flag in flags}
    (output / "README.md").write_text(
        f"""# Engine V2 manager and tactical profiles

**Status:** research feature build; no model promoted or deployed.

The build converts exact dated club games, manager labels, reported formation
shapes, player lineups, and appearances into decision-time profiles for all
{len(profiles):,} canonical extensions.

## 2020–2023 availability

- manager stability: **{coverage['manager_stability_available']:.1%}**;
- formation/tactical stability: **{coverage['formation_stability_available']:.1%}**;
- player role under the current manager: **{coverage['current_manager_player_role_available']:.1%}**;
- player role change around the latest manager transition: **{coverage['manager_transition_role_change_available']:.1%}**.

Manager tenure means the trailing uninterrupted manager-coded match spell, not
an official appointment date. Formation modifiers are reduced to base numeric
shape. Zero player involvement is accepted only after lineup and appearance
coverage passes the 80% gate.

## Source rows retained

- club games: {counts['relevant_club_games']:,}
- lineup rows: {counts['relevant_lineup_rows']:,}
- appearance rows: {counts['relevant_appearance_rows']:,}

The 2024+ outcome holdout is not read by this build.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    before = {item["path"]: sha256(ROOT / item["path"]) for item in freeze["files"]}
    profiles, audits, counts = build_profiles()
    dictionary = feature_dictionary()
    checks = build_checks(profiles, audits)
    profiles.to_csv(output / "manager_tactical_profiles.csv", index=False)
    audits.to_csv(output / "asof_evidence_audit.csv", index=False)
    dictionary.to_csv(output / "feature_dictionary.csv", index=False)
    checks.to_csv(output / "build_checks.csv", index=False)
    write_json(output / "feature_contract.json", contract_payload())
    write_readme(output, profiles, counts)
    sources = [
        EXTENSIONS, GAMES, LINEUPS, APPEARANCES,
        ROOT / "models" / "engine_v2" / "manager_tactical_feature_contract.py",
        ROOT / "scripts" / "build_engine_v2_manager_tactical_profiles.py",
    ]
    pd.DataFrame(
        [{"source": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in sources]
    ).to_csv(output / "source_manifest.csv", index=False)
    after = {path: sha256(ROOT / path) for path in before}
    summary = {
        **counts,
        "contract_version": CONTRACT_VERSION,
        "profile_rows": len(profiles),
        "development_availability": {flag: float(development) for flag, development in {
            flag: profiles.loc[profiles["signed_year"].between(2020, 2023), flag].mean()
            for flag in [
                "manager_stability_available", "formation_stability_available",
                "current_manager_player_role_available", "manager_transition_role_change_available",
            ]
        }.items()},
        "final_holdout_opened": False,
        "deployment_changed": False,
        "v1_unchanged": before == after,
        "checks_passed": int(checks["passed"].sum()),
        "checks_total": len(checks),
    }
    write_json(output / "run_summary.json", summary)
    manifest = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "output_manifest.csv":
            manifest.append({"output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest).to_csv(output / "output_manifest.csv", index=False)
    if not checks["passed"].all():
        raise RuntimeError(checks.loc[~checks["passed"], ["check", "observed", "expected"]].to_dict("records"))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

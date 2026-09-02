"""Build decision-time player role and squad competition profiles.

Zero involvement is emitted only when the club's own lineup/appearance source
coverage passes the frozen gate. Otherwise the relevant block is unavailable.
No post-signing record or 2024+ outcome is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.engine_v2.role_squad_feature_contract import (  # noqa: E402
    CONTRACT_VERSION,
    LOOKBACK_DAYS,
    MAX_VALUATION_AGE_DAYS,
    MIN_ACTIVE_SELECTIONS,
    MIN_CLUB_GAMES,
    MIN_KNOWN_POSITION_VALUES,
    MIN_SOURCE_GAME_COVERAGE,
    MIN_SPLIT_GAMES,
    MODEL_FEATURES,
    PRIOR_DAYS,
    RECENT_DAYS,
    ROLE_LEVEL_FEATURES,
    ROLE_TRAJECTORY_FEATURES,
    SQUAD_ONFIELD_FEATURES,
    SQUAD_VALUE_INCOMING_FEATURES,
    contract_payload,
)


DEFAULT_OUTPUT = ROOT / "Data" / "processed" / "engine_v2_role_squad_profiles"
P = ROOT / "Data" / "processed"
EXTENSIONS = P / "capology_contracts" / "canonical_extension_events.csv"
LINEUPS = P / "transfermarkt_clean" / "tables" / "game_lineups_clean.csv"
APPEARANCES = P / "transfermarkt_clean" / "tables" / "appearances_clean.csv"
GAMES = P / "transfermarkt_clean" / "tables" / "games_clean.csv"
PLAYERS = P / "canonical_integration" / "canonical_player_dimension.csv"
VALUATIONS = P / "canonical_integration" / "canonical_valuation_history.csv"
TRANSFERS = P / "canonical_integration" / "canonical_transfer_history.csv"
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


def broad_position(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if "goal" in text or text in {"k", "gk"}:
        return "Goalkeeper"
    if any(token in text for token in ["back", "defender", "defence", "sweeper"]):
        return "Defender"
    if "midfield" in text or text in {"m", "cm", "dm", "am"}:
        return "Midfield"
    if any(token in text for token in ["winger", "forward", "striker", "attack"]):
        return "Attack"
    return None


def normalized_entropy(values: pd.Series) -> float:
    counts = values.dropna().astype(str).value_counts()
    if len(counts) < 2:
        return 0.0
    probability = counts.to_numpy(float) / counts.sum()
    return float(-(probability * np.log(probability)).sum() / np.log(len(counts)))


def percentile_at_or_below(values: list[float], target: float) -> float:
    if not values:
        return np.nan
    array = np.asarray(values, dtype=float)
    return float(np.mean(array <= target))


def load_anchors() -> pd.DataFrame:
    columns = [
        "capology_extension_event_id", "canonical_player_id", "canonical_club_id",
        "canonical_position", "signed_date", "league",
    ]
    frame = pd.read_csv(EXTENSIONS, usecols=columns, low_memory=False)
    frame["signed_date"] = pd.to_datetime(frame["signed_date"], errors="raise")
    frame["canonical_player_id"] = pd.to_numeric(frame["canonical_player_id"], errors="coerce")
    frame["canonical_club_id"] = pd.to_numeric(frame["canonical_club_id"], errors="coerce")
    frame["canonical_position"] = frame["canonical_position"].map(broad_position)
    return frame.sort_values(["signed_date", "capology_extension_event_id"]).reset_index(drop=True)


def player_positions() -> pd.DataFrame:
    frame = pd.read_csv(
        PLAYERS,
        usecols=["canonical_player_id", "canonical_sub_position", "canonical_position"],
        low_memory=False,
    )
    frame["player_broad_position"] = frame["canonical_sub_position"].map(broad_position)
    frame["player_broad_position"] = frame["player_broad_position"].fillna(frame["canonical_position"].map(broad_position))
    return frame[["canonical_player_id", "player_broad_position"]]


def load_games(anchors: pd.DataFrame) -> pd.DataFrame:
    clubs = set(anchors["canonical_club_id"].dropna().astype(int))
    earliest = anchors["signed_date"].min() - pd.Timedelta(days=LOOKBACK_DAYS)
    latest = anchors["signed_date"].max()
    games = pd.read_csv(
        GAMES,
        usecols=["game_id", "date", "home_club_id", "away_club_id", "is_national_team_game"],
        low_memory=False,
    )
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    games = games.loc[games["date"].between(earliest, latest, inclusive="left") & ~as_bool(games["is_national_team_game"])].copy()
    home = games[["game_id", "date", "home_club_id"]].rename(columns={"home_club_id": "club_id"})
    away = games[["game_id", "date", "away_club_id"]].rename(columns={"away_club_id": "club_id"})
    clubs_games = pd.concat([home, away], ignore_index=True)
    clubs_games["club_id"] = pd.to_numeric(clubs_games["club_id"], errors="coerce")
    return clubs_games.loc[clubs_games["club_id"].isin(clubs)].drop_duplicates(["game_id", "club_id"]).sort_values(["club_id", "date", "game_id"])


def load_lineups(anchors: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    clubs = set(anchors["canonical_club_id"].dropna().astype(int))
    earliest = anchors["signed_date"].min() - pd.Timedelta(days=LOOKBACK_DAYS)
    latest = anchors["signed_date"].max()
    frame = pd.read_csv(
        LINEUPS,
        usecols=["game_lineups_id", "date", "game_id", "player_id", "club_id", "type", "position", "team_captain", "is_starting_lineup", "is_substitute"],
        low_memory=False,
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["club_id"] = pd.to_numeric(frame["club_id"], errors="coerce")
    frame["player_id"] = pd.to_numeric(frame["player_id"], errors="coerce")
    frame = frame.loc[frame["club_id"].isin(clubs) & frame["date"].between(earliest, latest, inclusive="left")].copy()
    frame["is_starting_lineup"] = as_bool(frame["is_starting_lineup"])
    frame["is_substitute"] = as_bool(frame["is_substitute"])
    frame["team_captain"] = pd.to_numeric(frame["team_captain"], errors="coerce").fillna(0).gt(0)
    frame["observed_broad_position"] = frame["position"].map(broad_position)
    frame = frame.merge(positions.rename(columns={"canonical_player_id": "player_id"}), on="player_id", how="left", validate="many_to_one")
    frame["broad_position"] = frame["observed_broad_position"].fillna(frame["player_broad_position"])
    return frame.sort_values(["club_id", "date", "game_id", "player_id"]).reset_index(drop=True)


def load_appearances(anchors: pd.DataFrame) -> pd.DataFrame:
    clubs = set(anchors["canonical_club_id"].dropna().astype(int))
    earliest = anchors["signed_date"].min() - pd.Timedelta(days=LOOKBACK_DAYS)
    latest = anchors["signed_date"].max()
    frame = pd.read_csv(
        APPEARANCES,
        usecols=["appearance_id", "game_id", "player_id", "player_club_id", "date", "minutes_played"],
        low_memory=False,
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["player_club_id"] = pd.to_numeric(frame["player_club_id"], errors="coerce")
    frame["player_id"] = pd.to_numeric(frame["player_id"], errors="coerce")
    frame["minutes_played"] = pd.to_numeric(frame["minutes_played"], errors="coerce").fillna(0).clip(lower=0, upper=130)
    return frame.loc[frame["player_club_id"].isin(clubs) & frame["date"].between(earliest, latest, inclusive="left")].sort_values(["player_club_id", "date", "game_id", "player_id"]).reset_index(drop=True)


def load_incoming_transfers(anchors: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    clubs = set(anchors["canonical_club_id"].dropna().astype(int))
    earliest = anchors["signed_date"].min() - pd.Timedelta(days=RECENT_DAYS)
    latest = anchors["signed_date"].max()
    frame = pd.read_csv(
        TRANSFERS,
        usecols=["canonical_transfer_event_id", "canonical_player_id", "transfer_date", "to_club_id", "canonical_transfer_type", "canonical_market_value_eur", "event_conflict", "transfer_type_conflict"],
        low_memory=False,
    )
    frame["transfer_date"] = pd.to_datetime(frame["transfer_date"], errors="coerce")
    frame["to_club_id"] = pd.to_numeric(frame["to_club_id"], errors="coerce")
    clean = ~as_bool(frame["event_conflict"]) & ~as_bool(frame["transfer_type_conflict"])
    typed = frame["canonical_transfer_type"].isin(["transfer", "loan", "loan_return"])
    frame = frame.loc[clean & typed & frame["to_club_id"].isin(clubs) & frame["transfer_date"].between(earliest, latest, inclusive="left")].copy()
    frame["canonical_market_value_eur"] = pd.to_numeric(frame["canonical_market_value_eur"], errors="coerce")
    frame = frame.merge(positions, on="canonical_player_id", how="left", validate="many_to_one")
    return frame.sort_values(["to_club_id", "transfer_date", "canonical_transfer_event_id"]).reset_index(drop=True)


def load_valuations(player_ids: set[int], anchors: pd.DataFrame) -> pd.DataFrame:
    earliest = anchors["signed_date"].min() - pd.Timedelta(days=MAX_VALUATION_AGE_DAYS)
    latest = anchors["signed_date"].max()
    frame = pd.read_csv(
        VALUATIONS,
        usecols=["canonical_valuation_id", "canonical_player_id", "valuation_date", "canonical_market_value_eur", "valuation_date_valid"],
        low_memory=False,
    )
    frame["valuation_date"] = pd.to_datetime(frame["valuation_date"], errors="coerce")
    frame["canonical_market_value_eur"] = pd.to_numeric(frame["canonical_market_value_eur"], errors="coerce")
    valid = as_bool(frame["valuation_date_valid"])
    return frame.loc[
        frame["canonical_player_id"].isin(player_ids)
        & frame["valuation_date"].between(earliest, latest, inclusive="both")
        & frame["canonical_market_value_eur"].gt(0)
        & valid
    ].sort_values(["canonical_player_id", "valuation_date", "canonical_valuation_id"]).reset_index(drop=True)


def asof_value(group: pd.DataFrame, decision: pd.Timestamp) -> tuple[float, pd.Timestamp | None]:
    if group.empty:
        return np.nan, None
    eligible = group.loc[group["valuation_date"].le(decision)].tail(1)
    if eligible.empty:
        return np.nan, None
    row = eligible.iloc[0]
    age = (decision - row["valuation_date"]).days
    if age < 0 or age > MAX_VALUATION_AGE_DAYS:
        return np.nan, row["valuation_date"]
    return float(row["canonical_market_value_eur"]), row["valuation_date"]


def source_coverage(game_ids: set[int], evidence: pd.DataFrame) -> float:
    if not game_ids:
        return np.nan
    return float(evidence.loc[evidence["game_id"].isin(game_ids), "game_id"].nunique() / len(game_ids))


def rates_for_player(lineups: pd.DataFrame, apps: pd.DataFrame, player_id: int, game_count: int) -> dict[str, float]:
    selected = lineups.loc[lineups["player_id"].eq(player_id)]
    played = apps.loc[apps["player_id"].eq(player_id)]
    denominator = max(game_count, 1)
    return {
        "selection": float(selected["game_id"].nunique() / denominator),
        "start": float(selected["is_starting_lineup"].sum() / denominator),
        "substitute": float(selected["is_substitute"].sum() / denominator),
        "captain": float(selected["team_captain"].sum() / denominator),
        "opportunity": float(played["minutes_played"].sum() / (90 * denominator)),
    }


def apply_availability(record: dict[str, object], features: list[str], available: bool) -> None:
    if not available:
        for feature in features:
            record[feature] = np.nan


def profile_anchor(
    anchor: pd.Series,
    club_games: pd.DataFrame,
    club_lineups: pd.DataFrame,
    club_apps: pd.DataFrame,
    incoming: pd.DataFrame,
    valuation_groups: dict[int, pd.DataFrame],
) -> tuple[dict[str, object], dict[str, object]]:
    decision = anchor["signed_date"]
    start = decision - pd.Timedelta(days=LOOKBACK_DAYS)
    split = decision - pd.Timedelta(days=RECENT_DAYS)
    player_id = int(anchor["canonical_player_id"]) if pd.notna(anchor["canonical_player_id"]) else None
    target_position = anchor["canonical_position"]

    games = club_games.loc[club_games["date"].ge(start) & club_games["date"].lt(decision)]
    recent_games = games.loc[games["date"].ge(split)]
    prior_games = games.loc[games["date"].lt(split)]
    game_ids = set(games["game_id"].astype(int))
    recent_ids = set(recent_games["game_id"].astype(int))
    prior_ids = set(prior_games["game_id"].astype(int))
    lineups = club_lineups.loc[club_lineups["game_id"].isin(game_ids)]
    apps = club_apps.loc[club_apps["game_id"].isin(game_ids)]
    recent_lineups = lineups.loc[lineups["game_id"].isin(recent_ids)]
    prior_lineups = lineups.loc[lineups["game_id"].isin(prior_ids)]
    recent_apps = apps.loc[apps["game_id"].isin(recent_ids)]
    prior_apps = apps.loc[apps["game_id"].isin(prior_ids)]

    lineup_coverage = source_coverage(game_ids, lineups)
    appearance_coverage = source_coverage(game_ids, apps)
    recent_lineup_coverage = source_coverage(recent_ids, recent_lineups)
    prior_lineup_coverage = source_coverage(prior_ids, prior_lineups)
    recent_appearance_coverage = source_coverage(recent_ids, recent_apps)
    prior_appearance_coverage = source_coverage(prior_ids, prior_apps)
    role_level_available = bool(
        player_id is not None and len(games) >= MIN_CLUB_GAMES
        and pd.notna(lineup_coverage) and lineup_coverage >= MIN_SOURCE_GAME_COVERAGE
    )
    trajectory_available = bool(
        role_level_available and len(recent_games) >= MIN_SPLIT_GAMES and len(prior_games) >= MIN_SPLIT_GAMES
        and recent_lineup_coverage >= MIN_SOURCE_GAME_COVERAGE
        and prior_lineup_coverage >= MIN_SOURCE_GAME_COVERAGE
        and recent_appearance_coverage >= MIN_SOURCE_GAME_COVERAGE
        and prior_appearance_coverage >= MIN_SOURCE_GAME_COVERAGE
    )

    full_rates = rates_for_player(lineups, apps, player_id, len(games)) if player_id is not None and len(games) else {key: np.nan for key in ["selection", "start", "substitute", "captain", "opportunity"]}
    recent_rates = rates_for_player(recent_lineups, recent_apps, player_id, len(recent_games)) if player_id is not None and len(recent_games) else {key: np.nan for key in ["selection", "start", "substitute", "captain", "opportunity"]}
    prior_rates = rates_for_player(prior_lineups, prior_apps, player_id, len(prior_games)) if player_id is not None and len(prior_games) else {key: np.nan for key in ["selection", "start", "substitute", "captain", "opportunity"]}
    target_lineups = lineups.loc[lineups["player_id"].eq(player_id)] if player_id is not None else lineups.iloc[0:0]
    last_selection = target_lineups["date"].max() if len(target_lineups) else pd.NaT
    days_since = min((decision - last_selection).days, LOOKBACK_DAYS) if pd.notna(last_selection) else LOOKBACK_DAYS

    record: dict[str, object] = {
        "capology_extension_event_id": anchor["capology_extension_event_id"],
        "canonical_player_id": player_id if player_id is not None else np.nan,
        "canonical_club_id": int(anchor["canonical_club_id"]) if pd.notna(anchor["canonical_club_id"]) else np.nan,
        "signed_date": decision.date().isoformat(), "signed_year": int(decision.year),
        "league": anchor["league"], "canonical_position": target_position,
        "club_games_365d": len(games), "club_games_recent_180d": len(recent_games),
        "club_games_prior_185d": len(prior_games), "club_lineup_game_coverage_365d": lineup_coverage,
        "club_appearance_game_coverage_365d": appearance_coverage,
        "player_lineup_selections_365d": int(target_lineups["game_id"].nunique()),
        "player_starts_365d": int(target_lineups["is_starting_lineup"].sum()),
        "player_minutes_365d": float(apps.loc[apps["player_id"].eq(player_id), "minutes_played"].sum()) if player_id is not None else np.nan,
        "role_level_available": role_level_available,
        "role_trajectory_available": trajectory_available,
        "role_lineup_selection_rate_365d": full_rates["selection"],
        "role_start_rate_365d": full_rates["start"],
        "role_substitute_selection_rate_365d": full_rates["substitute"],
        "role_captain_rate_365d": full_rates["captain"],
        "role_position_entropy_365d": normalized_entropy(target_lineups["position"]),
        "role_days_since_last_selection_scaled_365d": float(days_since / LOOKBACK_DAYS),
        "role_recent_start_rate_180d": recent_rates["start"],
        "role_start_rate_change_recent_vs_prior": recent_rates["start"] - prior_rates["start"],
        "role_selection_rate_change_recent_vs_prior": recent_rates["selection"] - prior_rates["selection"],
        "role_opportunity_share_change_recent_vs_prior": recent_rates["opportunity"] - prior_rates["opportunity"],
    }
    apply_availability(record, ROLE_LEVEL_FEATURES, role_level_available)
    apply_availability(record, ROLE_TRAJECTORY_FEATURES, trajectory_available)

    # Reconstruct the decision-time squad from sufficiently repeated lineup
    # selections; this avoids treating one emergency bench appearance as a
    # stable positional competitor.
    if len(lineups):
        aggregate = lineups.groupby("player_id", as_index=True).agg(
            selections=("game_id", "nunique"), starts=("is_starting_lineup", "sum")
        )
        dominant_position = (
            lineups.dropna(subset=["broad_position"])
            .groupby(["player_id", "broad_position"]).size().rename("n").reset_index()
            .sort_values(["player_id", "n", "broad_position"], ascending=[True, False, True])
            .drop_duplicates("player_id").set_index("player_id")["broad_position"]
        )
        aggregate["broad_position"] = dominant_position
    else:
        aggregate = pd.DataFrame(columns=["selections", "starts", "broad_position"])
    minutes = apps.groupby("player_id")["minutes_played"].sum() if len(apps) else pd.Series(dtype=float)
    aggregate["minutes"] = minutes
    aggregate["minutes"] = aggregate["minutes"].fillna(0.0)
    active = aggregate.loc[aggregate["selections"].ge(MIN_ACTIVE_SELECTIONS) & aggregate["broad_position"].eq(target_position)].copy() if target_position else aggregate.iloc[0:0].copy()
    competitor_ids = [int(value) for value in active.index if player_id is None or int(value) != player_id]
    comparison_ids = sorted(set(competitor_ids + ([player_id] if player_id is not None else [])))
    starts_by_id = {int(index): float(value) for index, value in aggregate["starts"].items()}
    minutes_by_id = {int(index): float(value) for index, value in aggregate["minutes"].items()}
    target_starts = starts_by_id.get(player_id, 0.0) if player_id is not None else np.nan
    target_minutes = minutes_by_id.get(player_id, 0.0) if player_id is not None else np.nan
    start_rates = [starts_by_id.get(pid, 0.0) / max(len(games), 1) for pid in comparison_ids]
    opportunity_rates = [minutes_by_id.get(pid, 0.0) / (90 * max(len(games), 1)) for pid in comparison_ids]
    target_start_rate = target_starts / max(len(games), 1) if player_id is not None else np.nan
    target_opportunity = target_minutes / (90 * max(len(games), 1)) if player_id is not None else np.nan
    total_position_minutes = float(sum(minutes_by_id.get(pid, 0.0) for pid in comparison_ids))
    position_shares = [minutes_by_id.get(pid, 0.0) / total_position_minutes for pid in comparison_ids] if total_position_minutes > 0 else []
    onfield_available = bool(
        player_id is not None and target_position is not None and len(games) >= MIN_CLUB_GAMES
        and lineup_coverage >= MIN_SOURCE_GAME_COVERAGE
        and appearance_coverage >= MIN_SOURCE_GAME_COVERAGE
    )
    strong_competitors = sum(starts_by_id.get(pid, 0.0) / max(len(games), 1) >= 0.25 for pid in competitor_ids)
    record.update({
        "squad_onfield_available": onfield_available,
        "squad_same_position_competitor_count": len(competitor_ids),
        "squad_same_position_competitor_count_log1p": math.log1p(len(competitor_ids)),
        "squad_same_position_strong_competitor_count": int(strong_competitors),
        "squad_same_position_strong_competitor_count_log1p": math.log1p(strong_competitors),
        "player_same_position_start_percentile": percentile_at_or_below(start_rates, target_start_rate),
        "player_same_position_minutes_percentile": percentile_at_or_below(opportunity_rates, target_opportunity),
        "player_start_rate_gap_to_position_leader": max(start_rates, default=0.0) - target_start_rate,
        "player_opportunity_gap_to_position_leader": max(opportunity_rates, default=0.0) - target_opportunity,
        "player_share_of_position_minutes": target_minutes / total_position_minutes if total_position_minutes > 0 else 0.0,
        "squad_position_minutes_hhi": float(sum(share**2 for share in position_shares)) if position_shares else 0.0,
    })
    apply_availability(record, SQUAD_ONFIELD_FEATURES, onfield_available)

    values: dict[int, tuple[float, pd.Timestamp | None]] = {
        pid: asof_value(valuation_groups.get(pid, pd.DataFrame()), decision) for pid in comparison_ids
    }
    target_value, target_value_date = values.get(player_id, (np.nan, None)) if player_id is not None else (np.nan, None)
    known = {pid: value for pid, (value, _) in values.items() if pd.notna(value) and value > 0}
    known_values = list(known.values())
    incoming_window = incoming.loc[
        incoming["transfer_date"].ge(decision - pd.Timedelta(days=RECENT_DAYS))
        & incoming["transfer_date"].lt(decision)
        & incoming["player_broad_position"].eq(target_position)
    ].copy() if target_position else incoming.iloc[0:0].copy()
    if player_id is not None:
        incoming_window = incoming_window.loc[~incoming_window["canonical_player_id"].eq(player_id)]
    incoming_players = incoming_window.drop_duplicates("canonical_player_id")
    incoming_values = pd.to_numeric(incoming_players["canonical_market_value_eur"], errors="coerce")
    incoming_known = incoming_values.loc[incoming_values.gt(0)]
    higher_incoming = int((incoming_known > target_value).sum()) if pd.notna(target_value) else 0
    value_available = bool(onfield_available and pd.notna(target_value) and len(known_values) >= MIN_KNOWN_POSITION_VALUES)
    median_value = float(np.median(known_values)) if known_values else np.nan
    max_incoming_value = float(incoming_known.max()) if len(incoming_known) else 0.0
    record.update({
        "squad_value_incoming_available": value_available,
        "player_predecision_value_eur": target_value,
        "player_predecision_value_date": target_value_date.date().isoformat() if target_value_date is not None else None,
        "squad_position_known_value_players": len(known_values),
        "squad_position_known_value_players_log1p": math.log1p(len(known_values)),
        "player_same_position_value_percentile": percentile_at_or_below(known_values, target_value) if pd.notna(target_value) else np.nan,
        "player_value_to_position_median_log": math.log(target_value / median_value) if pd.notna(target_value) and median_value > 0 else np.nan,
        "incoming_same_position_players_180d": int(incoming_players["canonical_player_id"].nunique()),
        "incoming_same_position_players_180d_log1p": math.log1p(incoming_players["canonical_player_id"].nunique()),
        "incoming_same_position_known_value_players_180d": len(incoming_known),
        "incoming_same_position_higher_value_players_180d": higher_incoming,
        "incoming_same_position_higher_value_players_180d_log1p": math.log1p(higher_incoming),
        "incoming_same_position_max_value_ratio_log": (
            math.log(max_incoming_value / target_value)
            if pd.notna(target_value) and target_value > 0 and max_incoming_value > 0
            else (0.0 if pd.notna(target_value) else np.nan)
        ),
    })
    apply_availability(record, SQUAD_VALUE_INCOMING_FEATURES, value_available)

    latest_evidence_value_date = max((date for _, date in values.values() if date is not None), default=None)
    audit = {
        "capology_extension_event_id": anchor["capology_extension_event_id"],
        "signed_date": decision.date().isoformat(),
        "latest_game_date_used": games["date"].max().date().isoformat() if len(games) else None,
        "latest_lineup_date_used": lineups["date"].max().date().isoformat() if len(lineups) else None,
        "latest_appearance_date_used": apps["date"].max().date().isoformat() if len(apps) else None,
        "latest_incoming_transfer_date_used": incoming_window["transfer_date"].max().date().isoformat() if len(incoming_window) else None,
        "latest_valuation_date_used": latest_evidence_value_date.date().isoformat() if latest_evidence_value_date is not None else None,
        "role_level_available": role_level_available,
        "role_trajectory_available": trajectory_available,
        "squad_onfield_available": onfield_available,
        "squad_value_incoming_available": value_available,
    }
    return record, audit


def build_profiles() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    anchors = load_anchors()
    positions = player_positions()
    games = load_games(anchors)
    lineups = load_lineups(anchors, positions)
    appearances = load_appearances(anchors)
    incoming = load_incoming_transfers(anchors, positions)
    player_ids = set(lineups["player_id"].dropna().astype(int)) | set(incoming["canonical_player_id"].dropna().astype(int)) | set(anchors["canonical_player_id"].dropna().astype(int))
    valuations = load_valuations(player_ids, anchors)

    game_groups = {int(key): value for key, value in games.groupby("club_id")}
    lineup_groups = {int(key): value for key, value in lineups.groupby("club_id")}
    app_groups = {int(key): value for key, value in appearances.groupby("player_club_id")}
    incoming_groups = {int(key): value for key, value in incoming.groupby("to_club_id")}
    valuation_groups = {int(key): value for key, value in valuations.groupby("canonical_player_id")}
    empty_games, empty_lineups = games.iloc[0:0], lineups.iloc[0:0]
    empty_apps, empty_incoming = appearances.iloc[0:0], incoming.iloc[0:0]

    profiles, audits = [], []
    for _, anchor in anchors.iterrows():
        club = int(anchor["canonical_club_id"]) if pd.notna(anchor["canonical_club_id"]) else None
        record, audit = profile_anchor(
            anchor,
            game_groups.get(club, empty_games), lineup_groups.get(club, empty_lineups),
            app_groups.get(club, empty_apps), incoming_groups.get(club, empty_incoming),
            valuation_groups,
        )
        profiles.append(record)
        audits.append(audit)
    counts = {
        "anchor_rows": len(anchors), "relevant_club_games": len(games),
        "relevant_lineup_rows": len(lineups), "relevant_appearance_rows": len(appearances),
        "relevant_incoming_transfers": len(incoming), "relevant_valuation_rows": len(valuations),
    }
    return pd.DataFrame(profiles), pd.DataFrame(audits), counts


def feature_dictionary() -> pd.DataFrame:
    definitions = {
        "role_lineup_selection_rate_365d": "Matchday selections / observed club games.",
        "role_start_rate_365d": "Starting-XI selections / observed club games.",
        "role_substitute_selection_rate_365d": "Bench selections / observed club games.",
        "role_captain_rate_365d": "Captain selections / observed club games.",
        "role_position_entropy_365d": "Normalized entropy of detailed observed lineup positions.",
        "role_days_since_last_selection_scaled_365d": "Days since last selection, capped and divided by 365.",
        "role_recent_start_rate_180d": "Starting rate in the most recent 180 days.",
        "role_start_rate_change_recent_vs_prior": "Recent 180-day start rate minus preceding 185-day rate.",
        "role_selection_rate_change_recent_vs_prior": "Recent selection rate minus preceding selection rate.",
        "role_opportunity_share_change_recent_vs_prior": "Recent minutes opportunity share minus preceding share.",
        "squad_same_position_competitor_count_log1p": "log(1 + other same-position players with at least three selections).",
        "squad_same_position_strong_competitor_count_log1p": "log(1 + same-position competitors starting at least 25% of club games).",
        "player_same_position_start_percentile": "Player start-rate percentile among active same-position group.",
        "player_same_position_minutes_percentile": "Player minutes percentile among active same-position group.",
        "player_start_rate_gap_to_position_leader": "Position leader start rate minus player start rate.",
        "player_opportunity_gap_to_position_leader": "Position leader opportunity share minus player share.",
        "player_share_of_position_minutes": "Player minutes / active same-position minutes.",
        "squad_position_minutes_hhi": "HHI of minutes across the active same-position group.",
        "squad_position_known_value_players_log1p": "log(1 + active same-position players with valid predecision values).",
        "player_same_position_value_percentile": "Player predecision value percentile in active same-position group.",
        "player_value_to_position_median_log": "log(player value / position-group median value).",
        "incoming_same_position_players_180d_log1p": "log(1 + incoming same-position players in prior 180 days).",
        "incoming_same_position_higher_value_players_180d_log1p": "log(1 + incoming same-position players valued above target player).",
        "incoming_same_position_max_value_ratio_log": "log(maximum known incoming value / player value); zero when no incoming player has a known value.",
    }
    block = {}
    for name in ROLE_LEVEL_FEATURES:
        block[name] = "role_level"
    for name in ROLE_TRAJECTORY_FEATURES:
        block[name] = "role_trajectory"
    for name in SQUAD_ONFIELD_FEATURES:
        block[name] = "squad_onfield"
    for name in SQUAD_VALUE_INCOMING_FEATURES:
        block[name] = "squad_value_incoming"
    return pd.DataFrame([{"feature": name, "block": block[name], "definition": definitions[name], "timing": "decision-time; source-gated"} for name in MODEL_FEATURES])


def build_checks(profiles: pd.DataFrame, audits: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(check: str, passed: bool, observed: object, expected: object, note: str) -> None:
        rows.append({"check": check, "passed": bool(passed), "observed": observed, "expected": expected, "note": note})

    add("event_grain", len(profiles) == 2805 and profiles.capology_extension_event_id.is_unique, {"rows": len(profiles), "duplicates": int(profiles.capology_extension_event_id.duplicated().sum())}, {"rows": 2805, "duplicates": 0}, "Canonical extension grain is preserved.")
    signing = pd.to_datetime(audits.signed_date, errors="raise")
    for field in ["latest_game_date_used", "latest_lineup_date_used", "latest_appearance_date_used", "latest_incoming_transfer_date_used"]:
        dates = pd.to_datetime(audits[field], errors="coerce")
        add(f"{field}_predecision", bool((dates.dropna() < signing.loc[dates.notna()]).all()), int((dates >= signing).fillna(False).sum()), 0, "Operational evidence must be strictly before signing.")
    valuation_dates = pd.to_datetime(audits.latest_valuation_date_used, errors="coerce")
    add("valuation_not_future", bool((valuation_dates.dropna() <= signing.loc[valuation_dates.notna()]).all()), int((valuation_dates > signing).fillna(False).sum()), 0, "Valuations may be on, never after, signing date.")
    availability = {
        "role_level_available": ROLE_LEVEL_FEATURES,
        "role_trajectory_available": ROLE_TRAJECTORY_FEATURES,
        "squad_onfield_available": SQUAD_ONFIELD_FEATURES,
        "squad_value_incoming_available": SQUAD_VALUE_INCOMING_FEATURES,
    }
    for flag, features in availability.items():
        expected = profiles[flag].astype(bool)
        observed = profiles[features].notna().all(axis=1)
        add(f"{flag}_matches_features", expected.equals(observed), int((expected != observed).sum()), 0, "Availability is explicit and reconstructible.")
    add("rates_bounded", profiles[ROLE_LEVEL_FEATURES[:4]].stack().between(0, 1.05).all(), True, True, "Role rates stay within defensible bounds.")
    add("percentiles_bounded", profiles[["player_same_position_start_percentile", "player_same_position_minutes_percentile", "player_same_position_value_percentile"]].stack().between(0, 1).all(), True, True, "Relative ranks are valid percentiles.")
    add("no_targets", not any(column.startswith("target_") or "movement_state" in column for column in profiles), [column for column in profiles if column.startswith("target_") or "movement_state" in column], [], "Feature artifact contains no outcome.")
    finite = profiles[MODEL_FEATURES].to_numpy(float)
    add("finite_model_features", bool(np.isfinite(finite[~np.isnan(finite)]).all()), True, True, "No infinite model input.")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    unchanged = all(sha256(ROOT / item["path"]) == item["sha256"] for item in freeze["files"])
    add("v1_frozen", unchanged, unchanged, True, "All frozen V1 files remain byte-identical.")
    return pd.DataFrame(rows)


def write_readme(output: Path, profiles: pd.DataFrame, counts: dict[str, int]) -> None:
    dev = profiles.loc[profiles.signed_year.between(2020, 2023)]
    coverage = {flag: float(dev[flag].mean()) for flag in ["role_level_available", "role_trajectory_available", "squad_onfield_available", "squad_value_incoming_available"]}
    text = f"""# Engine V2 player-role and squad-competition profiles

**Status:** research feature build; no model promoted or deployed.

The build converts exact dated lineup selections, appearances, club games,
historical valuations, and incoming transfers into decision-time profiles for
all {len(profiles):,} canonical extensions.

## 2020–2023 availability

- current role structure: **{coverage['role_level_available']:.1%}**;
- split-window role trajectory: **{coverage['role_trajectory_available']:.1%}**;
- on-field positional competition: **{coverage['squad_onfield_available']:.1%}**;
- positional value and incoming competition: **{coverage['squad_value_incoming_available']:.1%}**.

A player's zero start, selection, or minutes rate is accepted only after the
club's lineup/appearance game coverage passes the 80% gate. Missing source
coverage becomes unavailable. Active positional competitors require at least
three prior-year lineup selections. Valuations are on or before signing and no
more than 365 days old.

## Source rows retained for the anchor universe

- club games: {counts['relevant_club_games']:,}
- lineup rows: {counts['relevant_lineup_rows']:,}
- appearance rows: {counts['relevant_appearance_rows']:,}
- incoming transfers: {counts['relevant_incoming_transfers']:,}
- valuation rows: {counts['relevant_valuation_rows']:,}

The 2024+ outcome holdout is not read by this build.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


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
    profiles.to_csv(output / "role_squad_profiles.csv", index=False)
    audits.to_csv(output / "asof_evidence_audit.csv", index=False)
    dictionary.to_csv(output / "feature_dictionary.csv", index=False)
    checks.to_csv(output / "build_checks.csv", index=False)
    write_json(output / "feature_contract.json", contract_payload())
    write_readme(output, profiles, counts)
    sources = [
        EXTENSIONS,
        LINEUPS,
        APPEARANCES,
        GAMES,
        PLAYERS,
        VALUATIONS,
        TRANSFERS,
        ROOT / "models" / "engine_v2" / "role_squad_feature_contract.py",
        ROOT / "scripts" / "build_engine_v2_role_squad_profiles.py",
    ]
    pd.DataFrame([{"source": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in sources]).to_csv(output / "source_manifest.csv", index=False)
    after = {path: sha256(ROOT / path) for path in before}
    summary = {
        **counts, "contract_version": CONTRACT_VERSION, "profile_rows": len(profiles),
        "development_availability": {flag: float(profiles.loc[profiles.signed_year.between(2020, 2023), flag].mean()) for flag in ["role_level_available", "role_trajectory_available", "squad_onfield_available", "squad_value_incoming_available"]},
        "final_holdout_opened": False, "deployment_changed": False,
        "v1_unchanged": before == after, "checks_passed": int(checks.passed.sum()), "checks_total": len(checks),
    }
    write_json(output / "run_summary.json", summary)
    manifest = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "output_manifest.csv":
            manifest.append({"output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest).to_csv(output / "output_manifest.csv", index=False)
    if not checks.passed.all():
        raise RuntimeError(checks.loc[~checks.passed, ["check", "observed", "expected"]].to_dict("records"))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

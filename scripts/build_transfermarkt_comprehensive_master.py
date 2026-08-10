"""Build a broad, leakage-labelled Transfermarkt transfer master.

Every generated component and the final master retain exactly one row per
``transfer_event_id``. High-granularity sources are aggregated into explicit
pre-transfer feature windows and post-transfer outcome windows before joining.
No transfer is filtered for missing data or model eligibility.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "Data" / "processed" / "transfermarkt_clean" / "tables"
MERGED = ROOT / "Data" / "processed" / "transfermarkt_merged"

CORE_PATH = MERGED / "transfer_core.csv"
VALUATIONS_PATH = TABLES / "player_valuations_clean.csv"
APPEARANCES_PATH = TABLES / "appearances_clean.csv"
LINEUPS_PATH = TABLES / "game_lineups_clean.csv"
EVENTS_PATH = TABLES / "game_events_clean.csv"
GAMES_PATH = TABLES / "games_clean.csv"
CLUB_GAMES_PATH = TABLES / "club_games_clean.csv"
COMPETITIONS_PATH = TABLES / "competitions_clean.csv"

VALUATION_OUTPUT = MERGED / "transfer_valuation_features.csv"
APPEARANCE_OUTPUT = MERGED / "transfer_appearance_features.csv"
LINEUP_OUTPUT = MERGED / "transfer_lineup_features.csv"
EVENT_OUTPUT = MERGED / "transfer_event_features.csv"
CLUB_OUTPUT = MERGED / "transfer_club_context.csv"
MASTER_OUTPUT = MERGED / "transfermarkt_comprehensive_master.csv"
DICTIONARY_OUTPUT = MERGED / "transfermarkt_comprehensive_field_dictionary.csv"
ASSUMPTIONS_OUTPUT = MERGED / "transfermarkt_comprehensive_assumptions.csv"
CHECKS_OUTPUT = MERGED / "transfermarkt_comprehensive_checks.csv"
SUMMARY_OUTPUT = MERGED / "transfermarkt_comprehensive_summary.json"
README_OUTPUT = MERGED / "COMPREHENSIVE_MASTER_README.md"

PRE_DAYS = 365
POST12_LAST_DAY = 364
POST24_LAST_DAY = 729
VALUATION_TOLERANCES = (60, 90, 120, 180)
SPORTING_MIN_DESTINATION_GAMES_12M = 10
SPORTING_MIN_DESTINATION_GAMES_24M = 20


def clean_id(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().str.replace(r"\.0$", "", regex=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_pct(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype("string").fillna("").str.casefold().eq("true")


def deterministic_mode(values: Iterable[str]) -> str:
    cleaned = [str(value) for value in values if pd.notna(value) and str(value).strip()]
    if not cleaned:
        return ""
    counts = Counter(cleaned)
    highest = max(counts.values())
    return sorted(value for value, count in counts.items() if count == highest)[0]


def read_filtered(
    path: Path,
    usecols: list[str],
    relevant_ids: set[str],
    id_columns: list[str],
    chunksize: int = 250_000,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, dtype=str, keep_default_na=False, chunksize=chunksize):
        mask = pd.Series(False, index=chunk.index)
        for column in id_columns:
            chunk[column] = clean_id(chunk[column])
            mask |= chunk[column].isin(relevant_ids)
        if mask.any():
            parts.append(chunk.loc[mask].copy())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=usecols)


def reference_frame(core: pd.DataFrame) -> pd.DataFrame:
    reference = core[["transfer_event_id", "player_id", "transfer_date", "from_club_id", "to_club_id"]].copy()
    reference["transfer_date"] = pd.to_datetime(reference["transfer_date"], errors="raise")
    for column in ["player_id", "from_club_id", "to_club_id"]:
        reference[column] = clean_id(reference[column])
    return reference


def add_valuation_window(
    result: dict[str, object],
    prefix: str,
    dates: np.ndarray,
    values: np.ndarray,
    left: int,
    right: int,
) -> None:
    window_values = values[left:right]
    window_dates = dates[left:right]
    if not len(window_values):
        result[f"{prefix}_observations"] = 0
        return
    first_value = float(window_values[0])
    last_value = float(window_values[-1])
    first_date = pd.Timestamp(window_dates[0])
    last_date = pd.Timestamp(window_dates[-1])
    result.update(
        {
            f"{prefix}_observations": int(len(window_values)),
            f"{prefix}_first_date": first_date,
            f"{prefix}_first_eur": first_value,
            f"{prefix}_last_date": last_date,
            f"{prefix}_last_eur": last_value,
            f"{prefix}_mean_eur": float(np.nanmean(window_values)),
            f"{prefix}_min_eur": float(np.nanmin(window_values)),
            f"{prefix}_max_eur": float(np.nanmax(window_values)),
            f"{prefix}_change_eur": last_value - first_value,
            f"{prefix}_change_pct": safe_pct(last_value - first_value, first_value),
            f"{prefix}_observation_span_days": int((last_date - first_date).days),
        }
    )


def slice_indices(dates: np.ndarray, start: pd.Timestamp | None, end: pd.Timestamp) -> tuple[int, int]:
    left = 0 if start is None else int(np.searchsorted(dates, np.datetime64(start), side="left"))
    right = int(np.searchsorted(dates, np.datetime64(end), side="right"))
    return left, right


def build_valuation_features(reference: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    valuations = pd.read_csv(
        VALUATIONS_PATH,
        usecols=["player_id", "date", "market_value_in_eur", "current_club_id", "player_club_domestic_competition_id"],
        dtype=str,
        keep_default_na=False,
    )
    valuations["player_id"] = clean_id(valuations["player_id"])
    valuations["current_club_id"] = clean_id(valuations["current_club_id"])
    valuations["date"] = pd.to_datetime(valuations["date"], errors="coerce")
    valuations["market_value_in_eur"] = pd.to_numeric(valuations["market_value_in_eur"], errors="coerce")
    valuations = valuations.dropna(subset=["date", "market_value_in_eur"]).sort_values(["player_id", "date"])
    relevant = set(reference["player_id"])
    valuations = valuations[valuations["player_id"].isin(relevant)].copy()
    groups = {player_id: group.reset_index(drop=True) for player_id, group in valuations.groupby("player_id", sort=False)}
    max_date = valuations["date"].max()
    min_date = valuations["date"].min()

    records: list[dict[str, object]] = []
    for row in reference.itertuples(index=False):
        result: dict[str, object] = {"transfer_event_id": row.transfer_event_id}
        group = groups.get(row.player_id)
        transfer_date = row.transfer_date
        target12 = transfer_date + pd.Timedelta(days=365)
        target24 = transfer_date + pd.Timedelta(days=730)
        result["valuation_post12_target_date"] = target12
        result["valuation_post24_target_date"] = target24
        result["valuation_post12_horizon_observable"] = bool(pd.notna(max_date) and target12 <= max_date)
        result["valuation_post24_horizon_observable"] = bool(pd.notna(max_date) and target24 <= max_date)

        if group is None or group.empty:
            result["valuation_history_available"] = False
            records.append(result)
            continue

        dates = group["date"].to_numpy(dtype="datetime64[ns]")
        values = group["market_value_in_eur"].to_numpy(dtype=float)
        transfer64 = np.datetime64(transfer_date)
        pre_position = int(np.searchsorted(dates, transfer64, side="right"))
        result["valuation_history_available"] = True
        result["valuation_observations_all"] = int(len(group))
        result["valuation_observations_before_transfer"] = pre_position

        if pre_position > 0:
            idx = pre_position - 1
            pre_date = pd.Timestamp(dates[idx])
            pre_value = float(values[idx])
            staleness = int((transfer_date - pre_date).days)
            result.update(
                {
                    "valuation_pre_date": pre_date,
                    "valuation_pre_eur": pre_value,
                    "valuation_pre_staleness_days": staleness,
                    "valuation_pre_current_club_id": group.iloc[idx]["current_club_id"],
                    "valuation_pre_competition_id": group.iloc[idx]["player_club_domestic_competition_id"],
                    "valuation_pre_is_origin_club": group.iloc[idx]["current_club_id"] == row.from_club_id,
                    "valuation_pre_within_90d": staleness <= 90,
                    "valuation_pre_within_180d": staleness <= 180,
                    "valuation_pre_within_365d": staleness <= 365,
                    "valuation_prior_peak_eur": float(np.nanmax(values[:pre_position])),
                }
            )
            for prefix, days in (("valuation_pre365", 365), ("valuation_pre730", 730)):
                start_date = np.datetime64(transfer_date - pd.Timedelta(days=days))
                start_position = int(np.searchsorted(dates, start_date, side="left"))
                add_valuation_window(result, prefix, dates, values, start_position, pre_position)
            add_valuation_window(result, "valuation_career_pre", dates, values, 0, pre_position)

        for horizon, target in ((12, target12), (24, target24)):
            target64 = np.datetime64(target)
            insertion = int(np.searchsorted(dates, target64, side="left"))
            candidates = [idx for idx in (insertion - 1, insertion) if 0 <= idx < len(dates)]
            if not candidates:
                continue
            idx = min(candidates, key=lambda candidate: abs(int((dates[candidate] - target64).astype("timedelta64[D]").astype(int))))
            observed_date = pd.Timestamp(dates[idx])
            gap = int((observed_date - target).days)
            value = float(values[idx])
            result[f"valuation_post{horizon}_nearest_date"] = observed_date
            result[f"valuation_post{horizon}_nearest_eur"] = value
            result[f"valuation_post{horizon}_gap_days"] = gap
            for tolerance in VALUATION_TOLERANCES:
                result[f"valuation_post{horizon}_within_{tolerance}d"] = abs(gap) <= tolerance
            if abs(gap) <= 90:
                result[f"valuation_post{horizon}_90d_eur"] = value

        records.append(result)

    features = pd.DataFrame(records)
    for horizon in (12, 24):
        post = features.get(f"valuation_post{horizon}_90d_eur")
        pre = features.get("valuation_pre_eur")
        if post is not None and pre is not None:
            features[f"valuation_post{horizon}_change_eur"] = post - pre
            features[f"valuation_post{horizon}_change_pct"] = ((post - pre) / pre).where(pre.gt(0))
    metadata = {
        "valuation_min_date": min_date,
        "valuation_max_date": max_date,
        "relevant_valuation_rows": len(valuations),
    }
    return features, metadata


def prepare_appearance_groups(reference: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    relevant = set(reference["player_id"])
    columns = [
        "appearance_id", "game_id", "player_id", "player_club_id", "date", "competition_id",
        "yellow_cards", "red_cards", "goals", "assists", "minutes_played",
    ]
    appearances = read_filtered(APPEARANCES_PATH, columns, relevant, ["player_id"])
    appearances["player_club_id"] = clean_id(appearances["player_club_id"])
    appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")
    appearances = appearances.dropna(subset=["date"])
    for column in ["yellow_cards", "red_cards", "goals", "assists", "minutes_played"]:
        appearances[column] = pd.to_numeric(appearances[column], errors="coerce").fillna(0)
    appearances = appearances.sort_values(["player_id", "date", "game_id"])
    groups = {player_id: group.reset_index(drop=True) for player_id, group in appearances.groupby("player_id", sort=False)}
    metadata = {
        "appearance_min_date": appearances["date"].min(),
        "appearance_max_date": appearances["date"].max(),
        "relevant_appearance_rows": len(appearances),
    }
    return groups, metadata


def aggregate_appearance_window(
    group: pd.DataFrame | None,
    start: pd.Timestamp | None,
    end: pd.Timestamp,
    club_id: str | None = None,
) -> dict[str, object]:
    empty = {
        "has_records": False, "records": 0, "games": 0, "minutes": 0.0, "goals": 0.0,
        "assists": 0.0, "yellow_cards": 0.0, "red_cards": 0.0, "clubs": 0,
        "competitions": 0, "first_date": pd.NaT, "last_date": pd.NaT,
    }
    if group is None or group.empty:
        return empty
    dates = group["date"].to_numpy(dtype="datetime64[ns]")
    left, right = slice_indices(dates, start, end)
    window = group.iloc[left:right]
    if club_id is not None:
        window = window[window["player_club_id"].eq(club_id)]
    if window.empty:
        return empty
    return {
        "has_records": True,
        "records": int(len(window)),
        "games": int(window["game_id"].nunique()),
        "minutes": float(window["minutes_played"].sum()),
        "goals": float(window["goals"].sum()),
        "assists": float(window["assists"].sum()),
        "yellow_cards": float(window["yellow_cards"].sum()),
        "red_cards": float(window["red_cards"].sum()),
        "clubs": int(window["player_club_id"].nunique()),
        "competitions": int(window["competition_id"].nunique()),
        "first_date": window["date"].min(),
        "last_date": window["date"].max(),
    }


def prefixed(result: dict[str, object], prefix: str, values: dict[str, object]) -> None:
    result.update({f"{prefix}_{key}": value for key, value in values.items()})


def build_appearance_features(
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    groups, metadata = prepare_appearance_groups(reference)
    records: list[dict[str, object]] = []
    for row in reference.itertuples(index=False):
        group = groups.get(row.player_id)
        transfer_date = row.transfer_date
        result: dict[str, object] = {"transfer_event_id": row.transfer_event_id}
        windows = {
            "appearance_career_pre": (None, transfer_date - pd.Timedelta(days=1), None),
            "appearance_pre365_all": (transfer_date - pd.Timedelta(days=PRE_DAYS), transfer_date - pd.Timedelta(days=1), None),
            "appearance_pre365_origin": (transfer_date - pd.Timedelta(days=PRE_DAYS), transfer_date - pd.Timedelta(days=1), row.from_club_id),
            "appearance_pre730_all": (transfer_date - pd.Timedelta(days=730), transfer_date - pd.Timedelta(days=1), None),
            "appearance_pre730_origin": (transfer_date - pd.Timedelta(days=730), transfer_date - pd.Timedelta(days=1), row.from_club_id),
            "appearance_post12_all": (transfer_date, transfer_date + pd.Timedelta(days=POST12_LAST_DAY), None),
            "appearance_post12_destination": (transfer_date, transfer_date + pd.Timedelta(days=POST12_LAST_DAY), row.to_club_id),
            "appearance_post24_all": (transfer_date, transfer_date + pd.Timedelta(days=POST24_LAST_DAY), None),
            "appearance_post24_destination": (transfer_date, transfer_date + pd.Timedelta(days=POST24_LAST_DAY), row.to_club_id),
        }
        for prefix, (start, end, club_id) in windows.items():
            prefixed(result, prefix, aggregate_appearance_window(group, start, end, club_id))
        for prefix in windows:
            minutes = result[f"{prefix}_minutes"]
            nineties = minutes / 90 if minutes else 0
            result[f"{prefix}_goals_per90"] = safe_pct(result[f"{prefix}_goals"], nineties)
            result[f"{prefix}_assists_per90"] = safe_pct(result[f"{prefix}_assists"], nineties)
        records.append(result)
    return pd.DataFrame(records), metadata


def prepare_lineup_groups(reference: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    relevant = set(reference["player_id"])
    columns = [
        "game_lineups_id", "game_id", "date", "player_id", "club_id", "type", "position",
        "team_captain", "is_starting_lineup", "is_substitute",
    ]
    lineups = read_filtered(LINEUPS_PATH, columns, relevant, ["player_id"])
    lineups["club_id"] = clean_id(lineups["club_id"])
    lineups["date"] = pd.to_datetime(lineups["date"], errors="coerce")
    lineups = lineups.dropna(subset=["date"]).sort_values(["player_id", "date", "game_id"])
    for column in ["team_captain", "is_starting_lineup", "is_substitute"]:
        lineups[column] = lineups[column].astype(str).str.casefold().eq("true")
    groups = {player_id: group.reset_index(drop=True) for player_id, group in lineups.groupby("player_id", sort=False)}
    metadata = {
        "lineup_min_date": lineups["date"].min(),
        "lineup_max_date": lineups["date"].max(),
        "relevant_lineup_rows": len(lineups),
    }
    return groups, metadata


def aggregate_lineup_window(
    group: pd.DataFrame | None,
    start: pd.Timestamp | None,
    end: pd.Timestamp,
    club_id: str | None = None,
) -> dict[str, object]:
    empty = {
        "has_records": False, "records": 0, "games": 0, "starts": 0, "substitute_selections": 0,
        "captain_selections": 0, "start_pct": np.nan, "distinct_positions": 0,
        "primary_position": "", "first_date": pd.NaT, "last_date": pd.NaT,
    }
    if group is None or group.empty:
        return empty
    dates = group["date"].to_numpy(dtype="datetime64[ns]")
    left, right = slice_indices(dates, start, end)
    window = group.iloc[left:right]
    if club_id is not None:
        window = window[window["club_id"].eq(club_id)]
    if window.empty:
        return empty
    starts = int(window["is_starting_lineup"].sum())
    records = int(len(window))
    return {
        "has_records": True,
        "records": records,
        "games": int(window["game_id"].nunique()),
        "starts": starts,
        "substitute_selections": int(window["is_substitute"].sum()),
        "captain_selections": int(window["team_captain"].sum()),
        "start_pct": safe_pct(starts, records),
        "distinct_positions": int(window.loc[window["position"].ne(""), "position"].nunique()),
        "primary_position": deterministic_mode(window["position"]),
        "first_date": window["date"].min(),
        "last_date": window["date"].max(),
    }


def build_lineup_features(reference: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    groups, metadata = prepare_lineup_groups(reference)
    records: list[dict[str, object]] = []
    for row in reference.itertuples(index=False):
        transfer_date = row.transfer_date
        group = groups.get(row.player_id)
        result: dict[str, object] = {"transfer_event_id": row.transfer_event_id}
        windows = {
            "lineup_career_pre": (None, transfer_date - pd.Timedelta(days=1), None),
            "lineup_pre365_all": (transfer_date - pd.Timedelta(days=PRE_DAYS), transfer_date - pd.Timedelta(days=1), None),
            "lineup_pre365_origin": (transfer_date - pd.Timedelta(days=PRE_DAYS), transfer_date - pd.Timedelta(days=1), row.from_club_id),
            "lineup_pre730_all": (transfer_date - pd.Timedelta(days=730), transfer_date - pd.Timedelta(days=1), None),
            "lineup_pre730_origin": (transfer_date - pd.Timedelta(days=730), transfer_date - pd.Timedelta(days=1), row.from_club_id),
            "lineup_post12_destination": (transfer_date, transfer_date + pd.Timedelta(days=POST12_LAST_DAY), row.to_club_id),
            "lineup_post24_destination": (transfer_date, transfer_date + pd.Timedelta(days=POST24_LAST_DAY), row.to_club_id),
        }
        for prefix, (start, end, club_id) in windows.items():
            prefixed(result, prefix, aggregate_lineup_window(group, start, end, club_id))
        records.append(result)
    return pd.DataFrame(records), metadata


def prepare_event_groups(reference: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    relevant = set(reference["player_id"])
    columns = ["game_event_id", "game_id", "date", "club_id", "player_id", "player_in_id", "player_assist_id", "event_type_normalized"]
    ledgers: list[pd.DataFrame] = []
    min_date = pd.NaT
    max_date = pd.NaT
    source_rows = 0
    for chunk in pd.read_csv(EVENTS_PATH, usecols=columns, dtype=str, keep_default_na=False, chunksize=250_000):
        for column in ["club_id", "player_id", "player_in_id", "player_assist_id"]:
            chunk[column] = clean_id(chunk[column])
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk = chunk.dropna(subset=["date"])
        source_rows += len(chunk)
        if not chunk.empty:
            chunk_min, chunk_max = chunk["date"].min(), chunk["date"].max()
            min_date = chunk_min if pd.isna(min_date) or chunk_min < min_date else min_date
            max_date = chunk_max if pd.isna(max_date) or chunk_max > max_date else max_date

        primary = chunk[chunk["player_id"].isin(relevant)].copy()
        if not primary.empty:
            ledger = primary[["player_id", "date", "club_id", "game_id"]].copy()
            ledger["primary_events"] = 1
            ledger["goal_events"] = primary["event_type_normalized"].eq("goals").astype(int).to_numpy()
            ledger["card_events"] = primary["event_type_normalized"].eq("cards").astype(int).to_numpy()
            ledger["shootout_events"] = primary["event_type_normalized"].eq("shootout").astype(int).to_numpy()
            ledger["substitution_out_events"] = primary["event_type_normalized"].eq("substitutions").astype(int).to_numpy()
            ledger["substitution_in_events"] = 0
            ledger["assist_events"] = 0
            ledgers.append(ledger)

        incoming = chunk[chunk["player_in_id"].isin(relevant) & chunk["event_type_normalized"].eq("substitutions")].copy()
        if not incoming.empty:
            ledger = incoming[["player_in_id", "date", "club_id", "game_id"]].rename(columns={"player_in_id": "player_id"})
            ledger[["primary_events", "goal_events", "card_events", "shootout_events", "substitution_out_events", "assist_events"]] = 0
            ledger["substitution_in_events"] = 1
            ledgers.append(ledger)

        assisted = chunk[chunk["player_assist_id"].isin(relevant)].copy()
        if not assisted.empty:
            ledger = assisted[["player_assist_id", "date", "club_id", "game_id"]].rename(columns={"player_assist_id": "player_id"})
            ledger[["primary_events", "goal_events", "card_events", "shootout_events", "substitution_out_events", "substitution_in_events"]] = 0
            ledger["assist_events"] = 1
            ledgers.append(ledger)

    if ledgers:
        events = pd.concat(ledgers, ignore_index=True)
        metric_columns = ["primary_events", "goal_events", "card_events", "shootout_events", "substitution_out_events", "substitution_in_events", "assist_events"]
        events = events.groupby(["player_id", "date", "club_id", "game_id"], as_index=False)[metric_columns].sum()
        events = events.sort_values(["player_id", "date", "game_id"])
        groups = {player_id: group.reset_index(drop=True) for player_id, group in events.groupby("player_id", sort=False)}
    else:
        events = pd.DataFrame()
        groups = {}
    metadata = {
        "event_min_date": min_date,
        "event_max_date": max_date,
        "source_event_rows": source_rows,
        "relevant_player_event_ledger_rows": len(events),
    }
    return groups, metadata


def aggregate_event_window(
    group: pd.DataFrame | None,
    start: pd.Timestamp | None,
    end: pd.Timestamp,
    club_id: str | None = None,
) -> dict[str, object]:
    metrics = ["primary_events", "goal_events", "card_events", "shootout_events", "substitution_out_events", "substitution_in_events", "assist_events"]
    empty = {"has_records": False, "games": 0, **{metric: 0 for metric in metrics}, "first_date": pd.NaT, "last_date": pd.NaT}
    if group is None or group.empty:
        return empty
    dates = group["date"].to_numpy(dtype="datetime64[ns]")
    left, right = slice_indices(dates, start, end)
    window = group.iloc[left:right]
    if club_id is not None:
        window = window[window["club_id"].eq(club_id)]
    if window.empty:
        return empty
    result: dict[str, object] = {"has_records": True, "games": int(window["game_id"].nunique())}
    result.update({metric: int(window[metric].sum()) for metric in metrics})
    result["first_date"] = window["date"].min()
    result["last_date"] = window["date"].max()
    return result


def build_event_features(reference: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    groups, metadata = prepare_event_groups(reference)
    records: list[dict[str, object]] = []
    for row in reference.itertuples(index=False):
        transfer_date = row.transfer_date
        group = groups.get(row.player_id)
        result: dict[str, object] = {"transfer_event_id": row.transfer_event_id}
        windows = {
            "event_career_pre": (None, transfer_date - pd.Timedelta(days=1), None),
            "event_pre365_all": (transfer_date - pd.Timedelta(days=PRE_DAYS), transfer_date - pd.Timedelta(days=1), None),
            "event_pre365_origin": (transfer_date - pd.Timedelta(days=PRE_DAYS), transfer_date - pd.Timedelta(days=1), row.from_club_id),
            "event_pre730_all": (transfer_date - pd.Timedelta(days=730), transfer_date - pd.Timedelta(days=1), None),
            "event_pre730_origin": (transfer_date - pd.Timedelta(days=730), transfer_date - pd.Timedelta(days=1), row.from_club_id),
            "event_post12_destination": (transfer_date, transfer_date + pd.Timedelta(days=POST12_LAST_DAY), row.to_club_id),
            "event_post24_destination": (transfer_date, transfer_date + pd.Timedelta(days=POST24_LAST_DAY), row.to_club_id),
        }
        for prefix, (start, end, club_id) in windows.items():
            prefixed(result, prefix, aggregate_event_window(group, start, end, club_id))
        records.append(result)
    return pd.DataFrame(records), metadata


def prepare_club_game_groups() -> tuple[dict[str, dict[str, np.ndarray]], dict[str, object]]:
    games = pd.read_csv(
        GAMES_PATH,
        usecols=["game_id", "date", "competition_id", "season", "is_national_team_game"],
        dtype=str,
        keep_default_na=False,
    )
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    competitions = pd.read_csv(
        COMPETITIONS_PATH,
        usecols=["competition_id", "name", "type", "sub_type", "country_name"],
        dtype=str,
        keep_default_na=False,
    ).rename(
        columns={
            "name": "competition_name",
            "type": "competition_type",
            "sub_type": "competition_sub_type",
            "country_name": "competition_country_name",
        }
    )
    club_games = pd.read_csv(
        CLUB_GAMES_PATH,
        usecols=["game_id", "club_id", "own_goals", "own_position", "opponent_id", "opponent_goals", "hosting", "is_win"],
        dtype=str,
        keep_default_na=False,
    )
    club_games["club_id"] = clean_id(club_games["club_id"])
    club_games["opponent_id"] = clean_id(club_games["opponent_id"])
    for column in ["own_goals", "opponent_goals", "own_position"]:
        club_games[column] = pd.to_numeric(club_games[column], errors="coerce")
    club_games["is_win"] = club_games["is_win"].astype(str).str.casefold().eq("true")
    club_games = club_games.merge(games, on="game_id", how="left", validate="many_to_one")
    club_games = club_games.merge(competitions, on="competition_id", how="left", validate="many_to_one")
    club_games = club_games.dropna(subset=["date"]).sort_values(["club_id", "date", "game_id"])
    array_columns = [
        "date", "game_id", "own_goals", "opponent_goals", "own_position", "is_win", "hosting",
        "competition_id", "competition_name", "competition_type",
    ]
    groups: dict[str, dict[str, np.ndarray]] = {}
    for club_id, group in club_games.groupby("club_id", sort=False):
        groups[club_id] = {
            column: (
                group[column].to_numpy(dtype="datetime64[ns]")
                if column == "date"
                else group[column].to_numpy()
            )
            for column in array_columns
        }
    metadata = {
        "club_game_min_date": club_games["date"].min(),
        "club_game_max_date": club_games["date"].max(),
        "club_game_rows": len(club_games),
    }
    return groups, metadata


def aggregate_club_window(
    group: dict[str, np.ndarray] | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, object]:
    empty = {
        "has_records": False, "games": 0, "wins": 0, "draws": 0, "losses": 0,
        "goals_for": 0.0, "goals_against": 0.0, "goal_difference": 0.0, "points": 0,
        "win_pct": np.nan, "points_per_game": np.nan, "avg_recorded_position": np.nan,
        "home_games": 0, "away_games": 0, "competitions": 0, "domestic_league_games": 0,
        "domestic_cup_games": 0, "international_games": 0, "dominant_competition_id": "",
        "dominant_competition_name": "", "dominant_competition_type": "",
        "first_date": pd.NaT, "last_date": pd.NaT,
    }
    if group is None or not len(group["date"]):
        return empty
    dates = group["date"]
    left, right = slice_indices(dates, start, end)
    if right <= left:
        return empty
    game_ids = group["game_id"][left:right]
    own_goals = group["own_goals"][left:right].astype(float)
    opponent_goals = group["opponent_goals"][left:right].astype(float)
    positions = group["own_position"][left:right].astype(float)
    wins_array = group["is_win"][left:right].astype(bool)
    hosting = group["hosting"][left:right]
    competition_ids = group["competition_id"][left:right]
    competition_names = group["competition_name"][left:right]
    competition_types = group["competition_type"][left:right]
    games_count = int(len(np.unique(game_ids)))
    wins = int(wins_array.sum())
    draws = int(np.sum(own_goals == opponent_goals))
    losses = games_count - wins - draws
    goals_for = float(np.nansum(own_goals))
    goals_against = float(np.nansum(opponent_goals))
    dominant_id = deterministic_mode(competition_ids)
    dominant_mask = competition_ids == dominant_id if dominant_id else np.zeros(len(competition_ids), dtype=bool)
    return {
        "has_records": True,
        "games": games_count,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "goal_difference": goals_for - goals_against,
        "points": wins * 3 + draws,
        "win_pct": safe_pct(wins, games_count),
        "points_per_game": safe_pct(wins * 3 + draws, games_count),
        "avg_recorded_position": float(np.nanmean(positions)) if np.any(~np.isnan(positions)) else np.nan,
        "home_games": int(np.sum(hosting == "Home")),
        "away_games": int(np.sum(hosting == "Away")),
        "competitions": int(len(np.unique(competition_ids[competition_ids != ""]))),
        "domestic_league_games": int(np.sum(competition_types == "domestic_league")),
        "domestic_cup_games": int(np.sum(competition_types == "domestic_cup")),
        "international_games": int(np.sum(np.isin(competition_types, ["international_cup", "national_team_competition"]))),
        "dominant_competition_id": dominant_id,
        "dominant_competition_name": deterministic_mode(competition_names[dominant_mask]),
        "dominant_competition_type": deterministic_mode(competition_types[dominant_mask]),
        "first_date": pd.Timestamp(dates[left]),
        "last_date": pd.Timestamp(dates[right - 1]),
    }


def build_club_context(reference: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    groups, metadata = prepare_club_game_groups()
    records: list[dict[str, object]] = []
    for row in reference.itertuples(index=False):
        transfer_date = row.transfer_date
        result: dict[str, object] = {"transfer_event_id": row.transfer_event_id}
        windows = {
            "club_origin_pre365": (row.from_club_id, transfer_date - pd.Timedelta(days=365), transfer_date - pd.Timedelta(days=1)),
            "club_origin_pre730": (row.from_club_id, transfer_date - pd.Timedelta(days=730), transfer_date - pd.Timedelta(days=1)),
            "club_destination_pre365": (row.to_club_id, transfer_date - pd.Timedelta(days=365), transfer_date - pd.Timedelta(days=1)),
            "club_destination_pre730": (row.to_club_id, transfer_date - pd.Timedelta(days=730), transfer_date - pd.Timedelta(days=1)),
            "club_destination_post12": (row.to_club_id, transfer_date, transfer_date + pd.Timedelta(days=POST12_LAST_DAY)),
            "club_destination_post24": (row.to_club_id, transfer_date, transfer_date + pd.Timedelta(days=POST24_LAST_DAY)),
        }
        for prefix, (club_id, start, end) in windows.items():
            prefixed(result, prefix, aggregate_club_window(groups.get(club_id), start, end))
        records.append(result)
    return pd.DataFrame(records), metadata


def add_master_flags(master: pd.DataFrame, metadata: dict[str, object]) -> pd.DataFrame:
    transfer_dates = pd.to_datetime(master["transfer_date"], errors="coerce")
    birth_dates = pd.to_datetime(master["player_date_of_birth"], errors="coerce")
    master["player_age_at_transfer"] = (transfer_dates - birth_dates).dt.days / 365.2425
    master["player_age_at_transfer_plausible_15_45"] = master["player_age_at_transfer"].between(15, 45)
    master["has_positive_reported_fee"] = master["transfer_fee_status"].eq("positive_reported")
    master["movement_type_unresolved"] = True
    same_club = as_bool(master["is_same_club_id"])

    club_game_max = metadata.get("club_game_max_date")
    club_game_min = metadata.get("club_game_min_date")
    valuation_max = metadata.get("valuation_max_date")
    sporting_max = club_game_max if pd.notna(club_game_max) else pd.NaT
    master["transfer_date_before_first_club_game"] = transfer_dates.lt(club_game_min) if pd.notna(club_game_min) else False
    master["transfer_date_after_latest_club_game"] = transfer_dates.gt(club_game_max) if pd.notna(club_game_max) else False
    master["transfer_date_after_latest_valuation"] = transfer_dates.gt(valuation_max) if pd.notna(valuation_max) else False
    master["sporting_post12_horizon_observable"] = transfer_dates.add(pd.Timedelta(days=365)).le(sporting_max) if pd.notna(sporting_max) else False
    master["sporting_post24_horizon_observable"] = transfer_dates.add(pd.Timedelta(days=730)).le(sporting_max) if pd.notna(sporting_max) else False

    master["has_pretransfer_valuation_365d"] = master.get("valuation_pre_within_365d", False).fillna(False).astype(bool)
    master["has_post12_valuation_90d"] = master.get("valuation_post12_within_90d", False).fillna(False).astype(bool)
    master["has_post24_valuation_90d"] = master.get("valuation_post24_within_90d", False).fillna(False).astype(bool)
    master["has_pre365_appearance_records"] = master["appearance_pre365_all_has_records"].fillna(False).astype(bool)
    master["has_destination_game_coverage_12m"] = master["club_destination_post12_games"].ge(SPORTING_MIN_DESTINATION_GAMES_12M)
    master["has_destination_game_coverage_24m"] = master["club_destination_post24_games"].ge(SPORTING_MIN_DESTINATION_GAMES_24M)
    master["eligible_sporting_12m_provisional"] = (
        ~same_club
        & master["has_pre365_appearance_records"]
        & master["has_destination_game_coverage_12m"]
    )
    master["eligible_sporting_24m_provisional"] = (
        master["eligible_sporting_12m_provisional"] & master["has_destination_game_coverage_24m"]
    )
    master["eligible_financial_12m_provisional"] = (
        ~same_club
        & master["has_positive_reported_fee"]
        & master["has_pretransfer_valuation_365d"]
        & master["has_post12_valuation_90d"]
    )
    master["eligible_financial_24m_provisional"] = (
        ~same_club
        & master["has_positive_reported_fee"]
        & master["has_pretransfer_valuation_365d"]
        & master["has_post24_valuation_90d"]
    )
    master["eligible_combined_24m_provisional"] = (
        master["eligible_sporting_24m_provisional"] & master["eligible_financial_24m_provisional"]
    )

    def reasons(row: pd.Series) -> str:
        values: list[str] = []
        if str(row["is_same_club_id"]).casefold() == "true":
            values.append("same_club_id")
        if not row["has_pre365_appearance_records"]:
            values.append("no_pre365_appearances")
        if not row["has_destination_game_coverage_24m"]:
            values.append("insufficient_destination_game_coverage_24m")
        if not row["has_positive_reported_fee"]:
            values.append("no_positive_reported_fee")
        if not row["has_pretransfer_valuation_365d"]:
            values.append("no_pretransfer_valuation_within_365d")
        if not row["has_post24_valuation_90d"]:
            values.append("no_post24_valuation_within_90d")
        values.append("movement_type_unresolved")
        return " | ".join(values)

    master["provisional_exclusion_reasons"] = master.apply(reasons, axis=1)
    return master


def build_checks(
    core: pd.DataFrame,
    components: dict[str, pd.DataFrame],
    master: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(check: str, actual: object, expected: object, passed: bool, notes: str) -> None:
        rows.append({"check": check, "actual": actual, "expected": expected, "status": "OK" if passed else "FAIL", "notes": notes})

    expected_rows = len(core)
    for name, component in components.items():
        add(f"{name} row preservation", len(component), expected_rows, len(component) == expected_rows, "Every component must contain one row per transfer event.")
        add(f"{name} transfer_event_id uniqueness", component["transfer_event_id"].nunique(), expected_rows, component["transfer_event_id"].is_unique, "Component key must be unique before joining.")
    add("Master row preservation", len(master), expected_rows, len(master) == expected_rows, "No transfer may be filtered or duplicated.")
    add("Master transfer_event_id uniqueness", master["transfer_event_id"].nunique(), expected_rows, master["transfer_event_id"].is_unique, "Final master grain is one transfer event.")
    core_preserved = all(master[column].astype("string").fillna("").equals(core[column].astype("string").fillna("")) for column in core.columns)
    add("Transfer core fields preserved", core_preserved, True, core_preserved, "All transfer-core values and row order must remain unchanged.")

    transfer_date = pd.to_datetime(master["transfer_date"], errors="coerce")
    timing_checks = {
        "Pre-valuation occurs after transfer": pd.to_datetime(master.get("valuation_pre_date"), errors="coerce") > transfer_date,
        "Pre-appearance window ends on/after transfer": pd.to_datetime(master.get("appearance_pre365_all_last_date"), errors="coerce") >= transfer_date,
        "Pre-lineup window ends on/after transfer": pd.to_datetime(master.get("lineup_pre365_all_last_date"), errors="coerce") >= transfer_date,
        "Pre-event window ends on/after transfer": pd.to_datetime(master.get("event_pre365_all_last_date"), errors="coerce") >= transfer_date,
        "Origin club pre-window ends on/after transfer": pd.to_datetime(master.get("club_origin_pre365_last_date"), errors="coerce") >= transfer_date,
        "Destination club pre-window ends on/after transfer": pd.to_datetime(master.get("club_destination_pre365_last_date"), errors="coerce") >= transfer_date,
    }
    for label, mask in timing_checks.items():
        count = int(mask.fillna(False).sum())
        add(label, count, 0, count == 0, "Blocking temporal-leakage check.")

    negative_columns = [
        column for column in master.columns
        if any(token in column for token in ["_games", "_records", "_minutes", "_goals", "_assists", "_wins", "_draws", "_losses"])
        and not any(token in column for token in ["goal_difference", "valuation", "per90"])
    ]
    negative_count = 0
    for column in negative_columns:
        values = pd.to_numeric(master[column], errors="coerce")
        negative_count += int(values.lt(0).sum())
    add("Negative count or activity metrics", negative_count, 0, negative_count == 0, "Counts, minutes, goals, and records cannot be negative.")
    add(
        "Combined sample subset of sporting",
        int((master["eligible_combined_24m_provisional"] & ~master["eligible_sporting_24m_provisional"]).sum()),
        0,
        not (master["eligible_combined_24m_provisional"] & ~master["eligible_sporting_24m_provisional"]).any(),
        "Eligibility reconciliation.",
    )
    add(
        "Combined sample subset of financial",
        int((master["eligible_combined_24m_provisional"] & ~master["eligible_financial_24m_provisional"]).sum()),
        0,
        not (master["eligible_combined_24m_provisional"] & ~master["eligible_financial_24m_provisional"]).any(),
        "Eligibility reconciliation.",
    )
    return pd.DataFrame(rows)


def describe_column(column: str, core_columns: set[str]) -> tuple[str, str, str, str]:
    if column in core_columns:
        return "transfer_core", "transfer_core.csv", "Preserved transfer-core field.", "Review the transfer-core dictionary."
    if column.startswith("valuation_pre") or column.startswith("valuation_career_pre") or "pretransfer_valuation" in column:
        return "pretransfer_valuation", "player_valuations_clean.csv", "Valuation information dated on or before transfer.", "Permitted feature subject to staleness flag."
    if column.startswith("valuation_post"):
        return "posttransfer_valuation", "player_valuations_clean.csv", "Valuation information centered on a post-transfer target date.", "Outcome/diagnostic only; never a prediction feature."
    if column.startswith("appearance_career_pre") or column.startswith("appearance_pre"):
        return "pretransfer_player_activity", "appearances_clean.csv", "Player appearance aggregate before transfer.", "Permitted pre-transfer feature."
    if column.startswith("appearance_post"):
        return "posttransfer_player_outcome", "appearances_clean.csv", "Player appearance aggregate after transfer.", "Outcome/diagnostic only."
    if column.startswith("lineup_pre") or column.startswith("lineup_career_pre"):
        return "pretransfer_selection", "game_lineups_clean.csv", "Lineup-selection aggregate before transfer.", "Permitted pre-transfer feature."
    if column.startswith("lineup_post"):
        return "posttransfer_selection_outcome", "game_lineups_clean.csv", "Destination lineup-selection aggregate after transfer.", "Outcome/diagnostic only."
    if column.startswith("event_pre") or column.startswith("event_career_pre"):
        return "pretransfer_event_activity", "game_events_clean.csv", "Recorded player-event aggregate before transfer.", "Permitted feature with event-coverage flag."
    if column.startswith("event_post"):
        return "posttransfer_event_outcome", "game_events_clean.csv", "Destination player-event aggregate after transfer.", "Outcome/diagnostic only."
    if column.startswith("club_origin_pre") or column.startswith("club_destination_pre"):
        return "pretransfer_club_context", "games_clean.csv + club_games_clean.csv + competitions_clean.csv", "Club-match aggregate strictly before transfer.", "Permitted pre-transfer feature."
    if column.startswith("club_destination_post"):
        return "posttransfer_club_context", "games_clean.csv + club_games_clean.csv + competitions_clean.csv", "Destination-club match aggregate after transfer.", "Coverage/outcome only."
    return "derived_flag_or_metric", "derived", "Derived timing, coverage, eligibility, or quality-control field.", "Follow field name and timing prefix; provisional flags are not predictors."


def build_dictionary(master: pd.DataFrame, core_columns: set[str]) -> pd.DataFrame:
    rows = []
    for order, column in enumerate(master.columns, start=1):
        group, source, definition, timing = describe_column(column, core_columns)
        text_values = master[column].astype("string")
        missing_mask = text_values.isna() | text_values.fillna("").str.strip().eq("")
        rows.append(
            {
                "column_order": order,
                "column": column,
                "group": group,
                "source": source,
                "definition": definition,
                "timing_or_leakage_rule": timing,
                "missing_count": int(missing_mask.sum()),
                "non_null_count": int((~missing_mask).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_assumptions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["master_grain", "one row per transfer_event_id", "No source fact is directly joined at a lower grain."],
            ["master_population", "all cleaned transfer events", "Missing features create flags, not row filters."],
            ["pretransfer_window_days", PRE_DAYS, "Days -365 through -1 relative to transfer."],
            ["medium_term_window_days", 730, "Days -730 through -1 retain two seasons of medium-term history."],
            ["career_pre_window", "all observations before transfer", "Career-to-date history is retained once available; the 365-day rule remains a recency/eligibility gate."],
            ["post12_window", "days 0 through 364", "Twelve-month sporting outcome window."],
            ["post24_window", "days 0 through 729", "Twenty-four-month sporting outcome window."],
            ["pretransfer_valuation", "latest on or before transfer", "As-of join prevents temporal leakage."],
            ["valuation_target_12m", "transfer date + 365 days", "Nearest observation retained with gap."],
            ["valuation_target_24m", "transfer date + 730 days", "Nearest observation retained with gap."],
            ["primary_valuation_tolerance_days", 90, "Qualified post-transfer value must fall within +/-90 days."],
            ["additional_valuation_tolerances", "60 | 120 | 180", "Sensitivity flags retained for later narrowing."],
            ["destination_game_coverage_12m", SPORTING_MIN_DESTINATION_GAMES_12M, "Provisional floor before interpreting no player appearances."],
            ["destination_game_coverage_24m", SPORTING_MIN_DESTINATION_GAMES_24M, "Provisional floor before interpreting no player appearances."],
            ["movement_type", "unresolved", "Reported zero is not automatically classified as free or loan."],
            ["posttransfer_columns", "outcome_or_diagnostic_only", "Never use post-transfer fields as prediction features."],
        ],
        columns=["assumption", "value", "rationale"],
    )


def write_readme(master: pd.DataFrame) -> None:
    README_OUTPUT.write_text(
        "# Transfermarkt comprehensive transfer master\n\n"
        "Generated by `scripts/build_transfermarkt_comprehensive_master.py`.\n\n"
        f"The master contains {len(master):,} rows and {len(master.columns):,} columns at exactly one row per `transfer_event_id`. No transfer is removed for missing features or outcome maturity.\n\n"
        "## Modular inputs\n\n"
        "- `transfer_core.csv`: transfer facts and historical player/club identities.\n"
        "- `transfer_valuation_features.csv`: as-of and 12/24-month valuation fields.\n"
        "- `transfer_appearance_features.csv`: career, pre-365, and post-transfer appearance aggregates.\n"
        "- `transfer_lineup_features.csv`: pre-transfer and destination selection aggregates.\n"
        "- `transfer_event_features.csv`: goals, cards, shootouts, substitutions, and assists from event records.\n"
        "- `transfer_club_context.csv`: origin/destination pre-transfer form and destination outcome coverage.\n\n"
        "## Timing rule\n\n"
        "The three historical layers are `pre365` (recent), `pre730` (medium-term), and `career_pre` (all recorded history before transfer). The 365-day layer is also used as a recency/eligibility gate; it does not truncate older history. Columns containing `post12` or `post24` are outcomes or diagnostics and must never be model features.\n\n"
        "## Known limitations\n\n"
        "Movement type remains unresolved; zeros are not classified as free transfers. Market-value change is a proxy, not accounting ROI. Snapshot-labelled player fields are references rather than historical as-of attributes. Source-supplied transfers outside the observed game/valuation date ranges are retained and explicitly flagged.\n",
        encoding="utf-8",
    )


def write_component(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8", date_format="%Y-%m-%d", lineterminator="\n")


def read_component(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"transfer_event_id": str}, low_memory=False)
    for column in frame.columns:
        values = set(frame[column].dropna().astype(str).unique())
        if values and values.issubset({"True", "False"}):
            frame[column] = frame[column].astype("string").map({"True": True, "False": False})
    return frame


def component_date_extent(frame: pd.DataFrame, prefix: str) -> dict[str, object]:
    date_columns = [column for column in frame.columns if column.endswith("_date")]
    if not date_columns:
        return {f"{prefix}_min_date": pd.NaT, f"{prefix}_max_date": pd.NaT}
    parsed = pd.concat([pd.to_datetime(frame[column], errors="coerce") for column in date_columns], ignore_index=True)
    return {f"{prefix}_min_date": parsed.min(), f"{prefix}_max_date": parsed.max()}


def main() -> None:
    MERGED.mkdir(parents=True, exist_ok=True)
    core = pd.read_csv(CORE_PATH, dtype=str, keep_default_na=False)
    reference = reference_frame(core)
    reuse_existing = "--reuse-existing" in sys.argv

    if reuse_existing and VALUATION_OUTPUT.exists():
        print("Reusing valuation features", flush=True)
        valuations = read_component(VALUATION_OUTPUT)
        source_dates = pd.to_datetime(pd.read_csv(VALUATIONS_PATH, usecols=["date"])["date"], errors="coerce")
        valuation_meta = {
            "valuation_min_date": source_dates.min(),
            "valuation_max_date": source_dates.max(),
            "relevant_valuation_rows": None,
        }
    else:
        print("Building valuation features", flush=True)
        valuations, valuation_meta = build_valuation_features(reference)
        write_component(valuations, VALUATION_OUTPUT)

    if reuse_existing and APPEARANCE_OUTPUT.exists():
        print("Reusing appearance features", flush=True)
        appearances = read_component(APPEARANCE_OUTPUT)
        appearance_meta = {**component_date_extent(appearances, "appearance"), "relevant_appearance_rows": None}
    else:
        print("Building appearance features", flush=True)
        appearances, appearance_meta = build_appearance_features(reference)
        write_component(appearances, APPEARANCE_OUTPUT)

    if reuse_existing and LINEUP_OUTPUT.exists():
        print("Reusing lineup features", flush=True)
        lineups = read_component(LINEUP_OUTPUT)
        lineup_meta = {**component_date_extent(lineups, "lineup"), "relevant_lineup_rows": None}
    else:
        print("Building lineup features", flush=True)
        lineups, lineup_meta = build_lineup_features(reference)
        write_component(lineups, LINEUP_OUTPUT)

    if reuse_existing and EVENT_OUTPUT.exists():
        print("Reusing event features", flush=True)
        events = read_component(EVENT_OUTPUT)
        event_meta = {
            **component_date_extent(events, "event"),
            "source_event_rows": None,
            "relevant_player_event_ledger_rows": None,
        }
    else:
        print("Building event features", flush=True)
        events, event_meta = build_event_features(reference)
        write_component(events, EVENT_OUTPUT)

    print("Building club context", flush=True)
    club_context, club_meta = build_club_context(reference)
    write_component(club_context, CLUB_OUTPUT)

    components = {
        "valuation": valuations,
        "appearance": appearances,
        "lineup": lineups,
        "event": events,
        "club_context": club_context,
    }
    metadata = {**valuation_meta, **appearance_meta, **lineup_meta, **event_meta, **club_meta}

    print("Assembling comprehensive master", flush=True)
    master = core.copy()
    for name, component in components.items():
        master = master.merge(component, on="transfer_event_id", how="left", validate="one_to_one")
    component_flags = pd.DataFrame(
        {f"component_{name}_joined": True for name in components},
        index=master.index,
    )
    master = pd.concat([master.copy(), component_flags], axis=1)
    master = add_master_flags(master, metadata)

    checks = build_checks(core, components, master)
    failed = checks[checks["status"].eq("FAIL")]
    if not failed.empty:
        raise RuntimeError("Comprehensive-master validation failed:\n" + failed.to_string(index=False))

    write_component(master, MASTER_OUTPUT)
    dictionary = build_dictionary(master, set(core.columns))
    write_component(dictionary, DICTIONARY_OUTPUT)
    assumptions = build_assumptions()
    write_component(assumptions, ASSUMPTIONS_OUTPUT)
    write_component(checks, CHECKS_OUTPUT)
    write_readme(master)

    summary: dict[str, object] = {
        "grain": "One cleaned Transfermarkt transfer event.",
        "primary_key": "transfer_event_id",
        "rows": len(master),
        "columns": len(master.columns),
        "unique_players": int(master["player_id"].nunique()),
        "unique_origin_clubs": int(master["from_club_id"].nunique()),
        "unique_destination_clubs": int(master["to_club_id"].nunique()),
        "transfer_date_min": str(pd.to_datetime(master["transfer_date"]).min().date()),
        "transfer_date_max": str(pd.to_datetime(master["transfer_date"]).max().date()),
        "eligible_sporting_12m_provisional": int(master["eligible_sporting_12m_provisional"].sum()),
        "eligible_sporting_24m_provisional": int(master["eligible_sporting_24m_provisional"].sum()),
        "eligible_financial_12m_provisional": int(master["eligible_financial_12m_provisional"].sum()),
        "eligible_financial_24m_provisional": int(master["eligible_financial_24m_provisional"].sum()),
        "eligible_combined_24m_provisional": int(master["eligible_combined_24m_provisional"].sum()),
        "transfers_before_first_club_game": int(master["transfer_date_before_first_club_game"].sum()),
        "transfers_after_latest_club_game": int(master["transfer_date_after_latest_club_game"].sum()),
        "transfers_after_latest_valuation": int(master["transfer_date_after_latest_valuation"].sum()),
        "checks_passed": int(checks["status"].eq("OK").sum()),
        "checks_failed": int(checks["status"].eq("FAIL").sum()),
        "source_metadata": {key: (str(value.date()) if isinstance(value, pd.Timestamp) and pd.notna(value) else value) for key, value in metadata.items()},
        "component_files": {
            path.name: {"rows": len(frame), "columns": len(frame.columns), "sha256": sha256_file(path)}
            for path, frame in [
                (VALUATION_OUTPUT, valuations),
                (APPEARANCE_OUTPUT, appearances),
                (LINEUP_OUTPUT, lineups),
                (EVENT_OUTPUT, events),
                (CLUB_OUTPUT, club_context),
            ]
        },
        "master_sha256": sha256_file(MASTER_OUTPUT),
        "master_bytes": MASTER_OUTPUT.stat().st_size,
    }
    SUMMARY_OUTPUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {MASTER_OUTPUT.relative_to(ROOT)}: {len(master):,} rows x {len(master.columns):,} columns", flush=True)
    print(f"All {len(checks)} validation checks passed", flush=True)
    print(f"Master SHA-256: {summary['master_sha256']}", flush=True)


if __name__ == "__main__":
    main()

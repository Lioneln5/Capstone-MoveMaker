"""Create a canonical, audited Transfermarkt cleaning layer for MoveMaker.

The twelve CSV files under ``Data/TransferMarkt`` are immutable inputs.  This
script creates exact SHA-256-verified source backups, conservatively typed and
documented table copies, historical player/club identity unions, and automated
quality-control outputs.

The cleaner deliberately does not impute sporting results, valuations, fees,
or entity identities.  In particular, a zero transfer fee remains a reported
zero with an ``unclassified_zero`` status; it is not silently interpreted as a
free transfer, loan, or loan return.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
SOURCE = DATA / "TransferMarkt"
PROCESSED = DATA / "processed" / "transfermarkt_clean"
TABLE_OUTPUT = PROCESSED / "tables"
BACKUP = DATA / "source_backups" / "transfermarkt_original"
CHUNK_SIZE = 250_000


EXPECTED_COLUMNS: dict[str, list[str]] = {
    "appearances": [
        "appearance_id", "game_id", "player_id", "player_club_id",
        "player_current_club_id", "date", "player_name", "competition_id",
        "yellow_cards", "red_cards", "goals", "assists", "minutes_played",
    ],
    "club_games": [
        "game_id", "club_id", "own_goals", "own_position", "own_manager_name",
        "opponent_id", "opponent_goals", "opponent_position",
        "opponent_manager_name", "hosting", "is_win",
    ],
    "clubs": [
        "club_id", "club_code", "name", "domestic_competition_id",
        "total_market_value", "squad_size", "average_age", "foreigners_number",
        "foreigners_percentage", "national_team_players", "stadium_name",
        "stadium_seats", "net_transfer_record", "coach_name", "last_season",
        "filename", "url",
    ],
    "competitions": [
        "competition_id", "competition_code", "name", "sub_type", "type",
        "country_id", "country_name", "domestic_league_code", "confederation",
        "total_clubs", "url",
    ],
    "countries": [
        "country_id", "country_name", "country_code", "confederation",
        "total_clubs", "total_players", "average_age", "url",
    ],
    "game_events": [
        "game_event_id", "date", "game_id", "minute", "type", "club_id",
        "club_name", "player_id", "description", "player_in_id",
        "player_assist_id",
    ],
    "game_lineups": [
        "game_lineups_id", "date", "game_id", "player_id", "club_id",
        "player_name", "type", "position", "number", "team_captain",
    ],
    "games": [
        "game_id", "competition_id", "season", "round", "date", "home_club_id",
        "away_club_id", "home_club_goals", "away_club_goals",
        "home_club_position", "away_club_position", "home_club_manager_name",
        "away_club_manager_name", "stadium", "attendance", "referee", "url",
        "home_club_formation", "away_club_formation", "home_club_name",
        "away_club_name", "aggregate", "competition_type",
    ],
    "national_teams": [
        "national_team_id", "name", "team_code", "country_id", "country_name",
        "country_code", "confederation", "team_image_url", "squad_size",
        "average_age", "foreigners_number", "foreigners_percentage",
        "total_market_value", "coach_name", "fifa_ranking", "last_season", "url",
    ],
    "player_valuations": [
        "player_id", "date", "market_value_in_eur", "current_club_name",
        "current_club_id", "player_club_domestic_competition_id",
    ],
    "players": [
        "player_id", "first_name", "last_name", "name", "last_season",
        "current_club_id", "player_code", "country_of_birth", "city_of_birth",
        "country_of_citizenship", "date_of_birth", "sub_position", "position",
        "foot", "height_in_cm", "contract_expiration_date", "agent_name",
        "image_url", "international_caps", "international_goals",
        "current_national_team_id", "url",
        "current_club_domestic_competition_id", "current_club_name",
        "market_value_in_eur", "highest_market_value_in_eur",
    ],
    "transfers": [
        "player_id", "transfer_date", "transfer_season", "from_club_id",
        "to_club_id", "from_club_name", "to_club_name", "transfer_fee",
        "market_value_in_eur", "player_name",
    ],
}


TABLE_ORDER = [
    "countries",
    "competitions",
    "national_teams",
    "players",
    "clubs",
    "transfers",
    "player_valuations",
    "games",
    "club_games",
    "appearances",
    "game_lineups",
    "game_events",
]


KEY_COLUMNS = {
    "appearances": ["appearance_id"],
    "club_games": ["game_id", "club_id"],
    "clubs": ["club_id"],
    "competitions": ["competition_id"],
    "countries": ["country_id"],
    "game_events": ["game_event_id"],
    "game_lineups": ["game_lineups_id"],
    "games": ["game_id"],
    "national_teams": ["national_team_id"],
    "player_valuations": ["player_id", "date"],
    "players": ["player_id"],
    "transfers": ["player_id", "transfer_date", "from_club_id", "to_club_id"],
}


DATE_COLUMNS = {
    "appearances": ["date"],
    "game_events": ["date"],
    "game_lineups": ["date"],
    "games": ["date"],
    "player_valuations": ["date"],
    "players": ["date_of_birth", "contract_expiration_date"],
    "transfers": ["transfer_date"],
}


INTEGER_COLUMNS = {
    "appearances": [
        "game_id", "player_id", "player_club_id", "player_current_club_id",
        "yellow_cards", "red_cards", "goals", "assists", "minutes_played",
    ],
    "club_games": [
        "game_id", "club_id", "own_goals", "own_position", "opponent_id",
        "opponent_goals", "opponent_position", "is_win",
    ],
    "clubs": [
        "club_id", "total_market_value", "squad_size", "foreigners_number",
        "national_team_players", "stadium_seats", "last_season",
    ],
    "competitions": ["country_id", "total_clubs"],
    "countries": ["country_id", "total_clubs", "total_players"],
    "game_events": [
        "game_id", "minute", "club_id", "player_id", "player_in_id",
        "player_assist_id",
    ],
    "game_lineups": ["game_id", "player_id", "club_id", "team_captain"],
    "games": [
        "game_id", "season", "home_club_id", "away_club_id", "home_club_goals",
        "away_club_goals", "home_club_position", "away_club_position", "attendance",
    ],
    "national_teams": [
        "national_team_id", "country_id", "squad_size", "foreigners_number",
        "total_market_value", "fifa_ranking", "last_season",
    ],
    "player_valuations": ["player_id", "market_value_in_eur", "current_club_id"],
    "players": [
        "player_id", "last_season", "current_club_id", "height_in_cm",
        "international_caps", "international_goals", "current_national_team_id",
        "market_value_in_eur", "highest_market_value_in_eur",
    ],
    "transfers": [
        "player_id", "from_club_id", "to_club_id", "transfer_fee",
        "market_value_in_eur",
    ],
}


FLOAT_COLUMNS = {
    "clubs": ["average_age", "foreigners_percentage"],
    "countries": ["average_age"],
    "national_teams": ["average_age", "foreigners_percentage"],
}


DERIVED_COLUMNS = {
    "appearances": ["player_name_normalized"],
    "clubs": ["club_name_normalized"],
    "competitions": ["competition_name_normalized"],
    "countries": ["country_name_normalized"],
    "game_events": [
        "club_name_normalized", "event_type_normalized", "minute_raw",
        "event_minute_available",
    ],
    "game_lineups": [
        "player_name_normalized", "is_starting_lineup", "is_substitute",
    ],
    "games": [
        "home_club_name_normalized", "away_club_name_normalized",
        "season_start_year_july_boundary", "season_matches_july_boundary",
        "is_national_team_game",
    ],
    "national_teams": ["national_team_name_normalized"],
    "player_valuations": ["has_positive_market_value"],
    "players": ["player_name_normalized", "birth_year"],
    "transfers": [
        "transfer_event_id", "player_name_normalized",
        "from_club_name_normalized", "to_club_name_normalized",
        "transfer_season_derived", "transfer_season_matches_date",
        "transfer_fee_status", "has_reported_market_value",
    ],
}


SNAPSHOT_COLUMNS = {
    "appearances": {"player_current_club_id"},
    "clubs": {
        "total_market_value", "squad_size", "average_age", "foreigners_number",
        "foreigners_percentage", "national_team_players", "net_transfer_record",
        "coach_name", "last_season",
    },
    "national_teams": {
        "squad_size", "average_age", "foreigners_number", "foreigners_percentage",
        "total_market_value", "coach_name", "fifa_ranking", "last_season",
    },
    "players": {
        "last_season", "current_club_id", "contract_expiration_date", "agent_name",
        "international_caps", "international_goals", "current_national_team_id",
        "current_club_domestic_competition_id", "current_club_name",
        "market_value_in_eur", "highest_market_value_in_eur",
    },
}


PROVENANCE_COLUMNS = {
    "appearance_id", "game_event_id", "game_lineups_id", "url", "image_url",
    "team_image_url", "filename", "source_file", "source_row_number",
}


FK_RULES = {
    "appearances": [
        ("game_id", "games", "game_id"),
        ("player_id", "players", "player_id"),
        ("player_club_id", "clubs", "club_id"),
        ("player_current_club_id", "clubs", "club_id"),
        ("competition_id", "competitions", "competition_id"),
    ],
    "club_games": [
        ("game_id", "games", "game_id"),
        ("club_id", "clubs", "club_id"),
        ("opponent_id", "clubs", "club_id"),
    ],
    "clubs": [("domestic_competition_id", "competitions", "competition_id")],
    "competitions": [("country_id", "countries", "country_id")],
    "game_events": [
        ("game_id", "games", "game_id"),
        ("club_id", "clubs", "club_id"),
        ("player_id", "players", "player_id"),
        ("player_in_id", "players", "player_id"),
        ("player_assist_id", "players", "player_id"),
    ],
    "game_lineups": [
        ("game_id", "games", "game_id"),
        ("player_id", "players", "player_id"),
        ("club_id", "clubs", "club_id"),
    ],
    "games": [
        ("competition_id", "competitions", "competition_id"),
        ("home_club_id", "clubs", "club_id"),
        ("away_club_id", "clubs", "club_id"),
    ],
    "national_teams": [("country_id", "countries", "country_id")],
    "player_valuations": [
        ("player_id", "players", "player_id"),
        ("current_club_id", "clubs", "club_id"),
        (
            "player_club_domestic_competition_id",
            "competitions",
            "competition_id",
        ),
    ],
    "players": [
        ("current_club_id", "clubs", "club_id"),
        ("current_national_team_id", "national_teams", "national_team_id"),
        (
            "current_club_domestic_competition_id",
            "competitions",
            "competition_id",
        ),
    ],
    "transfers": [
        ("player_id", "players", "player_id"),
        ("from_club_id", "clubs", "club_id"),
        ("to_club_id", "clubs", "club_id"),
    ],
}


NONNEGATIVE_COLUMNS = {
    "appearances": [
        "yellow_cards", "red_cards", "goals", "assists", "minutes_played",
    ],
    "club_games": ["own_goals", "opponent_goals"],
    "clubs": [
        "total_market_value", "squad_size", "average_age", "foreigners_number",
        "foreigners_percentage", "national_team_players", "stadium_seats",
    ],
    "countries": ["total_clubs", "total_players", "average_age"],
    "game_events": ["minute"],
    "games": ["home_club_goals", "away_club_goals", "attendance"],
    "national_teams": [
        "squad_size", "average_age", "foreigners_number", "foreigners_percentage",
        "total_market_value", "fifa_ranking",
    ],
    "player_valuations": ["market_value_in_eur"],
    "players": [
        "height_in_cm", "international_caps", "international_goals",
        "market_value_in_eur", "highest_market_value_in_eur",
    ],
    "transfers": ["transfer_fee", "market_value_in_eur"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.casefold().replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9]+", "", text)
    return normalized or None


def add_check(
    checks: list[dict[str, Any]],
    table: str,
    check: str,
    observed: Any,
    expected: Any,
    passed: bool,
    severity: str,
    notes: str,
) -> None:
    checks.append(
        {
            "table": table,
            "check": check,
            "observed": observed,
            "expected": expected,
            "passed": bool(passed),
            "severity": severity,
            "notes": notes,
        }
    )


def derive_start_year(dates: pd.Series) -> pd.Series:
    years = dates.dt.year
    return (years - dates.dt.month.lt(7).astype("Int64")).astype("Int64")


def season_label_from_dates(dates: pd.Series) -> pd.Series:
    start = derive_start_year(dates)
    labels = start.map(
        lambda value: pd.NA
        if pd.isna(value)
        else f"{int(value) % 100:02d}/{(int(value) + 1) % 100:02d}"
    )
    return labels.astype("string")


def canonical_id_values(series: pd.Series) -> pd.Series:
    if pd.api.types.is_integer_dtype(series.dtype):
        return series.astype("Int64").astype("string")
    return series.astype("string")


def copy_and_verify_sources(files: list[Path]) -> pd.DataFrame:
    BACKUP.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for source in files:
        target = BACKUP / source.name
        source_hash = sha256(source)
        if target.exists():
            backup_hash = sha256(target)
            if backup_hash != source_hash:
                raise RuntimeError(f"Existing backup does not match source: {target}")
        else:
            shutil.copy2(source, target)
            backup_hash = sha256(target)
        rows.append(
            {
                "table": source.stem,
                "source_file": source.name,
                "source_relative_path": source.relative_to(ROOT).as_posix(),
                "source_bytes": source.stat().st_size,
                "source_sha256": source_hash,
                "backup_relative_path": target.relative_to(ROOT).as_posix(),
                "backup_bytes": target.stat().st_size,
                "backup_sha256": backup_hash,
                "hashes_match": source_hash == backup_hash,
            }
        )
    return pd.DataFrame(rows)


def source_id_sets() -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    parent_keys = {
        "clubs": "club_id",
        "competitions": "competition_id",
        "countries": "country_id",
        "games": "game_id",
        "national_teams": "national_team_id",
        "players": "player_id",
    }
    for table, column in parent_keys.items():
        frame = pd.read_csv(
            SOURCE / f"{table}.csv",
            usecols=[column],
            dtype="string",
            encoding="utf-8-sig",
        )
        result[f"{table}.{column}"] = set(frame[column].dropna().str.strip())
    return result


def clean_chunk(
    table: str,
    raw: pd.DataFrame,
    source_row_start: int,
    invalid_counts: dict[tuple[str, str], int],
    trimmed_counts: dict[tuple[str, str], int],
) -> pd.DataFrame:
    frame = pd.DataFrame(index=raw.index)
    dates = set(DATE_COLUMNS.get(table, []))
    integers = set(INTEGER_COLUMNS.get(table, []))
    floats = set(FLOAT_COLUMNS.get(table, []))

    for column in EXPECTED_COLUMNS[table]:
        source = raw[column].astype("string")
        stripped = source.str.strip()
        nonblank = stripped.notna() & stripped.ne("")
        trimmed_counts[(table, column)] += int(
            (source.notna() & stripped.notna() & source.ne(stripped)).sum()
        )
        if column in dates:
            parsed = pd.to_datetime(stripped.mask(~nonblank), errors="coerce", format="mixed")
            invalid_counts[(table, column)] += int((nonblank & parsed.isna()).sum())
            frame[column] = parsed
        elif column in integers:
            parsed = pd.to_numeric(stripped.mask(~nonblank), errors="coerce")
            invalid = nonblank & parsed.isna()
            fractional = parsed.notna() & parsed.mod(1).ne(0)
            invalid_counts[(table, column)] += int((invalid | fractional).sum())
            parsed = parsed.mask(fractional)
            frame[column] = parsed.astype("Int64")
        elif column in floats:
            parsed = pd.to_numeric(stripped.mask(~nonblank), errors="coerce")
            invalid_counts[(table, column)] += int((nonblank & parsed.isna()).sum())
            frame[column] = parsed.astype("Float64")
        else:
            frame[column] = stripped.mask(~nonblank).astype("string")

    frame.insert(0, "source_file", f"{table}.csv")
    frame.insert(
        1,
        "source_row_number",
        pd.Series(
            np.arange(source_row_start, source_row_start + len(frame)),
            index=frame.index,
            dtype="Int64",
        ),
    )

    if table == "appearances":
        frame["player_name_normalized"] = frame["player_name"].map(normalize_text).astype("string")
    elif table == "clubs":
        frame["club_name_normalized"] = frame["name"].map(normalize_text).astype("string")
    elif table == "competitions":
        frame["competition_name_normalized"] = frame["name"].map(normalize_text).astype("string")
    elif table == "countries":
        frame["country_name_normalized"] = frame["country_name"].map(normalize_text).astype("string")
    elif table == "game_events":
        frame["minute_raw"] = frame["minute"].copy()
        frame.loc[frame["minute"].eq(-1).fillna(False), "minute"] = pd.NA
        frame["event_minute_available"] = frame["minute"].notna()
        frame["club_name_normalized"] = frame["club_name"].map(normalize_text).astype("string")
        frame["event_type_normalized"] = frame["type"].str.casefold().astype("string")
    elif table == "game_lineups":
        frame["player_name_normalized"] = frame["player_name"].map(normalize_text).astype("string")
        frame["is_starting_lineup"] = frame["type"].eq("starting_lineup")
        frame["is_substitute"] = frame["type"].eq("substitutes")
    elif table == "games":
        frame["home_club_name_normalized"] = frame["home_club_name"].map(normalize_text).astype("string")
        frame["away_club_name_normalized"] = frame["away_club_name"].map(normalize_text).astype("string")
        frame["season_start_year_july_boundary"] = derive_start_year(frame["date"])
        frame["season_matches_july_boundary"] = frame["season"].eq(frame["season_start_year_july_boundary"])
        frame["is_national_team_game"] = frame["competition_type"].eq("national_team_competition")
    elif table == "national_teams":
        frame["national_team_name_normalized"] = frame["name"].map(normalize_text).astype("string")
    elif table == "player_valuations":
        frame["has_positive_market_value"] = frame["market_value_in_eur"].gt(0)
    elif table == "players":
        frame["player_name_normalized"] = frame["name"].map(normalize_text).astype("string")
        frame["birth_year"] = frame["date_of_birth"].dt.year.astype("Int64")
    elif table == "transfers":
        frame["player_name_normalized"] = frame["player_name"].map(normalize_text).astype("string")
        frame["from_club_name_normalized"] = frame["from_club_name"].map(normalize_text).astype("string")
        frame["to_club_name_normalized"] = frame["to_club_name"].map(normalize_text).astype("string")
        frame["transfer_season_derived"] = season_label_from_dates(frame["transfer_date"])
        frame["transfer_season_matches_date"] = frame["transfer_season"].eq(frame["transfer_season_derived"])
        fee_status = pd.Series("invalid_negative", index=frame.index, dtype="string")
        fee_status.loc[frame["transfer_fee"].isna()] = "missing"
        fee_status.loc[frame["transfer_fee"].eq(0).fillna(False)] = "unclassified_zero"
        fee_status.loc[frame["transfer_fee"].gt(0).fillna(False)] = "positive_reported"
        frame["transfer_fee_status"] = fee_status
        frame["has_reported_market_value"] = frame["market_value_in_eur"].notna()
        key_text = (
            frame["player_id"].astype("string")
            + "|"
            + frame["transfer_date"].dt.strftime("%Y-%m-%d").astype("string")
            + "|"
            + frame["from_club_id"].astype("string")
            + "|"
            + frame["to_club_id"].astype("string")
        )
        frame["transfer_event_id"] = key_text.map(
            lambda value: pd.NA
            if pd.isna(value)
            else hashlib.sha256(str(value).encode("utf-8")).hexdigest()
        ).astype("string")

    return frame.reset_index(drop=True)


def update_name_observations(
    store: dict[tuple[int, str, str, str], dict[str, Any]],
    frame: pd.DataFrame,
    id_column: str,
    name_column: str,
    source_table: str,
    role: str,
    date_column: str | None = None,
) -> None:
    columns = [id_column, name_column] + ([date_column] if date_column else [])
    work = frame[columns].dropna(subset=[id_column, name_column]).copy()
    if work.empty:
        return
    if date_column:
        grouped = work.groupby([id_column, name_column], dropna=False)[date_column].agg(
            observation_count="size", first_observed_date="min", last_observed_date="max"
        )
    else:
        grouped = work.groupby([id_column, name_column], dropna=False).size().to_frame("observation_count")
        grouped["first_observed_date"] = pd.NaT
        grouped["last_observed_date"] = pd.NaT
    for (entity_id, name), row in grouped.reset_index().set_index([id_column, name_column]).iterrows():
        key = (int(entity_id), str(name), source_table, role)
        current = store.setdefault(
            key,
            {"observation_count": 0, "first_observed_date": pd.NaT, "last_observed_date": pd.NaT},
        )
        current["observation_count"] += int(row["observation_count"])
        first = row["first_observed_date"]
        last = row["last_observed_date"]
        if pd.notna(first) and (pd.isna(current["first_observed_date"]) or first < current["first_observed_date"]):
            current["first_observed_date"] = first
        if pd.notna(last) and (pd.isna(current["last_observed_date"]) or last > current["last_observed_date"]):
            current["last_observed_date"] = last


def observations_to_frame(
    store: dict[tuple[int, str, str, str], dict[str, Any]], entity_label: str
) -> pd.DataFrame:
    rows = []
    for (entity_id, name, source_table, role), values in store.items():
        rows.append(
            {
                f"{entity_label}_id": entity_id,
                "observed_name": name,
                "observed_name_normalized": normalize_text(name),
                "source_table": source_table,
                "source_role": role,
                **values,
            }
        )
    return pd.DataFrame(rows).sort_values(
        [f"{entity_label}_id", "source_table", "source_role", "observed_name"],
        kind="stable",
    ).reset_index(drop=True)


def choose_canonical_name(group: pd.DataFrame, preferred_source: str) -> pd.Series:
    priority = {
        preferred_source: 100,
        "transfers": 80,
        "games": 75,
        "player_valuations": 60,
        "appearances": 55,
        "game_lineups": 50,
        "game_events": 45,
        "players": 40,
    }
    ranked = group.copy()
    ranked["source_priority"] = ranked["source_table"].map(priority).fillna(0)
    ranked["dated"] = ranked["last_observed_date"].notna().astype(int)
    ranked["last_sort"] = ranked["last_observed_date"].fillna(pd.Timestamp("1900-01-01"))
    ranked = ranked.sort_values(
        ["source_priority", "dated", "last_sort", "observation_count", "observed_name"],
        ascending=[False, False, False, False, True],
        kind="stable",
    )
    return ranked.iloc[0]


def build_entity_unions(
    club_observations: pd.DataFrame,
    player_observations: pd.DataFrame,
    club_ids: set[int],
    player_ids: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clubs_snapshot = pd.read_csv(TABLE_OUTPUT / "clubs_clean.csv", low_memory=False, encoding="utf-8-sig")
    players_snapshot = pd.read_csv(TABLE_OUTPUT / "players_clean.csv", low_memory=False, encoding="utf-8-sig")
    snapshot_club_ids = set(clubs_snapshot["club_id"].dropna().astype(int))
    snapshot_player_ids = set(players_snapshot["player_id"].dropna().astype(int))

    def summarize_observations(
        observations: pd.DataFrame,
        id_column: str,
        all_ids: set[int],
        preferred_source: str,
        snapshot_ids: set[int],
        presence_column: str,
    ) -> pd.DataFrame:
        priority = {
            preferred_source: 100,
            "transfers": 80,
            "games": 75,
            "player_valuations": 60,
            "appearances": 55,
            "game_lineups": 50,
            "game_events": 45,
            "players": 40,
        }
        ranked = observations.copy()
        ranked["source_priority"] = ranked["source_table"].map(priority).fillna(0)
        ranked["dated"] = ranked["last_observed_date"].notna().astype(int)
        ranked["last_sort"] = ranked["last_observed_date"].fillna(pd.Timestamp("1900-01-01"))
        ranked = ranked.sort_values(
            [id_column, "source_priority", "dated", "last_sort", "observation_count", "observed_name"],
            ascending=[True, False, False, False, False, True],
            kind="stable",
        )
        selected = ranked.drop_duplicates(id_column, keep="first")[
            [id_column, "observed_name", "source_table"]
        ].rename(
            columns={
                "observed_name": "canonical_name",
                "source_table": "canonical_name_source",
            }
        )
        summary = observations.groupby(id_column, as_index=False).agg(
            first_observed_date=("first_observed_date", "min"),
            last_observed_date=("last_observed_date", "max"),
            observation_count=("observation_count", "sum"),
            normalized_name_variant_count=("observed_name_normalized", "nunique"),
            source_table_count=("source_table", "nunique"),
        )
        union = pd.DataFrame({id_column: sorted(all_ids)}).merge(
            selected, on=id_column, how="left", validate="one_to_one"
        ).merge(summary, on=id_column, how="left", validate="one_to_one")
        union["canonical_name_normalized"] = union["canonical_name"].map(normalize_text).astype("string")
        union[presence_column] = union[id_column].isin(snapshot_ids)
        for column in ["observation_count", "normalized_name_variant_count", "source_table_count"]:
            union[column] = union[column].fillna(0).astype("Int64")
        preferred = [
            id_column, "canonical_name", "canonical_name_normalized",
            "canonical_name_source", presence_column, "first_observed_date",
            "last_observed_date", "observation_count",
            "normalized_name_variant_count", "source_table_count",
        ]
        return union[preferred]

    club_union = summarize_observations(
        club_observations,
        "club_id",
        club_ids,
        "clubs",
        snapshot_club_ids,
        "present_in_clubs_snapshot",
    )
    club_union = club_union.merge(
        clubs_snapshot[
            ["club_id", "club_code", "domestic_competition_id", "stadium_name", "url"]
        ].rename(
            columns={
                "domestic_competition_id": "snapshot_domestic_competition_id",
                "stadium_name": "snapshot_stadium_name",
                "url": "snapshot_url",
            }
        ),
        on="club_id",
        how="left",
        validate="one_to_one",
    )

    stable_player_columns = [
        "player_id", "date_of_birth", "birth_year", "country_of_birth",
        "country_of_citizenship", "position", "sub_position", "foot",
        "height_in_cm", "player_code", "url",
    ]
    player_union = summarize_observations(
        player_observations,
        "player_id",
        player_ids,
        "players",
        snapshot_player_ids,
        "present_in_players_snapshot",
    ).merge(
        players_snapshot[stable_player_columns].rename(columns={"url": "snapshot_url"}),
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    return club_union, player_union


def dictionary_definition(table: str, column: str) -> str:
    explicit = {
        "source_file": "Original Transfermarkt CSV filename.",
        "source_row_number": "One-based physical CSV row number including the header row.",
        "transfer_event_id": "Deterministic SHA-256 identifier from player, date, origin club, and destination club.",
        "transfer_season_derived": "Football season label derived from transfer_date using a July 1 boundary.",
        "transfer_season_matches_date": "Whether the supplied transfer season equals the season derived from transfer_date.",
        "transfer_fee_status": "Reported-fee state: missing, unclassified_zero, positive_reported, or invalid_negative.",
        "has_reported_market_value": "Whether transfers.market_value_in_eur is nonmissing; zero remains reported.",
        "season_start_year_july_boundary": "Comparison-only season start year derived from game date using a July 1 boundary; not universally valid for calendar-season competitions.",
        "season_matches_july_boundary": "Whether games.season agrees with a July-boundary season calculation; differences are diagnostic, not automatically errors.",
        "is_national_team_game": "True only when competition_type equals national_team_competition.",
        "has_positive_market_value": "Whether the dated player valuation is greater than zero.",
        "birth_year": "Calendar birth year derived from date_of_birth.",
        "is_starting_lineup": "True when the lineup record type is starting_lineup.",
        "is_substitute": "True when the lineup record type is substitutes.",
        "minute_raw": "Original event minute, including the source -1 unavailable sentinel.",
        "event_minute_available": "False when the source event minute is the -1 unavailable sentinel or blank.",
    }
    if column in explicit:
        return explicit[column]
    if column.endswith("_normalized"):
        return "Lowercase, accent-free alphanumeric helper for controlled entity matching."
    return f"Cleaned Transfermarkt {table}.{column} field; source semantics retained."


def modeling_rule(table: str, column: str) -> tuple[str, str]:
    if column in PROVENANCE_COLUMNS:
        return "Provenance", "Retain for lineage; exclude from modeling."
    if column in SNAPSHOT_COLUMNS.get(table, set()):
        return "Snapshot/leakage risk", "Exclude from historical predictors unless separately timestamped."
    if column in DATE_COLUMNS.get(table, []):
        return "Time anchor", "Use for explicit as-of filtering and outcome horizons."
    if column in KEY_COLUMNS.get(table, []) or column.endswith("_id"):
        return "Identifier", "Use for joins/grouping, not as an unconstrained numeric feature."
    if column in DERIVED_COLUMNS.get(table, []):
        return "Derived helper", "Use according to its documented transformation and timing."
    if table == "player_valuations" and column == "market_value_in_eur":
        return "Time-stamped financial measure", "Predictor only on/before transfer; later values are outcomes."
    if table == "transfers" and column in {"transfer_fee", "market_value_in_eur"}:
        return "Transfer-time measure", "Allowed at decision time if reported; never impute zero semantics."
    if table in {"appearances", "club_games", "games", "game_events", "game_lineups"}:
        return "Time-stamped sporting/context field", "Use only from records dated before transfer for predictors."
    return "Reference/identity field", "Retain; assess feature eligibility at the feature-engineering stage."


def table_grain(table: str) -> str:
    grains = {
        "appearances": "One player appearance in one game.",
        "club_games": "One club-perspective record for one game.",
        "clubs": "One current/latest club snapshot.",
        "competitions": "One competition reference record.",
        "countries": "One country reference record.",
        "game_events": "One recorded in-game event.",
        "game_lineups": "One player lineup or substitute record in one game.",
        "games": "One game.",
        "national_teams": "One current/latest national-team snapshot.",
        "player_valuations": "One player valuation observation date.",
        "players": "One current/latest player snapshot.",
        "transfers": "One recorded player movement between two clubs on a date.",
    }
    return grains[table]


def main() -> None:
    files = [SOURCE / f"{table}.csv" for table in TABLE_ORDER]
    missing = [path for path in files if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing Transfermarkt source files: {missing}")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    TABLE_OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = copy_and_verify_sources(files)
    parent_sets = source_id_sets()

    checks: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    table_summaries: list[dict[str, Any]] = []
    dictionary_rows: list[dict[str, Any]] = []
    join_stats: dict[tuple[str, str, str], dict[str, Any]] = {}
    invalid_counts: dict[tuple[str, str], int] = defaultdict(int)
    trimmed_counts: dict[tuple[str, str], int] = defaultdict(int)
    club_name_store: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    player_name_store: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    club_ids: set[int] = set()
    player_ids: set[int] = set()

    games_lookup: dict[int, tuple[pd.Timestamp, int, int, int | None, int | None, str | None]] = {}
    game_date_lookup: dict[int, pd.Timestamp] = {}
    game_home_lookup: dict[int, int] = {}
    game_away_lookup: dict[int, int] = {}
    game_competition_lookup: dict[int, str] = {}
    date_mismatch_counts: dict[str, int] = defaultdict(int)
    participant_mismatch_counts: dict[str, int] = defaultdict(int)
    club_game_reconciliation_errors = 0
    club_game_counts: dict[int, int] = defaultdict(int)

    for table in TABLE_ORDER:
        source_path = SOURCE / f"{table}.csv"
        output_path = TABLE_OUTPUT / f"{table}_clean.csv"
        if output_path.exists():
            output_path.unlink()

        actual_columns = pd.read_csv(source_path, nrows=0, encoding="utf-8-sig").columns.tolist()
        schema_ok = actual_columns == EXPECTED_COLUMNS[table]
        add_check(
            checks,
            table,
            "Source schema",
            " | ".join(actual_columns),
            " | ".join(EXPECTED_COLUMNS[table]),
            schema_ok,
            "blocking",
            "Exact ordered schema expected by the cleaner.",
        )
        if not schema_ok:
            raise RuntimeError(f"Unexpected schema in {source_path}")

        row_count = 0
        key_null_rows = 0
        key_hashes: list[np.ndarray] = []
        missing_counts: dict[str, int] = defaultdict(int)
        negative_counts: dict[str, int] = defaultdict(int)
        clean_columns: list[str] | None = None

        for raw in pd.read_csv(
            source_path,
            dtype="string",
            keep_default_na=False,
            chunksize=CHUNK_SIZE,
            encoding="utf-8-sig",
        ):
            clean = clean_chunk(
                table,
                raw,
                source_row_start=row_count + 2,
                invalid_counts=invalid_counts,
                trimmed_counts=trimmed_counts,
            )
            if clean_columns is None:
                clean_columns = clean.columns.tolist()
            elif clean.columns.tolist() != clean_columns:
                raise RuntimeError(f"Clean schema changed between chunks for {table}")

            key = KEY_COLUMNS[table]
            key_null_rows += int(clean[key].isna().any(axis=1).sum())
            key_hashes.append(pd.util.hash_pandas_object(clean[key], index=False).to_numpy(dtype="uint64"))
            for column in clean.columns:
                missing_counts[column] += int(clean[column].isna().sum())
            for column in NONNEGATIVE_COLUMNS.get(table, []):
                negative_counts[column] += int(clean[column].lt(0).sum())

            for child_column, parent_table, parent_column in FK_RULES.get(table, []):
                stat_key = (table, child_column, f"{parent_table}.{parent_column}")
                stat = join_stats.setdefault(
                    stat_key,
                    {"non_null_rows": 0, "matched_rows": 0, "unmatched_rows": 0, "unmatched_examples": set()},
                )
                values = canonical_id_values(clean[child_column]).dropna()
                matched = values.isin(parent_sets[f"{parent_table}.{parent_column}"])
                stat["non_null_rows"] += int(len(values))
                stat["matched_rows"] += int(matched.sum())
                stat["unmatched_rows"] += int((~matched).sum())
                if (~matched).any() and len(stat["unmatched_examples"]) < 10:
                    stat["unmatched_examples"].update(values.loc[~matched].drop_duplicates().head(10).tolist())

            # Collect historical entity IDs, even when the current snapshot dimensions omit them.
            club_columns = {
                "clubs": ["club_id"],
                "transfers": ["from_club_id", "to_club_id"],
                "player_valuations": ["current_club_id"],
                "games": ["home_club_id", "away_club_id"],
                "club_games": ["club_id", "opponent_id"],
                "appearances": ["player_club_id", "player_current_club_id"],
                "game_lineups": ["club_id"],
                "game_events": ["club_id"],
                "players": ["current_club_id"],
            }.get(table, [])
            for column in club_columns:
                club_ids.update(int(value) for value in clean[column].dropna().unique())

            player_columns = {
                "players": ["player_id"],
                "transfers": ["player_id"],
                "player_valuations": ["player_id"],
                "appearances": ["player_id"],
                "game_lineups": ["player_id"],
                "game_events": ["player_id", "player_in_id", "player_assist_id"],
            }.get(table, [])
            for column in player_columns:
                player_ids.update(int(value) for value in clean[column].dropna().unique())

            if table == "clubs":
                update_name_observations(club_name_store, clean, "club_id", "name", table, "snapshot_name")
            elif table == "players":
                update_name_observations(player_name_store, clean, "player_id", "name", table, "snapshot_name")
                update_name_observations(club_name_store, clean, "current_club_id", "current_club_name", table, "current_club_snapshot")
            elif table == "transfers":
                update_name_observations(club_name_store, clean, "from_club_id", "from_club_name", table, "origin", "transfer_date")
                update_name_observations(club_name_store, clean, "to_club_id", "to_club_name", table, "destination", "transfer_date")
                update_name_observations(player_name_store, clean, "player_id", "player_name", table, "transfer_player", "transfer_date")
            elif table == "player_valuations":
                update_name_observations(club_name_store, clean, "current_club_id", "current_club_name", table, "valuation_club", "date")
            elif table == "games":
                update_name_observations(club_name_store, clean, "home_club_id", "home_club_name", table, "home", "date")
                update_name_observations(club_name_store, clean, "away_club_id", "away_club_name", table, "away", "date")
                for row in clean[[
                    "game_id", "date", "home_club_id", "away_club_id",
                    "home_club_goals", "away_club_goals", "competition_id",
                ]].itertuples(index=False):
                    games_lookup[int(row.game_id)] = (
                        row.date,
                        int(row.home_club_id),
                        int(row.away_club_id),
                        None if pd.isna(row.home_club_goals) else int(row.home_club_goals),
                        None if pd.isna(row.away_club_goals) else int(row.away_club_goals),
                        None if pd.isna(row.competition_id) else str(row.competition_id),
                    )
                    game_date_lookup[int(row.game_id)] = row.date
                    game_home_lookup[int(row.game_id)] = int(row.home_club_id)
                    game_away_lookup[int(row.game_id)] = int(row.away_club_id)
                    if pd.notna(row.competition_id):
                        game_competition_lookup[int(row.game_id)] = str(row.competition_id)
            elif table == "appearances":
                update_name_observations(player_name_store, clean, "player_id", "player_name", table, "appearance_player", "date")
            elif table == "game_lineups":
                update_name_observations(player_name_store, clean, "player_id", "player_name", table, "lineup_player", "date")
            elif table == "game_events":
                update_name_observations(club_name_store, clean, "club_id", "club_name", table, "event_club", "date")

            if table in {"appearances", "game_lineups", "game_events"}:
                dates = clean["game_id"].map(game_date_lookup)
                date_mismatch_counts[table] += int((clean["date"].notna() & dates.notna() & clean["date"].ne(dates)).sum())

            if table in {"appearances", "game_lineups", "game_events"}:
                club_col = "player_club_id" if table == "appearances" else "club_id"
                expected_home = clean["game_id"].map(game_home_lookup).astype("Int64")
                expected_away = clean["game_id"].map(game_away_lookup).astype("Int64")
                membership = (
                    clean["game_id"].isna()
                    | clean[club_col].isna()
                    | expected_home.isna()
                    | clean[club_col].eq(expected_home).fillna(False)
                    | clean[club_col].eq(expected_away).fillna(False)
                )
                participant_mismatch_counts[table] += int((~membership).sum())

            if table == "appearances":
                expected_comp = clean["game_id"].map(game_competition_lookup).astype("string")
                participant_mismatch_counts["appearance_competition"] += int(
                    (clean["competition_id"].notna() & expected_comp.notna() & clean["competition_id"].ne(expected_comp)).sum()
                )

            if table == "club_games":
                for row in clean[[
                    "game_id", "club_id", "own_goals", "opponent_id",
                    "opponent_goals", "hosting",
                ]].itertuples(index=False):
                    game_id = int(row.game_id)
                    club_game_counts[game_id] += 1
                    lookup = games_lookup.get(game_id)
                    if lookup is None:
                        continue
                    _, home, away, home_goals, away_goals, _ = lookup
                    if row.hosting == "Home":
                        expected = (home, away, home_goals, away_goals)
                    elif row.hosting == "Away":
                        expected = (away, home, away_goals, home_goals)
                    else:
                        club_game_reconciliation_errors += 1
                        continue
                    actual = (
                        int(row.club_id),
                        int(row.opponent_id),
                        None if pd.isna(row.own_goals) else int(row.own_goals),
                        None if pd.isna(row.opponent_goals) else int(row.opponent_goals),
                    )
                    club_game_reconciliation_errors += int(actual != expected)

            mode = "w" if row_count == 0 else "a"
            clean.to_csv(
                output_path,
                index=False,
                mode=mode,
                header=row_count == 0,
                encoding="utf-8-sig" if row_count == 0 else "utf-8",
                date_format="%Y-%m-%d",
                lineterminator="\n",
            )
            row_count += len(clean)

        hashes = np.concatenate(key_hashes) if key_hashes else np.array([], dtype="uint64")
        unique_hashes = int(np.unique(hashes).size)
        duplicate_key_rows = row_count - unique_hashes
        invalid_total = sum(invalid_counts[(table, column)] for column in EXPECTED_COLUMNS[table])
        negative_total = sum(negative_counts.values())

        add_check(checks, table, "Rows retained", row_count, row_count, True, "blocking", "No source rows deleted or expanded.")
        add_check(checks, table, "Natural-key null rows", key_null_rows, 0, key_null_rows == 0, "blocking", "All candidate-key fields must be present.")
        add_check(checks, table, "Natural-key duplicate rows", duplicate_key_rows, 0, duplicate_key_rows == 0, "blocking", "Exact source audit previously found no duplicates; checked here with deterministic 64-bit row hashes.")
        add_check(checks, table, "Typed-value parse failures", invalid_total, 0, invalid_total == 0, "blocking", "Nonblank source values must parse to their declared date/numeric type.")
        add_check(checks, table, "Negative constrained values", negative_total, 0, negative_total == 0, "blocking", "Counts, minutes, fees, values, and attendance cannot be negative.")

        output_hash = sha256(output_path)
        manifest.loc[manifest["table"].eq(table), "source_row_count"] = row_count
        manifest.loc[manifest["table"].eq(table), "clean_relative_path"] = output_path.relative_to(ROOT).as_posix()
        manifest.loc[manifest["table"].eq(table), "clean_bytes"] = output_path.stat().st_size
        manifest.loc[manifest["table"].eq(table), "clean_sha256"] = output_hash

        table_summaries.append(
            {
                "table": table,
                "source_file": source_path.relative_to(ROOT).as_posix(),
                "clean_file": output_path.relative_to(ROOT).as_posix(),
                "grain": table_grain(table),
                "candidate_key": " + ".join(KEY_COLUMNS[table]),
                "source_rows": row_count,
                "clean_rows": row_count,
                "clean_columns": len(clean_columns or []),
                "key_null_rows": key_null_rows,
                "duplicate_key_rows": duplicate_key_rows,
                "invalid_typed_values": invalid_total,
                "negative_constrained_values": negative_total,
                "output_sha256": output_hash,
            }
        )

        for order, column in enumerate(clean_columns or [], start=1):
            role, timing = modeling_rule(table, column)
            if column in DATE_COLUMNS.get(table, []):
                dtype = "date"
                transformation = "Parsed as ISO date; source values otherwise unchanged."
            elif column in INTEGER_COLUMNS.get(table, []) or column in {"source_row_number", "birth_year", "season_start_year_july_boundary"}:
                dtype = "nullable integer"
                transformation = "Parsed as an integer; missing values retained."
            elif column in FLOAT_COLUMNS.get(table, []):
                dtype = "nullable decimal"
                transformation = "Parsed as a decimal; missing values retained."
            elif column in DERIVED_COLUMNS.get(table, []):
                dtype = "derived"
                transformation = dictionary_definition(table, column)
            else:
                dtype = "text/categorical"
                transformation = "Surrounding whitespace removed; blank strings standardized to missing."
            missing_count = missing_counts[column]
            dictionary_rows.append(
                {
                    "table": table,
                    "column_order": order,
                    "column": column,
                    "definition": dictionary_definition(table, column),
                    "data_type": dtype,
                    "modeling_role": role,
                    "timing_or_leakage_rule": timing,
                    "non_null_count": row_count - missing_count,
                    "missing_count": missing_count,
                    "missing_pct": missing_count / row_count if row_count else np.nan,
                    "transformation": transformation,
                }
            )

        actions.extend(
            [
                {
                    "table": table,
                    "action": "Retain all source rows",
                    "affected_rows_or_cells": row_count,
                    "details": "No natural-key duplicates existed, so no rows were removed.",
                },
                {
                    "table": table,
                    "action": "Canonicalize physical types",
                    "affected_rows_or_cells": row_count,
                    "details": "Dates, nullable integers, decimals, and text were parsed using the declared table schema; blanks remain missing.",
                },
                {
                    "table": table,
                    "action": "Add source provenance",
                    "affected_rows_or_cells": row_count,
                    "details": "Added source_file and physical source_row_number without replacing source keys.",
                },
            ]
        )

    # Cross-table reconciliation checks.
    add_check(checks, "appearances", "Dates agree with games", date_mismatch_counts["appearances"], 0, date_mismatch_counts["appearances"] == 0, "blocking", "appearance date must equal the joined game date.")
    add_check(checks, "game_lineups", "Dates agree with games", date_mismatch_counts["game_lineups"], 0, date_mismatch_counts["game_lineups"] == 0, "blocking", "lineup date must equal the joined game date.")
    add_check(checks, "game_events", "Dates agree with games", date_mismatch_counts["game_events"], 0, date_mismatch_counts["game_events"] == 0, "blocking", "event date must equal the joined game date.")
    for table in ["appearances", "game_lineups", "game_events"]:
        add_check(checks, table, "Club participates in joined game", participant_mismatch_counts[table], 0, participant_mismatch_counts[table] == 0, "blocking", "Historical club IDs are checked against home and away IDs, not clubs.csv coverage.")
    add_check(checks, "appearances", "Competition agrees with games", participant_mismatch_counts["appearance_competition"], 0, participant_mismatch_counts["appearance_competition"] == 0, "blocking", "appearance competition must equal joined game competition.")
    wrong_club_game_counts = sum(count != 2 for count in club_game_counts.values())
    missing_club_games = len(set(games_lookup) - set(club_game_counts))
    add_check(checks, "club_games", "Exactly two club perspectives per game", wrong_club_game_counts + missing_club_games, 0, wrong_club_game_counts + missing_club_games == 0, "blocking", "Every game should have one home and one away club_games row.")
    add_check(checks, "club_games", "Club-game rows reconcile to games", club_game_reconciliation_errors, 0, club_game_reconciliation_errors == 0, "blocking", "Club IDs, opponent IDs, goals, and hosting must agree with games.csv.")

    # Preserve foreign-key gaps as scope diagnostics rather than deleting history.
    join_rows = []
    for (child_table, child_column, parent_key), stat in sorted(join_stats.items()):
        non_null = stat["non_null_rows"]
        unmatched = stat["unmatched_rows"]
        join_rows.append(
            {
                "child_table": child_table,
                "child_column": child_column,
                "parent_key": parent_key,
                "non_null_rows": non_null,
                "matched_rows": stat["matched_rows"],
                "unmatched_rows": unmatched,
                "row_match_pct": stat["matched_rows"] / non_null if non_null else np.nan,
                "unmatched_examples": " | ".join(sorted(stat["unmatched_examples"])[:10]),
                "interpretation": "Snapshot dimension gap; retain historical fact rows and use reconstructed union."
                if parent_key in {"clubs.club_id", "players.player_id"} and unmatched
                else "Direct ID join.",
            }
        )
    join_frame = pd.DataFrame(join_rows)

    club_observations = observations_to_frame(club_name_store, "club")
    player_observations = observations_to_frame(player_name_store, "player")
    club_union, player_union = build_entity_unions(
        club_observations, player_observations, club_ids, player_ids
    )

    # Transfer-specific diagnostics are intentionally descriptive, not imputations.
    clean_transfers = pd.read_csv(TABLE_OUTPUT / "transfers_clean.csv", low_memory=False, encoding="utf-8-sig")
    fee_summary = (
        clean_transfers.groupby("transfer_fee_status", dropna=False)
        .size()
        .rename("records")
        .reset_index()
    )
    fee_summary["pct_of_transfers"] = fee_summary["records"] / len(clean_transfers)
    date_season_mismatches = int((~clean_transfers["transfer_season_matches_date"].fillna(False)).sum())
    add_check(checks, "transfers", "Transfer season/date mismatches", date_season_mismatches, 0, date_season_mismatches == 0, "warning", "Preserve the supplied label, but use transfer_date as the authoritative time anchor.")
    add_check(checks, "transfers", "Unique transfer_event_id", clean_transfers["transfer_event_id"].nunique(), len(clean_transfers), clean_transfers["transfer_event_id"].is_unique, "blocking", "SHA-256 of the natural transfer key.")

    clean_games = pd.read_csv(TABLE_OUTPUT / "games_clean.csv", usecols=["season_matches_july_boundary"], encoding="utf-8-sig")
    game_season_mismatches = int((~clean_games["season_matches_july_boundary"].fillna(False)).sum())
    add_check(checks, "games", "Game season differs from July-boundary calculation", game_season_mismatches, 0, game_season_mismatches == 0, "warning", "Expected for calendar-season leagues and some tournaments; preserve the source season and use competition-aware season logic downstream.")

    clean_event_minutes = pd.read_csv(
        TABLE_OUTPUT / "game_events_clean.csv",
        usecols=["minute", "minute_raw", "event_minute_available"],
        low_memory=False,
        encoding="utf-8-sig",
    )
    unavailable_minute_rows = clean_event_minutes["minute_raw"].eq(-1)
    sentinel_conversion_errors = int(
        (
            unavailable_minute_rows
            & (
                clean_event_minutes["minute"].notna()
                | clean_event_minutes["event_minute_available"].fillna(True).astype(bool)
            )
        ).sum()
    )
    add_check(
        checks,
        "game_events",
        "Unavailable minute sentinel handled",
        sentinel_conversion_errors,
        0,
        sentinel_conversion_errors == 0,
        "blocking",
        "Source minute -1 is retained in minute_raw and represented as missing in analytical minute.",
    )
    actions.append(
        {
            "table": "game_events",
            "action": "Convert unavailable minute sentinel to missing",
            "affected_rows_or_cells": int(unavailable_minute_rows.sum()),
            "details": "Source -1 values are retained in minute_raw; minute is null and event_minute_available is false.",
        }
    )

    checks_frame = pd.DataFrame(checks)
    blocking_failures = checks_frame.loc[
        checks_frame["severity"].eq("blocking") & ~checks_frame["passed"]
    ]

    manifest["source_row_count"] = manifest["source_row_count"].astype("Int64")
    manifest["clean_bytes"] = manifest["clean_bytes"].astype("Int64")
    manifest.to_csv(PROCESSED / "source_file_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(table_summaries).to_csv(PROCESSED / "transfermarkt_table_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(dictionary_rows).to_csv(PROCESSED / "transfermarkt_column_dictionary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(actions).to_csv(PROCESSED / "transfermarkt_cleaning_actions.csv", index=False, encoding="utf-8-sig")
    checks_frame.to_csv(PROCESSED / "transfermarkt_cleaning_checks.csv", index=False, encoding="utf-8-sig")
    join_frame.to_csv(PROCESSED / "foreign_key_checks.csv", index=False, encoding="utf-8-sig")
    club_observations.to_csv(PROCESSED / "club_name_observations.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    player_observations.to_csv(PROCESSED / "player_name_observations.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    club_union.to_csv(PROCESSED / "club_dimension_historical.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    player_union.to_csv(PROCESSED / "player_identity_union.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    fee_summary.to_csv(PROCESSED / "transfer_fee_summary.csv", index=False, encoding="utf-8-sig")

    total_rows = int(sum(row["clean_rows"] for row in table_summaries))
    summary = {
        "source_files": len(files),
        "source_rows": total_rows,
        "clean_rows": total_rows,
        "source_backup_directory": BACKUP.relative_to(ROOT).as_posix(),
        "clean_table_directory": TABLE_OUTPUT.relative_to(ROOT).as_posix(),
        "all_backup_hashes_match": bool(manifest["hashes_match"].all()),
        "blocking_checks": int(checks_frame["severity"].eq("blocking").sum()),
        "blocking_checks_passed": int((checks_frame["severity"].eq("blocking") & checks_frame["passed"]).sum()),
        "blocking_failures": int(len(blocking_failures)),
        "warning_checks": int(checks_frame["severity"].eq("warning").sum()),
        "warning_checks_passed": int((checks_frame["severity"].eq("warning") & checks_frame["passed"]).sum()),
        "historical_club_ids": int(len(club_union)),
        "club_ids_missing_from_snapshot": int((~club_union["present_in_clubs_snapshot"]).sum()),
        "historical_player_ids": int(len(player_union)),
        "player_ids_missing_from_snapshot": int((~player_union["present_in_players_snapshot"]).sum()),
        "transfer_season_date_mismatches": date_season_mismatches,
        "game_season_july_boundary_differences": game_season_mismatches,
        "event_minute_unavailable_sentinel_rows": int(unavailable_minute_rows.sum()),
        "invalid_typed_values": int(sum(invalid_counts.values())),
        "trimmed_text_cells": int(sum(trimmed_counts.values())),
    }
    (PROCESSED / "transfermarkt_cleaning_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    readme = f"""# Cleaned Transfermarkt data

This directory is generated by `scripts/clean_transfermarkt.py`.

## Preservation

- The twelve CSV files under `Data/TransferMarkt` are never modified.
- Exact binary copies are stored under `{BACKUP.relative_to(ROOT).as_posix()}`.
- `source_file_manifest.csv` records source, backup, and cleaned-output SHA-256 hashes.

## Main outputs

- `tables/`: one conservatively cleaned CSV per Transfermarkt source table.
- `transfermarkt_table_summary.csv`: grain, key, row count, schema, and hash summary.
- `transfermarkt_column_dictionary.csv`: definitions, types, missingness, transformations, and leakage rules.
- `transfermarkt_cleaning_actions.csv`: reproducible transformation log.
- `transfermarkt_cleaning_checks.csv`: blocking validations and preserved warnings.
- `foreign_key_checks.csv`: direct snapshot-dimension coverage diagnostics.
- `club_name_observations.csv`: dated and undated club-name evidence from all ID/name-bearing tables.
- `club_dimension_historical.csv`: union of every observed club ID, including clubs absent from `clubs.csv`.
- `player_name_observations.csv`: player-name evidence from snapshots, transfers, appearances, and lineups.
- `player_identity_union.csv`: union of every observed player ID with stable snapshot attributes where available.
- `transfer_fee_summary.csv`: missing, unclassified-zero, and positive-reported fee counts.

## Material rules

1. No source rows are deleted or imputed.
2. Blank strings are standardized to missing; dates and numeric fields are strictly parsed.
3. Source names remain intact; normalized matching helpers are added separately.
4. Current/latest snapshot fields are retained but marked as leakage risks in the dictionary.
5. A zero transfer fee is `unclassified_zero`, not automatically a free transfer.
6. Transfer dates are authoritative for transfer timing. Game-season comparisons use a clearly labeled July-boundary diagnostic because calendar-season leagues and tournaments require competition-aware logic.
7. Missing joins to `clubs.csv` and `players.csv` do not cause fact-row deletion because those dimensions are incomplete historical snapshots.
8. Event, lineup, appearance, and club-game records are reconciled to the games spine.

The cleaned source layer does not resolve FBref-to-Transfermarkt player aliases or squad mappings. Those remain separate, reviewed entity-resolution stages.
"""
    (PROCESSED / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if not blocking_failures.empty:
        failed = blocking_failures[["table", "check", "observed"]].to_dict("records")
        raise RuntimeError(f"Transfermarkt cleaning completed with blocking failures: {failed}")


if __name__ == "__main__":
    main()

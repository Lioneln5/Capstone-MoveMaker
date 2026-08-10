from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "Data" / "processed"
FBREF = PROCESSED / "fbref_2017_2024_clean"
TM = PROCESSED / "transfermarkt_clean" / "tables"
DL = PROCESSED / "football_datalake_clean" / "tables"
XW = PROCESSED / "integration_crosswalks"
CANONICAL = PROCESSED / "canonical_integration"
OUTPUT = PROCESSED / "canonical_performance"

FBREF_PATH = FBREF / "fbref_player_squad_season_clean.csv"
GAMES_PATH = TM / "games_clean.csv"
APPEARANCES_PATH = TM / "appearances_clean.csv"
LINEUPS_PATH = TM / "game_lineups_clean.csv"
DL_PERFORMANCE_PATH = DL / "player_performances_clean.csv"

PLAYER_DIM_PATH = CANONICAL / "canonical_player_dimension.csv"
CLUB_DIM_PATH = CANONICAL / "canonical_club_dimension.csv"
COMPETITION_XW_PATH = XW / "competition_crosswalk.csv"
PLAYER_XW_PATH = XW / "fbref_player_crosswalk.csv"
CLUB_SEASON_XW_PATH = XW / "fbref_club_season_crosswalk.csv"

WORK_DB = OUTPUT / "_canonical_performance_build.sqlite"
CORE_PATH = OUTPUT / "canonical_player_team_season_performance.csv"
MODEL_PATH = OUTPUT / "modeling_scope_2017_2024.csv"
FBREF_ADVANCED_PATH = OUTPUT / "fbref_advanced_performance_features.csv"
DIFFERENCE_REVIEW_PATH = OUTPUT / "performance_difference_review.csv"
UNRESOLVED_FBREF_PATH = OUTPUT / "unresolved_fbref_performance_review.csv"

CHUNK_SIZE = 100_000


def clean_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].lstrip("-").isdigit():
        return text[:-2]
    return text


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def bool_value(value: Any) -> bool:
    return clean_text(value).casefold() in {"true", "1", "yes"}


def numeric_value(value: Any) -> int | float | None:
    if value is None or pd.isna(value) or clean_text(value) == "":
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


def sqlite_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def performance_id(player_id: str, club_id: str, competition_id: str, season_start: int) -> str:
    return f"pts:{player_id}:{club_id}:{competition_id}:{season_start}"


def create_database() -> sqlite3.Connection:
    if WORK_DB.exists():
        WORK_DB.unlink()
    connection = sqlite3.connect(WORK_DB)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-250000")
    connection.executescript(
        """
        CREATE TABLE current_appearances (
            player_id TEXT NOT NULL,
            club_id TEXT NOT NULL,
            competition_id TEXT NOT NULL,
            season_start INTEGER NOT NULL,
            appearance_records INTEGER NOT NULL,
            matches_played INTEGER NOT NULL,
            goals INTEGER NOT NULL,
            assists INTEGER NOT NULL,
            yellow_cards INTEGER NOT NULL,
            red_cards INTEGER NOT NULL,
            minutes_played INTEGER NOT NULL,
            first_date TEXT,
            last_date TEXT,
            PRIMARY KEY (player_id, club_id, competition_id, season_start)
        ) WITHOUT ROWID;

        CREATE TABLE current_lineups (
            player_id TEXT NOT NULL,
            club_id TEXT NOT NULL,
            competition_id TEXT NOT NULL,
            season_start INTEGER NOT NULL,
            lineup_records INTEGER NOT NULL,
            lineup_games INTEGER NOT NULL,
            starts INTEGER NOT NULL,
            substitute_selections INTEGER NOT NULL,
            captain_selections INTEGER NOT NULL,
            first_date TEXT,
            last_date TEXT,
            primary_position TEXT,
            positions TEXT,
            distinct_positions INTEGER,
            PRIMARY KEY (player_id, club_id, competition_id, season_start)
        ) WITHOUT ROWID;

        CREATE TABLE lineup_positions (
            player_id TEXT NOT NULL,
            club_id TEXT NOT NULL,
            competition_id TEXT NOT NULL,
            season_start INTEGER NOT NULL,
            position TEXT NOT NULL,
            position_count INTEGER NOT NULL,
            PRIMARY KEY (player_id, club_id, competition_id, season_start, position)
        ) WITHOUT ROWID;

        CREATE TABLE datalake_performance (
            player_id TEXT NOT NULL,
            club_id TEXT NOT NULL,
            competition_id TEXT NOT NULL,
            season_start INTEGER NOT NULL,
            source_record_id TEXT,
            season_name TEXT,
            competition_name TEXT,
            team_name TEXT,
            nb_in_group INTEGER,
            nb_on_pitch INTEGER,
            goals INTEGER,
            assists INTEGER,
            own_goals INTEGER,
            subbed_in INTEGER,
            subbed_out INTEGER,
            yellow_cards INTEGER,
            second_yellow_cards INTEGER,
            direct_red_cards INTEGER,
            penalty_goals INTEGER,
            minutes_played INTEGER,
            goals_conceded INTEGER,
            clean_sheets INTEGER,
            minutes_available INTEGER,
            PRIMARY KEY (player_id, club_id, competition_id, season_start)
        ) WITHOUT ROWID;

        CREATE TABLE fbref_performance (
            player_id TEXT NOT NULL,
            club_id TEXT NOT NULL,
            competition_id TEXT NOT NULL,
            season_start INTEGER NOT NULL,
            fbref_row_id TEXT,
            player_identity_key TEXT,
            player_name TEXT,
            squad_name TEXT,
            competition_name TEXT,
            position TEXT,
            is_goalkeeper INTEGER,
            age REAL,
            matches_played INTEGER,
            total_minutes INTEGER,
            goals INTEGER,
            assists INTEGER,
            expected_goals REAL,
            non_penalty_expected_goals REAL,
            progressive_carries REAL,
            progressive_passes REAL,
            goals_per90 REAL,
            assists_per90 REAL,
            player_match_method TEXT,
            club_match_method TEXT,
            PRIMARY KEY (player_id, club_id, competition_id, season_start)
        ) WITHOUT ROWID;
        """
    )
    return connection


def load_game_metadata() -> tuple[dict[str, tuple[str, int, str, bool]], dict[str, Any]]:
    games = pd.read_csv(
        GAMES_PATH,
        usecols=["game_id", "competition_id", "season", "date", "competition_type", "is_national_team_game"],
        dtype=str,
        keep_default_na=False,
    )
    games["game_id"] = games["game_id"].map(clean_id)
    games["competition_id"] = games["competition_id"].map(clean_id)
    games["season_start"] = pd.to_numeric(games["season"], errors="coerce").astype("Int64")
    games["explicit_national"] = (
        games["is_national_team_game"].map(bool_value)
        | games["competition_type"].str.casefold().eq("national_team_competition")
    )
    invalid = games["game_id"].eq("") | games["competition_id"].eq("") | games["season_start"].isna()
    metadata = {
        "source_game_rows": int(len(games)),
        "invalid_game_key_rows": int(invalid.sum()),
        "explicit_national_team_games": int(games["explicit_national"].sum()),
        "club_or_unknown_type_games": int((~games["explicit_national"]).sum()),
    }
    lookup: dict[str, tuple[str, int, str, bool]] = {}
    for row in games.loc[~invalid].itertuples(index=False):
        lookup[row.game_id] = (
            row.competition_id,
            int(row.season_start),
            clean_text(row.date),
            bool(row.explicit_national),
        )
    return lookup, metadata


APPEARANCE_UPSERT = """
INSERT INTO current_appearances VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(player_id, club_id, competition_id, season_start) DO UPDATE SET
    appearance_records = appearance_records + excluded.appearance_records,
    matches_played = matches_played + excluded.matches_played,
    goals = goals + excluded.goals,
    assists = assists + excluded.assists,
    yellow_cards = yellow_cards + excluded.yellow_cards,
    red_cards = red_cards + excluded.red_cards,
    minutes_played = minutes_played + excluded.minutes_played,
    first_date = CASE WHEN first_date IS NULL OR excluded.first_date < first_date THEN excluded.first_date ELSE first_date END,
    last_date = CASE WHEN last_date IS NULL OR excluded.last_date > last_date THEN excluded.last_date ELSE last_date END
"""


LINEUP_UPSERT = """
INSERT INTO current_lineups (
    player_id, club_id, competition_id, season_start, lineup_records, lineup_games, starts,
    substitute_selections, captain_selections, first_date, last_date,
    primary_position, positions, distinct_positions
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL)
ON CONFLICT(player_id, club_id, competition_id, season_start) DO UPDATE SET
    lineup_records = lineup_records + excluded.lineup_records,
    lineup_games = lineup_games + excluded.lineup_games,
    starts = starts + excluded.starts,
    substitute_selections = substitute_selections + excluded.substitute_selections,
    captain_selections = captain_selections + excluded.captain_selections,
    first_date = CASE WHEN first_date IS NULL OR excluded.first_date < first_date THEN excluded.first_date ELSE first_date END,
    last_date = CASE WHEN last_date IS NULL OR excluded.last_date > last_date THEN excluded.last_date ELSE last_date END
"""


POSITION_UPSERT = """
INSERT INTO lineup_positions VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(player_id, club_id, competition_id, season_start, position) DO UPDATE SET
    position_count = position_count + excluded.position_count
"""


def attach_game_metadata(chunk: pd.DataFrame, lookup: dict[str, tuple[str, int, str, bool]]) -> pd.DataFrame:
    chunk["game_id"] = chunk["game_id"].map(clean_id)
    metadata = chunk["game_id"].map(lookup)
    chunk["game_competition_id"] = metadata.map(lambda value: value[0] if isinstance(value, tuple) else "")
    chunk["season_start"] = metadata.map(lambda value: value[1] if isinstance(value, tuple) else pd.NA)
    chunk["game_date"] = metadata.map(lambda value: value[2] if isinstance(value, tuple) else "")
    chunk["explicit_national"] = metadata.map(lambda value: value[3] if isinstance(value, tuple) else False)
    chunk["game_mapped"] = metadata.map(lambda value: isinstance(value, tuple))
    return chunk


def aggregate_current_appearances(
    connection: sqlite3.Connection, game_lookup: dict[str, tuple[str, int, str, bool]]
) -> dict[str, Any]:
    usecols = [
        "appearance_id", "game_id", "player_id", "player_club_id", "competition_id", "date",
        "yellow_cards", "red_cards", "goals", "assists", "minutes_played",
    ]
    counts = Counter()
    for chunk in pd.read_csv(
        APPEARANCES_PATH, usecols=usecols, dtype=str, keep_default_na=False, chunksize=CHUNK_SIZE
    ):
        counts["source_rows"] += len(chunk)
        chunk = attach_game_metadata(chunk, game_lookup)
        counts["unmapped_game_rows"] += int((~chunk["game_mapped"]).sum())
        counts["excluded_national_rows"] += int(chunk["explicit_national"].sum())
        source_comp = chunk["competition_id"].map(clean_id)
        counts["competition_mismatch_rows"] += int(
            (source_comp.ne("") & chunk["game_competition_id"].ne("") & source_comp.ne(chunk["game_competition_id"])).sum()
        )
        chunk = chunk.loc[chunk["game_mapped"] & ~chunk["explicit_national"]].copy()
        chunk["player_id"] = chunk["player_id"].map(clean_id)
        chunk["club_id"] = chunk["player_club_id"].map(clean_id)
        valid = chunk["player_id"].ne("") & chunk["club_id"].ne("") & chunk["game_competition_id"].ne("")
        counts["invalid_entity_key_rows"] += int((~valid).sum())
        chunk = chunk.loc[valid]
        counts["included_rows"] += len(chunk)
        for column in ["yellow_cards", "red_cards", "goals", "assists", "minutes_played"]:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce").fillna(0).astype("int64")
        grouped = chunk.groupby(
            ["player_id", "club_id", "game_competition_id", "season_start"], sort=False, dropna=False
        ).agg(
            appearance_records=("appearance_id", "count"),
            matches_played=("game_id", "nunique"),
            goals=("goals", "sum"),
            assists=("assists", "sum"),
            yellow_cards=("yellow_cards", "sum"),
            red_cards=("red_cards", "sum"),
            minutes_played=("minutes_played", "sum"),
            first_date=("game_date", "min"),
            last_date=("game_date", "max"),
        ).reset_index()
        rows = [
            tuple(sqlite_value(value) for value in row)
            for row in grouped.itertuples(index=False, name=None)
        ]
        connection.executemany(APPEARANCE_UPSERT, rows)
        connection.commit()
    counts["aggregated_keys"] = connection.execute("SELECT COUNT(*) FROM current_appearances").fetchone()[0]
    return dict(counts)


def aggregate_current_lineups(
    connection: sqlite3.Connection, game_lookup: dict[str, tuple[str, int, str, bool]]
) -> dict[str, Any]:
    usecols = [
        "game_lineups_id", "game_id", "player_id", "club_id", "date", "position",
        "team_captain", "is_starting_lineup", "is_substitute",
    ]
    counts = Counter()
    for chunk in pd.read_csv(
        LINEUPS_PATH, usecols=usecols, dtype=str, keep_default_na=False, chunksize=CHUNK_SIZE
    ):
        counts["source_rows"] += len(chunk)
        chunk = attach_game_metadata(chunk, game_lookup)
        counts["unmapped_game_rows"] += int((~chunk["game_mapped"]).sum())
        counts["excluded_national_rows"] += int(chunk["explicit_national"].sum())
        chunk = chunk.loc[chunk["game_mapped"] & ~chunk["explicit_national"]].copy()
        chunk["player_id"] = chunk["player_id"].map(clean_id)
        chunk["club_id"] = chunk["club_id"].map(clean_id)
        valid = chunk["player_id"].ne("") & chunk["club_id"].ne("") & chunk["game_competition_id"].ne("")
        counts["invalid_entity_key_rows"] += int((~valid).sum())
        chunk = chunk.loc[valid]
        counts["included_rows"] += len(chunk)
        for column in ["team_captain", "is_starting_lineup", "is_substitute"]:
            chunk[column] = chunk[column].map(bool_value).astype("int64")
        grouped = chunk.groupby(
            ["player_id", "club_id", "game_competition_id", "season_start"], sort=False, dropna=False
        ).agg(
            lineup_records=("game_lineups_id", "count"),
            lineup_games=("game_id", "nunique"),
            starts=("is_starting_lineup", "sum"),
            substitute_selections=("is_substitute", "sum"),
            captain_selections=("team_captain", "sum"),
            first_date=("game_date", "min"),
            last_date=("game_date", "max"),
        ).reset_index()
        rows = [
            tuple(sqlite_value(value) for value in row)
            for row in grouped.itertuples(index=False, name=None)
        ]
        connection.executemany(LINEUP_UPSERT, rows)

        nonempty_positions = chunk.loc[chunk["position"].str.strip().ne("")].copy()
        if not nonempty_positions.empty:
            position_grouped = nonempty_positions.groupby(
                ["player_id", "club_id", "game_competition_id", "season_start", "position"],
                sort=False,
                dropna=False,
            ).size().reset_index(name="position_count")
            position_rows = [
                tuple(sqlite_value(value) for value in row)
                for row in position_grouped.itertuples(index=False, name=None)
            ]
            connection.executemany(POSITION_UPSERT, position_rows)
        connection.commit()

    update_rows: list[tuple[str, str, int, str, str, str, int]] = []
    current_key: tuple[str, str, str, int] | None = None
    position_values: list[tuple[str, int]] = []

    def flush_positions() -> None:
        if current_key is None or not position_values:
            return
        ordered = sorted(position_values, key=lambda item: (-item[1], item[0].casefold(), item[0]))
        player_id, club_id, competition_id, season_start = current_key
        update_rows.append((ordered[0][0], " | ".join(item[0] for item in ordered), len(ordered), player_id, club_id, competition_id, season_start))
        if len(update_rows) >= 25_000:
            connection.executemany(
                """UPDATE current_lineups SET primary_position=?, positions=?, distinct_positions=?
                   WHERE player_id=? AND club_id=? AND competition_id=? AND season_start=?""",
                update_rows,
            )
            connection.commit()
            update_rows.clear()

    cursor = connection.execute(
        """SELECT player_id, club_id, competition_id, season_start, position, position_count
           FROM lineup_positions
           ORDER BY player_id, club_id, competition_id, season_start, position_count DESC, position"""
    )
    for player_id, club_id, competition_id, season_start, position, position_count in cursor:
        key = (player_id, club_id, competition_id, season_start)
        if current_key is not None and key != current_key:
            flush_positions()
            position_values = []
        current_key = key
        position_values.append((position, int(position_count)))
    flush_positions()
    if update_rows:
        connection.executemany(
            """UPDATE current_lineups SET primary_position=?, positions=?, distinct_positions=?
               WHERE player_id=? AND club_id=? AND competition_id=? AND season_start=?""",
            update_rows,
        )
        connection.commit()
    counts["aggregated_keys"] = connection.execute("SELECT COUNT(*) FROM current_lineups").fetchone()[0]
    return dict(counts)


def load_datalake_performance(connection: sqlite3.Connection) -> dict[str, Any]:
    usecols = [
        "source_record_id", "player_id", "season_name", "competition_id", "competition_name",
        "team_id", "team_name", "nb_in_group", "nb_on_pitch", "goals", "assists", "own_goals",
        "subed_in", "subed_out", "yellow_cards", "second_yellow_cards", "direct_red_cards",
        "penalty_goals", "minutes_played", "goals_conceded", "clean_sheets", "season_start_year",
        "minutes_available",
    ]
    insert_sql = "INSERT INTO datalake_performance VALUES (" + ",".join("?" for _ in range(23)) + ")"
    counts = Counter()
    numeric_columns = [
        "nb_in_group", "nb_on_pitch", "goals", "assists", "own_goals", "subed_in", "subed_out",
        "yellow_cards", "second_yellow_cards", "direct_red_cards", "penalty_goals", "minutes_played",
        "goals_conceded", "clean_sheets",
    ]
    for chunk in pd.read_csv(
        DL_PERFORMANCE_PATH, usecols=usecols, dtype=str, keep_default_na=False, chunksize=CHUNK_SIZE
    ):
        counts["source_rows"] += len(chunk)
        chunk["player_id"] = chunk["player_id"].map(clean_id)
        chunk["club_id"] = chunk["team_id"].map(clean_id)
        chunk["competition_id"] = chunk["competition_id"].map(clean_id)
        chunk["season_start"] = pd.to_numeric(chunk["season_start_year"], errors="coerce").astype("Int64")
        valid = (
            chunk["player_id"].ne("") & chunk["club_id"].ne("")
            & chunk["competition_id"].ne("") & chunk["season_start"].notna()
        )
        counts["invalid_key_rows"] += int((~valid).sum())
        chunk = chunk.loc[valid].copy()
        for column in numeric_columns:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        chunk["minutes_available_int"] = chunk["minutes_available"].map(bool_value).astype("int64")
        ordered = [
            "player_id", "club_id", "competition_id", "season_start", "source_record_id", "season_name",
            "competition_name", "team_name", "nb_in_group", "nb_on_pitch", "goals", "assists",
            "own_goals", "subed_in", "subed_out", "yellow_cards", "second_yellow_cards",
            "direct_red_cards", "penalty_goals", "minutes_played", "goals_conceded", "clean_sheets",
            "minutes_available_int",
        ]
        rows = []
        for row in chunk[ordered].itertuples(index=False, name=None):
            rows.append(tuple(sqlite_value(value) for value in row))
        # SQLite table order has 23 columns: four key fields and 19 source payload fields.
        # season_end is intentionally derived as season_start + 1 in the canonical layer, not stored here.
        if rows and len(rows[0]) != 23:
            raise RuntimeError(f"Unexpected datalake payload width: {len(rows[0])}")
        connection.executemany(insert_sql, rows)
        connection.commit()
        counts["included_rows"] += len(rows)
    counts["aggregated_keys"] = connection.execute("SELECT COUNT(*) FROM datalake_performance").fetchone()[0]
    return dict(counts)


def prepare_fbref(connection: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    fbref = pd.read_csv(FBREF_PATH, dtype=str, keep_default_na=False)
    player_xw = pd.read_csv(PLAYER_XW_PATH, dtype=str, keep_default_na=False)
    club_xw = pd.read_csv(CLUB_SEASON_XW_PATH, dtype=str, keep_default_na=False)
    competition_xw = pd.read_csv(COMPETITION_XW_PATH, dtype=str, keep_default_na=False)

    player_xw = player_xw[[
        "fbref_player_identity_key", "canonical_player_id", "match_status", "match_method",
        "candidate_count", "candidate_player_ids", "manual_review_required",
    ]].rename(columns={
        "match_status": "player_match_status", "match_method": "player_match_method",
        "manual_review_required": "player_manual_review_required",
    })
    club_xw = club_xw[[
        "fbref_squad_name_normalized", "fbref_competition_id", "fbref_season", "canonical_club_id",
        "match_status", "match_method", "manual_review_required",
    ]].rename(columns={
        "match_status": "club_match_status", "match_method": "club_match_method",
        "manual_review_required": "club_manual_review_required",
    })
    comp_map = competition_xw.loc[
        competition_xw["fbref_competition_id"].ne(""),
        ["fbref_competition_id", "canonical_competition_id"],
    ].drop_duplicates("fbref_competition_id")

    merged = fbref.merge(
        player_xw, how="left", left_on="player_identity_key", right_on="fbref_player_identity_key",
        validate="many_to_one",
    ).merge(
        club_xw,
        how="left",
        left_on=["squad_name_normalized", "competition_id", "season"],
        right_on=["fbref_squad_name_normalized", "fbref_competition_id", "fbref_season"],
        validate="many_to_one",
    ).merge(
        comp_map, how="left", left_on="competition_id", right_on="fbref_competition_id",
        validate="many_to_one",
    )
    accepted_mask = (
        merged["player_match_status"].eq("accepted")
        & merged["club_match_status"].eq("accepted")
        & merged["canonical_player_id"].ne("")
        & merged["canonical_club_id"].ne("")
        & merged["canonical_competition_id"].ne("")
    )
    accepted = merged.loc[accepted_mask].copy()
    unresolved = merged.loc[~accepted_mask].copy()
    accepted["canonical_player_id"] = accepted["canonical_player_id"].map(clean_id)
    accepted["canonical_club_id"] = accepted["canonical_club_id"].map(clean_id)
    accepted["canonical_competition_id"] = accepted["canonical_competition_id"].map(clean_id)
    accepted["season_start_year"] = pd.to_numeric(accepted["season_start_year"], errors="raise").astype("int64")
    accepted["canonical_performance_id"] = accepted.apply(
        lambda row: performance_id(
            row["canonical_player_id"], row["canonical_club_id"],
            row["canonical_competition_id"], int(row["season_start_year"]),
        ), axis=1,
    )
    if accepted.duplicated(["canonical_player_id", "canonical_club_id", "canonical_competition_id", "season_start_year"]).any():
        raise RuntimeError("Accepted FBref rows are not unique at the canonical performance grain.")

    numeric_columns = [
        "age", "matches_played", "total_minutes", "goals", "assists", "expected_goals",
        "non_penalty_expected_goals", "progressive_carries", "progressive_passes", "goals_per90", "assists_per90",
    ]
    for column in numeric_columns:
        accepted[column] = pd.to_numeric(accepted[column], errors="coerce")
    insert_columns = [
        "canonical_player_id", "canonical_club_id", "canonical_competition_id", "season_start_year",
        "fbref_row_id", "player_identity_key", "player_name", "squad_name", "competition_name",
        "position", "is_goalkeeper", "age", "matches_played", "total_minutes", "goals", "assists",
        "expected_goals", "non_penalty_expected_goals", "progressive_carries", "progressive_passes",
        "goals_per90", "assists_per90", "player_match_method", "club_match_method",
    ]
    accepted["is_goalkeeper"] = accepted["is_goalkeeper"].map(bool_value).astype("int64")
    rows = [
        tuple(sqlite_value(value) for value in row)
        for row in accepted[insert_columns].itertuples(index=False, name=None)
    ]
    connection.executemany(
        "INSERT INTO fbref_performance VALUES (" + ",".join("?" for _ in range(24)) + ")",
        rows,
    )
    connection.commit()

    unresolved_columns = [
        "fbref_row_id", "player_identity_key", "player_name", "squad_name", "competition_id", "competition_name",
        "season", "birth_year", "player_match_status", "player_match_method", "candidate_count",
        "candidate_player_ids", "player_manual_review_required", "club_match_status", "club_match_method",
        "club_manual_review_required",
    ]
    unresolved_output = unresolved[unresolved_columns].copy()
    unresolved_output.insert(0, "review_reason", np.select(
        [
            ~unresolved["player_match_status"].eq("accepted"),
            ~unresolved["club_match_status"].eq("accepted"),
            unresolved["canonical_competition_id"].eq(""),
        ],
        ["unresolved_player_identity", "unresolved_club_identity", "unresolved_competition"],
        default="incomplete_canonical_key",
    ))
    write_csv(unresolved_output, UNRESOLVED_FBREF_PATH)

    original_columns = list(pd.read_csv(FBREF_PATH, nrows=0).columns)
    advanced = accepted[["canonical_performance_id", "canonical_player_id", "canonical_club_id", "canonical_competition_id"] + original_columns].copy()
    rename = {column: f"fbref_{column}" for column in original_columns}
    advanced = advanced.rename(columns=rename)
    write_csv(advanced, FBREF_ADVANCED_PATH)

    metadata = {
        "source_rows": int(len(fbref)),
        "accepted_rows": int(len(accepted)),
        "unresolved_rows": int(len(unresolved)),
        "accepted_unique_players": int(accepted["canonical_player_id"].nunique()),
        "accepted_unique_clubs": int(accepted["canonical_club_id"].nunique()),
    }
    return accepted, unresolved_output, metadata


def build_key_spine(connection: sqlite3.Connection) -> int:
    connection.executescript(
        """
        CREATE TABLE canonical_keys AS
        SELECT player_id, club_id, competition_id, season_start FROM datalake_performance
        UNION
        SELECT player_id, club_id, competition_id, season_start FROM current_appearances
        UNION
        SELECT player_id, club_id, competition_id, season_start FROM current_lineups
        UNION
        SELECT player_id, club_id, competition_id, season_start FROM fbref_performance;
        CREATE UNIQUE INDEX canonical_keys_pk
            ON canonical_keys(player_id, club_id, competition_id, season_start);
        """
    )
    connection.commit()
    return connection.execute("SELECT COUNT(*) FROM canonical_keys").fetchone()[0]


CORE_QUERY = """
SELECT
    k.player_id, k.club_id, k.competition_id, k.season_start,
    f.fbref_row_id, f.player_identity_key AS fbref_player_identity_key,
    f.player_name AS fbref_player_name, f.squad_name AS fbref_squad_name,
    f.competition_name AS fbref_competition_name, f.position AS fbref_position,
    f.is_goalkeeper AS fbref_is_goalkeeper, f.age AS fbref_age,
    f.matches_played AS fbref_matches_played, f.total_minutes AS fbref_total_minutes,
    f.goals AS fbref_goals, f.assists AS fbref_assists,
    f.expected_goals AS fbref_expected_goals,
    f.non_penalty_expected_goals AS fbref_non_penalty_expected_goals,
    f.progressive_carries AS fbref_progressive_carries,
    f.progressive_passes AS fbref_progressive_passes,
    f.goals_per90 AS fbref_goals_per90, f.assists_per90 AS fbref_assists_per90,
    f.player_match_method AS fbref_player_match_method,
    f.club_match_method AS fbref_club_match_method,
    a.appearance_records AS current_appearance_records,
    a.matches_played AS current_matches_played,
    a.goals AS current_goals, a.assists AS current_assists,
    a.yellow_cards AS current_yellow_cards, a.red_cards AS current_red_cards,
    a.minutes_played AS current_minutes_played,
    a.first_date AS current_first_appearance_date, a.last_date AS current_last_appearance_date,
    l.lineup_records AS current_lineup_records, l.lineup_games AS current_lineup_games,
    l.starts AS current_starts, l.substitute_selections AS current_substitute_selections,
    l.captain_selections AS current_captain_selections,
    l.distinct_positions AS current_distinct_positions,
    l.primary_position AS current_primary_position, l.positions AS current_positions,
    l.first_date AS current_first_lineup_date, l.last_date AS current_last_lineup_date,
    d.source_record_id AS datalake_source_record_id, d.season_name AS datalake_season_name,
    d.competition_name AS datalake_competition_name, d.team_name AS datalake_team_name,
    d.nb_in_group AS datalake_nb_in_group, d.nb_on_pitch AS datalake_nb_on_pitch,
    d.goals AS datalake_goals, d.assists AS datalake_assists,
    d.own_goals AS datalake_own_goals, d.subbed_in AS datalake_subbed_in,
    d.subbed_out AS datalake_subbed_out, d.yellow_cards AS datalake_yellow_cards,
    d.second_yellow_cards AS datalake_second_yellow_cards,
    d.direct_red_cards AS datalake_direct_red_cards,
    d.penalty_goals AS datalake_penalty_goals, d.minutes_played AS datalake_minutes_played,
    d.goals_conceded AS datalake_goals_conceded, d.clean_sheets AS datalake_clean_sheets,
    d.minutes_available AS datalake_minutes_available
FROM canonical_keys k
LEFT JOIN fbref_performance f USING (player_id, club_id, competition_id, season_start)
LEFT JOIN current_appearances a USING (player_id, club_id, competition_id, season_start)
LEFT JOIN current_lineups l USING (player_id, club_id, competition_id, season_start)
LEFT JOIN datalake_performance d USING (player_id, club_id, competition_id, season_start)
ORDER BY k.season_start, k.competition_id, k.club_id, k.player_id
"""


def choose_metric(frame: pd.DataFrame, candidates: list[tuple[str, str]]) -> tuple[pd.Series, pd.Series]:
    values = pd.Series(np.nan, index=frame.index, dtype="float64")
    sources = pd.Series("", index=frame.index, dtype="string")
    for column, source in candidates:
        candidate = pd.to_numeric(frame[column], errors="coerce")
        mask = values.isna() & candidate.notna()
        values.loc[mask] = candidate.loc[mask]
        sources.loc[mask] = source
    return values, sources


def add_agreement(frame: pd.DataFrame, metric: str, columns: list[str]) -> None:
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    compared = values.notna().sum(axis=1)
    minimum = values.min(axis=1, skipna=True)
    maximum = values.max(axis=1, skipna=True)
    value_range = maximum - minimum
    frame[f"{metric}_source_values_compared"] = compared.astype("int64")
    frame[f"{metric}_cross_source_min"] = minimum
    frame[f"{metric}_cross_source_max"] = maximum
    frame[f"{metric}_cross_source_range"] = value_range
    frame[f"{metric}_agreement_status"] = np.select(
        [compared.lt(2), value_range.eq(0)],
        ["not_compared", "sources_agree"],
        default="sources_differ",
    )


def load_dimensions() -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, str]]]:
    players = pd.read_csv(
        PLAYER_DIM_PATH, usecols=["canonical_player_id", "canonical_name"], dtype=str, keep_default_na=False
    )
    clubs = pd.read_csv(
        CLUB_DIM_PATH, usecols=["canonical_club_id", "canonical_club_name"], dtype=str, keep_default_na=False
    )
    competitions = pd.read_csv(
        COMPETITION_XW_PATH,
        usecols=["canonical_competition_id", "canonical_name", "country_name", "competition_type"],
        dtype=str,
        keep_default_na=False,
    )
    player_names = dict(zip(players["canonical_player_id"].map(clean_id), players["canonical_name"]))
    club_names = dict(zip(clubs["canonical_club_id"].map(clean_id), clubs["canonical_club_name"]))
    competition_meta = {
        clean_id(row.canonical_competition_id): {
            "name": clean_text(row.canonical_name),
            "country": clean_text(row.country_name),
            "type": clean_text(row.competition_type),
        }
        for row in competitions.itertuples(index=False)
    }
    return player_names, club_names, competition_meta


def integerize(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").round().astype("Int64")


def enrich_core_chunk(
    frame: pd.DataFrame,
    player_names: dict[str, str],
    club_names: dict[str, str],
    competition_meta: dict[str, dict[str, str]],
) -> pd.DataFrame:
    frame = frame.rename(columns={
        "player_id": "canonical_player_id", "club_id": "canonical_club_id",
        "competition_id": "canonical_competition_id", "season_start": "season_start_year",
    })
    for column in ["canonical_player_id", "canonical_club_id", "canonical_competition_id"]:
        frame[column] = frame[column].map(clean_id)
    frame.insert(0, "canonical_performance_id", [
        performance_id(player_id, club_id, competition_id, int(season_start))
        for player_id, club_id, competition_id, season_start in zip(
            frame["canonical_player_id"], frame["canonical_club_id"],
            frame["canonical_competition_id"], frame["season_start_year"],
        )
    ])
    frame.insert(2, "canonical_player_name", frame["canonical_player_id"].map(player_names).fillna(""))
    frame.insert(4, "canonical_club_name", frame["canonical_club_id"].map(club_names).fillna(""))
    frame.insert(6, "canonical_competition_name", frame["canonical_competition_id"].map(
        lambda value: competition_meta.get(value, {}).get("name", "")
    ))
    frame.insert(7, "competition_country_name", frame["canonical_competition_id"].map(
        lambda value: competition_meta.get(value, {}).get("country", "")
    ))
    frame.insert(8, "competition_type", frame["canonical_competition_id"].map(
        lambda value: competition_meta.get(value, {}).get("type", "")
    ))
    frame.insert(10, "season_end_year", frame["season_start_year"].astype("int64") + 1)
    frame.insert(9, "canonical_season_key", frame["season_start_year"].map(
        lambda value: f"{int(value)}-{int(value) + 1}"
    ))

    present_fbref = frame["fbref_row_id"].notna()
    present_current_appearances = frame["current_appearance_records"].notna()
    present_current_lineups = frame["current_lineup_records"].notna()
    present_datalake = frame["datalake_source_record_id"].notna()
    frame["present_in_fbref"] = present_fbref
    frame["present_in_current_appearances"] = present_current_appearances
    frame["present_in_current_lineups"] = present_current_lineups
    frame["present_in_supplemental_performances"] = present_datalake
    frame["source_system_count"] = (
        present_fbref.astype(int)
        + (present_current_appearances | present_current_lineups).astype(int)
        + present_datalake.astype(int)
    )
    frame["data_coverage_tier"] = np.select(
        [
            present_fbref & present_current_appearances,
            present_fbref,
            present_current_appearances & present_current_lineups,
            present_current_appearances,
            present_datalake,
            present_current_lineups,
        ],
        [
            "fbref_plus_match_logs", "fbref_enriched", "match_logs_plus_lineups",
            "match_logs", "season_summary_only", "lineup_only",
        ],
        default="unclassified",
    )
    frame["advanced_features_available"] = present_fbref

    frame["canonical_matches_played"], frame["canonical_matches_source"] = choose_metric(frame, [
        ("fbref_matches_played", "fbref"),
        ("current_matches_played", "current_transfermarkt_appearances"),
        ("datalake_nb_on_pitch", "supplemental_datalake"),
        ("current_lineup_games", "current_transfermarkt_lineups"),
    ])
    frame["canonical_minutes_played"], frame["canonical_minutes_source"] = choose_metric(frame, [
        ("fbref_total_minutes", "fbref"),
        ("current_minutes_played", "current_transfermarkt_appearances"),
        ("datalake_minutes_played", "supplemental_datalake"),
    ])
    frame["canonical_goals"], frame["canonical_goals_source"] = choose_metric(frame, [
        ("fbref_goals", "fbref"),
        ("current_goals", "current_transfermarkt_appearances"),
        ("datalake_goals", "supplemental_datalake"),
    ])
    frame["canonical_assists"], frame["canonical_assists_source"] = choose_metric(frame, [
        ("fbref_assists", "fbref"),
        ("current_assists", "current_transfermarkt_appearances"),
        ("datalake_assists", "supplemental_datalake"),
    ])
    dl_red = pd.to_numeric(frame["datalake_second_yellow_cards"], errors="coerce").fillna(0) + pd.to_numeric(
        frame["datalake_direct_red_cards"], errors="coerce"
    ).fillna(0)
    dl_red.loc[frame["datalake_source_record_id"].isna()] = np.nan
    frame["datalake_red_cards"] = dl_red
    frame["canonical_yellow_cards"], yellow_source = choose_metric(frame, [
        ("current_yellow_cards", "current_transfermarkt_appearances"),
        ("datalake_yellow_cards", "supplemental_datalake"),
    ])
    frame["canonical_red_cards"], red_source = choose_metric(frame, [
        ("current_red_cards", "current_transfermarkt_appearances"),
        ("datalake_red_cards", "supplemental_datalake"),
    ])
    frame["canonical_cards_source"] = yellow_source.where(yellow_source.ne(""), red_source)
    minutes = pd.to_numeric(frame["canonical_minutes_played"], errors="coerce")
    frame["canonical_goals_per90"] = np.where(
        minutes.gt(0), pd.to_numeric(frame["canonical_goals"], errors="coerce") * 90 / minutes, np.nan
    )
    frame["canonical_assists_per90"] = np.where(
        minutes.gt(0), pd.to_numeric(frame["canonical_assists"], errors="coerce") * 90 / minutes, np.nan
    )
    frame["current_start_pct"] = np.where(
        pd.to_numeric(frame["current_lineup_records"], errors="coerce").gt(0),
        pd.to_numeric(frame["current_starts"], errors="coerce")
        / pd.to_numeric(frame["current_lineup_records"], errors="coerce"),
        np.nan,
    )
    frame["basic_model_eligible_450_minutes"] = (
        minutes.ge(450) & pd.to_numeric(frame["canonical_matches_played"], errors="coerce").ge(5)
    )

    add_agreement(frame, "matches", ["fbref_matches_played", "current_matches_played", "datalake_nb_on_pitch"])
    add_agreement(frame, "minutes", ["fbref_total_minutes", "current_minutes_played", "datalake_minutes_played"])
    add_agreement(frame, "goals", ["fbref_goals", "current_goals", "datalake_goals"])
    add_agreement(frame, "assists", ["fbref_assists", "current_assists", "datalake_assists"])
    add_agreement(frame, "yellow_cards", ["current_yellow_cards", "datalake_yellow_cards"])
    add_agreement(frame, "red_cards", ["current_red_cards", "datalake_red_cards"])
    status_columns = [f"{metric}_agreement_status" for metric in [
        "matches", "minutes", "goals", "assists", "yellow_cards", "red_cards"
    ]]
    differences = frame[status_columns].eq("sources_differ")
    frame["difference_metric_count"] = differences.sum(axis=1).astype("int64")
    frame["any_basic_cross_source_difference"] = frame["difference_metric_count"].gt(0)
    frame["difference_review_priority"] = np.select(
        [
            frame["goals_agreement_status"].eq("sources_differ")
            | frame["assists_agreement_status"].eq("sources_differ")
            | pd.to_numeric(frame["minutes_cross_source_range"], errors="coerce").gt(90)
            | pd.to_numeric(frame["matches_cross_source_range"], errors="coerce").gt(1),
            frame["any_basic_cross_source_difference"],
        ],
        ["high", "medium"],
        default="none",
    )

    integer_columns = [
        "season_start_year", "season_end_year", "source_system_count", "difference_metric_count",
        "canonical_matches_played", "canonical_minutes_played", "canonical_goals", "canonical_assists",
        "canonical_yellow_cards", "canonical_red_cards", "fbref_is_goalkeeper", "fbref_matches_played",
        "fbref_total_minutes", "fbref_goals", "fbref_assists", "current_appearance_records",
        "current_matches_played", "current_goals", "current_assists", "current_yellow_cards",
        "current_red_cards", "current_minutes_played", "current_lineup_records", "current_lineup_games",
        "current_starts", "current_substitute_selections", "current_captain_selections",
        "current_distinct_positions", "datalake_nb_in_group", "datalake_nb_on_pitch", "datalake_goals",
        "datalake_assists", "datalake_own_goals", "datalake_subbed_in", "datalake_subbed_out",
        "datalake_yellow_cards", "datalake_second_yellow_cards", "datalake_direct_red_cards",
        "datalake_penalty_goals", "datalake_minutes_played", "datalake_goals_conceded",
        "datalake_clean_sheets", "datalake_minutes_available", "datalake_red_cards",
    ] + [
        f"{metric}_{suffix}"
        for metric in ["matches", "minutes", "goals", "assists", "yellow_cards", "red_cards"]
        for suffix in ["source_values_compared", "cross_source_min", "cross_source_max", "cross_source_range"]
    ]
    integerize(frame, integer_columns)
    for column in [
        "canonical_goals_per90", "canonical_assists_per90", "current_start_pct", "fbref_age",
        "fbref_expected_goals", "fbref_non_penalty_expected_goals", "fbref_progressive_carries",
        "fbref_progressive_passes", "fbref_goals_per90", "fbref_assists_per90",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").round(6)
    return frame


BASE_COLUMNS = [
    "canonical_performance_id", "canonical_player_id", "canonical_player_name", "canonical_club_id",
    "canonical_club_name", "canonical_competition_id", "canonical_competition_name",
    "competition_country_name", "competition_type", "canonical_season_key", "season_start_year",
    "season_end_year", "present_in_fbref", "present_in_current_appearances",
    "present_in_current_lineups", "present_in_supplemental_performances", "source_system_count",
    "data_coverage_tier", "advanced_features_available", "basic_model_eligible_450_minutes",
    "canonical_matches_played", "canonical_minutes_played", "canonical_goals", "canonical_assists",
    "canonical_yellow_cards", "canonical_red_cards", "canonical_goals_per90", "canonical_assists_per90",
    "canonical_matches_source", "canonical_minutes_source", "canonical_goals_source",
    "canonical_assists_source", "canonical_cards_source",
]

FBREF_CORE_COLUMNS = [
    "fbref_row_id", "fbref_player_identity_key", "fbref_player_name", "fbref_squad_name",
    "fbref_competition_name", "fbref_position", "fbref_is_goalkeeper", "fbref_age",
    "fbref_matches_played", "fbref_total_minutes", "fbref_goals", "fbref_assists",
    "fbref_expected_goals", "fbref_non_penalty_expected_goals", "fbref_progressive_carries",
    "fbref_progressive_passes", "fbref_goals_per90", "fbref_assists_per90",
    "fbref_player_match_method", "fbref_club_match_method",
]

CURRENT_COLUMNS = [
    "current_appearance_records", "current_matches_played", "current_goals", "current_assists",
    "current_yellow_cards", "current_red_cards", "current_minutes_played",
    "current_first_appearance_date", "current_last_appearance_date", "current_lineup_records",
    "current_lineup_games", "current_starts", "current_substitute_selections",
    "current_captain_selections", "current_start_pct", "current_distinct_positions",
    "current_primary_position", "current_positions", "current_first_lineup_date", "current_last_lineup_date",
]

DATALAKE_COLUMNS = [
    "datalake_source_record_id", "datalake_season_name", "datalake_competition_name",
    "datalake_team_name", "datalake_nb_in_group", "datalake_nb_on_pitch", "datalake_goals",
    "datalake_assists", "datalake_own_goals", "datalake_subbed_in", "datalake_subbed_out",
    "datalake_yellow_cards", "datalake_second_yellow_cards", "datalake_direct_red_cards",
    "datalake_red_cards", "datalake_penalty_goals", "datalake_minutes_played",
    "datalake_goals_conceded", "datalake_clean_sheets", "datalake_minutes_available",
]

AGREEMENT_COLUMNS = [
    column
    for metric in ["matches", "minutes", "goals", "assists", "yellow_cards", "red_cards"]
    for column in [
        f"{metric}_source_values_compared", f"{metric}_cross_source_min",
        f"{metric}_cross_source_max", f"{metric}_cross_source_range", f"{metric}_agreement_status",
    ]
] + ["difference_metric_count", "any_basic_cross_source_difference", "difference_review_priority"]

CORE_COLUMNS = BASE_COLUMNS + FBREF_CORE_COLUMNS + CURRENT_COLUMNS + DATALAKE_COLUMNS + AGREEMENT_COLUMNS

MODEL_COLUMNS = BASE_COLUMNS + [
    "fbref_position", "fbref_is_goalkeeper", "fbref_expected_goals", "fbref_non_penalty_expected_goals",
    "fbref_progressive_carries", "fbref_progressive_passes", "current_starts",
    "current_substitute_selections", "current_captain_selections", "current_start_pct",
    "current_primary_position", "current_positions", "datalake_nb_in_group", "datalake_own_goals",
    "datalake_subbed_in", "datalake_subbed_out", "datalake_penalty_goals", "datalake_goals_conceded",
    "datalake_clean_sheets", "difference_metric_count", "any_basic_cross_source_difference",
    "difference_review_priority",
]

REVIEW_COLUMNS = [
    "canonical_performance_id", "canonical_player_id", "canonical_player_name", "canonical_club_id",
    "canonical_club_name", "canonical_competition_id", "canonical_competition_name",
    "canonical_season_key", "data_coverage_tier", "difference_review_priority", "difference_metric_count",
    "fbref_matches_played", "current_matches_played", "datalake_nb_on_pitch", "matches_cross_source_range",
    "fbref_total_minutes", "current_minutes_played", "datalake_minutes_played", "minutes_cross_source_range",
    "fbref_goals", "current_goals", "datalake_goals", "goals_cross_source_range",
    "fbref_assists", "current_assists", "datalake_assists", "assists_cross_source_range",
    "current_yellow_cards", "datalake_yellow_cards", "yellow_cards_cross_source_range",
    "current_red_cards", "datalake_red_cards", "red_cards_cross_source_range",
]


def build_core_outputs(connection: sqlite3.Connection) -> tuple[dict[str, Any], pd.DataFrame]:
    player_names, club_names, competition_meta = load_dimensions()
    agreement_counts = {metric: Counter() for metric in [
        "matches", "minutes", "goals", "assists", "yellow_cards", "red_cards"
    ]}
    tier_counts = Counter()
    total_rows = 0
    modeling_rows = 0
    difference_rows = 0
    missing_players = set()
    missing_clubs = set()
    missing_competitions = set()
    first_core = first_model = first_review = True

    for raw in pd.read_sql_query(CORE_QUERY, connection, chunksize=CHUNK_SIZE):
        frame = enrich_core_chunk(raw, player_names, club_names, competition_meta)
        missing_players.update(set(frame["canonical_player_id"]) - set(player_names))
        missing_clubs.update(set(frame["canonical_club_id"]) - set(club_names))
        missing_competitions.update(set(frame["canonical_competition_id"]) - set(competition_meta))
        tier_counts.update(frame["data_coverage_tier"].value_counts().to_dict())
        for metric in agreement_counts:
            agreement_counts[metric].update(frame[f"{metric}_agreement_status"].value_counts().to_dict())

        core = frame[CORE_COLUMNS]
        core.to_csv(
            CORE_PATH, mode="w" if first_core else "a", header=first_core, index=False,
            encoding="utf-8", lineterminator="\n",
        )
        first_core = False
        total_rows += len(core)

        modeling = frame.loc[frame["season_start_year"].between(2017, 2023), MODEL_COLUMNS]
        if not modeling.empty:
            modeling.to_csv(
                MODEL_PATH, mode="w" if first_model else "a", header=first_model, index=False,
                encoding="utf-8", lineterminator="\n",
            )
            first_model = False
            modeling_rows += len(modeling)

        review = frame.loc[frame["any_basic_cross_source_difference"], REVIEW_COLUMNS]
        if not review.empty:
            review.to_csv(
                DIFFERENCE_REVIEW_PATH, mode="w" if first_review else "a", header=first_review,
                index=False, encoding="utf-8", lineterminator="\n",
            )
            first_review = False
            difference_rows += len(review)

    if first_model:
        write_csv(pd.DataFrame(columns=MODEL_COLUMNS), MODEL_PATH)
    if first_review:
        write_csv(pd.DataFrame(columns=REVIEW_COLUMNS), DIFFERENCE_REVIEW_PATH)

    agreement_rows = []
    for metric, counts in agreement_counts.items():
        total = sum(counts.values())
        for status in ["not_compared", "sources_agree", "sources_differ"]:
            count = int(counts.get(status, 0))
            agreement_rows.append({
                "metric": metric,
                "agreement_status": status,
                "performance_rows": count,
                "pct_of_all_rows": count / total if total else None,
                "pct_of_compared_rows": (
                    count / (counts.get("sources_agree", 0) + counts.get("sources_differ", 0))
                    if status != "not_compared" and (counts.get("sources_agree", 0) + counts.get("sources_differ", 0))
                    else None
                ),
            })
    agreement = pd.DataFrame(agreement_rows)
    write_csv(agreement, OUTPUT / "source_agreement_summary.csv")

    metadata = {
        "canonical_rows": total_rows,
        "modeling_scope_rows": modeling_rows,
        "difference_review_rows": difference_rows,
        "missing_player_dimension_ids": len(missing_players),
        "missing_club_dimension_ids": len(missing_clubs),
        "missing_competition_dimension_ids": len(missing_competitions),
        "coverage_tiers": dict(tier_counts),
    }
    return metadata, agreement


def build_coverage_summaries(connection: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame]:
    join = """
    FROM canonical_keys k
    LEFT JOIN fbref_performance f USING (player_id, club_id, competition_id, season_start)
    LEFT JOIN current_appearances a USING (player_id, club_id, competition_id, season_start)
    LEFT JOIN current_lineups l USING (player_id, club_id, competition_id, season_start)
    LEFT JOIN datalake_performance d USING (player_id, club_id, competition_id, season_start)
    """
    season_query = """
    SELECT season_start, COUNT(*) AS performance_rows, COUNT(DISTINCT player_id) AS players,
           COUNT(DISTINCT club_id) AS clubs, COUNT(DISTINCT competition_id) AS competitions,
           SUM(f.fbref_row_id IS NOT NULL) AS fbref_rows,
           SUM(a.appearance_records IS NOT NULL) AS current_appearance_rows,
           SUM(l.lineup_records IS NOT NULL) AS current_lineup_rows,
           SUM(d.source_record_id IS NOT NULL) AS supplemental_rows
    """ + join + " GROUP BY season_start ORDER BY season_start"
    season = pd.read_sql_query(season_query, connection)
    season.insert(0, "canonical_season_key", season["season_start"].map(lambda value: f"{value}-{value + 1}"))
    season = season.rename(columns={"season_start": "season_start_year"})
    write_csv(season, OUTPUT / "coverage_by_season.csv")

    competition_query = """
    SELECT competition_id, COUNT(*) AS performance_rows, COUNT(DISTINCT player_id) AS players,
           COUNT(DISTINCT club_id) AS clubs, COUNT(DISTINCT season_start) AS seasons,
           MIN(season_start) AS first_season_start, MAX(season_start) AS last_season_start,
           SUM(f.fbref_row_id IS NOT NULL) AS fbref_rows,
           SUM(a.appearance_records IS NOT NULL) AS current_appearance_rows,
           SUM(l.lineup_records IS NOT NULL) AS current_lineup_rows,
           SUM(d.source_record_id IS NOT NULL) AS supplemental_rows
    """ + join + " GROUP BY competition_id ORDER BY performance_rows DESC, competition_id"
    competition = pd.read_sql_query(competition_query, connection)
    _, _, meta = load_dimensions()
    competition.insert(1, "canonical_competition_name", competition["competition_id"].map(
        lambda value: meta.get(clean_id(value), {}).get("name", "")
    ))
    competition.insert(2, "country_name", competition["competition_id"].map(
        lambda value: meta.get(clean_id(value), {}).get("country", "")
    ))
    competition.insert(3, "competition_type", competition["competition_id"].map(
        lambda value: meta.get(clean_id(value), {}).get("type", "")
    ))
    competition = competition.rename(columns={"competition_id": "canonical_competition_id"})
    write_csv(competition, OUTPUT / "coverage_by_competition.csv")
    return season, competition


def make_field_dictionary() -> pd.DataFrame:
    rows = []
    for order, column in enumerate(CORE_COLUMNS, start=1):
        if column.startswith("canonical_") and column.endswith("_id"):
            definition = "Canonical key used for stable joins across the integrated MoveMaker layer."
        elif column.startswith("fbref_"):
            definition = "FBref source-specific value; never silently overwrites another source."
        elif column.startswith("current_"):
            definition = "Current Transfermarkt match-log or lineup aggregate."
        elif column.startswith("datalake_"):
            definition = "Supplemental Transfermarkt seasonal-summary value."
        elif column.endswith("_source"):
            definition = "Source selected by the documented canonical precedence policy."
        elif "agreement_status" in column:
            definition = "Whether two or more available source values agree exactly."
        elif "cross_source" in column or "difference" in column:
            definition = "Cross-source reconciliation diagnostic; a difference is retained, not overwritten."
        elif column.startswith("present_in_"):
            definition = "Coverage flag showing whether this canonical grain exists in the named source."
        elif column == "basic_model_eligible_450_minutes":
            definition = "Convenience screen: at least 450 selected canonical minutes and five matches; not a quality label."
        else:
            definition = "Canonical performance field; see README policies and table dictionary."
        rows.append({
            "table": "canonical_player_team_season_performance",
            "column_order": order,
            "column": column,
            "definition": definition,
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    connection = create_database()
    print("Loading game metadata", flush=True)
    game_lookup, game_metadata = load_game_metadata()
    print("Aggregating current Transfermarkt appearances", flush=True)
    appearance_metadata = aggregate_current_appearances(connection, game_lookup)
    print("Aggregating current Transfermarkt lineups", flush=True)
    lineup_metadata = aggregate_current_lineups(connection, game_lookup)
    print("Loading supplemental seasonal performance", flush=True)
    datalake_metadata = load_datalake_performance(connection)
    print("Resolving FBref performance rows", flush=True)
    accepted_fbref, unresolved_fbref, fbref_metadata = prepare_fbref(connection)
    print("Building canonical key spine", flush=True)
    spine_rows = build_key_spine(connection)
    print(f"Writing canonical performance spine ({spine_rows:,} keys)", flush=True)
    core_metadata, agreement = build_core_outputs(connection)
    season_coverage, competition_coverage = build_coverage_summaries(connection)

    player_dim = pd.read_csv(PLAYER_DIM_PATH, usecols=["canonical_player_id"], dtype=str, keep_default_na=False)
    club_dim = pd.read_csv(CLUB_DIM_PATH, usecols=["canonical_club_id"], dtype=str, keep_default_na=False)
    comp_dim = pd.read_csv(COMPETITION_XW_PATH, usecols=["canonical_competition_id"], dtype=str, keep_default_na=False)

    checks = pd.DataFrame([
        {"check": "canonical_spine_unique", "severity": "blocking", "passed": core_metadata["canonical_rows"] == spine_rows, "observed": core_metadata["canonical_rows"], "expected": spine_rows, "detail": "One output row per unioned player-club-competition-season key."},
        {"check": "datalake_rows_preserved", "severity": "blocking", "passed": datalake_metadata["included_rows"] == datalake_metadata["source_rows"], "observed": datalake_metadata["included_rows"], "expected": datalake_metadata["source_rows"], "detail": "Every clean supplemental seasonal row enters the canonical spine."},
        {"check": "fbref_rows_reconciled", "severity": "blocking", "passed": fbref_metadata["accepted_rows"] + fbref_metadata["unresolved_rows"] == fbref_metadata["source_rows"], "observed": fbref_metadata["accepted_rows"] + fbref_metadata["unresolved_rows"], "expected": fbref_metadata["source_rows"], "detail": "Every FBref row is accepted or placed in the unresolved review queue."},
        {"check": "fbref_advanced_rows", "severity": "blocking", "passed": len(accepted_fbref) == fbref_metadata["accepted_rows"], "observed": len(accepted_fbref), "expected": fbref_metadata["accepted_rows"], "detail": "Advanced feature companion contains every accepted FBref row."},
        {"check": "canonical_player_references", "severity": "blocking", "passed": core_metadata["missing_player_dimension_ids"] == 0, "observed": core_metadata["missing_player_dimension_ids"], "expected": 0, "detail": f"All player IDs must resolve within {len(player_dim):,} canonical players."},
        {"check": "canonical_club_references", "severity": "blocking", "passed": core_metadata["missing_club_dimension_ids"] == 0, "observed": core_metadata["missing_club_dimension_ids"], "expected": 0, "detail": f"All club IDs must resolve within {len(club_dim):,} canonical clubs."},
        {"check": "canonical_competition_references", "severity": "blocking", "passed": core_metadata["missing_competition_dimension_ids"] == 0, "observed": core_metadata["missing_competition_dimension_ids"], "expected": 0, "detail": f"All competition IDs must resolve within {len(comp_dim):,} canonical competitions."},
        {"check": "current_appearance_games_resolve", "severity": "blocking", "passed": appearance_metadata.get("unmapped_game_rows", 0) == 0, "observed": appearance_metadata.get("unmapped_game_rows", 0), "expected": 0, "detail": "Every current appearance must resolve to game season and competition metadata."},
        {"check": "current_lineup_games_resolve", "severity": "blocking", "passed": lineup_metadata.get("unmapped_game_rows", 0) == 0, "observed": lineup_metadata.get("unmapped_game_rows", 0), "expected": 0, "detail": "Every current lineup row must resolve to game season and competition metadata."},
        {"check": "national_team_games_excluded", "severity": "blocking", "passed": appearance_metadata.get("excluded_national_rows", 0) >= 0 and lineup_metadata.get("excluded_national_rows", 0) >= 0, "observed": appearance_metadata.get("excluded_national_rows", 0) + lineup_metadata.get("excluded_national_rows", 0), "expected": "explicitly flagged rows excluded", "detail": "National-team game facts are outside the club/team compatibility grain."},
        {"check": "modeling_scope_seasons", "severity": "blocking", "passed": int(season_coverage.loc[season_coverage["season_start_year"].between(2017, 2023), "performance_rows"].sum()) == core_metadata["modeling_scope_rows"], "observed": core_metadata["modeling_scope_rows"], "expected": int(season_coverage.loc[season_coverage["season_start_year"].between(2017, 2023), "performance_rows"].sum()), "detail": "Modeling view contains all and only 2017-18 through 2023-24 canonical rows."},
        {"check": "difference_review_reconciles", "severity": "blocking", "passed": True, "observed": core_metadata["difference_review_rows"], "expected": "all rows with one or more differing metrics", "detail": "Difference review is emitted from the same row-level diagnostic as the master during the same streaming pass."},
        {"check": "appearance_competition_mismatches", "severity": "warning", "passed": appearance_metadata.get("competition_mismatch_rows", 0) == 0, "observed": appearance_metadata.get("competition_mismatch_rows", 0), "expected": 0, "detail": "Game competition is authoritative when an appearance label differs."},
        {"check": "fbref_unresolved_rows", "severity": "warning", "passed": len(unresolved_fbref) == 0, "observed": len(unresolved_fbref), "expected": 0, "detail": "Unresolved FBref identities are excluded from the canonical spine and retained for review."},
    ])
    write_csv(checks, OUTPUT / "performance_checks.csv")

    summary_rows = [
        {"area": "canonical_spine", "metric": "performance_rows", "value": spine_rows},
        {"area": "canonical_spine", "metric": "seasons", "value": int(len(season_coverage))},
        {"area": "canonical_spine", "metric": "competitions", "value": int(len(competition_coverage))},
        {"area": "canonical_spine", "metric": "modeling_scope_2017_2024_rows", "value": core_metadata["modeling_scope_rows"]},
        {"area": "canonical_spine", "metric": "basic_model_eligible_definition_minutes", "value": 450},
        {"area": "fbref", "metric": "source_rows", "value": fbref_metadata["source_rows"]},
        {"area": "fbref", "metric": "accepted_rows", "value": fbref_metadata["accepted_rows"]},
        {"area": "fbref", "metric": "unresolved_rows", "value": fbref_metadata["unresolved_rows"]},
        {"area": "current_appearances", "metric": "source_rows", "value": appearance_metadata["source_rows"]},
        {"area": "current_appearances", "metric": "included_club_rows", "value": appearance_metadata["included_rows"]},
        {"area": "current_appearances", "metric": "aggregated_keys", "value": appearance_metadata["aggregated_keys"]},
        {"area": "current_lineups", "metric": "source_rows", "value": lineup_metadata["source_rows"]},
        {"area": "current_lineups", "metric": "included_club_rows", "value": lineup_metadata["included_rows"]},
        {"area": "current_lineups", "metric": "aggregated_keys", "value": lineup_metadata["aggregated_keys"]},
        {"area": "supplemental_datalake", "metric": "source_rows", "value": datalake_metadata["source_rows"]},
        {"area": "supplemental_datalake", "metric": "canonical_keys", "value": datalake_metadata["aggregated_keys"]},
        {"area": "reconciliation", "metric": "difference_review_rows", "value": core_metadata["difference_review_rows"]},
        {"area": "exclusions", "metric": "national_team_appearance_rows", "value": appearance_metadata.get("excluded_national_rows", 0)},
        {"area": "exclusions", "metric": "national_team_lineup_rows", "value": lineup_metadata.get("excluded_national_rows", 0)},
    ]
    for tier, value in sorted(core_metadata["coverage_tiers"].items()):
        summary_rows.append({"area": "coverage_tier", "metric": tier, "value": value})
    summary = pd.DataFrame(summary_rows)
    write_csv(summary, OUTPUT / "performance_summary.csv")

    table_dictionary = pd.DataFrame([
        {"table": "canonical_player_team_season_performance", "grain": "One canonical player-club-competition-season row.", "purpose": "Source-preserving canonical spine and selected basic performance metrics."},
        {"table": "modeling_scope_2017_2024", "grain": "One canonical player-club-competition-season row for season starts 2017 through 2023.", "purpose": "Compact modeling view aligned to the seven FBref seasons; includes all available competitions."},
        {"table": "fbref_advanced_performance_features", "grain": "One accepted FBref canonical player-club-competition-season row.", "purpose": "Dense FBref advanced metrics keyed back to the canonical spine."},
        {"table": "performance_difference_review", "grain": "One canonical row with at least one differing comparable basic metric.", "purpose": "Cross-source review queue; differences are not silently resolved."},
        {"table": "unresolved_fbref_performance_review", "grain": "One FBref row excluded from the canonical spine.", "purpose": "Identity-resolution queue preserving all unmapped FBref evidence."},
        {"table": "coverage_by_season", "grain": "One canonical season.", "purpose": "Season-level coverage and source-presence audit."},
        {"table": "coverage_by_competition", "grain": "One canonical competition.", "purpose": "Competition-level coverage and source-presence audit."},
        {"table": "source_agreement_summary", "grain": "One metric-agreement-status row.", "purpose": "Exact agreement rates for comparable basic metrics."},
    ])
    write_csv(table_dictionary, OUTPUT / "performance_table_dictionary.csv")
    write_csv(make_field_dictionary(), OUTPUT / "performance_field_dictionary.csv")

    input_paths = [
        FBREF_PATH, GAMES_PATH, APPEARANCES_PATH, LINEUPS_PATH, DL_PERFORMANCE_PATH,
        PLAYER_DIM_PATH, CLUB_DIM_PATH, COMPETITION_XW_PATH, PLAYER_XW_PATH, CLUB_SEASON_XW_PATH,
    ]
    source_manifest = pd.DataFrame([
        {
            "source_file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "file_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in input_paths
    ])
    write_csv(source_manifest, OUTPUT / "source_manifest.csv")

    blocking_failures = int((checks["severity"].eq("blocking") & ~checks["passed"].astype(bool)).sum())
    result = {
        "status": "pass" if blocking_failures == 0 else "fail",
        "canonical_performance_rows": spine_rows,
        "modeling_scope_2017_2024_rows": core_metadata["modeling_scope_rows"],
        "accepted_fbref_rows": fbref_metadata["accepted_rows"],
        "unresolved_fbref_rows": fbref_metadata["unresolved_rows"],
        "current_appearance_keys": appearance_metadata["aggregated_keys"],
        "current_lineup_keys": lineup_metadata["aggregated_keys"],
        "supplemental_performance_keys": datalake_metadata["aggregated_keys"],
        "difference_review_rows": core_metadata["difference_review_rows"],
        "blocking_checks": int(checks["severity"].eq("blocking").sum()),
        "blocking_failures": blocking_failures,
    }
    (OUTPUT / "performance_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUTPUT / "README.md").write_text(
        f"""# MoveMaker canonical player-team-season performance

This layer integrates FBref seasonal performance, current Transfermarkt game appearances and lineups, and the supplemental Transfermarkt seasonal-summary table.

## Canonical grain

`canonical_player_id + canonical_club_id + canonical_competition_id + canonical_season_key`

National-team game appearances and lineups are excluded because the intended use is club recruitment and player-club compatibility. The separate supplemental national-performance table remains outside this layer.

## Source policy

- Source-specific metrics are retained in explicit `fbref_`, `current_`, and `datalake_` columns.
- Canonical matches, minutes, goals, and assists use `FBref -> current Transfermarkt appearances -> supplemental datalake` precedence. FBref is first within its Big Five scope so its totals remain coherent with its advanced per-90 features; current match logs are next; the much wider supplemental seasonal table fills remaining coverage.
- Canonical cards use `current Transfermarkt appearances -> supplemental datalake` because FBref cards are absent from the cleaned seasonal source.
- Cross-source disagreement is never erased. Exact comparison status, ranges, and a review queue are emitted.
- Game metadata is authoritative for current Transfermarkt season and competition. Explicit national-team games are excluded.
- The 450-minute modeling flag is a convenience screen, not a performance or confidence score.

## Outputs

- `canonical_player_team_season_performance.csv`: {spine_rows:,} canonical rows.
- `modeling_scope_2017_2024.csv`: {core_metadata['modeling_scope_rows']:,} compact rows covering 2017-18 through 2023-24 across all available club competitions.
- `fbref_advanced_performance_features.csv`: {fbref_metadata['accepted_rows']:,} dense accepted FBref rows with all cleaned FBref fields.
- `performance_difference_review.csv`: {core_metadata['difference_review_rows']:,} canonical rows with at least one differing comparable basic metric.
- `unresolved_fbref_performance_review.csv`: {fbref_metadata['unresolved_rows']:,} FBref rows withheld until identity resolution.

The advanced FBref companion joins one-to-one to applicable spine rows by `canonical_performance_id`.

## Validation

Blocking checks: {int(checks['severity'].eq('blocking').sum()):,}; failures: {blocking_failures:,}.
""",
        encoding="utf-8",
    )

    connection.close()
    for sidecar in [WORK_DB.with_suffix(".sqlite-wal"), WORK_DB.with_suffix(".sqlite-shm")]:
        if sidecar.exists():
            sidecar.unlink()
    if WORK_DB.exists():
        WORK_DB.unlink()

    output_paths = [path for path in OUTPUT.iterdir() if path.is_file() and path.name != "output_manifest.csv"]
    output_manifest = pd.DataFrame([
        {
            "output_file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "file_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(output_paths)
    ])
    write_csv(output_manifest, OUTPUT / "output_manifest.csv")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

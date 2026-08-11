"""Audit the supplemental Transfermarkt datalake before cleaning or merging.

The source CSVs under ``football-datasets-main/datalake`` are immutable. This
script profiles their schemas and natural keys, checks entity coverage against
the existing MoveMaker Transfermarkt layer, reconciles overlapping valuations,
transfers, profiles, and player-season performance, and writes reviewable audit
artifacts without changing any source or existing processed file.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "football-datasets-main" / "datalake" / "transfermarkt"
CURRENT = ROOT / "Data" / "TransferMarkt"
CURRENT_CLEAN = ROOT / "Data" / "processed" / "transfermarkt_clean"
OUTPUT = ROOT / "Data" / "processed" / "football_datalake_audit"
CHUNK_SIZE = 125_000
AUDIT_DATE = date(2026, 8, 7)


TABLES: dict[str, dict[str, Any]] = {
    "player_injuries": {
        "path": "player_injuries/player_injuries.csv",
        "grain": "One reported player injury spell.",
        "key": ["player_id", "season_name", "injury_reason", "from_date", "end_date"],
        "advertised_rows": 143_195,
        "dates": ["from_date", "end_date"],
        "numeric": ["player_id", "days_missed", "games_missed"],
        "player_fields": ["player_id"],
        "disposition": "keep_separate_fact",
    },
    "player_latest_market_value": {
        "path": "player_latest_market_value/player_latest_market_value.csv",
        "grain": "One latest valuation snapshot per player.",
        "key": ["player_id"],
        "advertised_rows": None,
        "dates": ["date_unix"],
        "numeric": ["player_id", "value"],
        "player_fields": ["player_id"],
        "disposition": "retain_raw_derive_from_history",
    },
    "player_market_value": {
        "path": "player_market_value/player_market_value.csv",
        "grain": "One player valuation observation date.",
        "key": ["player_id", "date_unix"],
        "advertised_rows": 901_457,
        "dates": ["date_unix"],
        "numeric": ["player_id", "value"],
        "player_fields": ["player_id"],
        "disposition": "union_canonical_valuation_history",
    },
    "player_national_performances": {
        "path": "player_national_performances/player_national_performances.csv",
        "grain": "One player-national-team career summary.",
        "key": ["player_id", "team_id"],
        "advertised_rows": 92_701,
        "dates": ["debut"],
        "numeric": ["player_id", "team_id", "matches", "goals", "shirt_number", "coach_id", "debut_game_id"],
        "player_fields": ["player_id"],
        "club_fields": ["team_id"],
        "disposition": "keep_separate_fact",
    },
    "player_performances": {
        "path": "player_performances/player_performances.csv",
        "grain": "One player-team-competition-season performance summary.",
        "key": ["player_id", "season_name", "competition_id", "team_id"],
        "advertised_rows": 1_878_719,
        "numeric": ["player_id", "team_id", "nb_in_group", "nb_on_pitch", "goals", "assists", "own_goals", "subed_in", "subed_out", "yellow_cards", "second_yellow_cards", "direct_red_cards", "penalty_goals", "minutes_played", "goals_conceded", "clean_sheets"],
        "player_fields": ["player_id"],
        "club_fields": ["team_id"],
        "competition_fields": ["competition_id"],
        "disposition": "keep_separate_fact_reconcile_to_appearances",
    },
    "player_profiles": {
        "path": "player_profiles/player_profiles.csv",
        "grain": "One current/latest player profile snapshot.",
        "key": ["player_id"],
        "advertised_rows": 92_671,
        "dates": ["date_of_birth", "joined", "contract_expires", "date_of_last_contract_extension", "contract_there_expires", "date_of_death"],
        "numeric": ["player_id", "height", "current_club_id", "player_agent_id", "on_loan_from_club_id"],
        "player_fields": ["player_id"],
        "club_fields": ["current_club_id", "on_loan_from_club_id"],
        "disposition": "merge_as_dimension_evidence",
    },
    "player_teammates_played_with": {
        "path": "player_teammates_played_with/player_teammates_played_with.csv",
        "grain": "One directed player-to-teammate relationship.",
        "key": ["player_id", "teammate_player_id"],
        "advertised_rows": 1_257_342,
        "numeric": ["player_id", "teammate_player_id", "ppg_played_with", "joint_goal_participation", "minutes_played_with"],
        "player_fields": ["player_id", "teammate_player_id"],
        "disposition": "keep_separate_edge_table",
    },
    "team_children": {
        "path": "team_children/team_children.csv",
        "grain": "One parent-club to child-club relationship.",
        "key": ["parent_team_id", "child_team_id"],
        "advertised_rows": 7_695,
        "dates": ["_last_modified_at"],
        "numeric": ["parent_team_id", "child_team_id"],
        "club_fields": ["parent_team_id", "child_team_id"],
        "disposition": "keep_separate_bridge",
    },
    "team_competitions_seasons": {
        "path": "team_competitions_seasons/team_competitions_seasons.csv",
        "grain": "One club-competition-season standings summary.",
        "key": ["club_id", "competition_id", "season_id"],
        "advertised_rows": 196_378,
        "numeric": ["club_id", "season_draws", "season_goal_difference", "season_goals_against", "season_goals_for", "season_id", "season_league_level_level_number", "season_league_season_id", "season_losses", "season_manager", "season_manager_manager_id", "season_points", "season_points_against", "season_points_for", "season_rank", "season_total_matches", "season_wins"],
        "club_fields": ["club_id"],
        "competition_fields": ["competition_id", "season_league_competition_id"],
        "disposition": "keep_separate_fact",
    },
    "team_details": {
        "path": "team_details/team_details.csv",
        "grain": "One current/latest club profile snapshot.",
        "key": ["club_id"],
        "advertised_rows": 2_175,
        "dates": ["_last_modified_at"],
        "numeric": ["club_id", "season_id"],
        "club_fields": ["club_id"],
        "competition_fields": ["competition_id"],
        "disposition": "merge_as_dimension_evidence",
    },
    "transfer_history": {
        "path": "transfer_history/transfer_history.csv",
        "grain": "One recorded player movement between clubs on a date.",
        "key": ["player_id", "transfer_date", "from_team_id", "to_team_id"],
        "advertised_rows": 1_101_440,
        "dates": ["transfer_date"],
        "numeric": ["player_id", "from_team_id", "to_team_id", "value_at_transfer", "transfer_fee"],
        "player_fields": ["player_id"],
        "club_fields": ["from_team_id", "to_team_id"],
        "disposition": "union_canonical_transfer_history",
    },
}


def source_path(table: str) -> Path:
    return SOURCE / TABLES[table]["path"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_id(value: Any) -> str:
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none"}:
        return ""
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def date_value(value: Any) -> str:
    text = str(value).strip()
    return text[:10] if text else ""


def text_value(value: Any) -> str:
    return str(value).strip().casefold()


def number_value(value: Any) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def number_equal(left: Any, right: Any) -> bool | None:
    a, b = number_value(left), number_value(right)
    if a is None or b is None:
        return None
    return math.isclose(a, b, rel_tol=0, abs_tol=1e-9)


def read_string_chunks(path: Path, usecols: list[str] | None = None):
    yield from pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        chunksize=CHUNK_SIZE,
        usecols=usecols,
        low_memory=False,
    )


def read_id_set(path: Path, column: str) -> set[str]:
    result: set[str] = set()
    for chunk in read_string_chunks(path, [column]):
        result.update(canonical_id(value) for value in chunk[column] if str(value).strip())
    result.discard("")
    return result


def csv_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        yield from csv.DictReader(handle)


def write_csv(name: str, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path = OUTPUT / name
    if columns is None:
        columns = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_reference_sets() -> dict[str, set[str]]:
    profiles = read_id_set(source_path("player_profiles"), "player_id")
    teams = read_id_set(source_path("team_details"), "club_id")
    new_competitions: set[str] = set()
    for table, field in [
        ("team_details", "competition_id"),
        ("team_competitions_seasons", "competition_id"),
        ("team_competitions_seasons", "season_league_competition_id"),
        ("player_performances", "competition_id"),
    ]:
        new_competitions |= read_id_set(source_path(table), field)
    current_players = read_id_set(CURRENT / "players.csv", "player_id")
    identity_union = read_id_set(CURRENT_CLEAN / "player_identity_union.csv", "player_id")
    current_clubs = read_id_set(CURRENT / "clubs.csv", "club_id")
    club_union = read_id_set(CURRENT_CLEAN / "club_dimension_historical.csv", "club_id")
    current_competitions = read_id_set(CURRENT / "competitions.csv", "competition_id")
    fbref_mapped: set[str] = set()
    for row in csv_rows(ROOT / "Data" / "processed" / "player_crosswalk.csv"):
        if row.get("match_status") == "unique" and row.get("player_id"):
            fbref_mapped.add(canonical_id(row["player_id"]))
    return {
        "new_player_profiles": profiles,
        "new_team_details": teams,
        "new_competitions": new_competitions,
        "current_player_snapshot": current_players,
        "current_player_identity_union": identity_union,
        "current_club_snapshot": current_clubs,
        "current_club_union": club_union,
        "current_competitions": current_competitions,
        "fbref_unique_mapped_players": fbref_mapped,
    }


def profile_tables(references: dict[str, set[str]]):
    manifests: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    columns_out: list[dict[str, Any]] = []
    keys_out: list[dict[str, Any]] = []
    foreign_out: list[dict[str, Any]] = []
    id_sets: dict[tuple[str, str], set[str]] = {}

    for table, config in TABLES.items():
        path = source_path(table)
        first_line = path.open("r", encoding="utf-8-sig", errors="replace").readline().strip()
        pointer = first_line == "version https://git-lfs.github.com/spec/v1"
        header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
        row_count = 0
        key_null_rows = 0
        key_duplicate_rows = 0
        seen_key_hashes: set[int] = set()
        missing = Counter()
        invalid_numeric = Counter()
        invalid_date = Counter()
        zero_count = Counter()
        mojibake = Counter()
        minima: dict[str, Any] = {}
        maxima: dict[str, Any] = {}
        samples: dict[str, list[str]] = defaultdict(list)
        fk_stats: dict[tuple[str, str, str], dict[str, Any]] = {}

        fk_specs: list[tuple[str, str, str]] = []
        for field in config.get("player_fields", []):
            fk_specs += [
                (field, "new_player_profiles", "player"),
                (field, "current_player_identity_union", "player"),
            ]
        for field in config.get("club_fields", []):
            fk_specs += [
                (field, "new_team_details", "club"),
                (field, "current_club_union", "club"),
            ]
        for field in config.get("competition_fields", []):
            fk_specs += [(field, "current_competitions", "competition")]
        for field, reference, entity in fk_specs:
            fk_stats[(field, reference, entity)] = {
                "nonblank_rows": 0,
                "matched_rows": 0,
                "unique_values": set(),
                "matched_unique_values": set(),
            }

        for chunk in read_string_chunks(path):
            row_count += len(chunk)
            blank_frame = chunk.apply(lambda series: series.astype(str).str.strip().eq(""))
            for column in header:
                series = chunk[column].astype(str).str.strip()
                missing[column] += int(blank_frame[column].sum())
                zero_count[column] += int(series.eq("0").sum() + series.eq("0.0").sum())
                mojibake[column] += int(series.str.contains(r"Ã|Â|â|�", regex=True, na=False).sum())
                for value in series[~series.eq("")].head(8):
                    if value not in samples[column] and len(samples[column]) < 5:
                        samples[column].append(value[:120])

            for column in config.get("numeric", []):
                if column not in chunk:
                    continue
                series = chunk[column].astype(str).str.strip()
                parsed = pd.to_numeric(series.where(~series.eq("")), errors="coerce")
                invalid_numeric[column] += int((~series.eq("") & parsed.isna()).sum())
                if parsed.notna().any():
                    low, high = float(parsed.min()), float(parsed.max())
                    minima[column] = low if column not in minima else min(minima[column], low)
                    maxima[column] = high if column not in maxima else max(maxima[column], high)

            for column in config.get("dates", []):
                if column not in chunk:
                    continue
                series = chunk[column].astype(str).str.strip()
                parsed = pd.to_datetime(series.where(~series.eq("")), errors="coerce")
                invalid_date[column] += int((~series.eq("") & parsed.isna()).sum())
                if parsed.notna().any():
                    low = parsed.min().isoformat()
                    high = parsed.max().isoformat()
                    minima[column] = low if column not in minima else min(minima[column], low)
                    maxima[column] = high if column not in maxima else max(maxima[column], high)

            key_columns = config["key"]
            key_blank = blank_frame[key_columns].any(axis=1)
            key_null_rows += int(key_blank.sum())
            valid_keys = chunk.loc[~key_blank, key_columns].astype(str).apply(lambda s: s.str.strip())
            hashes = pd.util.hash_pandas_object(valid_keys, index=False).to_numpy(dtype=np.uint64)
            unique_hashes, counts = np.unique(hashes, return_counts=True)
            key_duplicate_rows += int(np.maximum(counts - 1, 0).sum())
            key_duplicate_rows += sum(1 for value in unique_hashes if int(value) in seen_key_hashes)
            seen_key_hashes.update(int(value) for value in unique_hashes)

            for field, reference, entity in fk_specs:
                if field not in chunk:
                    continue
                values = chunk[field].map(canonical_id)
                nonblank = ~values.eq("")
                matched = nonblank & values.isin(references[reference])
                stats = fk_stats[(field, reference, entity)]
                stats["nonblank_rows"] += int(nonblank.sum())
                stats["matched_rows"] += int(matched.sum())
                stats["unique_values"].update(values[nonblank].unique().tolist())
                stats["matched_unique_values"].update(values[matched].unique().tolist())
                id_sets.setdefault((table, field), set()).update(values[nonblank].unique().tolist())

        advertised = config.get("advertised_rows")
        advertised_variance = None if advertised is None else row_count - advertised
        manifest = {
            "table": table,
            "source_file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "file_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "is_git_lfs_pointer": pointer,
            "data_rows": row_count,
            "columns": len(header),
        }
        manifests.append(manifest)
        summaries.append({
            **manifest,
            "grain": config["grain"],
            "natural_key": " + ".join(config["key"]),
            "key_null_rows": key_null_rows,
            "duplicate_key_rows": key_duplicate_rows,
            "advertised_rows": advertised,
            "advertised_row_variance": advertised_variance,
            "disposition": config["disposition"],
            "mojibake_cells": sum(mojibake.values()),
            "invalid_numeric_values": sum(invalid_numeric.values()),
            "invalid_date_values": sum(invalid_date.values()),
        })
        keys_out.append({
            "table": table,
            "natural_key": " + ".join(config["key"]),
            "rows": row_count,
            "key_null_rows": key_null_rows,
            "unique_key_count": len(seen_key_hashes),
            "duplicate_key_rows": key_duplicate_rows,
            "status": "pass" if key_null_rows == 0 and key_duplicate_rows == 0 else "review",
        })
        for column in header:
            expected_type = "date" if column in config.get("dates", []) else "numeric" if column in config.get("numeric", []) else "text"
            columns_out.append({
                "table": table,
                "column": column,
                "expected_type": expected_type,
                "rows": row_count,
                "nonblank_count": row_count - missing[column],
                "missing_count": missing[column],
                "missing_pct": missing[column] / row_count if row_count else None,
                "zero_literal_count": zero_count[column],
                "invalid_typed_values": invalid_date[column] if expected_type == "date" else invalid_numeric[column] if expected_type == "numeric" else 0,
                "min_value": minima.get(column),
                "max_value": maxima.get(column),
                "mojibake_cells": mojibake[column],
                "sample_values": " | ".join(samples[column]),
            })
        for (field, reference, entity), stats in fk_stats.items():
            nonblank_rows = stats["nonblank_rows"]
            unique_values = len(stats["unique_values"])
            foreign_out.append({
                "table": table,
                "field": field,
                "entity_type": entity,
                "reference_set": reference,
                "nonblank_rows": nonblank_rows,
                "matched_rows": stats["matched_rows"],
                "row_match_pct": stats["matched_rows"] / nonblank_rows if nonblank_rows else None,
                "unique_values": unique_values,
                "matched_unique_values": len(stats["matched_unique_values"]),
                "unique_match_pct": len(stats["matched_unique_values"]) / unique_values if unique_values else None,
            })
    return manifests, summaries, columns_out, keys_out, foreign_out, id_sets


def profile_overlap() -> list[dict[str, Any]]:
    current_players = {canonical_id(row["player_id"]): row for row in csv_rows(CURRENT / "players.csv")}
    mappings: dict[str, tuple[str, str, Callable[[Any], Any]]] = {
        "date_of_birth": ("date_of_birth", "date_of_birth", date_value),
        "country_of_birth": ("country_of_birth", "country_of_birth", text_value),
        "citizenship": ("citizenship", "country_of_citizenship", text_value),
        "foot": ("foot", "foot", text_value),
        "height": ("height", "height_in_cm", number_value),
        "current_club_id": ("current_club_id", "current_club_id", canonical_id),
        "contract_expires": ("contract_expires", "contract_expiration_date", date_value),
        "agent_name": ("player_agent_name", "agent_name", text_value),
    }
    stats = {field: Counter() for field in mappings}
    for row in csv_rows(source_path("player_profiles")):
        current = current_players.get(canonical_id(row["player_id"]))
        if current is None:
            continue
        for field, (new_column, current_column, normalizer) in mappings.items():
            left, right = row.get(new_column, ""), current.get(current_column, "")
            if not str(left).strip() or not str(right).strip():
                stats[field]["one_or_both_missing"] += 1
                continue
            stats[field]["both_nonblank"] += 1
            if normalizer(left) == normalizer(right):
                stats[field]["matches"] += 1
            else:
                stats[field]["differs"] += 1
    output = []
    for field, counts in stats.items():
        both = counts["both_nonblank"]
        output.append({
            "field": field,
            "both_nonblank": both,
            "matches": counts["matches"],
            "differs": counts["differs"],
            "match_pct": counts["matches"] / both if both else None,
            "one_or_both_missing": counts["one_or_both_missing"],
            "interpretation": "snapshot_sensitive" if field in {"current_club_id", "contract_expires", "agent_name"} else "stable_identity_attribute",
        })
    return output


def valuation_overlap() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    current_values: dict[tuple[str, str], str] = {}
    current_players: set[str] = set()
    for row in csv_rows(CURRENT / "player_valuations.csv"):
        key = (canonical_id(row["player_id"]), date_value(row["date"]))
        current_values[key] = row.get("market_value_in_eur", "")
        current_players.add(key[0])
    conflicts: list[dict[str, Any]] = []
    new_players: set[str] = set()
    latest_from_history: dict[str, tuple[str, str]] = {}
    shared_keys = matching_values = differing_values = 0
    for row in csv_rows(source_path("player_market_value")):
        player_id = canonical_id(row["player_id"])
        valuation_date = date_value(row["date_unix"])
        new_value = row.get("value", "")
        new_players.add(player_id)
        previous = latest_from_history.get(player_id)
        if previous is None or valuation_date > previous[0]:
            latest_from_history[player_id] = (valuation_date, new_value)
        key = (player_id, valuation_date)
        if key not in current_values:
            continue
        shared_keys += 1
        equal = number_equal(new_value, current_values[key])
        if equal:
            matching_values += 1
        else:
            differing_values += 1
            conflicts.append({
                "player_id": player_id,
                "valuation_date": valuation_date,
                "new_value": new_value,
                "current_value": current_values[key],
                "difference": (number_value(new_value) or 0) - (number_value(current_values[key]) or 0),
            })
    latest_rows = 0
    latest_exact = 0
    latest_date_mismatch = 0
    latest_value_mismatch = 0
    for row in csv_rows(source_path("player_latest_market_value")):
        latest_rows += 1
        player_id = canonical_id(row["player_id"])
        derived = latest_from_history.get(player_id)
        if derived is None:
            continue
        date_match = date_value(row["date_unix"]) == derived[0]
        value_match = bool(number_equal(row.get("value"), derived[1]))
        latest_exact += int(date_match and value_match)
        latest_date_mismatch += int(not date_match)
        latest_value_mismatch += int(not value_match)
    summary = {
        "current_valuation_players": len(current_players),
        "new_valuation_players": len(new_players),
        "shared_valuation_players": len(current_players & new_players),
        "current_valuation_rows": len(current_values),
        "new_valuation_rows": sum(1 for _ in csv_rows(source_path("player_market_value"))),
        "shared_player_date_keys": shared_keys,
        "matching_player_date_values": matching_values,
        "differing_player_date_values": differing_values,
        "latest_rows": latest_rows,
        "latest_exact_vs_history": latest_exact,
        "latest_date_mismatch": latest_date_mismatch,
        "latest_value_mismatch": latest_value_mismatch,
    }
    overlap = [
        {"area": "valuations", "metric": key, "value": value, "interpretation": ""}
        for key, value in summary.items()
    ]
    return overlap, conflicts, summary


def transfer_overlap() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    current_transfers: dict[tuple[str, str, str, str], dict[str, str]] = {}
    current_players: set[str] = set()
    for row in csv_rows(CURRENT / "transfers.csv"):
        key = (
            canonical_id(row["player_id"]),
            date_value(row["transfer_date"]),
            canonical_id(row["from_club_id"]),
            canonical_id(row["to_club_id"]),
        )
        current_transfers[key] = row
        current_players.add(key[0])
    conflicts: list[dict[str, Any]] = []
    new_players: set[str] = set()
    rows = shared_keys = fee_matches = value_matches = fee_diffs = value_diffs = 0
    for row in csv_rows(source_path("transfer_history")):
        rows += 1
        key = (
            canonical_id(row["player_id"]),
            date_value(row["transfer_date"]),
            canonical_id(row["from_team_id"]),
            canonical_id(row["to_team_id"]),
        )
        new_players.add(key[0])
        current = current_transfers.get(key)
        if current is None:
            continue
        shared_keys += 1
        fee_equal = number_equal(row.get("transfer_fee"), current.get("transfer_fee"))
        value_equal = number_equal(row.get("value_at_transfer"), current.get("market_value_in_eur"))
        fee_matches += int(fee_equal is True)
        fee_diffs += int(fee_equal is False)
        value_matches += int(value_equal is True)
        value_diffs += int(value_equal is False)
        if fee_equal is not True or value_equal is not True:
            conflicts.append({
                "player_id": key[0],
                "transfer_date": key[1],
                "from_club_id": key[2],
                "to_club_id": key[3],
                "new_from_name": row.get("from_team_name"),
                "new_to_name": row.get("to_team_name"),
                "new_transfer_type": row.get("transfer_type"),
                "new_transfer_fee": row.get("transfer_fee"),
                "current_transfer_fee": current.get("transfer_fee"),
                "fee_equal": fee_equal,
                "new_value_at_transfer": row.get("value_at_transfer"),
                "current_market_value": current.get("market_value_in_eur"),
                "value_equal": value_equal,
            })
    summary = {
        "current_transfer_rows": len(current_transfers),
        "new_transfer_rows": rows,
        "current_transfer_players": len(current_players),
        "new_transfer_players": len(new_players),
        "shared_transfer_players": len(current_players & new_players),
        "shared_transfer_event_keys": shared_keys,
        "matching_fees_on_shared_keys": fee_matches,
        "differing_fees_on_shared_keys": fee_diffs,
        "matching_values_on_shared_keys": value_matches,
        "differing_values_on_shared_keys": value_diffs,
    }
    overlap = [
        {"area": "transfers", "metric": key, "value": value, "interpretation": ""}
        for key, value in summary.items()
    ]
    return overlap, conflicts, summary


def tm_season_label(value: Any) -> str:
    try:
        start = int(float(str(value)))
    except ValueError:
        return str(value).strip()
    return f"{start % 100:02d}/{(start + 1) % 100:02d}"


def performance_reconciliation() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    game_seasons: dict[str, str] = {}
    for row in csv_rows(CURRENT / "games.csv"):
        game_seasons[canonical_id(row["game_id"])] = tm_season_label(row.get("season"))
    current_agg: dict[tuple[str, str, str, str], list[float]] = {}
    metrics = ["nb_on_pitch", "goals", "assists", "yellow_cards", "red_cards", "minutes_played"]
    for chunk in read_string_chunks(CURRENT / "appearances.csv"):
        chunk["season_name"] = chunk["game_id"].map(lambda value: game_seasons.get(canonical_id(value), ""))
        chunk["player_id"] = chunk["player_id"].map(canonical_id)
        chunk["player_club_id"] = chunk["player_club_id"].map(canonical_id)
        for column in ["goals", "assists", "yellow_cards", "red_cards", "minutes_played"]:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce").fillna(0)
        grouped = chunk.groupby(["player_id", "season_name", "competition_id", "player_club_id"], dropna=False).agg(
            nb_on_pitch=("appearance_id", "count"),
            goals=("goals", "sum"),
            assists=("assists", "sum"),
            yellow_cards=("yellow_cards", "sum"),
            red_cards=("red_cards", "sum"),
            minutes_played=("minutes_played", "sum"),
        ).reset_index()
        for row in grouped.itertuples(index=False):
            key = (str(row.player_id), str(row.season_name), str(row.competition_id), str(row.player_club_id))
            values = [float(getattr(row, metric)) for metric in metrics]
            if key not in current_agg:
                current_agg[key] = values
            else:
                current_agg[key] = [a + b for a, b in zip(current_agg[key], values)]

    shared_keys = 0
    new_keys = 0
    metric_stats = {metric: Counter() for metric in metrics}
    samples: list[dict[str, Any]] = []
    new_player_ids: set[str] = set()
    for row in csv_rows(source_path("player_performances")):
        new_keys += 1
        player_id = canonical_id(row["player_id"])
        new_player_ids.add(player_id)
        key = (player_id, row.get("season_name", ""), row.get("competition_id", ""), canonical_id(row.get("team_id")))
        current = current_agg.get(key)
        if current is None:
            continue
        shared_keys += 1
        new_values = {
            "nb_on_pitch": number_value(row.get("nb_on_pitch")),
            "goals": number_value(row.get("goals")),
            "assists": number_value(row.get("assists")),
            "yellow_cards": number_value(row.get("yellow_cards")),
            "red_cards": (number_value(row.get("direct_red_cards")) or 0) + (number_value(row.get("second_yellow_cards")) or 0),
            "minutes_played": number_value(row.get("minutes_played")),
        }
        row_differs = False
        for index, metric in enumerate(metrics):
            new_value = new_values[metric]
            if new_value is None:
                metric_stats[metric]["new_missing"] += 1
                continue
            metric_stats[metric]["compared"] += 1
            if math.isclose(new_value, current[index], rel_tol=0, abs_tol=1e-9):
                metric_stats[metric]["exact"] += 1
            else:
                metric_stats[metric]["differs"] += 1
                metric_stats[metric]["absolute_difference_sum"] += abs(new_value - current[index])
                row_differs = True
        if row_differs and len(samples) < 1_000:
            samples.append({
                "player_id": key[0], "season_name": key[1], "competition_id": key[2], "team_id": key[3],
                **{f"new_{metric}": new_values[metric] for metric in metrics},
                **{f"current_{metric}": current[index] for index, metric in enumerate(metrics)},
            })
    output: list[dict[str, Any]] = []
    for metric, counts in metric_stats.items():
        compared = counts["compared"]
        output.append({
            "metric": metric,
            "shared_keys": shared_keys,
            "compared_values": compared,
            "exact_matches": counts["exact"],
            "different_values": counts["differs"],
            "exact_match_pct": counts["exact"] / compared if compared else None,
            "mean_abs_difference_when_different": counts["absolute_difference_sum"] / counts["differs"] if counts["differs"] else 0,
            "new_missing_on_shared_keys": counts["new_missing"],
        })
    summary = {
        "current_aggregated_performance_keys": len(current_agg),
        "new_performance_rows": new_keys,
        "shared_performance_keys": shared_keys,
        "new_performance_players": len(new_player_ids),
    }
    return output, samples, summary


def build_overlap_rows(
    references: dict[str, set[str]],
    id_sets: dict[tuple[str, str], set[str]],
    valuation_rows: list[dict[str, Any]],
    transfer_rows: list[dict[str, Any]],
    performance_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = valuation_rows + transfer_rows
    entity_specs = [
        ("player_profiles", "player_id", "current_player_snapshot"),
        ("player_profiles", "player_id", "current_player_identity_union"),
        ("player_profiles", "player_id", "fbref_unique_mapped_players"),
        ("player_injuries", "player_id", "fbref_unique_mapped_players"),
        ("player_market_value", "player_id", "fbref_unique_mapped_players"),
        ("player_performances", "player_id", "fbref_unique_mapped_players"),
        ("player_teammates_played_with", "player_id", "fbref_unique_mapped_players"),
        ("team_details", "club_id", "current_club_union"),
        ("team_competitions_seasons", "club_id", "current_club_union"),
    ]
    for table, field, reference in entity_specs:
        source = id_sets.get((table, field), set())
        target = references[reference]
        rows.append({
            "area": f"{table}.{field}",
            "metric": f"unique_ids_overlapping_{reference}",
            "value": len(source & target),
            "interpretation": f"{len(source)} source IDs; {len(target)} reference IDs",
        })
    rows += [
        {"area": "performances", "metric": key, "value": value, "interpretation": ""}
        for key, value in performance_summary.items()
    ]
    return rows


def recommendation_rows(
    summaries: list[dict[str, Any]],
    columns: list[dict[str, Any]],
    valuation_summary: dict[str, Any],
    transfer_summary: dict[str, Any],
    performance_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    summary_by_table = {row["table"]: row for row in summaries}
    column_by_key = {(row["table"], row["column"]): row for row in columns}
    recommendations = [
        {
            "priority": "P0",
            "table": "transfer_history",
            "issue": "Zero transfer fees are semantically ambiguous.",
            "evidence": f"{column_by_key[('transfer_history', 'transfer_fee')]['zero_literal_count']:,} rows contain literal zero fees; README examples previously displayed dashes for some of these records.",
            "recommendation": "Retain zero as unclassified_zero unless transfer_type or another explicit field proves free transfer; never impute free status from zero alone.",
            "merge_policy": "Union by natural event key with source-specific fee columns and conflict flags.",
        },
        {
            "priority": "P0",
            "table": "player_profiles",
            "issue": "Current club, contract, agent, and status are snapshot fields.",
            "evidence": "Profile agreement is substantially lower for snapshot-sensitive fields than for date of birth, foot, and height.",
            "recommendation": "Keep source snapshot provenance and exclude untimestamped current/latest values from historical predictors.",
            "merge_policy": "Merge as dated evidence; do not overwrite the existing identity union.",
        },
        {
            "priority": "P0",
            "table": "player_performances",
            "issue": "Season summaries have a different grain from game-level appearances and FBref.",
            "evidence": f"{performance_summary['new_performance_rows']:,} new rows and {performance_summary['shared_performance_keys']:,} keys reconcile to current appearance aggregates.",
            "recommendation": "Clean as a separate player-team-competition-season fact and reconcile shared metrics before feature use.",
            "merge_policy": "Do not append to appearances or silently coalesce with FBref.",
        },
        {
            "priority": "P1",
            "table": "player_latest_market_value",
            "issue": "Latest valuation is a derived and leakage-prone snapshot.",
            "evidence": f"{valuation_summary['latest_exact_vs_history']:,} of {valuation_summary['latest_rows']:,} latest rows exactly match the maximum-date history record.",
            "recommendation": "Retain raw for audit only and derive latest/as-of values from canonical valuation history.",
            "merge_policy": "Exclude from canonical union as an independent fact source.",
        },
        {
            "priority": "P1",
            "table": "player_market_value",
            "issue": "Overlapping player-date observations can disagree.",
            "evidence": f"{valuation_summary['differing_player_date_values']:,} of {valuation_summary['shared_player_date_keys']:,} shared player-date keys have different values.",
            "recommendation": "Preserve both values and source timestamps; resolve only through an explicit source-priority or version rule.",
            "merge_policy": "Union with source provenance and conflict flags.",
        },
        {
            "priority": "P1",
            "table": "transfer_history",
            "issue": "Overlapping transfer events can disagree on fee or market value.",
            "evidence": f"{transfer_summary['shared_transfer_event_keys']:,} event keys overlap; {transfer_summary['differing_fees_on_shared_keys']:,} fee comparisons differ.",
            "recommendation": "Review fee/value conflicts and preserve both source fields rather than overwriting.",
            "merge_policy": "Canonical event spine plus one-to-many source evidence if conflicts remain.",
        },
        {
            "priority": "P1",
            "table": "all_text_fields",
            "issue": "Mojibake is present in names and locations.",
            "evidence": f"{sum(row['mojibake_cells'] for row in summaries):,} cells contain common mojibake markers.",
            "recommendation": "Preserve raw text and add a separately normalized UTF-8 helper only when a reversible repair is verified.",
            "merge_policy": "Never replace source display strings silently.",
        },
        {
            "priority": "P1",
            "table": "team_competitions_seasons",
            "issue": "Local row count differs materially from repository documentation.",
            "evidence": f"Local file has {summary_by_table['team_competitions_seasons']['data_rows']:,} rows versus 196,378 advertised.",
            "recommendation": "Treat the local file as the authoritative received snapshot and record the variance in manifests and documentation.",
            "merge_policy": "Do not infer or fabricate missing seasons.",
        },
        {
            "priority": "P2",
            "table": "player_national_performances",
            "issue": "The debut field is unavailable in the received file.",
            "evidence": f"{column_by_key[('player_national_performances', 'debut')]['missing_count']:,} of {summary_by_table['player_national_performances']['data_rows']:,} debut values are blank.",
            "recommendation": "Retain as missing and do not reconstruct debut dates from README samples.",
            "merge_policy": "Use matches/goals/team/career state only unless another dated source is joined.",
        },
        {
            "priority": "P2",
            "table": "player_teammates_played_with",
            "issue": "Relationship metrics are sparse.",
            "evidence": f"minutes_played_with missing {column_by_key[('player_teammates_played_with', 'minutes_played_with')]['missing_pct']:.1%}; joint_goal_participation missing {column_by_key[('player_teammates_played_with', 'joint_goal_participation')]['missing_pct']:.1%}.",
            "recommendation": "Keep missing values distinct from zero and document metric availability flags.",
            "merge_policy": "Separate directed edge table; aggregate only with explicit direction rules.",
        },
    ]
    for row in summaries:
        if row["duplicate_key_rows"]:
            recommendations.append({
                "priority": "P1",
                "table": row["table"],
                "issue": "Natural key is not unique.",
                "evidence": f"{row['duplicate_key_rows']:,} duplicate-key rows on {row['natural_key']}.",
                "recommendation": "Preserve raw rows, inspect differing attributes, and deduplicate only with a documented deterministic rule.",
                "merge_policy": "Block many-to-one joins until uniqueness is resolved.",
            })
    return recommendations


def write_readme(summary: dict[str, Any]) -> None:
    text = f"""# Supplemental football datalake audit

Generated by `scripts/audit_football_datalake.py` on {AUDIT_DATE.isoformat()}.

## Scope

- {summary['tables']} received Transfermarkt CSV tables
- {summary['source_rows']:,} source rows
- Existing MoveMaker Transfermarkt and FBref-derived identity layers used only for read-only reconciliation
- No raw or existing processed file modified

## Decision

Audit before merge. Preserve every received CSV as immutable source data. Union only valuations and transfer events into canonical histories; merge profiles and team details as source-specific dimension evidence; retain all other tables at their native grains.

## Outputs

- `source_file_manifest.csv`: source sizes, hashes, schemas, and Git LFS resolution state
- `table_summary.csv`: grain, natural key, row-count reconciliation, typing and encoding findings
- `column_dictionary.csv`: missingness, type validation, ranges, samples, zero and mojibake counts
- `key_duplicate_checks.csv`: natural-key integrity
- `foreign_key_checks.csv`: player, club, and competition coverage
- `overlap_summary.csv`: entity and fact-table overlap metrics
- `profile_overlap_summary.csv`: stable and snapshot field agreement
- `valuation_conflicts.csv`: shared player-date values that disagree
- `transfer_conflicts.csv`: shared transfer events with fee/value disagreements or missing comparisons
- `performance_reconciliation_summary.csv`: new season totals versus current appearance aggregates
- `performance_reconciliation_samples.csv`: bounded discrepancy examples
- `cleaning_recommendations.csv`: prioritized cleaning and integration rules
- `audit_summary.json`: machine-readable headline results
"""
    (OUTPUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    references = load_reference_sets()
    manifests, summaries, columns, keys, foreign, id_sets = profile_tables(references)
    profile = profile_overlap()
    valuation_rows, valuation_conflicts, valuation_summary = valuation_overlap()
    transfer_rows, transfer_conflicts, transfer_summary = transfer_overlap()
    performance, performance_samples, performance_summary = performance_reconciliation()
    overlap = build_overlap_rows(references, id_sets, valuation_rows, transfer_rows, performance_summary)
    recommendations = recommendation_rows(summaries, columns, valuation_summary, transfer_summary, performance_summary)

    write_csv("source_file_manifest.csv", manifests)
    write_csv("table_summary.csv", summaries)
    write_csv("column_dictionary.csv", columns)
    write_csv("key_duplicate_checks.csv", keys)
    write_csv("foreign_key_checks.csv", foreign)
    write_csv("overlap_summary.csv", overlap)
    write_csv("profile_overlap_summary.csv", profile)
    write_csv("valuation_conflicts.csv", valuation_conflicts, ["player_id", "valuation_date", "new_value", "current_value", "difference"])
    write_csv("transfer_conflicts.csv", transfer_conflicts, ["player_id", "transfer_date", "from_club_id", "to_club_id", "new_from_name", "new_to_name", "new_transfer_type", "new_transfer_fee", "current_transfer_fee", "fee_equal", "new_value_at_transfer", "current_market_value", "value_equal"])
    write_csv("performance_reconciliation_summary.csv", performance)
    write_csv("performance_reconciliation_samples.csv", performance_samples)
    write_csv("cleaning_recommendations.csv", recommendations)

    summary = {
        "audit_date": AUDIT_DATE.isoformat(),
        "tables": len(summaries),
        "source_rows": sum(row["data_rows"] for row in summaries),
        "all_lfs_objects_resolved": all(not row["is_git_lfs_pointer"] for row in summaries),
        "tables_with_duplicate_keys": [row["table"] for row in summaries if row["duplicate_key_rows"]],
        "tables_with_row_count_variance": [row["table"] for row in summaries if row["advertised_row_variance"] not in {None, 0}],
        "mojibake_cells": sum(row["mojibake_cells"] for row in summaries),
        "valuation_overlap": valuation_summary,
        "transfer_overlap": transfer_summary,
        "performance_reconciliation": performance_summary,
        "recommendation_count": len(recommendations),
        "blocking_recommendation_count": sum(row["priority"] == "P0" for row in recommendations),
    }
    (OUTPUT / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

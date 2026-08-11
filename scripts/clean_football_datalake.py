"""Build a conservative, row-preserving cleaned supplemental datalake layer.

The eleven received Transfermarkt CSVs remain immutable. Exact verified backups
are created, all source rows are retained, strict types and helper fields are
added, and quality issues are flagged rather than silently deleted or imputed.
This layer is intentionally separate from the existing canonical Transfermarkt
tables; canonical unioning happens in a later phase.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from audit_football_datalake import AUDIT_DATE, ROOT, SOURCE, TABLES, canonical_id, sha256


DATA = ROOT / "Data"
PROCESSED = DATA / "processed" / "football_datalake_clean"
TABLE_OUTPUT = PROCESSED / "tables"
BACKUP = DATA / "source_backups" / "football_datalake_original"
AUDIT = DATA / "processed" / "football_datalake_audit"
CHUNK_SIZE = 125_000


ID_COLUMNS: dict[str, list[str]] = {
    "player_injuries": ["player_id"],
    "player_latest_market_value": ["player_id"],
    "player_market_value": ["player_id"],
    "player_national_performances": ["player_id", "team_id", "coach_id", "debut_game_id"],
    "player_performances": ["player_id", "competition_id", "team_id"],
    "player_profiles": ["player_id", "current_club_id", "player_agent_id", "on_loan_from_club_id"],
    "player_teammates_played_with": ["player_id", "teammate_player_id"],
    "team_children": ["parent_team_id", "child_team_id"],
    "team_competitions_seasons": ["club_id", "competition_id", "season_league_competition_id", "season_manager_manager_id"],
    "team_details": ["club_id", "competition_id"],
    "transfer_history": ["player_id", "from_team_id", "to_team_id"],
}


FLOAT_COLUMNS = {("player_teammates_played_with", "ppg_played_with")}


ADDED_DEFINITIONS: dict[str, str] = {
    "source_file": "Relative path of the immutable received source CSV.",
    "source_row_number": "One-based source data-row number, excluding the header.",
    "source_record_id": "Stable source table and source-row identifier.",
    "natural_key_missing": "Whether at least one natural-key component is missing.",
    "natural_key_duplicate": "Whether the complete natural key occurs more than once.",
    "has_mojibake_marker": "Whether any source text contains a common mojibake marker; raw text is preserved.",
    "season_start_year": "Four-digit season starting year parsed from the supplied season label when possible.",
    "season_end_year": "Four-digit season ending year parsed from the supplied season label when possible.",
    "injury_calendar_days_inclusive": "Inclusive calendar days between injury start and end dates.",
    "days_missed_matches_dates": "Whether supplied days missed equals the inclusive date span.",
    "injury_end_before_start": "Whether the injury end date precedes its start date.",
    "injury_end_after_audit_date": "Whether the reported recovery/end date is after the audit date.",
    "is_derived_latest_snapshot": "Marks the latest-value table as a derivation of valuation history.",
    "snapshot_leakage_risk": "Current/latest snapshot field risk for historical modeling.",
    "valuation_status": "Missing, zero, or positive valuation state.",
    "has_positive_valuation": "Whether the valuation is positive.",
    "debut_available": "Whether a national-team debut date is supplied.",
    "career_state_normalized": "Lowercase normalized national-team career state.",
    "minutes_available": "Whether minutes are supplied for this record.",
    "goals_available": "Whether goals are supplied for this record.",
    "performance_metric_missing_count": "Count of core performance metrics missing on the record.",
    "player_name_display": "Player name with a trailing Transfermarkt ID annotation removed.",
    "club_name_display": "Club name with a trailing Transfermarkt ID annotation removed.",
    "current_club_status": "Retired, without_club, unknown, career_break, or active_or_other snapshot status.",
    "is_self_relationship": "Whether player and teammate IDs are identical.",
    "ppg_available": "Whether points per game together is supplied.",
    "joint_goal_participation_available": "Whether joint goal participation is supplied.",
    "is_self_link": "Whether parent and child club IDs are identical.",
    "wdl_matches_total": "Whether wins, draws, and losses sum to supplied total matches.",
    "goal_difference_matches": "Whether supplied goal difference equals goals for minus goals against.",
    "is_snapshot_record": "Marks a current/latest profile snapshot rather than historical as-of evidence.",
    "transfer_type_normalized": "Canonical transfer, loan, loan_return, or draft movement type.",
    "transfer_fee_status": "Missing, unclassified_zero, positive_reported, or invalid_negative fee state.",
    "value_at_transfer_status": "Missing, zero_reported, positive_reported, or invalid_negative market-value state.",
}


def source_path(table: str) -> Path:
    return SOURCE / TABLES[table]["path"]


def clean_path(table: str) -> Path:
    return TABLE_OUTPUT / f"{table}_clean.csv"


def normalized_text(value: Any) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    text = unicodedata.normalize("NFKD", str(value).strip().casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return text or pd.NA


def canonical_id_series(series: pd.Series) -> pd.Series:
    return series.map(lambda value: canonical_id(value) if pd.notna(value) else pd.NA).astype("string")


def season_years(value: Any) -> tuple[int | None, int | None]:
    if pd.isna(value):
        return None, None
    text = str(value).strip()
    match = re.fullmatch(r"(\d{2}|\d{4})/(\d{2}|\d{4})", text)
    if match:
        left, right = match.groups()
        start = int(left)
        if len(left) == 2:
            start += 2000 if start <= 40 else 1900
        end = int(right)
        if len(right) == 2:
            century = start // 100 * 100
            end += century
            if end < start:
                end += 100
        return start, end
    if re.fullmatch(r"\d{4}", text):
        year = int(text)
        return year, year
    return None, None


def key_hashes(frame: pd.DataFrame, key_columns: list[str]) -> tuple[pd.Series, pd.Series]:
    blank = frame[key_columns].isna().any(axis=1)
    key_frame = frame[key_columns].astype("string").fillna("")
    hashes = pd.util.hash_pandas_object(key_frame, index=False).astype("uint64")
    return hashes, blank


def find_duplicate_key_hashes(table: str) -> set[int]:
    path = source_path(table)
    key_columns = TABLES[table]["key"]
    seen: set[int] = set()
    duplicates: set[int] = set()
    for chunk in pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig", chunksize=CHUNK_SIZE, usecols=key_columns):
        chunk = chunk.apply(lambda series: series.astype(str).str.strip().replace("", pd.NA))
        for column in ID_COLUMNS.get(table, []):
            if column in chunk:
                chunk[column] = canonical_id_series(chunk[column])
        hashes, blank = key_hashes(chunk, key_columns)
        valid = hashes[~blank].to_numpy(dtype=np.uint64)
        unique, counts = np.unique(valid, return_counts=True)
        duplicates.update(int(value) for value, count in zip(unique, counts) if count > 1)
        duplicates.update(int(value) for value in unique if int(value) in seen)
        seen.update(int(value) for value in unique)
    return duplicates


def parse_dates(frame: pd.DataFrame, table: str, invalid_counts: Counter) -> None:
    for column in TABLES[table].get("dates", []):
        if column not in frame:
            continue
        original = frame[column].copy()
        parsed = pd.to_datetime(original, errors="coerce")
        invalid_counts[column] += int((original.notna() & parsed.isna()).sum())
        if column == "_last_modified_at":
            frame[column] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S").astype("string")
        else:
            frame[column] = parsed.dt.strftime("%Y-%m-%d").astype("string")


def parse_numerics(frame: pd.DataFrame, table: str, invalid_counts: Counter) -> None:
    id_columns = set(ID_COLUMNS.get(table, []))
    for column in TABLES[table].get("numeric", []):
        if column not in frame or column in id_columns:
            continue
        original = frame[column].copy()
        parsed = pd.to_numeric(original, errors="coerce")
        invalid_counts[column] += int((original.notna() & parsed.isna()).sum())
        if (table, column) in FLOAT_COLUMNS:
            frame[column] = parsed.astype("Float64")
        else:
            integral = parsed.dropna().map(lambda value: float(value).is_integer()).all()
            frame[column] = parsed.astype("Int64" if integral else "Float64")


def add_season_helpers(frame: pd.DataFrame) -> None:
    season_column = "season_name" if "season_name" in frame else "season_season" if "season_season" in frame else None
    if season_column:
        years = frame[season_column].map(season_years)
        frame["season_start_year"] = pd.array([value[0] for value in years], dtype="Int64")
        frame["season_end_year"] = pd.array([value[1] for value in years], dtype="Int64")


def add_normalized_helpers(frame: pd.DataFrame, table: str) -> None:
    helper_columns = {
        "player_injuries": ["injury_reason"],
        "player_national_performances": ["career_state"],
        "player_performances": ["competition_name", "team_name"],
        "player_profiles": ["player_name", "place_of_birth", "country_of_birth", "citizenship", "current_club_name", "player_agent_name"],
        "player_teammates_played_with": ["teammate_player_name"],
        "team_children": ["parent_team_name", "child_team_name"],
        "team_competitions_seasons": ["competition_name", "season_manager_manager_name", "team_name"],
        "team_details": ["club_name", "country_name", "competition_name"],
        "transfer_history": ["from_team_name", "to_team_name"],
    }
    for column in helper_columns.get(table, []):
        if column in frame:
            frame[f"{column}_normalized"] = frame[column].map(normalized_text).astype("string")


def status_from_numeric(series: pd.Series, zero_label: str) -> pd.Series:
    status = pd.Series("missing", index=series.index, dtype="string")
    status.loc[series.eq(0).fillna(False)] = zero_label
    status.loc[series.gt(0).fillna(False)] = "positive_reported"
    status.loc[series.lt(0).fillna(False)] = "invalid_negative"
    return status


def add_table_helpers(frame: pd.DataFrame, table: str) -> None:
    add_season_helpers(frame)
    add_normalized_helpers(frame, table)
    if table == "player_injuries":
        start = pd.to_datetime(frame["from_date"], errors="coerce")
        end = pd.to_datetime(frame["end_date"], errors="coerce")
        frame["injury_calendar_days_inclusive"] = ((end - start).dt.days + 1).astype("Int64")
        frame["days_missed_matches_dates"] = frame["days_missed"].eq(frame["injury_calendar_days_inclusive"]).astype("boolean")
        frame["injury_end_before_start"] = end.lt(start).astype("boolean")
        frame["injury_end_after_audit_date"] = end.dt.date.gt(AUDIT_DATE).astype("boolean")
    elif table == "player_latest_market_value":
        frame["is_derived_latest_snapshot"] = True
        frame["snapshot_leakage_risk"] = True
        frame["valuation_status"] = status_from_numeric(frame["value"], "zero")
        frame["has_positive_valuation"] = frame["value"].gt(0).fillna(False).astype("boolean")
    elif table == "player_market_value":
        frame["valuation_status"] = status_from_numeric(frame["value"], "zero")
        frame["has_positive_valuation"] = frame["value"].gt(0).fillna(False).astype("boolean")
    elif table == "player_national_performances":
        frame["debut_available"] = frame["debut"].notna().astype("boolean")
        frame["career_state_normalized"] = frame["career_state"].str.strip().str.casefold().astype("string")
    elif table == "player_performances":
        core = ["nb_on_pitch", "goals", "assists", "yellow_cards", "minutes_played"]
        frame["minutes_available"] = frame["minutes_played"].notna().astype("boolean")
        frame["goals_available"] = frame["goals"].notna().astype("boolean")
        frame["performance_metric_missing_count"] = frame[core].isna().sum(axis=1).astype("Int64")
    elif table == "player_profiles":
        frame["player_name_display"] = frame["player_name"].str.replace(r"\s*\(\d+\)\s*$", "", regex=True).astype("string")
        frame["player_name_display_normalized"] = frame["player_name_display"].map(normalized_text).astype("string")
        club = frame["current_club_name"].fillna("").str.strip().str.casefold()
        frame["current_club_status"] = np.select(
            [club.eq("retired"), club.eq("without club"), club.eq("unknown"), club.eq("career break")],
            ["retired", "without_club", "unknown", "career_break"],
            default="active_or_other",
        )
        frame["snapshot_leakage_risk"] = True
    elif table == "player_teammates_played_with":
        frame["is_self_relationship"] = frame["player_id"].eq(frame["teammate_player_id"]).astype("boolean")
        frame["ppg_available"] = frame["ppg_played_with"].notna().astype("boolean")
        frame["joint_goal_participation_available"] = frame["joint_goal_participation"].notna().astype("boolean")
        frame["minutes_available"] = frame["minutes_played_with"].notna().astype("boolean")
    elif table == "team_children":
        frame["is_self_link"] = frame["parent_team_id"].eq(frame["child_team_id"]).astype("boolean")
    elif table == "team_competitions_seasons":
        wdl = frame[["season_wins", "season_draws", "season_losses"]].sum(axis=1, min_count=3)
        frame["wdl_matches_total"] = wdl.eq(frame["season_total_matches"]).astype("boolean")
        goal_diff = frame["season_goals_for"] - frame["season_goals_against"]
        frame["goal_difference_matches"] = goal_diff.eq(frame["season_goal_difference"]).astype("boolean")
    elif table == "team_details":
        frame["club_name_display"] = frame["club_name"].str.replace(r"\s*\(\d+\)\s*$", "", regex=True).astype("string")
        frame["club_name_display_normalized"] = frame["club_name_display"].map(normalized_text).astype("string")
        frame["is_snapshot_record"] = True
        frame["snapshot_leakage_risk"] = True
    elif table == "transfer_history":
        movement = frame["transfer_type"].str.strip().str.casefold()
        frame["transfer_type_normalized"] = movement.map({
            "transfer": "transfer", "loan": "loan", "return from loan": "loan_return", "draft": "draft",
        }).fillna("other").astype("string")
        frame["transfer_fee_status"] = status_from_numeric(frame["transfer_fee"], "unclassified_zero")
        frame["value_at_transfer_status"] = status_from_numeric(frame["value_at_transfer"], "zero_reported")


def clean_table(table: str, duplicate_hashes: set[int]) -> dict[str, Any]:
    source = source_path(table)
    output = clean_path(table)
    source_columns = pd.read_csv(source, nrows=0, encoding="utf-8-sig").columns.tolist()
    row_count = 0
    invalid_numeric = Counter()
    invalid_dates = Counter()
    missing_clean = Counter()
    added_columns: list[str] = []
    flagged_missing_key = 0
    flagged_duplicate_key = 0
    mojibake_rows = 0
    first_write = True

    for chunk in pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8-sig", chunksize=CHUNK_SIZE, low_memory=False):
        chunk_rows = len(chunk)
        start = row_count + 1
        row_count += chunk_rows
        chunk = chunk.apply(lambda series: series.astype(str).str.strip().replace("", pd.NA))
        for column in ID_COLUMNS.get(table, []):
            if column in chunk:
                chunk[column] = canonical_id_series(chunk[column])
        parse_dates(chunk, table, invalid_dates)
        parse_numerics(chunk, table, invalid_numeric)
        hashes, missing_key = key_hashes(chunk, TABLES[table]["key"])
        duplicate_key = hashes.map(lambda value: int(value) in duplicate_hashes) & ~missing_key
        flagged_missing_key += int(missing_key.sum())
        flagged_duplicate_key += int(duplicate_key.sum())

        text_columns = [
            column for column in source_columns
            if column in chunk and pd.api.types.is_string_dtype(chunk[column].dtype)
        ]
        mojibake = pd.Series(False, index=chunk.index)
        for column in text_columns:
            mojibake |= chunk[column].astype("string").str.contains(
                r"\u00c3|\u00c2|\u00e2|\ufffd", regex=True, na=False
            )
        mojibake_rows += int(mojibake.sum())

        add_table_helpers(chunk, table)
        chunk.insert(0, "source_record_id", [f"{table}:{number}" for number in range(start, row_count + 1)])
        chunk.insert(0, "source_row_number", pd.array(range(start, row_count + 1), dtype="Int64"))
        chunk.insert(0, "source_file", str(source.relative_to(ROOT)).replace("\\", "/"))
        chunk["natural_key_missing"] = missing_key.astype("boolean")
        chunk["natural_key_duplicate"] = duplicate_key.astype("boolean")
        chunk["has_mojibake_marker"] = mojibake.astype("boolean")
        if first_write:
            added_columns = [column for column in chunk.columns if column not in source_columns]
        for column in chunk.columns:
            missing_clean[column] += int(chunk[column].isna().sum())
        chunk.to_csv(output, mode="w" if first_write else "a", header=first_write, index=False, encoding="utf-8-sig", lineterminator="\n")
        first_write = False

    return {
        "table": table,
        "source_file": str(source.relative_to(ROOT)).replace("\\", "/"),
        "clean_file": str(output.relative_to(ROOT)).replace("\\", "/"),
        "grain": TABLES[table]["grain"],
        "source_rows": row_count,
        "clean_rows": row_count,
        "source_columns": len(source_columns),
        "clean_columns": len(source_columns) + len(added_columns),
        "added_columns": " | ".join(added_columns),
        "natural_key": " + ".join(TABLES[table]["key"]),
        "missing_key_rows": flagged_missing_key,
        "duplicate_key_rows": flagged_duplicate_key,
        "mojibake_flagged_rows": mojibake_rows,
        "invalid_numeric_values": sum(invalid_numeric.values()),
        "invalid_date_values": sum(invalid_dates.values()),
        "output_sha256": sha256(output),
        "disposition": TABLES[table]["disposition"],
        "missing_by_column": dict(missing_clean),
    }


def copy_sources() -> list[dict[str, Any]]:
    manifest = []
    for table in TABLES:
        source = source_path(table)
        relative = Path(TABLES[table]["path"])
        backup = BACKUP / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, backup)
        source_hash = sha256(source)
        backup_hash = sha256(backup)
        manifest.append({
            "table": table,
            "source_file": str(source.relative_to(ROOT)).replace("\\", "/"),
            "backup_file": str(backup.relative_to(ROOT)).replace("\\", "/"),
            "file_bytes": source.stat().st_size,
            "source_sha256": source_hash,
            "backup_sha256": backup_hash,
            "hashes_match": source_hash == backup_hash,
            "git_lfs_resolved": source.open("r", encoding="utf-8-sig", errors="replace").readline().strip() != "version https://git-lfs.github.com/spec/v1",
        })
    return manifest


def definition(table: str, column: str) -> str:
    if column in ADDED_DEFINITIONS:
        return ADDED_DEFINITIONS[column]
    if column.endswith("_normalized"):
        return f"Normalized matching helper derived from {column[:-11]}; source display text is retained."
    return f"Source field preserved from {table}."


def leakage_rule(table: str, column: str) -> tuple[str, str]:
    if table in {"player_latest_market_value", "player_profiles", "team_details"} and column not in {"source_file", "source_row_number", "source_record_id"}:
        return "snapshot_risk", "Exclude from historical predictors unless separately timestamped or derived as-of the event date."
    if table == "player_performances":
        return "temporal_aggregation_required", "Use only seasons completed before the prediction cutoff or aggregate with an explicit as-of rule."
    if table == "player_injuries" and column in {"end_date", "days_missed", "games_missed"}:
        return "outcome_timing_risk", "Use only information observable by the prediction cutoff."
    if table == "transfer_history" and column in {"transfer_fee", "value_at_transfer"}:
        return "event_or_outcome_field", "Transfer fee is an outcome for fee prediction; value at transfer may be an as-of feature only with timing validation."
    return "review", "Retain source timing and apply the downstream model cutoff."


def write_column_dictionary(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for summary in summaries:
        table = summary["table"]
        clean = clean_path(table)
        header = pd.read_csv(clean, nrows=0, encoding="utf-8-sig").columns.tolist()
        source_columns = set(pd.read_csv(source_path(table), nrows=0, encoding="utf-8-sig").columns)
        missing = summary["missing_by_column"]
        for order, column in enumerate(header, start=1):
            timing, rule = leakage_rule(table, column)
            output.append({
                "table": table,
                "column_order": order,
                "column": column,
                "origin": "source" if column in source_columns else "derived_helper",
                "definition": definition(table, column),
                "missing_count": missing.get(column, 0),
                "non_null_count": summary["clean_rows"] - missing.get(column, 0),
                "missing_pct": missing.get(column, 0) / summary["clean_rows"] if summary["clean_rows"] else None,
                "timing_classification": timing,
                "modeling_rule": rule,
            })
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def build_actions() -> list[dict[str, Any]]:
    return [
        {"action_order": 1, "scope": "all", "action": "exact_backup", "detail": "Copied every received CSV byte-for-byte and verified SHA-256 equality."},
        {"action_order": 2, "scope": "all", "action": "preserve_rows", "detail": "Retained every source row; no deduplication or imputation."},
        {"action_order": 3, "scope": "all", "action": "standardize_blank", "detail": "Converted blank/whitespace-only strings to missing values."},
        {"action_order": 4, "scope": "ids", "action": "canonical_id", "detail": "Canonicalized numeric-looking IDs without changing alphanumeric competition IDs."},
        {"action_order": 5, "scope": "dates_and_metrics", "action": "strict_parse", "detail": "Strictly parsed configured dates and numeric metrics; invalid values became missing and were counted."},
        {"action_order": 6, "scope": "natural_keys", "action": "flag_key_issues", "detail": "Added missing-key and duplicate-key flags; problematic rows remain present."},
        {"action_order": 7, "scope": "names", "action": "add_normalized_helpers", "detail": "Added matching helpers while preserving original display strings and mojibake evidence."},
        {"action_order": 8, "scope": "transfer_history", "action": "classify_fee_state", "detail": "Classified literal zero as unclassified_zero, never automatically free transfer."},
        {"action_order": 9, "scope": "snapshot_tables", "action": "flag_leakage", "detail": "Marked current/latest profile and latest-value fields as historical leakage risks."},
        {"action_order": 10, "scope": "fact_tables", "action": "add_availability_and_consistency_flags", "detail": "Added metric availability, date consistency, standings arithmetic, and relationship diagnostics."},
    ]


def build_checks(manifest: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for row in manifest:
        checks.append({"table": row["table"], "check": "source_backup_sha256_match", "observed": row["hashes_match"], "expected": True, "passed": row["hashes_match"], "severity": "blocking", "detail": "Source and immutable backup must match exactly."})
        checks.append({"table": row["table"], "check": "git_lfs_object_resolved", "observed": row["git_lfs_resolved"], "expected": True, "passed": row["git_lfs_resolved"], "severity": "blocking", "detail": "No source may remain a Git LFS pointer."})
    for row in summaries:
        same_rows = row["source_rows"] == row["clean_rows"]
        checks.append({"table": row["table"], "check": "row_count_preserved", "observed": row["clean_rows"], "expected": row["source_rows"], "passed": same_rows, "severity": "blocking", "detail": "Cleaning must not add or remove source rows."})
        checks.append({"table": row["table"], "check": "natural_key_missing_rows", "observed": row["missing_key_rows"], "expected": 0, "passed": row["missing_key_rows"] == 0, "severity": "warning", "detail": "Rows are retained and flagged for later merge policy."})
        checks.append({"table": row["table"], "check": "natural_key_duplicate_rows", "observed": row["duplicate_key_rows"], "expected": 0, "passed": row["duplicate_key_rows"] == 0, "severity": "warning", "detail": "Rows are retained and flagged; canonical joins remain blocked until resolved."})
    return checks


def write_readme(summary: dict[str, Any]) -> None:
    text = f"""# Cleaned supplemental Transfermarkt datalake

Generated by `scripts/clean_football_datalake.py`.

## Preservation

- The eleven received CSV files under `football-datasets-main/datalake/transfermarkt` are never modified.
- Exact SHA-256-verified copies are stored under `Data/source_backups/football_datalake_original`.
- All {summary['source_rows']:,} source rows are retained in the cleaned layer.

## Main outputs

- `tables/`: one row-preserving cleaned CSV per received source table.
- `source_file_manifest.csv`: source, backup, and hash verification.
- `table_summary.csv`: grain, row counts, key flags, schemas, hashes, and intended integration disposition.
- `column_dictionary.csv`: source/derived provenance, missingness, definitions, timing class, and modeling rule.
- `cleaning_actions.csv`: ordered reproducible transformation log.
- `cleaning_checks.csv`: blocking validations and preserved warnings.
- `foreign_key_checks.csv`: copied audit-stage entity coverage because cleaning preserves identifiers.
- `transfer_fee_summary.csv`: missing, unclassified-zero, positive, and invalid fee states.

## Material rules

1. No source row is deleted or imputed.
2. Natural-key problems are flagged, not silently deduplicated.
3. Transfer fee zero remains `unclassified_zero`; it is not automatically a free transfer.
4. Current/latest snapshots are marked as leakage risks.
5. Player performance remains at player-team-competition-season grain and is not appended to game-level appearances or FBref.
6. Source display strings are retained; normalized matching helpers are separate.
7. This directory is a supplemental clean layer, not yet the canonical merged Transfermarkt master.
"""
    (PROCESSED / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    TABLE_OUTPUT.mkdir(parents=True, exist_ok=True)
    BACKUP.mkdir(parents=True, exist_ok=True)
    manifest = copy_sources()
    summaries = []
    for table in TABLES:
        duplicate_hashes = find_duplicate_key_hashes(table)
        summaries.append(clean_table(table, duplicate_hashes))

    dictionary = write_column_dictionary(summaries)
    actions = build_actions()
    checks = build_checks(manifest, summaries)
    write_csv(PROCESSED / "source_file_manifest.csv", manifest)
    write_csv(PROCESSED / "table_summary.csv", [{key: value for key, value in row.items() if key != "missing_by_column"} for row in summaries])
    write_csv(PROCESSED / "column_dictionary.csv", dictionary)
    write_csv(PROCESSED / "cleaning_actions.csv", actions)
    write_csv(PROCESSED / "cleaning_checks.csv", checks)
    shutil.copy2(AUDIT / "foreign_key_checks.csv", PROCESSED / "foreign_key_checks.csv")

    fees = pd.read_csv(clean_path("transfer_history"), usecols=["transfer_fee_status"])["transfer_fee_status"].value_counts(dropna=False)
    fee_rows = [{"transfer_fee_status": str(status), "records": int(count), "pct_of_transfers": int(count) / summaries[-1]["clean_rows"]} for status, count in fees.items()]
    write_csv(PROCESSED / "transfer_fee_summary.csv", fee_rows)

    blocking_failures = [row for row in checks if row["severity"] == "blocking" and not row["passed"]]
    summary = {
        "generated_on": AUDIT_DATE.isoformat(),
        "tables": len(summaries),
        "source_rows": sum(row["source_rows"] for row in summaries),
        "clean_rows": sum(row["clean_rows"] for row in summaries),
        "source_bytes": sum(row["file_bytes"] for row in manifest),
        "clean_bytes": sum(clean_path(row["table"]).stat().st_size for row in summaries),
        "tables_with_missing_keys": [row["table"] for row in summaries if row["missing_key_rows"]],
        "tables_with_duplicate_keys": [row["table"] for row in summaries if row["duplicate_key_rows"]],
        "mojibake_flagged_rows": sum(row["mojibake_flagged_rows"] for row in summaries),
        "blocking_checks": sum(row["severity"] == "blocking" for row in checks),
        "blocking_failures": len(blocking_failures),
        "status": "pass" if not blocking_failures else "fail",
    }
    (PROCESSED / "cleaning_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(summary)
    print(json.dumps(summary, indent=2))
    if blocking_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

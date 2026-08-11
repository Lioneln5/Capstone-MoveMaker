"""Create a canonical, audited copy of the 2017-2024 FBref-style data.

The source CSV files are immutable inputs. This script creates:

* an exact source backup with SHA-256 verification;
* one combined player-squad-season dataset;
* seven season-specific cleaned extracts;
* a source manifest, cleaning action log, data dictionary, and checks.

The cleaning is intentionally conservative. Legitimate sporting zeros are
preserved. Values are changed to missing only where the source structure proves
that zero means unavailable or not applicable.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
SOURCE = DATA / "2017-2024"
PROCESSED = DATA / "processed" / "fbref_2017_2024_clean"
BY_SEASON = PROCESSED / "by_season"
BACKUP = DATA / "source_backups" / "fbref_2017_2024_original"

SOURCE_KEY = ["player", "squad", "comp", "season"]
CLEAN_KEY = ["player_name", "birth_year", "squad_name", "competition_id", "season"]

COMPETITION_IDS = {
    "Premier League": "GB1",
    "La Liga": "ES1",
    "Serie A": "IT1",
    "Bundesliga": "L1",
    "Ligue 1": "FR1",
}

CODE_NATIONALITIES = {
    "gp GLP": "Guadeloupe",
    "sr SUR": "Suriname",
    "cy CYP": "Cyprus",
    "gd GRN": "Grenada",
    "pa PAN": "Panama",
    "lt LTU": "Lithuania",
    "tt TRI": "Trinidad and Tobago",
    "gt GUA": "Guatemala",
    "bo BOL": "Bolivia",
    "cu CUB": "Cuba",
    "tz TAN": "Tanzania",
    "kn SKN": "Saint Kitts and Nevis",
    "fo FRO": "Faroe Islands",
    "mt MLT": "Malta",
}

COLUMN_RENAMES = {
    "rk": "source_rank",
    "player": "player_name",
    "nation": "nationality_raw",
    "pos": "position",
    "squad": "squad_name",
    "comp": "competition_name",
    "age": "age",
    "born": "birth_year",
    "Matches Played": "matches_played",
    "Avg Mins per Match": "total_minutes",
    "Goals": "goals",
    "Assists": "assists",
    "Goals & Assists": "goals_and_assists",
    "Non Penalty Goals": "non_penalty_goals",
    "Penalty Kicks Made": "penalty_kicks_made",
    "Expected Goals": "expected_goals",
    "Exp NPG": "non_penalty_expected_goals",
    "Progressive Carries": "progressive_carries",
    "Progressive Passes": "progressive_passes",
    "Goals p 90": "goals_per90",
    "Assists p 90": "assists_per90",
    "Tackles attempted": "tackles_attempted",
    "Tackles Won": "tackles_won",
    "% Dribbles tackled": "dribbles_tackled_pct",
    "Shots blocked": "shots_blocked",
    "Passes blocked": "passes_blocked",
    "Interceptions": "interceptions",
    "Clearances": "clearances",
    "Errors made": "errors_made",
    "Goals Against": "goals_against",
    "Goals against p 90": "goals_against_per90",
    "Saves": "saves",
    "Saves %": "saves_pct",
    "Clean Sheets": "clean_sheets",
    "% Clean sheets": "clean_sheets_pct",
    "% Penalty saves": "penalty_saves_pct",
    "Passes Completed": "passes_completed",
    "Passes Attempted": "passes_attempted",
    "Pass completion %": "pass_completion_pct",
    "Progressive passes distance": "progressive_pass_distance",
    "% Short pass completed": "short_pass_completion_pct",
    "% Medium passes completed": "medium_pass_completion_pct",
    "% Long passes completed": "long_pass_completion_pct",
    "Key passes": "key_passes",
    "1/3": "passes_into_final_third",
    "Passes into penalty area": "passes_into_penalty_area",
    "touches_def_pen": "touches_defensive_penalty_area",
    "Take ons attempted": "take_ons_attempted",
    "% Successful take-ons": "successful_take_ons_pct",
    "Times tackled during take-on": "tackled_during_take_on_pct",
    "carries final 3rd": "carries_into_final_third",
    "carries penalty area": "carries_into_penalty_area",
    "Possessions lost": "possessions_lost",
    "Total Shots": "total_shots",
    "% Shots on target": "shots_on_target_pct",
    "Shots p 90": "shots_per90",
    "Goals per shot": "goals_per_shot",
    "Goals per shot on target": "goals_per_shot_on_target",
    "% Aerial Duels won": "aerial_duels_won_pct",
    "Shot creating actions p 90": "shot_creating_actions_per90",
    "Goal creating actions p 90": "goal_creating_actions_per90",
    "Crosses Stopped": "crosses_stopped_pct",
    "season": "season",
}

DUPLICATE_SOURCE_COLUMNS = {
    "Goals Scored": "Goals",
    "carries_prgc": "Progressive Carries",
}

GOALKEEPER_COLUMNS = [
    "goals_against",
    "goals_against_per90",
    "saves",
    "saves_pct",
    "clean_sheets",
    "clean_sheets_pct",
    "penalty_saves_pct",
    "crosses_stopped_pct",
]

INTEGER_COLUMNS = [
    "source_rank",
    "age",
    "birth_year",
    "matches_played",
    "goals",
    "assists",
    "goals_and_assists",
    "non_penalty_goals",
    "penalty_kicks_made",
    "progressive_carries",
    "progressive_passes",
    "tackles_attempted",
    "tackles_won",
    "shots_blocked",
    "passes_blocked",
    "interceptions",
    "clearances",
    "errors_made",
    "goals_against",
    "saves",
    "clean_sheets",
    "passes_completed",
    "passes_attempted",
    "progressive_pass_distance",
    "key_passes",
    "passes_into_final_third",
    "passes_into_penalty_area",
    "touches_defensive_penalty_area",
    "take_ons_attempted",
    "carries_into_final_third",
    "carries_into_penalty_area",
    "possessions_lost",
    "total_shots",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.casefold().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", "", text)


def copy_and_verify_sources(files: list[Path]) -> pd.DataFrame:
    BACKUP.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in files:
        target = BACKUP / source.name
        source_hash = sha256(source)
        if target.exists():
            if sha256(target) != source_hash:
                raise RuntimeError(f"Existing backup does not match source: {target}")
        else:
            shutil.copy2(source, target)
        backup_hash = sha256(target)
        rows.append(
            {
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


def add_check(checks: list[dict], name: str, observed, expected, passed: bool, notes: str) -> None:
    checks.append(
        {
            "check": name,
            "observed": observed,
            "expected": expected,
            "passed": bool(passed),
            "notes": notes,
        }
    )


def build_dictionary(clean: pd.DataFrame) -> pd.DataFrame:
    definitions = {
        "fbref_row_id": "Deterministic SHA-256 identifier derived from the canonical player-squad-season key.",
        "player_squad_season_key": "Readable natural key: normalized player, birth year, normalized squad, competition ID, and season.",
        "player_identity_key": "Normalized player name plus reported birth year; helper for identity resolution, not a universal ID.",
        "source_file": "Original seasonal CSV filename.",
        "source_row_number": "One-based physical CSV row number including the header row.",
        "source_rank": "Rank value retained from the source; excluded from modeling.",
        "player_name": "Player display name retained with accents and original spelling.",
        "player_name_normalized": "Lowercase accent-free alphanumeric player name for controlled matching.",
        "nationality_raw": "Nationality exactly as supplied by the source.",
        "nationality": "Cleaned nationality display; code-like source values expanded to country/territory names.",
        "position": "Broad source position: GK, DF, MF, or FW.",
        "is_goalkeeper": "True when the source position is GK.",
        "squad_name": "Squad display name retained from the source.",
        "squad_name_normalized": "Lowercase accent-free alphanumeric squad name for controlled matching.",
        "competition_name": "Big Five domestic league name.",
        "competition_id": "Transfermarkt domestic competition identifier mapped from competition_name.",
        "season": "Season label in YYYY-YYYY format.",
        "season_start_year": "First calendar year in season.",
        "season_end_year": "Second calendar year in season.",
        "season_end_date": "Standardized season completion date used for leakage-safe joins (June 30).",
        "age": "Source-reported player age for the season.",
        "birth_year": "Source-reported birth year.",
        "age_birth_year_consistent": "Whether age is within the two plausible integer ages implied by season start year and birth year.",
        "total_minutes": "Total league minutes. Renamed from the misleading source label 'Avg Mins per Match'.",
        "tackled_during_take_on_pct": "Percentage of take-on attempts in which the player was tackled; corrected from a misleading count-style source label.",
        "crosses_stopped_pct": "Goalkeeper percentage of opponent crosses stopped; corrected from an incomplete source label.",
        "minutes_per_appearance": "Derived total_minutes divided by matches_played.",
        "nineties": "Derived total_minutes divided by 90.",
        "goalkeeper_metrics_available": "True for goalkeeper rows outside 2017-18, where goalkeeper values are present.",
        "dribbles_tackled_pct_available": "False in 2017-18 because the entire source field is zero/unavailable.",
    }
    transformations = {
        "total_minutes": "Renamed; values unchanged.",
        "tackled_during_take_on_pct": "Renamed to reflect percentage semantics; decimal values unchanged.",
        "crosses_stopped_pct": "Renamed to reflect percentage semantics; decimal values unchanged.",
        "nationality": "Known code-like values expanded; original retained in nationality_raw.",
        "minutes_per_appearance": "Derived with safe division.",
        "nineties": "Derived with safe division.",
    }
    for column in GOALKEEPER_COLUMNS:
        transformations[column] = (
            "Set missing for non-goalkeepers; also set missing for all 2017-18 rows because the source column is entirely zero."
        )
    transformations["dribbles_tackled_pct"] = (
        "Set missing for 2017-18 because the entire source column is zero; other seasons unchanged."
    )

    identity = {
        "fbref_row_id",
        "player_squad_season_key",
        "player_identity_key",
        "source_file",
        "source_row_number",
        "source_rank",
        "player_name",
        "player_name_normalized",
        "nationality_raw",
        "nationality",
        "position",
        "is_goalkeeper",
        "squad_name",
        "squad_name_normalized",
        "competition_name",
        "competition_id",
        "season",
        "season_start_year",
        "season_end_year",
        "season_end_date",
        "age",
        "birth_year",
        "age_birth_year_consistent",
        "goalkeeper_metrics_available",
        "dribbles_tackled_pct_available",
    }
    rows = []
    for order, column in enumerate(clean.columns, start=1):
        series = clean[column]
        friendly = column.replace("_", " ")
        definition = definitions.get(column, f"Cleaned FBref player-season field: {friendly}.")
        role = "Identity/provenance" if column in identity else "Seasonal performance feature"
        rows.append(
            {
                "column_order": order,
                "column": column,
                "definition": definition,
                "role": role,
                "dtype": str(series.dtype),
                "non_null_count": int(series.notna().sum()),
                "missing_count": int(series.isna().sum()),
                "missing_pct": float(series.isna().mean()),
                "distinct_count": int(series.nunique(dropna=True)),
                "transformation": transformations.get(column, "Renamed to canonical snake_case; source values otherwise retained."),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    files = sorted(SOURCE.glob("cleaned_*.csv"))
    if len(files) != 7:
        raise RuntimeError(f"Expected seven source CSV files, found {len(files)}")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    BY_SEASON.mkdir(parents=True, exist_ok=True)
    manifest = copy_and_verify_sources(files)

    source_frames: list[pd.DataFrame] = []
    checks: list[dict] = []
    actions: list[dict] = []
    expected_columns: list[str] | None = None
    source_rows = 0
    source_duplicate_rows = 0
    duplicate_equal = {column: True for column in DUPLICATE_SOURCE_COLUMNS}

    for path in files:
        frame = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
        source_rows += len(frame)
        if expected_columns is None:
            expected_columns = frame.columns.tolist()
        elif frame.columns.tolist() != expected_columns:
            raise RuntimeError(f"Schema differs in {path.name}")
        source_duplicate_rows += int(frame.duplicated(SOURCE_KEY).sum())
        for duplicate, canonical in DUPLICATE_SOURCE_COLUMNS.items():
            duplicate_equal[duplicate] &= bool(frame[duplicate].equals(frame[canonical]))
        frame.insert(0, "source_file", path.name)
        frame.insert(1, "source_row_number", np.arange(2, len(frame) + 2))
        source_frames.append(frame)

    source = pd.concat(source_frames, ignore_index=True)
    add_check(checks, "Source files", len(files), 7, len(files) == 7, "One file per season.")
    add_check(checks, "Source rows", len(source), 18243, len(source) == 18243, "All source rows retained.")
    add_check(checks, "Source natural-key duplicates", source_duplicate_rows, 0, source_duplicate_rows == 0, "+".join(SOURCE_KEY))
    add_check(checks, "Common 65-column source schema", len(expected_columns or []), 65, len(expected_columns or []) == 65, "Trace columns excluded.")
    for duplicate, canonical in DUPLICATE_SOURCE_COLUMNS.items():
        add_check(
            checks,
            f"Duplicate equality: {duplicate}",
            duplicate_equal[duplicate],
            True,
            duplicate_equal[duplicate],
            f"Required before dropping duplicate of {canonical}.",
        )
        if not duplicate_equal[duplicate]:
            raise RuntimeError(f"Cannot drop {duplicate}; it differs from {canonical}")

    clean = source.drop(columns=list(DUPLICATE_SOURCE_COLUMNS)).rename(columns=COLUMN_RENAMES).copy()
    unknown = [column for column in clean.columns if column not in {"source_file", "source_row_number"} | set(COLUMN_RENAMES.values())]
    if unknown:
        raise RuntimeError(f"Unmapped cleaned columns: {unknown}")

    text_columns = [
        "source_file",
        "player_name",
        "nationality_raw",
        "position",
        "squad_name",
        "competition_name",
        "season",
    ]
    for column in text_columns:
        clean[column] = clean[column].astype("string").str.strip()

    numeric_columns = [column for column in clean.columns if column not in text_columns]
    for column in numeric_columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    clean["nationality"] = clean["nationality_raw"].replace(CODE_NATIONALITIES)
    clean["player_name_normalized"] = clean["player_name"].map(normalize_text).astype("string")
    clean["squad_name_normalized"] = clean["squad_name"].map(normalize_text).astype("string")
    clean["competition_id"] = clean["competition_name"].map(COMPETITION_IDS).astype("string")
    clean["is_goalkeeper"] = clean["position"].eq("GK")
    clean["season_start_year"] = pd.to_numeric(clean["season"].str.slice(0, 4), errors="coerce").astype("Int64")
    clean["season_end_year"] = pd.to_numeric(clean["season"].str.slice(-4), errors="coerce").astype("Int64")
    clean["season_end_date"] = pd.to_datetime(clean["season"].str.slice(-4) + "-06-30", errors="coerce")
    implied_age = clean["season_start_year"] - clean["birth_year"]
    clean["age_birth_year_consistent"] = clean["age"].isin(pd.concat([implied_age - 1, implied_age], ignore_index=True))
    # The Series.isin expression above checks global membership, so replace it with row-wise logic.
    clean["age_birth_year_consistent"] = clean["age"].eq(implied_age) | clean["age"].eq(implied_age - 1)

    clean["minutes_per_appearance"] = (
        clean["total_minutes"] / clean["matches_played"].replace(0, np.nan)
    )
    clean["nineties"] = clean["total_minutes"] / 90.0

    non_goalkeeper = ~clean["is_goalkeeper"]
    first_season = clean["season"].eq("2017-2018")
    non_goalkeeper_cells = int(non_goalkeeper.sum()) * len(GOALKEEPER_COLUMNS)
    first_season_goalkeeper_cells = int((first_season & clean["is_goalkeeper"]).sum()) * len(GOALKEEPER_COLUMNS)
    clean.loc[non_goalkeeper | first_season, GOALKEEPER_COLUMNS] = np.nan
    clean.loc[first_season, "dribbles_tackled_pct"] = np.nan
    clean["goalkeeper_metrics_available"] = clean["is_goalkeeper"] & ~first_season
    clean["dribbles_tackled_pct_available"] = ~first_season

    clean["player_identity_key"] = (
        clean["player_name_normalized"] + "|" + clean["birth_year"].astype("Int64").astype("string")
    )
    clean["player_squad_season_key"] = (
        clean["player_identity_key"]
        + "|"
        + clean["squad_name_normalized"]
        + "|"
        + clean["competition_id"]
        + "|"
        + clean["season"]
    )
    clean["fbref_row_id"] = clean["player_squad_season_key"].map(
        lambda value: hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    )

    for column in INTEGER_COLUMNS:
        clean[column] = clean[column].round().astype("Int64")
    clean["source_row_number"] = clean["source_row_number"].astype("Int64")

    preferred = [
        "fbref_row_id",
        "player_squad_season_key",
        "player_identity_key",
        "source_file",
        "source_row_number",
        "source_rank",
        "player_name",
        "player_name_normalized",
        "nationality",
        "nationality_raw",
        "position",
        "is_goalkeeper",
        "squad_name",
        "squad_name_normalized",
        "competition_name",
        "competition_id",
        "season",
        "season_start_year",
        "season_end_year",
        "season_end_date",
        "age",
        "birth_year",
        "age_birth_year_consistent",
        "matches_played",
        "total_minutes",
        "minutes_per_appearance",
        "nineties",
    ]
    remaining = [column for column in clean.columns if column not in preferred]
    clean = clean[preferred + remaining]
    clean = clean.sort_values(
        ["season_start_year", "competition_name", "squad_name", "player_name", "birth_year"],
        kind="stable",
    ).reset_index(drop=True)

    key_duplicates = int(clean.duplicated(CLEAN_KEY).sum())
    add_check(checks, "Clean rows equal source rows", len(clean), len(source), len(clean) == len(source), "No row deletion or expansion.")
    add_check(checks, "Clean natural-key duplicates", key_duplicates, 0, key_duplicates == 0, "+".join(CLEAN_KEY))
    add_check(checks, "fbref_row_id uniqueness", clean["fbref_row_id"].nunique(), len(clean), clean["fbref_row_id"].is_unique, "SHA-256 canonical key.")
    add_check(checks, "Competition mapping completeness", int(clean["competition_id"].isna().sum()), 0, clean["competition_id"].notna().all(), "All five leagues mapped.")
    add_check(checks, "Season parsing completeness", int(clean["season_end_date"].isna().sum()), 0, clean["season_end_date"].notna().all(), "YYYY-YYYY to June 30.")
    add_check(checks, "Age/birth-year consistency", int((~clean["age_birth_year_consistent"]).sum()), 0, clean["age_birth_year_consistent"].all(), "Age equals start-year minus birth-year or one less.")
    add_check(
        checks,
        "Goals and assists identity",
        int(clean["goals_and_assists"].eq(clean["goals"] + clean["assists"]).sum()),
        len(clean),
        clean["goals_and_assists"].eq(clean["goals"] + clean["assists"]).all(),
        "goals_and_assists = goals + assists",
    )
    add_check(
        checks,
        "Non-penalty goals identity",
        int(clean["non_penalty_goals"].eq(clean["goals"] - clean["penalty_kicks_made"]).sum()),
        len(clean),
        clean["non_penalty_goals"].eq(clean["goals"] - clean["penalty_kicks_made"]).all(),
        "non_penalty_goals = goals - penalty_kicks_made",
    )
    add_check(
        checks,
        "Non-goalkeeper goalkeeper fields missing",
        int(clean.loc[~clean["is_goalkeeper"], GOALKEEPER_COLUMNS].notna().sum().sum()),
        0,
        not clean.loc[~clean["is_goalkeeper"], GOALKEEPER_COLUMNS].notna().any().any(),
        "Structural not-applicable values are null, not zero.",
    )
    add_check(
        checks,
        "2017-18 goalkeeper fields missing",
        int(clean.loc[first_season, GOALKEEPER_COLUMNS].notna().sum().sum()),
        0,
        not clean.loc[first_season, GOALKEEPER_COLUMNS].notna().any().any(),
        "Entire source fields were zero/unavailable.",
    )
    add_check(
        checks,
        "2017-18 dribble-tackle percentage missing",
        int(clean.loc[first_season, "dribbles_tackled_pct"].notna().sum()),
        0,
        clean.loc[first_season, "dribbles_tackled_pct"].isna().all(),
        "Entire source field was zero/unavailable.",
    )
    add_check(
        checks,
        "Negative minutes",
        int(clean["total_minutes"].lt(0).sum()),
        0,
        not clean["total_minutes"].lt(0).any(),
        "Sign validation.",
    )
    add_check(
        checks,
        "Backup hashes match",
        int(manifest["hashes_match"].sum()),
        len(manifest),
        manifest["hashes_match"].all(),
        "Exact binary backup verification.",
    )

    actions.extend(
        [
            {
                "action": "Preserve source files",
                "scope": "7 CSV files",
                "affected_rows_or_cells": 0,
                "details": "Sources were read-only; exact SHA-256-verified copies were created under Data/source_backups.",
            },
            {
                "action": "Retain all source rows",
                "scope": "Combined dataset",
                "affected_rows_or_cells": len(clean),
                "details": "No duplicate natural-key rows existed, so no rows were removed.",
            },
            {
                "action": "Drop duplicate field",
                "scope": "Goals Scored",
                "affected_rows_or_cells": len(clean),
                "details": "Exactly equal to Goals in all rows.",
            },
            {
                "action": "Drop duplicate field",
                "scope": "carries_prgc",
                "affected_rows_or_cells": len(clean),
                "details": "Exactly equal to Progressive Carries in all rows.",
            },
            {
                "action": "Correct misleading label",
                "scope": "Avg Mins per Match -> total_minutes",
                "affected_rows_or_cells": len(clean),
                "details": "Values behave as season total minutes; values were not altered.",
            },
            {
                "action": "Correct misleading label",
                "scope": "Times tackled during take-on -> tackled_during_take_on_pct",
                "affected_rows_or_cells": len(clean),
                "details": "Decimal values behave as percentages; values were not altered.",
            },
            {
                "action": "Correct incomplete label",
                "scope": "Crosses Stopped -> crosses_stopped_pct",
                "affected_rows_or_cells": len(clean),
                "details": "Goalkeeper decimal values behave as stopped-cross percentage; values were not altered.",
            },
            {
                "action": "Expand code-like nationalities",
                "scope": "nationality",
                "affected_rows_or_cells": int(clean["nationality_raw"].isin(CODE_NATIONALITIES).sum()),
                "details": "Original values retained in nationality_raw.",
            },
            {
                "action": "Convert structural zero to missing",
                "scope": "Goalkeeper fields for non-goalkeepers",
                "affected_rows_or_cells": non_goalkeeper_cells,
                "details": "Eight goalkeeper-only metrics are not applicable to outfield rows.",
            },
            {
                "action": "Convert unavailable zero to missing",
                "scope": "2017-18 goalkeeper fields",
                "affected_rows_or_cells": first_season_goalkeeper_cells,
                "details": "All eight source fields were zero for the entire season.",
            },
            {
                "action": "Convert unavailable zero to missing",
                "scope": "2017-18 dribbles_tackled_pct",
                "affected_rows_or_cells": int(first_season.sum()),
                "details": "The entire source field was zero; later-season legitimate zeros were retained.",
            },
            {
                "action": "Add traceable keys and helpers",
                "scope": "All rows",
                "affected_rows_or_cells": len(clean),
                "details": "Added source provenance, normalized names, natural keys, SHA-256 row IDs, season dates, nineties, and availability flags.",
            },
        ]
    )

    combined_path = PROCESSED / "fbref_player_squad_season_clean.csv"
    clean.to_csv(combined_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    season_counts = []
    for season, frame in clean.groupby("season", sort=True):
        filename = f"fbref_{season.replace('-', '_')}_clean.csv"
        path = BY_SEASON / filename
        frame.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
        season_counts.append({"season": season, "rows": len(frame), "output_file": path.relative_to(ROOT).as_posix()})

    manifest["source_row_count"] = [len(pd.read_csv(SOURCE / name, usecols=["rk"])) for name in manifest["source_file"]]
    manifest.to_csv(PROCESSED / "source_file_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(actions).to_csv(PROCESSED / "fbref_cleaning_actions.csv", index=False, encoding="utf-8-sig")
    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(PROCESSED / "fbref_cleaning_checks.csv", index=False, encoding="utf-8-sig")
    dictionary = build_dictionary(clean)
    dictionary.to_csv(PROCESSED / "fbref_column_dictionary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(season_counts).to_csv(PROCESSED / "fbref_season_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "source_files": len(files),
        "source_rows": len(source),
        "clean_rows": len(clean),
        "clean_columns": len(clean.columns),
        "source_duplicate_natural_key_rows": source_duplicate_rows,
        "clean_duplicate_natural_key_rows": key_duplicates,
        "dropped_duplicate_columns": list(DUPLICATE_SOURCE_COLUMNS),
        "nationality_values_expanded": int(clean["nationality_raw"].isin(CODE_NATIONALITIES).sum()),
        "goalkeeper_structural_null_cells": non_goalkeeper_cells + first_season_goalkeeper_cells,
        "dribbles_tackled_2017_18_null_cells": int(first_season.sum()),
        "checks_passed": int(checks_frame["passed"].sum()),
        "checks_total": len(checks_frame),
        "all_checks_passed": bool(checks_frame["passed"].all()),
        "combined_output": combined_path.relative_to(ROOT).as_posix(),
        "backup_directory": BACKUP.relative_to(ROOT).as_posix(),
    }
    (PROCESSED / "fbref_cleaning_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    readme = f"""# Cleaned FBref 2017-2024 data

This directory is generated by `scripts/clean_fbref_seasonal.py`.

## Preservation

- The seven CSV files under `Data/2017-2024` are never modified.
- Exact binary copies are stored under `{BACKUP.relative_to(ROOT).as_posix()}`.
- `source_file_manifest.csv` records and verifies SHA-256 hashes.

## Main output

- `fbref_player_squad_season_clean.csv`: {len(clean):,} rows at one player-squad-competition-season record per row.
- `by_season/`: seven season-specific copies with the same canonical schema.
- `fbref_column_dictionary.csv`: definitions, types, null counts, and transformations.
- `fbref_cleaning_actions.csv`: explicit transformation log.
- `fbref_cleaning_checks.csv`: machine-readable validation results.

## Material cleaning rules

1. No rows were deleted because the source natural key is unique.
2. `Goals Scored` and `carries_prgc` were removed only after proving exact equality to their retained counterparts.
3. Misleading source labels for total minutes, take-on tackled percentage, and goalkeeper crosses-stopped percentage were corrected; values were unchanged.
4. Legitimate zeros were preserved.
5. Goalkeeper-only fields were set missing for non-goalkeepers.
6. 2017-18 goalkeeper fields and dribble-tackle percentage were set missing because each source field was entirely zero/unavailable.
7. Display names remain intact; normalized helpers were added rather than replacing source identity text.

Player-to-Transfermarkt identity resolution and squad-to-club resolution remain separate reviewed crosswalk stages. They are not silently embedded in this source-cleaning layer.
"""
    (PROCESSED / "README.md").write_text(readme, encoding="utf-8")

    if not checks_frame["passed"].all():
        failed = checks_frame.loc[~checks_frame["passed"], "check"].tolist()
        raise RuntimeError(f"Cleaning completed with failed checks: {failed}")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

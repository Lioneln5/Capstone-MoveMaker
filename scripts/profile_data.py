"""Profile MoveMaker CSV sources and emit machine-readable audit artifacts.

The profiler is deliberately read-only with respect to source data. It processes
large files in chunks so the audit can be rerun as the data is refreshed.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs" / "movemaker_data_audit"
CHUNK_SIZE = 100_000
DISTINCT_CAP = 50_000
NULL_TOKENS = {"", "na", "n/a", "null", "none", "nan", "nat"}


TABLE_GRAINS = {
    "cleaned_season": "One row per player-squad-competition-season statistical record.",
    "appearances": "One row per player appearance in a game.",
    "club_games": "One row per club in a game (normally two club-perspective rows per game).",
    "clubs": "One row per club; mostly a current/latest scraped snapshot rather than a historical panel.",
    "competitions": "One row per competition.",
    "countries": "One row per country.",
    "game_events": "One row per recorded in-game event.",
    "game_lineups": "One row per player listed in a game lineup or substitutes list.",
    "games": "One row per game.",
    "national_teams": "One row per national team; mostly a current/latest scraped snapshot.",
    "player_valuations": "One row per player valuation observation date.",
    "players": "One row per player; current/latest attributes are snapshot fields.",
    "transfers": "One row per recorded player movement between clubs on a transfer date.",
}


PRIMARY_KEYS = {
    "cleaned_season": ["player", "squad", "comp", "season"],
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


TABLE_NOTES = {
    "cleaned_season": (
        "Detailed seasonal performance features. Rows have names rather than Transfermarkt IDs, "
        "so identity resolution is required. Use only seasons completed before the transfer date."
    ),
    "appearances": (
        "Useful for pre-transfer form and post-transfer sporting outcomes. player_club_id is the club "
        "at the appearance; player_current_club_id is a current snapshot and may leak future information."
    ),
    "club_games": "Team-perspective match results and context; suitable for rolling club-strength features when time-filtered.",
    "clubs": (
        "Club dimension. Several attributes describe the latest scrape, not the historical transfer date, "
        "and therefore should not be used naively in backtests."
    ),
    "competitions": "Competition dimension; generally safe descriptive context when definitions are stable.",
    "countries": "Country dimension; generally safe descriptive context when definitions are stable.",
    "game_events": "Event-level source for granular outcomes; aggregate by player/club and strictly respect event date.",
    "game_lineups": "Lineup/position source; can derive selection and role outcomes after transfer or pre-transfer role features.",
    "games": "Game spine for dates, results, clubs, competitions, and attendance; filter national-team games where club analysis is intended.",
    "national_teams": "National-team snapshot; current values/rankings are not historical unless separately dated.",
    "player_valuations": (
        "Time-stamped financial series. Nearest valuation on or before transfer can be a feature; later valuations "
        "can define financial return at a fixed horizon."
    ),
    "players": (
        "Player identity dimension. Birth, citizenship, foot, position, and height are useful; current club, contract, "
        "current value, caps, goals, agent, and last_season may reflect a later snapshot and can leak."
    ),
    "transfers": (
        "Likely modeling-event spine. Includes future-dated and non-fee movements, so scope rules are required. "
        "Zero fees may mean free transfer, loan, return, or undisclosed rather than a true zero purchase price."
    ),
}


SEASONAL_DEFINITIONS = {
    "rk": "Source row rank within the seasonal extract; not a stable player identifier.",
    "player": "Player display name.",
    "nation": "Player nationality shown by the seasonal source.",
    "pos": "Broad playing position code (for example GK, DF, MF, FW, or combinations).",
    "squad": "Club/squad represented during the season.",
    "comp": "Domestic league competition represented during the season.",
    "age": "Player age reported for the season.",
    "born": "Player birth year.",
    "Matches Played": "Number of matches in which the player appeared.",
    "Avg Mins per Match": "Reported minutes field. Audit indicates this behaves like total minutes, despite the column label.",
    "Goals": "Goals scored.",
    "Assists": "Assists recorded.",
    "Goals & Assists": "Combined goals and assists.",
    "Non Penalty Goals": "Goals excluding penalty kicks.",
    "Penalty Kicks Made": "Penalty kicks scored.",
    "Expected Goals": "Expected goals (xG).",
    "Exp NPG": "Non-penalty expected goals (npxG).",
    "Progressive Carries": "Progressive carries; appears duplicated by carries_prgc in these files.",
    "Progressive Passes": "Completed progressive passes.",
    "Goals p 90": "Goals per 90 minutes.",
    "Assists p 90": "Assists per 90 minutes.",
    "Tackles attempted": "Tackles attempted.",
    "Tackles Won": "Tackles won.",
    "% Dribbles tackled": "Percentage of dribblers tackled successfully.",
    "Shots blocked": "Opponent shots blocked.",
    "Passes blocked": "Opponent passes blocked.",
    "Interceptions": "Interceptions made.",
    "Clearances": "Clearances made.",
    "Errors made": "Errors leading to an opponent shot or similarly defined source error event.",
    "Goals Against": "Goals conceded while playing goalkeeper; zero-filled for many non-goalkeepers.",
    "Goals against p 90": "Goalkeeper goals conceded per 90; zero-filled for many non-goalkeepers.",
    "Saves": "Goalkeeper saves; zero-filled for many non-goalkeepers.",
    "Saves %": "Goalkeeper save percentage; zero-filled for many non-goalkeepers.",
    "Clean Sheets": "Goalkeeper clean sheets; zero-filled for many non-goalkeepers.",
    "% Clean sheets": "Goalkeeper clean-sheet percentage; zero-filled for many non-goalkeepers.",
    "% Penalty saves": "Goalkeeper penalty save percentage; zero-filled for many non-goalkeepers.",
    "Passes Completed": "Completed passes.",
    "Passes Attempted": "Attempted passes.",
    "Pass completion %": "Pass completion percentage.",
    "Progressive passes distance": "Total progressive distance of completed passes.",
    "% Short pass completed": "Short-pass completion percentage.",
    "% Medium passes completed": "Medium-pass completion percentage.",
    "% Long passes completed": "Long-pass completion percentage.",
    "Key passes": "Passes leading directly to a shot.",
    "1/3": "Completed passes entering the final third.",
    "Passes into penalty area": "Completed passes into the penalty area, subject to source definition.",
    "touches_def_pen": "Touches in the defensive penalty area.",
    "Take ons attempted": "Take-ons/dribbles attempted.",
    "% Successful take-ons": "Percentage of attempted take-ons completed successfully.",
    "Times tackled during take-on": "Times dispossessed by a tackle during a take-on attempt.",
    "carries_prgc": "Progressive carries; appears duplicated by Progressive Carries in these files.",
    "carries final 3rd": "Carries entering the final third.",
    "carries penalty area": "Carries entering the penalty area.",
    "Possessions lost": "Times possession was lost or miscontrolled, subject to source definition.",
    "Goals Scored": "Goals scored; appears duplicated by Goals in these files.",
    "Total Shots": "Total shots attempted.",
    "% Shots on target": "Percentage of shots on target.",
    "Shots p 90": "Shots attempted per 90 minutes.",
    "Goals per shot": "Goals per shot.",
    "Goals per shot on target": "Goals per shot on target.",
    "% Aerial Duels won": "Percentage of aerial duels won.",
    "Shot creating actions p 90": "Shot-creating actions per 90 minutes.",
    "Goal creating actions p 90": "Goal-creating actions per 90 minutes.",
    "Crosses Stopped": "Goalkeeper crosses stopped; zero-filled for many non-goalkeepers.",
    "season": "Season label in YYYY-YYYY form.",
}


COMMON_DEFINITIONS = {
    "appearance_id": "Composite identifier for a player appearance, typically game_id_player_id.",
    "game_id": "Transfermarkt game identifier.",
    "player_id": "Transfermarkt player identifier.",
    "club_id": "Transfermarkt club/team identifier.",
    "player_club_id": "Club represented by the player in this appearance.",
    "player_current_club_id": "Player's current club in the latest source snapshot; not historical at appearance date.",
    "date": "Observation or event date.",
    "player_name": "Player display name retained for readability; player_id is preferred for joins.",
    "competition_id": "Transfermarkt competition identifier.",
    "yellow_cards": "Yellow cards recorded in the appearance.",
    "red_cards": "Red cards recorded in the appearance.",
    "goals": "Goals recorded.",
    "assists": "Assists recorded.",
    "minutes_played": "Minutes played in the appearance.",
    "own_goals": "Goals scored by the focal club in the game.",
    "own_position": "Focal club's league position associated with the game, when available.",
    "own_manager_name": "Focal club manager name for the game.",
    "opponent_id": "Opponent club/team identifier.",
    "opponent_goals": "Goals scored by the opponent.",
    "opponent_position": "Opponent league position associated with the game, when available.",
    "opponent_manager_name": "Opponent manager name for the game.",
    "hosting": "Whether the focal club was Home or Away.",
    "is_win": "Binary indicator that the focal club won the game.",
    "club_code": "URL-friendly club code/slug.",
    "name": "Entity display name.",
    "domestic_competition_id": "Identifier of the club's domestic league competition.",
    "total_market_value": "Displayed aggregate market value; may be text-formatted or missing in snapshot tables.",
    "squad_size": "Number of players in the current/latest squad snapshot.",
    "average_age": "Average age in the current/latest squad snapshot.",
    "foreigners_number": "Number of foreign players in the current/latest squad snapshot.",
    "foreigners_percentage": "Foreign players as a percentage of the current/latest squad.",
    "national_team_players": "Number of current/latest squad members classified as national-team players.",
    "stadium_name": "Club stadium name.",
    "stadium_seats": "Stadium seating capacity.",
    "net_transfer_record": "Displayed net transfer balance, stored as formatted text in the source.",
    "coach_name": "Coach/manager name in the current/latest snapshot.",
    "last_season": "Latest season represented for the entity in the scrape.",
    "filename": "Upstream source filename retained for provenance.",
    "url": "Transfermarkt source URL.",
    "competition_code": "URL-friendly competition code/slug.",
    "sub_type": "Detailed competition subtype.",
    "type": "Record or competition type; interpretation depends on table.",
    "country_id": "Transfermarkt country identifier.",
    "country_name": "Country display name.",
    "domestic_league_code": "Domestic league identifier associated with a competition.",
    "confederation": "Football confederation/region.",
    "total_clubs": "Number of clubs represented in the snapshot.",
    "country_code": "Transfermarkt country/competition code.",
    "total_players": "Number of players represented in the snapshot.",
    "game_event_id": "Unique identifier for a recorded game event.",
    "minute": "Match minute of the event.",
    "club_name": "Club/team display name at the event.",
    "description": "Free-text event description.",
    "player_in_id": "Incoming player identifier for substitution events.",
    "player_assist_id": "Assisting player identifier for goal events, when available.",
    "game_lineups_id": "Unique identifier for a game lineup record.",
    "position": "Broad playing position or competition type, depending on table context.",
    "number": "Squad/shirt number listed for the game.",
    "team_captain": "Binary indicator that the player was listed as captain.",
    "season": "Competition season start year or season label, depending on source table.",
    "round": "Competition round or matchweek label.",
    "home_club_id": "Home club/team identifier.",
    "away_club_id": "Away club/team identifier.",
    "home_club_goals": "Home team goals.",
    "away_club_goals": "Away team goals.",
    "home_club_position": "Home club league position associated with the game, when available.",
    "away_club_position": "Away club league position associated with the game, when available.",
    "home_club_manager_name": "Home team manager name for the game.",
    "away_club_manager_name": "Away team manager name for the game.",
    "stadium": "Game stadium.",
    "attendance": "Reported attendance.",
    "referee": "Match referee name.",
    "home_club_formation": "Reported home formation.",
    "away_club_formation": "Reported away formation.",
    "home_club_name": "Home club/team display name.",
    "away_club_name": "Away club/team display name.",
    "aggregate": "Displayed aggregate or final score string.",
    "competition_type": "Competition class such as domestic league, cup, or national-team competition.",
    "national_team_id": "Transfermarkt national-team identifier.",
    "team_code": "URL-friendly national-team code/slug.",
    "team_image_url": "National-team image URL.",
    "fifa_ranking": "FIFA ranking in the latest snapshot.",
    "market_value_in_eur": "Market value in euros.",
    "current_club_name": "Club name associated with the valuation/source snapshot.",
    "current_club_id": "Club identifier associated with the valuation or latest player snapshot.",
    "player_club_domestic_competition_id": "Domestic competition identifier associated with the player's club.",
    "first_name": "Player first name.",
    "last_name": "Player last name.",
    "player_code": "URL-friendly player code/slug.",
    "country_of_birth": "Player country of birth.",
    "city_of_birth": "Player city of birth.",
    "country_of_citizenship": "Player citizenship/nationality in the latest profile.",
    "date_of_birth": "Player date of birth.",
    "sub_position": "Detailed playing position.",
    "foot": "Preferred foot.",
    "height_in_cm": "Player height in centimeters.",
    "contract_expiration_date": "Contract expiration date in the latest profile snapshot; potentially leaky historically.",
    "agent_name": "Player agent name in the latest profile snapshot.",
    "image_url": "Player image URL.",
    "international_caps": "International caps in the latest profile snapshot; potentially leaky historically.",
    "international_goals": "International goals in the latest profile snapshot; potentially leaky historically.",
    "current_national_team_id": "Current national-team identifier in the latest profile snapshot.",
    "current_club_domestic_competition_id": "Domestic competition of the player's current club in the latest snapshot.",
    "current_club_name": "Player's current club name in the latest snapshot.",
    "highest_market_value_in_eur": "Highest recorded market value in euros across the player's history; leaks future information in historical backtests.",
    "transfer_date": "Effective date of the recorded player movement.",
    "transfer_season": "Transfer season label in YY/YY form.",
    "from_club_id": "Origin club identifier.",
    "to_club_id": "Destination club identifier.",
    "from_club_name": "Origin club display name.",
    "to_club_name": "Destination club display name.",
    "transfer_fee": "Numeric transfer fee field. Zero requires interpretation because it can represent several movement types or missing disclosure.",
}


DATE_NAME_RE = re.compile(r"(^date$|_date$|date_|birth|expiration)", re.I)
ID_NAME_RE = re.compile(r"(^|_)id$|_id$|^id$", re.I)


def table_name(path: Path) -> str:
    return "cleaned_season" if path.parent.name == "2017-2024" else path.stem


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.casefold().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", "", value)


def is_null_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.casefold().isin(NULL_TOKENS)


def column_definition(table: str, column: str) -> tuple[str, str]:
    if table == "cleaned_season" and column in SEASONAL_DEFINITIONS:
        return SEASONAL_DEFINITIONS[column], "Curated"
    if column in COMMON_DEFINITIONS:
        return COMMON_DEFINITIONS[column], "Curated"
    friendly = column.replace("_", " ").strip()
    return f"Source field: {friendly}. Confirm exact upstream definition before modeling.", "Inferred"


def model_role(table: str, column: str) -> tuple[str, str, str]:
    lower = column.casefold()
    if lower in {"url", "image_url", "team_image_url", "filename"}:
        return "Provenance", "Exclude", "Metadata only"
    if lower.endswith("_id") or lower in {"appearance_id", "game_lineups_id", "game_event_id"}:
        return "Join key", "Conditional", "Use for linkage, grouping, and split controls; not as a numeric predictor"
    if table == "transfers":
        if column in {"transfer_fee", "market_value_in_eur"}:
            return "Core transfer input", "Feature", "Available at transfer only after zero/missing semantics are resolved"
        if column in {"transfer_date", "transfer_season"}:
            return "Time anchor", "Feature", "Defines the information cutoff and validation split"
        return "Transfer identity/context", "Conditional", "Use as event context or linkage"
    if table == "player_valuations":
        if column == "market_value_in_eur":
            return "Financial measure", "Feature/target", "Feature only at dates on or before transfer; target only at a declared future horizon"
        if column == "date":
            return "Time anchor", "Feature", "Required for as-of joins"
    if table in {"appearances", "game_events", "game_lineups", "games", "club_games"}:
        if column == "date":
            return "Time anchor", "Feature", "Enforce pre-transfer/post-transfer windows"
        return "Sporting measure/context", "Feature/target", "Aggregate only within a declared time window"
    if table in {"players", "clubs", "national_teams"}:
        leaky_words = ("current", "market_value", "highest", "contract", "caps", "international_goals", "last_season", "coach", "squad", "average_age", "foreigners")
        if any(word in lower for word in leaky_words):
            return "Snapshot attribute", "High leakage risk", "Latest-snapshot value is not historically valid without a timestamp"
        return "Entity attribute", "Conditional", "Prefer stable biographical/descriptive attributes"
    if table == "cleaned_season":
        if column in {"player", "squad", "comp", "nation", "pos", "season"}:
            return "Identity/context", "Conditional", "Use for entity resolution, grouping, and time alignment"
        if column in {"rk"}:
            return "Source artifact", "Exclude", "Row rank is not a stable identifier"
        return "Seasonal performance", "Feature", "Use only when the full season predates the transfer cutoff"
    return "Reference attribute", "Conditional", "Use according to the linked entity and time context"


def choose_inferred_type(column: str, non_null: int, numeric_ok: int, integer_ok: int, date_ok: int) -> str:
    if non_null == 0:
        return "empty"
    if DATE_NAME_RE.search(column) and date_ok / non_null >= 0.98:
        return "date/datetime"
    if numeric_ok / non_null >= 0.995:
        if ID_NAME_RE.search(column):
            return "identifier (numeric-coded)"
        return "integer" if integer_ok / non_null >= 0.995 else "decimal"
    return "identifier (text)" if ID_NAME_RE.search(column) else "text/categorical"


def profile_file(path: Path) -> tuple[dict, list[dict], dict]:
    table = table_name(path)
    stats: dict[str, dict] = {}
    total_rows = 0
    key_hashes: list[np.ndarray] = []
    parse_error = ""
    duplicate_equal_pairs: set[tuple[str, str]] | None = None

    try:
        reader = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )
        for chunk_index, chunk in enumerate(reader):
            if chunk_index == 0:
                for order, col in enumerate(chunk.columns, start=1):
                    stats[col] = {
                        "order": order,
                        "missing": 0,
                        "non_null": 0,
                        "zero": 0,
                        "negative": 0,
                        "numeric_ok": 0,
                        "integer_ok": 0,
                        "date_ok": 0,
                        "numeric_min": None,
                        "numeric_max": None,
                        "date_min": None,
                        "date_max": None,
                        "min_length": None,
                        "max_length": 0,
                        "samples": [],
                        "distinct_values": set(),
                        "distinct_capped": False,
                        "mojibake_hits": 0,
                    }
                if table == "cleaned_season":
                    duplicate_equal_pairs = {
                        ("Goals", "Goals Scored"),
                        ("Progressive Carries", "carries_prgc"),
                    }

            total_rows += len(chunk)
            for col in chunk.columns:
                s = chunk[col].astype(str)
                null_mask = is_null_series(s)
                valid = s[~null_mask].str.strip()
                st = stats[col]
                st["missing"] += int(null_mask.sum())
                st["non_null"] += int((~null_mask).sum())
                if not valid.empty:
                    lengths = valid.str.len()
                    chunk_min_len = int(lengths.min())
                    chunk_max_len = int(lengths.max())
                    st["min_length"] = chunk_min_len if st["min_length"] is None else min(st["min_length"], chunk_min_len)
                    st["max_length"] = max(st["max_length"], chunk_max_len)
                    if len(st["samples"]) < 5:
                        for value in pd.unique(valid):
                            if value not in st["samples"]:
                                st["samples"].append(value)
                            if len(st["samples"]) == 5:
                                break
                    if not st["distinct_capped"]:
                        unique_values = pd.unique(valid)
                        room = DISTINCT_CAP + 1 - len(st["distinct_values"])
                        st["distinct_values"].update(unique_values[:room].tolist())
                        if len(st["distinct_values"]) > DISTINCT_CAP:
                            st["distinct_capped"] = True
                            st["distinct_values"] = set(list(st["distinct_values"])[:DISTINCT_CAP])

                    numeric = pd.to_numeric(valid, errors="coerce")
                    numeric_mask = numeric.notna()
                    numeric_count = int(numeric_mask.sum())
                    st["numeric_ok"] += numeric_count
                    if numeric_count:
                        numeric_valid = numeric[numeric_mask].astype(float)
                        st["integer_ok"] += int(np.isclose(numeric_valid % 1, 0, atol=1e-9).sum())
                        st["zero"] += int(np.isclose(numeric_valid, 0, atol=1e-12).sum())
                        st["negative"] += int((numeric_valid < 0).sum())
                        cmin = float(numeric_valid.min())
                        cmax = float(numeric_valid.max())
                        st["numeric_min"] = cmin if st["numeric_min"] is None else min(st["numeric_min"], cmin)
                        st["numeric_max"] = cmax if st["numeric_max"] is None else max(st["numeric_max"], cmax)

                    if DATE_NAME_RE.search(col):
                        parsed = pd.to_datetime(valid, errors="coerce", format="mixed")
                        date_mask = parsed.notna()
                        date_count = int(date_mask.sum())
                        st["date_ok"] += date_count
                        if date_count:
                            dmin = parsed[date_mask].min().isoformat()
                            dmax = parsed[date_mask].max().isoformat()
                            st["date_min"] = dmin if st["date_min"] is None else min(st["date_min"], dmin)
                            st["date_max"] = dmax if st["date_max"] is None else max(st["date_max"], dmax)

                    if chunk_index == 0:
                        # Flag common *sequences* created by double-decoding UTF-8,
                        # while allowing legitimate letters such as Ângelo and João.
                        st["mojibake_hits"] += int(
                            valid.str.contains(r"�|Ã[©¨ª«±³º¼]|Â[£€]|â[€™“”–—‚]", regex=True).sum()
                        )

            key_cols = PRIMARY_KEYS.get(table, [])
            if all(col in chunk.columns for col in key_cols):
                key_frame = chunk[key_cols].astype(str).apply(lambda x: x.str.strip())
                hashes = pd.util.hash_pandas_object(key_frame, index=False).to_numpy(dtype=np.uint64)
                key_hashes.append(hashes)

            if duplicate_equal_pairs:
                for pair in list(duplicate_equal_pairs):
                    if pair[0] not in chunk.columns or pair[1] not in chunk.columns:
                        duplicate_equal_pairs.discard(pair)
                    elif not chunk[pair[0]].astype(str).equals(chunk[pair[1]].astype(str)):
                        duplicate_equal_pairs.discard(pair)
    except Exception as exc:  # pragma: no cover - retained in audit output
        parse_error = f"{type(exc).__name__}: {exc}"

    columns = []
    for col, st in stats.items():
        non_null = st["non_null"]
        inferred_type = choose_inferred_type(col, non_null, st["numeric_ok"], st["integer_ok"], st["date_ok"])
        numeric_field = inferred_type in {"integer", "decimal", "identifier (numeric-coded)"}
        if not numeric_field:
            st["numeric_min"] = None
            st["numeric_max"] = None
            st["zero"] = 0
            st["negative"] = 0
        distinct_count = len(st["distinct_values"])
        distinct_display = f">={DISTINCT_CAP:,}" if st["distinct_capped"] else f"{distinct_count:,}"
        definition, definition_status = column_definition(table, col)
        role, recommendation, timing = model_role(table, col)
        note_parts = []
        if st["mojibake_hits"]:
            note_parts.append("Possible encoding artifacts detected in initial chunk")
        if table == "cleaned_season" and col == "Avg Mins per Match":
            note_parts.append("Suspected mislabeled total-minutes field")
        if table == "cleaned_season" and col in {"Goals", "Goals Scored", "Progressive Carries", "carries_prgc"}:
            note_parts.append("Potential duplicate feature")
        if table == "cleaned_season" and col in {"Goals Against", "Goals against p 90", "Saves", "Saves %", "Clean Sheets", "% Clean sheets", "% Penalty saves", "Crosses Stopped"}:
            note_parts.append("Position-specific goalkeeper metric; zeros for outfield players should be treated as structural missingness")
        columns.append(
            {
                "source_group": "Seasonal 2017-2024" if table == "cleaned_season" else "Transfermarkt",
                "table": path.stem,
                "logical_table": table,
                "file": str(path.relative_to(ROOT)).replace("\\", "/"),
                "column_order": st["order"],
                "column": col,
                "definition": definition,
                "definition_status": definition_status,
                "inferred_type": inferred_type,
                "modeling_role": role,
                "recommendation": recommendation,
                "timing_rule": timing,
                "non_null_count": non_null,
                "missing_count": st["missing"],
                "missing_pct": (st["missing"] / total_rows) if total_rows else None,
                "zero_count": st["zero"],
                "zero_pct_of_non_null": (st["zero"] / non_null) if non_null else None,
                "negative_count": st["negative"],
                "distinct_count": None if st["distinct_capped"] else distinct_count,
                "distinct_display": distinct_display,
                "distinct_capped": st["distinct_capped"],
                "unique_pct": (distinct_count / non_null) if non_null and not st["distinct_capped"] else None,
                "numeric_min": st["numeric_min"],
                "numeric_max": st["numeric_max"],
                "date_min": st["date_min"],
                "date_max": st["date_max"],
                "min_text_length": st["min_length"],
                "max_text_length": st["max_length"],
                "sample_values": " | ".join(str(v)[:120] for v in st["samples"]),
                "notes": "; ".join(note_parts),
            }
        )

    key_cols = PRIMARY_KEYS.get(table, [])
    key_summary = {
        "table": path.stem,
        "logical_table": table,
        "candidate_key": " + ".join(key_cols),
        "row_count": total_rows,
        "distinct_key_count": None,
        "duplicate_key_rows": None,
        "key_unique": None,
    }
    if key_hashes:
        all_hashes = np.concatenate(key_hashes)
        distinct_keys = int(np.unique(all_hashes).size)
        key_summary.update(
            {
                "distinct_key_count": distinct_keys,
                "duplicate_key_rows": total_rows - distinct_keys,
                "key_unique": distinct_keys == total_rows,
            }
        )

    date_columns = [c for c in columns if c["date_min"]]
    overview = {
        "source_group": "Seasonal 2017-2024" if table == "cleaned_season" else "Transfermarkt",
        "table": path.stem,
        "logical_table": table,
        "file": str(path.relative_to(ROOT)).replace("\\", "/"),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
        "row_count": total_rows,
        "column_count": len(stats),
        "grain": TABLE_GRAINS[table],
        "candidate_key": " + ".join(key_cols),
        "duplicate_key_rows": key_summary["duplicate_key_rows"],
        "key_unique": key_summary["key_unique"],
        "date_coverage": "; ".join(f"{c['column']}: {c['date_min'][:10]} to {c['date_max'][:10]}" for c in date_columns),
        "parse_status": "OK" if not parse_error else "ERROR",
        "parse_error": parse_error,
        "modeling_notes": TABLE_NOTES[table],
        "duplicate_equal_pairs": "; ".join(" = ".join(pair) for pair in sorted(duplicate_equal_pairs or set())),
    }
    return overview, columns, key_summary


def load_parent_sets() -> dict[str, set[str]]:
    def values(path: Path, column: str) -> set[str]:
        result: set[str] = set()
        for chunk in pd.read_csv(path, usecols=[column], dtype=str, keep_default_na=False, encoding="utf-8-sig", chunksize=CHUNK_SIZE):
            result.update(v.strip() for v in chunk[column].astype(str) if v.strip())
        return result

    return {
        "players.player_id": values(DATA_DIR / "TransferMarkt" / "players.csv", "player_id"),
        "clubs.club_id": values(DATA_DIR / "TransferMarkt" / "clubs.csv", "club_id"),
        "competitions.competition_id": values(DATA_DIR / "TransferMarkt" / "competitions.csv", "competition_id"),
        "games.game_id": values(DATA_DIR / "TransferMarkt" / "games.csv", "game_id"),
        "countries.country_id": values(DATA_DIR / "TransferMarkt" / "countries.csv", "country_id"),
        "national_teams.national_team_id": values(DATA_DIR / "TransferMarkt" / "national_teams.csv", "national_team_id"),
    }


FOREIGN_KEYS = [
    ("appearances", "game_id", "games.game_id"),
    ("appearances", "player_id", "players.player_id"),
    ("appearances", "player_club_id", "clubs.club_id"),
    ("appearances", "player_current_club_id", "clubs.club_id"),
    ("appearances", "competition_id", "competitions.competition_id"),
    ("club_games", "game_id", "games.game_id"),
    ("club_games", "club_id", "clubs.club_id"),
    ("club_games", "opponent_id", "clubs.club_id"),
    ("clubs", "domestic_competition_id", "competitions.competition_id"),
    ("competitions", "country_id", "countries.country_id"),
    ("game_events", "game_id", "games.game_id"),
    ("game_events", "club_id", "clubs.club_id"),
    ("game_events", "player_id", "players.player_id"),
    ("game_events", "player_in_id", "players.player_id"),
    ("game_events", "player_assist_id", "players.player_id"),
    ("game_lineups", "game_id", "games.game_id"),
    ("game_lineups", "player_id", "players.player_id"),
    ("game_lineups", "club_id", "clubs.club_id"),
    ("games", "competition_id", "competitions.competition_id"),
    ("games", "home_club_id", "clubs.club_id"),
    ("games", "away_club_id", "clubs.club_id"),
    ("national_teams", "country_id", "countries.country_id"),
    ("player_valuations", "player_id", "players.player_id"),
    ("player_valuations", "current_club_id", "clubs.club_id"),
    ("player_valuations", "player_club_domestic_competition_id", "competitions.competition_id"),
    ("players", "current_club_id", "clubs.club_id"),
    ("players", "current_national_team_id", "national_teams.national_team_id"),
    ("players", "current_club_domestic_competition_id", "competitions.competition_id"),
    ("transfers", "player_id", "players.player_id"),
    ("transfers", "from_club_id", "clubs.club_id"),
    ("transfers", "to_club_id", "clubs.club_id"),
]


def profile_foreign_keys(parent_sets: dict[str, set[str]]) -> list[dict]:
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for table, column, parent in FOREIGN_KEYS:
        grouped[table].append((column, parent))
    results: list[dict] = []
    for table, relationships in grouped.items():
        path = DATA_DIR / "TransferMarkt" / f"{table}.csv"
        usecols = sorted({column for column, _ in relationships})
        counters = {column: {"non_null": 0, "matched": 0, "unmatched_values": set()} for column, _ in relationships}
        for chunk in pd.read_csv(path, usecols=usecols, dtype=str, keep_default_na=False, encoding="utf-8-sig", chunksize=CHUNK_SIZE):
            for column, parent in relationships:
                values = chunk[column].astype(str).str.strip()
                valid = values[~values.str.casefold().isin(NULL_TOKENS)]
                matched = valid.isin(parent_sets[parent])
                counters[column]["non_null"] += len(valid)
                counters[column]["matched"] += int(matched.sum())
                if len(counters[column]["unmatched_values"]) < 10:
                    counters[column]["unmatched_values"].update(pd.unique(valid[~matched])[:10].tolist())
        for column, parent in relationships:
            c = counters[column]
            results.append(
                {
                    "child_table": table,
                    "child_column": column,
                    "parent_key": parent,
                    "non_null_rows": c["non_null"],
                    "matched_rows": c["matched"],
                    "unmatched_rows": c["non_null"] - c["matched"],
                    "row_match_pct": c["matched"] / c["non_null"] if c["non_null"] else None,
                    "unmatched_examples": " | ".join(sorted(c["unmatched_values"])[:10]),
                    "join_guidance": "Direct ID join" if c["matched"] == c["non_null"] else "Investigate coverage/scope before enforcing referential integrity",
                }
            )
    return results


def profile_seasonal_matching() -> list[dict]:
    players_path = DATA_DIR / "TransferMarkt" / "players.csv"
    players = pd.read_csv(players_path, usecols=["player_id", "name", "date_of_birth"], dtype=str, keep_default_na=False, encoding="utf-8-sig")
    players["name_norm"] = players["name"].map(normalize_text)
    players["birth_year"] = players["date_of_birth"].str.slice(0, 4)
    name_counts = players.groupby("name_norm")["player_id"].nunique()
    name_birth_counts = players.groupby(["name_norm", "birth_year"])["player_id"].nunique()

    clubs = pd.read_csv(DATA_DIR / "TransferMarkt" / "clubs.csv", usecols=["club_id", "name"], dtype=str, keep_default_na=False, encoding="utf-8-sig")
    clubs["name_norm"] = clubs["name"].map(normalize_text)
    club_counts = clubs.groupby("name_norm")["club_id"].nunique()

    results = []
    for path in sorted((DATA_DIR / "2017-2024").glob("*.csv")):
        df = pd.read_csv(path, usecols=["player", "born", "squad", "season"], dtype=str, keep_default_na=False, encoding="utf-8-sig")
        player_norm = df["player"].map(normalize_text)
        squad_norm = df["squad"].map(normalize_text)
        name_matches = player_norm.map(name_counts).fillna(0)
        name_birth_index = pd.MultiIndex.from_arrays([player_norm, df["born"].astype(str).str.strip()])
        name_birth_matches = pd.Series(name_birth_counts.reindex(name_birth_index).fillna(0).to_numpy(), index=df.index)
        club_matches = squad_norm.map(club_counts).fillna(0)
        results.append(
            {
                "table": path.stem,
                "season": df["season"].mode().iat[0] if not df.empty else "",
                "rows": len(df),
                "unique_player_name_match_rows": int((name_matches == 1).sum()),
                "unique_player_name_match_pct": float((name_matches == 1).mean()),
                "ambiguous_player_name_rows": int((name_matches > 1).sum()),
                "unmatched_player_name_rows": int((name_matches == 0).sum()),
                "unique_player_name_birth_match_rows": int((name_birth_matches == 1).sum()),
                "unique_player_name_birth_match_pct": float((name_birth_matches == 1).mean()),
                "ambiguous_player_name_birth_rows": int((name_birth_matches > 1).sum()),
                "unmatched_player_name_birth_rows": int((name_birth_matches == 0).sum()),
                "unique_exact_club_name_match_rows": int((club_matches == 1).sum()),
                "unique_exact_club_name_match_pct": float((club_matches == 1).mean()),
                "unmatched_exact_club_name_rows": int((club_matches == 0).sum()),
                "guidance": "Player name + birth year is the safer deterministic bridge; club aliases/fuzzy review are still required.",
            }
        )
    return results


def profile_seasonal_consistency() -> tuple[list[dict], list[dict]]:
    paths = sorted((DATA_DIR / "2017-2024").glob("*.csv"))
    canonical = list(pd.read_csv(paths[0], nrows=0, encoding="utf-8-sig").columns)
    schema_rows = []
    qa_rows = []
    for path in paths:
        df = pd.read_csv(path, encoding="utf-8-sig")
        cols = list(df.columns)
        minutes = pd.to_numeric(df["Avg Mins per Match"], errors="coerce")
        matches = pd.to_numeric(df["Matches Played"], errors="coerce")
        goals = pd.to_numeric(df["Goals"], errors="coerce")
        assists = pd.to_numeric(df["Assists"], errors="coerce")
        ga = pd.to_numeric(df["Goals & Assists"], errors="coerce")
        npen = pd.to_numeric(df["Non Penalty Goals"], errors="coerce")
        pens = pd.to_numeric(df["Penalty Kicks Made"], errors="coerce")
        avg_if_label_true = minutes / matches.replace(0, np.nan)
        schema_rows.append(
            {
                "table": path.stem,
                "rows": len(df),
                "columns": len(cols),
                "schema_matches_first_file": cols == canonical,
                "missing_vs_canonical": " | ".join(c for c in canonical if c not in cols),
                "extra_vs_canonical": " | ".join(c for c in cols if c not in canonical),
                "season_values": " | ".join(sorted(df["season"].astype(str).unique())),
            }
        )
        qa_rows.extend(
            [
                {
                    "table": path.stem,
                    "check": "Goals & Assists = Goals + Assists",
                    "pass_rows": int(np.isclose(ga, goals + assists, equal_nan=False).sum()),
                    "eligible_rows": int((ga.notna() & goals.notna() & assists.notna()).sum()),
                },
                {
                    "table": path.stem,
                    "check": "Non Penalty Goals = Goals - Penalty Kicks Made",
                    "pass_rows": int(np.isclose(npen, goals - pens, equal_nan=False).sum()),
                    "eligible_rows": int((npen.notna() & goals.notna() & pens.notna()).sum()),
                },
                {
                    "table": path.stem,
                    "check": "Minutes field / matches <= 120 (supports total-minutes interpretation)",
                    "pass_rows": int((avg_if_label_true <= 120).sum()),
                    "eligible_rows": int(avg_if_label_true.notna().sum()),
                },
                {
                    "table": path.stem,
                    "check": "Goals equals Goals Scored",
                    "pass_rows": int((df["Goals"].astype(str) == df["Goals Scored"].astype(str)).sum()),
                    "eligible_rows": len(df),
                },
                {
                    "table": path.stem,
                    "check": "Progressive Carries equals carries_prgc",
                    "pass_rows": int((df["Progressive Carries"].astype(str) == df["carries_prgc"].astype(str)).sum()),
                    "eligible_rows": len(df),
                },
            ]
        )
    for row in qa_rows:
        row["pass_pct"] = row["pass_rows"] / row["eligible_rows"] if row["eligible_rows"] else None
    return schema_rows, qa_rows


def write_csv(name: str, records: list[dict]) -> None:
    pd.DataFrame(records).to_csv(OUT_DIR / name, index=False, encoding="utf-8-sig")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_paths = sorted(DATA_DIR.rglob("*.csv"), key=lambda p: str(p).casefold())
    overview: list[dict] = []
    columns: list[dict] = []
    keys: list[dict] = []
    for path in csv_paths:
        print(f"Profiling {path.relative_to(ROOT)}", flush=True)
        o, c, k = profile_file(path)
        overview.append(o)
        columns.extend(c)
        keys.append(k)

    print("Checking foreign-key coverage", flush=True)
    joins = profile_foreign_keys(load_parent_sets())
    print("Checking seasonal-to-Transfermarkt identity bridges", flush=True)
    seasonal_matching = profile_seasonal_matching()
    seasonal_schema, seasonal_qa = profile_seasonal_consistency()

    write_csv("table_inventory.csv", overview)
    write_csv("column_dictionary.csv", columns)
    write_csv("candidate_keys.csv", keys)
    write_csv("join_coverage.csv", joins)
    write_csv("seasonal_matching.csv", seasonal_matching)
    write_csv("seasonal_schema.csv", seasonal_schema)
    write_csv("seasonal_quality_checks.csv", seasonal_qa)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(DATA_DIR),
        "profile_settings": {"chunk_size": CHUNK_SIZE, "distinct_cap": DISTINCT_CAP},
        "overview": overview,
        "columns": columns,
        "keys": keys,
        "joins": joins,
        "seasonal_matching": seasonal_matching,
        "seasonal_schema": seasonal_schema,
        "seasonal_qa": seasonal_qa,
    }
    (OUT_DIR / "audit_profile.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Audit artifacts written to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

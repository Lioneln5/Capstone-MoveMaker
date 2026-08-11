"""Independent verification for the comprehensive Transfermarkt master."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "Data" / "processed" / "transfermarkt_clean" / "tables"
MERGED = ROOT / "Data" / "processed" / "transfermarkt_merged"
MASTER_PATH = MERGED / "transfermarkt_comprehensive_master.csv"
CORE_PATH = MERGED / "transfer_core.csv"
COMPONENT_PATHS = {
    "valuation": MERGED / "transfer_valuation_features.csv",
    "appearance": MERGED / "transfer_appearance_features.csv",
    "lineup": MERGED / "transfer_lineup_features.csv",
    "event": MERGED / "transfer_event_features.csv",
    "club_context": MERGED / "transfer_club_context.csv",
}
CSV_OUTPUT = MERGED / "transfermarkt_comprehensive_independent_verification.csv"
JSON_OUTPUT = MERGED / "transfermarkt_comprehensive_independent_verification.json"


def clean_id(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().str.replace(r"\.0$", "", regex=True)


def read_filtered(path: Path, usecols: list[str], ids: set[str], id_columns: list[str]) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, dtype=str, keep_default_na=False, chunksize=250_000):
        mask = pd.Series(False, index=chunk.index)
        for column in id_columns:
            chunk[column] = clean_id(chunk[column])
            mask |= chunk[column].isin(ids)
        if mask.any():
            parts.append(chunk.loc[mask].copy())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=usecols)


def numeric_equal(actual: object, expected: float, tolerance: float = 1e-8) -> bool:
    if pd.isna(actual) and pd.isna(expected):
        return True
    try:
        return bool(np.isclose(float(actual), float(expected), atol=tolerance, rtol=tolerance, equal_nan=True))
    except (TypeError, ValueError):
        return False


def semantic_frame_mismatches(left: pd.DataFrame, right: pd.DataFrame) -> int:
    mismatches = 0
    for column in right.columns:
        left_values = left[column].astype("string").fillna("")
        right_values = right[column].astype("string").fillna("")
        if column.endswith("_id") or column == "transfer_event_id":
            mismatches += int(clean_id(left_values).ne(clean_id(right_values)).sum())
            continue
        left_numeric = pd.to_numeric(left_values.replace("", pd.NA), errors="coerce")
        right_numeric = pd.to_numeric(right_values.replace("", pd.NA), errors="coerce")
        nonblank = left_values.ne("") | right_values.ne("")
        numeric_column = bool((left_numeric[nonblank].notna() & right_numeric[nonblank].notna()).all()) if nonblank.any() else False
        if numeric_column:
            equal = np.isclose(left_numeric.astype(float), right_numeric.astype(float), atol=1e-10, rtol=1e-10, equal_nan=True)
            mismatches += int((~equal).sum())
        else:
            mismatches += int(left_values.ne(right_values).sum())
    return mismatches


def main() -> None:
    master = pd.read_csv(MASTER_PATH, dtype=str, keep_default_na=False, low_memory=False)
    core = pd.read_csv(CORE_PATH, dtype=str, keep_default_na=False, low_memory=False)
    results: list[dict[str, object]] = []

    def add(check: str, actual: object, expected: object, passed: bool, notes: str) -> None:
        results.append({"check": check, "actual": actual, "expected": expected, "status": "OK" if passed else "FAIL", "notes": notes})

    add("Master row count", len(master), len(core), len(master) == len(core), "Independent CSV reload.")
    add("Master key uniqueness", master["transfer_event_id"].nunique(), len(master), master["transfer_event_id"].is_unique, "One row per transfer event.")
    core_equal = master[list(core.columns)].equals(core)
    add("Transfer core byte-level values preserved", core_equal, True, core_equal, "String reload comparison for every core field.")

    for name, path in COMPONENT_PATHS.items():
        component = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
        mismatch_count = semantic_frame_mismatches(master[list(component.columns)], component)
        add(f"{name} component values preserved", mismatch_count, 0, mismatch_count == 0, "Semantic comparison tolerates equivalent ID formatting and floating-point CSV precision.")

    sample_indices = np.unique(np.linspace(0, len(master) - 1, 60, dtype=int))
    sample = master.iloc[sample_indices].copy()
    sample["transfer_date"] = pd.to_datetime(sample["transfer_date"], errors="raise")
    sample_ids = set(sample["player_id"])

    valuations = read_filtered(
        TABLES / "player_valuations_clean.csv",
        ["player_id", "date", "market_value_in_eur"],
        sample_ids,
        ["player_id"],
    )
    valuations["date"] = pd.to_datetime(valuations["date"], errors="coerce")
    valuations["market_value_in_eur"] = pd.to_numeric(valuations["market_value_in_eur"], errors="coerce")
    valuation_groups = {pid: group.dropna(subset=["date", "market_value_in_eur"]).sort_values("date") for pid, group in valuations.groupby("player_id")}
    valuation_mismatches = 0
    for row in sample.itertuples(index=False):
        group = valuation_groups.get(row.player_id)
        if group is None or group.empty:
            continue
        pre = group[group["date"].le(row.transfer_date)]
        expected_date = pre.iloc[-1]["date"] if not pre.empty else pd.NaT
        expected_value = pre.iloc[-1]["market_value_in_eur"] if not pre.empty else np.nan
        actual_date = pd.to_datetime(getattr(row, "valuation_pre_date"), errors="coerce")
        actual_value = pd.to_numeric(pd.Series([getattr(row, "valuation_pre_eur")]), errors="coerce").iloc[0]
        if not ((pd.isna(expected_date) and pd.isna(actual_date)) or expected_date == actual_date):
            valuation_mismatches += 1
        if not numeric_equal(actual_value, expected_value):
            valuation_mismatches += 1
        if not pre.empty:
            career_first = pre.iloc[0]
            if not numeric_equal(getattr(row, "valuation_career_pre_first_eur"), career_first["market_value_in_eur"]):
                valuation_mismatches += 1
            if not numeric_equal(getattr(row, "valuation_career_pre_observations"), len(pre)):
                valuation_mismatches += 1
            pre730 = pre[pre["date"].ge(row.transfer_date - pd.Timedelta(days=730))]
            if not numeric_equal(getattr(row, "valuation_pre730_observations"), len(pre730)):
                valuation_mismatches += 1
        for horizon, days in ((12, 365), (24, 730)):
            target = row.transfer_date + pd.Timedelta(days=days)
            nearest_index = (group["date"] - target).abs().idxmin()
            nearest = group.loc[nearest_index]
            actual_nearest_date = pd.to_datetime(getattr(row, f"valuation_post{horizon}_nearest_date"), errors="coerce")
            actual_nearest_value = pd.to_numeric(pd.Series([getattr(row, f"valuation_post{horizon}_nearest_eur")]), errors="coerce").iloc[0]
            if nearest["date"] != actual_nearest_date or not numeric_equal(actual_nearest_value, nearest["market_value_in_eur"]):
                valuation_mismatches += 1
    add("Sample valuation as-of and horizon selections", valuation_mismatches, 0, valuation_mismatches == 0, f"Recomputed {len(sample)} deterministic transfer samples from cleaned valuation rows.")

    appearances = read_filtered(
        TABLES / "appearances_clean.csv",
        ["game_id", "player_id", "player_club_id", "date", "minutes_played", "goals", "assists"],
        sample_ids,
        ["player_id"],
    )
    appearances["player_club_id"] = clean_id(appearances["player_club_id"])
    appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")
    for column in ["minutes_played", "goals", "assists"]:
        appearances[column] = pd.to_numeric(appearances[column], errors="coerce").fillna(0)
    appearance_mismatches = 0
    for row in sample.itertuples(index=False):
        player_rows = appearances[appearances["player_id"].eq(row.player_id)]
        windows = {
            "appearance_career_pre": player_rows["date"].le(row.transfer_date - pd.Timedelta(days=1)),
            "appearance_pre365_all": player_rows["date"].between(row.transfer_date - pd.Timedelta(days=365), row.transfer_date - pd.Timedelta(days=1)),
            "appearance_pre730_all": player_rows["date"].between(row.transfer_date - pd.Timedelta(days=730), row.transfer_date - pd.Timedelta(days=1)),
            "appearance_post12_destination": player_rows["date"].between(row.transfer_date, row.transfer_date + pd.Timedelta(days=364)) & player_rows["player_club_id"].eq(row.to_club_id),
            "appearance_post24_destination": player_rows["date"].between(row.transfer_date, row.transfer_date + pd.Timedelta(days=729)) & player_rows["player_club_id"].eq(row.to_club_id),
        }
        for prefix, mask in windows.items():
            window = player_rows.loc[mask]
            expected = {
                "games": window["game_id"].nunique(),
                "minutes": window["minutes_played"].sum(),
                "goals": window["goals"].sum(),
                "assists": window["assists"].sum(),
            }
            for metric, value in expected.items():
                if not numeric_equal(getattr(row, f"{prefix}_{metric}"), value):
                    appearance_mismatches += 1
    add("Sample appearance aggregates", appearance_mismatches, 0, appearance_mismatches == 0, "Recomputed pre-365 and destination 12/24-month metrics.")

    lineups = read_filtered(
        TABLES / "game_lineups_clean.csv",
        ["game_id", "player_id", "club_id", "date", "is_starting_lineup", "is_substitute"],
        sample_ids,
        ["player_id"],
    )
    lineups["club_id"] = clean_id(lineups["club_id"])
    lineups["date"] = pd.to_datetime(lineups["date"], errors="coerce")
    lineups["is_starting_lineup"] = lineups["is_starting_lineup"].astype(str).str.casefold().eq("true")
    lineup_mismatches = 0
    for row in sample.itertuples(index=False):
        player_rows = lineups[lineups["player_id"].eq(row.player_id)]
        windows = {
            "lineup_career_pre": player_rows["date"].le(row.transfer_date - pd.Timedelta(days=1)),
            "lineup_pre365_all": player_rows["date"].between(row.transfer_date - pd.Timedelta(days=365), row.transfer_date - pd.Timedelta(days=1)),
            "lineup_pre730_all": player_rows["date"].between(row.transfer_date - pd.Timedelta(days=730), row.transfer_date - pd.Timedelta(days=1)),
            "lineup_post12_destination": player_rows["date"].between(row.transfer_date, row.transfer_date + pd.Timedelta(days=364)) & player_rows["club_id"].eq(row.to_club_id),
        }
        for prefix, mask in windows.items():
            window = player_rows.loc[mask]
            if not numeric_equal(getattr(row, f"{prefix}_games"), window["game_id"].nunique()):
                lineup_mismatches += 1
            if not numeric_equal(getattr(row, f"{prefix}_starts"), window["is_starting_lineup"].sum()):
                lineup_mismatches += 1
    add("Sample lineup aggregates", lineup_mismatches, 0, lineup_mismatches == 0, "Recomputed pre-transfer and destination-selection metrics.")

    events = read_filtered(
        TABLES / "game_events_clean.csv",
        ["player_id", "player_assist_id", "date", "club_id", "event_type_normalized"],
        sample_ids,
        ["player_id", "player_assist_id"],
    )
    events["club_id"] = clean_id(events["club_id"])
    events["date"] = pd.to_datetime(events["date"], errors="coerce")
    event_mismatches = 0
    for row in sample.itertuples(index=False):
        pre_mask = events["date"].between(row.transfer_date - pd.Timedelta(days=365), row.transfer_date - pd.Timedelta(days=1))
        pre730_mask = events["date"].between(row.transfer_date - pd.Timedelta(days=730), row.transfer_date - pd.Timedelta(days=1))
        career_mask = events["date"].le(row.transfer_date - pd.Timedelta(days=1))
        post_mask = events["date"].between(row.transfer_date, row.transfer_date + pd.Timedelta(days=364)) & events["club_id"].eq(row.to_club_id)
        expected_pre_goals = int((pre_mask & events["player_id"].eq(row.player_id) & events["event_type_normalized"].eq("goals")).sum())
        expected_pre_assists = int((pre_mask & events["player_assist_id"].eq(row.player_id)).sum())
        expected_post_goals = int((post_mask & events["player_id"].eq(row.player_id) & events["event_type_normalized"].eq("goals")).sum())
        expected_pre730_goals = int((pre730_mask & events["player_id"].eq(row.player_id) & events["event_type_normalized"].eq("goals")).sum())
        expected_career_goals = int((career_mask & events["player_id"].eq(row.player_id) & events["event_type_normalized"].eq("goals")).sum())
        if not numeric_equal(getattr(row, "event_pre365_all_goal_events"), expected_pre_goals):
            event_mismatches += 1
        if not numeric_equal(getattr(row, "event_pre365_all_assist_events"), expected_pre_assists):
            event_mismatches += 1
        if not numeric_equal(getattr(row, "event_post12_destination_goal_events"), expected_post_goals):
            event_mismatches += 1
        if not numeric_equal(getattr(row, "event_pre730_all_goal_events"), expected_pre730_goals):
            event_mismatches += 1
        if not numeric_equal(getattr(row, "event_career_pre_goal_events"), expected_career_goals):
            event_mismatches += 1
    add("Sample event aggregates", event_mismatches, 0, event_mismatches == 0, "Recomputed goal and assist event counts.")

    games = pd.read_csv(TABLES / "games_clean.csv", usecols=["game_id", "date"], dtype=str, keep_default_na=False)
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    club_ids = set(sample["from_club_id"]) | set(sample["to_club_id"])
    club_games = read_filtered(
        TABLES / "club_games_clean.csv",
        ["game_id", "club_id", "own_goals", "opponent_goals", "is_win"],
        club_ids,
        ["club_id"],
    )
    club_games = club_games.merge(games, on="game_id", how="left", validate="many_to_one")
    for column in ["own_goals", "opponent_goals"]:
        club_games[column] = pd.to_numeric(club_games[column], errors="coerce")
    club_games["is_win"] = club_games["is_win"].astype(str).str.casefold().eq("true")
    club_mismatches = 0
    for row in sample.itertuples(index=False):
        windows = {
            "club_origin_pre365": (row.from_club_id, row.transfer_date - pd.Timedelta(days=365), row.transfer_date - pd.Timedelta(days=1)),
            "club_destination_pre365": (row.to_club_id, row.transfer_date - pd.Timedelta(days=365), row.transfer_date - pd.Timedelta(days=1)),
            "club_destination_post12": (row.to_club_id, row.transfer_date, row.transfer_date + pd.Timedelta(days=364)),
        }
        for prefix, (club_id, start, end) in windows.items():
            window = club_games[club_games["club_id"].eq(club_id) & club_games["date"].between(start, end)]
            expected = {
                "games": window["game_id"].nunique(),
                "wins": window["is_win"].sum(),
                "goals_for": window["own_goals"].sum(),
                "goals_against": window["opponent_goals"].sum(),
            }
            for metric, value in expected.items():
                if not numeric_equal(getattr(row, f"{prefix}_{metric}"), value):
                    club_mismatches += 1
    add("Sample club-context aggregates", club_mismatches, 0, club_mismatches == 0, "Recomputed origin/destination dated game windows.")

    frame = pd.DataFrame(results)
    frame.to_csv(CSV_OUTPUT, index=False, encoding="utf-8", lineterminator="\n")
    summary = {
        "checks": len(frame),
        "passed": int(frame["status"].eq("OK").sum()),
        "failed": int(frame["status"].eq("FAIL").sum()),
        "sample_transfer_events": len(sample),
    }
    JSON_OUTPUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if summary["failed"]:
        raise RuntimeError("Independent verification failures:\n" + frame[frame["status"].eq("FAIL")].to_string(index=False))
    print(f"Independent verification passed all {summary['passed']} checks across {len(sample)} sampled transfers")


if __name__ == "__main__":
    main()

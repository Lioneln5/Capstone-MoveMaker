"""Build a compact, audited StatsBomb case-study layer for MoveMaker.

The raw StatsBomb Open Data checkout is treated as immutable.  This extractor
keeps only cataloged male matches from 2017-07-01 through 2024-06-30, writes a
slim event stream for visual case studies, derives player and team features,
and creates review-only links to the existing Transfermarkt transfer cohort.

The selected range deliberately covers the MoveMaker feature/outcome window.
It is not represented as a complete league panel: StatsBomb Open Data is a
selected-match sample during most modern Big Five seasons.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
SOURCE_ROOT = DATA / "Open-Data-Master" / "open-data-master" / "open-data-master"
SOURCE = SOURCE_ROOT / "data"
PROCESSED = DATA / "processed" / "statsbomb_case_studies"
OUTPUT = ROOT / "outputs" / "statsbomb_case_studies"

START_DATE = date(2017, 7, 1)
END_DATE = date(2024, 6, 30)
BIG_FIVE = {"Premier League", "La Liga", "Serie A", "1. Bundesliga", "Ligue 1"}

EVENT_COLUMNS = [
    "match_id", "match_date", "competition_name", "season_name", "scope_tier",
    "event_id", "event_index", "period", "timestamp", "minute", "second",
    "possession", "event_type", "play_pattern", "team_id", "team_name",
    "possession_team_id", "possession_team_name", "player_id", "player_name",
    "position_id", "position_name", "x", "y", "end_x", "end_y", "duration_seconds",
    "under_pressure", "counterpress", "outcome_name", "recipient_id", "recipient_name",
    "body_part_name", "technique_name", "pass_type", "pass_height", "pass_length",
    "pass_angle", "pass_cross", "pass_switch", "pass_shot_assist", "pass_goal_assist",
    "assisted_shot_xg", "shot_type", "shot_statsbomb_xg", "goalkeeper_type",
    "duel_type", "source_file",
]

CONTEXT_360_COLUMNS = [
    "match_id", "event_id", "visible_area_polygon_area", "visible_players",
    "visible_teammates", "visible_opponents", "actor_observations", "actor_is_keeper",
    "nearest_teammate_distance", "nearest_opponent_distance", "teammates_within_10",
    "opponents_within_5", "opponents_within_10", "source_file",
]

COUNT_FIELDS = [
    "total_events", "events_under_pressure", "passes", "passes_completed",
    "forward_passes", "progressive_passes", "passes_into_final_third",
    "passes_into_penalty_area", "crosses", "switches", "shot_assists", "goal_assists",
    "carries", "progressive_carries", "carries_into_final_third",
    "carries_into_penalty_area", "shots", "shots_on_target", "goals",
    "pressures", "counterpressures", "ball_recoveries", "interceptions", "clearances",
    "blocks", "duels", "duels_won", "tackles", "tackles_won", "dribbles",
    "successful_dribbles", "dispossessed", "miscontrols", "fouls_committed",
    "fouls_won", "goalkeeper_events", "saves", "counterattack_events",
    "set_piece_events", "location_event_count",
]

SUM_FIELDS = [
    "pass_distance", "pass_forward_distance", "carry_distance", "carry_forward_distance",
    "shot_xg", "xg_assisted", "location_x_sum", "location_y_sum",
]


def ensure_inputs() -> None:
    required = [
        SOURCE / "competitions.json",
        SOURCE / "matches",
        SOURCE / "events",
        SOURCE / "lineups",
        ROOT / "Data" / "TransferMarkt" / "players.csv",
        DATA / "processed" / "transfer_cohort_master.csv",
        DATA / "processed" / "club_dimension_union.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def nested_name(value: Any) -> str | None:
    return value.get("name") if isinstance(value, dict) else None


def nested_id(value: Any) -> int | None:
    return value.get("id") if isinstance(value, dict) else None


def normalized_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    return "".join(character for character in text if character.isalnum())


def normalized_country(value: Any) -> str:
    text = normalized_text(value)
    aliases = {
        "unitedstatesofamerica": "unitedstates",
        "korearepublic": "southkorea",
        "republicofkorea": "southkorea",
        "cotedivoire": "ivorycoast",
        "congodemocraticrepublic": "drcongo",
        "democraticrepublicofthecongo": "drcongo",
    }
    return aliases.get(text, text)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")


def clock_minutes(value: Any) -> float | None:
    if value in (None, ""):
        return None
    parts = str(value).split(":")
    try:
        if len(parts) == 2:
            return float(parts[0]) + float(parts[1]) / 60.0
        if len(parts) == 3:
            return float(parts[0]) * 60.0 + float(parts[1]) + float(parts[2]) / 60.0
    except ValueError:
        return None
    return None


def union_minutes(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def polygon_area(flat_coordinates: list[float]) -> float | None:
    if len(flat_coordinates) < 6 or len(flat_coordinates) % 2:
        return None
    points = list(zip(flat_coordinates[0::2], flat_coordinates[1::2]))
    return abs(sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
    )) / 2.0


def empty_metrics() -> dict[str, float]:
    return {field: 0.0 for field in COUNT_FIELDS + SUM_FIELDS}


def update_metrics(target: dict[str, Any], event: dict[str, Any], assisted_xg: float) -> None:
    event_type = nested_name(event.get("type")) or "Unknown"
    location = event.get("location") or []
    x = float(location[0]) if len(location) >= 2 else None
    y = float(location[1]) if len(location) >= 2 else None
    target["total_events"] += 1
    if event.get("under_pressure"):
        target["events_under_pressure"] += 1
    if event.get("counterpress"):
        target["counterpressures"] += 1
    if nested_name(event.get("play_pattern")) == "From Counter":
        target["counterattack_events"] += 1
    if nested_name(event.get("play_pattern")) in {
        "From Corner", "From Free Kick", "From Throw In", "From Goal Kick", "From Kick Off",
    }:
        target["set_piece_events"] += 1
    if x is not None and y is not None:
        target["location_event_count"] += 1
        target["location_x_sum"] += x
        target["location_y_sum"] += y

    if event_type == "Pass":
        item = event.get("pass") or {}
        end = item.get("end_location") or []
        end_x = float(end[0]) if len(end) >= 2 else None
        end_y = float(end[1]) if len(end) >= 2 else None
        length = float(item.get("length") or 0.0)
        target["passes"] += 1
        target["pass_distance"] += length
        if not item.get("outcome"):
            target["passes_completed"] += 1
        if x is not None and end_x is not None:
            gain = end_x - x
            target["pass_forward_distance"] += max(gain, 0.0)
            target["forward_passes"] += gain > 1.0
            target["progressive_passes"] += gain >= 10.0
            target["passes_into_final_third"] += x < 80.0 <= end_x
            target["passes_into_penalty_area"] += (
                x < 102.0 and end_x >= 102.0 and end_y is not None and 18.0 <= end_y <= 62.0
            )
        target["crosses"] += bool(item.get("cross"))
        target["switches"] += bool(item.get("switch"))
        target["shot_assists"] += bool(item.get("shot_assist"))
        target["goal_assists"] += bool(item.get("goal_assist"))
        target["xg_assisted"] += assisted_xg
    elif event_type == "Carry":
        item = event.get("carry") or {}
        end = item.get("end_location") or []
        end_x = float(end[0]) if len(end) >= 2 else None
        end_y = float(end[1]) if len(end) >= 2 else None
        target["carries"] += 1
        if x is not None and y is not None and end_x is not None and end_y is not None:
            target["carry_distance"] += math.dist((x, y), (end_x, end_y))
            gain = end_x - x
            target["carry_forward_distance"] += max(gain, 0.0)
            target["progressive_carries"] += gain >= 10.0
            target["carries_into_final_third"] += x < 80.0 <= end_x
            target["carries_into_penalty_area"] += x < 102.0 <= end_x and 18.0 <= end_y <= 62.0
    elif event_type == "Shot":
        item = event.get("shot") or {}
        outcome = nested_name(item.get("outcome")) or ""
        target["shots"] += 1
        target["shot_xg"] += float(item.get("statsbomb_xg") or 0.0)
        target["shots_on_target"] += outcome in {"Goal", "Saved", "Saved To Post", "Saved Off Target"}
        target["goals"] += outcome == "Goal"
    elif event_type == "Pressure":
        target["pressures"] += 1
    elif event_type == "Ball Recovery":
        target["ball_recoveries"] += 1
    elif event_type == "Interception":
        target["interceptions"] += 1
    elif event_type == "Clearance":
        target["clearances"] += 1
    elif event_type == "Block":
        target["blocks"] += 1
    elif event_type == "Duel":
        item = event.get("duel") or {}
        outcome = nested_name(item.get("outcome")) or ""
        duel_type = nested_name(item.get("type")) or ""
        won = outcome in {"Won", "Success In Play", "Success Out"}
        target["duels"] += 1
        target["duels_won"] += won
        target["tackles"] += duel_type == "Tackle"
        target["tackles_won"] += duel_type == "Tackle" and won
    elif event_type == "Dribble":
        target["dribbles"] += 1
        target["successful_dribbles"] += nested_name((event.get("dribble") or {}).get("outcome")) == "Complete"
    elif event_type == "Dispossessed":
        target["dispossessed"] += 1
    elif event_type == "Miscontrol":
        target["miscontrols"] += 1
    elif event_type == "Foul Committed":
        target["fouls_committed"] += 1
    elif event_type == "Foul Won":
        target["fouls_won"] += 1
    elif event_type == "Goal Keeper":
        item = event.get("goalkeeper") or {}
        target["goalkeeper_events"] += 1
        keeper_type = nested_name(item.get("type")) or ""
        outcome = nested_name(item.get("outcome")) or ""
        target["saves"] += "Save" in keeper_type or "Saved" in outcome


def event_to_row(
    event: dict[str, Any], match: dict[str, Any], source_file: str, assisted_xg: float
) -> dict[str, Any]:
    event_type = nested_name(event.get("type")) or "Unknown"
    detail_key = {
        "Pass": "pass", "Shot": "shot", "Carry": "carry", "Duel": "duel",
        "Goal Keeper": "goalkeeper", "Dribble": "dribble", "Interception": "interception",
    }.get(event_type)
    detail = event.get(detail_key) if detail_key else {}
    detail = detail if isinstance(detail, dict) else {}
    location = event.get("location") or []
    end = detail.get("end_location") or []
    outcome = nested_name(detail.get("outcome"))
    pass_item = event.get("pass") or {}
    shot_item = event.get("shot") or {}
    keeper_item = event.get("goalkeeper") or {}
    duel_item = event.get("duel") or {}
    recipient = pass_item.get("recipient") or {}
    return {
        "match_id": match["match_id"], "match_date": match["match_date"],
        "competition_name": match["competition_name"], "season_name": match["season_name"],
        "scope_tier": match["scope_tier"], "event_id": event.get("id"),
        "event_index": event.get("index"), "period": event.get("period"),
        "timestamp": event.get("timestamp"), "minute": event.get("minute"),
        "second": event.get("second"), "possession": event.get("possession"),
        "event_type": event_type, "play_pattern": nested_name(event.get("play_pattern")),
        "team_id": nested_id(event.get("team")), "team_name": nested_name(event.get("team")),
        "possession_team_id": nested_id(event.get("possession_team")),
        "possession_team_name": nested_name(event.get("possession_team")),
        "player_id": nested_id(event.get("player")), "player_name": nested_name(event.get("player")),
        "position_id": nested_id(event.get("position")), "position_name": nested_name(event.get("position")),
        "x": location[0] if len(location) >= 2 else None,
        "y": location[1] if len(location) >= 2 else None,
        "end_x": end[0] if len(end) >= 2 else None,
        "end_y": end[1] if len(end) >= 2 else None,
        "duration_seconds": event.get("duration"), "under_pressure": bool(event.get("under_pressure")),
        "counterpress": bool(event.get("counterpress")), "outcome_name": outcome,
        "recipient_id": recipient.get("id"), "recipient_name": recipient.get("name"),
        "body_part_name": nested_name(detail.get("body_part")),
        "technique_name": nested_name(detail.get("technique")),
        "pass_type": nested_name(pass_item.get("type")), "pass_height": nested_name(pass_item.get("height")),
        "pass_length": pass_item.get("length"), "pass_angle": pass_item.get("angle"),
        "pass_cross": bool(pass_item.get("cross")), "pass_switch": bool(pass_item.get("switch")),
        "pass_shot_assist": bool(pass_item.get("shot_assist")),
        "pass_goal_assist": bool(pass_item.get("goal_assist")),
        "assisted_shot_xg": assisted_xg or None,
        "shot_type": nested_name(shot_item.get("type")), "shot_statsbomb_xg": shot_item.get("statsbomb_xg"),
        "goalkeeper_type": nested_name(keeper_item.get("type")), "duel_type": nested_name(duel_item.get("type")),
        "source_file": source_file,
    }


def load_match_catalog() -> tuple[pd.DataFrame, list[dict[str, Any]], set[str]]:
    competitions = json.loads((SOURCE / "competitions.json").read_bytes())
    competition_map = {
        (str(row["competition_id"]), str(row["season_id"])): row for row in competitions
    }
    rows: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    catalog_ids: set[str] = set()
    for path in sorted((SOURCE / "matches").rglob("*.json")):
        competition = competition_map[(path.parent.name, path.stem)]
        for match in json.loads(path.read_bytes()):
            match_id = str(match["match_id"])
            match_date = date.fromisoformat(match["match_date"])
            international = bool(competition.get("competition_international"))
            in_scope = (
                competition.get("competition_gender") == "male"
                and START_DATE <= match_date <= END_DATE
            )
            if competition["competition_name"] in BIG_FIVE and not international:
                scope_tier = "primary_big_five_club"
            elif international:
                scope_tier = "secondary_international"
            else:
                scope_tier = "secondary_other_club"
            home = match.get("home_team") or {}
            away = match.get("away_team") or {}
            row = {
                "match_id": match_id, "match_date": match_date.isoformat(),
                "kick_off": match.get("kick_off"), "competition_id": competition["competition_id"],
                "season_id": competition["season_id"], "country_name": competition["country_name"],
                "competition_name": competition["competition_name"],
                "competition_gender": competition.get("competition_gender"),
                "competition_international": international, "season_name": competition["season_name"],
                "match_week": match.get("match_week"),
                "competition_stage": nested_name(match.get("competition_stage")),
                "home_team_id": home.get("home_team_id"), "home_team_name": home.get("home_team_name"),
                "away_team_id": away.get("away_team_id"), "away_team_name": away.get("away_team_name"),
                "home_score": match.get("home_score"), "away_score": match.get("away_score"),
                "stadium_name": nested_name(match.get("stadium")), "referee_name": nested_name(match.get("referee")),
                "selected_for_case_studies": in_scope, "scope_tier": scope_tier if in_scope else "excluded",
                "event_file_available": (SOURCE / "events" / f"{match_id}.json").exists(),
                "lineup_file_available": (SOURCE / "lineups" / f"{match_id}.json").exists(),
                "three_sixty_file_available": (SOURCE / "three-sixty" / f"{match_id}.json").exists(),
                "match_source_file": str(path.relative_to(SOURCE_ROOT)).replace("\\", "/"),
            }
            rows.append(row)
            catalog_ids.add(match_id)
            if in_scope:
                selected.append(row)
    return pd.DataFrame(rows).sort_values(["match_date", "match_id"]), selected, catalog_ids


def build_lineup_metadata(
    match_id: str, match_duration: float
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    path = SOURCE / "lineups" / f"{match_id}.json"
    teams = json.loads(path.read_bytes())
    metadata: dict[int, dict[str, Any]] = {}
    lineup_rows: list[dict[str, Any]] = []
    for team in teams:
        team_id = int(team["team_id"])
        for player in team.get("lineup", []):
            positions = player.get("positions") or []
            intervals: list[tuple[float, float]] = []
            position_durations: Counter[str] = Counter()
            for item in positions:
                start = clock_minutes(item.get("from")) or 0.0
                end = clock_minutes(item.get("to"))
                end = match_duration if end is None else end
                if end > start:
                    intervals.append((start, end))
                    position_durations[item.get("position") or "Unknown"] += end - start
            minutes = min(union_minutes(intervals), match_duration)
            primary_position = position_durations.most_common(1)[0][0] if position_durations else None
            row = {
                "match_id": match_id, "team_id": team_id, "team_name": team.get("team_name"),
                "statsbomb_player_id": int(player["player_id"]), "player_name": player.get("player_name"),
                "player_nickname": player.get("player_nickname"), "jersey_number": player.get("jersey_number"),
                "country_name": nested_name(player.get("country")), "appeared": bool(positions),
                "started": any(item.get("start_reason") == "Starting XI" for item in positions),
                "minutes_played_estimate": round(minutes, 4), "primary_position": primary_position,
                "positions_played": " | ".join(name for name, _ in position_durations.most_common()),
                "yellow_cards": sum(card.get("card_type") == "Yellow Card" for card in player.get("cards", [])),
                "second_yellow_cards": sum(card.get("card_type") == "Second Yellow" for card in player.get("cards", [])),
                "red_cards": sum(card.get("card_type") == "Red Card" for card in player.get("cards", [])),
                "lineup_source_file": f"data/lineups/{match_id}.json",
            }
            metadata[int(player["player_id"])] = row
            lineup_rows.append(row)
    return metadata, lineup_rows


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.astype(float).div(denominator.replace(0, np.nan).astype(float))
    return result.replace([np.inf, -np.inf], np.nan)


def finalize_feature_rates(frame: pd.DataFrame, player: bool) -> pd.DataFrame:
    frame["pass_completion_pct"] = safe_divide(frame["passes_completed"], frame["passes"])
    frame["progressive_pass_share"] = safe_divide(frame["progressive_passes"], frame["passes"])
    frame["directness_ratio"] = safe_divide(frame["pass_forward_distance"], frame["pass_distance"])
    frame["xg_per_shot"] = safe_divide(frame["shot_xg"], frame["shots"])
    frame["under_pressure_event_share"] = safe_divide(frame["events_under_pressure"], frame["total_events"])
    frame["average_action_x"] = safe_divide(frame["location_x_sum"], frame["location_event_count"])
    frame["average_action_y"] = safe_divide(frame["location_y_sum"], frame["location_event_count"])
    if player:
        for field in [
            "passes", "progressive_passes", "carries", "progressive_carries", "shots",
            "shot_xg", "xg_assisted", "pressures", "ball_recoveries", "interceptions",
            "tackles", "successful_dribbles", "dispossessed",
        ]:
            frame[f"{field}_per90"] = safe_divide(frame[field] * 90.0, frame["minutes_played_estimate"])
    else:
        frame["passes_per_possession"] = safe_divide(frame["passes"], frame["possession_sequences"])
        frame["shots_per_possession"] = safe_divide(frame["shots"], frame["possession_sequences"])
    return frame


def aggregate_periods(player_matches: pd.DataFrame, team_matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    player_group = [
        "statsbomb_player_id", "player_name", "country_name", "team_id", "team_name",
        "competition_id", "competition_name", "season_id", "season_name", "scope_tier",
    ]
    player_sum = [
        "appeared", "started", "minutes_played_estimate", "yellow_cards", "second_yellow_cards", "red_cards",
        *COUNT_FIELDS, *SUM_FIELDS,
    ]
    player_period = player_matches.loc[player_matches["appeared"] | player_matches["total_events"].gt(0)].copy()
    player_period["matches_in_lineup"] = 1
    player_sum.append("matches_in_lineup")
    player_period = player_period.groupby(player_group, dropna=False, as_index=False)[player_sum].sum()
    player_period = player_period.rename(columns={"appeared": "appearances", "started": "starts"})
    player_period = finalize_feature_rates(player_period, player=True)

    team_group = [
        "team_id", "team_name", "competition_id", "competition_name", "season_id", "season_name", "scope_tier",
    ]
    team_sum = [*COUNT_FIELDS, *SUM_FIELDS, "possession_sequences", "goals_for", "goals_against"]
    team_period = team_matches.copy()
    team_period["matches"] = 1
    team_sum.append("matches")
    team_period = team_period.groupby(team_group, dropna=False, as_index=False)[team_sum].sum()
    team_period = finalize_feature_rates(team_period, player=False)
    team_period["goal_difference"] = team_period["goals_for"] - team_period["goals_against"]
    return player_period, team_period


def build_transfer_candidates(
    player_matches: pd.DataFrame, team_matches: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tm_players = pd.read_csv(DATA / "TransferMarkt" / "players.csv", dtype=str, low_memory=False)
    tm_players["normalized_name"] = tm_players["name"].map(normalized_text)
    tm_name_counts = tm_players.groupby("normalized_name")["player_id"].nunique()
    tm_unique = tm_players.loc[tm_players["normalized_name"].map(tm_name_counts).eq(1)].copy()
    tm_unique = tm_unique.drop_duplicates("normalized_name").set_index("normalized_name")

    sb_identity = player_matches[["statsbomb_player_id", "player_name", "country_name"]].drop_duplicates()
    sb_identity["normalized_name"] = sb_identity["player_name"].map(normalized_text)
    sb_counts = sb_identity.groupby("normalized_name")["statsbomb_player_id"].nunique()
    sb_unique = sb_identity.loc[sb_identity["normalized_name"].map(sb_counts).eq(1)].copy()
    sb_unique = sb_unique.sort_values(["normalized_name", "statsbomb_player_id"]).drop_duplicates("normalized_name")
    sb_unique["transfermarkt_candidate_count"] = sb_unique["normalized_name"].map(tm_name_counts).fillna(0).astype(int)
    sb_unique["transfermarkt_player_id"] = sb_unique["normalized_name"].map(
        tm_unique["player_id"] if not tm_unique.empty else pd.Series(dtype=str)
    )
    sb_unique["transfermarkt_player_name"] = sb_unique["normalized_name"].map(
        tm_unique["name"] if not tm_unique.empty else pd.Series(dtype=str)
    )
    sb_unique["transfermarkt_country"] = sb_unique["normalized_name"].map(
        tm_unique["country_of_citizenship"] if not tm_unique.empty else pd.Series(dtype=str)
    )
    sb_unique["country_agreement"] = sb_unique.apply(
        lambda row: bool(normalized_country(row["country_name"]))
        and normalized_country(row["country_name"]) == normalized_country(row["transfermarkt_country"]), axis=1,
    )
    sb_unique["match_status"] = np.where(
        sb_unique["transfermarkt_candidate_count"].eq(1), "candidate_review_required", "unmatched_or_ambiguous"
    )
    sb_unique["match_method"] = "exact_normalized_name_only"
    sb_unique["confidence_tier"] = np.where(sb_unique["country_agreement"], "strong_candidate", "name_only_candidate")
    sb_unique["manual_review_status"] = "required"
    player_crosswalk = sb_unique.sort_values(["match_status", "player_name"])

    club_dimension = pd.read_csv(DATA / "processed" / "club_dimension_union.csv", dtype=str, low_memory=False)
    alias_to_ids: dict[str, set[str]] = defaultdict(set)
    for row in club_dimension.itertuples(index=False):
        for alias in str(row.known_names or "").split(" | "):
            if normalized_text(alias):
                alias_to_ids[normalized_text(alias)].add(str(row.club_id))
    sb_teams = team_matches[["team_id", "team_name"]].drop_duplicates().copy()
    sb_teams["normalized_name"] = sb_teams["team_name"].map(normalized_text)
    sb_teams["transfermarkt_candidate_ids"] = sb_teams["normalized_name"].map(
        lambda value: " | ".join(sorted(alias_to_ids.get(value, set())))
    )
    sb_teams["transfermarkt_candidate_count"] = sb_teams["normalized_name"].map(
        lambda value: len(alias_to_ids.get(value, set()))
    )
    sb_teams["transfermarkt_club_id"] = sb_teams.apply(
        lambda row: row["transfermarkt_candidate_ids"] if row["transfermarkt_candidate_count"] == 1 else None, axis=1
    )
    sb_teams["match_status"] = np.where(
        sb_teams["transfermarkt_candidate_count"].eq(1), "candidate_review_required", "unmatched_or_ambiguous"
    )
    sb_teams["match_method"] = "exact_normalized_alias_only"
    sb_teams["manual_review_status"] = "required"
    club_crosswalk = sb_teams.sort_values(["match_status", "team_name"])

    accepted_player_candidates = player_crosswalk.loc[
        player_crosswalk["transfermarkt_candidate_count"].eq(1)
    ].set_index("transfermarkt_player_id")
    player_dates: dict[int, list[date]] = defaultdict(list)
    for row in player_matches.loc[player_matches["appeared"]].itertuples(index=False):
        player_dates[int(row.statsbomb_player_id)].append(date.fromisoformat(str(row.match_date)))
    accepted_club_candidates = club_crosswalk.loc[
        club_crosswalk["transfermarkt_candidate_count"].eq(1)
    ].set_index("transfermarkt_club_id")
    club_dates: dict[int, list[date]] = defaultdict(list)
    for row in team_matches.itertuples(index=False):
        club_dates[int(row.team_id)].append(date.fromisoformat(str(row.match_date)))

    cohort = pd.read_csv(DATA / "processed" / "transfer_cohort_master.csv", dtype=str, low_memory=False)
    candidate_rows: list[dict[str, Any]] = []
    for row in cohort.to_dict("records"):
        transfer_date = date.fromisoformat(str(row["transfer_date"])[:10])
        lower = transfer_date - timedelta(days=365)
        upper = transfer_date + timedelta(days=365)
        sb_player_id: int | None = None
        identity: pd.Series | None = None
        if str(row["player_id"]) in accepted_player_candidates.index:
            identity = accepted_player_candidates.loc[str(row["player_id"])]
            if isinstance(identity, pd.DataFrame):
                identity = identity.iloc[0]
            sb_player_id = int(identity["statsbomb_player_id"])
        origin_sb_id = None
        destination_sb_id = None
        if str(row["from_club_id"]) in accepted_club_candidates.index:
            club = accepted_club_candidates.loc[str(row["from_club_id"])]
            if isinstance(club, pd.DataFrame):
                club = club.iloc[0]
            origin_sb_id = int(club["team_id"])
        if str(row["to_club_id"]) in accepted_club_candidates.index:
            club = accepted_club_candidates.loc[str(row["to_club_id"])]
            if isinstance(club, pd.DataFrame):
                club = club.iloc[0]
            destination_sb_id = int(club["team_id"])

        dates = player_dates.get(sb_player_id, []) if sb_player_id is not None else []
        origin_dates = club_dates.get(origin_sb_id, []) if origin_sb_id is not None else []
        destination_dates = club_dates.get(destination_sb_id, []) if destination_sb_id is not None else []
        pre_player = sum(lower <= value < transfer_date for value in dates)
        post_player = sum(transfer_date <= value <= upper for value in dates)
        pre_origin = sum(lower <= value < transfer_date for value in origin_dates)
        pre_destination = sum(lower <= value < transfer_date for value in destination_dates)
        if not any([pre_player, post_player, pre_origin, pre_destination]):
            continue
        if pre_player >= 3 and post_player >= 3 and pre_destination >= 3:
            use = "player_before_after_with_destination_context"
        elif pre_player >= 3 and post_player >= 3:
            use = "player_before_after"
        elif pre_origin >= 3 and pre_destination >= 3:
            use = "origin_destination_style_comparison"
        elif pre_player >= 1:
            use = "pre_transfer_player_snapshot"
        else:
            use = "club_context_only"
        score = min(pre_player, 10) * 3 + min(post_player, 10) * 2 + min(pre_origin, 10) + min(pre_destination, 10)
        candidate_rows.append({
            "transfer_event_id": row["transfer_event_id"], "transfermarkt_player_id": row["player_id"],
            "player_name": row["player_name"], "transfer_date": row["transfer_date"],
            "from_club_id": row["from_club_id"], "from_club_name": row["from_club_name"],
            "to_club_id": row["to_club_id"], "to_club_name": row["to_club_name"],
            "statsbomb_player_id": sb_player_id, "statsbomb_origin_team_id": origin_sb_id,
            "statsbomb_destination_team_id": destination_sb_id,
            "player_matches_pre365": pre_player, "player_matches_post365": post_player,
            "origin_matches_pre365": pre_origin, "destination_matches_pre365": pre_destination,
            "case_study_score": score, "recommended_case_study_use": use,
            "player_identity_country_agreement": bool(identity["country_agreement"]) if identity is not None else None,
            "player_link_method": "exact_normalized_name_only" if identity is not None else None,
            "manual_review_status": "required",
            "eligible_sporting_24m": row.get("eligible_sporting_24m"),
            "eligible_financial_24m_provisional": row.get("eligible_financial_24m_provisional"),
            "eligible_combined_24m_provisional": row.get("eligible_combined_24m_provisional"),
            "post24_dest_minutes": row.get("post24_dest_minutes"),
            "fee_value_roi_proxy_24m": row.get("fee_value_roi_proxy_24m"),
            "timing_note": "Pre365 counts are feature-safe; post365 fields are outcome diagnostics only.",
        })
    candidates = pd.DataFrame(candidate_rows)
    if not candidates.empty:
        candidates = candidates.sort_values(
            ["case_study_score", "player_matches_pre365", "player_matches_post365"], ascending=False
        )
    return player_crosswalk, club_crosswalk, candidates


def create_field_dictionary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    explicit = {
        "scope_tier": "Case-study tier: primary Big Five club, secondary other club, or secondary international.",
        "minutes_played_estimate": "Estimated minutes from lineup position intervals; final-whistle intervals use the final event time.",
        "progressive_passes": "Passes advancing at least 10 StatsBomb x-coordinate units.",
        "progressive_carries": "Carries advancing at least 10 StatsBomb x-coordinate units.",
        "xg_assisted": "StatsBomb xG of shots linked back to the player's key passes.",
        "shot_xg": "Sum of StatsBomb shot xG.",
        "possession_sequences": "Distinct StatsBomb possession IDs attributed to the team.",
        "possession_sequence_share": "Share of distinct match possessions attributed to the team; not time-based possession percentage.",
        "case_study_score": "Selection score only: weighted capped counts of pre/post player and club matches; not a model target.",
        "manual_review_status": "Identity mappings are candidates and must be reviewed before analytical use.",
        "source_file": "Relative immutable StatsBomb source JSON file.",
    }
    rows: list[dict[str, Any]] = []
    for table_name, frame in tables.items():
        for column in frame.columns:
            if column in explicit:
                definition = explicit[column]
            elif column.endswith("_per90"):
                definition = f"{column[:-6].replace('_', ' ').title()} per 90 estimated minutes."
            elif column.endswith("_pct"):
                definition = column.replace("_", " ").title() + " as a 0-1 ratio."
            else:
                definition = column.replace("_", " ").capitalize() + "."
            if column.startswith("post") or "post365" in column or "roi" in column:
                role = "Target/diagnostic"
                timing = "Post-transfer; never use as a prediction feature."
            elif column in {"match_id", "event_id", "statsbomb_player_id", "team_id", "transfer_event_id"}:
                role = "Identifier/join key"
                timing = "Use for linkage and provenance, not as a numeric predictor."
            elif "manual_review" in column or "candidate" in column or "available" in column:
                role = "Quality/control"
                timing = "Review or filter field; not a substantive predictor."
            else:
                role = "Feature/context"
                timing = "Safe only when the record date is strictly before the modeled transfer date."
            rows.append({
                "table_name": table_name, "column_name": column, "dtype": str(frame[column].dtype),
                "definition": definition, "modeling_role": role, "timing_rule": timing,
            })
    return pd.DataFrame(rows)


def main() -> None:
    ensure_inputs()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("Loading StatsBomb match catalog", flush=True)
    match_catalog, selected_matches, catalog_ids = load_match_catalog()
    selected_by_id = {str(row["match_id"]): row for row in selected_matches}
    event_path = PROCESSED / "events_case_study.csv.gz"
    context_path = PROCESSED / "three_sixty_event_context.csv.gz"

    player_rows: list[dict[str, Any]] = []
    team_rows: list[dict[str, Any]] = []
    lineup_output_rows: list[dict[str, Any]] = []
    event_type_counts: Counter[str] = Counter()
    selected_event_ids: set[str] = set()
    duplicate_event_ids = 0
    event_parse_errors: list[dict[str, Any]] = []
    selected_event_count = 0
    match_event_counts: dict[str, int] = {}

    print(f"Extracting {len(selected_matches):,} selected matches", flush=True)
    with gzip.open(event_path, "wt", encoding="utf-8", newline="", compresslevel=6) as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_COLUMNS)
        writer.writeheader()
        for sequence, match in enumerate(sorted(selected_matches, key=lambda row: (row["match_date"], row["match_id"])), 1):
            match_id = str(match["match_id"])
            source_path = SOURCE / "events" / f"{match_id}.json"
            try:
                events = json.loads(source_path.read_bytes())
            except Exception as exc:
                event_parse_errors.append({"match_id": match_id, "source_file": str(source_path), "error": str(exc)})
                continue
            match_event_counts[match_id] = len(events)
            selected_event_count += len(events)
            match_duration = max(
                (float(event.get("minute") or 0) + float(event.get("second") or 0) / 60.0 for event in events),
                default=90.0,
            )
            match_duration = max(match_duration, 90.0)
            lineup_metadata, current_lineups = build_lineup_metadata(match_id, match_duration)
            lineup_output_rows.extend(current_lineups)
            assisted_xg_by_pass: dict[str, float] = {}
            for event in events:
                shot = event.get("shot") or {}
                key_pass_id = shot.get("key_pass_id")
                if key_pass_id:
                    assisted_xg_by_pass[str(key_pass_id)] = float(shot.get("statsbomb_xg") or 0.0)

            team_metrics: dict[int, dict[str, Any]] = {}
            possession_sets: dict[int, set[int]] = defaultdict(set)
            player_metrics: dict[int, dict[str, Any]] = {}
            for side in ("home", "away"):
                team_id = int(match[f"{side}_team_id"])
                team_metrics[team_id] = {
                    **empty_metrics(), "team_id": team_id, "team_name": match[f"{side}_team_name"],
                }
            for lineup in current_lineups:
                player_id = int(lineup["statsbomb_player_id"])
                player_metrics[player_id] = {**empty_metrics(), **lineup}

            for event in events:
                event_id = str(event.get("id"))
                if event_id in selected_event_ids:
                    duplicate_event_ids += 1
                selected_event_ids.add(event_id)
                event_type = nested_name(event.get("type")) or "Unknown"
                event_type_counts[event_type] += 1
                assisted_xg = assisted_xg_by_pass.get(event_id, 0.0)
                team_id = nested_id(event.get("team"))
                player_id = nested_id(event.get("player"))
                possession_team_id = nested_id(event.get("possession_team"))
                possession_id = event.get("possession")
                if team_id is not None:
                    team_id = int(team_id)
                    if team_id not in team_metrics:
                        team_metrics[team_id] = {**empty_metrics(), "team_id": team_id, "team_name": nested_name(event.get("team"))}
                    update_metrics(team_metrics[team_id], event, assisted_xg)
                if possession_team_id is not None and possession_id is not None:
                    possession_sets[int(possession_team_id)].add(int(possession_id))
                if player_id is not None:
                    player_id = int(player_id)
                    if player_id not in player_metrics:
                        player_metrics[player_id] = {
                            **empty_metrics(),
                            "match_id": match_id, "team_id": team_id, "team_name": nested_name(event.get("team")),
                            "statsbomb_player_id": player_id, "player_name": nested_name(event.get("player")),
                            "player_nickname": None, "jersey_number": None, "country_name": None,
                            "appeared": True, "started": False, "minutes_played_estimate": np.nan,
                            "primary_position": nested_name(event.get("position")),
                            "positions_played": nested_name(event.get("position")),
                            "yellow_cards": 0, "second_yellow_cards": 0, "red_cards": 0,
                            "lineup_source_file": f"data/lineups/{match_id}.json",
                        }
                    update_metrics(player_metrics[player_id], event, assisted_xg)
                writer.writerow(event_to_row(
                    event, match, f"data/events/{match_id}.json", assisted_xg
                ))

            total_possessions = len(set().union(*possession_sets.values())) if possession_sets else 0
            team_ids = list(team_metrics)
            for team_id, metrics in team_metrics.items():
                opponent_id = next((value for value in team_ids if value != team_id), None)
                opponent = team_metrics.get(opponent_id, {})
                is_home = team_id == int(match["home_team_id"])
                goals_for = int(match["home_score"] if is_home else match["away_score"])
                goals_against = int(match["away_score"] if is_home else match["home_score"])
                row = {
                    **metrics, "match_id": match_id, "match_date": match["match_date"],
                    "competition_id": match["competition_id"], "competition_name": match["competition_name"],
                    "season_id": match["season_id"], "season_name": match["season_name"],
                    "scope_tier": match["scope_tier"], "is_home": is_home,
                    "opponent_team_id": opponent_id, "opponent_team_name": opponent.get("team_name"),
                    "goals_for": goals_for, "goals_against": goals_against,
                    "result": "W" if goals_for > goals_against else "D" if goals_for == goals_against else "L",
                    "possession_sequences": len(possession_sets.get(team_id, set())),
                    "possession_sequence_share": (
                        len(possession_sets.get(team_id, set())) / total_possessions if total_possessions else np.nan
                    ),
                    "opponent_passes": opponent.get("passes", 0.0),
                    "pressures_per_100_opponent_passes": (
                        metrics["pressures"] * 100.0 / opponent.get("passes", 0.0)
                        if opponent.get("passes", 0.0) else np.nan
                    ),
                    "xg_difference": metrics["shot_xg"] - opponent.get("shot_xg", 0.0),
                    "source_file": f"data/events/{match_id}.json",
                }
                team_rows.append(row)
            for metrics in player_metrics.values():
                row = {
                    **metrics, "match_date": match["match_date"],
                    "competition_id": match["competition_id"], "competition_name": match["competition_name"],
                    "season_id": match["season_id"], "season_name": match["season_name"],
                    "scope_tier": match["scope_tier"], "source_file": f"data/events/{match_id}.json",
                }
                player_rows.append(row)
            if sequence % 100 == 0:
                print(f"  processed {sequence:,}/{len(selected_matches):,} matches", flush=True)

    player_matches = finalize_feature_rates(pd.DataFrame(player_rows), player=True)
    team_matches = finalize_feature_rates(pd.DataFrame(team_rows), player=False)
    lineups = pd.DataFrame(lineup_output_rows)
    player_periods, team_periods = aggregate_periods(player_matches, team_matches)

    print("Building compact StatsBomb 360 context", flush=True)
    context_count = 0
    valid_360 = 0
    invalid_360: list[dict[str, Any]] = []
    with gzip.open(context_path, "wt", encoding="utf-8", newline="", compresslevel=6) as handle:
        writer = csv.DictWriter(handle, fieldnames=CONTEXT_360_COLUMNS)
        writer.writeheader()
        for path in sorted((SOURCE / "three-sixty").glob("*.json")):
            try:
                frames = json.loads(path.read_bytes())
                valid_360 += 1
            except Exception as exc:
                invalid_360.append({
                    "issue_type": "invalid_json", "source_file": str(path.relative_to(SOURCE_ROOT)).replace("\\", "/"),
                    "match_id": path.stem, "severity": "warning", "details": str(exc),
                })
                continue
            if path.stem not in selected_by_id:
                continue
            for frame in frames:
                freeze = frame.get("freeze_frame") or []
                actors = [item for item in freeze if item.get("actor") and len(item.get("location") or []) >= 2]
                actor = actors[0] if actors else None
                actor_location = actor.get("location") if actor else None
                teammate_distances: list[float] = []
                opponent_distances: list[float] = []
                if actor_location:
                    for item in freeze:
                        location = item.get("location") or []
                        if len(location) < 2 or item is actor:
                            continue
                        distance = math.dist(actor_location[:2], location[:2])
                        (teammate_distances if item.get("teammate") else opponent_distances).append(distance)
                writer.writerow({
                    "match_id": path.stem, "event_id": frame.get("event_uuid"),
                    "visible_area_polygon_area": polygon_area(frame.get("visible_area") or []),
                    "visible_players": len(freeze),
                    "visible_teammates": sum(bool(item.get("teammate")) for item in freeze),
                    "visible_opponents": sum(not bool(item.get("teammate")) for item in freeze),
                    "actor_observations": len(actors), "actor_is_keeper": bool(actor and actor.get("keeper")),
                    "nearest_teammate_distance": min(teammate_distances) if teammate_distances else None,
                    "nearest_opponent_distance": min(opponent_distances) if opponent_distances else None,
                    "teammates_within_10": sum(value <= 10.0 for value in teammate_distances),
                    "opponents_within_5": sum(value <= 5.0 for value in opponent_distances),
                    "opponents_within_10": sum(value <= 10.0 for value in opponent_distances),
                    "source_file": f"data/three-sixty/{path.name}",
                })
                context_count += 1

    print("Creating review-only transfer overlap candidates", flush=True)
    player_crosswalk, club_crosswalk, transfer_candidates = build_transfer_candidates(player_matches, team_matches)

    event_counts = pd.DataFrame(
        [{"event_type": name, "selected_event_count": count} for name, count in event_type_counts.most_common()]
    )
    match_catalog["selected_event_count"] = match_catalog["match_id"].map(match_event_counts).fillna(0).astype(int)
    coverage = match_catalog.groupby(
        ["country_name", "competition_name", "competition_gender", "competition_international",
         "season_name", "competition_id", "season_id"], as_index=False, dropna=False
    ).agg(
        cataloged_matches=("match_id", "size"),
        selected_matches=("selected_for_case_studies", "sum"),
        selected_events=("selected_event_count", "sum"),
        selected_360_matches=("three_sixty_file_available", lambda values: int(sum(values & match_catalog.loc[values.index, "selected_for_case_studies"]))),
    )
    coverage["selection_share"] = safe_divide(coverage["selected_matches"], coverage["cataloged_matches"])
    coverage = coverage.sort_values(["selected_matches", "competition_name", "season_name"], ascending=[False, True, True])

    event_files = list((SOURCE / "events").glob("*.json"))
    orphan_files = [path for path in event_files if path.stem not in catalog_ids]
    quality_issues = pd.DataFrame([
        {
            "issue_type": "orphan_event_and_lineup_files", "source_file": "data/events + data/lineups",
            "match_id": None, "severity": "warning",
            "details": f"{len(orphan_files)} event files ({sum(path.stat().st_size for path in orphan_files):,} bytes) lack current match metadata and were excluded.",
        },
        *invalid_360,
        {
            "issue_type": "sample_selection_bias", "source_file": "data/competitions.json + data/matches",
            "match_id": None, "severity": "warning",
            "details": "Modern Big Five data centers on Barcelona, PSG, and Bayer Leverkusen; extracted rows are not a complete league panel.",
        },
        {
            "issue_type": "identity_crosswalk_review", "source_file": "generated crosswalk candidates",
            "match_id": None, "severity": "warning",
            "details": "StatsBomb-to-Transfermarkt links use exact normalized names/aliases without StatsBomb birth dates and require manual review.",
        },
    ])

    checks: list[dict[str, Any]] = []
    def add_check(name: str, passed: bool, observed: Any, expectation: str, blocking: bool = True) -> None:
        checks.append({
            "check_name": name, "status": "PASS" if passed else "FAIL",
            "blocking": blocking, "observed": observed, "expectation": expectation,
        })
    add_check("selected_match_count_positive", len(selected_matches) > 0, len(selected_matches), "> 0")
    add_check("selected_matches_unique", len(selected_matches) == len({row["match_id"] for row in selected_matches}), len(selected_matches), "unique match_id")
    add_check("selected_event_parse_errors", not event_parse_errors, len(event_parse_errors), "0")
    add_check("selected_event_ids_unique", duplicate_event_ids == 0, duplicate_event_ids, "0 duplicates")
    add_check("team_match_two_rows_per_match", len(team_matches) == len(selected_matches) * 2, len(team_matches), f"{len(selected_matches) * 2}")
    add_check("player_match_key_unique", not player_matches.duplicated(["match_id", "statsbomb_player_id"]).any(), int(player_matches.duplicated(["match_id", "statsbomb_player_id"]).sum()), "0 duplicates")
    add_check("selected_event_row_reconciliation", selected_event_count == sum(event_type_counts.values()), selected_event_count, str(sum(event_type_counts.values())))
    add_check("pass_reconciliation", int(player_matches["passes"].sum()) == event_type_counts["Pass"], int(player_matches["passes"].sum()), str(event_type_counts["Pass"]))
    add_check("selected_360_json_valid", not any(issue["match_id"] in selected_by_id for issue in invalid_360), sum(issue["match_id"] in selected_by_id for issue in invalid_360), "0 selected invalid files")
    add_check("repository_360_json_valid", not invalid_360, len(invalid_360), "0 invalid files", blocking=False)
    checks_frame = pd.DataFrame(checks)

    tables = {
        "match_catalog": match_catalog,
        "competition_coverage": coverage,
        "lineups_case_study": lineups,
        "player_match_features": player_matches,
        "team_match_features": team_matches,
        "player_period_features": player_periods,
        "team_period_features": team_periods,
        "event_type_counts": event_counts,
        "statsbomb_player_crosswalk_candidates": player_crosswalk,
        "statsbomb_club_crosswalk_candidates": club_crosswalk,
        "transfer_case_study_candidates": transfer_candidates,
        "data_quality_issues": quality_issues,
        "extraction_checks": checks_frame,
    }
    field_dictionary = create_field_dictionary(tables)
    tables["field_dictionary"] = field_dictionary

    output_files: dict[str, tuple[Path, int]] = {
        "match_catalog": (PROCESSED / "match_catalog.csv", len(match_catalog)),
        "competition_coverage": (PROCESSED / "competition_coverage.csv", len(coverage)),
        "lineups_case_study": (PROCESSED / "lineups_case_study.csv", len(lineups)),
        "player_match_features": (PROCESSED / "player_match_features.csv", len(player_matches)),
        "team_match_features": (PROCESSED / "team_match_features.csv", len(team_matches)),
        "player_period_features": (PROCESSED / "player_period_features.csv", len(player_periods)),
        "team_period_features": (PROCESSED / "team_period_features.csv", len(team_periods)),
        "event_type_counts": (PROCESSED / "event_type_counts.csv", len(event_counts)),
        "statsbomb_player_crosswalk_candidates": (PROCESSED / "statsbomb_player_crosswalk_candidates.csv", len(player_crosswalk)),
        "statsbomb_club_crosswalk_candidates": (PROCESSED / "statsbomb_club_crosswalk_candidates.csv", len(club_crosswalk)),
        "transfer_case_study_candidates": (PROCESSED / "transfer_case_study_candidates.csv", len(transfer_candidates)),
        "data_quality_issues": (PROCESSED / "data_quality_issues.csv", len(quality_issues)),
        "extraction_checks": (PROCESSED / "extraction_checks.csv", len(checks_frame)),
        "field_dictionary": (PROCESSED / "field_dictionary.csv", len(field_dictionary)),
    }
    for table_name, (path, _) in output_files.items():
        write_csv(tables[table_name], path)

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "scope_start": START_DATE.isoformat(), "scope_end": END_DATE.isoformat(),
        "cataloged_matches": int(len(match_catalog)), "selected_matches": int(len(selected_matches)),
        "selected_events": int(selected_event_count), "selected_lineup_rows": int(len(lineups)),
        "player_match_rows": int(len(player_matches)), "team_match_rows": int(len(team_matches)),
        "player_period_rows": int(len(player_periods)), "team_period_rows": int(len(team_periods)),
        "selected_360_context_rows": int(context_count), "valid_repository_360_files": int(valid_360),
        "invalid_repository_360_files": int(len(invalid_360)), "orphan_event_files_excluded": int(len(orphan_files)),
        "transfer_case_study_candidates": int(len(transfer_candidates)),
        "blocking_checks_failed": int(((checks_frame["blocking"]) & checks_frame["status"].ne("PASS")).sum()),
        "selection_rule": "Cataloged male matches dated 2017-07-01 through 2024-06-30 inclusive.",
    }
    (PROCESSED / "extraction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest_rows = []
    for table_name, (path, row_count) in output_files.items():
        manifest_rows.append({
            "artifact": table_name, "relative_path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "format": path.suffix.lstrip("."), "rows": row_count, "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    for artifact, path, row_count in [
        ("events_case_study", event_path, selected_event_count),
        ("three_sixty_event_context", context_path, context_count),
        ("extraction_summary", PROCESSED / "extraction_summary.json", 1),
    ]:
        manifest_rows.append({
            "artifact": artifact, "relative_path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "format": "csv.gz" if path.name.endswith(".csv.gz") else path.suffix.lstrip("."),
            "rows": row_count, "bytes": path.stat().st_size, "sha256": sha256(path),
        })
    manifest = pd.DataFrame(manifest_rows).sort_values("artifact")
    write_csv(manifest, PROCESSED / "output_manifest.csv")

    top_cases = transfer_candidates.head(100).replace({np.nan: None}).to_dict("records") if not transfer_candidates.empty else []
    workbook_payload = {
        "summary": summary,
        "coverage": coverage.replace({np.nan: None}).to_dict("records"),
        "top_cases": top_cases,
        "quality": quality_issues.replace({np.nan: None}).to_dict("records"),
        "checks": checks_frame.replace({np.nan: None}).to_dict("records"),
        "manifest": manifest.replace({np.nan: None}).to_dict("records"),
        "event_types": event_counts.head(20).replace({np.nan: None}).to_dict("records"),
        "dictionary": field_dictionary.loc[
            field_dictionary["column_name"].isin({
                "match_id", "match_date", "scope_tier", "statsbomb_player_id", "team_id",
                "minutes_played_estimate", "progressive_passes", "progressive_carries",
                "shot_xg", "xg_assisted", "possession_sequences", "possession_sequence_share",
                "case_study_score", "manual_review_status", "source_file",
            })
        ].replace({np.nan: None}).to_dict("records"),
    }
    (OUTPUT / "workbook_payload.json").write_text(json.dumps(workbook_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = f"""# StatsBomb case-study extract

Generated from the immutable local StatsBomb Open Data checkout.

## Selection rule

- Cataloged matches only: a match must exist in `data/matches`.
- Male competitions only.
- Match dates from {START_DATE.isoformat()} through {END_DATE.isoformat()}, inclusive.
- All modern male competitions are retained for case studies and labeled as `primary_big_five_club`, `secondary_other_club`, or `secondary_international`.
- The {len(orphan_files):,} event files without current match metadata are excluded.
- Cross-source player and club links are review-only candidates, never automatically accepted identities.

## Result

- {len(selected_matches):,} selected matches from {len(match_catalog):,} cataloged matches.
- {selected_event_count:,} slim event rows in `events_case_study.csv.gz`.
- {len(player_matches):,} player-match feature rows and {len(team_matches):,} team-match feature rows.
- {len(player_periods):,} player-team-competition-season rows and {len(team_periods):,} team-competition-season rows.
- {context_count:,} compact StatsBomb 360 event-context rows.
- {len(transfer_candidates):,} transfer case-study candidate rows requiring identity review.

## Recommended files

- `transfer_case_study_candidates.csv`: shortlist transfers by available pre/post player and club evidence.
- `player_period_features.csv`: compact player style/performance profiles.
- `team_period_features.csv`: compact club/national-team style profiles.
- `player_match_features.csv` and `team_match_features.csv`: match-level drilldown.
- `events_case_study.csv.gz`: slim event stream for pitch maps, pass maps, shot maps, and timelines.
- `three_sixty_event_context.csv.gz`: compact spatial context joined by `event_id`.
- `match_catalog.csv`: full catalog with inclusion flags.
- `field_dictionary.csv`, `data_quality_issues.csv`, `extraction_checks.csv`, and `output_manifest.csv`: audit and provenance.

## Important cautions

1. Modern Big Five coverage is selected rather than league-complete; do not train the main compatibility model on this sample as if it were representative.
2. `progressive_passes` and `progressive_carries` use a documented 10-unit forward-gain heuristic, not a proprietary StatsBomb definition.
3. `minutes_played_estimate` is reconstructed from lineup intervals and final event time.
4. `possession_sequence_share` counts distinct possessions; it is not time-based possession percentage.
5. Player and club crosswalks lack StatsBomb birth-date evidence and remain `manual_review_status = required`.
6. Post-transfer fields in the transfer shortlist are targets/diagnostics and must never enter pre-transfer features.
7. Published work must credit StatsBomb and comply with the repository attribution/logo terms.

## Reproduction

Run with the project Python environment:

```bash
python scripts/extract_statsbomb_case_studies.py
```
"""
    (PROCESSED / "README.md").write_text(readme, encoding="utf-8")

    if summary["blocking_checks_failed"]:
        raise AssertionError(f"{summary['blocking_checks_failed']} blocking extraction checks failed")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

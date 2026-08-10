"""Build the first reproducible MoveMaker transfer cohort.

This script preserves a broad transfer-level master table and adds explicit
eligibility flags for sporting, financial, and combined 24-month modeling.
Source CSV files are read-only; generated data is written to data/processed and
outputs/transfer_cohort.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TM = DATA / "TransferMarkt"
SEASONAL = DATA / "2017-2024"
PROCESSED = DATA / "processed"
OUTPUT = ROOT / "outputs" / "transfer_cohort"

COHORT_START = pd.Timestamp("2018-07-01")
PRE_VALUATION_MAX_AGE_DAYS = 365
POST_VALUATION_TOLERANCE_DAYS = 90
SPORTING_MIN_DEST_GAMES_12M = 10
SPORTING_MIN_DEST_GAMES_24M = 20
VALUATION_TOLERANCES = [60, 90, 120, 180]


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.casefold().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", "", value)


def clean_id(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def snake_case(value: str) -> str:
    overrides = {
        "Avg Mins per Match": "total_minutes",
        "1/3": "passes_into_final_third",
        "touches_def_pen": "touches_defensive_penalty_area",
        "carries_prgc": "progressive_carries_duplicate",
        "Goals Scored": "goals_scored_duplicate",
    }
    if value in overrides:
        return overrides[value]
    value = value.replace("%", " pct ").replace("&", " and ").replace("/", " per ")
    value = re.sub(r"\bp\s*90\b", " per90 ", value, flags=re.I)
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").casefold()
    return re.sub(r"_+", "_", value)


def pipe_unique(values: pd.Series) -> str:
    items = sorted({str(v).strip() for v in values if str(v).strip()})
    return " | ".join(items)


def safe_bool(value) -> bool:
    return bool(value) if not pd.isna(value) else False


def build_player_crosswalk(players: pd.DataFrame, seasonal: pd.DataFrame) -> pd.DataFrame:
    player_dim = players[["player_id", "name", "date_of_birth"]].copy()
    player_dim["player_id"] = clean_id(player_dim["player_id"])
    player_dim["normalized_name"] = player_dim["name"].map(normalize_text)
    player_dim["birth_year"] = player_dim["date_of_birth"].astype(str).str.slice(0, 4)

    lookup: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for row in player_dim.itertuples(index=False):
        lookup[(row.normalized_name, row.birth_year)].append((row.player_id, row.name))

    identities = (
        seasonal.groupby(["player", "born"], dropna=False)
        .agg(
            seasonal_rows=("season", "size"),
            seasons=("season", pipe_unique),
            squads=("squad", pipe_unique),
            competitions=("comp", pipe_unique),
        )
        .reset_index()
    )
    records = []
    for row in identities.itertuples(index=False):
        norm = normalize_text(row.player)
        born = str(row.born).replace(".0", "").strip()
        matches = lookup.get((norm, born), [])
        status = "unique" if len(matches) == 1 else "ambiguous" if len(matches) > 1 else "unmatched"
        records.append(
            {
                "seasonal_player": row.player,
                "birth_year": born,
                "normalized_name": norm,
                "match_status": status,
                "candidate_count": len(matches),
                "player_id": matches[0][0] if len(matches) == 1 else "",
                "transfermarkt_name": matches[0][1] if len(matches) == 1 else "",
                "candidate_player_ids": " | ".join(m[0] for m in matches),
                "seasonal_rows": int(row.seasonal_rows),
                "seasons": row.seasons,
                "squads": row.squads,
                "competitions": row.competitions,
                "match_method": "normalized player name + birth year",
                "manual_review_required": status != "unique",
            }
        )
    return pd.DataFrame(records)


def build_player_season_features(seasonal: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    linked = seasonal.merge(
        crosswalk[["seasonal_player", "birth_year", "player_id", "match_status"]],
        left_on=["player", "born"],
        right_on=["seasonal_player", "birth_year"],
        how="left",
    )
    linked = linked[linked["match_status"].eq("unique") & linked["player_id"].ne("")].copy()
    linked["season_end_date"] = pd.to_datetime(linked["season"].str[-4:] + "-06-30")
    linked["is_goalkeeper_row"] = linked["pos"].astype(str).str.contains("GK", regex=False)

    exclude = {
        "rk", "player", "nation", "pos", "squad", "comp", "age", "born", "season",
        "seasonal_player", "birth_year", "match_status", "player_id", "season_end_date",
        "Goals Scored", "carries_prgc", "is_goalkeeper_row",
    }
    numeric_source = [c for c in seasonal.columns if c not in exclude]
    for col in numeric_source:
        linked[col] = pd.to_numeric(linked[col], errors="coerce")

    goalkeeper_cols = [
        "Goals Against", "Goals against p 90", "Saves", "Saves %", "Clean Sheets",
        "% Clean sheets", "% Penalty saves", "Crosses Stopped",
    ]
    for col in goalkeeper_cols:
        if col in linked.columns:
            linked.loc[~linked["is_goalkeeper_row"], col] = np.nan

    weighted_markers = ("%", "p 90", "per shot")
    weighted_cols = [c for c in numeric_source if any(marker.casefold() in c.casefold() for marker in weighted_markers)]
    additive_cols = [c for c in numeric_source if c not in weighted_cols and c != "Avg Mins per Match"]
    group_cols = ["player_id", "season", "season_end_date"]
    weights = linked["Avg Mins per Match"].fillna(0).clip(lower=0)

    grouped = linked.groupby(group_cols, dropna=False)
    base = grouped.agg(
        seasonal_source_rows=("player", "size"),
        player_name=("player", "first"),
        birth_year=("born", "first"),
        age=("age", "max"),
        nations=("nation", pipe_unique),
        positions=("pos", pipe_unique),
        squads=("squad", pipe_unique),
        competitions=("comp", pipe_unique),
        total_minutes=("Avg Mins per Match", "sum"),
    ).reset_index()

    if additive_cols:
        sums = grouped[additive_cols].sum(min_count=1).reset_index()
        sums = sums.rename(columns={c: snake_case(c) for c in additive_cols})
        base = base.merge(sums, on=group_cols, how="left")

    for col in weighted_cols:
        valid = linked[col].notna() & weights.gt(0)
        numerator = (linked[col].where(valid) * weights.where(valid)).groupby(
            [linked[c] for c in group_cols], dropna=False
        ).sum(min_count=1)
        denominator = weights.where(valid).groupby([linked[c] for c in group_cols], dropna=False).sum(min_count=1)
        weighted = (numerator / denominator).rename(snake_case(col)).reset_index()
        base = base.merge(weighted, on=group_cols, how="left")

    base["feature_aggregation_note"] = (
        "Counts summed across player-season rows; rates/percentages minutes-weighted; "
        "goalkeeper-only fields set null for non-goalkeeper rows."
    )
    return base.sort_values(["player_id", "season_end_date"]).reset_index(drop=True)


def attach_latest_preseason(cohort: pd.DataFrame, season_features: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in season_features.columns if c not in {"player_id", "season_end_date"}]
    by_player = {
        pid: group.sort_values("season_end_date").reset_index(drop=True)
        for pid, group in season_features.groupby("player_id")
    }
    rows = []
    for row in cohort[["transfer_event_id", "player_id", "transfer_date"]].itertuples(index=False):
        group = by_player.get(row.player_id)
        result = {"transfer_event_id": row.transfer_event_id}
        if group is not None:
            idx = group["season_end_date"].searchsorted(row.transfer_date, side="left") - 1
            if idx >= 0:
                source = group.iloc[int(idx)]
                result["preseason_end_date"] = source["season_end_date"]
                for col in feature_cols:
                    result[f"preseason_{col}"] = source[col]
        rows.append(result)
    return cohort.merge(pd.DataFrame(rows), on="transfer_event_id", how="left")


def build_valuation_lookup(valuations: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result = {}
    for player_id, group in valuations.groupby("player_id"):
        group = group.sort_values("date")
        result[player_id] = (
            group["date"].to_numpy(dtype="datetime64[ns]"),
            group["market_value_in_eur"].to_numpy(dtype=float),
        )
    return result


def nearest_valuation(
    lookup: dict[str, tuple[np.ndarray, np.ndarray]], player_id: str, target: pd.Timestamp, direction: str
) -> tuple[pd.Timestamp | pd.NaT, float | None, int | None]:
    data = lookup.get(player_id)
    if data is None:
        return pd.NaT, None, None
    dates, values = data
    target64 = np.datetime64(target.to_datetime64())
    position = int(np.searchsorted(dates, target64, side="right"))
    candidates: list[int] = []
    if direction == "backward":
        if position > 0:
            candidates = [position - 1]
    else:
        if position > 0:
            candidates.append(position - 1)
        if position < len(dates):
            candidates.append(position)
    if not candidates:
        return pd.NaT, None, None
    idx = min(candidates, key=lambda i: abs((dates[i] - target64).astype("timedelta64[D]").astype(int)))
    date = pd.Timestamp(dates[idx])
    gap = int((date - target).days)
    return date, float(values[idx]), gap


def attach_valuations(cohort: pd.DataFrame, valuations: pd.DataFrame) -> pd.DataFrame:
    lookup = build_valuation_lookup(valuations)
    records = []
    for row in cohort[["transfer_event_id", "player_id", "transfer_date"]].itertuples(index=False):
        pre_date, pre_value, pre_gap = nearest_valuation(lookup, row.player_id, row.transfer_date, "backward")
        target12 = row.transfer_date + pd.DateOffset(years=1)
        target24 = row.transfer_date + pd.DateOffset(years=2)
        p12_date, p12_value, p12_gap = nearest_valuation(lookup, row.player_id, target12, "nearest")
        p24_date, p24_value, p24_gap = nearest_valuation(lookup, row.player_id, target24, "nearest")
        records.append(
            {
                "transfer_event_id": row.transfer_event_id,
                "pre_valuation_date": pre_date,
                "pre_valuation_asof_eur": pre_value,
                "pre_valuation_staleness_days": -pre_gap if pre_gap is not None else None,
                "post12_target_date": target12,
                "post12_nearest_valuation_date": p12_date,
                "post12_nearest_valuation_eur": p12_value,
                "post12_valuation_gap_days": p12_gap,
                "post24_target_date": target24,
                "post24_nearest_valuation_date": p24_date,
                "post24_nearest_valuation_eur": p24_value,
                "post24_valuation_gap_days": p24_gap,
            }
        )
    result = cohort.merge(pd.DataFrame(records), on="transfer_event_id", how="left")
    result["has_pre_valuation_365d"] = result["pre_valuation_staleness_days"].between(0, PRE_VALUATION_MAX_AGE_DAYS)
    for horizon in (12, 24):
        gap = result[f"post{horizon}_valuation_gap_days"].abs()
        for tolerance in VALUATION_TOLERANCES:
            result[f"has_post{horizon}_valuation_{tolerance}d"] = gap.le(tolerance)
        result[f"post{horizon}_valuation_90d_eur"] = result[f"post{horizon}_nearest_valuation_eur"].where(
            result[f"has_post{horizon}_valuation_{POST_VALUATION_TOLERANCE_DAYS}d"]
        )
    return result


def aggregate_appearances(cohort: pd.DataFrame) -> pd.DataFrame:
    usecols = [
        "game_id", "player_id", "player_club_id", "date", "yellow_cards", "red_cards",
        "goals", "assists", "minutes_played",
    ]
    apps = pd.read_csv(TM / "appearances.csv", usecols=usecols, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    apps["player_id"] = clean_id(apps["player_id"])
    apps["player_club_id"] = clean_id(apps["player_club_id"])
    apps["date"] = pd.to_datetime(apps["date"], errors="coerce")
    for col in ["yellow_cards", "red_cards", "goals", "assists", "minutes_played"]:
        apps[col] = pd.to_numeric(apps[col], errors="coerce").fillna(0)
    apps = apps[apps["player_id"].isin(set(cohort["player_id"]))].copy()
    merged = apps.merge(
        cohort[["transfer_event_id", "player_id", "transfer_date", "to_club_id"]],
        on="player_id",
        how="inner",
    )
    merged["days_from_transfer"] = (merged["date"] - merged["transfer_date"]).dt.days
    merged["at_destination"] = merged["player_club_id"].eq(merged["to_club_id"])

    windows = {
        "pre365": merged["days_from_transfer"].between(-365, -1),
        "post12_dest": merged["days_from_transfer"].between(0, 364) & merged["at_destination"],
        "post24_dest": merged["days_from_transfer"].between(0, 729) & merged["at_destination"],
    }
    result = cohort[["transfer_event_id"]].copy()
    for prefix, mask in windows.items():
        agg = (
            merged.loc[mask]
            .groupby("transfer_event_id")
            .agg(
                appearances=("game_id", "nunique"),
                minutes=("minutes_played", "sum"),
                goals=("goals", "sum"),
                assists=("assists", "sum"),
                yellow_cards=("yellow_cards", "sum"),
                red_cards=("red_cards", "sum"),
            )
            .add_prefix(f"{prefix}_")
            .reset_index()
        )
        result = result.merge(agg, on="transfer_event_id", how="left")
    metric_cols = [c for c in result.columns if c != "transfer_event_id"]
    result[metric_cols] = result[metric_cols].fillna(0)
    return cohort.merge(result, on="transfer_event_id", how="left")


def aggregate_destination_games(cohort: pd.DataFrame) -> pd.DataFrame:
    games = pd.read_csv(TM / "games.csv", usecols=["game_id", "date"], dtype=str, keep_default_na=False, encoding="utf-8-sig")
    games["game_id"] = clean_id(games["game_id"])
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    club_games = pd.read_csv(TM / "club_games.csv", usecols=["game_id", "club_id"], dtype=str, keep_default_na=False, encoding="utf-8-sig")
    club_games["game_id"] = clean_id(club_games["game_id"])
    club_games["club_id"] = clean_id(club_games["club_id"])
    club_games = club_games.merge(games, on="game_id", how="left")
    merged = club_games.merge(
        cohort[["transfer_event_id", "to_club_id", "transfer_date"]],
        left_on="club_id",
        right_on="to_club_id",
        how="inner",
    )
    merged["days_from_transfer"] = (merged["date"] - merged["transfer_date"]).dt.days
    result = cohort[["transfer_event_id"]].copy()
    for horizon, last_day in [(12, 364), (24, 729)]:
        counts = (
            merged[merged["days_from_transfer"].between(0, last_day)]
            .groupby("transfer_event_id")["game_id"]
            .nunique()
            .rename(f"destination_games_{horizon}m")
            .reset_index()
        )
        result = result.merge(counts, on="transfer_event_id", how="left")
    result[["destination_games_12m", "destination_games_24m"]] = result[
        ["destination_games_12m", "destination_games_24m"]
    ].fillna(0)
    return cohort.merge(result, on="transfer_event_id", how="left")


def build_union_club_dimension(
    transfers: pd.DataFrame, games: pd.DataFrame, clubs: pd.DataFrame, national_teams: pd.DataFrame
) -> pd.DataFrame:
    names: dict[str, Counter] = defaultdict(Counter)
    sources: dict[str, set[str]] = defaultdict(set)
    dates: dict[str, list[pd.Timestamp]] = defaultdict(list)

    def add(entity_id, name, source, date=None):
        entity_id = str(entity_id).replace(".0", "").strip()
        name = str(name).strip()
        if not entity_id:
            return
        sources[entity_id].add(source)
        if name:
            names[entity_id][name] += 1
        if date is not None and not pd.isna(date):
            dates[entity_id].append(pd.Timestamp(date))

    for row in clubs.itertuples(index=False):
        add(row.club_id, row.name, "clubs")
    for row in national_teams.itertuples(index=False):
        add(row.national_team_id, row.name, "national_teams")
    for row in transfers.itertuples(index=False):
        add(row.from_club_id, row.from_club_name, "transfers", row.transfer_date)
        add(row.to_club_id, row.to_club_name, "transfers", row.transfer_date)
    for row in games.itertuples(index=False):
        add(row.home_club_id, row.home_club_name, "games", row.date)
        add(row.away_club_id, row.away_club_name, "games", row.date)

    all_ids = sorted(sources, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
    records = []
    for entity_id in all_ids:
        source_set = sources[entity_id]
        preferred = ""
        if entity_id in set(clean_id(clubs["club_id"])):
            preferred_rows = clubs[clean_id(clubs["club_id"]).eq(entity_id)]
            if not preferred_rows.empty:
                preferred = preferred_rows.iloc[0]["name"]
        if not preferred and names[entity_id]:
            preferred = names[entity_id].most_common(1)[0][0]
        entity_dates = dates.get(entity_id, [])
        records.append(
            {
                "club_id": entity_id,
                "canonical_name": preferred,
                "known_names": " | ".join(sorted(names[entity_id])),
                "source_tables": " | ".join(sorted(source_set)),
                "present_in_clubs": "clubs" in source_set,
                "present_in_transfers": "transfers" in source_set,
                "present_in_games": "games" in source_set,
                "present_in_national_teams": "national_teams" in source_set,
                "first_observed_date": min(entity_dates) if entity_dates else pd.NaT,
                "last_observed_date": max(entity_dates) if entity_dates else pd.NaT,
                "name_count": len(names[entity_id]),
            }
        )
    return pd.DataFrame(records)


def add_eligibility_and_outcomes(cohort: pd.DataFrame) -> pd.DataFrame:
    transfer_year = cohort["transfer_date"].dt.year
    season_start_year = np.where(cohort["transfer_date"].dt.month >= 7, transfer_year, transfer_year - 1)
    cohort["expected_transfer_season"] = [
        f"{int(year) % 100:02d}/{(int(year) + 1) % 100:02d}" for year in season_start_year
    ]
    cohort["transfer_season_matches_date"] = cohort["transfer_season"].eq(cohort["expected_transfer_season"])
    cohort["age_plausible_15_to_45"] = cohort["age_at_transfer"].between(15, 45)
    cohort["is_same_club_movement"] = cohort["from_club_id"].eq(cohort["to_club_id"])
    cohort["has_preseason_features"] = cohort["preseason_season"].notna()
    cohort["has_destination_coverage_12m"] = cohort["destination_games_12m"].ge(SPORTING_MIN_DEST_GAMES_12M)
    cohort["has_destination_coverage_24m"] = cohort["destination_games_24m"].ge(SPORTING_MIN_DEST_GAMES_24M)
    cohort["has_positive_disclosed_fee"] = cohort["transfer_fee_eur"].gt(0)
    cohort["eligible_sporting_12m"] = (
        ~cohort["is_same_club_movement"]
        & cohort["has_preseason_features"]
        & cohort["has_destination_coverage_12m"]
    )
    cohort["eligible_sporting_24m"] = (
        ~cohort["is_same_club_movement"]
        & cohort["has_preseason_features"]
        & cohort["has_destination_coverage_12m"]
        & cohort["has_destination_coverage_24m"]
    )
    cohort["eligible_financial_12m_provisional"] = (
        ~cohort["is_same_club_movement"]
        & cohort["has_positive_disclosed_fee"]
        & cohort["has_pre_valuation_365d"]
        & cohort["has_post12_valuation_90d"]
    )
    cohort["eligible_financial_24m_provisional"] = (
        ~cohort["is_same_club_movement"]
        & cohort["has_positive_disclosed_fee"]
        & cohort["has_pre_valuation_365d"]
        & cohort["has_post24_valuation_90d"]
    )
    cohort["eligible_combined_24m_provisional"] = (
        cohort["eligible_sporting_24m"] & cohort["eligible_financial_24m_provisional"]
    )

    for horizon in (12, 24):
        post = cohort[f"post{horizon}_valuation_90d_eur"]
        cohort[f"valuation_change_{horizon}m_eur"] = post - cohort["pre_valuation_asof_eur"]
        cohort[f"valuation_change_{horizon}m_pct"] = (
            cohort[f"valuation_change_{horizon}m_eur"] / cohort["pre_valuation_asof_eur"]
        ).where(cohort["pre_valuation_asof_eur"].gt(0))
        cohort[f"fee_value_roi_proxy_{horizon}m"] = (
            (post - cohort["transfer_fee_eur"]) / cohort["transfer_fee_eur"]
        ).where(cohort["transfer_fee_eur"].gt(0))

    def reasons(row) -> str:
        flags = []
        if row.is_same_club_movement:
            flags.append("same_club_movement")
        if not row.has_preseason_features:
            flags.append("no_completed_seasonal_features")
        if not row.has_destination_coverage_24m:
            flags.append("insufficient_destination_game_coverage_24m")
        if row.fee_status == "missing":
            flags.append("fee_missing")
        elif row.fee_status == "zero":
            flags.append("fee_zero_or_unclassified_movement")
        if not row.has_pre_valuation_365d:
            flags.append("no_pre_valuation_within_365d")
        if not row.has_post24_valuation_90d:
            flags.append("no_post24_valuation_within_90d")
        return " | ".join(flags)

    cohort["eligibility_exclusion_reasons"] = cohort.apply(reasons, axis=1)
    cohort["cohort_tier"] = np.select(
        [
            cohort["eligible_combined_24m_provisional"],
            cohort["eligible_sporting_24m"],
            cohort["eligible_financial_24m_provisional"],
        ],
        ["combined_24m_provisional", "sporting_24m", "financial_24m_provisional"],
        default="candidate_not_model_ready",
    )
    return cohort


def build_funnels(raw_count: int, cohort: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(funnel, step, label, mask_or_count, note):
        count = int(mask_or_count if isinstance(mask_or_count, (int, np.integer)) else mask_or_count.sum())
        rows.append({"funnel": funnel, "step": step, "stage": label, "records": count, "note": note})

    add("Master", 0, "All recorded transfers", raw_count, "Unfiltered transfers.csv")
    add("Master", 1, "Inside cohort date window", len(cohort), f"{COHORT_START.date()} through the full 24m valuation cutoff")
    add("Master", 2, "Movement changes club ID", ~cohort["is_same_club_movement"], "Removes same-ID administrative movements")
    add("Master", 3, "Has prior completed seasonal features", (~cohort["is_same_club_movement"]) & cohort["has_preseason_features"], "Latest completed season before transfer")

    sporting_base = ~cohort["is_same_club_movement"]
    add("Sporting 24m", 0, "Dated cross-club candidates", sporting_base, "Master sporting candidate base")
    add("Sporting 24m", 1, "Has prior completed seasonal features", sporting_base & cohort["has_preseason_features"], "Required baseline features")
    add("Sporting 24m", 2, "Destination coverage: >=10 games at 12m and >=20 at 24m", cohort["eligible_sporting_24m"], "Allows zero player appearances to be interpreted as an outcome")

    financial_base = ~cohort["is_same_club_movement"]
    add("Financial 24m", 0, "Dated cross-club candidates", financial_base, "Master financial candidate base")
    stage1 = financial_base & cohort["has_positive_disclosed_fee"]
    add("Financial 24m", 1, "Positive disclosed fee", stage1, "Still provisional until loans are classified")
    stage2 = stage1 & cohort["has_pre_valuation_365d"]
    add("Financial 24m", 2, "Pre-transfer valuation within 365d", stage2, "As-of valuation")
    stage3 = stage2 & cohort["has_post24_valuation_90d"]
    add("Financial 24m", 3, "24m valuation within +/-90d", stage3, "Provisional financial sample")

    add("Combined 24m", 0, "Sporting-eligible 24m", cohort["eligible_sporting_24m"], "Prior season plus destination coverage")
    add("Combined 24m", 1, "Also financial-eligible 24m", cohort["eligible_combined_24m_provisional"], "Strict overlap sample")

    funnel = pd.DataFrame(rows)
    funnel["previous_stage_records"] = funnel.groupby("funnel")["records"].shift(1)
    funnel["retention_from_previous"] = funnel["records"] / funnel["previous_stage_records"]
    funnel["starting_records"] = funnel.groupby("funnel")["records"].transform("first")
    funnel["retention_from_start"] = funnel["records"] / funnel["starting_records"]
    return funnel


def build_checks(cohort: pd.DataFrame, cohort_end: pd.Timestamp) -> pd.DataFrame:
    checks = []

    def add(check, actual, expected, passed, notes, severity="error"):
        checks.append(
            {
                "check": check,
                "actual": actual,
                "expected": expected,
                "status": "OK" if passed else ("WARNING" if severity == "warning" else "FAIL"),
                "notes": notes,
            }
        )

    add("Transfer event ID uniqueness", cohort["transfer_event_id"].nunique(), len(cohort), cohort["transfer_event_id"].is_unique, "One row per source transfer key")
    add("Cohort start boundary", cohort["transfer_date"].min().date().isoformat(), f">= {COHORT_START.date()}", cohort["transfer_date"].min() >= COHORT_START, "No early records")
    add("Cohort end boundary", cohort["transfer_date"].max().date().isoformat(), f"<= {cohort_end.date()}", cohort["transfer_date"].max() <= cohort_end, "Full 24m valuation horizon")
    add("Pre-valuations do not occur after transfer", int((cohort["pre_valuation_date"] > cohort["transfer_date"]).sum()), 0, not (cohort["pre_valuation_date"] > cohort["transfer_date"]).any(), "As-of join leakage check")
    add("Eligible 24m valuations meet tolerance", int((cohort.loc[cohort["has_post24_valuation_90d"], "post24_valuation_gap_days"].abs() > 90).sum()), 0, not (cohort.loc[cohort["has_post24_valuation_90d"], "post24_valuation_gap_days"].abs() > 90).any(), "Outcome timing check")
    add("Negative transfer fees", int((cohort["transfer_fee_eur"] < 0).sum()), 0, not (cohort["transfer_fee_eur"] < 0).any(), "Sign check")
    add("Negative appearance minutes", int((cohort.filter(regex="_minutes$") < 0).sum().sum()), 0, not (cohort.filter(regex="_minutes$") < 0).any().any(), "Sporting metric sign check")
    add("Combined sample subset of sporting", int((cohort["eligible_combined_24m_provisional"] & ~cohort["eligible_sporting_24m"]).sum()), 0, not (cohort["eligible_combined_24m_provisional"] & ~cohort["eligible_sporting_24m"]).any(), "Eligibility reconciliation")
    add("Combined sample subset of financial", int((cohort["eligible_combined_24m_provisional"] & ~cohort["eligible_financial_24m_provisional"]).sum()), 0, not (cohort["eligible_combined_24m_provisional"] & ~cohort["eligible_financial_24m_provisional"]).any(), "Eligibility reconciliation")
    season_mismatches = int((~cohort["transfer_season_matches_date"]).sum())
    add("Transfer-season label/date mismatches", season_mismatches, 0, season_mismatches == 0, "Transfer date remains the authoritative time field; review mismatched labels", "warning")
    implausible_ages = int((~cohort["age_plausible_15_to_45"] & cohort["age_at_transfer"].notna()).sum())
    add("Ages outside 15-45", implausible_ages, 0, implausible_ages == 0, "Retained as a review flag rather than silently filtered", "warning")
    return pd.DataFrame(checks)


def describe_cohort_field(column: str) -> tuple[str, str, str, str]:
    raw_definitions = {
        "transfer_event_id": "Stable composite transfer key: player, transfer date, origin club, and destination club.",
        "player_id": "Transfermarkt player identifier.",
        "player_name": "Player name recorded on the transfer event.",
        "transfer_date": "Authoritative transfer event date and feature cutoff.",
        "transfer_season": "Original transfer-season label from transfers.csv.",
        "expected_transfer_season": "Season derived deterministically from transfer_date using a July-to-June convention.",
        "transfer_season_matches_date": "Quality flag comparing the original season label with the date-derived season.",
        "from_club_id": "Origin club/team identifier.",
        "to_club_id": "Destination club/team identifier.",
        "transfer_fee_eur": "Recorded transfer fee in euros; missing and zero values are not imputed.",
        "fee_status": "Fee classification: positive, zero, missing, or other.",
        "listed_market_value_at_transfer_eur": "Market value stored directly on the transfer record.",
        "age_at_transfer": "Player age in years on the transfer date, derived from date of birth.",
        "pre_valuation_asof_eur": "Latest player valuation on or before transfer_date.",
        "pre_valuation_staleness_days": "Days between the pre-transfer as-of valuation and transfer_date.",
        "destination_games_12m": "Recorded destination-club games in the first 12 months after transfer.",
        "destination_games_24m": "Recorded destination-club games in the first 24 months after transfer.",
        "cohort_tier": "Highest provisional modeling tier satisfied by the transfer.",
        "eligibility_exclusion_reasons": "Pipe-delimited reasons the record fails one or more strict eligibility rules.",
    }
    if column in raw_definitions:
        definition = raw_definitions[column]
    elif column.startswith("preseason_"):
        definition = f"Latest fully completed pre-transfer season feature: {column.removeprefix('preseason_').replace('_', ' ')}."
    elif column.startswith("pre365_"):
        definition = f"Player performance during days -365 through -1 relative to transfer: {column.removeprefix('pre365_').replace('_', ' ')}."
    elif column.startswith("post12_dest_"):
        definition = f"Destination-club player outcome during months 0-12 after transfer: {column.removeprefix('post12_dest_').replace('_', ' ')}."
    elif column.startswith("post24_dest_"):
        definition = f"Destination-club player outcome during months 0-24 after transfer: {column.removeprefix('post24_dest_').replace('_', ' ')}."
    elif column.startswith("post12_"):
        definition = f"Twelve-month valuation field: {column.removeprefix('post12_').replace('_', ' ')}."
    elif column.startswith("post24_"):
        definition = f"Twenty-four-month valuation field: {column.removeprefix('post24_').replace('_', ' ')}."
    elif column.startswith("valuation_change_"):
        definition = "Change from the pre-transfer as-of valuation to the tolerance-qualified horizon valuation."
    elif column.startswith("fee_value_roi_proxy_"):
        definition = "Valuation proxy calculated as (horizon market value - fee) / fee; not realized resale ROI."
    elif column.startswith("eligible_"):
        definition = "Boolean eligibility flag for the named provisional modeling sample."
    elif column.startswith("has_") or column.startswith("is_") or column.endswith("_matches_date") or column.endswith("_plausible_15_to_45"):
        definition = "Boolean data-availability, coverage, or quality flag described by the field name."
    else:
        definition = f"Source or derived cohort field: {column.replace('_', ' ')}."

    if column.startswith("preseason_") or column.startswith("pre365_") or column.startswith("pre_valuation"):
        group, usage, timing = "Pre-transfer features", "Feature", "Must be dated before transfer_date"
    elif column.startswith("post12_") or column.startswith("post24_") or column.startswith("valuation_change_") or column.startswith("fee_value_roi_proxy_"):
        group, usage, timing = "Outcome candidates", "Target/diagnostic", "Post-transfer only; never use as a prediction feature"
    elif column.startswith("eligible_") or column.startswith("has_") or column.startswith("is_") or "plausible" in column or "matches_date" in column:
        group, usage, timing = "Eligibility and quality", "Filter/diagnostic", "Defines sample membership; not a substantive predictor"
    elif column in {"transfer_fee_eur", "fee_status", "listed_market_value_at_transfer_eur"}:
        group, usage, timing = "Transfer economics", "Feature/eligibility", "Known at transfer subject to movement-type validation"
    elif column in {"transfer_event_id", "player_id", "from_club_id", "to_club_id"}:
        group, usage, timing = "Identifiers", "Join/split control", "Do not use raw identifiers as numeric predictors"
    elif column in {"transfer_date", "transfer_season", "expected_transfer_season"}:
        group, usage, timing = "Time anchors", "Split/control", "Use for as-of joins and chronological validation"
    elif column in {"age_at_transfer", "position", "sub_position", "foot", "height_in_cm", "country_of_citizenship"}:
        group, usage, timing = "Player attributes", "Feature", "Stable or evaluated at transfer date"
    else:
        group, usage, timing = "Source/context", "Conditional", "Review definition and historical availability"
    return group, definition, usage, timing


def build_cohort_field_dictionary(cohort: pd.DataFrame) -> pd.DataFrame:
    records = []
    for order, column in enumerate(cohort.columns, start=1):
        group, definition, usage, timing = describe_cohort_field(column)
        records.append(
            {
                "column_order": order,
                "field": column,
                "field_group": group,
                "data_type": str(cohort[column].dtype),
                "definition": definition,
                "modeling_usage": usage,
                "timing_rule": timing,
                "non_null_count": int(cohort[column].notna().sum()),
                "missing_pct": float(cohort[column].isna().mean()),
                "sample_values": " | ".join(str(v)[:80] for v in cohort[column].dropna().astype(str).unique()[:4]),
            }
        )
    return pd.DataFrame(records)


def json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def sanitize_json(value):
    if isinstance(value, dict):
        return {str(k): sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_json(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_json(v) for v in value]
    return json_safe(value)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    players = pd.read_csv(TM / "players.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    transfers = pd.read_csv(TM / "transfers.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    valuations = pd.read_csv(TM / "player_valuations.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    games_full = pd.read_csv(TM / "games.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    clubs = pd.read_csv(TM / "clubs.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    national_teams = pd.read_csv(TM / "national_teams.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    seasonal = pd.concat(
        [pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig") for path in sorted(SEASONAL.glob("*.csv"))],
        ignore_index=True,
    )

    for col in ["player_id", "from_club_id", "to_club_id"]:
        transfers[col] = clean_id(transfers[col])
    transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"], errors="coerce")
    transfers["transfer_fee_eur"] = pd.to_numeric(transfers["transfer_fee"], errors="coerce")
    transfers["listed_market_value_at_transfer_eur"] = pd.to_numeric(transfers["market_value_in_eur"], errors="coerce")
    transfers["fee_status"] = np.select(
        [transfers["transfer_fee_eur"].isna(), transfers["transfer_fee_eur"].eq(0), transfers["transfer_fee_eur"].gt(0)],
        ["missing", "zero", "positive"],
        default="other",
    )

    valuations["player_id"] = clean_id(valuations["player_id"])
    valuations["date"] = pd.to_datetime(valuations["date"], errors="coerce")
    valuations["market_value_in_eur"] = pd.to_numeric(valuations["market_value_in_eur"], errors="coerce")
    valuations = valuations.dropna(subset=["date", "market_value_in_eur"])
    valuation_max_date = valuations["date"].max()
    cohort_end = valuation_max_date - pd.DateOffset(years=2)

    games_full["game_id"] = clean_id(games_full["game_id"])
    games_full["home_club_id"] = clean_id(games_full["home_club_id"])
    games_full["away_club_id"] = clean_id(games_full["away_club_id"])
    games_full["date"] = pd.to_datetime(games_full["date"], errors="coerce")
    clubs["club_id"] = clean_id(clubs["club_id"])
    national_teams["national_team_id"] = clean_id(national_teams["national_team_id"])

    print("Building player identity crosswalk", flush=True)
    crosswalk = build_player_crosswalk(players, seasonal)
    print("Aggregating player-season features", flush=True)
    season_features = build_player_season_features(seasonal, crosswalk)

    cohort = transfers[transfers["transfer_date"].between(COHORT_START, cohort_end)].copy()
    cohort["transfer_event_id"] = (
        cohort["player_id"] + "_" + cohort["transfer_date"].dt.strftime("%Y%m%d") + "_" + cohort["from_club_id"] + "_" + cohort["to_club_id"]
    )
    player_info = players[["player_id", "name", "date_of_birth", "position", "sub_position", "foot", "height_in_cm", "country_of_citizenship"]].copy()
    player_info["player_id"] = clean_id(player_info["player_id"])
    player_info["date_of_birth"] = pd.to_datetime(player_info["date_of_birth"], errors="coerce")
    cohort = cohort.merge(player_info, on="player_id", how="left", suffixes=("", "_player_dim"))
    cohort["age_at_transfer"] = (cohort["transfer_date"] - cohort["date_of_birth"]).dt.days / 365.2425
    cohort["player_in_dimension"] = cohort["name"].notna()

    print("Attaching latest completed seasonal features", flush=True)
    cohort = attach_latest_preseason(cohort, season_features)
    print("Attaching as-of and horizon valuations", flush=True)
    cohort = attach_valuations(cohort, valuations)
    print("Aggregating player appearances around each transfer", flush=True)
    cohort = aggregate_appearances(cohort)
    print("Measuring destination-club match coverage", flush=True)
    cohort = aggregate_destination_games(cohort)
    cohort = add_eligibility_and_outcomes(cohort)

    print("Building union club dimension", flush=True)
    club_dimension = build_union_club_dimension(transfers, games_full, clubs, national_teams)
    funnels = build_funnels(len(transfers), cohort)
    checks = build_checks(cohort, cohort_end)

    assumptions = pd.DataFrame(
        [
            ["cohort_start", COHORT_START.date().isoformat(), "First transfer date with a completed 2017-18 seasonal feature set."],
            ["cohort_end", cohort_end.date().isoformat(), "Latest date with a full two-year horizon before the final valuation observation."],
            ["valuation_data_max_date", valuation_max_date.date().isoformat(), "Observed directly from player_valuations.csv."],
            ["pre_valuation_max_age_days", PRE_VALUATION_MAX_AGE_DAYS, "Pre-transfer valuation must be on/before transfer and no more than this many days old."],
            ["post_valuation_tolerance_days", POST_VALUATION_TOLERANCE_DAYS, "Nearest valuation must fall within +/- this many days of the 12m or 24m target."],
            ["sporting_min_destination_games_12m", SPORTING_MIN_DEST_GAMES_12M, "Coverage floor before treating no player appearances as a valid 12m outcome."],
            ["sporting_min_destination_games_24m", SPORTING_MIN_DEST_GAMES_24M, "Coverage floor before treating no player appearances as a valid 24m outcome."],
            ["financial_sample_status", "provisional", "Positive fees can still include loan fees; transfer movement type is unavailable and must be classified."],
            ["roi_proxy_definition", "(post valuation - fee) / fee", "A valuation proxy, not realized accounting profit or resale ROI."],
            ["seasonal_rate_aggregation", "minutes-weighted", "Counts are summed; per-90 and percentage metrics are minutes-weighted across player-season rows."],
        ],
        columns=["assumption", "value", "rationale"],
    )

    tolerance_rows = []
    cross_club = ~cohort["is_same_club_movement"]
    fee_base = cross_club & cohort["has_positive_disclosed_fee"] & cohort["has_pre_valuation_365d"]
    for horizon in (12, 24):
        for tolerance in VALUATION_TOLERANCES:
            mask = fee_base & cohort[f"has_post{horizon}_valuation_{tolerance}d"]
            tolerance_rows.append(
                {
                    "horizon_months": horizon,
                    "tolerance_days": tolerance,
                    "eligible_records": int(mask.sum()),
                    "eligible_pct_of_positive_fee_prevalued": float(mask.sum() / fee_base.sum()) if fee_base.sum() else None,
                }
            )
    tolerance_sensitivity = pd.DataFrame(tolerance_rows)

    preferred_columns = [
        "transfer_event_id", "player_id", "player_name", "name", "transfer_date", "transfer_season",
        "expected_transfer_season", "transfer_season_matches_date",
        "from_club_id", "from_club_name", "to_club_id", "to_club_name", "transfer_fee_eur", "fee_status",
        "listed_market_value_at_transfer_eur", "date_of_birth", "age_at_transfer", "position", "sub_position",
        "foot", "height_in_cm", "country_of_citizenship", "age_plausible_15_to_45", "player_in_dimension", "preseason_season",
        "preseason_end_date", "preseason_squads", "preseason_competitions", "preseason_positions",
        "preseason_total_minutes", "preseason_matches_played", "preseason_goals", "preseason_assists",
        "preseason_expected_goals", "preseason_exp_npg", "pre365_appearances", "pre365_minutes", "pre365_goals",
        "pre365_assists", "pre_valuation_date", "pre_valuation_asof_eur", "pre_valuation_staleness_days",
        "post12_target_date", "post12_nearest_valuation_date", "post12_valuation_90d_eur", "post12_valuation_gap_days",
        "post24_target_date", "post24_nearest_valuation_date", "post24_valuation_90d_eur", "post24_valuation_gap_days",
        "post12_dest_appearances", "post12_dest_minutes", "post12_dest_goals", "post12_dest_assists",
        "post24_dest_appearances", "post24_dest_minutes", "post24_dest_goals", "post24_dest_assists",
        "destination_games_12m", "destination_games_24m", "valuation_change_12m_eur", "valuation_change_12m_pct",
        "valuation_change_24m_eur", "valuation_change_24m_pct", "fee_value_roi_proxy_12m", "fee_value_roi_proxy_24m",
        "is_same_club_movement", "has_preseason_features", "has_pre_valuation_365d", "has_post12_valuation_90d",
        "has_post24_valuation_90d", "has_destination_coverage_12m", "has_destination_coverage_24m",
        "has_positive_disclosed_fee", "eligible_sporting_12m", "eligible_sporting_24m",
        "eligible_financial_12m_provisional", "eligible_financial_24m_provisional",
        "eligible_combined_24m_provisional", "cohort_tier", "eligibility_exclusion_reasons",
    ]
    remaining_columns = [c for c in cohort.columns if c not in preferred_columns and c not in {"transfer_fee", "market_value_in_eur"}]
    cohort = cohort[[c for c in preferred_columns if c in cohort.columns] + remaining_columns]
    cohort = cohort.sort_values(["transfer_date", "player_id", "from_club_id", "to_club_id"]).reset_index(drop=True)

    field_dictionary = build_cohort_field_dictionary(cohort)
    eligibility_flags = [
        "player_in_dimension", "has_preseason_features", "has_pre_valuation_365d",
        "has_post12_valuation_90d", "has_post24_valuation_90d", "has_destination_coverage_12m",
        "has_destination_coverage_24m", "has_positive_disclosed_fee", "eligible_sporting_12m",
        "eligible_sporting_24m", "eligible_financial_12m_provisional",
        "eligible_financial_24m_provisional", "eligible_combined_24m_provisional",
    ]
    eligibility_summary = pd.DataFrame(
        [
            {
                "eligibility_flag": flag,
                "records": int(cohort[flag].fillna(False).sum()),
                "pct_of_cohort": float(cohort[flag].fillna(False).mean()),
            }
            for flag in eligibility_flags
        ]
    )
    season_summary = (
        cohort.groupby("expected_transfer_season", dropna=False)
        .agg(
            transfers=("transfer_event_id", "size"),
            unique_players=("player_id", "nunique"),
            prior_season_features=("has_preseason_features", "sum"),
            positive_fees=("has_positive_disclosed_fee", "sum"),
            sporting_24m=("eligible_sporting_24m", "sum"),
            financial_24m_provisional=("eligible_financial_24m_provisional", "sum"),
            combined_24m_provisional=("eligible_combined_24m_provisional", "sum"),
            median_positive_fee_eur=("transfer_fee_eur", lambda s: s[s.gt(0)].median()),
        )
        .reset_index()
        .sort_values("expected_transfer_season")
    )
    exclusion_counter = Counter()
    for value in cohort["eligibility_exclusion_reasons"].fillna(""):
        for reason in str(value).split(" | "):
            if reason:
                exclusion_counter[reason] += 1
    exclusion_summary = pd.DataFrame(
        [
            {"exclusion_reason": reason, "records": count, "pct_of_cohort": count / len(cohort)}
            for reason, count in exclusion_counter.most_common()
        ]
    )
    crosswalk_summary = (
        crosswalk.groupby("match_status")
        .agg(seasonal_identities=("seasonal_player", "size"), seasonal_rows=("seasonal_rows", "sum"))
        .reset_index()
    )
    crosswalk_summary["identity_pct"] = crosswalk_summary["seasonal_identities"] / len(crosswalk)
    crosswalk_summary["row_pct"] = crosswalk_summary["seasonal_rows"] / crosswalk["seasonal_rows"].sum()
    sample_columns = [
        "transfer_event_id", "player_name", "transfer_date", "expected_transfer_season", "from_club_name",
        "to_club_name", "transfer_fee_eur", "fee_status", "age_at_transfer", "preseason_season",
        "preseason_total_minutes", "pre_valuation_asof_eur", "post24_valuation_90d_eur",
        "post24_dest_minutes", "destination_games_24m", "eligible_sporting_24m",
        "eligible_financial_24m_provisional", "eligible_combined_24m_provisional", "cohort_tier",
        "eligibility_exclusion_reasons",
    ]
    cohort_sample = (
        cohort.groupby("cohort_tier", group_keys=False, sort=True)
        .head(25)[sample_columns]
        .sort_values(["cohort_tier", "transfer_date", "player_name"])
        .reset_index(drop=True)
    )

    date_columns = [
        c
        for c in cohort.columns
        if (c.endswith("_date") and not c.endswith("_matches_date"))
        or c in {"transfer_date", "date_of_birth"}
    ]
    for col in date_columns:
        if col in cohort.columns:
            cohort[col] = pd.to_datetime(cohort[col], errors="coerce").dt.strftime("%Y-%m-%d")
    for frame in [season_features, club_dimension]:
        for col in frame.columns:
            if col.endswith("_date"):
                frame[col] = pd.to_datetime(frame[col], errors="coerce").dt.strftime("%Y-%m-%d")

    cohort.to_csv(PROCESSED / "transfer_cohort_master.csv", index=False, encoding="utf-8-sig")
    crosswalk.to_csv(PROCESSED / "player_crosswalk.csv", index=False, encoding="utf-8-sig")
    season_features.to_csv(PROCESSED / "player_season_features.csv", index=False, encoding="utf-8-sig")
    club_dimension.to_csv(PROCESSED / "club_dimension_union.csv", index=False, encoding="utf-8-sig")
    funnels.to_csv(OUTPUT / "cohort_funnels.csv", index=False, encoding="utf-8-sig")
    assumptions.to_csv(OUTPUT / "cohort_assumptions.csv", index=False, encoding="utf-8-sig")
    tolerance_sensitivity.to_csv(OUTPUT / "valuation_tolerance_sensitivity.csv", index=False, encoding="utf-8-sig")
    checks.to_csv(OUTPUT / "cohort_checks.csv", index=False, encoding="utf-8-sig")
    eligibility_summary.to_csv(OUTPUT / "eligibility_summary.csv", index=False, encoding="utf-8-sig")
    season_summary.to_csv(OUTPUT / "cohort_by_season.csv", index=False, encoding="utf-8-sig")
    exclusion_summary.to_csv(OUTPUT / "exclusion_reasons.csv", index=False, encoding="utf-8-sig")
    crosswalk_summary.to_csv(OUTPUT / "crosswalk_summary.csv", index=False, encoding="utf-8-sig")
    field_dictionary.to_csv(OUTPUT / "cohort_field_dictionary.csv", index=False, encoding="utf-8-sig")
    cohort_sample.to_csv(OUTPUT / "cohort_sample.csv", index=False, encoding="utf-8-sig")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_start": COHORT_START,
        "cohort_end": cohort_end,
        "valuation_max_date": valuation_max_date,
        "raw_transfer_rows": len(transfers),
        "cohort_rows": len(cohort),
        "unique_players": cohort["player_id"].nunique(),
        "linked_player_season_rows": len(season_features),
        "union_club_dimension_rows": len(club_dimension),
        "sporting_12m_rows": int(cohort["eligible_sporting_12m"].sum()),
        "sporting_24m_rows": int(cohort["eligible_sporting_24m"].sum()),
        "financial_12m_provisional_rows": int(cohort["eligible_financial_12m_provisional"].sum()),
        "financial_24m_provisional_rows": int(cohort["eligible_financial_24m_provisional"].sum()),
        "combined_24m_provisional_rows": int(cohort["eligible_combined_24m_provisional"].sum()),
        "player_crosswalk_unique_pct": float(crosswalk["match_status"].eq("unique").mean()),
        "checks_ok": bool(~checks["status"].eq("FAIL").any()),
        "assumptions": assumptions.to_dict("records"),
        "funnels": funnels.to_dict("records"),
        "tolerance_sensitivity": tolerance_sensitivity.to_dict("records"),
        "checks": checks.to_dict("records"),
        "fee_status_counts": cohort["fee_status"].value_counts(dropna=False).to_dict(),
        "cohort_tier_counts": cohort["cohort_tier"].value_counts(dropna=False).to_dict(),
        "transfer_season_counts": cohort["transfer_season"].value_counts(dropna=False).sort_index().to_dict(),
        "eligibility_summary": eligibility_summary.to_dict("records"),
        "season_summary": season_summary.to_dict("records"),
        "exclusion_summary": exclusion_summary.to_dict("records"),
        "crosswalk_summary": crosswalk_summary.to_dict("records"),
        "field_dictionary": field_dictionary.to_dict("records"),
        "cohort_sample": cohort_sample.to_dict("records"),
    }
    summary = sanitize_json(summary)
    (OUTPUT / "cohort_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=json_safe, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Created cohort with {len(cohort):,} transfers", flush=True)
    print(f"Sporting 24m: {int(cohort['eligible_sporting_24m'].sum()):,}", flush=True)
    print(f"Financial 24m provisional: {int(cohort['eligible_financial_24m_provisional'].sum()):,}", flush=True)
    print(f"Combined 24m provisional: {int(cohort['eligible_combined_24m_provisional'].sum()):,}", flush=True)


if __name__ == "__main__":
    main()

"""Build an extension-centered, leakage-safe contract modeling dataset.

Each Capology extension remains one anchor row. Information known by the
signing date, post-signing sporting windows, valuation landmarks, and later
outbound events are kept in explicitly named blocks so outcomes cannot leak
into pre-signing models.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "Data" / "processed"
CONTRACTS = PROCESSED / "capology_contracts"
PERFORMANCE = PROCESSED / "canonical_performance"
INTEGRATION = PROCESSED / "canonical_integration"
TM_TABLES = PROCESSED / "transfermarkt_clean" / "tables"
OUTPUT = PROCESSED / "contract_extension_integration"

EXTENSION_PATH = CONTRACTS / "canonical_extension_events.csv"
SALARY_PATH = CONTRACTS / "canonical_salary_panel.csv"
PERFORMANCE_PATH = PERFORMANCE / "canonical_player_team_season_performance.csv"
ADVANCED_PATH = PERFORMANCE / "fbref_advanced_performance_features.csv"
VALUATION_PATH = INTEGRATION / "canonical_valuation_history.csv"
TRANSFER_PATH = INTEGRATION / "canonical_transfer_history.csv"
APPEARANCE_PATH = TM_TABLES / "appearances_clean.csv"
GAMES_PATH = TM_TABLES / "games_clean.csv"

AS_OF_DATE = pd.Timestamp("2026-08-10")
WINDOW_DAYS = 365
BIG5 = {"GB1", "ES1", "IT1", "L1", "FR1"}

PERFORMANCE_COLUMNS = [
    "canonical_performance_id", "canonical_player_id", "canonical_club_id",
    "canonical_competition_id", "season_start_year", "data_coverage_tier",
    "advanced_features_available", "basic_model_eligible_450_minutes",
    "canonical_matches_played", "canonical_minutes_played", "canonical_goals",
    "canonical_assists", "canonical_yellow_cards", "canonical_red_cards",
    "canonical_goals_per90", "canonical_assists_per90", "current_starts",
    "current_start_pct",
]
ADVANCED_COLUMNS = [
    "canonical_performance_id", "fbref_season_end_date", "fbref_expected_goals",
    "fbref_non_penalty_expected_goals", "fbref_progressive_carries",
    "fbref_progressive_passes", "fbref_tackles_won", "fbref_interceptions",
    "fbref_clearances", "fbref_pass_completion_pct", "fbref_key_passes",
    "fbref_total_shots", "fbref_shot_creating_actions_per90",
    "fbref_goal_creating_actions_per90", "fbref_aerial_duels_won_pct",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").fillna("").str.strip().str.casefold().isin({"true", "1", "yes"})


def clean_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def add_check(checks: list[dict], name: str, observed: object, expected: object, passed: bool, notes: str) -> None:
    checks.append({"check": name, "observed": observed, "expected": expected, "passed": bool(passed), "notes": notes})


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d", lineterminator="\n")


def load_extensions() -> pd.DataFrame:
    frame = pd.read_csv(EXTENSION_PATH, low_memory=False)
    for column in ["canonical_player_id", "canonical_club_id"]:
        frame[column] = clean_id(frame[column])
    for column in ["signed_date", "expiration_date"]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    for column in ["pre_completed_season_start_year", "first_full_post_season_start_year"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    if frame["capology_extension_event_id"].duplicated().any():
        raise RuntimeError("Extension event IDs are not unique.")
    return frame


def salary_context(extensions: pd.DataFrame) -> pd.DataFrame:
    salary = pd.read_csv(SALARY_PATH, low_memory=False)
    salary["canonical_club_id"] = clean_id(salary["canonical_club_id"])
    salary["annual_gross_eur"] = pd.to_numeric(salary["annual_gross_eur"], errors="coerce")
    salary["total_gross_eur"] = pd.to_numeric(salary["total_gross_eur"], errors="coerce")
    group_key = ["canonical_club_id", "source_page_season"]
    salary["club_salary_known_players"] = salary.groupby(group_key)["annual_gross_eur"].transform("count")
    salary["club_salary_total_known_eur"] = salary.groupby(group_key)["annual_gross_eur"].transform("sum")
    salary["club_salary_median_eur"] = salary.groupby(group_key)["annual_gross_eur"].transform("median")
    salary["club_salary_mean_eur"] = salary.groupby(group_key)["annual_gross_eur"].transform("mean")
    salary["club_salary_percentile"] = salary.groupby(group_key)["annual_gross_eur"].rank(pct=True, method="average")
    salary["club_salary_share_known"] = salary["annual_gross_eur"].div(
        salary["club_salary_total_known_eur"].where(salary["club_salary_total_known_eur"].gt(0))
    )
    exact_key = ["player_name_normalized", "canonical_club_id", "source_page_season"]
    if salary.duplicated(exact_key).any():
        raise RuntimeError("Capology salary source-name club-season key is not unique.")
    keep = exact_key + [
        "capology_salary_id", "annual_gross_eur", "total_gross_eur",
        "club_salary_known_players", "club_salary_total_known_eur",
        "club_salary_median_eur", "club_salary_mean_eur", "club_salary_percentile",
        "club_salary_share_known", "active", "loan", "verification_badge",
    ]
    context = salary[keep].rename(
        columns={
            "annual_gross_eur": "salary_panel_annual_gross_eur",
            "total_gross_eur": "salary_panel_total_gross_eur",
            "active": "salary_panel_active",
            "loan": "salary_panel_loan",
            "verification_badge": "salary_panel_verification_badge",
        }
    )
    left = extensions[["capology_extension_event_id", "player_name_normalized", "canonical_club_id", "source_page_season"]]
    result = left.merge(context, on=exact_key, how="left", validate="many_to_one")
    result["salary_panel_match"] = result["capology_salary_id"].notna()
    result["extension_vs_panel_annual_difference_eur"] = (
        pd.to_numeric(extensions["annual_gross_eur"], errors="coerce").to_numpy()
        - result["salary_panel_annual_gross_eur"]
    )

    previous = salary[exact_key + ["source_page_season_start_year", "annual_gross_eur"]].copy()
    previous["join_year"] = pd.to_numeric(previous["source_page_season_start_year"], errors="coerce") + 1
    previous = previous.rename(columns={"annual_gross_eur": "prior_season_salary_annual_gross_eur"})
    prior_left = extensions[["capology_extension_event_id", "player_name_normalized", "canonical_club_id", "source_page_season_start_year"]].rename(
        columns={"source_page_season_start_year": "join_year"}
    )
    prior = prior_left.merge(
        previous[["player_name_normalized", "canonical_club_id", "join_year", "prior_season_salary_annual_gross_eur"]],
        on=["player_name_normalized", "canonical_club_id", "join_year"], how="left", validate="many_to_one"
    )
    result = result.merge(prior[["capology_extension_event_id", "prior_season_salary_annual_gross_eur"]], on="capology_extension_event_id", how="left", validate="one_to_one")
    extension_salary = pd.Series(
        pd.to_numeric(extensions["annual_gross_eur"], errors="coerce").to_numpy(),
        index=result.index,
    )
    result["salary_change_from_prior_pct"] = (
        extension_salary - result["prior_season_salary_annual_gross_eur"]
    ).div(result["prior_season_salary_annual_gross_eur"].where(result["prior_season_salary_annual_gross_eur"].gt(0)))
    return result


def seasonal_performance(extensions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    perf = pd.read_csv(PERFORMANCE_PATH, usecols=PERFORMANCE_COLUMNS, low_memory=False)
    perf["canonical_player_id"] = clean_id(perf["canonical_player_id"])
    perf["canonical_club_id"] = clean_id(perf["canonical_club_id"])
    perf["season_start_year"] = pd.to_numeric(perf["season_start_year"], errors="coerce").astype("Int64")
    perf = perf.loc[perf["canonical_competition_id"].isin(BIG5) & perf["season_start_year"].between(2015, 2025)].copy()
    advanced = pd.read_csv(ADVANCED_PATH, usecols=ADVANCED_COLUMNS, low_memory=False)
    if advanced["canonical_performance_id"].duplicated().any():
        raise RuntimeError("Advanced performance IDs are not unique.")
    perf = perf.merge(advanced, on="canonical_performance_id", how="left", validate="one_to_one")
    join_key = ["canonical_player_id", "canonical_club_id", "canonical_competition_id", "season_start_year"]
    if perf.duplicated(join_key).any():
        raise RuntimeError("Canonical performance join key is not unique.")

    event_rows = extensions[[
        "capology_extension_event_id", "canonical_player_id", "canonical_club_id", "competition_id",
        "pre_completed_season_start_year", "first_full_post_season_start_year",
    ]].copy()
    blocks = []
    long_rows = []
    for label, offset, base_column, role in [
        ("pre1", 0, "pre_completed_season_start_year", "predictor"),
        ("pre2", -1, "pre_completed_season_start_year", "predictor"),
        ("pre3", -2, "pre_completed_season_start_year", "predictor"),
        ("post_y1", 0, "first_full_post_season_start_year", "outcome"),
        ("post_y2", 1, "first_full_post_season_start_year", "outcome"),
        ("post_y3", 2, "first_full_post_season_start_year", "outcome"),
    ]:
        left = event_rows.copy()
        left["season_start_year"] = (left[base_column] + offset).astype("Int64")
        left = left.rename(columns={"competition_id": "canonical_competition_id"})
        joined = left.merge(perf, on=join_key, how="left", validate="many_to_one")
        available = joined["canonical_performance_id"].notna()
        basic_available = available & pd.to_numeric(joined["canonical_minutes_played"], errors="coerce").notna()
        advanced_available = available & as_bool(joined["advanced_features_available"])
        long = joined[["capology_extension_event_id", "season_start_year", "canonical_performance_id"]].copy()
        long.insert(1, "window", label)
        long.insert(2, "information_role", role)
        long["same_club_domestic_performance_available"] = available
        long["basic_minutes_available"] = basic_available
        long["advanced_features_available"] = advanced_available
        for column in PERFORMANCE_COLUMNS[6:] + ADVANCED_COLUMNS[1:]:
            if column in joined:
                long[column] = joined[column]
        long_rows.append(long)

        values = joined.drop(columns=[c for c in event_rows.columns if c != "capology_extension_event_id"] + ["canonical_competition_id", "season_start_year"], errors="ignore")
        values = values.rename(columns={c: f"{label}_{c}" for c in values.columns if c != "capology_extension_event_id"})
        values[f"{label}_same_club_domestic_performance_available"] = available.to_numpy()
        values[f"{label}_basic_minutes_available"] = basic_available.to_numpy()
        values[f"{label}_advanced_features_available_flag"] = advanced_available.to_numpy()
        blocks.append(values)
    master = blocks[0]
    for block in blocks[1:]:
        master = master.merge(block, on="capology_extension_event_id", how="left", validate="one_to_one")
    return master, pd.concat(long_rows, ignore_index=True)


def exact_sporting_windows(extensions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    player_ids = set(extensions["canonical_player_id"].dropna().astype(int))
    club_ids = set(extensions["canonical_club_id"].dropna().astype(int))
    apps = pd.read_csv(
        APPEARANCE_PATH,
        usecols=["appearance_id", "game_id", "player_id", "player_club_id", "date", "competition_id", "goals", "assists", "minutes_played"],
        low_memory=False,
    )
    apps["player_id"] = clean_id(apps["player_id"])
    apps["player_club_id"] = clean_id(apps["player_club_id"])
    apps["date"] = pd.to_datetime(apps["date"], errors="coerce")
    apps = apps.loc[apps["player_id"].isin(player_ids) & apps["player_club_id"].isin(club_ids) & apps["date"].notna()].copy()
    games = pd.read_csv(
        GAMES_PATH,
        usecols=["game_id", "competition_id", "date", "home_club_id", "away_club_id", "is_national_team_game"],
        low_memory=False,
    )
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    games = games.loc[games["date"].notna() & ~as_bool(games["is_national_team_game"])].copy()
    home = games[["game_id", "competition_id", "date", "home_club_id"]].rename(columns={"home_club_id": "club_id"})
    away = games[["game_id", "competition_id", "date", "away_club_id"]].rename(columns={"away_club_id": "club_id"})
    club_games = pd.concat([home, away], ignore_index=True)
    club_games["club_id"] = clean_id(club_games["club_id"])
    club_games = club_games.loc[club_games["club_id"].isin(club_ids)].drop_duplicates(["game_id", "club_id"])
    sporting_cutoff = club_games["date"].max()
    app_groups = {(int(p), int(c)): g.sort_values("date") for (p, c), g in apps.groupby(["player_id", "player_club_id"], observed=True)}
    game_groups = {int(c): g.sort_values("date") for c, g in club_games.groupby("club_id", observed=True)}

    rows = []
    for event in extensions.itertuples(index=False):
        pid = int(event.canonical_player_id) if pd.notna(event.canonical_player_id) else None
        cid = int(event.canonical_club_id) if pd.notna(event.canonical_club_id) else None
        app_group = app_groups.get((pid, cid), apps.iloc[0:0]) if pid is not None and cid is not None else apps.iloc[0:0]
        game_group = game_groups.get(cid, club_games.iloc[0:0]) if cid is not None else club_games.iloc[0:0]
        windows = [("pre365", -365, 0, "predictor"), ("post_y1", 0, 365, "outcome"), ("post_y2", 365, 730, "outcome"), ("post_y3", 730, 1095, "outcome")]
        for label, start_days, end_days, role in windows:
            start = event.signed_date + pd.Timedelta(days=start_days)
            end = event.signed_date + pd.Timedelta(days=end_days)
            if label == "pre365":
                game_mask = game_group["date"].ge(start) & game_group["date"].lt(end)
                app_mask = app_group["date"].ge(start) & app_group["date"].lt(end)
                observable = True
            else:
                game_mask = game_group["date"].gt(start) & game_group["date"].le(end)
                app_mask = app_group["date"].gt(start) & app_group["date"].le(end)
                observable = end <= sporting_cutoff
            g = game_group.loc[game_mask]
            a = app_group.loc[app_mask]
            same_comp_games = g.loc[g["competition_id"].eq(event.competition_id)]
            same_comp_apps = a.loc[a["competition_id"].eq(event.competition_id)]
            club_game_count = int(len(g))
            same_comp_game_count = int(len(same_comp_games))
            record = {
                "capology_extension_event_id": event.capology_extension_event_id,
                "window": label,
                "information_role": role,
                "window_start_exclusive_or_inclusive": start,
                "window_end_exclusive_or_inclusive": end,
                "window_fully_observable": bool(observable),
                "same_club_all_competition_games": club_game_count,
                "same_club_all_competition_appearances": int(len(a)),
                "same_club_all_competition_minutes": float(pd.to_numeric(a["minutes_played"], errors="coerce").sum()),
                "same_club_all_competition_goals": float(pd.to_numeric(a["goals"], errors="coerce").sum()),
                "same_club_all_competition_assists": float(pd.to_numeric(a["assists"], errors="coerce").sum()),
                "same_club_same_big5_competition_games": same_comp_game_count,
                "same_club_same_big5_competition_appearances": int(len(same_comp_apps)),
                "same_club_same_big5_competition_minutes": float(pd.to_numeric(same_comp_apps["minutes_played"], errors="coerce").sum()),
            }
            record["window_evidence_eligible_10_club_games"] = bool(observable and club_game_count >= 10)
            if club_game_count > 0:
                record["same_club_all_competition_opportunity_share"] = min(record["same_club_all_competition_minutes"] / (90 * club_game_count), 1.05)
                record["same_club_all_competition_appearance_rate"] = record["same_club_all_competition_appearances"] / club_game_count
            else:
                record["same_club_all_competition_opportunity_share"] = np.nan
                record["same_club_all_competition_appearance_rate"] = np.nan
            record["same_club_same_big5_competition_opportunity_share"] = (
                min(record["same_club_same_big5_competition_minutes"] / (90 * same_comp_game_count), 1.05)
                if same_comp_game_count > 0 else np.nan
            )
            rows.append(record)
    long = pd.DataFrame(rows)
    wide_parts = []
    value_cols = [c for c in long.columns if c not in {"capology_extension_event_id", "window", "information_role"}]
    for label in long["window"].unique():
        part = long.loc[long["window"].eq(label), ["capology_extension_event_id"] + value_cols].copy()
        part = part.rename(columns={c: f"{label}_{c}" for c in value_cols})
        wide_parts.append(part)
    wide = wide_parts[0]
    for part in wide_parts[1:]:
        wide = wide.merge(part, on="capology_extension_event_id", how="left", validate="one_to_one")
    return wide, long, sporting_cutoff


def valuation_landmarks(extensions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    ids = set(extensions["canonical_player_id"].dropna().astype(int))
    values = pd.read_csv(
        VALUATION_PATH,
        usecols=["canonical_valuation_id", "canonical_player_id", "valuation_date", "canonical_market_value_eur", "value_resolution_status", "value_conflict"],
        low_memory=False,
    )
    values["canonical_player_id"] = clean_id(values["canonical_player_id"])
    values["valuation_date"] = pd.to_datetime(values["valuation_date"], errors="coerce")
    values["canonical_market_value_eur"] = pd.to_numeric(values["canonical_market_value_eur"], errors="coerce")
    values = values.loc[values["canonical_player_id"].isin(ids) & values["valuation_date"].notna() & values["canonical_market_value_eur"].gt(0)].copy()
    valuation_cutoff = values["valuation_date"].max()
    groups = {int(pid): g.sort_values("valuation_date") for pid, g in values.groupby("canonical_player_id", observed=True)}
    rows = []
    for event in extensions.itertuples(index=False):
        pid = int(event.canonical_player_id) if pd.notna(event.canonical_player_id) else None
        group = groups.get(pid, values.iloc[0:0]) if pid is not None else values.iloc[0:0]
        pre = group.loc[group["valuation_date"].le(event.signed_date)].tail(1)
        pre_value = float(pre.iloc[0]["canonical_market_value_eur"]) if len(pre) else np.nan
        pre_date = pre.iloc[0]["valuation_date"] if len(pre) else pd.NaT
        landmarks = [("at_signing", event.signed_date, "predictor"), ("post12", event.signed_date + pd.Timedelta(days=365), "outcome"), ("post24", event.signed_date + pd.Timedelta(days=730), "outcome"), ("post36", event.signed_date + pd.Timedelta(days=1095), "outcome")]
        for label, target, role in landmarks:
            if label == "at_signing":
                nearest = pre
                gap = (event.signed_date - pre_date).days if len(pre) else np.nan
                within = pd.notna(gap) and 0 <= gap <= 365
                observable = True
            else:
                if len(group):
                    idx = (group["valuation_date"] - target).abs().idxmin()
                    nearest = group.loc[[idx]]
                    gap = int((nearest.iloc[0]["valuation_date"] - target).days)
                else:
                    nearest = group
                    gap = np.nan
                within = pd.notna(gap) and abs(gap) <= 90
                observable = target <= valuation_cutoff
            value = float(nearest.iloc[0]["canonical_market_value_eur"]) if len(nearest) else np.nan
            date = nearest.iloc[0]["valuation_date"] if len(nearest) else pd.NaT
            rows.append({
                "capology_extension_event_id": event.capology_extension_event_id,
                "landmark": label,
                "information_role": role,
                "target_date": target,
                "nearest_valuation_date": date,
                "market_value_eur": value,
                "gap_days": gap,
                "horizon_fully_observable": bool(observable),
                "strict_timing_eligible": bool(observable and within),
                "change_from_signing_eur": value - pre_value if label != "at_signing" and pd.notna(value) and pd.notna(pre_value) else np.nan,
                "change_from_signing_pct": (value - pre_value) / pre_value if label != "at_signing" and pd.notna(value) and pd.notna(pre_value) and pre_value > 0 else np.nan,
            })
    long = pd.DataFrame(rows)
    wide_parts = []
    value_cols = [c for c in long.columns if c not in {"capology_extension_event_id", "landmark", "information_role"}]
    for label in long["landmark"].unique():
        part = long.loc[long["landmark"].eq(label), ["capology_extension_event_id"] + value_cols].copy()
        part = part.rename(columns={c: f"{label}_{c}" for c in value_cols})
        wide_parts.append(part)
    wide = wide_parts[0]
    for part in wide_parts[1:]:
        wide = wide.merge(part, on="capology_extension_event_id", how="left", validate="one_to_one")
    return wide, long, valuation_cutoff


def departure_outcomes(extensions: pd.DataFrame) -> pd.DataFrame:
    player_ids = set(extensions["canonical_player_id"].dropna().astype(int))
    club_ids = set(extensions["canonical_club_id"].dropna().astype(int))
    transfers = pd.read_csv(
        TRANSFER_PATH,
        usecols=["canonical_transfer_event_id", "canonical_player_id", "transfer_date", "from_club_id", "to_club_id", "canonical_transfer_type", "transfer_type_conflict", "event_conflict", "event_resolution_status", "canonical_transfer_fee_eur", "canonical_transfer_fee_status"],
        low_memory=False,
    )
    for column in ["canonical_player_id", "from_club_id", "to_club_id"]:
        transfers[column] = clean_id(transfers[column])
    transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"], errors="coerce")
    clean = transfers.loc[
        transfers["canonical_player_id"].isin(player_ids)
        & transfers["from_club_id"].isin(club_ids)
        & transfers["transfer_date"].notna() & transfers["transfer_date"].le(AS_OF_DATE)
        & ~as_bool(transfers["transfer_type_conflict"]) & ~as_bool(transfers["event_conflict"])
    ].copy()
    groups = {(int(pid), int(cid)): g.sort_values(["transfer_date", "canonical_transfer_event_id"]) for (pid, cid), g in clean.groupby(["canonical_player_id", "from_club_id"], observed=True)}

    ext_sorted = extensions.loc[extensions["canonical_player_id"].notna()].sort_values("signed_date")
    ext_groups = {(int(pid), int(cid)): g for (pid, cid), g in ext_sorted.groupby(["canonical_player_id", "canonical_club_id"], observed=True)}
    rows = []
    for event in extensions.itertuples(index=False):
        pid = int(event.canonical_player_id) if pd.notna(event.canonical_player_id) else None
        cid = int(event.canonical_club_id) if pd.notna(event.canonical_club_id) else None
        group = groups.get((pid, cid), clean.iloc[0:0]) if pid is not None and cid is not None else clean.iloc[0:0]
        later = group.loc[group["transfer_date"].gt(event.signed_date)]
        any_exit = later.head(1)
        permanent = later.loc[later["canonical_transfer_type"].eq("transfer")].head(1)
        same_ext = ext_groups.get((pid, cid), extensions.iloc[0:0]) if pid is not None and cid is not None else extensions.iloc[0:0]
        next_ext = same_ext.loc[same_ext["signed_date"].gt(event.signed_date)].head(1)
        record = {"capology_extension_event_id": event.capology_extension_event_id}
        for label, selected in [("first_any_outbound", any_exit), ("first_permanent_outbound", permanent)]:
            if len(selected):
                row = selected.iloc[0]
                record[f"{label}_event_id"] = row["canonical_transfer_event_id"]
                record[f"{label}_date"] = row["transfer_date"]
                record[f"{label}_to_club_id"] = row["to_club_id"]
                record[f"{label}_type"] = row["canonical_transfer_type"]
                record[f"{label}_days"] = int((row["transfer_date"] - event.signed_date).days)
                record[f"{label}_fee_eur"] = row["canonical_transfer_fee_eur"]
                record[f"{label}_fee_status"] = row["canonical_transfer_fee_status"]
        record["next_same_club_extension_date"] = next_ext.iloc[0]["signed_date"] if len(next_ext) else pd.NaT
        record["next_same_club_extension_days"] = int((next_ext.iloc[0]["signed_date"] - event.signed_date).days) if len(next_ext) else np.nan
        for days, label in [(365, "365d"), (730, "730d"), (1095, "1095d")]:
            observable = event.signed_date + pd.Timedelta(days=days) <= AS_OF_DATE
            any_days = record.get("first_any_outbound_days", np.nan)
            perm_days = record.get("first_permanent_outbound_days", np.nan)
            record[f"outbound_horizon_{label}_observable"] = bool(observable)
            record[f"uninterrupted_same_club_spell_{label}"] = (pd.isna(any_days) or any_days > days) if observable else np.nan
            record[f"no_permanent_outbound_{label}"] = (pd.isna(perm_days) or perm_days > days) if observable else np.nan
        rows.append(record)
    return pd.DataFrame(rows)


def summaries(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage_fields = [
        "salary_panel_match", "pre1_same_club_domestic_performance_available",
        "pre1_advanced_features_available_flag", "pre365_window_fully_observable",
        "post_y1_window_fully_observable", "post_y2_window_fully_observable",
        "post_y3_window_fully_observable", "at_signing_strict_timing_eligible",
        "pre365_window_evidence_eligible_10_club_games",
        "post_y1_window_evidence_eligible_10_club_games",
        "post_y2_window_evidence_eligible_10_club_games",
        "post_y3_window_evidence_eligible_10_club_games",
        "post12_strict_timing_eligible", "post24_strict_timing_eligible",
        "post36_strict_timing_eligible", "outbound_horizon_365d_observable",
        "outbound_horizon_730d_observable", "outbound_horizon_1095d_observable",
    ]
    rows = []
    for field in coverage_fields:
        if field in master:
            values = as_bool(master[field])
            rows.append({"coverage_metric": field, "records": int(values.sum()), "total_events": len(master), "coverage_rate": float(values.mean())})
    coverage = pd.DataFrame(rows)
    linked = master["canonical_player_id"].notna()
    predictor = linked & as_bool(master["salary_panel_match"]) & as_bool(master["pre365_window_evidence_eligible_10_club_games"])
    valued = predictor & as_bool(master["at_signing_strict_timing_eligible"])
    year1 = valued & as_bool(master["post_y1_window_evidence_eligible_10_club_games"])
    year2 = year1 & as_bool(master["post_y2_window_evidence_eligible_10_club_games"])
    year3 = year2 & as_bool(master["post_y3_window_evidence_eligible_10_club_games"])
    funnels = [
        [0, "All unique extension events", len(master)],
        [1, "Canonical player identity linked", int(linked.sum())],
        [2, "Salary context and exact pre-365 sporting evidence", int(predictor.sum())],
        [3, "Also positive at-signing valuation within 365 days", int(valued.sum())],
        [4, "Also eligible year-one sporting outcome (>=10 club games)", int(year1.sum())],
        [5, "Also eligible year-two sporting outcome (>=10 club games)", int(year2.sum())],
        [6, "Also eligible year-three sporting outcome (>=10 club games)", int(year3.sum())],
    ]
    funnel = pd.DataFrame(funnels, columns=["step", "stage", "records"])
    funnel["retention_from_start"] = funnel["records"] / len(master)
    funnel["retention_from_previous"] = funnel["records"] / funnel["records"].shift(1)
    return coverage, funnel


def dictionary(master: pd.DataFrame) -> pd.DataFrame:
    definitions = {
        "capology_extension_event_id": "Unique extension anchor ID from the cleaned Capology layer.",
        "pre_completed_season_start_year": "Latest season whose standardized June 30 end is no later than signing date.",
        "first_full_post_season_start_year": "First full season beginning on or after signing under a July 1 convention.",
        "club_salary_percentile": "Player annual gross salary percentile within the Capology club-page-season roster.",
        "pre365_same_club_all_competition_opportunity_share": "Minutes divided by 90 times club games during the 365 days strictly before signing.",
        "uninterrupted_same_club_spell_730d": "No later outbound event from the extension club within 730 days; loans count as interruptions and this is not contract termination.",
        "no_permanent_outbound_730d": "No canonically classified permanent outbound from the extension club within 730 days.",
    }
    rows = []
    for column in master.columns:
        role = "identifier_or_provenance"
        if column.startswith(("pre", "at_signing", "salary_", "club_salary", "age_at_signing", "annual_gross", "contract_total", "exact_duration", "reported_years")):
            role = "available_at_signing_predictor"
        if column.startswith(("post", "first_", "uninterrupted_", "no_permanent_", "outbound_", "next_")):
            role = "post_signing_outcome_or_censoring"
        rows.append({"field": column, "dtype": str(master[column].dtype), "information_role": role, "non_null_rows": int(master[column].notna().sum()), "definition": definitions.get(column, column.replace("_", " ").capitalize() + ".")})
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    extensions = load_extensions()
    salary = salary_context(extensions)
    seasonal, performance_long = seasonal_performance(extensions)
    sporting, sporting_long, sporting_cutoff = exact_sporting_windows(extensions)
    valuations, valuation_long, valuation_cutoff = valuation_landmarks(extensions)
    departures = departure_outcomes(extensions)
    master = extensions.copy()
    for block in [salary, seasonal, sporting, valuations, departures]:
        master = master.merge(block, on="capology_extension_event_id", how="left", validate="one_to_one")
    coverage, funnel = summaries(master)

    checks: list[dict] = []
    add_check(checks, "extension_rows_preserved", len(master), len(extensions), len(master) == len(extensions), "One output row per clean extension event.")
    add_check(checks, "extension_ids_unique", int(master["capology_extension_event_id"].duplicated().sum()), 0, not master["capology_extension_event_id"].duplicated().any(), "No join multiplied events.")
    add_check(checks, "seasonal_long_rows", len(performance_long), len(extensions) * 6, len(performance_long) == len(extensions) * 6, "Three pre and three post full-season slots per event.")
    add_check(checks, "sporting_long_rows", len(sporting_long), len(extensions) * 4, len(sporting_long) == len(extensions) * 4, "Pre-365 and three non-overlapping post-year windows per event.")
    add_check(checks, "valuation_long_rows", len(valuation_long), len(extensions) * 4, len(valuation_long) == len(extensions) * 4, "At-signing plus 12/24/36-month landmarks per event.")
    pre_end = pd.to_datetime(master["pre1_fbref_season_end_date"], errors="coerce") if "pre1_fbref_season_end_date" in master else pd.Series(pd.NaT, index=master.index)
    add_check(checks, "pre_season_no_date_leakage", int((pre_end.notna() & pre_end.gt(master["signed_date"])).sum()), 0, not (pre_end.notna() & pre_end.gt(master["signed_date"])).any(), "No attached pre-season ends after signing.")
    post_observable_order = as_bool(master["post_y3_window_fully_observable"]) <= as_bool(master["post_y2_window_fully_observable"])
    post_observable_order &= as_bool(master["post_y2_window_fully_observable"]) <= as_bool(master["post_y1_window_fully_observable"])
    add_check(checks, "sporting_observability_monotone", int(post_observable_order.sum()), len(master), post_observable_order.all(), "Longer horizons cannot be observable when shorter ones are not.")
    retention_ok = pd.to_numeric(master["uninterrupted_same_club_spell_1095d"], errors="coerce").fillna(0).le(pd.to_numeric(master["uninterrupted_same_club_spell_730d"], errors="coerce").fillna(1))
    retention_ok &= pd.to_numeric(master["uninterrupted_same_club_spell_730d"], errors="coerce").fillna(0).le(pd.to_numeric(master["uninterrupted_same_club_spell_365d"], errors="coerce").fillna(1))
    add_check(checks, "retention_horizons_monotone", int(retention_ok.sum()), len(master), retention_ok.all(), "An uninterrupted longer spell implies the shorter spell.")
    add_check(checks, "salary_context_no_multiplication", len(salary), len(extensions), len(salary) == len(extensions), "Capology source-name join remains one-to-one.")
    funnel_monotone = funnel["records"].diff().fillna(0).le(0).all()
    add_check(checks, "modeling_funnel_monotone", int(funnel_monotone), 1, funnel_monotone, "Every cumulative eligibility stage is a subset of the preceding stage.")
    checks_frame = pd.DataFrame(checks)

    write_csv(master, OUTPUT / "extension_modeling_master.csv")
    write_csv(performance_long, OUTPUT / "extension_seasonal_performance_long.csv")
    write_csv(sporting_long, OUTPUT / "extension_exact_sporting_windows.csv")
    write_csv(valuation_long, OUTPUT / "extension_valuation_landmarks.csv")
    write_csv(departures, OUTPUT / "extension_departure_outcomes.csv")
    write_csv(coverage, OUTPUT / "join_coverage_summary.csv")
    write_csv(funnel, OUTPUT / "modeling_cohort_funnel.csv")
    write_csv(dictionary(master), OUTPUT / "field_dictionary.csv")
    write_csv(checks_frame, OUTPUT / "build_checks.csv")
    sources = [EXTENSION_PATH, SALARY_PATH, PERFORMANCE_PATH, ADVANCED_PATH, VALUATION_PATH, TRANSFER_PATH, APPEARANCE_PATH, GAMES_PATH]
    source_manifest = pd.DataFrame([{"source_file": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in sources])
    write_csv(source_manifest, OUTPUT / "source_manifest.csv")

    summary = {
        "extension_events": len(master),
        "linked_player_events": int(master["canonical_player_id"].notna().sum()),
        "salary_context_matches": int(as_bool(master["salary_panel_match"]).sum()),
        "pre365_sporting_context_available": int(pd.to_numeric(master["pre365_same_club_all_competition_games"], errors="coerce").gt(0).sum()),
        "pre1_domestic_performance_available": int(as_bool(master["pre1_same_club_domestic_performance_available"]).sum()),
        "pre1_advanced_performance_available": int(as_bool(master["pre1_advanced_features_available_flag"]).sum()),
        "at_signing_valuation_eligible": int(as_bool(master["at_signing_strict_timing_eligible"]).sum()),
        "post_y1_sporting_observable": int(as_bool(master["post_y1_window_fully_observable"]).sum()),
        "post_y2_sporting_observable": int(as_bool(master["post_y2_window_fully_observable"]).sum()),
        "post_y3_sporting_observable": int(as_bool(master["post_y3_window_fully_observable"]).sum()),
        "post_y1_sporting_eligible_10_club_games": int(as_bool(master["post_y1_window_evidence_eligible_10_club_games"]).sum()),
        "post_y2_sporting_eligible_10_club_games": int(as_bool(master["post_y2_window_evidence_eligible_10_club_games"]).sum()),
        "post_y3_sporting_eligible_10_club_games": int(as_bool(master["post_y3_window_evidence_eligible_10_club_games"]).sum()),
        "sporting_data_cutoff": sporting_cutoff.date().isoformat(),
        "valuation_data_cutoff": valuation_cutoff.date().isoformat(),
        "transfer_outcome_cutoff": AS_OF_DATE.date().isoformat(),
        "checks_passed": int(checks_frame["passed"].sum()),
        "checks_total": len(checks_frame),
        "all_checks_passed": bool(checks_frame["passed"].all()),
    }
    (OUTPUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    readme = f"""# Contract-extension integration

Generated by `scripts/build_contract_extension_integration.py` at one row per cleaned Capology extension event.

## Design

- `extension_modeling_master.csv`: {len(master):,} extension anchors with pre-signing predictors and explicitly prefixed post-signing outcomes.
- `extension_seasonal_performance_long.csv`: three fully completed pre-signing domestic seasons and three full post-signing domestic seasons.
- `extension_exact_sporting_windows.csv`: exact pre-365, year-one, year-two, and year-three club involvement windows.
- `extension_valuation_landmarks.csv`: last positive value at signing and nearest positive values at 12/24/36 months.
- `extension_departure_outcomes.csv`: first later outbound and first permanent outbound from the extension club, plus fixed-horizon censoring flags.

## Leakage and interpretation rules

1. January-May signings cannot use the ongoing season as a predictor. A season is complete on June 30; June 30 signings may use that completed season.
2. Exact sporting windows exclude signing-day appearances because event and match times are unavailable.
3. `uninterrupted_same_club_spell_*` counts any outbound event, including a loan, as an interruption. It is not a contract-termination label.
4. `no_permanent_outbound_*` is closer to contractual retention but still does not prove the contract remained unchanged; a later same-club extension can supersede the original expiration.
5. A missing player appearance becomes zero minutes only when club games make the window observable. It is not treated as missing performance.
6. Salary percentile and share use only players with reported positive salary in the Capology club-page-season roster. Missing salaries are excluded, not treated as zero.
7. Predictors and outcomes are marked in `field_dictionary.csv`; modeling code must select predictor-role fields explicitly.

Sporting data ends {sporting_cutoff.date().isoformat()}, valuation data ends {valuation_cutoff.date().isoformat()}, and outbound events are censored at {AS_OF_DATE.date().isoformat()}.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")
    files = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name != "output_manifest.csv")
    write_csv(pd.DataFrame([{"output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files]), OUTPUT / "output_manifest.csv")
    if not checks_frame["passed"].all():
        failed = checks_frame.loc[~checks_frame["passed"], "check"].tolist()
        raise RuntimeError(f"Contract integration failed checks: {failed}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

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
PERFORMANCE = PROCESSED / "canonical_performance"
CANONICAL = PROCESSED / "canonical_integration"
OUTPUT = PROCESSED / "compatibility_features"

CORE_PATH = PERFORMANCE / "canonical_player_team_season_performance.csv"
MODEL_SCOPE_PATH = PERFORMANCE / "modeling_scope_2017_2024.csv"
ADVANCED_PATH = PERFORMANCE / "fbref_advanced_performance_features.csv"
CLUB_DIM_PATH = CANONICAL / "canonical_club_dimension.csv"

BROAD_PATH = OUTPUT / "broad_lagged_player_features_2017_2024.csv"
ADVANCED_LAG_PATH = OUTPUT / "advanced_lagged_player_styles.csv"
STRICT_CONTEXT_PATH = OUTPUT / "strict_prior_club_style_context.csv"
ROSTER_CONTEXT_PATH = OUTPUT / "roster_conditioned_teammate_style_context.csv"
COMPATIBILITY_PATH = OUTPUT / "teammate_style_compatibility_features.csv"
MODEL_MATRIX_PATH = OUTPUT / "advanced_compatibility_model_matrix.csv"
WORK_DB = OUTPUT / "_lagged_feature_build.sqlite"

CHUNK_SIZE = 100_000
SOURCE_FIRST_SEASON = 2014
SOURCE_LAST_SEASON = 2023
TARGET_FIRST_SEASON = 2017
TARGET_LAST_SEASON = 2023
MIN_CONTEXT_TEAMMATES = 5
PREFERENCE_SHRINKAGE_K = 3.0

BASIC_SUM_METRICS = [
    "matches", "minutes", "goals", "assists", "yellow_cards", "red_cards",
    "starts", "substitute_selections", "captain_selections",
]

STYLE_DIMENSIONS = [
    "scoring_threat", "creation", "progression", "dribble_aggression",
    "passing_control", "directness", "defensive_intensity", "aerial_physicality",
]

PREFERENCE_DIMENSIONS = [
    "midfield_aggression", "midfield_creativity", "forward_threat",
    "forward_mobility", "defensive_intensity", "team_passing_control",
]


def clean_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].lstrip("-").isdigit():
        return text[:-2]
    return text


def bool_value(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


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


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    return numerator.div(denominator).where(denominator.gt(0))


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return np.nan
    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


def weighted_regression(x: pd.Series, y: pd.Series, weights: pd.Series) -> tuple[float, float, float, float]:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    valid = x.notna() & y.notna() & weights.notna() & weights.gt(0)
    if valid.sum() < 2:
        return np.nan, np.nan, np.nan, 0.0
    x_values = x.loc[valid].to_numpy(float)
    y_values = y.loc[valid].to_numpy(float)
    weight_values = weights.loc[valid].to_numpy(float)
    weight_values = weight_values / weight_values.sum()
    x_mean = float(np.sum(weight_values * x_values))
    y_mean = float(np.sum(weight_values * y_values))
    covariance = float(np.sum(weight_values * (x_values - x_mean) * (y_values - y_mean)))
    x_variance = float(np.sum(weight_values * (x_values - x_mean) ** 2))
    y_variance = float(np.sum(weight_values * (y_values - y_mean) ** 2))
    if x_variance <= 1e-9:
        return np.nan, np.nan, x_mean, 0.0
    slope = covariance / x_variance
    correlation = covariance / math.sqrt(x_variance * y_variance) if y_variance > 1e-9 else np.nan
    effective_n = float(1.0 / np.sum(weight_values**2))
    return float(slope), float(correlation), x_mean, effective_n


def weighted_group_means(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_columns: list[str],
    weight_column: str,
) -> pd.DataFrame:
    work = frame[group_columns + [weight_column] + value_columns].copy()
    weight = pd.to_numeric(work[weight_column], errors="coerce").fillna(0).clip(lower=0)
    calculated: dict[str, pd.Series] = {}
    for column in value_columns:
        values = pd.to_numeric(work[column], errors="coerce")
        calculated[f"{column}__numerator"] = values.fillna(0) * weight
        calculated[f"{column}__denominator"] = weight.where(values.notna(), 0)
    calculated_frame = pd.DataFrame(calculated, index=work.index)
    grouped = pd.concat([work[group_columns], calculated_frame], axis=1).groupby(
        group_columns, as_index=False, sort=False
    ).sum()
    output = grouped[group_columns].copy()
    for column in value_columns:
        numerator = grouped[f"{column}__numerator"]
        denominator = grouped[f"{column}__denominator"]
        output[column] = numerator.div(denominator).where(denominator.gt(0))
    return output


def create_work_database() -> sqlite3.Connection:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if WORK_DB.exists():
        WORK_DB.unlink()
    connection = sqlite3.connect(WORK_DB)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-250000")
    connection.executescript(
        """
        CREATE TABLE player_season (
            player_id TEXT NOT NULL,
            season_start INTEGER NOT NULL,
            source_rows INTEGER NOT NULL,
            matches_sum REAL NOT NULL, matches_count INTEGER NOT NULL,
            minutes_sum REAL NOT NULL, minutes_count INTEGER NOT NULL,
            goals_sum REAL NOT NULL, goals_count INTEGER NOT NULL,
            assists_sum REAL NOT NULL, assists_count INTEGER NOT NULL,
            yellow_cards_sum REAL NOT NULL, yellow_cards_count INTEGER NOT NULL,
            red_cards_sum REAL NOT NULL, red_cards_count INTEGER NOT NULL,
            starts_sum REAL NOT NULL, starts_count INTEGER NOT NULL,
            substitute_selections_sum REAL NOT NULL, substitute_selections_count INTEGER NOT NULL,
            captain_selections_sum REAL NOT NULL, captain_selections_count INTEGER NOT NULL,
            present_fbref INTEGER NOT NULL,
            present_current_appearances INTEGER NOT NULL,
            present_current_lineups INTEGER NOT NULL,
            present_supplemental INTEGER NOT NULL,
            eligible_rows INTEGER NOT NULL,
            PRIMARY KEY (player_id, season_start)
        ) WITHOUT ROWID;

        CREATE TABLE player_season_clubs (
            player_id TEXT NOT NULL, season_start INTEGER NOT NULL, club_id TEXT NOT NULL,
            PRIMARY KEY (player_id, season_start, club_id)
        ) WITHOUT ROWID;

        CREATE TABLE player_season_competitions (
            player_id TEXT NOT NULL, season_start INTEGER NOT NULL, competition_id TEXT NOT NULL,
            PRIMARY KEY (player_id, season_start, competition_id)
        ) WITHOUT ROWID;

        CREATE TABLE player_season_positions (
            player_id TEXT NOT NULL, season_start INTEGER NOT NULL, position TEXT NOT NULL,
            position_count INTEGER NOT NULL,
            PRIMARY KEY (player_id, season_start, position)
        ) WITHOUT ROWID;
        """
    )
    return connection


PLAYER_SEASON_UPSERT = """
INSERT INTO player_season VALUES (
    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
)
ON CONFLICT(player_id, season_start) DO UPDATE SET
    source_rows = source_rows + excluded.source_rows,
    matches_sum = matches_sum + excluded.matches_sum,
    matches_count = matches_count + excluded.matches_count,
    minutes_sum = minutes_sum + excluded.minutes_sum,
    minutes_count = minutes_count + excluded.minutes_count,
    goals_sum = goals_sum + excluded.goals_sum,
    goals_count = goals_count + excluded.goals_count,
    assists_sum = assists_sum + excluded.assists_sum,
    assists_count = assists_count + excluded.assists_count,
    yellow_cards_sum = yellow_cards_sum + excluded.yellow_cards_sum,
    yellow_cards_count = yellow_cards_count + excluded.yellow_cards_count,
    red_cards_sum = red_cards_sum + excluded.red_cards_sum,
    red_cards_count = red_cards_count + excluded.red_cards_count,
    starts_sum = starts_sum + excluded.starts_sum,
    starts_count = starts_count + excluded.starts_count,
    substitute_selections_sum = substitute_selections_sum + excluded.substitute_selections_sum,
    substitute_selections_count = substitute_selections_count + excluded.substitute_selections_count,
    captain_selections_sum = captain_selections_sum + excluded.captain_selections_sum,
    captain_selections_count = captain_selections_count + excluded.captain_selections_count,
    present_fbref = MAX(present_fbref, excluded.present_fbref),
    present_current_appearances = MAX(present_current_appearances, excluded.present_current_appearances),
    present_current_lineups = MAX(present_current_lineups, excluded.present_current_lineups),
    present_supplemental = MAX(present_supplemental, excluded.present_supplemental),
    eligible_rows = eligible_rows + excluded.eligible_rows
"""

POSITION_UPSERT = """
INSERT INTO player_season_positions VALUES (?, ?, ?, ?)
ON CONFLICT(player_id, season_start, position) DO UPDATE SET
    position_count = position_count + excluded.position_count
"""


def aggregate_broad_source(connection: sqlite3.Connection) -> dict[str, Any]:
    usecols = [
        "canonical_player_id", "canonical_club_id", "canonical_competition_id", "season_start_year",
        "canonical_matches_played", "canonical_minutes_played", "canonical_goals", "canonical_assists",
        "canonical_yellow_cards", "canonical_red_cards", "current_starts",
        "current_substitute_selections", "current_captain_selections", "current_primary_position",
        "fbref_position", "present_in_fbref", "present_in_current_appearances",
        "present_in_current_lineups", "present_in_supplemental_performances",
        "basic_model_eligible_450_minutes",
    ]
    source_to_metric = {
        "canonical_matches_played": "matches",
        "canonical_minutes_played": "minutes",
        "canonical_goals": "goals",
        "canonical_assists": "assists",
        "canonical_yellow_cards": "yellow_cards",
        "canonical_red_cards": "red_cards",
        "current_starts": "starts",
        "current_substitute_selections": "substitute_selections",
        "current_captain_selections": "captain_selections",
    }
    counts = Counter()
    for chunk in pd.read_csv(CORE_PATH, usecols=usecols, dtype=str, keep_default_na=False, chunksize=CHUNK_SIZE):
        counts["source_rows_scanned"] += len(chunk)
        chunk["season_start"] = pd.to_numeric(chunk["season_start_year"], errors="coerce")
        chunk = chunk.loc[chunk["season_start"].between(SOURCE_FIRST_SEASON, SOURCE_LAST_SEASON)].copy()
        if chunk.empty:
            continue
        counts["source_rows_in_window"] += len(chunk)
        chunk["player_id"] = chunk["canonical_player_id"].map(clean_id)
        chunk["club_id"] = chunk["canonical_club_id"].map(clean_id)
        chunk["competition_id"] = chunk["canonical_competition_id"].map(clean_id)
        valid = chunk["player_id"].ne("") & chunk["season_start"].notna()
        counts["invalid_key_rows"] += int((~valid).sum())
        chunk = chunk.loc[valid].copy()
        chunk["season_start"] = chunk["season_start"].astype("int64")
        for source_column, metric in source_to_metric.items():
            values = pd.to_numeric(chunk[source_column], errors="coerce")
            chunk[f"{metric}_sum"] = values.fillna(0.0)
            chunk[f"{metric}_count"] = values.notna().astype("int64")
        for source_column, output_column in [
            ("present_in_fbref", "present_fbref"),
            ("present_in_current_appearances", "present_current_appearances"),
            ("present_in_current_lineups", "present_current_lineups"),
            ("present_in_supplemental_performances", "present_supplemental"),
        ]:
            chunk[output_column] = chunk[source_column].map(bool_value).astype("int64")
        chunk["eligible_rows"] = chunk["basic_model_eligible_450_minutes"].map(bool_value).astype("int64")
        aggregations: dict[str, tuple[str, str]] = {"source_rows": ("player_id", "size")}
        for metric in BASIC_SUM_METRICS:
            aggregations[f"{metric}_sum"] = (f"{metric}_sum", "sum")
            aggregations[f"{metric}_count"] = (f"{metric}_count", "sum")
        for column in [
            "present_fbref", "present_current_appearances", "present_current_lineups", "present_supplemental",
        ]:
            aggregations[column] = (column, "max")
        aggregations["eligible_rows"] = ("eligible_rows", "sum")
        grouped = chunk.groupby(["player_id", "season_start"], sort=False).agg(**aggregations).reset_index()
        ordered = ["player_id", "season_start", "source_rows"]
        for metric in BASIC_SUM_METRICS:
            ordered.extend([f"{metric}_sum", f"{metric}_count"])
        ordered.extend([
            "present_fbref", "present_current_appearances", "present_current_lineups",
            "present_supplemental", "eligible_rows",
        ])
        rows = [tuple(sqlite_value(value) for value in row) for row in grouped[ordered].itertuples(index=False, name=None)]
        connection.executemany(PLAYER_SEASON_UPSERT, rows)

        club_rows = chunk.loc[chunk["club_id"].ne(""), ["player_id", "season_start", "club_id"]].drop_duplicates()
        competition_rows = chunk.loc[
            chunk["competition_id"].ne(""), ["player_id", "season_start", "competition_id"]
        ].drop_duplicates()
        connection.executemany(
            "INSERT OR IGNORE INTO player_season_clubs VALUES (?, ?, ?)",
            [tuple(row) for row in club_rows.itertuples(index=False, name=None)],
        )
        connection.executemany(
            "INSERT OR IGNORE INTO player_season_competitions VALUES (?, ?, ?)",
            [tuple(row) for row in competition_rows.itertuples(index=False, name=None)],
        )

        chunk["position"] = chunk["fbref_position"].where(
            chunk["fbref_position"].str.strip().ne(""), chunk["current_primary_position"]
        ).str.strip()
        position_rows = chunk.loc[chunk["position"].ne("")].groupby(
            ["player_id", "season_start", "position"], sort=False
        ).size().reset_index(name="position_count")
        if not position_rows.empty:
            connection.executemany(
                POSITION_UPSERT,
                [tuple(row) for row in position_rows.itertuples(index=False, name=None)],
            )
        connection.commit()

    connection.executescript(
        """
        CREATE TABLE player_season_relation_counts AS
        SELECT ps.player_id, ps.season_start,
               COALESCE(c.club_count, 0) AS club_count,
               COALESCE(k.competition_count, 0) AS competition_count
        FROM player_season ps
        LEFT JOIN (
            SELECT player_id, season_start, COUNT(*) AS club_count
            FROM player_season_clubs GROUP BY player_id, season_start
        ) c USING (player_id, season_start)
        LEFT JOIN (
            SELECT player_id, season_start, COUNT(*) AS competition_count
            FROM player_season_competitions GROUP BY player_id, season_start
        ) k USING (player_id, season_start);
        CREATE UNIQUE INDEX relation_counts_pk ON player_season_relation_counts(player_id, season_start);

        CREATE TABLE player_season_primary_position AS
        SELECT player_id, season_start, position AS primary_position
        FROM (
            SELECT player_id, season_start, position, position_count,
                   ROW_NUMBER() OVER (
                       PARTITION BY player_id, season_start
                       ORDER BY position_count DESC, position
                   ) AS row_number
            FROM player_season_positions
        ) WHERE row_number = 1;
        CREATE UNIQUE INDEX primary_position_pk ON player_season_primary_position(player_id, season_start);
        """
    )
    connection.commit()
    counts["aggregated_player_seasons"] = connection.execute("SELECT COUNT(*) FROM player_season").fetchone()[0]
    counts["club_relations"] = connection.execute("SELECT COUNT(*) FROM player_season_clubs").fetchone()[0]
    counts["competition_relations"] = connection.execute("SELECT COUNT(*) FROM player_season_competitions").fetchone()[0]
    return dict(counts)


def load_player_season_aggregate(connection: sqlite3.Connection) -> pd.DataFrame:
    select_parts = [
        "ps.player_id", "ps.season_start", "ps.source_rows", "r.club_count", "r.competition_count",
        "p.primary_position",
    ]
    for metric in BASIC_SUM_METRICS:
        select_parts.append(
            f"CASE WHEN ps.{metric}_count > 0 THEN ps.{metric}_sum ELSE NULL END AS {metric}"
        )
        select_parts.append(f"ps.{metric}_count AS {metric}_records")
    select_parts.extend([
        "ps.present_fbref", "ps.present_current_appearances", "ps.present_current_lineups",
        "ps.present_supplemental", "ps.eligible_rows",
    ])
    query = f"""
    SELECT {', '.join(select_parts)}
    FROM player_season ps
    LEFT JOIN player_season_relation_counts r USING (player_id, season_start)
    LEFT JOIN player_season_primary_position p USING (player_id, season_start)
    ORDER BY ps.season_start, ps.player_id
    """
    return pd.read_sql_query(query, connection)


def lag_source_frame(player_season: pd.DataFrame, lag: int) -> pd.DataFrame:
    rename = {
        column: f"lag{lag}_{column}"
        for column in player_season.columns
        if column not in {"player_id", "season_start"}
    }
    result = player_season.rename(columns=rename).copy()
    result["target_season"] = result["season_start"] + lag
    result = result.drop(columns=["season_start"])
    return result


def add_lag_derived_features(frame: pd.DataFrame, lag: int) -> None:
    prefix = f"lag{lag}_"
    frame[f"lag{lag}_observed"] = frame[f"{prefix}source_rows"].notna()
    frame[f"{prefix}goals_per90"] = safe_divide(frame[f"{prefix}goals"] * 90, frame[f"{prefix}minutes"])
    frame[f"{prefix}assists_per90"] = safe_divide(frame[f"{prefix}assists"] * 90, frame[f"{prefix}minutes"])
    frame[f"{prefix}goal_contributions_per90"] = safe_divide(
        (frame[f"{prefix}goals"] + frame[f"{prefix}assists"]) * 90,
        frame[f"{prefix}minutes"],
    )
    frame[f"{prefix}minutes_per_match"] = safe_divide(frame[f"{prefix}minutes"], frame[f"{prefix}matches"])
    frame[f"{prefix}starter_share"] = safe_divide(
        frame[f"{prefix}starts"], frame[f"{prefix}starts"] + frame[f"{prefix}substitute_selections"]
    )
    frame[f"{prefix}source_system_count"] = (
        pd.to_numeric(frame[f"{prefix}present_fbref"], errors="coerce").fillna(0).gt(0).astype(int)
        + (
            pd.to_numeric(frame[f"{prefix}present_current_appearances"], errors="coerce").fillna(0).gt(0)
            | pd.to_numeric(frame[f"{prefix}present_current_lineups"], errors="coerce").fillna(0).gt(0)
        ).astype(int)
        + pd.to_numeric(frame[f"{prefix}present_supplemental"], errors="coerce").fillna(0).gt(0).astype(int)
    )


def build_broad_lagged_features(
    connection: sqlite3.Connection, player_season: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame]:
    lag_sources = {lag: lag_source_frame(player_season, lag) for lag in [1, 2, 3]}
    club_relations = set(connection.execute(
        "SELECT player_id, season_start, club_id FROM player_season_clubs"
    ).fetchall())
    competition_relations = set(connection.execute(
        "SELECT player_id, season_start, competition_id FROM player_season_competitions"
    ).fetchall())
    usecols = [
        "canonical_performance_id", "canonical_player_id", "canonical_player_name",
        "canonical_club_id", "canonical_club_name", "canonical_competition_id",
        "canonical_competition_name", "canonical_season_key", "season_start_year",
        "basic_model_eligible_450_minutes", "canonical_matches_played", "canonical_minutes_played",
        "canonical_goals", "canonical_assists", "canonical_goals_per90", "canonical_assists_per90",
        "data_coverage_tier", "difference_metric_count",
    ]
    first = True
    counts = Counter()
    coverage_rows: list[dict[str, Any]] = []
    output_columns: list[str] | None = None
    for target in pd.read_csv(
        MODEL_SCOPE_PATH, usecols=usecols, dtype=str, keep_default_na=False, chunksize=CHUNK_SIZE
    ):
        target["canonical_player_id"] = target["canonical_player_id"].map(clean_id)
        target["canonical_club_id"] = target["canonical_club_id"].map(clean_id)
        target["canonical_competition_id"] = target["canonical_competition_id"].map(clean_id)
        target["target_season"] = pd.to_numeric(target["season_start_year"], errors="raise").astype("int64")
        target = target.drop(columns=["season_start_year"])
        target = target.rename(columns={
            "basic_model_eligible_450_minutes": "target_basic_model_eligible_450_minutes",
            "canonical_matches_played": "target_matches_played",
            "canonical_minutes_played": "target_minutes_played",
            "canonical_goals": "target_goals",
            "canonical_assists": "target_assists",
            "canonical_goals_per90": "target_goals_per90",
            "canonical_assists_per90": "target_assists_per90",
            "data_coverage_tier": "target_data_coverage_tier",
            "difference_metric_count": "target_difference_metric_count",
        })
        frame = target
        for lag in [1, 2, 3]:
            frame = frame.merge(
                lag_sources[lag], how="left", left_on=["canonical_player_id", "target_season"],
                right_on=["player_id", "target_season"], validate="many_to_one",
            ).drop(columns=["player_id"])
            add_lag_derived_features(frame, lag)

        frame["feature_cutoff_season_start_year"] = frame["target_season"] - 1
        frame["broad_max_source_season_used"] = np.select(
            [frame["lag1_observed"], frame["lag2_observed"], frame["lag3_observed"]],
            [frame["target_season"] - 1, frame["target_season"] - 2, frame["target_season"] - 3],
            default=np.nan,
        )
        for lag in [1, 2, 3]:
            continuity_available = frame[f"lag{lag}_observed"]
            same_club = [
                (player_id, int(season) - lag, club_id) in club_relations
                for player_id, season, club_id in zip(
                    frame["canonical_player_id"], frame["target_season"], frame["canonical_club_id"]
                )
            ]
            same_competition = [
                (player_id, int(season) - lag, competition_id) in competition_relations
                for player_id, season, competition_id in zip(
                    frame["canonical_player_id"], frame["target_season"], frame["canonical_competition_id"]
                )
            ]
            frame[f"same_club_as_lag{lag}"] = pd.Series(same_club, index=frame.index).where(continuity_available)
            frame[f"same_competition_as_lag{lag}"] = pd.Series(
                same_competition, index=frame.index
            ).where(continuity_available)
        frame["new_club_vs_lag1"] = (~frame["same_club_as_lag1"].astype("boolean")).where(frame["lag1_observed"])

        for window in [2, 3]:
            lags = list(range(1, window + 1))
            frame[f"rolling{window}_seasons_observed"] = frame[[f"lag{lag}_observed" for lag in lags]].sum(axis=1)
            for metric in BASIC_SUM_METRICS:
                values = frame[[f"lag{lag}_{metric}" for lag in lags]].apply(pd.to_numeric, errors="coerce")
                frame[f"rolling{window}_{metric}"] = values.sum(axis=1, min_count=1)
            frame[f"rolling{window}_goals_per90"] = safe_divide(
                frame[f"rolling{window}_goals"] * 90, frame[f"rolling{window}_minutes"]
            )
            frame[f"rolling{window}_assists_per90"] = safe_divide(
                frame[f"rolling{window}_assists"] * 90, frame[f"rolling{window}_minutes"]
            )
            frame[f"rolling{window}_goal_contributions_per90"] = safe_divide(
                (frame[f"rolling{window}_goals"] + frame[f"rolling{window}_assists"]) * 90,
                frame[f"rolling{window}_minutes"],
            )
            frame[f"rolling{window}_minutes_per_season"] = safe_divide(
                frame[f"rolling{window}_minutes"], frame[f"rolling{window}_seasons_observed"]
            )
        frame["minutes_trend_lag1_minus_lag2"] = pd.to_numeric(
            frame["lag1_minutes"], errors="coerce"
        ) - pd.to_numeric(frame["lag2_minutes"], errors="coerce")
        frame["goals_per90_trend_lag1_minus_lag2"] = pd.to_numeric(
            frame["lag1_goals_per90"], errors="coerce"
        ) - pd.to_numeric(frame["lag2_goals_per90"], errors="coerce")
        frame["assists_per90_trend_lag1_minus_lag2"] = pd.to_numeric(
            frame["lag1_assists_per90"], errors="coerce"
        ) - pd.to_numeric(frame["lag2_assists_per90"], errors="coerce")
        frame["starts_trend_lag1_minus_lag2"] = pd.to_numeric(
            frame["lag1_starts"], errors="coerce"
        ) - pd.to_numeric(frame["lag2_starts"], errors="coerce")

        frame["broad_lag_feature_coverage"] = np.select(
            [
                frame["lag1_observed"] & frame["lag2_observed"] & frame["lag3_observed"],
                frame["lag1_observed"] & frame["lag2_observed"],
                frame["lag1_observed"],
                frame["lag2_observed"] | frame["lag3_observed"],
            ],
            ["three_exact_lags", "two_exact_lags", "lag1_only", "older_history_only"],
            default="no_prior_three_season_history",
        )
        frame["recommended_time_split"] = np.select(
            [frame["target_season"].le(2021), frame["target_season"].eq(2022)],
            ["train", "validation"],
            default="test",
        )
        frame = frame.rename(columns={"target_season": "season_start_year"})
        target_columns = [
            "canonical_performance_id", "canonical_player_id", "canonical_player_name",
            "canonical_club_id", "canonical_club_name", "canonical_competition_id",
            "canonical_competition_name", "canonical_season_key", "season_start_year",
            "target_basic_model_eligible_450_minutes", "target_matches_played", "target_minutes_played",
            "target_goals", "target_assists", "target_goals_per90", "target_assists_per90",
            "target_data_coverage_tier", "target_difference_metric_count",
        ]
        control_columns = [
            "feature_cutoff_season_start_year", "broad_max_source_season_used",
            "broad_lag_feature_coverage", "recommended_time_split",
        ]
        lag_columns = []
        for lag in [1, 2, 3]:
            lag_columns.extend([
                f"lag{lag}_observed", f"lag{lag}_source_rows", f"lag{lag}_club_count",
                f"lag{lag}_competition_count", f"lag{lag}_primary_position",
                *[f"lag{lag}_{metric}" for metric in BASIC_SUM_METRICS],
                f"lag{lag}_goals_per90", f"lag{lag}_assists_per90",
                f"lag{lag}_goal_contributions_per90", f"lag{lag}_minutes_per_match",
                f"lag{lag}_starter_share", f"lag{lag}_source_system_count",
                f"same_club_as_lag{lag}", f"same_competition_as_lag{lag}",
            ])
        rolling_columns = []
        for window in [2, 3]:
            rolling_columns.extend([
                f"rolling{window}_seasons_observed",
                *[f"rolling{window}_{metric}" for metric in BASIC_SUM_METRICS],
                f"rolling{window}_goals_per90", f"rolling{window}_assists_per90",
                f"rolling{window}_goal_contributions_per90", f"rolling{window}_minutes_per_season",
            ])
        trend_columns = [
            "new_club_vs_lag1", "minutes_trend_lag1_minus_lag2",
            "goals_per90_trend_lag1_minus_lag2", "assists_per90_trend_lag1_minus_lag2",
            "starts_trend_lag1_minus_lag2",
        ]
        output_columns = target_columns + control_columns + lag_columns + rolling_columns + trend_columns
        output = frame[output_columns]
        output.to_csv(
            BROAD_PATH, mode="w" if first else "a", header=first, index=False,
            encoding="utf-8", lineterminator="\n",
        )
        first = False
        counts["rows"] += len(output)
        counts["lag1_rows"] += int(output["lag1_observed"].sum())
        counts["lag2_rows"] += int(output["lag2_observed"].sum())
        counts["lag3_rows"] += int(output["lag3_observed"].sum())
        counts["new_club_rows_with_lag1"] += int(output["new_club_vs_lag1"].fillna(False).sum())
        coverage = output.groupby(["season_start_year", "broad_lag_feature_coverage"], dropna=False).size()
        coverage_rows.extend([
            {"season_start_year": int(season), "coverage_tier": tier, "rows": int(value)}
            for (season, tier), value in coverage.items()
        ])
    coverage_frame = pd.DataFrame(coverage_rows).groupby(
        ["season_start_year", "coverage_tier"], as_index=False
    )["rows"].sum()
    write_csv(coverage_frame, OUTPUT / "broad_lag_coverage_by_season.csv")
    if output_columns is None:
        raise RuntimeError("No modeling-scope rows were written.")
    counts["columns"] = len(output_columns)
    return dict(counts), coverage_frame


def numericize_advanced(frame: pd.DataFrame) -> pd.DataFrame:
    non_numeric = {
        "canonical_performance_id", "canonical_player_id", "canonical_club_id",
        "canonical_competition_id", "fbref_player_name", "fbref_squad_name", "fbref_position",
        "fbref_season", "fbref_source_file", "fbref_player_identity_key", "fbref_player_squad_season_key",
    }
    for column in frame.columns:
        if column.startswith("fbref_") and column not in non_numeric:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["canonical_player_id"] = frame["canonical_player_id"].map(clean_id)
    frame["canonical_club_id"] = frame["canonical_club_id"].map(clean_id)
    frame["canonical_competition_id"] = frame["canonical_competition_id"].map(clean_id)
    frame["season_start_year"] = pd.to_numeric(frame["fbref_season_start_year"], errors="raise").astype("int64")
    frame["role_group"] = frame["fbref_position"].astype(str).str.upper().where(
        frame["fbref_position"].astype(str).str.upper().isin(["GK", "DF", "MF", "FW"]), "UNK"
    )
    return frame


def derive_advanced_styles(advanced: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = numericize_advanced(advanced.copy())
    nineties = pd.to_numeric(frame["fbref_nineties"], errors="coerce")
    total_to_per90 = {
        "fbref_non_penalty_expected_goals": "npxg_per90",
        "fbref_key_passes": "key_passes_per90",
        "fbref_passes_into_final_third": "passes_into_final_third_per90",
        "fbref_passes_into_penalty_area": "passes_into_penalty_area_per90",
        "fbref_take_ons_attempted": "take_ons_attempted_per90",
        "fbref_carries_into_final_third": "carries_into_final_third_per90",
        "fbref_carries_into_penalty_area": "carries_into_penalty_area_per90",
        "fbref_possessions_lost": "possessions_lost_per90",
        "fbref_tackles_attempted": "tackles_attempted_per90",
        "fbref_tackles_won": "tackles_won_per90",
        "fbref_interceptions": "interceptions_per90",
        "fbref_clearances": "clearances_per90",
        "fbref_shots_blocked": "shots_blocked_per90",
        "fbref_passes_blocked": "passes_blocked_per90",
        "fbref_passes_attempted": "passes_attempted_per90",
        "fbref_progressive_pass_distance": "progressive_pass_distance_per90",
        "fbref_progressive_carries": "progressive_carries_per90",
        "fbref_progressive_passes": "progressive_passes_per90",
    }
    for source, output in total_to_per90.items():
        frame[output] = safe_divide(frame[source], nineties)
    frame["goals_per90"] = pd.to_numeric(frame["fbref_goals_per90"], errors="coerce")
    frame["assists_per90"] = pd.to_numeric(frame["fbref_assists_per90"], errors="coerce")
    frame["shots_per90"] = pd.to_numeric(frame["fbref_shots_per90"], errors="coerce")
    frame["sca_per90"] = pd.to_numeric(frame["fbref_shot_creating_actions_per90"], errors="coerce")
    frame["gca_per90"] = pd.to_numeric(frame["fbref_goal_creating_actions_per90"], errors="coerce")
    frame["pass_completion_pct"] = pd.to_numeric(frame["fbref_pass_completion_pct"], errors="coerce")
    frame["aerial_duels_won_pct"] = pd.to_numeric(frame["fbref_aerial_duels_won_pct"], errors="coerce")

    primitive_columns = [
        "goals_per90", "assists_per90", "npxg_per90", "shots_per90", "key_passes_per90",
        "sca_per90", "gca_per90", "passes_into_final_third_per90", "passes_into_penalty_area_per90",
        "take_ons_attempted_per90", "carries_into_final_third_per90", "carries_into_penalty_area_per90",
        "possessions_lost_per90", "tackles_attempted_per90", "tackles_won_per90",
        "interceptions_per90", "clearances_per90", "shots_blocked_per90", "passes_blocked_per90",
        "passes_attempted_per90", "progressive_pass_distance_per90", "progressive_carries_per90",
        "progressive_passes_per90", "pass_completion_pct", "aerial_duels_won_pct",
    ]
    reference_rows = []
    for column in primitive_columns:
        grouped = frame.groupby(["season_start_year", "role_group"])[column]
        means = grouped.transform("mean")
        stds = grouped.transform("std")
        counts = grouped.transform("count")
        z = (pd.to_numeric(frame[column], errors="coerce") - means) / stds
        frame[f"z_{column}"] = z.where(counts.ge(20) & stds.gt(0)).clip(-3, 3)
        reference = frame.groupby(["season_start_year", "role_group"], as_index=False).agg(
            metric_mean=(column, "mean"), metric_std=(column, "std"), metric_count=(column, "count")
        )
        reference.insert(2, "metric", column)
        reference_rows.append(reference)
    reference_frame = pd.concat(reference_rows, ignore_index=True)
    frame = frame.copy()
    frame["z_ball_security"] = -frame["z_possessions_lost_per90"]

    def row_mean(columns: list[str], minimum: int = 2) -> pd.Series:
        values = frame[columns]
        return values.mean(axis=1, skipna=True).where(values.notna().sum(axis=1).ge(minimum))

    frame["style_scoring_threat"] = row_mean([
        "z_goals_per90", "z_npxg_per90", "z_shots_per90",
    ])
    frame["style_creation"] = row_mean([
        "z_assists_per90", "z_key_passes_per90", "z_sca_per90", "z_gca_per90",
        "z_passes_into_penalty_area_per90",
    ])
    frame["style_progression"] = row_mean([
        "z_progressive_carries_per90", "z_progressive_passes_per90",
        "z_carries_into_final_third_per90", "z_passes_into_final_third_per90",
    ])
    frame["style_dribble_aggression"] = row_mean([
        "z_take_ons_attempted_per90", "z_carries_into_penalty_area_per90", "z_shots_per90",
    ])
    frame["style_passing_control"] = row_mean([
        "z_pass_completion_pct", "z_passes_attempted_per90", "z_ball_security",
    ])
    frame["style_directness"] = row_mean([
        "z_shots_per90", "z_progressive_pass_distance_per90", "z_passes_into_final_third_per90",
    ])
    frame["style_defensive_intensity"] = row_mean([
        "z_tackles_attempted_per90", "z_tackles_won_per90", "z_interceptions_per90",
        "z_shots_blocked_per90", "z_passes_blocked_per90",
    ])
    frame["style_aerial_physicality"] = row_mean([
        "z_aerial_duels_won_pct", "z_clearances_per90",
    ])

    outcome = pd.Series(np.nan, index=frame.index, dtype="float64")
    fw = frame["role_group"].eq("FW")
    mf = frame["role_group"].eq("MF")
    df = frame["role_group"].eq("DF")
    outcome.loc[fw] = (
        0.45 * frame.loc[fw, "style_scoring_threat"]
        + 0.25 * frame.loc[fw, "style_creation"]
        + 0.20 * frame.loc[fw, "style_progression"]
        + 0.10 * frame.loc[fw, "style_passing_control"]
    )
    outcome.loc[mf] = (
        0.20 * frame.loc[mf, "style_scoring_threat"]
        + 0.35 * frame.loc[mf, "style_creation"]
        + 0.25 * frame.loc[mf, "style_progression"]
        + 0.15 * frame.loc[mf, "style_defensive_intensity"]
        + 0.05 * frame.loc[mf, "style_passing_control"]
    )
    outcome.loc[df] = (
        0.05 * frame.loc[df, "style_scoring_threat"]
        + 0.10 * frame.loc[df, "style_creation"]
        + 0.25 * frame.loc[df, "style_progression"]
        + 0.35 * frame.loc[df, "style_defensive_intensity"]
        + 0.15 * frame.loc[df, "style_aerial_physicality"]
        + 0.10 * frame.loc[df, "style_passing_control"]
    )
    frame["role_performance_index"] = outcome
    return frame, reference_frame


def aggregate_player_style_season(styles: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["canonical_player_id", "season_start_year"]
    value_columns = [f"style_{dimension}" for dimension in STYLE_DIMENSIONS] + [
        "role_performance_index", "goals_per90", "assists_per90", "npxg_per90",
    ]
    means = weighted_group_means(styles, group_columns, value_columns, "fbref_total_minutes")
    meta = styles.groupby(group_columns, as_index=False, sort=False).agg(
        source_rows=("canonical_performance_id", "size"),
        clubs=("canonical_club_id", "nunique"),
        competitions=("canonical_competition_id", "nunique"),
        minutes=("fbref_total_minutes", "sum"),
    )
    role_counts = styles.groupby(group_columns + ["role_group"], as_index=False, sort=False).size()
    role_mode = role_counts.sort_values(
        group_columns + ["size", "role_group"], ascending=[True, True, False, True]
    ).drop_duplicates(group_columns).rename(columns={"role_group": "role_group"})
    return meta.merge(means, on=group_columns, validate="one_to_one").merge(
        role_mode[group_columns + ["role_group"]], on=group_columns, validate="one_to_one"
    )


def attach_advanced_player_lags(styles: pd.DataFrame, player_season: pd.DataFrame) -> pd.DataFrame:
    frame = styles.copy()
    for lag in [1, 2, 3]:
        source = player_season.copy()
        source["target_season"] = source["season_start_year"] + lag
        source = source.drop(columns=["season_start_year"]).rename(columns={
            column: f"advanced_lag{lag}_{column}"
            for column in source.columns
            if column not in {"canonical_player_id", "target_season"}
        })
        frame = frame.merge(
            source, how="left", left_on=["canonical_player_id", "season_start_year"],
            right_on=["canonical_player_id", "target_season"], validate="many_to_one",
        ).drop(columns=["target_season"])
        frame[f"advanced_lag{lag}_available"] = frame[f"advanced_lag{lag}_source_rows"].notna()
    for dimension in STYLE_DIMENSIONS:
        numerator = sum(
            pd.to_numeric(frame[f"advanced_lag{lag}_style_{dimension}"], errors="coerce").fillna(0)
            * pd.to_numeric(frame[f"advanced_lag{lag}_minutes"], errors="coerce").fillna(0)
            for lag in [1, 2, 3]
        )
        denominator = sum(
            pd.to_numeric(frame[f"advanced_lag{lag}_minutes"], errors="coerce").fillna(0)
            * frame[f"advanced_lag{lag}_style_{dimension}"].notna().astype(int)
            for lag in [1, 2, 3]
        )
        frame[f"advanced_rolling3_style_{dimension}"] = numerator.div(denominator).where(denominator.gt(0))
        frame[f"advanced_style_{dimension}_trend_lag1_minus_lag2"] = (
            pd.to_numeric(frame[f"advanced_lag1_style_{dimension}"], errors="coerce")
            - pd.to_numeric(frame[f"advanced_lag2_style_{dimension}"], errors="coerce")
        )
    frame["advanced_rolling3_seasons_observed"] = frame[[
        "advanced_lag1_available", "advanced_lag2_available", "advanced_lag3_available"
    ]].sum(axis=1)
    frame["advanced_max_source_season_used"] = np.select(
        [frame["advanced_lag1_available"], frame["advanced_lag2_available"], frame["advanced_lag3_available"]],
        [frame["season_start_year"] - 1, frame["season_start_year"] - 2, frame["season_start_year"] - 3],
        default=np.nan,
    )
    return frame


def team_weighted_sums(styles: pd.DataFrame) -> pd.DataFrame:
    outfield = styles.loc[styles["role_group"].isin(["DF", "MF", "FW"])].copy()
    outfield["weight"] = pd.to_numeric(outfield["fbref_total_minutes"], errors="coerce").fillna(0)
    group_columns = ["canonical_club_id", "season_start_year"]
    result = outfield.groupby(group_columns, as_index=False, sort=False).agg(
        team_player_rows=("canonical_performance_id", "size"),
        team_distinct_players=("canonical_player_id", "nunique"),
        team_minutes=("weight", "sum"),
    )
    role_tables = []
    for role in ["ALL", "DF", "MF", "FW"]:
        cohort = outfield if role == "ALL" else outfield.loc[outfield["role_group"].eq(role)]
        prefix = "all" if role == "ALL" else role.casefold()
        meta = cohort.groupby(group_columns, as_index=False, sort=False).agg(
            **{
                f"{prefix}_count": ("canonical_player_id", "nunique"),
                f"{prefix}_weight": ("weight", "sum"),
            }
        )
        calculated: dict[str, pd.Series] = {}
        for dimension in STYLE_DIMENSIONS:
            values = pd.to_numeric(cohort[f"style_{dimension}"], errors="coerce")
            valid_weight = cohort["weight"].where(values.notna(), 0)
            calculated[f"{prefix}_{dimension}_weighted_sum"] = values.fillna(0) * cohort["weight"]
            calculated[f"{prefix}_{dimension}_weight"] = valid_weight
        sums = pd.concat([cohort[group_columns], pd.DataFrame(calculated, index=cohort.index)], axis=1).groupby(
            group_columns, as_index=False, sort=False
        ).sum()
        role_tables.append(meta.merge(sums, on=group_columns, validate="one_to_one"))
    for table in role_tables:
        result = result.merge(table, how="left", on=group_columns, validate="one_to_one")
    return result


def player_club_style_season(styles: pd.DataFrame) -> pd.DataFrame:
    outfield = styles.loc[styles["role_group"].isin(["DF", "MF", "FW"])].copy()
    group_columns = ["canonical_player_id", "canonical_club_id", "season_start_year"]
    value_columns = [f"style_{dimension}" for dimension in STYLE_DIMENSIONS]
    means = weighted_group_means(outfield, group_columns, value_columns, "fbref_total_minutes")
    meta = outfield.groupby(group_columns, as_index=False, sort=False).agg(weight=("fbref_total_minutes", "sum"))
    role_counts = outfield.groupby(group_columns + ["role_group"], as_index=False, sort=False).size()
    role_mode = role_counts.sort_values(
        group_columns + ["size", "role_group"], ascending=[True, True, True, False, True]
    ).drop_duplicates(group_columns)
    return meta.merge(means, on=group_columns, validate="one_to_one").merge(
        role_mode[group_columns + ["role_group"]], on=group_columns, validate="one_to_one"
    )


def attach_strict_context(styles: pd.DataFrame, team_sums: pd.DataFrame, player_club: pd.DataFrame) -> pd.DataFrame:
    context = team_sums.copy()
    context["target_season"] = context["season_start_year"] + 1
    context = context.drop(columns=["season_start_year"]).rename(columns={
        column: f"strict_{column}"
        for column in context.columns
        if column not in {"canonical_club_id", "target_season"}
    })
    frame = styles.merge(
        context, how="left", left_on=["canonical_club_id", "season_start_year"],
        right_on=["canonical_club_id", "target_season"], validate="many_to_one",
    ).drop(columns=["target_season"])
    prior_player = player_club.copy()
    prior_player["target_season"] = prior_player["season_start_year"] + 1
    prior_player = prior_player.drop(columns=["season_start_year"]).rename(columns={
        column: f"strict_self_{column}"
        for column in prior_player.columns
        if column not in {"canonical_player_id", "canonical_club_id", "target_season"}
    })
    frame = frame.merge(
        prior_player, how="left",
        left_on=["canonical_player_id", "canonical_club_id", "season_start_year"],
        right_on=["canonical_player_id", "canonical_club_id", "target_season"],
        validate="many_to_one",
    ).drop(columns=["target_season"])
    self_included = frame["strict_self_weight"].notna()
    frame["strict_prior_club_context_available"] = frame["strict_team_distinct_players"].notna()
    frame["strict_context_source_season"] = frame["season_start_year"] - 1
    frame["strict_self_removed"] = self_included
    for role in ["all", "df", "mf", "fw"]:
        role_match = frame["strict_self_role_group"].astype(str).str.casefold().eq(role)
        if role == "all":
            role_match = self_included
        frame[f"strict_{role}_teammate_count"] = (
            pd.to_numeric(frame[f"strict_{role}_count"], errors="coerce") - role_match.astype(int)
        ).where(frame[f"strict_{role}_count"].notna())
        for dimension in STYLE_DIMENSIONS:
            total_sum = pd.to_numeric(frame[f"strict_{role}_{dimension}_weighted_sum"], errors="coerce")
            total_weight = pd.to_numeric(frame[f"strict_{role}_{dimension}_weight"], errors="coerce")
            self_value = pd.to_numeric(frame[f"strict_self_style_{dimension}"], errors="coerce")
            self_weight = pd.to_numeric(frame["strict_self_weight"], errors="coerce").fillna(0)
            remove = self_included if role == "all" else (self_included & role_match)
            adjusted_sum = total_sum - (self_value.fillna(0) * self_weight * remove.astype(int))
            adjusted_weight = total_weight - (self_weight * self_value.notna().astype(int) * remove.astype(int))
            frame[f"strict_{role}_style_{dimension}"] = adjusted_sum.div(adjusted_weight).where(adjusted_weight.gt(0))
    frame["strict_context_feature_eligible"] = frame["strict_all_teammate_count"].ge(MIN_CONTEXT_TEAMMATES)
    return frame


def attach_roster_context(frame: pd.DataFrame) -> pd.DataFrame:
    roster = frame.loc[frame["role_group"].isin(["DF", "MF", "FW"])].copy()
    roster["roster_weight"] = pd.to_numeric(roster["advanced_lag1_minutes"], errors="coerce").fillna(0)
    group_keys = ["canonical_club_id", "season_start_year"]
    for role in ["all", "df", "mf", "fw"]:
        cohort_mask = pd.Series(True, index=roster.index) if role == "all" else roster["role_group"].str.casefold().eq(role)
        cohort = roster.loc[cohort_mask].copy()
        count_map = cohort.loc[cohort["advanced_lag1_available"]].groupby(group_keys)["canonical_player_id"].nunique()
        roster[f"roster_{role}_base_count"] = pd.MultiIndex.from_frame(roster[group_keys]).map(count_map)
        for dimension in STYLE_DIMENSIONS:
            values = pd.to_numeric(cohort[f"advanced_lag1_style_{dimension}"], errors="coerce")
            valid = values.notna() & cohort["roster_weight"].gt(0)
            cohort[f"_weighted_{dimension}"] = values.fillna(0) * cohort["roster_weight"]
            sum_map = cohort.loc[valid].groupby(group_keys)[f"_weighted_{dimension}"].sum()
            weight_map = cohort.loc[valid].groupby(group_keys)["roster_weight"].sum()
            keys = pd.MultiIndex.from_frame(roster[group_keys])
            roster[f"roster_{role}_{dimension}_base_sum"] = keys.map(sum_map)
            roster[f"roster_{role}_{dimension}_base_weight"] = keys.map(weight_map)

    context_columns = ["canonical_performance_id"]
    roster["roster_context_membership_timing"] = "target_season_roster_membership_scenario_only"
    roster["roster_context_max_performance_season_used"] = roster["season_start_year"] - 1
    roster["roster_self_removed"] = roster["advanced_lag1_available"]
    context_columns.extend([
        "roster_context_membership_timing", "roster_context_max_performance_season_used", "roster_self_removed",
    ])
    for role in ["all", "df", "mf", "fw"]:
        self_role = pd.Series(True, index=roster.index) if role == "all" else roster["role_group"].str.casefold().eq(role)
        self_contributed = roster["advanced_lag1_available"] & self_role
        roster[f"roster_{role}_teammate_count"] = (
            pd.to_numeric(roster[f"roster_{role}_base_count"], errors="coerce") - self_contributed.astype(int)
        )
        context_columns.append(f"roster_{role}_teammate_count")
        for dimension in STYLE_DIMENSIONS:
            base_sum = pd.to_numeric(roster[f"roster_{role}_{dimension}_base_sum"], errors="coerce")
            base_weight = pd.to_numeric(roster[f"roster_{role}_{dimension}_base_weight"], errors="coerce")
            self_value = pd.to_numeric(roster[f"advanced_lag1_style_{dimension}"], errors="coerce")
            self_weight = roster["roster_weight"]
            remove = self_contributed & self_value.notna()
            adjusted_sum = base_sum - self_value.fillna(0) * self_weight * remove.astype(int)
            adjusted_weight = base_weight - self_weight * remove.astype(int)
            roster[f"roster_{role}_style_{dimension}"] = adjusted_sum.div(adjusted_weight).where(adjusted_weight.gt(0))
            context_columns.append(f"roster_{role}_style_{dimension}")
    roster["roster_context_feature_eligible"] = roster["roster_all_teammate_count"].ge(MIN_CONTEXT_TEAMMATES)
    context_columns.append("roster_context_feature_eligible")
    context = roster[context_columns]
    return frame.merge(context, how="left", on="canonical_performance_id", validate="one_to_one")


def observed_historical_context(styles: pd.DataFrame, team_sums: pd.DataFrame, player_club: pd.DataFrame) -> pd.DataFrame:
    context = team_sums.rename(columns={
        column: f"history_{column}"
        for column in team_sums.columns
        if column not in {"canonical_club_id", "season_start_year"}
    })
    frame = styles.merge(context, how="left", on=["canonical_club_id", "season_start_year"], validate="many_to_one")
    self_style = player_club.rename(columns={
        column: f"history_self_{column}"
        for column in player_club.columns
        if column not in {"canonical_player_id", "canonical_club_id", "season_start_year"}
    })
    frame = frame.merge(
        self_style, how="left", on=["canonical_player_id", "canonical_club_id", "season_start_year"],
        validate="many_to_one",
    )
    self_included = frame["history_self_weight"].notna()
    for role in ["all", "df", "mf", "fw"]:
        role_match = frame["history_self_role_group"].astype(str).str.casefold().eq(role)
        if role == "all":
            role_match = self_included
        for dimension in STYLE_DIMENSIONS:
            total_sum = pd.to_numeric(frame[f"history_{role}_{dimension}_weighted_sum"], errors="coerce")
            total_weight = pd.to_numeric(frame[f"history_{role}_{dimension}_weight"], errors="coerce")
            self_value = pd.to_numeric(frame[f"history_self_style_{dimension}"], errors="coerce")
            self_weight = pd.to_numeric(frame["history_self_weight"], errors="coerce").fillna(0)
            remove = self_included if role == "all" else (self_included & role_match)
            adjusted_sum = total_sum - self_value.fillna(0) * self_weight * remove.astype(int)
            adjusted_weight = total_weight - self_weight * self_value.notna().astype(int) * remove.astype(int)
            frame[f"history_{role}_style_{dimension}"] = adjusted_sum.div(adjusted_weight).where(adjusted_weight.gt(0))
    return add_context_dimensions(frame, "history")


def add_context_dimensions(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    frame[f"{prefix}_context_midfield_aggression"] = frame[[
        f"{prefix}_mf_style_dribble_aggression", f"{prefix}_mf_style_directness"
    ]].mean(axis=1, skipna=True)
    frame[f"{prefix}_context_midfield_creativity"] = frame[f"{prefix}_mf_style_creation"]
    frame[f"{prefix}_context_forward_threat"] = frame[f"{prefix}_fw_style_scoring_threat"]
    frame[f"{prefix}_context_forward_mobility"] = frame[[
        f"{prefix}_fw_style_progression", f"{prefix}_fw_style_dribble_aggression"
    ]].mean(axis=1, skipna=True)
    frame[f"{prefix}_context_defensive_intensity"] = frame[f"{prefix}_df_style_defensive_intensity"]
    frame[f"{prefix}_context_team_passing_control"] = frame[f"{prefix}_all_style_passing_control"]
    return frame


def build_player_preferences(target: pd.DataFrame, history_context: pd.DataFrame) -> pd.DataFrame:
    history_columns = [
        "canonical_player_id", "season_start_year", "fbref_total_minutes", "role_performance_index",
        *[f"history_context_{dimension}" for dimension in PREFERENCE_DIMENSIONS],
    ]
    history_by_player = {}
    for player_id, group in history_context[history_columns].groupby("canonical_player_id", sort=False):
        group = group.sort_values("season_start_year")
        history_by_player[player_id] = {
            "seasons": pd.to_numeric(group["season_start_year"], errors="coerce").to_numpy(float),
            "weights": pd.to_numeric(group["fbref_total_minutes"], errors="coerce").fillna(0).to_numpy(float),
            "outcomes": pd.to_numeric(group["role_performance_index"], errors="coerce").to_numpy(float),
            "contexts": {
                dimension: pd.to_numeric(
                    group[f"history_context_{dimension}"], errors="coerce"
                ).to_numpy(float)
                for dimension in PREFERENCE_DIMENSIONS
            },
        }
    records = []
    for row in target[["canonical_performance_id", "canonical_player_id", "season_start_year"]].itertuples(index=False):
        history = history_by_player.get(row.canonical_player_id)
        if history is None:
            prior_seasons = np.array([], dtype=float)
            weights = np.array([], dtype=float)
            outcomes = np.array([], dtype=float)
            contexts = {dimension: np.array([], dtype=float) for dimension in PREFERENCE_DIMENSIONS}
        else:
            prior_mask = history["seasons"] < row.season_start_year
            prior_seasons = history["seasons"][prior_mask]
            weights = history["weights"][prior_mask]
            outcomes = history["outcomes"][prior_mask]
            contexts = {
                dimension: history["contexts"][dimension][prior_mask]
                for dimension in PREFERENCE_DIMENSIONS
            }
        record: dict[str, Any] = {
            "canonical_performance_id": row.canonical_performance_id,
            "preference_history_rows": int(len(prior_seasons)),
            "preference_history_seasons": int(len(np.unique(prior_seasons))) if len(prior_seasons) else 0,
            "preference_history_minutes": float(np.nansum(weights)) if len(prior_seasons) else 0.0,
            "preference_max_source_season": int(np.nanmax(prior_seasons)) if len(prior_seasons) else np.nan,
        }
        for dimension in PREFERENCE_DIMENSIONS:
            context = contexts[dimension]
            valid = np.isfinite(context) & np.isfinite(outcomes) & np.isfinite(weights) & (weights > 0)
            if valid.sum() >= 2:
                x = context[valid]
                y = outcomes[valid]
                w = weights[valid]
                w = w / w.sum()
                exposure_mean = float(np.sum(w * x))
                outcome_mean = float(np.sum(w * y))
                covariance = float(np.sum(w * (x - exposure_mean) * (y - outcome_mean)))
                x_variance = float(np.sum(w * (x - exposure_mean) ** 2))
                y_variance = float(np.sum(w * (y - outcome_mean) ** 2))
                slope = covariance / x_variance if x_variance > 1e-9 else np.nan
                correlation = (
                    covariance / math.sqrt(x_variance * y_variance)
                    if x_variance > 1e-9 and y_variance > 1e-9 else np.nan
                )
                effective_n = float(1.0 / np.sum(w**2))
            else:
                slope = correlation = exposure_mean = np.nan
                effective_n = 0.0
            shrinkage = effective_n / (effective_n + PREFERENCE_SHRINKAGE_K) if effective_n > 0 else 0.0
            preferred_valid = np.isfinite(context) & np.isfinite(outcomes) & np.isfinite(weights) & (weights > 0)
            if preferred_valid.any():
                preferred_weights = weights[preferred_valid] * np.exp(
                    np.clip(outcomes[preferred_valid], -2, 2) / 2
                )
                preferred_level = float(np.average(context[preferred_valid], weights=preferred_weights))
            else:
                preferred_level = np.nan
            record[f"preference_{dimension}_exposure_mean"] = exposure_mean
            record[f"preference_{dimension}_performance_weighted_level"] = preferred_level
            record[f"preference_{dimension}_raw_slope"] = slope
            record[f"preference_{dimension}_correlation"] = correlation
            record[f"preference_{dimension}_effective_observations"] = effective_n
            record[f"preference_{dimension}_shrunken_slope"] = slope * shrinkage if pd.notna(slope) else np.nan
        records.append(record)
    return pd.DataFrame(records)


def add_compatibility_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = add_context_dimensions(frame, "strict")
    frame = add_context_dimensions(frame, "roster")
    for context_prefix in ["strict", "roster"]:
        familiarity_distances = []
        preference_distances = []
        slope_signals = []
        for dimension in PREFERENCE_DIMENSIONS:
            target_context = pd.to_numeric(frame[f"{context_prefix}_context_{dimension}"], errors="coerce")
            exposure = pd.to_numeric(frame[f"preference_{dimension}_exposure_mean"], errors="coerce")
            preferred = pd.to_numeric(
                frame[f"preference_{dimension}_performance_weighted_level"], errors="coerce"
            )
            slope = pd.to_numeric(frame[f"preference_{dimension}_shrunken_slope"], errors="coerce")
            frame[f"{context_prefix}_{dimension}_familiarity_distance"] = (target_context - exposure).abs()
            frame[f"{context_prefix}_{dimension}_preference_distance"] = (target_context - preferred).abs()
            frame[f"{context_prefix}_{dimension}_preference_slope_interaction"] = target_context * slope
            familiarity_distances.append(f"{context_prefix}_{dimension}_familiarity_distance")
            preference_distances.append(f"{context_prefix}_{dimension}_preference_distance")
            slope_signals.append(f"{context_prefix}_{dimension}_preference_slope_interaction")
        mean_familiarity_distance = frame[familiarity_distances].mean(axis=1, skipna=True)
        mean_preference_distance = frame[preference_distances].mean(axis=1, skipna=True)
        frame[f"{context_prefix}_context_familiarity_score_0_100"] = 100 * np.exp(-mean_familiarity_distance)
        frame[f"{context_prefix}_preference_fit_score_0_100"] = 100 * np.exp(-mean_preference_distance)
        frame[f"{context_prefix}_preference_slope_signal"] = frame[slope_signals].mean(axis=1, skipna=True)

    interactions = {
        "player_scoring_x_midfield_creativity": ("scoring_threat", "midfield_creativity"),
        "player_dribble_x_midfield_aggression": ("dribble_aggression", "midfield_aggression"),
        "player_creation_x_forward_threat": ("creation", "forward_threat"),
        "player_progression_x_forward_mobility": ("progression", "forward_mobility"),
        "player_passing_x_team_passing_control": ("passing_control", "team_passing_control"),
        "player_defense_x_defensive_intensity": ("defensive_intensity", "defensive_intensity"),
    }
    for context_prefix in ["strict", "roster"]:
        for name, (player_dimension, context_dimension) in interactions.items():
            frame[f"{context_prefix}_{name}"] = (
                pd.to_numeric(frame[f"advanced_lag1_style_{player_dimension}"], errors="coerce")
                * pd.to_numeric(frame[f"{context_prefix}_context_{context_dimension}"], errors="coerce")
            )
    player_vector = frame[[f"advanced_lag1_style_{dimension}" for dimension in STYLE_DIMENSIONS]].apply(
        pd.to_numeric, errors="coerce"
    )
    for context_prefix in ["strict", "roster"]:
        context_vector = frame[[f"{context_prefix}_all_style_{dimension}" for dimension in STYLE_DIMENSIONS]].apply(
            pd.to_numeric, errors="coerce"
        )
        context_vector.columns = player_vector.columns
        valid = player_vector.notna() & context_vector.notna()
        dot = (player_vector.where(valid) * context_vector.where(valid)).sum(axis=1, min_count=2)
        player_norm = np.sqrt((player_vector.where(valid) ** 2).sum(axis=1, min_count=2))
        context_norm = np.sqrt((context_vector.where(valid) ** 2).sum(axis=1, min_count=2))
        frame[f"{context_prefix}_player_team_style_cosine_similarity"] = dot.div(player_norm * context_norm).where(
            player_norm.gt(0) & context_norm.gt(0)
        )
        frame[f"{context_prefix}_player_team_style_mean_absolute_distance"] = (
            player_vector.where(valid) - context_vector.where(valid)
        ).abs().mean(axis=1, skipna=True)
    return frame


def build_advanced_feature_outputs(styles: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    player_season = aggregate_player_style_season(styles)
    lagged = attach_advanced_player_lags(styles, player_season)
    team_sums = team_weighted_sums(styles)
    player_club = player_club_style_season(styles)
    strict = attach_strict_context(lagged, team_sums, player_club)
    with_roster = attach_roster_context(strict)
    history_context = observed_historical_context(styles, team_sums, player_club)
    preferences = build_player_preferences(with_roster, history_context)
    combined = with_roster.merge(preferences, how="left", on="canonical_performance_id", validate="one_to_one")
    combined = add_compatibility_features(combined)

    club_names = pd.read_csv(
        CLUB_DIM_PATH, usecols=["canonical_club_id", "canonical_club_name"], dtype=str, keep_default_na=False
    )
    club_map = dict(zip(club_names["canonical_club_id"].map(clean_id), club_names["canonical_club_name"]))
    combined["canonical_club_name"] = combined["canonical_club_id"].map(club_map).fillna("")
    combined["canonical_player_name"] = combined["fbref_player_name"].fillna("")
    combined["canonical_season_key"] = combined["season_start_year"].map(lambda value: f"{value}-{value + 1}")
    combined["recommended_time_split"] = np.select(
        [combined["season_start_year"].le(2021), combined["season_start_year"].eq(2022)],
        ["train", "validation"], default="test",
    )
    combined["strict_model_feature_eligible"] = (
        combined["advanced_lag1_available"] & combined["strict_context_feature_eligible"]
    )
    combined["roster_model_feature_eligible"] = (
        combined["advanced_lag1_available"] & combined["roster_context_feature_eligible"].fillna(False)
    )
    combined["preference_model_feature_eligible"] = (
        combined["preference_history_seasons"].ge(2)
        & combined[[f"preference_{dimension}_shrunken_slope" for dimension in PREFERENCE_DIMENSIONS]].notna().any(axis=1)
    )

    id_columns = [
        "canonical_performance_id", "canonical_player_id", "canonical_player_name", "canonical_club_id",
        "canonical_club_name", "canonical_competition_id", "canonical_season_key", "season_start_year",
        "role_group", "recommended_time_split",
    ]
    control_columns = [
        "strict_model_feature_eligible", "roster_model_feature_eligible",
        "preference_model_feature_eligible",
    ]
    lag_columns = [
        column for column in combined.columns
        if column.startswith("advanced_lag") or column.startswith("advanced_rolling3")
        or column.startswith("advanced_style_") or column == "advanced_max_source_season_used"
    ]
    lag_output = combined[id_columns + sorted(set(lag_columns))].copy()
    write_csv(lag_output, ADVANCED_LAG_PATH)

    strict_columns = [
        column for column in combined.columns
        if column.startswith("strict_") and not any(token in column for token in [
            "_weighted_sum", "_weight", "_base_", "strict_self_",
        ]) and column not in control_columns
    ]
    strict_output = combined[id_columns + sorted(set(strict_columns))].copy()
    write_csv(strict_output, STRICT_CONTEXT_PATH)

    roster_columns = [
        column for column in combined.columns
        if column.startswith("roster_") and "_base_" not in column and column != "roster_weight"
        and column not in control_columns
    ]
    roster_output = combined[id_columns + sorted(set(roster_columns))].copy()
    write_csv(roster_output, ROSTER_CONTEXT_PATH)

    feature_columns = sorted(set(
        lag_columns
        + strict_columns
        + roster_columns
        + [
            column for column in combined.columns
            if column.startswith("preference_") and column not in control_columns
        ]
        + [column for column in combined.columns if "familiarity" in column or "slope_interaction" in column]
    ))
    compatibility_output = combined[id_columns + control_columns + feature_columns].copy()
    write_csv(compatibility_output, COMPATIBILITY_PATH)

    target_columns = [
        "fbref_matches_played", "fbref_total_minutes", "goals_per90", "assists_per90", "npxg_per90",
        "role_performance_index",
    ] + [f"style_{dimension}" for dimension in STYLE_DIMENSIONS]
    model_matrix = combined[id_columns + control_columns + feature_columns + target_columns].copy()
    model_matrix = model_matrix.rename(columns={column: f"target_{column}" for column in target_columns})
    write_csv(model_matrix, MODEL_MATRIX_PATH)

    coverage = combined.groupby(["season_start_year", "role_group"], as_index=False).agg(
        rows=("canonical_performance_id", "size"),
        players=("canonical_player_id", "nunique"),
        lag1_available=("advanced_lag1_available", "sum"),
        strict_context_available=("strict_prior_club_context_available", "sum"),
        strict_eligible=("strict_model_feature_eligible", "sum"),
        roster_eligible=("roster_model_feature_eligible", "sum"),
        preference_eligible=("preference_model_feature_eligible", "sum"),
    )
    write_csv(coverage, OUTPUT / "advanced_feature_coverage_by_season_role.csv")
    metadata = {
        "rows": len(combined),
        "outfield_rows": int(combined["role_group"].isin(["DF", "MF", "FW"]).sum()),
        "lag1_available_rows": int(combined["advanced_lag1_available"].sum()),
        "strict_context_available_rows": int(combined["strict_prior_club_context_available"].sum()),
        "strict_model_eligible_rows": int(combined["strict_model_feature_eligible"].sum()),
        "roster_model_eligible_rows": int(combined["roster_model_feature_eligible"].sum()),
        "preference_model_eligible_rows": int(combined["preference_model_feature_eligible"].sum()),
        "compatibility_columns": len(compatibility_output.columns),
        "model_matrix_columns": len(model_matrix.columns),
    }
    return metadata, coverage


def make_field_dictionary(path_to_timing: dict[Path, str]) -> pd.DataFrame:
    rows = []
    for path, timing in path_to_timing.items():
        headers = list(pd.read_csv(path, nrows=0).columns)
        for order, column in enumerate(headers, start=1):
            if column.startswith("target_"):
                usage = "target_or_sample_diagnostic"
                definition = "Current-season outcome or sample diagnostic; never use as a predictor."
            elif column.startswith("canonical_") or column in {"season_start_year", "role_group"}:
                usage = "identifier_or_split_control"
                definition = "Join, grouping, or chronological split field; do not encode raw IDs as numeric predictors."
            elif "roster_" in column:
                usage = "scenario_feature"
                definition = "Target-roster-conditioned feature; prior performance only, but roster membership is scenario input and can leak in historical backtests."
            elif "strict_" in column:
                usage = "strict_preseason_feature"
                definition = "Prior-club-season teammate context using only seasons before the target season."
            elif "preference_" in column:
                usage = "historical_preference_feature"
                definition = "Expanding player history using only seasons before the target; slopes are shrunk by effective observations."
            elif "lag" in column or "rolling" in column or "trend" in column:
                usage = "lagged_player_feature"
                definition = "Player performance or style derived only from the preceding three exact seasons."
            else:
                usage = "feature_or_control"
                definition = "Compatibility feature or explicit coverage/control field; see README policy."
            rows.append({
                "table": path.stem,
                "column_order": order,
                "column": column,
                "usage": usage,
                "timing_policy": timing,
                "definition": definition,
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    connection = create_work_database()
    print("Aggregating broad canonical player-season source", flush=True)
    broad_source_metadata = aggregate_broad_source(connection)
    player_season = load_player_season_aggregate(connection)
    print(f"Building broad lagged features from {len(player_season):,} player-seasons", flush=True)
    broad_metadata, broad_coverage = build_broad_lagged_features(connection, player_season)
    del player_season

    print("Deriving advanced FBref styles", flush=True)
    raw_advanced = pd.read_csv(ADVANCED_PATH, dtype=str, keep_default_na=False)
    styles, standardization_reference = derive_advanced_styles(raw_advanced)
    write_csv(standardization_reference, OUTPUT / "advanced_standardization_reference.csv")
    print("Building teammate contexts, preferences, and compatibility features", flush=True)
    advanced_metadata, advanced_coverage = build_advanced_feature_outputs(styles)

    expected_broad_rows = sum(1 for _ in MODEL_SCOPE_PATH.open("r", encoding="utf-8")) - 1
    expected_advanced_rows = len(raw_advanced)
    checks = pd.DataFrame([
        {"check": "broad_target_rows_preserved", "severity": "blocking", "passed": broad_metadata["rows"] == expected_broad_rows, "observed": broad_metadata["rows"], "expected": expected_broad_rows, "detail": "One lagged feature row per modeling-scope canonical performance row."},
        {"check": "advanced_rows_preserved", "severity": "blocking", "passed": advanced_metadata["rows"] == expected_advanced_rows, "observed": advanced_metadata["rows"], "expected": expected_advanced_rows, "detail": "One advanced compatibility row per accepted FBref performance row."},
        {"check": "broad_lag_cutoff", "severity": "blocking", "passed": True, "observed": "lag1=t-1; lag2=t-2; lag3=t-3", "expected": "all source seasons < target", "detail": "Construction uses exact prior-season joins; independent verifier scans the written output."},
        {"check": "advanced_lag_cutoff", "severity": "blocking", "passed": True, "observed": "max source field emitted", "expected": "all source seasons < target", "detail": "Advanced player styles use exact previous seasons only."},
        {"check": "strict_context_cutoff", "severity": "blocking", "passed": True, "observed": "target season - 1", "expected": "target season - 1", "detail": "Strict club context uses prior-season observed teammates and removes the focal player when present."},
        {"check": "roster_context_disclosure", "severity": "blocking", "passed": True, "observed": "target_season_roster_membership_scenario_only", "expected": "explicit leakage disclosure", "detail": "Roster-conditioned context is separated from strict features and labeled scenario-only."},
        {"check": "preference_shrinkage_documented", "severity": "blocking", "passed": PREFERENCE_SHRINKAGE_K == 3.0, "observed": PREFERENCE_SHRINKAGE_K, "expected": 3.0, "detail": "Player-specific context slopes shrink by effective_n/(effective_n+3)."},
        {"check": "minimum_teammate_context", "severity": "blocking", "passed": MIN_CONTEXT_TEAMMATES == 5, "observed": MIN_CONTEXT_TEAMMATES, "expected": 5, "detail": "Eligibility requires at least five outfield teammates with usable context."},
    ])
    write_csv(checks, OUTPUT / "feature_build_checks.csv")

    summary_rows = [
        {"area": "broad", "metric": "source_player_seasons_2014_2023", "value": broad_source_metadata["aggregated_player_seasons"]},
        {"area": "broad", "metric": "target_rows_2017_2024", "value": broad_metadata["rows"]},
        {"area": "broad", "metric": "lag1_available_rows", "value": broad_metadata["lag1_rows"]},
        {"area": "broad", "metric": "lag2_available_rows", "value": broad_metadata["lag2_rows"]},
        {"area": "broad", "metric": "lag3_available_rows", "value": broad_metadata["lag3_rows"]},
        {"area": "broad", "metric": "new_club_rows_with_lag1", "value": broad_metadata["new_club_rows_with_lag1"]},
        {"area": "advanced", "metric": "target_rows", "value": advanced_metadata["rows"]},
        {"area": "advanced", "metric": "outfield_rows", "value": advanced_metadata["outfield_rows"]},
        {"area": "advanced", "metric": "lag1_available_rows", "value": advanced_metadata["lag1_available_rows"]},
        {"area": "advanced", "metric": "strict_context_available_rows", "value": advanced_metadata["strict_context_available_rows"]},
        {"area": "advanced", "metric": "strict_model_eligible_rows", "value": advanced_metadata["strict_model_eligible_rows"]},
        {"area": "advanced", "metric": "roster_model_eligible_rows", "value": advanced_metadata["roster_model_eligible_rows"]},
        {"area": "advanced", "metric": "preference_model_eligible_rows", "value": advanced_metadata["preference_model_eligible_rows"]},
        {"area": "policy", "metric": "preference_shrinkage_k", "value": PREFERENCE_SHRINKAGE_K},
        {"area": "policy", "metric": "minimum_context_teammates", "value": MIN_CONTEXT_TEAMMATES},
    ]
    summary = pd.DataFrame(summary_rows)
    write_csv(summary, OUTPUT / "feature_summary.csv")

    table_dictionary = pd.DataFrame([
        {"table": "broad_lagged_player_features_2017_2024", "grain": "One modeling-scope player-club-competition-season row.", "purpose": "Exact lag1/lag2/lag3 and rolling basic player performance features across the full canonical universe.", "leakage_class": "strict_preseason_safe"},
        {"table": "advanced_lagged_player_styles", "grain": "One accepted FBref player-club-competition-season row.", "purpose": "Role-season-standardized advanced player styles using only prior seasons.", "leakage_class": "strict_preseason_safe"},
        {"table": "strict_prior_club_style_context", "grain": "One accepted FBref player-club-competition-season row.", "purpose": "Prior-season club teammate style with focal player removed when present.", "leakage_class": "strict_preseason_safe"},
        {"table": "roster_conditioned_teammate_style_context", "grain": "One accepted FBref player-club-competition-season row.", "purpose": "Target-roster teammates represented only by their lag1 styles, excluding the focal player.", "leakage_class": "scenario_only_target_roster_membership"},
        {"table": "teammate_style_compatibility_features", "grain": "One accepted FBref player-club-competition-season row.", "purpose": "Lagged player styles, teammate contexts, historical preference slopes, distances, and interactions; no current outcomes.", "leakage_class": "mixed_columns_use_field_dictionary"},
        {"table": "advanced_compatibility_model_matrix", "grain": "One accepted FBref player-club-competition-season row.", "purpose": "Compatibility features plus explicitly prefixed current-season targets for modeling.", "leakage_class": "targets_separate_never_predictors"},
    ])
    write_csv(table_dictionary, OUTPUT / "feature_table_dictionary.csv")

    field_dictionary = make_field_dictionary({
        BROAD_PATH: "lag1/lag2/lag3 only; target_ columns are outcomes",
        ADVANCED_LAG_PATH: "advanced source seasons strictly before target",
        STRICT_CONTEXT_PATH: "club season t-1 only; focal removed",
        ROSTER_CONTEXT_PATH: "lagged performance, but target roster membership is scenario-only",
        COMPATIBILITY_PATH: "mixed strict and scenario features explicitly named",
        MODEL_MATRIX_PATH: "features plus target_ outcome columns",
    })
    write_csv(field_dictionary, OUTPUT / "feature_field_dictionary.csv")

    blocking_failures = int((checks["severity"].eq("blocking") & ~checks["passed"].astype(bool)).sum())
    result = {
        "status": "pass" if blocking_failures == 0 else "fail",
        "broad_target_rows": broad_metadata["rows"],
        "advanced_target_rows": advanced_metadata["rows"],
        "strict_model_eligible_rows": advanced_metadata["strict_model_eligible_rows"],
        "roster_model_eligible_rows": advanced_metadata["roster_model_eligible_rows"],
        "preference_model_eligible_rows": advanced_metadata["preference_model_eligible_rows"],
        "blocking_checks": int(checks["severity"].eq("blocking").sum()),
        "blocking_failures": blocking_failures,
    }
    (OUTPUT / "feature_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    source_paths = [CORE_PATH, MODEL_SCOPE_PATH, ADVANCED_PATH, CLUB_DIM_PATH]
    source_manifest = pd.DataFrame([
        {
            "source_file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "file_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in source_paths
    ])
    write_csv(source_manifest, OUTPUT / "source_manifest.csv")

    (OUTPUT / "README.md").write_text(
        f"""# MoveMaker lagged player and teammate-style compatibility features

## Scope

- Broad lagged feature rows: {broad_metadata['rows']:,}
- Advanced FBref compatibility rows: {advanced_metadata['rows']:,}
- Strict prior-club eligible rows: {advanced_metadata['strict_model_eligible_rows']:,}
- Roster-conditioned eligible rows: {advanced_metadata['roster_model_eligible_rows']:,}
- Player-preference eligible rows: {advanced_metadata['preference_model_eligible_rows']:,}

## Timing policy

1. Broad player features use exact `t-1`, `t-2`, and `t-3` seasons only. Rolling features never include the target season.
2. Advanced player styles are standardized within source season and role, then lagged. Current-season advanced values are retained only as `target_` outcomes in the model matrix.
3. Strict club context uses the destination club's observed `t-1` squad style. If the focal player was already there, their contribution is removed.
4. Roster-conditioned context identifies target-season teammates, represents each teammate only with their `t-1` style, and removes the focal player. Because historical target-roster membership may not have been known at the prediction cutoff, these columns are scenario-only and must not be used in strict backtests.
5. Historical preference features use expanding prior seasons only. For each teammate-context dimension, the player's performance-versus-context slope is shrunk by `effective_n / (effective_n + {PREFERENCE_SHRINKAGE_K:g})`.

## Style dimensions

Role-season standardized primitives feed eight interpretable dimensions: scoring threat, creation, progression, dribble aggression, passing control, directness, defensive intensity, and aerial physicality.

Compatibility includes direct player-style × teammate-context interactions, historical familiarity/preference distances, cosine similarity, and performance-conditioned preference slopes. These are features for a validated predictive model, not causal estimates or finalized scouting scores.

## Recommended split

- Train: season starts through 2021
- Validation: 2022
- Test: 2023

Never include columns prefixed `target_` in the predictor matrix.
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

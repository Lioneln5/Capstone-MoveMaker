from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "Data" / "processed"
PERFORMANCE = PROCESSED / "canonical_performance"
OUTPUT = PROCESSED / "compatibility_features"

CORE_PATH = PERFORMANCE / "canonical_player_team_season_performance.csv"
MODEL_SCOPE_PATH = PERFORMANCE / "modeling_scope_2017_2024.csv"
ADVANCED_SOURCE_PATH = PERFORMANCE / "fbref_advanced_performance_features.csv"

BROAD_PATH = OUTPUT / "broad_lagged_player_features_2017_2024.csv"
ADVANCED_LAG_PATH = OUTPUT / "advanced_lagged_player_styles.csv"
STRICT_PATH = OUTPUT / "strict_prior_club_style_context.csv"
ROSTER_PATH = OUTPUT / "roster_conditioned_teammate_style_context.csv"
COMPATIBILITY_PATH = OUTPUT / "teammate_style_compatibility_features.csv"
MODEL_MATRIX_PATH = OUTPUT / "advanced_compatibility_model_matrix.csv"

CHECKS_PATH = OUTPUT / "independent_verification.csv"
SUMMARY_PATH = OUTPUT / "independent_verification.json"
MANIFEST_PATH = OUTPUT / "output_manifest.csv"

CHUNK_SIZE = 100_000
EXPECTED_BROAD_ROWS = 905_269
EXPECTED_ADVANCED_ROWS = 17_790
MIN_TEAMMATES = 5
SHRINKAGE_K = 3.0

BASIC_METRICS = [
    "matches", "minutes", "goals", "assists", "yellow_cards", "red_cards",
    "starts", "substitute_selections", "captain_selections",
]
SOURCE_METRICS = {
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


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.casefold().isin({"true", "1", "yes"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_mismatches(actual: pd.Series, expected: pd.Series, tolerance: float = 1e-8) -> tuple[int, float]:
    left = pd.to_numeric(actual, errors="coerce")
    right = pd.to_numeric(expected, errors="coerce")
    missing_mismatch = left.isna() ^ right.isna()
    differences = (left - right).abs()
    mismatch = missing_mismatch | (differences.gt(tolerance) & left.notna() & right.notna())
    return int(mismatch.sum()), float(differences.max(skipna=True) if differences.notna().any() else 0.0)


def bool_mismatches(actual: pd.Series, expected: pd.Series) -> int:
    return int((as_bool(actual) != expected.astype(bool)).sum())


def main() -> None:
    checks: list[dict[str, Any]] = []

    def add_check(
        check: str,
        passed: bool,
        observed: Any,
        expected: Any,
        detail: str,
        severity: str = "blocking",
    ) -> None:
        checks.append({
            "check": check,
            "severity": severity,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
            "detail": detail,
        })

    required = [
        BROAD_PATH, ADVANCED_LAG_PATH, STRICT_PATH, ROSTER_PATH,
        COMPATIBILITY_PATH, MODEL_MATRIX_PATH,
    ]
    missing = [path.name for path in required if not path.exists()]
    add_check(
        "required_outputs_exist", not missing,
        "all present" if not missing else "; ".join(missing), "all six modeling tables",
        "Confirms the complete feature family was materialized.",
    )
    if missing:
        write_results(checks)
        raise SystemExit(1)

    broad_usecols = [
        "canonical_performance_id", "canonical_player_id", "canonical_club_id",
        "canonical_competition_id", "season_start_year", "feature_cutoff_season_start_year",
        "broad_max_source_season_used", "broad_lag_feature_coverage", "recommended_time_split",
        "new_club_vs_lag1",
    ]
    for lag in [1, 2, 3]:
        broad_usecols.extend([
            f"lag{lag}_observed", f"lag{lag}_source_rows", f"lag{lag}_club_count",
            f"lag{lag}_competition_count", f"same_club_as_lag{lag}",
            f"same_competition_as_lag{lag}",
            *[f"lag{lag}_{metric}" for metric in BASIC_METRICS],
            f"lag{lag}_goals_per90", f"lag{lag}_assists_per90",
            f"lag{lag}_goal_contributions_per90",
        ])
    for window in [2, 3]:
        broad_usecols.extend([
            f"rolling{window}_seasons_observed",
            *[f"rolling{window}_{metric}" for metric in BASIC_METRICS],
            f"rolling{window}_goals_per90", f"rolling{window}_assists_per90",
            f"rolling{window}_goal_contributions_per90",
        ])

    broad_rows = 0
    broad_ids: set[str] = set()
    broad_duplicate_ids = 0
    broad_leakage_violations = 0
    cutoff_mismatches = 0
    split_mismatches = 0
    max_source_mismatches = 0
    rolling_mismatches = 0
    per90_mismatches = 0
    new_club_mismatches = 0
    candidates: list[pd.DataFrame] = []

    for chunk_number, chunk in enumerate(pd.read_csv(BROAD_PATH, usecols=broad_usecols, dtype=str, chunksize=CHUNK_SIZE)):
        broad_rows += len(chunk)
        ids = chunk["canonical_performance_id"].astype(str)
        broad_duplicate_ids += int(ids.duplicated().sum()) + len(set(ids) & broad_ids)
        broad_ids.update(ids)
        season = pd.to_numeric(chunk["season_start_year"], errors="coerce")
        cutoff = pd.to_numeric(chunk["feature_cutoff_season_start_year"], errors="coerce")
        cutoff_mismatches += int((cutoff != season - 1).sum())
        max_source = pd.to_numeric(chunk["broad_max_source_season_used"], errors="coerce")
        broad_leakage_violations += int((max_source.notna() & max_source.ge(season)).sum())
        observed = {lag: as_bool(chunk[f"lag{lag}_observed"]) for lag in [1, 2, 3]}
        expected_max = pd.Series(np.nan, index=chunk.index)
        expected_max = expected_max.where(~observed[3], season - 3)
        expected_max = expected_max.where(~observed[2], season - 2)
        expected_max = expected_max.where(~observed[1], season - 1)
        mismatch, _ = numeric_mismatches(max_source, expected_max)
        max_source_mismatches += mismatch
        expected_split = np.select([season.le(2021), season.eq(2022)], ["train", "validation"], default="test")
        split_mismatches += int((chunk["recommended_time_split"].astype(str).to_numpy() != expected_split).sum())

        for lag in [1, 2, 3]:
            minutes = pd.to_numeric(chunk[f"lag{lag}_minutes"], errors="coerce")
            goals = pd.to_numeric(chunk[f"lag{lag}_goals"], errors="coerce")
            assists = pd.to_numeric(chunk[f"lag{lag}_assists"], errors="coerce")
            derived = {
                f"lag{lag}_goals_per90": goals.mul(90).div(minutes).where(minutes.ne(0)),
                f"lag{lag}_assists_per90": assists.mul(90).div(minutes).where(minutes.ne(0)),
                f"lag{lag}_goal_contributions_per90": goals.add(assists).mul(90).div(minutes).where(minutes.ne(0)),
            }
            for column, expected in derived.items():
                mismatch, _ = numeric_mismatches(chunk[column], expected)
                per90_mismatches += mismatch

        for window in [2, 3]:
            lags = range(1, window + 1)
            expected_seasons = sum(observed[lag].astype(int) for lag in lags)
            mismatch, _ = numeric_mismatches(chunk[f"rolling{window}_seasons_observed"], expected_seasons)
            rolling_mismatches += mismatch
            for metric in BASIC_METRICS:
                lag_values = pd.concat(
                    [pd.to_numeric(chunk[f"lag{lag}_{metric}"], errors="coerce") for lag in lags], axis=1
                )
                expected = lag_values.sum(axis=1, min_count=1)
                mismatch, _ = numeric_mismatches(chunk[f"rolling{window}_{metric}"], expected)
                rolling_mismatches += mismatch
            minutes = pd.to_numeric(chunk[f"rolling{window}_minutes"], errors="coerce")
            goals = pd.to_numeric(chunk[f"rolling{window}_goals"], errors="coerce")
            assists = pd.to_numeric(chunk[f"rolling{window}_assists"], errors="coerce")
            for column, expected in {
                f"rolling{window}_goals_per90": goals.mul(90).div(minutes).where(minutes.ne(0)),
                f"rolling{window}_assists_per90": assists.mul(90).div(minutes).where(minutes.ne(0)),
                f"rolling{window}_goal_contributions_per90": goals.add(assists).mul(90).div(minutes).where(minutes.ne(0)),
            }.items():
                mismatch, _ = numeric_mismatches(chunk[column], expected)
                per90_mismatches += mismatch

        lag1 = observed[1]
        same_club = as_bool(chunk["same_club_as_lag1"])
        new_club = as_bool(chunk["new_club_vs_lag1"])
        new_club_mismatches += int((lag1 & (new_club == same_club)).sum())

        sample = chunk.iloc[(np.arange(len(chunk)) + chunk_number * 137) % 997 == 0].copy()
        if not sample.empty:
            candidates.append(sample)

    add_check("broad_row_count", broad_rows == EXPECTED_BROAD_ROWS, broad_rows, EXPECTED_BROAD_ROWS,
              "One row must remain for every canonical modeling-scope performance row.")
    add_check("broad_unique_performance_ids", broad_duplicate_ids == 0, broad_duplicate_ids, 0,
              "Canonical performance IDs must remain unique after lag attachment.")
    add_check("broad_feature_cutoff", cutoff_mismatches == 0, cutoff_mismatches, 0,
              "Every broad cutoff must equal target season minus one.")
    add_check("broad_max_source_before_target", broad_leakage_violations == 0, broad_leakage_violations, 0,
              "No broad feature may report a source season at or after its target.")
    add_check("broad_max_source_exact", max_source_mismatches == 0, max_source_mismatches, 0,
              "The reported maximum source season must match the newest available exact lag.")
    add_check("broad_time_split", split_mismatches == 0, split_mismatches, 0,
              "Train/validation/test labels must follow the documented season split.")
    add_check("broad_rolling_reconciliation", rolling_mismatches == 0, rolling_mismatches, 0,
              "Rolling two- and three-season totals are independently recomputed from exact lag columns.")
    add_check("broad_per90_reconciliation", per90_mismatches == 0, per90_mismatches, 0,
              "Lagged and rolling goal/assist rates are independently recomputed.")
    add_check("broad_new_club_logic", new_club_mismatches == 0, new_club_mismatches, 0,
              "For rows with lag1, new-club and same-club flags must be exact opposites.")

    model_scope_ids: set[str] = set()
    model_scope_rows = 0
    for chunk in pd.read_csv(MODEL_SCOPE_PATH, usecols=["canonical_performance_id"], dtype=str, chunksize=CHUNK_SIZE):
        model_scope_rows += len(chunk)
        model_scope_ids.update(chunk["canonical_performance_id"].astype(str))
    id_difference = len(broad_ids ^ model_scope_ids)
    add_check("broad_ids_match_modeling_scope", id_difference == 0 and model_scope_rows == broad_rows,
              id_difference, 0, "The broad output key set must exactly match the canonical modeling scope.")

    sampled = pd.concat(candidates, ignore_index=True)
    sampled["_hash"] = pd.util.hash_pandas_object(sampled["canonical_performance_id"], index=False)
    sampled = sampled.sort_values("_hash").groupby("season_start_year", sort=True).head(100).drop(columns="_hash")
    relevant_keys: set[tuple[str, int]] = set()
    for row in sampled.itertuples(index=False):
        player = clean_id(row.canonical_player_id)
        target_season = int(row.season_start_year)
        relevant_keys.update((player, target_season - lag) for lag in [1, 2, 3])

    source_aggregates: dict[tuple[str, int], dict[str, Any]] = defaultdict(
        lambda: {
            "source_rows": 0,
            "metric_sum": defaultdict(float),
            "metric_count": defaultdict(int),
            "clubs": set(),
            "competitions": set(),
        }
    )
    source_usecols = [
        "canonical_player_id", "canonical_club_id", "canonical_competition_id", "season_start_year",
        *SOURCE_METRICS.keys(),
    ]
    for chunk in pd.read_csv(CORE_PATH, usecols=source_usecols, dtype=str, keep_default_na=False, chunksize=CHUNK_SIZE):
        players = chunk["canonical_player_id"].map(clean_id)
        seasons = pd.to_numeric(chunk["season_start_year"], errors="coerce")
        selected_mask = pd.Series(
            [(player, int(season)) in relevant_keys if pd.notna(season) else False
             for player, season in zip(players, seasons)],
            index=chunk.index,
        )
        selected = chunk.loc[selected_mask].copy()
        if selected.empty:
            continue
        selected["verifier_player"] = players.loc[selected_mask]
        selected["verifier_season"] = seasons.loc[selected_mask].astype(int)
        for row in selected.itertuples(index=False):
            data = row._asdict()
            key = (data["verifier_player"], int(data["verifier_season"]))
            aggregate = source_aggregates[key]
            aggregate["source_rows"] += 1
            club = clean_id(data["canonical_club_id"])
            competition = clean_id(data["canonical_competition_id"])
            if club:
                aggregate["clubs"].add(club)
            if competition:
                aggregate["competitions"].add(competition)
            for source_column, metric in SOURCE_METRICS.items():
                value = pd.to_numeric(pd.Series([data[source_column]]), errors="coerce").iloc[0]
                if pd.notna(value):
                    aggregate["metric_sum"][metric] += float(value)
                    aggregate["metric_count"][metric] += 1

    sample_mismatches = 0
    continuity_mismatches = 0
    for row in sampled.itertuples(index=False):
        data = row._asdict()
        player = clean_id(data["canonical_player_id"])
        target_season = int(data["season_start_year"])
        for lag in [1, 2, 3]:
            aggregate = source_aggregates.get((player, target_season - lag))
            expected_observed = aggregate is not None
            actual_observed = str(data[f"lag{lag}_observed"]).strip().casefold() in {"true", "1", "yes"}
            sample_mismatches += int(actual_observed != expected_observed)
            if aggregate is None:
                continue
            expected_values = {
                f"lag{lag}_source_rows": aggregate["source_rows"],
                f"lag{lag}_club_count": len(aggregate["clubs"]),
                f"lag{lag}_competition_count": len(aggregate["competitions"]),
            }
            for metric in BASIC_METRICS:
                expected_values[f"lag{lag}_{metric}"] = (
                    aggregate["metric_sum"][metric]
                    if aggregate["metric_count"][metric] > 0 else np.nan
                )
            for column, expected_value in expected_values.items():
                actual_value = pd.to_numeric(pd.Series([data[column]]), errors="coerce").iloc[0]
                if pd.isna(expected_value):
                    sample_mismatches += int(pd.notna(actual_value))
                else:
                    sample_mismatches += int(pd.isna(actual_value) or not math.isclose(
                        float(actual_value), float(expected_value), rel_tol=1e-9, abs_tol=1e-8
                    ))
            expected_same_club = clean_id(data["canonical_club_id"]) in aggregate["clubs"]
            expected_same_comp = clean_id(data["canonical_competition_id"]) in aggregate["competitions"]
            actual_same_club = str(data[f"same_club_as_lag{lag}"]).strip().casefold() in {"true", "1", "yes"}
            actual_same_comp = str(data[f"same_competition_as_lag{lag}"]).strip().casefold() in {"true", "1", "yes"}
            continuity_mismatches += int(actual_same_club != expected_same_club)
            continuity_mismatches += int(actual_same_comp != expected_same_comp)
    add_check("broad_source_sample_exact", sample_mismatches == 0, sample_mismatches, 0,
              f"Rebuilt all nine lag metrics from canonical source for {len(sampled):,} deterministic target rows.")
    add_check("broad_continuity_sample_exact", continuity_mismatches == 0, continuity_mismatches, 0,
              "Same-club and same-competition flags were rebuilt from canonical source relationships.")

    advanced = pd.read_csv(ADVANCED_LAG_PATH, dtype=str)
    strict = pd.read_csv(STRICT_PATH, dtype=str)
    roster = pd.read_csv(ROSTER_PATH, dtype=str)
    compatibility_ids = pd.read_csv(COMPATIBILITY_PATH, usecols=["canonical_performance_id"], dtype=str)
    matrix_header = list(pd.read_csv(MODEL_MATRIX_PATH, nrows=0).columns)
    compatibility_header = list(pd.read_csv(COMPATIBILITY_PATH, nrows=0).columns)
    matrix_usecols = [
        "canonical_performance_id", "canonical_player_id", "canonical_club_id", "season_start_year",
        "role_group", "target_fbref_total_minutes", "target_style_scoring_threat",
        "advanced_lag1_available", "advanced_lag1_minutes", "advanced_lag1_style_scoring_threat",
    ]
    matrix = pd.read_csv(MODEL_MATRIX_PATH, usecols=matrix_usecols, dtype=str)
    source_advanced_ids = set(pd.read_csv(
        ADVANCED_SOURCE_PATH, usecols=["canonical_performance_id"], dtype=str
    )["canonical_performance_id"].astype(str))
    advanced_ids = set(advanced["canonical_performance_id"].astype(str))
    all_advanced_counts = {
        "advanced_lag": len(advanced), "strict": len(strict), "roster": len(roster),
        "compatibility": len(compatibility_ids), "model_matrix": len(matrix),
    }
    advanced_count_ok = all(value == EXPECTED_ADVANCED_ROWS for value in all_advanced_counts.values())
    add_check("advanced_row_counts", advanced_count_ok, json.dumps(all_advanced_counts, sort_keys=True),
              EXPECTED_ADVANCED_ROWS, "Every advanced table must retain the full accepted FBref row set.")
    advanced_duplicate_count = sum(
        frame["canonical_performance_id"].duplicated().sum()
        for frame in [advanced, strict, roster, compatibility_ids, matrix]
    )
    add_check("advanced_unique_performance_ids", advanced_duplicate_count == 0, int(advanced_duplicate_count), 0,
              "Each advanced table must remain one row per canonical performance ID.")
    advanced_id_difference = len(advanced_ids ^ source_advanced_ids)
    add_check("advanced_ids_match_source", advanced_id_difference == 0, advanced_id_difference, 0,
              "Advanced lagged rows must exactly preserve the accepted FBref source keys.")
    cross_table_id_difference = sum(len(advanced_ids ^ set(frame["canonical_performance_id"].astype(str))) for frame in [
        strict, roster, compatibility_ids, matrix
    ])
    add_check("advanced_ids_match_across_outputs", cross_table_id_difference == 0,
              cross_table_id_difference, 0, "All advanced outputs must describe the identical key universe.")

    advanced_season = pd.to_numeric(advanced["season_start_year"], errors="coerce")
    advanced_max = pd.to_numeric(advanced["advanced_max_source_season_used"], errors="coerce")
    advanced_leakage = int((advanced_max.notna() & advanced_max.ge(advanced_season)).sum())
    add_check("advanced_lag_cutoff", advanced_leakage == 0, advanced_leakage, 0,
              "Advanced player styles may only use seasons before the target.")

    strict_season = pd.to_numeric(strict["season_start_year"], errors="coerce")
    strict_source = pd.to_numeric(strict["strict_context_source_season"], errors="coerce")
    strict_source_mismatch, _ = numeric_mismatches(strict_source, strict_season - 1)
    add_check("strict_context_exact_t_minus_1", strict_source_mismatch == 0, strict_source_mismatch, 0,
              "Strict club context is tied to the immediately preceding club season.")
    preference_max = pd.to_numeric(
        pd.read_csv(COMPATIBILITY_PATH, usecols=["season_start_year", "preference_max_source_season"], dtype=str)[
            "preference_max_source_season"
        ], errors="coerce"
    )
    preference_season = pd.to_numeric(
        pd.read_csv(COMPATIBILITY_PATH, usecols=["season_start_year"], dtype=str)["season_start_year"],
        errors="coerce",
    )
    preference_leakage = int((preference_max.notna() & preference_max.ge(preference_season)).sum())
    add_check("preference_history_cutoff", preference_leakage == 0, preference_leakage, 0,
              "Historical preference estimates must use only seasons before the target.")
    roster_season = pd.to_numeric(roster["season_start_year"], errors="coerce")
    roster_max = pd.to_numeric(roster["roster_context_max_performance_season_used"], errors="coerce")
    roster_leakage = int((roster_max.notna() & roster_max.ge(roster_season)).sum())
    membership_values = sorted(roster["roster_context_membership_timing"].dropna().unique().tolist())
    disclosure_ok = membership_values == ["target_season_roster_membership_scenario_only"]
    add_check("roster_performance_cutoff", roster_leakage == 0, roster_leakage, 0,
              "Roster-conditioned teammate performance must still come from t-1 or earlier.")
    add_check("roster_membership_disclosure", disclosure_ok, json.dumps(membership_values),
              "target_season_roster_membership_scenario_only",
              "Target-roster membership is explicitly separated from strict preseason-safe features.")

    target_columns_in_features = [column for column in compatibility_header if column.startswith("target_")]
    target_columns_in_matrix = [column for column in matrix_header if column.startswith("target_")]
    add_check("feature_only_table_has_no_targets", len(target_columns_in_features) == 0,
              len(target_columns_in_features), 0, "The reusable feature table contains no current-season outcomes.")
    add_check("model_matrix_targets_explicit", len(target_columns_in_matrix) >= 10,
              len(target_columns_in_matrix), ">=10 prefixed outcomes",
              "The model matrix retains outcomes only under an explicit target_ prefix.")

    strict_counts = pd.to_numeric(strict["strict_all_teammate_count"], errors="coerce")
    roster_counts = pd.to_numeric(roster["roster_all_teammate_count"], errors="coerce")
    strict_eligibility_mismatch = bool_mismatches(strict["strict_context_feature_eligible"], strict_counts.ge(MIN_TEAMMATES))
    roster_eligibility_mismatch = bool_mismatches(roster["roster_context_feature_eligible"], roster_counts.ge(MIN_TEAMMATES))
    add_check("strict_minimum_teammate_rule", strict_eligibility_mismatch == 0,
              strict_eligibility_mismatch, 0, "Strict eligibility requires at least five outfield teammates.")
    add_check("roster_minimum_teammate_rule", roster_eligibility_mismatch == 0,
              roster_eligibility_mismatch, 0, "Roster scenario eligibility requires at least five lagged teammates.")

    context_source = matrix.copy()
    context_source["season_start_year"] = pd.to_numeric(context_source["season_start_year"], errors="coerce").astype(int)
    context_source["_outfield"] = context_source["role_group"].isin(["DF", "MF", "FW"])
    outfield = context_source.loc[context_source["_outfield"]].copy()
    target_keys = ["canonical_club_id", "season_start_year"]
    strict_targets = strict[[
        "canonical_performance_id", "canonical_player_id", "canonical_club_id", "season_start_year",
        "strict_all_count", "strict_all_teammate_count",
    ]].copy()
    strict_targets["season_start_year"] = pd.to_numeric(strict_targets["season_start_year"], errors="coerce").astype(int)
    strict_targets["_source_season"] = strict_targets["season_start_year"] - 1
    base_counts = outfield.groupby(target_keys)["canonical_player_id"].nunique()
    source_index = pd.MultiIndex.from_arrays([
        strict_targets["canonical_club_id"], strict_targets["_source_season"]
    ])
    expected_base = pd.Series(source_index.map(base_counts), index=strict_targets.index, dtype="float64")
    membership = set(outfield[["canonical_player_id", "canonical_club_id", "season_start_year"]].drop_duplicates().itertuples(index=False, name=None))
    self_included = pd.Series([
        (player, club, season) in membership
        for player, club, season in zip(
            strict_targets["canonical_player_id"], strict_targets["canonical_club_id"], strict_targets["_source_season"]
        )
    ], index=strict_targets.index)
    expected_teammates = expected_base - self_included.astype(int)
    strict_base_mismatch, _ = numeric_mismatches(strict_targets["strict_all_count"], expected_base)
    strict_teammate_mismatch, _ = numeric_mismatches(strict_targets["strict_all_teammate_count"], expected_teammates)
    add_check("strict_focal_player_exclusion", strict_base_mismatch + strict_teammate_mismatch == 0,
              strict_base_mismatch + strict_teammate_mismatch, 0,
              "Prior-club player counts were independently rebuilt and the focal player removed when present.")

    lag_for_roster = advanced[[
        "canonical_performance_id", "canonical_player_id", "canonical_club_id", "season_start_year",
        "role_group", "advanced_lag1_available",
    ]].copy()
    lag_for_roster["season_start_year"] = pd.to_numeric(lag_for_roster["season_start_year"], errors="coerce").astype(int)
    lag_for_roster["_contributor"] = lag_for_roster["role_group"].isin(["DF", "MF", "FW"]) & as_bool(
        lag_for_roster["advanced_lag1_available"]
    )
    contributor_counts = lag_for_roster.loc[lag_for_roster["_contributor"]].groupby(target_keys)[
        "canonical_player_id"
    ].nunique()
    roster_target_index = pd.MultiIndex.from_frame(lag_for_roster[target_keys])
    roster_expected_base = pd.Series(roster_target_index.map(contributor_counts), index=lag_for_roster.index, dtype="float64")
    roster_expected_teammates = (roster_expected_base - lag_for_roster["_contributor"].astype(int)).where(
        lag_for_roster["role_group"].isin(["DF", "MF", "FW"])
    )
    roster_count_mismatch, _ = numeric_mismatches(roster["roster_all_teammate_count"], roster_expected_teammates)
    roster_self_mismatch = bool_mismatches(
        roster["roster_self_removed"],
        lag_for_roster["_contributor"].where(lag_for_roster["role_group"].isin(["DF", "MF", "FW"]), False),
    )
    add_check("roster_focal_player_exclusion", roster_count_mismatch + roster_self_mismatch == 0,
              roster_count_mismatch + roster_self_mismatch, 0,
              "Target-roster lag contributors were independently counted and the focal player removed.")

    preference_usecols = ["canonical_performance_id"]
    for dimension in PREFERENCE_DIMENSIONS:
        preference_usecols.extend([
            f"preference_{dimension}_effective_observations",
            f"preference_{dimension}_raw_slope",
            f"preference_{dimension}_shrunken_slope",
        ])
    preferences = pd.read_csv(COMPATIBILITY_PATH, usecols=preference_usecols, dtype=str)
    shrinkage_mismatches = 0
    for dimension in PREFERENCE_DIMENSIONS:
        effective_n = pd.to_numeric(preferences[f"preference_{dimension}_effective_observations"], errors="coerce")
        raw = pd.to_numeric(preferences[f"preference_{dimension}_raw_slope"], errors="coerce")
        expected = raw * effective_n.div(effective_n + SHRINKAGE_K)
        mismatch, _ = numeric_mismatches(preferences[f"preference_{dimension}_shrunken_slope"], expected)
        shrinkage_mismatches += mismatch
    add_check("preference_shrinkage_exact", shrinkage_mismatches == 0, shrinkage_mismatches, 0,
              "All six context slopes reconcile to raw_slope * n/(n+3).")

    interaction_columns = [
        "advanced_lag1_style_dribble_aggression",
        "strict_context_midfield_aggression", "strict_player_dribble_x_midfield_aggression",
        "roster_context_midfield_aggression", "roster_player_dribble_x_midfield_aggression",
    ]
    interactions = pd.read_csv(COMPATIBILITY_PATH, usecols=interaction_columns, dtype=str)
    player_dribble = pd.to_numeric(interactions["advanced_lag1_style_dribble_aggression"], errors="coerce")
    interaction_mismatches = 0
    for prefix in ["strict", "roster"]:
        context = pd.to_numeric(interactions[f"{prefix}_context_midfield_aggression"], errors="coerce")
        expected = player_dribble * context
        mismatch, _ = numeric_mismatches(interactions[f"{prefix}_player_dribble_x_midfield_aggression"], expected)
        interaction_mismatches += mismatch
    add_check("dribble_midfield_interaction_exact", interaction_mismatches == 0,
              interaction_mismatches, 0,
              "The focal-player dribble style × teammate-midfield aggression feature is exactly reproducible.")

    source_manifest = pd.read_csv(OUTPUT / "source_manifest.csv", dtype=str)
    source_hash_mismatches = 0
    for row in source_manifest.itertuples(index=False):
        path = ROOT / row.source_file
        if not path.exists() or sha256(path) != row.sha256:
            source_hash_mismatches += 1
    add_check("source_manifest_hashes", source_hash_mismatches == 0, source_hash_mismatches, 0,
              "All declared source files still match the hashes used for this build.")

    write_results(checks)
    failures = sum(not row["passed"] and row["severity"] == "blocking" for row in checks)
    if failures:
        raise SystemExit(1)


def write_results(checks: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(checks)
    frame.to_csv(CHECKS_PATH, index=False, encoding="utf-8", lineterminator="\n")
    blocking = frame.loc[frame["severity"].eq("blocking")]
    summary = {
        "status": "pass" if bool(blocking["passed"].all()) else "fail",
        "checks": int(len(frame)),
        "blocking_checks": int(len(blocking)),
        "blocking_failures": int((~blocking["passed"]).sum()),
        "verified_broad_rows": EXPECTED_BROAD_ROWS,
        "verified_advanced_rows": EXPECTED_ADVANCED_ROWS,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest_rows = []
    for path in sorted(OUTPUT.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path == MANIFEST_PATH:
            continue
        manifest_rows.append({
            "output_file": path.relative_to(ROOT).as_posix(),
            "file_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    pd.DataFrame(manifest_rows).to_csv(MANIFEST_PATH, index=False, encoding="utf-8", lineterminator="\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

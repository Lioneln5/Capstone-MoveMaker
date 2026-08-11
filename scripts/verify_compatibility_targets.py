from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "Data" / "processed"
OUTPUT = PROCESSED / "compatibility_targets"

TRANSFER_COHORT_PATH = PROCESSED / "transfer_cohort_master.csv"
CANONICAL_TRANSFER_PATH = PROCESSED / "canonical_integration" / "canonical_transfer_history.csv"
ADVANCED_MATRIX_PATH = PROCESSED / "compatibility_features" / "advanced_compatibility_model_matrix.csv"

TARGETS_PATH = OUTPUT / "transfer_compatibility_targets.csv"
FEATURES_PATH = OUTPUT / "transfer_compatibility_strict_features.csv"
MODEL_PATH = OUTPUT / "transfer_compatibility_strict_model_matrix.csv"
CHECKS_PATH = OUTPUT / "independent_verification.csv"
SUMMARY_PATH = OUTPUT / "independent_verification.json"
MANIFEST_PATH = OUTPUT / "output_manifest.csv"

EXPECTED_TARGET_ROWS = 936
EXPECTED_CANDIDATE_ROWS = 944
MIN_TARGET_MINUTES = 450.0
MIN_BASELINE_MINUTES = 450.0
OUTFIELD_ROLES = {"DF", "MF", "FW"}
PRIMARY_TRANSFER_TYPES = {"transfer", "loan"}


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


def numeric_mismatches(actual: pd.Series, expected: pd.Series, tolerance: float = 1e-9) -> tuple[int, float]:
    left = pd.to_numeric(actual, errors="coerce")
    right = pd.to_numeric(expected, errors="coerce")
    missing = left.isna() ^ right.isna()
    difference = (left - right).abs()
    mismatch = missing | (left.notna() & right.notna() & difference.gt(tolerance))
    return int(mismatch.sum()), float(difference.max(skipna=True) if difference.notna().any() else 0.0)


def target_season(dates: pd.Series) -> pd.Series:
    return dates.dt.year.where(dates.dt.month.ge(6), dates.dt.year - 1).astype("Int64")


def window(dates: pd.Series) -> pd.Series:
    month = dates.dt.month
    return pd.Series(
        np.select(
            [month.between(6, 10), month.isin([1, 2])],
            ["summer_primary", "winter"],
            default="off_window",
        ),
        index=dates.index,
    )


def main() -> None:
    checks: list[dict[str, Any]] = []

    def add(
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

    required = [TARGETS_PATH, FEATURES_PATH, MODEL_PATH]
    missing = [path.name for path in required if not path.exists()]
    add(
        "required_target_outputs_exist",
        not missing,
        "all present" if not missing else "; ".join(missing),
        "targets, strict features, strict model matrix",
        "Confirms the first-version target family was fully materialized.",
    )
    if missing:
        write_results(checks)
        raise SystemExit(1)

    targets = pd.read_csv(TARGETS_PATH, dtype=str, keep_default_na=False)
    features = pd.read_csv(FEATURES_PATH, dtype=str, keep_default_na=False)
    model = pd.read_csv(MODEL_PATH, dtype=str, keep_default_na=False)

    add(
        "target_row_count",
        len(targets) == EXPECTED_TARGET_ROWS,
        len(targets),
        EXPECTED_TARGET_ROWS,
        "One selected inbound event per unique destination player-season.",
    )
    duplicate_targets = int(targets["canonical_performance_id"].duplicated().sum())
    add(
        "target_performance_ids_unique",
        duplicate_targets == 0,
        duplicate_targets,
        0,
        "Destination performance grain must not be overweighted by repeated transfer events.",
    )
    cross_output_difference = (
        len(set(targets["canonical_performance_id"]) ^ set(features["canonical_performance_id"]))
        + len(set(targets["canonical_performance_id"]) ^ set(model["canonical_performance_id"]))
    )
    add(
        "keys_match_across_target_outputs",
        cross_output_difference == 0 and len(features) == len(model) == len(targets),
        cross_output_difference,
        0,
        "Targets, strict predictors, and labeled model matrix must share the identical key set.",
    )

    transfer_columns = [
        "transfer_event_id", "player_id", "transfer_date", "from_club_id", "to_club_id"
    ]
    transfers = pd.read_csv(
        TRANSFER_COHORT_PATH, usecols=transfer_columns, dtype=str, keep_default_na=False
    )
    for column in ["player_id", "from_club_id", "to_club_id"]:
        transfers[column] = transfers[column].map(clean_id)
    transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"], errors="coerce")
    transfers["season_start_year"] = target_season(transfers["transfer_date"])
    transfers["expected_window"] = window(transfers["transfer_date"])

    advanced_columns = [
        "canonical_performance_id", "canonical_player_id", "canonical_club_id",
        "canonical_competition_id", "season_start_year", "role_group",
        "target_fbref_matches_played", "target_fbref_total_minutes", "target_role_performance_index",
        "advanced_lag1_role_performance_index", "advanced_lag1_minutes",
        "advanced_lag2_role_performance_index", "advanced_lag2_minutes",
        "advanced_lag3_role_performance_index", "advanced_lag3_minutes",
    ]
    advanced = pd.read_csv(
        ADVANCED_MATRIX_PATH, usecols=advanced_columns, dtype=str, keep_default_na=False
    )
    for column in ["canonical_player_id", "canonical_club_id", "canonical_competition_id"]:
        advanced[column] = advanced[column].map(clean_id)
    advanced["season_start_year"] = pd.to_numeric(advanced["season_start_year"], errors="raise").astype("int64")

    candidates = transfers.merge(
        advanced,
        how="inner",
        left_on=["player_id", "to_club_id", "season_start_year"],
        right_on=["canonical_player_id", "canonical_club_id", "season_start_year"],
        validate="many_to_many",
    )
    add(
        "source_join_candidate_count",
        len(candidates) == EXPECTED_CANDIDATE_ROWS,
        len(candidates),
        EXPECTED_CANDIDATE_ROWS,
        "Independently rebuilds transfer events matching destination FBref player-seasons.",
    )
    candidates["same_destination_event_count"] = candidates.groupby("canonical_performance_id")[
        "transfer_event_id"
    ].transform("size")
    selected = candidates.sort_values(
        ["canonical_performance_id", "transfer_date", "transfer_event_id"], kind="stable"
    ).drop_duplicates("canonical_performance_id", keep="first")
    selected = selected.set_index("canonical_performance_id")
    target_by_id = targets.set_index("canonical_performance_id")
    common_ids = target_by_id.index.intersection(selected.index)
    selected_event_mismatches = int((
        target_by_id.loc[common_ids, "transfer_event_id"].astype(str)
        != selected.loc[common_ids, "transfer_event_id"].astype(str)
    ).sum()) + len(set(target_by_id.index) ^ set(selected.index))
    add(
        "earliest_inbound_event_selection",
        selected_event_mismatches == 0,
        selected_event_mismatches,
        0,
        "The earliest inbound event is independently selected for every destination player-season.",
    )
    event_count_mismatch, _ = numeric_mismatches(
        target_by_id.loc[common_ids, "same_destination_event_count"],
        selected.loc[common_ids, "same_destination_event_count"],
    )
    expected_collapsed = pd.to_numeric(
        selected.loc[common_ids, "same_destination_event_count"], errors="coerce"
    ) - 1
    collapsed_mismatch, _ = numeric_mismatches(
        target_by_id.loc[common_ids, "later_same_destination_events_collapsed"], expected_collapsed
    )
    add(
        "later_event_collapse_counts",
        event_count_mismatch + collapsed_mismatch == 0,
        event_count_mismatch + collapsed_mismatch,
        0,
        "All repeated inbound events are counted and reconciled rather than silently dropped.",
    )

    target_dates = pd.to_datetime(targets["transfer_date"], errors="coerce")
    expected_season = target_season(target_dates)
    season_mismatch, _ = numeric_mismatches(targets["season_start_year"], expected_season)
    window_mismatch = int((targets["transfer_window"].astype(str) != window(target_dates)).sum())
    add(
        "target_season_assignment_from_date",
        season_mismatch == 0,
        season_mismatch,
        0,
        "June–December map to that season start; January–May map to the preceding start year.",
    )
    add(
        "transfer_window_assignment",
        window_mismatch == 0,
        window_mismatch,
        0,
        "June–October are primary summer arrivals; January–February are winter arrivals.",
    )

    canonical = pd.read_csv(
        CANONICAL_TRANSFER_PATH,
        usecols=[
            "canonical_player_id", "transfer_date", "from_club_id", "to_club_id",
            "canonical_transfer_type",
        ],
        dtype=str,
        keep_default_na=False,
    )
    for column in ["canonical_player_id", "from_club_id", "to_club_id"]:
        canonical[column] = canonical[column].map(clean_id)
    canonical["transfer_date"] = pd.to_datetime(canonical["transfer_date"], errors="coerce")
    selected_reset = selected.reset_index()
    canonical_join = selected_reset.merge(
        canonical,
        how="left",
        left_on=["player_id", "transfer_date", "from_club_id", "to_club_id"],
        right_on=["canonical_player_id", "transfer_date", "from_club_id", "to_club_id"],
        validate="many_to_one",
    ).set_index("canonical_performance_id")
    type_mismatch = int((
        target_by_id.loc[common_ids, "canonical_transfer_type"].fillna("").astype(str)
        != canonical_join.loc[common_ids, "canonical_transfer_type"].fillna("").astype(str)
    ).sum())
    add(
        "canonical_transfer_type_join",
        type_mismatch == 0,
        type_mismatch,
        0,
        "Transfer, loan, and loan-return classifications are independently matched from canonical history.",
    )

    club_matches = advanced.assign(
        _matches=pd.to_numeric(advanced["target_fbref_matches_played"], errors="coerce")
    ).groupby(["canonical_club_id", "canonical_competition_id", "season_start_year"])["_matches"].max()
    target_index = pd.MultiIndex.from_arrays([
        targets["canonical_club_id"].map(clean_id),
        targets["canonical_competition_id"].map(clean_id),
        pd.to_numeric(targets["season_start_year"], errors="coerce"),
    ])
    expected_matches = pd.Series(target_index.map(club_matches), index=targets.index, dtype="float64")
    opportunity_mismatch, _ = numeric_mismatches(targets["target_club_match_opportunities"], expected_matches)
    expected_available_minutes = expected_matches * 90.0
    available_mismatch, _ = numeric_mismatches(
        targets["target_available_player_minutes"], expected_available_minutes
    )
    add(
        "club_opportunity_denominator",
        opportunity_mismatch + available_mismatch == 0,
        opportunity_mismatch + available_mismatch,
        0,
        "Club match opportunities and ×90 denominator are rebuilt from the full advanced matrix.",
    )

    advanced_by_id = advanced.set_index("canonical_performance_id")
    expected_target_minutes = pd.to_numeric(
        advanced_by_id.loc[targets["canonical_performance_id"], "target_fbref_total_minutes"], errors="coerce"
    ).reset_index(drop=True)
    expected_role_index = pd.to_numeric(
        advanced_by_id.loc[targets["canonical_performance_id"], "target_role_performance_index"], errors="coerce"
    ).reset_index(drop=True)
    expected_share = expected_target_minutes.div(expected_available_minutes).where(expected_available_minutes.gt(0))
    minutes_mismatch, _ = numeric_mismatches(targets["target_destination_minutes"], expected_target_minutes)
    share_mismatch, max_share_difference = numeric_mismatches(
        targets["target_destination_minutes_share"], expected_share
    )
    role_index_mismatch, _ = numeric_mismatches(targets["target_role_performance_index"], expected_role_index)
    add(
        "destination_minutes_target_exact",
        minutes_mismatch == 0,
        minutes_mismatch,
        0,
        "Destination minutes are copied from the accepted destination FBref player-season.",
    )
    add(
        "destination_minutes_share_exact",
        share_mismatch == 0,
        f"mismatches={share_mismatch}; max_abs_difference={max_share_difference}",
        0,
        "Every opportunity target is independently divided by the club-season denominator.",
    )
    add(
        "role_performance_target_exact",
        role_index_mismatch == 0,
        role_index_mismatch,
        0,
        "Current role-performance targets exactly match the advanced destination outcomes.",
    )

    target_advanced = advanced_by_id.loc[targets["canonical_performance_id"]].reset_index()
    numerator = pd.Series(0.0, index=target_advanced.index)
    denominator = pd.Series(0.0, index=target_advanced.index)
    seasons = pd.Series(0, index=target_advanced.index, dtype="int64")
    for lag in [1, 2, 3]:
        performance = pd.to_numeric(
            target_advanced[f"advanced_lag{lag}_role_performance_index"], errors="coerce"
        )
        minutes = pd.to_numeric(target_advanced[f"advanced_lag{lag}_minutes"], errors="coerce")
        valid = performance.notna() & minutes.gt(0)
        numerator += performance.fillna(0) * minutes.where(valid, 0)
        denominator += minutes.where(valid, 0)
        seasons += valid.astype(int)
    expected_baseline = numerator.div(denominator).where(denominator.gt(0))
    expected_change = expected_role_index - expected_baseline
    baseline_mismatch, _ = numeric_mismatches(
        targets["prior3_role_performance_baseline"], expected_baseline
    )
    baseline_minutes_mismatch, _ = numeric_mismatches(
        targets["prior3_role_performance_baseline_minutes"], denominator
    )
    baseline_seasons_mismatch, _ = numeric_mismatches(
        targets["prior3_role_performance_baseline_seasons"], seasons
    )
    change_mismatch, max_change_difference = numeric_mismatches(
        targets["target_performance_change_rolling3"], expected_change
    )
    add(
        "prior3_performance_baseline_exact",
        baseline_mismatch + baseline_minutes_mismatch + baseline_seasons_mismatch == 0,
        baseline_mismatch + baseline_minutes_mismatch + baseline_seasons_mismatch,
        0,
        "Minutes-weighted t-1/t-2/t-3 role-performance baseline is independently rebuilt.",
    )
    add(
        "performance_change_target_exact",
        change_mismatch == 0,
        f"mismatches={change_mismatch}; max_abs_difference={max_change_difference}",
        0,
        "Adaptation target equals current role performance minus the rebuilt prior3 baseline.",
    )

    strict = as_bool(targets["strict_model_feature_eligible"])
    common = (
        targets["transfer_window"].eq("summer_primary")
        & targets["canonical_transfer_type"].isin(PRIMARY_TRANSFER_TYPES)
        & targets["role_group"].isin(OUTFIELD_ROLES)
        & strict
    )
    expected_opportunity = common & expected_share.notna()
    expected_performance = common & expected_target_minutes.ge(MIN_TARGET_MINUTES) & expected_role_index.notna()
    expected_adaptation = (
        expected_performance
        & expected_baseline.notna()
        & denominator.ge(MIN_BASELINE_MINUTES)
        & expected_change.notna()
    )
    opportunity_eligibility_mismatch = int((
        as_bool(targets["eligible_opportunity_model_primary"]) != expected_opportunity
    ).sum())
    performance_eligibility_mismatch = int((
        as_bool(targets["eligible_performance_model_primary"]) != expected_performance
    ).sum())
    adaptation_eligibility_mismatch = int((
        as_bool(targets["eligible_adaptation_model_primary"]) != expected_adaptation
    ).sum())
    add(
        "opportunity_primary_eligibility_exact",
        opportunity_eligibility_mismatch == 0,
        opportunity_eligibility_mismatch,
        0,
        "Primary opportunity cohort requires summer transfer/loan, outfield role, and strict features.",
    )
    add(
        "performance_primary_eligibility_exact",
        performance_eligibility_mismatch == 0,
        performance_eligibility_mismatch,
        0,
        "Performance cohort additionally requires 450 target minutes and a role-performance outcome.",
    )
    add(
        "adaptation_primary_eligibility_exact",
        adaptation_eligibility_mismatch == 0,
        adaptation_eligibility_mismatch,
        0,
        "Adaptation cohort additionally requires 450 prior-baseline minutes and a change outcome.",
    )
    reason_mismatch = 0
    for flag, reason_column in [
        ("eligible_opportunity_model_primary", "opportunity_model_exclusion_reasons"),
        ("eligible_performance_model_primary", "performance_model_exclusion_reasons"),
        ("eligible_adaptation_model_primary", "adaptation_model_exclusion_reasons"),
    ]:
        reason_mismatch += int((
            as_bool(targets[flag]) != targets[reason_column].astype(str).eq("")
        ).sum())
    add(
        "eligibility_reasons_reconcile",
        reason_mismatch == 0,
        reason_mismatch,
        0,
        "A row is eligible if and only if its model-specific exclusion reason is blank.",
    )
    partial_primary = int((
        ~targets["transfer_window"].eq("summer_primary")
        & (
            as_bool(targets["eligible_opportunity_model_primary"])
            | as_bool(targets["eligible_performance_model_primary"])
            | as_bool(targets["eligible_adaptation_model_primary"])
        )
    ).sum())
    loan_return_primary = int((
        targets["canonical_transfer_type"].eq("loan_return")
        & (
            as_bool(targets["eligible_opportunity_model_primary"])
            | as_bool(targets["eligible_performance_model_primary"])
            | as_bool(targets["eligible_adaptation_model_primary"])
        )
    ).sum())
    add(
        "partial_season_moves_excluded_from_primary",
        partial_primary == 0,
        partial_primary,
        0,
        "Winter/off-window moves remain available for sensitivity analysis but are excluded from primary targets.",
    )
    add(
        "loan_returns_excluded_from_primary",
        loan_return_primary == 0,
        loan_return_primary,
        0,
        "Returning loan players are not treated as new external recruitment decisions in the primary cohort.",
    )

    feature_target_columns = [column for column in features.columns if column.startswith("target_")]
    feature_roster_columns = [column for column in features.columns if column.startswith("roster_")]
    model_roster_columns = [column for column in model.columns if column.startswith("roster_")]
    add(
        "feature_only_separates_targets",
        not feature_target_columns,
        len(feature_target_columns),
        0,
        "Strict scoring features contain no target-season outcomes.",
    )
    add(
        "strict_outputs_exclude_roster_scenarios",
        not feature_roster_columns and not model_roster_columns,
        len(feature_roster_columns) + len(model_roster_columns),
        0,
        "Scenario-only target-roster membership is absent from both strict transfer matrices.",
    )

    model_by_id = model.set_index("canonical_performance_id")
    target_by_id = targets.set_index("canonical_performance_id")
    model_target_mismatches = 0
    for column in [
        "target_destination_minutes_share", "target_role_performance_index",
        "target_performance_change_rolling3",
    ]:
        mismatch, _ = numeric_mismatches(
            model_by_id.loc[target_by_id.index, column], target_by_id[column]
        )
        model_target_mismatches += mismatch
    add(
        "model_matrix_targets_match_target_table",
        model_target_mismatches == 0,
        model_target_mismatches,
        0,
        "The three labeled model-matrix outcomes exactly match the canonical target table.",
    )

    season = pd.to_numeric(targets["season_start_year"], errors="coerce")
    advanced_max = pd.to_numeric(targets["advanced_max_source_season_used"], errors="coerce")
    preference_max = pd.to_numeric(targets["preference_max_source_season"], errors="coerce")
    strict_source = pd.to_numeric(targets["strict_context_source_season"], errors="coerce")
    leakage = int((advanced_max.notna() & advanced_max.ge(season)).sum())
    leakage += int((preference_max.notna() & preference_max.ge(season)).sum())
    strict_source_mismatch = int((strict_source.notna() & strict_source.ne(season - 1)).sum())
    add(
        "predictor_timing_cutoffs",
        leakage + strict_source_mismatch == 0,
        leakage + strict_source_mismatch,
        0,
        "Lagged player, preference, and strict club context sources all precede the destination target season.",
    )

    source_manifest = pd.read_csv(OUTPUT / "source_manifest.csv", dtype=str)
    hash_mismatches = 0
    for row in source_manifest.itertuples(index=False):
        path = ROOT / row.source_file
        if not path.exists() or sha256(path) != row.sha256:
            hash_mismatches += 1
    add(
        "source_manifest_hashes",
        hash_mismatches == 0,
        hash_mismatches,
        0,
        "All declared source files match the hashes used to construct the target layer.",
    )

    write_results(checks)
    blocking_failures = sum(
        not row["passed"] and row["severity"] == "blocking" for row in checks
    )
    if blocking_failures:
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
        "verified_target_rows": EXPECTED_TARGET_ROWS,
        "verified_candidate_rows": EXPECTED_CANDIDATE_ROWS,
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
    pd.DataFrame(manifest_rows).to_csv(
        MANIFEST_PATH, index=False, encoding="utf-8", lineterminator="\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

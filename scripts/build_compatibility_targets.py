from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "Data" / "processed"
OUTPUT = PROCESSED / "compatibility_targets"

TRANSFER_COHORT_PATH = PROCESSED / "transfer_cohort_master.csv"
CANONICAL_TRANSFER_PATH = PROCESSED / "canonical_integration" / "canonical_transfer_history.csv"
COMPATIBILITY_PATH = PROCESSED / "compatibility_features" / "teammate_style_compatibility_features.csv"
ADVANCED_MATRIX_PATH = PROCESSED / "compatibility_features" / "advanced_compatibility_model_matrix.csv"

TARGETS_PATH = OUTPUT / "transfer_compatibility_targets.csv"
FEATURES_PATH = OUTPUT / "transfer_compatibility_strict_features.csv"
MODEL_MATRIX_PATH = OUTPUT / "transfer_compatibility_strict_model_matrix.csv"

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


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.casefold().isin({"true", "1", "yes"})


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numeric_denominator = pd.to_numeric(denominator, errors="coerce")
    return pd.to_numeric(numerator, errors="coerce").div(numeric_denominator).where(numeric_denominator.gt(0))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def transfer_window(dates: pd.Series) -> pd.Series:
    month = dates.dt.month
    return pd.Series(
        np.select(
            [month.between(6, 10), month.isin([1, 2])],
            ["summer_primary", "winter"],
            default="off_window",
        ),
        index=dates.index,
    )


def target_season_from_date(dates: pd.Series) -> pd.Series:
    start = dates.dt.year.where(dates.dt.month.ge(6), dates.dt.year - 1)
    return pd.to_numeric(start, errors="coerce").astype("Int64")


def exclusion_reasons(frame: pd.DataFrame, target: str) -> pd.Series:
    results: list[str] = []
    for row in frame.itertuples(index=False):
        reasons: list[str] = []
        if row.transfer_window != "summer_primary":
            reasons.append("not_primary_summer_window")
        if row.canonical_transfer_type not in PRIMARY_TRANSFER_TYPES:
            reasons.append("transfer_type_not_transfer_or_loan")
        if row.role_group not in OUTFIELD_ROLES:
            reasons.append("role_not_outfield")
        if not bool(row.strict_model_feature_eligible):
            reasons.append("strict_compatibility_features_unavailable")
        if target == "opportunity":
            if not bool(row.target_destination_minutes_share_available):
                reasons.append("destination_minutes_share_unavailable")
        if target in {"performance", "adaptation"}:
            if pd.isna(row.target_destination_minutes) or row.target_destination_minutes < MIN_TARGET_MINUTES:
                reasons.append("destination_minutes_below_450")
            if not bool(row.target_role_performance_index_available):
                reasons.append("role_performance_target_unavailable")
        if target == "adaptation":
            if not bool(row.prior3_role_performance_baseline_available):
                reasons.append("prior3_performance_baseline_unavailable")
            elif row.prior3_role_performance_baseline_minutes < MIN_BASELINE_MINUTES:
                reasons.append("prior3_baseline_minutes_below_450")
            if not bool(row.target_performance_change_rolling3_available):
                reasons.append("performance_change_target_unavailable")
        results.append(" | ".join(reasons))
    return pd.Series(results, index=frame.index)


def field_usage(column: str, table: str) -> tuple[str, str, str]:
    if column in {
        "transfer_event_id", "canonical_transfer_event_id", "canonical_performance_id",
        "canonical_player_id", "canonical_club_id", "canonical_competition_id",
    } or column.endswith("_name") or column in {"canonical_season_key", "season_start_year", "role_group"}:
        return (
            "identifier_or_split_control",
            "not_a_predictor",
            "Join, grouping, audit, or chronological split field; do not encode raw identifiers as numeric predictors.",
        )
    if column.startswith("target_"):
        return (
            "target_or_target_diagnostic",
            "outcome_only",
            "Current destination-season outcome or its calculation component; never use as a predictor.",
        )
    if column.startswith("eligible_") or column.endswith("_exclusion_reasons"):
        return (
            "sample_control",
            "not_a_predictor",
            "Eligibility or exclusion control used to define the modeling cohort.",
        )
    if column.startswith("transfer_") or column in {
        "age_at_transfer", "position", "sub_position", "foot", "height_in_cm",
        "listed_market_value_at_transfer_eur", "from_club_id", "from_club_name",
        "to_club_id", "to_club_name", "canonical_transfer_type",
        "canonical_transfer_fee_status", "canonical_transfer_fee_eur",
    }:
        return (
            "transfer_time_feature_or_control",
            "available_at_transfer",
            "Transfer-event attribute available at or before the move; validate missingness and policy before modeling.",
        )
    if column.startswith("strict_") or column.startswith("advanced_") or column.startswith("preference_"):
        return (
            "strict_compatibility_predictor",
            "strict_preseason_safe",
            "Lagged player or prior-club compatibility feature constructed without target-season outcomes.",
        )
    if column.startswith("prior3_") or column.startswith("lag1_role_"):
        return (
            "target_baseline_diagnostic",
            "outcome_calculation_only",
            "Prior-history component used to define or audit performance change; omit when predicting absolute performance.",
        )
    return (
        "feature_or_control",
        "review_dictionary",
        f"Field emitted in {table}; review timing and usage before modeling.",
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    transfer_columns = [
        "transfer_event_id", "player_id", "player_name", "transfer_date", "transfer_season",
        "expected_transfer_season", "transfer_season_matches_date", "from_club_id", "from_club_name",
        "to_club_id", "to_club_name", "transfer_fee_eur", "fee_status",
        "listed_market_value_at_transfer_eur", "age_at_transfer", "position", "sub_position",
        "foot", "height_in_cm", "country_of_citizenship", "cohort_tier",
    ]
    transfers = pd.read_csv(
        TRANSFER_COHORT_PATH, usecols=transfer_columns, dtype=str, keep_default_na=False
    )
    for column in ["player_id", "from_club_id", "to_club_id"]:
        transfers[column] = transfers[column].map(clean_id)
    transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"], errors="coerce")
    transfers["season_start_year"] = target_season_from_date(transfers["transfer_date"])
    transfers["transfer_window"] = transfer_window(transfers["transfer_date"])
    season_start = pd.to_datetime(
        transfers["season_start_year"].astype("Int64").astype(str) + "-07-01", errors="coerce"
    )
    transfers["transfer_days_from_target_season_start"] = (transfers["transfer_date"] - season_start).dt.days

    advanced_columns = [
        "canonical_performance_id", "canonical_player_id", "canonical_player_name",
        "canonical_club_id", "canonical_club_name", "canonical_competition_id",
        "canonical_season_key", "season_start_year", "role_group", "recommended_time_split",
        "strict_model_feature_eligible", "preference_model_feature_eligible",
        "advanced_lag1_available", "advanced_lag1_role_group",
        "advanced_lag1_role_performance_index", "advanced_lag1_minutes",
        "advanced_lag2_role_performance_index", "advanced_lag2_minutes",
        "advanced_lag3_role_performance_index", "advanced_lag3_minutes",
        "advanced_rolling3_seasons_observed", "advanced_max_source_season_used",
        "preference_max_source_season", "strict_context_source_season",
        "target_fbref_matches_played", "target_fbref_total_minutes", "target_role_performance_index",
    ]
    advanced = pd.read_csv(
        ADVANCED_MATRIX_PATH, usecols=advanced_columns, dtype=str, keep_default_na=False
    )
    for column in ["canonical_player_id", "canonical_club_id", "canonical_competition_id"]:
        advanced[column] = advanced[column].map(clean_id)
    advanced["season_start_year"] = pd.to_numeric(advanced["season_start_year"], errors="raise").astype("int64")

    candidate = transfers.merge(
        advanced,
        how="inner",
        left_on=["player_id", "to_club_id", "season_start_year"],
        right_on=["canonical_player_id", "canonical_club_id", "season_start_year"],
        validate="many_to_many",
    )
    candidate["same_destination_event_count"] = candidate.groupby("canonical_performance_id")[
        "transfer_event_id"
    ].transform("size")
    candidate = candidate.sort_values(
        ["canonical_performance_id", "transfer_date", "transfer_event_id"], kind="stable"
    )
    selected = candidate.drop_duplicates("canonical_performance_id", keep="first").copy()
    selected["selected_destination_event_policy"] = "earliest_inbound_event_mapped_to_destination_player_season"
    selected["later_same_destination_events_collapsed"] = selected["same_destination_event_count"] - 1

    canonical_transfer_columns = [
        "canonical_transfer_event_id", "canonical_player_id", "transfer_date", "from_club_id", "to_club_id",
        "canonical_transfer_type", "canonical_transfer_fee_status", "canonical_transfer_fee_eur",
        "event_resolution_status", "event_conflict",
    ]
    canonical_transfers = pd.read_csv(
        CANONICAL_TRANSFER_PATH, usecols=canonical_transfer_columns, dtype=str, keep_default_na=False
    )
    for column in ["canonical_player_id", "from_club_id", "to_club_id"]:
        canonical_transfers[column] = canonical_transfers[column].map(clean_id)
    canonical_transfers["transfer_date"] = pd.to_datetime(canonical_transfers["transfer_date"], errors="coerce")
    canonical_transfers = canonical_transfers.rename(columns={
        "canonical_player_id": "_canonical_transfer_player_id",
        "from_club_id": "_canonical_transfer_from_club_id",
        "to_club_id": "_canonical_transfer_to_club_id",
    })
    selected = selected.merge(
        canonical_transfers,
        how="left",
        left_on=["player_id", "transfer_date", "from_club_id", "to_club_id"],
        right_on=[
            "_canonical_transfer_player_id", "transfer_date",
            "_canonical_transfer_from_club_id", "_canonical_transfer_to_club_id",
        ],
        validate="many_to_one",
    ).drop(columns=[
        "_canonical_transfer_player_id", "_canonical_transfer_from_club_id", "_canonical_transfer_to_club_id"
    ])

    selected["target_club_match_opportunities"] = selected.set_index(
        ["canonical_club_id", "canonical_competition_id", "season_start_year"]
    ).index.map(
        advanced.assign(
            _matches=pd.to_numeric(advanced["target_fbref_matches_played"], errors="coerce")
        ).groupby(["canonical_club_id", "canonical_competition_id", "season_start_year"])["_matches"].max()
    )
    selected["target_available_player_minutes"] = selected["target_club_match_opportunities"] * 90.0
    selected["target_destination_minutes"] = pd.to_numeric(
        selected["target_fbref_total_minutes"], errors="coerce"
    )
    selected["target_destination_minutes_share"] = safe_divide(
        selected["target_destination_minutes"], selected["target_available_player_minutes"]
    )
    selected["target_destination_minutes_share_available"] = selected[
        "target_destination_minutes_share"
    ].notna()
    selected["target_destination_minutes_share_denominator_policy"] = (
        "destination_club_competition_season_max_player_appearances_x_90"
    )

    selected["target_role_performance_index"] = pd.to_numeric(
        selected["target_role_performance_index"], errors="coerce"
    )
    selected["target_role_performance_index_available"] = selected[
        "target_role_performance_index"
    ].notna()
    baseline_numerator = pd.Series(0.0, index=selected.index)
    baseline_denominator = pd.Series(0.0, index=selected.index)
    baseline_seasons = pd.Series(0, index=selected.index, dtype="int64")
    for lag in [1, 2, 3]:
        performance = pd.to_numeric(
            selected[f"advanced_lag{lag}_role_performance_index"], errors="coerce"
        )
        minutes = pd.to_numeric(selected[f"advanced_lag{lag}_minutes"], errors="coerce")
        valid = performance.notna() & minutes.gt(0)
        baseline_numerator = baseline_numerator + performance.fillna(0) * minutes.where(valid, 0)
        baseline_denominator = baseline_denominator + minutes.where(valid, 0)
        baseline_seasons = baseline_seasons + valid.astype(int)
    selected["prior3_role_performance_baseline"] = baseline_numerator.div(
        baseline_denominator
    ).where(baseline_denominator.gt(0))
    selected["prior3_role_performance_baseline_minutes"] = baseline_denominator
    selected["prior3_role_performance_baseline_seasons"] = baseline_seasons
    selected["prior3_role_performance_baseline_available"] = selected[
        "prior3_role_performance_baseline"
    ].notna()
    selected["target_performance_change_rolling3"] = (
        selected["target_role_performance_index"] - selected["prior3_role_performance_baseline"]
    )
    selected["target_performance_change_rolling3_available"] = selected[
        "target_performance_change_rolling3"
    ].notna()
    selected["target_performance_change_definition"] = (
        "target_role_performance_index_minus_minutes_weighted_prior3_role_performance_index"
    )
    selected["lag1_role_matches_target_role"] = (
        selected["advanced_lag1_role_group"].astype(str).eq(selected["role_group"].astype(str))
    ).where(bool_series(selected["advanced_lag1_available"]))

    selected["strict_model_feature_eligible"] = bool_series(selected["strict_model_feature_eligible"])
    selected["preference_model_feature_eligible"] = bool_series(selected["preference_model_feature_eligible"])
    selected["transfer_type_primary_eligible"] = selected["canonical_transfer_type"].isin(
        PRIMARY_TRANSFER_TYPES
    )
    selected["outfield_role_eligible"] = selected["role_group"].isin(OUTFIELD_ROLES)
    selected["primary_summer_window_eligible"] = selected["transfer_window"].eq("summer_primary")

    selected["opportunity_model_exclusion_reasons"] = exclusion_reasons(selected, "opportunity")
    selected["performance_model_exclusion_reasons"] = exclusion_reasons(selected, "performance")
    selected["adaptation_model_exclusion_reasons"] = exclusion_reasons(selected, "adaptation")
    selected["eligible_opportunity_model_primary"] = selected[
        "opportunity_model_exclusion_reasons"
    ].eq("")
    selected["eligible_performance_model_primary"] = selected[
        "performance_model_exclusion_reasons"
    ].eq("")
    selected["eligible_adaptation_model_primary"] = selected[
        "adaptation_model_exclusion_reasons"
    ].eq("")

    target_columns = [
        "transfer_event_id", "canonical_transfer_event_id", "canonical_performance_id",
        "canonical_player_id", "canonical_player_name", "canonical_club_id", "canonical_club_name",
        "canonical_competition_id", "canonical_season_key", "season_start_year", "role_group",
        "recommended_time_split", "transfer_date", "transfer_window",
        "transfer_days_from_target_season_start", "player_id", "player_name", "from_club_id",
        "from_club_name", "to_club_id", "to_club_name", "canonical_transfer_type",
        "canonical_transfer_fee_status", "canonical_transfer_fee_eur", "transfer_fee_eur", "fee_status",
        "listed_market_value_at_transfer_eur", "age_at_transfer", "position", "sub_position", "foot",
        "height_in_cm", "country_of_citizenship", "cohort_tier", "event_resolution_status",
        "event_conflict", "selected_destination_event_policy", "same_destination_event_count",
        "later_same_destination_events_collapsed", "target_club_match_opportunities",
        "target_available_player_minutes", "target_destination_minutes",
        "target_destination_minutes_share", "target_destination_minutes_share_available",
        "target_destination_minutes_share_denominator_policy", "target_role_performance_index",
        "target_role_performance_index_available", "prior3_role_performance_baseline",
        "prior3_role_performance_baseline_minutes", "prior3_role_performance_baseline_seasons",
        "prior3_role_performance_baseline_available", "target_performance_change_rolling3",
        "target_performance_change_rolling3_available", "target_performance_change_definition",
        "lag1_role_matches_target_role", "advanced_max_source_season_used",
        "preference_max_source_season", "strict_context_source_season",
        "strict_model_feature_eligible", "preference_model_feature_eligible",
        "transfer_type_primary_eligible", "outfield_role_eligible", "primary_summer_window_eligible",
        "eligible_opportunity_model_primary", "eligible_performance_model_primary",
        "eligible_adaptation_model_primary", "opportunity_model_exclusion_reasons",
        "performance_model_exclusion_reasons", "adaptation_model_exclusion_reasons",
    ]
    target_output = selected[target_columns].sort_values(
        ["season_start_year", "canonical_player_name", "canonical_performance_id"]
    ).reset_index(drop=True)
    write_csv(target_output, TARGETS_PATH)

    compatibility = pd.read_csv(COMPATIBILITY_PATH, dtype=str, keep_default_na=False)
    compatibility["canonical_performance_id"] = compatibility["canonical_performance_id"].astype(str)
    strict_predictor_columns = [
        column for column in compatibility.columns
        if not column.startswith("roster_")
    ]
    strict_compatibility = compatibility[strict_predictor_columns]
    transfer_feature_columns = [
        "canonical_performance_id", "transfer_event_id", "canonical_transfer_event_id",
        "transfer_date", "transfer_window", "transfer_days_from_target_season_start",
        "canonical_transfer_type", "canonical_transfer_fee_status", "canonical_transfer_fee_eur",
        "transfer_fee_eur", "fee_status", "listed_market_value_at_transfer_eur", "age_at_transfer",
        "position", "sub_position", "foot", "height_in_cm", "country_of_citizenship", "cohort_tier",
        "selected_destination_event_policy", "same_destination_event_count",
        "later_same_destination_events_collapsed", "transfer_type_primary_eligible",
        "outfield_role_eligible", "primary_summer_window_eligible",
    ]
    transfer_features = target_output[transfer_feature_columns]
    feature_output = transfer_features.merge(
        strict_compatibility, how="left", on="canonical_performance_id", validate="one_to_one"
    )
    write_csv(feature_output, FEATURES_PATH)

    label_columns = [
        "canonical_performance_id", "target_club_match_opportunities",
        "target_available_player_minutes", "target_destination_minutes",
        "target_destination_minutes_share", "target_destination_minutes_share_available",
        "target_role_performance_index", "target_role_performance_index_available",
        "prior3_role_performance_baseline", "prior3_role_performance_baseline_minutes",
        "prior3_role_performance_baseline_seasons", "prior3_role_performance_baseline_available",
        "target_performance_change_rolling3", "target_performance_change_rolling3_available",
        "lag1_role_matches_target_role", "eligible_opportunity_model_primary",
        "eligible_performance_model_primary", "eligible_adaptation_model_primary",
        "opportunity_model_exclusion_reasons", "performance_model_exclusion_reasons",
        "adaptation_model_exclusion_reasons",
    ]
    model_output = feature_output.merge(
        target_output[label_columns], how="left", on="canonical_performance_id", validate="one_to_one"
    )
    write_csv(model_output, MODEL_MATRIX_PATH)

    coverage = target_output.groupby(
        ["season_start_year", "transfer_window", "role_group"], as_index=False, dropna=False
    ).agg(
        target_rows=("canonical_performance_id", "size"),
        minutes_share_available=("target_destination_minutes_share_available", "sum"),
        role_performance_available=("target_role_performance_index_available", "sum"),
        performance_change_available=("target_performance_change_rolling3_available", "sum"),
        opportunity_primary_eligible=("eligible_opportunity_model_primary", "sum"),
        performance_primary_eligible=("eligible_performance_model_primary", "sum"),
        adaptation_primary_eligible=("eligible_adaptation_model_primary", "sum"),
    )
    write_csv(coverage, OUTPUT / "target_coverage_by_season_window_role.csv")

    definitions = pd.DataFrame([
        {
            "target": "target_destination_minutes_share",
            "model": "opportunity",
            "definition": "Destination-club league minutes divided by destination club-season match opportunities × 90.",
            "unit": "share_0_to_1",
            "primary_minimum_minutes": "none",
            "primary_population": "summer transfer/loan; outfield; strict compatibility eligible",
            "interpretation": "How much of a full destination-club league season the player occupied.",
        },
        {
            "target": "target_role_performance_index",
            "model": "performance",
            "definition": "Current destination-season FBref role-performance index standardized within season and role.",
            "unit": "role_season_standardized_index",
            "primary_minimum_minutes": int(MIN_TARGET_MINUTES),
            "primary_population": "summer transfer/loan; outfield; strict compatibility eligible",
            "interpretation": "Absolute post-move performance conditional on receiving a meaningful sample of minutes.",
        },
        {
            "target": "target_performance_change_rolling3",
            "model": "adaptation",
            "definition": "Current role-performance index minus the minutes-weighted role-performance index across available t-1/t-2/t-3 seasons.",
            "unit": "standardized_index_change",
            "primary_minimum_minutes": f"target {int(MIN_TARGET_MINUTES)}; prior baseline {int(MIN_BASELINE_MINUTES)}",
            "primary_population": "summer transfer/loan; outfield; strict compatibility eligible",
            "interpretation": "Whether destination performance improved or declined relative to established prior form.",
        },
    ])
    write_csv(definitions, OUTPUT / "target_definition_dictionary.csv")

    distribution_rows = []
    for model_name, eligibility_column, target_column in [
        ("opportunity", "eligible_opportunity_model_primary", "target_destination_minutes_share"),
        ("performance", "eligible_performance_model_primary", "target_role_performance_index"),
        ("adaptation", "eligible_adaptation_model_primary", "target_performance_change_rolling3"),
    ]:
        values = pd.to_numeric(
            target_output.loc[target_output[eligibility_column], target_column], errors="coerce"
        ).dropna()
        distribution_rows.append({
            "model": model_name,
            "target": target_column,
            "primary_rows": int(len(values)),
            "mean": float(values.mean()),
            "standard_deviation": float(values.std()),
            "minimum": float(values.min()),
            "p10": float(values.quantile(0.10)),
            "p25": float(values.quantile(0.25)),
            "median": float(values.median()),
            "p75": float(values.quantile(0.75)),
            "p90": float(values.quantile(0.90)),
            "maximum": float(values.max()),
        })
    write_csv(pd.DataFrame(distribution_rows), OUTPUT / "target_distribution_summary.csv")

    exclusion_rows = []
    for model_name, eligibility_column, reason_column in [
        ("opportunity", "eligible_opportunity_model_primary", "opportunity_model_exclusion_reasons"),
        ("performance", "eligible_performance_model_primary", "performance_model_exclusion_reasons"),
        ("adaptation", "eligible_adaptation_model_primary", "adaptation_model_exclusion_reasons"),
    ]:
        exclusion_rows.append({
            "model": model_name,
            "reason": "eligible_primary",
            "rows": int(target_output[eligibility_column].sum()),
        })
        reasons = target_output.loc[~target_output[eligibility_column], reason_column].str.split(" | ", regex=False).explode()
        for reason, count in reasons.value_counts().items():
            exclusion_rows.append({
                "model": model_name,
                "reason": reason,
                "rows": int(count),
            })
    write_csv(pd.DataFrame(exclusion_rows), OUTPUT / "target_exclusion_summary.csv")

    summary_rows = [
        ("cohort", "source_transfer_events", len(transfers)),
        ("cohort", "matched_transfer_event_rows", len(candidate)),
        ("cohort", "unique_destination_player_seasons", len(target_output)),
        ("cohort", "later_same_destination_events_collapsed", int(target_output["later_same_destination_events_collapsed"].sum())),
        ("target", "destination_minutes_share_available", int(target_output["target_destination_minutes_share_available"].sum())),
        ("target", "role_performance_index_available", int(target_output["target_role_performance_index_available"].sum())),
        ("target", "performance_change_rolling3_available", int(target_output["target_performance_change_rolling3_available"].sum())),
        ("primary_model", "opportunity_eligible", int(target_output["eligible_opportunity_model_primary"].sum())),
        ("primary_model", "performance_eligible", int(target_output["eligible_performance_model_primary"].sum())),
        ("primary_model", "adaptation_eligible", int(target_output["eligible_adaptation_model_primary"].sum())),
        ("policy", "minimum_target_minutes", MIN_TARGET_MINUTES),
        ("policy", "minimum_prior3_baseline_minutes", MIN_BASELINE_MINUTES),
    ]
    summary = pd.DataFrame(summary_rows, columns=["area", "metric", "value"])
    write_csv(summary, OUTPUT / "target_summary.csv")
    summary_json = {
        "target_rows": int(len(target_output)),
        "opportunity_primary_eligible": int(target_output["eligible_opportunity_model_primary"].sum()),
        "performance_primary_eligible": int(target_output["eligible_performance_model_primary"].sum()),
        "adaptation_primary_eligible": int(target_output["eligible_adaptation_model_primary"].sum()),
    }
    (OUTPUT / "target_summary.json").write_text(
        json.dumps(summary_json, indent=2) + "\n", encoding="utf-8"
    )

    checks = [
        {
            "check": "target_rows_unique_by_destination_performance",
            "severity": "blocking",
            "passed": not target_output["canonical_performance_id"].duplicated().any(),
            "observed": int(target_output["canonical_performance_id"].duplicated().sum()),
            "expected": 0,
            "detail": "Earliest inbound event selection must produce one target row per destination player-season.",
        },
        {
            "check": "matched_event_collapse_reconciles",
            "severity": "blocking",
            "passed": len(candidate) - len(target_output) == int(target_output["later_same_destination_events_collapsed"].sum()),
            "observed": len(candidate) - len(target_output),
            "expected": int(target_output["later_same_destination_events_collapsed"].sum()),
            "detail": "All extra same-destination events are explicitly recorded as collapsed later events.",
        },
        {
            "check": "destination_minutes_share_bounds",
            "severity": "blocking",
            "passed": bool(target_output["target_destination_minutes_share"].dropna().between(0, 1.05).all()),
            "observed": float(target_output["target_destination_minutes_share"].max()),
            "expected": "0 to 1.05",
            "detail": "Small tolerance permits source minute conventions; independent verifier recomputes every row.",
        },
        {
            "check": "performance_change_reconciles",
            "severity": "blocking",
            "passed": bool(np.allclose(
                target_output["target_performance_change_rolling3"],
                target_output["target_role_performance_index"] - target_output["prior3_role_performance_baseline"],
                equal_nan=True,
            )),
            "observed": "recomputed",
            "expected": "target index minus prior3 baseline",
            "detail": "Adaptation target is an arithmetic difference, not a separate opaque score.",
        },
        {
            "check": "feature_only_has_no_target_columns",
            "severity": "blocking",
            "passed": not any(column.startswith("target_") for column in feature_output.columns),
            "observed": sum(column.startswith("target_") for column in feature_output.columns),
            "expected": 0,
            "detail": "Scoring features remain physically separate from current-season outcomes.",
        },
        {
            "check": "strict_tables_have_no_roster_columns",
            "severity": "blocking",
            "passed": not any(column.startswith("roster_") for column in model_output.columns),
            "observed": sum(column.startswith("roster_") for column in model_output.columns),
            "expected": 0,
            "detail": "Primary transfer matrices exclude scenario-only target-roster membership features.",
        },
        {
            "check": "strict_feature_and_model_rows_preserved",
            "severity": "blocking",
            "passed": len(feature_output) == len(model_output) == len(target_output),
            "observed": f"features={len(feature_output)}; model={len(model_output)}; targets={len(target_output)}",
            "expected": "equal row counts",
            "detail": "Targets and predictors remain one-to-one on canonical_performance_id.",
        },
    ]
    checks_frame = pd.DataFrame(checks)
    write_csv(checks_frame, OUTPUT / "target_build_checks.csv")

    table_frames = {
        "transfer_compatibility_targets": target_output,
        "transfer_compatibility_strict_features": feature_output,
        "transfer_compatibility_strict_model_matrix": model_output,
    }
    dictionary_rows = []
    for table, frame in table_frames.items():
        for order, column in enumerate(frame.columns, start=1):
            usage, timing, definition = field_usage(column, table)
            dictionary_rows.append({
                "table": table,
                "column_order": order,
                "column": column,
                "usage": usage,
                "timing_policy": timing,
                "definition": definition,
            })
    write_csv(pd.DataFrame(dictionary_rows), OUTPUT / "target_field_dictionary.csv")

    readme = f"""# MoveMaker transfer compatibility targets — first version

## Grain and cohort

- Target rows: {len(target_output):,}
- Grain: one accepted FBref destination player-club-competition-season with one selected inbound transfer event.
- When multiple inbound events map to the same destination stint, the earliest event is selected and later events are counted in `later_same_destination_events_collapsed`.
- Target season assignment uses June–December as the season beginning that calendar year and January–May as the preceding season start.

## Three targets

1. `target_destination_minutes_share`: destination minutes divided by the destination club-season maximum observed player appearances × 90.
2. `target_role_performance_index`: current destination-season role-adjusted FBref performance index.
3. `target_performance_change_rolling3`: current role index minus the minutes-weighted prior t-1/t-2/t-3 role index.

## Primary cohort policy

Primary eligibility requires a June–October arrival, canonical transfer type `transfer` or `loan`, an outfield role, and strict compatibility features. Performance and adaptation additionally require at least {int(MIN_TARGET_MINUTES)} destination minutes. Adaptation also requires at least {int(MIN_BASELINE_MINUTES)} prior-baseline minutes.

Winter and off-window moves remain in the target table for sensitivity analyses, but full-season minutes share is not directly comparable for those partial-season arrivals.

## Leakage policy

- `transfer_compatibility_strict_features.csv` contains no `target_` columns and no `roster_` columns.
- `transfer_compatibility_strict_model_matrix.csv` adds explicitly prefixed targets and eligibility controls.
- Never use `target_`, `eligible_`, or exclusion-reason fields as predictors.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")

    source_rows = []
    for path in [TRANSFER_COHORT_PATH, CANONICAL_TRANSFER_PATH, COMPATIBILITY_PATH, ADVANCED_MATRIX_PATH]:
        source_rows.append({
            "source_file": path.relative_to(ROOT).as_posix(),
            "file_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    write_csv(pd.DataFrame(source_rows), OUTPUT / "source_manifest.csv")

    manifest_rows = []
    for path in sorted(OUTPUT.iterdir(), key=lambda item: item.name.casefold()):
        if path.is_file() and path.name != "output_manifest.csv":
            manifest_rows.append({
                "output_file": path.relative_to(ROOT).as_posix(),
                "file_bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    write_csv(pd.DataFrame(manifest_rows), OUTPUT / "output_manifest.csv")

    failures = int((~checks_frame["passed"]).sum())
    print(json.dumps({
        "status": "pass" if failures == 0 else "fail",
        **summary_json,
        "blocking_failures": failures,
    }, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

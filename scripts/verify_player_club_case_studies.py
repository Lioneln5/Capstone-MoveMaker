"""Independently verify historical player-club case-study artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Data" / "processed" / "player_club_case_studies"
PAIRS_PATH = ROOT / "Data" / "processed" / "subgroup_stability" / "subgroup_pair_predictions.csv"
MATRIX_PATH = ROOT / "Data" / "processed" / "compatibility_targets" / "transfer_compatibility_strict_model_matrix.csv"
TARGETS_PATH = ROOT / "Data" / "processed" / "compatibility_targets" / "transfer_compatibility_targets.csv"
MATURE_ORIGINS = ["origin_2020", "origin_2021", "origin_2022"]
STYLE_DIMENSIONS = [
    "aerial_physicality", "creation", "defensive_intensity", "directness",
    "dribble_aggression", "passing_control", "progression", "scoring_threat",
]
CONTEXT_DIMENSIONS = [
    "defensive_intensity", "forward_mobility", "forward_threat",
    "midfield_aggression", "midfield_creativity", "team_passing_control",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dual_target_cohort(pairs: pd.DataFrame) -> pd.DataFrame:
    pairs = pairs.loc[
        pairs["reference_type"].eq("same_family_baseline")
        & pairs["origin_id"].isin(MATURE_ORIGINS)
        & pairs["target_key"].isin(["opportunity", "performance"])
    ].copy()
    keys = [
        "origin_id", "canonical_performance_id", "canonical_player_name",
        "canonical_club_name", "role_group", "role_name",
        "canonical_competition_id", "competition_name", "evaluation_year",
    ]
    values = [
        "actual", "candidate_prediction", "baseline_prediction",
        "candidate_absolute_error", "baseline_absolute_error", "mae_improvement",
    ]
    wide = pairs.pivot(index=keys, columns="target_key", values=values)
    wide.columns = [f"{target}_{metric}" for metric, target in wide.columns]
    wide = wide.reset_index().dropna()
    for target in ["opportunity", "performance"]:
        wide[f"{target}_compatibility_lift"] = wide[f"{target}_candidate_prediction"] - wide[f"{target}_baseline_prediction"]
        wide[f"{target}_realized_vs_baseline"] = wide[f"{target}_actual"] - wide[f"{target}_baseline_prediction"]
        wide[f"{target}_candidate_better"] = wide[f"{target}_mae_improvement"].gt(0)
    wide["mean_two_target_mae_improvement"] = wide[["opportunity_mae_improvement", "performance_mae_improvement"]].mean(axis=1)
    positive = wide["opportunity_compatibility_lift"].gt(0) & wide["performance_compatibility_lift"].gt(0)
    negative = wide["opportunity_compatibility_lift"].lt(0) & wide["performance_compatibility_lift"].lt(0)
    opposite = wide["opportunity_compatibility_lift"].mul(wide["performance_compatibility_lift"]).lt(0)
    both_better = wide["opportunity_candidate_better"] & wide["performance_candidate_better"]
    both_worse = ~wide["opportunity_candidate_better"] & ~wide["performance_candidate_better"]
    wide["case_category"] = np.select(
        [positive & both_better, negative & both_better, opposite & both_better, positive & both_worse, negative & both_worse],
        ["confirmed_positive", "confirmed_warning", "validated_tradeoff", "false_positive", "false_warning"],
        default="mixed_unselected",
    )
    return wide


def independently_select(cohort: pd.DataFrame) -> pd.DataFrame:
    selected = []
    selected.append(
        cohort.loc[cohort["case_category"].eq("confirmed_positive")]
        .sort_values(["mean_two_target_mae_improvement", "canonical_player_name"], ascending=[False, True])
        .head(2)
    )
    for category, ascending, limit in [
        ("confirmed_warning", False, None),
        ("false_positive", True, 2),
        ("false_warning", True, None),
    ]:
        group = (
            cohort.loc[cohort["case_category"].eq(category)]
            .sort_values(["role_name", "mean_two_target_mae_improvement", "canonical_player_name"], ascending=[True, ascending, True])
            .groupby("role_name", sort=True, as_index=False)
            .head(1)
        )
        if limit is not None:
            group = group.sort_values(["mean_two_target_mae_improvement", "canonical_player_name"]).head(limit)
        selected.append(group)
    tradeoffs = cohort.loc[cohort["case_category"].eq("validated_tradeoff")].copy()
    tradeoffs["pattern"] = np.where(
        tradeoffs["opportunity_compatibility_lift"].lt(0),
        "lower opportunity / higher performance",
        "higher opportunity / lower performance",
    )
    tradeoffs = (
        tradeoffs.sort_values(["pattern", "mean_two_target_mae_improvement", "canonical_player_name"], ascending=[True, False, True])
        .groupby("pattern", sort=True, as_index=False)
        .head(1)
    )
    selected.append(tradeoffs)
    return pd.concat(selected, ignore_index=True)


def compare_numeric(left: pd.Series, right: pd.Series, tolerance: float = 1e-12) -> bool:
    return bool(np.allclose(pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce"), atol=tolerance, rtol=0, equal_nan=True))


def run() -> None:
    required = [
        "case_study_portfolio.csv", "case_target_predictions.csv",
        "case_player_style_profiles.csv", "case_context_preference_profiles.csv",
        "case_fit_summary.csv", "case_feature_coverage.csv",
        "case_statsbomb_supplement.csv", "case_category_summary.csv",
        "case_study_build_checks.csv", "case_study_run_summary.json", "source_manifest.csv",
    ]
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any, expected: Any, detail: str) -> None:
        checks.append({"check": name, "severity": "blocking", "passed": bool(passed), "observed": observed, "expected": expected, "detail": detail})

    missing = [name for name in required if not (OUTPUT / name).exists()]
    add("required_outputs_exist", not missing, ";".join(missing) or "all present", "all present", "Every declared case-study artifact exists.")
    if missing:
        raise RuntimeError(f"Missing case-study outputs: {missing}")

    build_checks = pd.read_csv(OUTPUT / "case_study_build_checks.csv")
    build_pass = build_checks["passed"].astype(str).str.lower().eq("true")
    add("build_checks_all_pass", bool(build_pass.all()), int((~build_pass).sum()), 0, "All construction-time invariants passed.")

    source_manifest = pd.read_csv(OUTPUT / "source_manifest.csv")
    bad_hashes = []
    for row in source_manifest.itertuples(index=False):
        source = ROOT / str(row.source_file)
        if not source.exists() or sha256(source) != str(row.sha256):
            bad_hashes.append(str(row.source_file))
    add("source_manifest_hashes_match", not bad_hashes, ";".join(bad_hashes) or "all match", "all match", "Verification uses the exact recorded source artifacts.")

    pairs = pd.read_csv(PAIRS_PATH, low_memory=False)
    matrix = pd.read_csv(MATRIX_PATH, low_memory=False)
    targets = pd.read_csv(TARGETS_PATH, low_memory=False)
    portfolio = pd.read_csv(OUTPUT / "case_study_portfolio.csv", low_memory=False)
    predictions = pd.read_csv(OUTPUT / "case_target_predictions.csv", low_memory=False)
    styles = pd.read_csv(OUTPUT / "case_player_style_profiles.csv", low_memory=False)
    contexts = pd.read_csv(OUTPUT / "case_context_preference_profiles.csv", low_memory=False)
    fit = pd.read_csv(OUTPUT / "case_fit_summary.csv", low_memory=False)
    coverage = pd.read_csv(OUTPUT / "case_feature_coverage.csv", low_memory=False)

    cohort = dual_target_cohort(pairs)
    add("dual_target_cohort_recomputed", len(cohort) == 124, len(cohort), 124, "The mature dual-target evaluation cohort independently reproduces.")

    expected_selection = independently_select(cohort)
    selected_ids = set(expected_selection["canonical_performance_id"])
    reported_ids = set(portfolio["canonical_performance_id"])
    add("deterministic_selection_recomputed", selected_ids == reported_ids, sorted(reported_ids), sorted(selected_ids), "The twelve reported cases exactly match the predeclared category and role quotas.")

    category_counts = portfolio["case_category"].value_counts().to_dict()
    expected_counts = {"confirmed_warning": 3, "false_warning": 3, "confirmed_positive": 2, "validated_tradeoff": 2, "false_positive": 2}
    add("case_category_counts_exact", category_counts == expected_counts, category_counts, expected_counts, "Success, warning, trade-off, and failure representation is exact.")

    pair_lookup = pairs.loc[
        pairs["reference_type"].eq("same_family_baseline")
        & pairs["origin_id"].isin(MATURE_ORIGINS)
        & pairs["canonical_performance_id"].isin(reported_ids)
    ].copy()
    merged_predictions = predictions.merge(
        pair_lookup,
        left_on=["canonical_performance_id", "target_key"],
        right_on=["canonical_performance_id", "target_key"],
        suffixes=("_reported", "_source"),
        validate="one_to_one",
    )
    prediction_checks = [
        compare_numeric(merged_predictions["realized_outcome"], merged_predictions["actual"]),
        compare_numeric(merged_predictions["compatibility_prediction"], merged_predictions["candidate_prediction"]),
        compare_numeric(merged_predictions["baseline_prediction_reported"], merged_predictions["baseline_prediction_source"]),
        compare_numeric(merged_predictions["mae_improvement_reported"], merged_predictions["mae_improvement_source"]),
    ]
    add("target_predictions_map_to_verified_pairs", len(merged_predictions) == 24 and all(prediction_checks), len(merged_predictions), "24 exact mappings", "Every case prediction maps exactly to the verified rolling-origin paired output.")

    target_lookup = targets.drop_duplicates("canonical_performance_id").set_index("canonical_performance_id")
    detail_failures = []
    for row in portfolio.itertuples(index=False):
        source = target_lookup.loc[row.canonical_performance_id]
        for output_column, source_column in [
            ("transfer_event_id", "transfer_event_id"),
            ("from_club_name", "from_club_name"),
            ("canonical_club_name", "canonical_club_name"),
            ("transfer_date", "transfer_date"),
        ]:
            if str(getattr(row, output_column)) != str(source[source_column]):
                detail_failures.append(f"{row.case_id}:{output_column}")
    add("transfer_details_map_to_targets", not detail_failures, ";".join(detail_failures) or "all match", "all match", "Names, clubs, dates, and transfer ids map to the canonical target table.")

    matrix_lookup = matrix.set_index("canonical_performance_id")
    style_failures = []
    for row in styles.itertuples(index=False):
        source = matrix_lookup.loc[row.canonical_performance_id]
        player = source[f"advanced_rolling3_style_{row.style_dimension}"]
        team = source[f"strict_all_style_{row.style_dimension}"]
        if not np.allclose([row.player_rolling3_style_z, row.destination_prior_team_style_z, row.player_minus_team_style_gap], [player, team, player - team], atol=1e-12, rtol=0, equal_nan=True):
            style_failures.append(f"{row.case_id}:{row.style_dimension}")
    add("style_profiles_recomputed", len(styles) == 96 and not style_failures, ";".join(style_failures) or "96 exact rows", "96 exact rows", "Every player/team style value and gap maps to the strict feature matrix.")

    context_failures = []
    for row in contexts.itertuples(index=False):
        source = matrix_lookup.loc[row.canonical_performance_id]
        expected = [
            source[f"strict_context_{row.context_dimension}"],
            source[f"preference_{row.context_dimension}_exposure_mean"],
            source[f"preference_{row.context_dimension}_performance_weighted_level"],
            source[f"preference_{row.context_dimension}_shrunken_slope"],
            source[f"strict_{row.context_dimension}_familiarity_distance"],
            source[f"strict_{row.context_dimension}_preference_distance"],
            source[f"strict_{row.context_dimension}_preference_slope_interaction"],
        ]
        observed = [
            row.destination_prior_context_z, row.historical_exposure_mean_z,
            row.historical_performance_weighted_preferred_z,
            row.historical_shrunken_performance_slope, row.familiarity_distance,
            row.preference_distance, row.destination_context_x_preference_slope,
        ]
        if not np.allclose(observed, expected, atol=1e-12, rtol=0, equal_nan=True):
            context_failures.append(f"{row.case_id}:{row.context_dimension}")
    add("context_preference_profiles_recomputed", len(contexts) == 72 and not context_failures, ";".join(context_failures) or "72 exact rows", "72 exact rows", "Every historical context, preferred level, slope, and distance maps to the strict feature matrix.")

    fit_failures = []
    for row in fit.itertuples(index=False):
        source = matrix_lookup.loc[row.canonical_performance_id]
        familiarity_distances = pd.to_numeric(pd.Series([source[f"strict_{dimension}_familiarity_distance"] for dimension in CONTEXT_DIMENSIONS]), errors="coerce")
        preference_distances = pd.to_numeric(pd.Series([source[f"strict_{dimension}_preference_distance"] for dimension in CONTEXT_DIMENSIONS]), errors="coerce")
        expected_familiarity = 100 * np.exp(-familiarity_distances.mean()) if familiarity_distances.notna().any() else np.nan
        expected_preference = 100 * np.exp(-preference_distances.mean()) if preference_distances.notna().any() else np.nan
        if not np.allclose([row.context_familiarity_score_0_100, row.preference_fit_score_0_100], [expected_familiarity, expected_preference], atol=1e-10, rtol=0, equal_nan=True):
            fit_failures.append(row.case_id)
    add("aggregate_fit_scores_recomputed", not fit_failures, ";".join(fit_failures) or "all match", "all match", "Familiarity and preference-fit scores independently reproduce from six component distances.")

    timing_ok = coverage["timing_safe"].astype(str).str.lower().eq("true").all()
    add("feature_timing_safe", timing_ok, int((~coverage["timing_safe"].astype(str).str.lower().eq("true")).sum()), 0, "Every case uses player, preference, and club-context seasons strictly before the outcome season.")

    verification = pd.DataFrame(checks)
    verification.to_csv(OUTPUT / "independent_verification.csv", index=False)
    failures = int((~verification["passed"]).sum())
    summary = {
        "status": "pass" if failures == 0 else "fail",
        "checks": len(verification),
        "blocking_checks": len(verification),
        "blocking_failures": failures,
        "verified_mature_cases": len(cohort),
        "verified_selected_cases": len(portfolio),
        "verified_target_predictions": len(predictions),
        "verified_style_rows": len(styles),
        "verified_context_rows": len(contexts),
    }
    (OUTPUT / "independent_verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_paths = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name != "output_manifest.csv")
    pd.DataFrame([{"output_file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size} for path in output_paths]).to_csv(OUTPUT / "output_manifest.csv", index=False)
    print(json.dumps(summary, indent=2))
    if failures:
        raise RuntimeError(f"Independent case-study verification failed {failures} checks.")


if __name__ == "__main__":
    run()

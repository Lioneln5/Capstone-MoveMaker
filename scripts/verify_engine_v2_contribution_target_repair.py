"""Independently verify Engine V2 contribution-target repair Run 1."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.engine_v2.complete_contribution_target_contract import (  # noqa: E402
    AVAILABLE_FIELD,
    REASON_FIELD,
    TARGET_FIELD,
    add_complete_contribution_target,
    year_fields,
)


MASTER = ROOT / "Data" / "processed" / "contract_extension_integration" / "extension_modeling_master.csv"
GAMES = ROOT / "Data" / "processed" / "transfermarkt_clean" / "tables" / "games_clean.csv"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"
OUTPUT = ROOT / "Data" / "processed" / "engine_v2_contribution_target_repair"
REGENERABLE_V1_REPORTS = {
    "Data/processed/deployment_models/live_request_integrity.csv",
    "Data/processed/deployment_models/live_request_integrity.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().str.casefold().isin(
        {"true", "1", "yes"}
    )


def independent_schedule_fields(
    evaluation: pd.DataFrame, games: pd.DataFrame, year: int
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Reconstruct raw counts, peer medians, peer counts, and 80% gate.

    This deliberately does not call the production target's coverage helper.
    """

    base = games[["game_id", "competition_id", "date", "home_club_id", "away_club_id"]].copy()
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    home = base[["game_id", "competition_id", "date", "home_club_id"]].rename(
        columns={"home_club_id": "club_id"}
    )
    away = base[["game_id", "competition_id", "date", "away_club_id"]].rename(
        columns={"away_club_id": "club_id"}
    )
    long = pd.concat([home, away], ignore_index=True)
    long["club_id"] = pd.to_numeric(long["club_id"], errors="coerce").astype("Int64")
    long = long.dropna(subset=["game_id", "competition_id", "date", "club_id"]).drop_duplicates(
        ["game_id", "club_id"]
    )
    groups = {str(key): group for key, group in long.groupby("competition_id", sort=False)}

    raw_counts: list[float] = []
    medians: list[float] = []
    peer_counts: list[int] = []
    for row in evaluation.itertuples(index=False):
        group = groups.get(str(getattr(row, "competition_id")))
        club = pd.to_numeric(pd.Series([getattr(row, "canonical_club_id_x")]), errors="coerce").iat[0]
        start = pd.to_datetime(
            getattr(row, f"post_y{year}_window_start_exclusive_or_inclusive"), errors="coerce"
        )
        end = pd.to_datetime(
            getattr(row, f"post_y{year}_window_end_exclusive_or_inclusive"), errors="coerce"
        )
        if group is None or pd.isna(club) or pd.isna(start) or pd.isna(end):
            raw_counts.append(np.nan)
            medians.append(np.nan)
            peer_counts.append(0)
            continue
        window = group.loc[group["date"].gt(start) & group["date"].le(end)]
        counts = window.groupby("club_id", observed=True)["game_id"].nunique()
        raw_counts.append(float(counts.get(int(club), 0)))
        medians.append(float(counts.median()) if len(counts) else np.nan)
        peer_counts.append(int(len(counts)))

    raw = pd.Series(raw_counts, index=evaluation.index, dtype="float64")
    median = pd.Series(medians, index=evaluation.index, dtype="float64")
    peers = pd.Series(peer_counts, index=evaluation.index, dtype="Int64")
    ratio = raw.div(median.where(median.gt(0)))
    eligible = median.notna() & peers.ge(10) & ratio.ge(0.80)
    return raw, median, peers, eligible


def synthetic_frame() -> pd.DataFrame:
    rows = 6
    frame = pd.DataFrame(
        {
            "post_y1_window_fully_observable": [True] * rows,
            "post_y2_window_fully_observable": [True] * rows,
            "post_y1_same_club_all_competition_games": [40, 40, 40, 40, 40, 10],
            "post_y2_same_club_all_competition_games": [40, 40, 40, 40, 40, 10],
            "post_y1_same_club_all_competition_minutes": [1080, 720, 1080, 1080, None, 1000],
            "post_y2_same_club_all_competition_minutes": [1080, 1080, 1080, 1080, 1080, 1000],
            "post_y1_window_evidence_eligible_10_club_games": [True] * rows,
            "post_y2_window_evidence_eligible_10_club_games": [True] * rows,
        }
    )
    for year in (1, 2):
        fields = year_fields(year)
        frame[fields.original_league_games] = [38] * rows
        frame[fields.peer_median_games] = [38.0] * rows
        frame[fields.peer_clubs] = [20] * rows
        frame[fields.coverage_ratio] = [1.0] * rows
        frame[fields.coverage_eligible] = [True] * rows
    frame.loc[2, year_fields(1).coverage_ratio] = 0.70
    frame.loc[2, year_fields(1).coverage_eligible] = False
    frame.loc[3, year_fields(2).coverage_ratio] = 0.70
    frame.loc[3, year_fields(2).coverage_eligible] = False
    return frame


def main() -> None:
    checks: list[dict[str, object]] = []

    def add(name: str, actual: object, expected: object, passed: bool, note: str) -> None:
        checks.append(
            {"check": name, "actual": actual, "expected": expected, "passed": bool(passed), "note": note}
        )

    required = {
        "README.md",
        "build_checks.csv",
        "cohort_comparison.csv",
        "coverage_threshold_sensitivity.csv",
        "league_coverage_summary.csv",
        "normalization_cases.csv",
        "output_manifest.csv",
        "run_summary.json",
        "schedule_exclusions.csv",
        "source_manifest.csv",
        "target_contract.json",
        "threshold_sensitivity.csv",
        "unavailability_reason_counts.csv",
        "v1_freeze_comparison.csv",
    }
    actual_files = {path.name for path in OUTPUT.iterdir() if path.is_file()}
    allowed_extra = {"independent_verification.csv", "independent_verification.json"}
    add("required_outputs_exact", sorted(actual_files), sorted(required), required.issubset(actual_files) and not (actual_files - required - allowed_extra), "Run 1 evidence package has only declared files.")

    output_manifest = pd.read_csv(OUTPUT / "output_manifest.csv")
    output_ok = True
    for row in output_manifest.itertuples(index=False):
        path = OUTPUT / row.output_file
        output_ok &= path.exists() and path.stat().st_size == int(row.bytes) and sha256(path) == row.sha256
    add("output_manifest_exact", int(output_ok), 1, output_ok, "Every generated evidence hash and byte count matches.")

    source_manifest = pd.read_csv(OUTPUT / "source_manifest.csv")
    source_ok = True
    for row in source_manifest.itertuples(index=False):
        path = ROOT / row.source_file
        source_ok &= path.exists() and path.stat().st_size == int(row.bytes) and sha256(path) == row.sha256
    add("source_manifest_exact", int(source_ok), 1, source_ok, "Master, raw games, and V1 freeze match the build inputs.")

    master = pd.read_csv(MASTER, low_memory=False)
    games = pd.read_csv(GAMES, low_memory=False)
    master["signed_date"] = pd.to_datetime(master["signed_date"], errors="coerce")
    master["expiration_date"] = pd.to_datetime(master["expiration_date"], errors="coerce")
    master["signed_year"] = master["signed_date"].dt.year.astype("Int64")
    development = master.loc[master["signed_year"].le(2023)].copy()
    evaluation = development.loc[development["signed_year"].between(2020, 2023)].copy()

    linked = evaluation["canonical_player_id"].notna()
    covers = evaluation["expiration_date"].ge(evaluation["signed_date"] + pd.Timedelta(days=730))
    pre_ok = as_bool(evaluation["pre365_window_evidence_eligible_10_club_games"])
    y1_ok = as_bool(evaluation["post_y1_window_evidence_eligible_10_club_games"])
    y2_ok = as_bool(evaluation["post_y2_window_evidence_eligible_10_club_games"])
    pre_share = pd.to_numeric(evaluation["pre365_same_club_all_competition_opportunity_share"], errors="coerce")
    y1_stored = pd.to_numeric(evaluation["post_y1_same_club_all_competition_opportunity_share"], errors="coerce")
    y2_stored = pd.to_numeric(evaluation["post_y2_same_club_all_competition_opportunity_share"], errors="coerce")
    legacy = linked & covers & pre_ok & y2_ok & pre_share.notna() & y2_stored.notna()
    symmetric = legacy & y1_ok & y1_stored.notna()
    add("legacy_reference_independent", int(legacy.sum()), 929, int(legacy.sum()) == 929, "Legacy cohort is independently reconstructed.")
    add("symmetric_reference_independent", int(symmetric.sum()), 922, int(symmetric.sum()) == 922, "Symmetric cohort is independently reconstructed.")

    independent: dict[int, tuple[pd.Series, pd.Series, pd.Series, pd.Series]] = {}
    coverage = symmetric.copy()
    stored_counts_exact = True
    for year in (1, 2):
        independent[year] = independent_schedule_fields(evaluation, games, year)
        raw, _, _, eligible = independent[year]
        stored = pd.to_numeric(
            evaluation[f"post_y{year}_same_club_same_big5_competition_games"], errors="coerce"
        )
        stored_counts_exact &= bool(np.isclose(raw, stored, atol=0.0, rtol=0.0).all())
        coverage &= eligible
    add("raw_original_league_counts_exact", int(stored_counts_exact), 1, stored_counts_exact, "Both annual raw game counts reproduce the master exactly.")
    add("complete_cohort_independent", int(coverage.sum()), 878, int(coverage.sum()) == 878, "Independent 80%-of-peer schedule gate leaves 878 rows.")
    add("schedule_exclusions_independent", int((symmetric & ~coverage).sum()), 44, int((symmetric & ~coverage).sum()) == 44, "Exactly 44 symmetric rows fail schedule continuity.")

    raw_y1 = pd.to_numeric(evaluation["post_y1_same_club_all_competition_minutes"], errors="coerce").div(
        90 * pd.to_numeric(evaluation["post_y1_same_club_all_competition_games"], errors="coerce")
    )
    raw_y2 = pd.to_numeric(evaluation["post_y2_same_club_all_competition_minutes"], errors="coerce").div(
        90 * pd.to_numeric(evaluation["post_y2_same_club_all_competition_games"], errors="coerce")
    )
    normalized_y1 = raw_y1.clip(0, 1)
    normalized_y2 = raw_y2.clip(0, 1)
    target = normalized_y1.ge(0.25) & normalized_y2.ge(0.25)
    add("complete_positive_count_independent", int(target.loc[coverage].sum()), 438, int(target.loc[coverage].sum()) == 438, "Independent bounded-share target has 438 positives.")
    annual_capped = int(raw_y1.loc[coverage].gt(1).sum() + raw_y2.loc[coverage].gt(1).sum())
    add("annual_share_caps_independent", annual_capped, 6, annual_capped == 6, "Six eligible annual values exceed one before normalization.")
    raw_target = raw_y1.ge(0.25) & raw_y2.ge(0.25)
    add("caps_do_not_relabel", int(raw_target.loc[coverage].eq(target.loc[coverage]).all()), 1, raw_target.loc[coverage].eq(target.loc[coverage]).all(), "Capping changes scale, not the policy label.")

    saved_exclusions = pd.read_csv(OUTPUT / "schedule_exclusions.csv")
    expected_ids = set(evaluation.loc[symmetric & ~coverage, "capology_extension_event_id"])
    add("saved_exclusion_ids_exact", len(saved_exclusions), 44, set(saved_exclusions["capology_extension_event_id"]) == expected_ids, "Saved cases are all and only independent schedule exclusions.")
    add("saved_exclusion_reasons_explicit", int(saved_exclusions[REASON_FIELD].str.contains("SCHEDULE_BELOW_80PCT_PEER").all()), 1, saved_exclusions[REASON_FIELD].str.contains("SCHEDULE_BELOW_80PCT_PEER").all(), "Every saved exclusion identifies the failed schedule gate.")

    thresholds = pd.read_csv(OUTPUT / "threshold_sensitivity.csv")
    expected_threshold_counts = {0.20: 466, 0.25: 438, 0.30: 409}
    threshold_ok = all(
        int(thresholds.loc[np.isclose(thresholds["threshold_each_year"], threshold), "positive_targets"].iloc[0]) == count
        for threshold, count in expected_threshold_counts.items()
    )
    add("threshold_sensitivity_exact", int(threshold_ok), 1, threshold_ok, "20%, 25%, and 30% positive counts are 466, 438, and 409.")

    coverage_table = pd.read_csv(OUTPUT / "coverage_threshold_sensitivity.csv")
    expected_coverage = {0.70: 883, 0.75: 882, 0.80: 878, 0.85: 875, 0.90: 872}
    coverage_ok = all(
        int(coverage_table.loc[np.isclose(coverage_table["minimum_peer_coverage"], threshold), "eligible_rows"].iloc[0]) == count
        for threshold, count in expected_coverage.items()
    )
    add("coverage_sensitivity_exact", int(coverage_ok), 1, coverage_ok, "Nearby coverage thresholds reproduce the declared sensitivity.")

    synthetic = add_complete_contribution_target(synthetic_frame())
    actual_synthetic = [None if pd.isna(value) else int(value) for value in synthetic[TARGET_FIELD]]
    expected_synthetic = [1, 0, None, None, None, 1]
    add("synthetic_complete_contract", actual_synthetic, expected_synthetic, actual_synthetic == expected_synthetic, "Positive, negative, both schedule failures, missing minutes, and >100% share behave correctly.")
    add("synthetic_cap_exact", float(synthetic.loc[5, year_fields(1).normalized_share]), 1.0, synthetic.loc[5, year_fields(1).normalized_share] == 1.0, "A synthetic >100% share is capped exactly at one.")
    add("synthetic_unavailable_has_reason", int(synthetic.loc[[2, 3, 4], REASON_FIELD].ne("AVAILABLE").all()), 1, synthetic.loc[[2, 3, 4], REASON_FIELD].ne("AVAILABLE").all(), "Synthetic refused rows retain explicit reasons.")

    malformed = synthetic_frame()
    malformed.loc[0, year_fields(1).coverage_eligible] = False
    mismatch_rejected = False
    try:
        add_complete_contribution_target(malformed)
    except ValueError:
        mismatch_rejected = True
    add("stale_coverage_flag_fails_closed", int(mismatch_rejected), 1, mismatch_rejected, "A stored coverage flag cannot disagree with primitives.")

    summary = json.loads((OUTPUT / "run_summary.json").read_text(encoding="utf-8"))
    add("final_holdout_not_opened", summary.get("final_holdout_evaluated"), False, summary.get("final_holdout_evaluated") is False, "Run 1 never evaluates 2024+ outcomes.")
    add("no_models_fitted", summary.get("models_fitted"), 0, summary.get("models_fitted") == 0, "Run 1 is target construction only.")
    add("no_deployment_change", summary.get("deployment_artifacts_changed"), 0, summary.get("deployment_artifacts_changed") == 0, "No deployed model artifact changes.")

    freeze_comparison = pd.read_csv(OUTPUT / "v1_freeze_comparison.csv")
    baseline_ok = True
    for artifact in freeze_comparison.itertuples(index=False):
        if artifact.artifact in REGENERABLE_V1_REPORTS:
            continue
        path = ROOT / artifact.artifact
        baseline_ok &= path.exists() and sha256(path) == artifact.run1_baseline_sha256
    add("v1_protected_files_unchanged_from_run1_baseline", int(baseline_ok), 1, baseline_ok, "Every protected V1 artifact retains its exact Run 1 hash; the two explicitly regenerable integrity reports are checked separately.")
    expected_preexisting = REGENERABLE_V1_REPORTS
    reported_preexisting = set(
        freeze_comparison.loc[
            ~as_bool(freeze_comparison["matches_frozen_record"]), "artifact"
        ]
    )
    add("preexisting_v1_freeze_divergences_explicit", sorted(reported_preexisting), sorted(expected_preexisting), reported_preexisting == expected_preexisting, "Two pre-existing generated integrity-report divergences are disclosed rather than attributed to Run 1.")

    audit_doc = (ROOT / "docs" / "ENGINE_V2_FUTURE_ROLE_TARGET_AUDIT.md").read_text(encoding="utf-8")
    critical_doc = (ROOT / "docs" / "ENGINE_V2_CRITICAL_ISSUES.md").read_text(encoding="utf-8")
    registry = (ROOT / "models" / "README.md").read_text(encoding="utf-8")
    doc_ok = (
        "Complete contribution-target repair implemented" in audit_doc
        and "TARGET-REPAIR RUN 1 COMPLETE" in critical_doc
        and "Complete contribution-target repair" in registry
    )
    add("durable_documentation_updated", int(doc_ok), 1, doc_ok, "Audit, critical ledger, and model registry record Run 1 and its remaining boundary.")

    result = pd.DataFrame(checks)
    result.to_csv(OUTPUT / "independent_verification.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    verification_summary = {
        "checks_passed": int(result["passed"].sum()),
        "checks_total": len(result),
        "all_checks_passed": bool(result["passed"].all()),
        "models_fitted": 0,
        "final_holdout_evaluated": False,
    }
    (OUTPUT / "independent_verification.json").write_text(
        json.dumps(verification_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not result["passed"].all():
        failed = result.loc[~result["passed"], ["check", "actual", "expected"]]
        raise RuntimeError("Independent verification failed:\n" + failed.to_string(index=False))
    print(f"All {len(result)}/{len(result)} independent verification checks pass.")


if __name__ == "__main__":
    main()

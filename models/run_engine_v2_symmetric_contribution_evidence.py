"""Implement and evidence the symmetric Year-1/Year-2 contribution gate.

This stage changes no frozen V1 source, model, prediction, or deployment
artifact.  It applies the Engine V2 target contract to development rows through
2023 and verifies that the seven previously exposed asymmetric evaluation rows
become unavailable rather than false binary outcomes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.engine_v2.contribution_target_contract import (  # noqa: E402
    AVAILABLE_FIELD,
    REASON_FIELD,
    TARGET_FIELD,
    add_symmetric_contribution_target,
    contract_payload,
)


MASTER = ROOT / "Data" / "processed" / "contract_extension_integration" / "extension_modeling_master.csv"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"
OUTPUT = ROOT / "Data" / "processed" / "engine_v2_symmetric_contribution_evidence"
DEVELOPMENT_CUTOFF_YEAR = 2023
EVALUATION_YEARS = (2020, 2023)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().str.casefold().isin(
        {"true", "1", "yes"}
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(MASTER, low_memory=False)
    frame["signed_date"] = pd.to_datetime(frame["signed_date"], errors="coerce")
    frame["expiration_date"] = pd.to_datetime(frame["expiration_date"], errors="coerce")
    frame["signed_year"] = frame["signed_date"].dt.year.astype("Int64")
    development = frame.loc[frame["signed_year"].le(DEVELOPMENT_CUTOFF_YEAR)].copy()
    repaired = add_symmetric_contribution_target(development)

    evaluation = repaired.loc[repaired["signed_year"].between(*EVALUATION_YEARS)].copy()
    linked = evaluation["canonical_player_id"].notna()
    contract_covers = evaluation["expiration_date"].ge(
        evaluation["signed_date"] + pd.Timedelta(days=730)
    )
    pre_ok = as_bool(evaluation["pre365_window_evidence_eligible_10_club_games"])
    y2_ok = as_bool(evaluation["post_y2_window_evidence_eligible_10_club_games"])
    pre = pd.to_numeric(
        evaluation["pre365_same_club_all_competition_opportunity_share"], errors="coerce"
    )
    y1 = pd.to_numeric(
        evaluation["post_y1_same_club_all_competition_opportunity_share"], errors="coerce"
    )
    y2 = pd.to_numeric(
        evaluation["post_y2_same_club_all_competition_opportunity_share"], errors="coerce"
    )
    legacy_cohort = linked & contract_covers & pre_ok & y2_ok & pre.notna() & y2.notna()
    symmetric_cohort = legacy_cohort & evaluation[AVAILABLE_FIELD]
    legacy_target = (y1.ge(0.25) & y2.ge(0.25)).astype(int)

    excluded = evaluation.loc[
        legacy_cohort & ~evaluation[AVAILABLE_FIELD],
        [
            "capology_extension_event_id", "signed_year", "player_name", "club_name", "league",
            "post_y1_window_fully_observable", "post_y1_same_club_all_competition_games",
            "post_y1_window_evidence_eligible_10_club_games",
            "post_y1_same_club_all_competition_opportunity_share",
            "post_y2_window_fully_observable", "post_y2_same_club_all_competition_games",
            "post_y2_window_evidence_eligible_10_club_games",
            "post_y2_same_club_all_competition_opportunity_share", REASON_FIELD,
        ],
    ].copy()
    excluded["legacy_binary_target"] = legacy_target.loc[excluded.index]

    scope_summary = pd.DataFrame(
        [
            {
                "scope": "all_development_rows_through_2023",
                "rows_before_gate": len(repaired),
                "rows_with_symmetric_outcome_evidence": int(repaired[AVAILABLE_FIELD].sum()),
                "rows_made_unavailable": int((~repaired[AVAILABLE_FIELD]).sum()),
                "positive_targets": int(pd.to_numeric(repaired[TARGET_FIELD], errors="coerce").sum()),
            },
            {
                "scope": "rolling_evaluation_reference_2020_2023",
                "rows_before_gate": int(legacy_cohort.sum()),
                "rows_with_symmetric_outcome_evidence": int(symmetric_cohort.sum()),
                "rows_made_unavailable": int((legacy_cohort & ~evaluation[AVAILABLE_FIELD]).sum()),
                "positive_targets": int(pd.to_numeric(evaluation.loc[symmetric_cohort, TARGET_FIELD], errors="coerce").sum()),
            },
        ]
    )
    reasons = (
        repaired.loc[~repaired[AVAILABLE_FIELD], REASON_FIELD]
        .value_counts(dropna=False)
        .rename_axis("unavailability_reason")
        .reset_index(name="rows")
    )

    checks: list[dict[str, object]] = []

    def add(name: str, actual: object, expected: object, passed: bool, note: str) -> None:
        checks.append(
            {"check": name, "actual": actual, "expected": expected, "passed": bool(passed), "note": note}
        )

    target = pd.to_numeric(repaired[TARGET_FIELD], errors="coerce")
    add("development_cutoff_enforced", int(repaired["signed_year"].max()), 2023, repaired["signed_year"].max() == 2023, "No 2024+ outcome enters this repair stage.")
    add("target_available_iff_symmetric_evidence", int(target.notna().eq(repaired[AVAILABLE_FIELD]).all()), 1, target.notna().eq(repaired[AVAILABLE_FIELD]).all(), "A label exists if and only if both annual gates and shares are valid.")
    add("unavailable_never_negative", int(target.loc[~repaired[AVAILABLE_FIELD]].notna().sum()), 0, target.loc[~repaired[AVAILABLE_FIELD]].isna().all(), "Ineligible evidence remains NA rather than becoming class zero.")
    add("legacy_evaluation_reference", int(legacy_cohort.sum()), 929, int(legacy_cohort.sum()) == 929, "Frozen asymmetric rolling-evaluation cohort is reproduced only as a comparison.")
    add("symmetric_evaluation_cohort", int(symmetric_cohort.sum()), 922, int(symmetric_cohort.sum()) == 922, "Both annual evidence requirements leave 922 evaluation rows.")
    add("asymmetric_rows_removed", len(excluded), 7, len(excluded) == 7, "All seven known Year-1 evidence defects become unavailable.")
    add("symmetric_positive_count", int(evaluation.loc[symmetric_cohort, TARGET_FIELD].sum()), 453, int(evaluation.loc[symmetric_cohort, TARGET_FIELD].sum()) == 453, "The repaired evaluation cohort has 453 positive labels.")
    unchanged = evaluation.loc[symmetric_cohort, TARGET_FIELD].astype(int).eq(legacy_target.loc[symmetric_cohort]).all()
    add("eligible_labels_unchanged", int(unchanged), 1, unchanged, "The gate removes invalid labels but does not relabel eligible rows.")
    add("reason_present_for_every_unavailable_row", int(repaired.loc[~repaired[AVAILABLE_FIELD], REASON_FIELD].ne("").all()), 1, repaired.loc[~repaired[AVAILABLE_FIELD], REASON_FIELD].ne("").all(), "Every unavailable target has an explicit reason.")

    checks_frame = pd.DataFrame(checks)
    write_csv(scope_summary, OUTPUT / "symmetric_evidence_summary.csv")
    write_csv(excluded, OUTPUT / "excluded_asymmetric_evaluation_cases.csv")
    write_csv(reasons, OUTPUT / "unavailability_reasons.csv")
    write_csv(checks_frame, OUTPUT / "build_checks.csv")
    (OUTPUT / "target_contract.json").write_text(
        json.dumps(contract_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    source_manifest = pd.DataFrame(
        [
            {
                "source_file": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in [MASTER, FREEZE]
        ]
    )
    write_csv(source_manifest, OUTPUT / "source_manifest.csv")

    summary = {
        "stage": "symmetric_year1_year2_contribution_evidence_repair",
        "development_cutoff_year": DEVELOPMENT_CUTOFF_YEAR,
        "final_holdout_evaluated": False,
        "models_fitted": 0,
        "deployment_artifacts_changed": 0,
        "legacy_evaluation_rows": int(legacy_cohort.sum()),
        "symmetric_evaluation_rows": int(symmetric_cohort.sum()),
        "evaluation_rows_made_unavailable": len(excluded),
        "symmetric_evaluation_positive_targets": int(evaluation.loc[symmetric_cohort, TARGET_FIELD].sum()),
        "remaining_release_status": "blocked_pending_schedule_completeness_normalization_development_rerun_and_later_final_evaluation",
        "checks_passed": int(checks_frame["passed"].sum()),
        "checks_total": len(checks_frame),
        "all_checks_passed": bool(checks_frame["passed"].all()),
    }
    (OUTPUT / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    readme = f"""# Engine V2 symmetric contribution-evidence repair

This stage implements the first required repair from the future-role target
audit. A contribution label is now available only when **both Year 1 and Year
2** independently have a fully observable window, at least 10 captured
extension-club matches, and a nonmissing same-club opportunity share.

## Result

- Frozen reference evaluation rows: {int(legacy_cohort.sum()):,}
- Symmetric-evidence evaluation rows: {int(symmetric_cohort.sum()):,}
- Rows changed from a binary label to unavailable: {len(excluded)}
- Eligible rows whose binary label changed: 0
- Positive labels in the symmetric evaluation cohort: {int(evaluation.loc[symmetric_cohort, TARGET_FIELD].sum()):,}

The seven excluded rows are not recoded as failures. Their target is nullable
and carries an explicit unavailability reason.

## Boundary

This is **target-repair step 1 only**. It does not repair incomplete schedules
after relegation, resolve the six shares above 100%, fit a candidate model,
open the 2024+ final cohort, or authorize deployment. The target is named
**sustained realized extension-club contribution**, not future role when
available.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")

    output_files = sorted(
        path
        for path in OUTPUT.iterdir()
        if path.is_file() and path.name not in {"output_manifest.csv", "independent_verification.csv"}
    )
    write_csv(
        pd.DataFrame(
            {
                "output_file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in output_files
        ),
        OUTPUT / "output_manifest.csv",
    )

    if not checks_frame["passed"].all():
        failed = checks_frame.loc[~checks_frame["passed"], "check"].tolist()
        raise RuntimeError(f"Symmetric contribution evidence build failed: {failed}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

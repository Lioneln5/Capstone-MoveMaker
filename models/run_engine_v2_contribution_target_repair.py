"""Build evidence for the complete Engine V2 contribution-target repair.

This run repairs the target contract only. It uses development extensions signed
through 2023, fits no model, does not inspect 2024+ outcomes, and changes no
deployment artifact.
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

from models.engine_v2.complete_contribution_target_contract import (  # noqa: E402
    AVAILABLE_FIELD,
    MINIMUM_ORIGINAL_LEAGUE_PEER_COVERAGE,
    POLICY_THRESHOLD,
    REASON_FIELD,
    TARGET_FIELD,
    add_complete_contribution_target,
    add_original_league_schedule_coverage,
    contract_payload,
    year_fields,
)


MASTER = ROOT / "Data" / "processed" / "contract_extension_integration" / "extension_modeling_master.csv"
GAMES = ROOT / "Data" / "processed" / "transfermarkt_clean" / "tables" / "games_clean.csv"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"
OUTPUT = ROOT / "Data" / "processed" / "engine_v2_contribution_target_repair"
DEVELOPMENT_CUTOFF_YEAR = 2023
EVALUATION_YEARS = (2020, 2023)
REFERENCE_COUNTS = {
    "legacy_evaluation_rows": 929,
    "symmetric_evaluation_rows": 922,
    "complete_evaluation_rows": 878,
    "complete_positive_targets": 438,
    "schedule_exclusions_after_symmetric_gate": 44,
    "annual_shares_capped": 6,
}


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


def reference_masks(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    linked = frame["canonical_player_id"].notna()
    contract_covers = frame["expiration_date"].ge(
        frame["signed_date"] + pd.Timedelta(days=730)
    )
    pre_ok = as_bool(frame["pre365_window_evidence_eligible_10_club_games"])
    y1_ok = as_bool(frame["post_y1_window_evidence_eligible_10_club_games"])
    y2_ok = as_bool(frame["post_y2_window_evidence_eligible_10_club_games"])
    pre = pd.to_numeric(
        frame["pre365_same_club_all_competition_opportunity_share"], errors="coerce"
    )
    y1 = pd.to_numeric(
        frame["post_y1_same_club_all_competition_opportunity_share"], errors="coerce"
    )
    y2 = pd.to_numeric(
        frame["post_y2_same_club_all_competition_opportunity_share"], errors="coerce"
    )
    legacy = linked & contract_covers & pre_ok & y2_ok & pre.notna() & y2.notna()
    symmetric = legacy & y1_ok & y1.notna()
    return legacy, symmetric


def cohort_row(name: str, mask: pd.Series, target: pd.Series) -> dict[str, object]:
    usable = target.loc[mask].dropna()
    positives = int(usable.sum())
    rows = int(mask.sum())
    return {
        "cohort": name,
        "rows": rows,
        "positive_targets": positives,
        "negative_targets": rows - positives,
        "positive_rate": positives / rows if rows else pd.NA,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(MASTER, low_memory=False)
    games = pd.read_csv(GAMES, low_memory=False)
    master["signed_date"] = pd.to_datetime(master["signed_date"], errors="coerce")
    master["expiration_date"] = pd.to_datetime(master["expiration_date"], errors="coerce")
    master["signed_year"] = master["signed_date"].dt.year.astype("Int64")

    # Restrict before computing any target evidence. No 2024+ outcome enters Run 1.
    development = master.loc[master["signed_year"].le(DEVELOPMENT_CUTOFF_YEAR)].copy()
    covered = add_original_league_schedule_coverage(development, games)
    repaired = add_complete_contribution_target(covered)
    evaluation = repaired.loc[repaired["signed_year"].between(*EVALUATION_YEARS)].copy()

    legacy, symmetric = reference_masks(evaluation)
    complete = symmetric & evaluation[AVAILABLE_FIELD]
    y1_old = pd.to_numeric(
        evaluation["post_y1_same_club_all_competition_opportunity_share"], errors="coerce"
    )
    y2_old = pd.to_numeric(
        evaluation["post_y2_same_club_all_competition_opportunity_share"], errors="coerce"
    )
    old_target = (y1_old.ge(POLICY_THRESHOLD) & y2_old.ge(POLICY_THRESHOLD)).astype("Int64")
    new_target = pd.to_numeric(evaluation[TARGET_FIELD], errors="coerce")

    cohorts = pd.DataFrame(
        [
            cohort_row("legacy_asymmetric_reference", legacy, old_target),
            cohort_row("symmetric_year1_year2_evidence", symmetric, old_target),
            cohort_row("complete_schedule_checked_normalized", complete, new_target),
        ]
    )

    identity = [
        "capology_extension_event_id",
        "canonical_player_id",
        "canonical_club_id_x",
        "signed_year",
        "player_name",
        "club_name",
        "league",
        "competition_id",
    ]
    schedule_columns: list[str] = []
    normalization_columns: list[str] = []
    for year in (1, 2):
        fields = year_fields(year)
        schedule_columns.extend(
            [
                fields.original_league_games,
                fields.peer_median_games,
                fields.peer_clubs,
                fields.coverage_ratio,
                fields.coverage_eligible,
            ]
        )
        normalization_columns.extend(
            [fields.club_games, fields.club_minutes, fields.raw_share, fields.normalized_share, fields.share_capped]
        )

    exclusions = evaluation.loc[
        symmetric & ~evaluation[AVAILABLE_FIELD],
        identity + schedule_columns + [REASON_FIELD],
    ].copy()
    exclusions["prior_symmetric_target"] = old_target.loc[exclusions.index]
    exclusions = exclusions.sort_values(["league", "signed_year", "club_name", "player_name"])

    capped_rows: list[pd.DataFrame] = []
    for year in (1, 2):
        fields = year_fields(year)
        mask = complete & evaluation[fields.share_capped]
        cases = evaluation.loc[mask, identity + normalization_columns].copy()
        cases.insert(len(identity), "annual_window", f"Year {year}")
        cases["raw_threshold_label_for_this_year"] = evaluation.loc[mask, fields.raw_share].ge(
            POLICY_THRESHOLD
        ).astype(int)
        cases["normalized_threshold_label_for_this_year"] = evaluation.loc[
            mask, fields.normalized_share
        ].ge(POLICY_THRESHOLD).astype(int)
        capped_rows.append(cases)
    normalization_cases = pd.concat(capped_rows, ignore_index=True)

    threshold_rows: list[dict[str, object]] = []
    y1_normalized = pd.to_numeric(evaluation[year_fields(1).normalized_share], errors="coerce")
    y2_normalized = pd.to_numeric(evaluation[year_fields(2).normalized_share], errors="coerce")
    policy_labels = y1_normalized.ge(POLICY_THRESHOLD) & y2_normalized.ge(POLICY_THRESHOLD)
    for threshold in (0.20, 0.25, 0.30):
        labels = y1_normalized.ge(threshold) & y2_normalized.ge(threshold)
        positives = int(labels.loc[complete].sum())
        threshold_rows.append(
            {
                "threshold_each_year": threshold,
                "eligible_rows": int(complete.sum()),
                "positive_targets": positives,
                "negative_targets": int(complete.sum()) - positives,
                "positive_rate": positives / int(complete.sum()),
                "labels_different_from_25pct": int(labels.loc[complete].ne(policy_labels.loc[complete]).sum()),
                "status": "declared_policy" if threshold == POLICY_THRESHOLD else "sensitivity_only",
            }
        )
    threshold_sensitivity = pd.DataFrame(threshold_rows)

    coverage_rows: list[dict[str, object]] = []
    for threshold in (0.70, 0.75, 0.80, 0.85, 0.90):
        coverage_mask = symmetric.copy()
        for year in (1, 2):
            fields = year_fields(year)
            ratio = pd.to_numeric(evaluation[fields.coverage_ratio], errors="coerce")
            peers = pd.to_numeric(evaluation[fields.peer_clubs], errors="coerce")
            coverage_mask &= ratio.ge(threshold) & peers.ge(10)
        coverage_rows.append(
            {
                "minimum_peer_coverage": threshold,
                "symmetric_reference_rows": int(symmetric.sum()),
                "eligible_rows": int(coverage_mask.sum()),
                "excluded_rows": int((symmetric & ~coverage_mask).sum()),
                "status": "declared_policy" if threshold == MINIMUM_ORIGINAL_LEAGUE_PEER_COVERAGE else "sensitivity_only",
            }
        )
    coverage_sensitivity = pd.DataFrame(coverage_rows)

    league_rows: list[dict[str, object]] = []
    for league, group in evaluation.loc[symmetric].groupby("league", dropna=False):
        available = group[AVAILABLE_FIELD]
        league_rows.append(
            {
                "league": league,
                "symmetric_reference_rows": len(group),
                "complete_eligible_rows": int(available.sum()),
                "schedule_excluded_rows": int((~available).sum()),
                "eligible_rate": float(available.mean()),
            }
        )
    league_summary = pd.DataFrame(league_rows).sort_values("league")

    reason_counts = (
        evaluation.loc[symmetric & ~evaluation[AVAILABLE_FIELD], REASON_FIELD]
        .str.split(";")
        .explode()
        .value_counts()
        .rename_axis("unavailability_reason")
        .reset_index(name="rows")
    )

    checks: list[dict[str, object]] = []

    def add(name: str, actual: object, expected: object, passed: bool, note: str) -> None:
        checks.append(
            {"check": name, "actual": actual, "expected": expected, "passed": bool(passed), "note": note}
        )

    add("development_cutoff_enforced", int(repaired["signed_year"].max()), 2023, repaired["signed_year"].max() == 2023, "No 2024+ extension outcome enters the repair.")
    add("legacy_evaluation_reference", int(legacy.sum()), REFERENCE_COUNTS["legacy_evaluation_rows"], int(legacy.sum()) == REFERENCE_COUNTS["legacy_evaluation_rows"], "Frozen asymmetric reference cohort is reproduced.")
    add("symmetric_evaluation_reference", int(symmetric.sum()), REFERENCE_COUNTS["symmetric_evaluation_rows"], int(symmetric.sum()) == REFERENCE_COUNTS["symmetric_evaluation_rows"], "Prior symmetric-evidence repair is reproduced.")
    add("complete_evaluation_cohort", int(complete.sum()), REFERENCE_COUNTS["complete_evaluation_rows"], int(complete.sum()) == REFERENCE_COUNTS["complete_evaluation_rows"], "Both exact annual windows now pass base evidence and schedule continuity.")
    add("schedule_exclusions_exact", len(exclusions), REFERENCE_COUNTS["schedule_exclusions_after_symmetric_gate"], len(exclusions) == REFERENCE_COUNTS["schedule_exclusions_after_symmetric_gate"], "Incomplete original-league schedules become unavailable, not negative.")
    add("complete_positive_count", int(new_target.loc[complete].sum()), REFERENCE_COUNTS["complete_positive_targets"], int(new_target.loc[complete].sum()) == REFERENCE_COUNTS["complete_positive_targets"], "25% in each year leaves 438 positive labels.")
    add("target_available_iff_complete_gate", int(new_target.notna().eq(evaluation[AVAILABLE_FIELD]).all()), 1, new_target.notna().eq(evaluation[AVAILABLE_FIELD]).all(), "Binary labels exist only when the full gate passes.")
    add("unavailable_never_negative", int(new_target.loc[~evaluation[AVAILABLE_FIELD]].notna().sum()), 0, new_target.loc[~evaluation[AVAILABLE_FIELD]].isna().all(), "Incomplete evidence is nullable rather than forced to class zero.")
    all_normalized = pd.concat([y1_normalized.loc[complete], y2_normalized.loc[complete]])
    add("normalized_shares_bounded", int(all_normalized.between(0, 1, inclusive="both").all()), 1, all_normalized.between(0, 1, inclusive="both").all(), "All eligible annual opportunity shares are within [0,1].")
    add("annual_shares_capped_exact", len(normalization_cases), REFERENCE_COUNTS["annual_shares_capped"], len(normalization_cases) == REFERENCE_COUNTS["annual_shares_capped"], "Six extra-time-derived annual shares above 100% are capped transparently.")
    add("normalization_does_not_change_binary_labels", int(normalization_cases["raw_threshold_label_for_this_year"].eq(normalization_cases["normalized_threshold_label_for_this_year"]).all()), 1, normalization_cases["raw_threshold_label_for_this_year"].eq(normalization_cases["normalized_threshold_label_for_this_year"]).all(), "Capping repairs scale validity without changing 25% labels.")
    add("every_exclusion_has_reason", int(exclusions[REASON_FIELD].str.len().gt(0).all()), 1, exclusions[REASON_FIELD].str.len().gt(0).all(), "Every unavailable reference row has an explicit reason.")
    add("declared_coverage_threshold_present", int(coverage_sensitivity["minimum_peer_coverage"].eq(MINIMUM_ORIGINAL_LEAGUE_PEER_COVERAGE).any()), 1, coverage_sensitivity["minimum_peer_coverage"].eq(MINIMUM_ORIGINAL_LEAGUE_PEER_COVERAGE).any(), "80% policy is shown beside nearby sensitivity values.")
    add("declared_target_threshold_present", int(threshold_sensitivity["threshold_each_year"].eq(POLICY_THRESHOLD).any()), 1, threshold_sensitivity["threshold_each_year"].eq(POLICY_THRESHOLD).any(), "25% policy is shown beside 20% and 30% sensitivity values.")

    checks_frame = pd.DataFrame(checks)
    write_csv(cohorts, OUTPUT / "cohort_comparison.csv")
    write_csv(exclusions, OUTPUT / "schedule_exclusions.csv")
    write_csv(normalization_cases, OUTPUT / "normalization_cases.csv")
    write_csv(threshold_sensitivity, OUTPUT / "threshold_sensitivity.csv")
    write_csv(coverage_sensitivity, OUTPUT / "coverage_threshold_sensitivity.csv")
    write_csv(league_summary, OUTPUT / "league_coverage_summary.csv")
    write_csv(reason_counts, OUTPUT / "unavailability_reason_counts.csv")
    write_csv(checks_frame, OUTPUT / "build_checks.csv")
    (OUTPUT / "target_contract.json").write_text(
        json.dumps(contract_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    sources = [MASTER, GAMES, FREEZE]
    source_manifest = pd.DataFrame(
        {
            "source_file": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sources
    )
    write_csv(source_manifest, OUTPUT / "source_manifest.csv")

    freeze_payload = json.loads(FREEZE.read_text(encoding="utf-8"))
    freeze_rows: list[dict[str, object]] = []
    for artifact in freeze_payload["files"]:
        path = ROOT / artifact["path"]
        actual_hash = sha256(path) if path.exists() else "MISSING"
        freeze_rows.append(
            {
                "artifact": artifact["path"],
                "expected_frozen_sha256": artifact["sha256"],
                "run1_baseline_sha256": actual_hash,
                "matches_frozen_record": actual_hash == artifact["sha256"],
            }
        )
    freeze_comparison = pd.DataFrame(freeze_rows)
    write_csv(freeze_comparison, OUTPUT / "v1_freeze_comparison.csv")
    freeze_divergences = freeze_comparison.loc[
        ~freeze_comparison["matches_frozen_record"], "artifact"
    ].tolist()

    summary = {
        "stage": "complete_contribution_target_repair_run_1",
        "development_cutoff_year": DEVELOPMENT_CUTOFF_YEAR,
        "evaluation_reference_years": list(EVALUATION_YEARS),
        "final_holdout_evaluated": False,
        "models_fitted": 0,
        "deployment_artifacts_changed": 0,
        "legacy_evaluation_rows": int(legacy.sum()),
        "symmetric_evaluation_rows": int(symmetric.sum()),
        "complete_evaluation_rows": int(complete.sum()),
        "schedule_exclusions_after_symmetric_gate": len(exclusions),
        "complete_positive_targets": int(new_target.loc[complete].sum()),
        "annual_shares_capped": len(normalization_cases),
        "preexisting_v1_freeze_divergences": freeze_divergences,
        "checks_passed": int(checks_frame["passed"].sum()),
        "checks_total": len(checks_frame),
        "all_checks_passed": bool(checks_frame["passed"].all()),
        "release_status": "target_contract_repaired_development_only_model_rerun_not_started",
    }
    (OUTPUT / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    readme = f"""# Engine V2 complete contribution-target repair — Run 1

Run 1 closes the known construction defects in the realized-contribution
target without fitting a model or opening 2024+ outcomes.

## Contract

A binary target exists only when both exact post-extension years are fully
observable, contain at least 10 captured extension-club matches, and retain at
least {MINIMUM_ORIGINAL_LEAGUE_PEER_COVERAGE:.0%} of the contemporaneous median
club's original-league schedule over the exact same dates. A failed schedule
gate makes the outcome unavailable; it is never treated as zero.

Opportunity is still all captured club competitions. Annual share is
`same-club minutes / (90 × captured same-club games)` and is capped to [0,1].
The cap affects {len(normalization_cases)} annual values and changes no 25%
binary label. The positive policy remains at least 25% in **each** year, with
20% and 30% sensitivities published separately.

## Development result

- Legacy asymmetric reference cohort: {int(legacy.sum()):,}
- Symmetric Year-1/Year-2 evidence cohort: {int(symmetric.sum()):,}
- Complete schedule-checked cohort: {int(complete.sum()):,}
- Rows made unavailable by schedule continuity: {len(exclusions):,}
- Positive targets under the complete 25% contract: {int(new_target.loc[complete].sum()):,}

The peer-relative rule avoids falsely rejecting league-wide interruptions such
as the shortened 2019–20 Ligue 1 season while failing closed when a particular
club exits the top-flight coverage universe.

## Boundary

This target represents **sustained realized contribution to the extension
club**, not tactical role when available. Injury, loan, transfer, suspension,
international duty and non-selection remain part of realized contribution.
Run 1 fits zero models, changes zero deployment artifacts, and does not inspect
2024+ outcomes. Rolling-origin model development is a later run.

## Pre-existing repository-state warning

The V1 freeze comparison found {len(freeze_divergences)} files already different
from the August 30 freeze record when this evidence package was built:
{chr(10).join(f'- `{path}`' for path in freeze_divergences)}

Both are generated live-request-integrity reports that were already modified
in the working tree before Run 1. Run 1 does not alter them. Their exact Run 1
baseline hashes are recorded in `v1_freeze_comparison.csv`; resolving or
formally re-versioning that separate integrity-report divergence remains a
repository-governance task.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")

    output_files = sorted(
        path
        for path in OUTPUT.iterdir()
        if path.is_file()
        and path.name not in {"output_manifest.csv", "independent_verification.csv", "independent_verification.json"}
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
        raise RuntimeError(f"Contribution-target repair build failed: {failed}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

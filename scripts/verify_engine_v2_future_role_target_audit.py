"""Independent mechanical verification of the future-role target audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "Data" / "processed" / "contract_extension_integration" / "extension_modeling_master.csv"
OUTPUT = ROOT / "Data" / "processed" / "engine_v2_future_role_target_audit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def main() -> None:
    checks: list[dict[str, object]] = []

    def add(name: str, actual: object, expected: object, passed: bool, note: str) -> None:
        checks.append({"check": name, "actual": actual, "expected": expected, "passed": bool(passed), "note": note})

    required = {
        "README.md", "run_summary.json", "cohort_flow.csv", "year1_evidence_asymmetry.csv",
        "threshold_sensitivity.csv", "definition_sensitivity.csv", "denominator_sensitivity.csv",
        "club_window_coverage_sensitivity.csv", "club_window_coverage_break_cases.csv",
        "movement_decomposition.csv", "subgroup_prevalence.csv", "injury_case_audit.csv",
        "injury_burden_decomposition.csv", "injury_available_denominator_summary.csv",
        "audit_findings.csv", "build_checks.csv", "source_manifest.csv", "output_manifest.csv",
    }
    actual_files = {path.name for path in OUTPUT.iterdir() if path.is_file()}
    unexpected = actual_files - required - {"independent_verification.csv"}
    outputs_ok = required.issubset(actual_files) and not unexpected
    add("required_outputs", sorted(actual_files), sorted(required), outputs_ok, "Required outputs exist; the verifier report is the only allowed extra file.")

    manifest = pd.read_csv(OUTPUT / "output_manifest.csv")
    manifest_ok = True
    for row in manifest.itertuples(index=False):
        path = OUTPUT / row.output_file
        manifest_ok &= path.exists() and path.stat().st_size == int(row.bytes) and sha256(path) == row.sha256
    add("output_manifest_exact", int(manifest_ok), 1, manifest_ok, "Every declared output hash and byte count matches.")

    source_manifest = pd.read_csv(OUTPUT / "source_manifest.csv")
    sources_ok = True
    for row in source_manifest.itertuples(index=False):
        path = ROOT / row.source_file
        sources_ok &= path.exists() and path.stat().st_size == int(row.bytes) and sha256(path) == row.sha256
    add("source_manifest_exact", int(sources_ok), 1, sources_ok, "All input sources remain byte-identical to the audited run.")

    frame = pd.read_csv(MASTER, low_memory=False)
    frame["signed_date"] = pd.to_datetime(frame["signed_date"], errors="coerce")
    frame["expiration_date"] = pd.to_datetime(frame["expiration_date"], errors="coerce")
    frame = frame.loc[frame["signed_date"].dt.year.between(2020, 2023)].copy()
    for column in [
        "pre365_same_club_all_competition_opportunity_share",
        "post_y1_same_club_all_competition_opportunity_share",
        "post_y2_same_club_all_competition_opportunity_share",
        "post_y1_same_club_all_competition_games", "post_y2_same_club_all_competition_games",
        "post_y1_same_club_all_competition_minutes", "post_y2_same_club_all_competition_minutes",
        "first_any_outbound_days",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    linked = frame["canonical_player_id"].notna()
    covers = frame["expiration_date"].ge(frame["signed_date"] + pd.Timedelta(days=730))
    pre_ok = as_bool(frame["pre365_window_evidence_eligible_10_club_games"])
    y1_ok = as_bool(frame["post_y1_window_evidence_eligible_10_club_games"])
    y2_ok = as_bool(frame["post_y2_window_evidence_eligible_10_club_games"])
    pre = frame["pre365_same_club_all_competition_opportunity_share"]
    y1 = frame["post_y1_same_club_all_competition_opportunity_share"]
    y2 = frame["post_y2_same_club_all_competition_opportunity_share"]
    current = linked & covers & pre_ok & y2_ok & pre.notna() & y2.notna()
    strict = current & y1_ok & y1.notna()
    data = frame.loc[strict].copy()
    data["target"] = (data["post_y1_same_club_all_competition_opportunity_share"].ge(0.25) & data["post_y2_same_club_all_competition_opportunity_share"].ge(0.25)).astype(int)
    add("current_cohort_recomputed", int(current.sum()), 929, int(current.sum()) == 929, "Current asymmetric evaluation cohort independently reconstructed.")
    add("strict_cohort_recomputed", int(strict.sum()), 922, int(strict.sum()) == 922, "Symmetric Year-1/Year-2 cohort independently reconstructed.")
    add("asymmetry_recomputed", int((current & ~strict).sum()), 7, int((current & ~strict).sum()) == 7, "Seven current rows lack valid Year-1 evidence.")
    add("strict_positive_count", int(data["target"].sum()), 453, int(data["target"].sum()) == 453, "25% both-years target count is exact.")

    asymmetry = pd.read_csv(OUTPUT / "year1_evidence_asymmetry.csv")
    expected_ids = set(frame.loc[current & ~strict, "capology_extension_event_id"])
    add("asymmetry_ids_exact", len(asymmetry), 7, set(asymmetry["capology_extension_event_id"]) == expected_ids, "Saved defect cases are the exact reconstructed rows.")

    thresholds = pd.read_csv(OUTPUT / "threshold_sensitivity.csv")
    threshold_ok = True
    for row in thresholds.itertuples(index=False):
        target = data["post_y1_same_club_all_competition_opportunity_share"].ge(row.threshold) & data["post_y2_same_club_all_competition_opportunity_share"].ge(row.threshold)
        threshold_ok &= int(target.sum()) == int(row.positive_outcomes) and np.isclose(target.mean(), row.positive_rate)
    add("threshold_table_recomputed", int(threshold_ok), 1, threshold_ok, "Every threshold count and rate is independently reproduced.")

    definitions = pd.read_csv(OUTPUT / "definition_sensitivity.csv").set_index("definition")
    weighted = (
        data["post_y1_same_club_all_competition_minutes"] + data["post_y2_same_club_all_competition_minutes"]
    ) / (90 * (data["post_y1_same_club_all_competition_games"] + data["post_y2_same_club_all_competition_games"]))
    weighted_target = weighted.ge(0.25)
    changed = int(weighted_target.ne(data["target"].eq(1)).sum())
    saved_changed = int(definitions.loc["game_weighted_730_day_share_at_least_25pct", "labels_changed_vs_current"])
    add("weighted_definition_recomputed", changed, saved_changed, changed == saved_changed == 146, "Cumulative two-year alternative changes 146 labels.")

    coverage = pd.read_csv(OUTPUT / "club_window_coverage_sensitivity.csv")
    coverage_ok = True
    for row in coverage.itertuples(index=False):
        eligible = data["post_y1_same_club_all_competition_games"].ge(row.minimum_observed_club_games_each_year) & data["post_y2_same_club_all_competition_games"].ge(row.minimum_observed_club_games_each_year)
        coverage_ok &= int(eligible.sum()) == int(row.eligible_records)
    add("coverage_table_recomputed", int(coverage_ok), 1, coverage_ok, "Every minimum-game sensitivity count is independently reproduced.")
    cases = pd.read_csv(OUTPUT / "club_window_coverage_break_cases.csv")
    break_mask = data["post_y1_same_club_all_competition_games"].lt(30) | data["post_y2_same_club_all_competition_games"].lt(30)
    add("coverage_break_ids_exact", len(cases), 47, set(cases["capology_extension_event_id"]) == set(data.loc[break_mask, "capology_extension_event_id"]), "All and only under-30-game windows are saved.")

    movement = pd.read_csv(OUTPUT / "movement_decomposition.csv")
    add("movement_partition", int(movement["records"].sum()), 922, int(movement["records"].sum()) == 922, "Movement states partition the strict cohort.")
    temporary = movement.loc[movement["movement_state"].eq("temporary_first_within_24m")].iloc[0]
    add("temporary_negative_reference", int(temporary["not_sustained_records"]), 182, int(temporary["not_sustained_records"]) == 182, "Temporary movement explains 182 negative labels.")

    injury_cases = pd.read_csv(OUTPUT / "injury_case_audit.csv")
    merged = injury_cases.merge(
        data[["capology_extension_event_id", "post_y1_same_club_all_competition_minutes", "post_y2_same_club_all_competition_minutes", "target"]],
        on="capology_extension_event_id", how="left", validate="one_to_one",
    )
    y1_expected = (merged["post_y1_same_club_all_competition_minutes"] / (90 * merged["y1_available_games"].replace(0, np.nan))).clip(upper=1.05)
    y2_expected = (merged["post_y2_same_club_all_competition_minutes"] / (90 * merged["y2_available_games"].replace(0, np.nan))).clip(upper=1.05)
    injury_formula = np.allclose(y1_expected, merged["y1_availability_adjusted_share"], equal_nan=True) and np.allclose(y2_expected, merged["y2_availability_adjusted_share"], equal_nan=True)
    add("injury_adjusted_share_formula", int(injury_formula), 1, injury_formula, "Saved adjusted shares use raw minutes and audited available-game denominators exactly.")
    add("injury_current_targets_join", int(merged["current_target"].eq(merged["target"]).all()), 1, merged["current_target"].eq(merged["target"]).all(), "Injury subset retains the exact strict target labels.")
    valid = merged["availability_adjusted_target"].notna()
    changed = int(merged.loc[valid, "availability_adjusted_target"].astype(bool).ne(merged.loc[valid, "current_target"].astype(bool)).sum())
    add("injury_label_change_reference", changed, 29, changed == 29, "Availability adjustment changes 29 auditable labels.")

    findings = pd.read_csv(OUTPUT / "audit_findings.csv")
    expected_findings = {f"FR-{number:02d}" for number in range(1, 10)}
    add("finding_ledger_complete", sorted(findings["finding_id"]), sorted(expected_findings), set(findings["finding_id"]) == expected_findings, "Nine unique target-validity findings are present.")
    add("overall_verdict_unique", int(findings["verdict"].eq("NOT_DEPLOYABLE_AS_FUTURE_ROLE").sum()), 1, int(findings["verdict"].eq("NOT_DEPLOYABLE_AS_FUTURE_ROLE").sum()) == 1, "One explicit overall product verdict exists.")

    summary = json.loads((OUTPUT / "run_summary.json").read_text(encoding="utf-8"))
    add("final_holdout_not_evaluated", summary.get("final_holdout_evaluated"), False, summary.get("final_holdout_evaluated") is False, "Run metadata explicitly records the holdout policy.")
    add("runner_checks_passed", summary.get("all_checks_passed"), True, summary.get("all_checks_passed") is True, "Runner completed all internal invariants.")
    readme = (OUTPUT / "README.md").read_text(encoding="utf-8")
    language_ok = "not defensible as\na pure future-role outcome" in readme and "Meaningful Retention" in readme and "Temporary Displacement" in readme
    add("readme_decision_language", int(language_ok), 1, language_ok, "README states the limitation and replacement outcome structure.")

    audit_doc = (ROOT / "docs" / "ENGINE_V2_FUTURE_ROLE_TARGET_AUDIT.md").read_text(encoding="utf-8")
    register = (ROOT / "docs" / "ENGINE_V2_CRITICAL_ISSUES.md").read_text(encoding="utf-8")
    project_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    add(
        "durable_audit_document",
        int(all(phrase in audit_doc for phrase in ["not defensible as a", "symmetric Year-1", "47 of 922", "29 labels", "146 of 922"])),
        1,
        all(phrase in audit_doc for phrase in ["not defensible as a", "symmetric Year-1", "47 of 922", "29 labels", "146 of 922"]),
        "Permanent audit document preserves the target verdict and core evidence.",
    )
    add(
        "critical_register_linked",
        int("E2-CRIT-004" in register and "OPEN — TARGET REPAIR REQUIRED — RELEASE BLOCKING" in register),
        1,
        "E2-CRIT-004" in register and "OPEN — TARGET REPAIR REQUIRED — RELEASE BLOCKING" in register,
        "Target failure is maintained as its own release blocker.",
    )
    add(
        "public_status_corrected",
        int("Meaningful Retention is also blocked until that component is rebuilt" in project_readme),
        1,
        "Meaningful Retention is also blocked until that component is rebuilt" in project_readme,
        "Top-level publication status reflects the later target audit.",
    )

    result = pd.DataFrame(checks)
    result.to_csv(OUTPUT / "independent_verification.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    if not result["passed"].all():
        failed = result.loc[~result["passed"], ["check", "actual", "expected"]]
        raise RuntimeError("Verification failed:\n" + failed.to_string(index=False))
    print(f"All {len(result)}/{len(result)} independent verification checks pass.")


if __name__ == "__main__":
    main()

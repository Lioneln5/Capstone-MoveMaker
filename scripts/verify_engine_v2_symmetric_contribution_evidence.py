"""Independent verification for the symmetric contribution-evidence repair."""

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
    TARGET_FIELD,
    add_symmetric_contribution_target,
)


MASTER = ROOT / "Data" / "processed" / "contract_extension_integration" / "extension_modeling_master.csv"
OUTPUT = ROOT / "Data" / "processed" / "engine_v2_symmetric_contribution_evidence"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"
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


def synthetic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "post_y1_window_fully_observable": [True] * 5,
            "post_y2_window_fully_observable": [True] * 5,
            "post_y1_same_club_all_competition_games": [40, 40, 9, 40, 40],
            "post_y2_same_club_all_competition_games": [40, 40, 40, 9, 40],
            "post_y1_window_evidence_eligible_10_club_games": [True, True, False, True, True],
            "post_y2_window_evidence_eligible_10_club_games": [True, True, True, False, True],
            "post_y1_same_club_all_competition_opportunity_share": [0.30, 0.20, 0.30, 0.30, None],
            "post_y2_same_club_all_competition_opportunity_share": [0.30, 0.30, 0.30, 0.30, 0.30],
        }
    )


def main() -> None:
    checks: list[dict[str, object]] = []

    def add(name: str, actual: object, expected: object, passed: bool, note: str) -> None:
        checks.append(
            {"check": name, "actual": actual, "expected": expected, "passed": bool(passed), "note": note}
        )

    required = {
        "README.md", "target_contract.json", "symmetric_evidence_summary.csv",
        "excluded_asymmetric_evaluation_cases.csv", "unavailability_reasons.csv",
        "build_checks.csv", "source_manifest.csv", "output_manifest.csv", "run_summary.json",
    }
    actual_files = {path.name for path in OUTPUT.iterdir() if path.is_file()}
    allowed_extra = {"independent_verification.csv"}
    add("required_outputs", sorted(actual_files), sorted(required), required.issubset(actual_files) and not (actual_files - required - allowed_extra), "The target-repair evidence package is complete.")

    output_manifest = pd.read_csv(OUTPUT / "output_manifest.csv")
    manifest_ok = True
    for row in output_manifest.itertuples(index=False):
        path = OUTPUT / row.output_file
        manifest_ok &= path.exists() and path.stat().st_size == int(row.bytes) and sha256(path) == row.sha256
    add("output_manifest_exact", int(manifest_ok), 1, manifest_ok, "Every declared output hash and byte count matches.")

    source_manifest = pd.read_csv(OUTPUT / "source_manifest.csv")
    source_ok = True
    for row in source_manifest.itertuples(index=False):
        path = ROOT / row.source_file
        source_ok &= path.exists() and path.stat().st_size == int(row.bytes) and sha256(path) == row.sha256
    add("source_manifest_exact", int(source_ok), 1, source_ok, "The master and V1 freeze record remain byte-identical to the repair run.")

    frame = pd.read_csv(MASTER, low_memory=False)
    frame["signed_date"] = pd.to_datetime(frame["signed_date"], errors="coerce")
    frame["expiration_date"] = pd.to_datetime(frame["expiration_date"], errors="coerce")
    frame["signed_year"] = frame["signed_date"].dt.year.astype("Int64")
    development = frame.loc[frame["signed_year"].le(2023)].copy()
    evaluation = development.loc[development["signed_year"].between(2020, 2023)].copy()

    linked = evaluation["canonical_player_id"].notna()
    covers = evaluation["expiration_date"].ge(evaluation["signed_date"] + pd.Timedelta(days=730))
    pre_ok = as_bool(evaluation["pre365_window_evidence_eligible_10_club_games"])
    y1_ok = as_bool(evaluation["post_y1_window_evidence_eligible_10_club_games"])
    y2_ok = as_bool(evaluation["post_y2_window_evidence_eligible_10_club_games"])
    pre = pd.to_numeric(evaluation["pre365_same_club_all_competition_opportunity_share"], errors="coerce")
    y1 = pd.to_numeric(evaluation["post_y1_same_club_all_competition_opportunity_share"], errors="coerce")
    y2 = pd.to_numeric(evaluation["post_y2_same_club_all_competition_opportunity_share"], errors="coerce")
    legacy = linked & covers & pre_ok & y2_ok & pre.notna() & y2.notna()
    strict = legacy & y1_ok & y1.notna()
    expected_target = (y1.ge(0.25) & y2.ge(0.25)).astype("Int64")

    repaired = add_symmetric_contribution_target(development)
    repaired_eval = repaired.loc[repaired["signed_year"].between(2020, 2023)]
    actual_target = pd.to_numeric(repaired_eval[TARGET_FIELD], errors="coerce")
    add("legacy_reference_recomputed", int(legacy.sum()), 929, int(legacy.sum()) == 929, "The comparison cohort is independently reconstructed.")
    add("strict_reference_recomputed", int(strict.sum()), 922, int(strict.sum()) == 922, "The symmetric cohort is independently reconstructed.")
    add("seven_rows_become_unavailable", int((legacy & actual_target.isna()).sum()), 7, int((legacy & actual_target.isna()).sum()) == 7, "The seven defective rows receive no binary target.")
    add("strict_positive_count", int(actual_target.loc[strict].sum()), 453, int(actual_target.loc[strict].sum()) == 453, "The strict cohort retains 453 positive labels.")
    add("strict_labels_exact", int(actual_target.loc[strict].astype(int).eq(expected_target.loc[strict].astype(int)).all()), 1, actual_target.loc[strict].astype(int).eq(expected_target.loc[strict].astype(int)).all(), "All eligible labels exactly match the declared 25% both-years rule.")
    add("availability_exact", int(repaired_eval[AVAILABLE_FIELD].eq(y1_ok & y2_ok & y1.notna() & y2.notna()).all()), 1, repaired_eval[AVAILABLE_FIELD].eq(y1_ok & y2_ok & y1.notna() & y2.notna()).all(), "Availability is exactly the conjunction of both annual evidence gates and shares.")

    excluded = pd.read_csv(OUTPUT / "excluded_asymmetric_evaluation_cases.csv")
    expected_ids = set(evaluation.loc[legacy & ~strict, "capology_extension_event_id"])
    add("excluded_ids_exact", len(excluded), 7, set(excluded["capology_extension_event_id"]) == expected_ids, "Saved cases are all and only the asymmetric rows.")

    synthetic = add_symmetric_contribution_target(synthetic_frame())
    expected_synthetic = [1, 0, None, None, None]
    actual_synthetic = [None if pd.isna(value) else int(value) for value in synthetic[TARGET_FIELD]]
    add("synthetic_symmetric_gate", actual_synthetic, expected_synthetic, actual_synthetic == expected_synthetic, "Positive, negative, Year-1 ineligible, Year-2 ineligible, and missing-share cases behave correctly.")

    malformed = synthetic_frame()
    malformed.loc[0, "post_y1_window_evidence_eligible_10_club_games"] = False
    mismatch_rejected = False
    try:
        add_symmetric_contribution_target(malformed)
    except ValueError:
        mismatch_rejected = True
    add("stored_flag_mismatch_fails_closed", int(mismatch_rejected), 1, mismatch_rejected, "A stale stored flag cannot silently bypass primitive evidence validation.")

    summary = json.loads((OUTPUT / "run_summary.json").read_text(encoding="utf-8"))
    add("final_holdout_not_opened", summary.get("final_holdout_evaluated"), False, summary.get("final_holdout_evaluated") is False, "The 2024+ cohort remains unopened.")
    add("no_models_fitted", summary.get("models_fitted"), 0, summary.get("models_fitted") == 0, "This is a target repair, not a model experiment.")
    add("no_deployment_change", summary.get("deployment_artifacts_changed"), 0, summary.get("deployment_artifacts_changed") == 0, "No model artifact was changed.")

    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    frozen_ok = True
    for artifact in freeze["files"]:
        if artifact["path"] in REGENERABLE_V1_REPORTS:
            continue
        path = ROOT / artifact["path"]
        frozen_ok &= path.exists() and sha256(path) == artifact["sha256"]
    add("frozen_v1_protected_hashes_exact", int(frozen_ok), 1, frozen_ok, "Every protected V1 artifact retains its recorded hash; regenerable integrity reports are not model artifacts.")

    audit_doc = (ROOT / "docs" / "ENGINE_V2_FUTURE_ROLE_TARGET_AUDIT.md").read_text(encoding="utf-8")
    critical = (ROOT / "docs" / "ENGINE_V2_CRITICAL_ISSUES.md").read_text(encoding="utf-8")
    registry = (ROOT / "models" / "README.md").read_text(encoding="utf-8")
    add("audit_documents_repair_status", int("Symmetric evidence repair implemented" in audit_doc and "TARGET-REPAIR STEP 1 IMPLEMENTED" in critical and "Symmetric contribution evidence" in registry), 1, "Symmetric evidence repair implemented" in audit_doc and "TARGET-REPAIR STEP 1 IMPLEMENTED" in critical and "Symmetric contribution evidence" in registry, "Durable documentation distinguishes the implemented gate from unresolved release blockers.")

    result = pd.DataFrame(checks)
    write_path = OUTPUT / "independent_verification.csv"
    result.to_csv(write_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    if not result["passed"].all():
        failed = result.loc[~result["passed"], ["check", "actual", "expected"]]
        raise RuntimeError("Independent verification failed:\n" + failed.to_string(index=False))
    print(f"All {len(result)}/{len(result)} independent verification checks pass.")


if __name__ == "__main__":
    main()

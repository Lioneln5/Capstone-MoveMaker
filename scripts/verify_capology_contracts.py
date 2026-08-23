"""Independently verify the canonical Capology salary and extension layer."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LEAGUES = ROOT / "Data" / "leagues"
OUTPUT = ROOT / "Data" / "processed" / "capology_contracts"

SALARY_EUR_FIELDS = [
    "weekly_gross_eur", "annual_gross_eur", "weekly_net_eur", "annual_net_eur",
    "bonus_gross_eur", "bonus_net_eur", "total_gross_eur", "total_net_eur",
    "adjusted_total_gross_eur", "adjusted_total_net_eur",
]
EXTENSION_EUR_FIELDS = SALARY_EUR_FIELDS + ["contract_total_gross_eur", "contract_total_net_eur"]
ALLOWED_PLAYER_METHODS = {
    "fbref_accepted_exact_unique_name",
    "fbref_accepted_enhanced_exact_unique_name",
    "canonical_identity_exact_unique_name",
    "unresolved",
}
ALLOWED_CLUB_METHODS = {
    "fbref_accepted_exact_alias",
    "reviewed_exact_capology_alias",
    "unresolved",
}

EXPECTED_SALARY_FILES = 35
EXPECTED_EXTENSION_FILES = 34
EXPECTED_SALARY_ROWS = 19_770
EXPECTED_EXTENSION_ROWS = 2_811
EXPECTED_EXTENSION_EVENTS = 2_805


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add(checks: list[dict], name: str, observed: object, expected: object, passed: bool, notes: str) -> None:
    checks.append({"check": name, "observed": observed, "expected": expected, "passed": bool(passed), "notes": notes})


def main() -> None:
    required = {
        "canonical_salary_panel.csv", "canonical_extension_events.csv", "unresolved_player_matches.csv",
        "unresolved_club_matches.csv", "salary_summary.csv", "extension_summary.csv",
        "capology_cleaning_actions.csv", "field_dictionary.csv", "source_manifest.csv",
        "build_checks.csv", "run_summary.json", "README.md", "output_manifest.csv",
    }
    checks: list[dict] = []
    missing = sorted(name for name in required if not (OUTPUT / name).exists())
    add(checks, "required_outputs_exist", len(missing), 0, not missing, f"Missing: {missing or 'none'}")
    if missing:
        raise RuntimeError(f"Cannot verify; missing outputs: {missing}")

    salary = pd.read_csv(OUTPUT / "canonical_salary_panel.csv")
    extension = pd.read_csv(OUTPUT / "canonical_extension_events.csv")
    source_manifest = pd.read_csv(OUTPUT / "source_manifest.csv")
    build_checks = pd.read_csv(OUTPUT / "build_checks.csv")
    output_manifest = pd.read_csv(OUTPUT / "output_manifest.csv")

    # Verify the immutable, manifest-pinned build inputs rather than blindly
    # enumerating the live Data/leagues folders. Those folders are append-only
    # inputs for current scoring and can legitimately contain seasons added
    # after this frozen research artifact was built.
    salary_manifest = source_manifest.loc[source_manifest["source_type"].eq("salary")].copy()
    extension_manifest = source_manifest.loc[source_manifest["source_type"].eq("extension")].copy()
    raw_salary_files = [ROOT / path for path in salary_manifest["backup_relative_path"]]
    raw_extension_files = [ROOT / path for path in extension_manifest["backup_relative_path"]]
    raw_salary_rows = int(salary_manifest["source_rows"].sum())
    raw_extension_rows = int(extension_manifest["source_rows"].sum())
    salary_schema_counts = set(salary_manifest["source_columns"].astype(int))
    extension_schema_counts = set(extension_manifest["source_columns"].astype(int))

    add(checks, "raw_salary_file_count", len(raw_salary_files), EXPECTED_SALARY_FILES, len(raw_salary_files) == EXPECTED_SALARY_FILES, "Manifest-pinned immutable salary sources.")
    add(checks, "raw_extension_file_count", len(raw_extension_files), EXPECTED_EXTENSION_FILES, len(raw_extension_files) == EXPECTED_EXTENSION_FILES, "Manifest-pinned immutable extension sources including five 2024-25 pages.")
    add(checks, "raw_salary_schema", sorted(salary_schema_counts), [39], salary_schema_counts == {39}, "All frozen salary sources have the audited schema.")
    add(checks, "raw_extension_schema", sorted(extension_schema_counts), [41], extension_schema_counts == {41}, "All frozen extension sources have the audited schema.")
    add(checks, "raw_salary_rows", raw_salary_rows, EXPECTED_SALARY_ROWS, raw_salary_rows == EXPECTED_SALARY_ROWS, "Manifest row counts for the frozen build inputs.")
    add(checks, "raw_extension_rows", raw_extension_rows, EXPECTED_EXTENSION_ROWS, raw_extension_rows == EXPECTED_EXTENSION_ROWS, "Manifest row counts for the frozen build inputs.")
    add(checks, "salary_provenance_conserves_rows", int(salary["source_row_count"].sum()), raw_salary_rows, int(salary["source_row_count"].sum()) == raw_salary_rows, "Collapsed row counts reconcile to raw salary rows.")
    add(checks, "extension_provenance_conserves_rows", int(extension["source_row_count"].sum()), raw_extension_rows, int(extension["source_row_count"].sum()) == raw_extension_rows, "Collapsed event row counts reconcile to raw extension rows.")
    add(checks, "extension_event_count", len(extension), EXPECTED_EXTENSION_EVENTS, len(extension) == EXPECTED_EXTENSION_EVENTS, "Six repeated adjacent-page events are collapsed.")

    salary_key_unique = not salary["capology_salary_id"].duplicated().any() and not salary["salary_natural_key"].duplicated().any()
    extension_key_unique = not extension["capology_extension_event_id"].duplicated().any() and not extension["extension_natural_key"].duplicated().any()
    add(checks, "salary_keys_unique", int(salary["capology_salary_id"].duplicated().sum()), 0, salary_key_unique, "Both hashed and readable keys are unique.")
    add(checks, "extension_keys_unique", int(extension["capology_extension_event_id"].duplicated().sum()), 0, extension_key_unique, "Both hashed and readable keys are unique.")

    signed = pd.to_datetime(extension["signed_date"], errors="coerce")
    expiration = pd.to_datetime(extension["expiration_date"], errors="coerce")
    recomputed_days = (expiration - signed).dt.days
    duration_ok = signed.notna().all() and expiration.notna().all() and recomputed_days.gt(0).all() and np.allclose(
        recomputed_days, extension["exact_duration_days"], equal_nan=False
    )
    add(checks, "extension_dates_and_duration", int(duration_ok), 1, duration_ok, "Exact duration independently recomputed from parsed dates.")
    year_duration_ok = np.allclose(extension["exact_duration_years"], recomputed_days / 365.25, rtol=0, atol=1e-10)
    add(checks, "extension_exact_years", int(year_duration_ok), 1, year_duration_ok, "Exact years use 365.25 days.")

    zero_count = int((salary[SALARY_EUR_FIELDS] == 0).sum().sum() + (extension[EXTENSION_EUR_FIELDS] == 0).sum().sum())
    add(checks, "literal_zero_salary_removed", zero_count, 0, zero_count == 0, "Reported zeros are flags plus missing values, not modeled salary.")
    bonus_consistent = salary["bonus_available"].astype(bool).eq(salary["bonus_gross_eur"].notna()).all() and extension["bonus_available"].astype(bool).eq(extension["bonus_gross_eur"].notna()).all()
    add(checks, "bonus_availability_consistent", int(bonus_consistent), 1, bonus_consistent, "Bonus availability exactly tracks reported gross bonus.")

    arithmetic_mask = extension["contract_total_gross_eur"].notna() & extension["total_gross_eur"].notna() & extension["reported_years"].notna()
    arithmetic_difference = (
        extension.loc[arithmetic_mask, "contract_total_gross_eur"]
        - extension.loc[arithmetic_mask, "total_gross_eur"] * extension.loc[arithmetic_mask, "reported_years"]
    ).abs()
    arithmetic_ok = arithmetic_difference.le(100).all()
    add(checks, "contract_total_arithmetic", int(arithmetic_ok), 1, arithmetic_ok, f"Checked {len(arithmetic_difference):,} non-null events within EUR 100.")

    player_methods_ok = set(salary["player_link_method"].dropna()).issubset(ALLOWED_PLAYER_METHODS) and set(extension["player_link_method"].dropna()).issubset(ALLOWED_PLAYER_METHODS)
    club_methods_ok = set(salary["club_link_method"].dropna()).issubset(ALLOWED_CLUB_METHODS) and set(extension["club_link_method"].dropna()).issubset(ALLOWED_CLUB_METHODS)
    add(checks, "player_link_methods_conservative", int(player_methods_ok), 1, player_methods_ok, "Only exact unique-name methods or unresolved are allowed.")
    add(checks, "club_link_methods_reviewed", int(club_methods_ok), 1, club_methods_ok, "Only accepted/reviewed exact aliases or unresolved are allowed.")
    extension_player_rate = extension["canonical_player_id"].notna().mean()
    add(checks, "extension_player_link_rate", round(extension_player_rate, 4), ">=0.95", extension_player_rate >= 0.95, "No fuzzy matching is required to exceed the threshold.")
    add(checks, "extension_club_link_complete", int(extension["canonical_club_id"].notna().sum()), len(extension), extension["canonical_club_id"].notna().all(), "Every extension club is resolved.")

    manifest_paths = {row.source_relative_path: row for row in source_manifest.itertuples(index=False)}
    expected_source_count = EXPECTED_SALARY_FILES + EXPECTED_EXTENSION_FILES
    source_set_ok = (
        len(manifest_paths) == expected_source_count
        and source_manifest["source_relative_path"].str.startswith("Data/leagues/").all()
        and source_manifest["source_relative_path"].str.endswith(".csv").all()
        and not source_manifest["source_relative_path"].duplicated().any()
    )
    add(checks, "source_manifest_file_set", len(manifest_paths), expected_source_count, source_set_ok, "Manifest contains one unique Data/leagues CSV path per frozen build input.")
    hash_failures = []
    for row in source_manifest.itertuples(index=False):
        backup = ROOT / row.backup_relative_path
        if (
            not backup.exists()
            or sha256(backup) != row.backup_sha256
            or row.source_sha256 != row.backup_sha256
            or int(backup.stat().st_size) != int(row.backup_bytes)
        ):
            hash_failures.append(row.source_relative_path)
    add(checks, "source_and_backup_hashes", len(hash_failures), 0, not hash_failures, f"Frozen backup failures: {hash_failures or 'none'}")

    # Validate runner artifacts that were present when its manifest was written.
    output_hash_failures = []
    for row in output_manifest.itertuples(index=False):
        path = OUTPUT / row.output_file
        if row.output_file in {"independent_verification.csv", "independent_verification.json"}:
            continue
        if not path.exists() or sha256(path) != row.sha256:
            output_hash_failures.append(row.output_file)
    add(checks, "runner_output_hashes", len(output_hash_failures), 0, not output_hash_failures, f"Failures: {output_hash_failures or 'none'}")
    build_ok = build_checks["passed"].astype(str).str.casefold().eq("true").all()
    add(checks, "runner_build_checks", int(build_ok), 1, build_ok, "All builder checks passed before independent verification.")

    result = pd.DataFrame(checks)
    result.to_csv(OUTPUT / "independent_verification.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    payload = {
        "checks_passed": int(result["passed"].sum()),
        "checks_total": len(result),
        "all_checks_passed": bool(result["passed"].all()),
        "raw_salary_rows": raw_salary_rows,
        "canonical_salary_rows": len(salary),
        "raw_extension_rows": raw_extension_rows,
        "canonical_extension_events": len(extension),
        "extension_player_link_rate": round(float(extension_player_rate), 6),
        "extension_club_link_rate": round(float(extension["canonical_club_id"].notna().mean()), 6),
    }
    (OUTPUT / "independent_verification.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    files = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name != "output_manifest.csv")
    manifest = pd.DataFrame([{"output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files])
    manifest.to_csv(OUTPUT / "output_manifest.csv", index=False, encoding="utf-8-sig", lineterminator="\n")

    if not result["passed"].all():
        failed = result.loc[~result["passed"], "check"].tolist()
        raise RuntimeError(f"Independent verification failed: {failed}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

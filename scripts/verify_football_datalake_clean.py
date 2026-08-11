"""Independently verify the cleaned supplemental datalake layer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from audit_football_datalake import ROOT, sha256


CLEAN = ROOT / "Data" / "processed" / "football_datalake_clean"


def rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def check(condition: bool, name: str, detail: str, results: list[dict[str, str]]) -> None:
    results.append({"check": name, "status": "pass" if condition else "fail", "detail": detail})


def main() -> None:
    results: list[dict[str, str]] = []
    required = [
        "source_file_manifest.csv", "table_summary.csv", "column_dictionary.csv",
        "cleaning_actions.csv", "cleaning_checks.csv", "foreign_key_checks.csv",
        "transfer_fee_summary.csv", "cleaning_summary.json", "README.md",
    ]
    check(all((CLEAN / name).is_file() for name in required), "required_outputs", "All required cleaning artifacts exist.", results)

    summary = json.loads((CLEAN / "cleaning_summary.json").read_text(encoding="utf-8"))
    manifests = list(rows(CLEAN / "source_file_manifest.csv"))
    tables = list(rows(CLEAN / "table_summary.csv"))
    checks = list(rows(CLEAN / "cleaning_checks.csv"))
    check(len(manifests) == 11 == len(tables), "table_cardinality", f"manifest={len(manifests)}, summaries={len(tables)}", results)

    hashes_ok = True
    for row in manifests:
        source = ROOT / row["source_file"]
        backup = ROOT / row["backup_file"]
        hashes_ok &= sha256(source) == row["source_sha256"] == sha256(backup) == row["backup_sha256"]
    check(hashes_ok, "source_backup_hashes", "Every live source and immutable backup matches the recorded SHA-256.", results)

    row_counts_ok = True
    record_sequence_ok = True
    flagged_missing: dict[str, int] = {}
    flagged_duplicate: dict[str, int] = {}
    output_hashes_ok = True
    required_helpers_ok = True
    for table in tables:
        name = table["table"]
        path = ROOT / table["clean_file"]
        expected = int(table["clean_rows"])
        count = missing = duplicate = 0
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required_helpers_ok &= {"source_file", "source_row_number", "source_record_id", "natural_key_missing", "natural_key_duplicate", "has_mojibake_marker"}.issubset(reader.fieldnames or [])
            for count, row in enumerate(reader, start=1):
                record_sequence_ok &= row["source_row_number"] == str(count)
                record_sequence_ok &= row["source_record_id"] == f"{name}:{count}"
                missing += row["natural_key_missing"].casefold() == "true"
                duplicate += row["natural_key_duplicate"].casefold() == "true"
        row_counts_ok &= count == expected == int(table["source_rows"])
        flagged_missing[name] = missing
        flagged_duplicate[name] = duplicate
        output_hashes_ok &= sha256(path) == table["output_sha256"]
    check(row_counts_ok, "row_count_preservation", f"All tables reconcile; total clean rows={summary['clean_rows']:,}", results)
    check(record_sequence_ok, "source_record_sequence", "Every clean row retains the expected source row number and stable source record ID.", results)
    check(required_helpers_ok, "required_helper_columns", "Every clean table contains provenance and key/encoding flags.", results)
    check(output_hashes_ok, "clean_output_hashes", "Every clean CSV matches its recorded SHA-256.", results)
    flags_ok = all(flagged_missing[row["table"]] == int(row["missing_key_rows"]) and flagged_duplicate[row["table"]] == int(row["duplicate_key_rows"]) for row in tables)
    check(flags_ok, "key_flag_reconciliation", "Row-level missing and duplicate flags match the table summary.", results)

    fee_counts = {row["transfer_fee_status"]: int(row["records"]) for row in rows(CLEAN / "transfer_fee_summary.csv")}
    recomputed: dict[str, int] = {}
    fee_logic_ok = True
    for row in rows(CLEAN / "tables" / "transfer_history_clean.csv"):
        status = row["transfer_fee_status"]
        recomputed[status] = recomputed.get(status, 0) + 1
        fee = row["transfer_fee"]
        if fee == "":
            fee_logic_ok &= status == "missing"
        elif float(fee) == 0:
            fee_logic_ok &= status == "unclassified_zero"
        elif float(fee) > 0:
            fee_logic_ok &= status == "positive_reported"
        else:
            fee_logic_ok &= status == "invalid_negative"
    check(fee_logic_ok and fee_counts == recomputed, "transfer_fee_states", f"Recomputed fee states={recomputed}", results)

    latest_ok = all(row["is_derived_latest_snapshot"].casefold() == "true" and row["snapshot_leakage_risk"].casefold() == "true" for row in rows(CLEAN / "tables" / "player_latest_market_value_clean.csv"))
    check(latest_ok, "latest_snapshot_flags", "Every latest-value row is marked derived and leakage-sensitive.", results)
    blocking_ok = all(row["passed"].casefold() == "true" for row in checks if row["severity"] == "blocking")
    check(blocking_ok and summary["blocking_failures"] == 0, "blocking_checks", "All blocking cleaning checks pass.", results)

    failed = [row for row in results if row["status"] == "fail"]
    payload = {"status": "pass" if not failed else "fail", "checks": results}
    with (CLEAN / "independent_verification.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "status", "detail"])
        writer.writeheader()
        writer.writerows(results)
    (CLEAN / "independent_verification.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

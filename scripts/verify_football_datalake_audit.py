"""Independently verify the supplemental datalake audit outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "football-datasets-main" / "datalake" / "transfermarkt"
AUDIT = ROOT / "Data" / "processed" / "football_datalake_audit"


def rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def data_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        return sum(1 for _ in reader)


def check(condition: bool, name: str, detail: str, results: list[dict[str, str]]) -> None:
    results.append({"check": name, "status": "pass" if condition else "fail", "detail": detail})


def main() -> None:
    required = [
        "source_file_manifest.csv", "table_summary.csv", "column_dictionary.csv",
        "key_duplicate_checks.csv", "foreign_key_checks.csv", "overlap_summary.csv",
        "profile_overlap_summary.csv", "valuation_conflicts.csv", "transfer_conflicts.csv",
        "performance_reconciliation_summary.csv", "performance_reconciliation_samples.csv",
        "cleaning_recommendations.csv", "audit_summary.json", "README.md",
    ]
    results: list[dict[str, str]] = []
    check(all((AUDIT / name).is_file() for name in required), "required_outputs", "All required audit artifacts exist.", results)

    summary = json.loads((AUDIT / "audit_summary.json").read_text(encoding="utf-8"))
    manifests = list(rows(AUDIT / "source_file_manifest.csv"))
    tables = list(rows(AUDIT / "table_summary.csv"))
    keys = list(rows(AUDIT / "key_duplicate_checks.csv"))
    performance = list(rows(AUDIT / "performance_reconciliation_summary.csv"))
    valuation_conflicts = list(rows(AUDIT / "valuation_conflicts.csv"))

    check(len(manifests) == 11 == len(tables) == len(keys), "table_cardinality", f"manifest={len(manifests)}, summaries={len(tables)}, key checks={len(keys)}", results)
    check(sum(int(row["data_rows"]) for row in manifests) == summary["source_rows"], "row_total_reconciliation", f"Total={summary['source_rows']:,}", results)
    check(all(row["is_git_lfs_pointer"].casefold() == "false" for row in manifests), "lfs_resolution", "No audited source remains a Git LFS pointer.", results)

    actual_counts = {}
    for row in manifests:
        path = ROOT / row["source_file"]
        actual_counts[row["table"]] = data_row_count(path)
    check(all(actual_counts[row["table"]] == int(row["data_rows"]) for row in manifests), "source_row_recount", "Independent CSV recount matches every manifest.", results)

    key_math_ok = all(
        int(row["rows"]) - int(row["key_null_rows"]) - int(row["unique_key_count"]) == int(row["duplicate_key_rows"])
        for row in keys
    )
    check(key_math_ok, "key_math", "rows - null-key rows - unique keys equals duplicate-key rows for every table.", results)
    check(len(valuation_conflicts) == summary["valuation_overlap"]["differing_player_date_values"], "valuation_conflict_reconciliation", f"Conflicts={len(valuation_conflicts):,}", results)
    check(len(performance) == 6 and len({row["shared_keys"] for row in performance}) == 1, "performance_metric_reconciliation", "All six metrics use one shared-key population.", results)
    check(int(performance[0]["shared_keys"]) == summary["performance_reconciliation"]["shared_performance_keys"], "performance_summary_reconciliation", f"Shared keys={performance[0]['shared_keys']}", results)

    failed = [row for row in results if row["status"] == "fail"]
    with (AUDIT / "independent_verification.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "status", "detail"])
        writer.writeheader()
        writer.writerows(results)
    payload = {"status": "pass" if not failed else "fail", "checks": results}
    (AUDIT / "independent_verification.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

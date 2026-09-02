"""Independent checks for the Capology club-financial-context source audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Data" / "processed" / "capology_club_financial_context_audit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add(checks: list[dict], name: str, observed: object, expected: object, passed: bool, note: str) -> None:
    checks.append(
        {
            "check": name,
            "observed": observed,
            "expected": expected,
            "passed": bool(passed),
            "note": note,
        }
    )


def main() -> None:
    required = {
        "README.md",
        "bonus_coverage_by_season.csv",
        "club_identity_audit.csv",
        "extension_coverage_summary.csv",
        "extension_prior_season_coverage.csv",
        "file_profile.csv",
        "numeric_coverage.csv",
        "paired_club_season_audit.csv",
        "parsed_payroll_audit_panel.csv",
        "parsed_transfer_window_audit_panel.csv",
        "source_checks.csv",
        "source_manifest.csv",
        "summary.json",
    }
    checks: list[dict] = []
    present = {path.name for path in OUTPUT.iterdir() if path.is_file()}
    add(checks, "required_outputs_present", len(required & present), len(required), required.issubset(present), "All audit deliverables exist.")

    manifest = pd.read_csv(OUTPUT / "source_manifest.csv")
    payroll = pd.read_csv(OUTPUT / "parsed_payroll_audit_panel.csv", low_memory=False)
    transfers = pd.read_csv(OUTPUT / "parsed_transfer_window_audit_panel.csv", low_memory=False)
    paired = pd.read_csv(OUTPUT / "paired_club_season_audit.csv", low_memory=False)
    source_checks = pd.read_csv(OUTPUT / "source_checks.csv")
    extension = pd.read_csv(OUTPUT / "extension_prior_season_coverage.csv", low_memory=False)
    extension_summary = pd.read_csv(OUTPUT / "extension_coverage_summary.csv")
    summary = json.loads((OUTPUT / "summary.json").read_text())

    add(checks, "manifest_rows", len(manifest), 110, len(manifest) == 110, "One manifest row per source CSV.")
    add(checks, "manifest_paths_unique", int(manifest["source_relative_path"].duplicated().sum()), 0, manifest["source_relative_path"].is_unique, "No source file is counted twice.")
    hash_mismatches = []
    for row in manifest.itertuples(index=False):
        path = ROOT / row.source_relative_path
        if not path.exists() or sha256(path) != row.source_sha256:
            hash_mismatches.append(row.source_relative_path)
    add(checks, "raw_source_hashes_stable", len(hash_mismatches), 0, not hash_mismatches, "Every raw CSV is byte-identical to its audit manifest hash.")

    add(checks, "source_checks_all_pass", int(source_checks["passed"].astype(bool).sum()), len(source_checks), source_checks["passed"].astype(bool).all(), "All primary audit checks pass independently loaded output.")
    add(checks, "panel_row_counts", f"{len(payroll)} payroll; {len(transfers)} transfer", "1074 each", len(payroll) == len(transfers) == 1074, "Expected club-season count.")
    key = ["league_folder", "season_start_year", "capology_club_slug"]
    add(checks, "payroll_key_unique", int(payroll.duplicated(key).sum()), 0, not payroll.duplicated(key).any(), "Payroll grain is unique.")
    add(checks, "transfer_key_unique", int(transfers.duplicated(key).sum()), 0, not transfers.duplicated(key).any(), "Transfer-window grain is unique.")
    pkeys = set(map(tuple, payroll[key].itertuples(index=False, name=None)))
    tkeys = set(map(tuple, transfers[key].itertuples(index=False, name=None)))
    add(checks, "source_family_keys_match", len(pkeys ^ tkeys), 0, pkeys == tkeys, "Exact paired club-season universe.")
    add(checks, "paired_merge_complete", int(paired["_merge"].eq("both").sum()), len(paired), paired["_merge"].eq("both").all(), "Outer merge contains no source-only records.")

    required_payroll = ["parsed_weekly_gross_eur", "parsed_annual_gross_eur", "parsed_total_gross_eur", "parsed_adjusted_total_gross_eur"]
    required_transfer = ["parsed_income_eur", "parsed_expense_eur", "parsed_balance_eur", "parsed_adjbalance_eur", "parsed_players", "parsed_foreign", "parsed_age"]
    missing = int(payroll[required_payroll].isna().sum().sum() + transfers[required_transfer].isna().sum().sum())
    add(checks, "required_parsed_values_complete", missing, 0, missing == 0, "Optional bonus excluded from completeness requirement.")
    balance_error = (transfers["parsed_balance_eur"] - transfers["parsed_income_eur"] + transfers["parsed_expense_eur"]).abs().max()
    add(checks, "balance_identity", float(balance_error), "<=2 EUR", bool(balance_error <= 2), "Income minus expense independently recomputed.")

    position_columns = [
        f"{position}_{basis}_{currency}"
        for currency in ["eur", "gbp", "usd"]
        for basis in ["gross", "net"]
        for position in ["k", "d", "m", "f"]
    ]
    position_values = []
    for relative in manifest.loc[manifest["source_family"].eq("payroll"), "source_relative_path"]:
        raw = pd.read_csv(ROOT / relative, usecols=position_columns, dtype=str)
        position_values.extend(raw[position_columns].fillna("<MISSING>").to_numpy().ravel().tolist())
    locked = sum("Locked" in str(value) for value in position_values)
    add(checks, "position_group_payroll_unavailable", locked, len(position_values), locked == len(position_values), "Every position-group cell is a paywalled Locked placeholder, not usable data.")

    unresolved = payroll.loc[payroll["canonical_club_id"].isna(), "capology_club_slug"].nunique()
    link_rate = payroll["canonical_club_id"].notna().mean()
    add(checks, "canonical_link_summary", f"{link_rate:.6f}; {unresolved} unresolved", f"{summary['canonical_link_rate']:.6f}; {summary['unresolved_unique_clubs']} unresolved", abs(link_rate - summary["canonical_link_rate"]) < 1e-12 and unresolved == summary["unresolved_unique_clubs"], "Summary agrees with panel.")

    add(checks, "extension_event_grain", int(extension["capology_extension_event_id"].duplicated().sum()), 0, extension["capology_extension_event_id"].is_unique, "One coverage row per canonical extension event.")
    add(checks, "extension_event_count", len(extension), 2805, len(extension) == 2805, "Exact canonical extension universe.")
    evaluation_ids = set(pd.read_csv(ROOT / "Data" / "processed" / "integrated_contract_profile" / "integration_evaluation_cohort.csv", usecols=["capology_extension_event_id"])["capology_extension_event_id"])
    observed_ids = set(extension.loc[extension["in_integrated_839_evaluation_cohort"].astype(bool), "capology_extension_event_id"])
    add(checks, "evaluation_cohort_membership_exact", len(observed_ids ^ evaluation_ids), 0, observed_ids == evaluation_ids and len(observed_ids) == 839, "The 839-row published evaluation cohort is identified exactly.")
    recomputed = []
    for scope, subset in {
        "all_canonical_extension_events": extension,
        "integrated_839_evaluation_cohort": extension.loc[extension["in_integrated_839_evaluation_cohort"].astype(bool)],
    }.items():
        recomputed.append((scope, len(subset), int(subset["prior_completed_season_panel_available"].astype(bool).sum())))
    reported = set(zip(extension_summary["scope"], extension_summary["events"], extension_summary["prior_completed_season_available"]))
    add(checks, "extension_coverage_summary_reconciles", len(set(recomputed) ^ reported), 0, set(recomputed) == reported, "Coverage summary exactly reconciles to event-level audit.")
    evaluation = extension.loc[extension["in_integrated_839_evaluation_cohort"].astype(bool)]
    missing_eval = evaluation.loc[~evaluation["prior_completed_season_panel_available"].astype(bool)]
    promoted_labels = missing_eval["coverage_status"].eq("missing_prior_top_flight_panel_likely_promoted").sum()
    add(checks, "evaluation_missingness_classified", int(promoted_labels), len(missing_eval), promoted_labels == len(missing_eval) == 44, "All 44 missing prior-season rows have an event-season panel, consistent with promoted-club coverage loss.")

    freeze_path = ROOT / "Data" / "processed" / "data_usage_audit_2026_09_02" / "v1_freeze_check.csv"
    freeze = pd.read_csv(freeze_path)
    freeze_bad = []
    for row in freeze.itertuples(index=False):
        path = ROOT / row.path
        if not path.exists() or sha256(path) != row.expected_sha256:
            freeze_bad.append(row.path)
    add(checks, "frozen_v1_artifacts_unchanged", len(freeze_bad), 0, not freeze_bad, "All 24 previously frozen V1 artifact hashes remain exact.")

    results = pd.DataFrame(checks)
    results.to_csv(OUTPUT / "independent_verification.csv", index=False)
    payload = {
        "checks_passed": int(results["passed"].sum()),
        "checks_total": len(results),
        "all_passed": bool(results["passed"].all()),
        "failed_checks": results.loc[~results["passed"], "check"].tolist(),
    }
    (OUTPUT / "independent_verification.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

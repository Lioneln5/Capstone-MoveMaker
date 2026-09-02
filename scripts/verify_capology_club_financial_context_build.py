"""Independent verification of the canonical Capology club-season build."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Data" / "processed" / "capology_club_financial_context"
sys.path.insert(0, str(ROOT / "scripts"))
from build_capology_contracts import CLUB_ALIASES  # noqa: E402


EXPECTED_ALIASES = {
    "bastia": 595,
    "carpi": 4102,
    "cesena": 1429,
    "cordoba": 993,
    "evian": 14171,
    "gazelecajaccio": 3558,
    "holsteinkiel": 269,
    "hullcity": 3008,
    "ingolstadt": 4795,
    "middlesbrough": 641,
    "nancy": 1159,
    "palermo": 458,
    "pescara": 2921,
    "qpr": 1039,
    "sportinggijon": 2448,
    "sunderland": 289,
}

EXPECTED_CANDIDATES = {
    "club_payroll_annual_gross_eur",
    "transfer_income_eur",
    "transfer_spend_eur",
    "transfer_balance_eur",
    "squad_average_age",
    "payroll_log_eur",
    "payroll_league_percentile",
    "payroll_to_league_median",
    "transfer_spend_log_eur",
    "transfer_income_log_eur",
    "transfer_spend_to_payroll",
    "transfer_income_to_payroll",
    "squad_foreign_share",
    "payroll_growth_1y",
    "payroll_cagr_3y",
    "transfer_spend_3y_mean_eur",
    "transfer_income_3y_mean_eur",
    "transfer_balance_3y_mean_eur",
    "transfer_spend_3y_std_eur",
    "wage_structure_median_eur",
    "wage_structure_max_eur",
    "wage_structure_top5_share",
    "wage_structure_hhi",
    "wage_structure_gini",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def money(value: object) -> float:
    if pd.isna(value) or str(value).strip() in {"", "-", "—"}:
        return np.nan
    text = re.sub(r"[^0-9,.-]", "", str(value)).replace(",", "")
    return float(text)


def club_slug(value: object) -> str | None:
    match = re.search(r"/club/([^/]+)/", str(value))
    return match.group(1) if match else None


def season_start(path: Path) -> int:
    match = re.search(r"(20\d{2})_(20\d{2})", path.name)
    if not match:
        raise ValueError(path)
    return int(match.group(1))


def add(checks: list[dict], name: str, observed: object, expected: object, passed: bool, note: str) -> None:
    checks.append({"check": name, "observed": observed, "expected": expected, "passed": bool(passed), "note": note})


def source_reconstruction(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    payroll_rows, transfer_rows = [], []
    for row in manifest.itertuples(index=False):
        path = ROOT / row.source_relative_path
        raw = pd.read_csv(path, low_memory=False)
        year = season_start(path)
        league = row.league
        for record in raw.itertuples(index=False):
            base = {
                "league": league,
                "season_start_year": year,
                "capology_club_slug": club_slug(record.club),
            }
            if row.source_family == "payroll":
                payroll_rows.append(
                    {
                        **base,
                        "club_payroll_weekly_gross_eur": money(record.weekly_gross_eur),
                        "club_payroll_annual_gross_eur": money(record.annual_gross_eur),
                        "club_payroll_bonus_gross_eur": money(record.bonus_gross_eur),
                        "club_payroll_total_gross_eur": money(record.total_gross_eur),
                    }
                )
            else:
                transfer_rows.append(
                    {
                        **base,
                        "transfer_income_eur": money(record.income_eur),
                        "transfer_spend_eur": money(record.expense_eur),
                        "transfer_balance_eur": money(record.balance_eur),
                        "squad_player_count": float(record.players),
                        "squad_foreign_player_count": float(record.foreign),
                        "squad_average_age": float(record.age),
                    }
                )
    return pd.DataFrame(payroll_rows), pd.DataFrame(transfer_rows)


def main() -> None:
    required = {
        "README.md",
        "build_checks.csv",
        "canonical_club_season_financial_context.csv",
        "extension_feature_coverage_summary.csv",
        "extension_prior_season_feature_coverage.csv",
        "field_dictionary.csv",
        "output_manifest.csv",
        "reviewed_club_aliases.csv",
        "run_summary.json",
        "source_manifest.csv",
    }
    checks: list[dict] = []
    present = {path.name for path in OUTPUT.iterdir() if path.is_file()}
    add(checks, "required_outputs_present", len(required & present), len(required), required.issubset(present), "Complete build deliverable set.")

    manifest = pd.read_csv(OUTPUT / "source_manifest.csv")
    aliases = pd.read_csv(OUTPUT / "reviewed_club_aliases.csv")
    panel = pd.read_csv(OUTPUT / "canonical_club_season_financial_context.csv", low_memory=False)
    coverage = pd.read_csv(OUTPUT / "extension_prior_season_feature_coverage.csv", low_memory=False)
    coverage_summary = pd.read_csv(OUTPUT / "extension_feature_coverage_summary.csv")
    dictionary = pd.read_csv(OUTPUT / "field_dictionary.csv")
    build_checks = pd.read_csv(OUTPUT / "build_checks.csv")
    output_manifest = pd.read_csv(OUTPUT / "output_manifest.csv")
    summary = json.loads((OUTPUT / "run_summary.json").read_text())

    add(checks, "source_manifest_count", len(manifest), 110, len(manifest) == 110 and manifest["source_relative_path"].is_unique, "Exactly 110 unique raw sources.")
    source_hash_bad = []
    for row in manifest.itertuples(index=False):
        path = ROOT / row.source_relative_path
        if not path.exists() or sha256(path) != row.source_sha256:
            source_hash_bad.append(row.source_relative_path)
    add(checks, "raw_source_hashes_exact", len(source_hash_bad), 0, not source_hash_bad, "All raw inputs remain byte-identical to build manifest.")

    alias_map = dict(zip(aliases["capology_alias_normalized"], aliases["canonical_club_id"]))
    add(checks, "reviewed_alias_map_exact", len(set(alias_map.items()) ^ set(EXPECTED_ALIASES.items())), 0, alias_map == EXPECTED_ALIASES, "All 16 reviewed aliases map to the predeclared canonical IDs.")
    shared = {key: CLUB_ALIASES.get(key) for key in EXPECTED_ALIASES}
    add(checks, "shared_alias_map_exact", len(set(shared.items()) ^ set(EXPECTED_ALIASES.items())), 0, shared == EXPECTED_ALIASES, "Shared Capology linker contains the reviewed map.")
    tm = pd.read_csv(ROOT / "Data" / "processed" / "transfermarkt_clean" / "tables" / "clubs_clean.csv", low_memory=False)
    tm = tm.loc[tm["club_id"].isin(EXPECTED_ALIASES.values())]
    evidence_ids = set(tm["club_id"].astype(int))
    url_ids = {
        int(row.canonical_club_id)
        for row in aliases.itertuples(index=False)
        if f"/verein/{int(row.canonical_club_id)}" in str(row.transfermarkt_url)
    }
    add(checks, "alias_transfermarkt_evidence", len(evidence_ids & url_ids), 16, evidence_ids == url_ids == set(EXPECTED_ALIASES.values()), "Every accepted alias has matching Transfermarkt club-ID URL evidence.")

    add(checks, "build_checks_all_pass", int(build_checks["passed"].astype(bool).sum()), len(build_checks), build_checks["passed"].astype(bool).all(), "Primary builder gates all pass.")
    key = ["canonical_club_id", "season_start_year"]
    add(checks, "canonical_grain", f"{len(panel)} rows; {panel['canonical_club_id'].nunique()} clubs", "1074 rows; 164 clubs", len(panel) == 1074 and panel["canonical_club_id"].nunique() == 164 and not panel.duplicated(key).any(), "One canonical row per club-season.")
    add(checks, "all_identity_links_complete", int(panel["canonical_club_id"].notna().sum()), len(panel), panel["canonical_club_id"].notna().all() and panel["canonical_club_name"].notna().all(), "No unresolved canonical identity.")
    add(checks, "stable_ids_unique", int(panel["canonical_club_season_id"].duplicated().sum()), 0, panel["canonical_club_season_id"].is_unique, "Stable IDs are unique.")

    raw_payroll, raw_transfer = source_reconstruction(manifest)
    source_key = ["league", "season_start_year", "capology_club_slug"]
    raw = raw_payroll.merge(raw_transfer, on=source_key, validate="one_to_one")
    compare_columns = [
        "club_payroll_weekly_gross_eur",
        "club_payroll_annual_gross_eur",
        "club_payroll_bonus_gross_eur",
        "club_payroll_total_gross_eur",
        "transfer_income_eur",
        "transfer_spend_eur",
        "transfer_balance_eur",
        "squad_player_count",
        "squad_foreign_player_count",
        "squad_average_age",
    ]
    compared = panel.merge(raw, on=source_key, how="outer", suffixes=("_built", "_raw"), indicator=True, validate="one_to_one")
    value_mismatch = 0
    for column in compare_columns:
        built = pd.to_numeric(compared[f"{column}_built"], errors="coerce")
        source = pd.to_numeric(compared[f"{column}_raw"], errors="coerce")
        value_mismatch += int((~np.isclose(built, source, equal_nan=True, rtol=0, atol=1e-9)).sum())
    add(checks, "raw_values_reconstructed_exactly", value_mismatch, 0, compared["_merge"].eq("both").all() and value_mismatch == 0, "Every retained raw value independently re-parses from source.")

    expected_percentile = panel.groupby(["league", "season_start_year"])["club_payroll_annual_gross_eur"].rank(method="average", pct=True)
    rank_error = np.abs(expected_percentile - panel["payroll_league_percentile"]).max()
    add(checks, "payroll_percentile_recomputed", float(rank_error), "<=1e-12", bool(rank_error <= 1e-12), "League-season ranks independently recomputed.")
    expected_median = panel.groupby(["league", "season_start_year"])["club_payroll_annual_gross_eur"].transform("median")
    median_error = np.abs(panel["payroll_to_league_median"] - panel["club_payroll_annual_gross_eur"] / expected_median).max()
    add(checks, "payroll_median_ratio_recomputed", float(median_error), "<=1e-12", bool(median_error <= 1e-12), "Contemporaneous league median ratio independently recomputed.")

    indexed = panel.set_index(key)
    growth_bad = 0
    history_bad = 0
    rolling_bad = 0
    for row in panel.itertuples(index=False):
        club_id, year = int(row.canonical_club_id), int(row.season_start_year)
        prior = (club_id, year - 1)
        has_prior = prior in indexed.index
        if bool(row.prior_top_flight_panel_available) != has_prior:
            history_bad += 1
        expected_growth = row.club_payroll_annual_gross_eur / indexed.loc[prior, "club_payroll_annual_gross_eur"] - 1 if has_prior else np.nan
        if not np.isclose(row.payroll_growth_1y, expected_growth, equal_nan=True, rtol=0, atol=1e-12):
            growth_bad += 1
        keys3 = [(club_id, year - offset) for offset in range(3)]
        has3 = all(item in indexed.index for item in keys3)
        if bool(row.history_3y_consecutive_available) != has3:
            history_bad += 1
        expected_spend = indexed.loc[keys3, "transfer_spend_eur"].mean() if has3 else np.nan
        if not np.isclose(row.transfer_spend_3y_mean_eur, expected_spend, equal_nan=True, rtol=0, atol=1e-6):
            rolling_bad += 1
    add(checks, "exact_prior_season_history", history_bad, 0, history_bad == 0, "History flags never bridge a missing top-flight season.")
    add(checks, "payroll_growth_recomputed", growth_bad, 0, growth_bad == 0, "Every one-year growth value independently recomputed.")
    add(checks, "rolling_spend_recomputed", rolling_bad, 0, rolling_bad == 0, "Every three-season spend mean independently recomputed.")
    add(checks, "lineage_not_future", int((panel["feature_history_max_season_start_year"] > panel["season_start_year"]).sum()), 0, (panel["feature_history_max_season_start_year"] <= panel["season_start_year"]).all(), "No feature lineage extends beyond row season.")
    adjusted = [column for column in panel if "adjusted" in column or "adjbalance" in column]
    add(checks, "retrospective_adjusted_fields_excluded", len(adjusted), 0, not adjusted, "No 2025-adjusted Capology field is retained.")

    expected_eligible = panel["known_player_wage_count"].ge(15) & panel["known_player_wage_coverage_ratio"].between(0.95, 1.05)
    eligibility_mismatch = int(expected_eligible.ne(panel["wage_structure_eligible"].astype(bool)).sum())
    emitted = ["wage_structure_median_eur", "wage_structure_max_eur", "wage_structure_top5_share", "wage_structure_hhi", "wage_structure_gini"]
    invalid_emitted = int(panel.loc[~expected_eligible, emitted].notna().sum().sum())
    add(checks, "wage_structure_gate_exact", eligibility_mismatch, 0, eligibility_mismatch == 0 and int(expected_eligible.sum()) == 666, "Coverage rule independently reproduces 666 eligible club-seasons.")
    add(checks, "wage_structure_invalid_rows_null", invalid_emitted, 0, invalid_emitted == 0, "No distribution feature escapes the coverage gate.")

    eligible_fields = set(dictionary.loc[dictionary["candidate_model_eligible"].astype(bool), "field"])
    add(checks, "candidate_feature_allowlist_exact", len(eligible_fields ^ EXPECTED_CANDIDATES), 0, eligible_fields == EXPECTED_CANDIDATES, "Field dictionary permits only preregistered candidates.")
    duplicates = int(dictionary["field"].duplicated().sum())
    add(checks, "field_dictionary_complete", f"{len(dictionary)} rows; {duplicates} duplicates", f"{len(panel.columns)} rows; 0 duplicates", len(dictionary) == len(panel.columns) and duplicates == 0 and set(dictionary["field"]) == set(panel.columns), "One dictionary row per table field.")

    add(checks, "extension_event_grain", f"{len(coverage)} rows", "2805 unique", len(coverage) == 2805 and coverage["capology_extension_event_id"].is_unique, "One coverage row per canonical extension event.")
    eval_rows = coverage.loc[coverage["in_integrated_839_evaluation_cohort"].astype(bool)]
    add(checks, "evaluation_feature_coverage", f"{int(eval_rows['prior_completed_season_feature_available'].sum())}/{len(eval_rows)}", "795/839", len(eval_rows) == 839 and int(eval_rows["prior_completed_season_feature_available"].sum()) == 795, "Decision-time-safe evaluation coverage is exact.")
    missing_eval = eval_rows.loc[~eval_rows["prior_completed_season_feature_available"].astype(bool)]
    promoted = missing_eval["missing_reason"].eq("prior_season_outside_big_five_top_flight_likely_promoted").sum()
    add(checks, "promoted_boundary_preserved", int(promoted), 44, promoted == len(missing_eval) == 44, "No signing-season value silently fills promoted-club prior-season gaps.")
    reported = coverage_summary.loc[coverage_summary["scope"].eq("integrated_839_evaluation_cohort")].iloc[0]
    add(checks, "coverage_summary_reconciles", f"{int(reported.feature_available)}/{int(reported.events)}", "795/839", int(reported.feature_available) == 795 and int(reported.events) == 839, "Summary reconciles to event table.")

    output_hash_bad = []
    for row in output_manifest.itertuples(index=False):
        path = OUTPUT / row.output_file
        if not path.exists() or sha256(path) != row.sha256:
            output_hash_bad.append(row.output_file)
    add(checks, "output_manifest_hashes_exact", len(output_hash_bad), 0, not output_hash_bad, "All builder outputs remain exact.")
    add(checks, "run_summary_reconciles", f"{summary['canonical_club_seasons']}; {summary['wage_structure_eligible_club_seasons']}", "1074; 666", summary["canonical_club_seasons"] == 1074 and summary["wage_structure_eligible_club_seasons"] == 666 and summary["unresolved_source_rows"] == 0, "Run summary matches canonical table.")

    freeze = pd.read_csv(ROOT / "Data" / "processed" / "data_usage_audit_2026_09_02" / "v1_freeze_check.csv")
    freeze_bad = []
    for row in freeze.itertuples(index=False):
        path = ROOT / row.path
        if not path.exists() or sha256(path) != row.expected_sha256:
            freeze_bad.append(row.path)
    add(checks, "frozen_v1_artifacts_unchanged", len(freeze_bad), 0, not freeze_bad, "All 24 frozen V1 hashes remain exact.")

    result = pd.DataFrame(checks)
    result.to_csv(OUTPUT / "independent_verification.csv", index=False)
    payload = {
        "checks_passed": int(result["passed"].sum()),
        "checks_total": len(result),
        "all_passed": bool(result["passed"].all()),
        "failed_checks": result.loc[~result["passed"], "check"].tolist(),
    }
    (OUTPUT / "independent_verification.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

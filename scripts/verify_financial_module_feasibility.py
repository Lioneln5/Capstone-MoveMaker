"""Independently verify the MoveMaker financial module feasibility audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "Data" / "processed" / "financial_module_feasibility"
MASTER_PATH = ROOT / "Data" / "processed" / "transfermarkt_merged" / "transfermarkt_comprehensive_master.csv"
CANONICAL_PATH = ROOT / "Data" / "processed" / "canonical_integration" / "canonical_transfer_history.csv"
TARGETS_PATH = ROOT / "Data" / "processed" / "compatibility_targets" / "transfer_compatibility_targets.csv"
RETENTION_PATH = ROOT / "Data" / "processed" / "retention_diagnostic" / "retention_targets.csv"
SOURCE_PATHS = [MASTER_PATH, CANONICAL_PATH, TARGETS_PATH, RETENTION_PATH]
EVALUATION_YEARS = [2020, 2021, 2022, 2023]


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").fillna("").str.strip().str.lower().isin({"true", "1", "yes"})


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add(checks: list[dict[str, Any]], check: str, passed: bool, observed: Any, expected: Any, notes: str) -> None:
    checks.append({"check": check, "passed": bool(passed), "observed": observed, "expected": expected, "notes": notes})


def verify() -> None:
    required = [
        "README.md", "run_summary.json", "financial_event_audit.csv",
        "module_feasibility_summary.csv", "module_funnels.csv",
        "valuation_timing_sensitivity.csv", "outcome_class_balance.csv",
        "coverage_by_transfer_year.csv", "module_overlap_matrix.csv",
        "evidence_overlap_summary.csv", "module_subgroup_coverage.csv",
        "build_checks.csv", "source_manifest.csv", "output_manifest.csv",
    ]
    checks: list[dict[str, Any]] = []
    missing = [name for name in required if not (OUTPUT_DIR / name).exists()]
    add(checks, "required_outputs_exist", not missing, ";".join(missing), "", "Every declared runner output exists.")
    if missing:
        raise RuntimeError(f"Missing outputs: {missing}")

    events = pd.read_csv(OUTPUT_DIR / "financial_event_audit.csv", low_memory=False)
    summary = pd.read_csv(OUTPUT_DIR / "module_feasibility_summary.csv", low_memory=False)
    funnels = pd.read_csv(OUTPUT_DIR / "module_funnels.csv", low_memory=False)
    sensitivity = pd.read_csv(OUTPUT_DIR / "valuation_timing_sensitivity.csv", low_memory=False)
    balance = pd.read_csv(OUTPUT_DIR / "outcome_class_balance.csv", low_memory=False)
    years = pd.read_csv(OUTPUT_DIR / "coverage_by_transfer_year.csv", low_memory=False)
    overlap = pd.read_csv(OUTPUT_DIR / "module_overlap_matrix.csv", low_memory=False)
    evidence = pd.read_csv(OUTPUT_DIR / "evidence_overlap_summary.csv", low_memory=False)
    build_checks = pd.read_csv(OUTPUT_DIR / "build_checks.csv")
    sources = pd.read_csv(OUTPUT_DIR / "source_manifest.csv")
    manifest = pd.read_csv(OUTPUT_DIR / "output_manifest.csv")
    run = json.loads((OUTPUT_DIR / "run_summary.json").read_text(encoding="utf-8"))

    add(checks, "event_key_unique", events["transfer_event_id"].is_unique, events["transfer_event_id"].nunique(), len(events), "One row per clean permanent event.")
    add(checks, "permanent_only", set(events["canonical_transfer_type"].dropna()) == {"transfer"}, ";".join(sorted(set(events["canonical_transfer_type"].dropna()))), "transfer", "Loans and returns are excluded.")
    add(checks, "cross_club_only", events["from_club_id"].astype(str).ne(events["to_club_id"].astype(str)).all(), int(events["from_club_id"].astype(str).eq(events["to_club_id"].astype(str)).sum()), 0, "Administrative same-club events are excluded.")
    add(checks, "no_type_conflicts", not as_bool(events["transfer_type_conflict"]).any(), int(as_bool(events["transfer_type_conflict"]).sum()), 0, "Canonical type is unambiguous.")
    add(checks, "no_event_conflicts", not as_bool(events["event_conflict"]).any(), int(as_bool(events["event_conflict"]).sum()), 0, "Canonical event is unambiguous.")

    reliable = as_bool(events["reliable_positive_inbound_fee"])
    recomputed_reliable = (
        events["canonical_transfer_fee_status"].eq("positive_reported")
        & pd.to_numeric(events["canonical_transfer_fee_eur"], errors="coerce").gt(0)
        & ~as_bool(events["transfer_fee_conflict"])
    )
    add(checks, "reliable_inbound_fee_recomputed", reliable.equals(recomputed_reliable), int(reliable.sum()), int(recomputed_reliable.sum()), "Paid-fee eligibility is exact.")
    add(checks, "zero_not_called_free", not events["canonical_transfer_fee_status"].astype(str).str.contains("free", case=False).any(), int(events["canonical_transfer_fee_status"].astype(str).str.contains("free", case=False).sum()), 0, "Literal zero remains unclassified.")

    for horizon in [12, 24]:
        change = (
            pd.to_numeric(events[f"valuation_post{horizon}_nearest_eur"], errors="coerce")
            - pd.to_numeric(events["valuation_pre_eur"], errors="coerce")
        ) / pd.to_numeric(events["valuation_pre_eur"], errors="coerce")
        stored = pd.to_numeric(events[f"market_value_change_{horizon}m_pct"], errors="coerce")
        comparable = stored.notna()
        add(checks, f"value_change_{horizon}m_recomputed", np.allclose(change[comparable], stored[comparable], atol=1e-12), float((change[comparable] - stored[comparable]).abs().max()), 0, "Percentage target recomputes from dated valuations.")
        for tolerance in [90, 180]:
            eligible = as_bool(events[f"market_value_{horizon}m_eligible_{tolerance}d"])
            valid = (
                pd.to_numeric(events["valuation_pre_eur"], errors="coerce").gt(0)
                & pd.to_numeric(events["valuation_pre_staleness_days"], errors="coerce").between(0, 365)
                & pd.to_numeric(events[f"valuation_post{horizon}_nearest_eur"], errors="coerce").gt(0)
                & pd.to_numeric(events[f"valuation_post{horizon}_gap_days"], errors="coerce").abs().le(tolerance)
            )
            add(checks, f"value_{horizon}m_{tolerance}d_rules", not (eligible & ~valid).any(), int((eligible & ~valid).sum()), 0, "Every eligible event satisfies timing/value rules.")

    opp = pd.to_numeric(events["appearance_post24_destination_minutes"], errors="coerce") / (
        90 * pd.to_numeric(events["club_destination_post24_games"], errors="coerce")
    )
    stored_opp = pd.to_numeric(events["destination_opportunity_share_24m"], errors="coerce")
    observed = stored_opp.notna()
    add(checks, "opportunity_recomputed", np.allclose(opp[observed], stored_opp[observed], atol=1e-12), float((opp[observed] - stored_opp[observed]).abs().max()), 0, "Opportunity share uses destination capacity.")
    add(checks, "opportunity_denominator_floor", pd.to_numeric(events.loc[observed, "club_destination_post24_games"], errors="coerce").ge(20).all(), pd.to_numeric(events.loc[observed, "club_destination_post24_games"], errors="coerce").min(), ">=20", "Zero minutes are interpretable.")

    module_flags = [column for column in events if column.startswith("eligible_")]
    expected_modules = set(summary["module_id"])
    observed_modules = {column.removeprefix("eligible_") for column in module_flags}
    add(checks, "module_flags_complete", observed_modules == expected_modules, ";".join(sorted(observed_modules)), ";".join(sorted(expected_modules)), "Each summary module has a row-level flag.")
    summary_map = summary.set_index("module_id")
    for module in sorted(expected_modules):
        count = int(as_bool(events[f"eligible_{module}"]).sum())
        add(checks, f"module_count_{module}", count == int(summary_map.loc[module, "eligible_rows"]), count, int(summary_map.loc[module, "eligible_rows"]), "Summary count matches event flags.")

    dual = as_bool(events["eligible_dual_capital_at_risk_24m"])
    add(checks, "dual_subset_value", not (dual & ~as_bool(events["eligible_market_value_downside_24m"])).any(), int((dual & ~as_bool(events["eligible_market_value_downside_24m"])).sum()), 0, "Dual cohort is nested.")
    add(checks, "dual_subset_opportunity", not (dual & ~as_bool(events["eligible_paid_transfer_underutilization_24m"])).any(), int((dual & ~as_bool(events["eligible_paid_transfer_underutilization_24m"])).sum()), 0, "Dual cohort is nested.")
    add(checks, "full_roi_zero", int(as_bool(events["eligible_full_accounting_roi"]).sum()) == 0, int(as_bool(events["eligible_full_accounting_roi"]).sum()), 0, "The audit never invents accounting ROI.")
    add(checks, "full_roi_decision", summary_map.loc["full_accounting_roi", "recommendation"] == "not_feasible", summary_map.loc["full_accounting_roi", "recommendation"], "not_feasible", "Structural gap is explicit.")
    add(checks, "resale_decision_cautious", summary_map.loc["fee_only_resale_return_4y", "recommendation"] == "exploratory_only", summary_map.loc["fee_only_resale_return_4y", "recommendation"], "exploratory_only", "Conditional resale is not presented as universal ROI.")

    for module in expected_modules:
        flag = as_bool(events[f"eligible_{module}"])
        event_year_counts = events.loc[flag, "transfer_year"].value_counts()
        for year in EVALUATION_YEARS:
            reported = int(summary_map.loc[module, f"rows_{year}"])
            add(checks, f"year_count_{module}_{year}", reported == int(event_year_counts.get(year, 0)), reported, int(event_year_counts.get(year, 0)), "Temporal coverage is exact.")

    strict_sensitivity = sensitivity.loc[
        sensitivity["scope"].eq("modeling_era_2017plus")
        & sensitivity["horizon_months"].eq(24)
        & sensitivity["pre_tolerance_days"].eq(365)
        & sensitivity["post_tolerance_days"].eq(90)
    ].iloc[0]
    value_count = int(as_bool(events["eligible_market_value_downside_24m"]).sum())
    add(checks, "strict_sensitivity_matches_module", int(strict_sensitivity["eligible_rows"]) == value_count, int(strict_sensitivity["eligible_rows"]), value_count, "Timing table and module definition reconcile.")

    diagonal = overlap.loc[overlap["module_left"].eq(overlap["module_right"])]
    diagonal_map = diagonal.set_index("module_left")["intersection_rows"]
    overlap_ok = all(int(diagonal_map[module]) == int(summary_map.loc[module, "eligible_rows"]) for module in expected_modules)
    add(checks, "overlap_diagonal", overlap_ok, overlap_ok, True, "Overlap matrix diagonal equals module size.")

    funnel_last = funnels.sort_values(["module_id", "step"]).groupby("module_id").tail(1).set_index("module_id")
    funnel_modules = ["market_value_downside_24m", "paid_transfer_underutilization_24m", "fee_benchmarking", "dual_capital_at_risk_24m", "fee_only_resale_return_4y", "full_accounting_roi"]
    funnel_ok = all(int(funnel_last.loc[module, "records"]) == int(summary_map.loc[module, "eligible_rows"]) for module in funnel_modules)
    add(checks, "funnel_final_counts", funnel_ok, funnel_ok, True, "Every funnel terminates at its module cohort.")

    year_total = years[[column for column in years if column in expected_modules]].sum()
    year_ok = all(int(year_total[module]) == int(summary_map.loc[module, "eligible_rows"]) for module in year_total.index)
    add(checks, "year_coverage_totals", year_ok, year_ok, True, "Annual counts reconcile to summary.")

    evidence_ok = True
    for _, row in evidence.iterrows():
        evidence_ok &= int(row["module_rows"]) == int(summary_map.loc[row["module_id"], "eligible_rows"])
        evidence_ok &= int(row["overlap_rows"]) <= int(row["module_rows"])
    add(checks, "evidence_overlap_bounds", evidence_ok, evidence_ok, True, "Evidence overlaps never exceed module size.")

    target_source = pd.read_csv(
        TARGETS_PATH,
        usecols=[
            "canonical_transfer_event_id", "eligible_performance_model_primary",
            "target_role_performance_index_available",
        ],
    )
    rich_ids = set(
        target_source.loc[
            as_bool(target_source["eligible_performance_model_primary"])
            & as_bool(target_source["target_role_performance_index_available"]),
            "canonical_transfer_event_id",
        ].astype(str)
    )
    rich_recomputed = events["canonical_transfer_event_id"].astype(str).isin(rich_ids)
    add(checks, "rich_performance_overlap_recomputed", rich_recomputed.equals(as_bool(events["rich_performance_available"])), int(rich_recomputed.sum()), int(as_bool(events["rich_performance_available"]).sum()), "Rich performance joins through canonical event ID, not incompatible legacy event keys.")

    retention_ids = set(pd.read_csv(RETENTION_PATH, usecols=["transfer_event_id"])["transfer_event_id"].astype(str))
    retention_recomputed = events["transfer_event_id"].astype(str).isin(retention_ids)
    add(checks, "retention_overlap_recomputed", retention_recomputed.equals(as_bool(events["retention_target_available"])), int(retention_recomputed.sum()), int(as_bool(events["retention_target_available"]).sum()), "Retention overlap joins through the shared master event ID.")

    balance_ok = (balance["positive_rows"] <= balance["eligible_rows"]).all() and balance["positive_rate"].dropna().between(0, 1).all()
    add(checks, "outcome_balance_bounds", balance_ok, balance_ok, True, "Threshold counts and rates are valid.")
    add(checks, "runner_build_checks_pass", as_bool(build_checks["passed"]).all(), int(as_bool(build_checks["passed"]).sum()), len(build_checks), "All internal invariants passed before output.")

    source_map = sources.set_index("source")
    source_ok = True
    for path in SOURCE_PATHS:
        key = path.relative_to(ROOT).as_posix()
        source_ok &= key in source_map.index
        source_ok &= int(source_map.loc[key, "size_bytes"]) == path.stat().st_size
        source_ok &= source_map.loc[key, "sha256"] == sha256_file(path)
    add(checks, "source_manifest_hashes", source_ok, source_ok, True, "Every source matches the audited run.")

    manifest_ok = True
    for _, row in manifest.iterrows():
        path = OUTPUT_DIR / row["file"]
        manifest_ok &= path.exists()
        manifest_ok &= path.stat().st_size == int(row["size_bytes"])
        manifest_ok &= sha256_file(path) == row["sha256"]
    add(checks, "output_manifest_hashes", manifest_ok, manifest_ok, True, "Every runner output matches its hash.")
    add(checks, "protected_outputs_unchanged", run["protected_outputs_unchanged"] is True, run["protected_outputs_unchanged"], True, "Existing verified model outputs were not modified.")

    result = pd.DataFrame(checks)
    result.to_csv(OUTPUT_DIR / "independent_verification.csv", index=False)
    failures = result.loc[~result["passed"]]
    payload = {
        "status": "pass" if failures.empty else "fail",
        "checks": len(result),
        "passed": int(result["passed"].sum()),
        "failed": len(failures),
        "failed_checks": failures["check"].tolist(),
    }
    (OUTPUT_DIR / "independent_verification.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)
    if not failures.empty:
        raise RuntimeError(f"Verification failed: {failures['check'].tolist()}")


if __name__ == "__main__":
    verify()

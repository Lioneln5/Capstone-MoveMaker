"""Independent verification for Engine V2 Phase 4 candidate evidence."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.engine_v2.result_contract import ENGINE_VERSION, validate_payload  # noqa: E402
from models.engine_v2.reliability_contract import (  # noqa: E402
    MAX_ACCEPTABLE_REFUSAL_RATE, MAX_SUPPORTED_CALIBRATION_GAP,
    MIN_SUBGROUP_EVENTS, MIN_SUBGROUP_NONEVENTS, MIN_SUBGROUP_ORIGINS,
    MIN_SUBGROUP_ROWS, MIN_SUPPORTED_AUC,
)

OUTPUT = ROOT / "Data" / "processed" / "engine_v2_candidate_contract"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"
VARIANTS = {"chronology_prevalence", "phase2_prior_volume", "core_available", "core_plus_age_curve"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    results: list[dict] = []

    def add(name: str, passed: bool, observed: object, expected: object, notes: str) -> None:
        results.append({"check": name, "passed": bool(passed), "observed": observed, "expected": expected, "notes": notes})

    all_cont = pd.read_csv(OUTPUT / "continuity_candidate_predictions.csv")
    comparison = pd.read_csv(OUTPUT / "continuity_candidate_comparison.csv")
    selected = pd.read_csv(OUTPUT / "continuity_selected_predictions.csv")
    cont_subgroup = pd.read_csv(OUTPUT / "continuity_subgroup_reliability.csv")
    wage = pd.read_csv(OUTPUT / "wage_benchmark_predictions.csv")
    wage_metrics = pd.read_csv(OUTPUT / "wage_benchmark_origin_metrics.csv")
    wage_subgroup = pd.read_csv(OUTPUT / "wage_benchmark_subgroup_reliability.csv")
    relationship = pd.read_csv(OUTPUT / "target_relationship_audit.csv")
    decisions = json.loads((OUTPUT / "phase4_decisions.json").read_text(encoding="utf-8"))
    holdout = json.loads((OUTPUT / "final_holdout_plan.json").read_text(encoding="utf-8"))
    examples = json.loads((OUTPUT / "result_contract_examples.json").read_text(encoding="utf-8"))
    sources = pd.read_csv(OUTPUT / "source_manifest.csv")
    summary = json.loads((OUTPUT / "run_summary.json").read_text(encoding="utf-8"))

    add("continuity_variants_exact", set(all_cont.variant) == VARIANTS, sorted(all_cont.variant.unique()), sorted(VARIANTS), "No undeclared repair candidate is present.")
    add("rolling_years_exact", set(selected.evaluation_year) == {2020, 2021, 2022, 2023} and set(wage.evaluation_year) == {2020, 2021, 2022, 2023}, sorted(selected.evaluation_year.unique()), [2020, 2021, 2022, 2023], "Both audits are chronological.")
    cont_key = ["origin_id", "capology_extension_event_id", "variant"]
    wage_key = ["origin_id", "capology_extension_event_id"]
    add("continuity_keys_unique", not all_cont.duplicated(cont_key).any(), int(all_cont.duplicated(cont_key).sum()), 0, "One result per candidate/event/origin.")
    add("wage_keys_unique", not wage.duplicated(wage_key).any(), int(wage.duplicated(wage_key).sum()), 0, "One wage peer estimate per event/origin.")
    add("continuity_probabilities_bounded", all_cont.prediction.between(0, 1).all(), [float(all_cont.prediction.min()), float(all_cont.prediction.max())], "[0,1]", "All probabilities are valid.")
    add("selected_is_age_curve", set(selected.variant) == {"core_plus_age_curve"}, sorted(selected.variant.unique()), ["core_plus_age_curve"], "The repaired observable candidate is selected.")
    expected_selected = all_cont.loc[all_cont.variant.eq("core_plus_age_curve")]
    merged = selected.merge(expected_selected[["origin_id", "capology_extension_event_id", "prediction"]], on=["origin_id", "capology_extension_event_id"], suffixes=("_saved", "_source"), validate="one_to_one")
    selected_diff = float((merged.prediction_saved - merged.prediction_source).abs().max())
    add("selected_probabilities_exact", selected_diff < 1e-12, selected_diff, "<1e-12", "Intervals/support did not alter point estimates.")

    comparison_diffs = []
    for row in comparison.itertuples(index=False):
        chunk = all_cont.loc[all_cont.variant.eq(row.candidate)]
        base = all_cont.loc[all_cont.variant.eq("chronology_prevalence")]
        keys = ["origin_id", "capology_extension_event_id", "canonical_player_id"]
        pair = base[keys + ["brier_loss"]].merge(chunk[keys + ["brier_loss"]], on=keys, suffixes=("_base", "_candidate"), validate="one_to_one")
        improvement = pair.brier_loss_base - pair.brier_loss_candidate
        comparison_diffs.extend([
            abs(row.pooled_brier - pair.brier_loss_candidate.mean()),
            abs(row.pooled_brier_improvement_vs_chronology - improvement.mean()),
            abs(row.pooled_roc_auc - roc_auc_score(chunk.actual, chunk.prediction)),
            abs(row.origin_wins - int(pair.assign(improvement=improvement).groupby("origin_id").improvement.mean().gt(0).sum())),
        ])
    add("continuity_comparisons_recompute", max(comparison_diffs) < 1e-12, max(comparison_diffs), "<1e-12", "Pooled candidate evidence derives from row-level predictions.")
    selected_row = comparison.loc[comparison.candidate.eq("core_plus_age_curve")].iloc[0]
    add("age_curve_ci_positive", selected_row.ci_low_95 > 0 and selected_row.origin_wins == 4, [float(selected_row.ci_low_95), int(selected_row.origin_wins)], "CI > 0 and 4/4", "Repair beats chronology across origins with positive cluster interval.")
    add("continuity_intervals_ordered", (selected.parameter_interval_low_80 <= selected.prediction).all() and (selected.prediction <= selected.parameter_interval_high_80).all(), int(((selected.parameter_interval_low_80 > selected.prediction) | (selected.prediction > selected.parameter_interval_high_80)).sum()), 0, "Point estimate lies inside each model-fit interval.")
    add("continuity_bootstraps_complete", selected.successful_bootstrap_fits.eq(250).all(), int(selected.successful_bootstrap_fits.min()), 250, "All cluster refits completed.")
    add("continuity_refusals_explained", selected.loc[selected.support_status.eq("refused"), "support_reasons"].fillna("").str.len().gt(0).all(), int(selected.support_status.eq("refused").sum()), "all explained", "No refused result is silent.")

    subgroup_diffs, subgroup_status_errors = [], []
    for row in cont_subgroup.itertuples(index=False):
        column = {"position": "canonical_position", "league": "league", "age": "age_band"}[row.group_type]
        chunk = selected.loc[selected[column].astype(str).eq(str(row.group))]
        n, events = len(chunk), int(chunk.actual.sum())
        nonevents, origins = n - events, chunk.origin_id.nunique()
        auc = roc_auc_score(chunk.actual, chunk.prediction) if events and nonevents else np.nan
        brier = chunk.brier_loss.mean()
        baseline = chunk.baseline_brier_loss.mean()
        gap = abs(chunk.actual.mean() - chunk.prediction.mean())
        subgroup_diffs.extend([abs(n-row.rows), abs(events-row.events), abs(nonevents-row.nonevents), abs(origins-row.origins), abs(brier-row.brier), abs(baseline-row.baseline_brier), abs(gap-row.calibration_gap)])
        if not (pd.isna(auc) and pd.isna(row.roc_auc)):
            subgroup_diffs.append(abs(auc-row.roc_auc))
        sufficient = n >= MIN_SUBGROUP_ROWS and events >= MIN_SUBGROUP_EVENTS and nonevents >= MIN_SUBGROUP_NONEVENTS and origins >= MIN_SUBGROUP_ORIGINS
        supported = bool(sufficient and auc >= MIN_SUPPORTED_AUC and gap <= MAX_SUPPORTED_CALIBRATION_GAP and brier < baseline)
        expected = "supported" if supported else "limited_reliability" if sufficient else "insufficient_evidence"
        if expected != row.reliability_status:
            subgroup_status_errors.append((row.group_type, row.group, row.reliability_status, expected))
    add("continuity_subgroup_metrics_recompute", max(subgroup_diffs) < 1e-12, max(subgroup_diffs), "<1e-12", "Subgroup metrics independently reconstruct.")
    add("continuity_subgroup_statuses_recompute", not subgroup_status_errors, subgroup_status_errors, [], "Predeclared subgroup gates reproduce.")
    league_limits = cont_subgroup.loc[(cont_subgroup.group_type == "league") & (cont_subgroup.reliability_status != "supported"), "group"].tolist()
    add("ligue1_limitation_explicit", league_limits == ["Ligue 1"] and decisions["continuity"]["limited_leagues"] == ["Ligue 1"], league_limits, ["Ligue 1"], "Failure is surfaced, not averaged away.")
    add("continuity_deployment_block_recomputes", decisions["continuity"]["phase4_decision"].startswith("blocked_big_five"), decisions["continuity"]["phase4_decision"], "blocked_big_five*", "A failed league gate blocks Big-Five deployment.")

    wage_metric_diffs = []
    for row in wage_metrics.itertuples(index=False):
        chunk = wage.loc[wage.origin_id.eq(row.origin_id)]
        wage_metric_diffs.extend([
            abs(row.log_mae - chunk.absolute_log_error.mean()),
            abs(row.baseline_log_mae - chunk.baseline_absolute_log_error.mean()),
            abs(row.interval_80_coverage - chunk.within_80_interval.mean()),
            abs(row.refused_rate - chunk.support_status.eq("refused").mean()),
        ])
    add("wage_origin_metrics_recompute", max(wage_metric_diffs) < 1e-12, max(wage_metric_diffs), "<1e-12", "Wage metrics derive from row-level evidence.")
    wage_subgroup_diffs, wage_status_errors = [], []
    for row in wage_subgroup.itertuples(index=False):
        column = {"position": "canonical_position", "league": "league", "age": "age_band", "prior_wage": "prior_wage_available"}[row.group_type]
        chunk = wage.loc[wage[column].astype(str).eq(str(row.group))]
        improvement = (chunk.baseline_absolute_log_error - chunk.absolute_log_error).mean()
        coverage = chunk.within_80_interval.mean()
        sufficient = len(chunk) >= MIN_SUBGROUP_ROWS and chunk.origin_id.nunique() >= MIN_SUBGROUP_ORIGINS
        expected = "supported" if sufficient and improvement > 0 and .70 <= coverage <= .90 else "limited_reliability" if sufficient else "insufficient_evidence"
        wage_subgroup_diffs.extend([abs(len(chunk)-row.rows), abs(improvement-row.mae_improvement_vs_chronology), abs(coverage-row.interval_80_coverage)])
        if expected != row.reliability_status:
            wage_status_errors.append((row.group_type, row.group, row.reliability_status, expected))
    add("wage_subgroup_metrics_recompute", max(wage_subgroup_diffs) < 1e-12, max(wage_subgroup_diffs), "<1e-12", "Wage subgroup evidence independently reconstructs.")
    add("wage_subgroup_statuses_recompute", not wage_status_errors, wage_status_errors, [], "Wage reliability gates reproduce.")
    pooled_improvement = (wage.baseline_absolute_log_error - wage.absolute_log_error).mean()
    pooled_coverage = wage.within_80_interval.mean()
    pooled_refusal = wage.support_status.eq("refused").mean()
    add("wage_decision_recomputes", pooled_improvement > 0 and .70 <= pooled_coverage <= .90 and pooled_refusal <= MAX_ACCEPTABLE_REFUSAL_RATE and decisions["wage_benchmark"]["phase4_decision"] == "advance_to_final_holdout", [pooled_improvement, pooled_coverage, pooled_refusal], "positive, [.70,.90], <=.15", "Peer benchmark may advance only to sealed holdout.")
    add("missing_prior_wage_not_silently_required", wage.loc[wage.prior_wage_available.eq(0), "support_reasons"].fillna("").str.contains("log_prior_season").sum() == 0, int(wage.loc[wage.prior_wage_available.eq(0), "support_reasons"].fillna("").str.contains("log_prior_season").sum()), 0, "Availability flag represents genuine missing context without zero-filling or automatic refusal.")

    add("relationship_table_totals", relationship.rows.sum() == 1385, int(relationship.rows.sum()), 1385, "Target overlap cohort reconciles.")
    simultaneous = relationship.loc[(relationship.sustained_meaningful_contribution == 1) & (relationship.any_outbound_24m == 1), "rows"].iloc[0]
    add("targets_not_mutually_exclusive", simultaneous == 87, int(simultaneous), 87, "Observed overlap prohibits probability normalization.")
    contract_errors = []
    for i, payload in enumerate(examples):
        try:
            validate_payload(payload)
        except ValueError as exc:
            contract_errors.append((i, str(exc)))
    add("contract_examples_validate", not contract_errors, contract_errors, [], "Every serialized example satisfies the result contract.")
    invalid = json.loads(json.dumps(examples[0]))
    invalid["modules"][1]["value"] = .5
    rejected = False
    try:
        validate_payload(invalid)
    except ValueError:
        rejected = True
    add("refused_numeric_leak_rejected", rejected, rejected, True, "Contract rejects a number on a refused module.")
    add("no_overall_score_or_recommendation", all(item["overall_score"] is None and item["recommendation"] is None for item in examples), True, True, "No universal output returns.")
    add("engine_version_exact", summary["engine_version"] == ENGINE_VERSION, summary["engine_version"], ENGINE_VERSION, "Version is explicit.")
    add("holdout_remains_sealed", holdout["status"] == "reserved_not_evaluated" and holdout["earliest_eligible_extension_year"] == 2024, holdout["status"], "reserved_not_evaluated", "No later-season tuning occurred.")
    add("overall_engine_not_ready", decisions["overall_engine"]["decision"] == "not_ready_for_deployment", decisions["overall_engine"]["decision"], "not_ready_for_deployment", "Component progress does not authorize the engine.")
    add("no_phase4_artifacts", not list(OUTPUT.rglob("*.joblib")), [str(p) for p in OUTPUT.rglob("*.joblib")], [], "Research evidence is not loadable production state.")

    bad_sources = []
    for row in sources.itertuples(index=False):
        path = ROOT / row.source
        if not path.exists() or sha256(path) != row.sha256:
            bad_sources.append(row.source)
    add("source_manifest_integrity", not bad_sources, bad_sources, [], "Declared inputs remain unchanged.")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    bad_frozen = [item["path"] for item in freeze["files"] if not (ROOT/item["path"]).exists() or sha256(ROOT/item["path"]) != item["sha256"]]
    add("v1_hashes_unchanged", not bad_frozen, bad_frozen, [], "Every frozen V1 file remains byte-identical.")
    add("runner_checks_reconcile", summary["checks_passed"] == summary["checks_total"] == 13 and summary["all_checks_passed"], summary, "13/13", "Runner checks and summary agree.")

    checks = pd.DataFrame(results)
    checks.to_csv(OUTPUT / "independent_verification.csv", index=False, lineterminator="\n")
    payload = {
        "checks_passed": int(checks.passed.sum()), "checks_total": len(checks),
        "all_checks_passed": bool(checks.passed.all()),
        "verified_scope": "phase4_continuity_repair_wage_benchmark_result_contract_holdout_and_v1_isolation",
    }
    (OUTPUT / "independent_verification.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_rows = []
    for path in sorted(OUTPUT.iterdir()):
        if path.is_file() and path.name != "output_manifest.csv":
            manifest_rows.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest_rows).to_csv(OUTPUT / "output_manifest.csv", index=False, lineterminator="\n")
    print(json.dumps(payload, indent=2))
    if not payload["all_checks_passed"]:
        raise SystemExit(f"Failed checks: {checks.loc[~checks.passed, 'check'].tolist()}")


if __name__ == "__main__":
    main()

"""Independently verify the final MoveMaker case-study metric freeze."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "Data" / "processed"
OUTPUT = PROCESSED / "case_study_metric_freeze"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_valid(path: Path, file_column: str, base: Path) -> bool:
    manifest = pd.read_csv(path)
    return all(
        (base / getattr(row, file_column)).is_file()
        and (base / getattr(row, file_column)).stat().st_size == row.bytes
        and sha256(base / getattr(row, file_column)) == row.sha256
        for row in manifest.itertuples(index=False)
    )


def clean_rerun_matches() -> bool:
    with tempfile.TemporaryDirectory(prefix=".case_study_freeze_verify_", dir=PROCESSED) as temp:
        rerun = Path(temp) / "freeze"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "models" / "run_case_study_metric_freeze.py"),
                "--output",
                str(rerun),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        expected = [
            "README.md", "build_checks.csv", "freeze_contract.json",
            "metric_dispositions.csv", "output_manifest.csv",
            "run_summary.json", "source_manifest.csv",
        ]
        return all(sha256(OUTPUT / name) == sha256(rerun / name) for name in expected)


def main() -> None:
    dispositions = pd.read_csv(OUTPUT / "metric_dispositions.csv").set_index("output_id")
    summary = json.loads((OUTPUT / "run_summary.json").read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []

    def add(check: str, passed: bool, observed: object, expected: object) -> None:
        rows.append({
            "check": check,
            "passed": bool(passed),
            "observed": json.dumps(observed, sort_keys=True) if isinstance(observed, (dict, list)) else observed,
            "expected": json.dumps(expected, sort_keys=True) if isinstance(expected, (dict, list)) else expected,
        })

    add(
        "output_manifest",
        manifest_valid(OUTPUT / "output_manifest.csv", "output_file", OUTPUT),
        True,
        True,
    )
    add(
        "source_manifest",
        manifest_valid(OUTPUT / "source_manifest.csv", "source", ROOT),
        True,
        True,
    )
    expected_ids = {
        "contract_arithmetic", "club_salary_structure_context",
        "sustained_realized_contribution_24m", "meaningful_retention_24m",
        "temporary_displacement_risk_24m", "permanent_separation_risk_24m",
        "generic_continuity_24m", "downside_25pct_12m", "downside_25pct_24m",
        "annual_wage_peer_benchmark", "contract_duration_peer_benchmark",
        "fixed_wage_commitment_peer_benchmark", "wage_to_market_value_peer_benchmark",
        "integrated_contract_score", "historical_risk_weighted_screening",
    }
    add("complete_output_set", set(dispositions.index) == expected_ids, sorted(dispositions.index), sorted(expected_ids))
    add("unique_output_set", dispositions.index.is_unique, not dispositions.index.is_unique, False)
    add("summary_count", summary["output_families"] == len(expected_ids), summary["output_families"], len(expected_ids))
    add("no_live_probabilities", not dispositions.live_predictive_probability_allowed.any(), int(dispositions.live_predictive_probability_allowed.sum()), 0)
    add("no_automated_decisions", not dispositions.automated_contract_decision_allowed.any(), int(dispositions.automated_contract_decision_allowed.sum()), 0)
    allowed_descriptive = {"contract_arithmetic", "club_salary_structure_context"}
    observed_descriptive = set(dispositions.loc[dispositions.descriptive_numeric_allowed].index)
    add("numeric_display_is_descriptive_only", observed_descriptive == allowed_descriptive, sorted(observed_descriptive), sorted(allowed_descriptive))

    contribution = pd.read_csv(
        PROCESSED / "engine_v2_contribution_reliability_audit" / "reliability_decision.csv"
    ).iloc[0]
    contribution_groups = pd.read_csv(
        PROCESSED / "engine_v2_contribution_reliability_audit" / "subgroup_reliability.csv"
    )
    core = contribution_groups.loc[contribution_groups.group_type.isin(["league", "position"])]
    published_contribution = dispositions.loc["sustained_realized_contribution_24m"]
    add("contribution_auc", np.isclose(published_contribution.pooled_auc, contribution.pooled_roc_auc, atol=1e-14), published_contribution.pooled_auc, contribution.pooled_roc_auc)
    add("contribution_brier", np.isclose(published_contribution.pooled_brier, contribution.pooled_brier, atol=1e-14), published_contribution.pooled_brier, contribution.pooled_brier)
    add("contribution_calibration", np.isclose(published_contribution.calibration_gap, contribution.pooled_calibration_gap, atol=1e-14), published_contribution.calibration_gap, contribution.pooled_calibration_gap)
    supported = int(core.reliability_status.eq("supported").sum())
    add("contribution_subgroups", int(published_contribution.supported_subgroups) == supported == 6, int(published_contribution.supported_subgroups), 6)
    limited = set(core.loc[~core.reliability_status.eq("supported"), "group"])
    add("contribution_limitations_named", limited == {"La Liga", "Serie A", "Midfield"}, sorted(limited), ["La Liga", "Midfield", "Serie A"])
    add("contribution_not_live", published_contribution.final_disposition == "retain_qualified_research_signal", published_contribution.final_disposition, "retain_qualified_research_signal")

    movement_dir = PROCESSED / "engine_v2_movement_scope_closeout"
    movement = pd.read_csv(movement_dir / "product_scope_decisions.csv").set_index("output_id")
    movement_evidence = pd.read_csv(movement_dir / "development_evidence.csv").set_index("output_id")
    add("generic_continuity_retired", dispositions.loc["generic_continuity_24m", "disposition_class"] == "retired" and movement.loc["generic_continuity_24m", "status"] == "retired", dispositions.loc["generic_continuity_24m", "final_disposition"], "retired")
    add("meaningful_retention_superseded", dispositions.loc["meaningful_retention_24m", "final_disposition"] == "definition_retained_current_model_not_finalized" and pd.isna(dispositions.loc["meaningful_retention_24m", "pooled_auc"]), dispositions.loc["meaningful_retention_24m", "final_disposition"], "definition retained; no current metric")
    for output_id in ["temporary_displacement_risk_24m", "permanent_separation_risk_24m"]:
        published = dispositions.loc[output_id]
        raw = movement_evidence.loc[output_id]
        add(f"{output_id}_auc", np.isclose(published.pooled_auc, raw.pooled_auc, atol=1e-14), published.pooled_auc, raw.pooled_auc)
        add(f"{output_id}_skill", np.isclose(published.brier_skill, raw.brier_skill, atol=1e-14), published.brier_skill, raw.brier_skill)

    value_dir = PROCESSED / "engine_v2_public_value_downside_repair"
    value = pd.read_csv(value_dir / "selection_decision.csv").set_index("endpoint")
    exposure = pd.read_csv(value_dir / "known_exposure_audit.csv").set_index("endpoint")
    for endpoint in ["downside_25pct_12m", "downside_25pct_24m"]:
        published = dispositions.loc[endpoint]
        raw = value.loc[endpoint]
        add(f"{endpoint}_auc", np.isclose(published.pooled_auc, raw.pooled_roc_auc, atol=1e-14), published.pooled_auc, raw.pooled_roc_auc)
        add(f"{endpoint}_brier", np.isclose(published.pooled_brier, raw.pooled_brier, atol=1e-14), published.pooled_brier, raw.pooled_brier)
        add(f"{endpoint}_subgroups", int(published.supported_subgroups) == int(raw.supported_broad_subgroups), int(published.supported_subgroups), int(raw.supported_broad_subgroups))
        add(f"{endpoint}_exposed_not_final", not bool(exposure.loc[endpoint, "permitted_for_v2_selection_or_promotion"]), bool(exposure.loc[endpoint, "permitted_for_v2_selection_or_promotion"]), False)
    add("value_horizon_policy", dispositions.loc["downside_25pct_12m", "final_disposition"].startswith("retain_primary") and dispositions.loc["downside_25pct_24m", "final_disposition"].startswith("freeze_secondary"), [dispositions.loc["downside_25pct_12m", "final_disposition"], dispositions.loc["downside_25pct_24m", "final_disposition"]], ["primary qualified", "secondary frozen"])

    phase5 = PROCESSED / "engine_v2_final_evaluation_gate"
    wage = pd.read_csv(phase5 / "wage_final_holdout_metrics.csv").set_index("scope").loc["all_player_disjoint"]
    wage_groups = pd.read_csv(phase5 / "wage_final_holdout_subgroups.csv")
    ligue1 = wage_groups.loc[wage_groups.group_type.eq("league") & wage_groups.group.eq("Ligue 1")].iloc[0]
    published_wage = dispositions.loc["annual_wage_peer_benchmark"]
    add("wage_final_rows", int(published_wage.final_test_rows) == int(wage.rows) == 235, int(published_wage.final_test_rows), 235)
    add("wage_pooled_coverage", np.isclose(published_wage.final_interval_80_coverage, wage.interval_80_coverage, atol=1e-14), published_wage.final_interval_80_coverage, wage.interval_80_coverage)
    add("wage_ligue1_failure", ligue1.final_gate_status == "failed" and published_wage.disposition_class == "rejected", {"status": ligue1.final_gate_status, "disposition": published_wage.disposition_class}, {"status": "failed", "disposition": "rejected"})
    historical_finance = {"contract_duration_peer_benchmark", "fixed_wage_commitment_peer_benchmark", "wage_to_market_value_peer_benchmark"}
    add("other_peer_benchmarks_historical_only", set(dispositions.loc[dispositions.disposition_class.eq("historical_only")].index) == historical_finance, sorted(dispositions.loc[dispositions.disposition_class.eq("historical_only")].index), sorted(historical_finance))

    integrated = pd.read_csv(
        PROCESSED / "integrated_contract_profile" / "integration_score_evaluation.csv"
    ).set_index("score")
    composite = integrated.loc["equal_weight_three_module", "pooled_auc"]
    component = integrated.loc["opportunity_component", "pooled_auc"]
    add("integrated_score_no_improvement", composite < component and dispositions.loc["integrated_contract_score", "disposition_class"] == "rejected", {"composite_auc": composite, "component_auc": component}, "composite lower and rejected")

    predictions = pd.read_csv(
        PROCESSED / "business_metric_translation_audit" / "low_contribution_exposure_predictions.csv"
    )
    capacity = int(dispositions.loc["historical_risk_weighted_screening", "review_capacity"])
    selected_count = int(predictions.nlargest(capacity, "selected_exposure_eur").actual_low_contribution.sum())
    baseline_count = int(predictions.nlargest(capacity, "fixed_wages_through_24m_eur").actual_low_contribution.sum())
    published_screen = dispositions.loc["historical_risk_weighted_screening"]
    add("screening_counts", selected_count == int(published_screen.historical_outcomes_identified) == 79 and baseline_count == int(published_screen.baseline_outcomes_identified) == 54, {"selected": selected_count, "baseline": baseline_count}, {"selected": 79, "baseline": 54})
    lift = selected_count / baseline_count - 1
    add("screening_lift", np.isclose(published_screen.relative_lift, lift, atol=1e-14), published_screen.relative_lift, lift)
    add("screening_qualified", "not causal savings" in published_screen.limitation.lower(), published_screen.limitation, "contains not causal savings")

    verification_paths = [
        PROCESSED / "engine_v2_contribution_reliability_audit" / "independent_verification.json",
        movement_dir / "independent_verification.json",
        value_dir / "independent_verification.json",
        phase5 / "independent_verification.json",
        PROCESSED / "contract_financial_exposure_benchmark" / "independent_verification.json",
        PROCESSED / "business_metric_translation_audit" / "independent_verification.json",
        PROCESSED / "integrated_contract_profile" / "independent_verification.json",
    ]
    upstream = {}
    for path in verification_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        upstream[str(path.relative_to(ROOT))] = bool(
            payload.get("all_checks_passed", payload.get("all_passed", False))
        )
    add("upstream_verifications", all(upstream.values()), upstream, {key: True for key in upstream})

    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    changed = [item["path"] for item in frozen["files"] if sha256(ROOT / item["path"]) != item["sha256"]]
    changed_models = [path for path in changed if path.endswith(".joblib")]
    expected_report_changes = {
        "Data/processed/deployment_models/live_request_integrity.csv",
        "Data/processed/deployment_models/live_request_integrity.json",
    }
    add("v1_serialized_models_byte_identical", not changed_models, changed_models, [])
    add("v1_freeze_manifest_exceptions_disclosed", set(changed) == expected_report_changes, changed, sorted(expected_report_changes))
    api_text = (ROOT / "api" / "main.py").read_text(encoding="utf-8")
    scoring_disabled = "SCORING_ENABLED: Final[bool] = False" in api_text
    add("scoring_remains_disabled", scoring_disabled, scoring_disabled, True)
    add("clean_rerun", clean_rerun_matches(), True, True)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "independent_verification.csv", index=False)
    result = {
        "freeze_version": summary["freeze_version"],
        "checks_passed": int(frame.passed.sum()),
        "checks_total": int(len(frame)),
        "all_checks_passed": bool(frame.passed.all()),
        "model_artifacts_created": 0,
        "deployment_changed": False,
    }
    (OUTPUT / "independent_verification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not frame.passed.all():
        raise RuntimeError(frame.loc[~frame.passed].to_dict("records"))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

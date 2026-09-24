"""Build the authoritative MoveMaker case-study metric disposition freeze.

This is a publication/closeout artifact, not a model-training runner.  It reads
the latest independently verified research outputs, resolves superseded claims,
and writes one compact table describing what may be retained, qualified,
deferred, retired, or rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "Data" / "processed"
DEFAULT_OUTPUT = PROCESSED / "case_study_metric_freeze"
FREEZE_VERSION = "movemaker_case_study_metric_freeze_v1_2026-09-10"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verified(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return bool(payload.get("all_checks_passed", payload.get("all_passed", False)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")

    contribution_dir = PROCESSED / "engine_v2_contribution_reliability_audit"
    movement_dir = PROCESSED / "engine_v2_movement_scope_closeout"
    value_dir = PROCESSED / "engine_v2_public_value_downside_repair"
    phase5_dir = PROCESSED / "engine_v2_final_evaluation_gate"
    finance_dir = PROCESSED / "contract_financial_exposure_benchmark"
    business_dir = PROCESSED / "business_metric_translation_audit"
    integration_dir = PROCESSED / "integrated_contract_profile"

    contribution = pd.read_csv(contribution_dir / "reliability_decision.csv").iloc[0]
    contribution_groups = pd.read_csv(contribution_dir / "subgroup_reliability.csv")
    movement = pd.read_csv(movement_dir / "product_scope_decisions.csv").set_index("output_id")
    movement_evidence = pd.read_csv(movement_dir / "development_evidence.csv").set_index("output_id")
    value = pd.read_csv(value_dir / "selection_decision.csv").set_index("endpoint")
    value_exposure = pd.read_csv(value_dir / "known_exposure_audit.csv").set_index("endpoint")
    wage = pd.read_csv(phase5_dir / "wage_final_holdout_metrics.csv").set_index("scope")
    wage_groups = pd.read_csv(phase5_dir / "wage_final_holdout_subgroups.csv")
    legacy_finance = pd.read_csv(finance_dir / "benchmark_decision_summary.csv").set_index("endpoint")
    deterministic = pd.read_csv(finance_dir / "deterministic_metric_catalog.csv")
    exposure = pd.read_csv(business_dir / "low_contribution_exposure_performance.csv").set_index("evaluation")
    exposure_comparison = pd.read_csv(business_dir / "low_contribution_exposure_comparison.csv")
    exposure_predictions = pd.read_csv(business_dir / "low_contribution_exposure_predictions.csv")
    integrated = pd.read_csv(integration_dir / "integration_score_evaluation.csv").set_index("score")
    contribution_sensitivity = pd.read_csv(
        PROCESSED / "engine_v2_contribution_model_rerun" / "player_disjoint_sensitivity.csv"
    ).set_index(["protocol", "candidate"])
    value_candidate_metrics = pd.read_csv(value_dir / "candidate_metrics.csv")

    core_groups = contribution_groups.loc[
        contribution_groups["group_type"].isin(["league", "position"])
    ]
    supported_core = int(core_groups["reliability_status"].eq("supported").sum())
    limited_core = ", ".join(
        core_groups.loc[
            ~core_groups["reliability_status"].eq("supported"), "group"
        ].astype(str)
    )
    ligue1 = wage_groups.loc[
        wage_groups["group_type"].eq("league") & wage_groups["group"].eq("Ligue 1")
    ].iloc[0]
    # Supported deterministic formulas include exact calculations and clearly
    # labelled proxies, but not sparse bonus data.
    exact_supported = deterministic.loc[
        deterministic["metric_class"].eq("deterministic_formula")
        & deterministic["status"].isin(["supported", "supported_proxy"])
    ]
    club_context = deterministic.loc[
        deterministic["metric_class"].eq("historical_observed_context")
    ]
    contribution_baseline = contribution_sensitivity.loc[
        ("player_disjoint", "B0_chronology_prevalence")
    ]
    contribution_selected = contribution_sensitivity.loc[
        ("player_disjoint", "B2_core_recent_involvement")
    ]

    def blank_metrics() -> dict[str, object]:
        return {
            "evaluation_rows": None,
            "pooled_auc": None,
            "pooled_brier": None,
            "brier_skill": None,
            "calibration_gap": None,
            "origin_wins": None,
            "supported_subgroups": None,
            "total_subgroups": None,
            "final_test_rows": None,
            "final_interval_80_coverage": None,
            "reference_rows": None,
            "review_capacity": None,
            "historical_outcomes_identified": None,
            "baseline_outcomes_identified": None,
            "relative_lift": None,
        }

    rows: list[dict[str, object]] = []

    def add(
        output_id: str,
        public_label: str,
        output_type: str,
        final_disposition: str,
        disposition_class: str,
        exact_question: str,
        evidence_summary: str,
        limitation: str,
        case_study_role: str,
        authoritative_source: str,
        *,
        descriptive_numeric_allowed: bool = False,
        reopen_condition: str = "none; frozen conclusion",
        metrics: dict[str, object] | None = None,
    ) -> None:
        base = {
            "freeze_version": FREEZE_VERSION,
            "output_id": output_id,
            "public_label": public_label,
            "output_type": output_type,
            "final_disposition": final_disposition,
            "disposition_class": disposition_class,
            "exact_question": exact_question,
            "evidence_summary": evidence_summary,
            "limitation": limitation,
            "case_study_role": case_study_role,
            "descriptive_numeric_allowed": descriptive_numeric_allowed,
            "live_predictive_probability_allowed": False,
            "automated_contract_decision_allowed": False,
            "reopen_condition": reopen_condition,
            "authoritative_source": authoritative_source,
        }
        base.update(blank_metrics())
        if metrics:
            base.update(metrics)
        rows.append(base)

    add(
        "contract_arithmetic",
        "Contract facts and commitment arithmetic",
        "deterministic_context",
        "retain_descriptive_supported",
        "retained",
        "What does the proposed wage and term mechanically commit the club to?",
        f"{len(exact_supported)} deterministic calculations or explicitly labelled proxies are supported.",
        "Fixed wages exclude taxes, agent fees, bonuses, clauses, options and contingent costs unless entered.",
        "A defensible non-predictive layer showing that useful decision support does not require a model.",
        "Data/processed/contract_financial_exposure_benchmark/deterministic_metric_catalog.csv",
        descriptive_numeric_allowed=True,
        metrics={"reference_rows": int(len(exact_supported))},
    )
    add(
        "club_salary_structure_context",
        "Observed club salary context",
        "observed_context",
        "retain_descriptive_with_coverage_warning",
        "retained",
        "Where does a proposed wage sit within the club's known public salary panel?",
        f"{len(club_context)} observed-context outputs are supported when the club/player panel matches.",
        "The known salary panel is incomplete and is not an audited club payroll.",
        "A contextual descriptive output, not a prediction or affordability judgment.",
        "Data/processed/contract_financial_exposure_benchmark/deterministic_metric_catalog.csv",
        descriptive_numeric_allowed=True,
        metrics={"reference_rows": int(len(club_context))},
    )
    add(
        "sustained_realized_contribution_24m",
        "Sustained realized contribution",
        "predictive_research",
        "retain_qualified_research_signal",
        "qualified",
        "Will the player record at least 25% same-club opportunity share in both Year 1 and Year 2?",
        (
            f"The repaired compact candidate achieved AUC {contribution.pooled_roc_auc:.3f}, "
            f"Brier {contribution.pooled_brier:.4f}, calibration gap "
            f"{contribution.pooled_calibration_gap:.3f}, and positive skill in "
            f"{int(contribution.positive_skill_origins)}/4 origins."
        ),
        f"Only {supported_core}/9 core league/position groups passed the 99% skill gate; limited groups: {limited_core}.",
        "Primary contribution case-study model, reported as rolling development evidence rather than a universal probability.",
        "Data/processed/engine_v2_contribution_reliability_audit/reliability_decision.csv",
        reopen_condition="New specification or data must clear the frozen all-group reliability contract, followed by untouched later evaluation.",
        metrics={
            "evaluation_rows": int(contribution_selected.evaluation_rows),
            "pooled_auc": float(contribution.pooled_roc_auc),
            "pooled_brier": float(contribution.pooled_brier),
            "brier_skill": float(contribution_baseline.mean_origin_brier - contribution.pooled_brier),
            "calibration_gap": float(contribution.pooled_calibration_gap),
            "origin_wins": int(contribution.positive_skill_origins),
            "supported_subgroups": supported_core,
            "total_subgroups": 9,
        },
    )
    add(
        "meaningful_retention_24m",
        "Meaningful Retention",
        "defined_outcome",
        "definition_retained_current_model_not_finalized",
        "deferred",
        movement.loc["meaningful_retention_24m", "definition"],
        "The football question is retained, but its earlier movement-model evidence predates the repaired contribution target.",
        "The joint endpoint must be rebuilt from the repaired contribution contract before any current probability can be claimed.",
        "A defensible target definition and explicit example of why a promising result can be superseded by target repair.",
        "Data/processed/engine_v2_movement_scope_closeout/product_scope_decisions.csv",
        reopen_condition="Reconstruct the joint target from repaired contribution evidence, rerun development gates, then obtain untouched later evaluation.",
    )
    temp = movement_evidence.loc["temporary_displacement_risk_24m"]
    add(
        "temporary_displacement_risk_24m",
        "Temporary Displacement Risk",
        "predictive_research",
        "retain_development_candidate",
        "qualified",
        movement.loc["temporary_displacement_risk_24m", "definition"],
        f"Development AUC {temp.pooled_auc:.3f}, Brier skill +{temp.brier_skill:.4f}, 4/4 origin wins.",
        "No sufficiently mature, player-disjoint, untouched final temporal cohort has authorized promotion.",
        "Secondary movement case-study model; loans are treated separately from permanent departure.",
        "Data/processed/engine_v2_movement_scope_closeout/development_evidence.csv",
        reopen_condition="Evaluate the frozen question on a sufficiently mature untouched later cohort.",
        metrics={
            "evaluation_rows": int(temp.evaluation_rows),
            "pooled_auc": float(temp.pooled_auc),
            "pooled_brier": float(temp.pooled_brier),
            "brier_skill": float(temp.brier_skill),
            "calibration_gap": float(temp.pooled_ece),
            "origin_wins": int(temp.origin_wins),
        },
    )
    permanent = movement_evidence.loc["permanent_separation_risk_24m"]
    add(
        "permanent_separation_risk_24m",
        "Permanent Separation",
        "research_question",
        "research_only_unavailable",
        "deferred",
        movement.loc["permanent_separation_risk_24m", "definition"],
        f"The diagnostic found weak pooled discrimination (AUC {permanent.pooled_auc:.3f}) and unreliable subgroup behavior.",
        "Current public-data features do not justify a personalized probability.",
        "Unavailable output documenting a real limit of the available evidence.",
        "Data/processed/engine_v2_movement_scope_closeout/development_evidence.csv",
        reopen_condition="A materially new decision-time-safe mechanism must show stable cross-league support.",
        metrics={
            "evaluation_rows": int(permanent.evaluation_rows),
            "pooled_auc": float(permanent.pooled_auc),
            "pooled_brier": float(permanent.pooled_brier),
            "brier_skill": float(permanent.brier_skill),
            "origin_wins": int(permanent.origin_wins),
        },
    )
    generic = movement_evidence.loc["generic_continuity_24m"]
    add(
        "generic_continuity_24m",
        "Generic club continuity",
        "retired_predictive_question",
        "retired_mixed_movement_mechanisms",
        "retired",
        movement.loc["generic_continuity_24m", "definition"],
        f"The old endpoint achieved only AUC {generic.pooled_auc:.3f} and combined football mechanisms with different meanings.",
        "Loans, loan returns and permanent exits cannot be interpreted as one coherent contract-stability risk.",
        "Central negative finding: endpoint validity matters before model accuracy.",
        "Data/processed/engine_v2_movement_scope_closeout/product_scope_decisions.csv",
        metrics={"evaluation_rows": int(generic.evaluation_rows), "pooled_auc": float(generic.pooled_auc)},
    )

    for endpoint, label, disposition, disposition_class, role, limitation, reopen in [
        (
            "downside_25pct_12m",
            "At least 25% public-value downside within 12 months",
            "retain_primary_qualified_research_candidate",
            "qualified",
            "Primary public-value case-study horizon.",
            "Goalkeeper skill is uncertain under the 99% subgroup gate, and available 2024+ outcomes were previously exposed.",
            "Repair goalkeeper reliability and evaluate only on genuinely untouched later evidence.",
        ),
        (
            "downside_25pct_24m",
            "At least 25% public-value downside within 24 months",
            "freeze_secondary_development_candidate",
            "qualified",
            "Secondary/provisional horizon demonstrating stronger development evidence but slower outcome maturity.",
            "The 2024 player-disjoint cohort has only 26 rows and was previously exposed; smallest league n=3.",
            "Wait for a sufficiently large, mature and genuinely untouched later cohort.",
        ),
    ]:
        item = value.loc[endpoint]
        exposed = value_exposure.loc[endpoint]
        selected_origins = value_candidate_metrics.loc[
            value_candidate_metrics["endpoint"].eq(endpoint)
            & value_candidate_metrics["candidate"].eq(item.selected_candidate)
        ]
        add(
            endpoint,
            label,
            "predictive_research",
            disposition,
            disposition_class,
            f"Will public market value decline by at least 25% within {'12' if endpoint.endswith('12m') else '24'} months?",
            (
                f"Proposal-free rolling evidence: AUC {item.pooled_roc_auc:.3f}, Brier "
                f"{item.pooled_brier:.4f}, calibration gap {item.pooled_calibration_gap:.3f}, "
                f"positive skill in {int(item.positive_skill_origins)}/4 origins, "
                f"{int(item.supported_broad_subgroups)}/{int(item.total_broad_subgroups)} broad groups supported."
            ),
            limitation,
            role,
            "Data/processed/engine_v2_public_value_downside_repair/selection_decision.csv",
            reopen_condition=reopen,
            metrics={
                "evaluation_rows": int(selected_origins["rows"].sum()),
                "pooled_auc": float(item.pooled_roc_auc),
                "pooled_brier": float(item.pooled_brier),
                "brier_skill": float(item.chronology_brier_improvement),
                "calibration_gap": float(item.pooled_calibration_gap),
                "origin_wins": int(item.positive_skill_origins),
                "supported_subgroups": int(item.supported_broad_subgroups),
                "total_subgroups": int(item.total_broad_subgroups),
                "final_test_rows": int(exposed.exposed_2024_player_disjoint_rows),
            },
        )

    all_wage = wage.loc["all_player_disjoint"]
    add(
        "annual_wage_peer_benchmark",
        "Historical annual-wage peer benchmark",
        "historical_predictive_benchmark",
        "reject_current_candidate_after_final_holdout",
        "rejected",
        "What annual fixed wage did broadly comparable historical extensions receive?",
        f"On the 235-row player-disjoint 2024 final cohort, log MAE was {all_wage.log_mae:.3f} and pooled 80% range coverage was {all_wage.interval_80_coverage:.1%}.",
        f"Ligue 1 range coverage was only {ligue1.interval_80_coverage:.1%}; the declared Big-Five gate failed.",
        "A failed generalization result; the current candidate must not be serialized or described as a validated Big-Five benchmark.",
        "Data/processed/engine_v2_final_evaluation_gate/wage_final_holdout_metrics.csv",
        reopen_condition="A new version must be specified without fitting to the failed 2024 labels and tested on later untouched evidence.",
        metrics={
            "final_test_rows": int(all_wage.rows),
            "final_interval_80_coverage": float(all_wage.interval_80_coverage),
            "brier_skill": float(all_wage.mae_improvement_vs_chronology),
        },
    )
    for endpoint, label in [
        ("contract_duration", "Historical contract-duration peer benchmark"),
        ("fixed_wage_commitment", "Historical fixed-wage-commitment peer benchmark"),
        ("wage_to_market_value", "Historical wage-to-public-value peer benchmark"),
    ]:
        item = legacy_finance.loc[endpoint]
        add(
            f"{endpoint}_peer_benchmark",
            label,
            "historical_predictive_benchmark",
            "retain_frozen_v1_historical_evidence_only",
            "historical_only",
            item.recommended_display,
            (
                f"V1 rolling evidence recorded {item.total_origin_wins}/4 origin wins and "
                f"{item.empirical_80_interval_coverage:.1%} empirical 80% range coverage."
            ),
            "These candidates did not receive a successful Engine V2 final evaluation and are not current live benchmarks.",
            "Historical evidence only; useful for explaining the research path, not for current personalized output.",
            "Data/processed/contract_financial_exposure_benchmark/benchmark_decision_summary.csv",
            reopen_condition="Specify a V2 feature contract and evaluate it on a valid untouched final cohort.",
            metrics={
                "final_interval_80_coverage": float(item.empirical_80_interval_coverage),
                "origin_wins": int(item.total_origin_wins),
            },
        )

    opportunity_auc = float(integrated.loc["opportunity_component", "pooled_auc"])
    composite_auc = float(integrated.loc["equal_weight_three_module", "pooled_auc"])
    add(
        "integrated_contract_score",
        "Integrated contract score",
        "composite_score",
        "rejected_no_incremental_value_and_obscures_disagreement",
        "rejected",
        "Should separate risks be compressed into one headline score?",
        f"The equal-weight three-module score achieved AUC {composite_auc:.3f} versus {opportunity_auc:.3f} for the opportunity component alone.",
        "The composite did not improve discrimination and erased decision-relevant disagreement between domains.",
        "Rejected design hypothesis; all retained signals remain separate.",
        "Data/processed/integrated_contract_profile/integration_score_evaluation.csv",
        metrics={"evaluation_rows": int(integrated.loc["equal_weight_three_module", "evaluation_rows"]), "pooled_auc": composite_auc},
    )
    pooled_exposure = exposure.loc["pooled"]
    selected_screen = exposure_comparison.loc[
        exposure_comparison["model"].eq("selected_phase1_model")
    ].iloc[0]
    review_capacity = int(selected_screen.top_quintile_rows)
    selected_review = exposure_predictions.nlargest(review_capacity, "selected_exposure_eur")
    baseline_review = exposure_predictions.nlargest(review_capacity, "fixed_wages_through_24m_eur")
    selected_outcomes = int(selected_review["actual_low_contribution"].sum())
    baseline_outcomes = int(baseline_review["actual_low_contribution"].sum())
    relative_lift = selected_outcomes / baseline_outcomes - 1
    add(
        "historical_risk_weighted_screening",
        "Historical risk-weighted review screening",
        "retrospective_decision_analysis",
        "retain_retrospective_case_study_result",
        "retained",
        "At a fixed review capacity, did risk-weighted prioritization find more later low-contribution outcomes than reviewing the largest contracts?",
        f"Across {int(pooled_exposure.rows)} historical extensions and €{pooled_exposure.total_24m_fixed_wage_commitment_eur / 1e9:.2f}B in two-year fixed wages, 182 risk-ranked reviews identified 79 outcomes versus 54 for largest-contract review ({relative_lift:.0%} more).",
        "This is retrospective prioritization evidence, not causal savings, confirmed loss, or a deployable current screening policy.",
        "Headline practical case-study result, explicitly tied to historical outcomes and equal review capacity.",
        "Data/processed/business_metric_translation_audit/low_contribution_exposure_performance.csv",
        metrics={
            "evaluation_rows": int(pooled_exposure.rows),
            "reference_rows": int(pooled_exposure.players),
            "review_capacity": review_capacity,
            "historical_outcomes_identified": selected_outcomes,
            "baseline_outcomes_identified": baseline_outcomes,
            "relative_lift": float(relative_lift),
        },
    )

    dispositions = pd.DataFrame(rows)
    dispositions.to_csv(output / "metric_dispositions.csv", index=False)

    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, observed: object, expected: object, note: str) -> None:
        checks.append({
            "check": name,
            "passed": bool(passed),
            "observed": json.dumps(observed, sort_keys=True) if isinstance(observed, (dict, list)) else observed,
            "expected": json.dumps(expected, sort_keys=True) if isinstance(expected, (dict, list)) else expected,
            "note": note,
        })

    expected_ids = {
        "contract_arithmetic", "club_salary_structure_context",
        "sustained_realized_contribution_24m", "meaningful_retention_24m",
        "temporary_displacement_risk_24m", "permanent_separation_risk_24m",
        "generic_continuity_24m", "downside_25pct_12m", "downside_25pct_24m",
        "annual_wage_peer_benchmark", "contract_duration_peer_benchmark",
        "fixed_wage_commitment_peer_benchmark", "wage_to_market_value_peer_benchmark",
        "integrated_contract_score", "historical_risk_weighted_screening",
    }
    check("all_expected_outputs", set(dispositions.output_id) == expected_ids, sorted(dispositions.output_id), sorted(expected_ids), "Exactly fifteen final output families must be resolved.")
    check("unique_outputs", dispositions.output_id.is_unique, dispositions.output_id.duplicated().sum(), 0, "Each output has one authoritative disposition.")
    check("no_live_predictive_probabilities", not dispositions.live_predictive_probability_allowed.any(), int(dispositions.live_predictive_probability_allowed.sum()), 0, "The freeze authorizes no live model probability.")
    check("no_automated_contract_decision", not dispositions.automated_contract_decision_allowed.any(), int(dispositions.automated_contract_decision_allowed.sum()), 0, "MoveMaker never recommends extend/do-not-extend.")
    check("descriptive_only_numeric", set(dispositions.loc[dispositions.descriptive_numeric_allowed, "output_id"]) == {"contract_arithmetic", "club_salary_structure_context"}, sorted(dispositions.loc[dispositions.descriptive_numeric_allowed, "output_id"]), ["club_salary_structure_context", "contract_arithmetic"], "Only non-predictive context is numerically displayable.")
    check("meaningful_retention_not_relabelled", dispositions.set_index("output_id").loc["meaningful_retention_24m", "final_disposition"] == "definition_retained_current_model_not_finalized", dispositions.set_index("output_id").loc["meaningful_retention_24m", "final_disposition"], "definition_retained_current_model_not_finalized", "The repaired contribution target supersedes the old joint-model result.")
    check("generic_continuity_retired", dispositions.set_index("output_id").loc["generic_continuity_24m", "disposition_class"] == "retired", dispositions.set_index("output_id").loc["generic_continuity_24m", "disposition_class"], "retired", "Mixed movement mechanisms remain retired.")
    check("integrated_score_rejected", dispositions.set_index("output_id").loc["integrated_contract_score", "disposition_class"] == "rejected", dispositions.set_index("output_id").loc["integrated_contract_score", "disposition_class"], "rejected", "No universal score is restored.")
    check("annual_wage_candidate_rejected", dispositions.set_index("output_id").loc["annual_wage_peer_benchmark", "disposition_class"] == "rejected", dispositions.set_index("output_id").loc["annual_wage_peer_benchmark", "disposition_class"], "rejected", "The failed sealed Ligue 1 gate remains binding.")

    verification_paths = [
        contribution_dir / "independent_verification.json",
        movement_dir / "independent_verification.json",
        value_dir / "independent_verification.json",
        phase5_dir / "independent_verification.json",
        finance_dir / "independent_verification.json",
        business_dir / "independent_verification.json",
        integration_dir / "independent_verification.json",
    ]
    upstream = {str(path.relative_to(ROOT)): verified(path) for path in verification_paths}
    check("all_upstream_verifications_pass", all(upstream.values()), upstream, {key: True for key in upstream}, "Every evidence family has a passing independent verification artifact.")
    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(output / "build_checks.csv", index=False)

    contract = {
        "freeze_version": FREEZE_VERSION,
        "scope": "incumbent-club contract-extension analytical case study",
        "freeze_date": "2026-09-10",
        "output_families": len(dispositions),
        "allowed_disposition_classes": sorted(dispositions.disposition_class.unique()),
        "live_predictive_probabilities_authorized": 0,
        "automated_contract_decisions_authorized": 0,
        "descriptive_numeric_outputs": sorted(dispositions.loc[dispositions.descriptive_numeric_allowed, "output_id"]),
        "interpretation": "A final case-study disposition is a stopping decision, not a production promotion.",
    }
    write_json(output / "freeze_contract.json", contract)
    summary = {
        "freeze_version": FREEZE_VERSION,
        "output_families": len(dispositions),
        "disposition_counts": {
            str(key): int(value)
            for key, value in dispositions.disposition_class.value_counts().sort_index().items()
        },
        "checks_passed": int(checks_frame.passed.sum()),
        "checks_total": int(len(checks_frame)),
        "all_checks_passed": bool(checks_frame.passed.all()),
        "upstream_verification_families": len(upstream),
        "live_predictive_probabilities_authorized": 0,
        "automated_contract_decisions_authorized": 0,
        "model_artifacts_created": 0,
        "deployment_changed": False,
    }
    write_json(output / "run_summary.json", summary)
    (output / "README.md").write_text(
        """# MoveMaker case-study metric freeze

This directory is the authoritative compact disposition layer for the final
MoveMaker analytical case study. It resolves fifteen current output families
using the latest independently verified evidence. A disposition can retain,
qualify, defer, retire, or reject an output; none authorizes a live predictive
probability or an automated contract decision.

`metric_dispositions.csv` is the canonical row-level summary. The source and
output manifests make the freeze auditable and reproducible. No model was fit,
serialized, promoted, or deployed by this closeout.
""",
        encoding="utf-8",
    )

    source_paths = [
        ROOT / "models" / "run_case_study_metric_freeze.py",
        contribution_dir / "reliability_decision.csv",
        contribution_dir / "subgroup_reliability.csv",
        contribution_dir / "independent_verification.json",
        movement_dir / "product_scope_decisions.csv",
        movement_dir / "development_evidence.csv",
        movement_dir / "independent_verification.json",
        value_dir / "selection_decision.csv",
        value_dir / "known_exposure_audit.csv",
        value_dir / "independent_verification.json",
        phase5_dir / "wage_final_holdout_metrics.csv",
        phase5_dir / "wage_final_holdout_subgroups.csv",
        phase5_dir / "independent_verification.json",
        finance_dir / "benchmark_decision_summary.csv",
        finance_dir / "deterministic_metric_catalog.csv",
        finance_dir / "independent_verification.json",
        business_dir / "low_contribution_exposure_performance.csv",
        business_dir / "low_contribution_exposure_comparison.csv",
        business_dir / "low_contribution_exposure_predictions.csv",
        business_dir / "independent_verification.json",
        integration_dir / "integration_score_evaluation.csv",
        integration_dir / "independent_verification.json",
        PROCESSED / "engine_v2_contribution_model_rerun" / "player_disjoint_sensitivity.csv",
        value_dir / "candidate_metrics.csv",
        ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json",
    ]
    pd.DataFrame([
        {"source": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in source_paths
    ]).to_csv(output / "source_manifest.csv", index=False)
    manifest_rows = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "output_manifest.csv":
            manifest_rows.append({"output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest_rows).to_csv(output / "output_manifest.csv", index=False)

    if not checks_frame.passed.all():
        raise RuntimeError(checks_frame.loc[~checks_frame.passed].to_dict("records"))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

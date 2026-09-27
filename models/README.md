# MoveMaker model and verification registry

Run commands from the repository root. Every finalized research stage has a dedicated verifier and compact evidence under `Data/processed/`.

## Engine V2 validity program

Frozen V1 public scoring is unconditionally disabled. Engine V2 work is isolated
from `Data/processed/deployment_models/` and must advance phase by phase.
Release-blocking failures are maintained separately in
[`docs/ENGINE_V2_CRITICAL_ISSUES.md`](../docs/ENGINE_V2_CRITICAL_ISSUES.md) and
must not be diluted into ordinary phase notes.

The original twelve findings that triggered the V2 program, together with an
explicit `FIXED`, `CONTAINED`, `DEFERRED`, or `OPEN` status for each, are
preserved in
[`docs/ENGINE_V2_ORIGINAL_AUDIT.md`](../docs/ENGINE_V2_ORIGINAL_AUDIT.md).
Verify that ledger and its shared Ridge-explainer repair with
`python scripts/verify_engine_v2_original_audit.py`.

| Stage | Status | Runner | Verifier | Evidence |
| --- | --- | --- | --- | --- |
| Phase 0: V1 freeze and containment | Complete | — | `python scripts/verify_engine_v1_freeze.py` | `docs/ENGINE_V1_FREEZE.md` |
| Phase 1: cohort and decision-time validity boundary | Complete; no model fitted | `python models/run_engine_v2_validity_boundary.py` | `python scripts/verify_engine_v2_validity_boundary.py` | `Data/processed/engine_v2_validity_boundary/` |
| Phase 2: endpoint and feature specification | Complete; candidates only, not deployed | `python models/run_engine_v2_feature_specification.py` | `python scripts/verify_engine_v2_feature_specification.py` | `Data/processed/engine_v2_feature_specification/` |
| Phase 3: calibration, uncertainty, subgroup, and OOD reliability | Complete; role/value candidates advance, continuity blocked, none deployed | `python models/run_engine_v2_calibration_reliability.py` | `python scripts/verify_engine_v2_calibration_reliability.py` | `Data/processed/engine_v2_calibration_reliability/` |
| Phase 4: continuity repair, wage audit, and versioned result contract | Complete; continuity still blocked by Ligue 1 reliability, none deployed | `python models/run_engine_v2_candidate_contract.py` | `python scripts/verify_engine_v2_candidate_contract.py` | `Data/processed/engine_v2_candidate_contract/` |
| Phase 5: final-evaluation readiness and sealed wage holdout | Complete and corrected; value 2024+ was already exposed by V1 verification, wage candidate failed Ligue 1 coverage, none deployed | `python models/run_engine_v2_final_evaluation_gate.py` | `python scripts/verify_engine_v2_final_evaluation_gate.py` | `Data/processed/engine_v2_final_evaluation_gate/` |
| Public-value downside repair | Development cleanup; proposal-free 12m primary and 24m secondary candidates use only player-disjoint 2020–2023 origins; no deployment | `python models/run_engine_v2_public_value_downside_repair.py` | `python scripts/verify_engine_v2_public_value_downside_repair.py` | `Data/processed/engine_v2_public_value_downside_repair/` |
| Movement-scope closeout | Complete; generic continuity retired, replacement questions separated, none deployed | `python models/run_engine_v2_movement_scope_closeout.py` | `python scripts/verify_engine_v2_movement_scope_closeout.py` | `Data/processed/engine_v2_movement_scope_closeout/` |
| Future-role target validity audit | Complete; current target blocked as a pure role outcome, no model fitted | `python models/run_engine_v2_future_role_target_audit.py` | `python scripts/verify_engine_v2_future_role_target_audit.py` | `Data/processed/engine_v2_future_role_target_audit/` |
| Symmetric contribution evidence | Complete; Year-1/Year-2 evidence enforced, seven invalid labels removed, remaining target repairs still blocked | `python models/run_engine_v2_symmetric_contribution_evidence.py` | `python scripts/verify_engine_v2_symmetric_contribution_evidence.py` | `Data/processed/engine_v2_symmetric_contribution_evidence/` |
| Complete contribution-target repair | Complete; schedule refusal and `[0,1]` normalization implemented, no model fitted or deployed | `python models/run_engine_v2_contribution_target_repair.py` | `python scripts/verify_engine_v2_contribution_target_repair.py` | `Data/processed/engine_v2_contribution_target_repair/` |
| Repaired contribution development rerun | Complete; compact recent-involvement candidate advanced to reliability audit, not deployed | `python models/run_engine_v2_contribution_model_rerun.py` | `python scripts/verify_engine_v2_contribution_model_rerun.py` | `Data/processed/engine_v2_contribution_model_rerun/` |
| Contribution reliability audit | Complete; pooled gates passed but the frozen all-group uncertainty gate failed, not deployed | `python models/run_engine_v2_contribution_reliability_audit.py` | `python scripts/verify_engine_v2_contribution_reliability_audit.py` | `Data/processed/engine_v2_contribution_reliability_audit/` |
| Final case-study metric freeze | Complete; fifteen output families resolved, no model fit or deployment | `python models/run_case_study_metric_freeze.py` | `python scripts/verify_case_study_metric_freeze.py` | `Data/processed/case_study_metric_freeze/` |

Phase 1 permits only research outcome scenarios conditional on an extension.
It does not validate extend/do-not-extend recommendations because the sources
do not observe the full set of players considered but not extended.

Phase 2 narrows V2 to three predictive headline candidates plus a separately
validated wage-benchmark module. It removes raw scoring rates, unsupported
advanced-event fields, and user-controlled commercial terms from the sporting
and public-value candidates. Passing Phase 2 means only that a feature
specification may proceed to calibration and subgroup testing.

Phase 3 requires calibrated probabilities, explicit model-fit uncertainty,
subgroup evidence thresholds, and supported/limited/refused input states. The
future-role and value-downside candidates passed the declared endpoint gate.
The 24-month continuity candidate did not: required prior-season minutes were
missing often enough to produce a 60.7% refusal rate. No Phase-3 result is a
deployment authorization.

Phase 4 replaces those sparse minutes with broadly available recent-involvement
features and adds a decision-time-valid nonlinear age term. Refusal falls to
5.6% and pooled performance improves, but Ligue 1 still has negative Brier
skill against chronology-only prevalence. Big-Five continuity therefore
remains blocked. The annual-wage peer benchmark passes its broad
position/league audit and may proceed to the sealed final holdout. The
versioned result contract requires values, uncertainty, support status,
subgroup status, data vintage, and refusal reasons to travel together and
prohibits an overall score or automated recommendation.

Phase 5 finds that 2024 is not a usable final test for the predictive risk
modules: role/continuity labels were previously exposed in aggregate, and the
player-disjoint role and value cohorts are underpowered. Only the annual-wage
benchmark qualified to open. It improved strongly in the pooled 235-row final
cohort but failed the frozen Ligue 1 interval-coverage gate (60.6% versus the
required 70%), so no artifact was authorized. The final result is immutable;
2024 may not be used to retune that candidate.

The movement-scope closeout resolves the former continuity blocker through
endpoint retirement rather than model repair. Meaningful Retention and
Temporary Displacement Risk are separate development candidates; Permanent
Separation is research-only and unavailable. The HTTP application has no V1
scoring override, and no movement/retention probability is deployed.

The future-role target audit then tests the contribution component itself.
It finds an asymmetric Year-1 evidence gate, materially incomplete annual club
schedules around relegation/top-flight exit, movement-driven labels, and a
semantic difference between realized contribution and role when available.
The frozen target remains historical evidence only. Meaningful Retention is
blocked until its realized-contribution component is rebuilt and all affected
development evidence is rerun.

The complete contribution-target repair then rebuilds that component as a
versioned nullable target. It requires both annual evidence gates, at least 80%
of the contemporaneous original-league peer schedule in each exact window, and
bounded opportunity shares. This closes target construction only: no model was
fit, and Meaningful Retention remains blocked pending a complete development
rerun and later untouched player-disjoint temporal evaluation.

The subsequent repaired development rerun selected the compact recent-
involvement candidate. Its reliability audit retained positive pooled signal,
calibration, uncertainty, refusal, drift, and monotonicity results, but only
six of nine core league/position groups cleared the frozen 99% skill gate.
That model is therefore qualified case-study evidence rather than a live
all-Big-Five probability. Meaningful Retention still requires a new joint-model
rerun because its earlier evidence predates this target repair.

The final case-study freeze resolves every current output family without
fitting another model. Its dispositions supersede older advancement language
where a later target repair, final holdout, or scope decision changed the
conclusion.

## Current incumbent-extension pipeline

| Stage | Runner or builder | Verifier | Evidence |
| --- | --- | --- | --- |
| Capology contracts | `python scripts/build_capology_contracts.py` | `python scripts/verify_capology_contracts.py` | `Data/processed/capology_contracts/` |
| Contract integration | `python scripts/build_contract_extension_integration.py` | `python scripts/verify_contract_extension_integration.py` | `Data/processed/contract_extension_integration/` |
| Frozen research source | `python scripts/build_frozen_contract_source_snapshot.py` | Verified by downstream source hashes | `Data/processed/frozen_contract_sources/` (local) |
| Phase 1: opportunity and contribution | `python models/run_extension_opportunity_diagnostic.py` | `python scripts/verify_extension_opportunity_diagnostic.py` | `Data/processed/extension_opportunity_diagnostic/` |
| Phase 2: continuity and outbound | `python models/run_extension_survival_diagnostic.py` | `python scripts/verify_extension_survival_diagnostic.py` | `Data/processed/extension_survival_diagnostic/` |
| Phase 3: public-value preservation | `python models/run_extension_value_preservation_diagnostic.py` | `python scripts/verify_extension_value_preservation_diagnostic.py` | `Data/processed/extension_value_preservation_diagnostic/` |
| Phase 4: financial peer benchmarks | `python models/run_contract_financial_exposure_benchmark.py` | `python scripts/verify_contract_financial_exposure_benchmark.py` | `Data/processed/contract_financial_exposure_benchmark/` |
| Phase 5: integrated profile | `python models/run_integrated_contract_profile.py` | `python scripts/verify_integrated_contract_profile.py` | `Data/processed/integrated_contract_profile/` |
| Business translation audit | `python models/run_business_metric_translation_audit.py` | `python scripts/verify_business_metric_translation_audit.py` | `Data/processed/business_metric_translation_audit/` |
| Product calculation layer | `python scripts/build_business_metric_product_layer.py` | `python scripts/verify_business_metric_product_layer.py` | `Data/processed/business_metric_product/` |

`models/business_metric_engine.py` formats deterministic contract facts and consumes model probabilities. It does not generate player predictions by itself.

## Frozen V1 scoring components

| Component | Purpose |
| --- | --- |
| `models/canonical_feature_retriever.py` | Resolves current player, club, performance, salary, and public-value features |
| `models/deployment_model_artifact.py` | Versioned serialized artifact contract |
| `models/deployment_scorer.py` | Reconstructs the 16 frozen historical endpoints offline |
| `models/feature_contribution_explainer.py` | Reconstructs and explains supported predictions |
| `models/player_search.py` | Player lookup for the application |
| `models/incumbent_extension_profile.py` | Integrates the typed extension profile |
| `api/main.py` | FastAPI routes and static application hosting |

Build and verify the deployed artifact package with:

```bash
python scripts/build_deployment_models.py
python scripts/verify_deployment_models.py
python scripts/verify_deployment_out_of_sample.py
python scripts/verify_live_request_integrity.py
```

The frozen V1 artifact manifest is under `Data/processed/deployment_models/`.
Do not overwrite it during an exploratory run. These components are retained
for regression testing and historical audit; they are not enabled by default.

## Post-production diagnostics

| Work | Status | Runner | Verifier | Evidence |
| --- | --- | --- | --- | --- |
| Contribution refinement | Historical experiment | `models/run_contribution_model_refinement.py` | `scripts/verify_contribution_model_refinement.py` | `Data/processed/contribution_model_refinement/` |
| C6 contribution repair | C6 deployed after explicit promotion and parity verification | `models/run_contribution_model_repair.py` | `scripts/verify_contribution_model_repair.py`, `scripts/verify_c6_deployment.py` | `Data/processed/contribution_model_repair/` |
| Live input refresh | Append-only live feature source | `scripts/build_live_input_refresh.py` | `scripts/verify_live_input_refresh.py` | `Data/processed/live_input_refresh/` |
| Role contextualization | Research only; no production promotion | `models/run_role_contextualization_diagnostic.py` | `scripts/verify_role_contextualization_diagnostic.py` | `Data/processed/role_contextualization_diagnostic/` |
| Local data-use audit | Refreshed after Capology club-season acquisition; 26/26 checks | `scripts/audit_data_usage_coverage.py` | `scripts/verify_data_usage_coverage.py` | `Data/processed/data_usage_audit_2026_09_02/` |
| Capology club financial context | Verified feature infrastructure; rejected as continuity repair | `scripts/build_capology_club_financial_context.py`, `models/run_engine_v2_club_financial_context_experiment.py` | `scripts/verify_capology_club_financial_context_build.py`, `scripts/verify_capology_club_financial_context.py`, `scripts/verify_engine_v2_club_financial_context_experiment.py` | `Data/processed/capology_club_financial_context/`, `Data/processed/engine_v2_club_financial_context_experiment/` |
| Movement-state decomposition | Development diagnostic; replacement semantics frozen | `models/run_engine_v2_movement_state_profile_diagnostic.py` | `scripts/verify_engine_v2_movement_state_profile_diagnostic.py` | `Data/processed/engine_v2_movement_state_profile_diagnostic/` |
| Lagged club behavior | Research only; no block promoted | `models/run_engine_v2_lagged_club_behavior_experiment.py` | `scripts/verify_engine_v2_lagged_club_behavior.py` | `Data/processed/engine_v2_lagged_club_behavior_experiment/` |
| Role trajectory and squad competition | Research/descriptive only; no block promoted | `models/run_engine_v2_role_squad_experiment.py` | `scripts/verify_engine_v2_role_squad.py` | `Data/processed/engine_v2_role_squad_experiment/` |
| Manager and tactical context | Research/descriptive only; no block promoted | `models/run_engine_v2_manager_tactical_experiment.py` | `scripts/verify_engine_v2_manager_tactical.py` | `Data/processed/engine_v2_manager_tactical_experiment/` |
| Repaired contribution target | Target construction complete; no model promotion | `models/run_engine_v2_contribution_target_repair.py` | `scripts/verify_engine_v2_contribution_target_repair.py` | `Data/processed/engine_v2_contribution_target_repair/` |
| Repaired contribution model rerun | Run 2 development candidate advanced to reliability audit; not deployed | `models/run_engine_v2_contribution_model_rerun.py` | `scripts/verify_engine_v2_contribution_model_rerun.py` | `Data/processed/engine_v2_contribution_model_rerun/` |
| Repaired contribution reliability audit | Run 3 passed pooled reliability but failed all-league/all-position uncertainty gate; not deployed | `models/run_engine_v2_contribution_reliability_audit.py` | `scripts/verify_engine_v2_contribution_reliability_audit.py` | `Data/processed/engine_v2_contribution_reliability_audit/` |

Do not promote candidate `.joblib` files directly. A production change requires backup, full-history refit through the declared cutoff, manifest update, cross-path scoring parity, endpoint smoke tests, explainability reconstruction, and independent verification.

## Earlier transfer-risk research track

These models remain useful historical research but do not define the incumbent-extension product:

| Research module | Runner | Verifier |
| --- | --- | --- |
| Retention diagnostic | `models/run_retention_diagnostic.py` | `scripts/verify_retention_diagnostic.py` |
| Market-value downside | `models/run_market_value_downside_model.py` | `scripts/verify_market_value_downside_model.py` |
| Market-value calibration | `models/run_market_value_calibration_experiment.py` | `scripts/verify_market_value_calibration_experiment.py` |
| Paid-transfer under-utilization | `models/run_paid_transfer_underutilization_model.py` | `scripts/verify_paid_transfer_underutilization_model.py` |
| Under-utilization refinement | `models/run_paid_transfer_underutilization_refinement.py` | `scripts/verify_paid_transfer_underutilization_refinement.py` |
| Reported-fee benchmark | `models/run_fee_benchmarking_model.py` | `scripts/verify_fee_benchmarking_model.py` |
| Component capital at risk | `models/run_capital_at_risk_integration.py` | `scripts/verify_capital_at_risk_integration.py` |
| One-season landmark update | `models/run_landmark_monitoring_experiment.py` | `scripts/verify_landmark_monitoring_experiment.py` |

Do not mix transfer-trained probabilities with extension decisions or describe these modules as tactical fit, precise price, expected loss, or ROI.

## Development rules

- Primary evaluation is rolling chronological, never a random split alone.
- Fit preprocessing and selection inside each training origin.
- Preserve missingness and data-vintage fields.
- Save candidates outside `deployment_models/` until formally promoted.
- Compare with the frozen chronological baseline and report uncertainty and origin wins.
- Read the scope-pivot journal and the relevant evidence README before modifying a stage.

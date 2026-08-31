# MoveMaker model and verification registry

Run commands from the repository root. Every finalized research stage has a dedicated verifier and compact evidence under `Data/processed/`.

## Engine V2 validity program

Frozen V1 public scoring is disabled by default. Engine V2 work is isolated
from `Data/processed/deployment_models/` and must advance phase by phase.

| Stage | Status | Runner | Verifier | Evidence |
| --- | --- | --- | --- | --- |
| Phase 0: V1 freeze and containment | Complete | — | `python scripts/verify_engine_v1_freeze.py` | `docs/ENGINE_V1_FREEZE.md` |
| Phase 1: cohort and decision-time validity boundary | Complete; no model fitted | `python models/run_engine_v2_validity_boundary.py` | `python scripts/verify_engine_v2_validity_boundary.py` | `Data/processed/engine_v2_validity_boundary/` |

Phase 1 permits only research outcome scenarios conditional on an extension.
It does not validate extend/do-not-extend recommendations because the sources
do not observe the full set of players considered but not extended.

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
| `models/deployment_scorer.py` | Scores the 16 deployed endpoints |
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

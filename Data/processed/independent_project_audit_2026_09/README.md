# Independent project audit — evidence package (2026-09-09)

Report: `docs/INDEPENDENT_PROJECT_AUDIT_2026-09.md`.

All files here were produced by the scripts under `audit_scripts/`, which are read-only against every existing artifact. No file under `Data/processed/deployment_models/` or any other stage directory was modified; the frozen V1 inventory (24 files) was hash-checked before and after the runs (`git_and_deployment_state.json`).

| File | Content |
| --- | --- |
| `issue_ledger.csv` / `.json` | 23 issues (MM-01 … MM-23) with severity, status, evidence, affected files, consequence, classification, proposed repair, repairability |
| `model_metric_status_table.csv` | Every model/metric classified twice: as historical research evidence and as a live product output on `main` |
| `contradictions_claims_vs_implementation.csv` | Claim vs implementation table with issue cross-references |
| `metric_reconstruction.csv` | Claimed vs independently reconstructed metrics for Phases 1–5, the business illustration, Engine V2 and the live re-score |
| `dataset_and_activity_inventory.csv` | Sources, derived layers, sizes, current use, active/dormant status |
| `deployed_artifact_feature_inventory.csv` | The 16 frozen artifacts: model class, feature counts, whether user-proposal terms / global scoring rates / advanced FBref fields are inputs, UI surface, SHA-256 |
| `cohort_target_and_overlap_audit.json` | Cohort sizes, target composition (departures among non-sustainers), player overlap across evaluation years, missingness, data-quality residues |
| `oos_verifier_maturity_bias.csv` | Row counts showing which post-2023 survival rows the project's OOS verifier admitted without full follow-up |
| `git_and_deployment_state.json` | Branch/commit state, live `/health` payload, freeze hash check, environment |
| `t1_scoring_time_perturbation_grid.csv` | 300 live-path scores: 10 real 2023 profiles × 6 wage multipliers × 5 terms |
| `t1_scoring_time_perturbation_summary.csv`, `t1_effect_of_proposed_years_wage_fixed.csv`, `t1_effect_of_proposed_wage_years_fixed_3.csv` | Condensed perturbation effects |
| `t2_oos_live_rescoring_predictions.csv` / `t2_oos_live_rescoring_metrics.json` | 547 post-2023 events scored through the live retriever + scorer with targets and maturity flags; calibration, per-league AUC, corrected-maturity survival metrics |
| `t3_present_day_demo_profiles.json` | Bellingham / Haaland / Mbappé profiles at decision date 2026-09-09 through `main`'s application path, plus arbitrary-club and out-of-scope probes |

Reproduce (repository root, local runtime data present):

```bash
python Data/processed/independent_project_audit_2026_09/audit_scripts/audit_live_pipeline_tests.py
python Data/processed/independent_project_audit_2026_09/audit_scripts/audit_present_day_demo_profiles.py
python Data/processed/independent_project_audit_2026_09/audit_scripts/build_audit_evidence.py
python Data/processed/independent_project_audit_2026_09/audit_scripts/build_audit_tables.py
```

`t3` depends on the run date; everything else is deterministic given the frozen artifacts and local tables.

# Contribution-model refinement experiment

Bounded research experiment testing whether pre-extension injury history and
evidence-based elite-player interactions improve the frozen Phase-1
`sustained_meaningful_contribution` (meaningful-contributor-through-Year-2) model.
This is NOT a redeployment: it does not modify
`Data/processed/extension_opportunity_diagnostic/`, `Data/processed/deployment_models/`,
the production scorer, or the web app.

## Step 1 -- baseline reproduction

Pooled out-of-time cohort: 929 rows across four rolling origins (2020-2023).
Observed rate 0.4887, mean predicted 0.5227, Brier 0.1869, AUC 0.7881, AP 0.7323 -- all within the declared tolerance of the published reference.
High-value (>=EUR75M) audit: n=22, observed 0.8636, predicted 0.8556.

## Step 2 -- injury linkage

4080 unique injury-source player names; 3982 linked to canonical identity (98.0% row-level link rate). 303 rows excluded on a club-history cross-check mismatch.
Linkage uses only the existing exact-unique-normalized-name crosswalk (`build_player_map`/`build_club_map`) already used elsewhere in this repository -- no fuzzy or forced matches.

## Step 6 -- decision

- `R1_injury`: advances=False; pooled Brier improvement -0.005213, 95% CI [-0.011938, 0.001709], 2/4 origin wins, terminal-two improvement 0.003596474706067823, ECE delta -0.0175, majority-cohort improvement -0.0037927740780714712.
- `R2_elite`: advances=False; pooled Brier improvement 0.002189, 95% CI [-0.000999, 0.005299], 3/4 origin wins, terminal-two improvement 0.0043285712970700715, ECE delta -0.0102, majority-cohort improvement 0.0025527076077869124.
- `R3_injury_elite`: advances=False; pooled Brier improvement -0.004189, 95% CI [-0.010683, 0.002328], 3/4 origin wins, terminal-two improvement 0.0040446045940631245, ECE delta -0.0100, majority-cohort improvement -0.002573519837601183.
- `R4_compact`: advances=False; pooled Brier improvement 0.002111, 95% CI [-0.003027, 0.007351], 3/4 origin wins, terminal-two improvement 0.005160821532721816, ECE delta -0.0101, majority-cohort improvement 0.0033345301317228852.

**Any candidate advanced: False.**

See `model_comparison_results.csv`, `calibration_audit.csv`, `elite_subgroup_audit.csv`, `subgroup_stability.csv`, `model_selection_decision.csv`, `target_decomposition_diagnostic.csv`, and `demo_player_comparison.csv` for full detail before making any product claim.

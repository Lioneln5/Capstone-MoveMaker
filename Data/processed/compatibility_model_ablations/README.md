# MoveMaker stronger-model compatibility ablations

This phase compares ridge regression, sparse elastic net, and a deterministic nonlinear gradient-boosted regression-tree implementation. Each learner is evaluated on five declared feature sets: player-history baseline, baseline plus raw club context, baseline plus historical player preferences, context plus preferences, and the full model including explicit fit interactions.

All hyperparameters and validation-selected specifications are chosen on 2022 MAE. Models are then refit on train+validation and scored once on 2023. The principal causal-safe reading is within-family full-versus-baseline error change; cross-family selected comparisons are predictive benchmarks, not causal estimates.

Use `ablation_feature_block_effects.csv` for the full model/feature grid, `ablation_family_bootstrap.csv` for paired within-family uncertainty, and `ablation_model_selection.csv` plus `ablation_selected_bootstrap.csv` for the validation-selected baseline/full comparison. Test cohorts remain small, so repeated effects across validation, test, role, and league matter more than a single point estimate.

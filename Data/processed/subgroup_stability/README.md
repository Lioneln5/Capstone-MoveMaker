# MoveMaker frozen-candidate subgroup stability

This phase uses the already-scored rolling-origin predictions and does not retrain from subgroup outcomes. After correcting the shared encoder and rerunning the 2022 validation selection, both opportunity and performance are frozen as full compatibility plus gradient-boosted trees.

The primary comparison is the candidate versus the same model family using only player-history baseline features. This isolates compatibility information from learner choice. A secondary comparison uses the best baseline family selected on each origin validation season.

Primary subgroup reporting uses the mature three origins. A cell is marked adequate when pooled n is at least 15, every included origin is present, and the minimum origin cell is at least 5. All subgroup cells remain in the outputs, but small cells are explicitly labeled and excluded from confirmatory FDR adjustment.

Paired uncertainty uses 5,000 bootstrap resamples stratified by origin. The primary mature, same-family, adequately sized role and competition cells share one Benjamini-Hochberg FDR family at q=10%. Heterogeneity is tested by permuting subgroup labels within each origin 5,000 times, with separate FDR adjustment across the four primary target-dimension tests.

Use subgroup_primary_results.csv for the main mature role and competition readout, subgroup_origin_results.csv for season-level behavior, subgroup_heterogeneity.csv for evidence that effects differ between roles or leagues, and subgroup_pair_predictions.csv for the auditable player-level paired errors.

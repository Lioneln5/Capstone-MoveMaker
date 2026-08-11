# MoveMaker rolling-origin stability testing

This phase runs four expanding-window origins: train through 2018 / validate 2019 / evaluate 2020, then advances one season at a time through the terminal train-through-2021 / validate-2022 / evaluate-2023 origin. At every origin, all ridge, elastic-net, and deterministic boosted-tree hyperparameters are selected using that origin's validation season only. The following season is scored after selection.

`rolling_policy_comparison_stability.csv` is the primary leakage-safe policy result: at each origin it separately selects the best baseline family and best full-compatibility family on validation data, then compares them on the next season. `rolling_family_stability_summary.csv` holds model family fixed while comparing full compatibility with the player-history baseline.

`rolling_locked_stability_summary.csv` is a retrospective design stress test for the current opportunity and performance candidates chosen in the preceding 2022-validation phase. It locks feature set and model family but retunes hyperparameters inside each historical origin. Because the design itself was chosen later, use this as a stability diagnostic rather than an unbiased prospective performance estimate. Adaptation remains diagnostic-only and is excluded from the locked-candidate decision.

`rolling_sensitivity_summary.csv` repeats policy, locked-design, and within-family comparisons across the mature three origins and terminal two origins. This prevents the smallest initial training window from dominating the interpretation.

Positive MAE improvement means the compatibility candidate made lower absolute error than its paired baseline. The declared stability label is `stable_positive` when the pooled improvement is positive, at least three of four origins improve, and the stratified bootstrap probability of improvement is at least 90%; `mixed_positive` when pooled improvement is positive and at least half the origins improve; `unstable_negative` when pooled improvement is non-positive and at most one origin improves; otherwise it is `mixed`.

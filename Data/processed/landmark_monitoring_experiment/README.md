# MoveMaker day-365 landmark monitoring experiment

This v2 experiment tests whether first-year evidence improves forecasts for the remaining second year. It does not replace or rewrite any finalized v1 model.

## Leakage-safe design

- Utilization cohort: 667 clean paid permanent transfers still at the destination on day 365, with at least 10 destination-club matches in days 366-730.
- Market-value cohort: 2,632 permanent transfers still at the destination on day 365, with a positive valuation observed on/before day 365 and no more than 180 days stale.
- Utilization targets cover only days 366-730, not the already-observed first year.
- Value targets measure change from the strictly as-of day-365 valuation to the existing 24-month outcome valuation.
- Four rolling origins evaluate 2020-2023. L2 logistic/ridge strength is selected on the preceding validation season only.
- Primary uncertainty uses 5,000 paired player-cluster bootstrap repetitions.

## Monitoring-increment results

- `year2_opportunity_share`: `validated_monitoring_increment`; mae improvement +0.027959, player-clustered 95% CI [+0.015882, +0.040028], 4/4 origin wins.
- `year2_underutilization_10pct`: `validated_monitoring_increment`; brier improvement +0.030663, player-clustered 95% CI [+0.015675, +0.045668], 4/4 origin wins, pooled candidate AUC 0.727.
- `year2_underutilization_25pct`: `validated_monitoring_increment`; brier improvement +0.036186, player-clustered 95% CI [+0.019723, +0.051920], 4/4 origin wins, pooled candidate AUC 0.737.
- `year2_value_log_ratio`: `validated_monitoring_increment`; mae improvement +0.005437, player-clustered 95% CI [+0.001111, +0.009578], 3/4 origin wins.
- `year2_value_downside_10pct`: `validated_monitoring_increment`; brier improvement +0.005641, player-clustered 95% CI [+0.002141, +0.009086], 4/4 origin wins, pooled candidate AUC 0.752.
- `year2_value_downside_25pct`: `validated_monitoring_increment`; brier improvement +0.004827, player-clustered 95% CI [+0.001785, +0.007886], 4/4 origin wins, pooled candidate AUC 0.729.
- `year2_value_downside_50pct`: `promising_but_uncertain`; brier improvement +0.002065, player-clustered 95% CI [-0.000238, +0.004420], 2/4 origin wins, pooled candidate AUC 0.724.

## Interpretation

A validated increment means the day-365 update improves prediction of second-year risk over the compact transfer-time profile on the same at-risk rows. It turns MoveMaker into a two-stage screen-and-monitor workflow. It does not imply that first-year involvement causes later outcomes, predict sale proceeds, or establish accounting ROI.

The threshold classifiers were fit independently for this signal test. Their raw probabilities are not yet a live scoring policy: the recommended utilization update has 12/365 under-10/under-25 ordering violations, while the recommended value update has 34/1,630 down-25/down-10 and 21/1,630 down-50/down-25 violations. Apply validation-only calibration and monotone ordering before deployment; see `probability_consistency_audit.csv`.

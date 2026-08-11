# MoveMaker 24-month market-value downside model

This experiment predicts estimated market-value change, not realized cash return or accounting ROI.

## Design

- Strict cohort: 4,371 permanent 2017+ transfers with recent positive pre-value and a positive 24-month value within +/-90 days.
- Continuous target: `log(post24 value / pre-transfer value)`; percentage change is retained for interpretation.
- Binary targets: declines of at least 10%, 25%, and 50%.
- Four rolling origins evaluate 2020-2023. Model family and hyperparameters are selected only on the preceding validation season.
- Primary uncertainty: paired player-cluster bootstrap with 5,000 repetitions.

## Feature blocks

- `M0_chronological_constant`: 0 raw/derived predictors.
- `M1_market_profile`: 11 raw/derived predictors.
- `M2_plus_valuation_trajectory`: 24 raw/derived predictors.
- `M3_plus_sporting_history`: 48 raw/derived predictors.
- `M4_plus_destination_context`: 92 raw/derived predictors.
- `M5_plus_origin_transition_context`: 152 raw/derived predictors.

## Results

- `value_log_ratio_24m`: `validated_predictive_signal`. Full vs chronological baseline mae improvement 0.192083, 95% CI [0.170841, 0.213051], 4/4 origin wins. Added history/context beyond market profile: -0.002865, CI [-0.007908, 0.002331].
- `downside_10pct`: `validated_predictive_signal`. Full vs chronological baseline brier improvement 0.091292, 95% CI [0.082896, 0.099864], 4/4 origin wins. Added history/context beyond market profile: -0.003138, CI [-0.005507, -0.000788].
- `downside_25pct`: `validated_predictive_signal`. Full vs chronological baseline brier improvement 0.089141, 95% CI [0.080270, 0.097950], 4/4 origin wins. Added history/context beyond market profile: 0.000138, CI [-0.002470, 0.002754].
- `downside_50pct`: `validated_predictive_signal`. Full vs chronological baseline brier improvement 0.059040, 95% CI [0.050553, 0.067765], 4/4 origin wins. Added history/context beyond market profile: 0.001190, CI [-0.001981, 0.004319].

## Interpretation guardrail

A validated result means the model ranks or estimates future Transfermarkt-style market-value downside better than a chronological constant baseline. It does not mean the model predicts sale proceeds, profit, wages, or full ROI.

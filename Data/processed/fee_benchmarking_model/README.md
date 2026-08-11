# MoveMaker reported-fee benchmarking model

This model estimates the disclosed permanent-transfer fee paid for comparable historical deals. It does not estimate total deal cost, future value, or ROI.

## Design

- Cohort: 1,486 clean permanent deals with a reliable positive reported fee and recent positive pre-transfer value.
- Target: natural log of reported fee; euro figures are obtained by exponentiating the prediction.
- Primary hurdles: chronological median and the player's recent pre-transfer market value.
- Nested temporal design: prior-year model selection, validation-year feature-block selection/range estimation, next-year evaluation.
- Empirical 80% and 90% ranges use validation residual magnitudes and are audited out of season.

## Feature blocks

- `M1_market_profile`: 10 predictors.
- `M2_plus_valuation_trajectory`: 22 predictors.
- `M3_plus_prior_opportunity`: 30 predictors.
- `M4_plus_player_performance`: 46 predictors.
- `M5_plus_compact_club_context`: 72 predictors.

## Validation-selected richer-policy diagnostics

- `origin_2019`: `M5_plus_compact_club_context` (validation MAE-log 0.722870).
- `origin_2020`: `M1_market_profile` (validation MAE-log 0.700460).
- `origin_2021`: `M2_plus_valuation_trajectory` (validation MAE-log 0.723913).
- `origin_2022`: `M2_plus_valuation_trajectory` (validation MAE-log 0.660300).

## Result

- Status: `validated_fee_benchmark`.
- Versus pre-transfer value: MAE-log improvement 0.063933, 95% CI [0.026541, 0.099825], 3/4 origin wins.
- Versus chronological median: improvement 0.595439, CI [0.521089, 0.671422], 4/4 wins.
- Compact benchmark MAE-log 0.704923, corresponding to an average multiplicative error factor of approximately 2.02x.
- Empirical interval coverage: 80% range 79.3%; 90% range 90.2%.
- The ranges are broad: median upper/lower ratio 8.5x for the 80% range and 19.7x for the 90% range. They are uncertainty guardrails, not precise fair-price bands.
- Recommended model scope: `M1_market_profile`; richer-policy increment over market/profile supported: `false`.

## Interpretation

For a proposed deal, compare the proposed disclosed fee with the model's historical benchmark and range. A premium is a comparables warning for negotiation/diligence, not proof of overpayment; a discount is not proof of value. Historical cards use the realized reported fee only to audit the benchmark.

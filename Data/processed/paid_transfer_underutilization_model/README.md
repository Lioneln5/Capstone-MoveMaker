# MoveMaker paid-transfer under-utilization model

This experiment estimates 24-month sporting opportunity after a paid permanent transfer. It is a capital-utilization screen, not an accounting ROI or tactical-quality model.

## Design

- Cohort: 1,074 clean paid permanent arrivals with at least 20 observable destination matches.
- Continuous target: share of destination-club match-minute capacity in the first 24 months.
- Binary targets: under 10% (severe) and under 25% (meaningful) of capacity.
- Four rolling origins evaluate the 2020-2023 seasons. Family, hyperparameters, and feature block are selected using only the preceding validation season.
- Binary policy probabilities are projected to guarantee P(under 10%) <= P(under 25%).
- Primary uncertainty uses 5,000 paired player-cluster bootstrap repetitions.

## Feature blocks

- `M0_chronological_constant`: 0 predictors.
- `M1_paid_deal_market_profile`: 11 predictors.
- `M2_plus_prior_opportunity`: 19 predictors.
- `M3_plus_player_performance`: 35 predictors.
- `M4_plus_destination_context`: 79 predictors.
- `M5_plus_origin_transition_context`: 139 predictors.

## Primary results

- `opportunity_share_24m`: `validated_predictive_signal`; mae improvement 0.021010, player-clustered 95% CI [0.010690, 0.031100], 4/4 origin wins.
- `underutilization_10pct`: `validated_predictive_signal`; brier improvement 0.035244, player-clustered 95% CI [0.021678, 0.049315], 4/4 origin wins.
- `underutilization_25pct`: `validated_predictive_signal`; brier improvement 0.027690, player-clustered 95% CI [0.012207, 0.042998], 4/4 origin wins.

## Risk separation

- Severe under-utilization: lowest predicted-risk quintile 18.6% observed vs highest 70.1%.
- Meaningful under-utilization: lowest predicted-risk quintile 41.5% observed vs highest 83.8%.
- Raw policy threshold violations: 48/584; ordered policy violations: 0.

## Guardrails

A positive result supports pre-transfer screening of likely under-use. It does not identify why the under-use occurred, prove player-club incompatibility, value the lost minutes in euros, measure performance quality, or estimate ROI. Absolute probabilities remain candidates for a separate calibration audit before live use.

# Opportunity transferability and context decomposition

## Cohort and pre-model audit

The frozen cohort contains 3,134 transfers and 1,072 players. The pooled evaluation contains 1,485 transfers and 782 players.

Origin pre365 opportunity: mean 0.331227, median 0.274232, zero rows 367, maximum 1.0277777777777777.
Origin pre730 opportunity: mean 0.292717, median 0.219102, zero rows 192, maximum 1.0044444444444445.
Full-cohort descriptive correlations with destination opportunity: pre365 Pearson 0.105106, Spearman 0.042028; pre730 Pearson 0.215128, Spearman 0.173379. These were not used for model selection.

## Frozen models

- `M0_mean_only`: 0 predictors.
- `M1_market_profile`: 5 predictors.
- `M2_plus_prior_opportunity`: 13 predictors.
- `M3_plus_player_performance`: 29 predictors.
- `M4_plus_destination_context`: 73 predictors.
- `M5_plus_origin_transition_context`: 133 predictors.

## Plain-language block results

1. Market/profile over mean: pooled MAE improvement 0.018986, player-clustered 95% CI [0.013929, 0.024232], `stable_positive`; mature-three 0.015798, CI [0.009881, 0.021581], `stable_positive`.
2. Prior opportunity beyond market/profile: pooled MAE improvement 0.003183, player-clustered 95% CI [0.001200, 0.005091], `stable_positive`; mature-three 0.002728, CI [0.000435, 0.005035], `stable_positive`.
3. Performance beyond prior opportunity: pooled MAE improvement 0.000624, player-clustered 95% CI [-0.001078, 0.002242], `mixed_positive`; mature-three 0.000369, CI [-0.001613, 0.002339], `mixed_positive`.
4. Destination context beyond player controls: pooled MAE improvement 0.006465, player-clustered 95% CI [0.002602, 0.010431], `stable_positive`; mature-three 0.006992, CI [0.002632, 0.011386], `mixed_positive`.
5. Origin/transition beyond destination context: pooled MAE improvement 0.006793, player-clustered 95% CI [0.004801, 0.008779], `stable_positive`; mature-three 0.007512, CI [0.005181, 0.009822], `stable_positive`.

The defensible revised angle is **both**. The evidence supports opportunity persistence plus a smaller contextual adjustment.

Persistence is not compatibility, and contextual improvement is predictive rather than causal. `opportunity_persistence_summary.csv` contains the evaluation-only persistence diagnostics.

## Integrity

- Comprehensive master SHA-256: `e8e9a778cedc2a1dfb6f063a6adbf3c12d52e25d944c0f4741e56db07db466c4`.
- Protected existing outputs unchanged during the run: `true`.
- Independent verification is recorded separately after the verifier runs.

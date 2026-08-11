# Coarse club-context and compatibility diagnostic

## Cohort and target

The audited cohort contains 3,134 transfer rows and 1,072 unique players. The four pooled evaluation seasons contain 1,485 rows and 782 unique players.

The sole target is `target_destination_minutes_opportunity_share_24m`: destination minutes in the 730 days after transfer divided by 90 times the destination club's observed games over the same horizon. All cohort rows have a positive denominator and a target in [0, 1.05].

## Frozen models

- `M0_player_history_baseline`: 192 raw/derived predictors; ridge only.
- `M1_plus_destination_context`: 236 raw/derived predictors; ridge only.
- `M2_plus_origin_transition_context`: 296 raw/derived predictors; ridge only.
- `M3_plus_coarse_fit`: 322 raw/derived predictors; ridge only.

All preprocessing and alpha selection are fit within each chronological origin. The first training window uses every eligible season through 2018. Player-clustered paired bootstrap intervals are primary; row-bootstrap intervals are secondary.

## Plain-language results

1. Destination context (M1 vs M0): MAE improvement 0.005190, player-clustered 95% CI [0.001750, 0.008613], `stable_positive`.
2. Origin/transition context (M2 vs M1): MAE improvement 0.000783, player-clustered 95% CI [-0.001433, 0.002960], `mixed_positive`.
3. Fixed player-by-context fit terms (M3 vs M2): pooled MAE improvement 0.001197, player-clustered 95% CI [0.000190, 0.002272], `stable_positive`; mature-three improvement 0.000516, CI [-0.000104, 0.001136], `stable_positive`.
4. Stability is evaluated in pooled-four, mature-three, and terminal-two windows in `coarse_club_context_results.csv`.
5. This evidence **leaves unchanged** confidence: The coarse-fit effect is not stable enough to change confidence that a compatibility-shaped signal exists.

The existing rich-cohort ridge opportunity result was positive (MAE improvement 0.044804, 95% CI [0.029543, 0.060442], pooled n 207). Only direction and stability should be compared: this diagnostic uses a 24-month target, a broader cohort, and coarser context proxies.

## Integrity

- Comprehensive master SHA-256: `e8e9a778cedc2a1dfb6f063a6adbf3c12d52e25d944c0f4741e56db07db466c4`.
- Protected existing output trees unchanged during the run: `true`.
- Independent verification is recorded in `independent_verification.json` and `.csv` after the separate verifier runs.

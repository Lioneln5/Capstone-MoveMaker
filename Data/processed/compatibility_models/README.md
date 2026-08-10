# MoveMaker compatibility model benchmark — first version

## Experiment

Three independent continuous targets are modeled with a mean-only diagnostic, a market-aware player-history baseline, and the same baseline plus strict compatibility features. Models use standardized ridge regression, select alpha by 2022 validation MAE, then refit train+validation before a single 2023 test evaluation.

## Feature comparison

- Player-history baseline: 81 raw transfer-time, market, and lagged player-history features.
- Strict compatibility model: 202 raw features, adding historical teammate-style preferences and destination-club prior-season context/fit.
- Player IDs, club IDs, names, season labels, targets, target diagnostics, eligibility flags, exclusion reasons, and target-roster scenario fields are excluded.

## Interpretation

`model_test_comparison.csv` is the main held-out comparison. Positive `improvement_in_preferred_direction` means compatibility improved the metric. `model_test_bootstrap.csv` reports paired 95% bootstrap intervals and the probability that compatibility reduced test error. With only 37–58 test transfers per target, uncertainty must be emphasized over point estimates.

## Reproduction

Run `scripts/train_compatibility_models.py`, followed by `scripts/verify_compatibility_models.py`. The training script requires only the bundled pandas and NumPy runtime.

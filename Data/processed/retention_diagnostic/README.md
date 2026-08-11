# MoveMaker retention diagnostic

This is a first predictability test for retention, not a claim that exact contract duration is solved.

## Frozen questions

- **Sporting retention:** among canonically classified permanent arrivals, did the player earn at least 10% of destination club match-minute capacity during days 366-730?
- **Destination-spell retention:** was there no later outbound event from the original destination within 730 days?
- **365-day landmark update:** among players with no outbound event during the first 365 days, does observed first-year involvement improve prediction of second-year sporting retention?

An outbound loan ends the uninterrupted destination spell even though it does not necessarily end the player's contract. Therefore `destination_spell_retained_730d` must not be described as contract retention.

## Cohorts

- Permanent arrivals after conflict exclusions: 1,471
- Sporting-label rows: 1,432; positive rate: 51.9%
- Uninterrupted spell at 365 days: 74.3%
- Uninterrupted spell at 730 days: 42.7%
- Players in the 365-day landmark analysis: 1,074

## Pooled rolling-origin headlines (evaluation years 2020-2023)

- At-transfer sporting model, all pre-transfer blocks vs prevalence: Brier improvement +0.019281, 95% player-cluster CI [+0.009729, +0.028616], weighted-origin candidate AUC 0.586, `stable_positive`.
- At-transfer two-year spell model, all pre-transfer blocks vs prevalence: Brier improvement +0.008980, 95% player-cluster CI [+0.003343, +0.014711], weighted-origin candidate AUC 0.577, `stable_positive`.
- First-year involvement added at the 365-day landmark: Brier improvement +0.030729, 95% player-cluster CI [+0.017148, +0.044994], weighted-origin candidate AUC 0.713, `stable_positive`.

Positive Brier improvement means the candidate probabilities are better calibrated and more accurate than the named baseline. See `retention_comparison_results.csv` for every nested feature increment and evaluation window.

## Guardrails

- Only permanent arrivals are modeled; loan and loan-return counts remain visible in the cohort audit.
- All transfer-time predictors are available at or strictly before the move.
- First-year fields appear only in the declared 365-day landmark model.
- The 730-day administrative horizon is complete as of 2026-08-10 for every primary arrival.
- Exact uncensored tenure is not regressed as an ordinary numeric outcome. Fixed-horizon retention avoids treating right-censored spells as known durations.
- Protected existing outputs were unchanged: `True`.

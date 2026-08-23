# Extension survival and time-to-outbound diagnostic

This Phase-2 diagnostic estimates cumulative outbound risk with a discrete-time hazard model. It handles right-censoring and is not yet a live probability policy.

## Definitions

- Administrative censoring date: 2026-08-10.
- Any outbound: a loan, loan return, or permanent move ends uninterrupted club involvement.
- Permanent outbound: only a permanent transfer ends the permanent club relationship; loans do not.
- Scheduled contract expiration is a signing-time predictor and audit boundary, not a censor, because a renewed player may remain at the club.
- Eight half-year hazard intervals support cumulative 12-, 24-, 36-, and 48-month forecasts.

## Cohort and validation

- Primary cohort: 2,358 linked extension events.
- Observed any-outbound events: 1,848; permanent-outbound events: 1,690.
- Four rolling origins evaluate signing years 2020-2023; model selection uses the immediately preceding validation year.
- A direct horizon-specific classifier is tested against the discrete-time survival formulation.
- Paired uncertainty uses 5,000 player-cluster bootstrap repetitions.

## Decisions

- `any_outbound` at `24m`: `advance_survival_probability_candidate`; weighted-origin Brier 0.2191, weighted-origin AUC 0.710, baseline improvement 0.03632 (95% CI [0.02766, 0.04527]); contract economics add signal: `True`; survival beats direct classifier: `False`.
- `any_outbound` at `36m`: `advance_survival_probability_candidate`; weighted-origin Brier 0.2026, weighted-origin AUC 0.699, baseline improvement 0.02103 (95% CI [0.01134, 0.03066]); contract economics add signal: `True`; survival beats direct classifier: `False`.
- `permanent_outbound` at `24m`: `advance_survival_probability_candidate`; weighted-origin Brier 0.2041, weighted-origin AUC 0.696, baseline improvement 0.02572 (95% CI [0.01856, 0.03313]); contract economics add signal: `True`; survival beats direct classifier: `False`.
- `permanent_outbound` at `36m`: `descriptive_or_exploratory_only`; weighted-origin Brier 0.2280, weighted-origin AUC 0.681, baseline improvement 0.02554 (95% CI [0.01517, 0.03599]); contract economics add signal: `True`; survival beats direct classifier: `False`.

Use the decision summary, comparisons, risk bands, subgroup results, and independent verification before making product claims.

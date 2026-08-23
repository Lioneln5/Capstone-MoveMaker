# Extension opportunity and role-trajectory diagnostic

This Phase-1 diagnostic tests whether future post-extension involvement is measurable and predictively supportable. It is not a production scoring model.

## Pre-registered cohorts and targets

- Primary Year-2 cohort: 1,385 extensions with linked identity, at least 10 club games in exact pre-365 and Year-2 windows, and a scheduled expiration no earlier than day 730.
- Initial-role cohort: 922 primary events with pre-extension opportunity share of at least 25%.
- Continuous outcome: exact Year-2 opportunity share.
- Major role decline: pre-share >=25%, absolute decline >=20 points, and Year-2 share <=50% of pre-share.
- Sustained meaningful contribution: opportunity share >=25% in both Year 1 and Year 2.
- Contract-covered minimal involvement: Year-2 share <10% while the uninterrupted same-club spell survives 730 days.

## Validation

Four rolling origins evaluate signing years 2020-2023. Hyperparameters are selected only on the immediately preceding validation year. Paired player-cluster bootstraps use 5,000 repetitions.

## Decisions

- `year2_opportunity_share`: `advance_predictive_candidate`; total loss improvement 0.058634, 95% CI [0.048506, 0.068649], 4/4 origin wins; contract economics add signal: `True`.
- `major_role_decline`: `descriptive_only_no_validated_incremental_signal`; total loss improvement 0.001361, 95% CI [-0.008455, 0.011284], 2/4 origin wins; contract economics add signal: `True`.
- `sustained_meaningful_contribution`: `advance_predictive_candidate`; total loss improvement 0.064251, 95% CI [0.051333, 0.076891], 4/4 origin wins; contract economics add signal: `True`.
- `contract_covered_minimal_involvement`: `exploratory_signal_calibration_or_ranking_failed`; total loss improvement 0.012042, 95% CI [0.003419, 0.020918], 4/4 origin wins; contract economics add signal: `False`.

See the decision summary, comparison results, risk bands, subgroup table, target audits, feature manifest, and independent verification before making product claims.

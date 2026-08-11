# MoveMaker transfer compatibility targets — first version

## Grain and cohort

- Target rows: 936
- Grain: one accepted FBref destination player-club-competition-season with one selected inbound transfer event.
- When multiple inbound events map to the same destination stint, the earliest event is selected and later events are counted in `later_same_destination_events_collapsed`.
- Target season assignment uses June–December as the season beginning that calendar year and January–May as the preceding season start.

## Three targets

1. `target_destination_minutes_share`: destination minutes divided by the destination club-season maximum observed player appearances × 90.
2. `target_role_performance_index`: current destination-season role-adjusted FBref performance index.
3. `target_performance_change_rolling3`: current role index minus the minutes-weighted prior t-1/t-2/t-3 role index.

## Primary cohort policy

Primary eligibility requires a June–October arrival, canonical transfer type `transfer` or `loan`, an outfield role, and strict compatibility features. Performance and adaptation additionally require at least 450 destination minutes. Adaptation also requires at least 450 prior-baseline minutes.

Winter and off-window moves remain in the target table for sensitivity analyses, but full-season minutes share is not directly comparable for those partial-season arrivals.

## Leakage policy

- `transfer_compatibility_strict_features.csv` contains no `target_` columns and no `roster_` columns.
- `transfer_compatibility_strict_model_matrix.csv` adds explicitly prefixed targets and eligibility controls.
- Never use `target_`, `eligible_`, or exclusion-reason fields as predictors.

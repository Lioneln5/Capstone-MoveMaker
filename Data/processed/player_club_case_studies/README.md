# MoveMaker historical player-club case studies

This layer converts the verified rolling-origin predictions into twelve leakage-safe historical transfer cases. The selection protocol is deterministic and intentionally includes successes, warnings, target trade-offs, and model failures.

## Primary cohort

- Mature rolling origins: origin_2020, origin_2021, and origin_2022.
- A transfer must have both opportunity and performance predictions under the frozen subgroup-phase candidates and their same-family player-history baselines.
- Opportunity candidate: full explicit fit plus gradient-boosted trees.
- Performance candidate: full explicit fit plus gradient-boosted trees.

## Case categories

- Confirmed positive fit: both forecasts move upward and both become more accurate.
- Confirmed warning: both forecasts move downward and both become more accurate.
- Validated target trade-off: opportunity and performance move in opposite directions and both become more accurate.
- False positive: both forecasts move upward and both become less accurate.
- False warning: both forecasts move downward and both become less accurate.

## Interpretation

Compatibility lift is the frozen compatibility prediction minus the same-family player-history prediction. Positive means destination context raised the forecast; negative means it lowered the forecast. The player/team style and historical preference profiles are descriptive diagnostics, not causal explanations or exact model contributions.

StatsBomb Open Data coverage is sparse and every candidate identity remains marked for manual review. It is reported only as a supplementary availability flag and never affects case selection or primary conclusions.

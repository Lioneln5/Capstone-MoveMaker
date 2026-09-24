# Engine V2 repaired contribution reliability — Run 3

Run 3 audits the fixed Run 2 compact candidate. It searches no new feature or
model family, opens no 2024+ outcome, writes no model binary, and changes no
deployment file.

## Result

- Selected calibration: `raw`
- Pooled Brier: 0.196245
- Pooled ECE: 0.035375
- Calibration gap: 0.011492
- Calibration slope: 1.051675
- Reliable origins: 4/4
- Positive-skill origins: 4/4
- Refused profiles: 4.56%
- Mean / P90 80% parameter-interval width: 0.1407 / 0.1772
- Supported broad league/position groups: 6/9
- Run 3 decision: `blocked_reliability_repair`

The two recent-involvement inputs have pooled Spearman correlation
0.948. The explanation policy is therefore
`group_only_recent_involvement`: they may be described together as recent involvement,
not as independent causal effects or separately stable importance values.

The complete tables retain calibration alternatives, bins, parameter
uncertainty, OOD/refusal reasons, 99% player-cluster subgroup intervals,
temporal drift, coefficient stability, redundancy ablations, monotonicity
stress tests, missingness behavior, hashes, and the final gate.

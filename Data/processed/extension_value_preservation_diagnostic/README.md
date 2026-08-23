# Extension market-value preservation diagnostic

This Phase-3 diagnostic tests 12- and 24-month Transfermarkt-style market-value change after an extension. It does not estimate sale proceeds, profit, accounting value, or ROI.

## Targets

- Continuous: log(post-extension value / signing value).
- Ordered downside: declines of at least 10%, 25%, and 50%.
- Value preservation is the complement of at least 10% downside: retaining at least 90% of signing value.
- Signing landmarks may be at most 365 days from the signing date; post landmarks may be at most 90 days from their target dates.

## Cohort and validation

- 12-month cohort: 2,162; 24-month cohort: 1,944.
- Four rolling origins evaluate 2020-2023; tuning occurs on the immediately preceding validation year.
- Paired uncertainty uses 5,000 player-cluster bootstrap repetitions.

## 12- and 24-month decisions

- `value_log_ratio_12m`: `advance_broad_value_range_candidate`; MAE 0.4636, Spearman 0.607; baseline improvement 0.07819 (95% CI [0.05072, 0.10636]); contract economics add signal: `False`.
- `downside_10pct_12m`: `advance_downside_probability_candidate`; Brier 0.1901, AUC 0.786; baseline improvement 0.05742 (95% CI [0.04761, 0.06765]); contract economics add signal: `True`.
- `downside_25pct_12m`: `advance_downside_probability_candidate`; Brier 0.1713, AUC 0.781; baseline improvement 0.04571 (95% CI [0.03476, 0.05696]); contract economics add signal: `True`.
- `downside_50pct_12m`: `descriptive_or_exploratory_only`; Brier 0.0976, AUC 0.778; baseline improvement 0.01158 (95% CI [0.00362, 0.01973]); contract economics add signal: `True`.
- `value_log_ratio_24m`: `advance_broad_value_range_candidate`; MAE 0.6208, Spearman 0.663; baseline improvement 0.20708 (95% CI [0.17091, 0.24247]); contract economics add signal: `False`.
- `downside_10pct_24m`: `advance_downside_probability_candidate`; Brier 0.1654, AUC 0.837; baseline improvement 0.08290 (95% CI [0.07090, 0.09469]); contract economics add signal: `False`.
- `downside_25pct_24m`: `advance_downside_probability_candidate`; Brier 0.1717, AUC 0.821; baseline improvement 0.07806 (95% CI [0.06582, 0.08970]); contract economics add signal: `False`.
- `downside_50pct_24m`: `advance_downside_probability_candidate`; Brier 0.1646, AUC 0.805; baseline improvement 0.05506 (95% CI [0.04364, 0.06647]); contract economics add signal: `False`.

Probability thresholds were fit independently for signal testing. Inspect the ordering audit and apply validation-only calibration plus monotone ordering before any joint live display.

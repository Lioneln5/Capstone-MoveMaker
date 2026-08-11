# Market-value downside calibration and ordinal-consistency experiment

The frozen 11-feature model is unchanged. Calibration is learned only from the prior validation season and evaluated on the next season.

## Selected strategies

- `origin_2019`: `raw_projected` (selector mean Brier 0.132719).
- `origin_2020`: `raw_projected` (selector mean Brier 0.136783).
- `origin_2021`: `platt_projected` (selector mean Brier 0.141958).
- `origin_2022`: `empirical_distributional` (selector mean Brier 0.147360).

## Decision

Raw independent classifiers contain 157 threshold-order violations; the selected policy contains 0.

- `downside_10pct`: Brier improvement 0.004812, 95% CI [0.002717, 0.006908], 3/4 origin wins, AUC change -0.000287, ECE 0.084993 -> 0.057720; `replace_with_selected_calibrated_policy`.
- `downside_25pct`: Brier improvement 0.006130, 95% CI [0.003854, 0.008335], 3/4 origin wins, AUC change 0.000589, ECE 0.081657 -> 0.053656; `replace_with_selected_calibrated_policy`.
- `downside_50pct`: Brier improvement 0.004184, 95% CI [0.002535, 0.005863], 3/4 origin wins, AUC change -0.000472, ECE 0.069910 -> 0.049792; `replace_with_selected_calibrated_policy`.

Calibration can improve probability accuracy but cannot create new ranking information. If the replacement gate fails, retain the frozen model for ranking and communicate ordered risk bands rather than exact probabilities.

# MoveMaker component-based capital-at-risk integration

This layer combines frozen out-of-season component scores. It does not train a new raw-feature joint model and does not output expected financial loss or ROI.

## Cohort

- Strict audited overlap: 827 paid transfers; 110 suffered both >=25% estimated-value downside and <10% opportunity.
- Fully out-of-season component overlap: 446 transfers and 397 players; 78 severe dual events (17.5%).

## Frozen integration

- Value component: calibrated probability of >=25% estimated market-value downside.
- Severe opportunity component: ordered under-10% risk score, used only as a rank.
- Meaningful opportunity component: calibrated under-25% probability.
- Joint priority: geometric mean of within-origin component percentiles, forcing both components to matter.
- Proposed/reported fee and fee-benchmark premium are displayed separately. They do not alter the primary joint risk score.

## Result

- Severe value-only AUC: 0.557.
- Severe under-use-only AUC: 0.530.
- Severe geometric joint AUC: 0.604, improvement over value +0.047, 95% CI [-0.033, +0.124], 3/4 origin wins.
- Median-cut high/high quadrant: 59 rows, 28.8% severe dual-event rate versus 17.5% overall.
- Decision: `useful_matrix_but_incremental_gain_uncertain`.

## Interpretation

Use the matrix to distinguish value risk, opportunity risk, and deals exposed to both. The combined ranking is a triage aid: its incremental AUC gain is not strong enough to claim a calibrated joint probability. Keep the underlying component outputs visible and explain which dimension drives each flag.

Any euro-weighted `capital_priority_index` is reported fee multiplied by a rank-based priority score. It is not expected loss, recoverable value, total deal cost, profit, or ROI.

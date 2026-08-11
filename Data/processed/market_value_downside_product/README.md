# MoveMaker calibrated market-value downside product layer

This directory converts the independently validated downside model into presentation-ready historical outputs without retraining or changing it.

## Product output

Each of the 2,686 out-of-season historical decision cards contains calibrated and logically ordered probabilities for estimated-value declines of at least 10%, 25%, and 50%, plus a primary 25% risk band. The lowest-risk band has a 7.1% observed 25% decline rate and the highest-risk band has a 85.7% rate.

`reported_fee_exposure_index_eur` equals a positive reported fee multiplied by the calibrated 25% downside probability. It is a prioritization index, **not** expected loss, sale proceeds, profit, or ROI.

## Artifacts

- `historical_decision_cards.csv`: complete out-of-season scoring layer.
- `risk_driver_profiles.csv`: descriptive behavior by position, age, competition, pre-value, fee evidence, and season; not causal importance.
- `case_study_portfolio.csv`: 12 balanced correct-warning, correct-low-risk, false-alarm, and missed-downside examples.
- `scoring_contract.csv`: frozen inputs, outputs, timing rules, allowed uses, and forbidden claims.
- `model_card.json`: versioned evidence and limitations.

This is an acquisition screening and due-diligence prioritization tool. It does not establish tactical fit, realized financial return, or a causal effect of joining a club.

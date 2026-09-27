# Engine V2 public-value downside cleanup

This run removes proposal-time contract fields, raw goals/assists, and advanced event statistics; evaluates only a compact player/market/recent-involvement profile; and excludes every extension signed in 2024 or later.

## Results

- **downside_25pct_12m:** AUC 0.760, Brier 0.1794, chronology improvement +0.0384, calibration gap 0.010; decision `retain_research_candidate_reliability_repair_required`.
- **downside_25pct_24m:** AUC 0.823, Brier 0.1697, chronology improvement +0.0798, calibration gap 0.008; decision `freeze_development_candidate_waiting_pristine_later_test`.

## Interpretation boundary

The output estimates whether Transfermarkt's public market-value estimate falls by at least 25% after an incumbent-club extension. It does not estimate accounting impairment, realized sale proceeds, transfer ROI, or the wisdom of extending the player.

The 12-month endpoint is primary because it matures faster. The 24-month endpoint is secondary and provisional. Both available 2024+ cohorts were previously scored and their aggregate performance inspected for frozen V1, so neither can serve as an untouched final test for this repair.

No model or calibrator was serialized or deployed.

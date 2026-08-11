# MoveMaker financial module feasibility audit

This audit tests whether the current data can define sufficiently large, time-safe cohorts for commercial product modules. It does **not** establish predictive accuracy; every module that advances must still pass rolling-origin modeling against simple baselines.

## Decision

- **Advance market-value downside to modeling:** 4,371 strict 2017+ permanent transfers have positive pre-transfer value within 365 days and a positive 24-month value within +/-90 days. 35.5% declined by at least 25%.
- **Advance paid-transfer under-utilization to modeling:** 1,074 paid permanent arrivals have an interpretable 24-month destination opportunity outcome. 33.1% received under 10% of available match-minute capacity.
- **Advance fee benchmarking to modeling:** 1,486 paid permanent arrivals have a reliable positive fee and recent pre-transfer valuation. The target is reported fee, not total deal cost.
- **Test dual capital-at-risk only after its components:** 827 rows have both strict value and paid-opportunity outcomes; 13.3% meet both severe-risk thresholds.
- **Keep resale return exploratory:** only 337 fully followed acquisitions have both positive inbound and later positive outbound fees. This is conditional on a disclosed sale and is selection-biased.
- **Do not claim full ROI:** wages, bonuses, agent fees, add-ons, amortization, contract book value, and other operating costs are absent. No row is labeled as accounting ROI.

## Frozen definitions

- Primary era: transfers on or after 2017-07-01.
- Permanent arrival: canonical type `transfer`, cross-club, with no event or type conflict.
- Reliable paid fee: canonical status `positive_reported`, value > 0, and no fee conflict. Literal zero remains unclassified.
- Strict value target: positive valuation within 365 days before transfer and nearest positive valuation within +/-90 days of the 12- or 24-month target, with the horizon observable.
- Under-utilization: destination minutes divided by `90 * destination club games`; primary severe threshold is <10% over 24 months, with at least 20 destination games observed.
- Fee-only resale: first later permanent outbound from the acquired club within four years, restricted to acquisitions with full follow-up through the observed canonical transfer maximum of 2026-08-01 and positive disclosed fees on both legs.

## Feasibility gates

The audit reports a sample gate (>= 500 rows), a temporal gate (>= 75 rows in at least 3 of [2020, 2021, 2022, 2023]), an outcome-variation gate, and a target-definition gate. These gates authorize a model test, not a product-performance claim.

See `module_feasibility_summary.csv` for decisions, `module_funnels.csv` for attrition, `valuation_timing_sensitivity.csv` for strict/relaxed timing, `outcome_class_balance.csv` for thresholds, `evidence_overlap_summary.csv` for sporting/context/retention overlap, and `financial_event_audit.csv` for row-level provenance.

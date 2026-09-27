# MoveMaker business metric product-layer prototype

This package implements and verifies candidate business translations for incumbent-club extension profiles. Its formulas and response behavior are reproducible, but its card selection is not the finalized application design. A post-build review found that several valid calculations—especially salary percentile and commitment/public-value—may be too intuitive or insufficiently actionable to headline the product. It does not fit the Phase 1-4 models and it does not create an overall contract score.

## Implemented candidate outputs

- Fixed-wage commitment, commitment/public-value scale, age at expiration, and proposed wage change.
- Fixed-commitment difference from the Phase-4 peer estimate and an offer-aggressiveness percentile based on 1,504 frozen out-of-time residuals.
- Qualified low-contribution wage exposure: two-year fixed wages × modeled low-contribution probability.
- Primary paired 12-month public-value downside probability and euro threshold.
- Optional 24-month public-value downside scenarios with an enforced warning that available 2024+ outcomes were previously exposed in frozen V1 verification and that the repaired complete-feature, player-disjoint 2024 cohort contains only 26 outcomes, with as few as 3 in one Big Five league.
- Paired 36-month continuous-stay probability and scheduled post-36-month fixed wages.
- Matched known-panel club salary position and payroll share.

## Integration

Call `models.business_metric_engine.build_extension_business_profile(payload, sorted_reference_residuals)`. Load `sorted_offer_reference_residuals` from `offer_aggressiveness_reference.json`. The input JSON schema distinguishes required proposed-contract inputs from optional backend model outputs. The engine validates scope, probability bounds, horizon/threshold coherence, positive financial inputs, and missing-data behavior.

The personalized probability and Phase-4 peer inputs still require production model serialization. Until those scorers exist, use the 12 historical examples for the application layout and demonstration; do not pretend the engine can infer probabilities directly from the few visible user inputs.

Before application integration, apply the decision-utility standard documented in the repository README and scope-pivot journal: retain only outputs that reveal non-obvious information from the underlying data and can change a plausible club decision. Simple facts may remain as background contract context without being presented as MoveMaker's analytical contribution.

## Boundaries

The product is scoped to incumbent-club extensions. Low-contribution exposure is a qualified risk-weighted proxy, not expected accounting loss or wasted wages. Public-value and continuity outputs are paired scenarios, not multiplied losses. Offer percentile is historical aggressiveness, not fairness or optimality. No betting, new-club acquisition, ROI, or approve/reject claim is supported.

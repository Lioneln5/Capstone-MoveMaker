# MoveMaker contract-scope diagnostic checklist

Last updated: 2026-08-11

This checklist is an evidence-gate sequence, not a commitment to ship every
candidate output. A module advances only when its out-of-time performance,
uncertainty, calibration/ranking, and subgroup behavior support the claim.

## Phase 1 — future opportunity and role trajectory

- [x] Freeze contract-covered Year-1, Year-2, and Year-3 cohort rules.
- [x] Pre-register continuous opportunity and interpretable role outcomes.
- [x] Audit coverage, prevalence, role bands, and threshold sensitivity.
- [x] Evaluate a chronological baseline and fixed feature ladder on four rolling origins.
- [x] Isolate the incremental value of contract duration, wages, and relative financial context.
- [x] Check recent-window stability, risk-band ordering, and player-clustered uncertainty.
- [x] Independently verify targets, predictions, metrics, comparisons, decisions, and hashes.
- [x] Decision: advance Year-2 involvement range and sustained-contribution probability/tier.
- [x] Decision: reject major-role-decline probability; keep minimal involvement exploratory.

Artifacts: `Data/processed/extension_opportunity_diagnostic/`

## Phase 2 — contract survival and time to outbound movement

- [x] Freeze the event definition: permanent exit, any outbound move, and competing loan treatment.
- [x] Define time origin, censoring date, contract-expiration boundary, and renewal handling.
- [x] Audit uncensored events and at-risk counts at 12, 24, 36, and 48 months.
- [x] Compare survival/hazard models with fixed-horizon retention baselines.
- [x] Test whether wage and contract terms add stable signal beyond age, role, and history.
- [x] Evaluate calibration, horizon discrimination, risk-band ordering, and subgroup stability.
- [x] Independently verify targets, observability, probability ordering, metrics, comparisons, decisions, and hashes.
- [x] Decision: advance continuous-club survival probabilities at 24 and 36 months.
- [x] Decision: advance permanent-club survival probability at 24 months.
- [x] Decision: keep permanent-club survival at 36 months exploratory because calibration and recent-window stability fail the gate.
- [x] Decision: do not claim an exact departure date or recommended contract length; the validated scope is horizon probabilities.

Artifacts: `Data/processed/extension_survival_diagnostic/`

## Phase 3 — value preservation over the contract horizon

- [x] Freeze 12- and 24-month market-value outcomes and strict valuation-timing rules.
- [x] Separate continuous value change, 10% downside, 25% downside, and 50% severe downside.
- [x] Compare player/history baselines with wage and contract-exposure extensions.
- [x] Test age at expiration and commitment-to-market-value features without post-signing leakage.
- [x] Check calibration, ordered downside thresholds, recent-window stability, and subgroups.
- [x] Independently verify targets, metrics, comparisons, ordering violations, decisions, and hashes.
- [x] Decision: advance broad 12- and 24-month market-value preservation ranges.
- [x] Decision: advance 12-month 10% and 25% downside probabilities; keep 12-month 50% downside exploratory.
- [x] Decision: advance all three 24-month downside probabilities after calibration and monotone ordering.
- [x] Decision: contract economics adds short-term downside signal at 12 months but no validated incremental value at 24 months.
- [x] Decision: do not interpret public market-value preservation as sale proceeds, profit, accounting value, or ROI.

Artifacts: `Data/processed/extension_value_preservation_diagnostic/`

## Phase 4 — contract commitment and financial exposure

- [x] Validate annual wage, fixed wage commitment, age at expiration, club wage rank/share, and wage-change fields.
- [x] Define deterministic exposure metrics separately from predictive risk outputs.
- [x] Test annual wage, contract duration, wage-to-market-value, and fixed-wage-commitment benchmarks against historical peers.
- [x] Quantify uncertainty and identify where missing bonuses, options, clauses, taxes, signing costs, and agent fees limit the claim.
- [x] Independently verify formulas, cohorts, predictions, metrics, comparisons, benchmark bands, decisions, and hashes.
- [x] Decision: show age at expiration, fixed-wage commitment, wage/value, commitment/value, transfer premium, and fee-plus-fixed-wage commitment as input-driven calculations or explicitly labeled proxies.
- [x] Decision: advance broad historical peer ranges and offer-relative premiums/deltas for annual wage, contract duration, fixed-wage commitment, and wage-to-market-value.
- [x] Decision: do not label peer deviation as overpayment, optimal term, fair value, full acquisition cost, profit, or ROI.
- [x] Decision: keep reported bonuses optional; never interpret missing bonus, tax, fee, clause, or option fields as zero.

Artifacts: `Data/processed/contract_financial_exposure_benchmark/`

## Phase 5 — integrated player contract profile

- [x] Combine only modules that passed their individual evidence gates.
- [x] Keep deterministic calculations, predictive estimates, and unsupported context visibly separate.
- [x] Test whether integration improves decisions rather than merely producing a composite score.
- [x] Create web-app display names, plain-language explanations, ranges/tiers, and uncertainty warnings.
- [x] Run 12 historical case studies across correct, incorrect, and cross-domain-divergence categories.
- [x] Independently verify source joins, formulas, ranges, probability ordering, integration metrics, bootstraps, cases, claims, and hashes.
- [x] Decision: reject a headline overall contract score; neither equal-weight three-module integration nor opportunity-plus-outbound clears the incremental uncertainty gate.
- [x] Decision: display separate Opportunity, Continuity, Asset Value, Contract Facts, and Peer Context cards.
- [x] Decision: scope modeled outputs to incumbent-club extensions; do not apply them to new-club transfer compatibility.
- [x] Decision: deterministic app logic is ready, but personalized predictive outputs require full-history refits, serialized models, and backend feature retrieval before live scoring.
- [x] Freeze approved claims, prohibited claims, and the final product narrative.

Artifacts: `Data/processed/integrated_contract_profile/`

## Product translation audit — business-facing financial metrics

- [x] Keep exact contract calculations, historical benchmarks, model probabilities, and financial proxies as separate evidence classes.
- [x] Translate fixed commitment and historical peer deviation into concise euro, ratio, and percentile context.
- [x] Test risk-weighted low-contribution wage exposure on financially eligible out-of-time Phase-1 profiles.
- [x] Require player-clustered uncertainty, origin stability, capital-weighted probability improvement, aggregate calibration, and monotonic realized euro-exposure bands.
- [x] Decision: advance risk-weighted low-contribution fixed-wage exposure as a qualified proxy, not expected accounting loss or wasted wages.
- [x] Decision: pair public-value downside probability with the euro threshold magnitude; do not multiply them into expected cash loss.
- [x] Decision: pair 36-month continuity probability with scheduled fixed wages beyond month 36; do not multiply them into expected wage loss.
- [x] Decision: advance historical offer-aggressiveness percentile only after freezing and serializing the out-of-time residual reference distribution.
- [x] Decision: keep club salary percentile/share conditional on a matched known-salary panel.
- [x] Decision: continue rejecting an overall contract score, ROI, optimal-offer language, acquisition extrapolation, and betting claims.
- [x] Independently verify formulas, source/output hashes, out-of-time exposure metrics, bootstrap intervals, risk bands, coverage, cases, and claim boundaries.

Artifacts: `Data/processed/business_metric_translation_audit/`

## Business metric product layer — web-consumable implementation

- [x] Implement a pure extension-profile calculator with typed cards, evidence classes, components, warnings, and explicit availability.
- [x] Freeze the offer-aggressiveness empirical CDF from 1,504 historical out-of-time peer residuals.
- [x] Implement exact fixed commitment, commitment/public-value, wage change, age at expiration, and scheduled post-horizon wages.
- [x] Implement peer euro deviation, historical range position, and reproducible offer-aggressiveness percentile.
- [x] Implement qualified low-contribution wage exposure with its two-year commitment and probability components exposed.
- [x] Implement paired 25%/50% public-value scenarios without an expected-loss multiplication.
- [x] Implement paired 24/36-month continuity context without an expected-wage-loss multiplication.
- [x] Implement proposed club salary percentile and known-payroll share from a matched salary panel that explicitly excludes the player's prior wage.
- [x] Return `unavailable` for missing optional backend fields; never impute zero.
- [x] Reject new-club acquisition mode, betting mode, negative financial inputs, incoherent value probabilities, incoherent stay horizons, and an overall score.
- [x] Replay all 1,476 historical profiles with required financial inputs and generate 12 web-consumable historical examples across six outcome categories.
- [x] Independently verify formulas, empirical CDF, schemas, examples, evidence classes, invalid-input behavior, and hashes.

Artifacts: `Data/processed/business_metric_product/`

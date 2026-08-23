# Contract financial exposure and peer benchmark

Phase 4 separates exact input-driven calculations from historical extension benchmarks. It is not a full cost, fair-value, or optimal-contract engine.

## Deterministic layer

Supported calculations include age at expiration, fixed wage commitment, wage-to-market-value, commitment-to-market-value, transfer premium, and a proposed fee-plus-fixed-wage commitment proxy. See the metric catalog for exact formulas and required inputs.
Bonuses are optional and sparse. Missing bonuses, taxes, agent fees, signing fees, clauses, and options are never converted to zero.

## Historical benchmark decisions

- `annual_wage`: `advance_historical_peer_benchmark` using `B2_plus_prior_wage`; MAE 0.5193, R² 0.729, Spearman 0.889; median multiplicative error factor 1.47x; improvement vs median 0.5994 (95% CI [0.5557, 0.6427]); empirical 80% interval coverage 81.3%.
- `contract_duration`: `advance_historical_peer_benchmark` using `B5_compact_nonlinear`; MAE 0.8291, R² 0.427, Spearman 0.659; median absolute error 0.63 years; improvement vs median 0.3854 (95% CI [0.3460, 0.4237]); empirical 80% interval coverage 82.4%.
- `fixed_wage_commitment`: `advance_historical_peer_benchmark` using `B2_plus_prior_wage`; MAE 0.6881, R² 0.644, Spearman 0.832; median multiplicative error factor 1.69x; improvement vs median 0.5936 (95% CI [0.5423, 0.6455]); empirical 80% interval coverage 80.1%.
- `wage_to_market_value`: `advance_historical_peer_benchmark` using `B2_plus_prior_wage`; MAE 0.5120, R² 0.663, Spearman 0.835; median multiplicative error factor 1.45x; improvement vs median 0.4442 (95% CI [0.4024, 0.4870]); empirical 80% interval coverage 81.0%.

Peer premiums describe how unusual an offer is relative to historical extensions. They do not prove overpayment or recommend an optimal contract.

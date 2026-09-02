# Engine V2 Phase 3 — calibration and reliability

This package evaluates calibration, parameter uncertainty, subgroup reliability, and out-of-distribution refusal for the three Phase-2 candidates. No artifact was serialized or deployed.

## Endpoint decisions

- `any_outbound_24m`: calibration `raw`; Brier 0.2362; ECE 0.0321; refused-profile rate 60.7%; mean 80% parameter-interval width 0.123; supported subgroup views 88.2%; broad position/league gate failed; decision `blocked_reliability_repair`.
- `downside_25pct_24m`: calibration `platt`; Brier 0.1747; ECE 0.0708; refused-profile rate 2.3%; mean 80% parameter-interval width 0.165; supported subgroup views 82.4%; broad position/league gate passed; decision `advance_to_versioned_artifact_build`.
- `sustained_meaningful_contribution`: calibration `raw`; Brier 0.1942; ECE 0.0358; refused-profile rate 4.4%; mean 80% parameter-interval width 0.143; supported subgroup views 88.2%; broad position/league gate passed; decision `advance_to_versioned_artifact_build`.

## Interpretation

- `supported` means the profile lies inside the past-data density envelope; it does not mean the outcome is certain.
- `limited` means the profile is near the historical edge and must display an extrapolation warning and model-fit uncertainty.
- `refused` means a required feature is missing, a category is unseen, a numeric value is outside the observed range, or the profile's maximum robust feature deviation lies beyond the learned 99th-percentile threshold. V2 must not emit a probability in that state.
- Subgroup reliability is evaluated separately from individual profile support. A profile can be in-domain while its league/position/age subgroup remains statistically limited.

Passing Phase 3 permits only a separately reviewed, versioned artifact build. It does not restore public scoring.

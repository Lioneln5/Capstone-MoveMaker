# Engine V2 Phase 2 — endpoint and feature specification

This package converts the Phase-1 information boundary into model specifications and tests compact candidates. It does not deploy or serialize a model.

## Decisions

- Keep four modules separate: future role, 24-month club continuity, 24-month public-value downside, and historical wage benchmarking.
- Use a single headline threshold/horizon per predictive module rather than exposing V1's full endpoint grid.
- Exclude raw goals/assists rates. The prior role diagnostic found removal non-inferior for the headline endpoints, while role-aware variants did not reliably fix demo-player behavior.
- Exclude advanced event statistics. Earlier block tests did not establish incremental value beyond simpler information.
- Exclude proposed wage, term, and their derivatives from sporting and public-value models. Historical predictiveness can reflect club selection and negotiation; it does not make a user-controlled proposal a valid causal input.
- Keep proposed wage and term in the financial scenario layer only. The wage peer model must predict a historical benchmark without seeing the proposal it is asked to benchmark.

## Candidate results

- `sustained_meaningful_contribution`: `V2_core_involvement`; Brier improvement +0.059671 versus the chronology-only baseline, 95% cluster interval [+0.046513, +0.072587], historical-M5 comparison -0.004580, decision `advance_to_phase3_candidate`.
- `any_outbound_24m`: `V2_plus_prior_volume`; Brier improvement +0.017996 versus the chronology-only baseline, 95% cluster interval [+0.011265, +0.024646], historical-M5 comparison -0.016644, decision `advance_to_phase3_candidate`.
- `downside_25pct_24m`: `V2_core_involvement`; Brier improvement +0.080320 versus the chronology-only baseline, 95% cluster interval [+0.069426, +0.091476], historical-M5 comparison +0.002259, decision `advance_to_phase3_candidate`.

These are model-specification results, not evidence that public scoring can resume. Phase 3 must address calibration, uncertainty, subgroup/OOD refusals, and candidate artifacts under the Phase-1 request contract.

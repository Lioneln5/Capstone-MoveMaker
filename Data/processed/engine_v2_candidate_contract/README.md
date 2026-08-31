# Engine V2 Phase 4 — candidate contract

Phase 4 repairs continuity's input contract, audits the annual-wage peer benchmark, and freezes a refusal-safe response schema. It does not serialize or deploy a model.

## Decisions

- Continuity: `blocked_big_five_deployment_pending_new_context_or_untouched_holdout`. The selected compact age-curve candidate has Brier 0.2351, AUC 0.646, ECE 0.036, and refusal rate 5.6%. Its observable feature contract repairs the prior 60.7% refusal failure, but Ligue 1 still fails the predeclared subgroup-skill gate.
- Wage benchmark: `advance_to_final_holdout`. The historical peer estimate improves log-MAE over a chronology-only median by 0.599; its nominal 80% range covered 81.3% of unseen observations and refused 2.3% of profiles.
- Overall engine: `not_ready_for_deployment`. Continuity does not meet Big-Five subgroup coverage and the final later-season temporal holdout remains sealed.

## Contract rules

- Each module carries its own value, interval, support state, subgroup status, target, horizon, calibration, data vintage, artifact identity, and limitations.
- Refused or unavailable modules must return no numeric value and no interval.
- The engine cannot return an overall score or automated extend/do-not-extend recommendation.
- Contribution and outbound are not mutually exclusive in the labels: 87 overlapping historical events had both sustained contribution and an outbound event. Their probabilities must not be added or normalized.
- The wage output is historical peer context, not fair value, an optimal offer, or proof of savings.

## Holdout

The earliest candidate final holdout is extension year 2024, only after each endpoint's full outcome horizon is observable. Phase 4 does not open or inspect that holdout.

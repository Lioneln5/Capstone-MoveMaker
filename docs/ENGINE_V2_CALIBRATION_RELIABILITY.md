# Engine V2 Phase 3: calibration and reliability gate

Phase 3 asks a different question from Phase 2. Phase 2 established that the
compact candidates contained chronological signal using decision-time-valid
features. Phase 3 tests whether those signals can be expressed as probabilities
without concealing weak calibration, unstable estimates, unsupported
subgroups, or out-of-distribution inputs.

This remains a research stage. No model artifact was serialized, no API or HTML
path was changed, and public scoring remains paused.

## Reliability contract

The three Phase-2 candidates were evaluated over the same rolling 2020–2023
origins. For each origin, the base model was trained on earlier extension
events, its regularization was selected on the following year, and raw, Platt,
and isotonic probability mappings were evaluated on the next unseen year.

The audit adds four protections:

1. **Probability calibration.** Raw, Platt, and isotonic probabilities are
   compared using Brier score, log loss, expected calibration error (ECE), and
   observed-versus-predicted rate gaps. A calibrated variant must remain within
   0.002 Brier and 0.01 log loss of the raw model; near-ties favor the simpler
   method.
2. **Model-fit uncertainty.** Each out-of-time probability receives an 80%
   interval from 250 player-cluster bootstrap refits. This interval measures
   model-fit variation; it is not a prediction interval and does not represent
   every source of uncertainty.
3. **Input support.** Missing required inputs, unseen categories, numeric values
   outside the historical range, and extreme robust feature deviations trigger
   an explicit `limited` or `refused` state. A refused profile receives no V2
   probability.
4. **Subgroup reliability.** Position, league, age, and support-state results
   must have at least 50 rows, 10 events, 10 non-events, and three evaluation
   origins before they can be labeled supported. Supported groups must also
   achieve AUC at least 0.55, calibration gap at most 0.10, and positive Brier
   skill against chronology-only prevalence.

An endpoint passes the Phase-3 gate only when pooled ECE is no more than 0.08,
the refusal rate is no more than 15%, mean 80% model-fit interval width is no
more than 0.35, at least 75% of audited subgroup views are supported, and every
broad position and Big-Five league group is supported. Passing permits only a
separately reviewed, versioned artifact build; it does not authorize deployment.

## Results

| Endpoint | Selected probability mapping | Brier | ECE | Refused profiles | Mean 80% model-fit width | Phase-3 decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Sustained meaningful contribution | Raw | 0.1942 | 0.0358 | 4.4% | 0.143 | Advance to versioned artifact build |
| Any outbound within 24 months | Raw | 0.2362 | 0.0321 | 60.7% | 0.123 | **Blocked for reliability repair** |
| At least 25% public-value downside within 24 months | Platt | 0.1747 | 0.0708 | 2.3% | 0.165 | Advance to versioned artifact build |

The continuity result is the most important Phase-3 finding. Its pooled
probabilities were calibrated within the declared threshold, but the Phase-2
candidate depends on minutes in the prior two completed seasons. Those fields
are absent for much of the eligible historical cohort. Under the no-silent-
imputation rule, 947 of 1,560 evaluated profiles would have to be refused.
Good aggregate discrimination cannot make an operationally unavailable input
contract deployable.

The future-role and value-downside candidates clear the endpoint-level gate,
but their subgroup results remain qualified:

- Future role has insufficient evidence for age 34+ (`n=46`) and for refused
  profiles (`n=41`).
- Value downside is limited for players age 21 or younger and age 34+ because
  ranking within those age bands did not reach the declared AUC threshold,
  even though pooled performance passed. Refused profiles also have
  insufficient evidence (`n=31`).
- Continuity is additionally limited for Ligue 1 and age 34+ profiles, aside
  from the much larger input-availability failure.

These labels are product behavior, not footnotes. A later interface must show
supported, limited, or refused status beside each module and must never convert
missing inputs to zero or substitute a confident number for an unsupported
profile.

## What Phase 3 does not establish

- It does not validate an extend/do-not-extend recommendation.
- It does not make the three probabilities comparable or combinable into one
  score.
- It does not prove causal effects for age, league, position, or public value.
- It does not establish reliability for every age or league subgroup.
- It does not authorize the historical wage benchmark; that deterministic
  module requires its own applicability and reference-range checks.
- It does not reserve a final untouched deployment test. Calibration choices
  were specified from the rolling 2020–2023 evidence and must be frozen before
  evaluation on a later holdout.

## Reproduce and verify

```bash
python models/run_engine_v2_calibration_reliability.py
python scripts/verify_engine_v2_calibration_reliability.py
```

Compact evidence is stored under
`Data/processed/engine_v2_calibration_reliability/`. Regenerable row-level
probability tables remain local under the repository publication policy.

## Next decision

Phase 4 should not simply serialize all three candidates. It should:

1. freeze the future-role and value-downside specifications and probability
   mappings;
2. repair continuity with a feature contract that is both decision-time valid
   and broadly available, then rerun Phases 2 and 3 for that endpoint;
3. define a versioned result schema that carries probability, model-fit
   interval, support state, subgroup reliability, data vintage, and refusal
   reason together; and
4. reserve later-season data as the untouched final temporal test before any
   scoring path is enabled.

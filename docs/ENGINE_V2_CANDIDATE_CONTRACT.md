# Engine V2 Phase 4: repaired candidates and result contract

Phase 4 converts the surviving research signals into a candidate product
contract without deploying them. It repairs continuity's unusable input
requirements, independently validates the annual-wage peer benchmark, and
defines the complete response envelope a future V2 API must return.

No model artifact was serialized. The frozen V1 API, HTML, and deployment
package remain unchanged.

## Continuity repair

Phase 3 selected a continuity model that used minutes from the two prior
completed seasons. Those inputs were missing often enough that 60.7% of
historical evaluation profiles would have been refused under the no-silent-
imputation rule. Phase 4 compared four chronological candidates:

| Candidate | Pooled Brier | Improvement vs chronology | 95% player-cluster CI | Origin wins | AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prior-volume Phase-2 candidate | 0.2362 | +0.0164 | [0.0095, 0.0229] | 4/4 | 0.641 |
| Broadly available core | 0.2369 | +0.0156 | [0.0093, 0.0219] | 4/4 | 0.639 |
| Core plus nonlinear age | **0.2351** | **+0.0174** | **[0.0111, 0.0236]** | **4/4** | **0.646** |

The selected repair uses age, broad position, league, public market value,
recent same-club opportunity/appearance rates, and a squared distance from age
27. The nonlinear age term is derived solely from age at signing and captures
the observed U-shaped age relationship. It is decision-time valid and does not
restore goals, assists, post-extension outcomes, or commercial proposal terms.

The repair lowers the refused-profile rate to 5.6%, with pooled ECE 0.036 and a
mean 80% model-fit interval width of 0.115. It does **not** pass the full
Big-Five gate: Ligue 1 has AUC 0.550 but negative Brier skill (-0.0066) against
chronology-only prevalence. Phase 4 therefore blocks Big-Five continuity
deployment. It does not weaken the gate, hide the league, or create a manual
league adjustment.

## Wage benchmark

The annual-wage peer model uses player context, public value, and explicitly
available prior-wage history. Across unseen 2020–2023 extension events:

- pooled log-MAE was 0.5193 versus 1.1187 for a chronology-only median;
- the nominal 80% historical reference range covered 81.3% of observations;
- 2.3% of profiles were refused;
- every broad position and Big-Five league group passed the declared audit.

The missing-prior-wage subgroup remains limited: it had 66.5% interval
coverage. Missing prior wage is represented by an availability flag and is
never converted to zero. The benchmark can advance to the sealed final
holdout, but it must be described as historical peer context—not a fair wage,
an optimal offer, or evidence of savings.

## Versioned response contract

`models/engine_v2/result_contract.py` freezes version
`2.0.0-candidate`. Every module result must include:

- endpoint, target, horizon, value, and unit;
- `supported`, `limited`, `refused`, or `unavailable` status;
- model-fit or historical-reference interval where applicable;
- calibration method for probability outputs;
- machine-readable support/refusal reasons;
- subgroup reliability, source vintage, artifact identity, and limitations.

A refused or unavailable module must contain no numeric value and no interval.
The response contract prohibits an overall score and an automated
extend/do-not-extend recommendation.

This separation is not cosmetic. In the 1,385-event overlap cohort, 87 players
both sustained meaningful same-club contribution in Years 1 and 2 and had a
recorded outbound event within 24 months. The targets are not mutually
exclusive, so adding or normalizing their independently fitted probabilities
would impose a false relationship.

## Final holdout remains sealed

Candidate selection and calibration used evidence through evaluation year
2023. Extension year 2024 is the earliest potential final temporal holdout,
but it may be opened only after each endpoint's outcome horizon is completely
observable. No feature, calibration, support rule, or subgroup gate may change
after that holdout is opened.

## Decision

- Future role: specification frozen, pending final holdout.
- Public-value downside: specification frozen, pending final holdout.
- Wage benchmark: advances to final holdout with missing-prior-wage warning.
- Continuity: blocked from Big-Five deployment pending new context or a
  pre-frozen later holdout that resolves the Ligue 1 limitation.
- Overall Engine V2: **not ready for deployment**.

## Reproduce and verify

```bash
python models/run_engine_v2_candidate_contract.py
python scripts/verify_engine_v2_candidate_contract.py
```

The independent verifier reconstructs 33 checks from row-level evidence under
`Data/processed/engine_v2_candidate_contract/`.

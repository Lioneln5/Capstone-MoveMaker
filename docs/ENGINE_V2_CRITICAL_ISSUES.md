# ENGINE V2 CRITICAL-ISSUES REGISTER

> **RELEASE-BLOCKING REGISTER — NOT A PROGRESS LOG**
>
> An issue belongs here only when ignoring it could produce a materially
> misleading user-facing result, invalidate a core product claim, contaminate
> final evaluation, or make the engine unsafe to deploy. An issue remains here
> until it is independently verified as resolved or the affected product scope
> is explicitly removed.

## Open critical issues

| ID | Severity | Component | Status | Deployment consequence |
| --- | --- | --- | --- | --- |
| **E2-CRIT-001** | **CRITICAL / RELEASE BLOCKER** | Personalized 24-month club continuity | **OPEN** | **Engine V2 cannot claim supported Big-Five continuity scoring** |
| **E2-CRIT-002** | **CRITICAL / RELEASE BLOCKER** | Final temporal validation for predictive risk modules | **OPEN** | **Future-role, continuity, and value-downside artifacts cannot be promoted** |
| **E2-CRIT-003** | **CRITICAL / RELEASE BLOCKER** | Historical annual-wage benchmark | **OPEN** | **The current Big-Five wage benchmark cannot be serialized or deployed** |

---

## E2-CRIT-001 — Personalized continuity lacks reliable cross-league signal

### Status

**OPEN — NOT REPAIRED — RELEASE BLOCKING**

- First exposed: Engine V2 Phase 3
- Re-tested: Engine V2 Phase 4
- Affected endpoint: `any_outbound_24m`
- Affected product card: personalized club-continuity / contract-horizon
  stability
- Current deployment authorization: **none**

### The problem

The Phase-3 continuity candidate relied on minutes from the two prior completed
seasons. Those inputs were unavailable often enough that **947 of 1,560
evaluation profiles (60.7%)** would have been refused under the no-silent-
imputation rule.

Phase 4 repaired that availability failure by removing the sparse volume inputs
and adding a decision-time-valid nonlinear age term. The repaired candidate
reduced refusal to **5.6%** and retained positive pooled chronological skill.
It did **not** repair cross-league reliability:

| Evidence | Result |
| --- | ---: |
| Repaired candidate pooled Brier | 0.2351 |
| Repaired candidate pooled AUC | 0.646 |
| Pooled Brier improvement vs chronology | +0.0174 |
| Player-cluster 95% CI vs chronology | [0.0111, 0.0236] |
| Refused-profile rate | 5.6% |
| Ligue 1 evaluation rows | 299 |
| Ligue 1 AUC | 0.550 |
| Ligue 1 Brier skill vs chronology | **−0.0066** |

The repaired candidate is therefore worse in Ligue 1 than using the applicable
chronological prevalence baseline. A positive pooled average cannot override
that failed league-level gate.

The numerical improvement over the old Phase-2 prior-volume candidate is also
small: Brier changed from **0.2362 to 0.2351**. Phase 4 is a meaningful coverage
and safety repair, not evidence of a transformational predictive repair.

### Why this is critical

If this module were exposed as a supported Big-Five personalized probability,
the interface could present a precise-looking Ligue 1 continuity score even
though the model has not demonstrated positive predictive skill there. That
would be materially more confident than the evidence permits and would
undermine the central Engine V2 promise that unsupported numbers are refused
rather than cosmetically presented.

This issue blocks:

- a universal Big-Five continuity claim;
- promotion of the current continuity candidate into a V2 artifact;
- enabling the complete four-card V2 profile as though every card has passed;
- opening a final holdout and then changing continuity features in response to
  the result.

It does **not** automatically invalidate the separately modeled future-role,
public-value-downside, or historical-wage-benchmark modules.

### Prohibited workarounds

The following do **not** resolve this issue:

- weakening or removing the league subgroup gate;
- relying on pooled AUC while omitting the Ligue 1 result;
- applying a manual superstar, club, league, age, goal, or market-value bonus;
- silently substituting a historical league rate and labeling it personalized;
- silently excluding Ligue 1 while continuing to advertise Big-Five coverage;
- combining continuity with another probability to conceal the weak endpoint;
- opening the final temporal holdout before the candidate and rules are frozen;
- tuning on the final holdout after observing its results.

### Permitted temporary product behavior

Until the issue is resolved, only the following behaviors are defensible:

1. keep personalized continuity disabled for the entire V2 engine; or
2. explicitly narrow continuity to independently supported leagues and return
   `unavailable` for Ligue 1 with the machine-readable reason
   `SUBGROUP_RELIABILITY_LIMITED:Ligue 1`.

Option 2 also requires changing every product, documentation, and interface
claim from Big-Five continuity coverage to the exact supported league list.
Other independently validated cards may continue through their own gates.

### Required remediation work

The next continuity investigation must be predeclared and should test genuinely
new explanatory context rather than repeatedly rearranging the same player
profile:

- leakage-safe historical club loan and permanent-exit propensity;
- prior club extension-to-exit behavior calculated only from earlier events;
- squad depth and positional competition at the decision date;
- club/squad turnover and manager instability;
- club competitive level and trajectory;
- hierarchical league/club effects with shrinkage rather than manual offsets;
- target decomposition where substantively justified, while preserving exact
  loan/permanent definitions.

Any new source must meet the same decision-time, coverage, identity-linkage,
missingness, and historical/live semantic-parity rules as existing V2 features.

### Resolution gate

**E2-CRIT-001 may be closed only when one of these outcomes is independently
verified:**

#### Resolution A — model repaired

- Candidate features and target are frozen before final evaluation.
- Rolling chronological development results retain positive pooled Brier skill
  with a player-cluster 95% interval excluding zero.
- Every advertised league meets the predeclared minimum rows/events/origins,
  AUC, calibration-gap, and positive Brier-skill requirements.
- Refusal remains at or below 15% without silent imputation.
- A later, fully observable untouched temporal holdout passes the same gates.
- A separate verifier reconstructs the metrics and confirms V1 isolation.

#### Resolution B — product scope narrowed

- Unsupported leagues are removed from the continuity claim everywhere.
- The API contract returns no personalized value for excluded leagues.
- The interface explains the limitation before displaying any adjacent output.
- Documentation and tests prove that excluded requests cannot bypass refusal.

#### Resolution C — endpoint retired

- Personalized continuity is removed from the candidate engine and public
  claims.
- No proxy or composite score reintroduces the same unsupported claim under a
  different label.

### Evidence

- [Phase 3 reliability gate](ENGINE_V2_CALIBRATION_RELIABILITY.md)
- [Phase 4 candidate contract](ENGINE_V2_CANDIDATE_CONTRACT.md)
- `Data/processed/engine_v2_calibration_reliability/phase3_decisions.csv`
- `Data/processed/engine_v2_candidate_contract/continuity_candidate_comparison.csv`
- `Data/processed/engine_v2_candidate_contract/continuity_subgroup_reliability.csv`
- `Data/processed/engine_v2_candidate_contract/phase4_decisions.json`
- `Data/processed/engine_v2_candidate_contract/independent_verification.json`

### Closure record

Not yet eligible. When resolved, append the resolution date, commit, verifier,
artifact or scope change, and exact evidence that satisfied every applicable
closure condition. Do not delete this entry.

---

## E2-CRIT-002 — No valid final temporal test currently exists for the predictive risk modules

### Status

**OPEN — NOT REPAIRED — RELEASE BLOCKING**

- First exposed: Engine V2 Phase 5
- Affected modules: future role, club continuity, public-value downside
- Current deployment authorization: **none**

### The problem

The planned 2024 final cohort is neither uniformly mature nor uniformly
pristine. After enforcing the frozen requirement that no player appear in both
development data through 2023 and the final test, the mature 2024 cohorts are:

| Module | Eligible 2024 rows | Player-disjoint rows | Smallest league |
| --- | ---: | ---: | ---: |
| Future role | 81 | 56 | 6 |
| Club continuity | 290 | 176 | 23 |
| Public-value downside | 41 | 26 | 3 |

The future-role and continuity labels for all 81 jointly eligible 2024 cases
were also included in Phase 4's aggregate target-relationship audit. Phase 4
did not score those cases, reveal player-level outcomes, or measure 2024 model
performance, but the labels were inspected in aggregate. They are therefore
not a pristine final holdout.

Value downside remains sealed, but 26 player-disjoint cases—with as few as
three in one league—cannot support a final pooled plus Big-Five subgroup claim.

### Why this is critical

Opening these cohorts and reporting one favorable pooled number would convert
partial follow-up and tiny league samples into false finality. It would also
leave no untouched data on which to detect overfitting after candidate
selection. No predictive Engine V2 artifact can be called finally validated
under those conditions.

### Prohibited workarounds

- treating the aggregate-exposed role/continuity labels as pristine;
- keeping repeat-player events in both development and final partitions;
- pooling leagues to hide single-digit or very small subgroup samples;
- opening value-downside outcomes now and weakening gates after seeing them;
- using 2024 to redesign features, calibration, thresholds, or refusal rules;
- describing rolling 2020–2023 development evidence as a final external test.

### Required resolution

Freeze the candidate and all gates, then wait for a later cohort whose full
outcome horizon is observable. Before opening it, record source cutoffs,
eligibility, player-disjoint counts, league/position counts, and a code-search
audit proving its labels have not been inspected. The final test must pass the
same pooled, calibration, refusal, and advertised-subgroup gates without any
post-result tuning.

### Evidence

- [Phase 5 final-evaluation gate](ENGINE_V2_FINAL_EVALUATION_GATE.md)
- `Data/processed/engine_v2_final_evaluation_gate/holdout_readiness.csv`
- `Data/processed/engine_v2_final_evaluation_gate/prior_exposure_audit.csv`
- `Data/processed/engine_v2_final_evaluation_gate/independent_verification.json`

### Closure record

Not eligible. Do not close this issue merely because more calendar time has
passed; maturity, non-exposure, identity separation, and subgroup counts must
all be independently verified before the later labels are opened.

---

## E2-CRIT-003 — The annual-wage benchmark failed its sealed Big-Five holdout gate

### Status

**OPEN — FINAL CANDIDATE FAILED — RELEASE BLOCKING**

- First exposed: Engine V2 Phase 5
- Affected module: historical annual-wage peer benchmark
- Candidate: `B2_plus_prior_wage`
- Current deployment authorization: **none**

### The problem

The frozen candidate was evaluated once on 235 player-disjoint 2024 extension
events. It improved pooled log-MAE from 0.8359 to 0.5046, achieved 77.0%
coverage for its nominal 80% reference range, and refused 2.1% of cases.

It still failed the release gate. Ligue 1 reference-range coverage was only
**60.6%**, below the frozen 70% lower bound. A malformed one-row position label
was also surfaced and refused rather than silently normalized after opening
the holdout. The pooled improvement cannot override the advertised-league
failure.

### Why this is critical

Deploying the benchmark would display a precise-looking historical wage range
in a league where the interval missed too often in the final test. Widening the
range or changing the candidate using these same 2024 outcomes would tune on
the final holdout and invalidate the claimed test.

### Prohibited workarounds

- widening only Ligue 1 intervals using 2024 residuals;
- lowering the 70% coverage gate after seeing the result;
- citing only pooled MAE and omitting league coverage;
- silently mapping or dropping failed rows after the holdout was opened;
- promoting the Phase-4 candidate artifact despite the final failure;
- relabeling the output “directional” while still displaying an unsupported
  personalized range.

### Required resolution

The current candidate remains permanently failed on this holdout. A successor
must be declared as a new version, motivated without fitting to the 2024 final
labels, and evaluated on a later untouched cohort. Alternatively, narrow the
advertised scope before building an artifact and return `unavailable` outside
the independently supported league set.

### Evidence

- [Phase 5 final-evaluation gate](ENGINE_V2_FINAL_EVALUATION_GATE.md)
- `Data/processed/engine_v2_final_evaluation_gate/wage_final_holdout_metrics.csv`
- `Data/processed/engine_v2_final_evaluation_gate/wage_final_holdout_subgroups.csv`
- `Data/processed/engine_v2_final_evaluation_gate/phase5_decisions.json`
- `Data/processed/engine_v2_final_evaluation_gate/independent_verification.json`

### Closure record

Not eligible. A post-holdout patch to the same candidate does not close this
issue; only a new predeclared version with later untouched evidence, explicit
scope narrowing, or endpoint retirement can do so.

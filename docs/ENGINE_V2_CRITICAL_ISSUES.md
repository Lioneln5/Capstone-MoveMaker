# ENGINE V2 CRITICAL-ISSUES REGISTER

The twelve findings from the original post-presentation engine audit are
preserved separately in
[`ENGINE_V2_ORIGINAL_AUDIT.md`](ENGINE_V2_ORIGINAL_AUDIT.md). This register is
reserved for later findings that independently block release or invalidate a
specific product claim.

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
| **E2-CRIT-002** | **CRITICAL / RELEASE BLOCKER** | Final temporal validation for predictive risk modules | **OPEN** | **Future-role, movement/retention, and value-downside artifacts cannot be promoted** |
| **E2-CRIT-003** | **CRITICAL / RELEASE BLOCKER** | Historical annual-wage benchmark | **OPEN** | **The current Big-Five wage benchmark cannot be serialized or deployed** |

---

## Resolved critical issues

| ID | Former severity | Component | Resolution | Closed |
| --- | --- | --- | --- | --- |
| **E2-CRIT-001** | **CRITICAL / RELEASE BLOCKER** | Personalized 24-month club continuity | **Resolution C — endpoint retired and product scope replaced by separate non-deployed questions** | **2026-09-02** |

---

## E2-CRIT-001 — Personalized continuity lacks reliable cross-league signal

### Status

**RESOLVED 2026-09-02 — ENDPOINT RETIRED, NOT MODEL-REPAIRED**

- First exposed: Engine V2 Phase 3
- Re-tested: Engine V2 Phase 4
- Re-tested with new club financial context: 2026-09-01
- Re-tested with lagged club movement and mature extension history: 2026-09-01
- Re-tested with player role trajectory and squad competition: 2026-09-01
- Re-tested with manager, formation, and player-manager context: 2026-09-01
- Affected endpoint: `any_outbound_24m`
- Affected product card: personalized club-continuity / contract-horizon
  stability
- Final deployment authorization: **none; endpoint retired**

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

### Club financial context repair attempt (2026-09-01)

A preregistered rolling-origin experiment tested prior-season payroll level,
transfer intensity, squad composition, their combined block, and coverage-gated
wage concentration. None passed. Payroll and wage structure were statistically
indistinguishable from the matched Phase-4 baseline; recruitment, squad, and
combined context worsened pooled Brier. Operational refusal ranged from 16.0%
to 22.2%, above the 15% release gate.

The covered Ligue 1 subset was easier than the full cohort, but this was not a
feature repair. The locked Phase-4 model's Ligue 1 Brier skill changed from
−0.0066 on all 299 cases to −0.0009 on the 270 cases with prior club context;
the 29 missing cases had −0.0597 skill. On matched covered cases, every core
club-context block slightly worsened Ligue 1 Brier. Excluding promoted or
missing-context clubs therefore cannot be represented as repaired Big-Five
continuity.

### Movement-state decomposition finding (2026-09-01)

The blended endpoint has now been decomposed without opening the final holdout.
Temporary-first movement is materially learnable across every league (pooled
AUC 0.777, positive Brier skill in 4/4 origins, Ligue 1 AUC 0.710). A strict
joint outcome requiring both uninterrupted stay and sustained meaningful use is
also stronger (pooled AUC 0.759 and positive skill in every league).

The release blocker is more specific than previously understood: permanent
relationship separation remains below the league gate in Ligue 1 and
Bundesliga, while the three-state no-outbound/permanent separation remains weak
in Ligue 1. This supports redesigning the product contract around separate
movement states; it does **not** close the issue or authorize the diagnostic
models.

### Lagged club-behavior repair attempt (2026-09-01)

Strictly pre-decision club loan, sale, turnover, role-conditioned movement, and
mature prior-extension profiles were built and tested across the decomposed
endpoints. General and role-conditioned movement history produced small,
football-coherent temporary-movement improvements, but neither cleared the
four-block adjusted player-cluster interval. They remain research-only.

Mature prior-extension history is not a usable repair in its current form. Its
three-state Brier change was **−0.0123** with an adjusted interval entirely on
the harmful side, and 27.0% of evaluation cases were operationally refused.
The combined behavior block was also repeatedly harmful. Because the extension
panel begins in 2018, these rates accumulate structurally across origins and do
not form a stable club fingerprint. This result prohibits presenting earlier
same-club extension outcomes as a proven continuity feature or silently
combining them with the smaller temporary-movement signal.

At this stage the blocker remained open and motivated the later player-role,
squad, manager and tactical-context tests recorded below.

### Player role and squad-competition repair attempt (2026-09-01)

Decision-time role structure, role trajectory, on-field same-position
competition, and value/incoming competition profiles were built with 85.6–92.1%
availability across the canonical 2020–2023 cohort. The endpoint-linked
combined profile covered 84.5%, but operational refusal reached 18.5%.

None of 20 binary comparisons or five coherent three-state comparisons passed
the multiplicity-adjusted evidence gate. Role trajectory and incoming
same-position competition showed small, football-coherent temporary-movement
hints, but adjusted intervals crossed zero. The 24-feature combined block
worsened permanent-first, permanent-separation, and strict meaningful-stay
loss. The profiles are therefore research/descriptive infrastructure, not
validated risk modifiers.

This result also clarifies why the block did not repair continuity: the compact
baseline already contains recent same-club opportunity and appearance share,
while many new role ranks are highly correlated re-expressions of that same
involvement. Later work must avoid adding further versions of the same role
signal.

### Manager and tactical-context repair attempt (2026-09-01)

Observed manager stability, reported formation-shape stability, player role
under the current manager, and role change around the latest manager transition
were constructed from pre-signing game, lineup and appearance records. Manager
availability reached 95.1% and formation/current-manager-role availability
reached 87.9% across canonical development extensions. Independent verification
reconstructed 252 raw feature values and reproduced the full build and
experiment byte-for-byte.

None of the five predeclared blocks passed. Formation stability caused repeated
harm for temporary movement and strict meaningful stay. Manager stability
caused repeated harm for permanent-first movement and strict meaningful stay.
The only mildly positive direction was player role under the current manager
for the three-state outcome (+0.00156 multiclass Brier), but its adjusted 99%
interval ranged from −0.00259 to +0.00570. Current-manager start and opportunity
rates were also 0.95–0.96 correlated with the baseline opportunity measure.

The transition-role block was inconclusive and operationally refused 23.2% of
cases. Only 373 prior-year manager changes had sufficient covered games on both
sides for a real player-role comparison. The full 21-feature combination was
negative for every binary endpoint and refused approximately 30% of cases.

This is not a coverage or parsing loophole that permits another manual
adjustment. It is evidence that these public manager/base-formation summaries
do not provide stable incremental prediction beyond recent involvement. They
remain descriptive infrastructure only. This attempt did not repair
continuity; the later scope decision therefore retired the endpoint, and none
of these fields may be presented as a validated risk modifier.

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

### Product behavior considered before closure

Before the endpoint-retirement decision, only the following temporary
behaviors were considered defensible:

1. keep personalized continuity disabled for the entire V2 engine; or
2. explicitly narrow continuity to independently supported leagues and return
   `unavailable` for Ligue 1 with the machine-readable reason
   `SUBGROUP_RELIABILITY_LIMITED:Ligue 1`.

Option 2 also requires changing every product, documentation, and interface
claim from Big-Five continuity coverage to the exact supported league list.
Other independently validated cards may continue through their own gates.

### Required remediation work

Further work must be predeclared and must not repeat already rejected broad
feature blocks. Lagged club movement, mature extension history, squad depth,
role trajectory, manager stability and base-formation stability have now been
tested. Remaining defensible directions include:

- player-relative financial status and proposal-induced wage-tier movement;
- promoted-club transition context and prior top-flight experience;
- club competitive trajectory;
- explicitly time-safe injury availability around the decision;
- narrow interactions motivated by a concrete temporary-displacement or
  meaningful-retention mechanism; and
- hierarchical league/club effects with shrinkage rather than manual offsets.

The decomposed temporary-displacement and strict meaningful-retention targets
should remain separate. Permanent separation remains research-only until it is
repaired or retired.

Any new source must meet the same decision-time, coverage, identity-linkage,
missingness, and historical/live semantic-parity rules as existing V2 features.

### Resolution gate used for closure

**E2-CRIT-001 could be closed only when one of these outcomes was independently
verified. Resolution C was ultimately selected and verified:**

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
- [Club financial context experiment](ENGINE_V2_CLUB_FINANCIAL_CONTEXT_EXPERIMENT.md)
- [Movement-state and profile diagnostic](ENGINE_V2_MOVEMENT_STATE_AND_PROFILE_DIAGNOSTIC.md)
- [Lagged club-behavior diagnostic](ENGINE_V2_LAGGED_CLUB_BEHAVIOR_DIAGNOSTIC.md)
- [Player-role and squad-competition diagnostic](ENGINE_V2_ROLE_SQUAD_DIAGNOSTIC.md)
- [Manager and tactical-context diagnostic](ENGINE_V2_MANAGER_TACTICAL_DIAGNOSTIC.md)
- `Data/processed/engine_v2_calibration_reliability/phase3_decisions.csv`
- `Data/processed/engine_v2_candidate_contract/continuity_candidate_comparison.csv`
- `Data/processed/engine_v2_candidate_contract/continuity_subgroup_reliability.csv`
- `Data/processed/engine_v2_candidate_contract/phase4_decisions.json`
- `Data/processed/engine_v2_candidate_contract/independent_verification.json`
- `Data/processed/engine_v2_club_financial_context_experiment/selection_decision.csv`
- `Data/processed/engine_v2_club_financial_context_experiment/independent_verification.json`
- `Data/processed/engine_v2_movement_state_profile_diagnostic/independent_verification.json`
- `Data/processed/engine_v2_lagged_club_behavior_experiment/independent_verification.json`
- `Data/processed/engine_v2_role_squad_experiment/independent_verification.json`
- `Data/processed/engine_v2_manager_tactical_experiment/independent_verification.json`

### Closure record

**Closed through Resolution C on 2026-09-02.**

- Generic `any_outbound_24m` continuity is removed from the candidate product
  and public probability claims.
- Meaningful Retention and Temporary Displacement Risk are separately defined
  development candidates and remain non-deployed.
- Permanent Separation is research-only and returns no personalized value.
- The API fails closed without an environment-variable override.
- The public interface explains the scope change rather than displaying the
  old Contract-Horizon Stability card.
- `models/engine_v2/movement_product_scope_contract.py` prohibits generic
  continuity, numeric movement outputs and proxy composites.
- `scripts/verify_engine_v2_movement_scope_closeout.py` independently checks
  the contract, public copy, fail-closed API, compact evidence, clean rerun,
  sealed holdout and frozen V1 hashes.

Permanent-separation research moved to
`ENGINE_V2_MOVEMENT_RESEARCH_BACKLOG.md`. It is no longer a whole-engine
release blocker. Do not delete this entry; it records why scope retirement was
necessary.

---

## E2-CRIT-002 — No valid final temporal test currently exists for the predictive risk modules

### Status

**OPEN — NOT REPAIRED — RELEASE BLOCKING**

- First exposed: Engine V2 Phase 5
- Affected modules: future role, Meaningful Retention, Temporary Displacement,
  public-value downside
- Current deployment authorization: **none**

### The problem

The planned 2024 final cohort is neither uniformly mature nor uniformly
pristine. After enforcing the frozen requirement that no player appear in both
development data through 2023 and the final test, the mature 2024 cohorts are:

| Module | Eligible 2024 rows | Player-disjoint rows | Smallest league |
| --- | ---: | ---: | ---: |
| Future role | 81 | 56 | 6 |
| Historical blended continuity labels | 290 | 176 | 23 |
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

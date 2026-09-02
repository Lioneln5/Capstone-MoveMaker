# Engine V2 movement-state and profile diagnostic

**Status:** development diagnostic complete; target decomposition supported for
further research; no candidate promoted or deployed.

## Why this diagnostic was run

The original `any_outbound_24m` endpoint combined permanent transfers, loans,
loan returns, and unresolved movement records. Those events do not represent a
single football decision. This diagnostic preserved the raw labels, tested a
coherent three-state first-movement model, and separately tested a strict
sporting outcome requiring both uninterrupted stay and meaningful involvement.

Only the established 2020–2023 rolling development origins were evaluated. The
2024+ final holdout remained sealed.

## What the target contains

| First movement within 24 months | Rows | Share |
| --- | ---: | ---: |
| No outbound movement | 772 | 49.5% |
| Permanent transfer first | 451 | 28.9% |
| Loan first | 295 | 18.9% |
| Loan return first | 35 | 2.2% |
| Unresolved first type | 7 | 0.4% |

The three-state diagnostic groups loan and loan-return as
`temporary_first_24m` but retains their raw counts. The seven unresolved events
are excluded rather than silently assigned.

The direct permanent-relationship endpoint is not identical to
`permanent_first_24m`: 88 loan-first cases also had a permanent departure within
24 months, and five unresolved-first cases had a permanent departure. This is
why first-event state and eventual permanent separation answer different
questions.

## Relationship with meaningful contribution

| First movement | Sustained contribution rate |
| --- | ---: |
| No outbound | 62.4% |
| Loan first | 5.1% |
| Loan return first | 25.7% |
| Permanent transfer first | 12.9% |
| Unresolved | 0.0% |

The distinction is substantively strong. A temporary interruption is not just a
noisy version of permanent departure, and contract retention is not equivalent
to sporting use.

## Rolling-origin results

All models below use only the compact Phase-4 player/context inputs: age,
position, league, market value, recent opportunity and appearance share, plus
the nonlinear age term.

| Endpoint | Event rate | AUC | Brier skill vs chronology | Player-cluster 95% CI | Origin wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Any outbound within 24m | 50.5% | 0.646 | +0.0174 | [+0.0110, +0.0239] | 4/4 |
| Permanent relationship ended within 24m | 34.9% | 0.628 | +0.0127 | [+0.0055, +0.0200] | 4/4 |
| Temporary movement first | 21.2% | **0.777** | **+0.0291** | **[+0.0209, +0.0375]** | 4/4 |
| Strict meaningful stay | 30.9% | **0.759** | **+0.0352** | **[+0.0260, +0.0443]** | 4/4 |

`Strict meaningful stay` means no outbound event within 24 months and at least
25% same-club opportunity share in both Year 1 and Year 2.

The coherent three-state model—no outbound, temporary first, permanent
first—also beat the rolling chronology baseline in all four origins:

- 1,553 modelable evaluation events;
- multiclass Brier improvement: **+0.0615**;
- player-cluster 95% interval: **[+0.0450, +0.0772]**;
- multiclass log loss: **0.941** versus **1.045**;
- macro one-vs-rest AUC: **0.695**.

These are development results, not a release authorization. The multiclass
candidate has not yet passed final calibration, refusal, uncertainty, and
untouched-holdout gates.

## The Ligue 1 failure is now localized

| Ligue 1 endpoint/state | AUC | Brier skill vs chronology |
| --- | ---: | ---: |
| Temporary first | **0.710–0.716** | **+0.0156 to +0.0172** |
| Strict meaningful stay | **0.681** | **+0.0070** |
| Permanent relationship ended | 0.532 | **−0.0039** |
| Permanent first | 0.561 | **−0.0018** |
| No outbound | 0.543 | **−0.0114** |

The original Ligue 1 problem is not that every continuity-related outcome is
unpredictable. Temporary displacement and meaningful stay retain positive
skill. The unresolved problem is distinguishing permanent separation from
continued club relationship in Ligue 1. Bundesliga also has slightly negative
skill for the direct permanent-relationship endpoint.

This strongly supports separating movement mechanisms instead of continuing to
optimize one blended `any_outbound` probability.

## Club financial aggregates still do not repair the problem

The six prior-season payroll, transfer-intensity, squad-age, and foreign-share
features were retested on each decomposed endpoint using identical eligible
rows:

| Endpoint | Brier change from added club context |
| --- | ---: |
| Any outbound | −0.00144 |
| Permanent relationship ended | **−0.00183** |
| Temporary first | **−0.00383** |
| Strict meaningful stay | −0.00199 |

All estimates are harmful; permanent and temporary intervals exclude zero on
the harmful side. Aggregate club size/activity is therefore not a substitute
for player-relative squad status or historical club behavior.

## Initial data-use finding

At the time of the initial decomposition, no. The compact development baseline
used only nine distinct input fields. Much of the existing data supported
identity, eligibility, outcomes or historical research, while several football
profiles with plausible decision relevance had not yet been constructed for
the extension engine. The follow-up records below document which of those gaps
were subsequently built and tested.

This does **not** justify adding every raw column. The missing step is to convert
dated observations into coherent, decision-time-valid profiles.

### Initial P0 profile map

1. **Player role trajectory** — start/bench share, captaincy, positional
   entropy, role changes, and short-versus-long trend. Dated lineup coverage is
   88.3% of development extensions; 75.2% have at least ten prior-year lineup
   rows.
2. **Player-relative squad competition** — same-position minutes/start rank,
   competitor count, age/value/wage gaps, and incoming positional competition.
   The raw lineups, appearances, valuations, and salaries exist; the
   decision-date squad profile does not.
3. **Lagged club movement behavior** — prior loan-out rate, permanent-sale rate,
   extension-to-loan/sale rate, and age/position-conditioned turnover from 1.1M
   canonical transfer events and earlier extension histories.
4. **Manager and tactical stability** — manager tenure/change count, formation
   stability, and matches under the current manager. Pre-signing manager and
   formation coverage is approximately 96%; 91% have at least ten manager-coded
   games.
5. **Player-relative financial status** — club and same-position wage rank,
   value/payroll share, wage-to-minutes mismatch, and proposal-induced wage-tier
   movement. At least 15 prior-season wages exist for 85.4% of development
   extensions.
6. **Promotion transition** — second-tier payroll/transfer level, promotion
   flag, squad churn, and prior top-flight experience. Current Big-Five-only
   context is systematically missing for promoted clubs.

### P1 profiles after the core build

- valuation momentum, volatility, staleness, peak drawdown, and squad-relative
  value rank from 1.4M dated valuations;
- historical injury burden from the larger 143K ID-linked injury source;
- club competitive trajectory from standings and 177K club-game rows;
- role-normalized advanced style from the accepted 17,790-row FBref panel;
- dated teammate/cohesion networks reconstructed from matches rather than the
  unsafe career-aggregate teammate snapshot;
- explicitly proposal-conditioned contract scenarios.

### What should remain case-study-only

StatsBomb contains extensive event and spatial information, but modern Big-Five
coverage is selective and only 0.18% of development extensions have at least
three unreviewed prior-year same-club matches. It is valuable for examples and
mechanism studies, not a universal Big-Five production feature source.

## Original follow-up sequence

This was the preregistered sequence at the time of the diagnostic. Steps 1–4
were subsequently completed and rejected as predictive repairs; their evidence
is preserved below. The sequence is retained as analytical history, not as an
active to-do list.

1. Freeze the three-state movement semantics and separately retain strict
   meaningful stay.
2. Build lagged club loan/sale/extension behavior because it directly matches
   the decomposed outcomes.
3. Build player role trajectory and position-specific squad competition.
4. Add manager/tactical stability.
5. Add player-relative financial status.
6. Resolve promoted-club context before claiming universal Big-Five coverage.

Each block should be tested separately, then in a preregistered combined model.
The aim is not to maximize columns; it is to represent the football mechanisms
that lead to temporary displacement, permanent exit, or meaningful retention.

### Follow-up status: movement semantics and club behavior

Steps 1 and 2 were completed on 2026-09-01. The movement definitions are now a
versioned machine-readable contract. General and role-conditioned club movement
history produced a small but inconclusive temporary-movement signal. Mature
prior-extension rates caused repeated harm and 27–28% operational refusal
because the history panel accumulates structurally over calendar time.

The current mature-extension feature specification is rejected. General and
role-conditioned club movement features remain research-only inputs for a
later combined test. The next build is player role trajectory plus
player-relative squad competition. See
[`ENGINE_V2_LAGGED_CLUB_BEHAVIOR_DIAGNOSTIC.md`](ENGINE_V2_LAGGED_CLUB_BEHAVIOR_DIAGNOSTIC.md).

### Follow-up status: role trajectory and squad competition

Step 3 was completed on 2026-09-01. Four point-in-time profile blocks and one
combined block were tested against all decomposed movement outcomes. All 25
comparisons remained inconclusive after the frozen five-test correction. Small
temporary-movement hints from role trajectory and incoming positional
competition did not clear zero, while the full 24-feature block increased
permanent-separation and meaningful-stay loss and exceeded the 15% refusal
gate.

The lack of incremental gain is substantively interpretable: the compact
baseline already includes recent same-club opportunity and appearance share,
and the new current-role and squad-rank measures are strongly correlated with
those facts. The profiles remain useful descriptive infrastructure, but no
role/squad feature is authorized as a risk modifier. The next new mechanism is
manager/tactical stability. See
[`ENGINE_V2_ROLE_SQUAD_DIAGNOSTIC.md`](ENGINE_V2_ROLE_SQUAD_DIAGNOSTIC.md).

### Follow-up status: manager and tactical context

Step 4 was completed on 2026-09-01. Manager stability, reported base-formation
stability, player role under the current manager and role change around the
latest observed manager transition were constructed and independently
verified. The simpler blocks had good source coverage, so this was not merely a
missing-data failure.

None produced reliable incremental prediction. Manager and formation
stability caused repeated harm on several binary endpoints. Current-manager
role was directionally positive only for the three-state model, by +0.00156
multiclass Brier, with an adjusted interval crossing zero; its start and
opportunity measures were 0.95–0.96 correlated with the baseline opportunity
share. The transition block was inconclusive and operationally refused 23.2%
of cases. These profiles remain descriptive/research context, not validated
risk modifiers. See
[`ENGINE_V2_MANAGER_TACTICAL_DIAGNOSTIC.md`](ENGINE_V2_MANAGER_TACTICAL_DIAGNOSTIC.md).

## Evidence

- `Data/processed/engine_v2_movement_state_profile_diagnostic/state_definition_audit.csv`
- `Data/processed/engine_v2_movement_state_profile_diagnostic/binary_endpoint_pooled_metrics.csv`
- `Data/processed/engine_v2_movement_state_profile_diagnostic/binary_endpoint_subgroup_metrics.csv`
- `Data/processed/engine_v2_movement_state_profile_diagnostic/movement_state_pooled_metrics.csv`
- `Data/processed/engine_v2_movement_state_profile_diagnostic/movement_state_class_metrics.csv`
- `Data/processed/engine_v2_movement_state_profile_diagnostic/profile_opportunity_register.csv`
- `Data/processed/engine_v2_movement_state_profile_diagnostic/independent_verification.json`

Independent verification passes 28/28 checks, including exact named-class
probability reconstruction, a clean byte-reproducible rerun, and all 24 frozen
V1 hashes.

The final product-scope decision is recorded in
[`ENGINE_V2_MOVEMENT_SCOPE_CLOSEOUT.md`](ENGINE_V2_MOVEMENT_SCOPE_CLOSEOUT.md):
generic continuity is retired; Meaningful Retention and Temporary Displacement
remain non-deployed development candidates; Permanent Separation is
research-only and unavailable.

# Engine V2 player-role and squad-competition diagnostic

**Status:** decision-time profiles built and independently verified; all tested
predictive blocks remain inconclusive; nothing promoted or deployed.

## Question

The continuity engine previously treated recent involvement mostly as an
individual player property. This diagnostic tested a more football-specific
idea: does an extension decision become more predictable when the model knows
whether the player is gaining or losing a role and where that player sits
relative to same-position teammates?

The build used only evidence available at the signing date. It did not read a
2024+ outcome or write a model binary.

## Profiles created

All windows are left-closed and right-open, so signing-day and post-signing
records cannot enter.

| Block | Examples | Development availability |
| --- | --- | ---: |
| Current role structure | prior-365-day selection, start, bench and captain rates; observed-position entropy; recency of last selection | 92.1% of 1,602 canonical extensions |
| Role trajectory | recent-180-day start rate and recent-versus-prior changes in starts, selections and opportunity share | 85.6% |
| On-field positional competition | active same-position competitors, strong competitors, start/minutes percentile, gap to leader, minutes share and concentration | 92.0% |
| Positional value and incoming competition | same-position value percentile, value relative to positional median, and incoming/higher-valued same-position players | 88.2% |

For the actual endpoint-linked development frame, coverage was 94.7%, 88.0%,
94.5%, and 90.7%, respectively. The combined 24-feature profile was available
for 84.5%.

A zero role or competitor count is accepted only after the club's underlying
game and lineup/appearance coverage passes the 80% source gate. Otherwise the
feature block is unavailable. Active competitors require at least three
prior-year selections. Player and teammate values must be known on or before
the decision and no more than 365 days old.

## Evaluation design

Five blocks were frozen before the outcomes were evaluated:

1. current role structure;
2. role trajectory;
3. on-field same-position competition;
4. value and incoming same-position competition; and
5. their combined 24-feature profile.

Each block was added to the compact Phase-4 baseline and compared with a fresh
baseline trained on the exact same feature-eligible rows. Four rolling origins
evaluated 2020, 2021, 2022, and 2023. Binary tests covered temporary movement
first, permanent movement first, eventual permanent separation, and strict
meaningful stay. The same blocks were also tested against the coherent
three-state movement outcome.

The primary comparison was paired Brier loss. Five candidate tests within each
endpoint family used a 99% player-cluster bootstrap interval. Unavailable and
unsupported profiles count as operational refusals rather than being silently
imputed.

## Results

### Best directional results

| Endpoint | Block | Paired Brier improvement | Adjusted 99% interval | Origin wins | Operational refusal |
| --- | --- | ---: | ---: | ---: | ---: |
| Temporary first | Combined | +0.00090 | [−0.00180, +0.00368] | 2/4 | 18.5% |
| Temporary first | Value/incoming competition | +0.00067 | [−0.00151, +0.00298] | 2/4 | 11.2% |
| Temporary first | Role trajectory | +0.00022 | [−0.00093, +0.00130] | 3/4 | 15.8% |
| Strict meaningful stay | Role trajectory | +0.00006 | [−0.00308, +0.00338] | 2/4 | 15.8% |

These are hypotheses for later redesign, not passed candidates. Every interval
contains zero.

### Broad result

- **20 of 20 binary block/endpoint comparisons were inconclusive** under the
  frozen evidence rule.
- **5 of 5 three-state comparisons were inconclusive.** Current-role and
  value/incoming blocks were directionally positive by +0.00167 and +0.00141
  multiclass Brier, but their adjusted intervals crossed zero.
- The combined block was negative for permanent-first (−0.00289), eventual
  permanent separation (−0.00357), and strict meaningful stay (−0.00327).
  It also exceeded the 15% operational-refusal gate.
- No individual block repaired the continuity release blocker or earned a path
  to the final holdout.

## Football interpretation

The result does **not** mean player role or squad competition is irrelevant.
It says these particular one-year summaries do not add stable incremental
prediction after the baseline already knows recent same-club opportunity share
and appearance rate.

That distinction matters. Current start rate, role rank, minutes rank, and gap
to the positional leader mostly re-express the same observed involvement. In
the development data, start rate correlates 0.87 with both same-position start
and minutes percentile, and −0.92 with opportunity gap to the position leader.
Adding several versions of the same football fact creates little new
information and increases estimation variance.

The most coherent remaining hint is narrower: changes in role and incoming
same-position competition may help identify **temporary displacement**, such
as a loan used to clear a depth-chart blockage. That mechanism is different
from permanent separation. Even there, the measured gains are tiny and not
reliable enough to expose to users.

There are also football mechanisms this snapshot cannot observe. A player may
be extended precisely because the club intends to preserve or expand a role;
an incoming competitor may be rotation depth rather than a replacement; a
high-value player can be both important to the club and attractive to buyers.
Manager changes, injuries, tactical fit, player preference, and contract terms
can reverse the apparent meaning of the same depth-chart state. Those omitted
mechanisms help explain why plausible profile coefficients do not translate
into stable out-of-time loss reduction.

## What the profiles are still good for

The verified tables are useful research infrastructure and potentially useful
descriptive context. They can support:

- role-trajectory charts and player/squad explanations;
- interaction tests with manager change, injury availability, or incoming
  player role;
- narrower temporary-movement research; and
- quality-control checks that distinguish a genuine zero role from missing
  lineup evidence.

They must not be presented as validated risk modifiers, manually added to a
score, or used to justify opening the final holdout.

## Verification

Independent verification passed **15/15** checks:

- 96 role, trajectory, competition, valuation, and incoming-transfer values
  were reconstructed directly from raw tables with maximum absolute error
  `1.11e-16`;
- binary loss and named three-state probabilities reconstructed exactly;
- clean profile and full experiment reruns were byte-identical;
- all 24 frozen Engine V1 files remained byte-identical before and after; and
- the latest evaluated outcome year remained 2023.

## Next action

Do not add more correlated role ranks to this block. Manager/tactical context
was subsequently built and tested, but it also failed to produce reliable
incremental improvement; manager and formation stability caused repeated harm
on several endpoints. If a later combined model retests these role features,
it should use a reduced, predeclared representation rather than all 24 columns.
See `ENGINE_V2_MANAGER_TACTICAL_DIAGNOSTIC.md`.

## Evidence

- `models/engine_v2/role_squad_feature_contract.py`
- `scripts/build_engine_v2_role_squad_profiles.py`
- `models/run_engine_v2_role_squad_experiment.py`
- `scripts/verify_engine_v2_role_squad.py`
- `Data/processed/engine_v2_role_squad_profiles/role_squad_profiles.csv`
- `Data/processed/engine_v2_role_squad_profiles/feature_dictionary.csv`
- `Data/processed/engine_v2_role_squad_experiment/binary_candidate_comparison.csv`
- `Data/processed/engine_v2_role_squad_experiment/movement_state_candidate_comparison.csv`
- `Data/processed/engine_v2_role_squad_experiment/independent_verification.json`

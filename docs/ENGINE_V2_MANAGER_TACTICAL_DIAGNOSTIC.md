# Engine V2 manager and tactical-context diagnostic

**Status:** four decision-time profile families built and independently
verified; none improved the movement/retention candidates reliably; nothing promoted
or deployed.

## Question

The previous role diagnostic could not observe whether a player's involvement
belonged to a stable football context. This diagnostic tested four distinct
mechanisms:

1. whether the club had stable observed management;
2. whether its reported formation shape was stable;
3. the player's role specifically under the current manager; and
4. how that role changed around the latest observed manager transition.

The manager and formation records already existed in the Transfermarkt game
table. No new source was scraped. The build used games, lineups and appearances
strictly before each extension signing.

## What was constructed

| Block | Decision-time measures | 2020–2023 availability |
| --- | --- | ---: |
| Manager stability | distinct managers and adjacent changes; current-manager match share, observed spell matches/days, and a change within 90 days | 95.1% |
| Formation/tactical stability | distinct base shapes, normalized entropy, match-to-match changes, dominant-shape share, current-shape spell, and recent-versus-prior distribution shift | 87.9% |
| Player under current manager | selection, start, opportunity and captain rates during the current observed manager spell | 87.9% |
| Role change around latest transition | transition indicator and recency; post-versus-pre change in selection, starts and opportunity share | 78.3% |

The manager label is Unicode-normalized before consecutive spells are formed.
An "observed manager spell" is the trailing run of games carrying the same
manager name; it is not an official appointment or employment-tenure record.
Textual formation modifiers are reduced to the leading numeric base shape, so
`4-3-3 Attacking` and `4-3-3 Defending` are both `4-3-3`.

Manager and formation coverage is not the principal failure. Across the 1,602
canonical development extensions, the mean prior-year evidence was 40.4 club
games, 40.4 manager-coded games and 40.1 formation-coded games. Availability
was also reasonably stable across years and leagues. Ligue 1 had the lowest
formation availability at 82.2%, but manager availability was still 93.4%.

The transition block is narrower. Among 1,523 extensions with a usable manager
history, 616 had an observed manager change in the prior year. Only 373 of
those changes had at least five covered matches on both sides, so just 60.6%
could support a real pre/post role comparison. No-change cases retain explicit
structural zeros and a separate transition indicator; missing evidence is not
silently treated as no change.

## Evaluation design

Five variants were frozen before evaluation:

1. manager stability;
2. formation/tactical stability;
3. player role under the current manager;
4. player role change around the latest transition; and
5. all 21 features combined.

Every candidate was added to the compact baseline containing age, nonlinear
age, broad position, league, market value, recent same-club opportunity share
and recent same-club appearance rate. The comparison baseline was retrained on
the exact same feature-eligible rows.

Four rolling origins evaluated 2020, 2021, 2022 and 2023. The tests covered
temporary movement first, permanent movement first, eventual permanent
relationship separation, strict meaningful stay and the coherent three-state
movement outcome. Paired Brier improvement was the primary statistic. A 99%
player-cluster bootstrap interval controlled the five predeclared comparisons
within each endpoint family. Unsupported and unavailable rows counted as
operational refusals.

## Results

### Binary outcomes

| Endpoint | Block | Brier improvement | Adjusted 99% interval | Origin wins | Operational refusal | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Temporary first | Manager stability | −0.00205 | [−0.00480, +0.00060] | 0/4 | 9.5% | Inconclusive |
| Temporary first | Formation stability | **−0.00186** | **[−0.00367, −0.00004]** | 0/4 | 15.8% | Repeated harm |
| Temporary first | Current-manager role | −0.00094 | [−0.00346, +0.00096] | 1/4 | 13.7% | Inconclusive |
| Temporary first | Transition role change | −0.00074 | [−0.00232, +0.00098] | 2/4 | 23.2% | Inconclusive |
| Permanent first | Current-manager role | +0.00038 | [−0.00165, +0.00215] | 2/4 | 13.7% | Inconclusive |
| Permanent relationship ended | Current-manager role | −0.00019 | [−0.00234, +0.00176] | 1/4 | 13.7% | Inconclusive |
| Strict meaningful stay | Manager stability | **−0.00349** | **[−0.00650, −0.00057]** | 1/4 | 9.5% | Repeated harm |
| Strict meaningful stay | Formation stability | **−0.00233** | **[−0.00440, −0.00022]** | 0/4 | 15.9% | Repeated harm |
| Strict meaningful stay | Current-manager role | −0.00255 | [−0.00597, +0.00106] | 1/4 | 13.7% | Inconclusive |
| Strict meaningful stay | Transition role change | −0.00051 | [−0.00291, +0.00195] | 2/4 | 23.2% | Inconclusive |

The fourth repeated-harm result was manager stability for permanent-first
movement: −0.00205 with a 99% interval of [−0.00395, −0.00012]. All other
binary comparisons were inconclusive. The 21-feature combined block was
negative for every binary endpoint and refused approximately 30% of cases.

### Three-state outcome

| Block | Multiclass Brier improvement | Adjusted 99% interval | Origin wins | Result |
| --- | ---: | ---: | ---: | --- |
| Manager stability | −0.00374 | [−0.00997, +0.00251] | 1/4 | Inconclusive |
| Formation stability | −0.00047 | [−0.00518, +0.00429] | 2/4 | Inconclusive |
| Current-manager role | +0.00156 | [−0.00259, +0.00570] | 3/4 | Inconclusive |
| Transition role change | +0.00008 | [−0.00336, +0.00347] | 2/4 | Inconclusive |
| Combined | −0.00261 | [−0.01092, +0.00574] | 1/4 | Inconclusive |

The current-manager role result is the only mildly encouraging direction, but
the estimate is tiny and its adjusted interval includes meaningful harm and
benefit. It is not a validated improvement.

## Football interpretation

This result does **not** establish that managers or tactics are irrelevant to a
player's future. It establishes that these particular public, pre-signing
summaries do not improve predictions reliably after recent involvement is
already known.

### Why current-manager role adds little

The baseline already knows the player's prior-year same-club opportunity and
appearance rates. Current-manager start rate correlates **0.95** with baseline
opportunity share; current-manager opportunity share correlates **0.96** with
it. The two current-manager measures correlate **0.99** with each other. These
features mostly restate the same football fact over a manager-specific window,
so they add variance more readily than new information.

### Why manager stability is not a clean player-risk variable

A manager change is a club event, not an unambiguous player-level treatment.
It can precede displacement, rescue a marginalized player, trigger an
extension, or be unrelated to the player. In the usable descriptive sample,
an observed prior-year change coincided with slightly more temporary movement
(21.5% versus 19.0%) and less strict meaningful stay (29.0% versus 33.9%). The
rolling models could not turn that broad association into stable incremental
individual prediction. Coefficient signs were also unstable across origins.

### Why base formation shape is too coarse

A reported `4-3-3` does not identify pressing behavior, possession structure,
in-match changes or a specific player's assignment. Two teams—and two
managers—can use the same base shape in fundamentally different ways. The
formation block therefore measures lineup-shape regularity, not tactical fit.
Its repeated-harm results prohibit treating visual formation stability as a
validated risk adjustment.

### Why transition deltas remain inconclusive

Real pre/post comparisons are available for only 373 changed-manager
extensions. The block also refuses 23.2% operationally once feature and
support requirements are applied. Injuries, schedule strength, transfer-window
timing and the new manager's short observation period can all alter selection
rates without representing a durable role decision. The current data cannot
separate those mechanisms reliably.

## Decision

- **Reject manager stability and formation stability as predictive risk
  modifiers in their present form.** Some tests show repeated harm.
- **Do not promote current-manager role.** Its small three-state direction is
  unverified and largely redundant with recent involvement.
- **Do not promote manager-transition role change.** Its estimate is unstable,
  its eligible changed-manager sample is limited, and refusal is too high.
- **Retain all four profile families as descriptive/research infrastructure.**
  They are useful for explanations, case studies and later interaction work,
  provided they are labeled as observed context rather than causal effects.
- **Do not restore the old personalized continuity card.** This diagnostic did
  not repair permanent separation or establish a validated all-purpose
  continuity probability.

This is evidence against continuing to add broad feature blocks merely because
they are football-plausible. Any later use should begin with a narrower
hypothesis and a materially different information source, not another combined
model containing all 21 columns.

## Verification

Independent verification passed **16/16** checks:

- 252 manager, formation, current-role and transition-role values were
  reconstructed from raw game, lineup and appearance tables;
- the maximum absolute reconstruction error was `4.44e-16`;
- both changed-manager and no-change cases were sampled;
- profile and complete experiment reruns were byte-identical;
- binary loss and named state probabilities reconstructed exactly;
- the latest evaluated outcome year remained 2023; and
- all 24 frozen Engine V1 files remained byte-identical before and after.

## Evidence

- `models/engine_v2/manager_tactical_feature_contract.py`
- `scripts/build_engine_v2_manager_tactical_profiles.py`
- `models/run_engine_v2_manager_tactical_experiment.py`
- `scripts/verify_engine_v2_manager_tactical.py`
- `Data/processed/engine_v2_manager_tactical_profiles/manager_tactical_profiles.csv`
- `Data/processed/engine_v2_manager_tactical_profiles/feature_dictionary.csv`
- `Data/processed/engine_v2_manager_tactical_experiment/binary_candidate_comparison.csv`
- `Data/processed/engine_v2_manager_tactical_experiment/movement_state_candidate_comparison.csv`
- `Data/processed/engine_v2_manager_tactical_experiment/independent_verification.json`

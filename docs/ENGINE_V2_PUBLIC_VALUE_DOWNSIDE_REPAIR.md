# Engine V2 public-value downside repair

**Run date:** 2026-09-09

**Status:** development evidence only; no model serialized or deployed

**Primary research horizon:** 12 months
**Secondary provisional horizon:** 24 months

## What was repaired

The frozen V1 value models mixed several different kinds of information. Some
variants used raw goals and assists without role context; others used proposed
contract duration, future expiration age, proposed wages, and commitment
ratios. Those proposal fields describe the scenario a user is considering, but
historical training rows contained the contract that was actually signed.
Treating the two as the same predictor lets the interface imply that changing
an offer changes a player's underlying value-downside probability even though
that causal interpretation was never established.

The repaired candidate uses only six fields known at the decision date:

- age at signing;
- broad position;
- league;
- log public market value at signing;
- prior-365-day same-club opportunity share; and
- prior-365-day same-club appearance rate.

It excludes goals, assists, advanced event statistics, salary, proposed
contract terms, and every post-extension predictor. Required inputs must be
observed; they are not silently converted to zero.

The target is one question at two horizons: did Transfermarkt's public market
value estimate decline by at least 25% from the signing-date estimate by the
12- or 24-month landmark? It is not accounting impairment, a realized transfer
fee, cash loss, expected loss, ROI, or a recommendation to extend the player.

## Leakage finding and containment

Commit `1db685f` ran the frozen V1 out-of-sample verifier on every eligible
extension signed after 2023. On 2026-08-22 it attached the realized labels,
scored the V1 artifacts, and inspected aggregate performance for 381 12-month
outcomes and 41 24-month outcomes. Phase 5 later called public value “sealed.”
That statement was wrong and is now corrected.

The repair therefore applies a permanent rule:

- 2024+ outcomes may describe the exact frozen V1 models that already scored
  them;
- they may not select features, model form, regularization, calibration,
  support rules, thresholds, or release gates for Engine V2; and
- they may not be presented as an untouched final test of the repaired model.

Deleting outputs, rolling back Git, or starting a memory-free task cannot make
inspected outcomes pristine again. The repaired run evaluates only signed
years 2020–2023. A final release decision requires genuinely later, fully
mature and previously uninspected outcomes.

## Evaluation design

Four expanding rolling origins evaluate 2020, 2021, 2022, and 2023. For every
origin, any player appearing in the evaluation year is removed from:

1. hyperparameter-training rows;
2. the preceding calibration/validation year; and
3. the final model-fitting rows.

The six-field candidate is compared with chronology-only prevalence and a
four-field player/market profile. Regularization is selected on the preceding
validation year by Brier score. Raw, Platt, and isotonic probabilities are
also compared using validation-only calibrators. Raw probabilities win the
predeclared simplicity/non-inferiority rule at both horizons.

Uncertainty includes 5,000 player-cluster loss resamples, 99% player-cluster
subgroup skill intervals, and 100 full nested refits per horizon/origin for 80%
parameter-instability intervals. The run also learns explicit supported,
limited, and refused profile regions from each origin's fitting data.

## Development results

| Evidence | 12 months | 24 months |
| --- | ---: | ---: |
| Rolling evaluation rows | 1,418 | 1,354 |
| Event rate | 31.7% | 47.6% |
| ROC AUC | 0.760 | 0.823 |
| Average precision | 0.542 | 0.783 |
| Brier score | 0.1794 | 0.1697 |
| Brier improvement vs chronology | +0.0384 | +0.0798 |
| 95% interval for chronology improvement | [+0.0287, +0.0482] | [+0.0680, +0.0916] |
| Brier improvement beyond market profile | +0.0032 | +0.0040 |
| 95% interval beyond market profile | [+0.0003, +0.0061] | [+0.0013, +0.0067] |
| Origins beating chronology | 4 of 4 | 4 of 4 |
| Calibration gap | 1.0 percentage point | 0.8 percentage points |
| Adaptive ECE | 0.0497 | 0.0642 |
| Refusal rate | 1.7% | 1.4% |
| Mean 80% refit-interval width | 13.6 percentage points | 13.1 percentage points |
| Broad league/position groups supported | 8 of 9 | 9 of 9 |

The recent-involvement block adds only a small amount beyond age, position,
league, and starting value, but its player-cluster interval remains above zero
at both horizons. Most predictive power still comes from the player/market
profile—especially age and starting public value—not from performance detail.

## Football interpretation

The model is mostly identifying exposure to public-valuation depreciation:

- Older players had higher downside risk at every origin. This is consistent
  with shortening career horizon and valuation aging; it does not claim that
  their on-field performance must already be declining.
- Higher starting public values also had higher downside risk at every origin.
  Expensive players have more nominal and reputational valuation to lose and
  may show ceiling or mean-reversion effects. This is association, not proof
  that a high valuation causes the decline.
- Greater prior opportunity share was associated with lower downside risk at
  every origin. A player already trusted for a large share of club minutes is
  less exposed than a marginal player, all else equal.
- Appearance rate and opportunity share are extremely redundant (Spearman
  0.958–0.959). Their explanation must be grouped as **recent involvement**;
  separate feature-importance or causal claims are prohibited. Appearance-rate
  coefficients also change sign at 24 months, reinforcing that boundary.

This explains why goals and assists are unnecessary here. The value signal is
not “forwards who score are safe.” It is primarily age, current valuation
level, and whether the player already holds a meaningful club role, with broad
league and position context.

## Why the horizons have different decisions

The 12-month candidate passes the pooled development gates, all five league
groups, and three of four position groups. Goalkeepers are the exception:
their Brier skill is +0.0252, AUC is 0.735, and calibration gap is 4.4 points,
but the strict 99% skill interval crosses zero [-0.0015, +0.0518]. The correct
label is limited reliability, not failure or no signal. Until repaired or
supported by more independent evidence, 12 months remains a research
candidate rather than a frozen release candidate.

The 24-month candidate clears all declared development and broad-subgroup
gates, so its specification can be frozen pending a genuinely later final
test. It is still the secondary product horizon because 24-month outcomes
mature slowly: under the repaired complete-feature rule, the exposed 2024
cohort contains only 26 player-disjoint cases and just 3 in Ligue 1. A stronger
historical metric does not erase that deployment bottleneck.

The 12-month horizon remains primary for product research because it produces
usable evidence sooner and supports a much larger future validation cohort,
not because its current model is stronger than the 24-month model.

## Defensible and non-defensible claims

Defensible now:

- a compact proposal-free profile contains repeatable rolling-development
  signal for 25% public-value downside at both horizons;
- it improves modestly beyond a basic age/position/league/value profile;
- raw probabilities are better supported than extra calibration complexity;
- 24-month development reliability is stronger than 12-month reliability; and
- public-value risk should remain separate from contribution, movement, wage,
  and the human extension decision.

Not defensible now:

- calling either repaired probability production-validated;
- citing the exposed 2024+ metrics as V2 final performance;
- deploying or serializing either repaired candidate;
- claiming 12-month goalkeeper reliability is established;
- assigning independent meaning to appearance rate and opportunity share;
- interpreting feature contributions causally; or
- translating public-value downside into guaranteed cash loss, transfer
  proceeds, accounting impairment, or an extend/do-not-extend instruction.

## Reproduce and verify

```bash
python models/run_engine_v2_public_value_downside_repair.py
python scripts/verify_engine_v2_public_value_downside_repair.py
```

The independent verifier reconstructs origin cohorts, evaluation-player
exclusions, selected regularization, raw predictions, probability rows,
metrics, exposure counts, subgroup/parameter intervals, artifact boundaries,
source/output manifests, and frozen V1 model hashes. All 34 checks pass.

Detailed local evidence is under
`Data/processed/engine_v2_public_value_downside_repair/`. Row-level predictions
remain local under the repository publication policy; compact decisions,
aggregate metrics, manifests, and verification reports are the publishable
evidence.

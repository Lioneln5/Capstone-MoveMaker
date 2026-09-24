# Engine V2 repaired contribution model — Run 2

**Run date:** 2026-09-08

**Development outcomes used:** extension events signed in 2020–2023

**2024+ outcomes inspected:** no

**Serialized candidate produced:** no
**Deployment changed:** no

## Run 2 development result

The repaired **sustained realized extension-club contribution** target retains
a repeatable pre-extension signal. The fixed compact candidate,
`B2_core_recent_involvement`, advances to a separate reliability audit. It is
not authorized for deployment and it does not recommend whether a club should
extend a player.

This result answers a deliberately narrow question: conditional on observing
an incumbent-club extension, can information available before signing
distinguish players who will record at least 25% of captured extension-club
opportunity in **each** of the next two exact years? It does not estimate a
healthy player's tactical role, explain why contribution later changed, or
measure the causal effect of the extension.

## Frozen design

The design and gates were written before repaired-target performance was read.

- Four expanding rolling origins evaluate 2020, 2021, 2022, and 2023 extension
  events. Each origin selects regularization using only the immediately prior
  validation year.
- The primary protocol removes every evaluation player from all fitting rows.
  It also removes validation players from the earlier rows used for tuning.
- Brier score is the primary loss because the proposed output is a probability,
  not merely a rank. ROC AUC, average precision, log loss, adaptive expected
  calibration error, and calibration gap are secondary diagnostics.
- Loss differences use 5,000 player-cluster bootstrap draws, preserving the
  dependence between multiple extension events for the same player.
- The model family is L2 logistic regression. No nonlinear-family search was
  conducted in this run.
- No goals, assists, advanced-event statistics, wage fields, proposed terms,
  manager/tactical features, club-behavior features, or post-extension
  predictors enter any candidate.

The candidate ladder was:

| Candidate | Information available before extension |
| --- | --- |
| `B0_chronology_prevalence` | Historical outcome rate available at that origin |
| `B1_market_profile` | Age, broad position, league, and signing-time public value |
| `B2_core_recent_involvement` | B1 plus prior-365-day same-club opportunity and appearance rates |
| `B3_plus_prior_volume` | B2 plus canonical minutes in the two earlier season summaries |

`B2` was fixed as the primary candidate. `B3` could replace it only if its
player-cluster interval was entirely positive and it won at least three of four
origins.

## Main evidence

Across 878 player-disjoint development evaluations with 438 positive outcomes:

| Comparison or metric | Result |
| --- | ---: |
| B2 Brier score | 0.196245 |
| B2 ROC AUC | 0.761841 |
| B2 average precision | 0.703974 |
| B2 adaptive ECE | 0.035375 |
| B2 absolute calibration gap | 0.011492 |
| B2 Brier improvement vs chronology | +0.054602 |
| 95% player-cluster interval | [+0.041976, +0.067080] |
| Origin wins vs chronology | 4 / 4 |
| B2 Brier improvement vs market profile | +0.017690 |
| B3 Brier improvement vs B2 | -0.000061 |
| B3-vs-B2 95% interval | [-0.002197, +0.002123] |
| B3 origin wins vs B2 | 2 / 4 |

The ordinary temporal sensitivity, which permits earlier events from players
who later reappear in evaluation, produces a similar B2 Brier score of
0.194910. The stricter player-disjoint result therefore does not depend on
repeat-player leakage.

## Football interpretation

The evidence supports three claims.

1. **Pre-extension club involvement is informative.** A player's opportunity
   and appearance rates over the immediately preceding 365 days add useful
   information beyond age, broad position, league, and public value.
2. **The signal is about current club utilization, not scoring output.** Goals
   and assists are excluded, avoiding a global scoring-rate penalty that would
   compare unlike positional roles. The compact result therefore survives
   without treating scoring as equally meaningful for a midfielder, striker,
   defender, and goalkeeper.
3. **Longer historical volume does not add established incremental value.**
   Prior-season minute summaries do not improve the primary model. They are
   also sparse—roughly half of `pre2_canonical_minutes_played` is missing in
   later-origin fitting and evaluation samples—so Run 2 rejects their inclusion
   rather than silently treating missing history as evidence.

These are predictive associations. Recent involvement may proxy the manager's
trust, squad status, fitness, contract timing, or other club knowledge that is
not directly observed. Run 2 cannot identify which mechanism causes the later
outcome.

## Stability and limitations

B2 beats chronology in every origin, but reliability is not uniform:

| Evaluation year | Rows | Brier | ROC AUC | Adaptive ECE | Calibration gap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2020 | 212 | 0.185981 | 0.798337 | 0.104848 | 0.004839 |
| 2021 | 225 | 0.187182 | 0.787617 | 0.046346 | 0.027766 |
| 2022 | 206 | 0.199515 | 0.739368 | 0.104428 | 0.025928 |
| 2023 | 235 | 0.211314 | 0.739638 | 0.080439 | 0.042427 |

The later Brier scores are weaker, and adaptive ECE exceeds 0.08 in two
origins. The favorable pooled calibration gap is not enough to erase that
variation. Calibration method, probability reliability, drift, missingness,
and out-of-distribution behavior must therefore be audited before a candidate
can be frozen.

Preliminary pooled diagnostics show positive chronology-relative Brier skill
with sufficient cases in all five Big Five leagues and all four broad position
groups. They are encouraging diagnostics, not release gates. The age-34+ group
has only 42 observations and is explicitly marked `insufficient_evidence`.

The repaired target itself is a policy-defined composite of realized use and
availability. Injury, suspension, international duty, movement, and tactical
selection can all affect realized opportunity. The output must not be renamed
“future role” or “role when available.”

## Decision and next gate

Run 2 selects `B2_core_recent_involvement` and records
`advance_to_reliability_audit`. That decision means:

- preserve this exact feature set as the compact candidate;
- assess calibration, temporal drift, missingness, subgroup uncertainty,
  coefficient stability, and refusal/OOD behavior without opening 2024+;
- freeze any surviving probability specification before a later untouched
  cohort is opened; and
- require a later mature, player-disjoint temporal evaluation before any live
  personalized probability is authorized.

Run 2 writes no model binary. Existing deployment files are hash-identical
before and after the run.

The required reliability audit is now complete. Run 3 retained the raw
probability mapping and passed pooled calibration, uncertainty, refusal, drift,
and monotonicity gates, but failed the stricter all-league/all-position gate.
La Liga, Serie A, and Midfield had positive point-estimate skill with 99%
player-cluster intervals crossing zero. The candidate is therefore blocked
from freezing while that advertised-scope uncertainty remains unresolved. See
the [Run 3 reliability result](ENGINE_V2_CONTRIBUTION_RELIABILITY_RUN3.md).

## Reproduction

```bash
python models/run_engine_v2_contribution_model_rerun.py
python scripts/verify_engine_v2_contribution_model_rerun.py
```

The complete evidence package is stored in
`Data/processed/engine_v2_contribution_model_rerun/`.

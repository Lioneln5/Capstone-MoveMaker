# Engine V2 repaired contribution reliability — Run 3

**Run date:** 2026-09-09

**Candidate:** `B2_core_recent_involvement`

**Development outcomes used:** 2020–2023 extension events

**2024+ outcomes inspected:** no

**Model or calibrator serialized:** no
**Deployment changed:** no

## Run 3 reliability result

The compact repaired-contribution candidate is reliable at the pooled level,
but it does **not** clear the declared Big-Five and broad-position subgroup
gate. Run 3 therefore records `blocked_reliability_repair`.

This is a narrower failure than “the model has no signal.” The candidate
passes 14 of 15 predeclared reliability gates. La Liga, Serie A, and Midfield
retain positive point-estimate Brier skill, usable discrimination, and modest
calibration gaps, but their 99% player-cluster Brier-skill intervals cross
zero. The current evidence cannot support the stronger claim that improvement
is dependable in every advertised league and position group.

## Frozen audit design

Run 3 searches no new feature set or model family. It begins with the exact
player-disjoint Run 2 candidate and compares only three probability mappings:
raw logistic probabilities, Platt scaling, and isotonic calibration. Each
calibrator is fitted using the preceding validation year, never its evaluation
year.

The audit includes:

- four player-disjoint rolling evaluation origins;
- pooled and origin calibration metrics and equal-frequency reliability bins;
- 250 nested player-cluster refits per origin, including training,
  regularization selection, and calibration variation;
- explicit supported, limited, and refused input states learned inside each
  historical origin;
- 99% player-cluster Brier-skill intervals for league, broad position, age,
  and support-state groups;
- temporal feature-drift and target-rate checks;
- coefficient-sign and monotonic perturbation tests;
- missing-input behavior; and
- ablations of the two highly correlated recent-involvement fields.

## Calibration and uncertainty

Raw probabilities remain selected. Platt scaling slightly improves Brier
(`0.195266` versus `0.196245`) but worsens adaptive ECE (`0.040903` versus
`0.035375`). Isotonic calibration worsens Brier and log loss materially and
produces a pooled calibration slope of only `0.262`. Under the frozen
non-inferiority, ECE, and simplicity rule, the raw mapping is the most
defensible choice.

| Reliability measure | Result | Gate |
| --- | ---: | ---: |
| Pooled Brier | 0.196245 | Diagnostic |
| Pooled ROC AUC | 0.761841 | Diagnostic |
| Pooled adaptive ECE | 0.035375 | ≤0.08 |
| Absolute calibration gap | 0.011492 | ≤0.05 |
| Calibration slope | 1.051675 | 0.75–1.25 |
| Origins meeting ECE/gap limits | 4 / 4 | ≥3 / 4 |
| Origins with positive chronology-relative Brier skill | 4 / 4 | ≥3 / 4 |
| Mean 80% model-process interval width | 0.140705 | ≤0.20 |
| 90th-percentile interval width | 0.177218 | ≤0.30 |
| Extreme scorable probabilities | 0.0% | ≤5% |

The 80% intervals measure variation in the fitted modeling process. They are
not outcome-prediction intervals and do not include every source of football,
measurement, or future-environment uncertainty.

## Input support and missingness

Forty of 878 evaluation profiles are refused, a rate of 4.56%. Seventy-one are
limited but remain scorable with an extrapolation warning. Refusal declines
from 8.02% in the 2020 origin to 0.85% in 2023 as the historical support
envelope expands.

Missing signing-time public value is the largest refusal source. It occurs in
28 evaluation rows, with only two positive outcomes. This group is too small
and too unbalanced to validate the model's imputation/missingness behavior, so
Run 3 refuses these profiles instead of silently treating the imputed median as
ordinary evidence. Other refusals arise from values beyond the historical
range or above the learned 99th-percentile robust-distance envelope.

The latest-origin population has low drift in both recent-involvement fields:
PSI is `0.024` for opportunity share and `0.056` for appearance rate. League
composition and public value show moderate, not high, drift.

## Subgroup result

Subgroup gates are evaluated on non-refused profiles. “Supported” requires at
least 75 rows, 15 events, 15 non-events, three origins, ROC AUC of at least
0.60, a calibration gap no greater than 0.10, and a **99% player-cluster**
interval for chronology-relative Brier skill entirely above zero.

| Required group | Rows | AUC | Brier skill | 99% skill interval | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Bundesliga | 182 | 0.857 | +0.0873 | [+0.0560, +0.1164] | Supported |
| La Liga | 161 | 0.687 | +0.0333 | [−0.0055, +0.0714] | Limited reliability |
| Ligue 1 | 168 | 0.724 | +0.0440 | [+0.0079, +0.0775] | Supported |
| Premier League | 164 | 0.724 | +0.0434 | [+0.0033, +0.0793] | Supported |
| Serie A | 163 | 0.704 | +0.0295 | [−0.0176, +0.0723] | Limited reliability |
| Attack | 182 | 0.743 | +0.0460 | [+0.0131, +0.0785] | Supported |
| Defender | 286 | 0.708 | +0.0369 | [+0.0093, +0.0638] | Supported |
| Goalkeeper | 113 | 0.894 | +0.1227 | [+0.0778, +0.1639] | Supported |
| Midfield | 257 | 0.689 | +0.0303 | [−0.0014, +0.0620] | Limited reliability |

The three limited groups are not shown to be harmful. Their point estimates
are positive. The failure is that the evidence is not precise enough at the
declared 99% standard to defend a universal Big-Five and all-position claim.
The age-34+ group remains insufficient with 39 non-refused rows.

## Explanation constraint

Opportunity share and appearance rate have pooled Spearman correlation
`0.948`. Dropping either produces predictions correlated `0.988–0.996` with
the full model across origins, and neither ablation is uniformly better or
worse. Both fitted coefficients remain positive in all four origins, and a
five-percentage-point increase in either field never decreases the predicted
probability in the monotonicity stress test.

This supports the combined concept **recent club involvement**, but not stable
claims about how much independent importance belongs to appearances versus
minutes. Any later explanation layer must combine the two inputs into one
group contribution. Separate causal or feature-importance narratives are
prohibited.

## Decision boundary and next work

Run 3 does not justify suppressing the model because it lacks pooled signal,
nor does it justify displaying a universal probability. The defensible status
is:

- pooled development reliability: passed;
- input support/refusal policy: passed;
- model-process uncertainty: passed;
- all-league/all-position reliability: failed;
- deployment authorization: none.

A successor reliability repair may test a preregistered partial-pooling or
interaction specification, or explicitly narrow the advertised scope. It must
be versioned as new development work because the weak groups are now known.
Whatever survives must still be frozen before a later mature, untouched,
player-disjoint temporal cohort is opened.

## Reproduction

```bash
python models/run_engine_v2_contribution_reliability_audit.py
python scripts/verify_engine_v2_contribution_reliability_audit.py
```

Evidence is stored in
`Data/processed/engine_v2_contribution_reliability_audit/`.

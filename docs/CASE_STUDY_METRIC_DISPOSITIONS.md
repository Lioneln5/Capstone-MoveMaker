# MoveMaker final metric dispositions

**Freeze:** `movemaker_case_study_metric_freeze_v1_2026-09-10`

**Scope:** incumbent-club contract-extension analytical case study
**Authorization:** no live predictive probability and no automated
extend/do-not-extend decision

## What “final” means

This is a stopping decision for the current case study, not a claim that every
metric became production-ready. A final disposition may retain a descriptive
calculation, preserve qualified research evidence, defer an unresolved
question, retire an invalid endpoint, or reject a model. Negative and limited
results are part of the project rather than loose ends to hide.

The machine-readable authority is
`Data/processed/case_study_metric_freeze/metric_dispositions.csv`. Every number
below is reconstructed from an independently verified source artifact.

## Final disposition table

| Output | Final disposition | Evidence that survives | Binding limitation / public claim |
| --- | --- | --- | --- |
| Contract facts and commitment arithmetic | **Retain — descriptive** | Six exact calculations or explicitly labelled ratios/proxies | Fixed wage is not total contract cost; no affordability, ROI, or optimality claim |
| Club salary structure | **Retain — descriptive with warning** | Salary percentile and share of the known public wage panel when matched | Public panel is incomplete and not audited payroll |
| Sustained realized contribution | **Retain — qualified research signal** | Repaired compact model: AUC 0.762, Brier 0.1962, calibration gap 0.011, positive skill in 4/4 rolling origins | Only 6/9 core league/position groups pass the frozen 99% skill gate; La Liga, Serie A, and Midfield are limited |
| Meaningful Retention | **Definition retained; model not finalized** | Coherent target: no outbound within 24 months plus at least 25% same-club opportunity share in both exact anniversary years | Earlier joint-model evidence predates the repaired contribution target and is superseded |
| Temporary Displacement Risk | **Retain — development candidate** | AUC 0.777, Brier skill +0.0291, positive in 4/4 rolling origins | No sufficiently mature, player-disjoint, untouched final cohort |
| Permanent Separation | **Research-only; unavailable** | The distinct football question is preserved | AUC 0.628 and unreliable league behavior do not justify a personalized probability |
| Generic club continuity | **Retired** | Its failure motivated the movement redesign | Loans, loan returns, and permanent exits are different mechanisms and cannot support one “stability” interpretation |
| 25% public-value downside within 12 months | **Retain — primary qualified research candidate** | Proposal-free model: AUC 0.760, Brier 0.1794, calibration gap 0.010, positive in 4/4 origins; 8/9 broad groups supported | Goalkeeper skill remains uncertain and available 2024+ outcomes were already exposed |
| 25% public-value downside within 24 months | **Freeze — secondary development candidate** | AUC 0.823, Brier 0.1697, calibration gap 0.008, positive in 4/4 origins; 9/9 broad groups supported | Only 26 player-disjoint 2024 rows, smallest league n=3, and the cohort was previously exposed |
| Annual-wage peer benchmark | **Reject current candidate** | Pooled 2024 holdout log MAE 0.505 and 77.0% empirical range coverage | Ligue 1 coverage was 60.6%; the declared Big-Five final gate failed |
| Contract-duration peer benchmark | **Frozen V1 historical evidence only** | Four rolling-origin wins and 82.4% empirical range coverage | No successful Engine V2 final evaluation |
| Fixed-wage-commitment peer benchmark | **Frozen V1 historical evidence only** | Four rolling-origin wins and 80.1% empirical range coverage | No successful Engine V2 final evaluation |
| Wage-to-public-value peer benchmark | **Frozen V1 historical evidence only** | Four rolling-origin wins and 81.0% empirical range coverage | No successful Engine V2 final evaluation |
| Integrated contract score | **Rejected** | Equal-weight three-module AUC 0.683 versus 0.688 for the opportunity component alone | It did not improve discrimination and obscured disagreement between risks |
| Historical risk-weighted review screening | **Retain — retrospective case-study result** | Across 908 historical extensions and €5.22B in two-year fixed wages, 182 risk-ranked reviews identified 79 later low-contribution outcomes versus 54 from largest-contract review: 46% more | Historical prioritization result, not causal savings, confirmed loss, or a current deployable policy |

## What can be shown publicly

The case study can display deterministic contract facts, transparently
qualified historical context, aggregate development results, rejected
hypotheses, and representative historical examples. It must not display any
current player’s model probability as an authorized live score.

MoveMaker does not decide whether a contract should be extended. A sporting
director would combine these research signals with medical information,
scouting, tactical plans, squad depth, contract clauses, and internal finance.

## Verification contract

The closeout runner reads the authoritative contribution, movement,
public-value, wage, integration, and retrospective-screening artifacts. It
fits no model and creates no serialized artifact. The independent verifier:

1. reconstructs every published metric from its source output;
2. verifies all output statuses and supersession rules;
3. confirms the seven upstream evidence families have passing independent
   verification reports;
4. checks that every frozen V1 model binary remains byte-identical;
5. separately discloses that the two live-request integrity reports were
   intentionally regenerated after HTTP scoring was disabled;
6. confirms no predictive output or automated contract decision is
   authorized; and
7. reproduces the complete freeze byte-for-byte in a clean temporary run.

Reproduce the closeout with:

```bash
python models/run_case_study_metric_freeze.py --output /tmp/movemaker_case_study_metric_freeze
python scripts/verify_case_study_metric_freeze.py
```

The runner refuses to overwrite a non-empty freeze directory. That makes this
version immutable by default; a successor requires a new version or an
explicit reviewed replacement.

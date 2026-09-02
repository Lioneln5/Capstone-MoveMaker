# Engine V2 club financial context experiment

**Status:** completed development experiment; no candidate advanced; continuity
remains release-blocked.

## Question

Does strictly prior-season club payroll, transfer activity, squad composition,
or wage concentration repair the Phase-4 `any_outbound_24m` continuity model?

The experiment used the established rolling origins ending in evaluation years
2020, 2021, 2022, and 2023. The 2024+ final holdout remained sealed. For every
feature block, the Phase-4 baseline was retrained on the identical eligible
train, validation, and evaluation rows. Missing prior-season club context was
returned as unavailable; it was never imputed or scored.

Five blocks were frozen before outcome evaluation:

1. payroll level and within-league payroll rank;
2. transfer spend and income relative to payroll;
3. squad age and foreign-player share;
4. the combined six-feature club-context block; and
5. wage Gini and top-five wage share as a coverage-limited diagnostic.

Because five variants were tested, advancement required a Bonferroni-adjusted
99% player-cluster bootstrap interval above zero, at least three of four origin
wins, no more than 15% operational refusal, acceptable calibration, and support
for every advertised league.

## Result

| Candidate | Coverage | Operational refusal | Brier change vs matched baseline | Origin wins | Adjusted 99% CI | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Payroll resource level | 87.8% | 17.3% | **+0.000003** | 3/4 | [−0.000367, +0.000369] | Reject |
| Recruitment intensity | 87.8% | 16.0% | −0.000231 | 2/4 | [−0.001043, +0.000596] | Reject |
| Squad context | 87.8% | 16.7% | −0.001648 | 2/4 | [−0.003516, +0.000154] | Reject |
| Combined core context | 87.8% | 17.4% | −0.001440 | 2/4 | [−0.003362, +0.000448] | Reject |
| Wage structure diagnostic | 82.0% | 22.2% | **+0.000016** | 2/4 | [−0.000637, +0.000650] | Reject / diagnostic only |

Positive Brier change means lower loss. The two positive estimates are
effectively zero and their intervals span both benefit and harm. The combined
block found three additional outbound events at a pooled top-20% review
capacity, but it worsened probability accuracy, won only two origins, and
failed coverage/refusal requirements. That isolated ranking count is therefore
not an advancement result.

## Ligue 1 interpretation

The context-covered Ligue 1 subset looks less weak than the full Phase-4 cohort,
but the new features did not cause that improvement:

| Phase-4 model scope | Rows | AUC | Brier skill vs chronology |
| --- | ---: | ---: | ---: |
| Full Ligue 1 cohort | 299 | 0.550 | **−0.0066** |
| Prior-context available only | 270 | 0.569 | **−0.0009** |
| Prior-context missing | 29 | 0.476 | **−0.0597** |

The source gate disproportionately removes the most difficult cases—clubs with
no prior Big-Five top-flight season, generally promoted clubs. Retraining on the
covered cohort can make the formal league gate appear positive, but that is a
scope/selection effect. On the matched covered rows, payroll features worsened
Ligue 1 Brier by 0.00017, recruitment features by 0.00019, squad context by
0.00106, and the combined block by 0.00136. The club financial features did not
repair Ligue 1 continuity.

## Consequence

The new canonical Capology table is valid and useful descriptive context, but
these aggregates do not contain the missing personalized continuity signal.
They must not be promoted merely because they are financially intuitive.

The next continuity work should target mechanisms closer to outbound movement:

- leakage-safe club loan and permanent-exit propensity;
- prior extension-to-exit behavior;
- positional squad depth and competition at the exact decision date;
- manager change, tactical turnover, and squad churn;
- promoted-club / second-tier context rather than excluding those cases; and
- hierarchical league/club effects with shrinkage, evaluated without weakening
  the existing subgroup gates.

## Evidence

- `Data/processed/engine_v2_club_financial_context_experiment/preregistration.json`
- `Data/processed/engine_v2_club_financial_context_experiment/candidate_comparison.csv`
- `Data/processed/engine_v2_club_financial_context_experiment/origin_metrics.csv`
- `Data/processed/engine_v2_club_financial_context_experiment/subgroup_metrics.csv`
- `Data/processed/engine_v2_club_financial_context_experiment/coverage_selection_sensitivity.csv`
- `Data/processed/engine_v2_club_financial_context_experiment/selection_decision.csv`
- `Data/processed/engine_v2_club_financial_context_experiment/independent_verification.json`

Independent verification passes 29/29 checks, including a byte-reproducible
clean-directory rerun and all 24 frozen V1 artifact hashes.

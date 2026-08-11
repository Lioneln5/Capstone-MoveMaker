# MoveMaker model-family reassessment

Date: 2026-08-10

## Decision summary

| Target | Original full-model selection | Recommendation | Plain-language conclusion |
|---|---|---|---|
| Opportunity | `full_explicit_fit + elastic_net` | **KEEP** | The original validation margin was thin, but elastic net subsequently improved on its own baseline in all four rolling origins, retained a positive 95% interval, remained positive across the mature three origins, and had the lowest pooled full-model MAE. |
| Performance | `full_explicit_fit + gradient_boosted_trees` | **KEEP** | Ridge has the larger all-origin improvement over its own baseline, but that headline is driven by one unusually weak early ridge baseline. Boosted trees won all four origins, were positive with a 95% interval above zero across the mature three origins, and had the lowest pooled full-model MAE. |
| Adaptation | `full_explicit_fit + elastic_net` (diagnostic only) | **KEEP DIAGNOSTIC-ONLY; DO NOT SWITCH TO RIDGE** | Elastic net did not reproduce a positive compatibility effect. Ridge looks positive only when the smallest first origin is included; no family has a stable positive effect across the mature three origins. Adaptation should remain diagnostic rather than receive a newly frozen confirmatory family. |

No frozen opportunity or performance family should be switched on the present evidence. Therefore this audit does not recommend rerunning `subgroup_stability` or `player_club_case_studies`.

## What was compared

The original family winner for each target was the `full_explicit_fit` family with the lowest 2022 validation MAE. The family margin below is the runner-up full-model validation MAE minus the winner's validation MAE. Positive within-family improvement means the full compatibility model had lower MAE than the same learner using only player-history features.

The rolling comparison is deliberately within family. A large ridge improvement can mean that full ridge repaired a weak ridge baseline; it does not by itself mean that full ridge was the most accurate full model. For that reason, the audit reports both the compatibility improvement and the pooled absolute MAE of the full candidate.

## Original 2022 validation selection

| Target | Selected family | Selected full validation MAE | Runner-up family | Runner-up validation MAE | Winning margin | Same-family validation MAE improvement | Margin / selected-family rolling origin SD |
|---|---|---:|---|---:|---:|---:|---:|
| Opportunity | Elastic net | 0.200005 | Ridge | 0.203569 | 0.003563 | 0.001153 | 0.43x |
| Performance | Gradient-boosted trees | 0.163121 | Elastic net | 0.182999 | 0.019879 | 0.024265 | 2.97x |
| Adaptation | Elastic net | 0.181995 | Ridge | 0.198384 | 0.016390 | 0.010383 | 2.31x |

The opportunity choice was the thin one: its 0.003563 validation advantage was only 43% of elastic net's across-origin improvement standard deviation. The performance and adaptation validation margins were larger relative to the selected families' own rolling variation, but a one-season margin can still fail to generalize, as adaptation demonstrates.

## Four-origin rolling evidence

| Target | Family | Pooled n | Full candidate pooled MAE | Pooled MAE improvement | 95% CI | Origins won | Origin improvement SD | Stability label |
|---|---|---:|---:|---:|---|---:|---:|---|
| Opportunity | Ridge | 207 | 0.213209 | 0.044804 | [0.029543, 0.060442] | 4/4 | 0.084679 | `stable_positive` |
| Opportunity | Elastic net | 207 | 0.208643 | 0.014088 | [0.003035, 0.024732] | 4/4 | 0.008283 | `stable_positive` |
| Opportunity | Gradient-boosted trees | 207 | 0.219155 | -0.005166 | [-0.012329, 0.002330] | 0/4 | 0.003718 | `unstable_negative` |
| Performance | Ridge | 172 | 0.192608 | 0.158404 | [0.119592, 0.197163] | 3/4 | 0.281034 | `stable_positive` |
| Performance | Elastic net | 172 | 0.185163 | -0.005800 | [-0.020228, 0.008293] | 2/4 | 0.016653 | `mixed` |
| Performance | Gradient-boosted trees | 172 | 0.181657 | 0.008802 | [-0.000852, 0.018251] | 4/4 | 0.006688 | `stable_positive` |
| Adaptation | Ridge | 162 | 0.232449 | 0.047997 | [0.031995, 0.063763] | 3/4 | 0.100984 | `stable_positive` |
| Adaptation | Elastic net | 162 | 0.193643 | -0.005249 | [-0.016891, 0.006011] | 1/4 | 0.007110 | `unstable_negative` |
| Adaptation | Gradient-boosted trees | 162 | 0.188207 | -0.002517 | [-0.010671, 0.005560] | 1/4 | 0.003866 | `unstable_negative` |

## Target-by-target reassessment

### Opportunity — KEEP elastic net

The validation-selected elastic net was not the family with the largest four-origin improvement; ridge's 0.044804 improvement is larger than elastic net's 0.014088. Both, however, exclude zero and win all four origins. Elastic net is substantially more consistent: its origin improvement standard deviation is 0.008283 versus ridge's 0.084679, and its pooled full-model MAE is better, 0.208643 versus 0.213209.

The original 0.003563 family margin was thin at 0.43 times elastic net's rolling standard deviation, so the single validation split alone was weak evidence. The rolling results now supply the missing support. Across the mature three origins, elastic net remains `stable_positive`: improvement 0.010093, 95% CI [0.000438, 0.019791], with 3/3 origins won. Mature-three ridge falls to 0.003951 with a CI crossing zero, [-0.005683, 0.013660]. Elastic net is therefore the better-supported frozen opportunity family.

### Performance — KEEP gradient-boosted trees

This is the main apparent conflict. The selected boosted-tree result is `stable_positive` and wins 4/4 origins, but its all-origin 95% CI narrowly crosses zero: improvement 0.008802, CI [-0.000852, 0.018251]. Ridge is the only all-origin family whose CI clearly excludes zero, with a much larger pooled improvement of 0.158404 and 3/4 origins won.

That ridge result is not robust to the project's declared mature-origin sensitivity check. Its first-origin improvement is 0.563901, while one later origin is negative. Removing the smallest initial training origin reduces ridge to a mature-three improvement of 0.001437 with CI [-0.010522, 0.013534]; the terminal-two estimate is -0.000458. Boosted trees strengthen across the mature three origins to 0.011363 with CI [0.000336, 0.022046] and 3/3 wins. They also have the best all-origin full-model MAE, 0.181657, versus 0.192608 for ridge.

The original boosted-tree validation advantage over elastic net was 0.019879, or 2.97 times the boosted-tree origin improvement standard deviation. It was not a razor-thin choice, and the mature rolling evidence supports keeping it.

### Adaptation — keep diagnostic-only; do not switch to ridge

The original elastic-net choice does not reproduce: pooled improvement is -0.005249, its CI crosses zero, and it wins only 1/4 origins. Ridge is the only all-origin family with a positive CI, so the validation-selected family is explicitly not the apparent all-origin robustness winner.

The ridge headline is nevertheless driven by the smallest first origin, where improvement is 0.196940. Ridge's origin improvement standard deviation is 0.100984, so the original 0.016390 elastic-net validation margin is only 0.16 times the instability of the ridge alternative. Across the mature three origins ridge becomes negative overall at -0.004134 with CI [-0.015972, 0.007781]. Elastic net is also negative, -0.006593, and boosted trees are -0.001670; all three mature-origin intervals cross zero.

Ridge also has the worst all-origin full-model MAE for adaptation, 0.232449, despite the large improvement over its own baseline. The evidence therefore does not justify switching adaptation to ridge. The honest decision is to retain adaptation as diagnostic-only and avoid presenting any family as a stable compatibility candidate.

## Audit checks

- The three `validation_selected_full` rows exactly match the minimum validation MAE among the three `full_explicit_fit` families for their targets.
- The nine pooled MAE improvements were independently reconstructed from the four origin-level evaluation rows; the maximum absolute difference from the published summary was `2.78e-17`.
- The nine origin improvement standard deviations were independently reconstructed; the maximum absolute difference was `5.55e-17`.
- No model was retrained and no upstream or downstream pipeline output was modified for this reassessment.

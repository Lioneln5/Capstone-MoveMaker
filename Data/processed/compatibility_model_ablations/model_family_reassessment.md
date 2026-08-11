# MoveMaker model-family reassessment — corrected encoder rerun

Date: 2026-08-10

This document replaces the pre-correction reassessment. Every model and rolling-origin result cited below was regenerated after `Preprocessor.transform` was fixed to preserve the mixed numeric/categorical feature order learned by `fit_preprocessor`.

## Decision summary

| Target | Corrected 2022 full-model selection | Corrected recommendation | Plain-language conclusion |
|---|---|---|---|
| Opportunity | `full_explicit_fit + gradient_boosted_trees` | **Do not advance as a general compatibility candidate** | Rich compatibility information is mixed at best for opportunity. The corrected frozen tree candidate is worse than its same-family baseline across the four origins, while model-selection policy gains are small and uncertain. |
| Performance | `full_explicit_fit + gradient_boosted_trees` | **Advance with temporal and subgroup guardrails** | Performance is the only target with a repeatable rich-compatibility benefit under the corrected rolling evaluation. Both ridge and boosted trees are positive across the mature origins. |
| Adaptation | `full_explicit_fit + ridge` | **Diagnostic only** | No selection-policy advantage survives. Family-specific positive directions remain too dependent on learner and origin to freeze as a product claim. |

## Corrected single-validation selection

The 2022 validation winner among `full_explicit_fit` families is:

| Target | Selected family | Validation MAE | 2023 test MAE | 2023 test R² |
|---|---|---:|---:|---:|
| Opportunity | Gradient-boosted trees | 0.206166 | 0.219748 | -0.384480 |
| Performance | Gradient-boosted trees | 0.166252 | 0.187223 | -0.056836 |
| Adaptation | Ridge | 0.193738 | 0.171943 | -0.094545 |

These test slices contain only 37–58 rows and are not headline evidence.

## Corrected four-origin within-family evidence

Positive improvement means the full compatibility specification has lower pooled MAE than the same learner using player-history baseline features.

| Target | Family | Pooled n | Baseline MAE | Full MAE | MAE improvement | 95% CI | Origins won | Stability |
|---|---|---:|---:|---:|---:|---|---:|---|
| Opportunity | Ridge | 207 | 0.225358 | 0.222359 | +0.002999 | [-0.007879, +0.014107] | 1/4 | `mixed` |
| Opportunity | Elastic net | 207 | 0.229505 | 0.217304 | +0.012201 | [+0.001037, +0.023838] | 2/4 | `mixed_positive` |
| Opportunity | Gradient-boosted trees | 207 | 0.211337 | 0.214737 | -0.003400 | [-0.009841, +0.003317] | 1/4 | `unstable_negative` |
| Performance | Ridge | 172 | 0.191177 | 0.177773 | +0.013405 | [+0.005459, +0.020917] | 4/4 | `stable_positive` |
| Performance | Elastic net | 172 | 0.192489 | 0.188206 | +0.004283 | [-0.004960, +0.014258] | 3/4 | `mixed_positive` |
| Performance | Gradient-boosted trees | 172 | 0.182205 | 0.174383 | +0.007821 | [-0.002782, +0.018372] | 3/4 | `stable_positive` |
| Adaptation | Ridge | 162 | 0.181517 | 0.185427 | -0.003910 | [-0.012700, +0.005108] | 1/4 | `unstable_negative` |
| Adaptation | Elastic net | 162 | 0.189727 | 0.187143 | +0.002584 | [-0.003206, +0.008194] | 2/4 | `mixed_positive` |
| Adaptation | Gradient-boosted trees | 162 | 0.187554 | 0.182572 | +0.004983 | [-0.002068, +0.012134] | 4/4 | `stable_positive` |

## Corrected rolling model-selection policy

This comparison is closer to the actual procedure: each origin independently selects the best baseline-only and full-feature specification on its validation season.

| Target | Pooled n | Baseline MAE | Full MAE | Improvement | 95% CI | Origins won | Decision |
|---|---:|---:|---:|---:|---|---:|---|
| Opportunity | 207 | 0.215776 | 0.213562 | +0.002214 | [-0.005791, +0.010518] | 3/4 | Small, uncertain benefit |
| Performance | 172 | 0.186427 | 0.173867 | +0.012560 | [+0.000291, +0.024308] | 4/4 | Stable positive evidence |
| Adaptation | 162 | 0.185686 | 0.186333 | -0.000647 | [-0.011052, +0.009792] | 2/4 | No benefit |

## Mature-origin sensitivity

- Opportunity: no family has a mature-three confidence interval above zero. The corrected frozen boosted-tree candidate is negative overall.
- Performance: ridge improves MAE by +0.013635 with CI [+0.005379, +0.022062]; boosted trees improve by +0.013019 with CI [+0.000346, +0.025251]. Both win 3/3 origins.
- Adaptation: boosted trees are directionally positive (+0.005986) but the CI still crosses zero [-0.000913, +0.012764]. Ridge is negative.

## Subgroup consequence

The corrected 2022 validation selection freezes boosted trees for both opportunity and performance in the refreshed subgroup layer. Opportunity is positive in only one of six adequately sized mature subgroups, with no supported-positive subgroup and one supported-negative subgroup. Performance is positive in four of five adequately sized mature subgroups, with two supported-positive subgroups and none supported-negative.

## Bottom line

The encoder correction does not restore a universal compatibility score. It does, however, overturn the earlier conclusion that every target carried a stable ridge compatibility effect. The defensible rich-data claim is narrower: pre-transfer compatibility features contain a repeatable signal for destination performance, while opportunity remains weak/mixed and adaptation remains diagnostic.

The separate 3,134-row coarse diagnostic still finds small contextual effects for broad opportunity prediction. That large-sample context result should not be conflated with the richer 2017–2024 compatibility panel.

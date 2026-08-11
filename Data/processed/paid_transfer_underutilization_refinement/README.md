# Paid-transfer under-utilization calibration and compact refinement

This experiment uses an earlier season for raw-model selection, a later validation season for compact-block/calibration selection, and the following untouched season for evaluation.

## Selected joint policies

- `origin_2019`: `C3_plus_compact_destination_context` + `isotonic_projected` (selector mean Brier 0.168588).
- `origin_2020`: `C3_plus_compact_destination_context` + `platt_projected` (selector mean Brier 0.216573).
- `origin_2021`: `C3_plus_compact_destination_context` + `empirical_distributional` (selector mean Brier 0.192331).
- `origin_2022`: `C3_plus_compact_destination_context` + `platt_projected` (selector mean Brier 0.191664).

## Evaluation decision

- `underutilization_10pct`: existing-vs-calibrated Brier improvement -0.005106, 95% CI [-0.016317, 0.005915], ECE 0.113429 -> 0.124662; `retain_existing_ordered_ranking_and_use_risk_bands_not_absolute_probabilities`.
- `underutilization_25pct`: existing-vs-calibrated Brier improvement 0.012381, 95% CI [-0.001768, 0.026285], ECE 0.116576 -> 0.107922; `use_compact_calibrated_policy_for_probabilities_retain_existing_as_ranking_reference`.
- `opportunity_share_24m`: `replace_existing_continuous_policy_with_fixed_compact_context`.

## Guardrail

Calibration can correct probability scale but cannot create discrimination or establish causation. Compact context is retained only when validation selects it; no result should be described as proof of tactical compatibility, financial loss, or ROI.

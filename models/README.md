# MoveMaker finalized model runners

This folder contains one primary Python entry point for each finalized modeling stage. Run every command from the project root.

| Model stage | Entry point | Verified output |
| --- | --- | --- |
| Retention diagnostic | `python models/run_retention_diagnostic.py` | `Data/processed/retention_diagnostic/` |
| Market-value downside | `python models/run_market_value_downside_model.py` | `Data/processed/market_value_downside_model/` |
| Market-value calibration | `python models/run_market_value_calibration_experiment.py` | `Data/processed/market_value_calibration_experiment/` |
| Paid-transfer under-utilization | `python models/run_paid_transfer_underutilization_model.py` | `Data/processed/paid_transfer_underutilization_model/` |
| Under-utilization refinement | `python models/run_paid_transfer_underutilization_refinement.py` | `Data/processed/paid_transfer_underutilization_refinement/` |
| Reported-fee benchmark | `python models/run_fee_benchmarking_model.py` | `Data/processed/fee_benchmarking_model/` |
| Component capital-at-risk integration | `python models/run_capital_at_risk_integration.py` | `Data/processed/capital_at_risk_integration/` |

Independent verification, data preparation, audits, diagnostics, and product-artifact builders remain in `scripts/`. The model runners import those shared utilities without duplicating them.

The evidence, approved claims, rejected claims, and modeling stopping decisions are documented in `CODEX.md`.

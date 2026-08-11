# MoveMaker

MoveMaker is a football-transfer analytics capstone that tests which pre-transfer risks can be predicted credibly from public historical data.

The project began as a player-club compatibility and return-on-investment concept. Chronological evaluation did not support one universal compatibility or ROI score. The final scope is a modular transfer due-diligence system:

- calibrated 24-month estimated market-value downside;
- expected playing opportunity and under-utilization risk;
- transfer-time retention priors with a one-season monitoring update;
- historical reported-fee benchmarking; and
- an explainable value-risk versus opportunity-risk matrix.

MoveMaker is a screening and diligence-prioritization tool. It does not predict accounting ROI, realized sale proceeds, exact tenure, tactical causation, or a uniquely correct transfer fee.

## Headline findings

| Module | Primary verified result | Product interpretation |
| --- | --- | --- |
| Market-value downside | AUC 0.851 for a decline of at least 25%; Brier 0.2474 → 0.1584 | Strong historical downside-risk screen |
| Playing opportunity | MAE 0.2102 → 0.1818 after compact refinement | Broad opportunity estimate, not exact minutes |
| Severe under-utilization | Brier 0.2356 → 0.2003; event rate 18.6% → 70.1% across extreme quintiles | Relative risk tiers; individual calibration rejected |
| Meaningful under-utilization | Refined Brier 0.2296 → 0.2172; calibration improved | Historical probability bands with guardrails |
| Retention | Transfer-time AUC 0.586; one-season landmark AUC 0.713 | More useful as staged monitoring |
| Reported-fee benchmark | MAE-log 0.7049, approximately 2.02× multiplicative error | Coarse comparables and negotiation screen |
| Capital-risk integration | Joint AUC 0.604; high/high severe rate 28.8% vs 17.5% overall | Explainable triage matrix, not a joint probability |

Every result above uses rolling chronological evaluation and independently verified artifacts. See [CODEX.md](CODEX.md) for the complete technical handoff and claim boundaries.

## Repository layout

```text
models/                 Finalized model entry points
scripts/                Data builders, shared utilities, diagnostics, and verifiers
docs/                   Data audit, experiment prompts, journal, and paper evidence
Data/processed/         Curated summary evidence only in Git; large artifacts stay local
CODEX.md                Full project history, decisions, metrics, and guardrails
```

The model index and commands are documented in [models/README.md](models/README.md).

## Environment

The verified local environment uses Python 3.13. Install the core analytical dependencies with:

```bash
python -m pip install -r requirements.txt
```

Core models depend on NumPy, pandas, SciPy, and scikit-learn. Optional `.mjs` workbook builders use the Codex artifact runtime and are not required to reproduce the model metrics.

## Running and verification

Run commands from the repository root. For example:

```bash
python models/run_market_value_downside_model.py
python scripts/verify_market_value_downside_model.py
```

Each finalized model has a corresponding verifier under `scripts/`. Model runners require the locally prepared upstream data described in `CODEX.md`; the full raw and generated data estate is intentionally excluded from ordinary Git because it exceeds 23 GB.

The committed `Data/processed/` files are compact decision summaries, comparison tables, model cards, manifests, and independent-verification evidence. They are sufficient to audit reported conclusions but not to reconstruct the entire source database.

## Research narrative

- [August 11 scope-pivot journal](docs/journal/2026-08-11_scope_pivot_journal.md)
- [Five-page report evidence bank](docs/research/2026-08-11_report_evidence_bank.md)
- [Data-readiness audit](docs/DATA_READINESS_AUDIT.md)

The central finding is an evidence-led scope decision: public data did not validate the promised universal compatibility/ROI engine, but it did support narrower and commercially interpretable transfer-risk modules.

## Data and claim guardrails

- Never use post-transfer outcomes as predictors.
- Use chronological, not random, evaluation as the primary test.
- Treat public market value as an estimate, not realized cash.
- Treat reported fee as incomplete deal-price evidence, not total cost.
- Do not infer tactical causation from predictive context variables.
- Do not present fee-weighted risk indices as expected loss or ROI.

Raw datasets, source backups, large canonical tables, row-level predictions, and generated workbooks are excluded from Git. They should use documented external storage or Git LFS if the team later decides to distribute them.

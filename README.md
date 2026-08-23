# MoveMaker

MoveMaker is an explainable decision-support system for incumbent-club football contract extensions. A user selects a current player and enters proposed wage and term information; the application returns separate, evidence-backed views of:

- meaningful contribution through Year 2;
- continuous stay at the current club;
- 24-month public market-value downside;
- proposed fixed-wage commitment versus historical extension peers; and
- qualified financial exposure tied to low contribution.

MoveMaker is a screening and prioritization tool. It does **not** produce one overall success score, recommend whether a club should extend a player, predict tactical compatibility, or estimate accounting loss or ROI.

**Explore the project:** [final presentation](slides/final/MoveMaker_Final_Presentation.html) · [application](html/index.html) · [model card](docs/MODEL_CARD.md) · [data provenance](docs/DATA_PROVENANCE.md) · [analytical journal](docs/journal/2026-08-11_scope_pivot_journal.md)

## Why this product exists

The project did not begin with contract extensions. Its scope changed twice because the evidence rejected broader claims:

1. **Player and club compatibility.** The original concept profiled player-player and player-club relationships, then combined those profiles with transfer and financial information to estimate the likely sporting and financial return of a signing. Rich feature requirements reduced the complete cohort to only 63 cases, and larger-cohort diagnostics did not recover a stable general compatibility signal.
2. **Modular transfer-risk screening.** The first pivot separated opportunity, retention, value-downside, fee, and capital-risk questions. Several individual signals were useful, but they could not defensibly be compressed into one transfer-success or ROI score.
3. **Incumbent-club extension decisions.** Big-Five Capology salary and extension data made wage commitment and historical contract context observable. That supported a narrower, financially concrete question: *what should a club investigate before extending a player, and how aggressive are the proposed terms relative to comparable historical extensions?*

The scope pivot is the central research result: more data and more modeling did not justify more certainty. They identified which decisions public data can support and which claims must remain out of scope.

## Headline evidence

All primary results use rolling chronological evaluation and saved independent-verification artifacts.

| Decision signal | Verified evidence | Product use |
| --- | --- | --- |
| Year-2 opportunity | MAE 0.2369, R² 0.249, Spearman 0.512; improvement +0.0586, 95% CI [0.0485, 0.0686], 4/4 origins | Broad future-role outlook, not exact minutes |
| Sustained meaningful contribution | AUC 0.788, AP 0.732, Brier 0.1869; improvement +0.0643, 4/4 origins | Probability of sustaining at least 25% same-club opportunity share in both Year 1 and Year 2 |
| Any outbound by 24 months | AUC 0.710, Brier 0.2191; improvement +0.0363, 4/4 origins | Continuous-stay risk, not an exact departure date |
| Public-value downside by 24 months | AUC 0.821 for a decline of at least 25% | Public valuation risk, not sale proceeds or accounting loss |
| Fixed-wage commitment benchmark | R² 0.644, Spearman 0.832; ~80% empirical reference-range coverage | Proposed commitment versus historical peers, not a “fair” wage |
| Integrated contract score | Three-module AUC 0.6830 versus 0.6885 for opportunity risk alone; difference CI crosses zero | Rejected; signals remain separate |

The financial scale is material. Across 908 out-of-time extension profiles with known terms, €5.22B in two-year fixed wages was scheduled; €1.82B, or 34.8%, was attached to players who did not sustain meaningful contribution through Year 2. At the same 182-extension review capacity, risk-weighted screening identified 79 historical low-contribution outcomes versus 54 from selecting the largest contracts alone—46% more—while concentrating less wage capital in the flagged group. These are historical exposure and prioritization results, not confirmed losses or projected savings.

## Current product

The local application consists of a FastAPI scoring service and a static HTML front end:

```text
api/main.py                         API
html/index.html                     application interface
models/deployment_scorer.py         deployed endpoint orchestration
models/canonical_feature_retriever.py
Data/processed/deployment_models/   serialized model artifacts and manifest
```

The deployed profile exposes 16 separately validated endpoints rather than a composite score. The sustained-contribution endpoint uses the later C6 compact model, which removed unstable goals/assists-per-90 inputs and passed the declared non-inferiority and deployment-parity checks. The role-contextualization diagnostic is research-only and has not replaced the remaining production endpoints.

### Run locally

```bash
python -m pip install -r requirements.txt
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Then open [http://127.0.0.1:8000/](http://127.0.0.1:8000/). The service and source data are local; collaborators cannot run a standalone copy of the HTML without also running the API and retaining the required local artifacts.

## Repository map

```text
api/                    FastAPI service
html/                   user-facing extension profile
models/                 research runners, production scoring, and feature retrieval
scripts/                builders and independent verifiers
docs/                   journal, report evidence, diagnostics, and publication policy
Data/processed/         compact auditable evidence plus local-only generated artifacts
slides/final/           final presentation assets
```

See [models/README.md](models/README.md) for the experiment and verifier registry, the [scope-pivot journal](docs/journal/2026-08-11_scope_pivot_journal.md) for the analytical history, and [docs/GITHUB_PUBLICATION_MANIFEST.md](docs/GITHUB_PUBLICATION_MANIFEST.md) for what belongs in Git.

## Reproducibility and evidence

Each finalized stage has:

- a runner under `models/` or builder under `scripts/`;
- a dedicated verifier under `scripts/`;
- a source manifest, output manifest, build checks, decision summary, and independent-verification report under `Data/processed/`.

The compact Git evidence is intended to audit reported conclusions. Full raw data, large canonical tables, row-level training matrices, candidate artifacts, and predictions remain local because of size, licensing, and reviewability.

This repository is therefore an **auditable publication baseline**, not a zero-data rebuild bundle. See the [data-provenance statement](docs/DATA_PROVENANCE.md) for source boundaries and the [model card](docs/MODEL_CARD.md) for intended use, evaluation, and limitations.

## Claim boundaries

- Extension probabilities apply to incumbent-club extensions, not hypothetical destination-club transfers.
- “Low contribution” means failing to sustain at least 25% same-club opportunity share in both Year 1 and Year 2; it does not necessarily mean fewer than 25% of all possible minutes across the full two-year window.
- Outbound means any recorded move away from the incumbent club in the stated horizon, including loan or permanent movement where the endpoint says “any outbound.”
- Public market value is a public estimate, not a realized transfer fee, cash flow, or accounting valuation.
- Fixed wages exclude bonuses, taxes, agent fees, clauses, and options unless explicitly stated.
- Risk-weighted wage exposure is a prioritization proxy, not expected loss, wasted wages, or projected savings.
- Historical peer estimates describe observed comparable contracts; they are not optimal terms or fair value.
- Missing inputs remain unavailable rather than being silently converted to zero.

## Continuing the project

The presentation milestone is complete, but development is not. The next work should improve model validity and deployability rather than widen the claims:

1. finish publishing the current reproducible baseline in reviewable commits;
2. audit and, where justified, replace globally specified scoring-rate features with role-aware or removed variants;
3. refresh current performance and salary inputs without contaminating frozen out-of-time evidence;
4. strengthen uncertainty, subgroup, and out-of-distribution displays in the application;
5. package the API and front end for a reproducible public demo; and
6. add richer injury, squad-competition, option, clause, and internal club data only when coverage supports a new claim.

## Research record

- [Scope-pivot journal](docs/journal/2026-08-11_scope_pivot_journal.md)
- [Research-paper evidence bank](docs/research/2026-08-11_report_evidence_bank.md)
- [Data-readiness audit](docs/DATA_READINESS_AUDIT.md)
- [Contract diagnostic checklist](docs/diagnostics/contract_scope_diagnostic_checklist.md)
- [Final presentation](slides/final/MoveMaker_Final_Presentation.html)

## License and reuse

No open-source license is currently granted for this repository. The code and documentation are visible for review, but reuse rights are not implied. Third-party football data is not redistributed and remains subject to its original providers' terms.

MoveMaker’s core lesson is simple: the strongest product was not the broadest model. It was the one whose claims survived chronological testing and could be translated into a concrete financial decision.

# MoveMaker

**Football Contract Extension Decision Support**

MoveMaker is an explainable research program for incumbent-club football contract extensions. Its frozen V1 prototype evaluated separate views of:

- meaningful contribution through Year 2;
- outbound movement from the current club, an endpoint now retired from the candidate product;
- 24-month public market-value downside;
- proposed fixed-wage commitment versus historical extension peers; and
- qualified financial exposure tied to low contribution.

MoveMaker is a screening and prioritization tool. It does **not** produce one overall success score, recommend whether a club should extend a player, predict tactical compatibility, or estimate accounting loss or ROI.

> **Current status (2026-09-02):** Engine V1 is frozen for historical audit and HTTP scoring is unconditionally disabled. Engine V2 retired the generic continuity endpoint after decomposing it into distinct football questions. **Meaningful Retention** and **Temporary Displacement Risk** remain development candidates awaiting a later untouched temporal test; **Permanent Separation** is research-only and unavailable. This product-scope decision resolves the former continuity release blocker through endpoint retirement, not model repair. The two remaining release blockers concern final temporal validation and the historical wage benchmark. Nothing from V2 is deployed. See the [movement-scope contract](docs/ENGINE_V2_MOVEMENT_SCOPE_CLOSEOUT.md), [critical-issues register](docs/ENGINE_V2_CRITICAL_ISSUES.md), and [V1 freeze record](docs/ENGINE_V1_FREEZE.md).

**Explore the project:** [hosted research site](https://movemaker-production.up.railway.app/) · [final presentation](slides/final/MoveMaker_Final_Presentation.html) · [model card](docs/MODEL_CARD.md) · [data provenance](docs/DATA_PROVENANCE.md) · [analytical journal](docs/journal/2026-08-11_scope_pivot_journal.md)

[![MoveMaker live decision profile showing Jude Bellingham's four separate contract-extension signals](docs/assets/movemaker-live-profile.png)](https://movemaker-production.up.railway.app/)

*Historical V1 example: a three-year extension profile for Jude Bellingham. The pictured continuity card has since been retired; the image is retained as part of the project’s analytical history.*

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
| Any outbound by 24 months | AUC 0.710, Brier 0.2191; improvement +0.0363, 4/4 origins | Frozen V1 historical evidence; endpoint retired because it blends temporary and permanent movement |
| Meaningful Retention | AUC 0.759; Brier skill +0.0352, 95% CI [+0.0260, +0.0443], 4/4 origins | Engine V2 development candidate; no outbound plus meaningful same-club use in both years; not deployed |
| Temporary Displacement Risk | AUC 0.777; Brier skill +0.0291, 95% CI [+0.0209, +0.0375], 4/4 origins | Engine V2 development candidate; temporary first movement; not deployed |
| Permanent Separation | AUC 0.628; negative league-level skill in Bundesliga and Ligue 1 | Research-only and unavailable as a personalized output |
| Public-value downside by 24 months | AUC 0.821 for a decline of at least 25% | Public valuation risk, not sale proceeds or accounting loss |
| Fixed-wage commitment benchmark | R² 0.644, Spearman 0.832; ~80% empirical reference-range coverage | Proposed commitment versus historical peers, not a “fair” wage |
| Integrated contract score | Three-module AUC 0.6830 versus 0.6885 for opportunity risk alone; difference CI crosses zero | Rejected; signals remain separate |

The financial scale is material. Across 908 out-of-time extension profiles with known terms, €5.22B in two-year fixed wages was scheduled; €1.82B, or 34.8%, was attached to players who did not sustain meaningful contribution through Year 2. At the same 182-extension review capacity, risk-weighted screening identified 79 historical low-contribution outcomes versus 54 from selecting the largest contracts alone—46% more—while concentrating less wage capital in the flagged group. These are historical exposure and prioritization results, not confirmed losses or projected savings.

## Current product

The application combines a FastAPI service with a static HTML front end. The public research site is hosted on Railway in fail-closed research-record mode. HTTP scoring cannot be enabled by environment variable:

```text
api/main.py                         API
html/index.html                     application interface
models/deployment_scorer.py         frozen V1 scoring code retained for offline audit
models/canonical_feature_retriever.py
Data/processed/deployment_models/   serialized model artifacts and manifest
```

The frozen V1 package contains 16 separately evaluated endpoints rather than a composite score. Those artifacts are preserved byte-for-byte, but they are not served. Engine V2 uses a separate movement product contract and has authorized no live movement/retention probability.

### Run the research site locally

```bash
python -m pip install -r requirements.txt
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Then open [http://127.0.0.1:8000/](http://127.0.0.1:8000/). The site does not load models or runtime data. Frozen V1 regression checks run through the registered offline verifiers rather than an HTTP scoring switch.

For hosted deployment, see the [Railway deployment guide](docs/RAILWAY_DEPLOYMENT.md). The current fail-closed research site needs no runtime data volume. The historical V1 scoring setup used separately supplied runtime tables that were never committed to Git.

## Repository map

```text
api/                    FastAPI service
html/                   user-facing extension profile
models/                 research runners, production scoring, and feature retrieval
scripts/                builders and independent verifiers
docs/                   journal, report evidence, diagnostics, and publication policy
Data/processed/         compact auditable evidence; large row-level artifacts stay local
slides/final/           final presentation assets
```

See [models/README.md](models/README.md) for the experiment and verifier registry, the [scope-pivot journal](docs/journal/2026-08-11_scope_pivot_journal.md) for the analytical history, and [docs/GITHUB_PUBLICATION_MANIFEST.md](docs/GITHUB_PUBLICATION_MANIFEST.md) for what belongs in Git.

## Project team

MoveMaker was developed as a four-person capstone project. The repository and deployed demo represent the shared project deliverable.

- **Lionel** — Project Coordinator and ML Lead
- **Evelyn** — Scrum Master
- **Derick** — Deliverable Architect
- **Rene** — Tech Auditor

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
- Historical “any outbound” means any recorded loan or permanent move away from the incumbent club in the stated horizon. Engine V2 retired that blended product endpoint and keeps movement mechanisms separate.
- Meaningful Retention is a joint outcome: no outbound movement plus at least 25% same-club opportunity share in both Year 1 and Year 2. It is not a rename of the contribution-only V1 endpoint.
- Public market value is a public estimate, not a realized transfer fee, cash flow, or accounting valuation.
- Fixed wages exclude bonuses, taxes, agent fees, clauses, and options unless explicitly stated.
- Risk-weighted wage exposure is a prioritization proxy, not expected loss, wasted wages, or projected savings.
- Historical peer estimates describe observed comparable contracts; they are not optimal terms or fair value.
- Missing inputs remain unavailable rather than being silently converted to zero.

## Continuing the project

The presentation milestone is complete, but development is not. No current V2
candidate is authorized for an artifact. Generic continuity has been retired,
the two replacement development candidates lack a valid final cohort, and the
wage benchmark failed final Ligue 1 coverage. The next work should improve
validity and deployability rather than widen the claims:

1. preserve the failed/contaminated 2024 evidence and reserve a later, fully mature, player-disjoint cohort before opening another final temporal test;
2. keep permanent-separation research deferred until a materially new, decision-time-safe mechanism justifies a frozen candidate;
3. declare any successor wage candidate as a new version without fitting to the failed 2024 final labels, then test it only on later untouched evidence;
4. build versioned artifacts and API parity checks only for modules that pass their immutable final evaluation;
5. refresh current performance and salary inputs without contaminating frozen chronological evidence; and
6. add richer injury, squad-competition, option, clause, and internal club data only when coverage supports a new claim.

## Research record

- [Original twelve-problem Engine V2 audit](docs/ENGINE_V2_ORIGINAL_AUDIT.md) — permanent status ledger for the post-presentation stress-test findings
- [Scope-pivot journal](docs/journal/2026-08-11_scope_pivot_journal.md)
- [Research-paper evidence bank](docs/research/2026-08-11_report_evidence_bank.md) — historical transfer-risk pivot
- [Data-readiness audit](docs/DATA_READINESS_AUDIT.md) — historical pre-extension scope
- [Contract diagnostic checklist](docs/diagnostics/contract_scope_diagnostic_checklist.md)
- [Final presentation](slides/final/MoveMaker_Final_Presentation.html)

## License and reuse

No open-source license is currently granted for this repository. The code and documentation are visible for review, but reuse rights are not implied. Third-party football data is not redistributed and remains subject to its original providers' terms.

MoveMaker’s core lesson is simple: the strongest product was not the broadest model. It was the one whose claims survived chronological testing and could be translated into a concrete financial decision.

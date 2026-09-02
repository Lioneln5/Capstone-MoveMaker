# GitHub publication manifest

MoveMaker’s GitHub repository should make the research, product architecture, and reported evidence reviewable without publishing the full local data lake or presenting third-party data as redistributable.

## Publish

- Root documentation: `README.md` and `requirements.txt`.
- Research and production code under `models/`, `scripts/`, `api/`, and `html/`.
- Documentation under `docs/`, including the scope-pivot journal, report evidence bank, diagnostic checklist, and application notes.
- The final presentation and selected portfolio screenshots/assets, not every intermediate deck.
- Public model and data documentation: `docs/MODEL_CARD.md` and `docs/DATA_PROVENANCE.md`.
- Compact processed evidence needed to audit claims:
  - stage `README.md` files;
  - run/decision summaries;
  - feature and metric manifests;
  - chronological-origin aggregate metrics;
  - calibration, subgroup, coverage, and sensitivity summaries;
  - source/output manifests;
  - build checks; and
  - independent-verification CSV/JSON reports.
- Engine V2 movement-scope closeout artifacts: the product-scope contract,
  decision table, aggregate development evidence, non-blocking research
  backlog, checks, summaries, and independent verification.
- Small production model artifacts and their manifest when required for the application.

## Keep local or distribute separately

- Raw or scraped third-party datasets, including Transfermarkt, FBref, Capology, injury, and copied football repositories.
- Credentials, tokens, local environment files, editor settings, browser downloads, and machine-specific launch configurations.
- Large canonical, clean, integrated, and training tables that builders can reproduce locally.
- Row-level targets, model matrices, predictions, tuning grids, bootstrap samples, and generated case-card tables unless a specific compact example is needed.
- Player/event-level Engine V2 mechanism profiles and as-of evidence audits;
  publish their feature dictionaries, coverage summaries and verifiers instead.
- Machine-specific file inventories, table schemas, raw StatsBomb payload
  catalogs, extension-level overlap rows, and canonical Capology club-season
  panels; publish the refreshed audit summary, classifications, aggregate
  coverage and independent verification instead.
- Candidate/rejected serialized models, temporary backups, caches, `.DS_Store`, and copied `.git` directories.
- Superseded PowerPoint/PDF drafts, render diagnostics, inspect logs, and temporary slide assets.
- Generated audit workbooks and `outputs/` exports; their maintained builders and compact evidence remain public.
- Internal prompt drafts and uncurated exploratory notebooks superseded by the analytical journal and maintained scripts.

## Evidence standard

A published quantitative claim must be traceable to:

1. an included aggregate decision or metric artifact;
2. a passing independent-verification report;
3. a documented cohort, time horizon, and target definition; and
4. language that respects the approved claim boundary.

Exploratory discussion must remain labeled exploratory. A builder check proves that code produced its declared artifact; it does not by itself establish business usefulness. A candidate passing a research gate is not deployed until the production artifact, manifest, parity, explainability, and smoke tests are updated.

## Data-vintage policy

Frozen out-of-time evidence and live application inputs serve different purposes:

- frozen research snapshots support performance claims and must remain hash-stable;
- later salary, extension, valuation, performance, or injury updates may support current scoring;
- appended rows must not revise historical metrics until outcomes mature and a new evaluation is intentionally versioned.

## Commit structure

Prefer reviewable commits in this order:

1. current documentation and verified contract-research pipeline;
2. production API, HTML, feature retrieval, scoring code, and frozen deployed artifacts;
3. C6 repair and live-input refresh;
4. role-contextualization and other research-only diagnostics; and
5. final presentation/portfolio assets.

Do not blanket-stage the working tree. Before every commit, inspect file sizes, staged names, `git diff --cached --check`, and staged content for credentials or local paths.

## Reproduction

The Git repository contains enough compact evidence to audit the reported decisions, not enough third-party data to rebuild the entire project from zero. Full reproduction requires separately obtaining the documented sources, preserving the expected local directory structure, running the upstream canonical builders, and then running each stage and its verifier in the order listed in `models/README.md`.

The published final deck is `slides/final/MoveMaker_Final_Presentation.html`. No open-source license is currently granted; visibility of the repository does not imply permission to reuse the code, documentation, or third-party data.

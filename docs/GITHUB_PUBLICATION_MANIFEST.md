# GitHub publication manifest

This repository publishes the MoveMaker analysis code, technical documentation,
research narrative, and compact evidence needed to inspect the project's claims.
It intentionally does not publish the full local data lake or row-level model
outputs.

## Included

- `models/`: one implementation and one independent verifier for each finalized
  commercial model.
- `scripts/`: data preparation, compatibility diagnostics, case-study, and
  verification code.
- `docs/`: the scope-pivot journal, report evidence bank, and this manifest.
- `CODEX.md`: current technical handoff and project guardrails.
- `README.md` and `requirements.txt`: setup, project map, and reproducibility
  metadata.
- Selected `Data/processed/` evidence: decision summaries, aggregate metrics,
  temporal-origin results, subgroup checks, feature manifests, build checks,
  model/scoring contracts, source/output manifests, and independent-verification
  reports.

## Excluded

- Raw and third-party source datasets (`Data/TransferMarkt/`,
  `Data/2017-2024/`, `Data/Open-Data-Master/`, `football-datasets-main/`).
- Large canonical and cleaned tables that can be rebuilt locally.
- Row-level training targets, predictions, validation predictions, tuning grids,
  and generated decision-card tables.
- `outputs/`, caches, local environments, editor settings, and copied
  repositories.

The exclusions keep the repository reviewable and avoid treating locally
collected third-party data as redistributable. They do not change the reported
metrics: the included independent-verification reports check the compact
evidence against the locally generated artifacts.

## Evidence policy

Published findings must be traceable to an included decision summary or metric
table and to a passing independent-verification report. Exploratory discussion
is labeled as such in the journal and is not promoted to a validated result.

To reproduce the full pipeline, obtain the source datasets separately, preserve
the documented local directory structure, and run the builders and models in
the order described by `CODEX.md` and the module READMEs.

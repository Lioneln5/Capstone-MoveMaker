# Data provenance and publication boundaries

## Source families

MoveMaker was developed from several public football-data families:

- **Transfermarkt-derived data:** player identity, transfers, clubs, public market-value estimates, and related historical records.
- **FBref:** seasonal playing-time and performance statistics.
- **Capology:** publicly reported salary, contract-extension, and club payroll information.
- **Injury research data:** public injury-history records used in bounded coverage and feasibility diagnostics.

These sources differ in ownership, methodology, coverage, and update frequency. MoveMaker treats them as observed public estimates, not authoritative club records.

## What this repository publishes

The repository includes research and production code, model manifests, compact aggregate metrics, build checks, decision summaries, independent-verification reports, the local application, and the final presentation. These artifacts are intended to make the project's claims and decision trail auditable.

## What this repository excludes

Raw and scraped third-party datasets are not redistributed. Large canonical tables, joined training matrices, row-level predictions, tuning grids, candidate model artifacts, and most generated case cards remain local because of licensing, size, privacy, and reviewability concerns.

As a result, a clone can inspect the published logic and verify included compact evidence, but it cannot rebuild the complete data lake from zero without separately obtaining the source data.

## Frozen evidence versus live inputs

MoveMaker separates two data roles:

1. **Frozen research snapshots** support historical out-of-time performance claims and should remain hash-stable.
2. **Live application inputs** may use newer public salary, valuation, performance, or contract records to score a current player.

A live refresh does not retroactively improve or alter frozen evaluation metrics. New data should enter a new evaluation version only after the relevant outcomes have matured.

## Quality and linkage controls

The pipeline uses canonical player and club identifiers, documented name matching, source manifests, coverage checks, missingness audits, chronological eligibility rules, and independent verification. Missing fields remain unavailable rather than being silently imputed as observed zero unless an explicitly documented model pipeline performs imputation.

Public data can still contain name ambiguity, incomplete seasons, schema changes, stale valuations, and inconsistent league coverage. Those limitations are part of the product's displayed uncertainty and claim boundaries.

## Reuse and attribution

No open-source license is currently granted for this repository. Third-party data remains subject to the original provider's terms. Anyone reproducing the project must acquire and use those sources under their own authorized access and should cite the underlying providers as well as this project.

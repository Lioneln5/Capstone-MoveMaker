# Capology club financial context audit

> **Identity-resolution update (2026-09-01):** This directory preserves the
> initial pre-resolution audit snapshot. Its 97.7% link rate and 16 unresolved
> aliases have since been resolved with reviewed Transfermarkt club-ID evidence.
> The current canonical result is
> `Data/processed/capology_club_financial_context/`, which links 1,074/1,074
> club-seasons and passes independent verification. Do not treat this older
> unresolved-alias count as current status.

This directory is an **audit-only, non-deployed** assessment of the 110 raw
Capology payroll and transfer-window CSVs covering the Big Five leagues from
2014-15 through 2024-25. Raw files were read and hashed but never modified.

## What passed

- All 110 expected files exist and parse: 55 payroll plus 55 transfer-window.
- Each source family uses one consistent schema.
- Each has 1,074 unique club-season rows, and their keys match exactly.
- Required club-level euro values are complete and pass arithmetic checks.
- 1,049/1,074 club-season rows (97.7%) link through existing conservative canonical aliases.

## Material limitations

1. The keeper/defender/midfielder/forward payroll fields are paywalled `Locked`
   placeholders. This source supports **club-level payroll context only**.
2. Bonus values are absent before 2021-22 and complete only from 2022-23 onward.
   Do not treat missing historical bonus as zero.
3. Capology payroll is estimated, not audited club accounts.
4. The panel covers top-flight Big Five membership. A leakage-safe prior-season
   join is therefore structurally missing for newly promoted clubs.
5. Sixteen historical club names remain unresolved under conservative exact
   mapping. They are quarantined in `club_identity_audit.csv`; no fuzzy links
   were accepted.

## Leakage-safe extension coverage

Use only the last completed season at the decision date. On this rule the panel
covers 2,449/2,805
all extension events (87.3%) and
795/839
events in the integrated 2020-2023 evaluation cohort
(94.8%). The latter's missing cases are overwhelmingly
clubs absent from the previous top-flight panel, consistent with promotion.

Do **not** use a full current-season transfer-window total for an extension
signed during that season. The safe first candidate is prior-season payroll,
prior-season transfer spending/income/net balance, club payroll rank/percentile,
and multi-season changes calculated strictly from seasons completed before the
signing date. Any imputation must be learned within each training origin.

## Status

These outputs are not a production feature table and are not approved inputs to
V1 or V2. Canonical alias review, source-definition documentation, and a
rolling-origin feature experiment are still required.

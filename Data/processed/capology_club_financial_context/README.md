# Canonical Capology club financial context

This immutable build contains **1,074 canonical club-seasons** across the
Big Five from 2014-15 through 2024-25. All 16 previously unresolved Capology
aliases were reviewed against Transfermarkt club IDs and top-flight source
evidence; no fuzzy club match remains.

The table uses nominal euro payroll and transfer values. Capology's
retrospective 2025-adjusted fields and paywalled position-payroll placeholders
are intentionally excluded. All history features require exact consecutive
seasons and never bridge a relegation/promotion gap.

Player-level wage distributions are derived from the existing canonical salary
panel but emitted only when at least 15 wages are present and their sum is within
5% of the new club-payroll total. **666**
club-seasons clear that gate. Missing or non-reconciling distributions remain
null rather than silently estimated.

For extension decisions, join on `canonical_club_id` and
`pre_completed_season_start_year`. This covers
**795/839 (94.8%)**
of the integrated evaluation cohort. The 44 missing cases are the previously
identified promoted-club boundary; signing-season totals remain prohibited as a
fill.

Status: candidate feature source only. No model was trained, no final holdout was
opened, and no V1, API, HTML, or deployment artifact was changed.

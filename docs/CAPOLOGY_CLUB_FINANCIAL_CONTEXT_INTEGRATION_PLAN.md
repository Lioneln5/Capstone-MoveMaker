# Capology club financial context: integration decision

**Status:** canonical source build complete and independently verified;
**candidate research input only—not approved for V1, V2, API, or display use**.

The raw source now contains 55 Big Five club-payroll files and 55 corresponding
transfer-window files from 2014-15 through 2024-25. The independent audit passes
19/19 checks. Detailed machine-readable evidence is in
`Data/processed/capology_club_financial_context_audit/`.

## What the source actually adds

The useful new grain is one club-season row. Each family contains 1,074 rows
covering 164 distinct Capology club slugs.

Payroll supplies:

- estimated weekly and annual gross payroll;
- estimated gross bonus where Capology publishes it;
- estimated total gross payroll;
- Capology's inflation-adjusted total.

Transfer-window supplies:

- aggregate transfer income and expenditure;
- net transfer balance;
- Capology's inflation-adjusted balance;
- squad player count, foreign-player count, and average age.

The source does **not** expose payroll allocation by goalkeeper, defense,
midfield, or forward. Those columns contain paywalled `Locked` placeholders in
every file. Any position-specific wage-structure feature must therefore be
derived from the existing player-level salary panel, not these new tables.

## Coverage and identity

- 110/110 expected files are present and parse successfully.
- Both source families cover the exact same 1,074 club-season keys.
- All required club-level euro fields are present and internally coherent.
- 1,074/1,074 rows now link through the conservative canonical club mapping.
- Sixteen previously unresolved historical club names were individually
  reviewed against Capology slugs/codes and Transfermarkt club-ID URLs. The
  accepted map is versioned in
  `Data/processed/capology_club_financial_context/reviewed_club_aliases.csv`;
  no fuzzy match was accepted.
- Bonus data is absent before 2021-22, partial in 2021-22, and complete from
  2022-23 onward. Historical missing bonus cannot be interpreted as zero.

### Identity warning discovered during review

`integration_crosswalks/club_identity_crosswalk.csv:name_aliases` is not a safe
club-identity field. It contains many unrelated movement-history club names
under the same canonical row. It was **not used** to resolve these aliases and
must not be used for future club joins. The accepted mappings instead require
the Capology source slug/code plus a matching Transfermarkt club-ID URL and
historical competition identity. The current exact Capology linker and the new
canonical table follow that rule.

## Timing boundary and promoted clubs

The feature join must use `pre_completed_season_start_year`, already calculated
from the exact extension signing date. A full signing-season payroll or transfer
window must not be used because these pages are retrospective full-season totals
and can include activity after the extension decision.

Under the prior-completed-season rule:

| Scope | Covered | Total | Coverage |
| --- | ---: | ---: | ---: |
| All canonical extension events | 2,449 | 2,805 | 87.3% |
| Integrated 2020-2023 evaluation cohort | 795 | 839 | 94.8% |

All 44 missing evaluation cases have a panel row in the signing season but not
the prior season. The clubs are consistent with newly promoted teams, which were
outside the scraped Big Five top-flight universe in the prior year. Using the
signing-season figure to fill these rows would create a systematic leakage
exception precisely for promoted clubs.

Preferred resolution order:

1. Add the corresponding second-tier historical payroll and transfer-window
   pages if Capology provides comparable coverage.
2. If that source is unavailable, retain missingness explicitly and test a
   promoted-club indicator.
3. Learn any imputation only inside each rolling training origin. Never use a
   global full-sample median and never replace missing financial values with
   zero.
4. Report performance and calibration separately for promoted/missing-context
   cases at the final untouched evaluation gate.

## Candidate feature contract

The first experiment should be deliberately small. These are candidates, not a
license to add every available column.

### Club resource level

- `pre1_payroll_log_eur`: log of prior-season nominal gross payroll.
- `pre1_payroll_league_percentile`: rank within league and season.
- `pre1_payroll_to_league_median`: scale relative to contemporaneous peers.
- `proposed_wage_share_of_pre1_payroll`: extension annual gross wage divided by
  prior-season club payroll.

### Recruitment and liquidity context

- `pre1_transfer_spend_log_eur`.
- `pre1_transfer_income_log_eur`.
- `pre1_transfer_net_balance_eur`, with sign preserved.
- `pre1_transfer_spend_to_payroll`.
- trailing three-completed-season spend, income, balance, and volatility, only
  when those seasons precede the signing date.

### Wage-structure features derived jointly with existing player salaries

- known-player wage sum divided by Capology club payroll, as a coverage ratio;
- median, maximum, top-five share, HHI, and Gini of known player wages, emitted
  only above a preregistered minimum coverage threshold;
- proposed player wage percentile and share of the known wage bill;
- distance from the club's existing wage tiers.

These structure features require a separate coverage audit. A concentration
statistic calculated from a thin or selectively missing player list can be more
misleading than having no feature.

## Features excluded from the first experiment

- **Current/signing-season totals:** post-decision leakage risk.
- **Capology's 2025-adjusted euro values:** these embed a retrospective future
  price-base adjustment. Use nominal amounts with within-season ranks or a
  documented origin-safe deflator instead.
- **Position payroll fields:** unavailable (`Locked`).
- **Bonus as zero before publication begins:** false historical signal.
- **Raw club name as a model feature:** identity is for joining and subgroup
  checks, not memorizing club labels.
- **All financial fields at once:** redundant currencies and algebraic
  identities would create collinearity without adding information.

## Required experiment before advancement

1. ~~Review and version the 16 unresolved club aliases.~~ **Complete.**
2. ~~Build an immutable canonical club-season feature table with source hashes,
   definitions, and decision-time availability metadata.~~ **Complete:** 1,074
   rows, 64 documented fields, 13/13 build checks and 29/29 independent checks.
3. ~~Compare a locked existing baseline against small preregistered feature
   blocks: resource level, transfer context, and wage structure.~~ **Complete.**
4. ~~Use rolling origins; learn transforms, imputers, thresholds, and feature
   selection from training data only.~~ **Complete.** Missing club context was
   not imputed; the candidate was unavailable instead.
5. ~~Evaluate discrimination, calibration, coverage, stability by league and
   promoted-club status, and practical decision lift.~~ **Complete.**
6. ~~Advance only features that improve repeated-origin evidence without
   creating a material subgroup failure.~~ **Complete: no feature block
   advanced.** The final untouched gate remains sealed.

The canonical table is in
`Data/processed/capology_club_financial_context/canonical_club_season_financial_context.csv`.
It emits player-wage distribution features for 666 club-seasons only when at
least 15 player wages reconcile to within 5% of the club payroll; unsafe rows
remain null. This data is a credible answer to the missing club-resource-context
problem, but it is not evidence by itself that predictive performance will
improve. The controlled experiment is now complete: payroll context was
effectively neutral, recruitment and squad blocks worsened pooled Brier, and
every block failed the advancement gate. See
[the rolling-origin experiment](ENGINE_V2_CLUB_FINANCIAL_CONTEXT_EXPERIMENT.md).

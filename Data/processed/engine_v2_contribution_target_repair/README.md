# Engine V2 complete contribution-target repair — Run 1

Run 1 closes the known construction defects in the realized-contribution
target without fitting a model or opening 2024+ outcomes.

## Contract

A binary target exists only when both exact post-extension years are fully
observable, contain at least 10 captured extension-club matches, and retain at
least 80% of the contemporaneous median
club's original-league schedule over the exact same dates. A failed schedule
gate makes the outcome unavailable; it is never treated as zero.

Opportunity is still all captured club competitions. Annual share is
`same-club minutes / (90 × captured same-club games)` and is capped to [0,1].
The cap affects 6 annual values and changes no 25%
binary label. The positive policy remains at least 25% in **each** year, with
20% and 30% sensitivities published separately.

## Development result

- Legacy asymmetric reference cohort: 929
- Symmetric Year-1/Year-2 evidence cohort: 922
- Complete schedule-checked cohort: 878
- Rows made unavailable by schedule continuity: 44
- Positive targets under the complete 25% contract: 438

The peer-relative rule avoids falsely rejecting league-wide interruptions such
as the shortened 2019–20 Ligue 1 season while failing closed when a particular
club exits the top-flight coverage universe.

## Boundary

This target represents **sustained realized contribution to the extension
club**, not tactical role when available. Injury, loan, transfer, suspension,
international duty and non-selection remain part of realized contribution.
Run 1 fits zero models, changes zero deployment artifacts, and does not inspect
2024+ outcomes. Rolling-origin model development is a later run.

## Pre-existing repository-state warning

The V1 freeze comparison found 2 files already different
from the August 30 freeze record when this evidence package was built:
- `Data/processed/deployment_models/live_request_integrity.csv`
- `Data/processed/deployment_models/live_request_integrity.json`

Both are generated live-request-integrity reports that were already modified
in the working tree before Run 1. Run 1 does not alter them. Their exact Run 1
baseline hashes are recorded in `v1_freeze_comparison.csv`; resolving or
formally re-versioning that separate integrity-report divergence remains a
repository-governance task.

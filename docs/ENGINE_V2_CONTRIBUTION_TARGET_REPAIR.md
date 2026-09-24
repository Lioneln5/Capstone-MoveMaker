# Engine V2 complete contribution-target repair — Run 1

**Implementation date:** 2026-09-07

**Development outcomes used:** extension events signed through 2023

**2024+ final outcomes inspected:** no

**Models fitted:** none
**Deployment changes:** none

## Decision

Run 1 repairs the known construction defects in the contribution target. The
versioned target is now defensible as **sustained realized contribution to the
extension club**. It is not a measure of tactical role when available and it
does not recommend whether a contract should be extended.

Target construction is repaired; predictive validation is not complete. No
probability may be displayed from this target until affected development
models are rerun and a frozen survivor later passes untouched temporal and
player-disjoint evaluation.

## Versioned contract

The contract is
`engine_v2_sustained_realized_contribution_v2_2026-09-07`.

A binary label is available only when **both** exact post-extension years:

1. are fully observable;
2. contain at least 10 captured extension-club matches and valid minutes;
3. contain an original-Big-Five league schedule at least 80% as complete as
   the median club observed in the same competition over the exact same dates;
   and
4. have a normalized opportunity share in `[0,1]`.

The opportunity share remains:

`captured extension-club minutes / (90 × captured extension-club games)`

and is capped to `[0,1]`. The positive policy remains at least 25% in each
year. An unavailable year makes the whole target unavailable; it is never
coerced to a negative outcome.

## Why the schedule rule is peer-relative

A flat annual match floor confuses incomplete data with league-wide calendar
interruptions. Ligue 1's shortened 2019–20 season is the clearest case: a club
can have fewer than 30 original-league matches while still having schedule
coverage equal to its peers.

Run 1 instead compares the extension club with contemporaneous clubs in the
same original league and exact anniversary window. This keeps genuinely
complete shortened-season evidence while refusing club-specific breaks caused
by relegation, top-flight exit, or missing lower-tier coverage. At least 10
peer clubs are required. Eighty percent is a declared completeness boundary,
not a fitted performance threshold.

The original-league schedule is used only as the observable continuity check.
The contribution denominator still contains all captured extension-club
competitions, so domestic cups and European competition remain part of
realized club opportunity.

## Development evidence

| Cohort | Rows | Positive labels |
| --- | ---: | ---: |
| Legacy asymmetric reference | 929 | 454 |
| Symmetric Year-1/Year-2 evidence | 922 | 453 |
| Complete schedule-checked and normalized | 878 | 438 |

Forty-four rows that passed the symmetric gate fail schedule continuity. They
are now unavailable rather than negative. The exclusions are distributed
across every Big Five league: Bundesliga 4, La Liga 9, Ligue 1 7, Premier
League 11, and Serie A 13.

Six eligible annual opportunity shares exceed 100% before normalization
because extra-time minutes can exceed the `90 × games` denominator. All six
are capped at 100%; no 25% binary label changes.

Nearby schedule-boundary sensitivity is:

| Required peer coverage | Eligible rows |
| ---: | ---: |
| 70% | 883 |
| 75% | 882 |
| **80%** | **878** |
| 85% | 875 |
| 90% | 872 |

Nearby contribution-policy sensitivity on the 878 eligible rows is:

| Annual floor | Positive labels | Positive rate |
| ---: | ---: | ---: |
| 20% | 466 | 53.1% |
| **25%** | **438** | **49.9%** |
| 30% | 409 | 46.6% |

These sensitivities show that neither declared boundary is hiding a single
fragile cliff, but they do not prove that 80% or 25% is uniquely optimal.

## Semantic boundary

The target intentionally measures realized club contribution. Injury,
suspension, international duty, loan, permanent departure and tactical
non-selection do not reduce its opportunity denominator. This makes the
outcome useful for the club's realized extension exposure, but unsuitable for
claims about how much a healthy and available player would have played.

A role-when-available target requires comprehensive availability data and is
still unavailable. Temporary Displacement remains a separate outcome.
Meaningful Retention may use this repaired contribution component only after
its development evidence is rerun.

## Remaining release work

Run 1 closes target construction, not endpoint validation. The first required
development rerun is now complete: the fixed compact recent-involvement
candidate retained positive player-disjoint rolling-origin Brier skill and
advanced to a reliability audit. See the
[Run 2 development result](ENGINE_V2_CONTRIBUTION_MODEL_RUN2.md).

The remaining steps are:

1. evaluate calibration, temporal drift, uncertainty, missingness,
   league/position/age subgroups, and refusal/OOD behavior without opening
   2024+ outcomes;
2. freeze any surviving probability specification before looking at later
   outcomes; and
3. require a later untouched, player-disjoint temporal test before any live
   probability is authorized.

## Reproduction

```bash
python models/run_engine_v2_contribution_target_repair.py
python scripts/verify_engine_v2_contribution_target_repair.py
```

Evidence is stored in
`Data/processed/engine_v2_contribution_target_repair/`. The package includes
the versioned machine-readable contract, every exclusion, normalization cases,
threshold sensitivities, source and output hashes, build checks, and an
independent reconstruction of the schedule gate.

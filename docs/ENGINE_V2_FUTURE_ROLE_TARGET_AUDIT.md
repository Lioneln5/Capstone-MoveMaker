# Engine V2 future-role / meaningful-contribution target audit

**Audit date:** 2026-09-02

**Scope:** 2020-2023 development outcomes only

**Final 2024+ evaluation:** not performed

**Model fitting:** none
**Deployment changes:** none

## Decision

The frozen `target_sustained_meaningful_contribution` is **not defensible as a
pure future-role target** and cannot be promoted into Engine V2 under that
name.

Its strongest defensible interpretation is narrower: realized use by the
extension club, normalized by the extension club's observed match schedule.
At the audit date, even that interpretation required two repairs before
reuse; both construction repairs are now implemented in the later Run 1
record below:

1. symmetric Year-1 and Year-2 evidence eligibility; and
2. complete schedules after relegation or another explicit refusal rule for
   incomplete club windows.

The audit supports the existing movement-scope split:

- **Meaningful Retention** is the joint business outcome, but it must be
  rebuilt on a repaired contribution component before further validation;
- **Temporary Displacement** remains a separate movement outcome; and
- **role when available** is a different target that is not currently
  supportable with complete availability data.

No finding in this audit authorizes a model or a live probability.

## Symmetric evidence repair implemented

**Implementation date:** 2026-09-04

The first required repair is now enforced by
`models/engine_v2/contribution_target_contract.py`. An Engine V2 contribution
label is nullable and is available only when both Year 1 and Year 2
independently have a fully observable window, at least 10 captured
extension-club matches, and a nonmissing opportunity share. Stored evidence
flags are checked against those primitive conditions and a disagreement fails
closed.

On the frozen 2020-2023 development evaluation rows, this changes the eligible
cohort from 929 to 922. The seven previously invalid rows become unavailable;
none are recoded as failures, and the 922 eligible labels remain unchanged.
The implementation and independent verifier are recorded under
`Data/processed/engine_v2_symmetric_contribution_evidence/`.

This closes only the asymmetric-evidence subproblem. The target and Meaningful
Retention remain release-blocked pending club-schedule completeness repair,
normalization at 100%, a complete development rerun, and later untouched final
temporal evaluation.

## Complete contribution-target repair implemented

**Implementation date:** 2026-09-07

Run 1 completes the remaining target-construction repairs in the versioned
`models/engine_v2/complete_contribution_target_contract.py`. Original-league
schedule coverage must now reach at least 80% of the contemporaneous exact-
window peer median in both years. This preserves complete league-wide
interruptions, including shortened COVID-era Ligue 1 windows, while refusing
club-specific top-flight coverage breaks. Annual opportunity shares are also
recomputed from minutes and captured games and capped to `[0,1]`.

On the 922-row symmetric 2020-2023 reference cohort, 44 incomplete-schedule
rows become unavailable, leaving 878 eligible labels and 438 positives. Six
annual shares are capped at 100% and no binary label changes because of that
normalization. The 2024+ cohort was not opened and no model was fitted.

This closes the construction defects identified by this audit. It does **not**
authorize Meaningful Retention or another probability: all affected rolling-
origin development evidence must now be rerun under the frozen contract, and
any survivor still requires later untouched player-disjoint temporal evidence.
See [the Run 1 repair record](ENGINE_V2_CONTRIBUTION_TARGET_REPAIR.md) and
`Data/processed/engine_v2_contribution_target_repair/`.

## Exact current construction

For each extension, the integration layer constructs two exact anniversary
windows:

- Year 1: after signing through day 365; and
- Year 2: after day 365 through day 730.

The numerator is minutes played for the extension club in captured
competitions. The denominator is `90 × captured extension-club matches` in the
same window. The label is one only when both annual opportunity shares are at
least 25%.

This is not “25% of the player's available minutes over 24 months.” It is two
separate floors against observed club-match capacity. Injury, suspension,
international duty, loan, permanent departure and tactical non-selection do
not reduce the denominator.

## Audit population

The current frozen rolling-origin evaluation contains 929 events signed from
2020 through 2023. Requiring the missing symmetric Year-1 evidence check leaves
922 events. Of those, 453 (49.1%) meet the current 25% floor in both years.

The 2024+ final cohort was neither scored nor included in any calculation or
output.

## Findings

### 1. Year-1 evidence is assigned asymmetrically

The current cohort requires at least 10 observed club matches before signing
and in Year 2, but not in Year 1. Seven of the 929 evaluated rows fail the
matching Year-1 requirement:

- four have only two to nine captured matches; and
- three have zero captured matches and a missing Year-1 share.

All seven nevertheless receive a binary target. The three missing values
become negative because `NaN >= 0.25` evaluates false. One under-covered row is
even assigned a positive label from only eight Year-1 matches. This is a target
construction defect, not a modeling limitation.

**Required repair:** a target is missing unless both Year-1 and Year-2 windows
pass the same evidence rule and both shares are observed.

### 2. The 25% floor is a policy boundary, not a discovered breakpoint

The cutoff is coherent as the declared boundary between fringe and rotation
involvement, but no external football outcome establishes 25% as uniquely
correct.

| Annual floor in both years | Positive rate | Labels changed versus 25% |
| ---: | ---: | ---: |
| 20% | 52.4% | 30 / 922 |
| 25% | 49.1% | — |
| 30% | 45.9% | 30 / 922 |

Only 2.8% of cases sit within 2.5 percentage points of the 25% boundary on the
weaker of their two years. The label is therefore not dominated by tiny
rounding differences, but it remains a chosen materiality rule.

**Decision:** retain 25% only when the product and documentation call it an
explicit policy threshold and expose sensitivity. Do not describe it as an
empirically proven football law.

### 3. All-competition club-match capacity is coherent, but not availability

For realized contribution to the extension club, all captured club
competitions are more aligned with the question than domestic-league matches
alone. On the 916 rows with adequate original-league observations in both
years, switching to original-Big-Five-league-only capacity changes 21 labels
(2.3%). Median annual share differences are 0.8 to 1.4 percentage points;
90th-percentile differences are roughly 5.6 to 6.1 points.

This supports keeping an all-competition denominator for **realized club
contribution**. It does not make it an availability-adjusted role measure.

### 4. Injury meaningfully changes the target for a minority of players

The validated partial injury source supports a constrained diagnostic subset
of 703 rows; 701 retain a nonzero denominator after adjustment. Removing only
extension-club matches that fall inside validated, same-club injury spells
changes 29 labels, all from not-sustained to sustained. The positive rate moves
from 57.2% to 61.3% in that selected subset.

High burdens show the intuitive relationship that lighter burdens do not:

| Two-year injury burden | Rows | Not-sustained rate |
| --- | ---: | ---: |
| 0 days | 132 | 46.2% |
| 1-30 days | 175 | 37.7% |
| 31-90 days | 201 | 33.8% |
| 91-180 days | 115 | 49.6% |
| 181+ days | 80 | 62.5% |

This is descriptive, not causal. The injury source is partial, and suspension,
international duty, illness outside the recorded source, and tactical
non-selection remain unobserved. A partly adjusted denominator would therefore
create a new inconsistency.

**Decision:** injuries should continue to count in a realized-contribution
outcome. A pure role-when-available target must remain unavailable until a
complete and time-safe availability layer exists.

### 5. Movement explains most negative labels

| First movement state within 24 months | Rows | Not sustained | Within-state failure rate | Share of all failures |
| --- | ---: | ---: | ---: | ---: |
| No outbound | 553 | 162 | 29.3% | 34.5% |
| Temporary first | 200 | 182 | 91.0% | 38.8% |
| Permanent first | 168 | 124 | 73.8% | 26.4% |
| Other/unresolved first | 1 | 1 | 100.0% | 0.2% |

All outbound-first states account for 65.5% of negative contribution labels.
The target is therefore mostly not a clean measure of the player's role while
remaining at the club. A loan is a genuine failure of immediate contribution
to the parent club, but it is simultaneously a distinct football mechanism.

**Decision:** do not market this label as pure future role. Keep Temporary
Displacement separate and use Meaningful Retention for the explicitly joint
outcome.

### 6. Relegation and club-season coverage break the annual denominator

The current minimum of 10 captured club matches is too weak for a nominal
365-day opportunity share. Normal strict-cohort windows contain approximately
40-50 club matches, but 47 of 922 events have fewer than 30 captured matches in
at least one year. The affected clubs include Burnley, Cagliari, Hertha Berlin,
Metz, Mallorca, Schalke, Sheffield United, Spezia and Watford around seasons in
which top-flight continuity or source coverage changes.

This pattern is consistent with relegation/top-flight exit and the source no
longer capturing a complete lower-tier schedule. It cannot be repaired by
pretending the observed 10-19 matches were the club's entire annual capacity.
The pooled target rate barely changes when these rows are excluded, but that
does not save the individual labels.

| Minimum captured club matches in each year | Eligible | Excluded | Positive rate |
| ---: | ---: | ---: | ---: |
| 10 | 922 | 0 | 49.1% |
| 20 | 900 | 22 | 49.6% |
| 30 | 875 | 47 | 50.2% |
| 34 | 848 | 74 | 50.5% |

**Required repair:** integrate lower-tier schedules for affected extension
clubs or mark the outcome unavailable under a frozen completeness rule. A
30-game rule is useful for sensitivity, not automatically the final gate.

### 7. “Both years” and “two years total” are materially different outcomes

The current conjunction enforces persistence: one weak year fails the label
even if the other year is strong. A game-weighted 730-day average instead
measures cumulative contribution and changes 146 of 922 labels (15.8%). An
unweighted average changes 143 labels.

Neither definition is universally superior:

- the two annual floors answer whether involvement was sustained; and
- the cumulative share answers how much of the two-year opportunity was
  realized overall.

**Decision:** keep separate annual floors only for a target explicitly named
“sustained.” Never paraphrase it as less than 25% of total available minutes
over 24 months.

### 8. Six annual shares exceed 100%

The integration denominator assumes 90 minutes for every match while player
minutes can include extra time. Values are capped at 1.05, so six strict-cohort
Year-1/Year-2 shares exceed 100%. This does not change the 25% binary labels,
but it violates the ordinary meaning of a normalized share and affects future
continuous-role work.

**Required repair:** cap normalized shares at 1.00 or construct exact
match-minute capacity.

## Product contract after this audit

| Question | Defensible status |
| --- | --- |
| Did the player deliver sustained realized involvement for the extension club? | Rebuildable after symmetric evidence and club-schedule repairs; 25% remains a declared policy floor |
| Was the player meaningfully retained? | Primary joint candidate, but blocked until its contribution component is repaired and then revalidated |
| Was the player temporarily displaced? | Separate development candidate; unchanged by this target audit |
| What was the player's role when medically/administratively available? | Unavailable; current availability data are incomplete |
| Did the player permanently separate from the club? | Research-only under the existing movement closeout |

## Reproducibility

Run:

```bash
python models/run_engine_v2_future_role_target_audit.py
python scripts/verify_engine_v2_future_role_target_audit.py
```

The runner emits the full evidence set under
`Data/processed/engine_v2_future_role_target_audit/`. All 12 build checks and
all 25 independent verification checks must pass. The source manifest covers
the contract-extension master, Transfermarkt game and appearance tables, and
the validated partial injury source.

# Engine V2 future-role / meaningful-contribution target audit

This audit calculates and emits results only for 2020-2023 development
outcomes. It does not evaluate the 2024+ final cohort, fit a model, or change
any deployment artifact.

## Verdict

The existing `target_sustained_meaningful_contribution` is **not defensible as
a pure future-role outcome**. It is a composite of realized same-club use,
availability, loans/permanent departures, and—on 47 rows—an
annual club-match denominator with fewer than 30 observed matches in at least
one year. It can remain lineage evidence, but it must not be promoted under the
name “future role.”

A repaired **realized same-club contribution** outcome is defensible after:

1. requiring symmetric Year-1 and Year-2 evidence;
2. resolving lower-tier/relegated-club schedules or refusing incomplete years;
3. making the 25% floor an explicit policy threshold rather than an empirical
   law; and
4. capping shares at 100% (or using exact match-minute capacity).

Meaningful Retention should remain the joint primary extension outcome,
Temporary Displacement should remain separate, and a pure “role when
available” outcome should remain unavailable until availability coverage is
complete enough to support it.

## What the numbers mean

- Current frozen evaluation logic selects 929
  rows; the symmetric correction leaves 922. The 7
  removed rows were labeled without the required Year-1 evidence.
- At 25%, 453 of 922 strict rows
  meet the floor in both contract-anniversary years.
- A cumulative, game-weighted two-year definition changes
  146 labels. That is not a
  better version of the same question; it is a different question.
- Temporary-first movement explains
  38.8% of all negative
  labels, and all outbound-first states explain
  65.5%. The current label is
  therefore not a clean selection-role measure.
- On 703 rows with independently constrained
  injury follow-up, an availability-adjusted club-match denominator changes
  29 labels. This is a sensitivity
  test, not a replacement target: suspensions and other unavailability remain
  missing.

See `audit_findings.csv` for the decision ledger and the other CSVs for each
mechanical decomposition.

# Engine V2 public-value horizon policy

**Adopted:** 2026-09-05; exposure correction 2026-09-09

**Scope:** incumbent-club extension public-value downside
**Authorization:** product containment only; no live Engine V2 scoring authorized

## Decision

The 12-month public-value downside estimate is the primary horizon. The
24-month estimate remains an available secondary option and must never be
silently suppressed, but it must display a prominent provisional-evidence
warning wherever a personalized value is shown.

## Evidence behind the decision

The current strict outcome definition requires a positive valuation at signing
and a positive future valuation within 90 days of the exact 12- or 24-month
landmark. The future landmark must be fully observable. Players used in
development through 2023 are removed from the later validation cohort.

Reconstruction from the current integration layer produces:

| Horizon | Eligible 2024 outcomes | Player-disjoint outcomes | Big Five league counts after identity removal | Smallest league |
| --- | ---: | ---: | --- | ---: |
| 12 months | 337 | 212 before the repaired complete-feature rule; 209 after | IT1 51; L1 48; GB1 46; ES1 41; FR1 23 after the repaired feature rule | 23 |
| 24 months | 41 | 26 | GB1 8; L1 6; ES1 5; IT1 4; FR1 3 | 3 |

The frozen V1 rolling-development evidence remains positive at both horizons.
However, the frozen V1 out-of-sample verifier also scored the available 2024+
value outcomes and inspected aggregate performance on 2026-08-22. Those later
cohorts are therefore not pristine Engine V2 tests. The 24-month horizon has an
additional follow-up and subgroup-size constraint; this is not evidence that
the earlier historical signal disappeared.

The proposal-free Engine V2 cleanup strengthens the distinction. In
player-disjoint 2020–2023 rolling development, the 12-month candidate reached
AUC 0.760 with a 1.0-point calibration gap and positive skill in all four
origins, but its goalkeeper 99% skill interval crossed zero. The 24-month
candidate reached AUC 0.823 with a 0.8-point calibration gap and supported all
nine broad league/position groups. This does not reverse the product horizon:
12 months remains primary because it matures sooner, while 24 months remains
secondary because later independent support accumulates much more slowly.

## Required interface behavior

1. Default to the 12-month, 25% downside scenario.
2. Describe 12 months as the primary research horizon, not as finally production-validated; disclose that its available 2024+ outcomes were previously exposed in V1 verification and that goalkeeper skill remains uncertain under the strict subgroup gate.
3. Keep 24 months selectable.
4. When 24 months is selected, display this substance without euphemism:
   historical rolling-origin evidence is positive; the available 2024+
   outcomes were previously exposed in V1 verification; only 26
   player-disjoint 2024 outcomes are currently observable; the smallest Big
   Five league has 3; the estimate is provisional and not production-validated.
5. Do not offer the 12-month 50% downside endpoint. It remained exploratory and
   was not approved for display.
6. Continue to describe every value output as public-estimate downside, never
   accounting loss, guaranteed transfer proceeds, impairment, or ROI.

## Reopening the 24-month decision

The 24-month warning can be reconsidered only after the frozen candidate is
evaluated on a sufficiently large later, player-disjoint and previously
uninspected cohort under the predeclared pooled and Big-Five subgroup gates.
The same non-exposure requirement applies to final validation of the 12-month
candidate. Rearranging or deleting already inspected historical results does
not create a pristine final holdout.

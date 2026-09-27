# Engine V2 Phase 5: final-evaluation readiness and sealed wage test

Phase 5 asks a narrower question than “does the model score well?”: **is there
an honest, sufficiently mature, player-disjoint later cohort on which a final
claim can be made without changing the candidate afterward?**

No Engine V2 artifact was serialized or deployed. Frozen V1 files remain
byte-identical.

## Holdout readiness

The candidate final year was 2024. A player was removed from a module's final
cohort if that player appeared in development data through 2023. This enforces
the Phase-4 rule that identity clusters cannot cross development and test.

| Module | 2024 eligible before identity removal | Player-disjoint rows | Smallest league | Decision |
| --- | ---: | ---: | ---: | --- |
| Future role | 81 | 56 | 6 | Not opened: not pristine and underpowered |
| Club continuity | 290 | 176 | 23 | Not opened: blocked and not pristine |
| 24-month value downside | 41 | 26 | 3 | Not opened in Phase 5: previously exposed and underpowered |
| Annual-wage peer benchmark | 376 | 235 | 33 | Opened once as the sealed final test |

Future-role and continuity labels for the 81 overlapping eligible 2024 events
were included in Phase 4's aggregate target-relationship table. Phase 4 did
not inspect 2024 predictions or performance metrics, but those labels are no
longer pristine. Phase 5 records that error rather than repeating the earlier
claim that the entire 2024 holdout was untouched.

A later exposure audit corrected the original Phase-5 record: the frozen V1
out-of-sample verifier had already attached 2024+ value labels, scored the V1
models, and inspected aggregate performance on 2026-08-22. The value target
was therefore not pristine when Phase 5 ran. At 24 months it is also
underpowered: 26 player-disjoint rows cannot support pooled and Big-Five
subgroup gates. Phase 5 itself did not rescore the target.

## Immutable wage result

The frozen `B2_plus_prior_wage` candidate was fit through 2023 using the
predeclared validation recipe and evaluated once on 235 player-disjoint 2024
extensions.

| Final evidence | Result |
| --- | ---: |
| Candidate log-MAE | 0.5046 |
| Chronology-only median log-MAE | 0.8359 |
| Log-MAE improvement | +0.3313 |
| Nominal 80% reference-range coverage | 77.0% |
| Refusal rate | 2.1% |
| Ligue 1 reference-range coverage | **60.6%** |

The pooled result is useful, but the candidate fails the frozen league gate:
Ligue 1 coverage is below the required 70% lower bound. The immutable Phase-5
decision is therefore `block_artifact_build`. The holdout may not be used to
retune the same candidate, widen its interval, or relax the gate.

This does not prove historical wage context is useless. It shows that the
current interval method is not reliable enough across every advertised league
to support a Big-Five user-facing benchmark.

## Product decision

- Future role: no final claim; wait for a genuinely later, fully mature cohort.
- Continuity: still release-blocked by E2-CRIT-001; 2024 is not a repair set.
- Value downside: 2024+ is permanently restricted to confirmation of the exact
  frozen V1 artifacts. Engine V2 requires a genuinely later, mature and
  uninspected final cohort.
- Wage benchmark: final pooled performance was positive, but the current
  Big-Five artifact is blocked by the failed Ligue 1 coverage gate.
- Overall Engine V2: **not ready for deployment**.

These blockers are recorded independently as E2-CRIT-002 and E2-CRIT-003 in
the [critical-issues register](ENGINE_V2_CRITICAL_ISSUES.md).

> **Later scope note:** a subsequent football-validity audit added
> E2-CRIT-004 for the future-role / meaningful-contribution target itself.
> That later blocker is separate from this Phase-5 holdout-readiness finding.

> **Later product-containment note (2026-09-05):** this immutable Phase-5
> wage result remains unchanged. The product now uses **12-month public-value
> downside as its primary horizon**. The 24-month estimate remains available
> as provisional detail. A 2026-09-09 correction additionally records that
> both value horizons' 2024+ outcomes were already exposed by V1 verification;
> the 24-month player-disjoint cohort has only 26 outcomes, including 3 in the
> smallest Big Five league. This is containment and transparent disclosure,
> not a claim that either Engine V2 horizon is production-validated.

## Reproduce and verify

```bash
python models/run_engine_v2_final_evaluation_gate.py
python scripts/verify_engine_v2_final_evaluation_gate.py
```

The verifier independently reconstructs the player-disjoint wage cohort,
validation-selected Ridge parameter, every final prediction, interval width,
pooled metric, subgroup result, release decision, and frozen-V1 hashes. All 26
checks pass after the exposure correction.

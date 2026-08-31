# Engine V2 Phase 5: final-evaluation gate

Phase 5 audits whether 2024 can honestly serve as a final temporal test. It opens only the annual-wage benchmark holdout; no other 2024 target is scored.

## Readiness decision

- **future_role:** `do_not_open_not_pristine` — 56 player-disjoint eligible rows; minimum league count 6.
- **club_continuity:** `do_not_open_not_pristine` — 176 player-disjoint eligible rows; minimum league count 23.
- **public_value_downside:** `do_not_open_underpowered` — 26 player-disjoint eligible rows; minimum league count 3.
- **wage_benchmark:** `open_sealed_2024_holdout` — 235 player-disjoint eligible rows; minimum league count 33.

## Sealed wage result

The frozen B2 prior-wage candidate was evaluated once on 235 player-disjoint 2024 extensions. Log-MAE was 0.5046 versus 0.8359 for chronology-only median; the nominal 80% range covered 77.0%; refusal was 2.1%.
The immutable release decision is `block_artifact_build`. The candidate was not serialized or deployed.

## Why the remaining holdouts stayed closed

Future-role and continuity labels for 81 eligible 2024 events were already included in Phase 4's aggregate joint-target table. No 2024 predictions or performance metrics were inspected, but these labels are no longer pristine. After also removing players seen through 2023, only 56 future-role rows remain. Public-value downside remains unexposed, but only 26 player-disjoint rows are mature. Neither can support the frozen pooled and Big-Five subgroup gates.

Phase 5 therefore records readiness and refusal; it does not manufacture a final result from an underpowered sample.

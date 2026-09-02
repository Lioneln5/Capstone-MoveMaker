# Engine V2 movement and retention scope closeout

**Decision date:** 2026-09-02
**Status:** generic continuity retired; replacement questions frozen as
separate research candidates; no movement/retention output deployed.

## Final product decision

| Product output | Exact outcome | Status | Numeric display |
| --- | --- | --- | --- |
| Generic club continuity | Any loan or permanent outbound movement within 24 months | **Retired** | Prohibited |
| Meaningful Retention | No outbound within 24 months and at least 25% same-club opportunity share in both Year 1 and Year 2 | Development candidate | Prohibited until later final validation |
| Temporary Displacement Risk | Loan or loan return is the first resolved interruption within 24 months | Development candidate | Prohibited until later final validation |
| Permanent Separation | Permanent relationship ends within 24 months, even if a loan happened first | Research-only / unavailable | Prohibited |
| Sustained Meaningful Contribution | At least 25% same-club opportunity share in both years without the no-outbound requirement | Frozen V1 historical evidence | Prohibited in the current HTTP product |

Meaningful Retention is a joint sporting-retention outcome. It must not be
implemented by relabeling the older contribution-only probability.

## Why this closes the continuity blocker

`E2-CRIT-001` existed because the product claimed a personalized Big-Five
continuity probability despite a failed Ligue 1 reliability gate. Repeated
repair attempts using club finances, lagged club behavior, squad competition,
role trajectory, manager stability, formation stability and manager-transition
context did not produce a reliable replacement.

The issue is resolved through **Resolution C — endpoint retired**, not through
model repair:

- generic continuity is absent from the current product scope and public UI;
- the HTTP application has no scoring-enable override;
- no proxy or composite is permitted to reconstruct the retired score;
- Permanent Separation explicitly returns no personalized probability; and
- the two promising replacement questions remain labeled development
  candidates rather than deployed products.

This does not imply that permanent separation is impossible to model. It means
the unresolved research question no longer blocks unrelated Engine V2 work.

## Evidence supporting the replacement questions

Across the established 2020–2023 rolling development origins:

| Endpoint | Rows | AUC | Brier skill vs chronology | Player-cluster 95% interval | Origin wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Meaningful Retention | 1,560 | 0.759 | +0.0352 | [+0.0260, +0.0443] | 4/4 |
| Temporary Displacement Risk | 1,553 | 0.777 | +0.0291 | [+0.0209, +0.0375] | 4/4 |
| Permanent Separation | 1,560 | 0.628 | +0.0127 pooled | [+0.0055, +0.0200] | 4/4 |

Permanent Separation remains unavailable because pooled improvement conceals
negative league-level skill in Bundesliga and Ligue 1. The first two rows are
development evidence only. The planned 2024 final cohort is exposed or
underpowered, so neither may be promoted until a later untouched, mature,
player-disjoint cohort exists.

## Enforcement

The machine-readable contract is
`models/engine_v2/movement_product_scope_contract.py`. It freezes status,
target mapping, display authorization and reason codes. The closeout runner
publishes compact evidence to
`Data/processed/engine_v2_movement_scope_closeout/`; the independent verifier
checks product copy, API refusal, evidence reconstruction, V1 hashes and clean
rerun reproducibility.

Historical model code, raw `any_outbound` labels and earlier diagnostics remain
unchanged where needed for lineage. Historical evidence is not an active
product claim.

## What reopens this research

- **Meaningful Retention / Temporary Displacement:** a fully mature, untouched,
  player-disjoint temporal cohort capable of supporting the declared pooled
  and league-level gates.
- **Permanent Separation:** a materially new, time-safe mechanism plus a frozen
  candidate with credible cross-league improvement.

Additional versions of already rejected broad context blocks do not qualify as
materially new evidence.

## Evidence index

- `models/engine_v2/movement_target_contract.py`
- `models/engine_v2/movement_product_scope_contract.py`
- `models/run_engine_v2_movement_scope_closeout.py`
- `scripts/verify_engine_v2_movement_scope_closeout.py`
- `docs/ENGINE_V2_MOVEMENT_STATE_AND_PROFILE_DIAGNOSTIC.md`
- `docs/ENGINE_V2_LAGGED_CLUB_BEHAVIOR_DIAGNOSTIC.md`
- `docs/ENGINE_V2_ROLE_SQUAD_DIAGNOSTIC.md`
- `docs/ENGINE_V2_MANAGER_TACTICAL_DIAGNOSTIC.md`
- `Data/processed/engine_v2_movement_scope_closeout/product_scope_contract.json`
- `Data/processed/engine_v2_movement_scope_closeout/independent_verification.json`

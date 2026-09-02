# Engine V2 original twelve-problem audit

**Audit date:** 2026-08-30  
**Status review:** 2026-09-02

This is the durable ledger for the twelve problems found in the adversarial
post-presentation audit of the public MoveMaker engine. It preserves the
original findings separately from the later Engine V2 phase structure and the
smaller register of release-blocking failures.

## Status definitions

- **FIXED:** the underlying defect has been removed from the applicable V2
  design or shared implementation and is independently verifiable.
- **CONTAINED:** the unsafe behavior cannot reach the public because HTTP
  scoring is disabled or the output was retired, but end-to-end V2 runtime
  implementation remains incomplete.
- **DEFERRED:** the limitation is explicitly outside the current validated
  claim and requires a separate future data or research program.
- **OPEN:** an actionable defect remains neither repaired nor safely isolated.

A `FIXED` status does not authorize deployment. Engine V2 still requires a
valid final temporal evaluation and explicit artifact-promotion decision.

## Original findings and current disposition

| # | Original problem | Status | Current disposition and remaining condition |
| ---: | --- | --- | --- |
| 1 | Proposed contract terms improperly changed sporting-risk estimates. | **FIXED** | V2 predictive feature contracts exclude proposed wage, duration, commitment, salary rank/share and other commercial-scenario fields. Proposal terms remain deterministic financial inputs only. The feature-specification verifier enforces the separation. |
| 2 | The model could not determine whether a club should extend someone because every training row represented an observed extension. | **DEFERRED** | V2 is explicitly limited to historical outcome scenarios conditional on an extension. No unobserved “considered but not extended” population is fabricated. A separately designed observable renewal-eligibility cohort remains future research. |
| 3 | Goals and assists remained globally and incorrectly specified in 12 of 16 V1 endpoints. | **FIXED** | The frozen V1 artifacts remain unchanged for lineage but cannot score publicly. V2 headline candidates exclude raw goals and assists; the earlier role-contextualization evidence is retained rather than converted into manual positional bonuses. |
| 4 | The API scored arbitrary player-club-league combinations instead of enforcing incumbent-extension scope. | **CONTAINED** | The V2 validity boundary hard-refuses unverified player, club, league and incumbency evidence, and the current HTTP service refuses every scoring request. A canonical V2 resolver and scoring adapter are not yet deployed. |
| 5 | Omitting the optional normalized player-name field materially changed predictions and salary matching. | **CONTAINED** | Caller-supplied display or normalized names are not part of the V2 validity or predictive feature contract. The inactive V1 request/retrieval code still contains the optional field; canonical server-side identity resolution must replace it before V2 serving. |
| 6 | Extreme and unsupported inputs received precise-looking probabilities. | **CONTAINED** | Phase 3 implements missing-input, unseen-category, observed-range and robust-distance support checks with explicit refusal and no numeric result. Those checks are independently verified but are not yet connected to a live V2 HTTP scorer. |
| 7 | Strong pooled metrics concealed weak league, age, position and contract subgroups. | **FIXED** | V2 advancement gates independently audit advertised subgroups and block rather than average away failures. These gates exposed the continuity and Ligue 1 wage failures. Predictive weakness may remain, but it can no longer qualify through a pooled metric alone. |
| 8 | Independent overlapping models contradicted one another and required post-prediction clamps. | **CONTAINED** | V2 removed the 16-endpoint public grid, narrowed repeated thresholds and horizons, tested a coherent movement-state formulation, and retired generic continuity. Replacement movement questions remain non-deployed pending final evidence. |
| 9 | The API returned outcomes whose forecast horizon exceeded the proposed contract. | **CONTAINED** | The V2 boundary restricts proposals to two through six years and the candidate product uses declared 24-month questions rather than ordinary 36-month displays. The old permissive schema cannot score publicly; the rule still needs to travel through the future V2 API. |
| 10 | “Current club” was inferred from latest appearance and could disagree with the contract-holding club. | **CONTAINED** | V2 requires dated, verified incumbency evidence and refuses stale or ambiguous relationships. A live resolver reconciling contracts, salary panels, transfers, loans, rosters and appearances remains to be implemented. |
| 11 | Stale or incomplete current data produced warnings while ordinary scores continued. | **CONTAINED** | V2 applies module-specific freshness and coverage refusals for sporting data, public value, salary panels and incumbency evidence. The rules are verified, but no V2 scoring API currently consumes them end to end. |
| 12 | The shared feature-contribution explainer assumed classifier-shaped intercepts and failed on scalar Ridge intercepts. | **FIXED** | `raw_contributions` now normalizes both coefficient and intercept shapes with NumPy before reconstruction. A dedicated regression verifier exercises both Ridge-style and LogisticRegression-style representations. |

## Summary

| Status | Issues | Count |
| --- | --- | ---: |
| **FIXED** | 1, 3, 7, 12 | 4 |
| **CONTAINED** | 4, 5, 6, 8, 9, 10, 11 | 7 |
| **DEFERRED** | 2 | 1 |
| **OPEN** | None | 0 |

The absence of an `OPEN` item in this original ledger does not mean the engine
is ready for deployment. Contained items require runtime completion, and later
research exposed additional release blockers recorded in
[`ENGINE_V2_CRITICAL_ISSUES.md`](ENGINE_V2_CRITICAL_ISSUES.md), including the
missing final temporal cohort and the failed Big-Five wage-benchmark gate.

## Verification

Run:

```bash
python scripts/verify_engine_v2_original_audit.py
```

The verifier checks the twelve unique issue rows, allowed status vocabulary,
summary counts, registry linkage, the shared Ridge-shape repair, and unchanged
behavior for classifier-shaped linear artifacts.

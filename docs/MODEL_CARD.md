# MoveMaker model card

## Model purpose

MoveMaker is an explainable research program for **incumbent-club contract extensions in the Big Five European leagues**. Frozen V1 evidence covers future role, outbound movement, public-value downside, and historical wage context. Engine V2 has not authorized any live scoring output.

The authoritative final case-study stopping decisions are recorded in the
[metric disposition freeze](CASE_STUDY_METRIC_DISPOSITIONS.md). That freeze
supersedes earlier advancement language when a later target repair, final gate,
or scope decision changed the interpretation.

**Current status:** Engine V1 is a frozen research prototype and HTTP scoring
is unconditionally disabled. Engine V2 retired generic continuity. Temporary
Displacement Risk remains a development candidate; Permanent Separation is
research-only and unavailable. A later target audit rejected pure future role.
The realized-contribution target is now construction-complete under a new
versioned contract. A compact recent-involvement candidate retained positive
player-disjoint rolling-origin development skill after the repair. Its Run 3
audit passed pooled calibration, uncertainty, refusal, drift, and monotonicity
gates but failed the declared all-league/all-position uncertainty gate.
Meaningful Retention remains non-live, and all predictive promotion still
requires reliability repair and later untouched temporal evidence.
The metrics below document historical or rolling-development evidence, not a
live product.

The product deliberately avoids a single success score. Each output answers a different decision question and should be interpreted with its own target definition, time horizon, and uncertainty.

## Intended users and uses

Intended users include club recruitment, sporting, finance, and contract-planning staff. Appropriate uses are:

- prioritizing extension cases for deeper review;
- comparing proposed fixed-wage commitment with historical peer extensions;
- investigating where role, movement, and public-value evidence disagree; and
- supporting—not replacing—staff judgment and internal information.

## Out-of-scope uses

MoveMaker is not validated for:

- player-to-player or player-to-club tactical compatibility;
- hypothetical destination-club transfers;
- automated extend/do-not-extend decisions;
- exact future minutes, departure dates, transfer fees, or realized sale proceeds;
- accounting loss, expected loss, ROI, or projected savings;
- betting or investment recommendations; or
- leagues, seasons, and player profiles outside the documented data support.

## Frozen V1 decision signals

The application exposes 16 separately validated endpoints. The main user-facing signals are:

| Signal | Target and horizon | Verified out-of-time evidence | Interpretation boundary |
| --- | --- | --- | --- |
| Year-2 opportunity | Same-club opportunity share in Year 2 | MAE 0.2369; R² 0.249; Spearman 0.512; improvement +0.0586, 95% CI [0.0485, 0.0686], 4/4 origins | Historical evidence only; target coverage is not cleared for a pure role claim |
| Sustained meaningful contribution | At least 25% same-club opportunity share in both Year 1 and Year 2 | AUC 0.788; AP 0.732; Brier 0.1869; improvement +0.0643, 4/4 origins | Historical composite of realized use, availability, movement and club-schedule coverage; not deployable as pure role |
| Generic continuity | Any recorded outbound movement by 24 months | AUC 0.710; Brier 0.2191; improvement +0.0363, 4/4 origins | **Retired V1 endpoint**; blended temporary and permanent movement |
| Public-value downside, primary horizon | Public market-value decline of at least 25% by 12 months | Repaired proposal-free rolling evidence: AUC 0.760, Brier 0.1794, calibration gap 0.010, positive skill in 4/4 origins; goalkeeper 99% skill interval crosses zero; 209 player-disjoint 2024 cases exist under the repaired complete-feature rule but were previously exposed | Primary research horizon; development candidate needing goalkeeper reliability repair and a pristine final test |
| Public-value downside, provisional horizon | Public market-value decline of at least 25% by 24 months | Repaired proposal-free rolling evidence: AUC 0.823, Brier 0.1697, calibration gap 0.008, positive skill in 4/4 origins; 9/9 broad groups supported; 26 player-disjoint 2024 cases exist but were previously exposed, smallest league n=3 | Frozen development candidate; secondary/provisional horizon awaiting a sufficiently large pristine final test |
| Wage commitment benchmark | Proposed annual fixed wage × term versus historical extensions | R² 0.644; Spearman 0.832; approximately 80% empirical reference-range coverage | Historical context, not an optimal or “fair” wage |

An attempted integrated contract score was rejected: its AUC was 0.6830 versus 0.6885 for opportunity risk alone, with the difference interval crossing zero. Production therefore preserves the signals separately.

## Engine V2 movement and retention scope

| Output | Exact definition | Current authorization |
| --- | --- | --- |
| Meaningful Retention | No outbound within 24 months and at least 25% same-club opportunity share in both Year 1 and Year 2 | Contribution component rebuilt and its compact candidate passed Run 2 development gates, but Run 3 found insufficient 99% skill certainty in La Liga, Serie A, and Midfield; joint endpoint remains unmodeled and non-live |
| Temporary Displacement Risk | Loan or loan return is the first resolved outbound interruption within 24 months | Development candidate; later untouched temporal validation required |
| Permanent Separation | Permanent relationship with the extension club ends within 24 months, even if a loan occurred first | Research-only; no personalized probability |
| Generic continuity | Any loan or permanent outbound movement | Retired; no proxy or composite may reintroduce it |

No row in this table is authorized for live numeric display. Meaningful
Retention is not the contribution-only V1 target under a new name.

## Evaluation design

Primary performance claims use rolling chronological, out-of-time evaluation rather than random train/test splitting. Earlier extension cohorts train the model and later cohorts evaluate it, approximating how the product would have behaved on future decisions. Saved independent-verification artifacts check the declared cohorts, metrics, manifests, model selection, and deployment parity.

The sustained-contribution endpoint uses the frozen C6 compact model. C6 removed unstable goals- and assists-per-90 inputs and advanced through a declared compact non-inferiority route; it was not claimed to be statistically superior. Later role-contextualization candidates remain research-only and did not replace other V1 endpoints.

## Inputs and missingness

Inputs combine public player profile, role/opportunity, contract, wage, club, league, and market-value information where supported. The live retriever keeps unavailable fields missing rather than silently converting them to zero. The interface warns when a profile is sparse or outside the historical peer range.

Current live inputs and frozen evaluation evidence have different vintages. Newer data may refresh a live player profile without rewriting historical out-of-time results.

## Known limitations and risks

- Public salary, contract, injury, and market-value data are estimates and have uneven coverage.
- Fixed wages exclude bonuses, taxes, agent fees, clauses, and options unless explicitly stated.
- Public market value is not an accounting valuation or guaranteed transfer price.
- The available 2024+ public-value labels and V1 performance were already inspected. They may confirm only the exact frozen V1 artifacts and cannot select, calibrate, or promote Engine V2. The repaired value candidate is limited to pre-2024 rolling-development evidence until a genuinely later untouched cohort matures.
- Historical “any outbound” includes loan or permanent movement and does not identify the cause; this is why Engine V2 retired it as a product endpoint.
- Low contribution is threshold-defined and may include injury, tactical, developmental, or market-driven cases.
- The frozen V1 contribution cohort does not apply a symmetric Year-1 evidence gate and includes incomplete club schedules. Engine V2 repairs both defects with nullable labels, peer-relative schedule completeness, and bounded shares. A compact replacement candidate passed Run 2 development gates but failed Run 3's all-league/all-position uncertainty gate; no replacement model is authorized. See the [target audit](ENGINE_V2_FUTURE_ROLE_TARGET_AUDIT.md), [Run 1 repair](ENGINE_V2_CONTRIBUTION_TARGET_REPAIR.md), [Run 2 development rerun](ENGINE_V2_CONTRIBUTION_MODEL_RUN2.md), and [Run 3 reliability audit](ENGINE_V2_CONTRIBUTION_RELIABILITY_RUN3.md).
- Subgroup and extreme-value samples can be small; star-player outputs may be extrapolations.
- Manager changes, squad competition, medical detail, contract options, release clauses, bonuses, and internal club valuations are not comprehensively observed.
- Model probabilities support prioritization; they do not establish causality.

## Human oversight

Every output should be reviewed alongside medical information, scouting judgment, tactical plans, squad depth, contract clauses, and internal financial assumptions. Users should investigate disagreement between modules rather than averaging them into an overall score.

## Versioning and audit trail

Frozen V1 artifacts and their original manifest live under `Data/processed/deployment_models/`. Their byte-level Phase 0 inventory is recorded in `docs/releases/engine_v1_freeze_2026-08-30.json` and verified by `scripts/verify_engine_v1_freeze.py`. Research runners and independent verifiers are registered in `models/README.md`. V2 must use separate artifact paths; any future live replacement requires a newly versioned evaluation, deployment parity checks, explainability checks, and smoke tests.

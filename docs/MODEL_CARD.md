# MoveMaker model card

## Model purpose

MoveMaker is an explainable decision-support system for **incumbent-club contract extensions in the Big Five European leagues**. It helps a club screen a proposed extension by presenting separate estimates for future role, club continuity, public-value downside, and historical wage context.

**Current status:** Engine V1 is a frozen research prototype. Public scoring is
disabled by default while Engine V2 validity repairs are evaluated. The
metrics below document the frozen V1 evidence; they are not a claim that the
current public site is a production club-decision engine.

The product deliberately avoids a single success score. Each output answers a different decision question and should be interpreted with its own target definition, time horizon, and uncertainty.

## Intended users and uses

Intended users include club recruitment, sporting, finance, and contract-planning staff. Appropriate uses are:

- prioritizing extension cases for deeper review;
- comparing proposed fixed-wage commitment with historical peer extensions;
- identifying where role, continuity, and public-value risk disagree; and
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
| Year-2 opportunity | Same-club opportunity share in Year 2 | MAE 0.2369; R² 0.249; Spearman 0.512; improvement +0.0586, 95% CI [0.0485, 0.0686], 4/4 origins | Broad future-role outlook, not exact minutes |
| Sustained meaningful contribution | At least 25% same-club opportunity share in both Year 1 and Year 2 | AUC 0.788; AP 0.732; Brier 0.1869; improvement +0.0643, 4/4 origins | A probability of meeting the defined involvement threshold |
| Club continuity | Any recorded outbound movement by 24 months | AUC 0.710; Brier 0.2191; improvement +0.0363, 4/4 origins | Continuous-stay risk, not an exact departure date or reason |
| Public-value downside | Public market-value decline of at least 25% by 24 months | AUC 0.821 | Public valuation risk, not cash loss or sale proceeds |
| Wage commitment benchmark | Proposed annual fixed wage × term versus historical extensions | R² 0.644; Spearman 0.832; approximately 80% empirical reference-range coverage | Historical context, not an optimal or “fair” wage |

An attempted integrated contract score was rejected: its AUC was 0.6830 versus 0.6885 for opportunity risk alone, with the difference interval crossing zero. Production therefore preserves the signals separately.

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
- “Any outbound” can include loan or permanent movement; it does not identify the cause.
- Low contribution is threshold-defined and may include injury, tactical, developmental, or market-driven cases.
- Subgroup and extreme-value samples can be small; star-player outputs may be extrapolations.
- Manager changes, squad competition, medical detail, contract options, release clauses, bonuses, and internal club valuations are not comprehensively observed.
- Model probabilities support prioritization; they do not establish causality.

## Human oversight

Every output should be reviewed alongside medical information, scouting judgment, tactical plans, squad depth, contract clauses, and internal financial assumptions. Users should investigate disagreement between modules rather than averaging them into an overall score.

## Versioning and audit trail

Frozen V1 artifacts and their original manifest live under `Data/processed/deployment_models/`. Their byte-level Phase 0 inventory is recorded in `docs/releases/engine_v1_freeze_2026-08-30.json` and verified by `scripts/verify_engine_v1_freeze.py`. Research runners and independent verifiers are registered in `models/README.md`. V2 must use separate artifact paths; any future live replacement requires a newly versioned evaluation, deployment parity checks, explainability checks, and smoke tests.

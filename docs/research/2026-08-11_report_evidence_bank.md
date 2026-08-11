# MoveMaker research-paper evidence bank

## Intended use

This document is a source bank for a five-page capstone research report. It condenses the August 11 development session into report-ready research questions, methods, verified results, limitations, business implications, and draft prose. It is not an external-literature bibliography. Before final submission, add and cite academic or industry sources required by the course.

All numerical claims below were reconciled to local project artifacts with passing independent verification. Conversational ideas are included only when identified as motivation or decision context.

## Recommended paper title

**From Compatibility Prediction to Transfer-Risk Screening: An Evidence-Led Scope Pivot in Football Recruitment Analytics**

Alternative commercial title:

**MoveMaker: A Modular Due-Diligence System for Market-Value Downside, Player Under-Utilization, and Transfer-Fee Context**

## One-sentence thesis

Detailed public player and club data did not support a universal player-club compatibility or ROI model, but chronological out-of-sample testing supported a narrower modular product that screens estimated market-value downside, playing-opportunity under-utilization, retention, reported-fee context, and overlapping transfer risk.

## Abstract-ready summary

This project investigated whether public pre-transfer football data could predict player-club compatibility and the overall success or return on investment of professional player transfers. The original rich compatibility design produced only 37–58 observations in its single final test slices and 162–207 pooled evaluation observations after four rolling chronological origins. Corrected evaluation found a repeatable compatibility increment for destination performance but not for playing opportunity or adaptation, rejecting the feasibility of a universal compatibility score. The project therefore conducted a financial feasibility audit and reframed the product as modular transfer-risk screening. A compact 11-field market/profile model strongly ranked 24-month estimated market-value downside, achieving AUC 0.851 for declines of at least 25% and reducing Brier score from 0.2474 to 0.1584. Paid-transfer models improved prediction of continuous playing opportunity and severe under-utilization, while a compact destination-context refinement further reduced opportunity-share MAE from 0.1892 to 0.1818. A reported-fee benchmark improved MAE-log by 0.0639 over pre-transfer market value, although its empirical ranges remained too broad for precise fair-price claims. A final component risk matrix concentrated severe dual events but did not establish a calibrated joint probability. The resulting MoveMaker concept is an explainable due-diligence system rather than a compatibility, expected-loss, or accounting-ROI engine.

## Research problem and scope evolution

### Original research question

> Can pre-transfer player characteristics and player-destination compatibility features predict the sporting success and financial return of a professional football transfer?

### Revised primary research question

> Which commercially relevant transfer risks can be predicted credibly from the public pre-transfer data available to MoveMaker?

### Final component questions

1. Can pre-transfer evidence predict a substantial decline in estimated market value within 24 months?
2. Can a paid signing’s 24-month playing opportunity and risk of severe or meaningful under-utilization be estimated?
3. Can transfer-time retention estimates be improved after observing first-year involvement?
4. Can a reported transfer fee be benchmarked against comparable historical deals?
5. Does combining independently validated value and opportunity components produce additional joint-risk discrimination?

### Hypotheses and outcomes

| Hypothesis | Evidence outcome | Final interpretation |
|---|---|---|
| Rich compatibility features materially improve opportunity prediction | Not supported as a general claim | Rolling policy improvement 0.002214 MAE; CI crosses zero |
| Rich compatibility features improve destination performance | Narrowly supported | Improvement 0.012560; CI excludes zero; 4/4 wins |
| Compatibility predicts adaptation | Not supported | Improvement -0.000647; CI crosses zero |
| Market/profile data predicts estimated-value downside | Strongly supported | AUC approximately 0.85 across downside thresholds |
| Pre-transfer data predicts under-utilization | Supported for screening | All primary endpoints beat chronological constants in 4/4 origins |
| Compact destination context adds opportunity information | Supported selectively | Continuous opportunity and under-25 improved; under-10 calibration rejected |
| Reported fees can be benchmarked beyond pre-transfer value | Supported, imprecise | MAE-log improvement 0.063933; average error factor about 2.02× |
| A combined capital-risk score is incrementally validated | Not supported over the strongest component | Joint AUC rises, but primary improvement CI crosses zero |

## Data and analytical design

### Data foundation

The project integrates Transfermarkt-style transfer, valuation, appearance, lineup, event, club-game, and identity data with a cleaned Big Five player-season performance panel covering 2017–2018 through 2023–2024. The comprehensive transfer master contains 35,139 transfer events and 577 fields. Canonical integration layers reconcile player, club, transfer-event, valuation, and performance identities.

Database size was not treated as model sample size. Every model applied target-specific requirements for pre-transfer evidence, transfer semantics, post-transfer horizon observability, and chronological evaluation. Consequently, final eligible cohorts ranged from 827 joint-risk cases to 4,371 market-value cases.

### Leakage controls

- Predictors were restricted to information available at or before the transfer date.
- Post-transfer minutes, appearances, performance, and valuations were targets or diagnostics, never predictors.
- Models used chronological rather than random evaluation.
- Feature preprocessing and hyperparameter selection were fitted inside each temporal origin.
- Evaluation seasons remained untouched until final scoring.
- Player-clustered bootstrap intervals accounted for repeated transfers by the same player.
- Model and data outputs were checked by separate independent-verification scripts.

### Rolling-origin evaluation

The primary commercial models used four evaluation origins covering the 2020, 2021, 2022, and 2023 seasons. Depending on the experiment, an earlier season selected model family and hyperparameters, a later validation season selected feature blocks or calibration policy, and the next season served as untouched evaluation data.

This design answers a prospective question: how well would the procedure have performed if it had been used on the next season’s transfers using only earlier information?

### Why a random split was not primary

Football markets, leagues, valuations, club behavior, and transfer conditions change over time. Random splitting would allow a model to learn from later market conditions while predicting earlier transfers and would overstate deployment realism. Chronological origins impose a more demanding and commercially relevant test.

## Metric interpretation for the paper

| Metric | Meaning | Direction |
|---|---|---|
| MAE | Average absolute prediction error | Lower is better |
| MAE-log | Average absolute error after logging monetary values | Lower is better; exponentiation gives a multiplicative error factor |
| Brier score | Mean squared error of a predicted probability against a binary outcome | Lower is better |
| AUC | Probability that an event case is ranked above a non-event case | 0.5 is random; 1.0 is perfect |
| Average precision | Concentration of positive events near the top of a ranking | Compare with event prevalence |
| ECE | Difference between predicted and observed frequencies across probability bands | Lower is better |
| R² | Share of continuous variation explained relative to a mean benchmark | Higher is better |
| 95% clustered CI | Uncertainty interval accounting for repeated players | An improvement is strongly supported when the interval remains above zero |
| Origin wins | Number of evaluation seasons in which the candidate improved | More wins indicate temporal repeatability |

Throughout the project’s result tables, a positive “improvement” is defined so that positive values favor the candidate model even when the underlying loss metric is lower-is-better.

## Evidence that rejected the original universal compatibility scope

### Rich compatibility evaluation

| Target | Pooled evaluation n | Baseline MAE | Full/policy MAE | Improvement | Player-clustered 95% CI | Origin wins | Decision |
|---|---:|---:|---:|---:|---|---:|---|
| Opportunity | 207 | 0.215776 | 0.213562 | +0.002214 | [-0.005791, +0.010518] | 3/4 | Small and uncertain |
| Performance | 172 | 0.186427 | 0.173867 | +0.012560 | [+0.000291, +0.024308] | 4/4 | Narrow stable-positive finding |
| Adaptation | 162 | 0.185686 | 0.186333 | -0.000647 | [-0.011052, +0.009792] | 2/4 | No benefit |

The single 2023 test slices contained only 37–58 rows depending on target and were explicitly rejected as headline evidence. The rolling-origin results are more credible, but they still do not validate one general compatibility score across outcomes.

### Large-sample opportunity decomposition

The broader coarse cohort contained 3,134 transfers, with 1,485 pooled evaluation transfers.

| Added information block | MAE improvement | 95% player-clustered CI | Interpretation |
|---|---:|---|---|
| Market/profile over mean | +0.018986 | [+0.013929, +0.024232] | Largest basic signal |
| Prior opportunity | +0.003183 | [+0.001200, +0.005091] | Small persistence signal |
| Player performance | +0.000624 | [-0.001078, +0.002242] | Unsupported increment |
| Destination context | +0.006465 | [+0.002602, +0.010431] | Supported contextual adjustment |
| Origin/transition context | +0.006793 | [+0.004801, +0.008779] | Supported contextual adjustment |

Separate coarse player-by-context fit terms improved MAE by only 0.001197, CI [0.000190, 0.002272]. This supported the conclusion that future opportunity reflects ordinary player/market information plus club context, not a dominant hidden compatibility construct.

## Feasibility gate for the commercial pivot

| Proposed module | Eligible rows | Players | Key event rate | Gate decision |
|---|---:|---:|---:|---|
| 24-month estimated-value downside | 4,371 | 2,076 | 35.5% decline at least 25% | Advance |
| Paid-transfer under-utilization | 1,074 | 811 | 33.1% below 10% opportunity | Advance |
| Reported-fee benchmarking | 1,486 | 1,025 | Continuous positive fee | Advance |
| Severe dual capital-at-risk | 827 | 640 | 13.3% joint event | Advance with caveat |
| Fee-only resale return | 337 | 258 | Selected disclosed-sale sample | Exploratory only |
| Full accounting ROI | 0 | 0 | Required cost fields absent | Not feasible |

The audit is evidence of target feasibility, not predictive performance. Each advanced module was subsequently required to beat a simple chronological baseline out of season.

## Final model results

### 1. Retention diagnostic

| Forecast | Comparison | Brier improvement | 95% CI | AUC | Finding |
|---|---|---:|---|---:|---|
| Sporting retention at transfer | Pre-transfer blocks vs prevalence | +0.019281 | [+0.009729, +0.028616] | 0.586 | Supported but modest |
| Uninterrupted two-year spell at transfer | Pre-transfer blocks vs prevalence | +0.008980 | [+0.003343, +0.014711] | 0.577 | Supported but modest |
| Year-two involvement at day 365 | Add first-year involvement | +0.030729 | [+0.017148, +0.044994] | 0.713 | Material monitoring update |

Interpretation: retention works better as a staged model than as a confident transfer-time duration forecast. The destination-spell target is not contract retention, and exact tenure remains unresolved.

### 2. Market-value downside

Strict cohort: 4,371 transfers; pooled evaluation: 2,686 transfers.

| Endpoint | Baseline loss | Compact M1 loss | Improvement | 95% CI | AUC / R² | Origin wins |
|---|---:|---:|---:|---|---:|---:|
| Log value ratio at 24 months | MAE 0.743482 | MAE 0.548534 | +0.194948 | [+0.172762, +0.217230] | R² ≈ 0.443 | 4/4 |
| Decline at least 10% | Brier 0.255960 | Brier 0.161530 | +0.094430 | [+0.085548, +0.103493] | AUC 0.854 | 4/4 |
| Decline at least 25% | Brier 0.247443 | Brier 0.158439 | +0.089004 | [+0.079560, +0.098234] | AUC 0.851 | 4/4 |
| Decline at least 50% | Brier 0.187703 | Brier 0.129853 | +0.057850 | [+0.049266, +0.066479] | AUC 0.848 | 4/4 |

For the primary 25% endpoint, observed downside rose from 6.7% in the lowest-risk quintile to 85.3% in the highest. The 152-field full model did not reliably improve over the 11-field market/profile block; its incremental improvement for the 25% target was only 0.000138 with CI [-0.002470, 0.002754].

### 3. Market-value probability calibration

| Endpoint | Brier improvement over raw | 95% CI | Origin wins | Raw ECE | Calibrated ECE | AUC change |
|---|---:|---|---:|---:|---:|---:|
| 10% downside | +0.004812 | [+0.002717, +0.006908] | 3/4 | 0.084993 | 0.057720 | -0.000287 |
| 25% downside | +0.006130 | [+0.003854, +0.008335] | 3/4 | 0.081657 | 0.053656 | +0.000589 |
| 50% downside | +0.004184 | [+0.002535, +0.005863] | 3/4 | 0.069910 | 0.049792 | -0.000472 |

Threshold-order violations fell from 157 to zero. Calibration improved probability reliability while leaving ranking ability almost unchanged. The calibrated 25%-downside bands ranged from 7.1% observed in the lowest quintile to 85.7% in the highest.

### 4. Paid-transfer under-utilization

Cohort: 1,074 paid permanent arrivals; pooled evaluation: 584 transfers.

| Endpoint | Baseline loss | Selected-policy loss | Improvement | Relative improvement | 95% CI | Wins |
|---|---:|---:|---:|---:|---|---:|
| Opportunity share | MAE 0.210239 | MAE 0.189229 | +0.021010 | 10.0% | [+0.010690, +0.031100] | 4/4 |
| Under 10% opportunity | Brier 0.235575 | Brier 0.200331 | +0.035244 | 15.0% | [+0.021678, +0.049315] | 4/4 |
| Under 25% opportunity | Brier 0.257315 | Brier 0.229625 | +0.027690 | 10.8% | [+0.012207, +0.042998] | 4/4 |

Observed under-10 risk rose from 18.6% to 70.1% between the lowest and highest predicted-risk quintiles; under-25 risk rose from 41.5% to 83.8%. These outcomes measure involvement, not performance quality or tactical fit.

### 5. Under-utilization refinement

| Decision point | Existing result | Refined result | Change | 95% CI | Decision |
|---|---:|---:|---:|---|---|
| Continuous opportunity MAE | 0.189229 | 0.181804 | +0.007426 improvement | [+0.002321, +0.012728] | Replace with compact context |
| Under-10 Brier | 0.200331 | 0.205437 | -0.005106 improvement | [-0.016317, +0.005915] | Reject calibrated replacement |
| Under-25 Brier | 0.229625 | 0.217244 | +0.012381 improvement | [-0.001768, +0.026285] | Use calibrated historical bands; retain old ranking reference |

For under-25, AUC improved by 0.034728 and ECE improved from 0.116576 to 0.107922. For under-10, ECE worsened from 0.113429 to 0.124662. The asymmetry is part of the final model design, not an inconsistency to hide.

### 6. Reported-fee benchmark

| Measure | Result |
|---|---:|
| Pooled evaluation deals | 738 |
| Compact model MAE-log | 0.704923 |
| Approximate multiplicative error factor | 2.02× |
| Improvement over pre-transfer value | +0.063933 |
| 95% CI | [+0.026541, +0.099825] |
| Origin wins | 3/4 |
| Improvement over chronological median | +0.595439 |
| 80% empirical range coverage | 79.3% |
| 90% empirical range coverage | 90.2% |
| Median 80% upper/lower ratio | 8.5× |
| Median 90% upper/lower ratio | 19.7× |

The model validates a historical comparables screen, not a precise fair-price engine. The richer policy’s increment over the compact market/profile model was 0.006504 with CI [-0.004426, 0.017436], so the 10-field model remained the recommended scope.

### 7. Capital-at-risk integration

Fully out-of-sample overlap: 446 transfers, 397 players, and 78 severe joint events (17.5%).

| Score | Severe-event AUC | Average precision |
|---|---:|---:|
| Value component | 0.557 | 0.192 |
| Severe under-use component | 0.530 | 0.209 |
| Geometric joint priority | 0.604 | 0.265 |

The joint score improved AUC over value by 0.047, but the player-clustered CI [-0.033, 0.124] crossed zero. It improved over under-use by 0.075 with CI [0.001, 0.146]. The high/high median quadrant had a 28.8% severe-event rate compared with 17.5% overall, but joint quintiles were not strictly monotonic. The final status was `useful_matrix_but_incremental_gain_uncertain`, and probability claims were prohibited.

## Cross-model findings

### Finding 1: the strongest signal was financial-market risk, not tactical compatibility

Market-value downside AUCs near 0.85 substantially exceeded the discrimination of transfer-time retention or the final joint-risk integration. Age, pre-transfer valuation, position, competition, timing, and fee/value evidence captured more repeatable signal than the 141 additional sporting and context fields.

### Finding 2: playing opportunity was predictable in broad bands

The under-utilization models consistently beat chronological baselines and sharply separated low- and high-risk quintiles. Their continuous MAE remained approximately 18 percentage points, so they should support broad planning ranges and risk tiers rather than precise minute forecasts.

### Finding 3: club context mattered, but “compatibility” was too strong a label

Destination and transition context added repeatable opportunity information in the large cohort and compact refinement. Detailed player-performance and fixed fit interactions did not provide a comparably stable general increment. Context should therefore be described as predictive adjustment rather than causal fit.

### Finding 4: post-transfer monitoring can be more informative than transfer-time prediction

Retention AUC improved from 0.586 at transfer to 0.713 when first-year involvement became available. MoveMaker can create value not only before signing but also as a portfolio-monitoring system.

### Finding 5: integration improved usability more than raw predictive power

The final matrix made multiple risks interpretable on one decision card, but it did not establish an incremental calibrated probability beyond the strongest component. Its value is triage and explanation.

## Business proposition

### Product statement

MoveMaker is a pre-transfer due-diligence and post-transfer monitoring system that helps recruitment and finance teams identify acquisitions requiring deeper review. It does this through separate evidence modules rather than one opaque success score.

### Proposed workflow

1. Enter a candidate player, destination club, proposed reported fee, and transfer date.
2. Review calibrated estimated-value downside bands.
3. Review expected opportunity and under-utilization tiers.
4. Compare the proposed fee with historical comparable deals and broad uncertainty.
5. Place the deal in a value-risk versus opportunity-risk matrix.
6. Escalate high-capital or high/high-risk deals for scouting, medical, tactical, contract, and negotiation review.
7. Update retention and involvement risk after the player’s first season.

### User value

- Prioritizes limited scouting and diligence resources.
- Makes separate sporting and financial concerns visible.
- Provides historical reference points for fee negotiation.
- Supports portfolio monitoring after acquisition.
- Preserves human judgment by showing component drivers and uncertainty.

### What the product does not replace

- Human scouting and tactical evaluation.
- Medical and injury review.
- Contract, wage, and agent-cost analysis.
- Legal or accounting due diligence.
- Negotiation judgment.

## Approved and prohibited claims

| Approved wording | Prohibited wording |
|---|---|
| Estimated market-value downside risk | Expected cash loss or resale proceeds |
| Playing-opportunity estimate | Guaranteed minutes |
| Under-utilization risk tier | Proof of tactical incompatibility |
| Historical reported-fee benchmark | Objectively correct or fair fee |
| Retention prior and one-year update | Exact contract duration |
| Capital-risk prioritization matrix | Calibrated joint failure probability |
| Diligence prioritization index | ROI, profit, or accounting return |

## Limitations section for the paper

1. **Coverage:** detailed performance data covers the Big Five beginning in 2017 and does not comprehensively represent feeder leagues.
2. **Selection:** strict rich-feature cohorts are much smaller than the underlying database, limiting universal compatibility claims.
3. **Unobserved causes:** injuries, suspensions, availability, tactical instructions, and managerial changes are absent.
4. **Financial incompleteness:** wages, bonuses, agent fees, add-ons, amortization, contract book value, and complete resale proceeds are unavailable.
5. **Measurement:** public market values are estimates, and reported fees may be incomplete or undisclosed.
6. **External validity:** evaluation spans four historical origins; future market or league drift requires monitoring.
7. **Joint-event size:** only 78 severe joint events appear in the fully out-of-sample component overlap.
8. **Deployment:** historical validation does not itself constitute a frozen live scorer with ongoing drift checks.
9. **Causality:** predictive context effects do not prove that club context caused the outcome.

## Suggested five-page report structure

### Page 1 — Problem, original hypothesis, and scope pivot

- Introduce transfer uncertainty and the original compatibility/ROI ambition.
- Explain why database volume did not translate into an adequate rich evaluation cohort.
- State the revised research question and modular product thesis.

### Page 2 — Data and methodology

- Describe the transfer, valuation, performance, and club-context sources.
- Explain pre-transfer timing rules, target-specific cohorts, rolling origins, baselines, and player-clustered uncertainty.
- Define MAE, Brier, AUC, calibration, and confidence intervals briefly.

### Page 3 — Compatibility rejection and strongest component findings

- Present the corrected compatibility table.
- Explain the large-sample context decomposition.
- Present market-value downside and under-utilization results.

### Page 4 — Commercial modules and integration

- Cover retention update, fee benchmark, calibration/refinement decisions, and capital-risk matrix.
- Explain the final product workflow and business value.

### Page 5 — Limitations, implications, and conclusion

- State why the product is not ROI, fair value, or tactical compatibility.
- Emphasize evidence-led narrowing as a research contribution.
- Conclude with historical decision cards, presentation synthesis, and future data requirements.

## Polished report-ready paragraphs

### Introduction paragraph

Professional football transfers combine sporting uncertainty with substantial financial exposure. Clubs must judge not only whether a player is talented, but whether that player will receive meaningful opportunity, retain market value, remain involved, and justify the acquisition fee. MoveMaker initially sought to predict overall transfer success through player-club compatibility features. However, rigorous chronological evaluation showed that the available public data could not support one comprehensive compatibility or return-on-investment score. The project therefore adopted an evidence-led scope pivot: rather than force several outcomes into a single label, it tested a set of narrower transfer-risk modules and retained only the capabilities that improved on simple historical baselines out of season.

### Data and methodology paragraph

The analysis combined cleaned transfer, valuation, appearance, lineup, event, club-game, and Big Five player-season performance data in a canonical transfer-event framework. Every predictor was required to exist at or before the transfer date, while post-transfer minutes, performance, and valuations were reserved for targets and diagnostics. Models were evaluated through four rolling chronological origins covering the 2020–2023 seasons. Model family and hyperparameters were selected using earlier periods, and the following season remained untouched for evaluation. Performance differences were assessed with 5,000 paired player-cluster bootstrap repetitions so that repeated transfers by the same player did not create overstated confidence.

### Scope-pivot paragraph

The initial compatibility analysis demonstrated why a large database does not guarantee a large modeling sample. Once fresh pre-transfer performance, club context, outcome observability, and chronological testing were required simultaneously, the single final test contained only 37–58 transfers per target. Four-origin pooling increased the evidence to 162–207 observations, but the corrected results remained outcome-specific. Compatibility features improved destination-performance MAE by 0.0126 with a confidence interval above zero, while opportunity improved by only 0.0022 with an interval crossing zero and adaptation did not improve. A broader 3,134-transfer diagnostic found small but repeatable destination-context effects, yet explicitly compatibility-shaped fit terms added only 0.0012 MAE improvement. These findings rejected a universal compatibility product while preserving a narrower conclusion that context can make modest predictive adjustments.

### Strongest-results paragraph

The strongest commercial signal was estimated market-value downside. On 2,686 out-of-season evaluations, the compact 11-field market/profile model achieved AUC 0.851 for a decline of at least 25% and reduced Brier score from 0.2474 to 0.1584. The observed decline rate ranged from 6.7% in the lowest-risk quintile to 85.3% in the highest. Paid-transfer under-utilization was also modelable: continuous opportunity-share MAE improved from 0.2102 to 0.1892, while severe under-utilization Brier score improved from 0.2356 to 0.2003. The highest severe-risk quintile experienced under-utilization in 70.1% of cases compared with 18.6% in the lowest. These modules provided meaningful screening separation even though they did not explain the causes of a player’s outcome.

### Calibration and disciplined rejection paragraph

Several experiments showed the importance of distinguishing ranking from calibrated probability. Validation-only calibration improved the 25%-downside Brier score by 0.0061 and reduced expected calibration error from 0.0817 to 0.0537 while eliminating all threshold-order violations. In contrast, the proposed calibrated replacement for severe under-10% opportunity worsened Brier score and calibration error, so it was rejected even though the underlying ranking remained better than chronological prevalence. The project retained that model only for relative risk tiers. This asymmetric decision reduced apparent product breadth but strengthened the credibility of the final claims.

### Business paragraph

The resulting MoveMaker product is a modular transfer due-diligence screen. It presents estimated market-value downside, expected playing opportunity, under-utilization risk, historical reported-fee context, and a two-axis matrix showing whether a deal is exposed to value risk, opportunity risk, both, or neither. A fee benchmark can highlight unusual premiums or discounts relative to comparable historical deals, but its average multiplicative error of approximately 2.02 and very broad empirical ranges prevent precise fair-price claims. The final capital-risk integration concentrated severe outcomes in the high/high quadrant, where 28.8% of deals experienced both substantial value decline and severe under-use compared with 17.5% overall. Because the combined score did not reliably outperform the value component, it is presented as a diligence-priority matrix rather than a new failure probability.

### Limitations paragraph

The final scope remains constrained by the public data. Detailed performance coverage is concentrated in the Big Five from 2017 onward, while injuries, suspensions, managerial decisions, wages, bonuses, agent fees, add-ons, amortization, and contract book value are missing. Public market values are estimates rather than realized sale proceeds, and reported fees may omit material terms. These limitations prevent causal compatibility, precise fair-value, expected-loss, and accounting-ROI claims. They also mean that historical validation must be followed by a frozen live scoring procedure and drift monitoring before operational deployment.

### Conclusion paragraph

MoveMaker’s principal research contribution is the scope decision produced by the evidence. The project did not validate its original promise of a comprehensive player-club compatibility and ROI engine. It instead established that public transfer data can support a strong estimated-value downside model, useful opportunity and under-utilization screening, modest retention priors with stronger post-season updates, and a coarse reported-fee benchmark. Integrating these components creates an explainable workflow for prioritizing human diligence without pretending to eliminate transfer uncertainty. The project therefore demonstrates that a rejected hypothesis can lead to a more credible and commercially relevant analytical product when model claims are narrowed to match the available evidence.

## Suggested figures and tables

For a five-page paper, use no more than three primary visuals:

1. **Scope-pivot funnel:** universal compatibility/ROI hypothesis → compatibility evidence failure → feasibility audit → four-module due-diligence product.
2. **Primary results table:** market downside, under-utilization, retention update, fee benchmark, and integration metrics.
3. **Capital-risk 2×2 matrix:** value risk versus opportunity risk, emphasizing the 28.8% high/high severe-event rate and the probability guardrail.

Optional appendix visuals:

- 25%-downside risk quintiles.
- Severe under-utilization risk quintiles.
- Raw versus calibrated probability reliability.
- Correct warning, false alarm, missed event, and correct low-risk historical cases.

## Evidence and citation ledger

Use these local artifacts for project-method and result citations:

- Compatibility reassessment: `Data/processed/compatibility_model_ablations/model_family_reassessment.md`
- Coarse context: `Data/processed/coarse_club_context_diagnostic/README.md`
- Opportunity decomposition: `Data/processed/opportunity_transferability_decomposition/README.md`
- Retention: `Data/processed/retention_diagnostic/README.md`
- Financial feasibility: `Data/processed/financial_module_feasibility/module_feasibility_summary.csv`
- Market downside: `Data/processed/market_value_downside_model/market_value_decision_summary.csv`
- Market calibration: `Data/processed/market_value_calibration_experiment/calibration_decision_summary.csv`
- Market product layer: `Data/processed/market_value_downside_product/README.md`
- Under-utilization: `Data/processed/paid_transfer_underutilization_model/underutilization_decision_summary.csv`
- Under-utilization refinement: `Data/processed/paid_transfer_underutilization_refinement/refinement_decision_summary.csv`
- Fee benchmark: `Data/processed/fee_benchmarking_model/fee_benchmark_decision_summary.csv`
- Capital-risk integration: `Data/processed/capital_at_risk_integration/integration_decision_summary.csv`
- Verification status: each corresponding `independent_verification.json`
- Full project history and claim guardrails: `CODEX.md`

## Verification-status summary

| Stage | Independent checks |
|---|---:|
| Compatibility models | 31/31 |
| Compatibility ablations | 24/24 |
| Rolling-origin stability | 25/25 |
| Coarse context | 19/19 |
| Opportunity decomposition | 19/19 |
| Retention | 32/32 |
| Financial feasibility | 64/64 |
| Market-value downside | 32/32 |
| Market calibration | 32/32 |
| Downside product layer | 30/30 |
| Paid-transfer under-utilization | 38/38 |
| Under-utilization refinement | 38/38 |
| Fee benchmark | 41/41 |
| Capital-at-risk integration | 37/37 |

## Final writing guardrail

The paper’s strongest story is not “we built many models.” It is:

> We tested an ambitious compatibility hypothesis, found that the available evidence did not support the promised universal score, identified which narrower outcomes were genuinely predictable, and converted those validated components into an explainable business product with explicit limitations.

Maintain that sequence. It turns the scope change into the research finding rather than treating it as an embarrassment to hide.

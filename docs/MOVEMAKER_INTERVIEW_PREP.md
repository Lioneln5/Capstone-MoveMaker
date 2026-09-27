# MoveMaker interview preparation guide

This is an interview-first explanation of the project. It separates what was
built for the capstone presentation from what the later audit established, so
you can describe the work confidently without making claims the evidence no
longer supports.

## If you only have ten minutes, remember these five points

1. **The problem changed because the evidence changed.** MoveMaker began as a
   player-club compatibility and transfer-ROI idea, but the available data did
   not support one universal success score. The final capstone narrowed the
   product to explainable decision support for incumbent-club contract
   extensions.
2. **The data engineering was as important as the modeling.** The project
   connected public identity, transfer, match, performance, valuation, salary,
   contract, payroll, and injury sources into point-in-time player and club
   histories.
3. **The models were intentionally modular.** Regularized regression handled
   continuous outcomes, logistic regression handled binary risks, and a
   discrete-time logistic hazard model handled movement over time. Gradient
   boosting was tested, but only selected where its nonlinear improvement was
   justified.
4. **Validation was chronological.** Four rolling origins evaluated extension
   decisions from 2020 through 2023. Hyperparameters were selected on the
   immediately preceding year, preprocessing was fitted inside each training
   fold, and uncertainty was estimated with player-clustered bootstraps.
5. **A later audit froze the original scoring engine.** The public site remains
   online as a research record, but live V1 scoring is disabled while targets
   and final validation are repaired. Finding and containing those problems is
   part of the project, not something to hide.

## Thirty-second answer

> MoveMaker is an explainable football analytics project for contract-extension
> due diligence. I led the analytical and technical work on a four-person
> capstone team. We integrated millions of public records across player
> performance, transfers, valuations, salaries, contracts, clubs, and injuries,
> then used chronological backtesting to ask what could actually be predicted
> at the moment of an extension decision. The most important result was that a
> single compatibility or recommendation score was not defensible, so we built
> separate modules for future involvement, movement, public-value downside, and
> wage benchmarking. After the presentation I stress-tested the engine, found
> target and subgroup weaknesses, froze live scoring, and began a stricter V2
> validation program.

## Two-minute project walkthrough

> The original concept was much broader. We wanted to profile how players
> performed with different teammates and clubs, combine that with financial
> information, and estimate the sporting and financial ROI of a transfer. Once
> we required all of the necessary features, however, the clean compatibility
> cohort collapsed to only 63 cases. Larger-cohort tests also failed to produce
> a stable universal signal. Rather than force the original idea, we treated
> that failure as evidence and narrowed the question.
>
> We first moved to modular transfer-risk screening, and then Capology salary
> and extension data allowed us to focus on one more concrete decision: what
> should a club investigate before extending one of its current players, and
> how aggressive are the proposed wage and term relative to historical peers?
>
> I built and coordinated a pipeline that linked Transfermarkt-derived match,
> transfer, and valuation histories; FBref performance; Capology salary and
> extension records; StatsBomb event and lineup data for the earlier
> compatibility work; and bounded injury data. Each extension became one
> decision-time anchor. Predictors only used information available by the
> signing date, while future contribution, movement, and valuation outcomes
> were constructed in explicit post-signing windows.
>
> We compared regularized linear and logistic models with gradient boosting.
> Simpler models usually held up, were easier to explain, and were more stable
> across years. The final prototype therefore used Ridge for continuous
> outcomes, logistic regression for binary outcomes, a discrete-time logistic
> hazard model for movement, and gradient boosting only for contract duration.
> We rejected an overall recommendation score because combining the modules did
> not outperform the strongest individual signal.
>
> The capstone result showed useful historical screening power, especially for
> public-value downside and low future contribution. But a post-presentation
> audit found that the original contribution target sometimes mixed true role,
> injury, loans, and incomplete club schedules, while the generic continuity
> target mixed temporary loans with permanent departures. I froze V1, retired
> generic continuity, separated temporary displacement from permanent
> separation, and blocked any new deployment until the target and final
> temporal validation are repaired. That process is probably the best example
> of what I learned: a model is only useful if the football meaning of its
> target and the reliability of its probabilities survive scrutiny.

## What the product is—and is not

MoveMaker is a **screening and prioritization tool**. Its intended user is a
club analyst, recruitment team, sporting department, or finance team deciding
which extension cases deserve deeper investigation.

It is not:

- an automated “extend” or “do not extend” decision;
- a tactical compatibility model;
- a single overall player-quality or success score;
- an estimate of accounting loss, transfer ROI, or guaranteed savings; or
- a substitute for medical, scouting, contract, and internal club information.

That boundary matters. The project can rank historical cases and provide
reference ranges, but it cannot observe the full set of players a club
considered and chose not to extend. It therefore cannot identify the causal
effect of offering an extension or learn an unbiased extend-versus-do-not-
extend policy.

## The analytical journey

### 1. Original hypothesis: player and club compatibility

The original plan was to build player-player and player-club profiles from
co-performance, role, and statistical relationships. Those profiles would be
combined with financial information to estimate the expected sporting and
financial return of a transfer.

Why it failed:

- the richest complete-case cohort contained only 63 transfers;
- the targets represented different concepts—opportunity, performance, and
  adaptation—and did not behave like one universal definition of success;
- added compatibility information was not stable across all outcomes and
  chronological test periods; and
- a broad ROI claim required fee, wage, resale, and accounting information
  that public data did not observe consistently.

### 2. First pivot: modular transfer-risk screening

The project separated the original question into opportunity, retention,
public-value downside, reported-fee, and capital-exposure modules. This found
real signal, particularly in compact market/profile variables, but the modules
still could not be compressed into one defensible transfer-success score.

### 3. Final capstone scope: incumbent-player extension decisions

Capology salary and extension histories created a cleaner decision anchor: the
date a current club extended a player. The prototype asked four separate
questions:

- How much meaningful involvement did the player sustain after signing?
- Did the player remain with the club, temporarily leave, or permanently
  separate?
- Did the player's public market value decline materially?
- How did the proposed wage and contract term compare with historical peers?

### 4. Post-presentation V2: audit before expansion

The first prototype was presentation-ready, but a full adversarial audit found
problems that made it inappropriate to continue calling the engine production-
ready. V1 is now frozen and its HTTP scoring routes fail closed. V2 is a
research program with no authorized live probability yet.

Current outcome status:

| Outcome | Current status | Why |
| --- | --- | --- |
| Meaningful Retention | Blocked | Its contribution component must be rebuilt before the model can be revalidated. |
| Temporary Displacement Risk | Development candidate | Promising rolling-origin evidence, but no untouched final temporal test yet. |
| Permanent Separation | Research-only | Pooled performance hides negative reliability in the Bundesliga and Ligue 1. |
| Generic continuity | Retired | It combined loans and permanent exits into one ambiguous event. |
| Public-value downside | Historical evidence only | Strong V1 ranking evidence; any V2 release still needs the complete release process. |
| Wage benchmark | Blocked from release | Its final Ligue 1 interval coverage missed the predeclared gate. |

## Data used

### Main source families

| Source family | Information contributed | How it was used |
| --- | --- | --- |
| Transfermarkt-derived data | Player and club IDs, dates of birth, positions, appearances, minutes, games, transfers, formations/managers, and public market values | Identity spine, exact playing-time windows, club-game denominators, departure outcomes, valuation landmarks, and later context research |
| Supplemental Transfermarkt datalake | Additional transfer, performance, profile, teammate, injury, and club-season histories | Canonical unions, identity resolution, earlier compatibility work, injury and context diagnostics |
| FBref | Basic and advanced seasonal performance, including xG, passing, progression, defensive, aerial, and creation metrics | Earlier compatibility research, historical feature ladders, and tests of whether advanced performance added incremental signal |
| Capology | Player salaries, extensions, contract duration, club payroll, and transfer-window financial context | Extension decision anchors, wage features, historical peer benchmarks, club wage structure, and lagged club-finance experiments |
| StatsBomb open data | Event and lineup data | Already collected and processed for the earlier compatibility and case-study infrastructure; not selected for the current extension models because overlap with the extension cohort was sparse and no stable incremental signal was validated |
| Injury data | Public injury histories from two differently scoped sources | Coverage and target-validity diagnostics; not yet a sufficiently complete, decision-time-safe production feature |

The local data audit counted more than seven million rows in the primary
Transfermarkt match/event collection alone. The canonical layers include
approximately 190,000 player identities, 46,000 club identities, 1.4 million
dated valuations, 1.1 million transfer events, and 2.1 million player-team-
competition-season performance rows. These totals describe data engineering
scale; they should not be interpreted as millions of independent training
examples.

### The unit of analysis

For the extension product, one row represents **one incumbent-club extension
event**. The extension signing date is the anchor.

- Pre-signing fields are candidate predictors.
- Exact 365-day and completed-season windows summarize prior involvement.
- Post-signing Year 1 and Year 2 windows construct future contribution labels.
- The nearest eligible 12- and 24-month valuations construct public-value
  outcomes.
- Later transfer events construct outbound-movement outcomes.
- Salary and contract records construct annual wage, fixed commitment, wage
  change, and club-relative wage context.

The integrated historical extension table contains 2,805 deduplicated events;
each endpoint uses the subset whose predictor and outcome windows are actually
observable.

## How the merge pipeline worked

### Short interview answer

> I treated every extension as a dated anchor rather than joining everything by
> player name and season. I resolved stable player and club IDs first, then
> joined each source through explicit one-to-one or many-to-one contracts. For
> every extension, I calculated pre-signing features and post-signing outcomes
> in non-overlapping windows. Every build wrote coverage checks, source hashes,
> a field dictionary, and a cohort funnel, so a join could not silently multiply
> records or leak future information.

### Technical version

1. Clean each source independently and preserve source IDs and provenance.
2. Resolve canonical player and club identities using stable IDs where
   available; otherwise use normalized names, reviewed aliases, and explicit
   unresolved states.
3. Deduplicate Capology extension records into a unique
   `capology_extension_event_id`.
4. Join salary rows on normalized player, canonical club, and source season;
   derive prior wage, club wage percentile, and known payroll share.
5. Join canonical seasonal performance on player, club, competition, and
   season with `many_to_one` validation.
6. Use match-level appearances and club games to build exact pre-365, Year-1,
   and Year-2 windows. The denominator is captured club games multiplied by 90
   minutes—not the player's available minutes.
7. Find the last eligible valuation at signing and nearest eligible future
   valuations inside declared timing tolerances.
8. Match later outbound transfer events from the extension club to create
   censored movement outcomes.
9. Merge every block back to the unique extension ID with `one_to_one`
   validation.
10. Assert that row counts, unique IDs, temporal ordering, and eligibility
    funnels remain valid; publish hashes and independent verifier outputs.

One important lesson from the later audit is that “the code ran and the join
did not duplicate rows” is necessary but not sufficient. A football-valid
denominator also requires complete club schedules, including relegation and
lower-tier matches. That semantic defect is now a release blocker.

## Models used and why

### Core model families

| Model | Use | Why it was chosen |
| --- | --- | --- |
| Ridge regression | Continuous opportunity share, public-value log ratio, annual wage, fixed-wage commitment, and wage-to-value benchmarks | Handles correlated features and small-to-medium tabular cohorts, shrinks unstable coefficients, and remains interpretable |
| L2 logistic regression | Binary contribution and public-value downside endpoints | Produces probabilities, supports calibration and per-feature log-odds explanations, and was generally as good as or more stable than more complex candidates |
| Discrete-time logistic hazard model | Historical 24/36-month outbound movement | Converts time-to-event data into interval rows, handles censoring and different horizons, and guarantees a coherent survival curve within one fitted endpoint |
| Gradient-boosting regressor | Historical contract-duration benchmark | Captures nonlinear relationships where validation justified the extra complexity; it was not adopted as the default everywhere |
| Chronology-only constant/median | Baseline | Represents what would be known from prior historical prevalence or central tendency without player-specific features |

### What preprocessing did

The custom mixed-type preprocessor was fitted separately inside each training
origin. It:

- median-imputed numeric fields using training data only;
- added missingness indicators when a training feature contained missing data;
- one-hot encoded categorical variables with an explicit `__OTHER__` level;
- standardized encoded columns using training-fold means and scales; and
- removed constant encoded columns.

This order-preserving custom preprocessor also fixed a legacy bug in which
interleaved numeric and categorical fields could become misaligned.

### Why not use the most complex model?

> The dataset is tabular, temporally structured, and not enormous at the
> extension-event level. My goal was not to maximize an in-sample score; it was
> to find stable, calibrated, explainable improvements over a chronological
> baseline. We tested gradient boosting alongside regularized linear models,
> but complexity only earned a place when it improved later-year validation.
> In most endpoints the simpler model held up, was easier to debug, and made
> counterintuitive coefficients visible.

That last point mattered. A black-box model could have hidden the football-
invalid relationship between goals per 90 and continuity. Interpretability
helped reveal it.

## What each historical module did

### 1. Future involvement

The V1 prototype contained:

- a Ridge model for Year-2 same-club opportunity share; and
- an L2 logistic model for “sustained meaningful contribution,” originally
  defined as reaching at least 25% same-club opportunity share in both Year 1
  and Year 2.

The compact contribution model used age, broad position, league, market value,
recent opportunity/appearance rates, contract terms, and wage context. A repair
removed recent goals and assists per 90 because those global rates created
football-invalid behavior and added no validated incremental value.

The later target audit found that the target still mixes realized use with
availability and movement. It also found asymmetric evidence requirements,
incomplete club schedules around relegation, and six opportunity shares above
100%. Therefore it remains historical evidence, not a deployable pure “future
role” probability.

### 2. Movement

The historical model expanded each extension into time intervals and used
logistic regression to estimate the hazard of an outbound event. Multiplying
the interval survival terms produced a stay probability at 24 or 36 months.

The generic outcome was later retired because “any outbound” treats a loan,
loan return, and permanent transfer as versions of the same failure. V2 now
distinguishes:

- **Meaningful Retention:** no outbound plus meaningful contribution;
- **Temporary Displacement:** a loan or loan return is the first interruption;
- **Permanent Separation:** the permanent relationship ends within 24 months.

Temporary Displacement is the most promising development candidate. Permanent
Separation remains research-only because cross-league reliability is not good
enough.

### 3. Public-value downside

Ridge modeled the future log ratio of public market value, while separate L2
logistic models estimated the probability of declines of at least 10%, 25%,
and 50% at 12 and 24 months. Displayed threshold probabilities were projected
into a logically ordered sequence when independently fitted models produced a
minor contradiction.

The main historical 24-month, 25%-downside endpoint achieved AUC 0.821 in the
extension research. This means it ranked a randomly selected downside case
above a non-downside case about 82% of the time; it does not mean the model was
82% “accurate,” and it does not predict cash loss or sale proceeds.

### 4. Wage and contract peer context

Ridge models estimated log annual wage, fixed-wage commitment, and wage-to-
market-value relative to historical extension peers. Gradient boosting modeled
contract duration. Residual distributions produced empirical reference ranges.

These are peer benchmarks, not optimal contracts or fair-value estimates. The
wage model looked strong in pooled final testing but failed its predeclared
Ligue 1 interval-coverage gate—60.6% versus the required 70%—so it was not
authorized for V2 deployment.

### 5. No overall recommendation score

This is the correct answer if asked how the recommendation score was
structured:

> We ultimately did not use one recommendation score. We tested an integrated
> rank, but the three-module version achieved AUC 0.6830 compared with 0.6885
> for opportunity risk alone, and the uncertainty interval for the difference
> crossed zero. Combining distinct risks made the output harder to interpret
> without improving prediction, so the product kept separate cards and let the
> human decision-maker see which risk drove a flag.

The historical “risk-weighted screening” result was a prioritization exercise,
not an extend/do-not-extend score. For this specific audited proxy, two-year
fixed wages were multiplied by the modeled probability of low contribution.
At a fixed capacity of 182 reviews, ranking by that probability-weighted wage
exposure found 79 historical low-contribution outcomes versus 54 when simply
reviewing the 182 largest contracts—a 46% increase. The result demonstrates
better historical triage at equal capacity; it does not prove savings,
causality, or a correct recommendation in every case.

## How validation worked

### Short interview answer

> I avoided a random train-test split because football markets, coverage, and
> player cohorts change over time. I used four rolling origins: earlier years
> trained the model, the immediately preceding year selected the model and
> hyperparameters, and the next year was evaluated as unseen future data. I
> compared every candidate with a chronology-only baseline, reported both
> ranking and probability quality, bootstrapped by player, and checked league,
> age, and position subgroups. I also wrote separate verification scripts that
> reconstructed cohorts, metrics, hashes, and product behavior.

### Exact rolling design

| Origin | Train through | Select/tune on | Evaluate on |
| --- | ---: | ---: | ---: |
| 1 | 2018 | 2019 | 2020 |
| 2 | 2019 | 2020 | 2021 |
| 3 | 2020 | 2021 | 2022 |
| 4 | 2021 | 2022 | 2023 |

After selection, each origin refit on its training plus validation data before
predicting the evaluation year. Preprocessing was refit within the origin.

### Metrics

For binary outcomes:

- **Brier score:** mean squared error of the probability; lower is better.
- **Log loss:** heavily penalizes confident wrong probabilities.
- **AUC:** ranking discrimination; not accuracy.
- **Average precision:** useful when the event class is less common.
- **Calibration/ECE:** whether predicted probabilities match observed rates.

For continuous outcomes:

- **MAE/RMSE:** typical and squared prediction error;
- **R-squared:** improvement in explained variation; and
- **Spearman correlation:** quality of rank ordering.

The project used 5,000 player-clustered bootstrap repetitions for key paired
comparisons. Clustering by player matters because the same player can appear in
more than one historical event; treating those events as independent would
make uncertainty look too small.

### Validation gates beyond one headline metric

A candidate also had to survive:

- performance across all four origins;
- league, age, and position subgroup checks;
- calibration and probability reliability;
- missing-input and out-of-distribution refusal rules;
- temporal leakage and feature-availability checks;
- logical consistency between related probabilities;
- immutable source and artifact hashes; and
- parity between research predictions and the application scoring path.

## Application and API architecture

```text
Public source files
        |
        v
Source-specific cleaners and canonical identity tables
        |
        v
One-row-per-extension modeling master
        |
        v
Rolling-origin research runners and independent verifiers
        |
        v
Versioned scikit-learn artifacts + JSON manifest + SHA-256 hashes
        |
        v
Canonical feature retriever -> deployment scorer -> profile builder
        |
        v
FastAPI routes -> static HTML/CSS/JavaScript interface -> Railway
```

### How did the API connect to the model?

> In V1, FastAPI accepted a typed Pydantic request containing canonical player,
> club, competition, decision-date, wage, term, and optional market-value
> inputs. A feature retriever assembled the point-in-time player and club
> features. The scorer loaded a versioned JSON manifest and cached joblib
> artifacts, applied the artifact's fitted preprocessor, generated each module's
> prediction, and returned a typed JSON profile with support status,
> missing-input warnings, ranges, and explanations. The HTML front end called
> those routes and rendered separate cards. The service was packaged with
> Uvicorn and deployed on Railway.
>
> After the audit, I changed the HTTP layer to fail closed. The static research
> site and `/health` endpoint remain available, but all search and profile routes
> return a 503 before any V1 model or data is loaded. There is deliberately no
> environment-variable switch that can accidentally re-enable the old engine.

### How were explanations produced?

For supported Ridge and logistic models, the explainer transformed the request
with the exact artifact preprocessor, multiplied each encoded value by its
coefficient, grouped encoded terms back to their raw features, and reconciled
the contributions with the model intercept. For hazard models it repeated that
calculation across time intervals and connected interval hazards to cumulative
survival.

A post-presentation audit caught a shape assumption in the explainer:
logistic regression stores `intercept_` as a length-one array, while Ridge can
store it as a scalar. The shared code now normalizes both representations and
has a dedicated regression verifier.

## Libraries and tools

Core implementation:

- **Python** for all data and modeling work;
- **pandas** and **NumPy** for cleaning, joining, feature construction, and
  metric reconstruction;
- **scikit-learn** for Ridge, logistic regression, gradient boosting,
  preprocessing logic, and evaluation metrics;
- **SciPy** for supporting statistical calculations;
- **joblib** for serialized model artifacts;
- **FastAPI** and **Pydantic** for typed API routes and validation;
- **Uvicorn** as the application server; and
- **HTML, CSS, and JavaScript** for the interface, hosted on **Railway**.

The project also used Git/GitHub for versioning and pull requests, plus
independent Python verifier scripts and SHA-256 manifests for reproducibility.

## Likely interview questions and natural answers

### What was the problem?

> Clubs make contract extensions under uncertainty. They are committing wages
> and contract years without knowing whether a player's role, club
> relationship, or public market value will hold. Our original problem was even
> broader—predicting player-club compatibility and transfer ROI—but the data did
> not support that claim. We narrowed the product to showing separate, evidence-
> backed risks around one observable decision: extending a current player.

### What data did you use?

> We linked public Transfermarkt-derived player, match, transfer, and valuation
> histories with FBref performance and Capology salary, extension, payroll, and
> transfer-window data. We also had StatsBomb event and lineup data for the
> earlier compatibility work and two injury sources for coverage diagnostics.
> The difficult part was not obtaining a large row count; it was resolving
> identities, preserving decision-time validity, and knowing which source was
> complete enough for each target.

### What did you personally build?

Use only the responsibilities you can defend from your own work. A strong
version is:

> It was officially a four-person capstone, and I was the Project Coordinator
> and ML Lead. I drove the scope pivots and analytical decisions, built and
> integrated the core data and modeling pipelines, implemented the
> chronological evaluation and verification framework, translated the models
> into product modules, and led the API and deployment work. My teammates
> contributed to the shared deliverable in their assigned roles, so I describe
> my ownership specifically rather than calling the entire project solo.

If they probe, name one concrete file or system for each claimed responsibility:

- integration: `scripts/build_contract_extension_integration.py`;
- modeling: the phase runners under `models/`;
- verification: paired scripts under `scripts/verify_*.py`;
- application: `models/deployment_scorer.py` and `api/main.py`;
- product/research judgment: the scope-pivot journal and Engine V2 audit docs.

### What was the hardest technical issue?

> Identity resolution and temporal validity were harder than choosing a model.
> Every source used different player names, club aliases, season formats, and
> coverage windows. A merge can be syntactically correct and still be
> analytically wrong if it links the wrong club, multiplies an event, or uses a
> post-signing value as a predictor. I built canonical IDs, reviewed alias
> tables, explicit join cardinality checks, point-in-time windows, coverage
> funnels, and source hashes. The later target audit taught me that even those
> controls have to be paired with football-semantic checks, such as whether a
> relegated club's full schedule is actually captured.

An alternative answer is the target-design problem:

> The hardest conceptual issue was defining success without hiding several
> mechanisms inside one label. Low minutes can mean a tactical demotion, injury,
> loan, transfer, or missing competition coverage. Separating those concepts
> was more difficult—and more important—than tuning the classifier.

### Why did you choose those models?

> I used regularized linear and logistic models as the default because the
> extension cohort was modest, the inputs were structured tabular data, and the
> application needed interpretable probabilities. Ridge stabilized correlated
> financial and performance features; logistic regression gave inspectable
> probability models; and discrete-time logistic hazards handled censored
> movement histories. I tested gradient boosting rather than assuming linearity,
> but only kept it for contract duration, where later-year validation justified
> the complexity.

### What failed?

> Several things failed, and they changed the product. The universal
> compatibility idea did not have a large or stable enough cohort. The overall
> recommendation score did not outperform the strongest individual module.
> Large advanced-stat blocks added essentially no incremental predictive value
> in key extension models. Role-aware scoring interactions did not beat simply
> removing goals and assists from important endpoints. Club payroll, squad
> competition, manager stability, formation stability, and lagged club behavior
> were football-plausible but did not produce stable incremental improvement.
> Finally, the generic continuity target and the future-contribution target
> failed later semantic and subgroup audits, so I retired or blocked them.

### How did you validate the result?

> I used rolling chronological origins rather than a random split, tuned only on
> the prior year, compared with a chronology-only baseline, and reported
> calibration as well as ranking. I used player-clustered bootstraps for paired
> uncertainty and checked every league and major player subgroup. Separate
> verifier programs reconstructed row counts, target rates, metrics, hashes,
> and API/artifact parity. Later I added release gates so a strong pooled score
> cannot conceal a failed league or an excessive refusal rate.

### What would you change now?

> First, I would define and freeze the target contract before building more
> features. I would reconstruct contribution from complete club schedules with
> symmetric Year-1 and Year-2 evidence, cap the denominator correctly, and
> report realized contribution separately from role when available. Second, I
> would preserve Temporary Displacement and Permanent Separation as separate
> questions rather than rebuild generic continuity. Third, I would reserve a
> later, mature, player-disjoint cohort as an untouched final test. Only after
> those steps would I revisit richer injury, manager, squad, or financial
> features.

### Where did AI help?

> I used AI as a paired programmer and adversarial reviewer. It accelerated
> scaffolding, repetitive data checks, debugging, documentation, interface
> iteration, and the generation of candidate diagnostics. It was especially
> useful for asking, “What assumption have we not tested?” and then turning that
> question into a reproducible experiment.

### What still required your judgment?

> AI could write or critique code, but it could not decide what a football
> outcome should mean or whether a statistically useful pattern was acceptable
> in the product. I had to notice that goals per 90 were being interpreted
> globally across positions, distinguish loans from permanent separation,
> decide not to claim ROI or causal recommendations, reject a universal score,
> define the release gates, and freeze the system when later evidence exposed a
> target defect. I treated AI output as a hypothesis, not as evidence; the
> evidence came from the actual data, saved artifacts, and independent checks.

### Did AI introduce any risk?

> Yes. AI can produce code that is locally plausible while carrying forward a
> bad target definition or an overbroad product claim. The position-insensitive
> scoring-rate problem is a good example. My response was to add independent
> verifiers, perturbation tests, feature contracts, immutable hashes, and a
> fail-closed deployment state. The lesson was that faster coding increases the
> need for human problem formulation and adversarial validation.

### What result are you proudest of?

> I am proudest that we did not force the original hypothesis. The evidence
> caused two scope pivots, and we still produced a coherent product. From a
> modeling standpoint, public-value downside was the strongest historical
> signal. From an engineering standpoint, the chronological and independently
> verified pipeline is the stronger accomplishment. From a judgment standpoint,
> freezing a polished prototype after finding defects was more responsible than
> leaving an impressive but overstated demo online.

### Was the project production-ready?

> The capstone prototype was fully wired and deployed, but the post-presentation
> audit showed that “production-ready” was too strong. The software path worked;
> the problem was whether every target and subgroup claim was reliable enough
> for real decisions. I now describe V1 as a frozen research prototype. The
> public site is deliberately fail closed, and V2 will not serve probabilities
> until target repair and untouched temporal validation are complete.

## Code-adjacent follow-ups

### How did you prevent leakage?

> Every feature had an information role and cutoff. The extension signing date
> anchored the row. Pre-signing completed seasons and exact prior-365-day data
> could be predictors; Year-1/Year-2 minutes, future valuations, and later
> transfers could only be outcomes. Model preprocessing and hyperparameter
> selection happened within each rolling origin. I also asserted that attached
> pre-season end dates never exceeded the signing date.

### How did you handle missing data?

> Missing did not automatically mean zero. If the club schedule proved a window
> was observable and the player had no appearance, zero minutes could be a real
> observation. Otherwise the value stayed missing. The model preprocessor used
> training-fold medians, explicit missingness indicators, and unknown-category
> handling. Core missing inputs could force the application to return
> unavailable rather than fabricate a confident score.

### How did you serialize and version models?

> Each joblib artifact stored its model, fitted preprocessor, feature list,
> endpoint, units, hyperparameters, validation metadata, training-row count,
> data hash, model vintage, and library version. A JSON manifest mapped
> endpoints to artifacts and recorded SHA-256 hashes. The scorer loaded through
> that contract rather than reconstructing a feature list from application code.

### How did the hazard model produce a 24-month probability?

> I expanded every eligible extension into fixed time intervals and trained a
> logistic classifier for the probability of an event in each interval,
> conditional on surviving to that interval. At scoring time, I predicted the
> interval hazards and multiplied one minus each hazard through 24 months. That
> product is the estimated survival probability; one minus it is cumulative
> event risk.

### What did “incremental predictive improvement” mean?

> Every feature block was added to a named simpler model or chronology-only
> baseline on the same evaluation rows. Incremental improvement was the paired
> reduction in the relevant loss—for example, baseline Brier score minus
> candidate Brier score. A positive number meant the added block improved
> probability accuracy. It did not mean the feature explained that percentage
> of the entire outcome, and it was not accepted without uncertainty and
> cross-origin checks.

### Why was AUC not enough?

> AUC only measures ranking. A model can rank well and still give unreliable
> probabilities, fail in one league, or depend on inputs that are usually
> missing. That is why I also used Brier score, log loss, calibration error,
> subgroup reliability, origin wins, and refusal rates. The Ligue 1 continuity
> and wage results are examples where a pooled headline was not enough to
> authorize deployment.

## Numbers worth remembering

Do not try to memorize every metric. These are enough for most interviews:

- **2,805** deduplicated extension events in the integrated historical spine.
- **Four rolling origins**, evaluating decisions made in **2020–2023**.
- **908** historical extension profiles in the presentation's financial-impact
  analysis, representing **€5.22B** in scheduled two-year fixed wages.
- At equal capacity for **182 reviews**, historical risk screening found **79**
  low-contribution outcomes versus **54** from largest-contract review—**46%
  more**, not 46 percentage points and not projected savings.
- Historical 24-month, 25% public-value-downside model: **AUC 0.821**.
- The overall score was rejected: **0.6830 AUC** versus **0.6885** for
  opportunity risk alone, with an uncertainty interval crossing zero.
- Temporary Displacement development evidence: **AUC 0.777**, but not finally
  validated or deployed.

## Claims to avoid

Avoid saying:

- “The model tells clubs whether to extend a player.”
- “We predicted ROI” or “saved €1.82B.”
- “AUC 0.821 means 82.1% accuracy.”
- “All 7.2 million rows trained one model.”
- “Advanced stats do not matter in football.”
- “The production model is currently live.”
- “The current engine is position-aware.”
- “I built the entire team project alone.”

Prefer:

- “It prioritizes cases for human review.”
- “It found historical exposure associated with the target.”
- “It ranked downside cases above non-cases.”
- “The sources formed a large data backbone; endpoint cohorts were much
  smaller.”
- “Those advanced-stat blocks added no validated incremental signal for these
  targets, cohorts, and horizons.”
- “The public research site is live; scoring is deliberately disabled.”
- “The scoring-rate defect was contained in one model and remains part of the
  frozen V1 audit elsewhere.”
- “I was the Project Coordinator and ML Lead, and I personally owned these
  specific systems.”

## A strong closing thought

> The main lesson was that model development is not a straight line from more
> data to more certainty. The most valuable work was deciding what the evidence
> allowed us to claim, changing the product when it did not support the original
> idea, and building enough verification that we could detect and contain our
> own mistakes.

# MoveMaker: building a defensible football contract-risk case study

**Role:** Project Coordinator and ML Lead on a four-person capstone team

**Domain:** football analytics, contract extensions, risk screening

**Stack:** Python, pandas, NumPy, scikit-learn, SciPy, joblib, FastAPI, Pydantic, HTML/CSS/JavaScript, Railway

**Status:** frozen analytical case study; the public site is a research record and live predictive scoring is disabled

[Research site](https://movemaker-production.up.railway.app/) · [Interactive historical prototype](https://movemaker-production.up.railway.app/prototype.html) · [Final presentation](slides/final/MoveMaker_Final_Presentation.html) · [Metric dispositions](docs/CASE_STUDY_METRIC_DISPOSITIONS.md) · [Model card](docs/MODEL_CARD.md)

## Executive summary

MoveMaker investigates a narrow but financially important question: **what risks should a football club examine before extending an existing player's contract?** It does not recommend “extend” or “do not extend.” It produces evidence that a sporting director could consider alongside scouting, medical information, tactical plans, squad depth, contract clauses, and internal financial data.

The project began with a much broader ambition: estimate player-player and player-club compatibility, then combine sporting fit with financial information to calculate transfer return on investment. That concept did not survive contact with the data. The feature-rich complete cohort collapsed to 63 transfers, broader experiments did not recover a stable universal compatibility signal, and the separate risks could not defensibly be compressed into one score.

We therefore narrowed the problem twice—first to modular transfer risks and then to incumbent-club contract extensions. The final work joined public transfer, valuation, performance, wage, contract, lineup, event, club, and injury sources; built a time-aware extension dataset; evaluated models through rolling chronological origins; and preserved both successful and failed hypotheses.

The central result was not a perfect prediction system. It was a defensible boundary around what the available evidence could support:

- contract arithmetic and observed club salary context can be shown descriptively;
- realized contribution, temporary displacement, and 12-month public-value downside have qualified research signal;
- generic continuity and a single integrated contract score should be retired;
- permanent separation and several contract benchmarks remain unresolved or historical-only; and
- no current-player probability should be presented as an authorized live score.

## The problem

A contract extension commits future wages while the club still faces several distinct uncertainties:

- Will the player continue to receive meaningful playing opportunity?
- Will the player remain at the club, leave temporarily on loan, or separate permanently?
- Could the player's public market value decline during the contract horizon?
- How aggressive are the proposed wage and term relative to observed historical extensions?
- How much fixed-wage commitment is associated with the decision?

Those questions are related, but they are not interchangeable. A young player sent on loan may still be a valuable long-term asset. An older starter may continue contributing on the pitch while losing resale value. A high wage can be affordable to one club and destabilizing to another. Treating all of those mechanisms as one “good contract” score would hide the reason for the risk and imply a decision the data cannot make.

MoveMaker's final design therefore treats the outputs as **separate decision modules**, not as a recommendation engine.

## How the scope changed

### 1. Player and club compatibility

The original hypothesis was that player profiles could describe how well footballers performed with particular teammates and club environments. Those profiles, combined with transfer and financial information, would estimate the sporting and financial return of a proposed signing.

This failed for two reasons. First, the richest required features left only 63 complete transfer cases. Second, increasing coverage did not produce a stable universal compatibility signal. The honest conclusion was that the available public data did not justify one general measure of “fit” or transfer success.

### 2. Modular transfer-risk screening

The first pivot separated the broad transfer question into playing opportunity, retention, market-value downside, fee, and capital-risk outcomes. Some individual endpoints contained useful signal, but the mechanisms behaved differently and could not be combined into a defensible ROI number.

This phase established an important design principle: when the data supports several narrower claims but not one broad claim, preserve the narrower claims.

### 3. Incumbent-club contract-extension risk

Public Capology wage and extension histories made a more concrete decision observable. Instead of predicting whether a hypothetical transfer would succeed, the project could study known players extending with their current clubs and measure what happened afterward.

The decision unit became one player, one incumbent club, and one extension date. Predictors were restricted to information available before that date; outcomes were measured in exact future windows. This was narrower than the original idea, but much easier to define, test, and explain.

## Data and analytical unit

The research integrated public data derived from Transfermarkt, a supplemental Transfermarkt datalake, FBref, Capology, StatsBomb open event and lineup data, and public injury histories.

The source collections included approximately:

- 190,000 canonical player identities;
- 46,000 canonical club identities;
- 1.4 million dated public valuations;
- 1.1 million transfer events;
- 2.1 million player-team-competition-season performance rows; and
- more than 7 million records in the primary Transfermarkt match/event collection.

These figures describe data-engineering scale, not millions of independent model examples. After identity resolution, temporal alignment, deduplication, and eligibility rules, the integrated historical extension spine contained 2,805 unique incumbent-club extension events. Individual endpoints used smaller cohorts depending on future-outcome availability and evidence requirements.

### Canonical merge pipeline

The central engineering problem was that the sources did not share reliable universal identifiers. The pipeline:

1. cleaned each source independently while preserving source IDs and provenance;
2. resolved canonical player and club IDs using stable identifiers, normalized names, reviewed aliases, and explicit unresolved states;
3. deduplicated extension records into unique event IDs;
4. joined salary data by player, club, and season to derive prior wage and observed club salary context;
5. joined performance data with explicit row-cardinality checks;
6. reconstructed exact pre-extension, Year-1, and Year-2 windows from match-level appearances and club schedules;
7. joined dated valuation landmarks and future transfer events to construct outcomes; and
8. emitted cohort funnels, coverage checks, field dictionaries, source manifests, output manifests, and hashes.

One of the most important lessons was that a syntactically successful join can still be wrong in football terms. For example, incomplete schedules after relegation could make a player's opportunity share look artificially high or low. The repaired contribution target therefore requires evidence in both post-extension years, checks club-schedule coverage against same-league peers, and bounds opportunity shares to physically valid values.

## Outcomes that were modeled

### Sustained realized contribution

The repaired target asks whether the player reached at least 25% of the extension club's available playing opportunity in **each** of two exact post-signing anniversary years. It requires sufficient evidence for both years and measures realized involvement—not player quality, tactical suitability, or what would have happened without injury.

### Meaningful Retention

This is a stricter joint concept: the player records no outbound movement within 24 months **and** reaches the contribution threshold in both Year 1 and Year 2. The definition survived the audit, but the earlier model result did not: it predated the repaired contribution target and was superseded. A current model is not finalized.

### Temporary Displacement Risk

This isolates temporary first movement, such as a loan, instead of treating every departure as the same event. That distinction matters because a developmental loan has a different football meaning from a permanent exit.

### Permanent Separation

This asks whether the player permanently leaves the incumbent club. It remains research-only because the available model showed weak and inconsistent league-level performance.

### Public-value downside

This measures whether a player's public estimated market value declines by at least 25% within 12 or 24 months. Public valuation is an external estimate—not a transfer fee, accounting valuation, cash flow, or realized loss.

### Contract and wage context

The system calculates deterministic fixed-wage and term facts, observed salary percentile, and share of the known public wage panel. Historical peer models explored annual wage, contract duration, fixed commitment, and wage-to-public-value context. These describe comparable observed contracts; they do not estimate an optimal wage or total contract cost.

## Why these models

The project tested several model families rather than assuming that greater complexity would perform better.

- **L2-regularized logistic regression** was used for binary contribution and public-value outcomes. It handles correlated predictors, provides stable probability estimates in modest cohorts, and supports transparent feature contributions.
- **Discrete-time logistic hazard models** were used for time-to-movement questions so that event timing and censoring could be represented directly.
- **Ridge regression** was used for continuous outcomes such as opportunity share, value change, and historical wage or commitment benchmarks. Regularization reduced instability from correlated football and financial features.
- **Gradient boosting** was tested where nonlinear relationships were plausible and appeared in the historical contract-duration benchmark, but it was not assumed to be superior.
- **Chronology-only prevalence or median baselines** represented what could be predicted without player-specific features.

Within each rolling origin, preprocessing was fitted only on the training data: median imputation, missingness flags, one-hot encoding with an `__OTHER__` category, standardization, and constant-feature removal. This kept later seasons from influencing earlier transformations.

In most endpoints, simpler regularized models held up as well as or better than more complex alternatives and were easier to audit. Advanced on-ball statistics such as expected goals and progressive actions added essentially no stable incremental improvement for these particular extension cohorts, targets, and horizons. That does **not** mean advanced football statistics are generally unimportant; it means they did not improve these models beyond the available profile, playing-time, and contract information.

## Validation design

Random train/test splitting would allow later football seasons to inform predictions about earlier ones. MoveMaker instead used four rolling chronological evaluations:

| Origin | Training data | Validation season | Evaluation season |
| --- | --- | --- | --- |
| 1 | through 2018 | 2019 | 2020 |
| 2 | through 2019 | 2020 | 2021 |
| 3 | through 2020 | 2021 | 2022 |
| 4 | through 2021 | 2022 | 2023 |

After model and parameter selection, each candidate was refitted on its training and validation seasons before evaluation. Later repaired work also enforced player-disjoint evaluation where the cohort permitted it.

Binary models were assessed with Brier score, log loss, ROC AUC, average precision, calibration error, and improvement over a chronology-only baseline. Continuous models used MAE, RMSE, R², rank correlation, and empirical range coverage where relevant. Evaluation also included player-clustered bootstrap uncertainty, origin-by-origin consistency, league/position/age subgroup checks, out-of-distribution and refusal behavior, logical consistency checks, artifact hashes, and output-parity tests.

At freeze, **194 upstream verification checks and 42 final-freeze checks passed**. The freeze runner fits no models; it reconstructs every published metric from independently verified artifacts, checks supersession rules, confirms frozen binaries remain byte-identical, and reproduces the compact case-study outputs in a clean temporary run.

## Final evidence and dispositions

“Final” means a stopping decision for this case study, not that every output became production-ready. The complete fifteen-output authority is the [final metric disposition table](docs/CASE_STUDY_METRIC_DISPOSITIONS.md).

### Retained descriptive context

- **Contract arithmetic:** six exact calculations or explicitly labeled proxies survived. Fixed wages exclude bonuses, taxes, agent fees, clauses, and options unless stated; they are not total contract cost.
- **Club salary structure:** observed salary percentile and share of the known public wage panel survived with a warning that the panel is incomplete and is not audited club payroll.

### Qualified research signals

- **Sustained realized contribution:** AUC 0.762, Brier score 0.1962, calibration gap 0.011, and positive baseline-relative skill in all four rolling origins. Only 6 of 9 core league/position groups passed the frozen 99% skill gate; La Liga, Serie A, and midfield are explicitly limited.
- **Temporary Displacement Risk:** AUC 0.777 and Brier skill improvement of 0.0291, positive in all four origins. It remains a development candidate because there is no sufficiently mature, player-disjoint, untouched final cohort.
- **12-month 25% public-value downside:** AUC 0.760, Brier score 0.1794, calibration gap 0.010, positive skill in all four origins, and support in 8 of 9 broad groups. Goalkeeper performance remains uncertain, and the available 2024+ outcomes had already been exposed during prior model development.
- **24-month 25% public-value downside:** stronger development metrics—AUC 0.823 and Brier score 0.1697—but only 26 player-disjoint 2024 rows, with three observations in the smallest league subgroup, and the cohort was previously exposed. It is therefore secondary and provisional, not stronger final proof.

### Definitions or questions retained without a finalized model

- **Meaningful Retention:** the football definition survives, but the earlier performance estimate is superseded because it used the unrepaired contribution target.
- **Permanent Separation:** the question survives, but AUC 0.628 and negative league-level skill in the Bundesliga and Ligue 1 do not justify a personalized probability.

### Historical evidence only

Contract-duration, fixed-wage-commitment, and wage-to-public-value peer benchmarks won across the historical V1 rolling origins and had empirical range coverage of 82.4%, 80.1%, and 81.0%, respectively. They did not complete a successful Engine V2 final evaluation, so they are preserved as historical evidence rather than current outputs.

### Rejected or retired

- **Generic club continuity:** retired after the audit showed that loans, loan returns, and permanent exits were being blended into one apparently intuitive but semantically unstable “stability” number. Its repaired diagnostic AUC was 0.646.
- **Annual-wage candidate:** rejected despite a pooled 2024 log MAE of 0.505 and 77.0% range coverage because Ligue 1 coverage was only 60.6%, failing the predeclared Big-Five reliability gate.
- **Integrated contract score:** rejected. Its AUC of 0.683 was below the opportunity component alone at 0.688, and combining the modules hid why they disagreed.

The negative findings are part of the result. They demonstrate that a model can be statistically functional and still be unsuitable for the claim a user would naturally infer from its label.

## Retrospective decision-screening result

The historical V1 prototype was also tested as a limited-review prioritization exercise:

- 908 historical extension profiles with known terms;
- €5.22 billion in scheduled two-year fixed wages;
- a fixed capacity of 182 manual reviews;
- 79 later low-contribution outcomes found by risk-weighted ranking; and
- 54 found by reviewing the largest contracts alone.

That is **46% more historical low-contribution outcomes identified at equal review capacity**. The risk-ranked group also concentrated less wage capital than the largest-contract group.

This result is useful as a workflow case study, but its scope is deliberately narrow. It uses the historical V1 target, does not prove that the flagged contracts were mistakes, and does not estimate causal savings, confirmed loss, or a deployable current policy.

## Application architecture

The historical prototype connected the research pipeline to an explainable web interface:

```text
public sources
    -> source-specific cleaning and canonical identities
    -> dated extension master and endpoint cohorts
    -> rolling model runners and independent verifiers
    -> serialized artifacts, manifests, and hashes
    -> canonical feature retrieval and scoring
    -> profile builder
    -> FastAPI service
    -> HTML/CSS/JavaScript interface
    -> Railway deployment
```

After the post-presentation stress test uncovered target, interpretation, subgroup, and holdout-governance problems, the V1 artifacts were frozen for audit and HTTP scoring was disabled unconditionally. The hosted application now serves the research narrative, an interactive exhibit built from twelve saved historical profiles, and the health endpoint without loading models or live runtime datasets. Visitors can inspect the original product experience and reveal each player's recorded outcomes, while the interface labels which claims were retained, qualified, or retired. No environment variable can silently restore live scoring.

This fail-closed change was intentional: deployment is an engineering achievement, but keeping an inadequately governed probability live would weaken the analytical work.

## What I personally built

MoveMaker was officially a four-person capstone, and the repository represents the shared deliverable. My role was **Project Coordinator and ML Lead**. My specific ownership included:

- driving the two scope pivots and defining the narrower decision questions;
- building and integrating core cleaning, identity-resolution, canonical merge, feature, and outcome pipelines;
- selecting and implementing model families, chronological evaluation, subgroup checks, and verification contracts;
- translating model outputs into separate explainable product modules rather than one recommendation score;
- implementing and integrating the FastAPI scoring architecture and Railway deployment;
- leading the post-presentation adversarial audit, target repairs, endpoint dispositions, and fail-closed freeze; and
- translating the work into the presentation, research site, model card, and auditable repository evidence.

My teammates contributed to the shared project in their assigned roles. I describe my own implementation and analytical ownership specifically rather than presenting the entire capstone as solo work.

## The hardest technical and analytical issues

### Identity resolution and temporal validity

Joining names across football sources is deceptively difficult: spelling, accents, abbreviations, club renaming, transfers, and season conventions differ. A false match can produce a plausible row with the wrong player or club. The pipeline therefore prioritized stable source IDs, canonical mappings, reviewed aliases, unresolved states, row-count invariants, provenance, and dated joins.

The second challenge was preventing information from the future from leaking into an extension-date prediction. Every feature needed an availability date, every outcome needed an exact post-signing window, and preprocessing needed to be re-fitted inside each origin.

### Defining outcomes that match football reality

The most consequential problems were not syntax errors. They were semantic errors:

- goals and assists were originally treated globally even though their meaning differs by position;
- the old contribution target could accept incomplete Year-1/Year-2 evidence;
- schedule gaps after relegation could distort opportunity denominators;
- generic continuity mixed loans with permanent exits; and
- an accurate-looking public-value result could still lack an untouched final cohort.

The response was not to rename the outputs. We tested role-aware scoring features, removed them where that performed better, repaired the contribution evidence contract, separated movement mechanisms, and downgraded or retired outputs whose interpretation was not supportable.

## What failed—and why that improved the project

- **Universal compatibility:** insufficient complete cases and no stable general signal.
- **One integrated score:** no improvement over the strongest component and worse interpretability.
- **Advanced-stat feature groups:** no stable incremental value for the chosen extension targets and cohorts.
- **Role-aware goal/assist interactions:** did not consistently beat removing the scoring-rate features, showing that an intuitively correct adjustment can still fail empirically.
- **Generic continuity:** a convenient label for multiple incompatible movement mechanisms.
- **Permanent separation:** weak overall discrimination and inconsistent league behavior.
- **Annual wage model:** acceptable pooled behavior concealed a serious Ligue 1 reliability failure.
- **A pristine 2024 public-value holdout:** lost when the cohort was inspected during iterative development; the project records the exposure rather than relabeling it as untouched.

These failures changed both the product and the research practice. Modelability, interpretability, subgroup reliability, and holdout governance became separate gates. Passing one no longer implied passing the others.

## Where AI helped—and where human judgment remained essential

AI tools were used as a paired programmer and adversarial reviewer. They helped scaffold repetitive pipeline code, propose diagnostic checks, debug joins and deployment issues, generate verification routines, and iterate on documentation and interface copy.

AI did not determine the final claims. Human judgment was required to:

- recognize that position-insensitive scoring rates were football-incoherent;
- distinguish temporary loans from permanent separation;
- decide that realized contribution was not the same as player quality;
- reject a universal compatibility or ROI claim despite the original project ambition;
- set chronological, subgroup, and evidence-quality gates;
- identify when a technically valid result had an invalid user interpretation;
- disclose the exposed value cohort; and
- freeze live scoring when the post-presentation audit invalidated parts of the product story.

The evidence in the repository comes from saved data artifacts, model outputs, hashes, and independent checks—not from accepting generated explanations as proof.

## What the project can and cannot claim

MoveMaker can support:

- a reproducible case study in multi-source sports data integration;
- time-aware construction of incumbent-club extension cohorts;
- qualified research evidence for several separate risk questions;
- descriptive contract and observed public salary context;
- a retrospective example of using risk ranking to prioritize limited review capacity; and
- an auditable account of failed models, repaired targets, and narrowed claims.

MoveMaker cannot support:

- an automated extend/do-not-extend recommendation;
- a universal player-success, compatibility, or transfer-ROI score;
- an authorized current-player probability;
- causal claims about wages saved or contracts avoided;
- a permanent-separation probability with reliable Big-Five behavior;
- an untouched final validation claim for the public-value models; or
- the assertion that public wage observations equal complete audited club payroll.

## What I would change with more time or new data

I would reopen a model only when its documented evidence bottleneck can genuinely change—not simply to try more algorithms on the same labels.

Priority follow-ups would be:

1. acquire a genuinely untouched later public-valuation cohort and register it before any scoring or tuning;
2. accumulate a mature, player-disjoint final cohort for temporary displacement;
3. rebuild permanent separation around clearer contract-expiry, sale, release, and censoring mechanisms;
4. improve relegation and lower-tier schedule coverage for contribution outcomes;
5. evaluate wage benchmarks with more complete club payroll panels and league-specific reliability rules; and
6. incorporate trusted internal medical, tactical, squad-plan, clause, and finance data if the work were conducted inside a club.

## Reproducing the frozen conclusions

The compact final freeze can be reconstructed without fitting a model:

```bash
python models/run_case_study_metric_freeze.py --output /tmp/movemaker_case_study_metric_freeze
python scripts/verify_case_study_metric_freeze.py
```

For the complete experiment and verifier registry, see [models/README.md](models/README.md). For data-source boundaries and redistribution limits, see [data provenance](docs/DATA_PROVENANCE.md). For exact metric-level status and reopening conditions, see the [final disposition table](docs/CASE_STUDY_METRIC_DISPOSITIONS.md).

## Final takeaway

MoveMaker began by asking whether public data could predict the success and financial return of a football signing. The strongest work came from abandoning that overbroad promise.

The final case study shows how to turn fragmented public football data into a temporally valid analytical system, how to test whether a result survives across seasons and subgroups, and how to withdraw a metric when its meaning is weaker than its presentation. Its clearest lesson is that responsible modeling is not only about improving predictive performance—it is also about deciding which questions the evidence is allowed to answer.

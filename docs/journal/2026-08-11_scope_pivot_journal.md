# MoveMaker project journal — August 11, 2026 scope pivot

## Purpose and evidence standard

This journal records the development session in which MoveMaker changed from an
ambitious player-club compatibility and return-on-investment concept into a
modular transfer-risk due-diligence product. It is chronological because the
sequence matters: the final product was not chosen first and justified after
the fact. Each narrower module was proposed only after the preceding evidence
showed what the data could and could not support.

Two kinds of statements appear below. **Verified results** are tied to saved
artifacts and passing independent-verification reports. **Exploratory decisions**
describe questions, interpretations, and proposed product directions from the
working conversation. Exploratory ideas are not treated as empirical findings.
No models were rerun to write this journal.

## 1. Starting point: the original promise met the real sample

The project began with a broad question: could pre-transfer player data and
player-destination compatibility features predict sporting success and
financial return? The available folders contained gigabytes of transfer,
valuation, appearance, lineup, event, club, and seasonal player data, so the
initial intuition was that model sample size would be large.

The first major lesson was that database volume is not the same as an eligible
modeling cohort. A credible compatibility row simultaneously required a known
transfer event, fresh pre-transfer player evidence, destination-club context,
an observable post-transfer target, and a clean chronological evaluation date.
Detailed Big Five performance coverage begins in 2017, earlier financial data
could not supply all required fields, and statistics outside the Big Five were
inconsistent. Once the strict conditions were intersected, individual 2023
test slices contained only 37–58 transfers depending on the target. The often
quoted figure of roughly 63 rows described this bottleneck correctly even
though exact counts varied by endpoint and specification.

At this point the working hypothesis was not declared false solely because the
sample was small. Instead, the feature constraints were relaxed in controlled
steps and the evaluation was broadened across historical origins. This
distinguished two possible failures: a rich-design sample-size failure versus
the absence of a general compatibility signal.

## 2. Correcting the evaluation before drawing conclusions

During the diagnostic work, a shared mixed-type preprocessing defect was found
in the encoder used by several compatibility models. The affected models were
rerun after correction. This mattered methodologically: all later scope
decisions were based on corrected chronological outputs rather than the earlier
bug-affected estimates.

The corrected rich compatibility evaluation pooled four chronological origins.
It produced 207 opportunity observations, 172 destination-performance
observations, and 162 adaptation observations. These are better than one tiny
final-season slice, but still small relative to the apparent size of the source
data.

| Target | Baseline MAE | Full/policy MAE | Improvement | Player-clustered 95% CI | Origin wins |
|---|---:|---:|---:|---|---:|
| Opportunity | 0.215776 | 0.213562 | +0.002214 | [-0.005791, +0.010518] | 3/4 |
| Destination performance | 0.186427 | 0.173867 | +0.012560 | [+0.000291, +0.024308] | 4/4 |
| Adaptation | 0.185686 | 0.186333 | -0.000647 | [-0.011052, +0.009792] | 2/4 |

**Verified interpretation:** rich compatibility information showed a narrow,
repeatable increment for destination performance. It did not establish a
general opportunity or adaptation benefit, and it did not validate one score
that meant “compatibility” across outcomes. The project therefore rejected a
universal compatibility product, not the weaker statement that some contextual
variables can be predictive.

## 3. Large-sample signal check: what survives looser constraints?

The next experiment deliberately asked a more basic question. Rather than
requiring every rich feature, could broad player, market, prior-opportunity,
club, and transition context predict future opportunity in a much larger
cohort? The coarse diagnostic contained 3,134 transfers and 1,485 pooled
out-of-season evaluation transfers.

| Information added | MAE improvement | 95% player-clustered CI |
|---|---:|---|
| Market/profile over historical mean | +0.018986 | [+0.013929, +0.024232] |
| Prior opportunity | +0.003183 | [+0.001200, +0.005091] |
| Player performance | +0.000624 | [-0.001078, +0.002242] |
| Destination context | +0.006465 | [+0.002602, +0.010431] |
| Origin/transition context | +0.006793 | [+0.004801, +0.008779] |

Explicit coarse player-by-context fit terms added only 0.001197 MAE
improvement, with CI [0.000190, 0.002272]. The working discussion had wondered
whether opportunity was predictable only because it persisted from the prior
club. The decomposition answered more carefully: prior opportunity contributed
a small stable increment, but basic market/profile evidence was larger and both
destination and transition context added information. Future opportunity was
not merely a copy of prior minutes.

**Scope consequence:** MoveMaker could estimate playing opportunity in broad
bands, but the evidence did not justify calling that estimate tactical fit.
Club context should be framed as a predictive adjustment, not a causal
compatibility diagnosis.

## 4. Retention as the first adjacent commercial outcome

Before settling entirely on opportunity, the project tested whether it could
predict sustained involvement and length of stay. Exact tenure proved too
ambitious because destination-spell records are not contract records and exits
can reflect many unobserved causes. Retention nevertheless produced useful
staged evidence.

| Forecast point | Brier improvement | 95% CI | AUC |
|---|---:|---|---:|
| Sporting retention at transfer | +0.019281 | [+0.009729, +0.028616] | 0.586 |
| Uninterrupted two-year spell at transfer | +0.008980 | [+0.003343, +0.014711] | 0.577 |
| Year-two involvement after day 365 | +0.030729 | [+0.017148, +0.044994] | 0.713 |

**Verified interpretation:** transfer-time retention is a modest prior, not a
confident duration forecast. The stronger product opportunity is post-transfer
monitoring: after first-year involvement becomes known, the estimate of
year-two involvement improves materially. This expanded the product concept
from a one-time recruitment score to a possible portfolio-monitoring workflow.

## 5. The business challenge forced an explicit feasibility gate

The team then had to answer a practical capstone concern: a business proposal
needed a financial angle, not only a finding about minutes or retention. The
working conversation proposed several modules, but no feature was admitted to
the product merely because it sounded useful. A financial feasibility audit
first measured whether each target could be constructed at adequate scale.

| Candidate module | Eligible rows | Players | Event or target | Decision |
|---|---:|---:|---|---|
| 24-month estimated-value downside | 4,371 | 2,076 | 35.5% declined at least 25% | Advance |
| Paid-transfer under-utilization | 1,074 | 811 | 33.1% below 10% opportunity | Advance |
| Reported-fee benchmark | 1,486 | 1,025 | Positive reported fee | Advance |
| Severe dual capital risk | 827 | 640 | 13.3% joint event | Advance with caveat |
| Fee-only resale return | 337 | 258 | Disclosed-sale subset | Exploratory only |
| Full accounting ROI | 0 | 0 | Complete cost/return fields | Not feasible |

The audit did not prove prediction. It established which questions had enough
observable outcomes to deserve out-of-season modeling. Full ROI was rejected
because wages, bonuses, agent fees, add-ons, amortization, contract book value,
and complete realized resale proceeds were absent. Estimated market value was
retained only as a market-risk proxy, never relabeled as cash return.

## 6. Market-value downside became the strongest module

The first financial model estimated the 24-month log value ratio and the risk
of declines of at least 10%, 25%, and 50%. The strict cohort contained 4,371
transfers and the four pooled evaluation origins contained 2,686 transfers.

| Endpoint | Baseline loss | Compact-model loss | Improvement | 95% CI | Discrimination |
|---|---:|---:|---:|---|---:|
| Log value ratio | MAE 0.743482 | MAE 0.548534 | +0.194948 | [+0.172762, +0.217230] | R² about 0.443 |
| Decline at least 10% | Brier 0.255960 | 0.161530 | +0.094430 | [+0.085548, +0.103493] | AUC 0.854 |
| Decline at least 25% | Brier 0.247443 | 0.158439 | +0.089004 | [+0.079560, +0.098234] | AUC 0.851 |
| Decline at least 50% | Brier 0.187703 | 0.129853 | +0.057850 | [+0.049266, +0.066479] | AUC 0.848 |

Every endpoint beat its chronological baseline in all four origins. For the
primary 25% endpoint, the observed rate rose from 6.7% in the lowest-risk
quintile to 85.3% in the highest. Importantly, the compact 11-field
market/profile model was already sufficient. The 152-field full model improved
the 25% endpoint by only 0.000138 over compact, CI [-0.002470, 0.002754]. This
was evidence against complexity for its own sake.

The subsequent calibration experiment improved probability reliability without
materially changing ranking. For the 25% endpoint, calibration improved Brier
score by 0.006130, CI [+0.003854, +0.008335], reduced expected calibration error
from 0.081657 to 0.053656, and eliminated threshold-order violations from 157 to
zero. Comparable Brier improvements were 0.004812 at the 10% threshold and
0.004184 at 50%. The calibrated product layer therefore reports ordered risk
bands, model drivers, a scoring contract, and historical case studies.

**Commercial claim:** the module screens estimated market-value downside. It
does not predict accounting loss, sale proceeds, or ROI.

## 7. Paid-transfer under-utilization and compact refinement

The next module focused on paid permanent arrivals because an expensive player
who receives little opportunity is commercially relevant even when the cause is
unknown. The cohort contained 1,074 transfers, with 584 pooled evaluations.

| Endpoint | Baseline | Selected model | Improvement | 95% CI | Wins |
|---|---:|---:|---:|---|---:|
| Opportunity share | MAE 0.210239 | 0.189229 | +0.021010 | [+0.010690, +0.031100] | 4/4 |
| Below 10% opportunity | Brier 0.235575 | 0.200331 | +0.035244 | [+0.021678, +0.049315] | 4/4 |
| Below 25% opportunity | Brier 0.257315 | 0.229625 | +0.027690 | [+0.012207, +0.042998] | 4/4 |

The severe-risk quintiles separated clearly: observed under-10 risk increased
from 18.6% in the lowest predicted-risk quintile to 70.1% in the highest.
Under-25 risk increased from 41.5% to 83.8%.

A compact destination-context refinement then improved continuous opportunity
MAE from 0.189229 to 0.181804, an improvement of 0.007426 with CI
[+0.002321, +0.012728]. The probability findings were deliberately asymmetric.
The refined under-25 model improved Brier score by 0.012381, although its CI
[-0.001768, +0.026285] crossed zero; its AUC rose 0.034728 and ECE improved from
0.116576 to 0.107922. Conversely, the proposed calibrated under-10 replacement
worsened Brier score by 0.005106 and ECE rose from 0.113429 to 0.124662, so that
replacement was rejected.

**Frozen design:** use the compact context model for continuous opportunity,
use calibrated historical bands cautiously for meaningful under-use, and keep
the older severe-under-use ranking as a relative screening reference. Do not
convert under-utilization into a claim of tactical incompatibility.

## 8. Reported-fee benchmarking

The fee model asked whether historical pre-transfer evidence could benchmark a
reported transfer fee better than either a chronological median or the player’s
pre-transfer estimated value. Across 738 pooled evaluation deals, the compact
10-field model achieved MAE-log 0.704923, an approximate multiplicative error
factor of 2.02. It improved over pre-transfer value by 0.063933, CI
[+0.026541, +0.099825], and won three of four origins. It improved over the
chronological median by 0.595439.

The empirical ranges had correct aggregate coverage—79.3% for the nominal 80%
range and 90.2% for the nominal 90% range—but they were very broad. Median
upper-to-lower ratios were 8.5× and 19.7× respectively. Richer fields added only
0.006504 over the compact block, CI [-0.004426, +0.017436].

**Commercial claim:** this is a comparables and anomaly screen that supplies
historical fee context. It is not a precise valuation engine and cannot label a
fee objectively fair or unfair.

## 9. Component-based capital-at-risk integration

The final experiment combined fully out-of-sample value-downside and severe
under-use components. Their overlap contained 446 transfers, 397 players, and
78 severe joint events, a 17.5% event rate.

| Component or integration | Severe-event AUC | Average precision |
|---|---:|---:|
| Value risk | 0.557 | 0.192 |
| Severe under-use risk | 0.530 | 0.209 |
| Geometric joint priority | 0.604 | 0.265 |

The joint AUC exceeded value alone by 0.047, but CI [-0.033, +0.124] crossed
zero. It exceeded under-use alone by 0.075, CI [+0.001, +0.146]. The high/high
median quadrant contained a 28.8% severe-event rate versus 17.5% overall, but
joint quintiles were not strictly monotonic.

**Stopping decision:** the integration is useful as a two-axis prioritization
matrix and explanation layer. It is not an independently validated calibrated
joint probability and does not support an expected-loss claim.

## 10. Final MoveMaker product

The evidence supports an explainable due-diligence workflow rather than one
opaque transfer-success score:

1. Screen calibrated estimated market-value downside.
2. Estimate broad playing-opportunity and under-utilization tiers.
3. Compare the proposed reported fee with historical comparables and wide
   uncertainty ranges.
4. Place the transaction in a value-risk versus opportunity-risk matrix.
5. Escalate high-capital or high/high-risk cases for scouting, medical,
   tactical, contract, legal, and negotiation review.
6. Update retention, utilization, and estimated market-value risk after the
   first season.

This product has a defensible business use: it prioritizes limited diligence
resources, separates financial and sporting concerns, and records uncertainty.
It complements rather than replaces scouts, recruitment staff, medical teams,
finance teams, or negotiators.

Approved language includes *estimated market-value downside*, *playing-
opportunity estimate*, *under-utilization risk tier*, *historical reported-fee
benchmark*, *retention prior/update*, and *capital-risk prioritization matrix*.
Prohibited language includes *guaranteed minutes*, *proof of tactical fit*,
*fair fee*, *expected cash loss*, *joint failure probability*, *profit*, and
*accounting ROI*.

## 11. What the day changed

The original hypothesis was rejected in its promised universal form. That was
not a failure to produce a capstone; it was the central research result. The
project demonstrated that:

- rich public features do not automatically create a large or stable
  compatibility sample;
- detailed compatibility effects are outcome-specific rather than universal;
- ordinary market/profile and club-transition context predict broad opportunity
  better than an abstract fit score;
- estimated market-value downside is the strongest repeatable financial signal;
- under-utilization and post-year-one retention can support sporting-risk and
  monitoring workflows;
- fee benchmarking is useful only with honest, broad uncertainty; and
- integration improves decision usability more clearly than predictive power.

The final scope is narrower than the initial promise but more commercially
credible because every retained module has a defined target, chronological
baseline, uncertainty estimate, and claim boundary.

## 12. Limitations and unresolved work

Detailed performance coverage is concentrated in the Big Five from 2017
onward. Feeder leagues and earlier seasons remain incomplete. Injuries,
suspensions, managerial changes, tactical instructions, wages, bonuses, agent
fees, add-ons, amortization, contract book value, and complete resale proceeds
are absent. Public market values and reported fees are imperfect estimates.
Evaluation covers four historical origins, and the integrated severe-event
sample contains only 78 positive cases. Predictive context effects are not
causal effects.

The next phase should therefore synthesize rather than immediately add more
models: freeze live scoring inputs, build representative historical decision
cards, create presentation visuals, test interfaces with intended users, and
define drift and recalibration checks. New modeling is justified only if new
data directly addresses a known limitation—for example, contracts, wages,
injuries, or broader reliable league performance—not merely because more
algorithms are available.

## 13. Verification and artifact ledger

The saved project reports passing independent checks for compatibility models
(31/31), compatibility ablations (24/24), rolling-origin stability (25/25),
coarse context (19/19), opportunity decomposition (19/19), retention (32/32),
financial feasibility (64/64), market-value downside (32/32), market
calibration (32/32), downside product layer (30/30), paid-transfer
under-utilization (38/38), refinement (38/38), fee benchmarking (41/41), and
capital-at-risk integration (37/37). That is 462 passing checks across the
listed stages.

The primary narrative source is this journal. Exact report tables and polished
paper language are in `docs/research/2026-08-11_report_evidence_bank.md`.
Module decisions live under the corresponding `Data/processed/` directories,
with `independent_verification.json` and `.csv` files preserving verification
status. These artifacts—not exploratory conversation—are authoritative for
numeric claims.

## 14. Follow-on experiment: MoveMaker became a two-stage monitoring system

After the original journal was assembled, an external critique identified the
retention landmark result as the most commercially promising pattern: first-
year involvement had improved year-two retention prediction much more than the
transfer-time features alone. That suggestion was treated as an exploratory
hypothesis, not as evidence. A new v2 experiment tested whether the same pattern
also held for paid-transfer utilization and estimated market-value downside.

The experiment deliberately avoided two easy forms of leakage. Players with a
recorded outbound event by day 365 were removed because their destination
status was already known. Market value at the landmark was reconstructed from
canonical history using the latest positive observation on or before day 365;
the existing nearest-to-12-month field could fall after day 365 and was not
used as a predictor. Targets covered only the remaining second year: days
366-730 for opportunity, and the as-of day-365 value to the 24-month value for
financial risk.

The utilization analysis retained 667 paid permanent transfers, representing
551 players. Adding six first-year involvement measures to the compact
transfer-time profile improved year-two opportunity-share MAE by 0.027959, CI
[+0.015882, +0.040028], with four of four origin wins. For severe under-10%
opportunity, Brier score improved by 0.030663, CI [+0.015675, +0.045668], and
AUC reached 0.727. For under-25% opportunity, Brier score improved by 0.036186,
CI [+0.019723, +0.051920], and AUC reached 0.737. Both binary increments won all
four origins.

The market-value analysis retained 2,632 transfers, representing 1,730 players.
The complete monitoring update improved day-365-to-24-month log-ratio MAE by
0.005437, CI [+0.001111, +0.009578]. Brier improvements were 0.005641 for a 10%
decline, CI [+0.002141, +0.009086], and 0.004827 for a 25% decline, CI
[+0.001785, +0.007886]. Their AUCs were 0.752 and 0.729 respectively. The 50%
decline update improved Brier by 0.002065, but CI [-0.000238, +0.004420]
crossed zero and only two of four origins improved; it was therefore retained
as promising but uncertain rather than promoted.

This changed the product story in a material but disciplined way. MoveMaker is
not only a pre-signing due-diligence screen. It can also act as an early-warning
system after one season, updating a club's estimate of second-year utilization
and moderate market-value downside while the asset is still at the club. The
result does not establish causality, cash loss, or accounting ROI. All 13 build
checks and 16 independent verification checks passed, and all finalized v1
outputs remained unchanged.

One productization step remains. Because the threshold classifiers were fit
independently, the recommended utilization predictions contained 12/365 raw
under-10/under-25 ordering violations. The value update contained 34/1,630
down-25/down-10 and 21/1,630 down-50/down-25 violations. These do not invalidate
the endpoint-specific signal test, but the raw probabilities must be calibrated
and monotonically ordered before a live dashboard presents them together. The
new verification brings the cumulative documented total to 478 passing checks.

## 15. New contract data reopened the scope without reviving the rejected hypothesis

The first scope pivot had produced a credible transfer-risk due-diligence
workflow, but the business story still felt incomplete. In the working
conversation, the team and course staff returned repeatedly to contract length,
salary commitment, age at expiration, wage relative to market value, and the
question of how long a player could remain useful at their current club. Those
ideas were exploratory product questions at that stage. The earlier transfer
data could not answer them because it did not contain a sufficiently complete
historical wage-and-contract layer.

The user then collected Big-Five Capology salary panels and extension pages for
2017-2024. Missing or unavailable pages were kept visible rather than imputed:
extension pages were unavailable for 2017-18 and the Premier League 2022-23
page was unavailable, while bonus coverage was sparse before 2022. Payroll and
transfer-window snapshots were also considered in the exploratory discussion,
but the extension events and player salary panels were the data that directly
reopened a defensible modeling question.

Cleaning preserved 19,770 player-club-season salary observations and reduced
2,431 raw extension rows to 2,425 unique extension events. Conservative player
linkage resolved 2,358 events, or 97.2%; unresolved identities were not forced.
The later integration matched 2,162 extension events to salary-panel context,
provided pre-365 sporting context for 2,404, and found a strictly timed positive
signing valuation for 2,285. The modeling master retained one row per extension
and explicitly separated pre-signing predictors from post-signing targets.

This changed the unit of analysis and therefore the product claim. The new
question was not whether a player would fit a different destination club. It
was whether public evidence available when an **incumbent club extends its own
player** could support a structured contract-decision profile. Exact signing
and expiration dates made contract duration and age at expiration observable;
salary made fixed-wage exposure measurable; later appearances, outbound moves,
and valuations supplied sporting, continuity, and asset-value outcomes.

The group deliberately did not work backward from attractive interface labels
such as “transfer premium” or “contract risk.” It first created a five-phase
diagnostic checklist. A candidate output would advance only if its chronological
performance, uncertainty, calibration or ranking, and subgroup behavior
supported the language intended for the application. The five phases below are
verified analytical results. Suggestions made before these tests—including a
single risk score, optimal contract length, and full financial feasibility—were
exploratory hypotheses, not findings.

## 16. Phase 1 — future opportunity and role trajectory

### Question and design

Phase 1 asked whether an incumbent club could estimate the player's involvement
after an extension and whether a large decline in role could be converted into
a dependable probability. Four outcomes were frozen before evaluation:

1. continuous Year-2 opportunity share;
2. a major role decline among players with at least 25% pre-extension
   opportunity;
3. meaningful contribution in both Years 1 and 2, defined as at least 25%
   opportunity in each year; and
4. minimal Year-2 involvement below 10% while the uninterrupted same-club spell
   remained contract-covered through day 730.

The primary cohort contained 1,385 extensions with a linked player, at least 10
club games in the exact pre-365 and Year-2 windows, and a scheduled expiration
no earlier than day 730. The major-decline subset contained 922 events. The
pooled rolling-origin evaluation covered 929 primary events and 603 initial-role
events across signing years 2020-2023. Hyperparameters were selected only on
the preceding validation year, and uncertainty used 5,000 paired
player-cluster bootstrap repetitions.

In the full eligible cohorts, mean Year-2 opportunity was 35.7% and the median
was 31.9%. Major role decline occurred in 327 of 922 cases (35.5%), sustained
meaningful contribution in 694 of 1,385 (50.1%), and contract-covered minimal
involvement in 120 of 1,385 (8.7%).

### Verified findings and decisions

| Endpoint | Selected-model result | Improvement over baseline | Stability | Decision |
|---|---|---|---:|---|
| Year-2 opportunity share | MAE 0.2369; R² 0.249; Spearman 0.512 | MAE +0.05863, 95% CI [+0.04851, +0.06865] | 4/4 origin wins; monotonic bands | Advance as a broad range/tier |
| Major role decline | AUC 0.581; AP 0.450; Brier 0.2282 | Brier +0.00136, 95% CI [-0.00846, +0.01128] | 2/4 wins; nonmonotonic bands | Reject probability output; descriptive only |
| Sustained meaningful contribution | AUC 0.788; AP 0.732; Brier 0.1869 | Brier +0.06425, 95% CI [+0.05133, +0.07689] | 4/4 wins; monotonic bands | Advance probability/tier candidate |
| Contract-covered minimal involvement | AUC 0.759; AP 0.361; Brier 0.0704 | Brier +0.01204, 95% CI [+0.00342, +0.02092] | 4/4 wins; nonmonotonic bands | Exploratory only |

The ordered bands give the advanced outputs an intuitive interpretation:
observed Year-2 opportunity increased from 10.8% in the lowest band to 55.8%
in the highest, while observed sustained contribution increased from 7.5% to
75.8%. Contract economics added small pooled improvements to Year-2 opportunity
(+0.00395 MAE) and sustained contribution (+0.00604 Brier), but those
increments were not stable in every recent-window sensitivity test. Wage and
term are therefore secondary adjustments, not the core explanation.

**Product change after Phase 1:** MoveMaker could display an `Expected Year-2
involvement range` and a `Meaningful contributor through Year 2` probability or
tier. It could not promise exact minutes, a dependable major-role-decline
probability, or a production-ready severe-minimal-involvement probability. This
was the first evidence that the extension product should communicate broad
horizon outcomes rather than dramatic but unstable warning labels. All 9 build
checks and all 32 independent verification checks passed.

## 17. Phase 2 — continuity and time to outbound movement

### Question and design

Phase 2 asked how likely the player was to remain involved with the incumbent
club over the proposed contract horizon. It replaced a fixed 24-month retention
label with a discrete-time hazard design so right-censored cases could
contribute every fully observed half-year interval.

Two event definitions were kept separate. `Any outbound` treats a loan, loan
return, or permanent move as the end of uninterrupted club involvement.
`Permanent outbound` ends only when the permanent club relationship ends, so a
loan does not count as permanent separation. The administrative censoring date
was 2026-08-10. Scheduled contract expiration remained a signing-time predictor
and audit boundary, not a censor: 719 of 1,848 any-outbound events and 878 of
1,690 permanent-outbound events were recorded after the listed expiration,
consistent with renewals or later recorded club spells.

The cohort contained 2,358 linked extensions, 1,848 observed any-outbound
events, and 1,690 permanent-outbound events. All 2,358 cases had complete 12-
and 24-month administrative follow-up; 36-month evaluation used mature
2020-2022 origins only. At 24 months, 48.5% had any outbound event and 32.5%
had a permanent outbound event.

### Verified findings and decisions

| Event definition and horizon | Brier | AUC | Brier improvement over time-only baseline | Wins | Decision |
|---|---:|---:|---|---:|---|
| Any outbound, 24 months | 0.2191 | 0.710 | +0.03632, 95% CI [+0.02766, +0.04527] | 4/4 | Advance |
| Any outbound, 36 months | 0.2026 | 0.699 | +0.02103, 95% CI [+0.01134, +0.03066] | 3/3 | Advance |
| Permanent outbound, 24 months | 0.2041 | 0.696 | +0.02572, 95% CI [+0.01856, +0.03313] | 4/4 | Advance |
| Permanent outbound, 36 months | 0.2280 | 0.681 | +0.02554, 95% CI [+0.01517, +0.03599] | 3/3 | Exploratory only |

The 24- and 36-month any-outbound bands were monotonic, with observed event
rates rising from 28.2% to 76.9% and from 51.7% to 88.1%, respectively.
Permanent-outbound 24-month bands rose from 15.7% to 56.4%. The permanent
36-month model ranked groups but missed the mean observed event rate by more
than five percentage points and failed the terminal-window stability gate.

Salary and contract evidence added repeatable Brier improvements beyond player
and performance history: +0.01983 for any outbound at 24 months, +0.01079 at 36
months, +0.01110 for permanent outbound at 24 months, and +0.01071 at 36
months; every clustered interval excluded zero. These are predictive
associations, not proof that a wage or term causes departure. The survival
formulation did not beat direct fixed-horizon classifiers significantly on
Brier score; its value was coherent use of censoring and probabilities that
cannot imply lower cumulative outbound risk at a longer horizon.

**Product change after Phase 2:** MoveMaker gained candidate probabilities of
continuous club involvement through 24 and 36 months and of a permanent club
relationship through 24 months. The test rejected an exact departure date,
exact tenure, automatic recommended term, and a validated permanent-stay
probability at 36 months. The output became a horizon-specific continuity card,
not a prediction of “the player will stay 2.8 years.” All 12 build checks and
all 33 independent verification checks passed.

## 18. Phase 3 — public market-value preservation

### Question and design

Phase 3 asked whether an extension decision could be accompanied by a measured
public-asset-value downside outlook. It froze continuous log value change and
nested declines of at least 10%, 25%, and 50% at 12 and 24 months. Signing
valuations had to be positive and within 365 days of signing; follow-up
valuations had to be positive and within 90 days of the target date. The target
is Transfermarkt-style public market value, not sale price or accounting value.

The strict cohorts contained 2,162 extensions at 12 months and 1,944 at 24
months. Pooled out-of-time evaluations contained 1,432 and 1,367 cases across
2020-2023. In the full 24-month cohort, 54.8% lost at least 10% of signing
value, 46.7% lost at least 25%, and 30.2% lost at least 50%.

### Verified findings and decisions

| Horizon and endpoint | Selected-model result | Improvement over chronological baseline | Wins | Decision |
|---|---|---|---:|---|
| 12m log value ratio | MAE 0.4636; Spearman 0.607 | MAE +0.07819, 95% CI [+0.05072, +0.10636] | 4/4 | Advance broad range |
| 12m downside ≥10% | Brier 0.1901; AUC 0.786 | +0.05742, 95% CI [+0.04761, +0.06765] | 4/4 | Advance probability |
| 12m downside ≥25% | Brier 0.1713; AUC 0.781 | +0.04571, 95% CI [+0.03476, +0.05696] | 4/4 | Advance probability |
| 12m downside ≥50% | Brier 0.0976; AUC 0.778 | +0.01158, 95% CI [+0.00362, +0.01973] | 2/4 | Exploratory only |
| 24m log value ratio | MAE 0.6208; Spearman 0.663 | MAE +0.20708, 95% CI [+0.17091, +0.24247] | 4/4 | Advance broad range |
| 24m downside ≥10% | Brier 0.1654; AUC 0.837 | +0.08290, 95% CI [+0.07090, +0.09469] | 4/4 | Advance probability |
| 24m downside ≥25% | Brier 0.1717; AUC 0.821 | +0.07806, 95% CI [+0.06582, +0.08970] | 4/4 | Advance probability |
| 24m downside ≥50% | Brier 0.1646; AUC 0.805 | +0.05506, 95% CI [+0.04364, +0.06647] | 4/4 | Advance probability |

All eight selected-model risk-band sequences were monotonic. At 24 months,
observed 10%, 25%, and 50% downside rose across the five bands from 17.2% to
90.5%, 15.7% to 86.1%, and 8.0% to 65.7%. The continuous 24-month MAE implies
an approximate 1.86× multiplicative error factor, so it supports a broad range
rather than a precise future euro value.

Contract economics improved the three 12-month downside probabilities, with
increments of +0.00737, +0.01197, and +0.00262 Brier and confidence intervals
excluding zero. No 24-month contract-economics increment was validated. The
durable longer-horizon signal came mainly from age, current value, position,
league, and player history. Because the three thresholds were trained
independently, 129 of 1,367 raw 24-month prediction rows violated probability
ordering; the endpoint tests remained valid, but the probabilities required a
monotone display projection before appearing together.

**Product change after Phase 3:** MoveMaker gained broad 12- and 24-month
public-value ranges, 12-month 10% and 25% downside probabilities, and all three
24-month downside probabilities. It withheld the 12-month 50% probability. The
commercial language was fixed as *public-value preservation/downside*, never
expected sale proceeds, accounting loss, profit, or ROI. All 10 build checks
and all 32 independent verification checks passed.

## 19. Phase 4 — contract facts and historical financial peer context

### Question and design

Phase 4 separated two kinds of output that had been conflated in early product
discussion. Input-driven arithmetic answers “what does this proposed offer
commit?” Historical benchmarking answers “how unusual is it relative to
similar observed extensions?” Neither answers whether the contract is optimal.

The source remained 2,425 extension events. Usable cohorts were 2,265 for
annual wage, 2,358 for duration, 2,265 for fixed-wage commitment, and 2,222 for
wage-to-market-value. Four chronological origins covered 2020-2023 and produced
36,264 row-level predictions across four endpoints and six fixed model
variants. Current extension terms and their derivatives were prohibited from
the benchmark predictors. The ladder began with age, position, league, and
signing market value, added prior-season wage, and then tested prior
opportunity, sporting history, and a compact nonlinear alternative.

### Verified findings and decisions

| Historical benchmark | Selected model | MAE | R² | Spearman | Improvement over chronological median | 80% range coverage |
|---|---|---:|---:|---:|---|---:|
| Annual fixed wage | Profile + prior wage | 0.5193 log EUR | 0.729 | 0.889 | +0.5994, 95% CI [+0.5557, +0.6427] | 81.3% |
| Contract duration | Compact nonlinear sporting | 0.8291 years | 0.427 | 0.659 | +0.3854, 95% CI [+0.3460, +0.4237] | 82.4% |
| Fixed-wage commitment | Profile + prior wage | 0.6881 log EUR | 0.644 | 0.832 | +0.5936, 95% CI [+0.5423, +0.6455] | 80.1% |
| Wage-to-market-value | Profile + prior wage | 0.5120 log ratio | 0.663 | 0.835 | +0.4442, 95% CI [+0.4024, +0.4870] | 81.0% |

Every selected model beat the time-aware median in all four evaluation years,
every clustered confidence interval excluded zero, and every five-band sequence
was monotonic. The ranges remained intentionally broad. Median errors were
approximately 1.47× for annual wage, 0.63 years for duration, 1.69× for fixed
commitment, and 1.45× for wage-to-market-value. Prior wage added stable signal
for wage (+0.08060 MAE), fixed commitment (+0.07297), and wage/value
(+0.08247). Sporting history did not improve those three; only duration
benefited from opportunity, sporting history, and the nonlinear form.

The deterministic layer can calculate, when required inputs are supplied:

- age at expiration;
- annual fixed wage and fixed-wage commitment (`annual wage × contract years`);
- annual wage and fixed commitment relative to public market value;
- transfer premium relative to public market value; and
- `fee + fixed wages` as an explicitly partial acquisition-commitment proxy.

Matched salary panels can also provide a player percentile and share of known
club payroll. They are not audited complete payrolls. The data do not contain
employer taxes, agent/intermediary fees, signing-on fees, release clauses,
unilateral options, or reliable historical performance-based pay. Bonus was
reported for 759 events and was essentially unavailable before 2022; missing
bonus was never converted to zero.

**Product change after Phase 4:** The financial story became concrete and
application-friendly: show the proposed wage, term, and fixed commitment beside
a broad historical peer range and clearly labeled premium or delta. This
supports statements such as “22% above the historical peer benchmark,” not
“22% overpaid.” Phase 4 rejected a financial-feasibility score, optimal wage or
term, fair-value claim, complete acquisition cost, expected profit, and ROI.
All 10 build checks and all 33 independent verification checks passed.

## 20. Phase 5 — integration tested, and the headline score was rejected

### Question and design

Phase 5 did not assume that four useful modules should be averaged. It asked
whether combining them improved a defined triage decision. The historical
profile layer contained 1,560 unique out-of-time extension events. Availability
was 929 for opportunity, 1,560 for 24-month survival, 1,560 for 36-month
survival, 1,432 for 12-month value, 1,367 for 24-month value, and 1,504 for the
financial peer layer. A complete opportunity/survival/value profile existed for
839 extensions across 2020-2023.

The integration target was adverse outcomes in at least two of three domains:
no sustained meaningful contribution through Year 2, any outbound movement by
24 months, or at least a 25% public-value decline by 24 months. This occurred in
42.6% of complete cases. Component predictions were converted to within-year
ranks before transparent combinations; no new raw-feature meta-model was fit.

### Verified composite test

| Triage signal | AUC | Top-quintile event rate | Lift over cohort rate |
|---|---:|---:|---:|
| Opportunity risk | 0.6885 | 68.5% | 1.61× |
| 24m outbound risk | 0.6606 | 60.1% | 1.41× |
| 24m value-downside risk | 0.4597 | 45.2% | 1.06× |
| Equal-weight three-module rank | 0.6830 | 69.2% | 1.63× |
| Opportunity + outbound rank | 0.6986 | 67.3% | 1.58× |
| Maximum component rank | 0.6468 | 58.2% | 1.37× |

The equal-weight score was worse than opportunity alone by 0.00543 AUC, with a
player-clustered 95% CI of [-0.03947, +0.02849]. Opportunity plus outbound was
better by only 0.01019, CI [-0.00953, +0.02991]. Both comparisons won three of
four origins, but neither interval excluded zero. The overall contract score
was therefore rejected. This does not invalidate the value model on its own
downside target; it shows that value risk did not improve this particular
two-of-three composite triage outcome.

The integration also resolved the raw value-threshold display problem. A
monotone projection reduced 50 pairwise violations at 12 months and 134 at 24
months to zero, while changing mean Brier score by only -0.00020 and -0.00006.
Evaluation-derived 80% reference widths remained broad: 0.3699 opportunity-
share points and log widths of 0.7213 at 12 months and 0.9920 at 24 months,
equivalent to approximate multiplicative factors of 2.06 and 2.70 around the
value point estimate. They are historical reference ranges, not formal
individual guarantees.

Twelve deterministic case studies—two each for aligned high risk, aligned low
risk, false-positive stress, false-negative stress, value-only divergence, and
sporting/continuity-only divergence—were retained to show failures and
cross-domain disagreement as well as successes.

**Product change after Phase 5:** Integration improved organization and
explanation, not validated predictive power. MoveMaker would not hide distinct
sporting, continuity, value, and financial signals inside a headline number.
The rejected score became an explicit product requirement: `Do not display`.
All 11 build checks and all 33 independent verification checks passed.

## 21. Final contract-extension product and five-card interface

The five phases support one coherent commercial proposition:

> MoveMaker is an explainable incumbent-club contract-extension decision-support
> tool. It shows broad future involvement, horizon-specific club continuity,
> public market-value downside, fixed financial commitment, and how the proposed
> offer compares with historical peers.

The final application profile contains five separate cards:

1. **Contract Facts** — age at expiration, fixed-wage commitment, wage/value,
   commitment/value, and optional fee-based arithmetic;
2. **Historical Peer Context** — broad wage, duration, fixed-commitment, and
   wage/value ranges plus clearly labeled deviations from peers;
3. **Future Opportunity** — a broad Year-2 opportunity reference range and a
   sustained-contribution probability or tier;
4. **Club Continuity** — continuous stay through 24/36 months and permanent
   club relationship through 24 months; and
5. **Public-Value Outlook** — broad 12/24-month value ranges and ordered
   downside probabilities at the supported thresholds.

The separate cards preserve decision-relevant disagreement. A player may have
a strong opportunity outlook but high public-value downside; another may have
ordinary financial terms but elevated outbound risk. Those are different
questions for coaching, sporting, and finance staff. An average score would
erase the reason a case needs review.

### Claims now supported

- The proposed fixed-wage commitment and selected ratios can be computed from
  guarded user inputs.
- A proposed wage or term can be described as above or below a broad historical
  extension-peer range.
- The models can estimate broad Year-2 involvement, sustained contribution,
  horizon-specific stay probabilities, and public-value preservation/downside
  for incumbent-club extensions.
- The dashboard can identify aligned risks and cross-domain trade-offs that
  deserve human review.

### Claims explicitly prohibited

- a validated overall “good contract/bad contract” or success score;
- an optimal wage, optimal term, objectively fair value, or proof of
  overpayment;
- exact future minutes, an exact departure date, or guaranteed retention;
- complete acquisition cost from fee plus wages;
- realized resale proceeds, accounting loss, profit, or ROI from public market
  value; and
- applying the extension models to a player joining a new club or interpreting
  a low probability as tactical incompatibility.

The scope boundary is commercially important. A proposed transfer fee may be
entered for transparent transfer-premium or fee-plus-fixed-wage arithmetic, but
that does not turn an incumbent-extension model into a new-club transfer model.

## 22. Productionization boundary and next handoff

The research product is validated offline, not yet a live scoring service.
Deterministic contract calculations are ready for application logic. Observed
salary-panel context is ready when a backend match exists, and coarse static
peer tables can be displayed with sample-size and range warnings. Personalized
Phase 1-4 predictions still require:

1. a declared final training cutoff and full-history refit;
2. serialized preprocessors and model artifacts;
3. canonical backend retrieval of age, position, league, prior wage,
   opportunity, performance, and valuation features;
4. scoring-parity tests against the frozen offline calculations;
5. preservation of probability ordering and horizon coherence; and
6. version, drift, and recalibration monitoring.

If a required backend field is unavailable, the application must return
`unavailable`; it must never silently substitute zero. This is why Phase 5 is
the end of broad diagnostic modeling, not the end of engineering work.

Across the contract program, Phase 1 passed 32 independent checks, Phase 2
passed 33, Phase 3 passed 32, Phase 4 passed 33, and Phase 5 passed 33. The five
phases therefore add 163 independently verified checks. Added to the 478 checks
documented through the day-365 monitoring experiment, the journal's cumulative
ledger is 641 passing checks. The authoritative evidence remains the phase
decision summaries and independent-verification artifacts under
`Data/processed/`; conversational proposals remain historical context rather
than empirical support.

## 23. Production, model repair, and presentation completion

The production boundary described above was subsequently crossed in a reviewed
engineering stage. The selected Phase 1-4 endpoints were refit through the
declared cutoff, serialized behind a canonical feature retriever, exposed by a
FastAPI service, and connected to the extension-profile HTML application. The
live product preserves the Phase-5 decision: it exposes separate explained
signals and no overall contract-success score.

A presentation rehearsal then revealed that elite demo players received
counterintuitive sustained-contribution scores. This was investigated as a
model-integrity problem, not hidden with a display override. The audit found a
live league-name encoding mismatch and an unstable global goals/assists-per-90
specification. The mapping bug was corrected, and the C6 compact candidate
removed the two unstable per-90 inputs. C6 cleared the pre-declared compact
non-inferiority route, reduced missing-input dependence, passed deployment
parity and smoke tests, and replaced only the sustained-contribution artifact.
It did not add a superstar bonus, probability floor, or manual correction.

Later current-season performance and salary files were added through an
append-only live-input refresh. The frozen out-of-time research datasets and
metrics remained unchanged. A separate role-contextualization diagnostic found
that 12 of 16 deployed endpoints still used globally specified scoring-rate
features. Its endpoint-level results were mixed and no candidate was promoted;
that work remains an explicit next research stage rather than a hidden
production change.

The final presentation reframed the technical story around a business
decision: the team began with player/player and player/club compatibility plus
transfer return, rejected that promise when the evidence failed, learned that
risk dimensions behaved differently, and used newly collected salary and
extension data to build a narrower incumbent-club decision profile. The final
financial illustration quantified €5.22B in two-year fixed wages across 908
out-of-time extensions, of which €1.82B was associated with low Year-2
contribution. At equal capacity to review 182 extensions, risk-weighted
screening identified 79 historical low-contribution outcomes versus 54 from
reviewing the largest contracts alone. The claim remained screening efficiency,
not confirmed loss or projected savings.

The team completed and delivered the final presentation successfully. This is
not the end of MoveMaker: the post-presentation phase focuses on repository
reproducibility, role-aware or removal-based model repair, current-data
ingestion, uncertainty and out-of-distribution behavior, and a deployable public
demo. The repository README and stage-specific evidence files now describe the
current product state; this journal remains the chronological research record.

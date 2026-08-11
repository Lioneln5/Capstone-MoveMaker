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
6. Update retention and involvement risk after the first season.

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

The primary narrative source is `CODEX.md`. Exact report tables and polished
paper language are in `docs/research/2026-08-11_report_evidence_bank.md`.
Module decisions live under the corresponding `Data/processed/` directories,
with `independent_verification.json` and `.csv` files preserving verification
status. These artifacts—not exploratory conversation—are authoritative for
numeric claims.

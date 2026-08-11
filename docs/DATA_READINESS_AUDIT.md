# MoveMaker Data Readiness Audit

**Audit date:** 2026-08-06  
**Question:** What data do we actually have, how much is model-worthy under the project rules, what is not ready, and what should we acquire next?

## Executive verdict

Yes, MoveMaker has enough data to begin modeling, but not every intended model is equally ready.

- A **Transfermarkt-only sporting model is viable now**. There are **4,856 horizon-safe 24-month transfer rows** with recent player appearances and sufficient destination-club match coverage.
- A **coarse compatibility extension is also viable**. **3,134** of those rows also have at least 10 pre-transfer games for both the origin and destination clubs. The club information is mostly form/strength context, not yet a complete tactical-style representation.
- A **richer FBref-player sporting model is plausible but not final**. The current legacy cohort contains **970** horizon-safe sporting rows with a fresh immediately preceding season; **730** also have both pre-transfer club contexts. This is enough for restrained, interpretable modeling, but the cohort must be rebuilt from the cleaned FBref table and reviewed crosswalks.
- A **financial/value-return model has 1,788 horizon-safe provisional rows**, but zero rows are currently confirmed as paid permanent transfers. Movement type is unresolved, so this sample is suitable for pipeline development and sensitivity analysis, not a final claim about transfer ROI.
- The current Transfermarkt-only 24-month sporting/financial intersection is **847 rows**, or **699** with both club contexts. The rich FBref intersection is only **216 provisional rows**. A single complex “overall success” model would therefore be fragile and is not recommended.
- StatsBomb Open Data is useful for demonstrations and case studies, not the main training panel. Only **32 transfers** have at least three selected pre-transfer player matches, and none has at least three pre-transfer matches for the player, origin club, and destination club simultaneously.

The honest position is:

> We can build and evaluate sporting baselines now. We can begin a compatibility experiment after final feature engineering. We cannot yet produce a defensible realized-ROI or unified overall-success model.

## 1. What “model-worthy” means in this audit

A row count in a master file is not automatically a training sample. This audit applies the project’s time and coverage rules.

### Common requirements

A modeling row must:

1. represent one unique cross-club transfer event;
2. use only information observed before the transfer as features;
3. have an observable outcome horizon;
4. have enough destination-club coverage to distinguish a true zero outcome from missing coverage; and
5. exclude current-snapshot fields that would leak later information.

### Transfermarkt sporting model

The strict 24-month sporting audit requires:

- at least one recorded player appearance in days -365 through -1;
- at least 10 destination-club games in months 0-12;
- at least 20 destination-club games in months 0-24; and
- the full 24-month sporting horizon to fall within the available game-data range.

The existing `eligible_sporting_24m_provisional` flag does not itself require the full horizon to be observable. It identifies 5,274 rows; adding the explicit horizon rule reduces the defensible count to 4,856. The final cohort builder should incorporate this horizon condition directly.

### Rich FBref-player sporting model

The rich version additionally requires:

- the immediately preceding completed season;
- that season ending 1 to 365 days before the transfer;
- a unique/reviewed player identity match; and
- the same destination coverage and horizon rules.

An older season is not silently substituted when the immediate season is absent.

### Financial/value-return model

The provisional 24-month financial audit requires:

- a positive reported fee;
- a market valuation on or before the transfer that is no more than 365 days old;
- a valuation within ±90 days of the 24-month target; and
- the full valuation horizon to be observable.

This still does **not** establish that a movement was a paid permanent transfer. Until movement type is available, the financial cohort remains provisional.

### Compatibility model

Compatibility requires both a player profile and a destination-club profile known before the transfer. For a conservative coverage diagnostic, this audit requires at least 10 pre-transfer games for both the origin and destination clubs. Ten games is an audit threshold, not yet a formally approved modeling rule.

## 2. Data assets currently available

| Data layer | Local coverage | State | Modeling use |
|---|---:|---|---|
| Transfermarkt transfers | 35,139 transfer events; 4,345 players; 1993-07-01 to 2030-06-30 | Cleaned and in the comprehensive master | Event spine, fee field, origin/destination IDs |
| Transfermarkt valuations | 507,815 player-date observations | Cleaned and time-windowed | Pre-transfer value and 12/24-month value targets |
| Transfermarkt appearances | 1,894,350 player-game appearances | Cleaned and aggregated into 365-day, 730-day, and career-pre layers | Recent form and sporting outcomes |
| Transfermarkt lineups | 3,179,016 lineup/substitute records | Cleaned and aggregated | Starts, bench selection, captaincy, role evidence |
| Transfermarkt events | 1,274,469 recorded events | Cleaned and aggregated | Goals, assists, cards, substitutions and other event counts |
| Transfermarkt games/club games | 88,958 games and 177,916 club-game rows | Cleaned and aggregated | Origin/destination form, competition and outcome coverage |
| Transfermarkt dimensions | Players, clubs, competitions, countries, national teams | Cleaned; historical unions built | Identity/context, with snapshot leakage warnings |
| Comprehensive Transfermarkt master | 35,139 rows × 577 columns | Built and independently verified | Broad transfer-level source for cohort construction |
| Cleaned FBref-style Big Five panel | 18,243 player-squad-competition-season rows; 2017-18 through 2023-24 | Cleaned, documented and verified | Detailed player performance and position-specific metrics |
| Player crosswalk | 5,967 seasonal identities | Existing but requires reviewed update | FBref-to-Transfermarkt identity link |
| StatsBomb case-study layer | 642 selected matches and 2,298,900 selected events | Extracted and verified | Case studies and visual explanations, not main training data |

The twelve cleaned Transfermarkt source tables contain **7,208,921 rows** in total. This is a substantial factual base, but the relevant modeling unit remains a transfer event, not a raw event or lineup row.

## 3. What is merged, partially merged, and still separate

### Fully incorporated into the comprehensive Transfermarkt master

- transfer facts;
- player identity/reference fields;
- pre-transfer valuations and fixed-horizon valuation targets;
- 365-day, 730-day, and career-pre player histories;
- post-transfer appearance, lineup and event outcomes;
- pre-transfer origin/destination club form;
- post-transfer destination-club coverage; and
- eligibility and leakage-control flags.

### Processed but not finalized in the comprehensive modeling pipeline

- the cleaned FBref player-squad-season table;
- the FBref-to-Transfermarkt player crosswalk;
- aggregated FBref player-season features;
- the legacy `transfer_cohort_master.csv`, whose builder still uses the old seasonal inputs and old prior-season rule; and
- the StatsBomb case-study tables and manual-review crosswalk candidates.

### Not yet built or finalized

- a reviewed FBref squad-season-to-Transfermarkt club crosswalk;
- final player-season features rebuilt from the cleaned FBref source;
- explicit player-to-destination compatibility features;
- a frozen sporting-success label or composite score;
- confirmed permanent/loan/free/return movement classification;
- a final financial training cohort;
- chronological train/validation/test fold assignments; and
- one final leakage-safe model matrix containing only approved predictors and targets.

## 4. Actual feature coverage in the broad master

The broad master deliberately retains missing rows. Coverage is therefore uneven by date and source.

| Coverage condition | Transfer rows | Share of 35,139 |
|---|---:|---:|
| Pre-transfer valuation within 365 days | 20,547 | 58.5% |
| Any pre-365 appearance record | 10,300 | 29.3% |
| Pre-365 appearance for the origin club | 7,346 | 20.9% |
| Any pre-365 lineup record | 15,557 | 44.3% |
| Any pre-365 recorded event | 12,355 | 35.2% |
| Any origin-club pre-365 games | 14,424 | 41.0% |
| Any destination-club pre-365 games | 15,606 | 44.4% |
| At least 10 pre-365 games for both clubs | 4,146 | 11.8% |
| Positive reported fee | 2,584 | 7.4% |
| Reported zero, movement unclassified | 20,064 | 57.1% |
| Missing fee | 12,491 | 35.6% |

Important interpretations:

- An absent player event is not necessarily a zero. Event data is selective, so event-derived fields should generally supplement appearance/lineup features rather than gate the sample.
- Position-dependent missingness is structural. Goalkeeper fields should remain unavailable for outfield players rather than causing entire columns or rows to be discarded.
- The master contains 15 future-dated transfers after 2026-08-06 and 1 same-club-ID movement. Neither belongs in a historical training cohort.
- The player/club snapshot fields are useful identifiers but some are unsafe historical predictors.

## 5. Model-worthy sample sizes

### Transfermarkt-only cohorts

| Candidate model | Existing encoded count | Horizon-safe count | With both pre-transfer club contexts |
|---|---:|---:|---:|
| Sporting, 12 months | 5,430 | 5,391 | Not audited here |
| Sporting, 24 months | 5,274 | **4,856** | **3,134** |
| Financial/value return, 12 months | 1,999 provisional | **1,999 provisional** | Not required |
| Financial/value return, 24 months | 1,790 provisional | **1,788 provisional** | Not required |
| Sporting + financial, 24 months | 847 provisional | **847 provisional** | **699 provisional** |

The 4,856-row sporting cohort spans 2012-08-01 through 2024-07-05 and contains 1,656 unique players. It is large enough for baseline models, chronological validation, subgroup diagnostics and a controlled compatibility comparison.

The financial cohort is numerically adequate for a modest model, but it is not semantically final. All 1,788 rows still inherit unresolved movement-type risk.

### Rich FBref-player cohorts

The existing 2018-2024 legacy cohort contains 17,060 dated transfer candidates.

| Rich-cohort stage | Rows |
|---|---:|
| Any attached prior seasonal features | 1,890 |
| Immediately preceding season | 1,312 |
| Immediately preceding season and 1-365 days fresh | **1,294** |
| Fresh and sporting-eligible at 12 months | 979 |
| Fresh and horizon-safe sporting-eligible at 24 months | **970** |
| Fresh 24-month sporting plus both club contexts | **730** |
| Fresh sporting plus provisional financial eligibility at 24 months | **216** |

The 970-row sample is the best estimate of the current rich sporting ceiling before rebuilding the cohort from the cleaned source and reviewed mappings. The 730-row sample is the closest current approximation to a rich compatibility cohort, but it is not a finalized training file.

Chronological coverage is usable but not large: the fresh 24-month sporting counts are 92 in 2018, 158 in 2019, 154 in 2020, 140 in 2021, 184 in 2022, 211 in 2023 and 31 in early 2024. This supports simple/regularized models more comfortably than highly parameterized models.

### Player identity resolution

The existing player crosswalk contains:

- 5,483 uniquely accepted seasonal identities;
- 470 unmatched identities;
- 14 ambiguous identities; and
- 17,043 of 18,243 seasonal rows linked uniquely, or 93.42%.

This is high-precision coverage, but it is not the finished resolver. There are 247 transfer attention events involving 145 player-season combinations where Transfermarkt indicates relevant Big Five participation but the seasonal row did not link. These need reproducible alias overrides and manual review.

### StatsBomb overlap

StatsBomb should not be counted as a broad model-training expansion:

- 612 transfer case-study candidates exist;
- 146 have at least one selected player match in the preceding 365 days;
- 32 have at least three;
- 11 have at least five;
- 4 have at least three player matches before and after the transfer; and
- 0 have at least three preceding matches for the player, origin club and destination club.

It remains valuable for narratives, visualizations and sanity-checking what “fit” can look like at event level.

## 6. What can genuinely be modeled now

### Ready enough to begin

1. **Transfermarkt sporting baseline**
   - Candidate sample: 4,856.
   - Possible targets: destination minutes, appearances, starts, lineup inclusion, minutes share, or position-adjusted contribution at 12/24 months.
   - Suitable first models: regularized regression, gradient-boosted trees and simple classification after a target is frozen.

2. **Transfermarkt club-context extension**
   - Candidate sample: 3,134 with both origin and destination histories.
   - Current club variables capture form, points, goals and competition mix.
   - This can test whether destination context adds predictive value, although it is not yet a complete tactical compatibility test.

3. **Rich FBref sporting experiment**
   - Candidate ceiling: 970 fresh/horizon-safe rows, or 730 with both club contexts.
   - This is enough for a restrained capstone experiment if features are reduced, position handling is explicit and chronological folds are used.

### Only provisional

1. **Financial/value-return model**
   - Candidate sample: 1,788.
   - Target available: change in market value or fee-relative market-value proxy.
   - It is not realized ROI and should not be presented as such.

2. **Combined sporting/financial model**
   - Candidate sample: 847 Transfermarkt-only or 216 with fresh FBref data.
   - The intersection is better used for validation or a decision layer than for one high-dimensional model.

### Not ready for an honest final claim

- paid-permanent-transfer ROI;
- realized resale profitability;
- wage-adjusted or amortization-adjusted return;
- injury-adjusted success;
- a single “overall transfer success” classifier; or
- a high-resolution tactical compatibility model across the full transfer universe.

## 7. The biggest gaps

There are three different “biggest gaps,” depending on what is being judged.

### Biggest data gap for the stated ROI objective: movement and financial semantics

Only 2,584 transfers have a positive reported fee, while 20,064 are reported as zero and 12,491 are missing. More importantly, the local source does not reliably distinguish:

- permanent purchase;
- loan and loan fee;
- loan return;
- free transfer;
- youth/reserve promotion;
- undisclosed fee; and
- option or obligation to buy.

Because movement type is unresolved for every row, the number of **confirmed** paid-permanent financial training rows is currently zero. This is the most serious gap if the project continues to use the word “ROI.”

The data also lacks realized resale proceeds, wages, signing/agent fees, contract amortization and other accounting costs. Market-value change is only a proxy.

### Biggest integration gap for the compatibility hypothesis: club mapping and club style

The transfer master already contains club IDs and rolling club form, but the detailed seasonal player table has not been joined to a reviewed squad-season club dimension. We still need:

- a reviewed FBref squad + competition + season to Transfermarkt club-ID crosswalk;
- historical aliases and renamed clubs;
- player and destination team features on the same scale;
- explicit compatibility calculations defined before model results are examined; and
- broader tactical/style variables than win rate and goals alone.

Without this work, we can test “club context,” but not the strongest version of “player-club tactical fit.”

### Biggest sample-coverage gap: the detailed player panel is narrow

The FBref-style data covers the Big Five leagues from 2017-18 through 2023-24. It is structurally strong inside that universe but does not cover players arriving from most other leagues. This is why only 1,294 of the 17,060 legacy candidates have a strictly fresh immediate prior season.

The dominant problem is source-universe mismatch, not mass deletion during cleaning.

## 8. Data to look for now

### Priority 1 — movement type and fee classification

Acquire or construct a dated transfer ledger keyed to the existing natural transfer key with:

- `player_id`;
- transfer date;
- origin and destination club IDs;
- permanent / loan / loan return / free / youth / other type;
- reported fee and currency;
- whether the fee is guaranteed, an add-on, a loan fee or an option/obligation;
- contract/loan end date when applicable;
- source URL, acquisition date and confidence; and
- manual-review status.

Begin with the 2,584 positive-fee rows and especially the 1,788 financial-eligible candidates. This produces more immediate value than collecting another large event table.

### Priority 2 — historical team-season style data and club crosswalk

Seek team-season data for at least 2017-18 through 2023-24, preferably extending through 2025-26 and beyond the Big Five. Useful fields include:

- possession;
- passing volume, length and directness;
- progressive passes/carries;
- pressing or defensive-action proxies;
- shots and expected goals for/against;
- field tilt or territorial dominance;
- crossing frequency;
- transition speed;
- formation and role usage;
- squad age and stability; and
- league/competition strength.

The source must include stable team-season identity fields so it can be mapped reproducibly to Transfermarkt club IDs.

### Priority 3 — broader and newer player-season coverage

Look for:

- 2024-25 and 2025-26 player seasons;
- leagues that commonly feed transfers into the Big Five;
- the same definitions across seasons and competitions;
- minutes and role/position context; and
- clear provenance and reuse terms.

Broader coverage will increase the fresh pre-transfer sample much more than resolving only the remaining name aliases.

### Priority 4 — availability and contract economics

For stronger causal interpretation and risk scoring, seek dated:

- injury episodes and days unavailable;
- matchday availability/suspensions;
- contract start and expiry as known at the transfer date;
- wages or wage bands;
- agent/signing fees where legally and practically available; and
- subsequent exit transfer and realized fee.

These are not required for the first sporting baseline, but they are important for a credible financial or risk model.

### Lower priority right now — more granular event data

The existing StatsBomb layer is already sufficiently rich for case studies. Its overlap is too sparse to solve the main sample problem. More event detail should be pursued only if it adds broad, consistent league-season coverage or directly supports a predeclared compatibility feature.

## 9. Required work before final modeling

Recommended order:

1. Freeze separate sporting and financial target definitions.
2. Add explicit horizon observability to the final eligibility rules.
3. Rebuild the player crosswalk from the cleaned FBref table and resolve high-value attention events.
4. Build the reviewed squad-season-to-club crosswalk.
5. Rebuild fresh player-season features using only the immediately prior completed season.
6. Define compatibility features before inspecting model results.
7. Produce a leakage-safe model matrix and mark every field as feature, target, diagnostic or prohibited snapshot.
8. Assign chronological train/validation/test folds.
9. Train the Transfermarkt sporting baseline first, followed by the club-context extension on the identical rows and folds.
10. Keep the financial model provisional until movement type is resolved.

## 10. Recommended modeling architecture

Do not force all objectives into the smallest intersection.

1. **Sporting model:** train on the larger horizon-safe sporting cohort.
2. **Compatibility test:** compare the sporting baseline against additional destination-context/fit features on exactly the same eligible sample and chronological folds.
3. **Financial proxy model:** train separately on classified paid transfers with reliable valuations.
4. **Decision layer:** combine calibrated sporting, financial, compatibility, risk and confidence outputs.

This structure preserves far more usable data and keeps the project’s claims honest.

## 11. Bottom line

MoveMaker is not “missing all the data.” It has a strong cleaned transfer and match-history foundation and enough rows for meaningful sporting models.

The project is currently limited by **semantic and integration gaps**, not raw file volume:

- movement type and true financial outcomes are missing;
- club style and reviewed club identity mapping are unfinished;
- detailed player coverage is narrow outside the Big Five; and
- the final success labels and model matrix are not frozen.

The most defensible immediate milestone is a **4,856-row Transfermarkt sporting baseline**, followed by a **3,134-row club-context comparison**. In parallel, the team should classify transfer movement types and complete the club/player crosswalks. That sequence creates a credible model soon without pretending the financial and tactical data are more complete than they are.

## Audit sources and reproducibility

Primary local evidence:

- `Data/processed/transfermarkt_merged/transfermarkt_comprehensive_master.csv`
- `Data/processed/transfermarkt_merged/transfermarkt_comprehensive_summary.json`
- `Data/processed/transfermarkt_clean/transfermarkt_table_summary.csv`
- `Data/processed/transfermarkt_clean/transfer_fee_summary.csv`
- `Data/processed/fbref_2017_2024_clean/fbref_player_squad_season_clean.csv`
- `Data/processed/fbref_2017_2024_clean/fbref_cleaning_summary.json`
- `Data/processed/player_crosswalk.csv`
- `Data/processed/transfer_cohort_master.csv`
- `Data/processed/statsbomb_case_studies/extraction_summary.json`
- `Data/processed/statsbomb_case_studies/transfer_case_study_candidates.csv`

Recomputed metrics are stored in `outputs/data_readiness/model_readiness_metrics.json` and generated by `scripts/audit_model_readiness.py`.

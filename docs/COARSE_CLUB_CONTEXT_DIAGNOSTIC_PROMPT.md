# MoveMaker — coarse club-context and compatibility diagnostic (large-n signal check)

Read `CODEX.md` section 0A first, then:

- `docs/DATA_READINESS_AUDIT.md`, especially **Transfermarkt sporting, 24 months, both club contexts**;
- `Data/processed/transfermarkt_merged/COMPREHENSIVE_MASTER_README.md`;
- `Data/processed/transfermarkt_merged/transfermarkt_comprehensive_field_dictionary.csv`;
- `scripts/audit_model_readiness.py`; and
- `Data/processed/rolling_origin_stability/README.md` plus the ridge/preprocessing/bootstrap functions used by `scripts/run_rolling_origin_stability.py`.

## Purpose and interpretation boundary

The current rich compatibility pipeline tests tactical/style compatibility on a small strict cohort. This task asks a related, coarser question on the larger Transfermarkt-only cohort:

1. Does pre-transfer destination-club context add out-of-sample predictive value beyond player history?
2. Do a small number of predeclared player-by-context and origin-to-destination fit terms add value beyond club-context main effects?

These are separate claims. Additive club-context improvement alone is evidence that environment matters; it is **not** evidence of player-club compatibility. The second comparison is the compatibility-shaped diagnostic.

This task cannot prove or disprove the richer tactical-fit hypothesis. A positive coarse-fit result raises confidence that a compatibility-shaped signal exists. A null result only says that these coarse Transfermarkt proxies do not add stable predictive value; it does not rule out compatibility measured with richer tactical data.

## Scope limits

This is a bounded, single-purpose diagnostic. Do not extend it into a second full pipeline.

- Do not modify, retrain, or rerun anything under `Data/processed/compatibility_*`, `Data/processed/rolling_origin_stability/`, `Data/processed/subgroup_stability/`, or `Data/processed/player_club_case_studies/`.
- Create only a new diagnostic script, verifier, and the outputs declared below.
- Ridge regression only. Do not fit elastic net, trees, neural networks, or other model families.
- Use the four fixed nested feature sets below. Do not run feature-set searches, automated feature selection, interaction mining, or alternate-target searches.
- Do not generate an `.xlsx` workbook, plots, or PNG previews.
- Do not alter any source or existing processed file.
- Stop and report when the declared outputs and independent verification are complete. Do not begin a follow-on phase.

## Step 1 — Reconstruct the audited cohort exactly

Use:

`Data/processed/transfermarkt_merged/transfermarkt_comprehensive_master.csv`

Reconstruct the cohort using the exact logic in `scripts/audit_model_readiness.py`:

```text
eligible_sporting_24m_provisional
AND sporting_post24_horizon_observable
AND club_origin_pre365_games >= 10
AND club_destination_pre365_games >= 10
```

Blocking expectations:

- exactly **3,134 transfer rows**;
- exactly **1,072 unique players**;
- transfer dates from **2012-12-31 through 2024-07-05**.

If any expectation fails, stop and report the discrepancy. Do not silently redefine the cohort.

Derive `season_start_year` from `transfer_date` using the same July boundary used by the data-readiness audit: the calendar year for July–December and the preceding year for January–June. Do not trust the source-supplied transfer-season string as the sole time-split field. Report row and unique-player counts by derived season before modeling.

## Step 2 — Use one predeclared normalized opportunity target

Define the sole target as:

```text
target_destination_minutes_opportunity_share_24m =
    appearance_post24_destination_minutes
    / (90 * club_destination_post24_games)
```

Both inputs are post-transfer outcome/coverage fields and may be used **only** to construct this target. Neither may enter any predictor set.

This share is the closest available Transfermarkt-only analogue to the existing opportunity target: it measures the fraction of the original destination club's observed match-minute opportunities taken by the player during the 730 days after transfer. A subsequent departure from the original destination remains part of the outcome rather than being censored away.

Run these blocking target checks before fitting:

- `club_destination_post24_games > 0` for all 3,134 cohort rows;
- the target is nonmissing for all 3,134 rows;
- all target values lie in `[0, 1.05]`;
- zero minutes with a positive destination-club denominator is retained as a genuine zero outcome, not converted to missing;
- missing or nonpositive denominators are never treated as zero.

The current verified master is expected to satisfy all checks. If it does not, stop and report rather than falling back to raw minutes, games, starts, or another target.

For prediction, apply the same `[0, 1.05]` output bounds used for the existing opportunity target. Report unbounded and bounded metrics if the reused methodology already supports both; use bounded predictions for the declared comparison.

## Step 3 — Freeze an explicit feature manifest

Before fitting anything, write the raw and derived predictors for each feature set to `coarse_club_context_feature_manifest.csv`. The four feature sets must be nested and must score exactly the same rows within every origin.

Use an explicit allowlist based on the comprehensive field dictionary. Do not select features merely because they pass a name blacklist.

### M0 — `player_history_baseline`

Allow only information known by the transfer date:

- age at transfer, derived from `player_date_of_birth` and `transfer_date` when valid;
- `transfer_fee`, `transfer_fee_status`, and `market_value_in_eur`;
- numeric pre-transfer valuation summaries from the `pre365`, `pre730`, and `career_pre` windows;
- pre-transfer player appearance, lineup-selection, and event aggregates from the `pre365`, `pre730`, and `career_pre` windows;
- the historically observed `lineup_pre365_all_primary_position` (or the corresponding origin value when the all-club value is unavailable), mapped once into a frozen broad position group: goalkeeper, defender, midfielder, attacker, unknown; and
- pre-transfer coverage flags associated with those allowed histories.

Exclude raw first/last observation dates, raw entity IDs, names, URLs, source row numbers, current-club identifiers, high-cardinality competition identifiers/names, and all snapshot-labelled player attributes, including snapshot position.

### M1 — `plus_destination_context`

M0 plus destination-club `pre365` and `pre730` context main effects:

- games, wins, draws, losses, goals for, goals against, goal difference, and points;
- win percentage and points per game;
- goals for, goals against, and goal difference per game, derived with safe division;
- home/away game counts;
- competition count; and
- domestic-league, domestic-cup, and international-game counts and shares.

Do not include destination post-transfer fields or destination club identity/name. Do not include dominant competition ID/name. A coarse competition-type category may be used only if its missing/unseen-category behavior is explicitly frozen and fold-safe.

### M2 — `plus_origin_transition_context`

M1 plus the origin-club equivalents and these fixed destination-minus-origin transition terms, separately for `pre365` and `pre730`:

- points per game;
- win percentage;
- goals for per game;
- goals against per game;
- goal difference per game;
- domestic-league game share;
- domestic-cup game share; and
- international-game share.

The 3,134-row cohort already requires adequate pre-transfer context for both clubs. Do not drop rows to add origin context; handle remaining field-level missingness inside each training fold.

### M3 — `plus_coarse_fit`

M2 plus only the following predeclared fit interactions. Construct the player-rate inputs from `pre365` data, falling back to the corresponding `pre730` rate only when the `pre365` denominator is unavailable; add an indicator recording that fallback.

- prior start percentage × destination points per game;
- prior minutes per appearance × destination points per game;
- prior goals per 90 × destination goals for per game;
- prior assists per 90 × destination goals for per game;
- prior start percentage × destination-minus-origin points-per-game difference;
- prior minutes per appearance × destination-minus-origin points-per-game difference; and
- broad historical position-group indicators interacted with destination points per game, destination goals for per game, destination goals against per game, and destination-minus-origin points-per-game difference.

Create these terms for the `pre365` context only to keep the compatibility test small and predeclared. Do not generate pairwise interactions beyond this list.

The headline comparisons are:

1. M1 vs M0: incremental destination-context signal;
2. M2 vs M1: incremental origin/transition-context signal; and
3. M3 vs M2: incremental compatibility-shaped fit signal.

Also report M3 vs M0 as the total coarse contextual improvement, but do not substitute it for the M3-vs-M2 fit test.

## Step 4 — Enforce leakage-safe preprocessing

Reuse the preprocessing and ridge implementation from the existing compatibility-model scripts wherever shapes allow:

- fit numeric medians on the training fold only;
- add numeric missingness indicators based on training-fold behavior;
- learn categorical levels on the training fold only and map unseen levels to `__OTHER__`;
- fit standardization parameters on the training fold only;
- remove zero-variance encoded columns using the training fold only;
- select ridge alpha on the validation season only using the existing ridge alpha grid and MAE-first tie-breaking; and
- after alpha selection, refit preprocessing and ridge on train plus validation before scoring the evaluation season.

Do not use complete-case deletion, global imputation, target-informed imputation, zero-filling of genuinely missing measurements, or preprocessing fitted on validation/evaluation data.

Within every origin, block unless M0–M3 evaluate the exact same transfer-event rows in the same order.

## Step 5 — Use four expanding-window origins without discarding 2012–2017

Use these four origins:

1. train on every eligible season through 2018; validate 2019; evaluate 2020;
2. train through 2019; validate 2020; evaluate 2021;
3. train through 2020; validate 2021; evaluate 2022;
4. train through 2021; validate 2022; evaluate 2023.

Unlike the current rich-feature script's `FIRST_TRAIN_YEAR = 2018` filtering, the first training window here must begin with the earliest eligible derived season in the 3,134-row cohort. This is necessary for the intended large-n diagnostic. Reuse the existing methodology, but do not accidentally discard the 2012–2017 rows.

Report train, validation, and evaluation counts in both transfer rows and unique players for every origin. Rows from 2024 are not an undeclared fifth evaluation origin.

## Step 6 — Estimate uncertainty at the player level

Reuse the existing paired row-bootstrap results as a secondary comparability check, with 5,000 deterministic repetitions and stratification by evaluation origin.

Use a **paired player-cluster bootstrap** for the primary confidence interval because 3,134 transfers represent 1,072 players:

- for each pooled comparison, sample unique evaluation `player_id` clusters with replacement;
- include all scored transfer rows belonging to each sampled player across the included origins;
- preserve the pairing of actual, baseline prediction, and candidate prediction;
- calculate the MAE improvement for each draw; and
- use 5,000 deterministic draws for the 95% interval and probability of improvement.

For origin-specific results, cluster within that origin. For pooled, mature-three, and terminal-two results, cluster across all included evaluation rows by player so repeated transfers by the same player remain in the same resampled cluster.

Use the player-clustered interval and probability for the declared stability label. Keep the existing label rules:

- `stable_positive`: pooled improvement > 0, at least three of four origins improve, and bootstrap probability of improvement >= 0.90;
- `mixed_positive`: pooled improvement > 0 and at least half of origins improve;
- `unstable_negative`: pooled improvement <= 0 and at most one origin improves;
- otherwise `mixed`.

Report pooled-four, mature-three (evaluation years 2021–2023), and terminal-two (2022–2023) summaries for every headline comparison. Include pooled n, unique players, baseline and candidate MAE, absolute and relative MAE improvement, 95% player-clustered CI, probability candidate is better, origin win rate, origin-improvement standard deviation, worst/best origin, and stability label.

## Step 7 — Outputs

Write only to:

`Data/processed/coarse_club_context_diagnostic/`

Required files:

- `coarse_club_context_results.csv` — pooled-four, mature-three, and terminal-two summaries for the declared comparisons;
- `coarse_club_context_origin_results.csv` — origin-level row/unique-player counts, selected alpha, and metrics;
- `coarse_club_context_predictions.csv` — paired evaluation predictions needed for independent reconstruction;
- `coarse_club_context_feature_manifest.csv` — feature set, raw/derived feature, source fields, timing classification, and rationale;
- `README.md` — cohort, target, feature sets, methodology, results, and interpretation boundary;
- `independent_verification.csv`; and
- `independent_verification.json`.

The README must answer plainly:

1. Does destination context improve prediction over player history at this larger n?
2. Does origin/transition context add further value?
3. Do the fixed player-by-context terms improve over context main effects?
4. Are those effects stable across origins, mature-three, and terminal-two windows?
5. Does the evidence raise, lower, or leave unchanged confidence that a genuine compatibility-shaped signal exists?

Compare only the **direction and stability**, not the raw magnitude as if the targets were identical, with the existing rich-cohort ridge opportunity result: MAE improvement `0.044804`, 95% CI `[0.029543, 0.060442]`, pooled n `207`. The horizons, cohorts, and exact targets differ.

## Step 8 — Independent blocking verification

Create a separate verifier that reconstructs results from saved predictions rather than trusting values emitted by the training script. It must block unless all of the following pass:

- reconstructed cohort matches 3,134 rows and 1,072 players exactly;
- the declared normalized target is independently reconstructed and bounded;
- every evaluation row is identical across M0–M3 within an origin;
- the feature manifest is a true allowlist and every modeled column is represented in it;
- no `post12`, `post24`, `target_`, `eligible_`, `current_`, `latest_`, snapshot-labelled, identifier, name, URL, or raw date field entered a predictor matrix;
- all raw source fields and derived predictor inputs are known strictly before the transfer date;
- all imputers, categorical levels, scalers, variance filters, and alpha choices were fit without evaluation data;
- no row was removed because a candidate feature was missing;
- only ridge regression was fit;
- row-bootstrap and player-clustered bootstrap use paired errors and deterministic seeds;
- pooled, mature-three, terminal-two, and origin-level metrics reconstruct from saved predictions;
- the comprehensive-master hash matches the recorded upstream hash before and after the run; and
- no protected existing output directory or file changed.

If any blocking check fails, mark verification as failed and report the failure rather than presenting a substantive conclusion.

## Final response after execution

When finished, report only:

- the actual cohort and pooled evaluation sizes in rows and unique players;
- M1-vs-M0 pooled effect, player-clustered 95% CI, and stability label;
- M3-vs-M2 pooled and mature-three effects, player-clustered 95% CIs, and stability labels; and
- one careful sentence stating whether this raises, lowers, or leaves unchanged confidence that a compatibility-shaped signal exists.

Stop there.

# MoveMaker — opportunity transferability and context decomposition

Read these files before doing anything:

- `docs/journal/2026-08-11_scope_pivot_journal.md`;
- `docs/COARSE_CLUB_CONTEXT_DIAGNOSTIC_PROMPT.md`;
- `Data/processed/coarse_club_context_diagnostic/README.md`;
- `Data/processed/coarse_club_context_diagnostic/independent_verification.json`;
- `scripts/run_coarse_club_context_diagnostic.py` and `scripts/verify_coarse_club_context_diagnostic.py`;
- `scripts/audit_model_readiness.py`; and
- `Data/processed/transfermarkt_merged/transfermarkt_comprehensive_field_dictionary.csv`.

## Purpose

The large-n diagnostic established that destination-club context provides a small, stable improvement when forecasting post-transfer opportunity, while fixed player-by-context compatibility terms do not generalize. It did not determine where the baseline predictability comes from.

This task asks:

> How much of a player's post-transfer opportunity is explained by market/profile information, how much is explained by the transferability of prior playing opportunity, and how much additional information comes from player performance, destination context, and origin-to-destination transition context?

The primary hypothesis is not that compatibility exists. The primary hypothesis is:

> Recent playing opportunity is partially transferable between clubs, and destination/transition context may provide a smaller adjustment after controlling for that prior status.

This analysis must distinguish persistence from context. It must not create or label any output as a compatibility score.

## Strict scope

This is the first bounded analysis in the revised project scope.

- One primary target: 24-month destination opportunity share.
- One model family for inferential comparisons: ridge regression, plus a mean-only benchmark.
- Six fixed nested model specifications declared below.
- No feature search, interaction search, alternate target, classification model, 12-month robustness model, tree model, subgroup search, or visualization in this task.
- Do not modify or rerun the existing compatibility pipeline or the completed coarse club-context diagnostic.
- Do not produce an `.xlsx`, plot, PNG, dashboard, scope memo, or follow-on analysis.
- Create only the two new scripts and declared outputs below.
- If a blocking data or verification check fails, stop and report it rather than changing the cohort, target, feature blocks, or model design.

## Step 1 — Reconstruct the same audited cohort and destination target

Use:

`Data/processed/transfermarkt_merged/transfermarkt_comprehensive_master.csv`

Apply exactly:

```text
eligible_sporting_24m_provisional
AND sporting_post24_horizon_observable
AND club_origin_pre365_games >= 10
AND club_destination_pre365_games >= 10
```

Blocking expectations:

- 3,134 transfer rows;
- 1,072 unique players;
- transfer dates from 2012-12-31 through 2024-07-05; and
- the comprehensive-master SHA-256 matches its verified summary.

Derive `season_start_year` from `transfer_date`: use the calendar year for July–December and the preceding year for January–June.

Use the identical primary target from the completed coarse diagnostic:

```text
destination_opportunity_share_post24 =
    appearance_post24_destination_minutes
    / (90 * club_destination_post24_games)
```

Blocking target expectations:

- positive denominator and nonmissing target for all 3,134 rows;
- all values in `[0, 1.05]`;
- genuine zero-minute outcomes retained as zero; and
- exact row-level agreement with the target saved in the completed coarse diagnostic wherever its evaluation predictions provide the same transfer event.

Do not substitute raw minutes or another outcome.

## Step 2 — Construct and audit comparable prior-opportunity measures

Construct these eight pre-transfer usage measures:

```text
origin_opportunity_share_pre365 =
    appearance_pre365_origin_minutes / (90 * club_origin_pre365_games)

origin_opportunity_share_pre730 =
    appearance_pre730_origin_minutes / (90 * club_origin_pre730_games)

origin_appearance_rate_pre365 =
    appearance_pre365_origin_games / club_origin_pre365_games

origin_appearance_rate_pre730 =
    appearance_pre730_origin_games / club_origin_pre730_games

origin_start_rate_pre365 =
    lineup_pre365_origin_starts / club_origin_pre365_games

origin_start_rate_pre730 =
    lineup_pre730_origin_starts / club_origin_pre730_games

origin_substitute_selection_rate_pre365 =
    lineup_pre365_origin_substitute_selections / club_origin_pre365_games

origin_substitute_selection_rate_pre730 =
    lineup_pre730_origin_substitute_selections / club_origin_pre730_games
```

Use safe division: missing or nonpositive denominators produce missing, never zero. Do not cap the observed measures.

Blocking expectations for the current verified cohort:

- every denominator is positive;
- all eight measures are nonmissing for all 3,134 rows;
- all values fall in `[0, 1.05]`;
- `origin_opportunity_share_pre365` has 367 genuine zeros and maximum `1.0277777777777777`; and
- `origin_opportunity_share_pre730` has 192 genuine zeros and maximum `1.0044444444444445`.

Write a pre-model audit section in the README with distributions and full-cohort descriptive correlations between prior opportunity and the destination target. These full-cohort correlations are descriptive only and must not be used for feature/model selection.

## Step 3 — Freeze six nested models

Write `opportunity_transferability_feature_manifest.csv` before fitting. It must list feature set, order, raw/derived status, complete raw source lineage, timing classification, and rationale.

All models must evaluate exactly the same rows within every chronological origin. Missing predictors are handled inside the fold; no complete-case deletion is allowed.

### M0 — `mean_only`

No predictors. At each origin, predict the training mean for validation; after the design is fixed, predict the train-plus-validation mean for evaluation.

Purpose: establish the no-information benchmark.

### M1 — `market_profile`

M0 plus exactly these five transfer-time/profile variables:

1. `age_at_transfer`, derived from `player_date_of_birth` and `transfer_date`, retaining ages 15–45 and treating other values as missing;
2. `historical_position_group`, derived from `lineup_pre365_all_primary_position`, falling back to `lineup_pre365_origin_primary_position`, mapped once to goalkeeper/defender/midfielder/attacker/unknown;
3. `transfer_fee`;
4. `transfer_fee_status`; and
5. `market_value_in_eur`.

Do not add valuation histories, player statistics, club identity, competition identity, current snapshot position, preferred foot, height, nationality, or other profile fields.

Purpose: quantify what broad market status and basic profile explain before sporting history enters.

### M2 — `plus_prior_opportunity`

M1 plus exactly the eight normalized usage measures declared in Step 2.

Do not also add their raw minutes, games, starts, substitute counts, or club-game denominators as predictors. The normalized variables are the frozen representation of prior opportunity.

Purpose and primary comparison: M2 vs M1 tests whether prior playing opportunity is transferable beyond market/profile information.

### M3 — `plus_player_performance`

M2 plus exactly these 16 pre-transfer performance rates:

For each window/context below:

```text
pre365_all
pre365_origin
pre730_all
pre730_origin
```

include:

- appearance goals per 90;
- appearance assists per 90;
- yellow cards per 90; and
- red cards per 90.

Use the existing goals/assists-per-90 columns. Derive card rates as cards divided by appearance minutes and multiplied by 90. If minutes are zero or unavailable, leave the derived rate missing and let the fold-fitted preprocessor handle it. Never zero-fill an unavailable rate.

Do not add event-ledger counts, lineup fields, valuation history, or other performance variables.

Purpose: M3 vs M2 tests whether the coarse Transfermarkt performance available here adds predictive value beyond prior selection and usage. Interpret this block cautiously because it is much richer for attacking output than defensive contribution.

### M4 — `plus_destination_context`

M3 plus the frozen destination-club `pre365` and `pre730` main-effect block used in the completed coarse diagnostic:

- games, wins, draws, losses, goals for, goals against, goal difference, and points;
- win percentage and points per game;
- safely derived goals for, goals against, and goal difference per game;
- home/away games and competition count; and
- domestic-league, domestic-cup, and international-game counts and safely derived shares.

Do not include destination identity/name, dominant competition identity/name, or any post-transfer club field.

Purpose: M4 vs M3 tests whether destination context improves forecasts after controlling for market profile, prior opportunity, and player performance.

### M5 — `plus_origin_transition_context`

M4 plus the corresponding origin-club `pre365` and `pre730` main effects and the same fixed destination-minus-origin transition terms used in the coarse diagnostic:

- points per game;
- win percentage;
- goals for per game;
- goals against per game;
- goal difference per game;
- domestic-league game share;
- domestic-cup game share; and
- international-game share.

Do not add player-by-context interactions. They were already tested and are outside this task.

Purpose: M5 vs M4 tests whether origin and transition context add information beyond destination context alone.

Expected raw/derived predictor counts are:

```text
M0 = 0
M1 = 5
M2 = 13
M3 = 29
M4 = 73
M5 = 133
```

Block if the manifest does not match these counts or the feature sets are not perfectly nested.

## Step 4 — Leakage-safe chronological modeling

Reuse the cohort, preprocessing, ridge, bounded prediction, rolling-origin, and deterministic seed logic from the completed coarse diagnostic where applicable.

Use the same four origins:

1. train on every eligible season through 2018; validate 2019; evaluate 2020;
2. train through 2019; validate 2020; evaluate 2021;
3. train through 2020; validate 2021; evaluate 2022;
4. train through 2021; validate 2022; evaluate 2023.

The first training window begins with the earliest eligible season, not 2018.

For M1–M5:

- numeric medians, numeric missingness indicators, category levels, scaling, and zero-variance removal are learned on training data only;
- unseen validation/evaluation categories map to `__OTHER__`;
- choose ridge alpha from the existing seven-value ridge grid using validation MAE, then RMSE, then deterministic hyperparameter serialization;
- refit the preprocessor and selected ridge on train plus validation before evaluating the next season; and
- bound predictions to `[0, 1.05]`.

For M0, use the corresponding training or train-plus-validation mean without pretending an alpha was selected.

Do not use global preprocessing, target-informed imputation, evaluation data in tuning, feature selection, complete-case deletion, or a different row set for any model.

## Step 5 — Declared comparisons and inference

The primary nested comparisons are:

1. M1 vs M0 — market/profile information;
2. M2 vs M1 — prior-opportunity transferability;
3. M3 vs M2 — incremental coarse player performance;
4. M4 vs M3 — incremental destination context; and
5. M5 vs M4 — incremental origin/transition context.

Also report M5 vs M0 as total improvement, but do not use it to replace the block-specific comparisons.

For each comparison, report:

- pooled-four evaluation years 2020–2023;
- mature-three years 2021–2023; and
- terminal-two years 2022–2023.

Use paired MAE improvement as primary. Include baseline/candidate MAE, relative improvement, origin win rate, origin-improvement standard deviation, and worst/best origin.

Use 5,000 deterministic paired player-cluster bootstrap draws for the primary 95% interval and probability of improvement. Cluster all evaluation rows belonging to the same `player_id` together across included origins. Retain the origin-stratified paired row bootstrap only as a secondary comparability result.

Use the same stability labels as the completed coarse diagnostic. Do not describe a result as meaningful merely because a point estimate is positive; interpretation must account for interval width, origin consistency, and mature-window behavior.

## Step 6 — Persistence diagnostics

Create `opportunity_persistence_summary.csv` using evaluation rows only. For pooled-four, mature-three, terminal-two, and each individual evaluation origin, report:

- row count and unique players;
- Pearson and Spearman correlation between `origin_opportunity_share_pre365` and actual destination opportunity;
- MAE from naively using the origin share itself as the destination prediction, bounded to `[0, 1.05]`; and
- mean origin and destination shares.

These are descriptive diagnostics. They do not replace the M2-vs-M1 test because M2 controls for market/profile information and tunes only on prior seasons.

## Step 7 — Outputs

Create:

- `scripts/run_opportunity_transferability_decomposition.py`; and
- `scripts/verify_opportunity_transferability_decomposition.py`.

Write outputs only to:

`Data/processed/opportunity_transferability_decomposition/`

Required files:

- `opportunity_transferability_results.csv`;
- `opportunity_transferability_origin_results.csv`;
- `opportunity_transferability_predictions.csv`;
- `opportunity_transferability_feature_manifest.csv`;
- `opportunity_persistence_summary.csv`;
- `README.md`;
- `independent_verification.csv`; and
- `independent_verification.json`.

The README must state plainly:

1. how much market/profile information improves over the mean;
2. whether prior opportunity improves beyond market/profile information;
3. whether performance adds beyond prior opportunity;
4. whether destination context adds beyond complete player controls;
5. whether transition context adds beyond destination context;
6. whether each effect survives mature-three and terminal-two windows; and
7. whether the defensible project angle is opportunity persistence, contextual adjustment, both, or neither.

Do not call persistence compatibility. Do not call a positive contextual adjustment causal.

## Step 8 — Independent blocking verification

The independent verifier must reconstruct the cohort, destination target, all eight prior-opportunity measures, evaluation rows, origin metrics, paired comparisons, both bootstrap methods, persistence diagnostics, and stability labels from source/saved predictions.

It must block unless:

- cohort, date range, player count, source hash, target, and prior-opportunity expectations match exactly;
- M0–M5 use identical evaluation events within every origin;
- all six feature manifests match the declared counts and nesting;
- every modeled feature appears in the allowlist manifest with complete raw source lineage;
- no post-transfer, target, eligibility, snapshot, identity, name, URL, or raw-date field enters any predictor matrix;
- every predictor or derived input is available no later than the transfer date;
- preprocessing and hyperparameter selection are fold-fitted and evaluation-safe;
- only mean-only and ridge models were fit;
- no candidate-feature missingness removed a row;
- every saved metric and bootstrap result reconstructs from predictions;
- the comprehensive master remains byte-identical; and
- every existing directory under `Data/processed/compatibility_*`, `rolling_origin_stability`, `subgroup_stability`, `player_club_case_studies`, and `coarse_club_context_diagnostic` remains byte-identical before and after.

Save failed verification artifacts and stop without a substantive conclusion if any blocking check fails.

## Final response after execution

Report only:

- cohort and pooled evaluation sizes in rows and unique players;
- M2-vs-M1 pooled and mature-three MAE effects, player-clustered 95% intervals, and stability labels;
- M3-vs-M2, M4-vs-M3, and M5-vs-M4 pooled effects, player-clustered intervals, and stability labels;
- M5-vs-M0 total pooled MAE improvement;
- pooled evaluation correlation between prior 365-day origin opportunity and destination opportunity; and
- one sentence stating whether the evidence supports opportunity persistence, contextual adjustment, both, or neither as the revised project angle.

Stop there. Do not start the 12-month, classification, nonlinear-model, subgroup, visualization, or scope-memo phases.

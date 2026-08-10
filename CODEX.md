# MoveMaker project handoff for Codex

Last updated: 2026-08-09

Read this file before modifying data, rebuilding cohorts, or starting modeling. It records the project objective, source structure, completed work, verified findings, modeling decisions, unresolved risks, and recommended next steps.

## 0A. August 9 pipeline update — supersedes section 0 below

This section is newer than "0. Laptop handoff snapshot" (dated 2026-08-06) and overrides it and everything below when they conflict. It was written by Claude during an audit-only review session (no files were modified during that review) to keep this handoff accurate before the next Codex session. Verify the counts below against the referenced files rather than trusting them blindly.

### What was built since the 2026-08-06 snapshot

**A second Transfermarkt-style source was ingested and reconciled (canonical integration layer).** A supplemental datalake (`football-datasets-main/datalake/transfermarkt`, 11 CSVs, 5,605,055 source rows) was audited (`scripts/audit_football_datalake.py` → `Data/processed/football_datalake_audit/`), cleaned (`scripts/clean_football_datalake.py` → `Data/processed/football_datalake_clean/`), and merged against the existing Transfermarkt collection and FBref identities into a canonical identity/performance layer:

- `scripts/build_integration_crosswalks.py` → `Data/processed/integration_crosswalks/`: 170,991 canonical Transfermarkt player IDs; 5,735 of 5,967 FBref player identities auto-accepted; 46,490 canonical club IDs; 684 of 684 FBref club-season mappings accepted.
- `scripts/build_canonical_integration.py` → `Data/processed/canonical_integration/`: 190,592 canonical players; 46,490 canonical clubs; 1,408,834 canonical valuation rows; 1,106,278 canonical transfer-event rows. 14 blocking checks, 0 failures.
- `scripts/build_canonical_performance.py` → `Data/processed/canonical_performance/`: `canonical_player_team_season_performance.csv` (2,084,606 rows), `modeling_scope_2017_2024.csv` (905,269 rows, the base population for compatibility modeling below), `fbref_advanced_performance_features.csv` (17,790 rows). 12 blocking checks, 0 failures.

**The compatibility-modeling pipeline (the project's central hypothesis test) was built end-to-end**, in this order, each stage independently verified before the next started:

1. `scripts/build_compatibility_targets.py` → `Data/processed/compatibility_targets/`: 936 target rows across three targets — `target_destination_minutes_share` ("opportunity"), `target_role_performance_index` ("performance"), `target_performance_change_rolling3` ("adaptation"). Primary cohort requires June–October arrival, transfer/loan type, outfield role, strict compatibility features available; performance/adaptation additionally require 450+ destination minutes. 27/27 checks pass.
2. `scripts/build_compatibility_features.py` → `Data/processed/compatibility_features/`: lagged player and teammate-style compatibility features, strictly `t-1`/`t-2`/`t-3` only. 31/31 checks pass.
3. `scripts/train_compatibility_models.py` → `Data/processed/compatibility_models/`: ridge-only first pass, player-history baseline (81 features) vs strict compatibility (202 features), single train/2022-validation/2023-test chronological split. **Test n is only 37–58 rows per target** — treat single-split results as noisy, not definitive. 31/31 checks pass.
4. `scripts/train_compatibility_ablations.py` → `Data/processed/compatibility_model_ablations/`: same comparison across ridge, elastic net, and gradient-boosted trees, 5 feature-set variants each, family/feature-set selected on 2022 validation MAE. 24/24 checks pass.
5. `scripts/run_rolling_origin_stability.py` → `Data/processed/rolling_origin_stability/`: 4 expanding-window origins (train-through-2018/val-2019/eval-2020 through train-through-2021/val-2022/eval-2023), pooled test n 162–207 per target — much more trustworthy than the single-split numbers above. 25/25 checks pass.
6. `scripts/analyze_subgroup_stability.py` → `Data/processed/subgroup_stability/`: role/competition subgroup tests with FDR correction, run against **frozen candidates** chosen in step 3/4: opportunity = full_explicit_fit + elastic_net, performance = full_explicit_fit + gradient_boosted_trees. 11/11 checks pass.
7. `scripts/build_player_club_case_studies.py` → `Data/processed/player_club_case_studies/`: 12 deterministically selected historical cases (successes, warnings, trade-offs, and model failures) built from the frozen candidates' rolling-origin predictions. 12/12 checks pass.

### Honest reading of the compatibility results so far — do not skip this

The single train/val/test split (step 3 above) is too small to be definitive: baseline test R² is negative for opportunity and performance, and for adaptation the bootstrap probability that compatibility even helps is 0.48 (a coin flip). Do not quote step 3's numbers as a headline result.

The rolling-origin pooled comparison (step 5, n≈162–207) is more informative and shows a real pattern worth acting on: **ridge regression shows a CI-excluding-zero positive compatibility effect on all three targets** (opportunity, performance, adaptation), while elastic net and gradient-boosted trees are inconsistent — e.g. gradient-boosted trees is `unstable_negative` for opportunity even though it is `stable_positive` for performance. See `Data/processed/rolling_origin_stability/rolling_family_stability_summary.csv`.

**Open question, not yet resolved:** the frozen candidates carried into subgroup stability and case studies (step 6/7) were chosen in step 4 using a single small 2022-validation slice, where elastic net beat ridge on the opportunity target by a validation-MAE margin of only about 0.0036 — likely inside the noise floor at that sample size. Ridge was not re-checked against the later, larger rolling-origin evidence before being passed over. A two-phase audit prompt (model-family reassessment using existing rolling-origin evidence, then a cohort-filter sensitivity diagnostic relaxing the June–October/outfield/450-minute primary-cohort filters one at a time) is expected to run next. Its outputs, if produced, should land at `Data/processed/compatibility_model_ablations/model_family_reassessment.md` and `Data/processed/compatibility_targets/cohort_filter_sensitivity.csv`. Read those files first if they exist — they may already answer whether the frozen candidates should change.

### Git tracking status — read before assuming anything is backed up

This repository has exactly two commits: `de59132` ("Add cleaned FBref player-season dataset") and `09b6731` ("Add validated Transfermarkt transfer core"), both from before this canonical-integration and compatibility-modeling work started. **Everything described above, this file included, is currently untracked in the working tree** — nothing from the past week of work (cleaning, canonical integration, or the compatibility pipeline) is committed or pushed. Do not assume `git log`, GitHub, or any remote reflects the current state. A deliberate staged-commit plan (scripts and docs first; large `Data/processed/` and `outputs/` artifacts handled separately given GitHub's file-size limits, including the 92 MiB `transfermarkt_comprehensive_master.csv` already flagged below) is still pending and should be treated as real risk — there is currently no backup of this work besides the local disk.

## 0. Laptop handoff snapshot — authoritative latest state (2026-08-06, superseded above where it conflicts)

This section is the newest handoff and overrides older counts or recommendations elsewhere in this file when they conflict. The rest of the document preserves the detailed project history and rationale.

### First actions on the laptop

1. Open the repository root and read this entire `CODEX.md`.
2. Read `docs/DATA_READINESS_AUDIT.md`.
3. Read `Data/processed/transfermarkt_merged/COMPREHENSIVE_MASTER_README.md`.
4. Run `git status --short` before changing anything. Files copied manually from the Windows machine may be untracked even though they exist locally.
5. Confirm the files listed under "Files that must arrive on the laptop" below are present before rebuilding or modeling.
6. Do not restart cleaning or merging from scratch. The verified outputs already exist.

### Cross-platform path warning

The current project directory is named `Data/` with an uppercase `D`. macOS can be case-sensitive. Keep that spelling consistent during the laptop handoff; do not create a second lowercase `data/` directory.

Some older scripts, especially the provisional cohort workflow, may still contain lowercase `data` paths or older source assumptions. Inspect them before running. The newer cleaning and comprehensive-master scripts use the current `Data/` layout.

### Git state at handoff

- Repository: `https://github.com/Lioneln5/Capstone-MoveMaker.git`
- Branch: `main`
- Windows HEAD and `origin/main`: `09b6731` (`Add validated Transfermarkt transfer core`)
- Previous tracked commit: `de59132` (`Add cleaned FBref player-season dataset`)
- `transfer_core.csv` and its committed support artifacts are on GitHub.
- The combined cleaned FBref player-season CSV is on GitHub.
- Most comprehensive-master, audit, StatsBomb and supporting files are currently local/untracked and are being copied manually to the laptop.
- Copying files does not add them to Git. Inspect and stage deliberately later.
- The comprehensive master is **96,465,583 bytes (92.0 MiB)**, close to GitHub's single-file limit and above its large-file warning threshold. Keep it local unless Git LFS or another storage decision is made explicitly.

Do not assume that the laptop working tree exactly matches the Windows working tree merely because both use the same remote.

### Latest comprehensive Transfermarkt master

Primary file:

```text
Data/processed/transfermarkt_merged/transfermarkt_comprehensive_master.csv
```

Current verified state:

- 35,139 rows;
- 577 columns;
- one row per cleaned transfer event;
- 4,345 unique players;
- 92.0 MiB;
- SHA-256: `e8e9a778cedc2a1dfb6f063a6adbf3c12d52e25d944c0f4741e56db07db466c4`;
- all 22 build-time validation checks passed;
- all 13 independent checks passed across 60 deterministic sample transfers; and
- two consecutive canonical rebuilds produced the same hash.

The master was overwritten in place when the historical layers were added. No alternative historical-master CSV was created.

Player history is represented at three pre-transfer horizons:

- `pre365`: recent observations from days -365 through -1;
- `pre730`: medium-term observations from days -730 through -1; and
- `career_pre`: all recorded observations strictly before the transfer.

The 365-day rule is a recency/eligibility gate. It does **not** truncate the older 730-day or career history once the recency rule is met.

The three layers apply to valuations, appearances, lineups and player events. Club context currently has pre-365 and pre-730 windows.

Supporting files:

```text
Data/processed/transfermarkt_merged/transfer_valuation_features.csv
Data/processed/transfermarkt_merged/transfer_appearance_features.csv
Data/processed/transfermarkt_merged/transfer_lineup_features.csv
Data/processed/transfermarkt_merged/transfer_event_features.csv
Data/processed/transfermarkt_merged/transfer_club_context.csv
Data/processed/transfermarkt_merged/transfermarkt_comprehensive_field_dictionary.csv
Data/processed/transfermarkt_merged/transfermarkt_comprehensive_assumptions.csv
Data/processed/transfermarkt_merged/transfermarkt_comprehensive_checks.csv
Data/processed/transfermarkt_merged/transfermarkt_comprehensive_summary.json
Data/processed/transfermarkt_merged/transfermarkt_comprehensive_independent_verification.csv
Data/processed/transfermarkt_merged/transfermarkt_comprehensive_independent_verification.json
Data/processed/transfermarkt_merged/COMPREHENSIVE_MASTER_README.md
```

Builders/verifiers:

```text
scripts/build_transfermarkt_comprehensive_master.py
scripts/verify_transfermarkt_comprehensive_master.py
```

Use the `--reuse-existing` option only when the component feature CSVs have already been verified and the intention is to reassemble the master. A full rebuild regenerates the components from the cleaned Transfermarkt tables.

### Latest data-readiness audit

Primary report:

```text
docs/DATA_READINESS_AUDIT.md
```

Reproducible metrics:

```text
outputs/data_readiness/model_readiness_metrics.json
scripts/audit_model_readiness.py
```

The metrics script is deterministic and reads the current comprehensive master, legacy cohort and player crosswalk without altering them.

### Honest model-readiness conclusion

MoveMaker has enough data to begin sporting models. It does not yet have a final realized-ROI or unified overall-success dataset.

Current horizon-safe counts:

| Candidate cohort | Rows | Interpretation |
|---|---:|---|
| Transfermarkt sporting, 12 months | 5,391 | Recent player evidence plus destination coverage and observable horizon |
| Transfermarkt sporting, 24 months | 4,856 | Best current broad sporting baseline |
| Transfermarkt sporting, 24 months, both club contexts | 3,134 | Coarse club-context/compatibility experiment |
| Transfermarkt financial, 24 months | 1,788 | Provisional; movement type is unresolved |
| Transfermarkt sporting + financial, 24 months | 847 | Provisional intersection |
| Transfermarkt combined with both club contexts | 699 | Provisional and relatively small |
| Fresh immediate-prior-season FBref sporting, 24 months | 970 | Rich player-feature ceiling before the cleaned rebuild |
| Fresh FBref sporting with both club contexts | 730 | Closest current rich compatibility approximation |
| Fresh FBref sporting + provisional financial | 216 | Too small for one complex unified model |

Important eligibility correction:

- The current master flag `eligible_sporting_24m_provisional` is true for 5,274 rows.
- It does not itself require the complete 24-month horizon to be observable.
- Adding that requirement reduces the defensible count to 4,856.
- The final cohort/model builder must include horizon observability explicitly.

### What is genuinely modelable now

1. A Transfermarkt sporting baseline using the 4,856 horizon-safe rows.
2. A controlled club-context extension using the 3,134 rows with useful histories for both clubs.
3. A restrained rich-player experiment after rebuilding the FBref cohort; the current ceiling is roughly 970 rows, or 730 with both club contexts.
4. A provisional market-value-return model for pipeline development, clearly labeled as a value proxy rather than realized ROI.

Use chronological validation. Do not use a random split as the primary evaluation. Compare baseline and compatibility models on the exact same rows and folds.

### What is not yet final

- No single sporting-success or overall-success label has been frozen.
- No final leakage-safe feature matrix has been built.
- The legacy `Data/processed/transfer_cohort_master.csv` still comes from the old cohort builder, older seasonal input and old prior-season rule.
- The cleaned FBref table is not yet incorporated into a final rebuilt transfer cohort.
- The FBref player crosswalk needs reviewed aliases and a rebuild from the cleaned source.
- A reviewed FBref squad-season-to-Transfermarkt club crosswalk does not yet exist.
- Explicit player-to-destination compatibility features have not been frozen.
- Transfer movement type remains unresolved.
- Realized resale proceeds, wages, agent fees, signing costs and accounting amortization are unavailable.
- StatsBomb coverage is too selective to train the main compatibility model.

### Biggest gaps

The largest data gap for the stated ROI objective is transfer movement and financial semantics:

- 2,584 transfers have a positive reported fee;
- 20,064 contain an unclassified reported zero;
- 12,491 have a missing fee; and
- zero rows are currently confirmed as paid permanent transfers because the local data lacks reliable permanent/loan/free/return classification.

The largest integration gap for the compatibility hypothesis is the historical club mapping/style layer. Transfermarkt club form exists, but the detailed FBref player panel has not been linked to reviewed squad-season club identities and comparable team-style profiles.

The largest coverage gap is that the detailed FBref-style panel covers only the Big Five from 2017-18 through 2023-24. This is the main reason the rich fresh-season cohort is much smaller than the broad Transfermarkt cohort.

### Data acquisition priorities

Seek data in this order:

1. Transfer movement type and fee classification keyed to player/date/from/to club.
2. Historical team-season style data and stable club identifiers.
3. Broader and newer player-season coverage, including 2024-25 and 2025-26 and feeder leagues outside the Big Five.
4. Dated injuries, suspensions and availability.
5. Historical contract terms and wages/wage bands.
6. Subsequent exit transfers and realized fees.

Do not prioritize another narrow event dataset before resolving movement type and club identity. The local StatsBomb layer is already sufficient for case studies but has inadequate transfer overlap for the main training panel.

### Next recommended work sequence

1. Verify that every copied file arrived intact on the laptop.
2. Decide and formally document the sporting target(s), beginning with an interpretable 12- or 24-month outcome.
3. Add explicit horizon observability to the final eligibility rules.
4. Rebuild the player resolver from the cleaned FBref source and create reviewed alias overrides/review outputs.
5. Build the squad-season-to-club crosswalk.
6. Rebuild fresh player-season features using only the immediately preceding completed season ending 1-365 days before transfer.
7. Define compatibility features before examining results.
8. Build a leakage-safe model matrix with every field classified as predictor, target, diagnostic or prohibited.
9. Assign chronological folds and train the sporting baseline first.
10. Classify movement type before presenting a financial model as a final ROI analysis.

### Non-negotiable handoff rules

- Do not modify raw source files.
- Do not redo verified cleaning manually in Excel.
- Do not silently force uncertain player or club matches.
- Do not treat reported zero fees as free transfers.
- Do not call market-value change realized ROI.
- Do not use any `post12` or `post24` field as a predictor.
- Do not use current/latest snapshot attributes in historical backtests unless proven safe as of the transfer date.
- Do not replace missing player performance with zero.
- Do not delete position-dependent columns merely because they are sparse overall.
- Do not substitute a two- or three-year-old player season when the immediate prior season is missing.
- Do not upload the 92.0 MiB comprehensive master to ordinary Git without an explicit storage decision.
- Preserve row-level provenance, keys, checks and field dictionaries.
- Report counts before and after every model-cohort filter.

### Files that must arrive on the laptop

At minimum, copy and verify:

```text
CODEX.md
docs/DATA_READINESS_AUDIT.md
scripts/audit_model_readiness.py
scripts/build_transfermarkt_comprehensive_master.py
scripts/verify_transfermarkt_comprehensive_master.py
Data/processed/transfermarkt_merged/
Data/processed/transfermarkt_clean/
Data/processed/fbref_2017_2024_clean/
Data/processed/player_crosswalk.csv
Data/processed/player_season_features.csv
Data/processed/transfer_cohort_master.csv
outputs/data_readiness/model_readiness_metrics.json
```

Copy the raw Transfermarkt and seasonal source folders too if the laptop must reproduce full cleaning/builds rather than only continue modeling from processed files. Copy the 15 GB StatsBomb checkout only if case-study work will continue locally; its processed extract is sufficient for most immediate tasks.

### Instruction to the next Codex session

Treat the latest processed files and audit as the starting point. Inspect and verify before acting, then continue from the next requested task. Do not repeat completed cleaning, rebuild the master without a reason, or narrow the broad master by deleting sparse or position-dependent data.

## 1. Project objective

MoveMaker is a summer data analytics capstone investigating:

> Can we predict the return on investment and overall success of a professional football player transfer before the transfer occurs?

The main hypothesis is that transfer success depends on both traditional player characteristics and the player's compatibility with the destination club.

The intended comparison is:

1. A baseline model using player quality, age, value, fee, position, and recent performance.
2. A compatibility model containing the same baseline variables plus player-destination-club fit features.

Do not force the hypothesis to succeed. Test whether compatibility improves genuinely out-of-sample predictions using chronological validation.

The eventual application should accept a player, destination club, proposed fee, and possibly transfer season, then return predicted sporting success, financial return, compatibility, risks, confidence, and an interpretable recommendation.

## 2. Non-negotiable analytical rules

- Never modify raw source files in place.
- Every clean or modeled table must be reproducible from a script.
- Preserve natural keys, source filenames, and row-level provenance.
- Never force an uncertain player or club match.
- Store match method, confidence, evidence, and manual-review status.
- Use only information available before the transfer date as model features.
- Post-transfer appearances, minutes, goals, valuations, and destination performance are targets or diagnostics, never predictors.
- Use chronological train/validation/test splits. Do not use random splitting as the primary evaluation.
- Missing performance is not zero performance.
- Do not silently backfill an old player season as current form.
- Financial market-value change is a proxy, not realized accounting ROI.
- Treat transfers with missing/zero fees and unresolved loan/free/return status separately from paid permanent transfers.

## 3. Mac setup warning: directory capitalization

The Windows workspace currently contains `Data/` with an uppercase `D`.

Some older scripts use `ROOT / "data"`, while the new FBref cleaning script uses `ROOT / "Data"`. Windows is case-insensitive, but a case-sensitive macOS volume will treat these as different directories.

Before running scripts on the Mac:

1. Choose one directory spelling.
2. Use it consistently in every script.
3. Do not create both `Data/` and `data/`.

Recommended convention: rename the directory to lowercase `data/` and update the new cleaning script and this document's relative paths accordingly. Alternatively, update the older scripts to use `Data/`.

The Windows project folder was not initialized as a Git repository at the time of this handoff. Initialize Git on the Mac or copy it into the intended repository before making substantial changes. Consider Git LFS or external versioned storage for large data files.

Suggested Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pandas numpy
```

## 4. Source data inventory

The audit covered 19 CSV files, 7,227,164 total rows, and 618 physical column entries.

### Seasonal Big Five data

Location: `Data/2017-2024/`

| File | Rows | Columns |
|---|---:|---:|
| `cleaned_2017-18.csv` | 2,549 | 65 |
| `cleaned_2018-19.csv` | 2,503 | 65 |
| `cleaned_2019-20.csv` | 2,558 | 65 |
| `cleaned_2020-21.csv` | 2,634 | 65 |
| `cleaned_2021-22.csv` | 2,689 | 65 |
| `cleaned_2022-23.csv` | 2,687 | 65 |
| `cleaned_2023-24.csv` | 2,623 | 65 |

Total: 18,243 player-squad-competition-season rows.

The source natural key is:

```text
player + squad + comp + season
```

No duplicate natural-key rows were found.

Competitions present in every season:

- Premier League
- La Liga
- Serie A
- Bundesliga
- Ligue 1

Expected club coverage is complete: 98 club-seasons per year through 2022-23 and 96 in 2023-24 after Ligue 1 contracted to 18 teams.

The columns resemble historical FBref advanced-stat tables. However, an explicit original-source URL, extraction script, acquisition date, and license/provenance record were not found in the workspace. Do not state FBref/Opta provenance as certain until the user confirms it.

### Transfermarkt data

Location: `Data/TransferMarkt/`

| Table | Rows | Grain / key |
|---|---:|---|
| `appearances.csv` | 1,894,350 | One player appearance; `appearance_id` |
| `club_games.csv` | 177,916 | One club-game; `game_id + club_id` |
| `clubs.csv` | 796 | Current/latest club snapshot; `club_id` |
| `competitions.csv` | 65 | One competition; `competition_id` |
| `countries.csv` | 124 | One country; `country_id` |
| `game_events.csv` | 1,274,469 | One event; `game_event_id` |
| `game_lineups.csv` | 3,179,016 | One player lineup/substitute record; `game_lineups_id` |
| `games.csv` | 88,958 | One game; `game_id` |
| `national_teams.csv` | 124 | Current/latest national team snapshot; `national_team_id` |
| `player_valuations.csv` | 507,815 | One player-date valuation; `player_id + date` |
| `players.csv` | 50,149 | Current/latest player snapshot; `player_id` |
| `transfers.csv` | 35,139 | One movement; player/date/from/to natural key |

The Transfermarkt files align with the public `dcaribou/transfermarkt-datasets` structure, but source provenance should still be documented in the project.

## 5. Completed audit artifacts

Script:

```text
scripts/profile_data.py
```

Outputs:

```text
outputs/movemaker_data_audit/
```

Important files include:

- `MoveMaker_Data_Dictionary.xlsx`
- `column_dictionary.csv`
- `table_inventory.csv`
- `candidate_keys.csv`
- `seasonal_quality_checks.csv`
- `join_coverage.csv`
- `README.md`

Verified audit conclusions:

- All 19 CSV files parsed successfully.
- All proposed source keys were unique in the current extracts.
- All seven seasonal files use the same 65-column schema.
- Tested arithmetic identities passed across the seasonal records.
- `Avg Mins per Match` behaves as total season minutes, not average minutes.
- `Goals Scored` exactly duplicates `Goals`.
- `carries_prgc` exactly duplicates `Progressive Carries`.
- Several goalkeeper-only fields were zero-filled on outfield players.
- Nine 2017-18 fields were entirely zero/unavailable: dribble-tackle percentage plus eight goalkeeper fields.
- Current-snapshot player/club attributes are unsafe for historical backtesting unless timestamped.

## 6. Completed FBref-style source cleaning

Cleaning script:

```text
scripts/clean_fbref_seasonal.py
```

Run with:

```bash
python scripts/clean_fbref_seasonal.py
```

Main output:

```text
Data/processed/fbref_2017_2024_clean/fbref_player_squad_season_clean.csv
```

Supporting outputs:

```text
Data/processed/fbref_2017_2024_clean/by_season/
Data/processed/fbref_2017_2024_clean/fbref_cleaning_actions.csv
Data/processed/fbref_2017_2024_clean/fbref_cleaning_checks.csv
Data/processed/fbref_2017_2024_clean/fbref_cleaning_summary.json
Data/processed/fbref_2017_2024_clean/fbref_column_dictionary.csv
Data/processed/fbref_2017_2024_clean/fbref_season_summary.csv
Data/processed/fbref_2017_2024_clean/source_file_manifest.csv
Data/processed/fbref_2017_2024_clean/README.md
```

Exact source backups:

```text
Data/source_backups/fbref_2017_2024_original/
```

Cleaning result:

- 18,243 rows retained.
- 81 cleaned/documented columns.
- Zero clean natural-key duplicates.
- Zero source rows deleted.
- All seven source files backed up and verified with SHA-256 hashes.
- `Goals Scored` and `carries_prgc` removed after proving exact equality.
- `Avg Mins per Match` renamed `total_minutes`; values unchanged.
- `Times tackled during take-on` renamed `tackled_during_take_on_pct`; values unchanged.
- `Crosses Stopped` renamed `crosses_stopped_pct`; values unchanged.
- Forty-nine code-like nationality values expanded while preserving `nationality_raw`.
- Goalkeeper-only fields converted to missing for non-goalkeepers.
- All 2017-18 goalkeeper fields converted to missing because the source columns were entirely zero.
- 2017-18 dribble-tackle percentage converted to missing because the source column was entirely zero.
- Legitimate sporting zeros were retained.
- Normalized player/squad helpers, readable natural keys, SHA-256 row IDs, season dates, nineties, and availability flags were added.
- All 19 cleaning checks passed.
- Independent reconciliation found zero unintended metric changes.
- Rerunning the script produced the identical combined-output hash.

Do not repeat this cleaning manually in Excel. Modify the cleaning script, rerun it, and extend the action/check files if a rule changes.

## 6A. Completed Transfermarkt source cleaning

Cleaning script:

```text
scripts/clean_transfermarkt.py
```

Independent verifier:

```text
scripts/verify_transfermarkt_clean.py
```

Run with the project Python environment:

```bash
python scripts/clean_transfermarkt.py
python scripts/verify_transfermarkt_clean.py
```

Main cleaned table directory:

```text
Data/processed/transfermarkt_clean/tables/
```

Exact source backups:

```text
Data/source_backups/transfermarkt_original/
```

Important supporting outputs:

```text
Data/processed/transfermarkt_clean/source_file_manifest.csv
Data/processed/transfermarkt_clean/transfermarkt_table_summary.csv
Data/processed/transfermarkt_clean/transfermarkt_column_dictionary.csv
Data/processed/transfermarkt_clean/transfermarkt_cleaning_actions.csv
Data/processed/transfermarkt_clean/transfermarkt_cleaning_checks.csv
Data/processed/transfermarkt_clean/foreign_key_checks.csv
Data/processed/transfermarkt_clean/club_name_observations.csv
Data/processed/transfermarkt_clean/club_dimension_historical.csv
Data/processed/transfermarkt_clean/player_name_observations.csv
Data/processed/transfermarkt_clean/player_identity_union.csv
Data/processed/transfermarkt_clean/transfer_fee_summary.csv
Data/processed/transfermarkt_clean/README.md
outputs/transfermarkt_cleaning/MoveMaker_TransferMarkt_Cleaning_Audit.xlsx
outputs/transfermarkt_cleaning/independent_verification.csv
outputs/transfermarkt_cleaning/independent_verification.json
```

Verified cleaning result:

- All 12 Transfermarkt source tables were backed up and verified with SHA-256 hashes.
- All 7,208,921 Transfermarkt rows were retained; no rows were deleted or expanded.
- All source and clean natural keys remain complete and unique.
- All declared nonblank numeric and date values parse after accounting for source timestamp formatting.
- All 83 blocking cleaning checks passed.
- A separate verifier passed all 76 recomputed hash, row-count, key-preservation, provenance, sentinel, and union-coverage checks.
- Source fields remain in their original order, with `source_file` and `source_row_number` added for lineage.
- Display names are preserved; normalized matching helpers are added rather than replacing source identity text.
- Current/latest player, club, appearance-current-club, and national-team snapshot fields are retained but documented as historical leakage risks.
- Transfer dates are authoritative. There are 233 source transfer-season/date differences, preserved as warnings rather than overwritten labels.
- A July-boundary game-season comparison finds 1,940 differences. These are not automatically source errors because calendar-season leagues and tournaments need competition-aware season logic.
- The 17,575 `game_events.minute == -1` source sentinels are preserved in `minute_raw`; analytical `minute` is missing and `event_minute_available` is false.
- The cleaned historical club union contains 8,784 IDs, including 7,988 absent from the 796-row current/latest `clubs.csv` snapshot.
- The cleaned historical player union contains 123,095 IDs, including 72,946 absent from the 50,149-row current/latest `players.csv` snapshot.
- All appearance, lineup, event, and club-game records reconcile to the games spine on dates, participating clubs, competition where applicable, goals, opponents, and hosting.

Transfer fee states remain unresolved by design:

- 2,584 positive reported fees (7.35%).
- 20,064 reported zeros (57.10%), stored as `unclassified_zero`.
- 12,491 missing fees (35.55%).

Never relabel a reported zero as a free transfer without movement-type evidence. The source does not reliably distinguish free transfers, loans, loan returns, and undisclosed movements.

The historical union dimensions prevent legitimate fact rows from being deleted simply because the limited current `clubs.csv` or `players.csv` snapshot omits their IDs. They do not replace the reviewed FBref-player and squad-to-club crosswalks.

## 6B. Completed StatsBomb case-study extraction

Extractor:

```text
scripts/extract_statsbomb_case_studies.py
```

Workbook builder:

```text
scripts/build_statsbomb_case_study_workbook.mjs
```

Processed outputs:

```text
Data/processed/statsbomb_case_studies/
```

Audit workbook:

```text
outputs/statsbomb_case_studies/StatsBomb_Case_Study_Audit.xlsx
```

The immutable 15 GB StatsBomb checkout was reduced to a 194.49 MB case-study layer using this rule:

- current match-catalog entries only;
- male competitions only;
- match dates from 2017-07-01 through 2024-06-30, inclusive;
- modern matches labeled as `primary_big_five_club`, `secondary_other_club`, or `secondary_international`;
- uncataloged event/lineup files excluded; and
- StatsBomb-to-Transfermarkt links stored only as manual-review candidates.

Extraction result:

- 642 selected matches from 3,961 cataloged matches;
- 2,298,900 slim event rows in compressed CSV form;
- 27,702 player-match feature rows;
- 1,284 team-match feature rows;
- 6,192 player-team-competition-season feature rows;
- 309 team-competition-season feature rows;
- 963,072 compact StatsBomb 360 event-context rows; and
- 612 transfer case-study candidate rows.

All nine blocking extraction checks passed. Event IDs, player-match keys, team-match row counts, selected event totals, and pass totals reconcile. One repository-wide nonblocking validation warning remains: `data/three-sixty/3845506.json` contains invalid JSON/null-byte corruption. The 274 event/lineup files without current match metadata were documented and excluded.

The transfer overlap confirms this is an optional case-study layer, not the main compatibility-model source:

- 146 candidate transfers have at least one selected player match in the preceding 365 days;
- 32 have at least three;
- 11 have at least five;
- 4 have at least three matches both before and after the transfer; and
- zero have at least three preceding matches for both origin and destination clubs.

Recommended uses are player/club profiles, timelines, pitch maps, pass/shot/carry visualizations, and carefully reviewed transfer narratives. Do not treat the sample as a representative modern Big Five panel. Post-transfer shortlist fields are targets/diagnostics and must never be used as predictors.

## 7. Player identity audit and current crosswalk

Existing crosswalk:

```text
Data/processed/player_crosswalk.csv
```

Existing player-season output:

```text
Data/processed/player_season_features.csv
```

The first crosswalk used:

```text
normalized player name + birth year
```

It accepted a mapping only when exactly one Transfermarkt player matched.

Crosswalk identity counts:

- 5,483 unique accepted identities.
- 470 unmatched identities.
- 14 ambiguous identities.
- 17,043 of 18,243 seasonal rows linked uniquely (93.42%).
- Accepted identities are one-to-one: no accepted seasonal identity maps to multiple IDs, and no accepted ID is reused for multiple seasonal identities.

Direct Transfermarkt linkage checks:

- All 4,182 cohort player IDs exist in `players.csv`.
- All 34,309 Transfermarkt transfer rows associated with cohort players have exact normalized player-name agreement.
- All 222,146 relevant appearance rows have exact normalized player-name agreement.
- No incorrect accepted player identity was detected.

The problem is high-precision but incomplete matching, not evidence of false accepted matches.

### The 247 attention events

There are 247 transfer events where Transfermarkt confirms a prior-season Big Five appearance but the seasonal row did not link. These represent:

- 109 unique players.
- 145 unique player-season combinations.

Examples of clear aliases:

- Martín Aguirregabiria / Martin Agirregabiria
- Andreaw Gravillon / Andrew Gravillon
- Marios Ikonomou / Marios Oikonomou
- Dimitrios Siovas / Dimitris Siovas
- Mohamed Daramy / Mohammed Daramy
- Triantafyllos Pasalidis / Triantafyllos Pasaridīs

Those six aliases alone would populate or change preseason features for 27 transfer events.

Other causes include:

- missing accents;
- preferred versus full names;
- abbreviations or single-word names;
- source birth-year disagreement (for example, Dwight Gayle differs across sources);
- missing date of birth (for example, Shinji Okazaki in the local Transfermarkt player dimension);
- two distinct players sharing the same normalized name and birth year;
- very-low-minute players represented inconsistently.

Do not solve these by editing the master CSV in Excel. Build a reproducible resolver that outputs:

```text
Data/processed/player_alias_overrides.csv
Data/processed/player_match_review.csv
```

An automatic match should require multiple independent signals, such as name similarity, date/birth-year compatibility, nationality, season, club overlap, appearances, minutes, goals, and assists. Statistical fingerprint alone was tested and rejected as unsafe: exact stat fingerprints had only about 79% validation precision because low-minute players frequently share identical stat lines.

Store, at minimum:

- seasonal identity;
- Transfermarkt player ID;
- match status;
- match method;
- confidence tier;
- candidate count;
- name evidence;
- birth-date evidence;
- club/season evidence;
- statistical reconciliation evidence;
- manual reviewer/decision;
- notes.

## 8. Club identity work still required

The existing union club dimension is:

```text
Data/processed/club_dimension_union.csv
```

It contains 8,299 club/team IDs observed across club, transfer, game, and national-team sources.

The original audit found weak exact club-name matching because FBref-style names and Transfermarkt names differ. Examples include abbreviations, accents, sponsorship names, and historical names.

Before building compatibility features, create a reviewed squad-season-to-club mapping with:

- seasonal squad name;
- competition and season;
- Transfermarkt club ID;
- canonical name;
- alias used;
- match method and confidence;
- manual-review status.

Do not use the 796-row `clubs.csv` as a complete historical club master. It is a limited current/latest snapshot. Use the union of transfers, games, club games, and clubs.

## 9. Transfer cohort already built, but it is provisional

Builder:

```text
scripts/build_transfer_cohort.py
```

Existing master:

```text
Data/processed/transfer_cohort_master.csv
```

Audit outputs:

```text
outputs/transfer_cohort/
```

Existing cohort definition:

- Transfer dates: 2018-07-01 through 2024-02-27.
- Start date supplies a completed 2017-18 baseline season.
- End date supplies a full 24-month horizon through the final local valuation observation on 2026-02-27.
- Primary stored ID: `transfer_event_id`.
- Natural key: `player_id + transfer_date + from_club_id + to_club_id`.

Existing cohort size:

- 17,060 transfer rows.
- 4,182 unique players.
- 149 fields.
- 1,890 transfers with any prior completed seasonal features under the old rule.
- 1,326 sporting 12-month eligible.
- 1,314 sporting 24-month eligible.
- 1,058 provisional financial 12-month eligible.
- 957 provisional financial 24-month eligible.
- 244 provisional combined 24-month eligible.

Independent checks performed:

- All 17,060 cohort events match exactly one raw Transfermarkt transfer.
- No duplicate natural keys or event IDs.
- All player dimension fields match the raw player table.
- All attached valuation player/date/value triples exist exactly in the raw valuation table.
- No duplicate player-date valuation conflicts.
- No pre-transfer valuation occurs after the transfer.
- Independent valuation selection found zero errors.
- All 18 appearance-derived cells for every transfer were independently recalculated with zero discrepancies.

Important: the current builder still uses the original seasonal files and the old "latest available prior season" rule. Do not treat the existing cohort as final. Update it to consume the new cleaned seasonal dataset and the reviewed crosswalks before modeling.

## 10. Preseason freshness decision

The final rich-cohort rule should be:

> Use player performance only from the immediately preceding completed season, ending no more than 365 days before the transfer.

Implementation concept:

```python
preseason_freshness_days = (transfer_date - preseason_end_date).days

has_fresh_preseason = (
    preseason_season == immediately_previous_season
    and 1 <= preseason_freshness_days <= 365
)
```

If the immediate prior season is unavailable:

- keep the transfer in the broad master;
- leave rich preseason metrics missing;
- set `has_fresh_preseason = False`;
- record an exclusion reason;
- do not substitute a two- or three-year-old season.

Current diagnostic:

- 1,890 transfers have some prior seasonal attachment.
- 1,312 use the immediately preceding season.
- 578 use an older season and must be marked stale under the final rule.

After the 247 attention events are resolved, the theoretical fresh-sample ceiling is about 1,559 transfers before destination-club and outcome filters. The actual total will be lower after confidence and club-profile requirements.

## 11. Why broad coverage is low

The seasonal source covers only the Big Five domestic leagues. The Transfermarkt transfer log is much broader.

Exclusive prior-season coverage diagnosis across the 17,060 transfer rows:

| Situation | Transfers | Share |
|---|---:|---:|
| Immediate prior Big Five season linked | 1,312 | 7.69% |
| Missing link despite Transfermarkt Big Five appearance | 247 | 1.45% |
| Played only outside the Big Five in prior season | 2,875 | 16.85% |
| No prior-season appearance; older accepted stats exist | 369 | 2.16% |
| Accepted seasonal data exists only in the same/later season | 968 | 5.67% |
| No accepted seasonal identity and no tracked prior-season appearance | 11,289 | 66.17% |

This is primarily a source-universe mismatch, not proof that the Big Five seasonal panel is incomplete.

Among transfers where Transfermarkt confirms a prior-season Big Five league appearance, 1,281 of 1,528 linked to the immediate seasonal data under the original exact rule (83.84%).

## 12. Data availability conclusion

The local seasonal panel is structurally complete by Big Five club-season and broad enough to begin a rich sporting/compatibility cohort after entity resolution.

It is not an exhaustive archival copy of every football variable:

- some source labels were misleading;
- some missing values were encoded as zero;
- 2017-18 goalkeeper/defensive coverage has known gaps;
- injuries and availability are absent;
- wages and contract duration are absent;
- movement type is unresolved;
- realized resale proceeds and accounting profit are absent;
- explicit data provenance/licensing is not yet documented.

In January 2026, Sports Reference announced that its advanced soccer data provider terminated the feed and required deletion of advanced data. Do not assume the old advanced tables can simply be downloaded again. Preserve and hash the local source snapshots.

StatsBomb Open Data covers selected competitions and seasons, not the complete Big Five 2017-24 panel. Do not replace the current panel with a narrower event dataset unless the research scope deliberately changes.

## 13. Recommended modeling architecture

Do not force one model onto the 244-row strict intersection.

Recommended components:

1. Sporting-success model on the larger rich sporting cohort.
2. Financial/value-return model on paid permanent transfers with reliable fee and valuation timing.
3. Compatibility model comparing the baseline sporting model against additional player-club fit features on the exact same sample.
4. Decision layer combining calibrated sporting probability, financial/value return, compatibility, risk flags, and confidence.

Potential sporting targets:

- destination minutes/appearances in 12 or 24 months;
- destination minutes share when club-game coverage is available;
- starts/selection if lineups are used;
- goals/assists or position-adjusted contribution;
- retention/continued destination participation;
- a predeclared composite sporting-success score.

Financial target currently available:

- market-value change at fixed horizons;
- fee-relative market-value proxy.

Do not call the financial proxy realized ROI without resale proceeds, wages, agent fees, contract amortization, and other accounting inputs.

## 14. Recommended next work sequence

Proceed in this order:

1. Resolve the 145 attention player-season identities into an override/review crosswalk.
2. Rebuild the full player crosswalk from the cleaned seasonal table.
3. Build the seasonal squad-to-Transfermarkt club crosswalk.
4. Aggregate cleaned player-squad rows into leakage-safe player-season features.
5. Build pre-transfer club-season profiles for source and destination clubs.
6. Define compatibility features before examining model results.
7. Apply the immediately preceding season/365-day freshness rule.
8. Rebuild the transfer master from the cleaned data and reviewed crosswalks.
9. Produce an attrition funnel after every eligibility rule.
10. Resolve permanent/loan/free/return movement types before final financial modeling.
11. Define sporting and financial labels formally.
12. Train a simple baseline first, then the compatibility extension on the identical chronological folds.

## 15. Instructions for a new Codex session

When continuing this project:

1. Read this entire file.
2. Read `Data/processed/fbref_2017_2024_clean/README.md`.
3. Read `outputs/movemaker_data_audit/README.md` and `outputs/transfer_cohort/README.md`.
4. Inspect scripts before rerunning them; older cohort code uses the old seasonal source and freshness rule.
5. Check directory capitalization before executing anything on macOS.
6. Do not overwrite raw files or reviewed crosswalk decisions.
7. Report exact row counts before and after every filter or join.
8. Treat unexpected increases in row count as possible many-to-many join errors.
9. Independently reconcile a sample and all key aggregates after a rebuild.
10. Document every assumption in a data dictionary or decision log.

If an accepted mapping cannot be supported by independent evidence, leave it unresolved. High-precision incomplete data is preferable to a larger contaminated cohort.

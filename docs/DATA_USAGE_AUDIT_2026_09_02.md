# Local data coverage and actual-use audit — 2026-09-02

## Verdict

**MoveMaker has materially more data, and more previously engineered context,
than its current extension models use. Data acquisition is not the first step
for several of the proposed improvements.**

The strongest unused or underused local inputs are match-level manager and
formation records, lineup/role trajectories, historical club wage structure,
the larger ID-linked injury-spell source, and club-season standings. StatsBomb
is already present and substantially processed, but remains a sparse,
unreviewed-linkage case-study source relative to the current extension cohort.

This is an inventory/use/availability diagnostic, not a new model experiment.
No training, new predictions, holdout outcome rates, deployment, source repair,
or deletion was performed. Existing release restrictions remain in force.

## What was actually inspected

- **10,899 local files**, **25.43 GB logical size** (23.68 GiB), across `Data/`,
  the supplemental `football-datasets-main/` checkout, `outputs/`, `output/`,
  and the separate local `git/` metadata directory. This includes backups,
  repeated derived tables, reports, and metadata; it is NOT unique data volume.
- **1,171 CSV/CSV.gz tables**: headers and every row parsed for row counts.
- Full column completeness for primary cleaned tables, seasonal sources,
  canonical performance, salary/extensions, the extension master, and overlays.
- Every StatsBomb event, lineup, and 360 JSON payload parsed; individual payload
  hashes recorded. One pre-existing malformed 360 file explicitly recorded.
- Current build/research/runtime code and notebooks searched for references;
  source manifests read to provide lineage evidence.
- **All 16 frozen V1 artifacts** inspected for their actual feature lists,
  separately from current V2 candidate specifications.
- Availability joins against all **2,805 extension events** and separately the
  **2,240 events signed through 2023**. No endpoint outcome was used to select
  these availability denominators.
- **26/26 verification checks passed**, including an independent reconstruction
  of the pre-signing manager-game windows. All **24 frozen V1 files** retain
  their recorded SHA-256 hashes.

The file-by-file inventory and all 74 inventoried folder families are accounted
for in the evidence directory. Reports and source READMEs were not accepted as
substitutes for recounting the underlying data.

## Status definitions

1. **Engine input:** reaches a selected model feature or constructs a target.
2. **Context/infrastructure:** used for identification, display, joins, coverage,
   or business arithmetic, but not necessarily predictive modeling.
3. **Research-only:** processed or tested in earlier experiments, not selected
   for the current engine.
4. **Dormant substantive content:** stored/cleaned but not integrated into the
   current extension modeling path. A cleaner or identity crosswalk reading a
   table is not equivalent to using its substantive information.
5. **Intentionally withheld:** retrospective, unreviewed, unsuitable-scope, or
   insufficient-coverage data; non-use may be correct.
6. **Backup/evidence:** preserved copies and experiment results, not a new pool
   of unique observations waiting to be trained on.

"Runtime" below refers to the retained V1 implementation as historical code.
The API is unconditionally fail-closed, skips engine loading and has no
environment-variable override. V2 has research candidates, not
promoted/deployed artifacts. No remote service state was queried by this audit.

## 1. Primary source inventory and actual use

### A. Transfermarkt match/event collection — 7,208,921 raw rows

| Local table | Rows | What is used | What is not used by current V2 candidates |
| --- | ---: | --- | --- |
| `appearances.csv` | 1,894,350 | Exact pre/post player minutes and appearances; goals/assists in V1; seasonal basic performance | Short-window role trajectories, broader workload/context summaries |
| `game_lineups.csv` | 3,179,016 | Seasonal starts, bench selections, captaincy and positions in canonical performance/legacy research | Match-by-match role changes and positional squad competition |
| `game_events.csv` | 1,274,469 | Old comprehensive transfer master/event research | No direct current V2 event-feature input; current basic scoring histories use appearances |
| `games.csv` | 88,958 | Dates, club/competition IDs, club-game denominators; V1 lookup | Manager names and formations are present but not model inputs |
| `club_games.csv` | 177,916 | Historical transfer-club form/context research | Not a direct selected extension V2 feature source |
| `player_valuations.csv` | 507,815 | Canonical valuation history, pre-decision value and future value targets | Additional historical trajectory/context not in the compact candidates |
| `transfers.csv` | 35,139 | Canonical transfer union; old transfer experiments; extension departure outcomes | Club buying/selling/loan propensity is not a selected V2 predictor |
| `players.csv` | 50,149 | Canonical identities, date of birth, position; name resolution | Numerous profile/snapshot fields intentionally not historical features |
| `clubs.csv` | 796 | Club identities/aliases and historical context infrastructure | Snapshot fields are not historical squad/finance evidence |
| `competitions.csv` | 65 | Competition identity/type infrastructure | Not 65 independent model features |
| `countries.csv` | 124 | Cleaning/reference infrastructure | Not a selected predictor table |
| `national_teams.csv` | 124 | Reference/type/scope infrastructure | National-team dimension does not supply date-specific workload |

Source/cleaned versions are copies of the same observations, not additive
datasets. The tables have unequal temporal and competition coverage.

**Manager coverage:** 88,114 of 88,958 games have each home/away manager name
(99.1%). Home formation is present in 80,758 games (90.8%); away formation in
80,885 (90.9%). Source-wide percentages include matches outside the extension
cohort; the pre-signing cohort coverage below is more relevant.

### B. Supplemental Transfermarkt datalake — 5,605,055 raw rows

These files live outside `Data/`, under
`football-datasets-main/datalake/transfermarkt/`, with cleaned copies in
`Data/processed/football_datalake_clean/tables/`.

| Table | Rows | Actual disposition |
| --- | ---: | --- |
| `player_market_value` | 901,429 | Unioned into canonical valuation history |
| `transfer_history` | 1,101,440 | Unioned into canonical transfers; movement semantics and extension outcomes |
| `player_performances` | 1,878,719 | Incorporated into canonical player-team-competition-season performance |
| `player_profiles` | 92,671 | Stable identity evidence integrated; current contract/agent/club fields preserved separately as snapshots |
| `team_details` | 2,175 | Club identity and snapshot evidence |
| `player_latest_market_value` | 69,441 | Retained reference snapshot; canonical latest values derived from dated history instead |
| `player_injuries` | **143,195** | Cleaned and season-crosswalk/reference use; old notebook exploration. **Not integrated into current extension model features** |
| `player_teammates_played_with` | **1,257,342** | Cleaned directed teammate-edge table; **not current extension features** |
| `player_national_performances` | **92,701** | Cleaned national-team career summaries; outside canonical club performance and current models |
| `team_competitions_seasons` | **58,247** | Club/competition/season alias infrastructure; standings and manager metrics **not current extension features** |
| `team_children` | **7,695** | Club alias/crosswalk infrastructure; parent/reserve/youth hierarchy **not current extension context** |

Important qualifications:

- The teammate table is a **career-aggregate edge table without historical
  observation cutoffs**, not a date-stamped co-playing event log. Reported
  joint minutes exist on 453,189 of 1,257,342 edges (36.0%); joint goal
  participation on 222,165 (17.7%). Historical features require reconstruction
  from date-stamped records or independently dated snapshots.
- National-team summaries are career totals, not automatically valid prior-year
  international workload.
- Manager names exist in 47,686 club-season rows (81.9%); season rank in all
  58,247. Final-season results and a season-level manager entry cannot be used
  as though known at an earlier signing. Some season labels extend into the
  future; the full table is not a ready historical training matrix.
- Profile snapshots already contain contract expiry, options, last-extension
  dates, agents, loan-parent clubs, height, and preferred foot. For example,
  contract options are populated for 3,571 profiles and last-extension dates
  for 8,363. The missing piece is often **historical validity/completeness**,
  not total absence of those fields.

### C. FBref-style seasonal performance

| Layer | Actual coverage | Use |
| --- | --- | --- |
| Seven 2017–2024 files | **18,243 player-squad-season rows** | Cleaning, crosswalks, canonical performance, old compatibility models, V1 performance histories |
| Accepted advanced companion | **17,790 rows × 85 columns**, including metadata | Rich player performance stored; V1 M5 uses a small advanced block; current V2 selected candidates use none of it |
| Unresolved advanced rows | **453 rows** | Explicit review queue, not accepted model joins |
| 2024–25 full export | **2,854 rows × 267 columns** | Feeds live overlay builder, which selects only a subset |
| 2025–26 full export | **2,839 rows × 102 columns** | Feeds live overlay builder; schema/coverage differs from previous year |
| Matching light exports | **2,854 and 2,839 rows** | Stored alternative exports, not consumed by that builder; do not add to full-export counts as new players |
| Live performance overlay | **5,018 linked rows**, 2,618 advanced / 2,400 basic | Retained V1 retrieval path only; not merged into frozen historical extension experiments |

Advanced data includes passing, progression, shooting, creation, defensive,
aerial, dribbling, and goalkeeper fields. "Advanced statistics were not
retained" does not mean they were never collected or never researched.

### D. Capology salary and extension data

| Layer | Actual contents | Use |
| --- | --- | --- |
| Raw salary pages | **50 files; 28,877 rows**, five leagues, 2017–18 through 2026–27 | Historical base plus newer live overlay |
| Historical salary panel | **35 files; 19,770 rows**, through 2023–24 | Prior-player wage, club salary context, V1 retrieval, V2 wage benchmark |
| Newer salary overlay | **15 files; 9,107 raw → 9,103 canonical rows**, 2024–25 through 2026–27 | Live-only salary refresh |
| Raw extension pages | **44 files; 3,235 rows**, 2018–19 through 2026–27 | Not all ingested |
| Historical extension events/master | **34 source files; 2,811 raw → 2,805 deduplicated events**, through 2024–25 pages | Current V2 development spine and historical endpoint construction |
| Later extension pages outside base | **10 files; 424 raw rows** (366 for 2025–26; 58 for 2026–27) | No base or live-overlay extension ingestion found; not deduplicated/validated/mature training additions |
| Raw club payroll pages | **55 files; 1,074 club-seasons**, five leagues, 2014–15 through 2024–25 | Feeds the verified lagged club-financial-context builder |
| Raw transfer-window pages | **55 files; 1,074 club-seasons**, same league-season coverage | Feeds the same builder; contains income, expense, balance and squad context |
| Canonical club financial context | **1,074 club-seasons; 164 clubs; 16 reviewed aliases; 0 unresolved rows** | Decision-time lagged feature source; 87.3% coverage across all 2,805 extension events |

The Premier League 2022–23 extension page is absent from this local historical
collection; all five 2017–18 extension pages are also absent. The later-added
payroll and transfer-window files have a separate canonical builder,
independent audit, and rolling-origin experiment. They were not silently added
to the legacy salary/extension build.

Existing `salary_context()` creates club known payroll, median/mean, salary
percentile, and wage share. **V1 uses percentile/share as relative financial
features and display context; the wage-benchmark candidate does not use club
payroll/median.** The new club-season layer instead uses the most recent fully
completed pre-decision season. Its resource, recruitment, squad and combined
blocks did not repair the historical continuity endpoint and were rejected as
predictive candidates; the canonical table remains useful descriptive
infrastructure.

### E. Separate injury-thesis dataset

`Data/injury/full_dataset_thesis - 1.csv` contains **15,603 injury rows** and is
NOT the 143,195-row ID-linked supplemental injury table.

It was used by contribution refinement and corrected repair diagnostics.
Existing corrected evidence reports 3,982 linked players and 13,569 retained
spells after date-aware linkage exclusions and duplicate handling. No injury
feature appears in the selected frozen V1 or current V2 feature lists.

Thus "we have tried injury data" must specify **which source, which linkage,
which experiment, and whether it was retained**. The larger supplemental source
has not had equivalent integration into current extension models. The older
notebook injury exploration does not establish that integration.

### F. StatsBomb Open Data — already acquired and processed

| Local asset | Recounted contents |
| --- | ---: |
| Competition-season catalog entries | 80 |
| Cataloged matches | 3,961 |
| Event payload files | **4,235** |
| Raw events across those files | **14,874,171** |
| Lineup payload files | **4,235** |
| Player lineup entries | **161,958** |
| Distinct lineup player IDs | **11,889** |
| 360 payload files | **426**, of which **425 parse successfully** |
| Records in valid 360 files | **1,381,467** |
| Event/lineup files without catalog match metadata | **274 of each** |

The pre-existing extraction selects cataloged male matches from 2017-07-01
through 2024-06-30. Its output contains:

- **642 matches; 2,298,900 slim events**;
- **27,702 player-match and 1,284 team-match feature rows**;
- **6,192 player-period and 309 team-period rows**;
- **963,072 spatial-context rows**;
- player/club crosswalk candidates and **612 transfer case-study candidates**.

These outputs support a case-study workbook and supplementary availability
flags in historical player-club case reports. They **do not supply any selected
V1/V2 model predictor**. Player/club links remain review-only.

The local catalog's modern male Big-Five coverage is highly selective:

- La Liga: 36, 34, 33, and 35 matches in 2017–18, 2018–19, 2019–20, and 2020–21.
- Bundesliga: 34 matches in 2023–24.
- Ligue 1: 26 matches in 2021–22 and 32 in 2022–23. **All 58 involve PSG.**
- No Premier League or Serie A domestic matches in this catalog from July 2017
  onward. Earlier 2015–16 seasons are present (380 Premier League, 380 Serie A,
  380 La Liga, 377 Ligue 1 matches), but excluded by the current extractor's
  date window.

The 642-match extraction is not the entire local repository; conversely, owning
the entire local snapshot does not establish complete Big-Five match coverage
or equivalence to the provider's latest upstream release.

`three-sixty/3845506.json` is malformed. This confirms the older extractor's
known invalid-file flag; the source was not repaired or changed.

## 2. What reaches the current engine

### Integrated data backbone

- 190,592 canonical player identities and 46,490 club identities.
- 1,408,834 canonical player-date valuations.
- 1,106,278 canonical transfer events.
- 2,084,606 canonical player-team-competition-season performance rows.

These are integrated outputs, not additive independent observations on top of
the raw-source counts. Of the performance rows, **1,718,982 are
`season_summary_only`**. Owning two million seasonal rows does not mean owning
two million match logs or a uniformly dense advanced-performance panel.

### Frozen V1 versus current V2

| Module | Frozen V1 input count | Selected V2 input count | Current V2 basis |
| --- | ---: | ---: | --- |
| Sustained contribution | 15 (C6) | 6 | Age, position, league, market value, recent opportunity share and appearance rate |
| Any outbound, 24 months | 27 | 7 | Same six plus a nonlinear age term |
| 25% value downside, 24 months | 27 | 6 | Same six basic/context inputs |
| Annual wage benchmark | 6 | 6 | Age, position, league, market value, prior wage and availability flag |

The union across these four V2 candidates is **nine distinct input fields**.
Other data still constructs labels, eligibility, identity joins, uncertainty,
and reference distributions. Nine fields is not itself proof of a bad model;
it shows exactly which candidate feature set has been evaluated so far.

V1 also retains other horizons/targets: total 16 fitted artifacts. Its duration
model uses 20 inputs; annual wage, commitment and wage-to-value benchmarks use
six each. None of the selected artifacts includes an injury or StatsBomb field.
The exact per-artifact list is in `selected_model_feature_usage.csv`, read from
the artifacts rather than inferred from function comments.

### Previously engineered, not ported into current extension candidates

The old compatibility pipeline already built:

- **905,269** broad lagged player-season feature rows;
- **17,790** advanced compatibility rows;
- **10,540** strict prior-club-context eligible rows;
- role-season-standardized styles, teammate-role context, historical player
  preferences, and player/context interactions.

These were actually used in earlier transfer-scope model comparisons, not just
stored raw. They were not imported into current extension V2. Target-roster
scenario fields were intentionally separated because historical roster
membership can leak future information. Existing role-normalized *style*
features elsewhere do not mean V1's global goals/assists rates were contextualized.

## 3. Coverage in the extension cohort, not just raw row totals

The following uses **all 2,240 extension events signed through 2023**, including
unlinked events in the denominator. These are not the smaller eligible cohorts
for a specific prediction target. Only linked player/club events can qualify
for the availability joins. Windows are strictly before signing.

| Potential context | Events with evidence | Share | What this does NOT establish |
| --- | ---: | ---: | --- |
| At least 10 prior-year club matches with manager names | **2,039** | **91.0%** | Manager appointment dates, causal coach effects, predictive gain |
| At least 10 prior-year same-player/same-club lineup rows | **1,684** | **75.2%** | Every row is a start or played appearance; complete squad records |
| Prior completed-season club panel with ≥15 non-null wages | **1,912** | **85.4%** | Complete payroll, publication-time validity, exclusion of focal player |
| At least one supplemental injury start in preceding year | **1,053** | **47.0%** | Population injury reporting coverage; missing records do NOT mean healthy |
| Any earlier recorded supplemental injury start | **1,578** | **70.4%** | Exhaustive injury history |
| Previous completed club-season standings row, any competition | **2,179** | **97.3%** | Same-league match, season publication date, historical manager tenure |
| Lifetime teammate-edge record | **2,131** | **95.1%** | Pre-signing teammate history; rows are retrospective career aggregates |
| StatsBomb prior-year same-club match via existing candidate links | **58** | **2.6%** | Reviewed/accepted identities or exhaustive linkage of all raw StatsBomb players |
| At least 3 such StatsBomb matches | **4** | **0.18%** | A broad repeated-match training cohort |

StatsBomb availability uses all cataloged male raw lineup matches, player
position participation, and the existing unique-name candidate crosswalks.
Those crosswalks were created for the older selected sample, so this is a
**candidate-link-based diagnostic, not a proven maximum attainable overlap**.
No candidate link was promoted to accepted status.

### Ligue 1 specifically — 418 development extension events

- Manager history threshold: **384/418 (91.9%)**.
- Prior club wage panel threshold: **370/418 (88.5%)**.
- Lineup threshold: **302/418 (72.2%)**.
- Prior-year supplemental injury-start record: **122/418 (29.2%)**.
- Candidate-linked StatsBomb same-club prior-year match: **34/418 (8.1%)**.

This supports testing locally available club context before seeking wholesale
new sources. It does **not** explain the Ligue 1 model errors by itself. In
particular the lower injury-record rate cannot distinguish reporting gaps from
different injury incidence or cohort composition.

## 4. Dormant, withheld, stale, or duplicated: specific findings

1. **Manager/formations:** high-coverage dated records exist and have now been
   constructed into decision-time profiles. Their predeclared blocks did not
   add stable movement/retention skill and remain descriptive only.
2. **Club wage structure:** the new lagged club-season layer now captures
   payroll, wage concentration and transfer/squad context. Its experiment did
   not repair continuity. Missing a player's prior wage still does not imply
   that the entire club's previous wage structure is missing.
3. **Larger injury source:** not equivalent to the thesis source used in recent
   repair work. It needs a proper historical reporting/availability assessment,
   endpoint linkage, and tested feature construction, not another download.
4. **Squad/role context:** decision-time player role trajectory and
   player-relative competition profiles have now been constructed and tested.
   They did not clear the multiplicity-adjusted evidence gate.
5. **Career snapshots:** teammate edges, national totals, contract/agent fields,
   and club hierarchy are not safe to activate unchanged for historical models.
6. **Later extensions:** 424 raw rows are outside the current master, not
   automatically 424 new eligible training examples; deduplication, vintage and
   outcome maturity must be checked in a separately versioned build.
7. **Capology build boundaries:** the legacy salary/extension builder remains
   version-gated to its 35 salary and 34 extension source files. The 55 payroll
   and 55 transfer-window pages enter only through the separate verified club-
   financial-context builder; no guard was bypassed.
8. **Legacy root matrices:** `model_dataset.csv` (1,025 × 227) and
   `model_dataset_with_injuries.csv` (1,025 × 237) sit at `Data/processed/` root.
   The notebook constructs the former and explores injury joins; no selected
   V1/V2 loader consumes either. No current checked-in writer/consumer for the
   latter was located. Provenance is less clear than for the versioned pipelines.
9. **Documentation gap:** `DATA_PROVENANCE.md` lists four source families and
   omits StatsBomb. `DATA_READINESS_AUDIT.md` is explicitly an August 6 historical
   audit and does not describe current extension/overlay use. Future source
   recommendations should start with this inventory, not those older summaries.
10. **Backups and generated evidence are not junk:** the 6+ GB processed tree
    contains merged copies, evidence tables, feature matrices and diagnostics.
    Raw rows, cleaned rows, canonical rows, and backup rows must never be summed
    into an inflated "unique records used" claim. This audit recommends no deletion.

## 5. Next work, constrained by the evidence

1. Preserve the completed club-finance, lagged-behavior, role/squad and
   manager/tactical mechanism tests as rejected or descriptive evidence. Do not
   recycle those broad blocks under new labels.
2. Reconcile the **two injury sources** using IDs/dates and measure duplicates,
   reporting windows and contradictions before any model experiment. Do not
   sum their records or infer a zero from absence.
3. If role-standardized or other advanced performance research is revisited,
   define a materially new football mechanism and extension-date-safe feature
   contract first. Do not port roster-conditioned or target fields into strict
   historical training.
4. Treat StatsBomb as **targeted validation/case-study material** until identity
   review and event-cohort coverage support a broader claim. Earlier full-season
   slices could support a separate research question, not a silent replacement
   of the current evaluation cohort.
5. Keep the movement product scope fixed: generic continuity is retired;
   Meaningful Retention and Temporary Displacement await a later untouched
   temporal cohort; Permanent Separation remains research-only. Preserve frozen
   artifacts and distinguish input freshness from matured outcome evidence.

No new source acquisition should be proposed as "missing" without checking
`data_layer_usage_register.csv`, the table schemas, and the relevant cohort
overlap first. No item above is evidence of predictive improvement yet.

## Evidence and reproduction

Evidence directory: `Data/processed/data_usage_audit_2026_09_02/`.

- `file_inventory.csv`, `folder_inventory.csv`: every inspected file and size.
- `table_inventory.csv`, `table_schemas.csv`: row counts and schemas for all CSVs.
- `source_column_completeness.csv`, `source_group_coverage.csv`: full-column
  missingness and selected league/season/source coverage distributions.
- `data_layer_usage_register.csv`: disposition of every inventory family.
- `lineage_manifest_evidence.csv`, `folder_reference_evidence.csv`,
  `code_string_evidence.csv`: source-manifest and static-code evidence.
- `selected_model_feature_usage.csv`: exact selected V1/V2 feature sets.
- `capology_file_usage.csv`, `capology_rebuild_preflight.json`: base/overlay/
  unconsumed file partition and reproducibility drift.
- `statsbomb_*`: catalog, payload hashes/counts, and competition-season coverage.
- `extension_source_overlap_summary.csv`: development/all-year cohort availability.
- `extension_source_overlap_rows.csv`: local event-level audit evidence, not
  intended as a public dataset or modeling matrix.
- `v1_freeze_check.csv`, `independent_verification.*`, `run_summary.json`:
  preservation, checks and scope limitations.

```bash
python scripts/audit_data_usage_coverage.py --output Data/processed/data_usage_audit_NEW
python scripts/verify_data_usage_coverage.py --audit Data/processed/data_usage_audit_NEW
```

Use a new output directory. The explicit resume option is only for a completed
CSV stage after an interrupted JSON scan and verifies source metadata first.
The initial run encountered the known malformed 360 file; the scanner was
updated to record invalid payloads and resumed without changing that source.

### Audit limitations

- This proves what is local, not completeness against today's upstream catalogs.
- Static references and manifests show dependency evidence, not executed runtime
  telemetry, fitted coefficients, or feature importance. Model-feature use is
  stronger evidence because the actual selected V1 artifacts were inspected.
- Full CSV row counts parse every record with the first column selected; full
  column completeness is computed for the named primary/core tables, not every
  duplicated diagnostic output. Non-CSV documents/images/binaries outside
  StatsBomb and the selected model artifacts are inventoried by path/size, not
  semantically parsed.
- Date extrema labeled `lexical_*` in column profiles are raw string extrema,
  not guaranteed chronological ranges. Overlap windows separately parse dates.
- Source preservation uses size/mtime for the whole inventory and full SHA-256
  for frozen V1 artifacts; StatsBomb payload hashes were also captured. This is
  not a full-content before/after hash of every large CSV.
- Cohort overlaps count availability, not publication-time validity, exhaustive
  reporting, representative sampling, or new predictive performance.

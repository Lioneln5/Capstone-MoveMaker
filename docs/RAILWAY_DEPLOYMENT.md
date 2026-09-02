# Railway deployment

MoveMaker deploys as one FastAPI research-record service. The repository
contains the explanatory site, frozen model artifacts, and compact evidence.
No runtime data volume is required while HTTP scoring remains disabled.

## Current safe default

Engine V1 is frozen for historical audit. Generic continuity is retired, and
no Engine V2 output is authorized for deployment. The API therefore has no
environment-variable scoring override. Railway starts the explanatory site
without runtime data and reports `scoring_enabled: false` from `/health`.

## Service configuration

The checked-in `railway.toml` supplies the Railpack start command and `/health`
health check. In Railway:

1. Deploy the repository-backed service with the checked-in `railway.toml`.
2. Do not set a scoring-enable variable; none is recognized.
3. A previously attached `/data` volume may remain, but the research site does
   not read it. It can be detached if it is not used by another service.

The historical upload commands and required serving-table layout are retained
below only for infrastructure provenance. They do not enable the current API:

   ```bash
   railway link
   railway volume list
   railway volume files upload Data/processed/canonical_integration/canonical_player_dimension.csv /processed/canonical_integration/canonical_player_dimension.csv
   railway volume files upload Data/processed/canonical_integration/canonical_club_dimension.csv /processed/canonical_integration/canonical_club_dimension.csv
   railway volume files upload Data/processed/canonical_integration/canonical_valuation_history.csv /processed/canonical_integration/canonical_valuation_history.csv
   railway volume files upload Data/processed/canonical_performance/canonical_player_team_season_performance.csv /processed/canonical_performance/canonical_player_team_season_performance.csv
   railway volume files upload Data/processed/canonical_performance/fbref_advanced_performance_features.csv /processed/canonical_performance/fbref_advanced_performance_features.csv
   railway volume files upload Data/processed/capology_contracts/canonical_salary_panel.csv /processed/capology_contracts/canonical_salary_panel.csv
   railway volume files upload Data/processed/contract_extension_integration/extension_modeling_master.csv /processed/contract_extension_integration/extension_modeling_master.csv
   railway volume files upload Data/processed/transfermarkt_clean/tables/appearances_clean.csv /processed/transfermarkt_clean/tables/appearances_clean.csv
   railway volume files upload Data/processed/transfermarkt_clean/tables/games_clean.csv /processed/transfermarkt_clean/tables/games_clean.csv
   railway volume files upload Data/processed/live_input_refresh/canonical_performance_overlay.csv /processed/live_input_refresh/canonical_performance_overlay.csv
   railway volume files upload Data/processed/live_input_refresh/canonical_salary_overlay.csv /processed/live_input_refresh/canonical_salary_overlay.csv
   ```

   The CLI addresses `/` as the volume root. Because Railway mounts that root
   at `/data` in the running container, `/processed/example.csv` in the CLI is
   available to the application as `/data/processed/example.csv`.

   The upload commands target the volume attached to the linked service. If a
   Railway project contains multiple services or volumes, pass the relevant
   `--service`, `--environment`, and volume selection shown by
   `railway volume list --json`.

4. Historical volume contents can still be verified locally with:

   ```bash
   python scripts/verify_railway_runtime_data.py --data-root Data
   ```

5. The checked-in bootstrap now starts the fail-closed API immediately. Confirm
`GET /health` reports `engine_loaded: false` and `scoring_enabled: false`.

## Required volume layout

The frozen V1 scorer historically required these files beneath `/data`:

| Path under `/data` | Purpose |
|---|---|
| `processed/canonical_integration/canonical_player_dimension.csv` | Player identity/search |
| `processed/canonical_integration/canonical_club_dimension.csv` | Current-club labels |
| `processed/canonical_integration/canonical_valuation_history.csv` | Decision-date market values |
| `processed/transfermarkt_clean/tables/appearances_clean.csv` | Current club and recent opportunity |
| `processed/transfermarkt_clean/tables/games_clean.csv` | Club-match filtering |
| `processed/canonical_performance/canonical_player_team_season_performance.csv` | Prior-season performance |
| `processed/canonical_performance/fbref_advanced_performance_features.csv` | Covered advanced inputs |
| `processed/capology_contracts/canonical_salary_panel.csv` | Salary and club-payroll context |
| `processed/contract_extension_integration/extension_modeling_master.csv` | Historical extension reference distribution |

These files are not loaded by the current hosted service. Frozen V1 regression
verification runs offline through the registered scripts.

## Operational note

The required tables are roughly 1.7 GB on disk and pandas expands them in
memory during startup. A Railway volume provides storage, not additional RAM.
After the first successful deployment, inspect peak memory usage. If the
service approaches its memory limit, the next optimization should be a compact
serving snapshot containing only the columns and rows read by the live API.

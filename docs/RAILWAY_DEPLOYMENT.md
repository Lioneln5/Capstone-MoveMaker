# Railway deployment

MoveMaker deploys as one FastAPI service. The repository contains application
code, serialized deployment models, and compact reference artifacts. Large
source tables remain outside Git and are mounted into the service through a
Railway volume.

## Service configuration

The checked-in `railway.toml` supplies the Railpack start command and `/health`
health check. In Railway:

1. Attach a volume to the API service at **`/data`**. Do not mount it at
   `/app/Data`, because that would hide model artifacts shipped in the image.
2. Add the service variable **`MOVEMAKER_DATA_ROOT=/data`**.
3. Upload the required local data directories into the volume:

   ```bash
   railway link
   railway volume list
   railway volume files upload Data/processed/canonical_integration/canonical_player_dimension.csv /data/processed/canonical_integration/canonical_player_dimension.csv
   railway volume files upload Data/processed/canonical_integration/canonical_club_dimension.csv /data/processed/canonical_integration/canonical_club_dimension.csv
   railway volume files upload Data/processed/canonical_integration/canonical_valuation_history.csv /data/processed/canonical_integration/canonical_valuation_history.csv
   railway volume files upload Data/processed/canonical_performance/canonical_player_team_season_performance.csv /data/processed/canonical_performance/canonical_player_team_season_performance.csv
   railway volume files upload Data/processed/canonical_performance/fbref_advanced_performance_features.csv /data/processed/canonical_performance/fbref_advanced_performance_features.csv
   railway volume files upload Data/processed/capology_contracts/canonical_salary_panel.csv /data/processed/capology_contracts/canonical_salary_panel.csv
   railway volume files upload Data/processed/contract_extension_integration/extension_modeling_master.csv /data/processed/contract_extension_integration/extension_modeling_master.csv
   railway volume files upload Data/processed/transfermarkt_clean/tables/appearances_clean.csv /data/processed/transfermarkt_clean/tables/appearances_clean.csv
   railway volume files upload Data/processed/transfermarkt_clean/tables/games_clean.csv /data/processed/transfermarkt_clean/tables/games_clean.csv
   railway volume files upload Data/processed/live_input_refresh/canonical_performance_overlay.csv /data/processed/live_input_refresh/canonical_performance_overlay.csv
   railway volume files upload Data/processed/live_input_refresh/canonical_salary_overlay.csv /data/processed/live_input_refresh/canonical_salary_overlay.csv
   ```

   The upload commands target the volume attached to the linked service. If a
   Railway project contains multiple services or volumes, pass the relevant
   `--service`, `--environment`, and volume selection shown by
   `railway volume list --json`.

4. Verify the volume contents from a Railway shell or locally before upload:

   ```bash
   python scripts/verify_railway_runtime_data.py --data-root Data
   ```

5. Redeploy and confirm `GET /health` returns a successful response before
   opening the site.

## Required volume layout

The API requires these files beneath `/data`:

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

The two files under `processed/live_input_refresh/` are optional fallbacks, but
should be uploaded for parity with the current local demo.

## Operational note

The required tables are roughly 1.7 GB on disk and pandas expands them in
memory during startup. A Railway volume provides storage, not additional RAM.
After the first successful deployment, inspect peak memory usage. If the
service approaches its memory limit, the next optimization should be a compact
serving snapshot containing only the columns and rows read by the live API.

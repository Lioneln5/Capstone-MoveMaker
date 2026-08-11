# MoveMaker canonical player-team-season performance

This layer integrates FBref seasonal performance, current Transfermarkt game appearances and lineups, and the supplemental Transfermarkt seasonal-summary table.

## Canonical grain

`canonical_player_id + canonical_club_id + canonical_competition_id + canonical_season_key`

National-team game appearances and lineups are excluded because the intended use is club recruitment and player-club compatibility. The separate supplemental national-performance table remains outside this layer.

## Source policy

- Source-specific metrics are retained in explicit `fbref_`, `current_`, and `datalake_` columns.
- Canonical matches, minutes, goals, and assists use `FBref -> current Transfermarkt appearances -> supplemental datalake` precedence. FBref is first within its Big Five scope so its totals remain coherent with its advanced per-90 features; current match logs are next; the much wider supplemental seasonal table fills remaining coverage.
- Canonical cards use `current Transfermarkt appearances -> supplemental datalake` because FBref cards are absent from the cleaned seasonal source.
- Cross-source disagreement is never erased. Exact comparison status, ranges, and a review queue are emitted.
- Game metadata is authoritative for current Transfermarkt season and competition. Explicit national-team games are excluded.
- The 450-minute modeling flag is a convenience screen, not a performance or confidence score.

## Outputs

- `canonical_player_team_season_performance.csv`: 2,084,606 canonical rows.
- `modeling_scope_2017_2024.csv`: 905,269 compact rows covering 2017-18 through 2023-24 across all available club competitions.
- `fbref_advanced_performance_features.csv`: 17,790 dense accepted FBref rows with all cleaned FBref fields.
- `performance_difference_review.csv`: 49,595 canonical rows with at least one differing comparable basic metric.
- `unresolved_fbref_performance_review.csv`: 453 FBref rows withheld until identity resolution.

The advanced FBref companion joins one-to-one to applicable spine rows by `canonical_performance_id`.

## Validation

Blocking checks: 12; failures: 0.

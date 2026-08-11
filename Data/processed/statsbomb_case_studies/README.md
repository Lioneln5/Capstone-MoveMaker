# StatsBomb case-study extract

Generated from the immutable local StatsBomb Open Data checkout.

## Selection rule

- Cataloged matches only: a match must exist in `data/matches`.
- Male competitions only.
- Match dates from 2017-07-01 through 2024-06-30, inclusive.
- All modern male competitions are retained for case studies and labeled as `primary_big_five_club`, `secondary_other_club`, or `secondary_international`.
- The 274 event files without current match metadata are excluded.
- Cross-source player and club links are review-only candidates, never automatically accepted identities.

## Result

- 642 selected matches from 3,961 cataloged matches.
- 2,298,900 slim event rows in `events_case_study.csv.gz`.
- 27,702 player-match feature rows and 1,284 team-match feature rows.
- 6,192 player-team-competition-season rows and 309 team-competition-season rows.
- 963,072 compact StatsBomb 360 event-context rows.
- 612 transfer case-study candidate rows requiring identity review.

## Recommended files

- `transfer_case_study_candidates.csv`: shortlist transfers by available pre/post player and club evidence.
- `player_period_features.csv`: compact player style/performance profiles.
- `team_period_features.csv`: compact club/national-team style profiles.
- `player_match_features.csv` and `team_match_features.csv`: match-level drilldown.
- `events_case_study.csv.gz`: slim event stream for pitch maps, pass maps, shot maps, and timelines.
- `three_sixty_event_context.csv.gz`: compact spatial context joined by `event_id`.
- `match_catalog.csv`: full catalog with inclusion flags.
- `field_dictionary.csv`, `data_quality_issues.csv`, `extraction_checks.csv`, and `output_manifest.csv`: audit and provenance.

## Important cautions

1. Modern Big Five coverage is selected rather than league-complete; do not train the main compatibility model on this sample as if it were representative.
2. `progressive_passes` and `progressive_carries` use a documented 10-unit forward-gain heuristic, not a proprietary StatsBomb definition.
3. `minutes_played_estimate` is reconstructed from lineup intervals and final event time.
4. `possession_sequence_share` counts distinct possessions; it is not time-based possession percentage.
5. Player and club crosswalks lack StatsBomb birth-date evidence and remain `manual_review_status = required`.
6. Post-transfer fields in the transfer shortlist are targets/diagnostics and must never enter pre-transfer features.
7. Published work must credit StatsBomb and comply with the repository attribution/logo terms.

## Reproduction

Run with the project Python environment:

```bash
python scripts/extract_statsbomb_case_studies.py
```

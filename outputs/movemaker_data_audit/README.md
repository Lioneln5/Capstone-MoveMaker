# MoveMaker data audit

Generated from the 19 CSV files under `data/` on 2026-08-02. The audit is read-only: none of the source CSVs were changed.

## Scope

- 19 files
- 7,227,164 rows
- 709.61 MB of CSV data
- 618 physical file-column entries, including repeated seasonal schemas
- Seven seasonal performance files covering 2017-18 through 2023-24
- Twelve Transfermarkt tables

## Primary deliverable

`MoveMaker_Data_Dictionary.xlsx` contains:

- an executive audit summary;
- table grain, candidate keys, coverage, and modeling cautions;
- a filterable column dictionary with definitions, inferred types, missingness, cardinality, sample values, modeling roles, and timing/leakage rules;
- Transfermarkt foreign-key coverage;
- deterministic seasonal-to-Transfermarkt matching results;
- candidate-key uniqueness checks;
- cross-season schema checks;
- seasonal arithmetic and duplicate-field checks; and
- modeling rules that should become pipeline assertions.

The accompanying CSV files expose each audit table in a machine-readable format. `audit_profile.json` contains the combined profile.

## Key findings

1. All 19 files parsed successfully. Every tested physical or composite candidate key is unique in the current extracts.
2. All seven seasonal files share the same 65-column schema. The tested arithmetic relationships pass for all 18,243 seasonal rows.
3. Normalized player name plus birth year uniquely links 17,043 of 18,243 seasonal rows to `players.csv` (93.42%). Remaining identities require resolution and review.
4. Exact normalized club names link only 2,167 of 18,243 seasonal rows to `clubs.csv` (11.88%). A reviewed club alias/crosswalk is mandatory.
5. `clubs.csv` is not a complete historical club master: it covers 38.7% of `transfers.from_club_id` rows and 43.9% of `transfers.to_club_id` rows. Build a union club dimension from all ID/name-bearing tables.
6. Transfer economics need classification before ROI modeling: 35.5% of transfer fees are missing, 57.1% of all rows contain a numeric-zero fee, and 38.9% of transfer market values are missing.
7. `transfers.csv` spans 1993-07-01 through 2030-06-30 and contains 15 dates after the audit date. Future commitments and out-of-cohort records must be filtered explicitly.
8. Latest-snapshot fields in `players.csv`, `clubs.csv`, and `national_teams.csv` are unsafe for historical backtests unless a historical timestamp can be established.
9. `Avg Mins per Match` behaves like total minutes, not an average. `Goals` exactly duplicates `Goals Scored`, and `Progressive Carries` exactly duplicates `carries_prgc` across all seasonal rows.
10. Goalkeeper-only seasonal measures are zero-filled for many outfield players. Those values should be modeled as structurally not applicable rather than genuine measured zeros.

## Recommended next build

1. Define the canonical transfer cohort and movement-type rules.
2. Build reviewed player and club crosswalks with match method, confidence, and review status.
3. Define sporting and financial targets at fixed post-transfer horizons.
4. Implement strict as-of joins and leakage assertions.
5. Quantify the surviving sample after every eligibility and completeness filter.
6. Compare a traditional baseline with a player-club compatibility model using chronological validation.

## Reproduction

Run `scripts/profile_data.py` with the bundled Python environment, then run `build_data_dictionary.mjs` with the bundled Node environment and `@oai/artifact-tool` available through the local `node_modules` junction.

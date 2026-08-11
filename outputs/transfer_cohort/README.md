# MoveMaker transfer cohort

Generated on 2026-08-03 from the audited CSV sources. Raw files under `data/2017-2024` and `data/TransferMarkt` were not modified.

## Cohort definition

- Transfer date window: 2018-07-01 through 2024-02-27
- Start rationale: first date with a fully completed 2017-18 seasonal feature set
- End rationale: latest date with a full 24-month horizon before the final valuation observation on 2026-02-27
- Pre-transfer valuation: latest observation on or before transfer, no more than 365 days old
- Outcome valuation: nearest observation to the 12- or 24-month target, within plus or minus 90 days
- Sporting coverage: at least 10 recorded destination games within 12 months and at least 20 within 24 months
- Prior-season features: latest season ending before the transfer date
- Financial ROI proxy: `(post-transfer market value - fee) / fee`; this is not realized resale profit

## Resulting samples

- Broad dated candidate cohort: 17,060 transfers and 4,182 players
- Transfers with prior completed seasonal features: 1,890
- Sporting 12-month eligible: 1,326
- Sporting 24-month eligible: 1,314
- Financial 12-month provisional: 1,058
- Financial 24-month provisional: 957
- Strict combined 24-month provisional: 244

The recommended strategy is to train sporting and financial models separately, then combine calibrated predictions in the decision layer. The 244-record intersection is useful for validation but is too small to support a highly complex combined model by itself.

## Processed datasets

- `data/processed/transfer_cohort_master.csv`: 17,060 transfer rows and 149 source, feature, outcome, and eligibility fields
- `data/processed/player_crosswalk.csv`: seasonal player identity resolution to Transfermarkt player IDs
- `data/processed/player_season_features.csv`: 16,322 linked player-season records with summed count metrics and minutes-weighted rate metrics
- `data/processed/club_dimension_union.csv`: 8,299 club/team IDs unioned across club, transfer, game, and national-team sources

## Important cautions

1. Financial eligibility is provisional. The source does not distinguish all permanent transfers, loan fees, loan returns, free transfers, and undisclosed movements.
2. A numeric zero fee is not treated as a paid transfer or silently imputed.
3. Post-transfer fields are targets or diagnostics and must never enter the pre-transfer feature matrix.
4. Transfer date is the authoritative time field. There are 174 cases where the source transfer-season label disagrees with the date-derived season.
5. There are 317 nonmissing ages outside the 15-45 review range. They remain visible rather than being silently removed; many represent youth movements.
6. The player crosswalk resolves 93.4% of seasonal rows uniquely. Ambiguous and unmatched identities remain available for manual review.

## Validation status

All blocking checks pass:

- transfer keys are unique;
- cohort dates remain inside the declared window;
- pre-transfer valuations do not occur after transfer;
- eligible horizon valuations meet the declared tolerance;
- no negative fees or appearance minutes were found; and
- combined eligibility reconciles to both component samples.

Warnings are preserved in the cohort workbook and do not silently change sample membership.

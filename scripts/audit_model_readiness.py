"""Recompute MoveMaker model-readiness counts from the current processed data.

This audit deliberately separates broad data availability, provisional eligibility,
and stricter horizon-safe/model-specific cohorts. It does not alter source or
processed datasets.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "Data/processed/transfermarkt_merged/transfermarkt_comprehensive_master.csv"
COHORT = ROOT / "Data/processed/transfer_cohort_master.csv"
CROSSWALK = ROOT / "Data/processed/player_crosswalk.csv"
OUTPUT = ROOT / "outputs/data_readiness/model_readiness_metrics.json"


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").str.casefold().map({"true": True, "false": False}).fillna(False)


def describe_mask(frame: pd.DataFrame, mask: pd.Series) -> dict[str, object]:
    selected = frame.loc[mask]
    dates = pd.to_datetime(selected["transfer_date"], errors="coerce")
    return {
        "rows": int(mask.sum()),
        "unique_players": int(selected["player_id"].nunique()),
        "date_min": None if selected.empty else dates.min().date().isoformat(),
        "date_max": None if selected.empty else dates.max().date().isoformat(),
        "pct_of_master": float(mask.mean()),
    }


def main() -> None:
    master = pd.read_csv(MASTER, low_memory=False)
    master["transfer_date"] = pd.to_datetime(master["transfer_date"], errors="coerce")

    same_club = as_bool(master["is_same_club_id"])
    plausible_age = as_bool(master["player_age_at_transfer_plausible_15_45"])
    sporting_h12 = as_bool(master["sporting_post12_horizon_observable"])
    sporting_h24 = as_bool(master["sporting_post24_horizon_observable"])
    valuation_h12 = as_bool(master["valuation_post12_horizon_observable"])
    valuation_h24 = as_bool(master["valuation_post24_horizon_observable"])
    encoded_s12 = as_bool(master["eligible_sporting_12m_provisional"])
    encoded_s24 = as_bool(master["eligible_sporting_24m_provisional"])
    encoded_f12 = as_bool(master["eligible_financial_12m_provisional"])
    encoded_f24 = as_bool(master["eligible_financial_24m_provisional"])
    encoded_c24 = as_bool(master["eligible_combined_24m_provisional"])

    strict_s12 = encoded_s12 & sporting_h12
    strict_s24 = encoded_s24 & sporting_h24
    strict_f12 = encoded_f12 & valuation_h12
    strict_f24 = encoded_f24 & valuation_h24
    strict_c24 = strict_s24 & strict_f24

    club_context = master["club_origin_pre365_games"].ge(10) & master["club_destination_pre365_games"].ge(10)
    strict_s24_club = strict_s24 & club_context
    strict_c24_club = strict_c24 & club_context

    cohort = pd.read_csv(COHORT, low_memory=False)
    cohort["transfer_date"] = pd.to_datetime(cohort["transfer_date"], errors="coerce")
    cohort["preseason_end_date"] = pd.to_datetime(cohort["preseason_end_date"], errors="coerce")
    season_start = cohort["transfer_date"].dt.year.where(
        cohort["transfer_date"].dt.month.ge(7), cohort["transfer_date"].dt.year - 1
    )
    expected_prior = (season_start - 1).astype("Int64").astype("string") + "-" + season_start.astype("Int64").astype("string")
    immediate_prior = cohort["preseason_season"].astype("string").eq(expected_prior)
    freshness_days = (cohort["transfer_date"] - cohort["preseason_end_date"]).dt.days
    fresh_prior = immediate_prior & freshness_days.between(1, 365)
    cohort_s12 = as_bool(cohort["eligible_sporting_12m"])
    cohort_s24 = as_bool(cohort["eligible_sporting_24m"])
    cohort_f24 = as_bool(cohort["eligible_financial_24m_provisional"])

    natural_key = ["player_id", "transfer_date", "from_club_id", "to_club_id"]
    master_context = master[natural_key + [
        "club_origin_pre365_games", "club_destination_pre365_games",
        "sporting_post24_horizon_observable",
    ]]
    rich = cohort.merge(master_context, on=natural_key, how="left", validate="one_to_one")
    rich_club_context = rich["club_origin_pre365_games"].ge(10) & rich["club_destination_pre365_games"].ge(10)
    rich_horizon = as_bool(rich["sporting_post24_horizon_observable"])
    rich_s24 = fresh_prior & cohort_s24 & rich_horizon
    rich_s24_club = rich_s24 & rich_club_context
    rich_combined24 = fresh_prior & cohort_s24 & cohort_f24 & rich_horizon

    crosswalk = pd.read_csv(CROSSWALK, low_memory=False)
    crosswalk_counts = crosswalk["match_status"].value_counts(dropna=False).to_dict()

    source_coverage = {
        "pre365_valuation": int(as_bool(master["has_pretransfer_valuation_365d"]).sum()),
        "pre365_appearances": int(as_bool(master["appearance_pre365_all_has_records"]).sum()),
        "pre365_origin_appearances": int(as_bool(master["appearance_pre365_origin_has_records"]).sum()),
        "pre365_lineups": int(as_bool(master["lineup_pre365_all_has_records"]).sum()),
        "pre365_events": int(as_bool(master["event_pre365_all_has_records"]).sum()),
        "origin_club_pre365_any": int(as_bool(master["club_origin_pre365_has_records"]).sum()),
        "destination_club_pre365_any": int(as_bool(master["club_destination_pre365_has_records"]).sum()),
        "both_clubs_pre365_10_games": int(club_context.sum()),
        "positive_reported_fee": int(master["transfer_fee_status"].eq("positive_reported").sum()),
        "unclassified_zero_fee": int(master["transfer_fee_status"].eq("unclassified_zero").sum()),
        "missing_fee": int(master["transfer_fee_status"].eq("missing").sum()),
        "future_dated_after_2026_08_06": int(master["transfer_date"].gt(pd.Timestamp("2026-08-06")).sum()),
        "same_club_id": int(same_club.sum()),
        "plausible_age_15_45": int(plausible_age.sum()),
    }

    masks = {
        "transfermarkt_sporting_12m_encoded": encoded_s12,
        "transfermarkt_sporting_12m_horizon_safe": strict_s12,
        "transfermarkt_sporting_24m_encoded": encoded_s24,
        "transfermarkt_sporting_24m_horizon_safe": strict_s24,
        "transfermarkt_sporting_24m_with_both_club_contexts": strict_s24_club,
        "transfermarkt_financial_12m_encoded_provisional": encoded_f12,
        "transfermarkt_financial_12m_horizon_safe_provisional": strict_f12,
        "transfermarkt_financial_24m_encoded_provisional": encoded_f24,
        "transfermarkt_financial_24m_horizon_safe_provisional": strict_f24,
        "transfermarkt_combined_24m_encoded_provisional": encoded_c24,
        "transfermarkt_combined_24m_horizon_safe_provisional": strict_c24,
        "transfermarkt_combined_24m_with_both_club_contexts_provisional": strict_c24_club,
        "transfermarkt_sporting_24m_horizon_safe_plausible_age": strict_s24 & plausible_age,
    }

    result = {
        "generated_from": {
            "master": str(MASTER.relative_to(ROOT)),
            "legacy_cohort": str(COHORT.relative_to(ROOT)),
            "player_crosswalk": str(CROSSWALK.relative_to(ROOT)),
        },
        "master": {
            "rows": int(len(master)),
            "columns": int(len(master.columns)),
            "unique_players": int(master["player_id"].nunique()),
            "date_min": master["transfer_date"].min().date().isoformat(),
            "date_max": master["transfer_date"].max().date().isoformat(),
        },
        "feature_coverage": source_coverage,
        "model_cohorts": {name: describe_mask(master, mask) for name, mask in masks.items()},
        "rich_fbref": {
            "legacy_cohort_rows": int(len(cohort)),
            "any_prior_season_rows": int(cohort["preseason_season"].notna().sum()),
            "immediate_prior_season_rows": int(immediate_prior.sum()),
            "fresh_immediate_prior_rows": int(fresh_prior.sum()),
            "fresh_sporting_12m_rows": int((fresh_prior & cohort_s12).sum()),
            "fresh_sporting_24m_horizon_safe_rows": int(rich_s24.sum()),
            "fresh_sporting_24m_with_both_club_contexts_rows": int(rich_s24_club.sum()),
            "fresh_combined_24m_horizon_safe_provisional_rows": int(rich_combined24.sum()),
        },
        "player_crosswalk": {
            "identities": int(len(crosswalk)),
            "status_counts": {str(key): int(value) for key, value in crosswalk_counts.items()},
            "seasonal_rows_unique_link": int(crosswalk.loc[crosswalk["match_status"].eq("unique"), "seasonal_rows"].sum()),
            "seasonal_rows_total": int(crosswalk["seasonal_rows"].sum()),
        },
        "annual_horizon_safe_counts": (
            pd.DataFrame({
                "year": master["transfer_date"].dt.year,
                "sporting_24m": strict_s24.astype(int),
                "financial_24m_provisional": strict_f24.astype(int),
                "combined_24m_provisional": strict_c24.astype(int),
                "sporting_24m_both_clubs": strict_s24_club.astype(int),
            })
            .dropna(subset=["year"])
            .groupby("year", as_index=False)
            .sum()
            .query("sporting_24m > 0 or financial_24m_provisional > 0")
            .astype({"year": int})
            .to_dict(orient="records")
        ),
        "rich_fbref_annual_counts": (
            pd.DataFrame({
                "year": cohort["transfer_date"].dt.year,
                "fresh_prior": fresh_prior.astype(int),
                "fresh_sporting_24m": rich_s24.astype(int),
                "fresh_sporting_24m_both_clubs": rich_s24_club.astype(int),
                "fresh_combined_24m_provisional": rich_combined24.astype(int),
            })
            .dropna(subset=["year"])
            .groupby("year", as_index=False)
            .sum()
            .query("fresh_prior > 0")
            .astype({"year": int})
            .to_dict(orient="records")
        ),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Audit which MoveMaker financial product modules are supportable by current data.

This is a coverage and target-definition audit, not a predictive model.  It
reconciles the comprehensive transfer master with canonical movement types and
fees, applies horizon-safe valuation rules, measures overlap with sporting
evidence, and distinguishes fee-only resale evidence from full accounting ROI.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = ROOT / "Data" / "processed" / "transfermarkt_merged" / "transfermarkt_comprehensive_master.csv"
CANONICAL_PATH = ROOT / "Data" / "processed" / "canonical_integration" / "canonical_transfer_history.csv"
TARGETS_PATH = ROOT / "Data" / "processed" / "compatibility_targets" / "transfer_compatibility_targets.csv"
RETENTION_PATH = ROOT / "Data" / "processed" / "retention_diagnostic" / "retention_targets.csv"
OUTPUT_DIR = ROOT / "Data" / "processed" / "financial_module_feasibility"

AS_OF_DATE = pd.Timestamp("2026-08-10")
MODELING_ERA_START = pd.Timestamp("2017-07-01")
RESALE_HORIZON_DAYS = 1460
EVALUATION_YEARS = [2020, 2021, 2022, 2023]
MIN_TOTAL_ROWS = 500
MIN_ROWS_PER_EVAL_YEAR = 75
MIN_ADEQUATE_EVAL_YEARS = 3

PROTECTED_DIRS = [
    ROOT / "Data" / "processed" / "compatibility_targets",
    ROOT / "Data" / "processed" / "coarse_club_context_diagnostic",
    ROOT / "Data" / "processed" / "opportunity_transferability_decomposition",
    ROOT / "Data" / "processed" / "retention_diagnostic",
]

MASTER_COLUMNS = [
    "transfer_event_id", "player_id", "transfer_date", "transfer_season",
    "from_club_id", "from_club_name", "to_club_id", "to_club_name",
    "player_snapshot_position", "player_snapshot_sub_position",
    "transfer_fee", "transfer_fee_status",
    "valuation_pre_date", "valuation_pre_eur", "valuation_pre_staleness_days",
    "valuation_pre_within_90d", "valuation_pre_within_180d", "valuation_pre_within_365d",
    "valuation_post12_target_date", "valuation_post12_horizon_observable",
    "valuation_post12_nearest_date", "valuation_post12_nearest_eur", "valuation_post12_gap_days",
    "valuation_post24_target_date", "valuation_post24_horizon_observable",
    "valuation_post24_nearest_date", "valuation_post24_nearest_eur", "valuation_post24_gap_days",
    "appearance_post12_destination_minutes", "club_destination_post12_games",
    "appearance_post24_destination_minutes", "club_destination_post24_games",
    "eligible_sporting_24m_provisional", "sporting_post24_horizon_observable",
    "club_origin_pre365_games", "club_destination_pre365_games",
    "club_destination_pre365_dominant_competition_id",
]

CANONICAL_COLUMNS = [
    "canonical_transfer_event_id", "canonical_player_id", "transfer_date",
    "from_club_id", "to_club_id", "canonical_transfer_type",
    "canonical_transfer_fee_eur", "canonical_transfer_fee_status",
    "transfer_fee_conflict", "transfer_type_conflict", "event_conflict",
    "event_resolution_status",
]


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").fillna("").str.strip().str.lower().isin({"true", "1", "yes"})


def clean_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype(str)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_tree_sha256() -> str:
    digest = hashlib.sha256()
    for directory in sorted(PROTECTED_DIRS):
        if not directory.exists():
            continue
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def position_group(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    if "goal" in text:
        return "goalkeeper"
    if any(token in text for token in ["back", "defender", "sweeper"]):
        return "defender"
    if "midfield" in text:
        return "midfielder"
    if any(token in text for token in ["winger", "forward", "striker", "attack"]):
        return "attacker"
    return "unknown"


def load_and_reconcile() -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    master = pd.read_csv(MASTER_PATH, usecols=MASTER_COLUMNS, low_memory=False)
    canonical = pd.read_csv(CANONICAL_PATH, usecols=CANONICAL_COLUMNS, low_memory=False)
    canonical = canonical.rename(columns={"canonical_player_id": "player_id"})
    for frame in [master, canonical]:
        for column in ["player_id", "from_club_id", "to_club_id"]:
            frame[column] = clean_id(frame[column])
        frame["transfer_date"] = pd.to_datetime(frame["transfer_date"], errors="coerce")

    natural_key = ["player_id", "transfer_date", "from_club_id", "to_club_id"]
    if canonical.duplicated(natural_key).any():
        raise RuntimeError("Canonical transfer natural key is not unique.")
    frame = master.merge(canonical, on=natural_key, how="left", validate="one_to_one")
    if frame["canonical_transfer_event_id"].isna().any():
        raise RuntimeError("Every comprehensive-master event must map to the canonical transfer layer.")

    for column in ["transfer_fee_conflict", "transfer_type_conflict", "event_conflict"]:
        frame[column] = as_bool(frame[column])
    frame["is_cross_club"] = frame["from_club_id"].ne(frame["to_club_id"])
    frame["is_clean_permanent"] = (
        frame["canonical_transfer_type"].eq("transfer")
        & ~frame["transfer_type_conflict"]
        & ~frame["event_conflict"]
        & frame["is_cross_club"]
        & frame["transfer_date"].le(AS_OF_DATE)
    )
    frame["is_modeling_era"] = frame["transfer_date"].ge(MODELING_ERA_START)
    frame["transfer_year"] = frame["transfer_date"].dt.year.astype("Int64")
    frame["season_start_year"] = frame["transfer_year"].where(
        frame["transfer_date"].dt.month.ge(7), frame["transfer_year"] - 1
    ).astype("Int64")
    frame["position_group"] = frame["player_snapshot_position"].map(position_group)
    frame["destination_competition"] = frame[
        "club_destination_pre365_dominant_competition_id"
    ].fillna("unknown").astype(str)

    frame["canonical_transfer_fee_eur"] = pd.to_numeric(
        frame["canonical_transfer_fee_eur"], errors="coerce"
    )
    frame["reliable_positive_inbound_fee"] = (
        frame["is_clean_permanent"]
        & ~frame["transfer_fee_conflict"]
        & frame["canonical_transfer_fee_status"].eq("positive_reported")
        & frame["canonical_transfer_fee_eur"].gt(0)
    )
    frame["fee_band"] = pd.cut(
        frame["canonical_transfer_fee_eur"],
        bins=[-np.inf, 0, 1_000_000, 5_000_000, 15_000_000, 30_000_000, np.inf],
        labels=["zero_or_missing", "0-1m", "1-5m", "5-15m", "15-30m", "30m+"],
        right=False,
    ).astype("string").fillna("zero_or_missing")

    for horizon in [12, 24]:
        pre = pd.to_numeric(frame["valuation_pre_eur"], errors="coerce")
        post = pd.to_numeric(frame[f"valuation_post{horizon}_nearest_eur"], errors="coerce")
        gap = pd.to_numeric(frame[f"valuation_post{horizon}_gap_days"], errors="coerce")
        staleness = pd.to_numeric(frame["valuation_pre_staleness_days"], errors="coerce")
        frame[f"market_value_change_{horizon}m_pct"] = (post - pre).div(pre.where(pre.gt(0)))
        frame[f"market_value_change_{horizon}m_eur"] = post - pre
        for tolerance in [60, 90, 120, 180]:
            frame[f"market_value_{horizon}m_eligible_{tolerance}d"] = (
                frame["is_clean_permanent"]
                & pre.gt(0)
                & staleness.between(0, 365)
                & as_bool(frame[f"valuation_post{horizon}_horizon_observable"])
                & post.gt(0)
                & gap.abs().le(tolerance)
            )

    for horizon, minimum_games in [(12, 10), (24, 20)]:
        minutes = pd.to_numeric(
            frame[f"appearance_post{horizon}_destination_minutes"], errors="coerce"
        )
        games = pd.to_numeric(frame[f"club_destination_post{horizon}_games"], errors="coerce")
        observed = frame["is_clean_permanent"] & games.ge(minimum_games) & minutes.notna()
        frame[f"opportunity_{horizon}m_observed"] = observed
        frame[f"destination_opportunity_share_{horizon}m"] = (
            minutes.div(90.0 * games.where(games.gt(0))).clip(lower=0, upper=1.05).where(observed)
        )

    frame["coarse_context_available"] = (
        as_bool(frame["eligible_sporting_24m_provisional"])
        & as_bool(frame["sporting_post24_horizon_observable"])
        & pd.to_numeric(frame["club_origin_pre365_games"], errors="coerce").ge(10)
        & pd.to_numeric(frame["club_destination_pre365_games"], errors="coerce").ge(10)
    )

    target_flags = pd.read_csv(
        TARGETS_PATH,
        usecols=[
            "canonical_transfer_event_id", "eligible_performance_model_primary",
            "target_role_performance_index_available",
        ],
        low_memory=False,
    )
    target_flags["rich_performance_available"] = (
        as_bool(target_flags["eligible_performance_model_primary"])
        & as_bool(target_flags["target_role_performance_index_available"])
    )
    target_flags = target_flags.groupby("canonical_transfer_event_id", as_index=False)[
        "rich_performance_available"
    ].max()
    retention_ids = pd.read_csv(RETENTION_PATH, usecols=["transfer_event_id"])
    retention_ids["retention_target_available"] = True
    frame = frame.merge(target_flags, on="canonical_transfer_event_id", how="left", validate="one_to_one")
    frame = frame.merge(retention_ids, on="transfer_event_id", how="left", validate="one_to_one")
    frame["rich_performance_available"] = frame["rich_performance_available"].astype("boolean").fillna(False).astype(bool)
    frame["retention_target_available"] = frame["retention_target_available"].astype("boolean").fillna(False).astype(bool)

    observed_canonical_max = canonical.loc[
        canonical["transfer_date"].le(AS_OF_DATE), "transfer_date"
    ].max()
    attach_resale_evidence(frame, canonical, observed_canonical_max)
    return frame, canonical, observed_canonical_max


def attach_resale_evidence(
    frame: pd.DataFrame, canonical: pd.DataFrame, observed_canonical_max: pd.Timestamp
) -> None:
    clean_outbound = canonical.loc[
        canonical["canonical_transfer_type"].eq("transfer")
        & ~as_bool(canonical["transfer_type_conflict"])
        & ~as_bool(canonical["event_conflict"])
        & canonical["from_club_id"].ne(canonical["to_club_id"])
        & canonical["transfer_date"].le(observed_canonical_max)
    ].sort_values(["player_id", "transfer_date", "canonical_transfer_event_id"])
    groups = {key: group for key, group in clean_outbound.groupby("player_id", sort=False)}

    records: list[dict[str, Any]] = []
    for row in frame.loc[frame["is_clean_permanent"]].itertuples(index=False):
        group = groups.get(row.player_id)
        outbound = None
        if group is not None:
            candidates = group.loc[
                group["transfer_date"].gt(row.transfer_date)
                & group["from_club_id"].eq(row.to_club_id)
                & group["transfer_date"].le(row.transfer_date + pd.Timedelta(days=RESALE_HORIZON_DAYS))
            ]
            if not candidates.empty:
                outbound = candidates.iloc[0]
        record: dict[str, Any] = {"transfer_event_id": row.transfer_event_id}
        if outbound is not None:
            record.update(
                {
                    "resale_outbound_event_id": outbound["canonical_transfer_event_id"],
                    "resale_outbound_date": outbound["transfer_date"],
                    "resale_outbound_to_club_id": outbound["to_club_id"],
                    "resale_outbound_fee_status": outbound["canonical_transfer_fee_status"],
                    "resale_outbound_fee_eur": outbound["canonical_transfer_fee_eur"],
                    "resale_outbound_fee_conflict": bool(outbound["transfer_fee_conflict"]),
                    "days_to_permanent_disposal": int(
                        (outbound["transfer_date"] - row.transfer_date).days
                    ),
                }
            )
        records.append(record)
    resale = pd.DataFrame(records)
    merged = frame[["transfer_event_id"]].merge(resale, on="transfer_event_id", how="left")
    for column in merged.columns[1:]:
        frame[column] = merged[column].to_numpy()

    frame["full_resale_followup_4y"] = (
        frame["transfer_date"] + pd.Timedelta(days=RESALE_HORIZON_DAYS)
    ).le(observed_canonical_max)
    frame["permanent_disposal_within_4y"] = frame["resale_outbound_event_id"].notna()
    outbound_fee = pd.to_numeric(frame["resale_outbound_fee_eur"], errors="coerce")
    frame["reliable_positive_outbound_fee"] = (
        frame["permanent_disposal_within_4y"]
        & frame["resale_outbound_fee_status"].eq("positive_reported")
        & ~as_bool(frame["resale_outbound_fee_conflict"])
        & outbound_fee.gt(0)
    )
    inbound_fee = pd.to_numeric(frame["canonical_transfer_fee_eur"], errors="coerce")
    frame["fee_only_resale_multiple"] = outbound_fee.div(inbound_fee.where(inbound_fee.gt(0)))
    frame["fee_only_resale_gain_eur"] = outbound_fee - inbound_fee


def module_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    universe = frame["is_clean_permanent"] & frame["is_modeling_era"]
    value24 = universe & frame["market_value_24m_eligible_90d"]
    opportunity24 = universe & frame["reliable_positive_inbound_fee"] & frame["opportunity_24m_observed"]
    fee_benchmark = (
        universe
        & frame["reliable_positive_inbound_fee"]
        & pd.to_numeric(frame["valuation_pre_eur"], errors="coerce").gt(0)
        & pd.to_numeric(frame["valuation_pre_staleness_days"], errors="coerce").between(0, 365)
    )
    dual = value24 & opportunity24
    resale = (
        universe
        & frame["reliable_positive_inbound_fee"]
        & frame["full_resale_followup_4y"]
        & frame["reliable_positive_outbound_fee"]
    )
    return {
        "market_value_downside_24m": value24,
        "paid_transfer_underutilization_24m": opportunity24,
        "fee_benchmarking": fee_benchmark,
        "dual_capital_at_risk_24m": dual,
        "fee_only_resale_return_4y": resale,
        "full_accounting_roi": pd.Series(False, index=frame.index),
    }


def build_funnels(frame: pd.DataFrame) -> pd.DataFrame:
    universe = frame["is_clean_permanent"] & frame["is_modeling_era"]
    pre365 = (
        pd.to_numeric(frame["valuation_pre_eur"], errors="coerce").gt(0)
        & pd.to_numeric(frame["valuation_pre_staleness_days"], errors="coerce").between(0, 365)
    )
    paid = frame["reliable_positive_inbound_fee"]
    rows: list[dict[str, Any]] = []

    def add(module: str, step: int, stage: str, mask: pd.Series, note: str) -> None:
        rows.append(
            {"module_id": module, "step": step, "stage": stage, "records": int(mask.sum()), "note": note}
        )

    add("market_value_downside_24m", 0, "Clean permanent arrivals in 2017+ era", universe, "Primary financial audit universe")
    add("market_value_downside_24m", 1, "Positive pre-transfer valuation within 365d", universe & pre365, "At-transfer baseline")
    add("market_value_downside_24m", 2, "24m horizon globally observable", universe & pre365 & as_bool(frame["valuation_post24_horizon_observable"]), "No future target date")
    add("market_value_downside_24m", 3, "Positive 24m valuation within +/-90d", universe & frame["market_value_24m_eligible_90d"], "Strict target cohort")

    add("paid_transfer_underutilization_24m", 0, "Clean permanent arrivals in 2017+ era", universe, "Primary financial audit universe")
    add("paid_transfer_underutilization_24m", 1, "Reliable positive inbound fee", universe & paid, "Zero is not inferred to mean free")
    add("paid_transfer_underutilization_24m", 2, "At least 20 destination club games by 24m", universe & paid & frame["opportunity_24m_observed"], "Makes zero player minutes interpretable")

    add("fee_benchmarking", 0, "Clean permanent arrivals in 2017+ era", universe, "Primary financial audit universe")
    add("fee_benchmarking", 1, "Reliable positive reported fee", universe & paid, "Target is disclosed fee, not total deal cost")
    add("fee_benchmarking", 2, "Positive pre-transfer valuation within 365d", universe & paid & pre365, "Strict benchmark cohort")

    add("dual_capital_at_risk_24m", 0, "Reliable positive reported fee", universe & paid, "Capital placed in scope")
    add("dual_capital_at_risk_24m", 1, "Strict 24m value outcome", universe & paid & frame["market_value_24m_eligible_90d"], "Value-risk label")
    add("dual_capital_at_risk_24m", 2, "Also has 24m opportunity outcome", universe & paid & frame["market_value_24m_eligible_90d"] & frame["opportunity_24m_observed"], "Joint commercial/sporting cohort")

    add("fee_only_resale_return_4y", 0, "Reliable positive inbound fee", universe & paid, "Acquisitions with disclosed spend")
    add("fee_only_resale_return_4y", 1, "Full four-year transfer follow-up", universe & paid & frame["full_resale_followup_4y"], "Avoids right-censoring")
    add("fee_only_resale_return_4y", 2, "Permanent disposal observed within four years", universe & paid & frame["full_resale_followup_4y"] & frame["permanent_disposal_within_4y"], "First later permanent outbound from destination")
    add("fee_only_resale_return_4y", 3, "Reliable positive outbound fee", universe & paid & frame["full_resale_followup_4y"] & frame["reliable_positive_outbound_fee"], "Conditional fee-only return sample")
    add("full_accounting_roi", 0, "Fee-only resale sample", universe & paid & frame["full_resale_followup_4y"] & frame["reliable_positive_outbound_fee"], "Still omits major costs")
    add("full_accounting_roi", 1, "Wages, bonuses, agent fees, amortization and book value available", pd.Series(False, index=frame.index), "Structurally unavailable in current public data")

    result = pd.DataFrame(rows)
    result["previous_records"] = result.groupby("module_id")["records"].shift(1)
    result["retention_from_previous"] = result["records"].div(result["previous_records"])
    result["starting_records"] = result.groupby("module_id")["records"].transform("first")
    result["retention_from_start"] = result["records"].div(result["starting_records"])
    return result


def build_module_summary(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> pd.DataFrame:
    definitions = {
        "market_value_downside_24m": (
            "Risk that estimated market value falls by at least 25% within 24 months.",
            "24m market-value change and decline >=25%",
            "advance_to_model_test",
            "Market value is an estimate, not realized cash.",
            "Run rolling-origin continuous change and threshold classification models.",
        ),
        "paid_transfer_underutilization_24m": (
            "Risk that a paid signing receives less than 10% of destination match-minute capacity over 24 months.",
            "24m destination opportunity share and share <10%",
            "advance_to_model_test",
            "Measures sporting capital utilization, not tactical quality or accounting return.",
            "Test paid-transfer under-utilization with market/profile, prior opportunity, and club context blocks.",
        ),
        "fee_benchmarking": (
            "Whether the disclosed acquisition fee is high or low relative to comparable player evidence.",
            "log positive reported inbound fee",
            "advance_to_model_test",
            "Reported fee omits add-ons and undisclosed total deal costs.",
            "Test rolling-origin log-fee prediction and comparable-transfer residuals.",
        ),
        "dual_capital_at_risk_24m": (
            "Whether a paid signing both loses at least 25% estimated value and receives under 10% opportunity.",
            "joint value-decline and under-utilization flag",
            "advance_with_caveat",
            "The joint severe-risk class is smaller than either component and needs class-balance checks.",
            "Model component risks separately first, then test a calibrated joint-risk layer.",
        ),
        "fee_only_resale_return_4y": (
            "Conditional fee gain or loss when a later permanent sale has a disclosed positive fee.",
            "outbound fee / inbound fee among disclosed positive sales",
            "exploratory_only",
            "Small, selected sample; zero/missing outbound fees are not equivalent to no proceeds.",
            "Use descriptively or as a narrow secondary model, never as portfolio-wide ROI.",
        ),
        "full_accounting_roi": (
            "True accounting return after transfer costs, wages, bonuses, agent fees and book value.",
            "full realized accounting ROI",
            "not_feasible",
            "Required cost and accounting fields do not exist in the current data.",
            "Do not model or claim full ROI without licensed club accounting data.",
        ),
    }
    rows = []
    for module, mask in masks.items():
        subset = frame.loc[mask]
        year_counts = subset["transfer_year"].value_counts()
        total_gate = len(subset) >= MIN_TOTAL_ROWS
        adequate_years = sum(int(year_counts.get(year, 0)) >= MIN_ROWS_PER_EVAL_YEAR for year in EVALUATION_YEARS)
        temporal_gate = adequate_years >= MIN_ADEQUATE_EVAL_YEARS
        if module == "market_value_downside_24m":
            target = subset["market_value_change_24m_pct"].le(-0.25)
            outcome_rate = float(target.mean()) if len(target) else np.nan
            outcome_gate = bool(len(target) and 0.10 <= outcome_rate <= 0.90)
        elif module == "paid_transfer_underutilization_24m":
            target = subset["destination_opportunity_share_24m"].lt(0.10)
            outcome_rate = float(target.mean()) if len(target) else np.nan
            outcome_gate = bool(len(target) and 0.10 <= outcome_rate <= 0.90)
        elif module == "dual_capital_at_risk_24m":
            target = subset["market_value_change_24m_pct"].le(-0.25) & subset[
                "destination_opportunity_share_24m"
            ].lt(0.10)
            outcome_rate = float(target.mean()) if len(target) else np.nan
            outcome_gate = bool(len(target) and 0.05 <= outcome_rate <= 0.95)
        elif module == "fee_benchmarking":
            target = np.log1p(subset["canonical_transfer_fee_eur"])
            outcome_rate = np.nan
            outcome_gate = bool(target.nunique() >= 50 and target.std() > 0)
        elif module == "fee_only_resale_return_4y":
            target = subset["fee_only_resale_multiple"]
            outcome_rate = float(target.ge(1).mean()) if len(target) else np.nan
            outcome_gate = bool(target.nunique() >= 25 and target.std() > 0)
        else:
            outcome_rate = np.nan
            outcome_gate = False

        question, primary_target, recommendation, caveat, next_test = definitions[module]
        definition_gate = module != "full_accounting_roi"
        rows.append(
            {
                "module_id": module,
                "business_question": question,
                "primary_target": primary_target,
                "eligible_rows": len(subset),
                "unique_players": subset["player_id"].nunique(),
                "date_min": subset["transfer_date"].min(),
                "date_max": subset["transfer_date"].max(),
                "outcome_positive_or_preservation_rate": outcome_rate,
                "sample_gate_500_rows": total_gate,
                "adequate_eval_years": adequate_years,
                "temporal_gate_3_years": temporal_gate,
                "outcome_variation_gate": outcome_gate,
                "definition_gate": definition_gate,
                "recommendation": recommendation,
                "major_caveat": caveat,
                "next_test": next_test,
                **{f"rows_{year}": int(year_counts.get(year, 0)) for year in EVALUATION_YEARS},
            }
        )
    return pd.DataFrame(rows)


def build_valuation_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pre_value = pd.to_numeric(frame["valuation_pre_eur"], errors="coerce")
    pre_age = pd.to_numeric(frame["valuation_pre_staleness_days"], errors="coerce")
    for scope, scope_mask in {
        "all_observed_history": frame["is_clean_permanent"],
        "modeling_era_2017plus": frame["is_clean_permanent"] & frame["is_modeling_era"],
    }.items():
        for horizon in [12, 24]:
            post = pd.to_numeric(frame[f"valuation_post{horizon}_nearest_eur"], errors="coerce")
            gap = pd.to_numeric(frame[f"valuation_post{horizon}_gap_days"], errors="coerce")
            for pre_tolerance in [90, 180, 365]:
                for post_tolerance in [60, 90, 120, 180]:
                    mask = (
                        scope_mask
                        & pre_value.gt(0)
                        & pre_age.between(0, pre_tolerance)
                        & as_bool(frame[f"valuation_post{horizon}_horizon_observable"])
                        & post.gt(0)
                        & gap.abs().le(post_tolerance)
                    )
                    change = frame.loc[mask, f"market_value_change_{horizon}m_pct"]
                    rows.append(
                        {
                            "scope": scope,
                            "horizon_months": horizon,
                            "pre_tolerance_days": pre_tolerance,
                            "post_tolerance_days": post_tolerance,
                            "eligible_rows": int(mask.sum()),
                            "unique_players": frame.loc[mask, "player_id"].nunique(),
                            "median_change_pct": change.median(),
                            "decline_10pct_rate": change.le(-0.10).mean(),
                            "decline_25pct_rate": change.le(-0.25).mean(),
                            "decline_50pct_rate": change.le(-0.50).mean(),
                            "preserve_75pct_rate": change.ge(-0.25).mean(),
                        }
                    )
    return pd.DataFrame(rows)


def build_outcome_balance(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for horizon in [12, 24]:
        for tolerance in [90, 180]:
            mask = (
                frame["is_modeling_era"]
                & frame[f"market_value_{horizon}m_eligible_{tolerance}d"]
            )
            change = frame.loc[mask, f"market_value_change_{horizon}m_pct"]
            for threshold in [0.10, 0.25, 0.50]:
                rows.append(
                    {
                        "module_id": f"market_value_downside_{horizon}m",
                        "cohort_rule": f"post_tolerance_{tolerance}d",
                        "threshold": threshold,
                        "threshold_definition": f"change <= -{threshold:.2f}",
                        "eligible_rows": len(change),
                        "positive_rows": int(change.le(-threshold).sum()),
                        "positive_rate": change.le(-threshold).mean(),
                    }
                )
    opportunity_mask = masks["paid_transfer_underutilization_24m"]
    opportunity = frame.loc[opportunity_mask, "destination_opportunity_share_24m"]
    for threshold in [0.10, 0.25, 0.50]:
        rows.append(
            {
                "module_id": "paid_transfer_underutilization_24m",
                "cohort_rule": "paid_permanent_and_20_destination_games",
                "threshold": threshold,
                "threshold_definition": f"opportunity_share < {threshold:.2f}",
                "eligible_rows": len(opportunity),
                "positive_rows": int(opportunity.lt(threshold).sum()),
                "positive_rate": opportunity.lt(threshold).mean(),
            }
        )
    dual_mask = masks["dual_capital_at_risk_24m"]
    dual_positive = (
        frame.loc[dual_mask, "market_value_change_24m_pct"].le(-0.25)
        & frame.loc[dual_mask, "destination_opportunity_share_24m"].lt(0.10)
    )
    rows.append(
        {
            "module_id": "dual_capital_at_risk_24m",
            "cohort_rule": "strict_value_and_paid_opportunity_overlap",
            "threshold": np.nan,
            "threshold_definition": "value decline >=25% AND opportunity share <10%",
            "eligible_rows": len(dual_positive),
            "positive_rows": int(dual_positive.sum()),
            "positive_rate": dual_positive.mean(),
        }
    )
    resale_mask = masks["fee_only_resale_return_4y"]
    multiple = frame.loc[resale_mask, "fee_only_resale_multiple"]
    rows.append(
        {
            "module_id": "fee_only_resale_return_4y",
            "cohort_rule": "full_followup_and_positive_disclosed_sale",
            "threshold": 1.0,
            "threshold_definition": "outbound fee >= inbound fee",
            "eligible_rows": len(multiple),
            "positive_rows": int(multiple.ge(1).sum()),
            "positive_rate": multiple.ge(1).mean(),
        }
    )
    return pd.DataFrame(rows)


def build_year_coverage(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> pd.DataFrame:
    years = sorted(int(year) for year in frame.loc[frame["is_clean_permanent"], "transfer_year"].dropna().unique())
    rows = []
    for year in years:
        row = {"transfer_year": year}
        year_mask = frame["transfer_year"].eq(year)
        row["clean_permanent_rows"] = int((year_mask & frame["is_clean_permanent"]).sum())
        for module, mask in masks.items():
            row[module] = int((year_mask & mask).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def build_overlap_matrix(masks: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for left_name, left in masks.items():
        for right_name, right in masks.items():
            intersection = int((left & right).sum())
            union = int((left | right).sum())
            rows.append(
                {
                    "module_left": left_name,
                    "module_right": right_name,
                    "intersection_rows": intersection,
                    "union_rows": union,
                    "jaccard": intersection / union if union else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_evidence_overlap(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> pd.DataFrame:
    evidence = {
        "sporting_opportunity_24m": frame["opportunity_24m_observed"],
        "coarse_club_context": frame["coarse_context_available"],
        "rich_performance_primary": frame["rich_performance_available"],
        "retention_target": frame["retention_target_available"],
    }
    rows = []
    for module, mask in masks.items():
        total = int(mask.sum())
        for evidence_name, evidence_mask in evidence.items():
            overlap = int((mask & evidence_mask).sum())
            rows.append(
                {
                    "module_id": module,
                    "evidence_layer": evidence_name,
                    "module_rows": total,
                    "overlap_rows": overlap,
                    "overlap_share": overlap / total if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_subgroup_coverage(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    dimensions = ["position_group", "destination_competition", "fee_band"]
    for module, mask in masks.items():
        subset = frame.loc[mask]
        for dimension in dimensions:
            counts = subset[dimension].fillna("unknown").astype(str).value_counts(dropna=False)
            for category, count in counts.items():
                rows.append(
                    {
                        "module_id": module,
                        "dimension": dimension,
                        "category": category,
                        "rows": int(count),
                        "share_of_module": count / len(subset) if len(subset) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def build_checks(frame: pd.DataFrame, canonical: pd.DataFrame, masks: dict[str, pd.Series]) -> pd.DataFrame:
    checks = []

    def add(check: str, passed: bool, observed: Any, expected: Any, notes: str) -> None:
        checks.append(
            {"check": check, "passed": bool(passed), "observed": observed, "expected": expected, "notes": notes}
        )

    add("master_key_unique", frame["transfer_event_id"].is_unique, frame["transfer_event_id"].nunique(), len(frame), "One row per comprehensive-master event.")
    add("canonical_match_complete", frame["canonical_transfer_event_id"].notna().all(), int(frame["canonical_transfer_event_id"].notna().sum()), len(frame), "Every master event has canonical movement evidence.")
    add("clean_permanent_has_no_type_conflicts", not frame.loc[frame["is_clean_permanent"], "transfer_type_conflict"].any(), int(frame.loc[frame["is_clean_permanent"], "transfer_type_conflict"].sum()), 0, "No ambiguous movement types enter financial modules.")
    add("reliable_inbound_fee_positive", frame.loc[frame["reliable_positive_inbound_fee"], "canonical_transfer_fee_eur"].gt(0).all(), float(frame.loc[frame["reliable_positive_inbound_fee"], "canonical_transfer_fee_eur"].min()), ">0", "Paid cohorts use only positive canonical fees.")
    add("reliable_inbound_fee_no_conflict", not frame.loc[frame["reliable_positive_inbound_fee"], "transfer_fee_conflict"].any(), int(frame.loc[frame["reliable_positive_inbound_fee"], "transfer_fee_conflict"].sum()), 0, "Conflicting fees are not used.")
    for horizon in [12, 24]:
        eligible = frame[f"market_value_{horizon}m_eligible_90d"]
        add(f"value_{horizon}m_pretransfer_timing", frame.loc[eligible, "valuation_pre_staleness_days"].between(0, 365).all(), frame.loc[eligible, "valuation_pre_staleness_days"].max(), "0..365", "Pre-transfer valuation is leakage-safe.")
        add(f"value_{horizon}m_post_tolerance", frame.loc[eligible, f"valuation_post{horizon}_gap_days"].abs().le(90).all(), frame.loc[eligible, f"valuation_post{horizon}_gap_days"].abs().max(), "<=90", "Strict outcome timing is enforced.")
    add("opportunity_bounds", frame.loc[frame["opportunity_24m_observed"], "destination_opportunity_share_24m"].between(0, 1.05).all(), f"{frame['destination_opportunity_share_24m'].min()}..{frame['destination_opportunity_share_24m'].max()}", "0..1.05", "Opportunity share uses club match-minute capacity.")
    add("resale_days_within_horizon", frame.loc[frame["permanent_disposal_within_4y"], "days_to_permanent_disposal"].between(1, RESALE_HORIZON_DAYS).all(), frame.loc[frame["permanent_disposal_within_4y"], "days_to_permanent_disposal"].max(), f"1..{RESALE_HORIZON_DAYS}", "Disposal timing follows the declared horizon.")
    add("full_roi_unavailable", int(masks["full_accounting_roi"].sum()) == 0, int(masks["full_accounting_roi"].sum()), 0, "No row is labeled as full accounting ROI.")
    add("dual_subset_value", not (masks["dual_capital_at_risk_24m"] & ~masks["market_value_downside_24m"]).any(), int((masks["dual_capital_at_risk_24m"] & ~masks["market_value_downside_24m"]).sum()), 0, "Dual-risk cohort is a strict subset.")
    add("dual_subset_opportunity", not (masks["dual_capital_at_risk_24m"] & ~masks["paid_transfer_underutilization_24m"]).any(), int((masks["dual_capital_at_risk_24m"] & ~masks["paid_transfer_underutilization_24m"]).sum()), 0, "Dual-risk cohort is a strict subset.")
    add("source_canonical_key_unique", not canonical.duplicated(["player_id", "transfer_date", "from_club_id", "to_club_id"]).any(), int(canonical.duplicated(["player_id", "transfer_date", "from_club_id", "to_club_id"]).sum()), 0, "Canonical source key remains unique.")
    return pd.DataFrame(checks)


def event_audit_columns(frame: pd.DataFrame, masks: dict[str, pd.Series]) -> pd.DataFrame:
    for module, mask in masks.items():
        frame[f"eligible_{module}"] = mask
    columns = [
        "transfer_event_id", "canonical_transfer_event_id", "player_id", "transfer_date",
        "transfer_year", "season_start_year", "from_club_id", "from_club_name",
        "to_club_id", "to_club_name", "position_group", "destination_competition",
        "canonical_transfer_type", "event_resolution_status", "transfer_type_conflict",
        "event_conflict", "canonical_transfer_fee_status", "canonical_transfer_fee_eur",
        "transfer_fee_conflict", "reliable_positive_inbound_fee", "fee_band",
        "valuation_pre_date", "valuation_pre_eur", "valuation_pre_staleness_days",
        "valuation_post12_target_date", "valuation_post12_nearest_date",
        "valuation_post12_nearest_eur", "valuation_post12_gap_days",
        "market_value_change_12m_eur", "market_value_change_12m_pct",
        "valuation_post24_target_date", "valuation_post24_nearest_date",
        "valuation_post24_nearest_eur", "valuation_post24_gap_days",
        "market_value_change_24m_eur", "market_value_change_24m_pct",
        "market_value_12m_eligible_90d", "market_value_12m_eligible_180d",
        "market_value_24m_eligible_90d", "market_value_24m_eligible_180d",
        "club_destination_post12_games", "appearance_post12_destination_minutes",
        "destination_opportunity_share_12m", "club_destination_post24_games",
        "appearance_post24_destination_minutes", "destination_opportunity_share_24m",
        "coarse_context_available", "rich_performance_available", "retention_target_available",
        "full_resale_followup_4y", "resale_outbound_event_id", "resale_outbound_date",
        "resale_outbound_to_club_id", "resale_outbound_fee_status", "resale_outbound_fee_eur",
        "resale_outbound_fee_conflict", "days_to_permanent_disposal",
        "permanent_disposal_within_4y", "reliable_positive_outbound_fee",
        "fee_only_resale_multiple", "fee_only_resale_gain_eur",
    ] + [f"eligible_{module}" for module in masks]
    return frame.loc[frame["is_clean_permanent"], columns].copy()


def make_readme(summary: pd.DataFrame, frame: pd.DataFrame, observed_max: pd.Timestamp) -> str:
    indexed = summary.set_index("module_id")
    def n(module: str) -> int:
        return int(indexed.loc[module, "eligible_rows"])
    value_rate = indexed.loc["market_value_downside_24m", "outcome_positive_or_preservation_rate"]
    under_rate = indexed.loc["paid_transfer_underutilization_24m", "outcome_positive_or_preservation_rate"]
    dual_rate = indexed.loc["dual_capital_at_risk_24m", "outcome_positive_or_preservation_rate"]
    return f"""# MoveMaker financial module feasibility audit

This audit tests whether the current data can define sufficiently large, time-safe cohorts for commercial product modules. It does **not** establish predictive accuracy; every module that advances must still pass rolling-origin modeling against simple baselines.

## Decision

- **Advance market-value downside to modeling:** {n('market_value_downside_24m'):,} strict 2017+ permanent transfers have positive pre-transfer value within 365 days and a positive 24-month value within +/-90 days. {value_rate:.1%} declined by at least 25%.
- **Advance paid-transfer under-utilization to modeling:** {n('paid_transfer_underutilization_24m'):,} paid permanent arrivals have an interpretable 24-month destination opportunity outcome. {under_rate:.1%} received under 10% of available match-minute capacity.
- **Advance fee benchmarking to modeling:** {n('fee_benchmarking'):,} paid permanent arrivals have a reliable positive fee and recent pre-transfer valuation. The target is reported fee, not total deal cost.
- **Test dual capital-at-risk only after its components:** {n('dual_capital_at_risk_24m'):,} rows have both strict value and paid-opportunity outcomes; {dual_rate:.1%} meet both severe-risk thresholds.
- **Keep resale return exploratory:** only {n('fee_only_resale_return_4y'):,} fully followed acquisitions have both positive inbound and later positive outbound fees. This is conditional on a disclosed sale and is selection-biased.
- **Do not claim full ROI:** wages, bonuses, agent fees, add-ons, amortization, contract book value, and other operating costs are absent. No row is labeled as accounting ROI.

## Frozen definitions

- Primary era: transfers on or after {MODELING_ERA_START.date()}.
- Permanent arrival: canonical type `transfer`, cross-club, with no event or type conflict.
- Reliable paid fee: canonical status `positive_reported`, value > 0, and no fee conflict. Literal zero remains unclassified.
- Strict value target: positive valuation within 365 days before transfer and nearest positive valuation within +/-90 days of the 12- or 24-month target, with the horizon observable.
- Under-utilization: destination minutes divided by `90 * destination club games`; primary severe threshold is <10% over 24 months, with at least 20 destination games observed.
- Fee-only resale: first later permanent outbound from the acquired club within four years, restricted to acquisitions with full follow-up through the observed canonical transfer maximum of {observed_max.date()} and positive disclosed fees on both legs.

## Feasibility gates

The audit reports a sample gate (>= {MIN_TOTAL_ROWS} rows), a temporal gate (>= {MIN_ROWS_PER_EVAL_YEAR} rows in at least {MIN_ADEQUATE_EVAL_YEARS} of {EVALUATION_YEARS}), an outcome-variation gate, and a target-definition gate. These gates authorize a model test, not a product-performance claim.

See `module_feasibility_summary.csv` for decisions, `module_funnels.csv` for attrition, `valuation_timing_sensitivity.csv` for strict/relaxed timing, `outcome_class_balance.csv` for thresholds, `evidence_overlap_summary.csv` for sporting/context/retention overlap, and `financial_event_audit.csv` for row-level provenance.
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    protected_before = protected_tree_sha256()
    frame, canonical, observed_canonical_max = load_and_reconcile()
    masks = module_masks(frame)

    outputs: dict[str, pd.DataFrame] = {
        "financial_event_audit.csv": event_audit_columns(frame, masks),
        "module_feasibility_summary.csv": build_module_summary(frame, masks),
        "module_funnels.csv": build_funnels(frame),
        "valuation_timing_sensitivity.csv": build_valuation_sensitivity(frame),
        "outcome_class_balance.csv": build_outcome_balance(frame, masks),
        "coverage_by_transfer_year.csv": build_year_coverage(frame, masks),
        "module_overlap_matrix.csv": build_overlap_matrix(masks),
        "evidence_overlap_summary.csv": build_evidence_overlap(frame, masks),
        "module_subgroup_coverage.csv": build_subgroup_coverage(frame, masks),
        "build_checks.csv": build_checks(frame, canonical, masks),
    }
    for name, table in outputs.items():
        table.to_csv(OUTPUT_DIR / name, index=False)

    source_manifest = pd.DataFrame(
        [
            {"source": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in [MASTER_PATH, CANONICAL_PATH, TARGETS_PATH, RETENTION_PATH]
        ]
    )
    source_manifest.to_csv(OUTPUT_DIR / "source_manifest.csv", index=False)
    outputs["source_manifest.csv"] = source_manifest

    module_summary = outputs["module_feasibility_summary.csv"]
    readme = make_readme(module_summary, frame, observed_canonical_max)
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    protected_after = protected_tree_sha256()
    checks = outputs["build_checks.csv"]
    if not checks["passed"].all():
        raise RuntimeError(f"Financial audit checks failed: {checks.loc[~checks['passed'], 'check'].tolist()}")
    if protected_before != protected_after:
        raise RuntimeError("A protected verified output tree changed during the financial audit.")

    run_summary = {
        "as_of_date": AS_OF_DATE.date().isoformat(),
        "modeling_era_start": MODELING_ERA_START.date().isoformat(),
        "observed_canonical_transfer_max_date": observed_canonical_max.date().isoformat(),
        "master_rows": len(frame),
        "clean_permanent_rows": int(frame["is_clean_permanent"].sum()),
        "modeling_era_clean_permanent_rows": int((frame["is_clean_permanent"] & frame["is_modeling_era"]).sum()),
        "module_rows": {key: int(mask.sum()) for key, mask in masks.items()},
        "protected_outputs_unchanged": True,
        "build_checks": len(checks),
        "build_checks_passed": int(checks["passed"].sum()),
    }
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    manifest_names = sorted(list(outputs) + ["README.md", "run_summary.json"])
    manifest = pd.DataFrame(
        [
            {"file": name, "size_bytes": (OUTPUT_DIR / name).stat().st_size, "sha256": sha256_file(OUTPUT_DIR / name)}
            for name in manifest_names
        ]
    )
    manifest.to_csv(OUTPUT_DIR / "output_manifest.csv", index=False)
    print(json.dumps(run_summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

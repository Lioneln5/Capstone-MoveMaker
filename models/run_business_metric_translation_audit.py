"""Audit business-facing translations of the verified extension profile.

This is a product-metric audit, not a new outcome-model search. It consumes
only out-of-time Phase 1-5 outputs and asks whether supported probabilities and
contract facts can be translated into concise, financially meaningful metrics
without turning proxies into accounting loss, ROI, or an overall deal score.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Data" / "processed" / "business_metric_translation_audit"
PHASE5 = ROOT / "Data" / "processed" / "integrated_contract_profile"
PHASE1 = ROOT / "Data" / "processed" / "extension_opportunity_diagnostic"
PHASE2 = ROOT / "Data" / "processed" / "extension_survival_diagnostic"
PHASE3 = ROOT / "Data" / "processed" / "extension_value_preservation_diagnostic"
PHASE4 = ROOT / "Data" / "processed" / "contract_financial_exposure_benchmark"
MASTER = ROOT / "Data" / "processed" / "contract_extension_integration" / "extension_modeling_master.csv"
M0 = "M0_chronological_baseline"
M5 = "M5_plus_relative_financial_context"
BOOTSTRAP_REPETITIONS = 5000
RANDOM_SEED = 20260812


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").fillna("").str.strip().str.casefold().isin({"true", "1", "yes"})


def load_verified_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for directory in [PHASE1, PHASE2, PHASE3, PHASE4, PHASE5]:
        payload = json.loads((directory / "independent_verification.json").read_text(encoding="utf-8"))
        if not payload.get("all_checks_passed"):
            raise RuntimeError(f"Required source has not passed independent verification: {directory}")

    profiles = pd.read_csv(PHASE5 / "historical_integrated_profiles.csv", low_memory=False)
    cohort = pd.read_csv(PHASE5 / "integration_evaluation_cohort.csv", low_memory=False)
    master = pd.read_csv(MASTER, low_memory=False)
    return profiles, cohort, master


def add_business_metrics(profiles: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    profile = profiles.copy()
    prior = master[[
        "capology_extension_event_id", "prior_season_salary_annual_gross_eur",
    ]].drop_duplicates("capology_extension_event_id")
    profile = profile.merge(prior, on="capology_extension_event_id", how="left", validate="one_to_one")

    annual = pd.to_numeric(profile["annual_gross_eur"], errors="coerce")
    duration = pd.to_numeric(profile["exact_duration_years"], errors="coerce")
    market = pd.to_numeric(profile["at_signing_market_value_eur"], errors="coerce").where(lambda x: x.gt(0))
    prior_wage = pd.to_numeric(profile["prior_season_salary_annual_gross_eur"], errors="coerce").where(lambda x: x.gt(0))
    peer_commitment = pd.to_numeric(profile["fixed_commitment_peer_estimate_eur"], errors="coerce").where(lambda x: x.gt(0))

    profile["prior_wage_change_eur"] = annual - prior_wage
    profile["prior_wage_change_pct"] = annual.div(prior_wage) - 1
    profile["fixed_commitment_peer_difference_eur"] = profile["fixed_wage_commitment_eur"] - peer_commitment
    profile["fixed_commitment_peer_residual_log"] = np.log(profile["fixed_wage_commitment_eur"].div(peer_commitment))
    eligible_peer = profile["fixed_commitment_peer_residual_log"].notna()
    profile["fixed_commitment_peer_percentile"] = np.nan
    profile.loc[eligible_peer, "fixed_commitment_peer_percentile"] = profile.loc[
        eligible_peer, "fixed_commitment_peer_residual_log"
    ].rank(pct=True, method="average")
    profile["fixed_commitment_peer_position"] = np.select(
        [
            profile["fixed_wage_commitment_eur"] < profile["fixed_commitment_peer_lower_eur"],
            profile["fixed_wage_commitment_eur"] > profile["fixed_commitment_peer_upper_eur"],
        ],
        ["below_historical_80pct_range", "above_historical_80pct_range"],
        default="within_historical_80pct_range",
    )
    profile.loc[
        profile[["fixed_wage_commitment_eur", "fixed_commitment_peer_lower_eur", "fixed_commitment_peer_upper_eur"]].isna().any(axis=1),
        "fixed_commitment_peer_position",
    ] = "unavailable"

    profile["fixed_wages_through_24m_eur"] = annual * duration.clip(lower=0, upper=2)
    profile["low_contribution_probability_24m"] = 1 - profile["sustained_contribution_probability"]
    profile["risk_weighted_low_contribution_wage_exposure_eur"] = (
        profile["fixed_wages_through_24m_eur"] * profile["low_contribution_probability_24m"]
    )

    profile["public_value_downside_floor_25pct_eur"] = market * 0.25
    profile["public_value_downside_floor_50pct_eur"] = market * 0.50

    profile["contract_years_beyond_24m"] = (duration - 2).clip(lower=0)
    profile["contract_years_beyond_36m"] = (duration - 3).clip(lower=0)
    profile["scheduled_fixed_wages_after_24m_eur"] = annual * profile["contract_years_beyond_24m"]
    profile["scheduled_fixed_wages_after_36m_eur"] = annual * profile["contract_years_beyond_36m"]
    profile["outbound_probability_24m"] = 1 - profile["continuous_stay_probability_24m"]
    profile["outbound_probability_36m"] = 1 - profile["continuous_stay_probability_36m"]

    # These display strings deliberately pair, rather than multiply, probability
    # and financial severity. They are useful scenarios, not expected-loss claims.
    profile["value_downside_25pct_display"] = np.where(
        profile["value_downside_25pct_probability_24m"].notna() & profile["public_value_downside_floor_25pct_eur"].notna(),
        profile["value_downside_25pct_probability_24m"].map(lambda x: f"{x:.0%}")
        + " probability of at least €"
        + profile["public_value_downside_floor_25pct_eur"].map(lambda x: f"{x:,.0f}")
        + " public-value downside within 24 months",
        "Unavailable",
    )
    profile["continuity_commitment_display"] = np.where(
        profile["continuous_stay_probability_36m"].notna() & profile["scheduled_fixed_wages_after_36m_eur"].notna(),
        profile["continuous_stay_probability_36m"].map(lambda x: f"{x:.0%}")
        + " continuous-stay probability through 36 months; €"
        + profile["scheduled_fixed_wages_after_36m_eur"].map(lambda x: f"{x:,.0f}")
        + " scheduled fixed wages after month 36",
        "Unavailable",
    )
    return profile


def bootstrap_improvements(frame: pd.DataFrame) -> dict[str, float]:
    clusters = [group.index.to_numpy() for _, group in frame.groupby("canonical_player_id", sort=True)]
    mae_delta = (
        np.abs(frame["baseline_exposure_eur"] - frame["actual_low_contribution_exposure_eur"])
        - np.abs(frame["selected_exposure_eur"] - frame["actual_low_contribution_exposure_eur"])
    ).to_numpy()
    weight = frame["fixed_wages_through_24m_eur"].to_numpy()
    actual = frame["actual_low_contribution"].to_numpy()
    baseline = frame["baseline_low_contribution_probability"].to_numpy()
    selected = frame["selected_low_contribution_probability"].to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    mae_draws: list[float] = []
    brier_draws: list[float] = []
    for _ in range(BOOTSTRAP_REPETITIONS):
        indices = np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))])
        mae_draws.append(float(mae_delta[indices].mean()))
        total_weight = float(weight[indices].sum())
        base_loss = float(np.sum(weight[indices] * (actual[indices] - baseline[indices]) ** 2) / total_weight)
        selected_loss = float(np.sum(weight[indices] * (actual[indices] - selected[indices]) ** 2) / total_weight)
        brier_draws.append(base_loss - selected_loss)
    mae_values = np.asarray(mae_draws)
    brier_values = np.asarray(brier_draws)
    return {
        "mae_improvement_ci_lower_95": float(np.quantile(mae_values, 0.025)),
        "mae_improvement_ci_upper_95": float(np.quantile(mae_values, 0.975)),
        "weighted_brier_improvement_ci_lower_95": float(np.quantile(brier_values, 0.025)),
        "weighted_brier_improvement_ci_upper_95": float(np.quantile(brier_values, 0.975)),
        "bootstrap_repetitions": len(mae_values),
        "player_clusters": len(clusters),
    }


def top_quintile_summary(frame: pd.DataFrame, score: str, label: str) -> dict[str, Any]:
    threshold = float(frame[score].quantile(0.80))
    top = frame[score].ge(threshold)
    realized = frame["actual_low_contribution_exposure_eur"]
    commitment = frame["fixed_wages_through_24m_eur"]
    capture = float(realized[top].sum() / realized.sum()) if realized.sum() else np.nan
    capital_share = float(commitment[top].sum() / commitment.sum()) if commitment.sum() else np.nan
    return {
        "model": label,
        "top_quintile_rows": int(top.sum()),
        "score_threshold_eur": threshold,
        "realized_exposure_capture": capture,
        "capital_share": capital_share,
        "capture_lift_vs_capital_share": capture / capital_share if capital_share else np.nan,
        "top_group_commitment_weighted_event_rate": float(
            np.sum(commitment[top] * frame.loc[top, "actual_low_contribution"]) / commitment[top].sum()
        ),
        "remainder_commitment_weighted_event_rate": float(
            np.sum(commitment[~top] * frame.loc[~top, "actual_low_contribution"]) / commitment[~top].sum()
        ),
    }


def audit_low_contribution_exposure(profile: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    predictions = pd.read_csv(PHASE1 / "model_predictions.csv", low_memory=False)
    predictions = predictions.loc[
        predictions["endpoint"].eq("sustained_meaningful_contribution")
        & predictions["model_variant"].isin([M0, M5])
    ].copy()
    wide = predictions.pivot(
        index=["capology_extension_event_id", "canonical_player_id", "evaluation_year"],
        columns="model_variant", values="prediction",
    ).reset_index()
    actual = predictions.loc[predictions["model_variant"].eq(M5), [
        "capology_extension_event_id", "actual",
    ]].rename(columns={"actual": "actual_sustained_contribution"})
    audit = wide.merge(actual, on="capology_extension_event_id", validate="one_to_one")
    audit = audit.merge(profile[[
        "capology_extension_event_id", "player_name", "club_name", "league", "canonical_position",
        "annual_gross_eur", "exact_duration_years", "fixed_wages_through_24m_eur",
    ]], on="capology_extension_event_id", validate="one_to_one")
    audit["actual_low_contribution"] = 1 - audit["actual_sustained_contribution"]
    audit["baseline_low_contribution_probability"] = 1 - audit[M0]
    audit["selected_low_contribution_probability"] = 1 - audit[M5]
    audit["actual_low_contribution_exposure_eur"] = audit["fixed_wages_through_24m_eur"] * audit["actual_low_contribution"]
    audit["baseline_exposure_eur"] = audit["fixed_wages_through_24m_eur"] * audit["baseline_low_contribution_probability"]
    audit["selected_exposure_eur"] = audit["fixed_wages_through_24m_eur"] * audit["selected_low_contribution_probability"]
    audit = audit.dropna(subset=[
        "fixed_wages_through_24m_eur", "actual_low_contribution",
        "baseline_low_contribution_probability", "selected_low_contribution_probability",
        "actual_low_contribution_exposure_eur", "baseline_exposure_eur", "selected_exposure_eur",
    ]).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for origin, group in [("pooled", audit), *[(str(year), g) for year, g in audit.groupby("evaluation_year")]]:
        commitment = group["fixed_wages_through_24m_eur"]
        actual_event = group["actual_low_contribution"]
        baseline_probability = group["baseline_low_contribution_probability"]
        selected_probability = group["selected_low_contribution_probability"]
        actual_exposure = group["actual_low_contribution_exposure_eur"]
        baseline_exposure = group["baseline_exposure_eur"]
        selected_exposure = group["selected_exposure_eur"]
        total_commitment = float(commitment.sum())
        baseline_mae = float(np.mean(np.abs(baseline_exposure - actual_exposure)))
        selected_mae = float(np.mean(np.abs(selected_exposure - actual_exposure)))
        baseline_brier = float(np.sum(commitment * (actual_event - baseline_probability) ** 2) / total_commitment)
        selected_brier = float(np.sum(commitment * (actual_event - selected_probability) ** 2) / total_commitment)
        rows.append({
            "evaluation": origin,
            "rows": len(group),
            "players": group["canonical_player_id"].nunique(),
            "total_24m_fixed_wage_commitment_eur": total_commitment,
            "realized_low_contribution_exposure_eur": float(actual_exposure.sum()),
            "baseline_predicted_exposure_eur": float(baseline_exposure.sum()),
            "selected_predicted_exposure_eur": float(selected_exposure.sum()),
            "baseline_exposure_mae_eur": baseline_mae,
            "selected_exposure_mae_eur": selected_mae,
            "exposure_mae_improvement_eur": baseline_mae - selected_mae,
            "baseline_capital_weighted_brier": baseline_brier,
            "selected_capital_weighted_brier": selected_brier,
            "capital_weighted_brier_improvement": baseline_brier - selected_brier,
            "selected_aggregate_calibration_ratio": float(selected_exposure.sum() / actual_exposure.sum()),
            "selected_exposure_spearman": float(selected_exposure.corr(actual_exposure, method="spearman")),
        })
    performance = pd.DataFrame(rows)
    audit["exposure_band"] = pd.qcut(
        audit["selected_exposure_eur"].rank(method="first"), 5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)
    band_rows = []
    for band, group in audit.groupby("exposure_band", observed=True):
        predicted = float(group["selected_exposure_eur"].mean())
        realized = float(group["actual_low_contribution_exposure_eur"].mean())
        band_rows.append({
            "exposure_band": int(band), "rows": len(group),
            "mean_predicted_exposure_eur": predicted,
            "mean_realized_exposure_eur": realized,
            "predicted_to_realized_ratio": predicted / realized if realized else np.nan,
            "mean_24m_fixed_wage_commitment_eur": float(group["fixed_wages_through_24m_eur"].mean()),
            "mean_low_contribution_probability": float(group["selected_low_contribution_probability"].mean()),
            "observed_low_contribution_rate": float(group["actual_low_contribution"].mean()),
        })
    bands = pd.DataFrame(band_rows)
    realized_bands_monotone = bool(bands["mean_realized_exposure_eur"].is_monotonic_increasing)
    bootstrap = bootstrap_improvements(audit)
    pooled = performance.loc[performance["evaluation"].eq("pooled")].iloc[0]
    origin_rows = performance.loc[~performance["evaluation"].eq("pooled")]
    origin_wins = int(origin_rows["exposure_mae_improvement_eur"].gt(0).sum())
    weighted_origin_wins = int(origin_rows["capital_weighted_brier_improvement"].gt(0).sum())
    concentration = pd.DataFrame([
        top_quintile_summary(audit, "fixed_wages_through_24m_eur", "capital_only"),
        top_quintile_summary(audit, "baseline_exposure_eur", "chronological_baseline"),
        top_quintile_summary(audit, "selected_exposure_eur", "selected_phase1_model"),
    ])
    capital_row = concentration.loc[concentration["model"].eq("capital_only")].iloc[0]
    selected_row = concentration.loc[concentration["model"].eq("selected_phase1_model")].iloc[0]
    improves_capital_targeting = (
        selected_row["realized_exposure_capture"] >= capital_row["realized_exposure_capture"]
        and selected_row["capital_share"] < capital_row["capital_share"]
    )
    supported = (
        pooled["exposure_mae_improvement_eur"] > 0
        and bootstrap["mae_improvement_ci_lower_95"] > 0
        and pooled["capital_weighted_brier_improvement"] > 0
        and bootstrap["weighted_brier_improvement_ci_lower_95"] > 0
        and origin_wins >= 3
        and weighted_origin_wins >= 3
        and 0.75 <= pooled["selected_aggregate_calibration_ratio"] <= 1.25
        and realized_bands_monotone
        and improves_capital_targeting
    )
    decision = "advance_risk_weighted_exposure_proxy" if supported else "show_probability_and_commitment_separately"
    comparison = pd.DataFrame([{
        **bootstrap,
        "origin_mae_wins": origin_wins,
        "origin_weighted_brier_wins": weighted_origin_wins,
        "realized_exposure_bands_monotone": realized_bands_monotone,
        "improves_targeting_over_capital_only": improves_capital_targeting,
        "decision": decision,
        "interpretation_boundary": "Risk-weighted fixed-wage exposure proxy; not expected accounting loss or wasted wages.",
    }])
    keep = [
        "capology_extension_event_id", "canonical_player_id", "evaluation_year", "player_name", "club_name",
        "league", "canonical_position", "annual_gross_eur", "exact_duration_years",
        "fixed_wages_through_24m_eur", "actual_low_contribution",
        "baseline_low_contribution_probability", "selected_low_contribution_probability",
        "actual_low_contribution_exposure_eur", "baseline_exposure_eur", "selected_exposure_eur",
    ]
    return audit[keep], performance, pd.concat([comparison, concentration], axis=0, ignore_index=True), bands, decision


def metric_coverage(profile: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "age_at_expiration", "fixed_wage_commitment_eur", "fixed_commitment_to_market_value",
        "prior_wage_change_pct", "fixed_commitment_peer_difference_eur", "fixed_commitment_peer_percentile",
        "risk_weighted_low_contribution_wage_exposure_eur", "public_value_downside_floor_25pct_eur",
        "value_downside_25pct_probability_24m", "scheduled_fixed_wages_after_24m_eur",
        "scheduled_fixed_wages_after_36m_eur", "continuous_stay_probability_36m",
        "club_salary_percentile", "club_salary_share_known",
    ]
    rows = []
    for field in fields:
        values = pd.to_numeric(profile[field], errors="coerce")
        eligible = values.notna()
        rows.append({
            "metric": field,
            "available_rows": int(eligible.sum()),
            "profile_rows": len(profile),
            "coverage": float(eligible.mean()),
            "p10": float(values[eligible].quantile(0.10)) if eligible.any() else np.nan,
            "median": float(values[eligible].median()) if eligible.any() else np.nan,
            "p90": float(values[eligible].quantile(0.90)) if eligible.any() else np.nan,
            "minimum": float(values[eligible].min()) if eligible.any() else np.nan,
            "maximum": float(values[eligible].max()) if eligible.any() else np.nan,
        })
    return pd.DataFrame(rows)


def scenario_audits(profile: pd.DataFrame, cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    value_rows = []
    eligible = cohort[[
        "at_signing_market_value_eur", "value_downside_25pct_probability_24m",
        "value_downside_50pct_probability_24m", "actual_downside_25pct_24m",
        "actual_downside_50pct_24m", "actual_value_log_ratio_24m",
    ]].notna().all(axis=1)
    value = cohort.loc[eligible].copy()
    value["downside_floor_25_eur"] = value["at_signing_market_value_eur"] * 0.25
    value["downside_floor_50_eur"] = value["at_signing_market_value_eur"] * 0.50
    value["risk_band"] = pd.qcut(
        value["value_downside_25pct_probability_24m"].rank(method="first"), 5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)
    for band, group in value.groupby("risk_band", observed=True):
        value_rows.append({
            "risk_band": int(band), "rows": len(group),
            "mean_predicted_25pct_downside_probability": float(group["value_downside_25pct_probability_24m"].mean()),
            "observed_25pct_downside_rate": float(group["actual_downside_25pct_24m"].mean()),
            "median_25pct_downside_floor_eur": float(group["downside_floor_25_eur"].median()),
            "mean_predicted_50pct_downside_probability": float(group["value_downside_50pct_probability_24m"].mean()),
            "observed_50pct_downside_rate": float(group["actual_downside_50pct_24m"].mean()),
            "median_50pct_downside_floor_eur": float(group["downside_floor_50_eur"].median()),
        })
    value_audit = pd.DataFrame(value_rows)

    survival = pd.read_csv(PHASE2 / "survival_predictions.csv", low_memory=False)
    survival = survival.loc[
        survival["model_variant"].eq(M5)
        & survival["endpoint"].eq("any_outbound")
        & survival["horizon"].isin(["24m", "36m"])
    ].copy()
    finance = profile[[
        "capology_extension_event_id", "annual_gross_eur", "exact_duration_years",
        "scheduled_fixed_wages_after_24m_eur", "scheduled_fixed_wages_after_36m_eur",
    ]]
    survival = survival.merge(finance, on="capology_extension_event_id", validate="many_to_one")
    continuity_rows = []
    for horizon, group in survival.groupby("horizon", observed=True):
        suffix = "24m" if horizon == "24m" else "36m"
        group = group.loc[group["actual_event_by_horizon"].notna()].copy()
        group["risk_band"] = pd.qcut(
            group["predicted_event_probability"].rank(method="first"), 5,
            labels=[1, 2, 3, 4, 5],
        ).astype(int)
        scheduled = f"scheduled_fixed_wages_after_{suffix}_eur"
        for band, band_frame in group.groupby("risk_band", observed=True):
            continuity_rows.append({
                "horizon": horizon, "outbound_risk_band": int(band), "rows": len(band_frame),
                "mean_predicted_outbound_probability": float(band_frame["predicted_event_probability"].mean()),
                "observed_outbound_rate": float(band_frame["actual_event_by_horizon"].mean()),
                "share_with_contract_beyond_horizon": float(band_frame[scheduled].gt(0).mean()),
                "median_scheduled_fixed_wages_after_horizon_eur": float(band_frame[scheduled].median()),
            })
    return value_audit, pd.DataFrame(continuity_rows)


def metric_catalog(exposure_decision: str) -> pd.DataFrame:
    exposure_status = "supported_proxy" if exposure_decision == "advance_risk_weighted_exposure_proxy" else "rejected_as_single_euro_metric"
    rows = [
        ("fixed_wage_commitment_eur", "Fixed-wage commitment", "capital_commitment", "exact_calculation", "annual fixed wage × proposed years", "EUR", "headline", "supported", "How much fixed wage is scheduled?", "Partial fixed-wage commitment; not total contract cost."),
        ("fixed_commitment_to_market_value", "Commitment / public value", "capital_commitment", "exact_ratio_proxy", "fixed-wage commitment / current public market value", "percent", "headline", "supported_proxy", "How large is the commitment relative to the public asset estimate?", "Not ROI, book value, or accounting exposure."),
        ("prior_wage_change_pct", "Proposed wage increase", "offer_change", "exact_calculation", "proposed annual wage / prior annual wage − 1", "percent", "detail", "supported_when_prior_wage_known", "How much does fixed pay change?", "Public prior wage may be missing or estimated."),
        ("fixed_commitment_peer_difference_eur", "Commitment versus historical peers", "peer_context", "historical_benchmark_translation", "proposed fixed commitment − modeled peer commitment", "EUR", "headline", "supported", "How unusual is the offer in euro terms?", "Historical deviation is not proof of overpayment."),
        ("fixed_commitment_peer_percentile", "Offer aggressiveness percentile", "peer_context", "historical_oos_reference", "percentile of log commitment residual in frozen out-of-time reference distribution", "percentile", "headline", "supported_after_frozen_reference_cdf", "Where does the offer sit among historical extension deviations?", "Not an optimality, fairness, or causal score."),
        ("risk_weighted_low_contribution_wage_exposure_eur", "Low-contribution wage exposure", "contribution_exposure", "probability_weighted_proxy", "fixed wages through 24m × P(not meaningful contributor through Year 2)", "EUR", "headline", exposure_status, "How much two-year fixed commitment is exposed to low contribution risk?", "Not expected accounting loss or wasted wages."),
        ("value_downside_25pct_display", "24-month public-value downside scenario", "asset_value", "paired_probability_scenario", "P(≥25% decline) displayed with 25% × current public market value", "probability_and_EUR", "headline", "supported_paired_display", "What is the probability and minimum threshold magnitude of a material public-value decline?", "Do not multiply into expected loss; not resale proceeds or impairment."),
        ("continuity_commitment_display", "Contract-continuity horizon", "continuity", "paired_probability_scenario", "36m continuous-stay probability displayed with scheduled fixed wages after month 36", "probability_and_EUR", "headline", "supported_paired_display", "How much contract extends beyond the modeled continuity horizon?", "Outbound movement can stop wages or produce a fee; this is not expected loss."),
        ("club_salary_percentile", "Club salary position", "salary_structure", "observed_context", "percentile among known club salary panel", "percentile", "detail", "supported_when_matched", "Where would the player sit in the club wage hierarchy?", "Known public salary panel may be incomplete."),
        ("club_salary_share_known", "Share of known club payroll", "salary_structure", "observed_context", "annual fixed wage / total known club fixed wages", "percent", "detail", "supported_when_matched", "How concentrated is known payroll in this player?", "Denominator is not audited total payroll."),
        ("overall_contract_score", "Overall contract score", "overall", "unsupported_composite", "none", "none", "never", "rejected", "None", "Phase 5 found no incremental support for a headline composite."),
    ]
    return pd.DataFrame(rows, columns=[
        "metric_key", "display_label", "metric_family", "evidence_class", "formula", "unit",
        "display_priority", "status", "decision_question", "interpretation_boundary",
    ])


def decision_summary(profile: pd.DataFrame, exposure_performance: pd.DataFrame, exposure_comparison: pd.DataFrame, exposure_decision: str) -> pd.DataFrame:
    pooled = exposure_performance.loc[exposure_performance["evaluation"].eq("pooled")].iloc[0]
    comparison = exposure_comparison.loc[exposure_comparison["decision"].notna()].iloc[0]
    peer_n = int(profile["fixed_commitment_peer_percentile"].notna().sum())
    value_n = int((profile["value_downside_25pct_probability_24m"].notna() & profile["public_value_downside_floor_25pct_eur"].notna()).sum())
    continuity_n = int((profile["continuous_stay_probability_36m"].notna() & profile["scheduled_fixed_wages_after_36m_eur"].notna()).sum())
    salary_n = int(profile["club_salary_percentile"].notna().sum())
    rows = [
        ("capital_commitment", len(profile), "advance", "Exact guarded contract arithmetic is ready for application logic."),
        ("historical_offer_context", peer_n, "advance_after_freezing_reference_cdf", "Show euro/percent deviation, range position, and a frozen historical offer-aggressiveness percentile; never call it fair price."),
        ("low_contribution_wage_exposure", int(pooled["rows"]), exposure_decision, f"Selected-vs-baseline exposure MAE improvement €{pooled['exposure_mae_improvement_eur']:,.0f}; 95% CI [€{comparison['mae_improvement_ci_lower_95']:,.0f}, €{comparison['mae_improvement_ci_upper_95']:,.0f}]; {int(comparison['origin_mae_wins'])}/4 origin wins."),
        ("public_value_downside_scenario", value_n, "advance_probability_plus_threshold_amount", "Show probability and threshold euro magnitude together; reject a probability-weighted expected-loss or ROI label."),
        ("contract_continuity_horizon", continuity_n, "advance_probability_plus_scheduled_commitment", "Show stay probability beside scheduled wages beyond 36 months; do not multiply them into expected wage loss."),
        ("club_salary_structure", salary_n, "advance_when_matched", "Show known-panel percentile/share with a public-data completeness warning."),
        ("overall_contract_score", 0, "reject", "No single score passed Phase 5 and financially translated modules remain intentionally separate."),
        ("bettor_use_case", 0, "reject_out_of_scope", "Multi-season extension outcomes are not match, odds, or betting-market predictions."),
    ]
    return pd.DataFrame(rows, columns=["metric_group", "eligible_rows", "decision", "evidence_and_boundary"])


def web_metric_specification(catalog: pd.DataFrame) -> pd.DataFrame:
    display = catalog.loc[~catalog["display_priority"].eq("never")].copy()
    display["display_rule"] = display["metric_key"].map({
        "fixed_wage_commitment_eur": "Always when annual fixed wage and term are valid.",
        "fixed_commitment_to_market_value": "Show when dated positive public market value is available.",
        "prior_wage_change_pct": "Show as secondary detail only when prior wage is known; never impute zero.",
        "fixed_commitment_peer_difference_eur": "Show with peer range, percentile, cohort size, and historical label.",
        "fixed_commitment_peer_percentile": "Use a frozen out-of-time residual CDF; return unavailable until production CDF is serialized.",
        "risk_weighted_low_contribution_wage_exposure_eur": "Show only if audit decision advances the proxy; always disclose probability and 24m commitment components.",
        "value_downside_25pct_display": "Show probability and threshold amount together, never as expected cash loss.",
        "continuity_commitment_display": "Show 36m stay probability and scheduled post-36m wages together, never as a multiplied loss.",
        "club_salary_percentile": "Show only with a matched same-club salary panel.",
        "club_salary_share_known": "Show only with a matched panel and known-payroll denominator warning.",
    })
    return display[[
        "metric_key", "display_label", "metric_family", "evidence_class", "display_priority",
        "status", "display_rule", "decision_question", "interpretation_boundary",
    ]]


def select_cases(profile: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    base_cases = pd.read_csv(PHASE5 / "historical_case_studies.csv", low_memory=False)
    translated = profile[[
        "capology_extension_event_id", "fixed_wage_commitment_eur", "fixed_commitment_to_market_value",
        "fixed_commitment_peer_difference_eur", "fixed_commitment_peer_percentile", "fixed_commitment_peer_position",
        "fixed_wages_through_24m_eur", "low_contribution_probability_24m",
        "risk_weighted_low_contribution_wage_exposure_eur", "value_downside_25pct_display",
        "public_value_downside_floor_25pct_eur", "continuous_stay_probability_36m",
        "scheduled_fixed_wages_after_36m_eur", "continuity_commitment_display",
        "club_salary_percentile", "club_salary_share_known",
    ]]
    actuals = cohort[[
        "capology_extension_event_id", "actual_low_contribution", "actual_outbound_24m",
        "actual_value_downside_25pct_24m",
    ]]
    cases = base_cases.merge(translated, on="capology_extension_event_id", how="left", validate="one_to_one")
    cases = cases.drop(columns=[column for column in actuals.columns if column != "capology_extension_event_id" and column in cases.columns])
    cases = cases.merge(actuals, on="capology_extension_event_id", how="left", validate="one_to_one")
    return cases


def build_checks(profile: pd.DataFrame, cohort: pd.DataFrame, exposure: pd.DataFrame, exposure_bands: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def add(name: str, observed: Any, expected: Any, passed: bool, notes: str) -> None:
        checks.append({"check": name, "observed": observed, "expected": expected, "passed": bool(passed), "notes": notes})

    add("profile_rows", len(profile), 1560, len(profile) == 1560, "One Phase-5 out-of-time extension profile per event.")
    add("profile_event_grain", int(profile["capology_extension_event_id"].duplicated().sum()), 0, profile["capology_extension_event_id"].is_unique, "No translated-profile duplication.")
    fixed_formula = np.allclose(profile["fixed_wage_commitment_eur"], profile["annual_gross_eur"] * profile["exact_duration_years"], equal_nan=True)
    add("fixed_commitment_formula", int(fixed_formula), 1, fixed_formula, "Annual wage × term reproduces Phase 5.")
    horizon_formula = np.allclose(profile["fixed_wages_through_24m_eur"], profile["annual_gross_eur"] * profile["exact_duration_years"].clip(upper=2), equal_nan=True)
    add("two_year_commitment_formula", int(horizon_formula), 1, horizon_formula, "Two-year exposure never exceeds entered contract duration.")
    exposure_formula = np.allclose(profile["risk_weighted_low_contribution_wage_exposure_eur"], profile["fixed_wages_through_24m_eur"] * (1 - profile["sustained_contribution_probability"]), equal_nan=True)
    add("low_contribution_exposure_formula", int(exposure_formula), 1, exposure_formula, "Probability-weighted proxy formula reproduces.")
    value_formula = np.allclose(profile["public_value_downside_floor_25pct_eur"], profile["at_signing_market_value_eur"] * .25, equal_nan=True)
    add("value_threshold_formula", int(value_formula), 1, value_formula, "Euro severity is exactly 25% of dated public value.")
    continuity_formula = np.allclose(profile["scheduled_fixed_wages_after_36m_eur"], profile["annual_gross_eur"] * (profile["exact_duration_years"] - 3).clip(lower=0), equal_nan=True)
    add("post_36m_commitment_formula", int(continuity_formula), 1, continuity_formula, "Post-horizon scheduled fixed wages reproduce.")
    peer_bounds = profile["fixed_commitment_peer_percentile"].dropna().between(0, 1).all()
    add("peer_percentile_bounds", int(peer_bounds), 1, peer_bounds, "Historical reference percentiles remain within [0,1].")
    probability_columns = ["low_contribution_probability_24m", "value_downside_25pct_probability_24m", "continuous_stay_probability_36m"]
    probability_ok = all(profile[column].dropna().between(0, 1).all() for column in probability_columns)
    add("probability_bounds", int(probability_ok), 1, probability_ok, "All translated probabilities remain valid.")
    event_semantics = np.array_equal(
        cohort["actual_downside_25pct_24m"].astype(int).to_numpy(),
        np.exp(cohort["actual_value_log_ratio_24m"]).le(.75 + 1e-12).astype(int).to_numpy(),
    )
    add("value_event_semantics", int(event_semantics), 1, event_semantics, "25% scenario event matches the saved 24-month log-value target.")
    expected_exposure_rows = int((
        profile["fixed_wages_through_24m_eur"].notna()
        & profile["sustained_contribution_probability"].notna()
    ).sum())
    add("exposure_evaluation_rows", len(exposure), expected_exposure_rows, len(exposure) == expected_exposure_rows, "All financially eligible Phase-1 out-of-time profiles enter the translation audit; missing wage rows remain unavailable.")
    exposure_monotone = exposure_bands["mean_realized_exposure_eur"].is_monotonic_increasing
    add("exposure_band_ordering", int(exposure_monotone), 1, exposure_monotone, "Realized low-contribution wage exposure increases across the five predicted euro-exposure bands.")
    no_expected_loss = not catalog["display_label"].str.casefold().str.contains("expected loss").any()
    add("no_expected_loss_label", int(no_expected_loss), 1, no_expected_loss, "No proxy is labeled expected loss.")
    composite_rejected = catalog.loc[catalog["metric_key"].eq("overall_contract_score"), "status"].eq("rejected").all()
    add("overall_score_rejected", int(composite_rejected), 1, composite_rejected, "No unsupported headline score is restored.")
    return pd.DataFrame(checks)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw_profiles, cohort, master = load_verified_sources()
    profile = add_business_metrics(raw_profiles, master)
    exposure_rows, exposure_performance, exposure_comparison, exposure_bands, exposure_decision = audit_low_contribution_exposure(profile)
    coverage = metric_coverage(profile)
    value_scenarios, continuity_scenarios = scenario_audits(profile, cohort)
    catalog = metric_catalog(exposure_decision)
    decisions = decision_summary(profile, exposure_performance, exposure_comparison, exposure_decision)
    web_spec = web_metric_specification(catalog)
    cases = select_cases(profile, cohort)
    checks = build_checks(profile, cohort, exposure_rows, exposure_bands, catalog)

    profile_columns = [
        "profile_scope", "capology_extension_event_id", "canonical_player_id", "evaluation_year",
        "player_name", "club_name", "league", "canonical_position", "signed_date", "age_at_signing",
        "age_at_expiration", "at_signing_market_value_eur", "annual_gross_eur", "exact_duration_years",
        "prior_season_salary_annual_gross_eur", "prior_wage_change_eur", "prior_wage_change_pct",
        "fixed_wage_commitment_eur", "fixed_commitment_to_market_value",
        "fixed_commitment_peer_estimate_eur", "fixed_commitment_peer_difference_eur",
        "fixed_commitment_peer_premium", "fixed_commitment_peer_percentile", "fixed_commitment_peer_position",
        "fixed_wages_through_24m_eur", "sustained_contribution_probability", "low_contribution_probability_24m",
        "risk_weighted_low_contribution_wage_exposure_eur", "value_downside_25pct_probability_24m",
        "value_downside_50pct_probability_24m", "public_value_downside_floor_25pct_eur",
        "public_value_downside_floor_50pct_eur", "value_downside_25pct_display",
        "continuous_stay_probability_24m", "continuous_stay_probability_36m", "outbound_probability_24m",
        "outbound_probability_36m", "contract_years_beyond_24m", "contract_years_beyond_36m",
        "scheduled_fixed_wages_after_24m_eur", "scheduled_fixed_wages_after_36m_eur",
        "continuity_commitment_display", "club_salary_percentile", "club_salary_share_known",
    ]
    outputs = {
        "translated_historical_profiles.csv": profile[profile_columns],
        "business_metric_catalog.csv": catalog,
        "business_metric_decision_summary.csv": decisions,
        "web_app_business_metric_specification.csv": web_spec,
        "metric_coverage_and_distribution.csv": coverage,
        "low_contribution_exposure_predictions.csv": exposure_rows,
        "low_contribution_exposure_performance.csv": exposure_performance,
        "low_contribution_exposure_comparison.csv": exposure_comparison,
        "low_contribution_exposure_bands.csv": exposure_bands,
        "value_downside_scenario_audit.csv": value_scenarios,
        "continuity_commitment_scenario_audit.csv": continuity_scenarios,
        "historical_business_profile_cases.csv": cases,
        "build_checks.csv": checks,
    }
    for name, frame in outputs.items():
        write_csv(frame, OUTPUT / name)

    source_paths = [
        MASTER,
        PHASE1 / "model_predictions.csv", PHASE1 / "independent_verification.json",
        PHASE2 / "survival_predictions.csv", PHASE2 / "independent_verification.json",
        PHASE3 / "value_predictions.csv", PHASE3 / "independent_verification.json",
        PHASE4 / "benchmark_predictions.csv", PHASE4 / "independent_verification.json",
        PHASE5 / "historical_integrated_profiles.csv", PHASE5 / "integration_evaluation_cohort.csv",
        PHASE5 / "historical_case_studies.csv", PHASE5 / "independent_verification.json",
    ]
    source_manifest = pd.DataFrame([{
        "source_file": str(path.relative_to(ROOT)), "rows": len(pd.read_csv(path, low_memory=False)) if path.suffix == ".csv" else np.nan,
        "bytes": path.stat().st_size, "sha256": sha256(path),
    } for path in source_paths])
    write_csv(source_manifest, OUTPUT / "source_manifest.csv")

    pooled = exposure_performance.loc[exposure_performance["evaluation"].eq("pooled")].iloc[0]
    comparison = exposure_comparison.loc[exposure_comparison["decision"].notna()].iloc[0]
    readme = f"""# Business metric translation audit

This audit translates the independently verified incumbent-club extension profile into concise financial and business metrics. It does not train or select a sixth outcome model.

## Decision

Exact fixed-wage commitment, commitment/public-value scale, historical offer context, paired public-value downside scenarios, paired contract-continuity scenarios, and matched club salary context advance to the web-product specification.

The low-contribution euro metric decision is **{exposure_decision}**. Across {int(pooled['rows']):,} out-of-time extensions, the selected Phase-1 probability changed exposure MAE by €{pooled['exposure_mae_improvement_eur']:,.0f} versus the chronological baseline, with player-clustered 95% CI [€{comparison['mae_improvement_ci_lower_95']:,.0f}, €{comparison['mae_improvement_ci_upper_95']:,.0f}] and {int(comparison['origin_mae_wins'])}/4 origin wins. Even if advanced, it is a probability-weighted fixed-wage exposure proxy—not expected accounting loss or wasted wages.

The highest selected-proxy quintile captures 54.1% of realized low-contribution exposure while containing 55.2% of two-year fixed commitment. A capital-size-only top quintile captures 53.5% while containing 64.6% of the capital. This capital-only comparison is part of the advancement gate; the proxy does not pass merely because large contracts create large euro values.

## Display architecture

The profile should answer: how much fixed capital is scheduled, how unusual the offer is, what contribution risk attaches to the first two years, what public-value downside scenario is plausible, how much contract extends beyond the continuity horizon, and where the proposed wage sits in the known club pay structure.

Probability and euro severity are multiplied only for the specifically audited low-contribution exposure proxy. Public-value and continuity metrics remain paired displays because a threshold amount or scheduled wage is not loss severity. No overall contract score is supported.

## Scope boundaries

- Incumbent-club extensions in the modeled Big-Five data only.
- Public market value is not sale proceeds, book value, impairment, profit, or ROI.
- Fixed-wage commitment omits bonuses, taxes, agent fees, options, clauses, and other costs.
- Historical peer deviation is not proof of overpayment or an optimal offer.
- These outputs are not betting predictions and are not validated for new-club acquisitions.

Read `business_metric_catalog.csv`, `business_metric_decision_summary.csv`, `web_app_business_metric_specification.csv`, the historical cases, and independent verification before implementation.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")

    summary = {
        "profile_scope": "incumbent_club_extension",
        "historical_out_of_time_profiles": len(profile),
        "low_contribution_exposure_rows": len(exposure_rows),
        "low_contribution_exposure_decision": exposure_decision,
        "value_scenario_rows": int((profile["value_downside_25pct_probability_24m"].notna() & profile["public_value_downside_floor_25pct_eur"].notna()).sum()),
        "continuity_scenario_rows": int((profile["continuous_stay_probability_36m"].notna() & profile["scheduled_fixed_wages_after_36m_eur"].notna()).sum()),
        "historical_case_studies": len(cases),
        "builder_checks_passed": int(as_bool(checks["passed"]).sum()),
        "builder_checks_total": len(checks),
        "all_builder_checks_passed": bool(as_bool(checks["passed"]).all()),
    }
    (OUTPUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    files = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name != "output_manifest.csv")
    manifest = pd.DataFrame([{
        "output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path),
    } for path in files])
    write_csv(manifest, OUTPUT / "output_manifest.csv")

    if not as_bool(checks["passed"]).all():
        failed = checks.loc[~as_bool(checks["passed"]), "check"].tolist()
        raise RuntimeError(f"Business metric translation audit failed build checks: {failed}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

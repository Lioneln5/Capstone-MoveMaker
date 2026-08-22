"""Independently verify the business metric translation audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Data" / "processed" / "business_metric_translation_audit"
PHASE1 = ROOT / "Data" / "processed" / "extension_opportunity_diagnostic"
PHASE5 = ROOT / "Data" / "processed" / "integrated_contract_profile"
MASTER = ROOT / "Data" / "processed" / "frozen_contract_sources" / "phase1_5_2026-08-11" / "contract_extension_integration" / "extension_modeling_master.csv"
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


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").fillna("").str.strip().str.casefold().isin({"true", "1", "yes"})


def close(left: Any, right: Any, tolerance: float = 1e-8) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return bool(np.isclose(float(left), float(right), atol=tolerance, rtol=tolerance))


def add(checks: list[dict[str, Any]], name: str, observed: Any, expected: Any, passed: bool, notes: str) -> None:
    checks.append({"check": name, "observed": observed, "expected": expected, "passed": bool(passed), "notes": notes})


def reconstruct_exposure(profile: pd.DataFrame) -> pd.DataFrame:
    predictions = pd.read_csv(PHASE1 / "model_predictions.csv", low_memory=False)
    predictions = predictions.loc[
        predictions["endpoint"].eq("sustained_meaningful_contribution")
        & predictions["model_variant"].isin([M0, M5])
    ]
    wide = predictions.pivot(
        index=["capology_extension_event_id", "canonical_player_id", "evaluation_year"],
        columns="model_variant", values="prediction",
    ).reset_index()
    actual = predictions.loc[predictions["model_variant"].eq(M5), [
        "capology_extension_event_id", "actual",
    ]].rename(columns={"actual": "actual_sustained_contribution"})
    frame = wide.merge(actual, on="capology_extension_event_id", validate="one_to_one")
    frame = frame.merge(profile[[
        "capology_extension_event_id", "player_name", "club_name", "league", "canonical_position",
        "annual_gross_eur", "exact_duration_years", "fixed_wages_through_24m_eur",
    ]], on="capology_extension_event_id", validate="one_to_one")
    frame["actual_low_contribution"] = 1 - frame["actual_sustained_contribution"]
    frame["baseline_low_contribution_probability"] = 1 - frame[M0]
    frame["selected_low_contribution_probability"] = 1 - frame[M5]
    frame["actual_low_contribution_exposure_eur"] = frame["fixed_wages_through_24m_eur"] * frame["actual_low_contribution"]
    frame["baseline_exposure_eur"] = frame["fixed_wages_through_24m_eur"] * frame["baseline_low_contribution_probability"]
    frame["selected_exposure_eur"] = frame["fixed_wages_through_24m_eur"] * frame["selected_low_contribution_probability"]
    required = [
        "fixed_wages_through_24m_eur", "actual_low_contribution",
        "baseline_low_contribution_probability", "selected_low_contribution_probability",
        "actual_low_contribution_exposure_eur", "baseline_exposure_eur", "selected_exposure_eur",
    ]
    return frame.dropna(subset=required).reset_index(drop=True)


def performance_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("pooled", frame), *[(str(year), group) for year, group in frame.groupby("evaluation_year")]]
    for origin, group in groups:
        commitment = group["fixed_wages_through_24m_eur"]
        actual_event = group["actual_low_contribution"]
        actual_exposure = group["actual_low_contribution_exposure_eur"]
        baseline_exposure = group["baseline_exposure_eur"]
        selected_exposure = group["selected_exposure_eur"]
        baseline_probability = group["baseline_low_contribution_probability"]
        selected_probability = group["selected_low_contribution_probability"]
        total = float(commitment.sum())
        base_mae = float(np.mean(np.abs(baseline_exposure - actual_exposure)))
        selected_mae = float(np.mean(np.abs(selected_exposure - actual_exposure)))
        base_brier = float(np.sum(commitment * (actual_event - baseline_probability) ** 2) / total)
        selected_brier = float(np.sum(commitment * (actual_event - selected_probability) ** 2) / total)
        rows.append({
            "evaluation": origin, "rows": len(group), "players": group["canonical_player_id"].nunique(),
            "total_24m_fixed_wage_commitment_eur": total,
            "realized_low_contribution_exposure_eur": float(actual_exposure.sum()),
            "baseline_predicted_exposure_eur": float(baseline_exposure.sum()),
            "selected_predicted_exposure_eur": float(selected_exposure.sum()),
            "baseline_exposure_mae_eur": base_mae, "selected_exposure_mae_eur": selected_mae,
            "exposure_mae_improvement_eur": base_mae - selected_mae,
            "baseline_capital_weighted_brier": base_brier,
            "selected_capital_weighted_brier": selected_brier,
            "capital_weighted_brier_improvement": base_brier - selected_brier,
            "selected_aggregate_calibration_ratio": float(selected_exposure.sum() / actual_exposure.sum()),
            "selected_exposure_spearman": float(selected_exposure.corr(actual_exposure, method="spearman")),
        })
    return pd.DataFrame(rows)


def bootstrap(frame: pd.DataFrame) -> tuple[float, float, float, float, int]:
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
    mae_draws, brier_draws = [], []
    for _ in range(BOOTSTRAP_REPETITIONS):
        indices = np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))])
        mae_draws.append(float(mae_delta[indices].mean()))
        total = float(weight[indices].sum())
        base_loss = float(np.sum(weight[indices] * (actual[indices] - baseline[indices]) ** 2) / total)
        selected_loss = float(np.sum(weight[indices] * (actual[indices] - selected[indices]) ** 2) / total)
        brier_draws.append(base_loss - selected_loss)
    mae = np.asarray(mae_draws)
    brier = np.asarray(brier_draws)
    return (
        float(np.quantile(mae, .025)), float(np.quantile(mae, .975)),
        float(np.quantile(brier, .025)), float(np.quantile(brier, .975)), len(clusters),
    )


def exposure_bands(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["exposure_band"] = pd.qcut(
        data["selected_exposure_eur"].rank(method="first"), 5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)
    rows = []
    for band, group in data.groupby("exposure_band", observed=True):
        predicted = float(group["selected_exposure_eur"].mean())
        realized = float(group["actual_low_contribution_exposure_eur"].mean())
        rows.append({
            "exposure_band": int(band), "rows": len(group),
            "mean_predicted_exposure_eur": predicted, "mean_realized_exposure_eur": realized,
            "predicted_to_realized_ratio": predicted / realized if realized else np.nan,
            "mean_24m_fixed_wage_commitment_eur": float(group["fixed_wages_through_24m_eur"].mean()),
            "mean_low_contribution_probability": float(group["selected_low_contribution_probability"].mean()),
            "observed_low_contribution_rate": float(group["actual_low_contribution"].mean()),
        })
    return pd.DataFrame(rows)


def top_quintile_summary(frame: pd.DataFrame, score: str, label: str) -> dict[str, Any]:
    threshold = float(frame[score].quantile(.80))
    top = frame[score].ge(threshold)
    realized = frame["actual_low_contribution_exposure_eur"]
    commitment = frame["fixed_wages_through_24m_eur"]
    capital_share = float(commitment[top].sum() / commitment.sum())
    return {
        "model": label,
        "top_quintile_rows": int(top.sum()),
        "score_threshold_eur": threshold,
        "realized_exposure_capture": float(realized[top].sum() / realized.sum()),
        "capital_share": capital_share,
        "capture_lift_vs_capital_share": float(realized[top].sum() / realized.sum()) / capital_share,
        "top_group_commitment_weighted_event_rate": float(np.sum(commitment[top] * frame.loc[top, "actual_low_contribution"]) / commitment[top].sum()),
        "remainder_commitment_weighted_event_rate": float(np.sum(commitment[~top] * frame.loc[~top, "actual_low_contribution"]) / commitment[~top].sum()),
    }


def compare_frames(left: pd.DataFrame, right: pd.DataFrame, keys: list[str], columns: list[str], tolerance: float = 1e-8) -> int:
    merged = left[keys + columns].merge(right[keys + columns], on=keys, suffixes=("_left", "_right"), validate="one_to_one")
    mismatches = abs(len(left) - len(merged)) + abs(len(right) - len(merged))
    for column in columns:
        lvalue, rvalue = merged[f"{column}_left"], merged[f"{column}_right"]
        if pd.api.types.is_numeric_dtype(lvalue) and pd.api.types.is_numeric_dtype(rvalue):
            mismatches += int((~np.isclose(lvalue, rvalue, atol=tolerance, rtol=tolerance, equal_nan=True)).sum())
        else:
            mismatches += int(lvalue.fillna("<NA>").astype(str).ne(rvalue.fillna("<NA>").astype(str)).sum())
    return mismatches


def main() -> None:
    checks: list[dict[str, Any]] = []
    profile = pd.read_csv(OUTPUT / "translated_historical_profiles.csv", low_memory=False)
    catalog = pd.read_csv(OUTPUT / "business_metric_catalog.csv", low_memory=False)
    decisions = pd.read_csv(OUTPUT / "business_metric_decision_summary.csv", low_memory=False)
    web_spec = pd.read_csv(OUTPUT / "web_app_business_metric_specification.csv", low_memory=False)
    coverage = pd.read_csv(OUTPUT / "metric_coverage_and_distribution.csv", low_memory=False)
    saved_exposure = pd.read_csv(OUTPUT / "low_contribution_exposure_predictions.csv", low_memory=False)
    saved_performance = pd.read_csv(OUTPUT / "low_contribution_exposure_performance.csv", low_memory=False)
    saved_comparison = pd.read_csv(OUTPUT / "low_contribution_exposure_comparison.csv", low_memory=False)
    saved_bands = pd.read_csv(OUTPUT / "low_contribution_exposure_bands.csv", low_memory=False)
    saved_value = pd.read_csv(OUTPUT / "value_downside_scenario_audit.csv", low_memory=False)
    saved_continuity = pd.read_csv(OUTPUT / "continuity_commitment_scenario_audit.csv", low_memory=False)
    cases = pd.read_csv(OUTPUT / "historical_business_profile_cases.csv", low_memory=False)
    build_checks = pd.read_csv(OUTPUT / "build_checks.csv", low_memory=False)
    original = pd.read_csv(PHASE5 / "historical_integrated_profiles.csv", low_memory=False)
    cohort = pd.read_csv(PHASE5 / "integration_evaluation_cohort.csv", low_memory=False)
    master = pd.read_csv(MASTER, low_memory=False)

    add(checks, "profile_row_count", len(profile), 1560, len(profile) == 1560, "Translated Phase-5 profile count is fixed.")
    add(checks, "profile_event_grain", int(profile["capology_extension_event_id"].duplicated().sum()), 0, profile["capology_extension_event_id"].is_unique, "One row per extension event.")
    add(checks, "profile_source_ids", len(set(profile["capology_extension_event_id"]) ^ set(original["capology_extension_event_id"])), 0, set(profile["capology_extension_event_id"]) == set(original["capology_extension_event_id"]), "No source profile is added or dropped.")

    annual = profile["annual_gross_eur"]
    duration = profile["exact_duration_years"]
    market = profile["at_signing_market_value_eur"].where(profile["at_signing_market_value_eur"].gt(0))
    formula_mismatches = 0
    formula_mismatches += int((~np.isclose(profile["fixed_wage_commitment_eur"], annual * duration, equal_nan=True)).sum())
    formula_mismatches += int((~np.isclose(profile["fixed_wages_through_24m_eur"], annual * duration.clip(upper=2), equal_nan=True)).sum())
    formula_mismatches += int((~np.isclose(profile["risk_weighted_low_contribution_wage_exposure_eur"], profile["fixed_wages_through_24m_eur"] * (1 - profile["sustained_contribution_probability"]), equal_nan=True)).sum())
    formula_mismatches += int((~np.isclose(profile["public_value_downside_floor_25pct_eur"], market * .25, equal_nan=True)).sum())
    formula_mismatches += int((~np.isclose(profile["public_value_downside_floor_50pct_eur"], market * .50, equal_nan=True)).sum())
    formula_mismatches += int((~np.isclose(profile["scheduled_fixed_wages_after_24m_eur"], annual * (duration - 2).clip(lower=0), equal_nan=True)).sum())
    formula_mismatches += int((~np.isclose(profile["scheduled_fixed_wages_after_36m_eur"], annual * (duration - 3).clip(lower=0), equal_nan=True)).sum())
    add(checks, "translated_formula_recomputation", formula_mismatches, 0, formula_mismatches == 0, "All central financial translations reproduce independently.")

    prior = master[["capology_extension_event_id", "prior_season_salary_annual_gross_eur"]].drop_duplicates("capology_extension_event_id")
    prior_check = profile[["capology_extension_event_id", "annual_gross_eur", "prior_wage_change_pct"]].merge(prior, on="capology_extension_event_id", validate="one_to_one")
    expected_change = prior_check["annual_gross_eur"].div(prior_check["prior_season_salary_annual_gross_eur"].where(lambda x: x.gt(0))) - 1
    prior_mismatch = int((~np.isclose(prior_check["prior_wage_change_pct"], expected_change, equal_nan=True)).sum())
    add(checks, "prior_wage_change_recomputation", prior_mismatch, 0, prior_mismatch == 0, "Missing prior wage remains missing and is never replaced with zero.")

    residual = np.log(profile["fixed_wage_commitment_eur"] / profile["fixed_commitment_peer_estimate_eur"])
    expected_percentile = residual.rank(pct=True, method="average")
    peer_mismatch = int((~np.isclose(profile["fixed_commitment_peer_percentile"], expected_percentile, equal_nan=True)).sum())
    add(checks, "peer_percentile_recomputation", peer_mismatch, 0, peer_mismatch == 0, "Frozen historical OOS residual percentile reproduces.")

    reconstructed = reconstruct_exposure(profile)
    exposure_columns = [
        "fixed_wages_through_24m_eur", "actual_low_contribution",
        "baseline_low_contribution_probability", "selected_low_contribution_probability",
        "actual_low_contribution_exposure_eur", "baseline_exposure_eur", "selected_exposure_eur",
    ]
    exposure_mismatch = compare_frames(
        reconstructed, saved_exposure,
        ["capology_extension_event_id", "canonical_player_id", "evaluation_year"], exposure_columns,
    )
    add(checks, "exposure_row_recomputation", exposure_mismatch, 0, exposure_mismatch == 0, "All financially eligible exposure rows and values reproduce from Phase 1.")

    expected_performance = performance_rows(reconstructed)
    performance_columns = [column for column in expected_performance.columns if column != "evaluation"]
    performance_mismatch = compare_frames(expected_performance, saved_performance, ["evaluation"], performance_columns)
    add(checks, "exposure_performance_recomputation", performance_mismatch, 0, performance_mismatch == 0, "Pooled and origin euro MAE, weighted Brier, calibration, and rank metrics reproduce.")

    lower, upper, brier_lower, brier_upper, clusters = bootstrap(reconstructed)
    comparison = saved_comparison.loc[saved_comparison["decision"].notna()].iloc[0]
    bootstrap_mismatch = sum([
        not close(lower, comparison["mae_improvement_ci_lower_95"]),
        not close(upper, comparison["mae_improvement_ci_upper_95"]),
        not close(brier_lower, comparison["weighted_brier_improvement_ci_lower_95"]),
        not close(brier_upper, comparison["weighted_brier_improvement_ci_upper_95"]),
        int(comparison["bootstrap_repetitions"]) != BOOTSTRAP_REPETITIONS,
        int(comparison["player_clusters"]) != clusters,
    ])
    add(checks, "exposure_bootstrap_recomputation", bootstrap_mismatch, 0, bootstrap_mismatch == 0, "Player-clustered uncertainty reproduces with the frozen seed.")

    expected_bands = exposure_bands(reconstructed)
    band_columns = [column for column in expected_bands.columns if column != "exposure_band"]
    band_mismatch = compare_frames(expected_bands, saved_bands, ["exposure_band"], band_columns)
    add(checks, "exposure_band_recomputation", band_mismatch, 0, band_mismatch == 0, "Five predicted euro-exposure bands reproduce.")
    band_monotone = expected_bands["mean_realized_exposure_eur"].is_monotonic_increasing
    add(checks, "exposure_realized_ordering", int(band_monotone), 1, band_monotone, "Realized euro exposure is monotonic across predicted bands.")

    expected_concentration = pd.DataFrame([
        top_quintile_summary(reconstructed, "fixed_wages_through_24m_eur", "capital_only"),
        top_quintile_summary(reconstructed, "baseline_exposure_eur", "chronological_baseline"),
        top_quintile_summary(reconstructed, "selected_exposure_eur", "selected_phase1_model"),
    ])
    saved_concentration = saved_comparison.loc[saved_comparison["model"].notna()].copy()
    concentration_columns = [column for column in expected_concentration.columns if column != "model"]
    concentration_mismatch = compare_frames(expected_concentration, saved_concentration, ["model"], concentration_columns)
    add(checks, "capital_targeting_recomputation", concentration_mismatch, 0, concentration_mismatch == 0, "Capital-only, baseline, and selected top-quintile exposure concentration reproduce.")
    capital_row = expected_concentration.loc[expected_concentration["model"].eq("capital_only")].iloc[0]
    selected_row = expected_concentration.loc[expected_concentration["model"].eq("selected_phase1_model")].iloc[0]
    improves_capital_targeting = (
        selected_row["realized_exposure_capture"] >= capital_row["realized_exposure_capture"]
        and selected_row["capital_share"] < capital_row["capital_share"]
    )
    add(checks, "capital_targeting_gate", int(improves_capital_targeting), 1, improves_capital_targeting, "The selected proxy captures at least as much realized exposure using a smaller share of reviewed two-year capital than contract size alone.")

    pooled = expected_performance.loc[expected_performance["evaluation"].eq("pooled")].iloc[0]
    origins = expected_performance.loc[~expected_performance["evaluation"].eq("pooled")]
    expected_decision = "advance_risk_weighted_exposure_proxy" if (
        pooled["exposure_mae_improvement_eur"] > 0 and lower > 0
        and pooled["capital_weighted_brier_improvement"] > 0 and brier_lower > 0
        and origins["exposure_mae_improvement_eur"].gt(0).sum() >= 3
        and origins["capital_weighted_brier_improvement"].gt(0).sum() >= 3
        and .75 <= pooled["selected_aggregate_calibration_ratio"] <= 1.25
        and band_monotone
        and improves_capital_targeting
    ) else "show_probability_and_commitment_separately"
    add(checks, "exposure_decision_recomputation", comparison["decision"], expected_decision, comparison["decision"] == expected_decision, "The predeclared business-metric gate reproduces.")

    value_monotone = saved_value["observed_25pct_downside_rate"].is_monotonic_increasing
    value_formula_ok = np.allclose(saved_value["median_50pct_downside_floor_eur"], 2 * saved_value["median_25pct_downside_floor_eur"], equal_nan=True)
    add(checks, "value_scenario_behavior", f"monotone={value_monotone}; severity={value_formula_ok}", "both true", value_monotone and value_formula_ok, "Risk bands separate outcomes and euro thresholds retain exact 25%/50% semantics.")
    continuity_ordering = all(group["observed_outbound_rate"].is_monotonic_increasing for _, group in saved_continuity.groupby("horizon"))
    add(checks, "continuity_scenario_behavior", int(continuity_ordering), 1, continuity_ordering, "Observed outbound rates increase across saved risk bands at both supported horizons.")

    expected_coverage = {}
    for column in coverage["metric"]:
        values = pd.to_numeric(profile[column], errors="coerce")
        eligible = values.notna()
        expected_coverage[column] = (int(eligible.sum()), float(eligible.mean()))
    coverage_mismatch = 0
    for row in coverage.itertuples(index=False):
        expected = expected_coverage[row.metric]
        coverage_mismatch += int(row.available_rows != expected[0]) + int(not close(row.coverage, expected[1]))
    add(checks, "metric_coverage_recomputation", coverage_mismatch, 0, coverage_mismatch == 0, "Saved metric availability reproduces from translated profiles.")

    catalog_map = dict(zip(catalog["metric_key"], catalog["status"]))
    required_status = {
        "fixed_wage_commitment_eur": "supported",
        "fixed_commitment_peer_percentile": "supported_after_frozen_reference_cdf",
        "risk_weighted_low_contribution_wage_exposure_eur": "supported_proxy",
        "value_downside_25pct_display": "supported_paired_display",
        "continuity_commitment_display": "supported_paired_display",
        "overall_contract_score": "rejected",
    }
    catalog_ok = all(catalog_map.get(key) == value for key, value in required_status.items())
    add(checks, "catalog_status_boundaries", int(catalog_ok), 1, catalog_ok, "Supported, paired, proxy, and rejected outputs remain distinct.")
    language = " ".join(catalog["interpretation_boundary"].fillna("").astype(str)).casefold()
    boundary_ok = all(term in language for term in ["not total contract cost", "not expected accounting loss", "not resale proceeds", "not expected loss", "not audited total payroll"])
    add(checks, "catalog_claim_boundaries", int(boundary_ok), 1, boundary_ok, "Core financial non-claims are explicit.")
    composite_absent = "overall_contract_score" not in set(web_spec["metric_key"])
    add(checks, "web_spec_no_composite", int(composite_absent), 1, composite_absent, "Rejected overall score is absent from the display specification.")
    bettor_rejected = decisions.loc[decisions["metric_group"].eq("bettor_use_case"), "decision"].eq("reject_out_of_scope").all()
    add(checks, "bettor_scope_rejected", int(bettor_rejected), 1, bettor_rejected, "No betting claim is inferred from extension outcomes.")

    add(checks, "historical_case_count", len(cases), 12, len(cases) == 12, "Twelve Phase-5 success, failure, and divergence cases are translated.")
    case_trace = cases["capology_extension_event_id"].is_unique and set(cases["capology_extension_event_id"]).issubset(set(profile["capology_extension_event_id"]))
    add(checks, "historical_case_traceability", int(case_trace), 1, case_trace, "Every case traces to a unique out-of-time profile.")

    source_manifest = pd.read_csv(OUTPUT / "source_manifest.csv", low_memory=False)
    source_failures = []
    for row in source_manifest.itertuples(index=False):
        path = ROOT / row.source_file
        if not path.exists() or sha256(path) != row.sha256:
            source_failures.append(row.source_file)
    add(checks, "source_hashes", len(source_failures), 0, not source_failures, f"Failures: {source_failures or 'none'}")
    output_manifest = pd.read_csv(OUTPUT / "output_manifest.csv", low_memory=False)
    output_failures = []
    for row in output_manifest.itertuples(index=False):
        if row.output_file in {"independent_verification.csv", "independent_verification.json"}:
            continue
        path = OUTPUT / row.output_file
        if not path.exists() or sha256(path) != row.sha256:
            output_failures.append(row.output_file)
    add(checks, "builder_output_hashes", len(output_failures), 0, not output_failures, f"Failures: {output_failures or 'none'}")
    builder_ok = as_bool(build_checks["passed"]).all()
    add(checks, "builder_checks", int(builder_ok), 1, builder_ok, "All builder checks pass.")

    results = pd.DataFrame(checks)
    results.to_csv(OUTPUT / "independent_verification.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    payload = {
        "checks_passed": int(as_bool(results["passed"]).sum()),
        "checks_total": len(results),
        "all_checks_passed": bool(as_bool(results["passed"]).all()),
        "historical_out_of_time_profiles": len(profile),
        "low_contribution_exposure_rows": len(reconstructed),
        "low_contribution_exposure_decision": expected_decision,
        "historical_business_cases": len(cases),
        "profile_scope": "incumbent_club_extension",
    }
    (OUTPUT / "independent_verification.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    files = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name != "output_manifest.csv")
    manifest = pd.DataFrame([{
        "output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path),
    } for path in files])
    manifest.to_csv(OUTPUT / "output_manifest.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    if not as_bool(results["passed"]).all():
        failed = results.loc[~as_bool(results["passed"]), "check"].tolist()
        raise RuntimeError(f"Independent business metric audit verification failed: {failed}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

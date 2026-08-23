"""Build the verified Phase-5 integrated incumbent-club contract profile.

This product layer joins only Phase 1-4 outputs that passed their own evidence
gates. It preserves factual calculations, historical peer benchmarks, modeled
estimates, and unsupported claims as separate classes. It also tests—but does
not assume—that a cross-module review score improves prioritization.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Data" / "processed" / "integrated_contract_profile"
MASTER = ROOT / "Data" / "processed" / "contract_extension_integration" / "extension_modeling_master.csv"
PHASES = {
    "phase1": ROOT / "Data" / "processed" / "extension_opportunity_diagnostic",
    "phase2": ROOT / "Data" / "processed" / "extension_survival_diagnostic",
    "phase3": ROOT / "Data" / "processed" / "extension_value_preservation_diagnostic",
    "phase4": ROOT / "Data" / "processed" / "contract_financial_exposure_benchmark",
}
RANDOM_SEED = 20260811
BOOTSTRAP_REPETITIONS = 5000
KEY = ["capology_extension_event_id", "canonical_player_id", "evaluation_year"]
M5 = "M5_plus_relative_financial_context"
P4_SELECTED = {
    "annual_wage": "B2_plus_prior_wage",
    "contract_duration": "B5_compact_nonlinear",
    "fixed_wage_commitment": "B2_plus_prior_wage",
    "wage_to_market_value": "B2_plus_prior_wage",
}


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


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def flatten_pivot(frame: pd.DataFrame, column: str, values: list[str]) -> pd.DataFrame:
    wide = frame.pivot(index=KEY, columns=column, values=values)
    wide.columns = ["_".join(map(str, pair)) for pair in wide.columns]
    return wide.reset_index()


def decreasing_projection(values: list[float]) -> list[float]:
    """Least-squares projection onto a non-increasing sequence via PAVA."""
    blocks: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        blocks.append({"sum": float(value), "weight": 1, "indices": [index]})
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if left["sum"] / left["weight"] >= right["sum"] / right["weight"]:
                break
            blocks[-2:] = [{
                "sum": left["sum"] + right["sum"],
                "weight": left["weight"] + right["weight"],
                "indices": left["indices"] + right["indices"],
            }]
    result = [0.0] * len(values)
    for block in blocks:
        mean = block["sum"] / block["weight"]
        for index in block["indices"]:
            result[index] = float(np.clip(mean, 0, 1))
    return result


def load_selected_layers() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for phase, directory in PHASES.items():
        verification = json.loads((directory / "independent_verification.json").read_text(encoding="utf-8"))
        if not verification.get("all_checks_passed"):
            raise RuntimeError(f"{phase} has not passed independent verification")

    p1 = pd.read_csv(PHASES["phase1"] / "model_predictions.csv", low_memory=False)
    p1 = p1.loc[p1["model_variant"].eq(M5) & p1["endpoint"].isin([
        "year2_opportunity_share", "sustained_meaningful_contribution",
    ])].copy()
    p1 = flatten_pivot(p1, "endpoint", ["actual", "prediction"])

    p2 = pd.read_csv(PHASES["phase2"] / "survival_predictions.csv", low_memory=False)
    p2 = p2.loc[p2["model_variant"].eq(M5)].copy()
    p2["endpoint_horizon"] = p2["endpoint"] + "_" + p2["horizon"]
    supported = {"any_outbound_24m", "any_outbound_36m", "permanent_outbound_24m"}
    p2 = p2.loc[p2["endpoint_horizon"].isin(supported)]
    p2 = flatten_pivot(p2, "endpoint_horizon", [
        "actual_event_by_horizon", "predicted_event_probability", "predicted_survival_probability",
    ])

    p3 = pd.read_csv(PHASES["phase3"] / "value_predictions.csv", low_memory=False)
    p3 = p3.loc[p3["model_variant"].eq(M5) & p3["endpoint"].isin([
        "value_log_ratio_12m", "downside_10pct_12m", "downside_25pct_12m",
        "value_log_ratio_24m", "downside_10pct_24m", "downside_25pct_24m", "downside_50pct_24m",
    ])].copy()
    p3 = flatten_pivot(p3, "endpoint", ["actual", "prediction"])

    p4 = pd.read_csv(PHASES["phase4"] / "benchmark_predictions.csv", low_memory=False)
    p4 = p4.loc[p4.apply(lambda row: row["model_variant"] == P4_SELECTED[row["endpoint"]], axis=1)].copy()
    p4 = flatten_pivot(p4, "endpoint", [
        "actual", "prediction", "interval_lower", "interval_upper", "observed_premium_vs_benchmark",
    ])
    return p1, p2, p3, p4


def add_reference_ranges(profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    specifications = [
        ("year2_opportunity", "actual_year2_opportunity_share", "prediction_year2_opportunity_share", "bounded_share"),
        ("value_multiplier_12m", "actual_value_log_ratio_12m", "prediction_value_log_ratio_12m", "log_ratio"),
        ("value_multiplier_24m", "actual_value_log_ratio_24m", "prediction_value_log_ratio_24m", "log_ratio"),
    ]
    for endpoint, actual, prediction, kind in specifications:
        eligible = profiles[actual].notna() & profiles[prediction].notna()
        residual = (profiles.loc[eligible, actual] - profiles.loc[eligible, prediction]).abs()
        width = float(residual.quantile(.80))
        lower = profiles[prediction] - width
        upper = profiles[prediction] + width
        if kind == "bounded_share":
            profiles[f"{endpoint}_estimate"] = profiles[prediction].clip(0, 1)
            profiles[f"{endpoint}_reference_lower"] = lower.clip(0, 1)
            profiles[f"{endpoint}_reference_upper"] = upper.clip(0, 1)
            actual_scale = profiles[actual]
        else:
            profiles[f"{endpoint}_estimate"] = np.exp(profiles[prediction])
            profiles[f"{endpoint}_reference_lower"] = np.exp(lower)
            profiles[f"{endpoint}_reference_upper"] = np.exp(upper)
            actual_scale = np.exp(profiles[actual])
        coverage = float(((actual_scale[eligible] >= profiles.loc[eligible, f"{endpoint}_reference_lower"]) &
                          (actual_scale[eligible] <= profiles.loc[eligible, f"{endpoint}_reference_upper"])).mean())
        rows.append({
            "endpoint": endpoint, "eligible_rows": int(eligible.sum()), "absolute_residual_q80": width,
            "observed_reference_coverage": coverage,
            "status": "evaluation_derived_reference_range_not_formal_prediction_interval",
        })
    return profiles, pd.DataFrame(rows)


def build_profiles() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p1, p2, p3, p4 = load_selected_layers()
    master = pd.read_csv(MASTER, low_memory=False)
    master_columns = [
        "capology_extension_event_id", "player_name", "club_name", "league", "canonical_position",
        "signed_date", "age_at_signing", "expiration_date", "at_signing_market_value_eur",
        "annual_gross_eur", "exact_duration_years", "bonus_available", "bonus_gross_eur",
        "club_salary_percentile", "club_salary_share_known", "prior_season_salary_annual_gross_eur",
    ]
    base = p2[[*KEY]].drop_duplicates().merge(master[master_columns], on="capology_extension_event_id", validate="one_to_one")
    profiles = base.merge(p1, on=KEY, how="left", validate="one_to_one")
    profiles = profiles.merge(p2, on=KEY, how="left", validate="one_to_one")
    profiles = profiles.merge(p3, on=KEY, how="left", validate="one_to_one")
    profiles = profiles.merge(p4, on=KEY, how="left", validate="one_to_one")
    profiles.insert(0, "profile_scope", "incumbent_club_extension")

    profiles["age_at_expiration"] = pd.to_numeric(profiles["age_at_signing"], errors="coerce") + pd.to_numeric(profiles["exact_duration_years"], errors="coerce")
    profiles["fixed_wage_commitment_eur"] = pd.to_numeric(profiles["annual_gross_eur"], errors="coerce") * pd.to_numeric(profiles["exact_duration_years"], errors="coerce")
    market = pd.to_numeric(profiles["at_signing_market_value_eur"], errors="coerce").where(lambda value: value.gt(0))
    profiles["wage_to_market_value"] = pd.to_numeric(profiles["annual_gross_eur"], errors="coerce").div(market)
    profiles["fixed_commitment_to_market_value"] = profiles["fixed_wage_commitment_eur"].div(market)

    profiles["sustained_contribution_probability"] = profiles["prediction_sustained_meaningful_contribution"]
    profiles["sustained_contribution_tier"] = pd.cut(
        profiles["sustained_contribution_probability"], [-np.inf, 1 / 3, 2 / 3, np.inf],
        labels=["low", "moderate", "high"],
    ).astype("string")
    profiles["continuous_stay_probability_24m"] = profiles["predicted_survival_probability_any_outbound_24m"]
    profiles["continuous_stay_probability_36m"] = profiles["predicted_survival_probability_any_outbound_36m"]
    profiles["permanent_relationship_probability_24m"] = profiles["predicted_survival_probability_permanent_outbound_24m"]

    for horizon, thresholds in [(12, [10, 25]), (24, [10, 25, 50])]:
        raw_columns = [f"prediction_downside_{threshold}pct_{horizon}m" for threshold in thresholds]
        ordered = profiles[raw_columns].apply(
            lambda row: decreasing_projection(row.tolist()) if row.notna().all() else [np.nan] * len(row), axis=1, result_type="expand",
        )
        for position, threshold in enumerate(thresholds):
            profiles[f"value_downside_{threshold}pct_probability_{horizon}m"] = ordered[position]
        profiles[f"value_preservation_probability_{horizon}m"] = 1 - profiles[f"value_downside_10pct_probability_{horizon}m"]

    profiles, interval_audit = add_reference_ranges(profiles)
    profiles["annual_wage_peer_estimate_eur"] = np.exp(profiles["prediction_annual_wage"])
    profiles["annual_wage_peer_lower_eur"] = np.exp(profiles["interval_lower_annual_wage"])
    profiles["annual_wage_peer_upper_eur"] = np.exp(profiles["interval_upper_annual_wage"])
    profiles["annual_wage_peer_premium"] = profiles["observed_premium_vs_benchmark_annual_wage"]
    profiles["contract_duration_peer_estimate_years"] = profiles["prediction_contract_duration"]
    profiles["contract_duration_peer_lower_years"] = profiles["interval_lower_contract_duration"].clip(lower=0)
    profiles["contract_duration_peer_upper_years"] = profiles["interval_upper_contract_duration"]
    profiles["contract_duration_peer_delta_years"] = profiles["actual_contract_duration"] - profiles["prediction_contract_duration"]
    profiles["fixed_commitment_peer_estimate_eur"] = np.exp(profiles["prediction_fixed_wage_commitment"])
    profiles["fixed_commitment_peer_lower_eur"] = np.exp(profiles["interval_lower_fixed_wage_commitment"])
    profiles["fixed_commitment_peer_upper_eur"] = np.exp(profiles["interval_upper_fixed_wage_commitment"])
    profiles["fixed_commitment_peer_premium"] = profiles["observed_premium_vs_benchmark_fixed_wage_commitment"]
    profiles["wage_value_peer_estimate"] = np.exp(profiles["prediction_wage_to_market_value"])
    profiles["wage_value_peer_lower"] = np.exp(profiles["interval_lower_wage_to_market_value"])
    profiles["wage_value_peer_upper"] = np.exp(profiles["interval_upper_wage_to_market_value"])
    profiles["wage_value_peer_premium"] = profiles["observed_premium_vs_benchmark_wage_to_market_value"]

    profiles["opportunity_module_available"] = profiles["sustained_contribution_probability"].notna() & profiles["year2_opportunity_estimate"].notna()
    profiles["survival_24m_module_available"] = profiles["continuous_stay_probability_24m"].notna() & profiles["permanent_relationship_probability_24m"].notna()
    profiles["survival_36m_module_available"] = profiles["continuous_stay_probability_36m"].notna()
    profiles["value_12m_module_available"] = profiles["value_downside_25pct_probability_12m"].notna()
    profiles["value_24m_module_available"] = profiles["value_downside_50pct_probability_24m"].notna()
    profiles["financial_peer_module_available"] = profiles["annual_wage_peer_estimate_eur"].notna() & profiles["contract_duration_peer_estimate_years"].notna()
    profiles["complete_three_risk_module_profile"] = profiles[[
        "opportunity_module_available", "survival_24m_module_available", "value_24m_module_available",
    ]].all(axis=1)
    profiles["headline_composite_score"] = np.nan
    profiles["headline_composite_status"] = "not_supported_show_modules_separately"

    coverage_rows = []
    for column in [
        "opportunity_module_available", "survival_24m_module_available", "survival_36m_module_available",
        "value_12m_module_available", "value_24m_module_available", "financial_peer_module_available",
        "complete_three_risk_module_profile",
    ]:
        coverage_rows.append({"availability_flag": column, "available_rows": int(as_bool(profiles[column]).sum()), "profile_rows": len(profiles), "coverage": float(as_bool(profiles[column]).mean())})
    return profiles, pd.DataFrame(coverage_rows), interval_audit


def bootstrap_auc_difference(frame: pd.DataFrame, candidate: str, baseline: str) -> dict[str, float]:
    clusters = [group.index.to_numpy() for _, group in frame.groupby("canonical_player_id", sort=True)]
    rng = np.random.default_rng(RANDOM_SEED + int(hashlib.sha256(candidate.encode()).hexdigest()[:8], 16))
    draws = []
    for _ in range(BOOTSTRAP_REPETITIONS):
        sampled = rng.integers(0, len(clusters), len(clusters))
        indices = np.concatenate([clusters[index] for index in sampled])
        target = frame.loc[indices, "multi_domain_adverse_outcome"].to_numpy()
        if np.unique(target).size < 2:
            continue
        draws.append(roc_auc_score(target, frame.loc[indices, candidate]) - roc_auc_score(target, frame.loc[indices, baseline]))
    values = np.asarray(draws)
    return {
        "cluster_count": len(clusters), "bootstrap_repetitions": len(values),
        "auc_difference_ci_lower_95": float(np.quantile(values, .025)),
        "auc_difference_ci_upper_95": float(np.quantile(values, .975)),
        "probability_candidate_better": float(np.mean(values > 0)),
    }


def integration_test(profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = [
        "sustained_contribution_probability", "continuous_stay_probability_24m",
        "value_downside_25pct_probability_24m", "actual_sustained_meaningful_contribution",
        "actual_event_by_horizon_any_outbound_24m", "actual_downside_25pct_24m",
    ]
    cohort = profiles.loc[profiles[required].notna().all(axis=1)].copy().reset_index(drop=True)
    cohort["opportunity_risk"] = 1 - cohort["sustained_contribution_probability"]
    cohort["outbound_risk_24m"] = 1 - cohort["continuous_stay_probability_24m"]
    cohort["value_downside_risk_24m"] = cohort["value_downside_25pct_probability_24m"]
    cohort["actual_low_contribution"] = 1 - cohort["actual_sustained_meaningful_contribution"]
    cohort["actual_outbound_24m"] = cohort["actual_event_by_horizon_any_outbound_24m"]
    cohort["actual_value_downside_25pct_24m"] = cohort["actual_downside_25pct_24m"]
    outcome_columns = ["actual_low_contribution", "actual_outbound_24m", "actual_value_downside_25pct_24m"]
    cohort["adverse_domain_count"] = cohort[outcome_columns].sum(axis=1).astype(int)
    cohort["multi_domain_adverse_outcome"] = cohort["adverse_domain_count"].ge(2).astype(int)
    risk_columns = ["opportunity_risk", "outbound_risk_24m", "value_downside_risk_24m"]
    # Freeze decimal precision before evaluation so exact rank ties survive CSV round-tripping.
    ranks = cohort.groupby("evaluation_year")[risk_columns].rank(pct=True, method="average").round(12)
    cohort["opportunity_risk_rank"] = ranks["opportunity_risk"]
    cohort["outbound_risk_rank"] = ranks["outbound_risk_24m"]
    cohort["value_downside_risk_rank"] = ranks["value_downside_risk_24m"]
    cohort["equal_weight_three_module_rank"] = ranks.mean(axis=1).round(12)
    cohort["opportunity_outbound_rank"] = ranks[["opportunity_risk", "outbound_risk_24m"]].mean(axis=1).round(12)
    cohort["maximum_component_rank"] = ranks.max(axis=1).round(12)

    scores = {
        "opportunity_component": "opportunity_risk_rank",
        "outbound_component": "outbound_risk_rank",
        "value_component": "value_downside_risk_rank",
        "equal_weight_three_module": "equal_weight_three_module_rank",
        "opportunity_plus_outbound": "opportunity_outbound_rank",
        "maximum_component": "maximum_component_rank",
    }
    target = cohort["multi_domain_adverse_outcome"]
    evaluation_rows = []
    for name, column in scores.items():
        threshold = float(cohort[column].quantile(.80))
        top = cohort[column].ge(threshold)
        origin_aucs = [roc_auc_score(group["multi_domain_adverse_outcome"], group[column]) for _, group in cohort.groupby("evaluation_year")]
        evaluation_rows.append({
            "score": name, "score_column": column, "evaluation_rows": len(cohort),
            "multi_domain_adverse_rate": float(target.mean()), "pooled_auc": float(roc_auc_score(target, cohort[column])),
            "minimum_origin_auc": float(min(origin_aucs)), "maximum_origin_auc": float(max(origin_aucs)),
            "top_quintile_rows": int(top.sum()), "top_quintile_event_rate": float(target[top].mean()),
            "top_quintile_lift": float(target[top].mean() / target.mean()),
        })
    evaluation = pd.DataFrame(evaluation_rows)
    best_component_row = evaluation.loc[evaluation["score"].isin([
        "opportunity_component", "outbound_component", "value_component",
    ])].sort_values(["pooled_auc", "score"], ascending=[False, True]).iloc[0]
    baseline_name, baseline_column = best_component_row["score"], best_component_row["score_column"]

    comparison_rows = []
    for candidate_name in ["equal_weight_three_module", "opportunity_plus_outbound"]:
        candidate_column = scores[candidate_name]
        pooled_delta = roc_auc_score(target, cohort[candidate_column]) - roc_auc_score(target, cohort[baseline_column])
        origin_deltas = []
        for _, group in cohort.groupby("evaluation_year"):
            origin_deltas.append(
                roc_auc_score(group["multi_domain_adverse_outcome"], group[candidate_column])
                - roc_auc_score(group["multi_domain_adverse_outcome"], group[baseline_column])
            )
        bootstrap = bootstrap_auc_difference(cohort, candidate_column, baseline_column)
        supported = pooled_delta > 0 and bootstrap["auc_difference_ci_lower_95"] > 0 and sum(value > 0 for value in origin_deltas) >= 3
        comparison_rows.append({
            "candidate": candidate_name, "baseline_best_component": baseline_name,
            "pooled_auc_difference": float(pooled_delta), "origin_wins": int(sum(value > 0 for value in origin_deltas)),
            "origin_count": len(origin_deltas), **bootstrap,
            "decision": "candidate_review_priority_tier" if supported else "reject_headline_composite_keep_modules_separate",
        })
    return cohort, evaluation, pd.DataFrame(comparison_rows)


def output_specification() -> pd.DataFrame:
    rows = [
        ("Identity", "profile_scope", "Decision scope", "context", "Phase 5", "supported", "Incumbent-club extension", "Models are not validated for a new-club acquisition."),
        ("Contract facts", "age_at_expiration", "Age at expiration", "factual_calculation", "Phase 4", "supported", "0.0 years", "Requires proposed term."),
        ("Contract facts", "fixed_wage_commitment_eur", "Fixed-wage commitment", "factual_calculation", "Phase 4", "supported_proxy", "€#,##0", "Annual fixed wage × proposed years; not total contract cost."),
        ("Contract facts", "wage_to_market_value", "Annual wage / market value", "factual_calculation", "Phase 4", "supported", "0.0%", "Market value is a public estimate."),
        ("Contract facts", "fixed_commitment_to_market_value", "Fixed commitment / market value", "factual_calculation", "Phase 4", "supported_proxy", "0.0%", "Not an ROI measure."),
        ("Acquisition-only arithmetic", "transfer_premium", "Transfer premium", "factual_calculation", "Phase 4", "supported_when_fee_entered", "0.0%", "Fee relative to public market value; not proof of overpayment."),
        ("Acquisition-only arithmetic", "estimated_acquisition_commitment", "Fee + fixed wages", "factual_calculation", "Phase 4", "supported_proxy_when_fee_entered", "€#,##0", "Excludes taxes, agents, signing fees, add-ons, clauses, and financing."),
        ("Peer context", "annual_wage_peer_range", "Historical annual-wage peer range", "historical_peer_benchmark", "Phase 4", "supported_broad_range", "€#,##0–€#,##0", "Historical extensions, not an optimal wage recommendation."),
        ("Peer context", "annual_wage_peer_premium", "Wage versus historical peers", "historical_peer_benchmark", "Phase 4", "supported", "+0%;−0%", "Unusual does not mean overpriced."),
        ("Peer context", "contract_duration_peer_range", "Historical term peer range", "historical_peer_benchmark", "Phase 4", "supported_broad_range", "0.0–0.0 years", "Not an optimal term recommendation."),
        ("Peer context", "contract_duration_peer_delta", "Term versus historical peers", "historical_peer_benchmark", "Phase 4", "supported", "+0.0;−0.0 years", "Historical deviation only."),
        ("Peer context", "fixed_commitment_peer_range", "Historical fixed-commitment range", "historical_peer_benchmark", "Phase 4", "supported_broad_range", "€#,##0–€#,##0", "Partial fixed-wage commitment only."),
        ("Opportunity", "year2_opportunity_reference_range", "Expected Year-2 opportunity range", "model_reference_range", "Phase 1", "supported_broad_range", "0%–0%", "Evaluation-derived reference range; not exact minutes."),
        ("Opportunity", "sustained_contribution_probability", "Meaningful contributor through Year 2", "model_probability", "Phase 1", "supported_candidate", "0%", "Predictive association, not tactical compatibility."),
        ("Continuity", "continuous_stay_probability_24m", "Continuous-club stay through 24 months", "model_probability", "Phase 2", "supported_candidate", "0%", "Loan or permanent outbound ends continuous stay."),
        ("Continuity", "continuous_stay_probability_36m", "Continuous-club stay through 36 months", "model_probability", "Phase 2", "supported_candidate", "0%", "Not an exact departure-date prediction."),
        ("Continuity", "permanent_relationship_probability_24m", "Permanent-club relationship through 24 months", "model_probability", "Phase 2", "supported_candidate", "0%", "An intervening loan does not end the permanent relationship."),
        ("Asset value", "value_multiplier_reference_range_12m", "12-month public-value range", "model_reference_range", "Phase 3", "supported_broad_range", "0.0×–0.0×", "Not realized resale proceeds."),
        ("Asset value", "value_preservation_probability_12m", "Preserve at least 90% of value at 12 months", "ordered_model_probability", "Phase 3", "supported_candidate", "0%", "Threshold probabilities are order-constrained for display."),
        ("Asset value", "value_downside_25pct_probability_12m", "Lose at least 25% of value at 12 months", "ordered_model_probability", "Phase 3", "supported_candidate", "0%", "Public-value downside, not accounting loss."),
        ("Asset value", "value_multiplier_reference_range_24m", "24-month public-value range", "model_reference_range", "Phase 3", "supported_broad_range", "0.0×–0.0×", "Not realized resale proceeds."),
        ("Asset value", "value_preservation_probability_24m", "Preserve at least 90% of value at 24 months", "ordered_model_probability", "Phase 3", "supported_candidate", "0%", "Threshold probabilities are order-constrained for display."),
        ("Asset value", "value_downside_25pct_probability_24m", "Lose at least 25% of value at 24 months", "ordered_model_probability", "Phase 3", "supported_candidate", "0%", "Public-value downside, not accounting loss."),
        ("Asset value", "value_downside_50pct_probability_24m", "Lose at least 50% of value at 24 months", "ordered_model_probability", "Phase 3", "supported_candidate", "0%", "Public-value downside, not accounting loss."),
        ("Salary context", "club_salary_percentile", "Club salary percentile", "observed_context", "Phase 4", "supported_when_matched", "0th percentile", "Known salary panel may be incomplete."),
        ("Salary context", "club_salary_share_known", "Share of known club payroll", "observed_context", "Phase 4", "supported_when_matched", "0.0%", "Denominator is not audited total payroll."),
        ("Overall", "headline_composite_score", "Overall contract score", "unsupported", "Phase 5", "rejected", "Do not display", "Integration did not validate a headline all-domain score."),
    ]
    return pd.DataFrame(rows, columns=[
        "section", "output_key", "display_label", "data_class", "source_phase", "status",
        "display_format", "plain_language_warning",
    ])


def input_contract() -> pd.DataFrame:
    rows = [
        ("player", "user_selection", "all sections", "required", "Resolve to canonical player ID."),
        ("current_club", "user_selection", "all modeled sections", "required", "Must be the incumbent club for extension-mode predictions."),
        ("proposed_annual_fixed_wage", "user_input", "contract facts and peer context", "required", "Positive gross annual amount with currency normalized to EUR."),
        ("proposed_contract_years", "user_input", "contract facts and peer context", "required", "Positive term; disclose whether options are included."),
        ("proposed_transfer_fee", "user_input", "acquisition-only arithmetic", "optional", "Does not authorize use of extension outcome models for a transfer."),
        ("current_market_value", "backend_or_user_input", "financial ratios and value models", "required", "Positive dated public estimate."),
        ("date_of_birth", "backend", "age features", "required", "Validated player identity match."),
        ("position", "backend", "all benchmark models", "required", "Canonical position category."),
        ("league", "backend", "all benchmark models", "required", "Validated only for modeled Big-Five scope."),
        ("prior_season_wage", "backend", "financial peer benchmarks", "recommended", "Missingness is modeled; do not substitute zero."),
        ("prior_opportunity_and_performance", "backend", "opportunity, survival, value", "required_for_models", "Retrieved from the canonical pre-extension history, not manually entered."),
        ("club_salary_panel", "backend", "relative financial context", "optional", "Known panel may be incomplete."),
    ]
    return pd.DataFrame(rows, columns=["input_field", "source", "used_for", "requirement", "validation_rule"])


def claims_catalog() -> pd.DataFrame:
    rows = [
        ("C01", "approved", "MoveMaker provides separate opportunity, continuity, public-value, and contract-cost context for incumbent-club extension decisions.", "Every displayed module passed its own out-of-time evidence gate."),
        ("C02", "qualified", "An offer is above or below historical extension peers.", "Peer deviation is descriptive and uncertain; it is not optimality or causation."),
        ("C03", "qualified", "The model estimates horizon-specific probabilities and broad ranges.", "Do not convert them into exact dates, minutes, sale proceeds, or guarantees."),
        ("C04", "prohibited", "The overall contract score predicts whether the deal will succeed.", "The three-module composite did not establish incremental value over the best component."),
        ("C05", "prohibited", "The model recommends the optimal wage or contract length.", "Historical peer benchmarks do not observe club utility, negotiation leverage, or causal contract effects."),
        ("C06", "prohibited", "Fee plus wages equals total acquisition cost.", "Taxes, agent fees, signing fees, add-ons, options, clauses, and financing are missing."),
        ("C07", "prohibited", "Market-value preservation equals resale profit or ROI.", "Public estimates are not realized proceeds or accounting values."),
        ("C08", "prohibited", "Extension-model probabilities apply to a player joining a new club.", "Phases 1-4 were evaluated on incumbent-club extension events."),
        ("C09", "prohibited", "A low probability means the player is tactically incompatible.", "The models predict outcomes, not causal tactical compatibility."),
    ]
    return pd.DataFrame(rows, columns=["claim_id", "classification", "claim", "evidence_boundary"])


def deployment_readiness() -> pd.DataFrame:
    rows = [
        ("Deterministic contract calculations", "ready_for_application_logic", "Implement directly from validated formulas and guarded inputs."),
        ("Observed salary-panel context", "ready_when_backend_match_exists", "Return unavailable when the player/club-season panel does not match."),
        ("Historical static peer tables", "ready_for_reference_display", "Can support coarse lookup cards; retain sample size and range warnings."),
        ("Personalized Phase-4 peer models", "offline_validated_not_yet_serialized", "Refit through the deployment cutoff and serialize preprocessing/model artifacts."),
        ("Phase-1 opportunity models", "offline_validated_not_yet_serialized", "Build the canonical backend feature payload and production scoring test."),
        ("Phase-2 survival models", "offline_validated_not_yet_serialized", "Preserve horizon coherence and the incumbent-club event definition."),
        ("Phase-3 value models", "offline_validated_not_yet_serialized", "Apply probability ordering and retain public-value warnings."),
        ("Headline composite score", "rejected", "Do not implement; show separate module cards."),
    ]
    return pd.DataFrame(rows, columns=["component", "deployment_status", "next_engineering_requirement"])


def select_case_studies(cohort: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    candidates = cohort.loc[
        as_bool(cohort["financial_peer_module_available"])
        & cohort["club_salary_percentile"].notna()
    ].copy()
    candidates["three_score"] = candidates["equal_weight_three_module_rank"]
    candidates["two_score"] = candidates["opportunity_outbound_rank"]
    definitions = [
        ("aligned_high_multidomain", candidates["multi_domain_adverse_outcome"].eq(1), "three_score", False),
        ("aligned_low_multidomain", candidates["multi_domain_adverse_outcome"].eq(0), "three_score", True),
        ("high_alert_without_multidomain", candidates["multi_domain_adverse_outcome"].eq(0), "three_score", False),
        ("missed_multidomain", candidates["multi_domain_adverse_outcome"].eq(1), "three_score", True),
        ("value_only_divergence", candidates["actual_value_downside_25pct_24m"].eq(1) & candidates["actual_low_contribution"].eq(0) & candidates["actual_outbound_24m"].eq(0), "value_downside_risk_rank", False),
        ("sporting_continuity_only", candidates["actual_value_downside_25pct_24m"].eq(0) & candidates["actual_low_contribution"].eq(1) & candidates["actual_outbound_24m"].eq(1), "two_score", False),
    ]
    selected, used = [], set()
    for category, mask, sort_column, ascending in definitions:
        subset = candidates.loc[mask & ~candidates["capology_extension_event_id"].isin(used)].sort_values(
            [sort_column, "capology_extension_event_id"], ascending=[ascending, True],
        )
        for row in subset.head(2).itertuples(index=False):
            selected.append({"case_category": category, **row._asdict()})
            used.add(row.capology_extension_event_id)
    cases = pd.DataFrame(selected)
    cases["case_interpretation"] = cases["case_category"].map({
        "aligned_high_multidomain": "High modeled concern aligned with adverse outcomes in at least two domains.",
        "aligned_low_multidomain": "Low modeled concern aligned with fewer than two adverse domains.",
        "high_alert_without_multidomain": "False-positive stress case: high combined rank without multi-domain adversity.",
        "missed_multidomain": "False-negative stress case: multi-domain adversity despite a low combined rank.",
        "value_only_divergence": "Value downside occurred without low contribution or 24-month outbound movement.",
        "sporting_continuity_only": "Low contribution and outbound movement occurred without 25% public-value downside.",
    })
    columns = [
        "case_category", "capology_extension_event_id", "canonical_player_id", "evaluation_year",
        "player_name", "club_name", "league", "canonical_position", "signed_date",
        "annual_gross_eur", "exact_duration_years", "at_signing_market_value_eur",
        "year2_opportunity_estimate", "sustained_contribution_probability",
        "continuous_stay_probability_24m", "permanent_relationship_probability_24m",
        "value_preservation_probability_24m", "value_downside_25pct_probability_24m",
        "value_downside_50pct_probability_24m", "annual_wage_peer_premium",
        "contract_duration_peer_delta_years", "fixed_commitment_peer_premium",
        "actual_low_contribution", "actual_outbound_24m", "actual_value_downside_25pct_24m",
        "adverse_domain_count", "multi_domain_adverse_outcome", "equal_weight_three_module_rank",
        "opportunity_outbound_rank", "case_interpretation",
    ]
    return cases[columns]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    profiles, coverage, interval_audit = build_profiles()
    integration_cohort, evaluation, comparisons = integration_test(profiles)
    spec = output_specification()
    inputs = input_contract()
    claims = claims_catalog()
    readiness = deployment_readiness()
    cases = select_case_studies(integration_cohort, profiles)

    ordering_rows = []
    for horizon, thresholds in [(12, [10, 25]), (24, [10, 25, 50])]:
        raw = [f"prediction_downside_{threshold}pct_{horizon}m" for threshold in thresholds]
        ordered = [f"value_downside_{threshold}pct_probability_{horizon}m" for threshold in thresholds]
        eligible = profiles[raw].notna().all(axis=1)
        raw_violations = sum((profiles.loc[eligible, raw[i + 1]] > profiles.loc[eligible, raw[i]]).sum() for i in range(len(raw) - 1))
        ordered_violations = sum((profiles.loc[eligible, ordered[i + 1]] > profiles.loc[eligible, ordered[i]] + 1e-12).sum() for i in range(len(ordered) - 1))
        raw_brier, ordered_brier = [], []
        for threshold, raw_column, ordered_column in zip(thresholds, raw, ordered):
            actual = profiles.loc[eligible, f"actual_downside_{threshold}pct_{horizon}m"]
            raw_brier.append(float(np.mean((actual - profiles.loc[eligible, raw_column]) ** 2)))
            ordered_brier.append(float(np.mean((actual - profiles.loc[eligible, ordered_column]) ** 2)))
        ordering_rows.append({
            "horizon": f"{horizon}m", "eligible_rows": int(eligible.sum()),
            "raw_pairwise_violations": int(raw_violations), "ordered_pairwise_violations": int(ordered_violations),
            "mean_raw_brier": float(np.mean(raw_brier)), "mean_ordered_brier": float(np.mean(ordered_brier)),
            "ordered_minus_raw_brier": float(np.mean(ordered_brier) - np.mean(raw_brier)),
        })
    ordering = pd.DataFrame(ordering_rows)

    checks = []
    def add(name: str, observed: object, expected: object, passed: bool, notes: str) -> None:
        checks.append({"check": name, "observed": observed, "expected": expected, "passed": bool(passed), "notes": notes})
    add("profile_event_grain", int(profiles["capology_extension_event_id"].duplicated().sum()), 0, not profiles["capology_extension_event_id"].duplicated().any(), "One out-of-time profile per extension event.")
    add("profile_rows", len(profiles), 1560, len(profiles) == 1560, "Phase-2 24-month evaluation cohort anchors profile coverage.")
    add("integration_cohort", len(integration_cohort), 839, len(integration_cohort) == 839, "Complete three-domain out-of-time overlap.")
    add("integration_years", integration_cohort["evaluation_year"].nunique(), 4, integration_cohort["evaluation_year"].nunique() == 4, "All four evaluation years are represented.")
    add("probability_ordering", int(ordering["ordered_pairwise_violations"].sum()), 0, ordering["ordered_pairwise_violations"].sum() == 0, "Displayed value-downside probabilities are coherent by threshold.")
    add("probability_ordering_brier_change", float(ordering["ordered_minus_raw_brier"].max()), "<=0.001", ordering["ordered_minus_raw_brier"].max() <= .001, "Coherence projection does not materially degrade mean Brier score.")
    formula_ok = np.allclose(profiles["fixed_wage_commitment_eur"], pd.to_numeric(profiles["annual_gross_eur"]) * pd.to_numeric(profiles["exact_duration_years"]), equal_nan=True)
    add("commitment_formula", int(formula_ok), 1, formula_ok, "Fixed-wage commitment is exact wage × duration arithmetic.")
    add("case_study_count", len(cases), 12, len(cases) == 12, "Two deterministic examples across six success/failure/divergence categories.")
    add("headline_composite_rejected", int(spec.loc[spec["output_key"].eq("headline_composite_score"), "status"].eq("rejected").all()), 1, spec.loc[spec["output_key"].eq("headline_composite_score"), "status"].eq("rejected").all(), "No unsupported all-domain score is exposed.")
    add("extension_scope_explicit", int(spec.loc[spec["output_key"].eq("profile_scope"), "plain_language_warning"].str.contains("new-club").all()), 1, spec.loc[spec["output_key"].eq("profile_scope"), "plain_language_warning"].str.contains("new-club").all(), "Transfer extrapolation is explicitly prohibited.")
    add("integration_decisions", " | ".join(comparisons["decision"]), "reject_headline_composite_keep_modules_separate", comparisons["decision"].eq("reject_headline_composite_keep_modules_separate").all(), "Neither tested integration score passes the incremental gate.")
    checks_frame = pd.DataFrame(checks)

    profile_columns = [
        "profile_scope", "capology_extension_event_id", "canonical_player_id", "evaluation_year", "player_name", "club_name", "league", "canonical_position", "signed_date",
        "age_at_signing", "age_at_expiration", "at_signing_market_value_eur", "annual_gross_eur", "exact_duration_years", "bonus_available", "bonus_gross_eur",
        "fixed_wage_commitment_eur", "wage_to_market_value", "fixed_commitment_to_market_value", "club_salary_percentile", "club_salary_share_known",
        "year2_opportunity_estimate", "year2_opportunity_reference_lower", "year2_opportunity_reference_upper", "sustained_contribution_probability", "sustained_contribution_tier",
        "continuous_stay_probability_24m", "continuous_stay_probability_36m", "permanent_relationship_probability_24m",
        "value_multiplier_12m_estimate", "value_multiplier_12m_reference_lower", "value_multiplier_12m_reference_upper", "value_preservation_probability_12m", "value_downside_25pct_probability_12m",
        "value_multiplier_24m_estimate", "value_multiplier_24m_reference_lower", "value_multiplier_24m_reference_upper", "value_preservation_probability_24m", "value_downside_25pct_probability_24m", "value_downside_50pct_probability_24m",
        "annual_wage_peer_estimate_eur", "annual_wage_peer_lower_eur", "annual_wage_peer_upper_eur", "annual_wage_peer_premium",
        "contract_duration_peer_estimate_years", "contract_duration_peer_lower_years", "contract_duration_peer_upper_years", "contract_duration_peer_delta_years",
        "fixed_commitment_peer_estimate_eur", "fixed_commitment_peer_lower_eur", "fixed_commitment_peer_upper_eur", "fixed_commitment_peer_premium",
        "wage_value_peer_estimate", "wage_value_peer_lower", "wage_value_peer_upper", "wage_value_peer_premium",
        "opportunity_module_available", "survival_24m_module_available", "survival_36m_module_available", "value_12m_module_available", "value_24m_module_available", "financial_peer_module_available", "complete_three_risk_module_profile",
        "headline_composite_score", "headline_composite_status",
    ]
    write_csv(profiles[profile_columns], OUTPUT / "historical_integrated_profiles.csv")
    write_csv(coverage, OUTPUT / "profile_coverage_audit.csv")
    write_csv(interval_audit, OUTPUT / "reference_interval_audit.csv")
    write_csv(integration_cohort, OUTPUT / "integration_evaluation_cohort.csv")
    write_csv(evaluation, OUTPUT / "integration_score_evaluation.csv")
    write_csv(comparisons, OUTPUT / "integration_score_comparisons.csv")
    write_csv(ordering, OUTPUT / "value_probability_ordering.csv")
    write_csv(spec, OUTPUT / "web_app_output_specification.csv")
    write_csv(inputs, OUTPUT / "web_app_input_contract.csv")
    write_csv(claims, OUTPUT / "approved_and_prohibited_claims.csv")
    write_csv(readiness, OUTPUT / "deployment_readiness.csv")
    write_csv(cases, OUTPUT / "historical_case_studies.csv")
    write_csv(checks_frame, OUTPUT / "build_checks.csv")

    source_rows = [{"source_file": MASTER.relative_to(ROOT).as_posix(), "bytes": MASTER.stat().st_size, "sha256": sha256(MASTER)}]
    phase_files = {
        "phase1": ["model_predictions.csv", "opportunity_decision_summary.csv", "independent_verification.json"],
        "phase2": ["survival_predictions.csv", "survival_decision_summary.csv", "independent_verification.json"],
        "phase3": ["value_predictions.csv", "value_decision_summary.csv", "independent_verification.json"],
        "phase4": ["benchmark_predictions.csv", "benchmark_decision_summary.csv", "deterministic_metric_catalog.csv", "independent_verification.json"],
    }
    for phase, directory in PHASES.items():
        for filename in phase_files[phase]:
            path = directory / filename
            source_rows.append({"source_file": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_csv(pd.DataFrame(source_rows), OUTPUT / "source_manifest.csv")

    best_component = evaluation.sort_values("pooled_auc", ascending=False).iloc[0]
    summary = {
        "profile_scope": "incumbent_club_extension", "historical_profile_rows": len(profiles),
        "complete_three_domain_evaluation_rows": len(integration_cohort),
        "multi_domain_adverse_rate": float(integration_cohort["multi_domain_adverse_outcome"].mean()),
        "best_evaluated_score": best_component["score"], "best_evaluated_auc": float(best_component["pooled_auc"]),
        "headline_composite_decision": "rejected_keep_modules_separate",
        "historical_case_studies": len(cases), "checks_passed": int(checks_frame["passed"].sum()),
        "checks_total": len(checks_frame), "all_checks_passed": bool(checks_frame["passed"].all()),
    }
    (OUTPUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    sample = cases.loc[cases["case_category"].eq("aligned_high_multidomain")].iloc[0]
    sample_profile = profiles.loc[profiles["capology_extension_event_id"].eq(sample["capology_extension_event_id"]), profile_columns].iloc[0]
    payload = json.loads(sample_profile.to_json(date_format="iso"))
    (OUTPUT / "sample_web_profile.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    comparison_text = "; ".join(
        f"{row.candidate} ΔAUC {row.pooled_auc_difference:+.4f} (95% CI [{row.auc_difference_ci_lower_95:+.4f}, {row.auc_difference_ci_upper_95:+.4f}]), {row.decision}"
        for row in comparisons.itertuples(index=False)
    )
    lines = [
        "# Integrated incumbent-club contract profile", "",
        "Phase 5 combines only independently verified Phase 1-4 candidates. Factual calculations, historical peer benchmarks, modeled estimates, and unsupported claims remain separate.", "",
        "## Scope", "",
        "The outcome models are validated for incumbent-club extension decisions. They must not be used as new-club transfer compatibility predictions. Acquisition fee arithmetic may be shown separately when a fee is entered.", "",
        "## Integration decision", "",
        f"The complete out-of-time overlap contains {len(integration_cohort):,} extensions across 2020-2023. The multi-domain adverse outcome (at least two of low Year-2 contribution, outbound by 24 months, or at least 25% public-value decline by 24 months) occurs in {integration_cohort['multi_domain_adverse_outcome'].mean():.1%} of cases.",
        f"{comparison_text}.",
        "Therefore the application should display separate module cards, not a headline overall contract score. The integration analysis is a product-routing diagnostic, not a new probability model.", "",
        "## Product readiness", "",
        "Deterministic contract calculations and matched observed salary context are ready for application logic. Personalized predictive modules are offline-validated but still require a full-history production refit, serialized preprocessing/model artifacts, and backend feature retrieval before live scoring.", "",
        "See the output specification, input contract, readiness table, claims catalog, historical case studies, and independent verification before implementation.",
    ]
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    files = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name != "output_manifest.csv")
    write_csv(pd.DataFrame([{"output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files]), OUTPUT / "output_manifest.csv")
    if not checks_frame["passed"].all():
        raise RuntimeError(f"Phase-5 build checks failed: {checks_frame.loc[~checks_frame['passed'], 'check'].tolist()}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

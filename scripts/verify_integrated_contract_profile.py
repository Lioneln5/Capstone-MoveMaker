"""Independently verify the Phase-5 integrated contract profile."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Data" / "processed" / "integrated_contract_profile"
MASTER = ROOT / "Data" / "processed" / "frozen_contract_sources" / "phase1_5_2026-08-11" / "contract_extension_integration" / "extension_modeling_master.csv"
PHASES = {
    "phase1": ROOT / "Data" / "processed" / "extension_opportunity_diagnostic",
    "phase2": ROOT / "Data" / "processed" / "extension_survival_diagnostic",
    "phase3": ROOT / "Data" / "processed" / "extension_value_preservation_diagnostic",
    "phase4": ROOT / "Data" / "processed" / "contract_financial_exposure_benchmark",
}
M5 = "M5_plus_relative_financial_context"
P4_SELECTED = {
    "annual_wage": "B2_plus_prior_wage", "contract_duration": "B5_compact_nonlinear",
    "fixed_wage_commitment": "B2_plus_prior_wage", "wage_to_market_value": "B2_plus_prior_wage",
}
RANDOM_SEED = 20260811
BOOTSTRAP_REPETITIONS = 5000


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


def add(checks: list[dict], name: str, observed: object, expected: object, passed: bool, notes: str) -> None:
    checks.append({"check": name, "observed": observed, "expected": expected, "passed": bool(passed), "notes": notes})


def close(left: object, right: object, tolerance: float = 1e-10) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return bool(np.isclose(float(left), float(right), atol=tolerance, rtol=tolerance))


def decreasing_projection(values: list[float]) -> list[float]:
    blocks = []
    for index, value in enumerate(values):
        blocks.append([float(value), 1, [index]])
        while len(blocks) >= 2 and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]:
            right, left = blocks.pop(), blocks.pop()
            blocks.append([left[0] + right[0], left[1] + right[1], left[2] + right[2]])
    result = [0.0] * len(values)
    for total, weight, indices in blocks:
        for index in indices:
            result[index] = float(np.clip(total / weight, 0, 1))
    return result


def bootstrap_difference(frame: pd.DataFrame, candidate: str, baseline: str) -> tuple[float, float, float, int]:
    clusters = [group.index.to_numpy() for _, group in frame.groupby("canonical_player_id", sort=True)]
    seed = RANDOM_SEED + int(hashlib.sha256(candidate.encode()).hexdigest()[:8], 16)
    rng, draws = np.random.default_rng(seed), []
    for _ in range(BOOTSTRAP_REPETITIONS):
        indices = np.concatenate([clusters[index] for index in rng.integers(0, len(clusters), len(clusters))])
        target = frame.loc[indices, "multi_domain_adverse_outcome"].to_numpy()
        if np.unique(target).size < 2:
            continue
        draws.append(roc_auc_score(target, frame.loc[indices, candidate]) - roc_auc_score(target, frame.loc[indices, baseline]))
    values = np.asarray(draws)
    return float(np.quantile(values, .025)), float(np.quantile(values, .975)), float(np.mean(values > 0)), len(values)


def main() -> None:
    required = {
        "historical_integrated_profiles.csv", "profile_coverage_audit.csv", "reference_interval_audit.csv",
        "integration_evaluation_cohort.csv", "integration_score_evaluation.csv", "integration_score_comparisons.csv",
        "value_probability_ordering.csv", "web_app_output_specification.csv", "web_app_input_contract.csv",
        "approved_and_prohibited_claims.csv", "deployment_readiness.csv", "historical_case_studies.csv",
        "sample_web_profile.json", "build_checks.csv", "source_manifest.csv", "output_manifest.csv",
        "run_summary.json", "README.md",
    }
    missing = sorted(name for name in required if not (OUTPUT / name).exists())
    if missing:
        raise RuntimeError(f"Missing Phase-5 artifacts: {missing}")
    checks: list[dict] = []
    add(checks, "required_outputs", len(missing), 0, not missing, "All declared Phase-5 outputs exist.")

    upstream_failures = []
    for phase, directory in PHASES.items():
        verification = json.loads((directory / "independent_verification.json").read_text(encoding="utf-8"))
        if not verification.get("all_checks_passed"):
            upstream_failures.append(phase)
    add(checks, "upstream_verifications", len(upstream_failures), 0, not upstream_failures, f"Failures: {upstream_failures or 'none'}")

    master = pd.read_csv(MASTER, low_memory=False)
    profiles = pd.read_csv(OUTPUT / "historical_integrated_profiles.csv", low_memory=False)
    coverage = pd.read_csv(OUTPUT / "profile_coverage_audit.csv")
    intervals = pd.read_csv(OUTPUT / "reference_interval_audit.csv")
    cohort = pd.read_csv(OUTPUT / "integration_evaluation_cohort.csv", low_memory=False)
    evaluation = pd.read_csv(OUTPUT / "integration_score_evaluation.csv")
    comparisons = pd.read_csv(OUTPUT / "integration_score_comparisons.csv")
    ordering = pd.read_csv(OUTPUT / "value_probability_ordering.csv")
    spec = pd.read_csv(OUTPUT / "web_app_output_specification.csv")
    inputs = pd.read_csv(OUTPUT / "web_app_input_contract.csv")
    claims = pd.read_csv(OUTPUT / "approved_and_prohibited_claims.csv")
    readiness = pd.read_csv(OUTPUT / "deployment_readiness.csv")
    cases = pd.read_csv(OUTPUT / "historical_case_studies.csv", low_memory=False)
    build_checks = pd.read_csv(OUTPUT / "build_checks.csv")

    add(checks, "source_row_count", len(master), 2425, len(master) == 2425, "Verified extension master remains unchanged in row count.")
    add(checks, "profile_row_count", len(profiles), 1560, len(profiles) == 1560, "Historical profile coverage matches the Phase-2 evaluation anchor.")
    add(checks, "profile_event_grain", int(profiles["capology_extension_event_id"].duplicated().sum()), 0, not profiles["capology_extension_event_id"].duplicated().any(), "One profile per extension event.")
    source_ids = set(master["capology_extension_event_id"])
    add(checks, "profile_source_subset", len(set(profiles["capology_extension_event_id"]) - source_ids), 0, set(profiles["capology_extension_event_id"]).issubset(source_ids), "Every profile traces to the verified extension master.")
    scope_ok = profiles["profile_scope"].eq("incumbent_club_extension").all()
    add(checks, "profile_scope", int(scope_ok), 1, scope_ok, "Every modeled profile is explicitly extension-only.")

    source_lookup = master.set_index("capology_extension_event_id").loc[profiles["capology_extension_event_id"]]
    wage = pd.to_numeric(source_lookup["annual_gross_eur"], errors="coerce").to_numpy()
    duration = pd.to_numeric(source_lookup["exact_duration_years"], errors="coerce").to_numpy()
    market = pd.to_numeric(source_lookup["at_signing_market_value_eur"], errors="coerce").to_numpy()
    age = pd.to_numeric(source_lookup["age_at_signing"], errors="coerce").to_numpy()
    formula_mismatches = 0
    formula_mismatches += int((~np.isclose(profiles["age_at_expiration"], age + duration, equal_nan=True)).sum())
    formula_mismatches += int((~np.isclose(profiles["fixed_wage_commitment_eur"], wage * duration, equal_nan=True)).sum())
    formula_mismatches += int((~np.isclose(profiles["wage_to_market_value"], wage / np.where(market > 0, market, np.nan), equal_nan=True)).sum())
    formula_mismatches += int((~np.isclose(profiles["fixed_commitment_to_market_value"], wage * duration / np.where(market > 0, market, np.nan), equal_nan=True)).sum())
    add(checks, "deterministic_formula_recomputation", formula_mismatches, 0, formula_mismatches == 0, "Age, fixed commitment, and market-value ratios reproduce from source inputs.")

    coverage_mismatches = 0
    for row in coverage.itertuples(index=False):
        observed = int(as_bool(profiles[row.availability_flag]).sum())
        coverage_mismatches += int(observed != row.available_rows)
        coverage_mismatches += int(not close(observed / len(profiles), row.coverage))
    expected_coverage = {
        "opportunity_module_available": 929, "survival_24m_module_available": 1560,
        "survival_36m_module_available": 1560, "value_12m_module_available": 1432,
        "value_24m_module_available": 1367, "financial_peer_module_available": 1504,
        "complete_three_risk_module_profile": 839,
    }
    observed_coverage = dict(zip(coverage["availability_flag"], coverage["available_rows"]))
    add(checks, "coverage_recomputation", coverage_mismatches, 0, coverage_mismatches == 0, "Saved module coverage matches row-level flags.")
    add(checks, "coverage_counts", observed_coverage, expected_coverage, observed_coverage == expected_coverage, "Phase-specific availability counts are fixed.")

    p1 = pd.read_csv(PHASES["phase1"] / "model_predictions.csv", low_memory=False)
    p1 = p1.loc[p1["model_variant"].eq(M5) & p1["endpoint"].eq("year2_opportunity_share")]
    p3 = pd.read_csv(PHASES["phase3"] / "value_predictions.csv", low_memory=False)
    p3 = p3.loc[p3["model_variant"].eq(M5) & p3["endpoint"].isin(["value_log_ratio_12m", "value_log_ratio_24m"])]
    interval_expected = {}
    p1_residual = (p1["actual"] - p1["prediction"]).abs()
    width = float(p1_residual.quantile(.80))
    lower, upper = (p1["prediction"] - width).clip(0, 1), (p1["prediction"] + width).clip(0, 1)
    interval_expected["year2_opportunity"] = (len(p1), width, float(((p1["actual"] >= lower) & (p1["actual"] <= upper)).mean()))
    for endpoint, group in p3.groupby("endpoint", observed=True):
        width = float((group["actual"] - group["prediction"]).abs().quantile(.80))
        interval_expected[endpoint.replace("value_log_ratio", "value_multiplier")] = (
            len(group), width, float(((group["actual"] >= group["prediction"] - width) & (group["actual"] <= group["prediction"] + width)).mean()),
        )
    interval_mismatches = 0
    for row in intervals.itertuples(index=False):
        expected = interval_expected[row.endpoint]
        interval_mismatches += int(row.eligible_rows != expected[0]) + int(not close(row.absolute_residual_q80, expected[1])) + int(not close(row.observed_reference_coverage, expected[2]))
    add(checks, "reference_interval_recomputation", interval_mismatches, 0, interval_mismatches == 0, "Evaluation-derived 80% reference widths and coverage reproduce from out-of-time predictions.")

    add(checks, "integration_cohort_count", len(cohort), 839, len(cohort) == 839, "Complete three-domain cohort is fixed.")
    add(checks, "integration_year_counts", cohort["evaluation_year"].value_counts().sort_index().to_dict(), {2020: 205, 2021: 211, 2022: 212, 2023: 211}, cohort["evaluation_year"].value_counts().sort_index().to_dict() == {2020: 205, 2021: 211, 2022: 212, 2023: 211}, "All evaluation origins have stable overlap.")
    adverse_count = (1 - cohort["actual_sustained_meaningful_contribution"] + cohort["actual_event_by_horizon_any_outbound_24m"] + cohort["actual_downside_25pct_24m"]).astype(int)
    target = adverse_count.ge(2).astype(int)
    target_ok = np.array_equal(adverse_count, cohort["adverse_domain_count"]) and np.array_equal(target, cohort["multi_domain_adverse_outcome"])
    add(checks, "integration_target_recomputation", int(target_ok), 1, target_ok, "Multi-domain outcome is exactly two or more adverse component outcomes.")
    add(checks, "integration_target_rate", float(target.mean()), 0.42550655542312277, close(target.mean(), 0.42550655542312277), "Observed multi-domain rate is reproduced.")

    raw_risk = pd.DataFrame({
        "opportunity_risk": 1 - cohort["sustained_contribution_probability"],
        "outbound_risk_24m": 1 - cohort["continuous_stay_probability_24m"],
        "value_downside_risk_24m": cohort["value_downside_25pct_probability_24m"],
    })
    ranks = raw_risk.groupby(cohort["evaluation_year"]).rank(pct=True, method="average").round(12)
    rank_mismatches = 0
    for source, saved in [
        ("opportunity_risk", "opportunity_risk_rank"), ("outbound_risk_24m", "outbound_risk_rank"),
        ("value_downside_risk_24m", "value_downside_risk_rank"),
    ]:
        rank_mismatches += int((~np.isclose(ranks[source], cohort[saved])).sum())
    rank_mismatches += int((~np.isclose(ranks.mean(axis=1).round(12), cohort["equal_weight_three_module_rank"])).sum())
    rank_mismatches += int((~np.isclose(ranks[["opportunity_risk", "outbound_risk_24m"]].mean(axis=1).round(12), cohort["opportunity_outbound_rank"])).sum())
    add(checks, "integration_rank_recomputation", rank_mismatches, 0, rank_mismatches == 0, "Within-year component and combined ranks reproduce.")

    score_columns = dict(zip(evaluation["score"], evaluation["score_column"]))
    score_mismatches = 0
    for row in evaluation.itertuples(index=False):
        score = cohort[row.score_column]
        threshold = score.quantile(.80)
        top = score.ge(threshold)
        auc = roc_auc_score(target, score)
        origin_aucs = [roc_auc_score(group["multi_domain_adverse_outcome"], group[row.score_column]) for _, group in cohort.groupby("evaluation_year")]
        score_mismatches += int(not close(auc, row.pooled_auc))
        score_mismatches += int(int(top.sum()) != row.top_quintile_rows)
        score_mismatches += int(not close(target[top].mean(), row.top_quintile_event_rate))
        score_mismatches += int(not close(min(origin_aucs), row.minimum_origin_auc))
    add(checks, "integration_score_recomputation", score_mismatches, 0, score_mismatches == 0, "AUCs, origin ranges, and top-quintile concentrations reproduce.")

    baseline_column = score_columns["opportunity_component"]
    comparison_mismatches = 0
    for row in comparisons.itertuples(index=False):
        candidate_column = score_columns[row.candidate]
        delta = roc_auc_score(target, cohort[candidate_column]) - roc_auc_score(target, cohort[baseline_column])
        origin_deltas = []
        for _, group in cohort.groupby("evaluation_year"):
            origin_deltas.append(roc_auc_score(group["multi_domain_adverse_outcome"], group[candidate_column]) - roc_auc_score(group["multi_domain_adverse_outcome"], group[baseline_column]))
        lower, upper, probability, repetitions = bootstrap_difference(cohort, candidate_column, baseline_column)
        supported = delta > 0 and lower > 0 and sum(value > 0 for value in origin_deltas) >= 3
        expected_decision = "candidate_review_priority_tier" if supported else "reject_headline_composite_keep_modules_separate"
        comparison_mismatches += int(not close(delta, row.pooled_auc_difference))
        comparison_mismatches += int(sum(value > 0 for value in origin_deltas) != row.origin_wins)
        comparison_mismatches += int(not close(lower, row.auc_difference_ci_lower_95)) + int(not close(upper, row.auc_difference_ci_upper_95))
        comparison_mismatches += int(not close(probability, row.probability_candidate_better)) + int(repetitions != row.bootstrap_repetitions)
        comparison_mismatches += int(expected_decision != row.decision)
    add(checks, "integration_comparison_recomputation", comparison_mismatches, 0, comparison_mismatches == 0, "Cluster-bootstrap uncertainty and no-composite decisions reproduce.")

    p3_binary = pd.read_csv(PHASES["phase3"] / "value_predictions.csv", low_memory=False)
    p3_binary = p3_binary.loc[p3_binary["model_variant"].eq(M5)]
    ordering_mismatches = 0
    for horizon, thresholds in [(12, [10, 25]), (24, [10, 25, 50])]:
        subset = p3_binary.loc[p3_binary["endpoint"].isin([f"downside_{value}pct_{horizon}m" for value in thresholds])]
        wide = subset.pivot(index=["capology_extension_event_id", "canonical_player_id", "evaluation_year"], columns="endpoint", values=["actual", "prediction"]).dropna()
        raw = wide["prediction"][[f"downside_{value}pct_{horizon}m" for value in thresholds]].to_numpy()
        actual = wide["actual"][[f"downside_{value}pct_{horizon}m" for value in thresholds]].to_numpy()
        adjusted = np.asarray([decreasing_projection(row.tolist()) for row in raw])
        raw_violations = int(sum((raw[:, i + 1] > raw[:, i]).sum() for i in range(len(thresholds) - 1)))
        ordered_violations = int(sum((adjusted[:, i + 1] > adjusted[:, i] + 1e-12).sum() for i in range(len(thresholds) - 1)))
        raw_brier = float(np.mean([np.mean((actual[:, i] - raw[:, i]) ** 2) for i in range(len(thresholds))]))
        ordered_brier = float(np.mean([np.mean((actual[:, i] - adjusted[:, i]) ** 2) for i in range(len(thresholds))]))
        saved = ordering.loc[ordering["horizon"].eq(f"{horizon}m")].iloc[0]
        ordering_mismatches += int(len(raw) != saved.eligible_rows) + int(raw_violations != saved.raw_pairwise_violations) + int(ordered_violations != saved.ordered_pairwise_violations)
        ordering_mismatches += int(not close(raw_brier, saved.mean_raw_brier)) + int(not close(ordered_brier, saved.mean_ordered_brier))
    add(checks, "value_ordering_recomputation", ordering_mismatches, 0, ordering_mismatches == 0, "PAVA ordering, violations, and Brier changes reproduce from Phase-3 predictions.")

    expected_keys = {
        "age_at_expiration", "fixed_wage_commitment_eur", "transfer_premium", "estimated_acquisition_commitment",
        "year2_opportunity_reference_range", "sustained_contribution_probability", "continuous_stay_probability_24m",
        "permanent_relationship_probability_24m", "value_preservation_probability_24m",
        "value_downside_50pct_probability_24m", "headline_composite_score",
    }
    add(checks, "output_specification_scope", len(expected_keys - set(spec["output_key"])), 0, expected_keys.issubset(set(spec["output_key"])), "The web specification contains all required product sections.")
    composite_rejected = spec.loc[spec["output_key"].eq("headline_composite_score"), "status"].eq("rejected").all()
    add(checks, "headline_composite_not_exposed", int(composite_rejected), 1, composite_rejected, "The rejected headline score is explicitly unavailable.")
    extension_warning = spec.loc[spec["output_key"].eq("profile_scope"), "plain_language_warning"].str.contains("new-club").all()
    add(checks, "extension_scope_warning", int(extension_warning), 1, extension_warning, "New-club extrapolation is prohibited in the app contract.")
    required_inputs = {"player", "current_club", "proposed_annual_fixed_wage", "proposed_contract_years", "current_market_value", "prior_opportunity_and_performance"}
    add(checks, "input_contract_scope", len(required_inputs - set(inputs["input_field"])), 0, required_inputs.issubset(set(inputs["input_field"])), "User and backend requirements are explicit.")
    prohibited_claims = claims.loc[claims["classification"].eq("prohibited"), "claim"].str.cat(sep=" ").casefold()
    claims_ok = all(term in prohibited_claims for term in ["overall contract score", "optimal wage", "total acquisition cost", "roi", "new club", "tactically incompatible"])
    add(checks, "prohibited_claims_scope", int(claims_ok), 1, claims_ok, "Core unsupported commercial and causal claims are frozen.")
    readiness_map = dict(zip(readiness["component"], readiness["deployment_status"]))
    readiness_ok = readiness_map.get("Deterministic contract calculations") == "ready_for_application_logic" and readiness_map.get("Headline composite score") == "rejected" and all("not_yet_serialized" in readiness_map[name] for name in ["Phase-1 opportunity models", "Phase-2 survival models", "Phase-3 value models"])
    add(checks, "deployment_readiness_boundaries", int(readiness_ok), 1, readiness_ok, "Offline validation is not mislabeled as live production readiness.")

    add(checks, "case_study_count", len(cases), 12, len(cases) == 12, "Twelve historical cases exist.")
    category_counts = cases["case_category"].value_counts().to_dict()
    categories_ok = len(category_counts) == 6 and set(category_counts.values()) == {2}
    add(checks, "case_study_categories", category_counts, "six categories × two", categories_ok, "Cases include successes, failures, and cross-domain divergence.")
    case_ids_ok = set(cases["capology_extension_event_id"]).issubset(set(cohort["capology_extension_event_id"])) and cases["capology_extension_event_id"].is_unique
    add(checks, "case_study_traceability", int(case_ids_ok), 1, case_ids_ok, "Every unique case traces to the complete out-of-time cohort.")
    sample = json.loads((OUTPUT / "sample_web_profile.json").read_text(encoding="utf-8"))
    sample_ok = sample.get("capology_extension_event_id") in set(cases["capology_extension_event_id"]) and sample.get("headline_composite_status") == "not_supported_show_modules_separately" and sample.get("headline_composite_score") is None
    add(checks, "sample_profile_boundary", int(sample_ok), 1, sample_ok, "Sample profile is historical and does not expose a composite score.")

    source_manifest = pd.read_csv(OUTPUT / "source_manifest.csv")
    expected_source_count = 14
    add(checks, "source_manifest_scope", len(source_manifest), expected_source_count, len(source_manifest) == expected_source_count, "Master plus exact Phase 1-4 predictions, decisions, catalogs, and verifications are pinned.")
    source_failures = []
    for row in source_manifest.itertuples(index=False):
        path = ROOT / row.source_file
        if not path.exists() or sha256(path) != row.sha256:
            source_failures.append(row.source_file)
    add(checks, "source_hashes", len(source_failures), 0, not source_failures, f"Failures: {source_failures or 'none'}")
    output_manifest = pd.read_csv(OUTPUT / "output_manifest.csv")
    output_failures = []
    for row in output_manifest.itertuples(index=False):
        if row.output_file in {"independent_verification.csv", "independent_verification.json"}:
            continue
        path = OUTPUT / row.output_file
        if not path.exists() or sha256(path) != row.sha256:
            output_failures.append(row.output_file)
    add(checks, "builder_output_hashes", len(output_failures), 0, not output_failures, f"Failures: {output_failures or 'none'}")
    build_ok = as_bool(build_checks["passed"]).all()
    add(checks, "builder_checks", int(build_ok), 1, build_ok, "All builder checks pass.")

    result = pd.DataFrame(checks)
    result.to_csv(OUTPUT / "independent_verification.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    decisions = dict(zip(comparisons["candidate"], comparisons["decision"]))
    payload = {
        "checks_passed": int(result["passed"].sum()), "checks_total": len(result),
        "all_checks_passed": bool(result["passed"].all()), "historical_profile_rows": len(profiles),
        "complete_integration_cohort": len(cohort), "integration_decisions_verified": decisions,
        "historical_case_studies": len(cases), "profile_scope": "incumbent_club_extension",
    }
    (OUTPUT / "independent_verification.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    files = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name != "output_manifest.csv")
    manifest = pd.DataFrame([{"output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files])
    manifest.to_csv(OUTPUT / "output_manifest.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    if not result["passed"].all():
        failed = result.loc[~result["passed"], "check"].tolist()
        raise RuntimeError(f"Independent Phase-5 verification failed: {failed}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

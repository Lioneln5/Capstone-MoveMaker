"""Independently verify the Phase-2 extension survival diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Data" / "processed" / "extension_survival_diagnostic"
FROZEN_SOURCE = ROOT / "Data" / "processed" / "frozen_contract_sources" / "phase1_5_2026-08-11" / "contract_extension_integration"
SOURCE = FROZEN_SOURCE / "extension_modeling_master.csv"
UPSTREAM_VERIFY = FROZEN_SOURCE / "independent_verification.json"
AS_OF_DATE = pd.Timestamp("2026-08-10")
HORIZONS = {"12m": 365, "24m": 730, "36m": 1095, "48m": 1460}
ENDPOINTS = {
    "any_outbound": "first_any_outbound_date",
    "permanent_outbound": "first_permanent_outbound_date",
}
SURVIVAL_VARIANTS = {
    "M0_chronological_baseline", "M1_player_market_profile", "M2_plus_prior_opportunity",
    "M3_plus_performance_history", "M4_plus_contract_terms",
    "M5_plus_relative_financial_context", "M6_compact_nonlinear",
}
DIRECT_VARIANT = "D5_direct_horizon_classifier"
EXPECTED_DECISIONS = {
    ("any_outbound", "24m"): "advance_survival_probability_candidate",
    ("any_outbound", "36m"): "advance_survival_probability_candidate",
    ("permanent_outbound", "24m"): "advance_survival_probability_candidate",
    ("permanent_outbound", "36m"): "descriptive_or_exploratory_only",
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


def add(checks: list[dict], name: str, observed: object, expected: object, passed: bool, notes: str) -> None:
    checks.append({"check": name, "observed": observed, "expected": expected, "passed": bool(passed), "notes": notes})


def metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "brier": brier_score_loss(actual, prediction),
        "log_loss": log_loss(actual, prediction, labels=[0, 1]),
        "actual_rate": actual.mean(),
        "predicted_rate": prediction.mean(),
        "roc_auc": roc_auc_score(actual, prediction),
        "average_precision": average_precision_score(actual, prediction),
    }


def main() -> None:
    required = {
        "survival_model_matrix.csv", "survival_target_audit.csv", "at_risk_horizon_audit.csv",
        "contract_expiry_boundary_audit.csv", "person_period_audit.csv", "feature_manifest.csv",
        "model_origin_horizon_metrics.csv", "survival_predictions.csv", "model_comparison_results.csv",
        "survival_risk_bands.csv", "subgroup_stability.csv", "survival_decision_summary.csv",
        "build_checks.csv", "source_manifest.csv", "output_manifest.csv", "run_summary.json", "README.md",
    }
    missing = sorted(name for name in required if not (OUTPUT / name).exists())
    if missing:
        raise RuntimeError(f"Missing survival artifacts: {missing}")
    checks: list[dict] = []
    add(checks, "required_outputs", len(missing), 0, not missing, "All declared artifacts exist.")

    upstream = json.loads(UPSTREAM_VERIFY.read_text(encoding="utf-8"))
    add(checks, "upstream_verification", upstream.get("all_checks_passed"), True, bool(upstream.get("all_checks_passed")), "The contract integration passed independent verification.")

    master = pd.read_csv(SOURCE, low_memory=False)
    matrix = pd.read_csv(OUTPUT / "survival_model_matrix.csv", low_memory=False)
    predictions = pd.read_csv(OUTPUT / "survival_predictions.csv", low_memory=False)
    origin_metrics = pd.read_csv(OUTPUT / "model_origin_horizon_metrics.csv", low_memory=False)
    comparisons = pd.read_csv(OUTPUT / "model_comparison_results.csv", low_memory=False)
    bands = pd.read_csv(OUTPUT / "survival_risk_bands.csv", low_memory=False)
    decisions = pd.read_csv(OUTPUT / "survival_decision_summary.csv", low_memory=False)
    features = pd.read_csv(OUTPUT / "feature_manifest.csv", low_memory=False)
    build_checks = pd.read_csv(OUTPUT / "build_checks.csv", low_memory=False)
    expiry_audit = pd.read_csv(OUTPUT / "contract_expiry_boundary_audit.csv", low_memory=False)

    add(checks, "source_row_count", len(master), 2425, len(master) == 2425, "Clean extension source row count is unchanged.")
    add(checks, "matrix_event_grain", int(matrix["capology_extension_event_id"].duplicated().sum()), 0, not matrix["capology_extension_event_id"].duplicated().any(), "One matrix row per extension event.")
    add(checks, "matrix_source_ids", len(set(matrix["capology_extension_event_id"])), len(set(master["capology_extension_event_id"])), set(matrix["capology_extension_event_id"]) == set(master["capology_extension_event_id"]), "Matrix preserves the exact source ID set.")

    signed = pd.to_datetime(master["signed_date"], errors="coerce")
    expiration = pd.to_datetime(master["expiration_date"], errors="coerce")
    followup = (AS_OF_DATE - signed).dt.days
    cohort = master["canonical_player_id"].notna() & signed.notna() & expiration.notna() & signed.lt(AS_OF_DATE) & followup.gt(0)
    add(checks, "primary_cohort_recomputation", int(cohort.sum()), 2358, int(cohort.sum()) == 2358, "Canonical identity and positive administrative follow-up are independently reproduced.")
    matrix_lookup = matrix.set_index("capology_extension_event_id")
    source_ids = master["capology_extension_event_id"]
    cohort_observed = as_bool(matrix_lookup.loc[source_ids, "survival_primary_cohort"].reset_index(drop=True))
    add(checks, "primary_cohort_flags", int((cohort.reset_index(drop=True) != cohort_observed).sum()), 0, cohort.reset_index(drop=True).equals(cohort_observed), "Saved cohort flags match source recomputation.")

    target_mismatches = 0
    expected_event_counts = {"any_outbound": 1848, "permanent_outbound": 1690}
    for endpoint, date_column in ENDPOINTS.items():
        event_date = pd.to_datetime(master[date_column], errors="coerce")
        event = event_date.notna() & event_date.le(AS_OF_DATE)
        event_days = (event_date - signed).dt.days
        observed = matrix_lookup.loc[source_ids]
        target_mismatches += int((pd.to_numeric(observed[f"{endpoint}_event"], errors="coerce").to_numpy() != event.astype(int).to_numpy()).sum())
        expected_followup = event_days.where(event, followup)
        target_mismatches += int((~np.isclose(pd.to_numeric(observed[f"{endpoint}_followup_days"], errors="coerce"), expected_followup, equal_nan=True)).sum())
        for label, horizon in HORIZONS.items():
            target = (event & event_days.le(horizon)).astype(int)
            observable = event | followup.ge(horizon)
            target_mismatches += int((pd.to_numeric(observed[f"{endpoint}_event_by_{label}"], errors="coerce").to_numpy() != target.to_numpy()).sum())
            target_mismatches += int((as_bool(observed[f"{endpoint}_horizon_observable_{label}"]).to_numpy() != observable.to_numpy()).sum())
        add(checks, f"{endpoint}_event_count", int(event.loc[cohort].sum()), expected_event_counts[endpoint], int(event.loc[cohort].sum()) == expected_event_counts[endpoint], "Dated events through the administrative cutoff are reproduced.")
    add(checks, "target_recomputation", target_mismatches, 0, target_mismatches == 0, "Event, follow-up, horizon target, and observability fields reproduce from dates.")

    expiration_rule = not as_bool(expiry_audit["expiration_used_as_censor"]).any()
    add(checks, "expiration_not_censor", int(expiration_rule), 1, expiration_rule, "Scheduled expiration is audited but does not truncate club-stay follow-up.")
    expiry_counts_ok = True
    for row in expiry_audit.itertuples(index=False):
        event_date = pd.to_datetime(master[ENDPOINTS[row.endpoint]], errors="coerce")
        event = event_date.notna() & event_date.le(AS_OF_DATE) & cohort
        before = int((event & event_date.le(expiration)).sum())
        after = int((event & event_date.gt(expiration)).sum())
        expiry_counts_ok &= before == row.events_before_or_on_expiry and after == row.events_after_expiry
    add(checks, "expiry_boundary_counts", int(expiry_counts_ok), 1, expiry_counts_ok, "Before/after-expiration event counts independently reproduce.")

    timing_safe = not features["feature"].str.startswith("post_").any()
    timing_safe &= features.loc[features["feature"].ne("hazard_interval"), "timing"].eq("known_at_extension_signing").all()
    add(checks, "feature_timing", int(timing_safe), 1, timing_safe, "No post-signing player information enters a model.")

    prediction_key = ["endpoint", "origin_id", "horizon", "model_variant", "capology_extension_event_id"]
    add(checks, "prediction_grain", int(predictions.duplicated(prediction_key).sum()), 0, not predictions.duplicated(prediction_key).any(), "One prediction per endpoint/origin/horizon/model/event.")
    add(checks, "prediction_row_count", len(predictions), 99072, len(predictions) == 99072, "Prediction count matches declared models and available direct comparators.")
    bounds = predictions["predicted_event_probability"].between(0, 1).all() and predictions["predicted_survival_probability"].between(0, 1).all()
    add(checks, "probability_bounds", int(bounds), 1, bounds, "Event and stay probabilities are valid.")
    complements = np.allclose(predictions["predicted_event_probability"] + predictions["predicted_survival_probability"], 1.0)
    add(checks, "probability_complements", int(complements), 1, complements, "Event risk and stay probability sum to one.")

    survival = predictions[predictions["model_variant"].isin(SURVIVAL_VARIANTS)]
    wide = survival.pivot_table(index=["endpoint", "origin_id", "model_variant", "capology_extension_event_id"], columns="horizon_days", values="predicted_event_probability")
    ordered = wide[365].le(wide[730]).all() and wide[730].le(wide[1095]).all() and wide[1095].le(wide[1460]).all()
    add(checks, "survival_curve_ordering", int(ordered), 1, ordered, "Cumulative outbound risk never decreases with horizon.")

    observability_mismatches = 0
    actual_mismatches = 0
    matrix_targets = matrix_lookup
    for (endpoint, origin_id, horizon), group in predictions.groupby(["endpoint", "origin_id", "horizon"], observed=True):
        days = HORIZONS[horizon]
        ids = group["capology_extension_event_id"].unique()
        signed_group = pd.to_datetime(matrix_targets.loc[ids, "signed_date"], errors="coerce")
        complete = bool(((AS_OF_DATE - signed_group).dt.days >= days).all())
        observability_mismatches += int(as_bool(group["horizon_fully_observable"]).nunique() != 1)
        observability_mismatches += int(as_bool(group["horizon_fully_observable"]).iat[0] != complete)
        if complete:
            expected = pd.to_numeric(matrix_targets.loc[group["capology_extension_event_id"], f"{endpoint}_event_by_{horizon}"], errors="coerce").to_numpy()
            actual_mismatches += int((~np.isclose(group["actual_event_by_horizon"].to_numpy(), expected)).sum())
        else:
            actual_mismatches += int(group["actual_event_by_horizon"].notna().sum())
    add(checks, "origin_horizon_observability", observability_mismatches, 0, observability_mismatches == 0, "Evaluation is scored only when the full signing-year cohort reaches the horizon.")
    add(checks, "prediction_actual_values", actual_mismatches, 0, actual_mismatches == 0, "Saved actuals match source targets, and incomplete origins remain unscored.")

    metric_key = ["endpoint", "origin_id", "horizon", "model_variant"]
    add(checks, "origin_metric_grain", int(origin_metrics.duplicated(metric_key).sum()), 0, not origin_metrics.duplicated(metric_key).any(), "One metric row per scored endpoint/origin/horizon/model.")
    add(checks, "origin_metric_row_count", len(origin_metrics), 208, len(origin_metrics) == 208, "All administratively complete origin/horizon/model combinations are reported.")
    metric_mismatches = 0
    scored = predictions[predictions["horizon_fully_observable"].eq(True)]
    for key, group in scored.groupby(metric_key, observed=True):
        saved = origin_metrics.loc[
            origin_metrics["endpoint"].eq(key[0]) & origin_metrics["origin_id"].eq(key[1])
            & origin_metrics["horizon"].eq(key[2]) & origin_metrics["model_variant"].eq(key[3])
        ].iloc[0]
        recalculated = metrics(group["actual_event_by_horizon"].to_numpy(int), group["predicted_event_probability"].to_numpy(float))
        metric_mismatches += sum(not np.isclose(saved[name], value, atol=1e-10, rtol=1e-10) for name, value in recalculated.items())
    add(checks, "origin_metric_recomputation", metric_mismatches, 0, metric_mismatches == 0, "All evaluation metrics reproduce from row-level predictions.")

    add(checks, "comparison_row_count", len(comparisons), 162, len(comparisons) == 2 * 3 * 3 * 9, "All endpoint/horizon/window/comparison combinations exist.")
    comparison_mismatches = 0
    for row in comparisons.itertuples(index=False):
        years = [int(value.strip()) for value in str(row.evaluation_years).split("|")]
        subset = predictions[
            predictions["endpoint"].eq(row.endpoint) & predictions["horizon"].eq(row.horizon)
            & predictions["evaluation_year"].isin(years) & predictions["horizon_fully_observable"].eq(True)
        ]
        keys = ["origin_id", "evaluation_year", "capology_extension_event_id", "canonical_player_id"]
        base = subset[subset["model_variant"].eq(row.baseline)][keys + ["primary_loss"]].rename(columns={"primary_loss": "base"})
        candidate = subset[subset["model_variant"].eq(row.candidate)][keys + ["primary_loss"]].rename(columns={"primary_loss": "candidate"})
        pair = base.merge(candidate, on=keys, validate="one_to_one")
        delta = pair["base"] - pair["candidate"]
        wins = int(pair.assign(delta=delta).groupby("evaluation_year")["delta"].mean().gt(0).sum())
        comparison_mismatches += int(len(pair) != row.evaluation_rows)
        comparison_mismatches += int(not np.isclose(delta.mean(), row.mean_brier_improvement, atol=1e-12, rtol=1e-12))
        comparison_mismatches += int(wins != row.origin_wins)
    add(checks, "comparison_recomputation", comparison_mismatches, 0, comparison_mismatches == 0, "Paired improvements, row counts, and origin wins reproduce.")

    add(checks, "risk_band_row_count", len(bands), 30, len(bands) == 2 * 3 * 5, "Five M5 risk bands exist for each endpoint at 12, 24, and 36 months.")
    band_mismatches = 0
    for _, group in bands.groupby(["endpoint", "horizon"], observed=True):
        monotone = bool(group.sort_values("risk_band")["observed_event_rate"].is_monotonic_increasing)
        band_mismatches += int(as_bool(group["observed_monotone"]).nunique() != 1)
        band_mismatches += int(as_bool(group["observed_monotone"]).iat[0] != monotone)
    add(checks, "risk_band_flags", band_mismatches, 0, band_mismatches == 0, "Monotonicity flags match observed rates.")

    decision_map = {(row.endpoint, row.horizon): row.status for row in decisions.itertuples(index=False)}
    add(checks, "decision_set", len(decision_map), 4, set(decision_map) == set(EXPECTED_DECISIONS), "Exactly two endpoints at two primary horizons are adjudicated.")
    add(checks, "decision_results", decision_map, EXPECTED_DECISIONS, decision_map == EXPECTED_DECISIONS, "Saved decisions match the evidence gates for this run.")
    decision_trace_mismatches = 0
    for row in decisions.itertuples(index=False):
        total = comparisons[
            comparisons["endpoint"].eq(row.endpoint) & comparisons["horizon"].eq(row.horizon)
            & comparisons["window"].eq("pooled_four") & comparisons["comparison"].eq("M5_vs_M0_total")
        ].iloc[0]
        contract = comparisons[
            comparisons["endpoint"].eq(row.endpoint) & comparisons["horizon"].eq(row.horizon)
            & comparisons["window"].eq("pooled_four") & comparisons["comparison"].eq("M5_vs_M3_all_contract_economics")
        ].iloc[0]
        method = comparisons[
            comparisons["endpoint"].eq(row.endpoint) & comparisons["horizon"].eq(row.horizon)
            & comparisons["window"].eq("pooled_four") & comparisons["comparison"].eq("M5_survival_vs_D5_direct")
        ].iloc[0]
        decision_trace_mismatches += int(not np.isclose(row.total_brier_improvement_vs_baseline, total.mean_brier_improvement))
        decision_trace_mismatches += int(not np.isclose(row.contract_brier_improvement, contract.mean_brier_improvement))
        decision_trace_mismatches += int(not np.isclose(row.survival_vs_direct_brier_improvement, method.mean_brier_improvement))
    add(checks, "decision_traceability", decision_trace_mismatches, 0, decision_trace_mismatches == 0, "Decision statistics trace to registered pooled comparisons.")

    source_manifest = pd.read_csv(OUTPUT / "source_manifest.csv")
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
    add(checks, "builder_checks", int(build_ok), 1, build_ok, "All builder checks passed.")

    result = pd.DataFrame(checks)
    result.to_csv(OUTPUT / "independent_verification.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    payload = {
        "checks_passed": int(result["passed"].sum()), "checks_total": len(result),
        "all_checks_passed": bool(result["passed"].all()),
        "source_extension_events": len(master), "survival_primary_cohort": int(cohort.sum()),
        "prediction_rows_verified": len(predictions),
        "decisions_verified": {f"{key[0]}_{key[1]}": value for key, value in decision_map.items()},
    }
    (OUTPUT / "independent_verification.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    files = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name != "output_manifest.csv")
    manifest = pd.DataFrame([{"output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files])
    manifest.to_csv(OUTPUT / "output_manifest.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    if not result["passed"].all():
        failed = result.loc[~result["passed"], "check"].tolist()
        raise RuntimeError(f"Independent survival verification failed: {failed}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

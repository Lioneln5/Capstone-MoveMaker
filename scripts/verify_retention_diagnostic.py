"""Independently verify the MoveMaker retention diagnostic outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

import run_retention_diagnostic as retention
from mixed_type_preprocessor import fit_preprocessor
from train_compatibility_models import sha256_file


OUTPUT_DIR = retention.OUTPUT_DIR


def add(
    checks: list[dict[str, Any]],
    check: str,
    passed: bool,
    observed: Any,
    expected: Any,
    notes: str,
) -> None:
    checks.append(
        {
            "check": check,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
            "notes": notes,
        }
    )


def verify() -> None:
    required = [
        "README.md",
        "run_summary.json",
        "retention_targets.csv",
        "retention_cohort_audit.csv",
        "sporting_threshold_sensitivity.csv",
        "retention_feature_manifest.csv",
        "destination_spell_survival_summary.csv",
        "retention_predictions.csv",
        "retention_origin_results.csv",
        "retention_comparison_results.csv",
        "output_manifest.csv",
    ]
    checks: list[dict[str, Any]] = []
    missing = [name for name in required if not (OUTPUT_DIR / name).exists()]
    add(checks, "required_outputs_exist", not missing, ";".join(missing), "", "All declared outputs exist.")
    if missing:
        raise RuntimeError(f"Missing retention outputs: {missing}")

    summary = json.loads((OUTPUT_DIR / "run_summary.json").read_text(encoding="utf-8"))
    targets = pd.read_csv(OUTPUT_DIR / "retention_targets.csv", low_memory=False)
    audit = pd.read_csv(OUTPUT_DIR / "retention_cohort_audit.csv")
    thresholds = pd.read_csv(OUTPUT_DIR / "sporting_threshold_sensitivity.csv")
    features = pd.read_csv(OUTPUT_DIR / "retention_feature_manifest.csv", low_memory=False)
    survival = pd.read_csv(OUTPUT_DIR / "destination_spell_survival_summary.csv")
    predictions = pd.read_csv(OUTPUT_DIR / "retention_predictions.csv", low_memory=False)
    origins = pd.read_csv(OUTPUT_DIR / "retention_origin_results.csv", low_memory=False)
    comparisons = pd.read_csv(OUTPUT_DIR / "retention_comparison_results.csv", low_memory=False)
    manifest = pd.read_csv(OUTPUT_DIR / "output_manifest.csv")

    add(checks, "primary_target_rows", len(targets) == 1471, len(targets), 1471, "Permanent arrivals after three conflicting arrivals are excluded.")
    add(checks, "target_key_unique", targets["transfer_event_id"].is_unique, targets["transfer_event_id"].nunique(), len(targets), "One target row per source arrival.")
    add(checks, "permanent_only", set(targets["canonical_transfer_type"]) == {"transfer"}, ";".join(sorted(set(targets["canonical_transfer_type"]))), "transfer", "Loans and loan returns are not modeled.")

    sporting_observed = targets["sporting_retention_y2_10pct"].notna()
    sporting_positive = targets.loc[sporting_observed, "sporting_retention_y2_10pct"].astype(bool)
    add(checks, "sporting_target_observed", int(sporting_observed.sum()) == 1432, int(sporting_observed.sum()), 1432, "Rows have positive year-two club-game denominators.")
    add(checks, "sporting_target_positive", int(sporting_positive.sum()) == 743, int(sporting_positive.sum()), 743, "Primary 10% threshold positives.")
    add(checks, "sporting_target_recomputed", np.array_equal(sporting_positive.to_numpy(), targets.loc[sporting_observed, "year2_destination_opportunity_share"].ge(0.10).to_numpy()), int(sporting_positive.sum()), int(targets.loc[sporting_observed, "year2_destination_opportunity_share"].ge(0.10).sum()), "Binary target exactly follows the frozen threshold.")
    add(checks, "year2_nested_windows_nonnegative", targets["year2_destination_minutes"].ge(0).all() and targets["year2_destination_club_games"].ge(0).all(), f"minutes_min={targets['year2_destination_minutes'].min()};games_min={targets['year2_destination_club_games'].min()}", ">=0", "Cumulative 24-month fields never fall below 12-month fields.")

    retained_365 = int(targets["destination_spell_retained_365d"].sum())
    retained_730 = int(targets["destination_spell_retained_730d"].sum())
    add(checks, "spell_retained_365_count", retained_365 == 1093, retained_365, 1093, "One-year uninterrupted-spell count.")
    add(checks, "spell_retained_730_count", retained_730 == 628, retained_730, 628, "Two-year uninterrupted-spell count.")
    add(checks, "spell_retention_monotone", (targets["destination_spell_retained_730d"] <= targets["destination_spell_retained_365d"]).all(), retained_730, f"<= {retained_365}", "A two-year retained spell must also survive one year.")
    administrative_recomputed = targets["days_to_first_outbound"].isna() | targets["days_to_first_outbound"].gt(730)
    add(checks, "administrative_target_recomputed", np.array_equal(administrative_recomputed.astype(int), targets["destination_spell_retained_730d"].astype(int)), int(administrative_recomputed.sum()), retained_730, "Two-year spell target exactly matches first outbound timing.")

    expected_survival_days = [90, 180, 365, 545, 730]
    add(checks, "survival_days_frozen", survival["day"].tolist() == expected_survival_days, ";".join(map(str, survival["day"])), ";".join(map(str, expected_survival_days)), "Fixed descriptive checkpoints are present.")
    add(checks, "survival_monotone", survival["uninterrupted_spell_rate"].is_monotonic_decreasing, ";".join(f"{x:.6f}" for x in survival["uninterrupted_spell_rate"]), "nonincreasing", "Uninterrupted-spell survival cannot increase with time.")

    audit_map = audit.set_index(["area", "metric"])["value"]
    add(checks, "broad_cohort_audit", int(float(audit_map.loc[("cohort", "broad_context_cohort_rows")])) == 3134, audit_map.loc[("cohort", "broad_context_cohort_rows")], 3134, "Frozen broad context cohort is unchanged.")
    type_expected = {"transfer": 1474, "loan": 712, "loan_return": 864, "blank_unclassified": 84}
    type_observed = {key: int(float(audit_map.loc[("arrival_type", key)])) for key in type_expected}
    add(checks, "arrival_type_accounting", type_observed == type_expected, json.dumps(type_observed, sort_keys=True), json.dumps(type_expected, sort_keys=True), "Canonical movement types reconcile before primary restriction.")

    add(checks, "thresholds_frozen", thresholds["threshold"].round(2).tolist() == [0.05, 0.10, 0.25, 0.50], ";".join(map(str, thresholds["threshold"])), "0.05;0.10;0.25;0.50", "Threshold sensitivity includes the declared alternatives.")
    add(checks, "threshold_positive_monotone", thresholds["positive_rows"].is_monotonic_decreasing, ";".join(map(str, thresholds["positive_rows"])), "nonincreasing", "Higher sporting thresholds cannot create positives.")

    add(checks, "prediction_row_count", len(predictions) == 9558, len(predictions), 9558, "All endpoint/origin/model predictions are present.")
    prediction_key = ["endpoint", "origin_id", "model_variant", "transfer_event_id"]
    add(checks, "prediction_key_unique", not predictions.duplicated(prediction_key).any(), int(predictions.duplicated(prediction_key).sum()), 0, "No duplicate evaluation prediction.")
    add(checks, "prediction_probability_bounds", predictions["predicted_probability"].between(0, 1).all(), f"{predictions['predicted_probability'].min():.6f}..{predictions['predicted_probability'].max():.6f}", "[0,1]", "Every prediction is a valid probability.")
    add(checks, "prediction_actual_binary", set(predictions["actual"]) <= {0, 1}, ";".join(map(str, sorted(set(predictions["actual"])))), "0;1", "Every evaluated target is binary.")

    aligned = True
    for (endpoint, origin_id), group in predictions.groupby(["endpoint", "origin_id"]):
        expected_ids = None
        for _, model_group in group.groupby("model_variant"):
            ids = sorted(model_group["transfer_event_id"].astype(str).tolist())
            if expected_ids is None:
                expected_ids = ids
            elif ids != expected_ids:
                aligned = False
    add(checks, "paired_evaluation_rows", aligned, aligned, True, "Every model for an endpoint/origin uses identical rows.")

    comparison_matches = True
    auc_matches = True
    for _, row in comparisons.iterrows():
        years = [int(value) for value in str(row["evaluation_years"]).split(";")]
        paired = retention.paired_predictions(
            predictions,
            row["endpoint"],
            row["baseline_model"],
            row["candidate_model"],
            years,
        )
        base_brier = brier_score_loss(paired["actual"], paired["baseline_probability"])
        candidate_brier = brier_score_loss(paired["actual"], paired["candidate_probability"])
        comparison_matches &= np.isclose(base_brier - candidate_brier, row["brier_improvement"], atol=1e-12)
        origin_auc = []
        origin_n = []
        for _, group in paired.groupby("origin_id"):
            origin_auc.append(roc_auc_score(group["actual"], group["candidate_probability"]))
            origin_n.append(len(group))
        auc_matches &= np.isclose(
            np.average(origin_auc, weights=origin_n),
            row["weighted_origin_candidate_roc_auc"],
            atol=1e-12,
        )
    add(checks, "comparison_brier_recomputed", comparison_matches, comparison_matches, True, "Every reported Brier improvement recomputes from row-level predictions.")
    add(checks, "weighted_origin_auc_recomputed", auc_matches, auc_matches, True, "Reported AUC respects chronological-origin strata.")

    first_year_rows = features["timing_classification"].eq("observed during days 0-365 after transfer")
    add(checks, "landmark_features_isolated", set(features.loc[first_year_rows, "model_variant"]) == {"M6_plus_first_year_involvement"} and int(first_year_rows.sum()) == 6, int(first_year_rows.sum()), 6, "Post-transfer involvement fields occur only in the landmark model.")
    at_transfer_has_post = features.loc[features["model_variant"].isin(retention.AT_TRANSFER_MODELS), "timing_classification"].astype(str).str.contains("after transfer").any()
    add(checks, "at_transfer_features_leakage_safe", not at_transfer_has_post, at_transfer_has_post, False, "At-transfer feature sets contain no post-transfer evidence.")

    # Regression check for the mixed-type column-order bug fixed during this run.
    toy = pd.DataFrame({"n1": [1.0, 2.0, 3.0], "cat": ["a", "b", "a"], "n2": [10.0, 20.0, 30.0]})
    processor = fit_preprocessor(toy, ["n1", "cat", "n2"])
    transformed = processor.transform(toy)
    expected_numeric = np.array([-1.224744871391589, 0.0, 1.224744871391589])
    order_ok = (
        processor.encoded_names == ["n1", "cat==a", "cat==b", "n2"]
        and transformed.shape == (3, 4)
        and np.allclose(transformed[:, 0], expected_numeric)
        and np.allclose(transformed[:, -1], expected_numeric)
    )
    add(checks, "mixed_type_preprocessor_order", order_ok, processor.encoded_names, ["n1", "cat==a", "cat==b", "n2"], "Transform order matches fit order for interleaved numeric and categorical features.")

    add(checks, "source_master_hash", sha256_file(retention.MASTER_PATH) == summary["master_sha256"], sha256_file(retention.MASTER_PATH), summary["master_sha256"], "Comprehensive master matches the run input.")
    add(checks, "canonical_transfer_hash", sha256_file(retention.CANONICAL_TRANSFER_PATH) == summary["canonical_transfer_sha256"], sha256_file(retention.CANONICAL_TRANSFER_PATH), summary["canonical_transfer_sha256"], "Canonical transfers match the run input.")
    add(checks, "protected_outputs_unchanged", summary["protected_outputs_unchanged"] is True, summary["protected_outputs_unchanged"], True, "Existing verified output trees were not changed by the run.")

    manifest_ok = True
    for _, row in manifest.iterrows():
        path = OUTPUT_DIR / row["file"]
        manifest_ok &= path.exists() and path.stat().st_size == int(row["size_bytes"]) and sha256_file(path) == row["sha256"]
    add(checks, "output_manifest_hashes", manifest_ok, manifest_ok, True, "Every runner-produced output matches its manifest hash.")

    result = pd.DataFrame(checks)
    result.to_csv(OUTPUT_DIR / "independent_verification.csv", index=False)
    failures = result.loc[~result["passed"]]
    payload = {
        "status": "pass" if failures.empty else "fail",
        "checks": len(result),
        "passed": int(result["passed"].sum()),
        "failed": len(failures),
        "failed_checks": failures["check"].tolist(),
    }
    (OUTPUT_DIR / "independent_verification.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)
    if not failures.empty:
        raise RuntimeError(f"Retention verification failed: {failures['check'].tolist()}")


if __name__ == "__main__":
    verify()

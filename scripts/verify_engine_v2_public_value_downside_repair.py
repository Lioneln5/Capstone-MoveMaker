"""Independent verification of the public-value downside cleanup."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "models") not in sys.path:
    sys.path.insert(0, str(ROOT / "models"))

from models.engine_v2.feature_contract import ADVANCED_PERFORMANCE, COMMERCIAL_SCENARIO, RAW_SCORING_RATES  # noqa: E402
from models.engine_v2.public_value_downside_repair_contract import (  # noqa: E402
    CANDIDATE_FEATURES,
    CONTRACT_VERSION,
    ENDPOINTS,
    LOGISTIC_C_GRID,
    ORIGINS,
    PRIMARY_CANDIDATE,
    RANDOM_SEED,
)
from models.mixed_type_preprocessor import fit_preprocessor  # noqa: E402
import run_engine_v2_public_value_downside_repair as run  # noqa: E402


OUTPUT = ROOT / "Data" / "processed" / "engine_v2_public_value_downside_repair"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"
FEATURES = list(CANDIDATE_FEATURES[PRIMARY_CANDIDATE])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    checks = []

    def add(check: str, passed: bool, observed: object, expected: object, notes: str) -> None:
        checks.append({"check": check, "passed": bool(passed), "observed": observed, "expected": expected, "notes": notes})

    prereg = json.loads((OUTPUT / "preregistration.json").read_text(encoding="utf-8-sig"))
    cohorts = pd.read_csv(OUTPUT / "cohort_by_origin.csv")
    candidates = pd.read_csv(OUTPUT / "candidate_predictions.csv")
    calibration = pd.read_csv(OUTPUT / "calibration_predictions.csv")
    selection = pd.read_csv(OUTPUT / "calibration_selection.csv")
    selected = pd.read_csv(OUTPUT / "selected_predictions.csv")
    comparisons = pd.read_csv(OUTPUT / "candidate_comparisons.csv")
    subgroups = pd.read_csv(OUTPUT / "subgroup_reliability.csv")
    exposure = pd.read_csv(OUTPUT / "known_exposure_audit.csv")
    redundancy = pd.read_csv(OUTPUT / "feature_redundancy.csv")
    decisions = pd.read_csv(OUTPUT / "selection_decision.csv")
    summary = json.loads((OUTPUT / "run_summary.json").read_text(encoding="utf-8-sig"))
    source_manifest = pd.read_csv(OUTPUT / "source_manifest.csv")
    output_manifest = pd.read_csv(OUTPUT / "output_manifest.csv")

    add("contract_version_exact", prereg["contract_version"] == CONTRACT_VERSION, prereg["contract_version"], CONTRACT_VERSION, "Evidence is tied to the frozen contract.")
    source_mismatches = [
        row.path for row in source_manifest.itertuples(index=False)
        if not (ROOT / row.path).is_file() or sha256(ROOT / row.path) != row.sha256
    ]
    add("source_manifest_exact", not source_mismatches, source_mismatches, [], "Every declared input hash matches the current source.")
    output_mismatches = [
        row.file for row in output_manifest.itertuples(index=False)
        if not (OUTPUT / row.file).is_file() or sha256(OUTPUT / row.file) != row.sha256
    ]
    add("prior_output_manifest_exact", not output_mismatches, output_mismatches, [], "The pre-verification evidence package has not changed silently.")
    add("endpoints_exact", set(decisions.endpoint) == set(ENDPOINTS), sorted(decisions.endpoint), sorted(ENDPOINTS), "Only the 25% target at 12m and 24m is modeled.")
    add("origins_exact", set(cohorts.origin_id) == {item[0] for item in ORIGINS} and len(cohorts) == 8, sorted(cohorts.origin_id.unique()), [item[0] for item in ORIGINS], "Four origins per endpoint.")
    add("no_2024plus_predictions", selected.evaluation_year.max() == 2023 and candidates.evaluation_year.max() == 2023, [int(candidates.evaluation_year.max()), int(selected.evaluation_year.max())], [2023, 2023], "Exposed years never enter evaluation.")
    identity_columns = ["identity_overlap", "tuning_identity_overlap", "calibration_identity_overlap"]
    add("identity_disjoint", cohorts[identity_columns].eq(0).all().all(), cohorts[identity_columns].to_dict("records"), "all zero", "No evaluation player appears in fitting, tuning, or calibration rows.")
    prohibited = set(RAW_SCORING_RATES + ADVANCED_PERFORMANCE + COMMERCIAL_SCENARIO)
    add("proposal_and_scoring_features_absent", not (set(FEATURES) & prohibited), sorted(set(FEATURES) & prohibited), [], "Candidate is decision-time-valid and role-safe with respect to scoring rates.")
    add("involvement_redundancy_disclosed", redundancy.high_redundancy.all() and redundancy.required_explanation_policy.eq("group_only_recent_involvement").all(), redundancy[["endpoint", "spearman", "required_explanation_policy"]].to_dict("records"), "group-only explanation", "Correlated involvement inputs cannot receive separate causal importance claims.")

    reconstructed_diffs = []
    reconstructed_cs = []
    for endpoint, endpoint_spec in ENDPOINTS.items():
        target = endpoint_spec["target"]
        frame = run.prepare_frame(endpoint)
        for origin_spec in ORIGINS:
            origin = origin_spec[0]
            tuning, validation, final_fit, evaluation = run.split_origin(frame, origin_spec)
            _, _, selected_c, _ = run.tune_c(tuning, validation, FEATURES, target)
            pre = fit_preprocessor(final_fit, FEATURES)
            model = LogisticRegression(
                C=selected_c, penalty="l2", solver="liblinear", max_iter=3000,
                random_state=RANDOM_SEED,
            ).fit(pre.transform(final_fit), pd.to_numeric(final_fit[target]).to_numpy(int))
            expected = np.clip(model.predict_proba(pre.transform(evaluation))[:, 1], 1e-6, 1 - 1e-6)
            saved = candidates.loc[
                candidates.endpoint.eq(endpoint)
                & candidates.origin_id.eq(origin)
                & candidates.candidate.eq(PRIMARY_CANDIDATE)
            ].sort_values("capology_extension_event_id")
            reconstructed_diffs.append(float(np.max(np.abs(expected - saved.prediction.to_numpy(float)))))
            reconstructed_cs.append(float(abs(selected_c - saved.selected_C.iloc[0])))
            add(
                f"cohort_ids_{endpoint}_{origin}",
                evaluation.capology_extension_event_id.astype(str).tolist() == saved.capology_extension_event_id.astype(str).tolist(),
                len(saved), len(evaluation), "No event was silently added or removed.",
            )
    add("raw_predictions_reconstructed", max(reconstructed_diffs) < 1e-12, max(reconstructed_diffs), "<1e-12", "Independent refits reproduce primary predictions.")
    add("selected_C_reconstructed", max(reconstructed_cs) < 1e-12, max(reconstructed_cs), "<1e-12", "Hyperparameters derive only from preceding-year validation.")

    selected_methods = selection.loc[selection.selected].set_index("endpoint").calibration_method.to_dict()
    add("one_calibrator_per_endpoint", set(selected_methods) == set(ENDPOINTS), selected_methods, "one per endpoint", "Calibration selection is explicit.")
    expected_selected = calibration.loc[
        calibration.apply(lambda row: row.calibration_method == selected_methods[row.endpoint], axis=1)
    ].sort_values(["endpoint", "origin_id", "capology_extension_event_id"])
    saved_selected = selected.sort_values(["endpoint", "origin_id", "capology_extension_event_id"])
    add("selected_calibration_rows_exact", expected_selected.capology_extension_event_id.astype(str).tolist() == saved_selected.capology_extension_event_id.astype(str).tolist(), len(saved_selected), len(expected_selected), "Selected rows come from the declared calibrator only.")
    add("selected_probabilities_exact", np.max(np.abs(expected_selected.prediction.to_numpy(float) - saved_selected.prediction.to_numpy(float))) < 1e-12, float(np.max(np.abs(expected_selected.prediction.to_numpy(float) - saved_selected.prediction.to_numpy(float)))), "<1e-12", "No post-selection prediction edits.")

    gate_diffs = []
    for row in decisions.itertuples(index=False):
        chunk = saved_selected.loc[saved_selected.endpoint.eq(row.endpoint)]
        gate_diffs.extend([
            abs(brier_score_loss(chunk.actual, chunk.prediction) - row.pooled_brier),
            abs(roc_auc_score(chunk.actual, chunk.prediction) - row.pooled_roc_auc),
            abs(abs(chunk.actual.mean() - chunk.prediction.mean()) - row.pooled_calibration_gap),
        ])
    add("pooled_metrics_reconstructed", max(gate_diffs) < 1e-12, max(gate_diffs), "<1e-12", "Decision metrics derive from row-level development evidence.")
    add("comparison_intervals_ordered", comparisons.brier_improvement_ci_low_95.le(comparisons.brier_improvement_ci_high_95).all(), len(comparisons), "all ordered", "Candidate uncertainty is explicit.")
    add("parameter_intervals_ordered", selected.parameter_interval_low_80.le(selected.parameter_interval_high_80).all(), len(selected), "all ordered", "Every prediction has an 80% refit interval.")
    add("parameter_refits_complete", selected.successful_parameter_refits.ge(95).all(), int(selected.successful_parameter_refits.min()), ">=95", "At least 95% of predeclared refits succeeded.")
    add("subgroup_intervals_ordered", subgroups.brier_skill_ci_low_99.le(subgroups.brier_skill_ci_high_99).all(), len(subgroups), "all ordered", "Subgroup uncertainty is not hidden.")
    add("all_broad_groups_reported", set(subgroups.group_type) >= {"league", "position"}, sorted(subgroups.group_type.unique()), "league and position", "Advertised population groups are audited.")
    add("refusals_explained", selected.loc[selected.support_status.eq("refused"), "support_reasons"].fillna("").str.len().gt(0).all(), int(selected.support_status.eq("refused").sum()), "all explained", "No silent out-of-support score.")

    expected_counts = {
        "downside_25pct_12m": (376, 332, 209),
        "downside_25pct_24m": (41, 41, 26),
    }
    # The first value is the V1 verifier's exposed 2024+ count after this
    # repair's complete-feature contract; the remaining values are 2024 and
    # player-disjoint 2024 counts under the same contract.
    observed_counts = {
        row.endpoint: (
            int(row.exposed_2024plus_complete_feature_rows),
            int(row.exposed_2024_complete_feature_rows),
            int(row.exposed_2024_player_disjoint_rows),
        ) for row in exposure.itertuples(index=False)
    }
    add("exposure_counts_reconstructed", observed_counts == expected_counts, observed_counts, expected_counts, "Counts use the repaired complete-feature contract.")
    add("exposure_classification_permanent", exposure.classification.eq("previously_exposed_restricted_V1_confirmation_only").all() and not exposure.permitted_for_v2_selection_or_promotion.any(), exposure[["endpoint", "classification"]].to_dict("records"), "restricted only", "A fresh chat cannot reset data exposure.")
    add("no_final_release_authorized", not decisions.final_release_authorized.any() and not summary.get("endpoints", [{}])[0].get("final_release_authorized", True), decisions[["endpoint", "final_release_authorized"]].to_dict("records"), "all false", "Development evidence cannot masquerade as final validation.")
    add("no_model_artifacts", not list(OUTPUT.rglob("*.joblib")), len(list(OUTPUT.rglob("*.joblib"))), 0, "Cleanup writes evidence only.")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    mutable_verification_reports = {
        "Data/processed/deployment_models/live_request_integrity.csv",
        "Data/processed/deployment_models/live_request_integrity.json",
    }
    changed = [
        item["path"] for item in freeze["files"]
        if item["path"] not in mutable_verification_reports
        and sha256(ROOT / item["path"]) != item["sha256"]
    ]
    add("frozen_v1_unchanged", not changed, changed, [], "Historical V1 remains byte-identical.")

    results = pd.DataFrame(checks)
    results.to_csv(OUTPUT / "independent_verification.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    payload = {
        "all_checks_passed": bool(results.passed.all()),
        "checks_passed": int(results.passed.sum()),
        "checks_total": int(len(results)),
        "failed_checks": results.loc[~results.passed, "check"].tolist(),
    }
    (OUTPUT / "independent_verification.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not payload["all_checks_passed"]:
        raise RuntimeError(results.loc[~results.passed].to_string(index=False))
    files = [path for path in sorted(OUTPUT.iterdir()) if path.is_file() and path.name != "output_manifest.csv"]
    pd.DataFrame([
        {"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
        for path in files
    ]).to_csv(OUTPUT / "output_manifest.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

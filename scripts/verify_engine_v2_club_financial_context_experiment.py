"""Independent verification for the club-context rolling-origin experiment."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Data" / "processed" / "engine_v2_club_financial_context_experiment"
RUNNER = ROOT / "models" / "run_engine_v2_club_financial_context_experiment.py"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    checks: list[dict] = []

    def check(name: str, passed: bool, observed, expected, note: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "observed": observed, "expected": expected, "note": note})

    required = {
        "README.md", "preregistration.json", "source_manifest.csv", "build_checks.csv",
        "paired_predictions.csv", "origin_metrics.csv", "coverage_by_origin.csv",
        "candidate_comparison.csv", "subgroup_metrics.csv", "decision_capacity_lift.csv",
        "selection_decision.csv", "coverage_selection_sensitivity.csv", "support_profiles.json",
        "run_summary.json", "output_manifest.csv",
    }
    present = {path.name for path in OUTPUT.iterdir() if path.is_file()}
    check("required_outputs_exist", required.issubset(present), sorted(required - present), [], "All declared evidence tables exist.")

    prereg = json.loads((OUTPUT / "preregistration.json").read_text())
    expected_origins = [["origin_2020", 2018, 2019, 2020], ["origin_2021", 2019, 2020, 2021], ["origin_2022", 2020, 2021, 2022], ["origin_2023", 2021, 2022, 2023]]
    check("origins_preregistered", prereg["origins"] == expected_origins, prereg["origins"], expected_origins, "Only the established four development origins are used.")
    check("holdout_policy_preregistered", prereg["final_holdout_policy"] == "2024 and later are not evaluated", prereg["final_holdout_policy"], "2024 and later are not evaluated", "Future holdout remains sealed.")
    check("missing_context_not_imputed", prereg["missing_context_policy"] == "unavailable; never imputed and never scored", prereg["missing_context_policy"], "unavailable; never imputed and never scored", "Promoted-club gaps cannot be silently filled.")

    predictions = pd.read_csv(OUTPUT / "paired_predictions.csv", low_memory=False)
    origins = pd.read_csv(OUTPUT / "origin_metrics.csv")
    coverage = pd.read_csv(OUTPUT / "coverage_by_origin.csv")
    comparison = pd.read_csv(OUTPUT / "candidate_comparison.csv")
    decisions = pd.read_csv(OUTPUT / "selection_decision.csv")
    variants = list(prereg["variant_blocks"])
    check("variant_family_exact", set(predictions.variant) == set(variants), sorted(predictions.variant.unique()), sorted(variants), "No post-hoc feature block was added or dropped.")
    check("prediction_key_unique", not predictions.duplicated(["variant", "origin_id", "capology_extension_event_id"]).any(), int(predictions.duplicated(["variant", "origin_id", "capology_extension_event_id"]).sum()), 0, "One paired prediction per candidate/event/origin.")
    check("development_years_only", sorted(predictions.evaluation_year.unique().tolist()) == [2020, 2021, 2022, 2023], sorted(predictions.evaluation_year.unique().tolist()), [2020, 2021, 2022, 2023], "No 2024+ outcome was evaluated.")

    actual = predictions.actual.to_numpy(float)
    candidate_loss = (actual - predictions.candidate_prediction.to_numpy(float)) ** 2
    baseline_loss = (actual - predictions.matched_baseline_prediction.to_numpy(float)) ** 2
    check("candidate_loss_recomputed", np.allclose(candidate_loss, predictions.candidate_brier_loss, atol=1e-14), float(np.max(np.abs(candidate_loss - predictions.candidate_brier_loss))), "<=1e-14", "Row losses reconstruct from raw predictions.")
    check("baseline_loss_recomputed", np.allclose(baseline_loss, predictions.matched_baseline_brier_loss, atol=1e-14), float(np.max(np.abs(baseline_loss - predictions.matched_baseline_brier_loss))), "<=1e-14", "Matched baseline losses reconstruct.")
    check("paired_improvement_recomputed", np.allclose(baseline_loss - candidate_loss, predictions.loss_improvement, atol=1e-14), float(np.max(np.abs((baseline_loss - candidate_loss) - predictions.loss_improvement))), "<=1e-14", "Paired improvement direction is baseline minus candidate.")

    pooled = predictions.groupby("variant").agg(
        candidate_brier=("candidate_brier_loss", "mean"),
        matched_baseline_brier=("matched_baseline_brier_loss", "mean"),
        matched_evaluation_rows=("actual", "size"),
    ).reset_index()
    pair = comparison.merge(pooled, on="variant", suffixes=("_reported", "_recomputed"), validate="one_to_one")
    check("pooled_brier_recomputed", np.allclose(pair.candidate_brier_reported, pair.candidate_brier_recomputed, atol=1e-14), float(np.max(np.abs(pair.candidate_brier_reported - pair.candidate_brier_recomputed))), "<=1e-14", "Candidate Brier is independently pooled.")
    check("pooled_baseline_recomputed", np.allclose(pair.matched_baseline_brier_reported, pair.matched_baseline_brier_recomputed, atol=1e-14), float(np.max(np.abs(pair.matched_baseline_brier_reported - pair.matched_baseline_brier_recomputed))), "<=1e-14", "Matched baseline Brier is independently pooled.")
    check("no_candidate_positive_adjusted_ci", comparison.ci_low_bonferroni_99.le(0).all(), comparison.loc[comparison.ci_low_bonferroni_99.gt(0), "variant"].tolist(), [], "No context block has multiplicity-adjusted positive evidence.")
    check("no_candidate_advanced", not decisions.advance_gate_passed.astype(bool).any(), decisions.loc[decisions.advance_gate_passed.astype(bool), "variant"].tolist(), [], "The mechanical gate rejected every block.")
    check("nothing_deployed", decisions.deployment_status.eq("not_deployed").all(), decisions.deployment_status.unique().tolist(), ["not_deployed"], "Experiment did not mutate production status.")

    cov = coverage.groupby("variant").agg(total=("evaluation_rows_total", "sum"), eligible=("feature_eligible_rows", "sum"), refused=("operational_refused_rows", "sum")).reset_index()
    cov["coverage"] = cov.eligible / cov.total
    cov["refusal"] = cov.refused / cov.total
    cov_pair = comparison.merge(cov, on="variant", validate="one_to_one")
    check("coverage_recomputed", np.allclose(cov_pair.feature_coverage_rate, cov_pair.coverage, atol=1e-14), float(np.max(np.abs(cov_pair.feature_coverage_rate - cov_pair.coverage))), "<=1e-14", "Coverage denominator includes all evaluation events.")
    check("refusal_recomputed", np.allclose(cov_pair.operational_refusal_rate, cov_pair.refusal, atol=1e-14), float(np.max(np.abs(cov_pair.operational_refusal_rate - cov_pair.refusal))), "<=1e-14", "Unavailable and OOD-refused rows are both counted.")
    check("context_rows_never_exceed_total", bool((coverage.feature_eligible_rows <= coverage.evaluation_rows_total).all()), int((coverage.feature_eligible_rows > coverage.evaluation_rows_total).sum()), 0, "Coverage arithmetic is coherent.")

    sensitivity = pd.read_csv(OUTPUT / "coverage_selection_sensitivity.csv")
    ligue = sensitivity.loc[sensitivity.league.eq("Ligue 1")].set_index("scope")
    check("ligue1_full_failure_reproduced", np.isclose(ligue.loc["full_locked_phase4", "locked_phase4_brier_skill"], -0.006583, atol=1e-6), ligue.loc["full_locked_phase4", "locked_phase4_brier_skill"], -0.006583, "The original release-blocking league failure is preserved.")
    check("coverage_filter_alone_does_not_repair_ligue1", ligue.loc["prior_context_available", "locked_phase4_brier_skill"] < 0, ligue.loc["prior_context_available", "locked_phase4_brier_skill"], "<0", "Filtering to context-covered Ligue 1 cases improves the number but does not create positive skill.")
    check("missing_ligue1_segment_is_materially_weak", ligue.loc["prior_context_missing", "locked_phase4_brier_skill"] < -0.05, ligue.loc["prior_context_missing", "locked_phase4_brier_skill"], "<-0.05", "The omitted promoted/missing-context segment is especially weak and cannot be hidden.")

    build = pd.read_csv(OUTPUT / "build_checks.csv")
    check("runner_checks_pass", build.passed.astype(bool).all(), int(build.passed.astype(bool).sum()), len(build), "Runner internal invariants all pass.")
    summary = json.loads((OUTPUT / "run_summary.json").read_text())
    check("baseline_reconstruction_exact", summary["locked_baseline_reconstruction"]["exact_within_1e_12"], summary["locked_baseline_reconstruction"]["max_absolute_prediction_difference"], "<=1e-12", "The Phase-4 reference is genuinely reconstructed.")
    check("summary_holdout_false", summary["final_holdout_opened"] is False, summary["final_holdout_opened"], False, "Summary does not imply final testing.")
    check("summary_deployment_false", summary["deployment_changed"] is False, summary["deployment_changed"], False, "Summary does not imply deployment.")

    source = pd.read_csv(OUTPUT / "source_manifest.csv")
    source_ok = all(sha256(ROOT / row.source) == row.sha256 for row in source.itertuples())
    check("source_hashes_current", source_ok, source_ok, True, "Inputs still match the experiment manifest.")
    freeze = json.loads(FREEZE.read_text())
    frozen_ok = all(sha256(ROOT / item["path"]) == item["sha256"] for item in freeze["files"])
    check("frozen_v1_hashes_unchanged", frozen_ok, frozen_ok, True, "All 24 frozen production artifacts remain byte-identical.")
    check("no_model_artifacts_in_output", not any(path.suffix in {".joblib", ".pkl", ".pickle"} for path in OUTPUT.iterdir()), [path.name for path in OUTPUT.iterdir() if path.suffix in {".joblib", ".pkl", ".pickle"}], [], "Only evidence, never candidate binaries, is emitted.")

    deterministic = [
        "README.md", "preregistration.json", "source_manifest.csv", "build_checks.csv",
        "paired_predictions.csv", "origin_metrics.csv", "coverage_by_origin.csv",
        "candidate_comparison.csv", "subgroup_metrics.csv", "decision_capacity_lift.csv",
        "selection_decision.csv", "support_profiles.json", "run_summary.json",
    ]
    with tempfile.TemporaryDirectory(prefix="movemaker_club_context_verify_") as temp:
        rerun = Path(temp) / "rerun"
        subprocess.run([sys.executable, str(RUNNER), "--output", str(rerun)], cwd=ROOT, check=True, capture_output=True, text=True)
        mismatches = [name for name in deterministic if sha256(OUTPUT / name) != sha256(rerun / name)]
    check("byte_reproducible_rerun", not mismatches, mismatches, [], "Independent clean-directory rerun reproduces every substantive artifact byte-for-byte.")

    result = pd.DataFrame(checks)
    result.to_csv(OUTPUT / "independent_verification.csv", index=False)
    payload = {"all_checks_passed": bool(result.passed.all()), "checks_passed": int(result.passed.sum()), "checks_total": len(result)}
    (OUTPUT / "independent_verification.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    manifest_rows = []
    for path in sorted(OUTPUT.iterdir()):
        if path.is_file() and path.name != "output_manifest.csv":
            manifest_rows.append({"output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest_rows).to_csv(OUTPUT / "output_manifest.csv", index=False)
    print(json.dumps(payload, indent=2))
    if not payload["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

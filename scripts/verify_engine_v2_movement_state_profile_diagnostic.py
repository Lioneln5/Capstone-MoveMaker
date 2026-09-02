"""Independent verifier for movement-state and profile diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Data" / "processed" / "engine_v2_movement_state_profile_diagnostic"
RUNNER = ROOT / "models" / "run_engine_v2_movement_state_profile_diagnostic.py"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"
STATES = ["no_outbound_24m", "temporary_first_24m", "permanent_first_24m"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    rows = []

    def check(name, passed, observed, expected, note):
        rows.append({"check": name, "passed": bool(passed), "observed": observed, "expected": expected, "note": note})

    required = {
        "README.md", "state_definition_audit.csv", "movement_state_counts.csv",
        "movement_contribution_interaction.csv", "binary_endpoint_predictions.csv",
        "binary_endpoint_origin_metrics.csv", "binary_endpoint_pooled_metrics.csv",
        "binary_endpoint_subgroup_metrics.csv", "movement_state_predictions.csv",
        "movement_state_origin_metrics.csv", "movement_state_pooled_metrics.csv",
        "movement_state_class_metrics.csv", "club_context_by_decomposed_endpoint.csv",
        "profile_opportunity_register.csv", "build_checks.csv", "source_manifest.csv",
        "run_summary.json", "output_manifest.csv",
    }
    present = {path.name for path in OUTPUT.iterdir() if path.is_file()}
    check("required_outputs", required.issubset(present), sorted(required - present), [], "All substantive evidence exists.")

    counts = pd.read_csv(OUTPUT / "movement_state_counts.csv")
    pooled_counts = counts.loc[counts.scope.eq("pooled")].set_index("movement_state_raw").rows.to_dict()
    expected_counts = {"no_outbound_24m": 772, "permanent_first_24m": 451, "loan_first_24m": 295, "loan_return_first_24m": 35, "unresolved_first_type_24m": 7}
    check("raw_state_counts", pooled_counts == expected_counts, pooled_counts, expected_counts, "Raw movement semantics are preserved before grouping.")
    interaction = pd.read_csv(OUTPUT / "movement_contribution_interaction.csv")
    strict_count = int(interaction.loc[interaction.movement_state_raw.eq("no_outbound_24m"), "sustained"].iloc[0])
    check("strict_meaningful_stay_count", strict_count == 482, strict_count, 482, "Joint no-outbound plus sustained-use outcome reconstructs.")

    binary = pd.read_csv(OUTPUT / "binary_endpoint_predictions.csv")
    expected_loss = np.square(binary.actual - binary.prediction)
    expected_baseline = np.square(binary.actual - binary.chronology_prediction)
    check("binary_loss_formula", np.allclose(expected_loss, binary.brier_loss, atol=1e-14), float(np.max(np.abs(expected_loss - binary.brier_loss))), "<=1e-14", "Binary row losses reconstruct.")
    check("binary_baseline_formula", np.allclose(expected_baseline, binary.chronology_brier_loss, atol=1e-14), float(np.max(np.abs(expected_baseline - binary.chronology_brier_loss))), "<=1e-14", "Chronology losses reconstruct.")
    check("binary_development_years", sorted(binary.evaluation_year.unique().tolist()) == [2020, 2021, 2022, 2023], sorted(binary.evaluation_year.unique().tolist()), [2020, 2021, 2022, 2023], "No later outcome enters binary evaluation.")
    pooled = pd.read_csv(OUTPUT / "binary_endpoint_pooled_metrics.csv")
    recomputed = binary.groupby("endpoint").agg(pooled_brier=("brier_loss", "mean"), chronology_brier=("chronology_brier_loss", "mean"), evaluation_rows=("actual", "size")).reset_index()
    pair = pooled.merge(recomputed, on="endpoint", suffixes=("_reported", "_recomputed"), validate="one_to_one")
    check("binary_pooled_recomputed", np.allclose(pair.pooled_brier_reported, pair.pooled_brier_recomputed, atol=1e-14), float(np.max(np.abs(pair.pooled_brier_reported - pair.pooled_brier_recomputed))), "<=1e-14", "Pooled binary metrics do not rely on runner summaries.")
    strong = pooled.set_index("endpoint")
    check("temporary_signal_positive_all_origins", strong.loc["temporary_first_24m", "origin_wins"] == 4 and strong.loc["temporary_first_24m", "player_cluster_ci_low_95"] > 0, strong.loc["temporary_first_24m", ["origin_wins", "player_cluster_ci_low_95"]].to_dict(), "4 wins and CI low >0", "Temporary-first target has repeated-origin signal.")
    check("strict_stay_signal_positive_all_origins", strong.loc["strict_meaningful_stay_24m", "origin_wins"] == 4 and strong.loc["strict_meaningful_stay_24m", "player_cluster_ci_low_95"] > 0, strong.loc["strict_meaningful_stay_24m", ["origin_wins", "player_cluster_ci_low_95"]].to_dict(), "4 wins and CI low >0", "Strict meaningful stay has repeated-origin signal.")

    multi = pd.read_csv(OUTPUT / "movement_state_predictions.csv")
    probability = multi[[f"probability_{state}" for state in STATES]].to_numpy(float)
    chronology = multi[[f"chronology_{state}" for state in STATES]].to_numpy(float)
    check("multinomial_probability_sum", np.allclose(probability.sum(axis=1), 1, atol=1e-12), float(np.max(np.abs(probability.sum(axis=1) - 1))), "<=1e-12", "Three-state probabilities are coherent.")
    check("multinomial_chronology_sum", np.allclose(chronology.sum(axis=1), 1, atol=1e-12), float(np.max(np.abs(chronology.sum(axis=1) - 1))), "<=1e-12", "Baseline probabilities are coherent.")
    onehot = np.column_stack([multi.actual_state.eq(state) for state in STATES]).astype(float)
    brier = np.square(onehot - probability).sum(axis=1)
    chronology_brier = np.square(onehot - chronology).sum(axis=1)
    check("multinomial_brier_formula", np.allclose(brier, multi.multiclass_brier_loss, atol=1e-14), float(np.max(np.abs(brier - multi.multiclass_brier_loss))), "<=1e-14", "Multiclass Brier reconstructs.")
    check("multinomial_baseline_formula", np.allclose(chronology_brier, multi.chronology_brier_loss, atol=1e-14), float(np.max(np.abs(chronology_brier - multi.chronology_brier_loss))), "<=1e-14", "Multiclass baseline reconstructs.")
    indices = np.array([STATES.index(state) for state in multi.actual_state], dtype=int)
    manual_log_loss = float(-np.log(probability[np.arange(len(multi)), indices]).mean())
    manual_chronology_log_loss = float(-np.log(chronology[np.arange(len(multi)), indices]).mean())
    multi_pooled = pd.read_csv(OUTPUT / "movement_state_pooled_metrics.csv").iloc[0]
    check("multinomial_log_loss_label_order", np.isclose(manual_log_loss, multi_pooled.multiclass_log_loss, atol=1e-14), manual_log_loss - multi_pooled.multiclass_log_loss, "<=1e-14", "Actual labels select their named probability columns.")
    check("chronology_log_loss_label_order", np.isclose(manual_chronology_log_loss, multi_pooled.chronology_log_loss, atol=1e-14), manual_chronology_log_loss - multi_pooled.chronology_log_loss, "<=1e-14", "Chronology log loss uses the same class order.")
    check("unresolved_excluded", len(multi) == 1553, len(multi), 1553, "Seven unresolved first types are not silently assigned.")
    check("multinomial_all_origin_wins", multi_pooled.origin_wins == 4 and multi_pooled.player_cluster_ci_low_95 > 0, {"origin_wins": multi_pooled.origin_wins, "ci_low": multi_pooled.player_cluster_ci_low_95}, "4 and >0", "Three-state model beats chronology repeatedly.")

    context = pd.read_csv(OUTPUT / "club_context_by_decomposed_endpoint.csv")
    check("club_context_not_misrepresented", context.brier_improvement.le(0).all(), context.loc[context.brier_improvement.gt(0), "endpoint"].tolist(), [], "Aggregate club context did not repair any decomposed endpoint.")
    profiles = pd.read_csv(OUTPUT / "profile_opportunity_register.csv")
    expected_completed_p0 = {"P01", "P02", "P03", "P04"}
    actual_completed_p0 = set(profiles.loc[profiles.priority.eq("P0_COMPLETE"), "profile_id"])
    check("completed_p0_profile_contract", actual_completed_p0 == expected_completed_p0, sorted(actual_completed_p0), sorted(expected_completed_p0), "Completed mechanism tests remain explicit and cannot be silently recycled.")
    check("remaining_profile_contract", profiles.set_index("profile_id").loc["P05", "priority"] == "P1_PARTIAL" and profiles.set_index("profile_id").loc["P11", "priority"] == "P0", {"P05": profiles.set_index("profile_id").loc["P05", "priority"], "P11": profiles.set_index("profile_id").loc["P11", "priority"]}, {"P05": "P1_PARTIAL", "P11": "P0"}, "Partial and unresolved directions remain distinguishable from completed tests.")
    statsbomb = profiles.loc[profiles.profile_id.eq("P12")].iloc[0]
    check("statsbomb_not_overclaimed", statsbomb.priority == "P3" and "not global" in statsbomb.relevant_outcomes, {"priority": statsbomb.priority, "relevance": statsbomb.relevant_outcomes}, "P3 case study", "Rich but selective coverage is not confused with global modelability.")

    build = pd.read_csv(OUTPUT / "build_checks.csv")
    check("runner_checks", build.passed.astype(bool).all(), int(build.passed.astype(bool).sum()), len(build), "Runner invariants pass.")
    summary = json.loads((OUTPUT / "run_summary.json").read_text())
    check("holdout_sealed", summary["final_holdout_opened"] is False, summary["final_holdout_opened"], False, "Final holdout remains sealed.")
    check("deployment_unchanged", summary["deployment_changed"] is False, summary["deployment_changed"], False, "No candidate was promoted.")
    source = pd.read_csv(OUTPUT / "source_manifest.csv")
    source_ok = all(sha256(ROOT / row.source) == row.sha256 for row in source.itertuples())
    check("source_hashes", source_ok, source_ok, True, "Source evidence still matches.")
    freeze = json.loads(FREEZE.read_text())
    freeze_ok = all(sha256(ROOT / item["path"]) == item["sha256"] for item in freeze["files"])
    check("v1_hashes", freeze_ok, freeze_ok, True, "All frozen production artifacts remain byte-identical.")
    check("no_model_binaries", not any(path.suffix in {".joblib", ".pkl", ".pickle"} for path in OUTPUT.iterdir()), [path.name for path in OUTPUT.iterdir() if path.suffix in {".joblib", ".pkl", ".pickle"}], [], "Diagnostic emits evidence only.")

    deterministic = sorted(required - {"output_manifest.csv"})
    with tempfile.TemporaryDirectory(prefix="movemaker_state_verify_") as temp:
        rerun = Path(temp) / "rerun"
        subprocess.run([sys.executable, str(RUNNER), "--output", str(rerun)], cwd=ROOT, check=True, capture_output=True, text=True)
        mismatches = [name for name in deterministic if sha256(OUTPUT / name) != sha256(rerun / name)]
    check("byte_reproducible", not mismatches, mismatches, [], "A clean-directory rerun reproduces every substantive output.")

    verification = pd.DataFrame(rows)
    verification.to_csv(OUTPUT / "independent_verification.csv", index=False)
    payload = {"all_checks_passed": bool(verification.passed.all()), "checks_passed": int(verification.passed.sum()), "checks_total": len(verification)}
    (OUTPUT / "independent_verification.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    manifest = []
    for path in sorted(OUTPUT.iterdir()):
        if path.is_file() and path.name != "output_manifest.csv":
            manifest.append({"output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest).to_csv(OUTPUT / "output_manifest.csv", index=False)
    print(json.dumps(payload, indent=2))
    if not payload["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

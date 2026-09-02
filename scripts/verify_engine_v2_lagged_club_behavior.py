"""Independent verification for the lagged club-behavior build and experiment."""

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
BUILD = ROOT / "Data" / "processed" / "engine_v2_lagged_club_behavior"
EXPERIMENT = ROOT / "Data" / "processed" / "engine_v2_lagged_club_behavior_experiment"
BUILDER = ROOT / "scripts" / "build_engine_v2_lagged_club_behavior.py"
RUNNER = ROOT / "models" / "run_engine_v2_lagged_club_behavior_experiment.py"
TRANSFER = ROOT / "Data" / "processed" / "canonical_integration" / "canonical_transfer_history.csv"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"
STATES = ["no_outbound_24m", "temporary_first_24m", "permanent_first_24m"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def update_manifest(directory: Path) -> None:
    rows = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name != "output_manifest.csv":
            rows.append({"output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(rows).to_csv(directory / "output_manifest.csv", index=False)


def direct_sample_reconstruction(profiles: pd.DataFrame) -> tuple[int, int, int]:
    sample = profiles.sort_values("capology_extension_event_id").iloc[::97].head(30).copy()
    transfers = pd.read_csv(
        TRANSFER,
        usecols=["canonical_transfer_event_id", "transfer_date", "from_club_id", "event_conflict"],
        low_memory=False,
    )
    transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"], errors="coerce")
    conflict = transfers["event_conflict"].fillna(False).astype(bool)
    transfers = transfers.loc[~conflict].copy()
    mismatches = 0
    for row in sample.itertuples():
        signed = pd.Timestamp(row.signed_date)
        expected = transfers.loc[
            transfers["from_club_id"].eq(row.canonical_club_id)
            & transfers["transfer_date"].ge(signed - pd.Timedelta(days=730))
            & transfers["transfer_date"].lt(signed),
            "canonical_transfer_event_id",
        ].nunique()
        mismatches += int(expected != row.club_outbound_events_730d)
    return len(sample), mismatches, int(sample.club_outbound_events_730d.sum())


def direct_mature_extension_reconstruction(profiles: pd.DataFrame) -> tuple[int, int]:
    sys.path[:0] = [str(ROOT), str(ROOT / "models")]
    import run_engine_v2_feature_specification as phase2  # noqa: PLC0415

    history = phase2.load_endpoint_frames()["any_outbound_24m"].copy()
    history["signed_date"] = pd.to_datetime(history["signed_date"], errors="raise")
    history["outcome_available_date"] = history["signed_date"] + pd.Timedelta(days=730)
    any_outbound = pd.to_numeric(history["any_outbound_event_by_24m"], errors="coerce")
    first_type = history["first_any_outbound_type"].astype("string")
    history["resolved_state"] = any_outbound.eq(0) | (any_outbound.eq(1) & first_type.isin(["transfer", "loan", "loan_return"]))
    sample = profiles.loc[profiles.signed_year.between(2020, 2023)].sort_values("capology_extension_event_id").iloc[::61].head(25)
    mismatches = 0
    for row in sample.itertuples():
        expected = history.loc[
            pd.to_numeric(history["canonical_club_id_x"]).eq(row.canonical_club_id)
            & history["outcome_available_date"].le(pd.Timestamp(row.signed_date))
            & history["resolved_state"]
        ]
        mismatches += int(len(expected) != row.club_mature_prior_extension_count)
    return len(sample), mismatches


def main() -> None:
    rows = []

    def check(name: str, passed: bool, observed: object, expected: object, note: str) -> None:
        rows.append({"check": name, "passed": bool(passed), "observed": observed, "expected": expected, "note": note})

    build_required = {
        "README.md", "lagged_club_behavior_features.csv", "asof_evidence_audit.csv",
        "feature_dictionary.csv", "movement_target_contract.json", "build_checks.csv",
        "source_manifest.csv", "run_summary.json", "output_manifest.csv",
    }
    experiment_required = {
        "README.md", "preregistration.json", "binary_predictions.csv",
        "binary_origin_metrics.csv", "binary_coverage_by_origin.csv",
        "binary_candidate_comparison.csv", "binary_league_metrics.csv",
        "movement_state_predictions.csv", "movement_state_origin_metrics.csv",
        "movement_state_coverage_by_origin.csv", "movement_state_candidate_comparison.csv",
        "movement_state_class_metrics.csv", "movement_state_league_metrics.csv",
        "added_feature_coefficients.csv", "build_checks.csv", "source_manifest.csv",
        "run_summary.json", "output_manifest.csv",
    }
    check("build_outputs_present", build_required.issubset({p.name for p in BUILD.iterdir()}), sorted(build_required - {p.name for p in BUILD.iterdir()}), [], "Feature evidence is complete.")
    check("experiment_outputs_present", experiment_required.issubset({p.name for p in EXPERIMENT.iterdir()}), sorted(experiment_required - {p.name for p in EXPERIMENT.iterdir()}), [], "Experiment evidence is complete.")

    profiles = pd.read_csv(BUILD / "lagged_club_behavior_features.csv", low_memory=False)
    audit = pd.read_csv(BUILD / "asof_evidence_audit.csv", low_memory=False)
    check("feature_event_grain", len(profiles) == 2805 and profiles.capology_extension_event_id.is_unique, {"rows": len(profiles), "duplicates": int(profiles.capology_extension_event_id.duplicated().sum())}, {"rows": 2805, "duplicates": 0}, "Canonical event grain is preserved.")
    latest_transfer = pd.to_datetime(audit.latest_transfer_date_used, errors="coerce")
    signed = pd.to_datetime(audit.signed_date, errors="raise")
    check("all_transfer_evidence_predecision", bool((latest_transfer.dropna() < signed.loc[latest_transfer.notna()]).all()), int((latest_transfer >= signed).fillna(False).sum()), 0, "No same-day or future transfer was used.")
    maturity = pd.to_datetime(audit.latest_mature_extension_outcome_available_date, errors="coerce")
    check("all_extension_outcomes_mature", bool((maturity.dropna() <= signed.loc[maturity.notna()]).all()), int((maturity > signed).fillna(False).sum()), 0, "No immature prior extension outcome was used.")
    sample_rows, transfer_mismatches, sample_total = direct_sample_reconstruction(profiles)
    check("direct_transfer_count_reconstruction", transfer_mismatches == 0, {"sample_rows": sample_rows, "mismatches": transfer_mismatches, "total_outbounds": sample_total}, {"mismatches": 0}, "Fixed sample counts are rebuilt from canonical events, not runner summaries.")
    mature_rows, mature_mismatches = direct_mature_extension_reconstruction(profiles)
    check("direct_mature_extension_reconstruction", mature_mismatches == 0, {"sample_rows": mature_rows, "mismatches": mature_mismatches}, {"mismatches": 0}, "Fixed sample maturity counts are independently rebuilt.")

    contract = json.loads((BUILD / "movement_target_contract.json").read_text(encoding="utf-8"))
    check("target_contract_frozen", contract["contract_version"] == "engine_v2_movement_targets_v1_2026-09-01" and contract["modeled_state_order"] == STATES, {"version": contract["contract_version"], "states": contract["modeled_state_order"]}, {"version": "engine_v2_movement_targets_v1_2026-09-01", "states": STATES}, "Movement semantics are explicit and serialized.")
    check("unresolved_policy_explicit", any(item["target"] == "movement_state_3" and "Exclude" in item["unresolved_policy"] for item in contract["definitions"]), True, True, "Unresolved first movements cannot be silently assigned.")

    binary = pd.read_csv(EXPERIMENT / "binary_predictions.csv", low_memory=False)
    expected_delta = np.square(binary.actual - binary.matched_baseline_prediction) - np.square(binary.actual - binary.candidate_prediction)
    check("binary_loss_reconstruction", np.allclose(expected_delta, binary.loss_improvement, atol=1e-14), float(np.max(np.abs(expected_delta - binary.loss_improvement))), "<=1e-14", "Paired Brier improvement reconstructs row by row.")
    key = ["endpoint", "variant", "origin_id", "capology_extension_event_id"]
    check("binary_prediction_key", not binary.duplicated(key).any(), int(binary.duplicated(key).sum()), 0, "One paired prediction per endpoint, block, origin, and event.")
    check("development_years_only", sorted(binary.evaluation_year.unique().tolist()) == [2020, 2021, 2022, 2023], sorted(binary.evaluation_year.unique().tolist()), [2020, 2021, 2022, 2023], "2024+ target outcomes remain sealed.")

    binary_comparison = pd.read_csv(EXPERIMENT / "binary_candidate_comparison.csv")
    counts = binary_comparison.evidence_classification.value_counts().to_dict()
    check("binary_evidence_classification", counts == {"inconclusive": 13, "repeated_harm": 3}, counts, {"inconclusive": 13, "repeated_harm": 3}, "No block is falsely promoted from small unadjusted changes.")
    mature = binary_comparison.loc[binary_comparison.variant.eq("C3_mature_extension_history")]
    check("mature_history_refusal_visible", mature.operational_refusal_rate.min() > 0.25, float(mature.operational_refusal_rate.min()), ">0.25", "Accumulating history creates a material modelability problem.")

    multi = pd.read_csv(EXPERIMENT / "movement_state_predictions.csv", low_memory=False)
    candidate = multi[[f"candidate_probability_{state}" for state in STATES]].to_numpy(float)
    baseline = multi[[f"matched_baseline_probability_{state}" for state in STATES]].to_numpy(float)
    onehot = np.column_stack([multi.actual_state.eq(state) for state in STATES]).astype(float)
    delta = np.square(onehot - baseline).sum(axis=1) - np.square(onehot - candidate).sum(axis=1)
    check("multinomial_probability_sum", np.allclose(candidate.sum(axis=1), 1, atol=1e-12), float(np.max(np.abs(candidate.sum(axis=1) - 1))), "<=1e-12", "Named state probabilities are coherent.")
    check("multinomial_loss_reconstruction", np.allclose(delta, multi.loss_improvement, atol=1e-14), float(np.max(np.abs(delta - multi.loss_improvement))), "<=1e-14", "Multiclass paired loss reconstructs.")
    multi_comparison = pd.read_csv(EXPERIMENT / "movement_state_candidate_comparison.csv")
    multi_counts = multi_comparison.evidence_classification.value_counts().to_dict()
    check("multinomial_evidence_classification", multi_counts == {"inconclusive": 2, "repeated_harm": 2}, multi_counts, {"inconclusive": 2, "repeated_harm": 2}, "Role-conditioned movement is retained as inconclusive, not overclaimed.")
    role = multi_comparison.loc[multi_comparison.variant.eq("C2_role_conditioned_movement")].iloc[0]
    check("role_signal_not_overclaimed", role.pooled_brier_improvement > 0 and role.ci_low_adjusted_98_75 <= 0, {"improvement": role.pooled_brier_improvement, "adjusted_ci_low": role.ci_low_adjusted_98_75}, "positive point estimate but adjusted CI crosses zero", "Small coherent signal remains development evidence only.")

    build_checks = pd.read_csv(BUILD / "build_checks.csv")
    experiment_checks = pd.read_csv(EXPERIMENT / "build_checks.csv")
    check("runner_checks", build_checks.passed.astype(bool).all() and experiment_checks.passed.astype(bool).all(), {"build": int(build_checks.passed.astype(bool).sum()), "experiment": int(experiment_checks.passed.astype(bool).sum())}, {"build": len(build_checks), "experiment": len(experiment_checks)}, "All internal invariants pass.")
    source = pd.concat([pd.read_csv(BUILD / "source_manifest.csv"), pd.read_csv(EXPERIMENT / "source_manifest.csv")]).drop_duplicates("source")
    source_ok = all(sha256(ROOT / row.source) == row.sha256 for row in source.itertuples())
    check("source_hashes", source_ok, source_ok, True, "All source evidence is unchanged.")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    freeze_ok = all(sha256(ROOT / item["path"]) == item["sha256"] for item in freeze["files"])
    check("v1_hashes", freeze_ok, freeze_ok, True, "All 24 frozen V1 files remain byte-identical.")
    binaries = [path.name for directory in [BUILD, EXPERIMENT] for path in directory.iterdir() if path.suffix in {".joblib", ".pkl", ".pickle"}]
    check("no_model_binaries", not binaries, binaries, [], "Research outputs contain evidence only.")

    build_deterministic = sorted(build_required - {"output_manifest.csv"})
    experiment_deterministic = sorted(experiment_required - {"output_manifest.csv"})
    with tempfile.TemporaryDirectory(prefix="movemaker_club_behavior_verify_") as temp:
        temp_root = Path(temp)
        rebuilt = temp_root / "build"
        rerun = temp_root / "experiment"
        subprocess.run([sys.executable, str(BUILDER), "--output", str(rebuilt)], cwd=ROOT, check=True, capture_output=True, text=True)
        build_mismatches = [name for name in build_deterministic if sha256(BUILD / name) != sha256(rebuilt / name)]
        subprocess.run([sys.executable, str(RUNNER), "--output", str(rerun)], cwd=ROOT, check=True, capture_output=True, text=True)
        experiment_mismatches = [name for name in experiment_deterministic if sha256(EXPERIMENT / name) != sha256(rerun / name)]
    check("build_byte_reproducible", not build_mismatches, build_mismatches, [], "Clean-directory feature build is byte reproducible.")
    check("experiment_byte_reproducible", not experiment_mismatches, experiment_mismatches, [], "Clean-directory experiment is byte reproducible.")

    verification = pd.DataFrame(rows)
    verification.to_csv(EXPERIMENT / "independent_verification.csv", index=False)
    payload = {
        "all_checks_passed": bool(verification.passed.all()),
        "checks_passed": int(verification.passed.sum()),
        "checks_total": len(verification),
    }
    (EXPERIMENT / "independent_verification.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    update_manifest(EXPERIMENT)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_checks_passed"]:
        print(verification.loc[~verification.passed].to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

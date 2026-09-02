"""Test decision-time manager and tactical feature blocks.

Development-only rolling-origin experiment. Every candidate is compared with
the compact Engine V2 baseline retrained on identical rows. The 2024+ final
outcome holdout remains sealed; no deployable artifact is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "models")]

from models.engine_v2.manager_tactical_feature_contract import (  # noqa: E402
    CONTRACT_VERSION as MANAGER_TACTICAL_CONTRACT_VERSION,
    FEATURE_BLOCKS,
)
from models.engine_v2.movement_target_contract import (  # noqa: E402
    CONTRACT_VERSION as MOVEMENT_CONTRACT_VERSION,
    add_movement_targets,
)
import run_engine_v2_candidate_contract as phase4  # noqa: E402
import run_engine_v2_feature_specification as phase2  # noqa: E402
import run_engine_v2_lagged_club_behavior_experiment as engine  # noqa: E402


DEFAULT_OUTPUT = ROOT / "Data" / "processed" / "engine_v2_manager_tactical_experiment"
PROFILE_DIR = ROOT / "Data" / "processed" / "engine_v2_manager_tactical_profiles"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"
ADJUSTED_CONFIDENCE = 1 - 0.05 / len(FEATURE_BLOCKS)


def load_data(profile_dir: Path = PROFILE_DIR) -> pd.DataFrame:
    data = phase4.add_age_curve(phase2.load_endpoint_frames()["any_outbound_24m"])
    data = add_movement_targets(data)
    feature_names = sorted({name for block in FEATURE_BLOCKS.values() for name in block})
    profiles = pd.read_csv(
        profile_dir / "manager_tactical_profiles.csv",
        usecols=["capology_extension_event_id"] + feature_names,
        low_memory=False,
    )
    return data.merge(profiles, on="capology_extension_event_id", how="left", validate="one_to_one")


def relabel_adjusted_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            "ci_low_adjusted_98_75": "ci_low_adjusted_99",
            "ci_high_adjusted_98_75": "ci_high_adjusted_99",
        }
    )


def build_checks(
    data: pd.DataFrame,
    binary_predictions: pd.DataFrame,
    multi_predictions: pd.DataFrame,
    binary_comparison: pd.DataFrame,
    multi_comparison: pd.DataFrame,
    before: dict[str, str],
    after: dict[str, str],
) -> pd.DataFrame:
    rows = []

    def add(check: str, passed: bool, observed: object, expected: object, note: str) -> None:
        rows.append({"check": check, "passed": bool(passed), "observed": observed, "expected": expected, "note": note})

    add("four_binary_endpoints", binary_predictions["endpoint"].nunique() == 4, binary_predictions["endpoint"].nunique(), 4, "All decomposed movement questions were tested.")
    add("five_predeclared_variants", binary_predictions["variant"].nunique() == len(FEATURE_BLOCKS), binary_predictions["variant"].nunique(), len(FEATURE_BLOCKS), "No post-outcome block was added.")
    add("four_rolling_origins", binary_predictions["origin_id"].nunique() == 4, binary_predictions["origin_id"].nunique(), 4, "Only established development origins were used.")
    maximum_year = max(binary_predictions["evaluation_year"].max(), multi_predictions["evaluation_year"].max())
    add("holdout_sealed", maximum_year == 2023, maximum_year, 2023, "No 2024+ target was evaluated.")
    binary_formula = np.square(binary_predictions["actual"] - binary_predictions["matched_baseline_prediction"]) - np.square(binary_predictions["actual"] - binary_predictions["candidate_prediction"])
    add("binary_loss_formula", np.allclose(binary_formula, binary_predictions["loss_improvement"], atol=1e-14), float(np.max(np.abs(binary_formula - binary_predictions["loss_improvement"]))), "<=1e-14", "Paired Brier deltas reconstruct.")
    state_columns = [f"candidate_probability_{state}" for state in engine.STATE_ORDER]
    probability_sum = multi_predictions[state_columns].sum(axis=1)
    add("multinomial_probability_sum", np.allclose(probability_sum, 1, atol=1e-12), float(np.max(np.abs(probability_sum - 1))), "<=1e-12", "Named movement-state probabilities are coherent.")
    add("matched_row_comparisons", binary_comparison["evaluation_rows_matched"].gt(0).all() and multi_comparison["evaluation_rows_matched"].gt(0).all(), True, True, "Every candidate baseline uses identical rows.")
    development = data.loc[data["signed_year"].between(2020, 2023)]
    feature_names = sorted({name for block in FEATURE_BLOCKS.values() for name in block})
    add("features_present", development[feature_names].notna().any().all(), int(development[feature_names].notna().any().sum()), len(feature_names), "Every frozen feature has development observations.")
    add("research_only", binary_comparison["deployment_status"].eq("not_deployed").all() and multi_comparison["deployment_status"].eq("not_deployed").all(), True, True, "This diagnostic cannot authorize deployment.")
    add("v1_frozen", before == after, sum(before[key] == after[key] for key in before), len(before), "All frozen V1 files remain byte-identical.")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile-dir", type=Path, default=PROFILE_DIR)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")

    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    before = {item["path"]: engine.sha256(ROOT / item["path"]) for item in freeze["files"]}
    engine.VARIANT_BLOCKS = {name: list(features) for name, features in FEATURE_BLOCKS.items()}
    engine.FAMILY_CONFIDENCE = ADJUSTED_CONFIDENCE
    data = load_data(args.profile_dir)
    binary_predictions, binary_origins, binary_coverage, binary_coefficients = engine.run_binary(data)
    multi_predictions, multi_origins, multi_coverage, multi_coefficients = engine.run_multinomial(data)
    binary_comparison, binary_leagues = engine.summarize_binary(binary_predictions, binary_origins, binary_coverage)
    multi_comparison, multi_classes, multi_leagues = engine.summarize_multinomial(multi_predictions, multi_origins, multi_coverage)
    binary_comparison = relabel_adjusted_intervals(binary_comparison)
    multi_comparison = relabel_adjusted_intervals(multi_comparison)
    after = {path: engine.sha256(ROOT / path) for path in before}
    checks = build_checks(data, binary_predictions, multi_predictions, binary_comparison, multi_comparison, before, after)

    outputs = {
        "binary_predictions.csv": binary_predictions,
        "binary_origin_metrics.csv": binary_origins,
        "binary_coverage_by_origin.csv": binary_coverage,
        "binary_candidate_comparison.csv": binary_comparison,
        "binary_league_metrics.csv": binary_leagues,
        "movement_state_predictions.csv": multi_predictions,
        "movement_state_origin_metrics.csv": multi_origins,
        "movement_state_coverage_by_origin.csv": multi_coverage,
        "movement_state_candidate_comparison.csv": multi_comparison,
        "movement_state_class_metrics.csv": multi_classes,
        "movement_state_league_metrics.csv": multi_leagues,
        "added_feature_coefficients.csv": pd.concat([binary_coefficients, multi_coefficients], ignore_index=True),
        "build_checks.csv": checks,
    }
    for name, frame in outputs.items():
        frame.to_csv(output / name, index=False)

    preregistration = {
        "manager_tactical_contract_version": MANAGER_TACTICAL_CONTRACT_VERSION,
        "movement_contract_version": MOVEMENT_CONTRACT_VERSION,
        "development_origins": [list(origin) for origin in phase2.ORIGINS],
        "baseline_features": engine.BASE_FEATURES,
        "variant_blocks": FEATURE_BLOCKS,
        "binary_endpoints": engine.BINARY_TARGETS,
        "multinomial_state_order": engine.STATE_ORDER,
        "primary_loss": "paired Brier loss versus baseline retrained on identical feature-eligible rows",
        "familywise_interval": "99% player-cluster bootstrap interval across five variants within each endpoint family",
        "support_policy": "feature-unavailable and unsupported rows count as operational refusals",
        "future_holdout_policy": "2024+ target outcomes remain sealed",
        "promotion_policy": "No candidate can be deployed from this diagnostic; a promising block requires later frozen combined and final temporal tests.",
    }
    engine.write_json(output / "preregistration.json", preregistration)
    sources = [
        args.profile_dir / "manager_tactical_profiles.csv",
        args.profile_dir / "feature_contract.json",
        ROOT / "models" / "engine_v2" / "manager_tactical_feature_contract.py",
        ROOT / "models" / "engine_v2" / "movement_target_contract.py",
        ROOT / "models" / "run_engine_v2_manager_tactical_experiment.py",
        ROOT / "models" / "run_engine_v2_lagged_club_behavior_experiment.py",
        ROOT / "models" / "run_engine_v2_candidate_contract.py",
        ROOT / "models" / "run_engine_v2_feature_specification.py",
        FREEZE,
    ]
    pd.DataFrame(
        [{"source": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": engine.sha256(path)} for path in sources]
    ).to_csv(output / "source_manifest.csv", index=False)
    summary = {
        "manager_tactical_contract_version": MANAGER_TACTICAL_CONTRACT_VERSION,
        "movement_contract_version": MOVEMENT_CONTRACT_VERSION,
        "development_evaluation_years": [2020, 2021, 2022, 2023],
        "binary_endpoints": list(engine.BINARY_TARGETS),
        "variants": list(FEATURE_BLOCKS),
        "binary_evidence_counts": binary_comparison["evidence_classification"].value_counts().to_dict(),
        "multinomial_evidence_counts": multi_comparison["evidence_classification"].value_counts().to_dict(),
        "final_holdout_opened": False,
        "deployment_changed": False,
        "model_binaries_written": False,
        "v1_unchanged": before == after,
        "checks_passed": int(checks["passed"].sum()),
        "checks_total": len(checks),
    }
    engine.write_json(output / "run_summary.json", summary)
    (output / "README.md").write_text(
        """# Engine V2 manager and tactical rolling-origin experiment

This development diagnostic tests whether observed manager stability, reported
formation-shape stability, player role under the current manager, and player
role change around the latest manager transition add signal beyond the compact
Phase-4 baseline. Each block and one predeclared combination are evaluated on
matched rows.

The 2024+ final holdout was not opened; no model was serialized or deployed.
""",
        encoding="utf-8",
    )
    manifest = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "output_manifest.csv":
            manifest.append({"output_file": path.name, "bytes": path.stat().st_size, "sha256": engine.sha256(path)})
    pd.DataFrame(manifest).to_csv(output / "output_manifest.csv", index=False)
    if not checks["passed"].all():
        raise RuntimeError(checks.loc[~checks["passed"], ["check", "observed", "expected"]].to_dict("records"))
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nBinary comparisons\n", binary_comparison.to_string(index=False))
    print("\nMultinomial comparisons\n", multi_comparison.to_string(index=False))


if __name__ == "__main__":
    main()

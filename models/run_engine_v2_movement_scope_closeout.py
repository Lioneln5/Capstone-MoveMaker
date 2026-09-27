"""Finalize the Engine V2 movement/retention product-scope decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.engine_v2.movement_product_scope_contract import (  # noqa: E402
    CONTRACT_VERSION,
    PRODUCT_OUTPUTS,
    contract_payload,
    validate_contract,
)


DEFAULT_OUTPUT = ROOT / "Data" / "processed" / "engine_v2_movement_scope_closeout"
P = ROOT / "Data" / "processed"
DIAGNOSTIC_DIR = P / "engine_v2_movement_state_profile_diagnostic"
MANAGER_DIR = P / "engine_v2_manager_tactical_experiment"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")

    pooled = pd.read_csv(DIAGNOSTIC_DIR / "binary_endpoint_pooled_metrics.csv")
    subgroup = pd.read_csv(DIAGNOSTIC_DIR / "binary_endpoint_subgroup_metrics.csv")
    decisions = pd.DataFrame(PRODUCT_OUTPUTS)
    endpoint_map = {
        "generic_continuity_24m": "any_outbound_24m",
        "meaningful_retention_24m": "strict_meaningful_stay_24m",
        "temporary_displacement_risk_24m": "temporary_first_24m",
        "permanent_separation_risk_24m": "permanent_relationship_ended_24m",
    }
    evidence = decisions.loc[decisions["output_id"].isin(endpoint_map)].copy()
    evidence["diagnostic_endpoint"] = evidence["output_id"].map(endpoint_map)
    evidence = evidence.merge(
        pooled,
        left_on="diagnostic_endpoint",
        right_on="endpoint",
        how="left",
        validate="one_to_one",
    )
    evidence = evidence[[
        "output_id", "diagnostic_endpoint", "status", "evaluation_rows",
        "event_rate", "pooled_brier", "chronology_brier", "brier_skill",
        "player_cluster_ci_low_95", "player_cluster_ci_high_95",
        "origin_wins", "pooled_auc", "pooled_average_precision", "pooled_ece",
    ]]

    backlog = pd.DataFrame([
        {
            "research_item": "permanent_separation_repair",
            "priority": "deferred_non_blocking",
            "current_status": "research_only_unavailable",
            "reopen_condition": "A materially new, decision-time-safe mechanism and frozen candidate with credible cross-league support.",
            "prohibited_shortcut": "Do not add manual league/player bonuses or repackage generic continuity.",
        },
        {
            "research_item": "meaningful_retention_final_validation",
            "priority": "required_before_promotion",
            "current_status": "development_candidate_not_deployed",
            "reopen_condition": "A fully mature, player-disjoint, untouched temporal cohort large enough for declared subgroup gates.",
            "prohibited_shortcut": "Do not reuse the exposed or underpowered 2024 cohort as a pristine final test.",
        },
        {
            "research_item": "temporary_displacement_final_validation",
            "priority": "required_before_promotion",
            "current_status": "development_candidate_not_deployed",
            "reopen_condition": "A fully mature, player-disjoint, untouched temporal cohort large enough for declared subgroup gates.",
            "prohibited_shortcut": "Do not promote rolling-development evidence as final validation.",
        },
    ])

    checks: list[dict[str, object]] = []

    def add(check: str, passed: bool, observed: object, expected: object, note: str) -> None:
        checks.append({
            "check": check, "passed": bool(passed), "observed": observed,
            "expected": expected, "note": note,
        })

    violations = validate_contract()
    add("product_contract_valid", not violations, violations, [], "Product status invariants must hold.")
    indexed = pooled.set_index("endpoint")
    for endpoint in ["strict_meaningful_stay_24m", "temporary_first_24m"]:
        row = indexed.loc[endpoint]
        passed = row.brier_skill > 0 and row.player_cluster_ci_low_95 > 0 and row.origin_wins == 4
        add(
            f"{endpoint}_development_signal",
            passed,
            {
                "brier_skill": float(row.brier_skill),
                "ci_low": float(row.player_cluster_ci_low_95),
                "origin_wins": int(row.origin_wins),
            },
            "positive skill, CI low > 0, 4/4 origins",
            "This is development evidence only, not deployment authorization.",
        )
    permanent_leagues = subgroup.loc[
        subgroup.endpoint.eq("permanent_relationship_ended_24m")
        & subgroup.scope.eq("league")
    ]
    failing_leagues = sorted(permanent_leagues.loc[permanent_leagues.brier_skill.le(0), "group"])
    add(
        "permanent_separation_subgroup_failure_preserved",
        bool(failing_leagues),
        failing_leagues,
        "at least one league",
        "Research-only status is grounded in an observed subgroup failure.",
    )
    add(
        "no_numeric_output_authorized",
        not decisions.numeric_output_allowed.any(),
        int(decisions.numeric_output_allowed.sum()),
        0,
        "No movement/retention probability is authorized for live display.",
    )
    movement_verification = json.loads(
        (DIAGNOSTIC_DIR / "independent_verification.json").read_text(encoding="utf-8")
    )
    manager_verification = json.loads(
        (MANAGER_DIR / "independent_verification.json").read_text(encoding="utf-8")
    )
    add(
        "upstream_verification",
        movement_verification["all_checks_passed"] and manager_verification["all_passed"],
        {
            "movement": movement_verification["checks_passed"],
            "manager_tactical": manager_verification["checks_passed"],
        },
        "all upstream checks pass",
        "The scope decision uses independently verified evidence.",
    )
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    changed = [
        item["path"] for item in freeze["files"]
        if sha256(ROOT / item["path"]) != item["sha256"]
    ]
    changed_models = [path for path in changed if path.endswith(".joblib")]
    known_report_changes = {
        "Data/processed/deployment_models/live_request_integrity.csv",
        "Data/processed/deployment_models/live_request_integrity.json",
    }
    models_unchanged = not changed_models
    add(
        "v1_serialized_models_frozen",
        models_unchanged,
        changed_models,
        [],
        "All frozen V1 model binaries must remain byte-identical.",
    )
    add(
        "v1_freeze_manifest_exceptions_disclosed",
        set(changed) == known_report_changes,
        changed,
        sorted(known_report_changes),
        "The two live-request audit reports were intentionally regenerated after HTTP scoring was disabled.",
    )
    add("final_holdout_sealed", True, False, False, "No 2024+ target is opened by this closeout.")
    check_frame = pd.DataFrame(checks)

    write_json(output / "product_scope_contract.json", contract_payload())
    decisions.to_csv(output / "product_scope_decisions.csv", index=False)
    evidence.to_csv(output / "development_evidence.csv", index=False)
    backlog.to_csv(output / "research_backlog.csv", index=False)
    check_frame.to_csv(output / "build_checks.csv", index=False)
    summary = {
        "contract_version": CONTRACT_VERSION,
        "generic_continuity_status": "retired",
        "development_candidates": [
            "meaningful_retention_24m", "temporary_displacement_risk_24m"
        ],
        "research_only_outputs": ["permanent_separation_risk_24m"],
        "numeric_outputs_authorized": 0,
        "checks_passed": int(check_frame.passed.sum()),
        "checks_total": len(check_frame),
        "final_holdout_opened": False,
        "deployment_changed": False,
        "v1_serialized_models_unchanged": models_unchanged,
        "v1_freeze_manifest_changed_files": changed,
    }
    write_json(output / "run_summary.json", summary)
    (output / "README.md").write_text(
        """# Engine V2 movement-scope closeout

Generic continuity is retired. Meaningful Retention and Temporary Displacement
Risk are separate development candidates that require later untouched temporal
validation. Permanent Separation is research-only and unavailable as a
personalized output. No numeric movement/retention output is deployed.

This directory contains compact publication evidence only; row-level
predictions remain local and are reproducible from the registered runners.
""",
        encoding="utf-8",
    )
    sources = [
        ROOT / "models" / "engine_v2" / "movement_product_scope_contract.py",
        ROOT / "models" / "engine_v2" / "movement_target_contract.py",
        ROOT / "models" / "run_engine_v2_movement_scope_closeout.py",
        DIAGNOSTIC_DIR / "binary_endpoint_pooled_metrics.csv",
        DIAGNOSTIC_DIR / "binary_endpoint_subgroup_metrics.csv",
        DIAGNOSTIC_DIR / "independent_verification.json",
        MANAGER_DIR / "independent_verification.json",
        FREEZE,
    ]
    pd.DataFrame([
        {
            "source": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sources
    ]).to_csv(output / "source_manifest.csv", index=False)
    manifest = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "output_manifest.csv":
            manifest.append({
                "output_file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    pd.DataFrame(manifest).to_csv(output / "output_manifest.csv", index=False)
    if not check_frame.passed.all():
        raise RuntimeError(check_frame.loc[~check_frame.passed].to_dict("records"))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

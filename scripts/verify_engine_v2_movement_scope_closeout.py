"""Independent verification for the Engine V2 movement-scope closeout."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.engine_v2.movement_product_scope_contract import (  # noqa: E402
    DEVELOPMENT_CANDIDATE,
    RESEARCH_ONLY,
    RETIRED,
    contract_payload,
    validate_contract,
)


P = ROOT / "Data" / "processed"
OUTPUT = P / "engine_v2_movement_scope_closeout"
DIAGNOSTIC = P / "engine_v2_movement_state_profile_diagnostic"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_valid(directory: Path, name: str = "output_manifest.csv") -> bool:
    manifest = pd.read_csv(directory / name)
    file_column = "output_file" if name == "output_manifest.csv" else "source"
    base = directory if name == "output_manifest.csv" else ROOT
    return all(
        (base / getattr(row, file_column)).is_file()
        and (base / getattr(row, file_column)).stat().st_size == row.bytes
        and sha256(base / getattr(row, file_column)) == row.sha256
        for row in manifest.itertuples(index=False)
    )


def api_fails_closed_with_override() -> dict[str, object]:
    script = """
import json
from fastapi import HTTPException
import api.main as main
status = None
try:
    main._require_scoring_enabled()
except HTTPException as exc:
    status = exc.status_code
print(json.dumps({
    'scoring_enabled': main.SCORING_ENABLED,
    'engine_loaded': main._profile_builder is not None,
    'health': main.health(),
    'refusal_status': status,
}))
"""
    environment = os.environ.copy()
    environment["MOVEMAKER_ENABLE_SCORING"] = "true"
    run = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(run.stdout.strip().splitlines()[-1])


def clean_rerun_matches() -> bool:
    with tempfile.TemporaryDirectory(prefix=".movement_scope_verify_", dir=P) as temp:
        rerun = Path(temp) / "closeout"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "models" / "run_engine_v2_movement_scope_closeout.py"),
                "--output", str(rerun),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        files = [
            "README.md", "build_checks.csv", "development_evidence.csv",
            "output_manifest.csv", "product_scope_contract.json",
            "product_scope_decisions.csv", "research_backlog.csv",
            "run_summary.json", "source_manifest.csv",
        ]
        return all(sha256(OUTPUT / name) == sha256(rerun / name) for name in files)


def main() -> None:
    rows: list[dict[str, object]] = []

    def add(check: str, passed: bool, observed: object, expected: object) -> None:
        rows.append({
            "check": check, "passed": bool(passed),
            "observed": observed, "expected": expected,
        })

    stored_contract = json.loads((OUTPUT / "product_scope_contract.json").read_text())
    decisions = pd.read_csv(OUTPUT / "product_scope_decisions.csv")
    evidence = pd.read_csv(OUTPUT / "development_evidence.csv")
    upstream = pd.read_csv(DIAGNOSTIC / "binary_endpoint_pooled_metrics.csv")
    decision_index = decisions.set_index("output_id")

    add("output_manifest", manifest_valid(OUTPUT), True, True)
    add("source_manifest", manifest_valid(OUTPUT, "source_manifest.csv"), True, True)
    add("contract_matches_source", stored_contract == contract_payload(), True, True)
    violations = validate_contract(stored_contract)
    add("contract_invariants", not violations, violations, [])
    add(
        "generic_continuity_retired",
        decision_index.loc["generic_continuity_24m", "status"] == RETIRED,
        decision_index.loc["generic_continuity_24m", "status"],
        RETIRED,
    )
    candidate_statuses = decision_index.loc[
        ["meaningful_retention_24m", "temporary_displacement_risk_24m"], "status"
    ]
    add(
        "replacement_candidates_not_deployed",
        candidate_statuses.eq(DEVELOPMENT_CANDIDATE).all(),
        candidate_statuses.to_dict(),
        DEVELOPMENT_CANDIDATE,
    )
    add(
        "permanent_separation_unavailable",
        decision_index.loc["permanent_separation_risk_24m", "status"] == RESEARCH_ONLY,
        decision_index.loc["permanent_separation_risk_24m", "status"],
        RESEARCH_ONLY,
    )
    add(
        "no_numeric_display",
        not decisions.numeric_output_allowed.any()
        and not decisions.public_probability_claim_allowed.any(),
        {
            "numeric": int(decisions.numeric_output_allowed.sum()),
            "public_claim": int(decisions.public_probability_claim_allowed.sum()),
        },
        {"numeric": 0, "public_claim": 0},
    )

    joined = evidence.merge(
        upstream,
        left_on="diagnostic_endpoint",
        right_on="endpoint",
        suffixes=("_published", "_raw"),
        validate="one_to_one",
    )
    numeric = [
        "evaluation_rows", "event_rate", "pooled_brier", "chronology_brier",
        "brier_skill", "player_cluster_ci_low_95", "player_cluster_ci_high_95",
        "origin_wins", "pooled_auc", "pooled_average_precision", "pooled_ece",
    ]
    maximum_error = max(
        float(np.max(np.abs(joined[f"{column}_published"] - joined[f"{column}_raw"])))
        for column in numeric
    )
    add(
        "development_evidence_reconstruction",
        len(joined) == 4 and maximum_error <= 1e-14,
        {"rows": len(joined), "maximum_error": maximum_error},
        {"rows": 4, "maximum_error": "<=1e-14"},
    )

    public_html = "\n".join(
        (ROOT / "html" / name).read_text(encoding="utf-8").lower()
        for name in ["index.html", "about.html", "how-it-works.html"]
    )
    forbidden = [
        "contract-horizon stability", "continuous-stay probability",
        "permanent relationship intact", "supports continuity",
    ]
    found = [phrase for phrase in forbidden if phrase in public_html]
    add("retired_public_claims_absent", not found, found, [])
    required_copy = [
        "meaningful retention", "temporary displacement risk",
        "permanent separation", "development candidate", "research only",
    ]
    missing_copy = [phrase for phrase in required_copy if phrase not in public_html]
    add("replacement_status_copy_present", not missing_copy, missing_copy, [])

    api = api_fails_closed_with_override()
    api_closed = (
        api["scoring_enabled"] is False
        and api["engine_loaded"] is False
        and api["health"]["scoring_enabled"] is False
        and api["refusal_status"] == 503
    )
    add("api_override_cannot_enable_scoring", api_closed, api, "false/false/false/503")
    code_text = (
        (ROOT / "api" / "main.py").read_text(encoding="utf-8")
        + (ROOT / "scripts" / "railway_bootstrap.py").read_text(encoding="utf-8")
    )
    add(
        "no_scoring_environment_bypass",
        "MOVEMAKER_ENABLE_SCORING" not in code_text,
        "MOVEMAKER_ENABLE_SCORING" in code_text,
        False,
    )

    register = (ROOT / "docs" / "ENGINE_V2_CRITICAL_ISSUES.md").read_text(encoding="utf-8")
    open_section = register.split("## Resolved critical issues", 1)[0]
    add(
        "critical_issue_closed_by_retirement",
        "E2-CRIT-001" not in open_section
        and "RESOLVED 2026-09-02 — ENDPOINT RETIRED" in register,
        {
            "in_open_section": "E2-CRIT-001" in open_section,
            "resolved_status": "RESOLVED 2026-09-02 — ENDPOINT RETIRED" in register,
        },
        {"in_open_section": False, "resolved_status": True},
    )
    add(
        "remaining_blockers_preserved",
        "E2-CRIT-002" in open_section and "E2-CRIT-003" in open_section,
        True,
        True,
    )

    origins = pd.read_csv(DIAGNOSTIC / "binary_endpoint_origin_metrics.csv")
    add("final_holdout_sealed", origins.evaluation_year.max() == 2023, origins.evaluation_year.max(), 2023)
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    frozen = all(sha256(ROOT / item["path"]) == item["sha256"] for item in freeze["files"])
    add("v1_frozen_before_rerun", frozen, frozen, True)

    local_only = [
        "Data/processed/engine_v2_manager_tactical_experiment/binary_predictions.csv",
        "Data/processed/engine_v2_manager_tactical_profiles/manager_tactical_profiles.csv",
        "Data/processed/engine_v2_role_squad_profiles/role_squad_profiles.csv",
        "Data/processed/engine_v2_lagged_club_behavior/lagged_club_behavior_features.csv",
    ]
    ignored_paths = {
        path: subprocess.run(
            ["git", "check-ignore", "--quiet", path], cwd=ROOT
        ).returncode == 0
        for path in local_only
    }
    tracked = subprocess.run(
        ["git", "ls-files", "--", *local_only],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
    add(
        "row_level_artifacts_local_only",
        all(ignored_paths.values()) and not tracked,
        {"ignored": ignored_paths, "tracked": tracked},
        {"ignored": "all true", "tracked": []},
    )

    rerun_match = clean_rerun_matches()
    add("clean_closeout_rerun", rerun_match, rerun_match, True)
    frozen_after = all(
        sha256(ROOT / item["path"]) == item["sha256"] for item in freeze["files"]
    )
    add("v1_frozen_after_rerun", frozen_after, frozen_after, True)

    checks = pd.DataFrame(rows)
    print(checks.to_string(index=False))
    print(f"\n{int(checks.passed.sum())}/{len(checks)} independent checks passed")
    verification = {
        "checks_passed": int(checks.passed.sum()),
        "checks_total": len(checks),
        "all_passed": bool(checks.passed.all()),
        "development_evidence_maximum_absolute_error": maximum_error,
        "api_override_cannot_enable_scoring": api_closed,
        "clean_closeout_rerun_byte_identical": rerun_match,
        "final_holdout_opened": False,
        "deployment_changed": False,
        "v1_unchanged": frozen_after,
        "checks": checks.to_dict("records"),
    }
    (OUTPUT / "independent_verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not checks.passed.all():
        raise RuntimeError(checks.loc[~checks.passed].to_dict("records"))


if __name__ == "__main__":
    main()

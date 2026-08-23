"""Controlled, single-endpoint deployment: promote C6 to be the production
`sustained_meaningful_contribution` artifact.

C6 = C3 (M2 + contract terms + relative financial context) minus BOTH
pre365_goals_per90 and pre365_assists_per90. It advanced in
Data/processed/contribution_model_repair/model_selection_decision.csv via
the COMPACT NON-INFERIORITY route, not statistical superiority: pooled
Brier improvement vs the prior M5 (=C0) artifact was +0.002246 with a
player-clustered 95% CI of [-0.000896, +0.005349] (does not exclude zero),
2/4 rolling-origin wins. It was promoted because it also removes the
previously-deployed model's severe, reproducible goals-per-90
face-invalidity while not degrading calibration, the terminal-two-year
window, or the majority cohort, and while cutting mean missing-input
dependence from 16.4% to 3.8%.

Scope discipline (all enforced structurally below, not just by convention):
  - Writes ONLY Data/processed/deployment_models/extension_opportunity_
    diagnostic/sustained_meaningful_contribution.joblib and the matching
    single entry inside Data/processed/deployment_models/model_manifest.json.
    All 15 other endpoint artifacts and their manifest entries are read
    but never rewritten.
  - Does NOT touch models/deployment_scorer.py, api/, html/, or slides/.
  - Backs up the pre-deployment artifact AND the full manifest before any
    write, so this is a clean single-file rollback if ever needed.
  - Refits using scripts/build_deployment_models.py's own refit_linear()
    (imported, not re-derived) -- the exact same recipe, hyperparameter
    grid, and train/validation/final-fit split every other deployed
    endpoint used -- with only the feature list swapped to C6's
    already-approved set.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
SCRIPTS_DIR = ROOT / "scripts"
for path in (MODELS_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_deployment_models as bdm  # noqa: E402 -- refit_linear, sha256_file, verify_deployment_cutoff_maturity, MASTER_PATH
import run_extension_opportunity_diagnostic as phase1  # noqa: E402
from deployment_model_artifact import ModelArtifact  # noqa: E402
from run_contribution_model_repair import C5_FEATURES, C6_FEATURES  # noqa: E402 -- already-approved feature set, not re-derived here

DEPLOYMENT_OUTPUT = ROOT / "Data" / "processed" / "deployment_models"
ARTIFACT_PATH = DEPLOYMENT_OUTPUT / "extension_opportunity_diagnostic" / "sustained_meaningful_contribution.joblib"
MANIFEST_PATH = DEPLOYMENT_OUTPUT / "model_manifest.json"
BACKUP_DIR = DEPLOYMENT_OUTPUT / "_backups" / f"pre_c6_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

PHASE = "extension_opportunity_diagnostic"
ENDPOINT = "sustained_meaningful_contribution"
TARGET = "target_sustained_meaningful_contribution"
COHORT_COL = "cohort_primary_y2"

SELECTION_NOTE = (
    "C6 advanced through the COMPACT NON-INFERIORITY route, NOT statistical superiority: pooled Brier improvement vs "
    "the prior M5 (=C0) artifact was +0.002246 with a player-clustered 95% CI of [-0.000896, +0.005349] (does not "
    "exclude zero), 2/4 rolling-origin wins. It was promoted because it also removed the previously-deployed model's "
    "severe, reproducible goals-per-90 face-invalidity (a smooth, monotonic, sign-stable-but-counterintuitive "
    "probability decrease as recent scoring rate increases -- see "
    "Data/processed/contribution_model_repair/goals_per90_perturbation_audit.csv), cut mean missing-input dependence "
    "from 16.4% to 3.8%, and did not degrade the terminal-two-year window, calibration, or the majority cohort. "
    "Full evidence: Data/processed/contribution_model_repair/."
)


def main() -> None:
    if not ARTIFACT_PATH.exists() or not MANIFEST_PATH.exists():
        raise FileNotFoundError("Expected existing production artifact/manifest not found; refusing to deploy.")
    if "pre365_goals_per90" in C6_FEATURES or "pre365_assists_per90" in C6_FEATURES:
        raise RuntimeError("C6_FEATURES unexpectedly contains a goals/assists-per-90 feature; refusing to deploy.")
    if "pre365_assists_per90" not in C5_FEATURES:
        raise RuntimeError("C5_FEATURES sanity check failed (should retain pre365_assists_per90); import may be stale.")

    print("Step 1: backing up the existing artifact and manifest BEFORE any write...")
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ARTIFACT_PATH, BACKUP_DIR / ARTIFACT_PATH.name)
    shutil.copy2(MANIFEST_PATH, BACKUP_DIR / MANIFEST_PATH.name)
    original_artifact_sha256 = bdm.sha256_file(ARTIFACT_PATH)
    original_manifest_sha256 = bdm.sha256_file(MANIFEST_PATH)
    (BACKUP_DIR / "backup_manifest.json").write_text(json.dumps({
        "backed_up_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "Pre-C6-deployment backup of sustained_meaningful_contribution.joblib and the full model_manifest.json.",
        "original_artifact_sha256": original_artifact_sha256, "original_manifest_sha256": original_manifest_sha256,
        "original_artifact_path": str(ARTIFACT_PATH.relative_to(ROOT)), "original_manifest_path": str(MANIFEST_PATH.relative_to(ROOT)),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"  backed up to {BACKUP_DIR.relative_to(ROOT)}")

    print("Step 2: refitting C6 through the exact production cutoff (build_deployment_models.refit_linear, unmodified)...")
    frame = phase1.load_frame()
    bdm.verify_deployment_cutoff_maturity(frame)
    source_hash = bdm.sha256_file(bdm.MASTER_PATH)
    data = frame.loc[frame[COHORT_COL] & frame[TARGET].notna()].copy()
    pre, model, hyperparameter, validation_metrics, training_rows, encoded_feature_count, interval_width = bdm.refit_linear(
        data, C6_FEATURES, TARGET, "binary", phase1.RIDGE_ALPHAS, phase1.LOGISTIC_C, clip_0_1_05=False,
    )
    artifact = ModelArtifact(
        phase=PHASE, endpoint=ENDPOINT, horizon=None, kind="binary", unit="probability",
        variant="C6_C3_minus_pre365_goals_assists", features=C6_FEATURES, preprocessor=pre, model=model,
        hyperparameter=hyperparameter, interval_abs_residual_q80=interval_width, training_rows=training_rows,
        encoded_feature_count=encoded_feature_count, validation_metrics=validation_metrics, source_data_sha256=source_hash,
        deployment_cutoff_signed_year=bdm.DEPLOYMENT_CUTOFF_SIGNED_YEAR,
        model_vintage=f"{bdm.MODEL_VINTAGE}_C6_compact_non_inferiority",
    )
    print(f"  training_rows={training_rows}, hyperparameter={hyperparameter}, validation_brier={validation_metrics.get('brier')}")
    assert set(artifact.features).isdisjoint({"pre365_goals_per90", "pre365_assists_per90"}), "Refit artifact must not contain either dropped feature."

    print("Step 3: writing ONLY the sustained_meaningful_contribution artifact...")
    joblib.dump(artifact, ARTIFACT_PATH)
    new_artifact_sha256 = bdm.sha256_file(ARTIFACT_PATH)
    print(f"  new artifact sha256: {new_artifact_sha256}")

    print("Step 4: patching ONLY the matching manifest entry; all 15 other entries and the top-level model_vintage are left untouched...")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    matched = 0
    for entry in manifest["artifacts"]:
        if entry["phase"] == PHASE and entry["endpoint"] == ENDPOINT:
            matched += 1
            entry.update({
                "variant": artifact.variant,
                "hyperparameter": json.dumps(hyperparameter, sort_keys=True),
                "feature_count": len(C6_FEATURES),
                "encoded_feature_count": encoded_feature_count,
                "training_rows": training_rows,
                "interval_abs_residual_q80": interval_width,
                "trained_at_utc": artifact.trained_at_utc,
                "artifact_sha256": new_artifact_sha256,
                "validation_brier": validation_metrics.get("brier"),
                "model_vintage": artifact.model_vintage,
                "selection_basis": "compact_non_inferiority",
                "selection_note": SELECTION_NOTE,
                "excludes_features_present_in_prior_deployment": ["pre365_goals_per90", "pre365_assists_per90"],
                "research_evidence_dir": "Data/processed/contribution_model_repair/",
                "pre_deployment_backup_dir": str(BACKUP_DIR.relative_to(ROOT)),
            })
    if matched != 1:
        raise RuntimeError(f"Expected exactly 1 matching manifest entry for {PHASE}/{ENDPOINT}, found {matched}; refusing to write a possibly-corrupt manifest.")
    manifest.setdefault("deployment_log", []).append({
        "deployed_at_utc": datetime.now(timezone.utc).isoformat(), "phase": PHASE, "endpoint": ENDPOINT,
        "action": "single_endpoint_promotion", "from_variant": "M5_plus_relative_financial_context", "to_variant": artifact.variant,
        "selection_basis": "compact_non_inferiority", "previous_artifact_sha256": original_artifact_sha256,
        "new_artifact_sha256": new_artifact_sha256, "previous_manifest_sha256": original_manifest_sha256,
        "backup_dir": str(BACKUP_DIR.relative_to(ROOT)),
    })
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"  manifest updated; deployment_log now has {len(manifest['deployment_log'])} entr{'y' if len(manifest['deployment_log']) == 1 else 'ies'}.")

    print("\nDeployment complete.")
    print(f"  artifact: {ARTIFACT_PATH.relative_to(ROOT)}")
    print(f"  backup:   {BACKUP_DIR.relative_to(ROOT)}")
    print("Run scripts/verify_c6_deployment.py next.")


if __name__ == "__main__":
    main()

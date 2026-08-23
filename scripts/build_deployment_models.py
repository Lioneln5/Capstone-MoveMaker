"""Full-history refit and serialization of the retained Phase 1-4 models.

This is deployment refitting, not a new experiment: every architecture,
feature set, hyperparameter grid, and selection rule below is copied
unchanged from the already-verified diagnostic scripts (imported directly,
not re-transcribed, wherever the diagnostic module exposes a reusable
function). The only thing that changes is *how much history* each model is
allowed to train on and that the fitted objects are persisted to disk
instead of being discarded after producing evaluation metrics.

Deployment cutoff
------------------
signed_year <= 2023 is the last cohort with 100% Year-2 (post_y2) sporting
observability and 100% 24-month valuation observability in the current
extension_modeling_master.csv (verified below at build time, not assumed).
2024 signings are only 34-75% mature and are excluded from every
outcome-dependent model (Phases 1-3) to avoid training on a
partially-censored cohort. Phase 4 (financial peer benchmarks) does not
require future-outcome maturity -- it benchmarks *current* contract terms
against *current* profile -- but is held to the same 2023 cutoff so the five
displayed cards share one frozen model vintage, per the deployment specification
to declare a single production cutoff rather than a per-card patchwork.

Recipe per endpoint (mirrors each diagnostic's own tune_predict/
fit_survival_model exactly, just with the window pushed to the deployment
cutoff instead of stopping at 2023's rolling-origin evaluation year):
  1. train      = eligible rows with signed_year <= 2022
  2. validation = eligible rows with signed_year == 2023
  3. grid-search the endpoint's already-selected hyperparameter family
     (Ridge alpha / LogisticRegression C / GradientBoostingRegressor grid)
     on validation loss, exactly as the diagnostic does
  4. final production model = refit preprocessor + model on train UNION
     validation (i.e. on 100% of signed_year <= 2023), using the selected
     hyperparameter -- no held-out row is wasted at deployment time
  5. serialize preprocessor + model + full metadata

Only the endpoints selected for the frozen five-card product are refit here
(see docs/journal/2026-08-11_scope_pivot_journal.md section 21 and
Data/processed/integrated_contract_profile/web_app_output_specification.csv):
rejected/exploratory endpoints (major_role_decline, contract_covered_
minimal_involvement, permanent_outbound-36m, downside_50pct_12m) are not
serialized.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from mixed_type_preprocessor import Preprocessor, fit_preprocessor  # noqa: E402
from deployment_model_artifact import ModelArtifact  # noqa: E402
import run_extension_opportunity_diagnostic as phase1  # noqa: E402
import run_extension_survival_diagnostic as phase2  # noqa: E402
import run_extension_value_preservation_diagnostic as phase3  # noqa: E402
import run_contract_financial_exposure_benchmark as phase4  # noqa: E402

OUTPUT = ROOT / "Data" / "processed" / "deployment_models"
MASTER_PATH = ROOT / "Data" / "processed" / "contract_extension_integration" / "extension_modeling_master.csv"

DEPLOYMENT_CUTOFF_SIGNED_YEAR = 2023
DEPLOYMENT_TRAIN_THROUGH_YEAR = 2022
DEPLOYMENT_VALIDATION_YEAR = 2023
MODEL_VINTAGE = f"deployment_v1_cutoff_{DEPLOYMENT_CUTOFF_SIGNED_YEAR}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_deployment_cutoff_maturity(frame: pd.DataFrame) -> None:
    """Refuse to build if the assumed cutoff isn't actually fully mature --
    catches the case where this script is rerun after the raw data changed
    without updating DEPLOYMENT_CUTOFF_SIGNED_YEAR above."""
    at_cutoff = frame.loc[frame["signed_year"].eq(DEPLOYMENT_CUTOFF_SIGNED_YEAR)]
    if at_cutoff.empty:
        raise RuntimeError(f"No signed extensions in {DEPLOYMENT_CUTOFF_SIGNED_YEAR}; cutoff is not usable.")
    for column in ["post_y2_window_fully_observable", "post24_horizon_fully_observable"]:
        observed = frame.loc[frame["signed_year"].eq(DEPLOYMENT_CUTOFF_SIGNED_YEAR), column]
        rate = observed.astype("string").str.casefold().eq("true").mean()
        if rate < 0.999:
            raise RuntimeError(
                f"{column} maturity for signed_year={DEPLOYMENT_CUTOFF_SIGNED_YEAR} is only {rate:.1%}; "
                "the deployment cutoff is stale and must be re-chosen, not silently reused."
            )
    beyond = frame.loc[frame["signed_year"].gt(DEPLOYMENT_CUTOFF_SIGNED_YEAR)]
    if len(beyond):
        print(f"  (note: {len(beyond)} rows with signed_year > {DEPLOYMENT_CUTOFF_SIGNED_YEAR} exist and are correctly excluded from training)")


# ModelArtifact must live in its own importable module (deployment_model_
# artifact.py), not here -- this script and models/deployment_scorer.py are
# different __main__ contexts, and pickle resolves a class by its defining
# module, so a class defined in a script run as __main__ can't be unpickled
# from a different entry point.


def save_artifact(artifact: ModelArtifact, manifest_rows: list[dict[str, Any]]) -> Path:
    subdir = OUTPUT / artifact.phase
    subdir.mkdir(parents=True, exist_ok=True)
    if artifact.horizon is None or str(artifact.horizon) in artifact.endpoint:
        name = artifact.endpoint
    else:
        name = f"{artifact.endpoint}_{artifact.horizon}"
    path = subdir / f"{name}.joblib"
    joblib.dump(artifact, path)
    manifest_rows.append({
        "phase": artifact.phase, "endpoint": artifact.endpoint, "horizon": artifact.horizon,
        "kind": artifact.kind, "variant": artifact.variant, "hyperparameter": json.dumps(artifact.hyperparameter, sort_keys=True),
        "feature_count": len(artifact.features), "encoded_feature_count": artifact.encoded_feature_count,
        "training_rows": artifact.training_rows, "interval_abs_residual_q80": artifact.interval_abs_residual_q80,
        "deployment_cutoff_signed_year": artifact.deployment_cutoff_signed_year, "model_vintage": artifact.model_vintage,
        "trained_at_utc": artifact.trained_at_utc, "artifact_path": path.relative_to(ROOT).as_posix(),
        "artifact_sha256": sha256_file(path),
        **{f"validation_{k}": v for k, v in artifact.validation_metrics.items()},
    })
    print(f"  saved {path.relative_to(ROOT)}  (train_rows={artifact.training_rows}, hyperparameter={artifact.hyperparameter})")
    return path


# ---------------------------------------------------------------------------
# Shared linear refit (Ridge / LogisticRegression), used by Phases 1 and 3.
# ---------------------------------------------------------------------------
def refit_linear(
    data: pd.DataFrame, features: list[str], target: str, kind: str,
    ridge_alphas: list[float], logistic_c: list[float], clip_0_1_05: bool = False,
) -> tuple[Preprocessor, Any, dict[str, Any], dict[str, Any], int, int, float | None]:
    """clip_0_1_05 must only be True for a target that is itself bounded in
    [0, 1.05] (Phase 1's opportunity-share target). Phase 3's continuous
    targets are unbounded log ratios -- clipping them would corrupt both
    hyperparameter selection and the validation-residual interval width."""
    train = data.loc[data["signed_year"].le(DEPLOYMENT_TRAIN_THROUGH_YEAR)].copy()
    validation = data.loc[data["signed_year"].eq(DEPLOYMENT_VALIDATION_YEAR)].copy()
    if min(len(train), len(validation)) < 30:
        raise RuntimeError(f"Insufficient deployment split for {target}: train={len(train)} validation={len(validation)}")

    pre = fit_preprocessor(train, features)
    x_train, x_validation = pre.transform(train), pre.transform(validation)
    y_train = pd.to_numeric(train[target]).to_numpy()
    y_validation = pd.to_numeric(validation[target]).to_numpy()

    def clip(values: np.ndarray) -> np.ndarray:
        return np.clip(values, 0, 1.05) if clip_0_1_05 else values

    candidates = []
    if kind == "continuous":
        for alpha in ridge_alphas:
            model = Ridge(alpha=alpha).fit(x_train, y_train)
            pred = clip(model.predict(x_validation))
            candidates.append(((mean_absolute_error(y_validation, pred), float(np.sqrt(mean_squared_error(y_validation, pred)))), {"alpha": alpha}, pred))
    else:
        y_train_int, y_validation_int = y_train.astype(int), y_validation.astype(int)
        for c_value in logistic_c:
            model = LogisticRegression(C=c_value, penalty="l2", solver="liblinear", max_iter=3000, random_state=phase1.RANDOM_SEED).fit(x_train, y_train_int)
            pred = np.clip(model.predict_proba(x_validation)[:, 1], 1e-6, 1 - 1e-6)
            candidates.append(((brier_score_loss(y_validation_int, pred), log_loss(y_validation_int, pred, labels=[0, 1])), {"C": c_value}, pred))
    _, hyperparameter, best_validation_pred = min(candidates, key=lambda item: item[0])

    fit = pd.concat([train, validation], ignore_index=True)
    fit_pre = fit_preprocessor(fit, features)
    x_fit = fit_pre.transform(fit)
    y_fit = pd.to_numeric(fit[target]).to_numpy()
    interval_width: float | None = None
    if kind == "continuous":
        model = Ridge(alpha=hyperparameter["alpha"]).fit(x_fit, y_fit)
        validation_metrics = {"mae": float(mean_absolute_error(y_validation, best_validation_pred))}
        interval_width = float(np.quantile(np.abs(y_validation - best_validation_pred), .80))
    else:
        model = LogisticRegression(C=hyperparameter["C"], penalty="l2", solver="liblinear", max_iter=3000, random_state=phase1.RANDOM_SEED).fit(x_fit, y_fit.astype(int))
        validation_metrics = {"brier": float(brier_score_loss(y_validation.astype(int), best_validation_pred))}
    return fit_pre, model, hyperparameter, validation_metrics, len(fit), x_fit.shape[1], interval_width


def build_phase1(source_hash: str, manifest_rows: list[dict[str, Any]]) -> None:
    print("Phase 1 -- opportunity")
    frame = phase1.load_frame()
    verify_deployment_cutoff_maturity(frame)
    features = phase1.FEATURE_SETS["M5_plus_relative_financial_context"]
    specs = [
        ("year2_opportunity_share", "cohort_primary_y2", "target_year2_opportunity_share", "continuous", "share_0_1"),
        ("sustained_meaningful_contribution", "cohort_primary_y2", "target_sustained_meaningful_contribution", "binary", "probability"),
    ]
    for endpoint, cohort_col, target, kind, unit in specs:
        data = frame.loc[frame[cohort_col] & frame[target].notna()].copy()
        pre, model, hp, val_metrics, rows, encoded, interval_width = refit_linear(
            data, features, target, kind, phase1.RIDGE_ALPHAS, phase1.LOGISTIC_C, clip_0_1_05=(kind == "continuous")
        )
        artifact = ModelArtifact(
            phase="extension_opportunity_diagnostic", endpoint=endpoint, horizon=None, kind=kind, unit=unit,
            variant="M5_plus_relative_financial_context", features=features, preprocessor=pre, model=model,
            hyperparameter=hp, interval_abs_residual_q80=interval_width, training_rows=rows, encoded_feature_count=encoded,
            validation_metrics=val_metrics, source_data_sha256=source_hash,
            deployment_cutoff_signed_year=DEPLOYMENT_CUTOFF_SIGNED_YEAR, model_vintage=MODEL_VINTAGE,
        )
        save_artifact(artifact, manifest_rows)


def build_phase3(source_hash: str, manifest_rows: list[dict[str, Any]]) -> None:
    print("Phase 3 -- value preservation")
    frame = phase3.load_frame()
    features = phase3.FEATURE_SETS["M5_plus_relative_financial_context"]
    endpoints = [
        "value_log_ratio_12m", "value_log_ratio_24m",
        "downside_10pct_12m", "downside_10pct_24m",  # Tier 2 (2026-08-12): both already documented as advancing candidates, just not deployed until now
        "downside_25pct_12m", "downside_25pct_24m", "downside_50pct_24m",
    ]  # excludes downside_50pct_12m (exploratory only, rejected for display)
    for endpoint in endpoints:
        spec = phase3.ENDPOINTS[endpoint]
        data = phase3.endpoint_frame(frame, endpoint)
        pre, model, hp, val_metrics, rows, encoded, interval_width = refit_linear(data, features, spec["target"], spec["kind"], phase3.RIDGE_ALPHAS, phase3.LOGISTIC_C)
        unit = "log_ratio" if spec["kind"] == "continuous" else "probability"
        artifact = ModelArtifact(
            phase="extension_value_preservation_diagnostic", endpoint=endpoint, horizon=spec["horizon"], kind=spec["kind"], unit=unit,
            variant="M5_plus_relative_financial_context", features=features, preprocessor=pre, model=model,
            hyperparameter=hp, interval_abs_residual_q80=interval_width, training_rows=rows, encoded_feature_count=encoded,
            validation_metrics=val_metrics, source_data_sha256=source_hash,
            deployment_cutoff_signed_year=DEPLOYMENT_CUTOFF_SIGNED_YEAR, model_vintage=MODEL_VINTAGE,
        )
        save_artifact(artifact, manifest_rows)


# ---------------------------------------------------------------------------
# Phase 2 -- discrete-time hazard survival model.
# ---------------------------------------------------------------------------
def refit_hazard(
    cohort: pd.DataFrame, endpoint: str, features: list[str],
) -> tuple[Preprocessor, Any, dict[str, Any], dict[str, Any], int, int]:
    train = cohort.loc[cohort["signed_year"].le(DEPLOYMENT_TRAIN_THROUGH_YEAR)].copy()
    validation = cohort.loc[cohort["signed_year"].eq(DEPLOYMENT_VALIDATION_YEAR)].copy()
    if min(len(train), len(validation)) < 30:
        raise RuntimeError(f"Insufficient deployment split for hazard/{endpoint}: train={len(train)} validation={len(validation)}")

    fit_features = features + ["hazard_interval"]
    train_periods = phase2.person_periods(train, endpoint)
    pre = fit_preprocessor(train_periods, fit_features)
    x_train = pre.transform(train_periods)
    y_train = train_periods["hazard_event"].to_numpy(int)

    candidates = []
    for c_value in phase2.C_GRID:
        model = LogisticRegression(C=c_value, penalty="l2", solver="liblinear", max_iter=3000, random_state=phase2.RANDOM_SEED).fit(x_train, y_train)
        cumulative = phase2.predict_hazard_curve(model, pre, validation, features)
        candidates.append((phase2.validation_integrated_brier(validation, endpoint, cumulative), {"C": c_value}))
    _, hyperparameter = min(candidates, key=lambda item: item[0])

    fit_subjects = pd.concat([train, validation], ignore_index=True)
    fit_periods = phase2.person_periods(fit_subjects, endpoint)
    fit_pre = fit_preprocessor(fit_periods, fit_features)
    x_fit = fit_pre.transform(fit_periods)
    y_fit = fit_periods["hazard_event"].to_numpy(int)
    model = LogisticRegression(C=hyperparameter["C"], penalty="l2", solver="liblinear", max_iter=3000, random_state=phase2.RANDOM_SEED).fit(x_fit, y_fit)

    val_cumulative = phase2.predict_hazard_curve(model, fit_pre, validation, features)
    validation_metrics = {"integrated_brier": phase2.validation_integrated_brier(validation, endpoint, val_cumulative)}
    return fit_pre, model, hyperparameter, validation_metrics, len(fit_periods), x_fit.shape[1]


def build_phase2(source_hash: str, manifest_rows: list[dict[str, Any]]) -> None:
    print("Phase 2 -- survival / continuity")
    frame = phase2.load_frame()
    cohort = frame.loc[frame["survival_primary_cohort"]].copy()
    features = phase2.FEATURE_SETS["M5_plus_relative_financial_context"]
    # (endpoint, horizons to serialize) -- permanent_outbound 36m is excluded:
    # it failed the calibration/recent-stability gate and is exploratory-only.
    targets = [("any_outbound", ["24m", "36m"]), ("permanent_outbound", ["24m"])]
    for endpoint, horizons in targets:
        pre, model, hp, val_metrics, rows, encoded = refit_hazard(cohort, endpoint, features)
        for horizon in horizons:
            artifact = ModelArtifact(
                phase="extension_survival_diagnostic", endpoint=endpoint, horizon=horizon, kind="hazard", unit="probability",
                variant="M5_plus_relative_financial_context", features=features, preprocessor=pre, model=model,
                hyperparameter=hp, interval_abs_residual_q80=None, training_rows=rows, encoded_feature_count=encoded,
                validation_metrics=val_metrics, source_data_sha256=source_hash,
            deployment_cutoff_signed_year=DEPLOYMENT_CUTOFF_SIGNED_YEAR, model_vintage=MODEL_VINTAGE,
            )
            save_artifact(artifact, manifest_rows)
    # INTERVAL_BOUNDS is shared by every horizon of a given endpoint's single
    # fitted hazard model; the scorer needs it to know which expanded-period
    # column index a horizon maps to, so persist it once alongside the model.
    (OUTPUT / "extension_survival_diagnostic").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "extension_survival_diagnostic" / "interval_bounds.json").write_text(
        json.dumps(phase2.INTERVAL_BOUNDS), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Phase 4 -- financial peer benchmarks (Ridge or GradientBoostingRegressor).
# ---------------------------------------------------------------------------
PHASE4_UNITS = {
    "annual_wage": "log_eur", "fixed_wage_commitment": "log_eur",
    "contract_duration": "years", "wage_to_market_value": "log_ratio",
}


def build_phase4(source_hash: str, manifest_rows: list[dict[str, Any]]) -> None:
    print("Phase 4 -- financial peer benchmarks")
    frame = phase4.load_frame()
    for endpoint, variant in phase4.SELECTED_MODELS.items():
        spec = phase4.ENDPOINTS[endpoint]
        features = phase4.FEATURE_SETS[variant]
        data = frame.loc[frame[spec["cohort"]]].copy()
        train = data.loc[data["signed_year"].le(DEPLOYMENT_TRAIN_THROUGH_YEAR)].copy()
        validation = data.loc[data["signed_year"].eq(DEPLOYMENT_VALIDATION_YEAR)].copy()
        if min(len(train), len(validation)) < 30:
            raise RuntimeError(f"Insufficient deployment split for {endpoint}: train={len(train)} validation={len(validation)}")

        pre = fit_preprocessor(train, features)
        x_train, x_validation = pre.transform(train), pre.transform(validation)
        y_train = pd.to_numeric(train[spec["target"]]).to_numpy()
        y_validation = pd.to_numeric(validation[spec["target"]]).to_numpy()
        nonlinear = variant == "B5_compact_nonlinear"
        candidates = []
        if nonlinear:
            grid = [
                {"n_estimators": 75, "learning_rate": .03, "max_depth": 1, "min_samples_leaf": 20},
                {"n_estimators": 100, "learning_rate": .03, "max_depth": 2, "min_samples_leaf": 25},
                {"n_estimators": 75, "learning_rate": .05, "max_depth": 2, "min_samples_leaf": 30},
            ]
            for i, params in enumerate(grid):
                model = GradientBoostingRegressor(loss="huber", random_state=phase4.stable_seed(endpoint, variant, str(i)), **params).fit(x_train, y_train)
                estimate = model.predict(x_validation)
                candidates.append(((mean_absolute_error(y_validation, estimate), float(np.sqrt(mean_squared_error(y_validation, estimate)))), params, estimate))
        else:
            for alpha in phase4.RIDGE_ALPHAS:
                model = Ridge(alpha=alpha).fit(x_train, y_train)
                estimate = model.predict(x_validation)
                candidates.append(((mean_absolute_error(y_validation, estimate), float(np.sqrt(mean_squared_error(y_validation, estimate)))), {"alpha": alpha}, estimate))
        _, hyperparameter, validation_estimate = min(candidates, key=lambda item: item[0])
        interval_width = float(np.quantile(np.abs(y_validation - validation_estimate), .80))

        fit = pd.concat([train, validation], ignore_index=True)
        fit_pre = fit_preprocessor(fit, features)
        x_fit = fit_pre.transform(fit)
        y_fit = pd.to_numeric(fit[spec["target"]]).to_numpy()
        model = (GradientBoostingRegressor(loss="huber", random_state=phase4.stable_seed(endpoint, variant, "fit"), **hyperparameter)
                 if nonlinear else Ridge(alpha=hyperparameter["alpha"]))
        model.fit(x_fit, y_fit)

        artifact = ModelArtifact(
            phase="contract_financial_exposure_benchmark", endpoint=endpoint, horizon=None, kind="continuous",
            unit=PHASE4_UNITS[endpoint],
            variant=variant, features=features, preprocessor=fit_pre, model=model, hyperparameter=hyperparameter,
            interval_abs_residual_q80=interval_width, training_rows=len(fit), encoded_feature_count=x_fit.shape[1],
            validation_metrics={"mae": float(mean_absolute_error(y_validation, validation_estimate))},
            source_data_sha256=source_hash,
            deployment_cutoff_signed_year=DEPLOYMENT_CUTOFF_SIGNED_YEAR, model_vintage=MODEL_VINTAGE,
        )
        save_artifact(artifact, manifest_rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(MASTER_PATH)
    manifest_rows: list[dict[str, Any]] = []
    build_phase1(source_hash, manifest_rows)
    build_phase2(source_hash, manifest_rows)
    build_phase3(source_hash, manifest_rows)
    build_phase4(source_hash, manifest_rows)

    manifest = {
        "model_vintage": MODEL_VINTAGE,
        "deployment_cutoff_signed_year": DEPLOYMENT_CUTOFF_SIGNED_YEAR,
        "train_through_year": DEPLOYMENT_TRAIN_THROUGH_YEAR,
        "validation_year": DEPLOYMENT_VALIDATION_YEAR,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_extension_modeling_master_sha256": source_hash,
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "artifacts": manifest_rows,
    }
    (OUTPUT / "model_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {len(manifest_rows)} model artifacts and model_manifest.json to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

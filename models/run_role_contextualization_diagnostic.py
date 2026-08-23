"""Role-contextualization diagnostic for goals/assists-per-90 features across
every currently deployed MoveMaker endpoint that uses them.

READ-ONLY with respect to production. This script:
  - loads the same source frames as the deployed diagnostics (via import of
    the existing, unmodified run_extension_*.py / run_contract_financial_
    exposure_benchmark.py modules -- every one of those files is guarded by
    `if __name__ == "__main__":`, so importing them here does not execute
    their own pipelines or touch their own output directories);
  - reuses their exact cohort/target/origin-split/tune_predict/
    fit_survival_model machinery for R0-R4, so "R0" is provably the same
    fitting procedure as the deployed models, just isolated to this script's
    own output directory;
  - never writes to Data/processed/deployment_models/, never imports or
    calls anything that would refit or overwrite a deployed artifact, and
    never touches models/deployment_scorer.py, api/, html/, or slides/
    (DeploymentScorer is imported READ-ONLY, for live-audit scoring only).

Candidates (built independently per affected endpoint):
  R0  Deployed reproduction  -- exact deployed raw feature list.
  R1  Remove scoring rates   -- R0 minus every goals/assists-per-90 feature.
  R2  Position-relative      -- raw scoring features replaced by a
                                 training-only position-relative transform
                                 (robust z-score or empirical percentile,
                                 selected per origin on validation only).
  R3  Position interactions  -- raw scoring features kept, plus explicit
                                 scoring x broad-position interaction terms
                                 (Attack/Midfield/Defender only).
  R4  Compact role-aware     -- R1's non-scoring features + whichever of
                                 R2/R3's scoring representation validated
                                 better in that origin.

Run from the repository root:
    python3 models/run_role_contextualization_diagnostic.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, mean_absolute_error,
    mean_squared_error, roc_auc_score,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

import run_extension_opportunity_diagnostic as opp          # noqa: E402
import run_extension_survival_diagnostic as surv            # noqa: E402
import run_extension_value_preservation_diagnostic as val   # noqa: E402
import run_contract_financial_exposure_benchmark as fin     # noqa: E402
from mixed_type_preprocessor import fit_preprocessor         # noqa: E402
from feature_contribution_explainer import label_for, sigmoid  # noqa: E402


def raw_contributions(artifact, fv: dict, drop_features: tuple = (), extra: dict | None = None):
    """Local, isolated copy of feature_contribution_explainer.raw_
    contributions with one bug fix: `model.intercept_[0]` crashes with
    IndexError for a Ridge model on a continuous target (Ridge.intercept_ is
    a plain Python float there, not a length-1 array like LogisticRegression's
    -- production never noticed because explain_headline_scores() only ever
    calls this function for the three binary/hazard cards). Fixed here via
    np.ravel(...)[0], which is correct for both cases; NOT patched in the
    shared models/feature_contribution_explainer.py file itself, per this
    diagnostic's isolation requirement -- see README.md's 'Bug found, not
    fixed' section. Everything else is byte-identical to the original."""
    row = {f: fv.get(f, np.nan) for f in artifact.features}
    if extra:
        row.update(extra)
    frame = pd.DataFrame([row])
    x = artifact.preprocessor.transform(frame)[0]
    coef = artifact.model.coef_[0] if np.ndim(artifact.model.coef_) > 1 else artifact.model.coef_
    intercept = float(np.ravel(artifact.model.intercept_)[0])
    encoded_raw = artifact.preprocessor.encoded_raw_features
    per_raw: dict[str, float] = {}
    for raw, xi, ci in zip(encoded_raw, x, coef):
        if raw in drop_features:
            continue
        per_raw[raw] = per_raw.get(raw, 0.0) + float(xi * ci)
    logit = intercept + sum(per_raw.values())
    ranked = sorted(per_raw.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return ranked, intercept, logit

OUT = ROOT / "Data" / "processed" / "role_contextualization_diagnostic"
CANDIDATES_DIR = OUT / "candidate_artifacts"
OUT.mkdir(parents=True, exist_ok=True)
CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = opp.RANDOM_SEED               # 20260811, shared project seed
BOOTSTRAP_REPETITIONS = opp.BOOTSTRAP_REPETITIONS   # 5000, project standard
ORIGINS = opp.ORIGINS
RIDGE_ALPHAS = opp.RIDGE_ALPHAS
LOGISTIC_C = opp.LOGISTIC_C

DEPLOYMENT_TRAIN_THROUGH_YEAR = 2022
DEPLOYMENT_VALIDATION_YEAR = 2023

BROAD_POSITIONS = ["Attack", "Midfield", "Defender", "Goalkeeper"]
INTERACTION_POSITIONS = ["Attack", "Midfield", "Defender"]  # GK excluded, see spec
MIN_ROLE_TRAIN_N = 30  # minimum training rows in a position group before it
                        # gets its own reference distribution; below this,
                        # global training fallback is used (and logged).

SCORING_FEATURES_ALL = [
    "pre365_goals_per90", "pre365_assists_per90",
    "pre1_canonical_goals_per90", "pre1_canonical_assists_per90",
    "pre2_canonical_goals_per90", "pre2_canonical_assists_per90",
]

manifest_path = ROOT / "Data" / "processed" / "deployment_models" / "model_manifest.json"
DEPLOYMENT_MANIFEST = json.loads(manifest_path.read_text(encoding="utf-8"))

# Not every hazard endpoint has both 24m and 36m artifacts deployed
# (permanent_outbound is 24m-only). Derived from the manifest, not assumed.
DEPLOYED_HAZARD_HORIZONS: dict[str, list[str]] = {}
for _a in DEPLOYMENT_MANIFEST["artifacts"]:
    if _a["phase"] == "extension_survival_diagnostic":
        DEPLOYED_HAZARD_HORIZONS.setdefault(_a["endpoint"], []).append(_a["horizon"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


PROTECTED_TARGETS = [
    "Data/processed/deployment_models", "models/deployment_scorer.py",
    "api", "html", "slides", "Data/processed/contribution_model_repair",
]


def capture_protected_fingerprint(path: Path) -> None:
    """Capture the current protected tree for a genuine before/after audit."""
    lines: list[str] = []
    for target in PROTECTED_TARGETS:
        lines.append(f"=== {target} ===")
        item = ROOT / target
        if item.is_dir():
            for file_path in sorted(
                p for p in item.rglob("*")
                if p.is_file() and p.name != ".DS_Store" and "__pycache__" not in p.parts
            ):
                lines.append(f"{sha256_file(file_path)}  {file_path.relative_to(ROOT)}")
        elif item.is_file():
            lines.append(f"{sha256_file(item)}  {target}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(frame: pd.DataFrame, name: str) -> None:
    path = OUT / name
    frame.to_csv(path, index=False)
    print(f"  wrote {path} ({len(frame)} rows)")


# =============================================================================
# PHASE A -- production feature inventory (derive, don't assume)
# =============================================================================

def load_deployed_artifacts() -> list[dict[str, Any]]:
    import joblib
    rows = []
    for entry in DEPLOYMENT_MANIFEST["artifacts"]:
        art = joblib.load(entry["artifact_path"])
        rows.append({
            "phase": entry["phase"], "endpoint": entry["endpoint"], "horizon": entry.get("horizon"),
            "kind": entry["kind"], "variant": entry["variant"], "features": list(art.features),
            "hyperparameter": entry["hyperparameter"], "artifact_path": entry["artifact_path"],
            "artifact_sha256": entry["artifact_sha256"],
        })
    return rows


def phase_a_feature_inventory(deployed: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for a in deployed:
        feats = a["features"]
        has = {sf: (sf in feats) for sf in SCORING_FEATURES_ALL}
        any_scoring = any(has.values())
        rows.append({
            "phase": a["phase"], "endpoint": a["endpoint"], "horizon": a["horizon"],
            "deployed_variant": a["variant"], "kind": a["kind"],
            "raw_feature_list": " | ".join(feats), "raw_feature_count": len(feats),
            **{f"has_{sf}": has[sf] for sf in SCORING_FEATURES_ALL},
            "uses_any_scoring_rate_feature": any_scoring,
            "has_canonical_position": "canonical_position" in feats,
            "has_canonical_sub_position": "canonical_sub_position" in feats,
            "has_position_scoring_interaction": False,   # deployed models: none do
            "scoring_rates_standardized_globally": any_scoring,  # fit_preprocessor
            # z-scores every numeric feature globally (see mixed_type_preprocessor.py
            # fit_preprocessor: means_all/scales_all computed over the whole encoded
            # matrix, not per group) -- true whenever a scoring feature is present.
            "scoring_rates_standardized_within_position": False,  # none deployed do
            "scoring_rates_shown_in_live_explanation_layer": (
                a["phase"] == "extension_opportunity_diagnostic" and a["endpoint"] == "sustained_meaningful_contribution"
            ) is False and any_scoring and a["endpoint"] in {"downside_25pct_24m", "any_outbound"},
            # ^ explain_headline_scores() (models/deployment_scorer.py) only builds
            # a per-feature logit breakdown for 3 cards: sustained_contribution
            # (C6, excludes scoring), downside_25pct_24m, and any_outbound@24m.
            # Every OTHER affected endpoint's scoring-rate coefficients exist in
            # the artifact but are never surfaced in any live "why" panel.
            "artifact_sha256": a["artifact_sha256"],
        })
    return pd.DataFrame(rows)


AFFECTED_ENDPOINTS: list[dict[str, Any]] = []  # populated in main() after inventory


# =============================================================================
# Role-relative feature engineering (training-only, leakage-safe)
# =============================================================================

def _position_series(frame: pd.DataFrame) -> pd.Series:
    pos = frame["canonical_position"].astype("string")
    # One known data-quality artifact: a single row reads
    # "Midfield - Central Midfield" instead of "Midfield" (see
    # role_coverage_audit.csv). Treated as Midfield (its unambiguous broad
    # group) rather than dropped or left to silently miss every role bucket.
    pos = pos.replace({"Midfield - Central Midfield": "Midfield"})
    return pos


def fit_role_reference(train: pd.DataFrame, scoring_features: list[str]) -> dict[str, Any]:
    """Training-rows-only reference distributions per (broad position, feature).
    Returns z-score stats (median/IQR) and percentile stats (sorted training
    values) for every scoring feature, plus a global training fallback used
    whenever a position group has < MIN_ROLE_TRAIN_N training rows."""
    pos = _position_series(train)
    ref: dict[str, Any] = {"per_position": {}, "global": {}, "fallback_log": []}
    for feat in scoring_features:
        vals = pd.to_numeric(train[feat], errors="coerce")
        finite = vals.dropna()
        g_median = float(finite.median()) if len(finite) else 0.0
        g_iqr = float(finite.quantile(.75) - finite.quantile(.25)) if len(finite) else 1.0
        g_iqr = g_iqr if g_iqr > 1e-9 else 1.0
        ref["global"][feat] = {
            "median": g_median, "iqr": g_iqr, "sorted": np.sort(finite.to_numpy()), "n": len(finite),
        }
        for group in BROAD_POSITIONS:
            mask = pos.eq(group)
            sub = vals[mask].dropna()
            key = (group, feat)
            if len(sub) >= MIN_ROLE_TRAIN_N:
                iqr = float(sub.quantile(.75) - sub.quantile(.25))
                iqr = iqr if iqr > 1e-9 else 1.0
                ref["per_position"][key] = {
                    "median": float(sub.median()), "iqr": iqr, "sorted": np.sort(sub.to_numpy()), "n": len(sub),
                }
            else:
                ref["per_position"][key] = None
                ref["fallback_log"].append({"position": group, "feature": feat, "train_n": int(len(sub))})
    return ref


def apply_role_relative(frame: pd.DataFrame, ref: dict[str, Any], scoring_features: list[str], method: str) -> pd.DataFrame:
    """method: 'zscore' or 'percentile'. NaNs pass through unchanged (the
    shared Preprocessor's own missing-indicator mechanism, unmodified,
    handles them downstream -- identical missingness policy to every
    deployed endpoint)."""
    pos = _position_series(frame)
    out = frame.copy()
    for feat in scoring_features:
        vals = pd.to_numeric(frame[feat], errors="coerce")
        result = np.full(len(frame), np.nan)
        for group in BROAD_POSITIONS:
            mask = pos.eq(group).to_numpy()
            if not mask.any():
                continue
            stats = ref["per_position"][(group, feat)] or ref["global"][feat]
            group_vals = vals.to_numpy()[mask]
            if method == "zscore":
                z = (group_vals - stats["median"]) / stats["iqr"]
                result[mask] = z
            else:
                sorted_train = stats["sorted"]
                if len(sorted_train) == 0:
                    result[mask] = np.nan
                else:
                    pct = np.searchsorted(sorted_train, group_vals, side="right") / len(sorted_train)
                    result[mask] = pct
        # rows with unmapped/missing position (NaN canonical_position) fall
        # back to the global training reference so they are not silently
        # dropped from the role-relative feature entirely.
        unmapped = ~pos.isin(BROAD_POSITIONS).to_numpy()
        if unmapped.any():
            stats = ref["global"][feat]
            group_vals = vals.to_numpy()[unmapped]
            if method == "zscore":
                result[unmapped] = (group_vals - stats["median"]) / stats["iqr"]
            else:
                sorted_train = stats["sorted"]
                result[unmapped] = (
                    np.searchsorted(sorted_train, group_vals, side="right") / len(sorted_train)
                    if len(sorted_train) else np.nan
                )
        suffix = "role_z" if method == "zscore" else "role_pct"
        out[f"{feat}__{suffix}"] = result
    return out


def add_interaction_features(frame: pd.DataFrame, scoring_features: list[str]) -> pd.DataFrame:
    """Raw scoring feature x broad-position indicator, Attack/Midfield/
    Defender only (Goalkeeper excluded per spec -- scoring values there are
    ~zero/near-invariant and would contribute an uninformative, near-
    collinear column)."""
    out = frame.copy()
    pos = _position_series(frame)
    for feat in scoring_features:
        vals = pd.to_numeric(frame[feat], errors="coerce")
        for group in INTERACTION_POSITIONS:
            indicator = pos.eq(group).astype(float)
            out[f"{feat}__x__{group}"] = vals * indicator
    return out


def candidate_feature_lists(deployed_features: list[str], role_method: str) -> dict[str, list[str]]:
    present_scoring = [f for f in SCORING_FEATURES_ALL if f in deployed_features]
    non_scoring = [f for f in deployed_features if f not in present_scoring]
    role_cols = [f"{f}__{'role_z' if role_method == 'zscore' else 'role_pct'}" for f in present_scoring]
    interaction_cols = [f"{f}__x__{g}" for f in present_scoring for g in INTERACTION_POSITIONS]
    return {
        "R0_deployed_reproduction": list(deployed_features),
        "R1_remove_scoring": non_scoring,
        "R2_position_relative": non_scoring + role_cols,
        "R3_position_interactions": non_scoring + present_scoring + interaction_cols,
        # R4 is finalized per-origin after seeing which of R2/R3 validates
        # better (see build_r4_for_origin below) -- placeholder here.
    }


# =============================================================================
# Endpoint registry -- built from the Phase A inventory, not assumed.
# =============================================================================

def build_endpoint_registry(deployed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reg = []
    for a in deployed:
        feats = a["features"]
        if not any(sf in feats for sf in SCORING_FEATURES_ALL):
            continue  # out of scope: no scoring-rate feature present
        endpoint, phase, horizon = a["endpoint"], a["phase"], a["horizon"]
        if phase == "extension_opportunity_diagnostic":
            entry = {
                "family": "opportunity", "phase": phase, "endpoint": endpoint, "horizon": None,
                "kind": "continuous", "deployed_features": feats,
                "target": opp.ENDPOINTS[endpoint]["target"],
                "cohort_fn": lambda frame, ep=endpoint: opp.endpoint_frame(frame, ep),
            }
        elif phase == "extension_survival_diagnostic":
            entry = {
                "family": "survival", "phase": phase, "endpoint": endpoint, "horizon": horizon,
                "kind": "hazard", "deployed_features": feats, "target": None, "cohort_fn": None,
            }
        elif phase == "extension_value_preservation_diagnostic":
            entry = {
                "family": "value_preservation", "phase": phase, "endpoint": endpoint, "horizon": horizon,
                "kind": a["kind"], "deployed_features": feats,
                "target": val.ENDPOINTS[endpoint]["target"],
                "cohort_fn": lambda frame, ep=endpoint: val.endpoint_frame(frame, ep),
            }
        elif phase == "contract_financial_exposure_benchmark":
            entry = {
                "family": "financial_exposure", "phase": phase, "endpoint": endpoint, "horizon": None,
                "kind": "continuous", "deployed_features": feats,
                "target": fin.ENDPOINTS[endpoint]["target"],
                "cohort_col": fin.ENDPOINTS[endpoint]["cohort"],
                "deployed_variant": a["variant"],  # B5_compact_nonlinear -> GBM family
            }
        else:
            raise RuntimeError(f"Unhandled phase in registry build: {phase}")
        entry["deployed_variant"] = entry.get("deployed_variant", a["variant"])
        entry["deployed_hyperparameter"] = a["hyperparameter"]
        reg.append(entry)
    # Collapse the two any_outbound horizon rows into one hazard entry (a
    # single fit_survival_model() call produces both 24m and 36m from the
    # same cumulative hazard curve).
    collapsed, seen_hazard = [], set()
    for entry in reg:
        if entry["family"] == "survival":
            if entry["endpoint"] in seen_hazard:
                continue
            seen_hazard.add(entry["endpoint"])
        collapsed.append(entry)
    return collapsed


# =============================================================================
# PHASE D/E -- candidate construction + rolling-origin, leakage-safe fitting
# =============================================================================

def continuous_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    corr = pd.Series(actual).corr(pd.Series(prediction), method="spearman")
    return {
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
        "spearman": float(corr) if pd.notna(corr) else np.nan,
    }


def binary_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    result = {
        "brier": float(brier_score_loss(actual, prediction)),
        "log_loss": float(log_loss(actual, prediction, labels=[0, 1])),
        "actual_rate": float(np.mean(actual)), "predicted_rate": float(np.mean(prediction)),
    }
    if len(np.unique(actual)) == 2:
        result["roc_auc"] = float(roc_auc_score(actual, prediction))
        result["average_precision"] = float(average_precision_score(actual, prediction))
    else:
        result["roc_auc"], result["average_precision"] = np.nan, np.nan
    return result


def calibration_error(actual: np.ndarray, prediction: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(prediction, bins) - 1, 0, n_bins - 1)
    errs, weights = [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        errs.append(abs(actual[mask].mean() - prediction[mask].mean()))
        weights.append(mask.sum())
    return float(np.average(errs, weights=weights)) if errs else np.nan


def fit_and_predict(
    train: pd.DataFrame, validation: pd.DataFrame, evaluation: pd.DataFrame,
    features: list[str], target: str, kind: str, seed_tag: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Ridge/LogisticRegression tune (train->validation) then refit on
    train+validation, predict on evaluation. Mirrors run_extension_
    opportunity_diagnostic.tune_predict's exact two-stage procedure (same
    hyperparameter grids, same RANDOM_SEED) so R0 here is provably the same
    fitting methodology used to build the deployed artifacts -- written
    locally (rather than calling tune_predict directly) only because this
    diagnostic also needs the fitted (preprocessor, model) object itself for
    later demo-player scoring, which tune_predict does not return."""
    y_train = pd.to_numeric(train[target]).to_numpy()
    y_validation = pd.to_numeric(validation[target]).to_numpy()
    if not features:
        anchor = float(np.mean(y_train)) if kind == "continuous" else float(np.mean(y_train))
        fit_mean = float(pd.concat([train, validation])[target].mean())
        prediction = np.repeat(fit_mean, len(evaluation))
        return prediction, {"selected_parameter": None, "encoded_feature_count": 0, "preprocessor": None, "model": None}

    pre = fit_preprocessor(train, features)
    x_train, x_validation = pre.transform(train), pre.transform(validation)
    candidates = []
    if kind == "continuous":
        for alpha in RIDGE_ALPHAS:
            model = Ridge(alpha=alpha).fit(x_train, y_train)
            pred = np.clip(model.predict(x_validation), 0, 1.05) if target == "target_year2_opportunity_share" else model.predict(x_validation)
            candidates.append(((mean_absolute_error(y_validation, pred),), {"alpha": alpha}))
    else:
        y_train_int = y_train.astype(int)
        for c_value in LOGISTIC_C:
            model = LogisticRegression(C=c_value, penalty="l2", solver="liblinear", max_iter=3000, random_state=RANDOM_SEED).fit(x_train, y_train_int)
            pred = np.clip(model.predict_proba(x_validation)[:, 1], 1e-6, 1 - 1e-6)
            candidates.append(((brier_score_loss(y_validation.astype(int), pred),), {"C": c_value}))
    _, params = min(candidates, key=lambda item: item[0])

    fit = pd.concat([train, validation], ignore_index=True)
    fit_pre = fit_preprocessor(fit, features)
    x_fit, x_eval = fit_pre.transform(fit), fit_pre.transform(evaluation)
    y_fit = pd.to_numeric(fit[target]).to_numpy()
    if kind == "continuous":
        model = Ridge(alpha=params["alpha"]).fit(x_fit, y_fit)
        prediction = model.predict(x_eval)
        if target == "target_year2_opportunity_share":
            prediction = np.clip(prediction, 0, 1.05)
    else:
        model = LogisticRegression(C=params["C"], penalty="l2", solver="liblinear", max_iter=3000, random_state=RANDOM_SEED).fit(x_fit, y_fit.astype(int))
        prediction = np.clip(model.predict_proba(x_eval)[:, 1], 1e-6, 1 - 1e-6)
    return prediction, {
        "selected_parameter": json.dumps(params, sort_keys=True), "encoded_feature_count": x_fit.shape[1],
        "preprocessor": fit_pre, "model": model,
    }


def fit_and_predict_gbm(
    train: pd.DataFrame, validation: pd.DataFrame, evaluation: pd.DataFrame,
    features: list[str], target: str, seed_tag: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """GradientBoostingRegressor variant, used only for contract_duration
    (the sole nonlinear/B5_compact_nonlinear endpoint in the deployed
    engine). Grid matches run_contract_financial_exposure_benchmark.py's
    own B5 grid exactly."""
    grid = [
        {"n_estimators": 75, "learning_rate": .03, "max_depth": 1, "min_samples_leaf": 20},
        {"n_estimators": 100, "learning_rate": .03, "max_depth": 2, "min_samples_leaf": 25},
        {"n_estimators": 75, "learning_rate": .05, "max_depth": 2, "min_samples_leaf": 30},
    ]
    y_train = pd.to_numeric(train[target]).to_numpy()
    y_validation = pd.to_numeric(validation[target]).to_numpy()
    pre = fit_preprocessor(train, features)
    x_train, x_validation = pre.transform(train), pre.transform(validation)
    candidates = []
    for i, params in enumerate(grid):
        seed = (RANDOM_SEED + int(hashlib.sha256(f"{seed_tag}|{i}".encode()).hexdigest()[:8], 16)) % (2**32 - 1)
        model = GradientBoostingRegressor(loss="huber", random_state=seed, **params).fit(x_train, y_train)
        pred = model.predict(x_validation)
        candidates.append(((mean_absolute_error(y_validation, pred),), params))
    _, params = min(candidates, key=lambda item: item[0])
    fit = pd.concat([train, validation], ignore_index=True)
    fit_pre = fit_preprocessor(fit, features)
    x_fit, x_eval = fit_pre.transform(fit), fit_pre.transform(evaluation)
    y_fit = pd.to_numeric(fit[target]).to_numpy()
    seed = (RANDOM_SEED + int(hashlib.sha256(f"{seed_tag}|fit".encode()).hexdigest()[:8], 16)) % (2**32 - 1)
    model = GradientBoostingRegressor(loss="huber", random_state=seed, **params).fit(x_fit, y_fit)
    prediction = model.predict(x_eval)
    return prediction, {
        "selected_parameter": json.dumps(params, sort_keys=True), "encoded_feature_count": x_fit.shape[1],
        "preprocessor": fit_pre, "model": model,
    }


def fit_survival_candidate(
    train: pd.DataFrame, validation: pd.DataFrame, evaluation: pd.DataFrame,
    endpoint: str, features: list[str], variant_tag: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Discrete-time hazard fit for one candidate feature list, reusing
    models/run_extension_survival_diagnostic.py's own person_periods/
    fit_survival_model/predict_hazard_curve unmodified. surv.FEATURE_SETS is
    a plain dict COPY made at survival-module-import time (`dict(phase1.
    FEATURE_SETS)`), never written back to any file -- registering this
    diagnostic's own candidate feature lists into that in-memory copy under
    a unique key lets person_periods() carry the new role-relative/
    interaction columns through its person-period expansion (which only
    keeps columns appearing somewhere in FEATURE_SETS) without touching
    run_extension_survival_diagnostic.py on disk or any other variant
    already registered there."""
    surv.FEATURE_SETS[variant_tag] = features
    try:
        cumulative, meta, encoded, fit_n = surv.fit_survival_model(train, validation, evaluation, endpoint, variant_tag, features)
    finally:
        del surv.FEATURE_SETS[variant_tag]
    return cumulative, {**meta, "encoded_feature_count": encoded, "fit_person_periods": fit_n}


def select_role_method(
    train: pd.DataFrame, validation: pd.DataFrame, non_scoring: list[str], scoring: list[str],
    target: str, kind: str, family: str, endpoint: str, ref: dict[str, Any],
) -> str:
    """Fits a quick train-only-referenced R2 candidate both ways and picks
    whichever validates better on THIS origin's validation year -- selection
    itself never touches the evaluation (test) year."""
    if not scoring:
        return "zscore"
    scored_train_z = apply_role_relative(train, ref, scoring, "zscore")
    scored_val_z = apply_role_relative(validation, ref, scoring, "zscore")
    scored_train_p = apply_role_relative(train, ref, scoring, "percentile")
    scored_val_p = apply_role_relative(validation, ref, scoring, "percentile")
    role_cols_z = [f"{f}__role_z" for f in scoring]
    role_cols_p = [f"{f}__role_pct" for f in scoring]
    feats_z, feats_p = non_scoring + role_cols_z, non_scoring + role_cols_p

    def quick_score(tr, va, feats):
        pre = fit_preprocessor(tr, feats)
        x_tr, x_va = pre.transform(tr), pre.transform(va)
        y_tr = pd.to_numeric(tr[target]).to_numpy()
        y_va = pd.to_numeric(va[target]).to_numpy()
        if kind == "continuous":
            model = Ridge(alpha=1.0).fit(x_tr, y_tr)
            pred = model.predict(x_va)
            return mean_absolute_error(y_va, pred)
        model = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=3000, random_state=RANDOM_SEED).fit(x_tr, y_tr.astype(int))
        pred = np.clip(model.predict_proba(x_va)[:, 1], 1e-6, 1 - 1e-6)
        return brier_score_loss(y_va.astype(int), pred)

    loss_z = quick_score(scored_train_z, scored_val_z, feats_z)
    loss_p = quick_score(scored_train_p, scored_val_p, feats_p)
    return "zscore" if loss_z <= loss_p else "percentile"


def select_r4_representation(
    train: pd.DataFrame, validation: pd.DataFrame, non_scoring: list[str], scoring: list[str],
    role_cols: list[str], interaction_cols: list[str], target: str, kind: str,
) -> str:
    """Chooses R2's role-relative scoring block or R3's interaction block
    for R4, using ONLY this origin's train/validation split."""
    if not scoring:
        return "role_relative"

    def quick_score(feats):
        pre = fit_preprocessor(train, feats)
        x_tr, x_va = pre.transform(train), pre.transform(validation)
        y_tr = pd.to_numeric(train[target]).to_numpy()
        y_va = pd.to_numeric(validation[target]).to_numpy()
        if kind == "continuous":
            model = Ridge(alpha=1.0).fit(x_tr, y_tr)
            return mean_absolute_error(y_va, model.predict(x_va))
        model = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=3000, random_state=RANDOM_SEED).fit(x_tr, y_tr.astype(int))
        return brier_score_loss(y_va.astype(int), np.clip(model.predict_proba(x_va)[:, 1], 1e-6, 1 - 1e-6))

    loss_role = quick_score(non_scoring + role_cols)
    loss_interaction = quick_score(non_scoring + scoring + interaction_cols)
    return "role_relative" if loss_role <= loss_interaction else "interactions"


# =============================================================================
# Main origin backtest: builds R0-R4 for every affected endpoint x origin.
# =============================================================================

def load_family_frames() -> dict[str, pd.DataFrame]:
    return {
        "opportunity": opp.load_frame(),
        "survival": surv.load_frame(),
        "value_preservation": val.load_frame(),
        "financial_exposure": fin.load_frame(),
    }


def run_origin_backtest(registry: list[dict[str, Any]], frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []
    r4_choice_rows: list[dict[str, Any]] = []
    role_method_rows: list[dict[str, Any]] = []

    for entry in registry:
        endpoint, family = entry["endpoint"], entry["family"]
        deployed_features = entry["deployed_features"]
        scoring = [f for f in SCORING_FEATURES_ALL if f in deployed_features]
        non_scoring = [f for f in deployed_features if f not in scoring]
        is_hazard = family == "survival"
        print(f"-- {family}/{endpoint} -- scoring features: {scoring}")

        for origin_id, train_end, validation_year, evaluation_year in ORIGINS:
            if is_hazard:
                base = frames["survival"].loc[frames["survival"]["survival_primary_cohort"]].copy()
            elif family == "opportunity":
                base = entry["cohort_fn"](frames["opportunity"])
            elif family == "value_preservation":
                base = entry["cohort_fn"](frames["value_preservation"])
            elif family == "financial_exposure":
                fr = frames["financial_exposure"]
                base = fr.loc[fr[entry["cohort_col"]] & fr[entry["target"]].notna()].copy()
            else:
                raise RuntimeError(family)

            train = base.loc[base["signed_year"].le(train_end)].copy()
            validation = base.loc[base["signed_year"].eq(validation_year)].copy()
            evaluation = base.loc[base["signed_year"].eq(evaluation_year)].copy()
            if min(len(train), len(validation), len(evaluation)) < 30:
                print(f"   skip {origin_id}: insufficient rows ({len(train)}/{len(validation)}/{len(evaluation)})")
                continue

            ref = fit_role_reference(train, scoring) if scoring else None
            for fb in (ref["fallback_log"] if ref else []):
                fallback_rows.append({"endpoint": endpoint, "origin_id": origin_id, **fb})

            role_method = select_role_method(
                train, validation, non_scoring, scoring,
                entry["target"] if not is_hazard else f"{endpoint}_event_by_24m",
                "continuous" if entry["kind"] == "continuous" else "binary",
                family, endpoint, ref,
            ) if scoring else "zscore"
            role_method_rows.append({"endpoint": endpoint, "origin_id": origin_id, "selected_role_method": role_method})

            def augmented(frame: pd.DataFrame) -> pd.DataFrame:
                if not scoring:
                    return frame
                out = apply_role_relative(frame, ref, scoring, role_method)
                out = add_interaction_features(out, scoring)
                return out

            train_a, validation_a, evaluation_a = augmented(train), augmented(validation), augmented(evaluation)
            role_cols = [f"{f}__{'role_z' if role_method == 'zscore' else 'role_pct'}" for f in scoring]
            interaction_cols = [f"{f}__x__{g}" for f in scoring for g in INTERACTION_POSITIONS]

            r4_rep = select_r4_representation(
                train_a, validation_a, non_scoring, scoring, role_cols, interaction_cols,
                entry["target"] if not is_hazard else f"{endpoint}_event_by_24m",
                "continuous" if entry["kind"] == "continuous" else "binary",
            ) if scoring else "role_relative"
            r4_choice_rows.append({"endpoint": endpoint, "origin_id": origin_id, "r4_representation": r4_rep})

            candidates: dict[str, list[str]] = {
                "R0_deployed_reproduction": list(deployed_features),
                "R1_remove_scoring": non_scoring,
                "R2_position_relative": non_scoring + role_cols,
                "R3_position_interactions": non_scoring + scoring + interaction_cols,
                "R4_compact_role_aware": non_scoring + (role_cols if r4_rep == "role_relative" else scoring + interaction_cols),
            }
            for cand_name, feats in candidates.items():
                manifest_rows.append({
                    "endpoint": endpoint, "origin_id": origin_id, "candidate": cand_name,
                    "feature_count": len(feats), "features": " | ".join(feats),
                    "role_method_used": role_method if "role" in cand_name.lower() or cand_name == "R4_compact_role_aware" else None,
                })

            if is_hazard:
                for cand_name, feats in candidates.items():
                    variant_tag = f"RC_{endpoint}_{cand_name}_{origin_id}"
                    cumulative, meta = fit_survival_candidate(train_a, validation_a, evaluation_a, endpoint, feats, variant_tag)
                    for label, horizon_days in surv.HORIZONS.items():
                        if label not in ("24m", "36m"):
                            continue
                        col = surv.horizon_column_index(horizon_days)
                        actual_col = f"{endpoint}_event_by_{label}"
                        if actual_col not in evaluation_a.columns or not surv.horizon_is_complete(evaluation_a, horizon_days):
                            continue
                        actual = evaluation_a[actual_col].to_numpy(int)
                        prediction = cumulative[:, col]
                        metrics = binary_metrics(actual, prediction)
                        metrics["calibration_error"] = calibration_error(actual, prediction)
                        metric_rows.append({
                            "endpoint": f"{endpoint}_{label}", "kind": "hazard", "origin_id": origin_id,
                            "candidate": cand_name, "train_end_year": train_end, "validation_year": validation_year,
                            "evaluation_year": evaluation_year, "train_rows": len(train), "validation_rows": len(validation),
                            "evaluation_rows": len(evaluation), "selected_parameter": meta.get("selected_parameter"),
                            "encoded_feature_count": meta.get("encoded_feature_count"),
                            **{f"evaluation_{k}": v for k, v in metrics.items()},
                        })
                        for i, (_, row) in enumerate(evaluation_a.reset_index(drop=True).iterrows()):
                            prediction_rows.append({
                                "endpoint": f"{endpoint}_{label}", "kind": "hazard", "origin_id": origin_id,
                                "evaluation_year": evaluation_year, "canonical_player_id": row["canonical_player_id"],
                                "canonical_position": row.get("canonical_position"), "age_band": row.get("age_band"),
                                "league": row.get("league"), "market_value_band": row.get("market_value_band"),
                                "candidate": cand_name, "actual": float(actual[i]), "prediction": float(prediction[i]),
                                "primary_loss": (float(actual[i]) - float(prediction[i])) ** 2,
                            })
            else:
                kind = entry["kind"]
                target = entry["target"]
                for cand_name, feats in candidates.items():
                    seed_tag = f"{endpoint}|{cand_name}|{origin_id}"
                    if entry.get("deployed_variant") == "B5_compact_nonlinear":
                        prediction, meta = fit_and_predict_gbm(train_a, validation_a, evaluation_a, feats, target, seed_tag)
                    else:
                        prediction, meta = fit_and_predict(train_a, validation_a, evaluation_a, feats, target, kind, seed_tag)
                    actual = pd.to_numeric(evaluation_a[target]).to_numpy()
                    metrics = continuous_metrics(actual, prediction) if kind == "continuous" else binary_metrics(actual.astype(int), prediction)
                    if kind == "binary":
                        metrics["calibration_error"] = calibration_error(actual.astype(int), prediction)
                    metric_rows.append({
                        "endpoint": endpoint, "kind": kind, "origin_id": origin_id, "candidate": cand_name,
                        "train_end_year": train_end, "validation_year": validation_year, "evaluation_year": evaluation_year,
                        "train_rows": len(train), "validation_rows": len(validation), "evaluation_rows": len(evaluation),
                        "selected_parameter": meta.get("selected_parameter"), "encoded_feature_count": meta.get("encoded_feature_count"),
                        **{f"evaluation_{k}": v for k, v in metrics.items()},
                    })
                    for i, (_, row) in enumerate(evaluation_a.reset_index(drop=True).iterrows()):
                        prediction_rows.append({
                            "endpoint": endpoint, "kind": kind, "origin_id": origin_id, "evaluation_year": evaluation_year,
                            "canonical_player_id": row.get("canonical_player_id"), "canonical_position": row.get("canonical_position"),
                            "age_band": row.get("age_band"), "league": row.get("league"),
                            "market_value_band": row.get("market_value_band"), "candidate": cand_name,
                            "actual": float(actual[i]), "prediction": float(prediction[i]),
                            "primary_loss": abs(float(actual[i]) - float(prediction[i])) if kind == "continuous" else (float(actual[i]) - float(prediction[i])) ** 2,
                        })

    return {
        "predictions": pd.DataFrame(prediction_rows),
        "origin_metrics": pd.DataFrame(metric_rows),
        "candidate_manifest": pd.DataFrame(manifest_rows),
        "fallback_log": pd.DataFrame(fallback_rows),
        "r4_choices": pd.DataFrame(r4_choice_rows),
        "role_method_choices": pd.DataFrame(role_method_rows),
    }


# =============================================================================
# Pooled, player-clustered-bootstrap comparison of each candidate vs R0.
# =============================================================================

def cluster_bootstrap(pairs: pd.DataFrame, seed: int) -> dict[str, float]:
    """Identical block-draw player-clustered bootstrap already used by
    run_extension_opportunity_diagnostic.cluster_bootstrap and its siblings
    -- same BOOTSTRAP_REPETITIONS (5000) and draw-in-blocks-of-250 pattern,
    reimplemented locally only because it needs to run against this
    script's own paired frame shape."""
    pair = pairs.copy()
    pair["improvement"] = pair["baseline_loss"] - pair["candidate_loss"]
    clusters = pair.groupby("canonical_player_id")["improvement"].agg(["sum", "size"])
    sums, sizes = clusters["sum"].to_numpy(float), clusters["size"].to_numpy(float)
    rng, draws = np.random.default_rng(seed), []
    for start in range(0, BOOTSTRAP_REPETITIONS, 250):
        n = min(250, BOOTSTRAP_REPETITIONS - start)
        idx = rng.integers(0, len(clusters), size=(n, len(clusters)))
        draws.extend((sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)).tolist())
    return {
        "cluster_count": int(len(clusters)),
        "ci_lower_95": float(np.quantile(draws, 0.025)),
        "ci_upper_95": float(np.quantile(draws, 0.975)),
        "probability_candidate_better": float(np.mean(np.asarray(draws) > 0)),
    }


def pooled_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint in predictions["endpoint"].unique():
        for window_name, years in (("pooled_four", [2020, 2021, 2022, 2023]), ("terminal_two", [2022, 2023])):
            base = predictions.loc[predictions["endpoint"].eq(endpoint) & predictions["evaluation_year"].isin(years)]
            baseline = base.loc[base["candidate"].eq("R0_deployed_reproduction"), ["origin_id", "evaluation_year", "canonical_player_id", "actual", "prediction", "primary_loss"]].rename(columns={"prediction": "baseline_pred", "primary_loss": "baseline_loss"})
            for candidate in ["R1_remove_scoring", "R2_position_relative", "R3_position_interactions", "R4_compact_role_aware"]:
                cand = base.loc[base["candidate"].eq(candidate), ["origin_id", "evaluation_year", "canonical_player_id", "prediction", "primary_loss"]].rename(columns={"prediction": "candidate_pred", "primary_loss": "candidate_loss"})
                pairs = baseline.merge(cand, on=["origin_id", "evaluation_year", "canonical_player_id"], how="inner")
                if pairs.empty:
                    continue
                origin_means = pairs.groupby("origin_id")[["candidate_loss", "baseline_loss"]].mean()
                origin_wins = (origin_means["candidate_loss"] < origin_means["baseline_loss"]).sum()
                n_origins = pairs["origin_id"].nunique()
                seed_key = f"{endpoint}|{window_name}|{candidate}"
                seed = (RANDOM_SEED + int(hashlib.sha256(seed_key.encode()).hexdigest()[:8], 16)) % (2**32 - 1)
                boot = cluster_bootstrap(pairs, seed)
                rows.append({
                    "endpoint": endpoint, "window": window_name, "candidate": candidate,
                    "n_rows": len(pairs), "n_players_clustered": boot["cluster_count"],
                    "pooled_baseline_loss": float(pairs["baseline_loss"].mean()),
                    "pooled_candidate_loss": float(pairs["candidate_loss"].mean()),
                    "pooled_loss_improvement": float(pairs["baseline_loss"].mean() - pairs["candidate_loss"].mean()),
                    "origin_wins": int(origin_wins), "n_origins": int(n_origins),
                    "bootstrap_ci_lower_95": boot["ci_lower_95"], "bootstrap_ci_upper_95": boot["ci_upper_95"],
                    "bootstrap_probability_candidate_better": boot["probability_candidate_better"],
                    "ci_excludes_zero_favoring_candidate": boot["ci_lower_95"] > 0,
                })
    return pd.DataFrame(rows)


def subgroup_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint in predictions["endpoint"].unique():
        base = predictions.loc[predictions["endpoint"].eq(endpoint) & predictions["evaluation_year"].isin([2020, 2021, 2022, 2023])]
        for candidate in base["candidate"].unique():
            cand = base.loc[base["candidate"].eq(candidate)]
            is_binary = set(pd.unique(cand["actual"])) <= {0.0, 1.0}
            for group_col in ["canonical_position", "age_band", "league", "market_value_band"]:
                if group_col not in cand.columns:
                    continue
                for group_value, g in cand.groupby(group_col, dropna=True):
                    if len(g) < 15:
                        continue
                    actual, pred = g["actual"].to_numpy(), g["prediction"].to_numpy()
                    m = binary_metrics(actual.astype(int), pred) if is_binary else continuous_metrics(actual, pred)
                    rows.append({
                        "endpoint": endpoint, "candidate": candidate, "group_type": group_col,
                        "group_value": str(group_value), "n_rows": len(g), **m,
                    })
    return pd.DataFrame(rows)


def calibration_audit(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint in predictions["endpoint"].unique():
        base = predictions.loc[predictions["endpoint"].eq(endpoint) & predictions["evaluation_year"].isin([2020, 2021, 2022, 2023])]
        is_binary = set(pd.unique(base["actual"])) <= {0.0, 1.0}
        if not is_binary:
            continue
        for candidate in base["candidate"].unique():
            cand = base.loc[base["candidate"].eq(candidate)]
            rows.append({
                "endpoint": endpoint, "candidate": candidate, "n_rows": len(cand),
                "calibration_error_10bin": calibration_error(cand["actual"].to_numpy(int), cand["prediction"].to_numpy()),
                "brier": float(brier_score_loss(cand["actual"].astype(int), cand["prediction"])),
            })
    return pd.DataFrame(rows)


# =============================================================================
# PHASE C -- position / role data audit
# =============================================================================

def phase_c_audits(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    frame = frames["opportunity"]  # canonical_position/sub_position are the same columns everywhere
    pos = _position_series(frame)

    coverage_rows = []
    for group in BROAD_POSITIONS + ["__unmapped__"]:
        mask = pos.eq(group) if group != "__unmapped__" else ~pos.isin(BROAD_POSITIONS)
        sub = frame.loc[mask]
        for feat in SCORING_FEATURES_ALL:
            vals = pd.to_numeric(sub[feat], errors="coerce")
            coverage_rows.append({
                "broad_position": group, "feature": feat, "n_rows": len(sub),
                "n_non_missing": int(vals.notna().sum()), "missing_rate": float(vals.isna().mean()) if len(sub) else np.nan,
            })
    coverage = pd.DataFrame(coverage_rows)

    dist_rows = []
    for group in BROAD_POSITIONS:
        sub = frame.loc[pos.eq(group)]
        for feat in SCORING_FEATURES_ALL:
            vals = pd.to_numeric(sub[feat], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            dist_rows.append({
                "broad_position": group, "feature": feat, "n": len(vals),
                "mean": float(vals.mean()), "median": float(vals.median()), "std": float(vals.std()),
                "p10": float(vals.quantile(.10)), "p25": float(vals.quantile(.25)),
                "p75": float(vals.quantile(.75)), "p90": float(vals.quantile(.90)), "p99": float(vals.quantile(.99)),
                "min": float(vals.min()), "max": float(vals.max()),
                "zero_rate": float((vals == 0).mean()),
            })
    distribution = pd.DataFrame(dist_rows)

    # canonical_sub_position audit -- reported, not adopted as the primary
    # grouping (see module docstring / README for the coverage rationale).
    sub_pos = frame["canonical_sub_position"].astype("string")
    sub_rows = []
    for value, count in sub_pos.value_counts(dropna=False).items():
        sub_rows.append({"canonical_sub_position": str(value), "n_rows": int(count)})
    sub_position_counts = pd.DataFrame(sub_rows)

    stability_rows = []
    for origin_id, train_end, validation_year, evaluation_year in ORIGINS:
        train = frame.loc[frame["signed_year"].le(train_end)]
        train_pos = _position_series(train)
        for group in BROAD_POSITIONS:
            n = int(train_pos.eq(group).sum())
            stability_rows.append({
                "origin_id": origin_id, "train_end_year": train_end, "broad_position": group,
                "train_n": n, "meets_min_role_n": n >= MIN_ROLE_TRAIN_N,
                "share_of_train": float(n / len(train)) if len(train) else np.nan,
            })
    origin_stability = pd.DataFrame(stability_rows)

    return {
        "coverage": coverage, "distribution": distribution,
        "sub_position_counts": sub_position_counts, "origin_stability": origin_stability,
    }


# =============================================================================
# "Deployment-equivalent" final candidate fits (train<=2022 + validation==
# 2023 combined) -- used for the live-production role audit (Phase B, R0
# side only, compared against the ACTUAL deployed artifact) and the demo-
# player comparison (Phase G, all of R0-R4). These are NEW, clearly-labeled
# research fits saved only under candidate_artifacts/, never written to
# Data/processed/deployment_models/.
# =============================================================================

def fit_final_hazard_candidate(cohort: pd.DataFrame, endpoint: str, features: list[str], variant_tag: str) -> dict[str, Any]:
    train = cohort.loc[cohort["signed_year"].le(DEPLOYMENT_TRAIN_THROUGH_YEAR)].copy()
    validation = cohort.loc[cohort["signed_year"].eq(DEPLOYMENT_VALIDATION_YEAR)].copy()
    fit_subjects = pd.concat([train, validation], ignore_index=True)
    surv.FEATURE_SETS[variant_tag] = features
    try:
        fit_features = features + ["hazard_interval"]
        train_periods = surv.person_periods(train, endpoint)
        pre = fit_preprocessor(train_periods, fit_features)
        x_train = pre.transform(train_periods)
        y_train = train_periods["hazard_event"].to_numpy(int)
        candidates = []
        for c_value in surv.C_GRID:
            model = LogisticRegression(C=c_value, penalty="l2", solver="liblinear", max_iter=3000, random_state=RANDOM_SEED).fit(x_train, y_train)
            cumulative = surv.predict_hazard_curve(model, pre, validation, features)
            candidates.append((surv.validation_integrated_brier(validation, endpoint, cumulative), {"C": c_value}))
        _, params = min(candidates, key=lambda item: item[0])
        fit_periods = surv.person_periods(fit_subjects, endpoint)
        fit_pre = fit_preprocessor(fit_periods, fit_features)
        model = LogisticRegression(C=params["C"], penalty="l2", solver="liblinear", max_iter=3000, random_state=RANDOM_SEED)
        model.fit(fit_pre.transform(fit_periods), fit_periods["hazard_event"].to_numpy(int))
    finally:
        del surv.FEATURE_SETS[variant_tag]
    return {"kind": "hazard", "preprocessor": fit_pre, "model": model, "features": features, "selected_parameter": params}


def build_final_candidates(registry: list[dict[str, Any]], frames: dict[str, pd.DataFrame]) -> dict[tuple[str, str], dict[str, Any]]:
    """Returns {(endpoint, candidate_name): fitted-candidate-dict}."""
    final: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in registry:
        endpoint, family = entry["endpoint"], entry["family"]
        deployed_features = entry["deployed_features"]
        scoring = [f for f in SCORING_FEATURES_ALL if f in deployed_features]
        non_scoring = [f for f in deployed_features if f not in scoring]
        is_hazard = family == "survival"

        if is_hazard:
            cohort = frames["survival"].loc[frames["survival"]["survival_primary_cohort"]].copy()
        elif family == "opportunity":
            cohort = entry["cohort_fn"](frames["opportunity"])
        elif family == "value_preservation":
            cohort = entry["cohort_fn"](frames["value_preservation"])
        else:
            fr = frames["financial_exposure"]
            cohort = fr.loc[fr[entry["cohort_col"]] & fr[entry["target"]].notna()].copy()

        train = cohort.loc[cohort["signed_year"].le(DEPLOYMENT_TRAIN_THROUGH_YEAR)].copy()
        validation = cohort.loc[cohort["signed_year"].eq(DEPLOYMENT_VALIDATION_YEAR)].copy()
        ref = fit_role_reference(pd.concat([train, validation], ignore_index=True), scoring) if scoring else None
        role_method = "zscore"
        if scoring:
            role_method = select_role_method(
                train, validation, non_scoring, scoring,
                entry["target"] if not is_hazard else f"{endpoint}_event_by_24m",
                "continuous" if entry["kind"] == "continuous" else "binary", family, endpoint, ref,
            )

        def augmented(frame: pd.DataFrame) -> pd.DataFrame:
            if not scoring:
                return frame
            out = apply_role_relative(frame, ref, scoring, role_method)
            return add_interaction_features(out, scoring)

        role_cols = [f"{f}__{'role_z' if role_method == 'zscore' else 'role_pct'}" for f in scoring]
        interaction_cols = [f"{f}__x__{g}" for f in scoring for g in INTERACTION_POSITIONS]
        train_a, validation_a = augmented(train), augmented(validation)
        r4_rep = select_r4_representation(
            train_a, validation_a, non_scoring, scoring, role_cols, interaction_cols,
            entry["target"] if not is_hazard else f"{endpoint}_event_by_24m",
            "continuous" if entry["kind"] == "continuous" else "binary",
        ) if scoring else "role_relative"

        candidates = {
            "R0_deployed_reproduction": list(deployed_features),
            "R1_remove_scoring": non_scoring,
            "R2_position_relative": non_scoring + role_cols,
            "R3_position_interactions": non_scoring + scoring + interaction_cols,
            "R4_compact_role_aware": non_scoring + (role_cols if r4_rep == "role_relative" else scoring + interaction_cols),
        }

        for cand_name, feats in candidates.items():
            if is_hazard:
                variant_tag = f"RCfinal_{endpoint}_{cand_name}"
                cohort_a = augmented(cohort)
                result = fit_final_hazard_candidate(cohort_a, endpoint, feats, variant_tag)
            else:
                fit_all = pd.concat([train_a, validation_a], ignore_index=True)
                target, kind = entry["target"], entry["kind"]
                y_fit = pd.to_numeric(fit_all[target]).to_numpy()
                if entry.get("deployed_variant") == "B5_compact_nonlinear":
                    pre = fit_preprocessor(fit_all, feats)
                    seed = (RANDOM_SEED + int(hashlib.sha256(f"final|{endpoint}|{cand_name}".encode()).hexdigest()[:8], 16)) % (2**32 - 1)
                    model = GradientBoostingRegressor(loss="huber", random_state=seed, n_estimators=100, learning_rate=.03, max_depth=2, min_samples_leaf=25).fit(pre.transform(fit_all), y_fit)
                else:
                    pre = fit_preprocessor(fit_all, feats)
                    x_fit = pre.transform(fit_all)
                    if kind == "continuous":
                        model = Ridge(alpha=1.0).fit(x_fit, y_fit)
                    else:
                        model = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=3000, random_state=RANDOM_SEED).fit(x_fit, y_fit.astype(int))
                result = {"kind": kind, "preprocessor": pre, "model": model, "features": feats}
            result.update({
                "endpoint": endpoint, "candidate": cand_name, "family": family, "role_method": role_method,
                "r4_representation": r4_rep, "role_reference": ref, "scoring_features": scoring,
            })
            final[(endpoint, cand_name)] = result
    return final


# =============================================================================
# Scoring an arbitrary live player through one final candidate model.
# =============================================================================

def _engineered_row(fv: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    row = dict(fv)
    scoring = candidate["scoring_features"]
    ref = candidate["role_reference"]
    position = fv.get("canonical_position")
    position = "Midfield" if position == "Midfield - Central Midfield" else position
    for feat in scoring:
        raw = fv.get(feat)
        suffix = "role_z" if candidate["role_method"] == "zscore" else "role_pct"
        if raw is not None and ref is not None:
            stats = (ref["per_position"].get((position, feat)) if position in BROAD_POSITIONS else None) or ref["global"][feat]
            if candidate["role_method"] == "zscore":
                row[f"{feat}__{suffix}"] = (raw - stats["median"]) / stats["iqr"]
            else:
                sorted_train = stats["sorted"]
                row[f"{feat}__{suffix}"] = (np.searchsorted(sorted_train, raw, side="right") / len(sorted_train)) if len(sorted_train) else np.nan
        else:
            row[f"{feat}__{suffix}"] = np.nan
        for group in INTERACTION_POSITIONS:
            row[f"{feat}__x__{group}"] = (raw if raw is not None else np.nan) * (1.0 if position == group else 0.0)
    return row


def score_with_candidate(fv: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    row = _engineered_row(fv, candidate)
    frame = pd.DataFrame([{f: row.get(f, np.nan) for f in candidate["features"]}])
    if candidate["kind"] == "hazard":
        n_periods = len(surv.INTERVAL_BOUNDS) - 1
        expanded = pd.concat([frame.assign(hazard_interval=f"period_{i}") for i in range(1, n_periods + 1)], ignore_index=True)
        hazard = np.clip(candidate["model"].predict_proba(candidate["preprocessor"].transform(expanded))[:, 1], 1e-6, 1 - 1e-6)
        cumulative = 1.0 - np.cumprod(1.0 - hazard)
        stay = {label: float(1.0 - cumulative[surv.horizon_column_index(days)]) for label, days in surv.HORIZONS.items() if label in ("24m", "36m")}
        return {"value": stay, "row": row}
    x = candidate["preprocessor"].transform(frame)
    if candidate["kind"] == "binary":
        value = float(np.clip(candidate["model"].predict_proba(x)[0, 1], 1e-6, 1 - 1e-6))
    else:
        value = float(candidate["model"].predict(x)[0])
    return {"value": value, "row": row}


def candidate_contribution(fv: dict[str, Any], candidate: dict[str, Any]) -> list[tuple[str, float]]:
    """Only meaningful for linear (Ridge/LogisticRegression) candidates --
    matches feature_contribution_explainer.raw_contributions' exact
    reconstruction. GBM candidates (contract_duration) return an empty list;
    see README for why the closed-form decomposition does not apply."""
    if not hasattr(candidate["model"], "coef_"):
        return []
    row = _engineered_row(fv, candidate)
    if candidate["kind"] == "hazard":
        row = {**row, "hazard_interval": "period_1"}
        frame = pd.DataFrame([{f: row.get(f, np.nan) for f in candidate["features"] + ["hazard_interval"]}])
    else:
        frame = pd.DataFrame([{f: row.get(f, np.nan) for f in candidate["features"]}])
    x = candidate["preprocessor"].transform(frame)[0]
    # Ridge.coef_ is shape (n_features,) for a 1-D continuous target --
    # coef_[0] would silently grab a single scalar instead of the full
    # coefficient vector. LogisticRegression.coef_ is shape (1, n_features).
    # np.ndim disambiguates instead of assuming the LogisticRegression shape
    # (see raw_contributions() above for the same bug found in the shared,
    # unmodified feature_contribution_explainer.py).
    coef = candidate["model"].coef_[0] if np.ndim(candidate["model"].coef_) > 1 else candidate["model"].coef_
    encoded_raw = candidate["preprocessor"].encoded_raw_features
    per_raw: dict[str, float] = {}
    for raw, xi, ci in zip(encoded_raw, x, coef):
        if raw == "hazard_interval":
            continue
        per_raw[raw] = per_raw.get(raw, 0.0) + float(xi * ci)
    return sorted(per_raw.items(), key=lambda kv: abs(kv[1]), reverse=True)


# =============================================================================
# PHASE B -- live production role audit (READ-ONLY: DeploymentScorer and
# CanonicalFeatureRetriever are imported and called, never modified).
# =============================================================================

DEMO_QUERIES = {
    "Jude Bellingham": "attacking_midfielder",
    "Erling Haaland": "forward",
    "Kylian Mbappé": "forward",  # the unaccented "Kylian Mbappe" returns
    # zero hits from PlayerSearchIndex.search() for this two-word query even
    # though "Mbappe" alone or the accented full name both resolve
    # correctly -- a minor, pre-existing search-robustness gap (same code
    # path the live site's /players/search uses), noted in README.md, not
    # itself a role-contextualization finding and out of scope to fix here.
}
DEFENDER_CANDIDATES = ["Virgil van Dijk", "William Saliba", "Ruben Dias"]
GOALKEEPER_CANDIDATES = ["Alisson Becker", "Ederson Moraes", "Thibaut Courtois", "Jan Oblak"]


def resolve_demo_players() -> list[dict[str, Any]]:
    from player_search import PlayerSearchIndex
    index = PlayerSearchIndex()
    index.warm_up()
    resolved = []
    for name in list(DEMO_QUERIES) + DEFENDER_CANDIDATES + GOALKEEPER_CANDIDATES:
        hits = index.search(name, limit=3)
        if hits:
            hit = hits[0]
            resolved.append({
                "query": name, "canonical_player_id": hit.canonical_player_id,
                "display_name": hit.display_name, "current_club_id": hit.current_club_id,
                "current_club_name": hit.current_club_name, "competition_id": hit.competition_id,
                "player_name_normalized": getattr(hit, "player_name_normalized", None),
            })
    return resolved


def phase_b_live_audit(registry: list[dict[str, Any]]) -> pd.DataFrame:
    from deployment_scorer import DeploymentScorer
    scorer = DeploymentScorer()

    players = resolve_demo_players()
    seen_roles: dict[str, dict] = {}
    demo_set: list[dict[str, Any]] = []
    for name in DEMO_QUERIES:
        hit = next((p for p in players if p["query"] == name), None)
        if hit:
            demo_set.append({**hit, "role_label": "headline_demo"})
    defender = next((p for p in players if p["query"] in DEFENDER_CANDIDATES), None)
    if defender:
        demo_set.append({**defender, "role_label": "representative_defender"})
    keeper = next((p for p in players if p["query"] in GOALKEEPER_CANDIDATES), None)
    if keeper:
        demo_set.append({**keeper, "role_label": "representative_goalkeeper"})

    rows = []
    decision_date = pd.Timestamp.today().normalize()
    for player in demo_set:
        features = scorer.retriever.get_extension_features(
            canonical_player_id=player["canonical_player_id"], canonical_club_id=player["current_club_id"],
            competition_id=player["competition_id"], decision_date=decision_date,
            proposed_annual_fixed_wage_eur=10_000_000, proposed_contract_years=3,
            player_name_normalized=player.get("player_name_normalized"),
        )
        if not features.in_scope:
            rows.append({
                "player": player["display_name"], "role_label": player["role_label"], "status": "refused",
                "refusal_reason": features.refusal_reason,
            })
            continue
        fv = features.features
        position = fv.get("canonical_position")

        for entry in registry:
            endpoint, family, horizon = entry["endpoint"], entry["family"], entry["horizon"]
            deployed_features = entry["deployed_features"]
            if family == "survival":
                # Not every hazard endpoint has both 24m and 36m deployed
                # artifacts (permanent_outbound is 24m-only per model_
                # manifest.json) -- only score horizons that actually exist.
                for art_horizon in DEPLOYED_HAZARD_HORIZONS.get(entry["endpoint"], []):
                    disp = scorer._score_hazard(entry["endpoint"], art_horizon, fv)
                    lbl = f"{endpoint}_{art_horizon}"
                    hazard_artifact = scorer._load("extension_survival_diagnostic", entry["endpoint"], art_horizon)
                    ranked, intercept, logit_offset = raw_contributions(hazard_artifact, fv, drop_features=("hazard_interval",), extra={"hazard_interval": "period_1"})
                    _record_live_row(rows, player, position, lbl, disp, ranked, deployed_features)
            elif family == "financial_exposure" and entry.get("deployed_variant") == "B5_compact_nonlinear":
                artifact = scorer._load(entry["phase"], entry["endpoint"], None)
                displayed = scorer._score_ridge_or_logistic(entry["phase"], entry["endpoint"], fv)
                _record_live_row(rows, player, position, endpoint, displayed, [], deployed_features, note="GBM (contract_duration): closed-form per-feature contribution not applicable -- see feature_contribution_explainer.py docstring.")
            else:
                horizon_arg = entry["horizon"]
                artifact = scorer._load(entry["phase"], entry["endpoint"], horizon_arg)
                displayed = scorer._score_ridge_or_logistic(entry["phase"], entry["endpoint"], fv, horizon_arg)
                ranked, intercept, logit = raw_contributions(artifact, fv)
                _record_live_row(rows, player, position, endpoint, displayed, ranked, deployed_features)
    return pd.DataFrame(rows)


def _record_live_row(rows, player, position, endpoint_label, displayed, ranked, deployed_features, note=None):
    value = displayed.get("value") if isinstance(displayed, dict) else None
    status = displayed.get("status") if isinstance(displayed, dict) else "unavailable"
    ranked_map = dict(ranked)
    scoring_present = [f for f in SCORING_FEATURES_ALL if f in deployed_features]
    for feat in scoring_present:
        contrib = ranked_map.get(feat)
        rows.append({
            "player": player["display_name"], "role_label": player["role_label"], "position": position,
            "endpoint": endpoint_label, "status": status, "deployed_value": value,
            "scoring_feature": feat, "feature_contribution_logit": contrib,
            "raises_result": (contrib is not None and contrib > 0),
            "lowers_result": (contrib is not None and contrib < 0),
            "note": note,
        })
    if not scoring_present:
        rows.append({
            "player": player["display_name"], "role_label": player["role_label"], "position": position,
            "endpoint": endpoint_label, "status": status, "deployed_value": value,
            "scoring_feature": None, "feature_contribution_logit": None,
            "raises_result": None, "lowers_result": None, "note": note,
        })


# =============================================================================
# PHASE F -- role-specific perturbation audit + origin sign-stability check.
# =============================================================================

def build_position_profiles(frame: pd.DataFrame, non_scoring: list[str]) -> dict[str, dict[str, Any]]:
    pos = _position_series(frame)
    profiles = {}
    for group in BROAD_POSITIONS:
        sub = frame.loc[pos.eq(group)]
        profile: dict[str, Any] = {}
        for feat in non_scoring:
            if feat in ("canonical_position",):
                profile[feat] = group
                continue
            series = sub[feat]
            if pd.api.types.is_numeric_dtype(series):
                profile[feat] = float(pd.to_numeric(series, errors="coerce").median())
            else:
                mode = series.mode(dropna=True)
                profile[feat] = str(mode.iloc[0]) if len(mode) else None
        profiles[group] = profile
    return profiles


FAMILY_TO_FRAME_KEY = {
    "opportunity": "opportunity", "survival": "survival",
    "value_preservation": "value_preservation", "financial_exposure": "financial_exposure",
}


def phase_f_perturbation(registry: list[dict[str, Any]], frames: dict[str, pd.DataFrame], final_candidates: dict[tuple[str, str], dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for entry in registry:
        endpoint, family = entry["endpoint"], entry["family"]
        deployed_features = entry["deployed_features"]
        scoring = [f for f in SCORING_FEATURES_ALL if f in deployed_features]
        if not scoring:
            continue
        non_scoring = [f for f in deployed_features if f not in scoring]
        # Each endpoint's non-scoring features must be read from ITS OWN
        # family frame (contract_duration's log_prior_season_annual_gross_eur
        # / prior_wage_available only exist in the financial_exposure frame,
        # not the opportunity frame every other affected endpoint shares).
        opp_frame = frames[FAMILY_TO_FRAME_KEY[family]]
        profiles = build_position_profiles(opp_frame, non_scoring)
        pos_all = _position_series(opp_frame)

        primary_feat = "pre365_goals_per90" if "pre365_goals_per90" in scoring else scoring[0]
        assist_feat = "pre365_assists_per90" if "pre365_assists_per90" in scoring else None

        for position in BROAD_POSITIONS:
            sub_vals = pd.to_numeric(opp_frame.loc[pos_all.eq(position), primary_feat], errors="coerce").dropna()
            if len(sub_vals) < MIN_ROLE_TRAIN_N:
                continue
            quantiles = {"p10": .10, "p25": .25, "p50": .50, "p75": .75, "p90": .90, "p99": .99}
            baseline_value = float(sub_vals.median())
            assist_baseline = float(pd.to_numeric(opp_frame.loc[pos_all.eq(position), assist_feat], errors="coerce").dropna().median()) if assist_feat else None

            candidate_names = ["R0_deployed_reproduction", "R2_position_relative", "R3_position_interactions", "R4_compact_role_aware"]
            baselines: dict[str, Any] = {}
            for cand_name in candidate_names:
                key = (endpoint, cand_name)
                if key not in final_candidates:
                    continue
                candidate = final_candidates[key]
                fv = dict(profiles[position])
                fv[primary_feat] = baseline_value
                if assist_feat:
                    fv[assist_feat] = assist_baseline
                result = score_with_candidate(fv, candidate)
                baselines[cand_name] = result["value"]

            for q_label, q in quantiles.items():
                raw_value = float(sub_vals.quantile(q))
                pct_within_position = float(q)
                for cand_name in candidate_names:
                    key = (endpoint, cand_name)
                    if key not in final_candidates:
                        continue
                    candidate = final_candidates[key]
                    fv = dict(profiles[position])
                    fv[primary_feat] = raw_value
                    if assist_feat:
                        fv[assist_feat] = assist_baseline
                    result = score_with_candidate(fv, candidate)
                    value = result["value"]
                    base = baselines.get(cand_name)
                    if isinstance(value, dict):
                        for horizon_label, v in value.items():
                            b = base.get(horizon_label) if isinstance(base, dict) else None
                            rows.append({
                                "endpoint": f"{endpoint}_{horizon_label}", "candidate": cand_name, "position": position,
                                "quantile": q_label, "raw_scoring_value": raw_value, "within_position_percentile": pct_within_position,
                                "predicted_value": v, "baseline_value": b,
                                "change_from_baseline": (v - b) if b is not None else None,
                                "large_effect": (abs(v - b) > 0.15) if b is not None else None,
                            })
                    else:
                        rows.append({
                            "endpoint": endpoint, "candidate": cand_name, "position": position,
                            "quantile": q_label, "raw_scoring_value": raw_value, "within_position_percentile": pct_within_position,
                            "predicted_value": value, "baseline_value": base,
                            "change_from_baseline": (value - base) if base is not None else None,
                            "large_effect": (abs(value - base) > 0.15) if base is not None else None,
                        })
    return pd.DataFrame(rows)


def origin_sign_stability(registry: list[dict[str, Any]], frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Fast per-origin R0 refit (deployed feature list) recording the raw
    goals-per-90 coefficient's sign, to check whether any observed negative
    relationship is stable across the four rolling origins or an artifact
    of one origin's sample."""
    rows = []
    for entry in registry:
        endpoint, family = entry["endpoint"], entry["family"]
        deployed_features = entry["deployed_features"]
        if "pre365_goals_per90" not in deployed_features:
            continue
        is_hazard = family == "survival"
        for origin_id, train_end, validation_year, evaluation_year in ORIGINS:
            if is_hazard:
                base = frames["survival"].loc[frames["survival"]["survival_primary_cohort"]]
                cohort = base.loc[base["signed_year"].le(validation_year)].copy()
                periods = surv.person_periods(cohort, endpoint)
                pre = fit_preprocessor(periods, deployed_features + ["hazard_interval"])
                x = pre.transform(periods)
                y = periods["hazard_event"].to_numpy(int)
            else:
                if family == "opportunity":
                    cohort_full = entry["cohort_fn"](frames["opportunity"])
                elif family == "value_preservation":
                    cohort_full = entry["cohort_fn"](frames["value_preservation"])
                else:
                    fr = frames["financial_exposure"]
                    cohort_full = fr.loc[fr[entry["cohort_col"]] & fr[entry["target"]].notna()]
                cohort = cohort_full.loc[cohort_full["signed_year"].le(validation_year)].copy()
                if len(cohort) < 30:
                    continue
                pre = fit_preprocessor(cohort, deployed_features)
                x = pre.transform(cohort)
                y = pd.to_numeric(cohort[entry["target"]]).to_numpy()
            try:
                if is_hazard or entry["kind"] == "binary":
                    model = LogisticRegression(C=0.01, penalty="l2", solver="liblinear", max_iter=3000, random_state=RANDOM_SEED).fit(x, y.astype(int))
                elif entry.get("deployed_variant") == "B5_compact_nonlinear":
                    continue  # GBM has no linear coefficient to sign-check
                else:
                    model = Ridge(alpha=1.0).fit(x, y)
            except Exception as exc:  # pragma: no cover - defensive
                rows.append({"endpoint": endpoint, "origin_id": origin_id, "error": str(exc)})
                continue
            encoded_raw = pre.encoded_raw_features
            coef = model.coef_[0] if hasattr(model.coef_, "shape") and model.coef_.ndim > 1 else model.coef_
            per_raw = {}
            for raw, ci in zip(encoded_raw, coef):
                per_raw[raw] = per_raw.get(raw, 0.0) + float(ci)
            rows.append({
                "endpoint": endpoint, "origin_id": origin_id, "train_through_year": validation_year,
                "n_rows": len(cohort), "pre365_goals_per90_coefficient": per_raw.get("pre365_goals_per90"),
                "sign": "negative" if per_raw.get("pre365_goals_per90", 0) < 0 else "positive",
            })
    return pd.DataFrame(rows)


# =============================================================================
# PHASE G -- demo player comparison across R0-R4 (final candidates).
# =============================================================================

def phase_g_demo_comparison(registry: list[dict[str, Any]], final_candidates: dict[tuple[str, str], dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    from deployment_scorer import DeploymentScorer
    scorer = DeploymentScorer()
    comparison_rows, contribution_rows = [], []
    decision_date = pd.Timestamp.today().normalize()

    for name in DEMO_QUERIES:
        players = resolve_demo_players()
        hit = next((p for p in players if p["query"] == name), None)
        if not hit:
            continue
        features = scorer.retriever.get_extension_features(
            canonical_player_id=hit["canonical_player_id"], canonical_club_id=hit["current_club_id"],
            competition_id=hit["competition_id"], decision_date=decision_date,
            proposed_annual_fixed_wage_eur=10_000_000, proposed_contract_years=3,
            player_name_normalized=hit.get("player_name_normalized"),
        )
        if not features.in_scope:
            continue
        fv = features.features
        position = fv.get("canonical_position")

        for entry in registry:
            endpoint = entry["endpoint"]
            scoring = [f for f in SCORING_FEATURES_ALL if f in entry["deployed_features"]]
            raw_scoring_values = {f: fv.get(f) for f in scoring}

            deployed_value = None
            if entry["family"] == "survival":
                deployed_value = {}
                for art_horizon in DEPLOYED_HAZARD_HORIZONS.get(endpoint, []):
                    deployed_value[art_horizon] = scorer._score_hazard(endpoint, art_horizon, fv).get("value")
            else:
                horizon_arg = entry["horizon"]
                d = scorer._score_ridge_or_logistic(entry["phase"], endpoint, fv, horizon_arg)
                deployed_value = d.get("value")

            candidate_values: dict[str, Any] = {"R0_deployed_(actual_production)": deployed_value}
            for cand_name in ["R1_remove_scoring", "R2_position_relative", "R3_position_interactions", "R4_compact_role_aware"]:
                key = (endpoint, cand_name)
                if key not in final_candidates:
                    continue
                candidate = final_candidates[key]
                result = score_with_candidate(fv, candidate)
                candidate_values[cand_name] = result["value"]
                contributions = candidate_contribution(fv, candidate)
                for rank, (feat, contrib) in enumerate(contributions[:5]):
                    contribution_rows.append({
                        "player": name, "endpoint": endpoint, "candidate": cand_name, "rank": rank + 1,
                        "feature": feat, "label": label_for(feat), "logit_or_coef_contribution": contrib,
                        "direction": "positive" if contrib > 0 else "negative",
                    })
                for rank, (feat, contrib) in enumerate(sorted(contributions, key=lambda kv: kv[1])[:5]):
                    contribution_rows.append({
                        "player": name, "endpoint": endpoint, "candidate": cand_name, "rank": f"bottom_{rank + 1}",
                        "feature": feat, "label": label_for(feat), "logit_or_coef_contribution": contrib,
                        "direction": "positive" if contrib > 0 else "negative",
                    })

            def scalar(v):
                if isinstance(v, dict):
                    return v.get("24m")
                return v

            r0 = scalar(candidate_values.get("R0_deployed_(actual_production)"))
            row = {
                "player": name, "position": position, "endpoint": endpoint,
                **{f"raw_{f}": raw_scoring_values.get(f) for f in SCORING_FEATURES_ALL},
            }
            for cand_name, v in candidate_values.items():
                sv = scalar(v)
                row[f"{cand_name}__value"] = sv
                if r0 is not None and sv is not None:
                    row[f"{cand_name}__abs_change_vs_R0"] = sv - r0
                    row[f"{cand_name}__pct_point_change_vs_R0"] = (sv - r0) * 100
            comparison_rows.append(row)
    return pd.DataFrame(comparison_rows), pd.DataFrame(contribution_rows)


# =============================================================================
# PHASE H -- decision rules (computed here for transparency; this script
# never deploys anything regardless of the outcome).
# =============================================================================

def phase_h_decisions(pooled: pd.DataFrame, subgroup: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint in pooled["endpoint"].unique():
        term = pooled.loc[pooled["endpoint"].eq(endpoint) & pooled["window"].eq("terminal_two")]
        four = pooled.loc[pooled["endpoint"].eq(endpoint) & pooled["window"].eq("pooled_four")]

        def gate(candidate: str) -> dict[str, Any]:
            t = term.loc[term["candidate"].eq(candidate)]
            f = four.loc[four["candidate"].eq(candidate)]
            if t.empty or f.empty:
                return {"eligible": False, "reason": "missing pooled comparison rows"}
            t, f = t.iloc[0], f.iloc[0]
            non_inferior_terminal = t["pooled_loss_improvement"] >= -0.005
            majority_origin_wins = f["origin_wins"] >= (f["n_origins"] / 2)
            no_harm_ci = f["bootstrap_ci_lower_95"] >= -0.01
            sub = subgroup.loc[subgroup["endpoint"].eq(endpoint) & subgroup["candidate"].eq(candidate)]
            r0_sub = subgroup.loc[subgroup["endpoint"].eq(endpoint) & subgroup["candidate"].eq("R0_deployed_reproduction")]
            degrades_group = False
            for _, r in sub.iterrows():
                base_rows = r0_sub.loc[r0_sub["group_type"].eq(r["group_type"]) & r0_sub["group_value"].eq(r["group_value"])]
                if base_rows.empty or r["n_rows"] < 30:
                    continue
                base = base_rows.iloc[0]
                metric = "brier" if "brier" in r and pd.notna(r.get("brier")) else "mae"
                if metric in r and metric in base and pd.notna(r[metric]) and pd.notna(base[metric]):
                    if r[metric] > base[metric] * 1.10:  # >10% worse on an adequately-sized group
                        degrades_group = True
            eligible = bool(non_inferior_terminal and majority_origin_wins and no_harm_ci and not degrades_group)
            return {
                "eligible": eligible, "terminal_two_improvement": float(t["pooled_loss_improvement"]),
                "origin_wins": int(f["origin_wins"]), "n_origins": int(f["n_origins"]),
                "ci_lower_95": float(f["bootstrap_ci_lower_95"]), "ci_upper_95": float(f["bootstrap_ci_upper_95"]),
                "degrades_a_position_group": bool(degrades_group),
            }

        gates = {c: gate(c) for c in ["R1_remove_scoring", "R2_position_relative", "R3_position_interactions", "R4_compact_role_aware"]}
        if gates["R4_compact_role_aware"].get("eligible"):
            decision, chosen = "ADVANCE_ROLE_AWARE", "R4_compact_role_aware"
        elif gates["R2_position_relative"].get("eligible") or gates["R3_position_interactions"].get("eligible"):
            chosen = "R2_position_relative" if gates["R2_position_relative"].get("eligible") else "R3_position_interactions"
            decision = "ADVANCE_ROLE_AWARE"
        elif gates["R1_remove_scoring"].get("eligible"):
            decision, chosen = "REMOVE_SCORING_FEATURES", "R1_remove_scoring"
        else:
            decision, chosen = "RETAIN_CURRENT_PENDING_MORE_DATA", None

        rows.append({
            "endpoint": endpoint, "decision": decision, "chosen_candidate": chosen,
            "R1_gate": json.dumps(gates["R1_remove_scoring"]), "R2_gate": json.dumps(gates["R2_position_relative"]),
            "R3_gate": json.dumps(gates["R3_position_interactions"]), "R4_gate": json.dumps(gates["R4_compact_role_aware"]),
            "note": "Computed decision gate only -- this script does not deploy anything regardless of outcome. Requires human review.",
        })
    return pd.DataFrame(rows)


# =============================================================================
# main
# =============================================================================

def main() -> None:
    import time
    t0 = time.time()
    print("=" * 70)
    print("Role-contextualization diagnostic -- START")
    print("=" * 70)
    capture_protected_fingerprint(OUT / "protected_artifacts_fingerprint_before.txt")

    print("\n[Phase A] production feature inventory")
    deployed = load_deployed_artifacts()
    inventory = phase_a_feature_inventory(deployed)
    write_csv(inventory, "production_scoring_feature_inventory.csv")
    c6_row = inventory.loc[inventory["endpoint"].eq("sustained_meaningful_contribution")].iloc[0]
    assert not c6_row["uses_any_scoring_rate_feature"], "C6 must NOT use scoring-rate features -- aborting."
    print(f"  confirmed: sustained_meaningful_contribution (C6) excludes scoring-rate features (variant={c6_row['deployed_variant']}). Out of scope, untouched.")

    registry = build_endpoint_registry(deployed)
    print(f"  affected endpoints derived from artifacts: {[e['endpoint'] + (('_' + e['horizon']) if e.get('horizon') else '') for e in registry]}")

    print("\n[Phase C] role/position data audit")
    frames = load_family_frames()
    audits = phase_c_audits(frames)
    write_csv(audits["coverage"], "role_coverage_audit.csv")
    write_csv(audits["distribution"], "role_scoring_distribution_audit.csv")
    write_csv(audits["origin_stability"], "role_origin_stability_audit.csv")
    write_csv(audits["sub_position_counts"], "role_sub_position_coverage.csv")

    print("\n[Phase B] live production role audit")
    live_audit = phase_b_live_audit(registry)
    write_csv(live_audit, "live_production_role_audit.csv")

    print("\n[Phase D/E] rolling-origin candidate backtest (R0-R4)")
    backtest = run_origin_backtest(registry, frames)
    write_csv(backtest["candidate_manifest"], "candidate_feature_manifest.csv")
    write_csv(backtest["origin_metrics"], "origin_model_metrics.csv")
    write_csv(backtest["fallback_log"], "role_reference_fallback_log.csv")
    write_csv(backtest["r4_choices"], "r4_representation_choices.csv")
    write_csv(backtest["role_method_choices"], "role_method_choices.csv")

    print("\n  pooled + player-clustered bootstrap comparison")
    pooled = pooled_comparison(backtest["predictions"])
    write_csv(pooled, "pooled_model_comparison.csv")

    print("  position/age/league/market-value subgroup metrics")
    subgroup = subgroup_metrics(backtest["predictions"])
    write_csv(subgroup, "position_subgroup_metrics.csv")

    print("  calibration audit")
    calib = calibration_audit(backtest["predictions"])
    write_csv(calib, "calibration_audit.csv")

    print("\n[final candidates] deployment-equivalent refits for live scoring")
    final_candidates = build_final_candidates(registry, frames)
    print(f"  built {len(final_candidates)} final (endpoint, candidate) fits")

    print("\n[Phase F] role-specific perturbation audit + origin sign stability")
    perturbation = phase_f_perturbation(registry, frames, final_candidates)
    write_csv(perturbation, "role_specific_perturbation_audit.csv")
    sign_stability = origin_sign_stability(registry, frames)
    write_csv(sign_stability, "goals_per90_origin_sign_stability.csv")

    print("\n[Phase G] demo player comparison (Bellingham, Haaland, Mbappe)")
    demo_comparison, demo_contributions = phase_g_demo_comparison(registry, final_candidates)
    write_csv(demo_comparison, "demo_player_role_comparison.csv")
    write_csv(demo_contributions, "demo_player_role_contributions.csv")

    print("\n[Phase H] decision rules")
    decisions = phase_h_decisions(pooled, subgroup)
    write_csv(decisions, "model_selection_decision.csv")

    print("\n[candidate artifacts] saving final candidates (labeled not_deployed_pending_review)")
    import joblib
    artifact_rows = []
    for (endpoint, cand_name), candidate in final_candidates.items():
        if cand_name == "R0_deployed_reproduction":
            continue  # R0 is a reproduction check, not a new artifact -- do not save a duplicate of deployed IP
        path = CANDIDATES_DIR / f"{endpoint}__{cand_name}.joblib"
        payload = {
            "endpoint": endpoint, "candidate": cand_name, "status_label": "not_deployed_pending_review",
            "kind": candidate["kind"], "features": candidate["features"], "preprocessor": candidate["preprocessor"],
            "model": candidate["model"], "role_method": candidate.get("role_method"),
            "r4_representation": candidate.get("r4_representation"),
        }
        joblib.dump(payload, path)
        artifact_rows.append({
            "endpoint": endpoint, "candidate": cand_name, "path": str(path.relative_to(ROOT)),
            "status_label": "not_deployed_pending_review", "sha256": sha256_file(path),
        })
    write_csv(pd.DataFrame(artifact_rows), "candidate_artifacts_manifest.csv")

    print("\n[source manifest]")
    source_files = [
        ROOT / "Data" / "processed" / "contract_extension_integration" / "extension_modeling_master.csv",
        ROOT / "Data" / "processed" / "deployment_models" / "model_manifest.json",
    ]
    source_rows = [{"path": str(p.relative_to(ROOT)), "sha256": sha256_file(p), "exists": p.exists()} for p in source_files]
    write_csv(pd.DataFrame(source_rows), "source_manifest.csv")

    elapsed = time.time() - t0
    summary = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "elapsed_seconds": elapsed,
        "random_seed": RANDOM_SEED,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "origins": [{"id": o[0], "train_end": o[1], "validation_year": o[2], "evaluation_year": o[3]} for o in ORIGINS],
        "affected_endpoints": [e["endpoint"] + (f"_{e['horizon']}" if e.get("horizon") else "") for e in registry],
        "sustained_meaningful_contribution_C6_out_of_scope_confirmed": True,
        "decisions": decisions.to_dict("records"),
        "deployment_manifest_model_vintage": DEPLOYMENT_MANIFEST["model_vintage"],
    }
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"  wrote {OUT / 'run_summary.json'}")

    capture_protected_fingerprint(OUT / "protected_artifacts_fingerprint_after.txt")

    print(f"\nDONE in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

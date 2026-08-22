"""Phase-3 diagnostic for post-extension market-value preservation.

The experiment pre-registers 12- and 24-month log value ratios and nested
downside events of at least 10%, 25%, and 50%. All valuation landmarks satisfy
the integration layer's strict timing rules. Results concern public market-
value estimates, not sale proceeds, accounting profit, or realized ROI.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, mean_absolute_error,
    mean_squared_error, r2_score, roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "models") not in sys.path:
    sys.path.insert(0, str(ROOT / "models"))
from mixed_type_preprocessor import fit_preprocessor
import run_extension_opportunity_diagnostic as phase1


SOURCE = phase1.SOURCE
UPSTREAM_VERIFY = ROOT / "Data" / "processed" / "contract_extension_integration" / "independent_verification.json"
OUTPUT = ROOT / "Data" / "processed" / "extension_value_preservation_diagnostic"
RANDOM_SEED = 20260811
BOOTSTRAP_REPETITIONS = 5000
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]
LOGISTIC_C = [0.01, 0.1, 1.0, 10.0]
ORIGINS = phase1.ORIGINS
WINDOWS = phase1.WINDOWS
FEATURE_SETS = phase1.FEATURE_SETS

ENDPOINTS: dict[str, dict[str, str]] = {}
for horizon in [12, 24]:
    ENDPOINTS[f"value_log_ratio_{horizon}m"] = {
        "kind": "continuous", "target": f"target_log_value_ratio_{horizon}m", "horizon": f"{horizon}m",
    }
    for threshold in [10, 25, 50]:
        ENDPOINTS[f"downside_{threshold}pct_{horizon}m"] = {
            "kind": "binary", "target": f"target_downside_{threshold}pct_{horizon}m", "horizon": f"{horizon}m",
        }

COMPARISONS = [
    ("M1_vs_M0", "M0_chronological_baseline", "M1_player_market_profile"),
    ("M2_vs_M1", "M1_player_market_profile", "M2_plus_prior_opportunity"),
    ("M3_vs_M2", "M2_plus_prior_opportunity", "M3_plus_performance_history"),
    ("M4_vs_M3_contract_terms", "M3_plus_performance_history", "M4_plus_contract_terms"),
    ("M5_vs_M4_relative_financial", "M4_plus_contract_terms", "M5_plus_relative_financial_context"),
    ("M5_vs_M3_all_contract_economics", "M3_plus_performance_history", "M5_plus_relative_financial_context"),
    ("M5_vs_M0_total", "M0_chronological_baseline", "M5_plus_relative_financial_context"),
    ("M6_vs_M5_nonlinearity", "M5_plus_relative_financial_context", "M6_compact_nonlinear"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: str) -> int:
    return (RANDOM_SEED + int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)) % (2**32 - 1)


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").fillna("").str.strip().str.casefold().isin({"true", "1", "yes"})


def load_frame() -> pd.DataFrame:
    upstream = json.loads(UPSTREAM_VERIFY.read_text(encoding="utf-8"))
    if not upstream.get("all_checks_passed"):
        raise RuntimeError("Contract extension integration has not passed independent verification.")
    frame = phase1.load_frame()
    frame["signed_date"] = pd.to_datetime(frame["signed_date"], errors="coerce")
    frame["signed_year"] = frame["signed_date"].dt.year.astype("Int64")
    at_value = pd.to_numeric(frame["at_signing_market_value_eur"], errors="coerce")
    base = frame["canonical_player_id"].notna() & as_bool(frame["at_signing_strict_timing_eligible"]) & at_value.gt(0)
    frame["value_base_cohort"] = base
    for horizon in [12, 24]:
        post = pd.to_numeric(frame[f"post{horizon}_market_value_eur"], errors="coerce")
        cohort = base & as_bool(frame[f"post{horizon}_strict_timing_eligible"]) & post.gt(0)
        ratio = post.div(at_value.where(at_value.gt(0)))
        frame[f"value_cohort_{horizon}m"] = cohort
        frame[f"target_value_ratio_{horizon}m"] = ratio
        frame[f"target_log_value_ratio_{horizon}m"] = np.log(ratio.where(ratio.gt(0)))
        frame[f"target_value_change_pct_{horizon}m"] = ratio - 1.0
        for threshold in [10, 25, 50]:
            boundary = np.nextafter(at_value * (1 - threshold / 100), np.inf)
            frame[f"target_downside_{threshold}pct_{horizon}m"] = post.le(boundary).astype(int)
        frame[f"target_value_preserved_90pct_{horizon}m"] = 1 - frame[f"target_downside_10pct_{horizon}m"]
    frame["age_band"] = pd.cut(frame["age_at_signing"], [-np.inf, 21, 25, 29, 33, np.inf], labels=["<=21", "22-25", "26-29", "30-33", "34+"])
    frame["contract_length_band"] = pd.cut(frame["exact_duration_years"], [-np.inf, 2, 3, 4, 5, np.inf], labels=["<2y", "2-<3y", "3-<4y", "4-<5y", "5y+"])
    return frame


def endpoint_frame(frame: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    horizon = ENDPOINTS[endpoint]["horizon"]
    return frame.loc[frame[f"value_cohort_{horizon}"] & frame[ENDPOINTS[endpoint]["target"]].notna()].copy()


def continuous_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    correlation = pd.Series(actual).corr(pd.Series(prediction), method="spearman")
    return {
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(actual, prediction))),
        "r2": float(r2_score(actual, prediction)),
        "spearman": float(correlation) if pd.notna(correlation) else np.nan,
        "median_absolute_error": float(np.median(np.abs(actual - prediction))),
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
        result["roc_auc"] = np.nan
        result["average_precision"] = np.nan
    return result


def tune_predict(train: pd.DataFrame, validation: pd.DataFrame, evaluation: pd.DataFrame, features: list[str], target: str, kind: str, variant: str, endpoint: str) -> tuple[np.ndarray, dict[str, Any], int]:
    y_train = pd.to_numeric(train[target]).to_numpy()
    y_validation = pd.to_numeric(validation[target]).to_numpy()
    if not features:
        anchor = float(np.median(y_train)) if kind == "continuous" else float(np.mean(y_train))
        validation_prediction = np.repeat(anchor, len(validation))
        fit_target = pd.to_numeric(pd.concat([train, validation])[target])
        refit_anchor = float(fit_target.median()) if kind == "continuous" else float(fit_target.mean())
        prediction = np.repeat(refit_anchor, len(evaluation))
        metrics = continuous_metrics(y_validation, validation_prediction) if kind == "continuous" else binary_metrics(y_validation.astype(int), validation_prediction)
        return prediction, {"selected_parameter": np.nan, **metrics}, 0

    pre = fit_preprocessor(train, features)
    x_train, x_validation = pre.transform(train), pre.transform(validation)
    nonlinear = variant == "M6_compact_nonlinear"
    candidates = []
    if kind == "continuous":
        if nonlinear:
            grid = [
                {"n_estimators": 75, "learning_rate": 0.03, "max_depth": 1, "min_samples_leaf": 20},
                {"n_estimators": 100, "learning_rate": 0.03, "max_depth": 2, "min_samples_leaf": 25},
                {"n_estimators": 75, "learning_rate": 0.05, "max_depth": 2, "min_samples_leaf": 30},
            ]
            for i, params in enumerate(grid):
                model = GradientBoostingRegressor(loss="huber", random_state=stable_seed(endpoint, variant, str(i)), **params).fit(x_train, y_train)
                estimate = model.predict(x_validation)
                candidates.append(((mean_absolute_error(y_validation, estimate), math.sqrt(mean_squared_error(y_validation, estimate))), params))
        else:
            for alpha in RIDGE_ALPHAS:
                model = Ridge(alpha=alpha).fit(x_train, y_train)
                estimate = model.predict(x_validation)
                candidates.append(((mean_absolute_error(y_validation, estimate), math.sqrt(mean_squared_error(y_validation, estimate))), {"alpha": alpha}))
    else:
        if nonlinear:
            grid = [
                {"n_estimators": 75, "learning_rate": 0.03, "max_depth": 1, "min_samples_leaf": 20},
                {"n_estimators": 100, "learning_rate": 0.03, "max_depth": 2, "min_samples_leaf": 25},
                {"n_estimators": 75, "learning_rate": 0.05, "max_depth": 2, "min_samples_leaf": 30},
            ]
            for i, params in enumerate(grid):
                model = GradientBoostingClassifier(loss="log_loss", random_state=stable_seed(endpoint, variant, str(i)), **params).fit(x_train, y_train.astype(int))
                estimate = np.clip(model.predict_proba(x_validation)[:, 1], 1e-6, 1 - 1e-6)
                candidates.append(((brier_score_loss(y_validation.astype(int), estimate), log_loss(y_validation.astype(int), estimate, labels=[0, 1])), params))
        else:
            for c_value in LOGISTIC_C:
                model = LogisticRegression(C=c_value, penalty="l2", solver="liblinear", max_iter=3000, random_state=RANDOM_SEED).fit(x_train, y_train.astype(int))
                estimate = np.clip(model.predict_proba(x_validation)[:, 1], 1e-6, 1 - 1e-6)
                candidates.append(((brier_score_loss(y_validation.astype(int), estimate), log_loss(y_validation.astype(int), estimate, labels=[0, 1])), {"C": c_value}))
    _, params = min(candidates, key=lambda item: item[0])
    fit = pd.concat([train, validation], ignore_index=True)
    fit_pre = fit_preprocessor(fit, features)
    x_fit, x_eval = fit_pre.transform(fit), fit_pre.transform(evaluation)
    y_fit = pd.to_numeric(fit[target]).to_numpy()
    if kind == "continuous":
        model = GradientBoostingRegressor(loss="huber", random_state=stable_seed(endpoint, variant, "fit"), **params) if nonlinear else Ridge(alpha=params["alpha"])
        model.fit(x_fit, y_fit)
        prediction = model.predict(x_eval)
        validation_model = GradientBoostingRegressor(loss="huber", random_state=stable_seed(endpoint, variant, "validation"), **params) if nonlinear else Ridge(alpha=params["alpha"])
        validation_estimate = validation_model.fit(x_train, y_train).predict(x_validation)
        validation_metrics = continuous_metrics(y_validation, validation_estimate)
    else:
        model = GradientBoostingClassifier(loss="log_loss", random_state=stable_seed(endpoint, variant, "fit"), **params) if nonlinear else LogisticRegression(C=params["C"], penalty="l2", solver="liblinear", max_iter=3000, random_state=RANDOM_SEED)
        model.fit(x_fit, y_fit.astype(int))
        prediction = np.clip(model.predict_proba(x_eval)[:, 1], 1e-6, 1 - 1e-6)
        validation_model = GradientBoostingClassifier(loss="log_loss", random_state=stable_seed(endpoint, variant, "validation"), **params) if nonlinear else LogisticRegression(C=params["C"], penalty="l2", solver="liblinear", max_iter=3000, random_state=RANDOM_SEED)
        validation_model.fit(x_train, y_train.astype(int))
        validation_estimate = np.clip(validation_model.predict_proba(x_validation)[:, 1], 1e-6, 1 - 1e-6)
        validation_metrics = binary_metrics(y_validation.astype(int), validation_estimate)
    return prediction, {"selected_parameter": json.dumps(params, sort_keys=True), **validation_metrics}, x_fit.shape[1]


def fit_models(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows, metric_rows = [], []
    for endpoint, spec in ENDPOINTS.items():
        data = endpoint_frame(frame, endpoint)
        target, kind = spec["target"], spec["kind"]
        for origin, train_end, validation_year, evaluation_year in ORIGINS:
            train = data.loc[data["signed_year"].le(train_end)].copy()
            validation = data.loc[data["signed_year"].eq(validation_year)].copy()
            evaluation = data.loc[data["signed_year"].eq(evaluation_year)].sort_values("capology_extension_event_id").copy()
            if min(len(train), len(validation), len(evaluation)) < 30:
                raise RuntimeError(f"Insufficient split {endpoint}/{origin}: {len(train)}/{len(validation)}/{len(evaluation)}")
            for variant, features in FEATURE_SETS.items():
                estimate, validation_metrics, encoded = tune_predict(train, validation, evaluation, features, target, kind, variant, endpoint)
                actual = pd.to_numeric(evaluation[target]).to_numpy()
                evaluation_metrics = continuous_metrics(actual, estimate) if kind == "continuous" else binary_metrics(actual.astype(int), estimate)
                metric_rows.append({
                    "endpoint": endpoint, "horizon": spec["horizon"], "kind": kind,
                    "origin_id": origin, "model_variant": variant, "train_end_year": train_end,
                    "validation_year": validation_year, "evaluation_year": evaluation_year,
                    "train_rows": len(train), "validation_rows": len(validation), "evaluation_rows": len(evaluation),
                    "raw_feature_count": len(features), "encoded_feature_count": encoded,
                    **{f"validation_{key}": value for key, value in validation_metrics.items()},
                    **{f"evaluation_{key}": value for key, value in evaluation_metrics.items()},
                })
                for i, (_, row) in enumerate(evaluation.iterrows()):
                    value, prediction = float(actual[i]), float(estimate[i])
                    prediction_rows.append({
                        "endpoint": endpoint, "horizon": spec["horizon"], "kind": kind,
                        "origin_id": origin, "evaluation_year": evaluation_year,
                        "capology_extension_event_id": row["capology_extension_event_id"],
                        "canonical_player_id": row["canonical_player_id"], "league": row["league"],
                        "canonical_position": row["canonical_position"], "age_band": row["age_band"],
                        "contract_length_band": row["contract_length_band"], "model_variant": variant,
                        "actual": value, "prediction": prediction,
                        "value_preservation_probability": 1.0 - prediction if endpoint.startswith("downside_10pct") else np.nan,
                        "primary_loss": abs(value - prediction) if kind == "continuous" else (value - prediction) ** 2,
                    })
    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)


def paired(predictions: pd.DataFrame, endpoint: str, baseline: str, candidate: str, years: list[int]) -> pd.DataFrame:
    subset = predictions.loc[predictions["endpoint"].eq(endpoint) & predictions["evaluation_year"].isin(years)]
    keys = ["endpoint", "origin_id", "evaluation_year", "capology_extension_event_id", "canonical_player_id"]
    left = subset.loc[subset["model_variant"].eq(baseline), keys + ["primary_loss"]].rename(columns={"primary_loss": "baseline_loss"})
    right = subset.loc[subset["model_variant"].eq(candidate), keys + ["primary_loss"]].rename(columns={"primary_loss": "candidate_loss"})
    return left.merge(right, on=keys, how="inner", validate="one_to_one")


def cluster_bootstrap(pairs: pd.DataFrame, seed: int) -> dict[str, float]:
    data = pairs.assign(improvement=pairs["baseline_loss"] - pairs["candidate_loss"])
    clusters = data.groupby("canonical_player_id")["improvement"].agg(["sum", "size"])
    sums, sizes = clusters["sum"].to_numpy(float), clusters["size"].to_numpy(float)
    rng, draws = np.random.default_rng(seed), []
    for start in range(0, BOOTSTRAP_REPETITIONS, 250):
        count = min(250, BOOTSTRAP_REPETITIONS - start)
        indices = rng.integers(0, len(clusters), size=(count, len(clusters)))
        draws.extend((sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)).tolist())
    values = np.asarray(draws)
    return {
        "cluster_count": len(clusters), "cluster_ci_lower_95": float(np.quantile(values, 0.025)),
        "cluster_ci_upper_95": float(np.quantile(values, 0.975)),
        "cluster_probability_candidate_better": float(np.mean(values > 0)),
    }


def comparisons(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint, spec in ENDPOINTS.items():
        for window, years in WINDOWS.items():
            for name, baseline, candidate in COMPARISONS:
                pairs = paired(predictions, endpoint, baseline, candidate, years)
                delta = pairs["baseline_loss"] - pairs["candidate_loss"]
                origin_delta = pairs.assign(delta=delta).groupby("evaluation_year")["delta"].mean()
                rows.append({
                    "endpoint": endpoint, "horizon": spec["horizon"], "kind": spec["kind"],
                    "window": window, "comparison": name, "baseline": baseline, "candidate": candidate,
                    "evaluation_years": " | ".join(map(str, sorted(pairs["evaluation_year"].unique()))),
                    "evaluation_rows": len(pairs), "mean_loss_improvement": float(delta.mean()),
                    "origin_wins": int(origin_delta.gt(0).sum()), "origin_count": len(origin_delta),
                    **cluster_bootstrap(pairs, stable_seed(endpoint, window, name)),
                })
    return pd.DataFrame(rows)


def risk_bands(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint, spec in ENDPOINTS.items():
        data = predictions.loc[predictions["endpoint"].eq(endpoint) & predictions["model_variant"].eq("M5_plus_relative_financial_context")].copy()
        data["risk_band"] = pd.qcut(data["prediction"].rank(method="first"), 5, labels=False) + 1
        grouped = data.groupby("risk_band", observed=True).agg(
            records=("actual", "size"), mean_prediction=("prediction", "mean"),
            observed_outcome=("actual", "mean"), mean_primary_loss=("primary_loss", "mean"),
        ).reset_index()
        grouped.insert(0, "kind", spec["kind"]); grouped.insert(0, "horizon", spec["horizon"]); grouped.insert(0, "endpoint", endpoint)
        grouped["observed_monotone"] = bool(grouped["observed_outcome"].is_monotonic_increasing)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def threshold_ordering(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in ["12m", "24m"]:
        for variant in ["M5_plus_relative_financial_context", "M6_compact_nonlinear"]:
            subset = predictions.loc[predictions["horizon"].eq(horizon) & predictions["kind"].eq("binary") & predictions["model_variant"].eq(variant)]
            wide = subset.pivot(index=["origin_id", "evaluation_year", "capology_extension_event_id"], columns="endpoint", values="prediction")
            p10, p25, p50 = wide[f"downside_10pct_{horizon}"], wide[f"downside_25pct_{horizon}"], wide[f"downside_50pct_{horizon}"]
            rows.append({
                "horizon": horizon, "model_variant": variant, "rows": len(wide),
                "down25_gt_down10_violations": int((p25 > p10).sum()),
                "down50_gt_down25_violations": int((p50 > p25).sum()),
                "any_order_violation_rows": int(((p25 > p10) | (p50 > p25)).sum()),
            })
    return pd.DataFrame(rows)


def subgroup_stability(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    source = predictions.loc[predictions["horizon"].eq("24m") & predictions["model_variant"].eq("M5_plus_relative_financial_context")]
    for endpoint, data in source.groupby("endpoint", observed=True):
        kind = data["kind"].iloc[0]
        for group_type, column in [("league", "league"), ("position", "canonical_position"), ("age_band", "age_band"), ("contract_length", "contract_length_band")]:
            for group, subset in data.groupby(column, observed=True, dropna=False):
                if len(subset) < 30:
                    continue
                actual, estimate = subset["actual"].to_numpy(), subset["prediction"].to_numpy()
                score = continuous_metrics(actual, estimate) if kind == "continuous" else binary_metrics(actual.astype(int), estimate)
                rows.append({"endpoint": endpoint, "kind": kind, "group_type": group_type, "group": str(group), "records": len(subset), **score})
    return pd.DataFrame(rows)


def target_audits(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, timing = [], []
    for horizon in [12, 24]:
        cohort = frame.loc[frame[f"value_cohort_{horizon}m"]]
        ratio = cohort[f"target_value_ratio_{horizon}m"]
        rows.append({
            "horizon": f"{horizon}m", "endpoint": "value_ratio", "kind": "continuous", "records": len(cohort),
            "mean": ratio.mean(), "std": ratio.std(), "p10": ratio.quantile(.1), "p25": ratio.quantile(.25),
            "median": ratio.median(), "p75": ratio.quantile(.75), "p90": ratio.quantile(.9),
            "positive_rows": np.nan, "positive_rate": np.nan,
        })
        for threshold in [10, 25, 50]:
            target = cohort[f"target_downside_{threshold}pct_{horizon}m"]
            rows.append({
                "horizon": f"{horizon}m", "endpoint": f"downside_{threshold}pct", "kind": "binary", "records": len(cohort),
                "mean": target.mean(), "std": np.nan, "p10": np.nan, "p25": np.nan, "median": np.nan, "p75": np.nan, "p90": np.nan,
                "positive_rows": int(target.sum()), "positive_rate": target.mean(),
            })
        timing.append({
            "horizon": f"{horizon}m", "eligible_rows": len(cohort),
            "at_signing_max_abs_gap_days": float(pd.to_numeric(cohort["at_signing_gap_days"], errors="coerce").abs().max()),
            "post_max_abs_gap_days": float(pd.to_numeric(cohort[f"post{horizon}_gap_days"], errors="coerce").abs().max()),
            "at_signing_tolerance_days": 365, "post_tolerance_days": 90,
        })
    return pd.DataFrame(rows), pd.DataFrame(timing)


def feature_manifest() -> pd.DataFrame:
    rows = []
    for variant, features in FEATURE_SETS.items():
        for feature in features:
            rows.append({
                "model_variant": variant, "feature": feature,
                "block": "contract_economics" if variant.startswith(("M4_", "M5_", "M6_")) and feature not in FEATURE_SETS["M3_plus_performance_history"] else "player_history",
                "timing": "known_at_extension_signing",
            })
    return pd.DataFrame(rows)


def decisions(comparison: pd.DataFrame, bands: pd.DataFrame, metrics: pd.DataFrame, ordering: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint, spec in ENDPOINTS.items():
        total = comparison.loc[comparison["endpoint"].eq(endpoint) & comparison["window"].eq("pooled_four") & comparison["comparison"].eq("M5_vs_M0_total")].iloc[0]
        contract = comparison.loc[comparison["endpoint"].eq(endpoint) & comparison["window"].eq("pooled_four") & comparison["comparison"].eq("M5_vs_M3_all_contract_economics")].iloc[0]
        nonlinear = comparison.loc[comparison["endpoint"].eq(endpoint) & comparison["window"].eq("pooled_four") & comparison["comparison"].eq("M6_vs_M5_nonlinearity")].iloc[0]
        monotone = bool(bands.loc[bands["endpoint"].eq(endpoint), "observed_monotone"].iloc[0])
        endpoint_metrics = metrics.loc[metrics["endpoint"].eq(endpoint) & metrics["model_variant"].eq("M5_plus_relative_financial_context")]
        if spec["kind"] == "continuous":
            performance = float(np.average(endpoint_metrics["evaluation_mae"], weights=endpoint_metrics["evaluation_rows"]))
            secondary = float(np.average(endpoint_metrics["evaluation_spearman"], weights=endpoint_metrics["evaluation_rows"]))
            calibrated = True
            supported = total["cluster_ci_lower_95"] > 0 and total["origin_wins"] >= 3 and monotone
            status = "advance_broad_value_range_candidate" if supported else "descriptive_or_exploratory_only"
            display = f"Expected {spec['horizon']} market-value preservation range" if supported else "none"
        else:
            performance = float(np.average(endpoint_metrics["evaluation_brier"], weights=endpoint_metrics["evaluation_rows"]))
            secondary = float(np.average(endpoint_metrics["evaluation_roc_auc"], weights=endpoint_metrics["evaluation_rows"]))
            calibration_error = float(np.average(endpoint_metrics["evaluation_predicted_rate"] - endpoint_metrics["evaluation_actual_rate"], weights=endpoint_metrics["evaluation_rows"]))
            calibrated = abs(calibration_error) <= .05
            supported = total["cluster_ci_lower_95"] > 0 and total["origin_wins"] >= 3 and monotone and calibrated
            status = "advance_downside_probability_candidate" if supported else "descriptive_or_exploratory_only"
            threshold = endpoint.split("_")[1]
            display = f"{spec['horizon']} probability of market-value decline of at least {threshold.replace('pct','%')}" if supported else "none"
        ordering_row = ordering.loc[ordering["horizon"].eq(spec["horizon"]) & ordering["model_variant"].eq("M5_plus_relative_financial_context")].iloc[0]
        rows.append({
            "endpoint": endpoint, "kind": spec["kind"], "status": status, "recommended_display": display,
            "weighted_origin_primary_metric": performance, "weighted_origin_secondary_metric": secondary,
            "total_loss_improvement_vs_baseline": total["mean_loss_improvement"],
            "total_ci_lower_95": total["cluster_ci_lower_95"], "total_ci_upper_95": total["cluster_ci_upper_95"],
            "total_origin_wins": total["origin_wins"], "risk_bands_monotone": monotone,
            "mean_calibration_error_within_5_points": calibrated,
            "contract_economics_add_signal": bool(contract["cluster_ci_lower_95"] > 0 and contract["origin_wins"] >= 3),
            "contract_incremental_improvement": contract["mean_loss_improvement"],
            "contract_ci_lower_95": contract["cluster_ci_lower_95"], "contract_ci_upper_95": contract["cluster_ci_upper_95"],
            "nonlinear_adds_signal": bool(nonlinear["cluster_ci_lower_95"] > 0 and nonlinear["origin_wins"] >= 3),
            "nonlinear_incremental_improvement": nonlinear["mean_loss_improvement"],
            "raw_m5_order_violation_rows_all_thresholds": int(ordering_row["any_order_violation_rows"]),
        })
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    target_audit, timing_audit = target_audits(frame)
    predictions, metrics = fit_models(frame)
    comparison = comparisons(predictions)
    bands = risk_bands(predictions)
    ordering = threshold_ordering(predictions)
    subgroups = subgroup_stability(predictions)
    decision = decisions(comparison, bands, metrics, ordering)

    target_columns = [
        "capology_extension_event_id", "canonical_player_id", "player_name", "league", "canonical_position",
        "signed_date", "signed_year", "at_signing_market_value_eur", "value_base_cohort",
    ]
    for horizon in [12, 24]:
        target_columns.extend([
            f"post{horizon}_market_value_eur", f"value_cohort_{horizon}m",
            f"target_value_ratio_{horizon}m", f"target_log_value_ratio_{horizon}m",
            f"target_value_change_pct_{horizon}m", f"target_value_preserved_90pct_{horizon}m",
            *[f"target_downside_{threshold}pct_{horizon}m" for threshold in [10, 25, 50]],
        ])
    target_columns.extend(list(dict.fromkeys(feature for values in FEATURE_SETS.values() for feature in values)))
    matrix = frame[target_columns].copy()

    checks = []
    def add(name: str, observed: object, expected: object, passed: bool, notes: str) -> None:
        checks.append({"check": name, "observed": observed, "expected": expected, "passed": bool(passed), "notes": notes})
    add("source_event_ids_unique", int(frame["capology_extension_event_id"].duplicated().sum()), 0, not frame["capology_extension_event_id"].duplicated().any(), "One row per extension event.")
    add("value_base_cohort", int(frame["value_base_cohort"].sum()), 2285, int(frame["value_base_cohort"].sum()) == 2285, "Linked event with strict positive signing valuation.")
    add("value_12m_cohort", int(frame["value_cohort_12m"].sum()), 2162, int(frame["value_cohort_12m"].sum()) == 2162, "Strict positive 12-month valuation.")
    add("value_24m_cohort", int(frame["value_cohort_24m"].sum()), 1944, int(frame["value_cohort_24m"].sum()) == 1944, "Strict positive 24-month valuation.")
    nested = True
    for horizon in [12, 24]:
        cohort = frame.loc[frame[f"value_cohort_{horizon}m"]]
        nested &= cohort[f"target_downside_50pct_{horizon}m"].le(cohort[f"target_downside_25pct_{horizon}m"]).all()
        nested &= cohort[f"target_downside_25pct_{horizon}m"].le(cohort[f"target_downside_10pct_{horizon}m"]).all()
    add("target_thresholds_nested", int(nested), 1, nested, "Severe downside implies moderate downside.")
    timing_ok = timing_audit["at_signing_max_abs_gap_days"].le(timing_audit["at_signing_tolerance_days"]).all() and timing_audit["post_max_abs_gap_days"].le(timing_audit["post_tolerance_days"]).all()
    add("valuation_timing_tolerances", int(timing_ok), 1, timing_ok, "All eligible landmarks satisfy declared gap tolerances.")
    add("prediction_key_unique", int(predictions.duplicated(["endpoint", "origin_id", "model_variant", "capology_extension_event_id"]).sum()), 0, not predictions.duplicated(["endpoint", "origin_id", "model_variant", "capology_extension_event_id"]).any(), "One prediction per endpoint/origin/model/event.")
    probabilities = predictions.loc[predictions["kind"].eq("binary"), "prediction"]
    add("probability_bounds", int(probabilities.between(0, 1).sum()), len(probabilities), probabilities.between(0, 1).all(), "Every downside forecast is a valid probability.")
    safe = feature_manifest()
    timing_features = safe["timing"].eq("known_at_extension_signing").all() and not safe["feature"].str.startswith("post").any()
    add("feature_timing_safe", int(timing_features), 1, timing_features, "No future valuation or post-extension performance feature enters the model.")
    add("decision_rows", len(decision), 8, len(decision) == 8, "One 12- and 24-month decision per continuous/ordered downside endpoint.")
    checks_frame = pd.DataFrame(checks)

    write_csv(matrix, OUTPUT / "value_preservation_model_matrix.csv")
    write_csv(target_audit, OUTPUT / "value_target_audit.csv")
    write_csv(timing_audit, OUTPUT / "valuation_timing_audit.csv")
    write_csv(feature_manifest(), OUTPUT / "feature_manifest.csv")
    write_csv(metrics, OUTPUT / "model_origin_metrics.csv")
    write_csv(predictions, OUTPUT / "value_predictions.csv")
    write_csv(comparison, OUTPUT / "model_comparison_results.csv")
    write_csv(bands, OUTPUT / "value_risk_bands.csv")
    write_csv(ordering, OUTPUT / "probability_ordering_audit.csv")
    write_csv(subgroups, OUTPUT / "subgroup_stability.csv")
    write_csv(decision, OUTPUT / "value_decision_summary.csv")
    write_csv(checks_frame, OUTPUT / "build_checks.csv")
    write_csv(pd.DataFrame([
        {"source_file": SOURCE.relative_to(ROOT).as_posix(), "bytes": SOURCE.stat().st_size, "sha256": sha256(SOURCE)},
        {"source_file": UPSTREAM_VERIFY.relative_to(ROOT).as_posix(), "bytes": UPSTREAM_VERIFY.stat().st_size, "sha256": sha256(UPSTREAM_VERIFY)},
    ]), OUTPUT / "source_manifest.csv")

    summary = {
        "source_extension_events": len(frame), "value_base_cohort": int(frame["value_base_cohort"].sum()),
        "value_12m_cohort": int(frame["value_cohort_12m"].sum()), "value_24m_cohort": int(frame["value_cohort_24m"].sum()),
        "evaluation_years": [2020, 2021, 2022, 2023], "prediction_rows": len(predictions),
        "checks_passed": int(checks_frame["passed"].sum()), "checks_total": len(checks_frame),
        "all_checks_passed": bool(checks_frame["passed"].all()),
        "decisions": dict(zip(decision["endpoint"], decision["status"])),
    }
    (OUTPUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Extension market-value preservation diagnostic", "",
        "This Phase-3 diagnostic tests 12- and 24-month Transfermarkt-style market-value change after an extension. It does not estimate sale proceeds, profit, accounting value, or ROI.", "",
        "## Targets", "",
        "- Continuous: log(post-extension value / signing value).",
        "- Ordered downside: declines of at least 10%, 25%, and 50%.",
        "- Value preservation is the complement of at least 10% downside: retaining at least 90% of signing value.",
        "- Signing landmarks may be at most 365 days from the signing date; post landmarks may be at most 90 days from their target dates.", "",
        "## Cohort and validation", "",
        f"- 12-month cohort: {int(frame['value_cohort_12m'].sum()):,}; 24-month cohort: {int(frame['value_cohort_24m'].sum()):,}.",
        "- Four rolling origins evaluate 2020-2023; tuning occurs on the immediately preceding validation year.",
        "- Paired uncertainty uses 5,000 player-cluster bootstrap repetitions.", "",
        "## 12- and 24-month decisions", "",
    ]
    for row in decision.itertuples(index=False):
        metric_name = "MAE" if row.kind == "continuous" else "Brier"
        secondary = "Spearman" if row.kind == "continuous" else "AUC"
        lines.append(
            f"- `{row.endpoint}`: `{row.status}`; {metric_name} {row.weighted_origin_primary_metric:.4f}, {secondary} {row.weighted_origin_secondary_metric:.3f}; "
            f"baseline improvement {row.total_loss_improvement_vs_baseline:.5f} (95% CI [{row.total_ci_lower_95:.5f}, {row.total_ci_upper_95:.5f}]); "
            f"contract economics add signal: `{row.contract_economics_add_signal}`."
        )
    lines.extend(["", "Probability thresholds were fit independently for signal testing. Inspect the ordering audit and apply validation-only calibration plus monotone ordering before any joint live display."])
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    files = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name != "output_manifest.csv")
    write_csv(pd.DataFrame([{"output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files]), OUTPUT / "output_manifest.csv")
    if not checks_frame["passed"].all():
        raise RuntimeError(f"Failed build checks: {checks_frame.loc[~checks_frame['passed'], 'check'].tolist()}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

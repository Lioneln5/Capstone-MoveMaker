"""Phase-2 diagnostic for post-extension club-spell survival.

The diagnostic separates uninterrupted club involvement (any outbound event
ends the spell) from permanent club departure (loans do not end the spell).
It fits discrete-time hazard models so right-censored extensions contribute
their observed person-periods, and compares them with direct fixed-horizon
classifiers on identical chronological evaluation cohorts.
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "models") not in sys.path:
    sys.path.insert(0, str(ROOT / "models"))
from mixed_type_preprocessor import fit_preprocessor
import run_extension_opportunity_diagnostic as phase1


SOURCE = phase1.SOURCE
UPSTREAM_VERIFY = ROOT / "Data" / "processed" / "contract_extension_integration" / "independent_verification.json"
OUTPUT = ROOT / "Data" / "processed" / "extension_survival_diagnostic"
AS_OF_DATE = pd.Timestamp("2026-08-10")
RANDOM_SEED = 20260811
BOOTSTRAP_REPETITIONS = 5000
C_GRID = [0.01, 0.1, 1.0, 10.0]
INTERVAL_BOUNDS = [0, 182, 365, 547, 730, 912, 1095, 1277, 1460]
HORIZONS = {"12m": 365, "24m": 730, "36m": 1095, "48m": 1460}
PRIMARY_HORIZONS = ["24m", "36m"]
ORIGINS = phase1.ORIGINS
WINDOWS = phase1.WINDOWS

ENDPOINTS = {
    "any_outbound": {
        "date": "first_any_outbound_date",
        "days": "first_any_outbound_days",
        "description": "First outbound move after extension; loans and permanent moves both interrupt continuous club involvement.",
        "display": "Continuous club-involvement survival probability",
    },
    "permanent_outbound": {
        "date": "first_permanent_outbound_date",
        "days": "first_permanent_outbound_days",
        "description": "First permanent outbound transfer after extension; intervening loans do not end the permanent relationship.",
        "display": "Permanent club-stay survival probability",
    },
}

FEATURE_SETS = dict(phase1.FEATURE_SETS)
SURVIVAL_VARIANTS = list(FEATURE_SETS)
DIRECT_VARIANT = "D5_direct_horizon_classifier"

COMPARISONS = [
    ("M1_vs_M0", "M0_chronological_baseline", "M1_player_market_profile"),
    ("M2_vs_M1", "M1_player_market_profile", "M2_plus_prior_opportunity"),
    ("M3_vs_M2", "M2_plus_prior_opportunity", "M3_plus_performance_history"),
    ("M4_vs_M3_contract_terms", "M3_plus_performance_history", "M4_plus_contract_terms"),
    ("M5_vs_M4_relative_financial", "M4_plus_contract_terms", "M5_plus_relative_financial_context"),
    ("M5_vs_M3_all_contract_economics", "M3_plus_performance_history", "M5_plus_relative_financial_context"),
    ("M5_vs_M0_total", "M0_chronological_baseline", "M5_plus_relative_financial_context"),
    ("M6_vs_M5_nonlinearity", "M5_plus_relative_financial_context", "M6_compact_nonlinear"),
    ("M5_survival_vs_D5_direct", DIRECT_VARIANT, "M5_plus_relative_financial_context"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: str) -> int:
    token = "|".join(parts).encode("utf-8")
    return (RANDOM_SEED + int(hashlib.sha256(token).hexdigest()[:8], 16)) % (2**32 - 1)


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").fillna("").str.strip().str.casefold().isin({"true", "1", "yes"})


def load_frame() -> pd.DataFrame:
    upstream = json.loads(UPSTREAM_VERIFY.read_text(encoding="utf-8"))
    if not upstream.get("all_checks_passed"):
        raise RuntimeError("Contract extension integration has not passed independent verification.")
    frame = phase1.load_frame()
    for column in ["signed_date", "expiration_date", "first_any_outbound_date", "first_permanent_outbound_date"]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["signed_year"] = frame["signed_date"].dt.year.astype("Int64")
    frame["administrative_followup_days"] = (AS_OF_DATE - frame["signed_date"]).dt.days
    frame["survival_primary_cohort"] = (
        frame["canonical_player_id"].notna()
        & frame["signed_date"].notna()
        & frame["expiration_date"].notna()
        & frame["signed_date"].lt(AS_OF_DATE)
        & frame["administrative_followup_days"].gt(0)
    )
    frame["age_band"] = pd.cut(
        frame["age_at_signing"], [-np.inf, 21, 25, 29, 33, np.inf],
        labels=["<=21", "22-25", "26-29", "30-33", "34+"],
    )
    frame["contract_length_band"] = pd.cut(
        frame["exact_duration_years"], [-np.inf, 2, 3, 4, 5, np.inf],
        labels=["<2y", "2-<3y", "3-<4y", "4-<5y", "5y+"],
    )
    for endpoint, spec in ENDPOINTS.items():
        event_date = frame[spec["date"]]
        event = event_date.notna() & event_date.le(AS_OF_DATE)
        event_days = (event_date - frame["signed_date"]).dt.days
        frame[f"{endpoint}_event"] = event.astype(int)
        frame[f"{endpoint}_event_days"] = event_days.where(event)
        frame[f"{endpoint}_followup_days"] = event_days.where(event, frame["administrative_followup_days"])
        frame[f"{endpoint}_event_before_or_on_expiry"] = (event & event_date.le(frame["expiration_date"])).astype(int)
        for label, horizon in HORIZONS.items():
            frame[f"{endpoint}_event_by_{label}"] = (event & event_days.le(horizon)).astype(int)
            frame[f"{endpoint}_horizon_observable_{label}"] = event | frame["administrative_followup_days"].ge(horizon)
    return frame


def person_periods(subjects: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    event_column = f"{endpoint}_event"
    event_days_column = f"{endpoint}_event_days"
    followup_column = f"{endpoint}_followup_days"
    base_columns = list(dict.fromkeys(
        ["capology_extension_event_id", "canonical_player_id", "signed_year"]
        + [feature for values in FEATURE_SETS.values() for feature in values]
    ))
    for row in subjects[base_columns + [event_column, event_days_column, followup_column]].to_dict("records"):
        followup = min(float(row[followup_column]), float(INTERVAL_BOUNDS[-1]))
        event_days = float(row[event_days_column]) if pd.notna(row[event_days_column]) else np.nan
        is_event = bool(row[event_column])
        for interval in range(1, len(INTERVAL_BOUNDS)):
            start, end = INTERVAL_BOUNDS[interval - 1], INTERVAL_BOUNDS[interval]
            event_here = is_event and pd.notna(event_days) and start < event_days <= end
            fully_observed = followup >= end
            if not event_here and not fully_observed:
                break
            period = {column: row.get(column) for column in base_columns}
            period["hazard_interval"] = f"period_{interval}"
            period["interval_start_days"] = start
            period["interval_end_days"] = end
            period["hazard_event"] = int(event_here)
            rows.append(period)
            if event_here:
                break
    return pd.DataFrame(rows)


def horizon_is_complete(subjects: pd.DataFrame, horizon: int) -> bool:
    return bool(subjects["administrative_followup_days"].ge(horizon).all())


def predictions_to_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    result = {
        "brier": float(brier_score_loss(actual, prediction)),
        "log_loss": float(log_loss(actual, prediction, labels=[0, 1])),
        "actual_rate": float(np.mean(actual)),
        "predicted_rate": float(np.mean(prediction)),
    }
    if len(np.unique(actual)) == 2:
        result["roc_auc"] = float(roc_auc_score(actual, prediction))
        result["average_precision"] = float(average_precision_score(actual, prediction))
    else:
        result["roc_auc"] = np.nan
        result["average_precision"] = np.nan
    return result


def predict_hazard_curve(model: Any, preprocessor: Any, subjects: pd.DataFrame, features: list[str]) -> np.ndarray:
    blocks = []
    for interval in range(1, len(INTERVAL_BOUNDS)):
        block = subjects.copy()
        block["hazard_interval"] = f"period_{interval}"
        blocks.append(block)
    expanded = pd.concat(blocks, ignore_index=True)
    hazard = np.clip(model.predict_proba(preprocessor.transform(expanded))[:, 1], 1e-6, 1 - 1e-6)
    hazard = hazard.reshape(len(INTERVAL_BOUNDS) - 1, len(subjects)).T
    return 1.0 - np.cumprod(1.0 - hazard, axis=1)


def horizon_column_index(horizon: int) -> int:
    return INTERVAL_BOUNDS[1:].index(horizon)


def validation_integrated_brier(subjects: pd.DataFrame, endpoint: str, cumulative: np.ndarray) -> float:
    losses = []
    for label, horizon in HORIZONS.items():
        if not horizon_is_complete(subjects, horizon):
            continue
        actual = subjects[f"{endpoint}_event_by_{label}"].to_numpy(int)
        losses.append(brier_score_loss(actual, cumulative[:, horizon_column_index(horizon)]))
    return float(np.mean(losses))


def fit_survival_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    evaluation: pd.DataFrame,
    endpoint: str,
    variant: str,
    features: list[str],
) -> tuple[np.ndarray, dict[str, Any], int, int]:
    train_periods = person_periods(train, endpoint)
    fit_features = features + ["hazard_interval"]
    pre = fit_preprocessor(train_periods, fit_features)
    x_train = pre.transform(train_periods)
    y_train = train_periods["hazard_event"].to_numpy(int)
    nonlinear = variant == "M6_compact_nonlinear"
    candidates = []
    if nonlinear:
        grid = [
            {"n_estimators": 75, "learning_rate": 0.03, "max_depth": 1, "min_samples_leaf": 30},
            {"n_estimators": 100, "learning_rate": 0.03, "max_depth": 2, "min_samples_leaf": 40},
            {"n_estimators": 75, "learning_rate": 0.05, "max_depth": 2, "min_samples_leaf": 50},
        ]
        for i, params in enumerate(grid):
            model = GradientBoostingClassifier(loss="log_loss", random_state=stable_seed(endpoint, variant, str(i)), **params).fit(x_train, y_train)
            cumulative = predict_hazard_curve(model, pre, validation, features)
            candidates.append((validation_integrated_brier(validation, endpoint, cumulative), params))
    else:
        for c_value in C_GRID:
            model = LogisticRegression(C=c_value, penalty="l2", solver="liblinear", max_iter=3000, random_state=RANDOM_SEED).fit(x_train, y_train)
            cumulative = predict_hazard_curve(model, pre, validation, features)
            candidates.append((validation_integrated_brier(validation, endpoint, cumulative), {"C": c_value}))
    validation_score, params = min(candidates, key=lambda item: item[0])

    fit_subjects = pd.concat([train, validation], ignore_index=True)
    fit_periods = person_periods(fit_subjects, endpoint)
    fit_pre = fit_preprocessor(fit_periods, fit_features)
    x_fit = fit_pre.transform(fit_periods)
    y_fit = fit_periods["hazard_event"].to_numpy(int)
    if nonlinear:
        model = GradientBoostingClassifier(loss="log_loss", random_state=stable_seed(endpoint, variant, "fit"), **params)
    else:
        model = LogisticRegression(C=params["C"], penalty="l2", solver="liblinear", max_iter=3000, random_state=RANDOM_SEED)
    model.fit(x_fit, y_fit)
    cumulative = predict_hazard_curve(model, fit_pre, evaluation, features)
    return cumulative, {"selected_parameter": json.dumps(params, sort_keys=True), "validation_integrated_brier": validation_score}, x_fit.shape[1], len(fit_periods)


def fit_direct_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    evaluation: pd.DataFrame,
    endpoint: str,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]], int]:
    features = FEATURE_SETS["M5_plus_relative_financial_context"]
    results, metadata, encoded = {}, {}, 0
    for label, horizon in HORIZONS.items():
        if not horizon_is_complete(train, horizon) or not horizon_is_complete(validation, horizon):
            continue
        target = f"{endpoint}_event_by_{label}"
        pre = fit_preprocessor(train, features)
        x_train, x_val = pre.transform(train), pre.transform(validation)
        y_train, y_val = train[target].to_numpy(int), validation[target].to_numpy(int)
        candidates = []
        for c_value in C_GRID:
            model = LogisticRegression(C=c_value, penalty="l2", solver="liblinear", max_iter=3000, random_state=RANDOM_SEED).fit(x_train, y_train)
            estimate = np.clip(model.predict_proba(x_val)[:, 1], 1e-6, 1 - 1e-6)
            candidates.append((brier_score_loss(y_val, estimate), c_value))
        validation_brier, c_value = min(candidates)
        fit = pd.concat([train, validation], ignore_index=True)
        fit_pre = fit_preprocessor(fit, features)
        model = LogisticRegression(C=c_value, penalty="l2", solver="liblinear", max_iter=3000, random_state=RANDOM_SEED).fit(fit_pre.transform(fit), fit[target].to_numpy(int))
        results[label] = np.clip(model.predict_proba(fit_pre.transform(evaluation))[:, 1], 1e-6, 1 - 1e-6)
        metadata[label] = {"selected_parameter": json.dumps({"C": c_value}), "validation_brier": validation_brier}
        encoded = fit_pre.transform(fit).shape[1]
    return results, metadata, encoded


def fit_all(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cohort = frame.loc[frame["survival_primary_cohort"]].copy()
    prediction_rows, metric_rows, period_rows = [], [], []
    for endpoint in ENDPOINTS:
        for origin, train_end, validation_year, evaluation_year in ORIGINS:
            train = cohort.loc[cohort["signed_year"].le(train_end)].copy()
            validation = cohort.loc[cohort["signed_year"].eq(validation_year)].copy()
            evaluation = cohort.loc[cohort["signed_year"].eq(evaluation_year)].sort_values("capology_extension_event_id").copy()
            if min(len(train), len(validation), len(evaluation)) < 30:
                raise RuntimeError(f"Insufficient split {endpoint}/{origin}: {len(train)}/{len(validation)}/{len(evaluation)}")
            for variant, features in FEATURE_SETS.items():
                cumulative, tuning, encoded, fit_period_count = fit_survival_model(train, validation, evaluation, endpoint, variant, features)
                period_rows.append({
                    "endpoint": endpoint, "origin_id": origin, "model_variant": variant,
                    "train_subjects": len(train), "validation_subjects": len(validation),
                    "fit_subjects": len(train) + len(validation), "fit_person_periods": fit_period_count,
                    "encoded_feature_count": encoded,
                })
                for label, horizon in HORIZONS.items():
                    estimate = cumulative[:, horizon_column_index(horizon)]
                    complete = horizon_is_complete(evaluation, horizon)
                    actual = evaluation[f"{endpoint}_event_by_{label}"].to_numpy(int) if complete else np.repeat(np.nan, len(evaluation))
                    if complete:
                        scores = predictions_to_metrics(actual.astype(int), estimate)
                        metric_rows.append({
                            "endpoint": endpoint, "origin_id": origin, "model_variant": variant,
                            "train_end_year": train_end, "validation_year": validation_year,
                            "evaluation_year": evaluation_year, "horizon": label, "horizon_days": horizon,
                            "evaluation_rows": len(evaluation), "horizon_fully_observable": True,
                            "raw_feature_count": len(features), "encoded_feature_count": encoded,
                            **tuning, **scores,
                        })
                    for i, (_, row) in enumerate(evaluation.iterrows()):
                        value = float(actual[i]) if complete else np.nan
                        prediction_rows.append({
                            "endpoint": endpoint, "origin_id": origin, "evaluation_year": evaluation_year,
                            "horizon": label, "horizon_days": horizon,
                            "horizon_fully_observable": complete,
                            "capology_extension_event_id": row["capology_extension_event_id"],
                            "canonical_player_id": row["canonical_player_id"], "league": row["league"],
                            "canonical_position": row["canonical_position"], "age_band": row["age_band"],
                            "contract_length_band": row["contract_length_band"],
                            "model_variant": variant, "actual_event_by_horizon": value,
                            "predicted_event_probability": float(estimate[i]),
                            "predicted_survival_probability": 1.0 - float(estimate[i]),
                            "primary_loss": (value - float(estimate[i])) ** 2 if complete else np.nan,
                        })
            direct, direct_metadata, direct_encoded = fit_direct_models(train, validation, evaluation, endpoint)
            for label, horizon in HORIZONS.items():
                if label not in direct:
                    continue
                estimate = direct[label]
                complete = horizon_is_complete(evaluation, horizon)
                actual = evaluation[f"{endpoint}_event_by_{label}"].to_numpy(int) if complete else np.repeat(np.nan, len(evaluation))
                if complete:
                    scores = predictions_to_metrics(actual.astype(int), estimate)
                    metric_rows.append({
                        "endpoint": endpoint, "origin_id": origin, "model_variant": DIRECT_VARIANT,
                        "train_end_year": train_end, "validation_year": validation_year,
                        "evaluation_year": evaluation_year, "horizon": label, "horizon_days": horizon,
                        "evaluation_rows": len(evaluation), "horizon_fully_observable": True,
                        "raw_feature_count": len(FEATURE_SETS["M5_plus_relative_financial_context"]),
                        "encoded_feature_count": direct_encoded,
                        "selected_parameter": direct_metadata[label]["selected_parameter"],
                        "validation_integrated_brier": np.nan,
                        "validation_brier": direct_metadata[label]["validation_brier"], **scores,
                    })
                for i, (_, row) in enumerate(evaluation.iterrows()):
                    value = float(actual[i]) if complete else np.nan
                    prediction_rows.append({
                        "endpoint": endpoint, "origin_id": origin, "evaluation_year": evaluation_year,
                        "horizon": label, "horizon_days": horizon,
                        "horizon_fully_observable": complete,
                        "capology_extension_event_id": row["capology_extension_event_id"],
                        "canonical_player_id": row["canonical_player_id"], "league": row["league"],
                        "canonical_position": row["canonical_position"], "age_band": row["age_band"],
                        "contract_length_band": row["contract_length_band"],
                        "model_variant": DIRECT_VARIANT, "actual_event_by_horizon": value,
                        "predicted_event_probability": float(estimate[i]),
                        "predicted_survival_probability": 1.0 - float(estimate[i]),
                        "primary_loss": (value - float(estimate[i])) ** 2 if complete else np.nan,
                    })
    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows), pd.DataFrame(period_rows)


def paired(predictions: pd.DataFrame, endpoint: str, horizon: str, baseline: str, candidate: str, years: list[int]) -> pd.DataFrame:
    subset = predictions.loc[
        predictions["endpoint"].eq(endpoint)
        & predictions["horizon"].eq(horizon)
        & predictions["evaluation_year"].isin(years)
        & predictions["horizon_fully_observable"].eq(True)
    ]
    keys = ["endpoint", "origin_id", "evaluation_year", "horizon", "capology_extension_event_id", "canonical_player_id"]
    left = subset.loc[subset["model_variant"].eq(baseline), keys + ["primary_loss"]].rename(columns={"primary_loss": "baseline_loss"})
    right = subset.loc[subset["model_variant"].eq(candidate), keys + ["primary_loss"]].rename(columns={"primary_loss": "candidate_loss"})
    return left.merge(right, on=keys, how="inner", validate="one_to_one")


def cluster_bootstrap(pairs: pd.DataFrame, seed: int) -> dict[str, float]:
    pairs = pairs.assign(improvement=pairs["baseline_loss"] - pairs["candidate_loss"])
    clusters = pairs.groupby("canonical_player_id")["improvement"].agg(["sum", "size"])
    sums, sizes = clusters["sum"].to_numpy(float), clusters["size"].to_numpy(float)
    rng, draws = np.random.default_rng(seed), []
    for start in range(0, BOOTSTRAP_REPETITIONS, 250):
        count = min(250, BOOTSTRAP_REPETITIONS - start)
        indices = rng.integers(0, len(clusters), size=(count, len(clusters)))
        draws.extend((sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)).tolist())
    values = np.asarray(draws)
    return {
        "cluster_count": len(clusters),
        "cluster_ci_lower_95": float(np.quantile(values, 0.025)),
        "cluster_ci_upper_95": float(np.quantile(values, 0.975)),
        "cluster_probability_candidate_better": float(np.mean(values > 0)),
    }


def comparisons(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint in ENDPOINTS:
        for horizon in ["12m", "24m", "36m"]:
            for window, years in WINDOWS.items():
                for name, baseline, candidate in COMPARISONS:
                    pairs = paired(predictions, endpoint, horizon, baseline, candidate, years)
                    if pairs.empty:
                        continue
                    delta = pairs["baseline_loss"] - pairs["candidate_loss"]
                    origin_delta = pairs.assign(delta=delta).groupby("evaluation_year")["delta"].mean()
                    rows.append({
                        "endpoint": endpoint, "horizon": horizon, "window": window,
                        "comparison": name, "baseline": baseline, "candidate": candidate,
                        "evaluation_years": " | ".join(map(str, sorted(pairs["evaluation_year"].unique()))),
                        "evaluation_rows": len(pairs), "mean_brier_improvement": float(delta.mean()),
                        "origin_wins": int(origin_delta.gt(0).sum()), "origin_count": len(origin_delta),
                        **cluster_bootstrap(pairs, stable_seed(endpoint, horizon, window, name)),
                    })
    return pd.DataFrame(rows)


def risk_bands(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint in ENDPOINTS:
        for horizon in ["12m", "24m", "36m"]:
            subset = predictions.loc[
                predictions["endpoint"].eq(endpoint)
                & predictions["horizon"].eq(horizon)
                & predictions["horizon_fully_observable"].eq(True)
                & predictions["model_variant"].eq("M5_plus_relative_financial_context")
            ].copy()
            subset["risk_band"] = pd.qcut(subset["predicted_event_probability"].rank(method="first"), 5, labels=False) + 1
            grouped = subset.groupby("risk_band", observed=True).agg(
                records=("actual_event_by_horizon", "size"),
                mean_prediction=("predicted_event_probability", "mean"),
                observed_event_rate=("actual_event_by_horizon", "mean"),
                mean_brier=("primary_loss", "mean"),
            ).reset_index()
            monotone = bool(grouped["observed_event_rate"].is_monotonic_increasing)
            grouped.insert(0, "horizon", horizon)
            grouped.insert(0, "endpoint", endpoint)
            grouped["observed_monotone"] = monotone
            rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def subgroup_stability(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    source = predictions.loc[
        predictions["horizon_fully_observable"].eq(True)
        & predictions["model_variant"].eq("M5_plus_relative_financial_context")
        & predictions["horizon"].isin(PRIMARY_HORIZONS)
    ]
    for (endpoint, horizon), data in source.groupby(["endpoint", "horizon"], observed=True):
        for group_type, column in [("league", "league"), ("position", "canonical_position"), ("age_band", "age_band"), ("contract_length", "contract_length_band")]:
            for group, subset in data.groupby(column, observed=True, dropna=False):
                if len(subset) < 30:
                    continue
                actual = subset["actual_event_by_horizon"].to_numpy(int)
                estimate = subset["predicted_event_probability"].to_numpy(float)
                metrics = predictions_to_metrics(actual, estimate)
                rows.append({"endpoint": endpoint, "horizon": horizon, "group_type": group_type, "group": str(group), "records": len(subset), **metrics})
    return pd.DataFrame(rows)


def target_audits(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cohort = frame.loc[frame["survival_primary_cohort"]].copy()
    audit_rows, horizon_rows, expiry_rows = [], [], []
    for endpoint, spec in ENDPOINTS.items():
        event = cohort[f"{endpoint}_event"].eq(1)
        audit_rows.append({
            "endpoint": endpoint, "records": len(cohort), "events": int(event.sum()),
            "censored": int((~event).sum()), "event_rate": float(event.mean()),
            "median_event_days": float(cohort.loc[event, f"{endpoint}_event_days"].median()),
            "description": spec["description"],
        })
        for label, horizon in HORIZONS.items():
            observable = cohort[f"{endpoint}_horizon_observable_{label}"]
            target = cohort[f"{endpoint}_event_by_{label}"]
            horizon_rows.append({
                "endpoint": endpoint, "horizon": label, "horizon_days": horizon,
                "records": len(cohort), "observable_or_event_rows": int(observable.sum()),
                "events_by_horizon": int(target.sum()),
                "event_rate_among_observable_or_event": float(target.loc[observable].mean()),
                "administratively_complete_for_full_cohort": bool(cohort["administrative_followup_days"].ge(horizon).all()),
            })
        event_date = cohort[spec["date"]]
        expiry_rows.append({
            "endpoint": endpoint, "events": int(event.sum()),
            "events_before_or_on_expiry": int((event & event_date.le(cohort["expiration_date"])).sum()),
            "events_after_expiry": int((event & event_date.gt(cohort["expiration_date"])).sum()),
            "expiration_used_as_censor": False,
            "reason": "Outcome is club stay, which can continue after a renewal; scheduled expiration is a signing-time predictor and audit boundary.",
        })
    return pd.DataFrame(audit_rows), pd.DataFrame(horizon_rows), pd.DataFrame(expiry_rows)


def feature_manifest() -> pd.DataFrame:
    rows = [{"model_variant": "all_survival_models", "feature": "hazard_interval", "block": "baseline_hazard", "timing": "elapsed_time_axis_not_player_information"}]
    for variant, features in FEATURE_SETS.items():
        for feature in features:
            rows.append({
                "model_variant": variant, "feature": feature,
                "block": "contract_economics" if variant.startswith(("M4_", "M5_", "M6_")) and feature not in FEATURE_SETS["M3_plus_performance_history"] else "player_history",
                "timing": "known_at_extension_signing",
            })
    for feature in FEATURE_SETS["M5_plus_relative_financial_context"]:
        rows.append({"model_variant": DIRECT_VARIANT, "feature": feature, "block": "direct_horizon_comparator", "timing": "known_at_extension_signing"})
    return pd.DataFrame(rows)


def decisions(comparison: pd.DataFrame, bands: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint, spec in ENDPOINTS.items():
        for horizon in PRIMARY_HORIZONS:
            total = comparison.loc[
                comparison["endpoint"].eq(endpoint) & comparison["horizon"].eq(horizon)
                & comparison["window"].eq("pooled_four") & comparison["comparison"].eq("M5_vs_M0_total")
            ].iloc[0]
            contract = comparison.loc[
                comparison["endpoint"].eq(endpoint) & comparison["horizon"].eq(horizon)
                & comparison["window"].eq("pooled_four") & comparison["comparison"].eq("M5_vs_M3_all_contract_economics")
            ].iloc[0]
            method = comparison.loc[
                comparison["endpoint"].eq(endpoint) & comparison["horizon"].eq(horizon)
                & comparison["window"].eq("pooled_four") & comparison["comparison"].eq("M5_survival_vs_D5_direct")
            ].iloc[0]
            monotone = bool(bands.loc[bands["endpoint"].eq(endpoint) & bands["horizon"].eq(horizon), "observed_monotone"].iloc[0])
            pooled = metrics.loc[
                metrics["endpoint"].eq(endpoint) & metrics["horizon"].eq(horizon)
                & metrics["model_variant"].eq("M5_plus_relative_financial_context")
            ]
            weighted_brier = float(np.average(pooled["brier"], weights=pooled["evaluation_rows"]))
            weighted_auc = float(np.average(pooled["roc_auc"], weights=pooled["evaluation_rows"]))
            calibrated = abs(float(np.average(pooled["predicted_rate"] - pooled["actual_rate"], weights=pooled["evaluation_rows"]))) <= 0.05
            supported = total["cluster_ci_lower_95"] > 0 and total["origin_wins"] >= 3 and monotone and calibrated
            rows.append({
                "endpoint": endpoint, "horizon": horizon,
                "status": "advance_survival_probability_candidate" if supported else "descriptive_or_exploratory_only",
                "recommended_display": f"{spec['display']} at {horizon}" if supported else "none",
                "weighted_origin_m5_brier": weighted_brier, "weighted_origin_m5_auc": weighted_auc,
                "total_brier_improvement_vs_baseline": total["mean_brier_improvement"],
                "total_ci_lower_95": total["cluster_ci_lower_95"], "total_ci_upper_95": total["cluster_ci_upper_95"],
                "total_origin_wins": total["origin_wins"], "risk_bands_monotone": monotone,
                "mean_calibration_error_within_5_points": calibrated,
                "contract_economics_add_signal": bool(contract["cluster_ci_lower_95"] > 0 and contract["origin_wins"] >= 3),
                "contract_brier_improvement": contract["mean_brier_improvement"],
                "contract_ci_lower_95": contract["cluster_ci_lower_95"], "contract_ci_upper_95": contract["cluster_ci_upper_95"],
                "survival_beats_direct_classifier": bool(method["cluster_ci_lower_95"] > 0 and method["origin_wins"] >= 3),
                "survival_vs_direct_brier_improvement": method["mean_brier_improvement"],
                "survival_vs_direct_ci_lower_95": method["cluster_ci_lower_95"], "survival_vs_direct_ci_upper_95": method["cluster_ci_upper_95"],
            })
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    cohort = frame.loc[frame["survival_primary_cohort"]].copy()
    target_audit, horizon_audit, expiry_audit = target_audits(frame)
    predictions, metrics, period_audit = fit_all(frame)
    comparison = comparisons(predictions)
    bands = risk_bands(predictions)
    subgroups = subgroup_stability(predictions)
    decision = decisions(comparison, bands, metrics)

    matrix_columns = [
        "capology_extension_event_id", "canonical_player_id", "player_name", "league", "canonical_position",
        "signed_date", "signed_year", "expiration_date", "administrative_followup_days", "survival_primary_cohort",
    ]
    for endpoint in ENDPOINTS:
        matrix_columns.extend([
            f"{endpoint}_event", f"{endpoint}_event_days", f"{endpoint}_followup_days",
            f"{endpoint}_event_before_or_on_expiry",
            *[f"{endpoint}_event_by_{label}" for label in HORIZONS],
            *[f"{endpoint}_horizon_observable_{label}" for label in HORIZONS],
        ])
    matrix_columns.extend(list(dict.fromkeys(feature for values in FEATURE_SETS.values() for feature in values)))
    matrix = frame[matrix_columns].copy()

    checks = []
    def add(name: str, observed: object, expected: object, passed: bool, notes: str) -> None:
        checks.append({"check": name, "observed": observed, "expected": expected, "passed": bool(passed), "notes": notes})
    add("source_event_ids_unique", int(frame["capology_extension_event_id"].duplicated().sum()), 0, not frame["capology_extension_event_id"].duplicated().any(), "One row per clean extension event.")
    add("survival_primary_cohort", len(cohort), 2358, len(cohort) == 2358, "Canonical identity and valid follow-up required.")
    add("any_outbound_events", int(cohort["any_outbound_event"].sum()), 1848, int(cohort["any_outbound_event"].sum()) == 1848, "Dated any-outbound events by the administrative cutoff.")
    add("permanent_outbound_events", int(cohort["permanent_outbound_event"].sum()), 1690, int(cohort["permanent_outbound_event"].sum()) == 1690, "Dated permanent-outbound events by the cutoff.")
    dates_valid = all((cohort.loc[cohort[f"{endpoint}_event"].eq(1), f"{endpoint}_event_days"] > 0).all() for endpoint in ENDPOINTS)
    add("event_dates_after_signing", int(dates_valid), 1, dates_valid, "Every modeled event occurs after extension signing.")
    nested = True
    for endpoint in ENDPOINTS:
        nested &= cohort[f"{endpoint}_event_by_12m"].le(cohort[f"{endpoint}_event_by_24m"]).all()
        nested &= cohort[f"{endpoint}_event_by_24m"].le(cohort[f"{endpoint}_event_by_36m"]).all()
        nested &= cohort[f"{endpoint}_event_by_36m"].le(cohort[f"{endpoint}_event_by_48m"]).all()
    add("horizon_targets_nested", int(nested), 1, nested, "Cumulative event targets never decrease over time.")
    add("prediction_key_unique", int(predictions.duplicated(["endpoint", "origin_id", "horizon", "model_variant", "capology_extension_event_id"]).sum()), 0, not predictions.duplicated(["endpoint", "origin_id", "horizon", "model_variant", "capology_extension_event_id"]).any(), "One prediction per endpoint/origin/horizon/model/event.")
    add("predictions_bounded", int(predictions["predicted_event_probability"].between(0, 1).sum()), len(predictions), predictions["predicted_event_probability"].between(0, 1).all(), "Every event probability is valid.")
    complements_ok = np.allclose(predictions["predicted_event_probability"] + predictions["predicted_survival_probability"], 1.0)
    add("event_survival_complements", int(complements_ok), 1, complements_ok, "Outbound risk and stay probability sum exactly to one.")
    ordering = predictions.loc[predictions["model_variant"].isin(SURVIVAL_VARIANTS)].pivot_table(index=["endpoint", "origin_id", "model_variant", "capology_extension_event_id"], columns="horizon_days", values="predicted_event_probability")
    order_ok = ordering[365].le(ordering[730]).all() and ordering[730].le(ordering[1095]).all() and ordering[1095].le(ordering[1460]).all()
    add("survival_horizon_ordering", int(order_ok), 1, order_ok, "Discrete hazards guarantee nondecreasing cumulative event probability.")
    safe_features = feature_manifest()
    timing_ok = safe_features.loc[safe_features["feature"].ne("hazard_interval"), "timing"].eq("known_at_extension_signing").all() and not safe_features["feature"].str.startswith("post_").any()
    add("feature_timing_safe", int(timing_ok), 1, timing_ok, "No post-extension player information enters the model.")
    add("decision_rows", len(decision), 4, len(decision) == 4, "Two endpoints at 24 and 36 months.")
    checks_frame = pd.DataFrame(checks)

    write_csv(matrix, OUTPUT / "survival_model_matrix.csv")
    write_csv(target_audit, OUTPUT / "survival_target_audit.csv")
    write_csv(horizon_audit, OUTPUT / "at_risk_horizon_audit.csv")
    write_csv(expiry_audit, OUTPUT / "contract_expiry_boundary_audit.csv")
    write_csv(period_audit, OUTPUT / "person_period_audit.csv")
    write_csv(feature_manifest(), OUTPUT / "feature_manifest.csv")
    write_csv(metrics, OUTPUT / "model_origin_horizon_metrics.csv")
    write_csv(predictions, OUTPUT / "survival_predictions.csv")
    write_csv(comparison, OUTPUT / "model_comparison_results.csv")
    write_csv(bands, OUTPUT / "survival_risk_bands.csv")
    write_csv(subgroups, OUTPUT / "subgroup_stability.csv")
    write_csv(decision, OUTPUT / "survival_decision_summary.csv")
    write_csv(checks_frame, OUTPUT / "build_checks.csv")
    write_csv(pd.DataFrame([
        {"source_file": SOURCE.relative_to(ROOT).as_posix(), "bytes": SOURCE.stat().st_size, "sha256": sha256(SOURCE)},
        {"source_file": UPSTREAM_VERIFY.relative_to(ROOT).as_posix(), "bytes": UPSTREAM_VERIFY.stat().st_size, "sha256": sha256(UPSTREAM_VERIFY)},
    ]), OUTPUT / "source_manifest.csv")

    summary = {
        "source_extension_events": len(frame), "survival_primary_cohort": len(cohort),
        "any_outbound_events": int(cohort["any_outbound_event"].sum()),
        "permanent_outbound_events": int(cohort["permanent_outbound_event"].sum()),
        "evaluation_years": [2020, 2021, 2022, 2023], "prediction_rows": len(predictions),
        "checks_passed": int(checks_frame["passed"].sum()), "checks_total": len(checks_frame),
        "all_checks_passed": bool(checks_frame["passed"].all()),
        "decisions": {f"{r.endpoint}_{r.horizon}": r.status for r in decision.itertuples(index=False)},
    }
    (OUTPUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Extension survival and time-to-outbound diagnostic", "",
        "This Phase-2 diagnostic estimates cumulative outbound risk with a discrete-time hazard model. It handles right-censoring and is not yet a live probability policy.", "",
        "## Definitions", "",
        f"- Administrative censoring date: {AS_OF_DATE.date()}.",
        "- Any outbound: a loan, loan return, or permanent move ends uninterrupted club involvement.",
        "- Permanent outbound: only a permanent transfer ends the permanent club relationship; loans do not.",
        "- Scheduled contract expiration is a signing-time predictor and audit boundary, not a censor, because a renewed player may remain at the club.",
        "- Eight half-year hazard intervals support cumulative 12-, 24-, 36-, and 48-month forecasts.", "",
        "## Cohort and validation", "",
        f"- Primary cohort: {len(cohort):,} linked extension events.",
        f"- Observed any-outbound events: {int(cohort['any_outbound_event'].sum()):,}; permanent-outbound events: {int(cohort['permanent_outbound_event'].sum()):,}.",
        "- Four rolling origins evaluate signing years 2020-2023; model selection uses the immediately preceding validation year.",
        "- A direct horizon-specific classifier is tested against the discrete-time survival formulation.",
        "- Paired uncertainty uses 5,000 player-cluster bootstrap repetitions.", "",
        "## Decisions", "",
    ]
    for row in decision.itertuples(index=False):
        lines.append(
            f"- `{row.endpoint}` at `{row.horizon}`: `{row.status}`; weighted-origin Brier {row.weighted_origin_m5_brier:.4f}, weighted-origin AUC {row.weighted_origin_m5_auc:.3f}, "
            f"baseline improvement {row.total_brier_improvement_vs_baseline:.5f} (95% CI [{row.total_ci_lower_95:.5f}, {row.total_ci_upper_95:.5f}]); "
            f"contract economics add signal: `{row.contract_economics_add_signal}`; survival beats direct classifier: `{row.survival_beats_direct_classifier}`."
        )
    lines.extend(["", "Use the decision summary, comparisons, risk bands, subgroup results, and independent verification before making product claims."])
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    files = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name != "output_manifest.csv")
    write_csv(pd.DataFrame([{"output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files]), OUTPUT / "output_manifest.csv")
    if not checks_frame["passed"].all():
        raise RuntimeError(f"Failed build checks: {checks_frame.loc[~checks_frame['passed'], 'check'].tolist()}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

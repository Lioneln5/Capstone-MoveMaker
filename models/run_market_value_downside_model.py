"""Model 24-month market-value change and downside risk with rolling origins.

The continuous target is log(post24 value / pre-transfer value), which avoids
letting extreme percentage appreciation dominate the fit.  Binary targets keep
their direct business definitions: declines of at least 10%, 25%, and 50%.
All predictors are available at transfer or derived strictly from pre-transfer
history.  Results authorize only market-value downside claims, never ROI.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
for dependency_dir in [ROOT / "models", ROOT / "scripts"]:
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

import run_coarse_club_context_diagnostic as coarse
import run_opportunity_transferability_decomposition as opportunity
from mixed_type_preprocessor import fit_preprocessor
from train_compatibility_models import (
    ALPHA_GRID,
    fit_ridge,
    metrics as regression_metrics,
    predict_ridge,
    sha256_file,
)


MASTER_PATH = ROOT / "Data" / "processed" / "transfermarkt_merged" / "transfermarkt_comprehensive_master.csv"
AUDIT_PATH = ROOT / "Data" / "processed" / "financial_module_feasibility" / "financial_event_audit.csv"
AUDIT_VERIFICATION_PATH = ROOT / "Data" / "processed" / "financial_module_feasibility" / "independent_verification.json"
OUTPUT_DIR = ROOT / "Data" / "processed" / "market_value_downside_model"

RANDOM_SEED = 20260810
BOOTSTRAP_REPETITIONS = 5000
ORIGINS = coarse.ORIGINS
WINDOWS = {
    "pooled_four": [2020, 2021, 2022, 2023],
    "mature_three": [2021, 2022, 2023],
    "terminal_two": [2022, 2023],
}
MODEL_ORDER = [
    "M0_chronological_constant",
    "M1_market_profile",
    "M2_plus_valuation_trajectory",
    "M3_plus_sporting_history",
    "M4_plus_destination_context",
    "M5_plus_origin_transition_context",
]
COMPARISONS = [
    ("M1_vs_M0_market_profile", MODEL_ORDER[0], MODEL_ORDER[1]),
    ("M2_vs_M1_valuation_trajectory", MODEL_ORDER[1], MODEL_ORDER[2]),
    ("M3_vs_M2_sporting_history", MODEL_ORDER[2], MODEL_ORDER[3]),
    ("M4_vs_M3_destination_context", MODEL_ORDER[3], MODEL_ORDER[4]),
    ("M5_vs_M4_origin_transition", MODEL_ORDER[4], MODEL_ORDER[5]),
    ("M5_vs_M1_all_incremental_history_context", MODEL_ORDER[1], MODEL_ORDER[5]),
    ("M5_vs_M0_total", MODEL_ORDER[0], MODEL_ORDER[5]),
]
ENDPOINTS = {
    "value_log_ratio_24m": {"kind": "regression", "target": "target_log_value_ratio_24m"},
    "downside_10pct": {"kind": "classification", "target": "target_downside_10pct"},
    "downside_25pct": {"kind": "classification", "target": "target_downside_25pct"},
    "downside_50pct": {"kind": "classification", "target": "target_downside_50pct"},
}
LOGISTIC_C_GRID = [0.01, 0.1, 1.0, 10.0]
GBT_CONFIGS = [
    {"n_estimators": 100, "learning_rate": 0.03, "max_depth": 2, "min_samples_leaf": 20},
    {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 2, "min_samples_leaf": 15},
    {"n_estimators": 150, "learning_rate": 0.03, "max_depth": 3, "min_samples_leaf": 20},
]
PROTECTED_DIRS = [
    ROOT / "Data" / "processed" / "financial_module_feasibility",
    ROOT / "Data" / "processed" / "retention_diagnostic",
    ROOT / "Data" / "processed" / "compatibility_targets",
    ROOT / "Data" / "processed" / "rolling_origin_stability",
    ROOT / "Data" / "processed" / "opportunity_transferability_decomposition",
]


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").fillna("").str.strip().str.lower().isin({"true", "1", "yes"})


def stable_seed(*parts: str) -> int:
    token = "|".join(parts).encode("utf-8")
    return (RANDOM_SEED + int(hashlib.sha256(token).hexdigest()[:8], 16)) % (2**32 - 1)


def protected_tree_sha256() -> str:
    digest = hashlib.sha256()
    for directory in sorted(PROTECTED_DIRS):
        if not directory.exists():
            continue
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def safe_log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    return np.log(num.where(num.gt(0)) / den.where(den.gt(0)))


def build_model_frame() -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    with AUDIT_VERIFICATION_PATH.open("r", encoding="utf-8") as handle:
        audit_verification = json.load(handle)
    if audit_verification.get("status") != "pass":
        raise RuntimeError("The upstream financial feasibility audit is not independently verified.")

    audit = pd.read_csv(
        AUDIT_PATH,
        usecols=[
            "transfer_event_id", "canonical_transfer_fee_eur", "canonical_transfer_fee_status",
            "valuation_pre_eur", "valuation_post24_nearest_eur", "market_value_change_24m_pct",
            "market_value_24m_eligible_90d", "eligible_market_value_downside_24m",
        ],
        low_memory=False,
    )
    eligible_ids = set(
        audit.loc[as_bool(audit["eligible_market_value_downside_24m"]), "transfer_event_id"].astype(str)
    )
    master = pd.read_csv(MASTER_PATH, low_memory=False)
    frame = master.loc[master["transfer_event_id"].astype(str).isin(eligible_ids)].copy()
    audit_fields = audit.loc[
        audit["transfer_event_id"].astype(str).isin(eligible_ids),
        ["transfer_event_id", "canonical_transfer_fee_eur", "canonical_transfer_fee_status"],
    ]
    frame = frame.merge(audit_fields, on="transfer_event_id", how="left", validate="one_to_one")
    if len(frame) != len(eligible_ids) or len(frame) != 4371:
        raise RuntimeError(f"Strict market-value cohort mismatch: {len(frame)} rows.")

    frame["transfer_date"] = pd.to_datetime(frame["transfer_date"], errors="coerce")
    frame["season_start_year"] = frame["transfer_date"].dt.year.where(
        frame["transfer_date"].dt.month.ge(7), frame["transfer_date"].dt.year - 1
    ).astype("Int64")
    frame["target_value_change_24m_pct"] = pd.to_numeric(
        frame["valuation_post24_change_pct"], errors="coerce"
    )
    frame["target_log_value_ratio_24m"] = safe_log_ratio(
        frame["valuation_post24_nearest_eur"], frame["valuation_pre_eur"]
    )
    pre_value = pd.to_numeric(frame["valuation_pre_eur"], errors="coerce")
    post_value = pd.to_numeric(frame["valuation_post24_nearest_eur"], errors="coerce")
    for threshold in [10, 25, 50]:
        boundary = np.nextafter(pre_value * (1 - threshold / 100), np.inf)
        frame[f"target_downside_{threshold}pct"] = post_value.le(boundary).astype(int)

    definitions: dict[str, dict[str, str]] = {}
    transfer_date = frame["transfer_date"]
    birth_date = pd.to_datetime(frame["player_date_of_birth"], errors="coerce")
    frame["age_at_transfer"] = ((transfer_date - birth_date).dt.days / 365.2425).where(
        lambda values: values.between(15, 45)
    )
    historical_position = frame["lineup_pre365_all_primary_position"].where(
        frame["lineup_pre365_all_primary_position"].notna(),
        frame["lineup_pre365_origin_primary_position"],
    )
    frame["historical_position_group"] = historical_position.map(coarse.position_group)
    frame["destination_competition_id"] = frame[
        "club_destination_pre365_dominant_competition_id"
    ].astype("string").fillna("unknown")
    frame["transfer_month_sin"] = np.sin(2 * np.pi * transfer_date.dt.month / 12)
    frame["transfer_month_cos"] = np.cos(2 * np.pi * transfer_date.dt.month / 12)
    frame["log_pre_value_eur"] = np.log1p(pd.to_numeric(frame["valuation_pre_eur"], errors="coerce"))
    frame["log_inbound_fee_eur"] = np.log1p(
        pd.to_numeric(frame["canonical_transfer_fee_eur"], errors="coerce").clip(lower=0)
    )
    frame["fee_reported_positive"] = pd.to_numeric(
        frame["canonical_transfer_fee_eur"], errors="coerce"
    ).gt(0).astype(float)
    frame["log_fee_to_pre_value"] = safe_log_ratio(
        frame["canonical_transfer_fee_eur"], frame["valuation_pre_eur"]
    )
    m1 = [
        "age_at_transfer", "historical_position_group", "destination_competition_id",
        "transfer_month_sin", "transfer_month_cos", "log_pre_value_eur",
        "log_inbound_fee_eur", "fee_reported_positive", "canonical_transfer_fee_status",
        "log_fee_to_pre_value", "valuation_pre_staleness_days",
    ]
    for feature in m1:
        definitions[feature] = {
            "feature_kind": "derived" if feature not in {"canonical_transfer_fee_status", "valuation_pre_staleness_days"} else "raw",
            "source_fields": feature,
            "timing_classification": "available at transfer",
            "rationale": "Market/profile control known at the transfer date.",
        }

    trajectory_specs = {
        "valuation_pre365_log_change": ("valuation_pre365_last_eur", "valuation_pre365_first_eur"),
        "valuation_pre730_log_change": ("valuation_pre730_last_eur", "valuation_pre730_first_eur"),
        "valuation_career_pre_log_change": ("valuation_career_pre_last_eur", "valuation_career_pre_first_eur"),
        "valuation_pre_to_prior_peak_log_ratio": ("valuation_pre_eur", "valuation_prior_peak_eur"),
        "valuation_pre365_log_range": ("valuation_pre365_max_eur", "valuation_pre365_min_eur"),
        "valuation_pre730_log_range": ("valuation_pre730_max_eur", "valuation_pre730_min_eur"),
    }
    trajectory_features = []
    for feature, (numerator, denominator) in trajectory_specs.items():
        frame[feature] = safe_log_ratio(frame[numerator], frame[denominator])
        trajectory_features.append(feature)
        definitions[feature] = {
            "feature_kind": "derived",
            "source_fields": f"{numerator};{denominator}",
            "timing_classification": "derived strictly pre-transfer valuation history",
            "rationale": "As-of valuation trajectory or dispersion signal.",
        }
    for source in [
        "valuation_pre365_observations", "valuation_pre730_observations",
        "valuation_career_pre_observations", "valuation_pre365_observation_span_days",
        "valuation_pre730_observation_span_days", "valuation_career_pre_observation_span_days",
        "valuation_pre_is_origin_club",
    ]:
        feature = f"log1p_{source}" if source != "valuation_pre_is_origin_club" else source
        if feature != source:
            frame[feature] = np.log1p(pd.to_numeric(frame[source], errors="coerce").clip(lower=0))
        trajectory_features.append(feature)
        definitions[feature] = {
            "feature_kind": "derived" if feature != source else "raw",
            "source_fields": source,
            "timing_classification": "strictly pre-transfer valuation history",
            "rationale": "Valuation coverage, horizon, or origin-club alignment control.",
        }
    m2 = m1 + trajectory_features

    sporting_features = []
    for feature, numerator, denominator, scale in opportunity.USAGE_SPECS:
        frame[feature] = coarse.safe_divide(
            frame[numerator], pd.to_numeric(frame[denominator], errors="coerce") * scale
        )
        sporting_features.append(feature)
        definitions[feature] = {
            "feature_kind": "derived",
            "source_fields": f"{numerator};{denominator}",
            "timing_classification": "derived strictly pre-transfer sporting history",
            "rationale": "Origin-club usage before transfer.",
        }
    for context in opportunity.PERFORMANCE_CONTEXTS:
        for suffix in ["goals_per90", "assists_per90"]:
            feature = f"appearance_{context}_{suffix}"
            sporting_features.append(feature)
            definitions[feature] = {
                "feature_kind": "raw", "source_fields": feature,
                "timing_classification": "strictly pre-transfer sporting history",
                "rationale": "Player output rate before transfer.",
            }
        minutes = f"appearance_{context}_minutes"
        for card in ["yellow_cards", "red_cards"]:
            source = f"appearance_{context}_{card}"
            feature = f"appearance_{context}_{card}_per90"
            frame[feature] = 90 * coarse.safe_divide(frame[source], frame[minutes])
            sporting_features.append(feature)
            definitions[feature] = {
                "feature_kind": "derived", "source_fields": f"{source};{minutes}",
                "timing_classification": "derived strictly pre-transfer sporting history",
                "rationale": "Player disciplinary rate before transfer.",
            }
    m3 = m2 + sporting_features

    destination_features = coarse.add_context_features(frame, "destination", definitions)
    m4 = m3 + destination_features
    origin_features = coarse.add_context_features(frame, "origin", definitions)
    transition_features = []
    for horizon in [365, 730]:
        for metric_name in coarse.TRANSITION_METRICS:
            destination = f"club_destination_pre{horizon}_{metric_name}"
            origin = f"club_origin_pre{horizon}_{metric_name}"
            feature = f"transition_pre{horizon}_{metric_name}_delta"
            frame[feature] = pd.to_numeric(frame[destination], errors="coerce") - pd.to_numeric(
                frame[origin], errors="coerce"
            )
            transition_features.append(feature)
            definitions[feature] = {
                "feature_kind": "derived", "source_fields": f"{destination};{origin}",
                "timing_classification": "derived strictly pre-transfer club context",
                "rationale": "Destination-minus-origin context contrast.",
            }
    m5 = m4 + origin_features + transition_features
    feature_sets = dict(zip(MODEL_ORDER, [[], m1, m2, m3, m4, m5]))
    for index in range(1, len(MODEL_ORDER)):
        if not set(feature_sets[MODEL_ORDER[index - 1]]).issubset(feature_sets[MODEL_ORDER[index]]):
            raise RuntimeError("Feature sets are not nested.")
    if any(len(features) != len(set(features)) for features in feature_sets.values()):
        raise RuntimeError("A feature set contains duplicates.")

    manifest_rows = [{
        "feature_set": MODEL_ORDER[0], "feature_order": 0, "feature": "",
        "feature_kind": "no_predictors", "source_fields": "",
        "timing_classification": "not applicable",
        "rationale": "Chronological median or prevalence benchmark.", "raw_feature_count": 0,
    }]
    for feature_set in MODEL_ORDER[1:]:
        features = feature_sets[feature_set]
        for order, feature in enumerate(features, start=1):
            manifest_rows.append({
                "feature_set": feature_set, "feature_order": order, "feature": feature,
                **definitions[feature], "raw_feature_count": len(features),
            })
    return frame.sort_values(["season_start_year", "transfer_date", "transfer_event_id"]), feature_sets, pd.DataFrame(manifest_rows)


def classification_metrics(actual: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    result: dict[str, float | int] = {
        "n": len(actual), "brier": brier_score_loss(actual, probability),
        "log_loss": log_loss(actual, probability, labels=[0, 1]),
        "actual_rate": float(actual.mean()), "predicted_rate": float(probability.mean()),
    }
    if len(np.unique(actual)) == 2:
        result["roc_auc"] = roc_auc_score(actual, probability)
        result["average_precision"] = average_precision_score(actual, probability)
    else:
        result["roc_auc"] = np.nan
        result["average_precision"] = np.nan
    return result


def candidate_specs(kind: str) -> list[tuple[str, dict[str, Any]]]:
    if kind == "regression":
        linear = [("ridge", {"alpha": alpha}) for alpha in ALPHA_GRID]
    else:
        linear = [("logistic", {"C": value}) for value in LOGISTIC_C_GRID]
    trees = [("gradient_boosted_trees", config) for config in GBT_CONFIGS]
    return linear + trees


def fit_candidate(
    train: pd.DataFrame,
    score: pd.DataFrame,
    features: list[str],
    target: str,
    kind: str,
    family: str,
    parameters: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, int]:
    processor = fit_preprocessor(train, features)
    x_train = processor.transform(train)
    x_score = processor.transform(score)
    y_train = train[target].to_numpy(dtype=float if kind == "regression" else int)
    if family == "ridge":
        intercept, beta = fit_ridge(x_train, y_train, float(parameters["alpha"]))
        prediction = predict_ridge(x_score, intercept, beta)
    elif family == "logistic":
        model = LogisticRegression(C=float(parameters["C"]), max_iter=3000, random_state=seed)
        model.fit(x_train, y_train)
        prediction = model.predict_proba(x_score)[:, 1]
    elif kind == "regression":
        model = GradientBoostingRegressor(loss="huber", random_state=seed, **parameters)
        model.fit(x_train, y_train)
        prediction = model.predict(x_score)
    else:
        model = GradientBoostingClassifier(loss="log_loss", random_state=seed, **parameters)
        model.fit(x_train, y_train)
        prediction = model.predict_proba(x_score)[:, 1]
    return np.asarray(prediction, dtype=float), x_train.shape[1]


def tune_and_fit(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    evaluation: pd.DataFrame,
    features: list[str],
    endpoint: str,
    target: str,
    kind: str,
    origin_id: str,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    tuning_rows = []
    candidates = []
    for family, parameters in candidate_specs(kind):
        seed = stable_seed(endpoint, origin_id, family, json.dumps(parameters, sort_keys=True))
        prediction, encoded_count = fit_candidate(
            train, validation, features, target, kind, family, parameters, seed
        )
        result = regression_metrics(validation[target], prediction) if kind == "regression" else classification_metrics(validation[target], prediction)
        primary = float(result["mae"] if kind == "regression" else result["brier"])
        secondary = float(result["rmse"] if kind == "regression" else result["log_loss"])
        row = {
            "endpoint": endpoint, "origin_id": origin_id, "model_family": family,
            "hyperparameters": json.dumps(parameters, sort_keys=True),
            "encoded_feature_count": encoded_count, "validation_primary_loss": primary,
            "validation_secondary_loss": secondary, **{f"validation_{key}": value for key, value in result.items()},
        }
        tuning_rows.append(row)
        candidates.append(((primary, secondary, family, row["hyperparameters"]), family, parameters))
    _, family, parameters = min(candidates, key=lambda item: item[0])
    fit_frame = pd.concat([train, validation], ignore_index=True)
    seed = stable_seed(endpoint, origin_id, family, json.dumps(parameters, sort_keys=True), "refit")
    prediction, encoded_count = fit_candidate(
        fit_frame, evaluation, features, target, kind, family, parameters, seed
    )
    selected = {
        "selected_family": family,
        "selected_hyperparameters": json.dumps(parameters, sort_keys=True),
        "encoded_feature_count": encoded_count,
    }
    return prediction, selected, tuning_rows


def fit_models(
    frame: pd.DataFrame, feature_sets: dict[str, list[str]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = []
    origin_rows = []
    tuning_rows = []
    for endpoint, spec in ENDPOINTS.items():
        kind, target = spec["kind"], spec["target"]
        for origin_id, train_end, validation_year, evaluation_year in ORIGINS:
            train = frame.loc[frame["season_start_year"].le(train_end)].copy()
            validation = frame.loc[frame["season_start_year"].eq(validation_year)].copy()
            evaluation = frame.loc[frame["season_start_year"].eq(evaluation_year)].copy()
            if min(len(train), len(validation), len(evaluation)) == 0:
                raise RuntimeError(f"Empty chronological split for {endpoint}/{origin_id}.")
            for model_variant in MODEL_ORDER:
                if model_variant == MODEL_ORDER[0]:
                    fit_target = pd.concat([train[target], validation[target]], ignore_index=True)
                    if kind == "regression":
                        prediction = np.repeat(float(fit_target.median()), len(evaluation))
                        selected = {"selected_family": "chronological_median", "selected_hyperparameters": "{}", "encoded_feature_count": 0}
                    else:
                        prediction = np.repeat(float(fit_target.mean()), len(evaluation))
                        selected = {"selected_family": "chronological_prevalence", "selected_hyperparameters": "{}", "encoded_feature_count": 0}
                else:
                    prediction, selected, tuning = tune_and_fit(
                        train, validation, evaluation, feature_sets[model_variant], endpoint,
                        target, kind, origin_id,
                    )
                    for row in tuning:
                        row["model_variant"] = model_variant
                    tuning_rows.extend(tuning)
                if kind == "classification":
                    prediction = np.clip(prediction, 1e-6, 1 - 1e-6)
                    result = classification_metrics(evaluation[target], prediction)
                else:
                    result = regression_metrics(evaluation[target], prediction)
                origin_rows.append({
                    "endpoint": endpoint, "endpoint_kind": kind, "origin_id": origin_id,
                    "train_end_year": train_end, "validation_year": validation_year,
                    "evaluation_year": evaluation_year, "train_n": len(train),
                    "validation_n": len(validation), "evaluation_n": len(evaluation),
                    "model_variant": model_variant, "raw_feature_count": len(feature_sets[model_variant]),
                    **selected, **result,
                })
                for row, predicted in zip(evaluation.itertuples(index=False), prediction):
                    predictions.append({
                        "endpoint": endpoint, "endpoint_kind": kind, "origin_id": origin_id,
                        "evaluation_year": evaluation_year, "transfer_event_id": row.transfer_event_id,
                        "player_id": row.player_id, "transfer_date": row.transfer_date,
                        "model_variant": model_variant, **selected,
                        "actual": getattr(row, target), "prediction": float(predicted),
                        "actual_value_change_pct": row.target_value_change_24m_pct,
                        "predicted_value_change_pct": math.expm1(float(predicted)) if kind == "regression" else np.nan,
                    })
    return pd.DataFrame(predictions), pd.DataFrame(origin_rows), pd.DataFrame(tuning_rows)


def paired_predictions(
    predictions: pd.DataFrame, endpoint: str, baseline: str, candidate: str, years: list[int]
) -> pd.DataFrame:
    subset = predictions.loc[
        predictions["endpoint"].eq(endpoint) & predictions["evaluation_year"].isin(years)
    ]
    keys = ["origin_id", "evaluation_year", "transfer_event_id", "player_id"]
    left = subset.loc[subset["model_variant"].eq(baseline), keys + ["actual", "prediction"]].rename(
        columns={"prediction": "baseline_prediction"}
    )
    right = subset.loc[subset["model_variant"].eq(candidate), keys + ["actual", "prediction"]].rename(
        columns={"actual": "candidate_actual", "prediction": "candidate_prediction"}
    )
    paired = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if not np.allclose(paired["actual"], paired["candidate_actual"], atol=1e-12):
        raise RuntimeError("Paired model actuals disagree.")
    return paired.drop(columns="candidate_actual")


def cluster_bootstrap(pairs: pd.DataFrame, kind: str, seed: int) -> dict[str, Any]:
    if kind == "regression":
        difference = np.abs(pairs["actual"] - pairs["baseline_prediction"]) - np.abs(
            pairs["actual"] - pairs["candidate_prediction"]
        )
    else:
        difference = np.square(pairs["actual"] - pairs["baseline_prediction"]) - np.square(
            pairs["actual"] - pairs["candidate_prediction"]
        )
    clusters = pd.DataFrame({"player_id": pairs["player_id"], "difference": difference}).groupby(
        "player_id", sort=True
    )["difference"].agg(["sum", "size"])
    sums = clusters["sum"].to_numpy(float)
    sizes = clusters["size"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_REPETITIONS, dtype=float)
    chunk = 250
    for start in range(0, BOOTSTRAP_REPETITIONS, chunk):
        stop = min(start + chunk, BOOTSTRAP_REPETITIONS)
        indices = rng.integers(0, len(clusters), size=(stop - start, len(clusters)))
        draws[start:stop] = sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)
    return {
        "cluster_count": len(clusters), "cluster_ci_lower_95": np.quantile(draws, 0.025),
        "cluster_ci_upper_95": np.quantile(draws, 0.975),
        "cluster_probability_candidate_better": np.mean(draws > 0),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS, "bootstrap_seed": seed,
    }


def stability_label(improvement: float, origin_values: list[float], probability: float) -> str:
    wins = sum(value > 0 for value in origin_values)
    if improvement > 0 and wins >= max(3, len(origin_values) - 1) and probability >= 0.90:
        return "stable_positive"
    if improvement > 0 and wins >= math.ceil(len(origin_values) / 2):
        return "mixed_positive"
    if improvement <= 0 and wins <= 1:
        return "unstable_negative"
    return "mixed"


def build_comparisons(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint, spec in ENDPOINTS.items():
        kind = spec["kind"]
        for comparison, baseline, candidate in COMPARISONS:
            for window, years in WINDOWS.items():
                paired = paired_predictions(predictions, endpoint, baseline, candidate, years)
                if kind == "regression":
                    base_loss = np.abs(paired["actual"] - paired["baseline_prediction"])
                    candidate_loss = np.abs(paired["actual"] - paired["candidate_prediction"])
                    metric_name = "mae"
                else:
                    base_loss = np.square(paired["actual"] - paired["baseline_prediction"])
                    candidate_loss = np.square(paired["actual"] - paired["candidate_prediction"])
                    metric_name = "brier"
                origin_values = []
                for _, group in paired.groupby("origin_id", sort=True):
                    if kind == "regression":
                        b = np.abs(group["actual"] - group["baseline_prediction"]).mean()
                        c = np.abs(group["actual"] - group["candidate_prediction"]).mean()
                    else:
                        b = np.square(group["actual"] - group["baseline_prediction"]).mean()
                        c = np.square(group["actual"] - group["candidate_prediction"]).mean()
                    origin_values.append(float(b - c))
                improvement = float(base_loss.mean() - candidate_loss.mean())
                bootstrap = cluster_bootstrap(
                    paired, kind, stable_seed(endpoint, comparison, window, "cluster")
                )
                rows.append({
                    "endpoint": endpoint, "endpoint_kind": kind, "comparison": comparison,
                    "baseline_model": baseline, "candidate_model": candidate,
                    "window": window, "evaluation_years": ";".join(map(str, years)),
                    "primary_metric": metric_name, "pooled_n": len(paired),
                    "unique_players": paired["player_id"].nunique(),
                    "baseline_loss": base_loss.mean(), "candidate_loss": candidate_loss.mean(),
                    "loss_improvement": improvement,
                    "relative_loss_improvement": improvement / base_loss.mean(),
                    "origin_wins": sum(value > 0 for value in origin_values),
                    "origin_count": len(origin_values), **bootstrap,
                    "stability_label": stability_label(
                        improvement, origin_values, float(bootstrap["cluster_probability_candidate_better"])
                    ),
                })
    return pd.DataFrame(rows)


def build_risk_bands(predictions: pd.DataFrame) -> pd.DataFrame:
    selected = predictions.loc[predictions["model_variant"].eq(MODEL_ORDER[-1])].copy()
    rows = []
    for endpoint, spec in ENDPOINTS.items():
        subset = selected.loc[selected["endpoint"].eq(endpoint)].copy()
        subset["risk_score"] = -subset["prediction"] if spec["kind"] == "regression" else subset["prediction"]
        subset["risk_band"] = subset.groupby("origin_id")["risk_score"].transform(
            lambda values: pd.qcut(values.rank(method="first"), 5, labels=False, duplicates="drop") + 1
        )
        for band, group in subset.groupby("risk_band", sort=True):
            rows.append({
                "endpoint": endpoint, "risk_band": int(band),
                "risk_band_label": "lowest_predicted_risk" if band == 1 else ("highest_predicted_risk" if band == 5 else f"band_{int(band)}"),
                "rows": len(group), "unique_players": group["player_id"].nunique(),
                "mean_prediction": group["prediction"].mean(), "mean_actual": group["actual"].mean(),
                "mean_actual_value_change_pct": group["actual_value_change_pct"].mean(),
            })
    return pd.DataFrame(rows)


def build_decisions(comparisons: pd.DataFrame, origin_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pooled = comparisons.loc[comparisons["window"].eq("pooled_four")].set_index(
        ["endpoint", "comparison"]
    )
    for endpoint, spec in ENDPOINTS.items():
        market = pooled.loc[(endpoint, "M1_vs_M0_market_profile")]
        incremental = pooled.loc[(endpoint, "M5_vs_M1_all_incremental_history_context")]
        total = pooled.loc[(endpoint, "M5_vs_M0_total")]
        validated = total["cluster_ci_lower_95"] > 0 and total["origin_wins"] >= 3
        incremental_validated = incremental["cluster_ci_lower_95"] > 0 and incremental["origin_wins"] >= 3
        if validated:
            status = "validated_predictive_signal"
        elif total["loss_improvement"] > 0:
            status = "promising_but_uncertain"
        else:
            status = "not_validated"
        endpoint_origins = origin_results.loc[
            origin_results["endpoint"].eq(endpoint) & origin_results["model_variant"].eq(MODEL_ORDER[-1])
        ]
        rows.append({
            "endpoint": endpoint, "endpoint_kind": spec["kind"], "status": status,
            "market_profile_signal_validated": bool(market["cluster_ci_lower_95"] > 0),
            "incremental_history_context_validated": bool(incremental_validated),
            "total_loss_improvement": total["loss_improvement"],
            "total_relative_loss_improvement": total["relative_loss_improvement"],
            "total_ci_lower_95": total["cluster_ci_lower_95"],
            "total_ci_upper_95": total["cluster_ci_upper_95"],
            "total_origin_wins": int(total["origin_wins"]),
            "incremental_over_market_improvement": incremental["loss_improvement"],
            "incremental_over_market_ci_lower_95": incremental["cluster_ci_lower_95"],
            "incremental_over_market_ci_upper_95": incremental["cluster_ci_upper_95"],
            "selected_family_counts_full_model": json.dumps(endpoint_origins["selected_family"].value_counts().to_dict(), sort_keys=True),
            "recommended_scope": MODEL_ORDER[-1] if incremental_validated else MODEL_ORDER[1],
        })
    return pd.DataFrame(rows)


def build_checks(frame: pd.DataFrame, feature_sets: dict[str, list[str]], predictions: pd.DataFrame) -> pd.DataFrame:
    checks = []
    def add(check: str, passed: bool, observed: Any, expected: Any, note: str) -> None:
        checks.append({"check": check, "passed": bool(passed), "observed": observed, "expected": expected, "note": note})
    add("cohort_rows", len(frame) == 4371, len(frame), 4371, "Matches strict financial audit cohort.")
    add("cohort_key_unique", frame["transfer_event_id"].is_unique, frame["transfer_event_id"].nunique(), len(frame), "One row per transfer.")
    add("continuous_target_complete", frame["target_log_value_ratio_24m"].notna().all(), int(frame["target_log_value_ratio_24m"].notna().sum()), len(frame), "Positive values imply finite log ratios.")
    add("thresholds_nested", (frame["target_downside_50pct"] <= frame["target_downside_25pct"]).all() and (frame["target_downside_25pct"] <= frame["target_downside_10pct"]).all(), True, True, "Severe downside classes are nested.")
    add("feature_sets_nested", all(set(feature_sets[MODEL_ORDER[i-1]]).issubset(feature_sets[MODEL_ORDER[i]]) for i in range(1, len(MODEL_ORDER))), True, True, "Each block only adds predictors.")
    forbidden = [feature for features in feature_sets.values() for feature in features if any(token in feature.lower() for token in ["post12", "post24", "target_", "eligible_"])]
    add("no_posttransfer_predictors", not forbidden, ";".join(sorted(set(forbidden))), "none", "No outcome or eligibility fields are predictors.")
    key = ["endpoint", "origin_id", "model_variant", "transfer_event_id"]
    add("prediction_key_unique", not predictions.duplicated(key).any(), int(predictions.duplicated(key).sum()), 0, "One prediction per endpoint/origin/model/event.")
    class_predictions = predictions.loc[predictions["endpoint_kind"].eq("classification"), "prediction"]
    add("probability_bounds", class_predictions.between(0, 1).all(), f"{class_predictions.min()}..{class_predictions.max()}", "0..1", "All downside forecasts are probabilities.")
    aligned = predictions.groupby(["endpoint", "origin_id", "model_variant"]).size().groupby(level=[0, 1]).nunique().eq(1).all()
    add("paired_evaluation_rows", aligned, aligned, True, "All models share evaluation rows.")
    return pd.DataFrame(checks)


def write_readme(frame: pd.DataFrame, decisions: pd.DataFrame, comparisons: pd.DataFrame, feature_sets: dict[str, list[str]]) -> None:
    lines = [
        "# MoveMaker 24-month market-value downside model", "",
        "This experiment predicts estimated market-value change, not realized cash return or accounting ROI.", "",
        "## Design", "",
        f"- Strict cohort: {len(frame):,} permanent 2017+ transfers with recent positive pre-value and a positive 24-month value within +/-90 days.",
        "- Continuous target: `log(post24 value / pre-transfer value)`; percentage change is retained for interpretation.",
        "- Binary targets: declines of at least 10%, 25%, and 50%.",
        "- Four rolling origins evaluate 2020-2023. Model family and hyperparameters are selected only on the preceding validation season.",
        "- Primary uncertainty: paired player-cluster bootstrap with 5,000 repetitions.", "",
        "## Feature blocks", "",
    ]
    for model in MODEL_ORDER:
        lines.append(f"- `{model}`: {len(feature_sets[model])} raw/derived predictors.")
    lines += ["", "## Results", ""]
    pooled = comparisons.loc[comparisons["window"].eq("pooled_four")].set_index(["endpoint", "comparison"])
    for row in decisions.itertuples(index=False):
        total = pooled.loc[(row.endpoint, "M5_vs_M0_total")]
        incremental = pooled.loc[(row.endpoint, "M5_vs_M1_all_incremental_history_context")]
        lines.append(
            f"- `{row.endpoint}`: `{row.status}`. Full vs chronological baseline {total['primary_metric']} improvement {total['loss_improvement']:.6f}, 95% CI [{total['cluster_ci_lower_95']:.6f}, {total['cluster_ci_upper_95']:.6f}], {int(total['origin_wins'])}/4 origin wins. Added history/context beyond market profile: {incremental['loss_improvement']:.6f}, CI [{incremental['cluster_ci_lower_95']:.6f}, {incremental['cluster_ci_upper_95']:.6f}]."
        )
    lines += [
        "", "## Interpretation guardrail", "",
        "A validated result means the model ranks or estimates future Transfermarkt-style market-value downside better than a chronological constant baseline. It does not mean the model predicts sale proceeds, profit, wages, or full ROI.", "",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    protected_before = protected_tree_sha256()
    frame, feature_sets, feature_manifest = build_model_frame()
    predictions, origin_results, tuning = fit_models(frame, feature_sets)
    comparisons = build_comparisons(predictions)
    risk_bands = build_risk_bands(predictions)
    decisions = build_decisions(comparisons, origin_results)
    checks = build_checks(frame, feature_sets, predictions)
    if not checks["passed"].all():
        raise RuntimeError(f"Build checks failed: {checks.loc[~checks['passed'], 'check'].tolist()}")

    target_columns = [
        "transfer_event_id", "player_id", "transfer_date", "season_start_year",
        "valuation_pre_eur", "valuation_post24_nearest_eur", "target_value_change_24m_pct",
        "target_log_value_ratio_24m", "target_downside_10pct", "target_downside_25pct",
        "target_downside_50pct",
    ]
    tables = {
        "market_value_downside_targets.csv": frame[target_columns],
        "market_value_feature_manifest.csv": feature_manifest,
        "validation_candidate_results.csv": tuning,
        "market_value_predictions.csv": predictions,
        "market_value_origin_results.csv": origin_results,
        "market_value_comparison_results.csv": comparisons,
        "market_value_risk_bands.csv": risk_bands,
        "market_value_decision_summary.csv": decisions,
        "build_checks.csv": checks,
    }
    for name, table in tables.items():
        table.to_csv(OUTPUT_DIR / name, index=False)
    source_manifest = pd.DataFrame([
        {"source": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in [MASTER_PATH, AUDIT_PATH, AUDIT_VERIFICATION_PATH]
    ])
    source_manifest.to_csv(OUTPUT_DIR / "source_manifest.csv", index=False)
    tables["source_manifest.csv"] = source_manifest
    write_readme(frame, decisions, comparisons, feature_sets)

    protected_after = protected_tree_sha256()
    if protected_before != protected_after:
        raise RuntimeError("A protected verified output changed during modeling.")
    run_summary = {
        "cohort_rows": len(frame), "unique_players": frame["player_id"].nunique(),
        "date_min": frame["transfer_date"].min().date().isoformat(),
        "date_max": frame["transfer_date"].max().date().isoformat(),
        "feature_counts": {model: len(features) for model, features in feature_sets.items()},
        "endpoints": list(ENDPOINTS), "origins": [item[0] for item in ORIGINS],
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "protected_outputs_unchanged": True, "build_checks": len(checks),
        "build_checks_passed": int(checks["passed"].sum()),
        "decisions": decisions.set_index("endpoint")["status"].to_dict(),
    }
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(run_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_names = sorted(list(tables) + ["README.md", "run_summary.json"])
    manifest = pd.DataFrame([
        {"file": name, "size_bytes": (OUTPUT_DIR / name).stat().st_size, "sha256": sha256_file(OUTPUT_DIR / name)}
        for name in manifest_names
    ])
    manifest.to_csv(OUTPUT_DIR / "output_manifest.csv", index=False)
    print(json.dumps(run_summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

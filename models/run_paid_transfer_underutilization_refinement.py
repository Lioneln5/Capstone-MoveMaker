"""Calibrate and compactly refine paid-transfer under-utilization forecasts.

The experiment freezes three small feature blocks and uses three temporal
layers for every rolling origin:

1. an internal prior season selects raw model family/hyperparameters;
2. the next validation season selects the compact block and calibration policy;
3. the following season remains untouched evaluation data.

Calibration is joint across the under-10% and under-25% targets and every
eligible policy guarantees P(under 10%) <= P(under 25%).
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
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
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

import run_market_value_downside_model as market
import run_paid_transfer_underutilization_model as under
from train_compatibility_models import sha256_file


MODEL_DIR = under.OUTPUT_DIR
MODEL_VERIFICATION_PATH = MODEL_DIR / "independent_verification.json"
OUTPUT_DIR = ROOT / "Data" / "processed" / "paid_transfer_underutilization_refinement"

RANDOM_SEED = 20260811
BOOTSTRAP_REPETITIONS = 5000
MIN_SUBGROUP_ROWS = 50
MIN_SUBGROUP_ORIGINS = 2
THRESHOLD_ENDPOINTS = ["underutilization_10pct", "underutilization_25pct"]
THRESHOLDS = {"underutilization_10pct": 0.10, "underutilization_25pct": 0.25}
CALIBRATION_STRATEGIES = [
    "raw_projected", "platt_projected", "isotonic_projected",
    "gaussian_distributional", "empirical_distributional",
]
COMPACT_MODELS = [
    "C1_paid_deal_market_profile",
    "C2_plus_prior_opportunity",
    "C3_plus_compact_destination_context",
]
COMPACT_DESTINATION_FEATURES = [
    "club_destination_pre365_games",
    "club_destination_pre365_win_pct",
    "club_destination_pre365_points_per_game",
    "club_destination_pre365_goals_for_per_game",
    "club_destination_pre365_goals_against_per_game",
    "club_destination_pre365_goal_difference_per_game",
    "club_destination_pre365_domestic_league_game_share",
    "club_destination_pre365_domestic_cup_game_share",
    "club_destination_pre365_international_game_share",
]
PROTECTED_DIRS = under.PROTECTED_DIRS + [MODEL_DIR]


def stable_seed(*parts: str) -> int:
    token = "|".join(parts).encode("utf-8")
    return (RANDOM_SEED + int(hashlib.sha256(token).hexdigest()[:8], 16)) % (2**32 - 1)


def protected_tree_sha256() -> str:
    digest = hashlib.sha256()
    for directory in sorted(set(PROTECTED_DIRS)):
        if not directory.exists():
            continue
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def compact_feature_sets(
    full_sets: dict[str, list[str]], full_manifest: pd.DataFrame
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    c1 = list(full_sets[under.FIXED_MODEL_ORDER[1]])
    c2 = list(full_sets[under.FIXED_MODEL_ORDER[2]])
    c3 = c2 + COMPACT_DESTINATION_FEATURES
    result = dict(zip(COMPACT_MODELS, [c1, c2, c3]))
    expected = dict(zip(COMPACT_MODELS, [11, 19, 28]))
    observed = {name: len(features) for name, features in result.items()}
    if observed != expected:
        raise RuntimeError(f"Compact feature count mismatch: {observed}")
    if not set(c1).issubset(c2) or not set(c2).issubset(c3):
        raise RuntimeError("Compact feature blocks are not nested.")
    if any(len(features) != len(set(features)) for features in result.values()):
        raise RuntimeError("Compact feature block contains duplicates.")

    definitions = full_manifest.dropna(subset=["feature"]).drop_duplicates("feature").set_index("feature")
    rows = []
    for model, features in result.items():
        for order, feature in enumerate(features, start=1):
            if feature not in definitions.index:
                raise RuntimeError(f"Compact feature missing from upstream manifest: {feature}")
            source = definitions.loc[feature]
            rows.append({
                "compact_model": model, "feature_order": order, "feature": feature,
                "feature_kind": source["feature_kind"], "source_fields": source["source_fields"],
                "timing_classification": source["timing_classification"],
                "rationale": source["rationale"], "raw_feature_count": len(features),
            })
    return result, pd.DataFrame(rows)


def select_raw_candidate(
    model_train: pd.DataFrame,
    model_selector: pd.DataFrame,
    features: list[str],
    target: str,
    kind: str,
    endpoint: str,
    origin_id: str,
    compact_model: str,
) -> tuple[str, dict[str, Any], pd.DataFrame]:
    rows = []
    candidates = []
    for family, parameters in market.candidate_specs(kind):
        parameter_text = json.dumps(parameters, sort_keys=True)
        prediction, encoded_count = market.fit_candidate(
            model_train, model_selector, features, target, kind, family, parameters,
            stable_seed(endpoint, origin_id, compact_model, family, parameter_text),
        )
        if kind == "regression":
            prediction = np.clip(prediction, 0, 1.05)
            result = market.regression_metrics(model_selector[target], prediction)
            primary, secondary = float(result["mae"]), float(result["rmse"])
        else:
            result = market.classification_metrics(model_selector[target], prediction)
            primary, secondary = float(result["brier"]), float(result["log_loss"])
        rows.append({
            "endpoint": endpoint, "origin_id": origin_id, "compact_model": compact_model,
            "model_family": family, "hyperparameters": parameter_text,
            "encoded_feature_count": encoded_count,
            "internal_selector_primary_loss": primary,
            "internal_selector_secondary_loss": secondary,
        })
        candidates.append(((primary, secondary, family, parameter_text), family, parameters))
    _, family, parameters = min(candidates, key=lambda item: item[0])
    return family, parameters, pd.DataFrame(rows)


def build_raw_predictions(
    frame: pd.DataFrame, feature_sets: dict[str, list[str]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    tuning_rows = []
    selected_rows = []
    for origin_id, train_end, validation_year, evaluation_year in under.ORIGINS:
        model_train = frame.loc[frame["season_start_year"].le(train_end - 1)].copy()
        model_selector = frame.loc[frame["season_start_year"].eq(train_end)].copy()
        validation = frame.loc[frame["season_start_year"].eq(validation_year)].sort_values(
            ["transfer_date", "transfer_event_id"]
        ).copy()
        evaluation = frame.loc[frame["season_start_year"].eq(evaluation_year)].sort_values(
            ["transfer_date", "transfer_event_id"]
        ).copy()
        if min(len(model_train), len(model_selector), len(validation), len(evaluation)) == 0:
            raise RuntimeError(f"Empty nested temporal layer for {origin_id}.")
        raw_train = pd.concat([model_train, model_selector], ignore_index=True)
        refit_train = pd.concat([raw_train, validation], ignore_index=True)
        for endpoint, spec in under.ENDPOINTS.items():
            target, kind = spec["target"], spec["kind"]
            for compact_model in COMPACT_MODELS:
                features = feature_sets[compact_model]
                family, parameters, tuning = select_raw_candidate(
                    model_train, model_selector, features, target, kind, endpoint,
                    origin_id, compact_model,
                )
                tuning_rows.extend(tuning.to_dict("records"))
                parameter_text = json.dumps(parameters, sort_keys=True)
                validation_prediction, validation_encoded = market.fit_candidate(
                    raw_train, validation, features, target, kind, family, parameters,
                    stable_seed(endpoint, origin_id, compact_model, "validation", family, parameter_text),
                )
                evaluation_prediction, evaluation_encoded = market.fit_candidate(
                    refit_train, evaluation, features, target, kind, family, parameters,
                    stable_seed(endpoint, origin_id, compact_model, "evaluation", family, parameter_text),
                )
                if kind == "regression":
                    validation_prediction = np.clip(validation_prediction, 0, 1.05)
                    evaluation_prediction = np.clip(evaluation_prediction, 0, 1.05)
                else:
                    validation_prediction = np.clip(validation_prediction, 1e-6, 1 - 1e-6)
                    evaluation_prediction = np.clip(evaluation_prediction, 1e-6, 1 - 1e-6)
                selected_rows.append({
                    "endpoint": endpoint, "origin_id": origin_id,
                    "internal_model_train_end_year": train_end - 1,
                    "internal_model_selector_year": train_end,
                    "calibration_validation_year": validation_year,
                    "evaluation_year": evaluation_year, "compact_model": compact_model,
                    "selected_family": family,
                    "selected_hyperparameters": parameter_text,
                    "validation_encoded_feature_count": validation_encoded,
                    "evaluation_encoded_feature_count": evaluation_encoded,
                })
                for split, source, values in [
                    ("validation", validation, validation_prediction),
                    ("evaluation", evaluation, evaluation_prediction),
                ]:
                    for row, predicted in zip(source.itertuples(index=False), values):
                        prediction_rows.append({
                            "split": split, "endpoint": endpoint, "endpoint_kind": kind,
                            "origin_id": origin_id, "evaluation_year": evaluation_year,
                            "transfer_event_id": row.transfer_event_id,
                            "player_id": row.player_id,
                            "transfer_date": row.transfer_date.date().isoformat(),
                            "compact_model": compact_model, "selected_family": family,
                            "selected_hyperparameters": parameter_text,
                            "actual": float(getattr(row, target)), "raw_prediction": float(predicted),
                        })
    return pd.DataFrame(prediction_rows), pd.DataFrame(tuning_rows), pd.DataFrame(selected_rows)


def wide_raw(long: pd.DataFrame, compact_model: str, split: str, origin_id: str) -> pd.DataFrame:
    subset = long.loc[
        long["compact_model"].eq(compact_model)
        & long["split"].eq(split)
        & long["origin_id"].eq(origin_id)
    ]
    keys = ["origin_id", "evaluation_year", "transfer_event_id", "player_id", "transfer_date"]
    prediction = subset.pivot(index=keys, columns="endpoint", values="raw_prediction").reset_index()
    actual = subset.pivot(index=keys, columns="endpoint", values="actual").reset_index()
    prediction = prediction.rename(columns={
        "opportunity_share_24m": "raw_share",
        "underutilization_10pct": "raw_under10",
        "underutilization_25pct": "raw_under25",
    })
    actual = actual.rename(columns={
        "opportunity_share_24m": "actual_share",
        "underutilization_10pct": "actual_under10",
        "underutilization_25pct": "actual_under25",
    })
    return prediction.merge(actual, on=keys, validate="one_to_one").sort_values(
        ["transfer_date", "transfer_event_id"]
    ).reset_index(drop=True)


def project_probabilities(probabilities: pd.DataFrame) -> pd.DataFrame:
    result = probabilities[["underutilization_10pct", "underutilization_25pct"]].copy()
    violation = result["underutilization_10pct"] > result["underutilization_25pct"]
    midpoint = (
        result["underutilization_10pct"] + result["underutilization_25pct"]
    ) / 2
    result.loc[violation, "underutilization_10pct"] = midpoint[violation]
    result.loc[violation, "underutilization_25pct"] = midpoint[violation]
    return result.clip(1e-6, 1 - 1e-6)


def fit_platt(probability: pd.Series, actual: pd.Series) -> LogisticRegression | None:
    if actual.nunique() < 2:
        return None
    p = np.clip(probability.to_numpy(float), 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, max_iter=3000, random_state=RANDOM_SEED)
    model.fit(x, actual.to_numpy(int))
    return model


def apply_platt(model: LogisticRegression | None, probability: pd.Series) -> np.ndarray:
    p = np.clip(probability.to_numpy(float), 1e-6, 1 - 1e-6)
    if model is None:
        return p
    return model.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]


def fit_isotonic(probability: pd.Series, actual: pd.Series) -> IsotonicRegression | None:
    if actual.nunique() < 2 or probability.nunique() < 2:
        return None
    return IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(
        probability.to_numpy(float), actual.to_numpy(int)
    )


def apply_isotonic(model: IsotonicRegression | None, probability: pd.Series) -> np.ndarray:
    p = probability.to_numpy(float)
    return p if model is None else model.predict(p)


def fit_calibrator(strategy: str, calibration: pd.DataFrame) -> dict[str, Any]:
    fitted: dict[str, Any] = {"strategy": strategy}
    if strategy == "platt_projected":
        fitted["under10"] = fit_platt(calibration["raw_under10"], calibration["actual_under10"])
        fitted["under25"] = fit_platt(calibration["raw_under25"], calibration["actual_under25"])
    elif strategy == "isotonic_projected":
        fitted["under10"] = fit_isotonic(calibration["raw_under10"], calibration["actual_under10"])
        fitted["under25"] = fit_isotonic(calibration["raw_under25"], calibration["actual_under25"])
    elif strategy.endswith("distributional"):
        residuals = calibration["actual_share"] - calibration["raw_share"]
        fitted["residual_mean"] = float(residuals.mean())
        fitted["residual_std"] = max(float(residuals.std(ddof=1)), 1e-6)
        fitted["residuals"] = np.sort(residuals.to_numpy(float))
    return fitted


def apply_calibrator(fitted: dict[str, Any], score: pd.DataFrame) -> pd.DataFrame:
    strategy = fitted["strategy"]
    if strategy == "raw_projected":
        probability = pd.DataFrame({
            "underutilization_10pct": score["raw_under10"].to_numpy(float),
            "underutilization_25pct": score["raw_under25"].to_numpy(float),
        }, index=score.index)
    elif strategy == "platt_projected":
        probability = pd.DataFrame({
            "underutilization_10pct": apply_platt(fitted["under10"], score["raw_under10"]),
            "underutilization_25pct": apply_platt(fitted["under25"], score["raw_under25"]),
        }, index=score.index)
    elif strategy == "isotonic_projected":
        probability = pd.DataFrame({
            "underutilization_10pct": apply_isotonic(fitted["under10"], score["raw_under10"]),
            "underutilization_25pct": apply_isotonic(fitted["under25"], score["raw_under25"]),
        }, index=score.index)
    elif strategy == "gaussian_distributional":
        probability = pd.DataFrame({
            endpoint: norm.cdf(
                (threshold - score["raw_share"].to_numpy(float) - fitted["residual_mean"])
                / fitted["residual_std"]
            )
            for endpoint, threshold in THRESHOLDS.items()
        }, index=score.index)
    else:
        residuals = fitted["residuals"]
        probability = pd.DataFrame({
            endpoint: (
                np.searchsorted(
                    residuals, threshold - score["raw_share"].to_numpy(float), side="right"
                ) + 0.5
            ) / (len(residuals) + 1)
            for endpoint, threshold in THRESHOLDS.items()
        }, index=score.index)
    return project_probabilities(probability)


def joint_score(probability: pd.DataFrame, actual: pd.DataFrame) -> tuple[float, float]:
    briers, log_losses = [], []
    for endpoint, suffix in [
        ("underutilization_10pct", "under10"),
        ("underutilization_25pct", "under25"),
    ]:
        y, p = actual[f"actual_{suffix}"], probability[endpoint]
        briers.append(brier_score_loss(y, p))
        log_losses.append(log_loss(y, p, labels=[0, 1]))
    return float(np.mean(briers)), float(np.mean(log_losses))


def select_and_apply_policies(
    raw_long: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selection_rows = []
    classification_rows = []
    continuous_rows = []
    for origin_id, _, validation_year, evaluation_year in under.ORIGINS:
        validation_by_model = {
            model: wide_raw(raw_long, model, "validation", origin_id) for model in COMPACT_MODELS
        }
        evaluation_by_model = {
            model: wide_raw(raw_long, model, "evaluation", origin_id) for model in COMPACT_MODELS
        }
        candidate_rows = []
        for model in COMPACT_MODELS:
            validation = validation_by_model[model]
            cut = int(math.floor(0.60 * len(validation)))
            calibration, selector = validation.iloc[:cut], validation.iloc[cut:]
            for strategy in CALIBRATION_STRATEGIES:
                fitted = fit_calibrator(strategy, calibration)
                probability = apply_calibrator(fitted, selector)
                brier, cross_entropy = joint_score(probability, selector)
                candidate_rows.append({
                    "origin_id": origin_id, "validation_year": validation_year,
                    "evaluation_year": evaluation_year, "compact_model": model,
                    "calibration_strategy": strategy, "validation_rows": len(validation),
                    "calibration_rows": len(calibration), "selector_rows": len(selector),
                    "selector_mean_brier": brier,
                    "selector_mean_log_loss": cross_entropy,
                })
        winner = min(
            candidate_rows,
            key=lambda row: (
                row["selector_mean_brier"], row["selector_mean_log_loss"],
                COMPACT_MODELS.index(row["compact_model"]),
                CALIBRATION_STRATEGIES.index(row["calibration_strategy"]),
            ),
        )
        for row in candidate_rows:
            row["selected_for_calibrated_policy"] = (
                row["compact_model"] == winner["compact_model"]
                and row["calibration_strategy"] == winner["calibration_strategy"]
            )
        selection_rows.extend(candidate_rows)

        selected_model = winner["compact_model"]
        validation = validation_by_model[selected_model]
        evaluation = evaluation_by_model[selected_model]
        calibrated = apply_calibrator(
            fit_calibrator(winner["calibration_strategy"], validation), evaluation
        )
        raw_projected = apply_calibrator(fit_calibrator("raw_projected", validation), evaluation)
        for strategy_name, probability in [
            ("selected_compact_raw_projected", raw_projected),
            ("selected_compact_calibrated", calibrated),
        ]:
            for endpoint, actual_column in [
                ("underutilization_10pct", "actual_under10"),
                ("underutilization_25pct", "actual_under25"),
            ]:
                for index, row in evaluation.iterrows():
                    classification_rows.append({
                        "strategy": strategy_name,
                        "selected_compact_model": selected_model,
                        "selected_calibration_strategy": winner["calibration_strategy"],
                        "endpoint": endpoint, "origin_id": origin_id,
                        "evaluation_year": evaluation_year,
                        "transfer_event_id": row["transfer_event_id"],
                        "player_id": row["player_id"], "transfer_date": row["transfer_date"],
                        "actual": int(row[actual_column]),
                        "probability": float(probability.loc[index, endpoint]),
                        "actual_share": float(row["actual_share"]),
                    })
        for model in COMPACT_MODELS:
            evaluation_candidate = evaluation_by_model[model]
            projected = apply_calibrator(
                fit_calibrator("raw_projected", validation_by_model[model]),
                evaluation_candidate,
            )
            for endpoint, actual_column in [
                ("underutilization_10pct", "actual_under10"),
                ("underutilization_25pct", "actual_under25"),
            ]:
                for index, row in evaluation_candidate.iterrows():
                    classification_rows.append({
                        "strategy": f"{model}_raw_projected",
                        "selected_compact_model": model,
                        "selected_calibration_strategy": "raw_projected",
                        "endpoint": endpoint, "origin_id": origin_id,
                        "evaluation_year": evaluation_year,
                        "transfer_event_id": row["transfer_event_id"],
                        "player_id": row["player_id"], "transfer_date": row["transfer_date"],
                        "actual": int(row[actual_column]),
                        "probability": float(projected.loc[index, endpoint]),
                        "actual_share": float(row["actual_share"]),
                    })

        continuous_candidates = []
        for model in COMPACT_MODELS:
            validation = validation_by_model[model]
            continuous_candidates.append((
                float(np.abs(validation["actual_share"] - validation["raw_share"]).mean()), model
            ))
        validation_mae, selected_continuous_model = min(
            continuous_candidates, key=lambda item: (item[0], COMPACT_MODELS.index(item[1]))
        )
        for model in COMPACT_MODELS:
            evaluation_candidate = evaluation_by_model[model]
            for _, row in evaluation_candidate.iterrows():
                continuous_rows.append({
                    "strategy": f"{model}_raw", "selected_compact_model": model,
                    "origin_id": origin_id, "evaluation_year": evaluation_year,
                    "transfer_event_id": row["transfer_event_id"], "player_id": row["player_id"],
                    "transfer_date": row["transfer_date"], "actual": row["actual_share"],
                    "prediction": row["raw_share"],
                    "selected_for_continuous_policy": model == selected_continuous_model,
                    "selected_validation_mae": validation_mae,
                })
    return pd.DataFrame(classification_rows), pd.DataFrame(continuous_rows), pd.DataFrame(selection_rows)


def attach_existing_predictions(
    classification: pd.DataFrame, continuous: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing = pd.read_csv(MODEL_DIR / "underutilization_predictions.csv", low_memory=False)
    class_existing = existing.loc[
        existing["model_variant"].isin([under.FIXED_MODEL_ORDER[0], under.ORDERED_POLICY_MODEL])
        & existing["endpoint"].isin(THRESHOLD_ENDPOINTS)
    ].copy()
    class_existing["strategy"] = np.where(
        class_existing["model_variant"].eq(under.FIXED_MODEL_ORDER[0]),
        "chronological_prevalence", "existing_ordered_policy",
    )
    class_existing = class_existing.rename(columns={"prediction": "probability"})
    class_existing["actual_share"] = class_existing["target_opportunity_share_24m"]
    class_existing["selected_compact_model"] = "not_applicable"
    class_existing["selected_calibration_strategy"] = "not_applicable"
    class_columns = classification.columns.tolist()
    classification = pd.concat(
        [classification, class_existing.reindex(columns=class_columns)], ignore_index=True
    )

    continuous_existing = existing.loc[
        existing["endpoint"].eq("opportunity_share_24m")
        & existing["model_variant"].isin([under.FIXED_MODEL_ORDER[0], under.POLICY_MODEL])
    ].copy()
    continuous_existing["strategy"] = np.where(
        continuous_existing["model_variant"].eq(under.FIXED_MODEL_ORDER[0]),
        "chronological_median", "existing_validation_selected_policy",
    )
    continuous_existing = continuous_existing.rename(columns={"prediction": "prediction"})
    continuous_existing["selected_compact_model"] = "not_applicable"
    continuous_existing["selected_for_continuous_policy"] = False
    continuous_existing["selected_validation_mae"] = np.nan
    continuous = pd.concat(
        [continuous, continuous_existing.reindex(columns=continuous.columns)], ignore_index=True
    )
    return classification, continuous


def expected_calibration_error(actual: pd.Series, probability: pd.Series, bins: int = 10) -> float:
    work = pd.DataFrame({"actual": actual.astype(float), "probability": probability.astype(float)})
    work["bin"] = pd.qcut(
        work["probability"].rank(method="first"), bins, labels=False, duplicates="drop"
    )
    grouped = work.groupby("bin").agg(
        n=("actual", "size"), actual=("actual", "mean"), probability=("probability", "mean")
    )
    return float(
        (grouped["n"] / len(work) * (grouped["actual"] - grouped["probability"]).abs()).sum()
    )


def classification_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy, endpoint, origin_id), group in predictions.groupby(
        ["strategy", "endpoint", "origin_id"], sort=True
    ):
        rows.append({
            "strategy": strategy, "endpoint": endpoint, "origin_id": origin_id,
            "evaluation_year": int(group["evaluation_year"].iloc[0]), "rows": len(group),
            "actual_rate": group["actual"].mean(),
            "predicted_rate": group["probability"].mean(),
            "brier": brier_score_loss(group["actual"], group["probability"]),
            "log_loss": log_loss(group["actual"], group["probability"], labels=[0, 1]),
            "roc_auc": roc_auc_score(group["actual"], group["probability"]),
            "average_precision": average_precision_score(group["actual"], group["probability"]),
            "ece_10bin": expected_calibration_error(group["actual"], group["probability"]),
        })
    return pd.DataFrame(rows)


def cluster_bootstrap(
    pairs: pd.DataFrame, baseline_column: str, candidate_column: str,
    kind: str, seed: int,
) -> dict[str, Any]:
    if kind == "classification":
        difference = np.square(pairs["actual"] - pairs[baseline_column]) - np.square(
            pairs["actual"] - pairs[candidate_column]
        )
    else:
        difference = np.abs(pairs["actual"] - pairs[baseline_column]) - np.abs(
            pairs["actual"] - pairs[candidate_column]
        )
    clusters = pd.DataFrame({"player_id": pairs["player_id"], "difference": difference}).groupby(
        "player_id", sort=True
    )["difference"].agg(["sum", "size"])
    sums, sizes = clusters["sum"].to_numpy(float), clusters["size"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_REPETITIONS)
    for start in range(0, BOOTSTRAP_REPETITIONS, 250):
        stop = min(start + 250, BOOTSTRAP_REPETITIONS)
        indices = rng.integers(0, len(clusters), size=(stop - start, len(clusters)))
        draws[start:stop] = sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)
    return {
        "cluster_count": len(clusters),
        "cluster_ci_lower_95": float(np.quantile(draws, 0.025)),
        "cluster_ci_upper_95": float(np.quantile(draws, 0.975)),
        "cluster_probability_candidate_better": float(np.mean(draws > 0)),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS, "bootstrap_seed": seed,
    }


def classification_comparisons(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs_to_test = [
        ("calibrated_vs_chronological", "chronological_prevalence", "selected_compact_calibrated"),
        ("calibrated_vs_existing_ordered", "existing_ordered_policy", "selected_compact_calibrated"),
        ("calibration_vs_selected_compact_raw", "selected_compact_raw_projected", "selected_compact_calibrated"),
        ("prior_opportunity_vs_market_profile", f"{COMPACT_MODELS[0]}_raw_projected", f"{COMPACT_MODELS[1]}_raw_projected"),
        ("compact_context_vs_prior_opportunity", f"{COMPACT_MODELS[1]}_raw_projected", f"{COMPACT_MODELS[2]}_raw_projected"),
        ("compact_context_vs_market_profile", f"{COMPACT_MODELS[0]}_raw_projected", f"{COMPACT_MODELS[2]}_raw_projected"),
    ]
    for comparison, baseline, candidate in pairs_to_test:
        for endpoint in THRESHOLD_ENDPOINTS:
            subset = predictions.loc[predictions["endpoint"].eq(endpoint)]
            keys = ["origin_id", "evaluation_year", "transfer_event_id", "player_id"]
            left = subset.loc[subset["strategy"].eq(baseline), keys + ["actual", "probability"]].rename(
                columns={"probability": "baseline_probability"}
            )
            right = subset.loc[subset["strategy"].eq(candidate), keys + ["probability"]].rename(
                columns={"probability": "candidate_probability"}
            )
            paired_all = left.merge(right, on=keys, validate="one_to_one")
            for window, years in under.WINDOWS.items():
                pairs = paired_all.loc[paired_all["evaluation_year"].isin(years)].copy()
                baseline_brier = brier_score_loss(pairs["actual"], pairs["baseline_probability"])
                candidate_brier = brier_score_loss(pairs["actual"], pairs["candidate_probability"])
                origin_improvements = []
                for _, group in pairs.groupby("origin_id"):
                    origin_improvements.append(
                        brier_score_loss(group["actual"], group["baseline_probability"])
                        - brier_score_loss(group["actual"], group["candidate_probability"])
                    )
                weights = [len(group) for _, group in pairs.groupby("origin_id")]
                baseline_auc = np.average([
                    roc_auc_score(group["actual"], group["baseline_probability"])
                    for _, group in pairs.groupby("origin_id")
                ], weights=weights)
                candidate_auc = np.average([
                    roc_auc_score(group["actual"], group["candidate_probability"])
                    for _, group in pairs.groupby("origin_id")
                ], weights=weights)
                bootstrap = cluster_bootstrap(
                    pairs, "baseline_probability", "candidate_probability", "classification",
                    stable_seed(comparison, endpoint, window),
                )
                rows.append({
                    "comparison": comparison, "baseline_strategy": baseline,
                    "candidate_strategy": candidate, "endpoint": endpoint, "window": window,
                    "evaluation_years": ";".join(map(str, years)), "pooled_n": len(pairs),
                    "unique_players": pairs["player_id"].nunique(),
                    "baseline_brier": baseline_brier, "candidate_brier": candidate_brier,
                    "brier_improvement": baseline_brier - candidate_brier,
                    "baseline_weighted_origin_auc": baseline_auc,
                    "candidate_weighted_origin_auc": candidate_auc,
                    "auc_change": candidate_auc - baseline_auc,
                    "origin_wins": sum(value > 0 for value in origin_improvements),
                    "origin_count": len(origin_improvements), **bootstrap,
                })
    return pd.DataFrame(rows)


def continuous_comparisons(predictions: pd.DataFrame) -> pd.DataFrame:
    selected = predictions.loc[
        predictions["strategy"].str.endswith("_raw")
        & predictions["selected_for_continuous_policy"].fillna(False)
    ].copy()
    selected["strategy"] = "selected_compact_policy"
    work = pd.concat([predictions, selected], ignore_index=True)
    rows = []
    for comparison, baseline, candidate in [
        ("compact_vs_chronological", "chronological_median", "selected_compact_policy"),
        ("compact_vs_existing_policy", "existing_validation_selected_policy", "selected_compact_policy"),
        ("fixed_compact_context_vs_existing_policy", "existing_validation_selected_policy", f"{COMPACT_MODELS[2]}_raw"),
        ("prior_opportunity_vs_market_profile", f"{COMPACT_MODELS[0]}_raw", f"{COMPACT_MODELS[1]}_raw"),
        ("compact_context_vs_prior_opportunity", f"{COMPACT_MODELS[1]}_raw", f"{COMPACT_MODELS[2]}_raw"),
        ("compact_context_vs_market_profile", f"{COMPACT_MODELS[0]}_raw", f"{COMPACT_MODELS[2]}_raw"),
    ]:
        keys = ["origin_id", "evaluation_year", "transfer_event_id", "player_id"]
        left = work.loc[work["strategy"].eq(baseline), keys + ["actual", "prediction"]].rename(
            columns={"prediction": "baseline_prediction"}
        )
        right = work.loc[work["strategy"].eq(candidate), keys + ["prediction"]].rename(
            columns={"prediction": "candidate_prediction"}
        )
        paired_all = left.merge(right, on=keys, validate="one_to_one")
        for window, years in under.WINDOWS.items():
            pairs = paired_all.loc[paired_all["evaluation_year"].isin(years)].copy()
            baseline_mae = np.abs(pairs["actual"] - pairs["baseline_prediction"]).mean()
            candidate_mae = np.abs(pairs["actual"] - pairs["candidate_prediction"]).mean()
            origin_improvements = [
                np.abs(group["actual"] - group["baseline_prediction"]).mean()
                - np.abs(group["actual"] - group["candidate_prediction"]).mean()
                for _, group in pairs.groupby("origin_id")
            ]
            bootstrap = cluster_bootstrap(
                pairs, "baseline_prediction", "candidate_prediction", "regression",
                stable_seed(comparison, window),
            )
            rows.append({
                "comparison": comparison, "baseline_strategy": baseline,
                "candidate_strategy": candidate, "window": window,
                "evaluation_years": ";".join(map(str, years)), "pooled_n": len(pairs),
                "unique_players": pairs["player_id"].nunique(),
                "baseline_mae": baseline_mae, "candidate_mae": candidate_mae,
                "mae_improvement": baseline_mae - candidate_mae,
                "origin_wins": sum(value > 0 for value in origin_improvements),
                "origin_count": len(origin_improvements), **bootstrap,
            })
    return pd.DataFrame(rows)


def build_consistency(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy, origin_id), group in predictions.groupby(["strategy", "origin_id"]):
        wide = group.pivot(index="transfer_event_id", columns="endpoint", values="probability")
        violations = wide["underutilization_10pct"] > wide["underutilization_25pct"] + 1e-12
        rows.append({
            "strategy": strategy, "origin_id": origin_id, "rows": len(wide),
            "order_violation_rows": int(violations.sum()),
            "order_violation_rate": float(violations.mean()),
        })
    return pd.DataFrame(rows)


def build_risk_bands(predictions: pd.DataFrame) -> pd.DataFrame:
    selected = predictions.loc[predictions["strategy"].isin([
        "existing_ordered_policy", "selected_compact_calibrated"
    ])].copy()
    selected["risk_band"] = selected.groupby(["strategy", "endpoint", "origin_id"])[
        "probability"
    ].transform(lambda values: pd.qcut(values.rank(method="first"), 5, labels=False) + 1)
    return selected.groupby(
        ["strategy", "endpoint", "risk_band"], as_index=False
    ).agg(
        rows=("actual", "size"), unique_players=("player_id", "nunique"),
        mean_probability=("probability", "mean"), observed_rate=("actual", "mean"),
        mean_actual_share=("actual_share", "mean"),
    )


def subgroup_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[[
        "transfer_event_id", "historical_position_group", "age_at_transfer", "fee_band",
        "destination_competition_id", "season_start_year",
    ]].copy()
    result["age_band"] = pd.cut(
        result["age_at_transfer"], [-np.inf, 21, 24, 27, 30, np.inf],
        labels=["21_or_younger", "22_to_24", "25_to_27", "28_to_30", "31_or_older"],
    )
    return result


def build_subgroup_stability(predictions: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["endpoint", "origin_id", "evaluation_year", "transfer_event_id", "player_id"]
    existing = predictions.loc[
        predictions["strategy"].eq("existing_ordered_policy"), keys + ["actual", "probability"]
    ].rename(columns={"probability": "existing_probability"})
    calibrated = predictions.loc[
        predictions["strategy"].eq("selected_compact_calibrated"), keys + ["probability"]
    ].rename(columns={"probability": "candidate_probability"})
    paired = existing.merge(calibrated, on=keys, validate="one_to_one").merge(
        subgroup_frame(frame), on="transfer_event_id", validate="many_to_one"
    )
    rows = []
    dimensions = [
        "historical_position_group", "age_band", "fee_band",
        "destination_competition_id", "evaluation_year",
    ]
    for endpoint in THRESHOLD_ENDPOINTS:
        endpoint_rows = paired.loc[paired["endpoint"].eq(endpoint)]
        for dimension in dimensions:
            for category, group in endpoint_rows.groupby(dimension, dropna=False, observed=True):
                adequate = len(group) >= MIN_SUBGROUP_ROWS and group["origin_id"].nunique() >= MIN_SUBGROUP_ORIGINS
                existing_brier = brier_score_loss(group["actual"], group["existing_probability"])
                candidate_brier = brier_score_loss(group["actual"], group["candidate_probability"])
                row = {
                    "endpoint": endpoint, "dimension": dimension, "category": str(category),
                    "rows": len(group), "unique_players": group["player_id"].nunique(),
                    "origin_count": group["origin_id"].nunique(),
                    "adequate_for_inference": adequate, "actual_rate": group["actual"].mean(),
                    "existing_brier": existing_brier, "candidate_brier": candidate_brier,
                    "brier_improvement": existing_brier - candidate_brier,
                }
                if adequate:
                    bootstrap = cluster_bootstrap(
                        group, "existing_probability", "candidate_probability", "classification",
                        stable_seed("subgroup", endpoint, dimension, str(category)),
                    )
                    row.update(bootstrap)
                    row["stability_label"] = (
                        "supported_positive" if bootstrap["cluster_ci_lower_95"] > 0
                        else ("supported_negative" if bootstrap["cluster_ci_upper_95"] < 0 else "uncertain")
                    )
                else:
                    row.update({
                        "cluster_count": np.nan, "cluster_ci_lower_95": np.nan,
                        "cluster_ci_upper_95": np.nan,
                        "cluster_probability_candidate_better": np.nan,
                        "bootstrap_repetitions": np.nan, "bootstrap_seed": np.nan,
                        "stability_label": "descriptive_only",
                    })
                rows.append(row)
    return pd.DataFrame(rows)


def weighted_ece(metrics: pd.DataFrame, strategy: str, endpoint: str) -> float:
    subset = metrics.loc[
        metrics["strategy"].eq(strategy) & metrics["endpoint"].eq(endpoint)
    ]
    return float(np.average(subset["ece_10bin"], weights=subset["rows"]))


def build_decisions(
    comparisons: pd.DataFrame,
    continuous: pd.DataFrame,
    metrics: pd.DataFrame,
    consistency: pd.DataFrame,
    subgroups: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    pooled = comparisons.loc[comparisons["window"].eq("pooled_four")].set_index(
        ["comparison", "endpoint"]
    )
    violations = int(consistency.loc[
        consistency["strategy"].eq("selected_compact_calibrated"), "order_violation_rows"
    ].sum())
    for endpoint in THRESHOLD_ENDPOINTS:
        baseline = pooled.loc[("calibrated_vs_chronological", endpoint)]
        replacement = pooled.loc[("calibrated_vs_existing_ordered", endpoint)]
        existing_ece = weighted_ece(metrics, "existing_ordered_policy", endpoint)
        calibrated_ece = weighted_ece(metrics, "selected_compact_calibrated", endpoint)
        supported_negative = int(subgroups.loc[
            subgroups["endpoint"].eq(endpoint)
            & subgroups["adequate_for_inference"]
            & subgroups["stability_label"].eq("supported_negative")
        ].shape[0])
        ranking_validated = (
            baseline["cluster_ci_lower_95"] > 0 and baseline["origin_wins"] >= 3
        )
        strict_replace = (
            replacement["cluster_ci_lower_95"] > 0
            and replacement["origin_wins"] >= 3
            and replacement["auc_change"] >= -0.005
            and calibrated_ece < existing_ece
            and violations == 0
            and supported_negative == 0
        )
        calibrated_noninferior = (
            replacement["cluster_ci_lower_95"] > -0.003
            and replacement["auc_change"] >= -0.01
            and calibrated_ece < existing_ece
            and violations == 0
            and supported_negative == 0
        )
        if strict_replace:
            recommendation = "replace_existing_with_compact_calibrated_policy"
        elif calibrated_noninferior:
            recommendation = "use_compact_calibrated_policy_for_probabilities_retain_existing_as_ranking_reference"
        else:
            recommendation = "retain_existing_ordered_ranking_and_use_risk_bands_not_absolute_probabilities"
        rows.append({
            "endpoint": endpoint, "ranking_vs_chronological_validated": ranking_validated,
            "baseline_brier_improvement": baseline["brier_improvement"],
            "baseline_ci_lower_95": baseline["cluster_ci_lower_95"],
            "baseline_ci_upper_95": baseline["cluster_ci_upper_95"],
            "baseline_origin_wins": int(baseline["origin_wins"]),
            "replacement_brier_improvement": replacement["brier_improvement"],
            "replacement_ci_lower_95": replacement["cluster_ci_lower_95"],
            "replacement_ci_upper_95": replacement["cluster_ci_upper_95"],
            "replacement_origin_wins": int(replacement["origin_wins"]),
            "replacement_auc_change": replacement["auc_change"],
            "existing_weighted_ece": existing_ece,
            "calibrated_weighted_ece": calibrated_ece,
            "ordered_probability_violations": violations,
            "adequate_supported_negative_subgroups": supported_negative,
            "strict_replacement_gate": strict_replace,
            "calibrated_noninferiority_gate": calibrated_noninferior,
            "recommendation": recommendation,
        })
    continuous_pooled = continuous.loc[
        continuous["comparison"].eq("fixed_compact_context_vs_existing_policy")
        & continuous["window"].eq("pooled_four")
    ].iloc[0]
    rows.append({
        "endpoint": "opportunity_share_24m",
        "ranking_vs_chronological_validated": bool(
            continuous.loc[
                continuous["comparison"].eq("compact_vs_chronological")
                & continuous["window"].eq("pooled_four"), "cluster_ci_lower_95"
            ].iloc[0] > 0
        ),
        "replacement_brier_improvement": np.nan,
        "replacement_ci_lower_95": continuous_pooled["cluster_ci_lower_95"],
        "replacement_ci_upper_95": continuous_pooled["cluster_ci_upper_95"],
        "replacement_origin_wins": int(continuous_pooled["origin_wins"]),
        "recommendation": (
            "replace_existing_continuous_policy_with_fixed_compact_context" if continuous_pooled["cluster_ci_lower_95"] > 0 and continuous_pooled["origin_wins"] >= 3
            else "retain_existing_continuous_policy"
        ),
    })
    return pd.DataFrame(rows)


def build_checks(
    raw: pd.DataFrame, feature_sets: dict[str, list[str]], raw_selections: pd.DataFrame,
    calibration_selection: pd.DataFrame, classification: pd.DataFrame,
    consistency: pd.DataFrame,
) -> pd.DataFrame:
    checks = []
    def add(check: str, passed: bool, observed: Any, expected: Any, note: str) -> None:
        checks.append({
            "check": check, "passed": bool(passed), "observed": observed,
            "expected": expected, "note": note,
        })
    add("compact_feature_counts", {name: len(value) for name, value in feature_sets.items()} == dict(zip(COMPACT_MODELS, [11, 19, 28])), json.dumps({name: len(value) for name, value in feature_sets.items()}, sort_keys=True), "11;19;28", "Frozen compact blocks.")
    add("raw_prediction_key_unique", not raw.duplicated(["split", "endpoint", "origin_id", "compact_model", "transfer_event_id"]).any(), raw.duplicated(["split", "endpoint", "origin_id", "compact_model", "transfer_event_id"]).sum(), 0, "Raw forecast grain unique.")
    add("nested_temporal_layers", (raw_selections["internal_model_train_end_year"] < raw_selections["internal_model_selector_year"]).all() and (raw_selections["internal_model_selector_year"] < raw_selections["calibration_validation_year"]).all() and (raw_selections["calibration_validation_year"] < raw_selections["evaluation_year"]).all(), True, True, "Model tuning, calibration, and evaluation strictly ordered.")
    selected = calibration_selection.loc[calibration_selection["selected_for_calibrated_policy"].astype(bool)]
    add("one_policy_per_origin", len(selected) == 4 and selected.groupby("origin_id").size().eq(1).all(), len(selected), 4, "Joint compact/calibration selection unique.")
    add("selection_candidates_complete", len(calibration_selection) == 4 * 3 * 5, len(calibration_selection), 60, "Every compact block/strategy tested in each origin.")
    add("classification_key_unique", not classification.duplicated(["strategy", "endpoint", "origin_id", "transfer_event_id"]).any(), classification.duplicated(["strategy", "endpoint", "origin_id", "transfer_event_id"]).sum(), 0, "Final forecast grain unique.")
    add("probability_bounds", classification["probability"].between(0, 1).all(), f"{classification['probability'].min()}..{classification['probability'].max()}", "0..1", "All probabilities bounded.")
    policy_violations = int(consistency.loc[consistency["strategy"].eq("selected_compact_calibrated"), "order_violation_rows"].sum())
    add("calibrated_policy_ordered", policy_violations == 0, policy_violations, 0, "Joint policy probabilities nested.")
    policy_rows = classification.loc[classification["strategy"].eq("selected_compact_calibrated")]
    add("evaluation_rows", policy_rows["transfer_event_id"].nunique() == 584, policy_rows["transfer_event_id"].nunique(), 584, "All frozen evaluation events scored.")
    forbidden = [feature for fields in feature_sets.values() for feature in fields if any(token in feature.lower() for token in ["post12", "post24", "target_", "eligible_"])]
    add("no_outcome_predictors", not forbidden, ";".join(sorted(set(forbidden))), "none", "Compact predictors are pre-transfer only.")
    return pd.DataFrame(checks)


def write_readme(
    selection: pd.DataFrame, comparisons: pd.DataFrame, continuous: pd.DataFrame,
    metrics: pd.DataFrame, decisions: pd.DataFrame, bands: pd.DataFrame,
) -> None:
    selected = selection.loc[selection["selected_for_calibrated_policy"].astype(bool)]
    lines = [
        "# Paid-transfer under-utilization calibration and compact refinement", "",
        "This experiment uses an earlier season for raw-model selection, a later validation season for compact-block/calibration selection, and the following untouched season for evaluation.", "",
        "## Selected joint policies", "",
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"- `{row.origin_id}`: `{row.compact_model}` + `{row.calibration_strategy}` (selector mean Brier {row.selector_mean_brier:.6f})."
        )
    lines += ["", "## Evaluation decision", ""]
    for row in decisions.itertuples(index=False):
        if row.endpoint == "opportunity_share_24m":
            lines.append(f"- `{row.endpoint}`: `{row.recommendation}`.")
        else:
            lines.append(
                f"- `{row.endpoint}`: existing-vs-calibrated Brier improvement {row.replacement_brier_improvement:.6f}, 95% CI [{row.replacement_ci_lower_95:.6f}, {row.replacement_ci_upper_95:.6f}], ECE {row.existing_weighted_ece:.6f} -> {row.calibrated_weighted_ece:.6f}; `{row.recommendation}`."
            )
    lines += ["", "## Guardrail", "",
        "Calibration can correct probability scale but cannot create discrimination or establish causation. Compact context is retained only when validation selects it; no result should be described as proof of tactical compatibility, financial loss, or ROI.",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    verification = json.loads(MODEL_VERIFICATION_PATH.read_text(encoding="utf-8"))
    if verification.get("status") != "pass":
        raise RuntimeError("The upstream under-utilization model is not independently verified.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    protected_before = protected_tree_sha256()
    frame, full_feature_sets, full_manifest = under.build_model_frame()
    feature_sets, manifest = compact_feature_sets(full_feature_sets, full_manifest)
    raw, tuning, raw_selections = build_raw_predictions(frame, feature_sets)
    classification, continuous_predictions, calibration_selection = select_and_apply_policies(raw)
    classification, continuous_predictions = attach_existing_predictions(
        classification, continuous_predictions
    )
    metrics = classification_metrics(classification)
    comparisons = classification_comparisons(classification)
    continuous = continuous_comparisons(continuous_predictions)
    consistency = build_consistency(classification)
    bands = build_risk_bands(classification)
    subgroups = build_subgroup_stability(classification, frame)
    decisions = build_decisions(comparisons, continuous, metrics, consistency, subgroups)
    checks = build_checks(
        raw, feature_sets, raw_selections, calibration_selection, classification, consistency
    )
    if not checks["passed"].all():
        raise RuntimeError(f"Build checks failed: {checks.loc[~checks['passed'], 'check'].tolist()}")

    tables = {
        "compact_feature_manifest.csv": manifest,
        "raw_model_tuning_results.csv": tuning,
        "raw_model_selections.csv": raw_selections,
        "raw_validation_evaluation_predictions.csv": raw,
        "calibration_policy_selection.csv": calibration_selection,
        "classification_predictions.csv": classification,
        "classification_origin_metrics.csv": metrics,
        "classification_comparison_results.csv": comparisons,
        "continuous_predictions.csv": continuous_predictions,
        "continuous_comparison_results.csv": continuous,
        "probability_consistency_audit.csv": consistency,
        "calibrated_risk_bands.csv": bands,
        "subgroup_stability.csv": subgroups,
        "refinement_decision_summary.csv": decisions,
        "build_checks.csv": checks,
    }
    for name, table in tables.items():
        table.to_csv(OUTPUT_DIR / name, index=False)
    source_paths = [
        MODEL_DIR / "underutilization_predictions.csv",
        MODEL_DIR / "underutilization_origin_results.csv",
        MODEL_DIR / "underutilization_targets.csv", MODEL_VERIFICATION_PATH,
        under.MASTER_PATH, under.AUDIT_PATH,
    ]
    sources = pd.DataFrame([
        {
            "source": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in source_paths
    ])
    sources.to_csv(OUTPUT_DIR / "source_manifest.csv", index=False)
    tables["source_manifest.csv"] = sources
    write_readme(calibration_selection, comparisons, continuous, metrics, decisions, bands)
    if protected_before != protected_tree_sha256():
        raise RuntimeError("A protected verified output changed during refinement.")

    selected = calibration_selection.loc[
        calibration_selection["selected_for_calibrated_policy"].astype(bool)
    ]
    run_summary = {
        "cohort_rows": len(frame), "evaluation_rows": 584,
        "compact_feature_counts": {name: len(value) for name, value in feature_sets.items()},
        "selected_joint_policies": {
            row.origin_id: {
                "compact_model": row.compact_model,
                "calibration_strategy": row.calibration_strategy,
            }
            for row in selected.itertuples(index=False)
        },
        "recommendations": decisions.set_index("endpoint")["recommendation"].to_dict(),
        "policy_order_violations": int(consistency.loc[
            consistency["strategy"].eq("selected_compact_calibrated"),
            "order_violation_rows",
        ].sum()),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "protected_outputs_unchanged": True,
        "build_checks": len(checks), "build_checks_passed": int(checks["passed"].sum()),
    }
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_names = sorted(list(tables) + ["README.md", "run_summary.json"])
    output_manifest = pd.DataFrame([
        {
            "file": name, "size_bytes": (OUTPUT_DIR / name).stat().st_size,
            "sha256": sha256_file(OUTPUT_DIR / name),
        }
        for name in manifest_names
    ])
    output_manifest.to_csv(OUTPUT_DIR / "output_manifest.csv", index=False)
    print(json.dumps(run_summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

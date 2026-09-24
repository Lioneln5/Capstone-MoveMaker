"""Run 3: reliability audit for the repaired contribution candidate.

This is a development-only audit. It does not search new features, inspect
2024+ outcomes, serialize a model, or modify deployment.
"""

from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "models") not in sys.path:
    sys.path.insert(0, str(ROOT / "models"))

from models.engine_v2.contribution_model_development_contract import (  # noqa: E402
    LOGISTIC_C_GRID,
    ORIGINS,
)
from models.engine_v2.complete_contribution_target_contract import (  # noqa: E402
    TARGET_FIELD,
)
from models.engine_v2.contribution_reliability_contract import (  # noqa: E402
    CALIBRATION_BRIER_NONINFERIORITY,
    CALIBRATION_ECE_NEAR_TIE,
    CALIBRATION_LOG_LOSS_NONINFERIORITY,
    CALIBRATION_METHODS,
    CALIBRATION_SIMPLICITY,
    CONTRACT_VERSION,
    HARD_REQUIRED_FEATURES,
    INVOLVEMENT_FEATURES,
    MAXIMUM_EXTREME_PROBABILITY_SHARE,
    MAXIMUM_LATEST_INVOLVEMENT_PSI,
    MAXIMUM_MEAN_PARAMETER_WIDTH_80,
    MAXIMUM_ORIGIN_CALIBRATION_GAP,
    MAXIMUM_ORIGIN_ECE,
    MAXIMUM_P90_PARAMETER_WIDTH_80,
    MAXIMUM_POOLED_CALIBRATION_GAP,
    MAXIMUM_POOLED_ECE,
    MAXIMUM_REFUSAL_RATE,
    MAXIMUM_SUBGROUP_CALIBRATION_GAP,
    MAXIMUM_TARGET_RATE_RANGE,
    MAXIMUM_CALIBRATION_SLOPE,
    MINIMUM_CALIBRATION_SLOPE,
    MINIMUM_POSITIVE_SKILL_ORIGINS,
    MINIMUM_RELIABLE_ORIGINS,
    MINIMUM_SUBGROUP_AUC,
    MINIMUM_SUBGROUP_EVENTS,
    MINIMUM_SUBGROUP_NONEVENTS,
    MINIMUM_SUBGROUP_ORIGINS,
    MINIMUM_SUBGROUP_ROWS,
    MODEL_CONTRACT_VERSION,
    OOD_LIMITED_PERCENTILE,
    OOD_REFUSAL_PERCENTILE,
    PARAMETER_BOOTSTRAP_REPETITIONS,
    PSI_HIGH_THRESHOLD,
    PSI_MODERATE_THRESHOLD,
    RANDOM_SEED,
    REDUNDANCY_SPEARMAN_THRESHOLD,
    REQUIRED_SUBGROUP_TYPES,
    SUBGROUP_BOOTSTRAP_REPETITIONS,
    SUBGROUP_INTERVAL_ALPHA,
    TARGET_CONTRACT_VERSION,
    contract_payload,
)
from models.engine_v2.feature_contract import V2_CORE_FEATURES  # noqa: E402
from models.mixed_type_preprocessor import fit_preprocessor  # noqa: E402
from models.run_engine_v2_contribution_model_rerun import prepare_model_frame  # noqa: E402


RUN2 = ROOT / "Data" / "processed" / "engine_v2_contribution_model_rerun"
OUTPUT = ROOT / "Data" / "processed" / "engine_v2_contribution_reliability_audit"
DEPLOYMENT = ROOT / "Data" / "processed" / "deployment_models"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"
TARGET = TARGET_FIELD
FEATURES = list(V2_CORE_FEATURES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: str) -> int:
    token = "|".join(parts).encode("utf-8")
    return (RANDOM_SEED + int(hashlib.sha256(token).hexdigest()[:8], 16)) % (2**32 - 1)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def write_json(payload: object, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def deployment_fingerprint() -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in sorted(DEPLOYMENT.rglob("*"))
        if path.is_file()
    }


def logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def fit_calibrator(method: str, probability: np.ndarray, actual: np.ndarray):
    if method == "raw":
        return None
    if method == "platt":
        return LogisticRegression(
            C=1e6,
            solver="lbfgs",
            max_iter=3000,
            random_state=RANDOM_SEED,
        ).fit(logit(probability).reshape(-1, 1), actual)
    if method == "isotonic":
        return IsotonicRegression(
            out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6
        ).fit(probability, actual)
    raise KeyError(method)


def apply_calibrator(method: str, calibrator, probability: np.ndarray) -> np.ndarray:
    raw = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    if method == "raw":
        return raw
    if method == "platt":
        return np.clip(
            calibrator.predict_proba(logit(raw).reshape(-1, 1))[:, 1],
            1e-6,
            1 - 1e-6,
        )
    return np.clip(calibrator.predict(raw), 1e-6, 1 - 1e-6)


def adaptive_ece(actual: np.ndarray, prediction: np.ndarray, bins: int = 10) -> float:
    ordered = pd.DataFrame({"actual": actual, "prediction": prediction}).sort_values(
        "prediction"
    )
    groups = np.array_split(np.arange(len(ordered)), min(bins, len(ordered)))
    return float(
        sum(
            len(indices)
            * abs(
                ordered.iloc[indices]["actual"].mean()
                - ordered.iloc[indices]["prediction"].mean()
            )
            for indices in groups
        )
        / len(ordered)
    )


def probability_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=int)
    prediction = np.clip(np.asarray(prediction, dtype=float), 1e-6, 1 - 1e-6)
    calibration = LogisticRegression(
        C=1e6, solver="lbfgs", max_iter=3000, random_state=RANDOM_SEED
    ).fit(logit(prediction).reshape(-1, 1), actual)
    return {
        "brier": float(brier_score_loss(actual, prediction)),
        "log_loss": float(log_loss(actual, prediction, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(actual, prediction)),
        "average_precision": float(average_precision_score(actual, prediction)),
        "adaptive_ece": adaptive_ece(actual, prediction),
        "actual_rate": float(actual.mean()),
        "predicted_rate": float(prediction.mean()),
        "calibration_gap": float(abs(actual.mean() - prediction.mean())),
        "calibration_intercept": float(calibration.intercept_[0]),
        "calibration_slope": float(calibration.coef_[0][0]),
    }


def split_origin(data: pd.DataFrame, train_end: int, validation_year: int, evaluation_year: int):
    base_train = data.loc[data["signed_year"].le(train_end)].copy()
    validation = data.loc[data["signed_year"].eq(validation_year)].copy()
    evaluation = data.loc[data["signed_year"].eq(evaluation_year)].sort_values(
        "capology_extension_event_id"
    ).copy()
    validation_players = set(validation["canonical_player_id"].dropna())
    tuning_train = base_train.loc[
        ~base_train["canonical_player_id"].isin(validation_players)
    ].copy()
    final_fit = pd.concat([base_train, validation], ignore_index=True)
    evaluation_players = set(evaluation["canonical_player_id"].dropna())
    final_fit = final_fit.loc[
        ~final_fit["canonical_player_id"].isin(evaluation_players)
    ].copy()
    return tuning_train, validation, final_fit, evaluation


def tune_c(tuning_train: pd.DataFrame, validation: pd.DataFrame, features: list[str]):
    pre = fit_preprocessor(tuning_train, features)
    x_train = pre.transform(tuning_train)
    x_validation = pre.transform(validation)
    y_train = pd.to_numeric(tuning_train[TARGET]).to_numpy(int)
    y_validation = pd.to_numeric(validation[TARGET]).to_numpy(int)
    trials = []
    for c_value in LOGISTIC_C_GRID:
        model = LogisticRegression(
            C=c_value,
            penalty="l2",
            solver="liblinear",
            max_iter=3000,
            random_state=RANDOM_SEED,
        ).fit(x_train, y_train)
        prediction = np.clip(model.predict_proba(x_validation)[:, 1], 1e-6, 1 - 1e-6)
        trials.append(
            (
                float(brier_score_loss(y_validation, prediction)),
                float(log_loss(y_validation, prediction, labels=[0, 1])),
                c_value,
                prediction,
            )
        )
    return min(trials, key=lambda item: (item[0], item[1], item[2]))


def fit_origin_models(data: pd.DataFrame):
    prediction_rows: list[dict[str, object]] = []
    calibration_fit_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    support_profiles: list[dict[str, object]] = []
    model_cache: dict[str, dict[str, object]] = {}
    for origin, train_end, validation_year, evaluation_year in ORIGINS:
        tuning, validation, final_fit, evaluation = split_origin(
            data, train_end, validation_year, evaluation_year
        )
        _, _, selected_c, validation_raw = tune_c(tuning, validation, FEATURES)
        validation_actual = pd.to_numeric(validation[TARGET]).to_numpy(int)
        calibrators = {
            method: fit_calibrator(method, validation_raw, validation_actual)
            for method in CALIBRATION_METHODS
        }
        pre = fit_preprocessor(final_fit, FEATURES)
        model = LogisticRegression(
            C=selected_c,
            penalty="l2",
            solver="liblinear",
            max_iter=3000,
            random_state=RANDOM_SEED,
        ).fit(pre.transform(final_fit), pd.to_numeric(final_fit[TARGET]).to_numpy(int))
        evaluation_raw = np.clip(
            model.predict_proba(pre.transform(evaluation))[:, 1], 1e-6, 1 - 1e-6
        )
        profile = build_support_profile(final_fit)
        profile.update(
            {
                "origin_id": origin,
                "train_end_year": train_end,
                "validation_year": validation_year,
                "evaluation_year": evaluation_year,
            }
        )
        support_profiles.append(profile)
        support = assess_support(evaluation, profile)
        baseline = float(pd.to_numeric(final_fit[TARGET]).mean())
        actual = pd.to_numeric(evaluation[TARGET]).to_numpy(int)
        for method in CALIBRATION_METHODS:
            estimates = apply_calibrator(method, calibrators[method], evaluation_raw)
            for i, (_, row) in enumerate(evaluation.iterrows()):
                prediction_rows.append(
                    {
                        "origin_id": origin,
                        "evaluation_year": evaluation_year,
                        "calibration_method": method,
                        "capology_extension_event_id": row["capology_extension_event_id"],
                        "canonical_player_id": row["canonical_player_id"],
                        "canonical_position": row["canonical_position"],
                        "league": row["league"],
                        "age_band": row["age_band"],
                        "actual": int(actual[i]),
                        "prediction": float(estimates[i]),
                        "raw_prediction": float(evaluation_raw[i]),
                        "baseline_prediction": baseline,
                        "brier_loss": float((actual[i] - estimates[i]) ** 2),
                        "baseline_brier_loss": float((actual[i] - baseline) ** 2),
                        **support.iloc[i].to_dict(),
                    }
                )
        for i, (_, row) in enumerate(validation.sort_values("capology_extension_event_id").iterrows()):
            # tune_c preserves the original validation order, so map explicitly.
            original_index = validation.index.get_loc(row.name)
            calibration_fit_rows.append(
                {
                    "origin_id": origin,
                    "validation_year": validation_year,
                    "capology_extension_event_id": row["capology_extension_event_id"],
                    "canonical_player_id": row["canonical_player_id"],
                    "actual": int(validation_actual[original_index]),
                    "raw_prediction": float(validation_raw[original_index]),
                    "selected_C": selected_c,
                }
            )
        for encoded, raw, coefficient in zip(
            pre.encoded_names, pre.encoded_raw_features, model.coef_.reshape(-1)
        ):
            coefficient_rows.append(
                {
                    "origin_id": origin,
                    "encoded_feature": encoded,
                    "raw_feature": raw,
                    "coefficient": float(coefficient),
                }
            )
        model_cache[origin] = {
            "tuning": tuning,
            "validation": validation,
            "final_fit": final_fit,
            "evaluation": evaluation,
            "selected_C": selected_c,
            "validation_raw": validation_raw,
            "calibrators": calibrators,
            "preprocessor": pre,
            "model": model,
            "profile": profile,
        }
    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(calibration_fit_rows),
        pd.DataFrame(coefficient_rows),
        support_profiles,
        model_cache,
    )


def build_support_profile(reference: pd.DataFrame) -> dict[str, object]:
    complete = reference.loc[
        reference[list(HARD_REQUIRED_FEATURES)].notna().all(axis=1)
    ].copy()
    profile: dict[str, object] = {
        "features": FEATURES,
        "complete_reference_rows": len(complete),
        "categorical": {},
        "numeric": {},
    }
    distances = pd.DataFrame(index=complete.index)
    for feature in FEATURES:
        if feature in {"canonical_position", "league"}:
            profile["categorical"][feature] = sorted(
                complete[feature].astype(str).unique().tolist()
            )
            continue
        values = pd.to_numeric(complete[feature], errors="coerce")
        median = float(values.median())
        q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
        scale = max((q3 - q1) / 1.349, 1e-8)
        profile["numeric"][feature] = {
            "median": median,
            "robust_scale": scale,
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "q01": float(values.quantile(0.01)),
            "q99": float(values.quantile(0.99)),
        }
        distances[feature] = (values - median).abs() / scale
    maximum = distances.max(axis=1)
    profile["distance_limited_threshold"] = float(maximum.quantile(OOD_LIMITED_PERCENTILE))
    profile["distance_refusal_threshold"] = float(maximum.quantile(OOD_REFUSAL_PERCENTILE))
    return profile


def assess_support(frame: pd.DataFrame, profile: dict[str, object]) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        refusal: list[str] = []
        limited: list[str] = []
        missing = [feature for feature in HARD_REQUIRED_FEATURES if pd.isna(row[feature])]
        if missing:
            refusal.append("MISSING_REQUIRED_FEATURE:" + "|".join(missing))
        for feature in ("canonical_position", "league"):
            if pd.notna(row[feature]) and str(row[feature]) not in profile["categorical"][feature]:
                refusal.append("UNSEEN_CATEGORY:" + feature)
        distances = []
        if not missing:
            for feature, spec in profile["numeric"].items():
                value = float(row[feature])
                distances.append(abs(value - spec["median"]) / spec["robust_scale"])
                if value < spec["minimum"] or value > spec["maximum"]:
                    refusal.append("OUTSIDE_OBSERVED_RANGE:" + feature)
                elif value < spec["q01"] or value > spec["q99"]:
                    limited.append("OUTSIDE_TRAINING_P01_P99:" + feature)
        distance = max(distances) if distances else np.nan
        if pd.notna(distance) and distance > profile["distance_refusal_threshold"]:
            refusal.append("ROBUST_DISTANCE_ABOVE_TRAINING_P99")
        elif pd.notna(distance) and distance > profile["distance_limited_threshold"]:
            limited.append("ROBUST_DISTANCE_ABOVE_TRAINING_P95")
        status = "refused" if refusal else "limited" if limited else "supported"
        rows.append(
            {
                "support_status": status,
                "support_reasons": ";".join(refusal + limited),
                "support_distance": distance,
            }
        )
    return pd.DataFrame(rows, index=frame.index)


def calibration_evidence(predictions: pd.DataFrame):
    metric_rows = []
    comparison_rows = []
    for (origin, year, method), chunk in predictions.groupby(
        ["origin_id", "evaluation_year", "calibration_method"]
    ):
        metric_rows.append(
            {
                "origin_id": origin,
                "evaluation_year": year,
                "calibration_method": method,
                "rows": len(chunk),
                **probability_metrics(chunk["actual"], chunk["prediction"]),
            }
        )
    metrics_frame = pd.DataFrame(metric_rows)
    raw = predictions.loc[predictions["calibration_method"].eq("raw")]
    raw_metrics = probability_metrics(raw["actual"], raw["prediction"])
    for method in CALIBRATION_METHODS:
        chunk = predictions.loc[predictions["calibration_method"].eq(method)]
        current = probability_metrics(chunk["actual"], chunk["prediction"])
        origin_wins = 0
        if method != "raw":
            for origin in chunk["origin_id"].unique():
                method_loss = chunk.loc[chunk["origin_id"].eq(origin), "brier_loss"].mean()
                raw_loss = raw.loc[raw["origin_id"].eq(origin), "brier_loss"].mean()
                origin_wins += int(method_loss < raw_loss)
        comparison_rows.append(
            {
                "calibration_method": method,
                "rows": len(chunk),
                **{f"pooled_{key}": value for key, value in current.items()},
                "brier_improvement_vs_raw": raw_metrics["brier"] - current["brier"],
                "log_loss_improvement_vs_raw": raw_metrics["log_loss"] - current["log_loss"],
                "ece_improvement_vs_raw": raw_metrics["adaptive_ece"] - current["adaptive_ece"],
                "origin_wins_vs_raw": origin_wins,
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    raw_row = comparison.loc[comparison["calibration_method"].eq("raw")].iloc[0]
    eligible = comparison.loc[
        comparison["pooled_brier"].le(
            raw_row["pooled_brier"] + CALIBRATION_BRIER_NONINFERIORITY
        )
        & comparison["pooled_log_loss"].le(
            raw_row["pooled_log_loss"] + CALIBRATION_LOG_LOSS_NONINFERIORITY
        )
    ].copy()
    best_ece = eligible["pooled_adaptive_ece"].min()
    eligible = eligible.loc[
        eligible["pooled_adaptive_ece"].le(best_ece + CALIBRATION_ECE_NEAR_TIE)
    ].copy()
    eligible["simplicity"] = eligible["calibration_method"].map(CALIBRATION_SIMPLICITY)
    selected_method = eligible.sort_values(["simplicity", "pooled_brier"]).iloc[0][
        "calibration_method"
    ]
    decision = comparison.copy()
    decision["within_brier_noninferiority"] = decision["pooled_brier"].le(
        raw_row["pooled_brier"] + CALIBRATION_BRIER_NONINFERIORITY
    )
    decision["within_log_loss_noninferiority"] = decision["pooled_log_loss"].le(
        raw_row["pooled_log_loss"] + CALIBRATION_LOG_LOSS_NONINFERIORITY
    )
    decision["within_ece_near_tie"] = decision["pooled_adaptive_ece"].le(
        best_ece + CALIBRATION_ECE_NEAR_TIE
    )
    decision["selected"] = decision["calibration_method"].eq(selected_method)
    return metrics_frame, comparison, decision, selected_method


def calibration_bins(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groupings = [("pooled", selected)] + [
        (origin, chunk) for origin, chunk in selected.groupby("origin_id")
    ]
    for scope, chunk in groupings:
        ordered = chunk.sort_values("prediction").reset_index(drop=True)
        assignments = np.concatenate(
            [np.repeat(index + 1, len(part)) for index, part in enumerate(np.array_split(ordered, 10))]
        )
        ordered["bin"] = assignments
        for bin_id, part in ordered.groupby("bin"):
            n = len(part)
            rate = float(part["actual"].mean())
            z = 1.959963984540054
            denominator = 1 + z**2 / n
            center = (rate + z**2 / (2 * n)) / denominator
            half = z * np.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2)) / denominator
            rows.append(
                {
                    "scope": scope,
                    "bin": bin_id,
                    "rows": n,
                    "minimum_prediction": float(part["prediction"].min()),
                    "maximum_prediction": float(part["prediction"].max()),
                    "mean_prediction": float(part["prediction"].mean()),
                    "actual_rate": rate,
                    "actual_rate_wilson_low_95": center - half,
                    "actual_rate_wilson_high_95": center + half,
                    "absolute_gap": abs(rate - float(part["prediction"].mean())),
                }
            )
    return pd.DataFrame(rows)


def cluster_bootstrap_frame(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    clusters = frame["canonical_player_id"].astype(str)
    unique = clusters.unique()
    sampled = rng.choice(unique, size=len(unique), replace=True)
    parts = [frame.loc[clusters.eq(cluster)] for cluster in sampled]
    return pd.concat(parts, ignore_index=True)


def parameter_interval_origin(
    data: pd.DataFrame,
    selected_method: str,
    origin_spec: tuple[str, int, int, int],
) -> tuple[pd.DataFrame, dict[str, object]]:
    origin, train_end, validation_year, evaluation_year = origin_spec
    tuning, validation, _, evaluation = split_origin(
        data, train_end, validation_year, evaluation_year
    )
    rng = np.random.default_rng(stable_seed(origin, selected_method, "nested_parameter"))
    draws = []
    for _ in range(PARAMETER_BOOTSTRAP_REPETITIONS):
        boot_tuning = cluster_bootstrap_frame(tuning, rng)
        boot_validation = cluster_bootstrap_frame(validation, rng)
        if (
            pd.to_numeric(boot_tuning[TARGET]).nunique() < 2
            or pd.to_numeric(boot_validation[TARGET]).nunique() < 2
        ):
            continue
        _, _, selected_c, validation_raw = tune_c(boot_tuning, boot_validation, FEATURES)
        calibrator = fit_calibrator(
            selected_method,
            validation_raw,
            pd.to_numeric(boot_validation[TARGET]).to_numpy(int),
        )
        boot_final = pd.concat([boot_tuning, boot_validation], ignore_index=True)
        pre = fit_preprocessor(boot_final, FEATURES)
        model = LogisticRegression(
            C=selected_c,
            penalty="l2",
            solver="liblinear",
            max_iter=3000,
            random_state=RANDOM_SEED,
        ).fit(pre.transform(boot_final), pd.to_numeric(boot_final[TARGET]).to_numpy(int))
        raw = model.predict_proba(pre.transform(evaluation))[:, 1]
        draws.append(apply_calibrator(selected_method, calibrator, raw))
    matrix = np.asarray(draws)
    if len(matrix) < int(PARAMETER_BOOTSTRAP_REPETITIONS * 0.95):
        raise RuntimeError(f"Too few successful parameter refits for {origin}: {len(matrix)}")
    low = np.quantile(matrix, 0.10, axis=0)
    high = np.quantile(matrix, 0.90, axis=0)
    width = high - low
    part = pd.DataFrame(
        {
            "origin_id": origin,
            "capology_extension_event_id": evaluation["capology_extension_event_id"].to_numpy(),
            "parameter_interval_low_80": low,
            "parameter_interval_high_80": high,
            "parameter_interval_width_80": width,
            "successful_parameter_refits": len(matrix),
        }
    )
    summary = {
        "origin_id": origin,
        "evaluation_year": evaluation_year,
        "rows": len(evaluation),
        "successful_parameter_refits": len(matrix),
        "mean_width_80": float(np.mean(width)),
        "median_width_80": float(np.median(width)),
        "p90_width_80": float(np.quantile(width, 0.90)),
        "maximum_width_80": float(np.max(width)),
    }
    return part, summary


def parameter_intervals(data: pd.DataFrame, selected_method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    results: dict[str, tuple[pd.DataFrame, dict[str, object]]] = {}
    with ThreadPoolExecutor(max_workers=len(ORIGINS)) as executor:
        futures = {
            executor.submit(parameter_interval_origin, data, selected_method, spec): spec[0]
            for spec in ORIGINS
        }
        for future in as_completed(futures):
            origin = futures[future]
            results[origin] = future.result()
            print(f"Completed {origin} nested parameter bootstrap.", flush=True)
    prediction_parts = [results[spec[0]][0] for spec in ORIGINS]
    summary_rows = [results[spec[0]][1] for spec in ORIGINS]
    intervals = pd.concat(prediction_parts, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    summary = pd.concat(
        [
            summary,
            pd.DataFrame(
                [
                    {
                        "origin_id": "pooled",
                        "evaluation_year": pd.NA,
                        "rows": len(intervals),
                        "successful_parameter_refits": int(
                            summary["successful_parameter_refits"].min()
                        ),
                        "mean_width_80": float(
                            intervals["parameter_interval_width_80"].mean()
                        ),
                        "median_width_80": float(
                            intervals["parameter_interval_width_80"].median()
                        ),
                        "p90_width_80": float(
                            intervals["parameter_interval_width_80"].quantile(0.90)
                        ),
                        "maximum_width_80": float(
                            intervals["parameter_interval_width_80"].max()
                        ),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return intervals, summary


def clustered_skill_interval(chunk: pd.DataFrame, seed_parts: tuple[str, ...]):
    grouped = chunk.assign(
        improvement=chunk["baseline_brier_loss"] - chunk["brier_loss"]
    ).groupby("canonical_player_id")["improvement"].agg(["sum", "size"])
    sums = grouped["sum"].to_numpy(float)
    sizes = grouped["size"].to_numpy(float)
    rng = np.random.default_rng(stable_seed(*seed_parts))
    draws = []
    for start in range(0, SUBGROUP_BOOTSTRAP_REPETITIONS, 250):
        count = min(250, SUBGROUP_BOOTSTRAP_REPETITIONS - start)
        indices = rng.integers(0, len(grouped), size=(count, len(grouped)))
        draws.extend((sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)).tolist())
    return (
        float(np.quantile(draws, SUBGROUP_INTERVAL_ALPHA / 2)),
        float(np.quantile(draws, 1 - SUBGROUP_INTERVAL_ALPHA / 2)),
    )


def subgroup_reliability(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_type, column in (
        ("league", "league"),
        ("position", "canonical_position"),
        ("age", "age_band"),
        ("support", "support_status"),
    ):
        population = selected if group_type == "support" else selected.loc[
            ~selected["support_status"].eq("refused")
        ]
        for group, chunk in population.groupby(column, dropna=False):
            events = int(chunk["actual"].sum())
            nonevents = len(chunk) - events
            auc = (
                float(roc_auc_score(chunk["actual"], chunk["prediction"]))
                if events and nonevents
                else np.nan
            )
            brier = float(chunk["brier_loss"].mean())
            baseline = float(chunk["baseline_brier_loss"].mean())
            gap = float(abs(chunk["actual"].mean() - chunk["prediction"].mean()))
            low, high = clustered_skill_interval(
                chunk, ("subgroup", group_type, str(group))
            )
            sufficient = bool(
                len(chunk) >= MINIMUM_SUBGROUP_ROWS
                and events >= MINIMUM_SUBGROUP_EVENTS
                and nonevents >= MINIMUM_SUBGROUP_NONEVENTS
                and chunk["origin_id"].nunique() >= MINIMUM_SUBGROUP_ORIGINS
            )
            supported = bool(
                sufficient
                and auc >= MINIMUM_SUBGROUP_AUC
                and gap <= MAXIMUM_SUBGROUP_CALIBRATION_GAP
                and low > 0
            )
            rows.append(
                {
                    "analysis_population": "all" if group_type == "support" else "non_refused",
                    "group_type": group_type,
                    "group": str(group),
                    "rows": len(chunk),
                    "player_clusters": chunk["canonical_player_id"].nunique(),
                    "events": events,
                    "nonevents": nonevents,
                    "origins": chunk["origin_id"].nunique(),
                    "actual_rate": float(chunk["actual"].mean()),
                    "predicted_rate": float(chunk["prediction"].mean()),
                    "calibration_gap": gap,
                    "brier": brier,
                    "chronology_brier": baseline,
                    "brier_skill": baseline - brier,
                    "brier_skill_ci_low_99": low,
                    "brier_skill_ci_high_99": high,
                    "roc_auc": auc,
                    "mean_parameter_width_80": float(
                        chunk["parameter_interval_width_80"].mean()
                    ),
                    "reliability_status": (
                        "supported"
                        if supported
                        else "limited_reliability"
                        if sufficient
                        else "insufficient_evidence"
                    ),
                }
            )
    return pd.DataFrame(rows)


def population_stability_index(reference: pd.Series, evaluation: pd.Series) -> float:
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    evaluation = pd.to_numeric(evaluation, errors="coerce").dropna()
    edges = np.unique(reference.quantile(np.linspace(0, 1, 11)).to_numpy(float))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = pd.cut(reference, edges, include_lowest=True).value_counts(sort=False) / len(reference)
    eval_share = pd.cut(evaluation, edges, include_lowest=True).value_counts(sort=False) / len(evaluation)
    ref_share = ref_share.clip(lower=1e-6)
    eval_share = eval_share.clip(lower=1e-6)
    return float(((eval_share - ref_share) * np.log(eval_share / ref_share)).sum())


def temporal_drift(model_cache: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for origin, cache in model_cache.items():
        reference, evaluation = cache["final_fit"], cache["evaluation"]
        for feature in FEATURES:
            if feature in {"canonical_position", "league"}:
                ref = reference[feature].astype("string").fillna("__MISSING__").value_counts(normalize=True)
                ev = evaluation[feature].astype("string").fillna("__MISSING__").value_counts(normalize=True)
                categories = ref.index.union(ev.index)
                value = float(0.5 * (ref.reindex(categories, fill_value=0) - ev.reindex(categories, fill_value=0)).abs().sum())
                metric = "total_variation"
                status = "high" if value > 0.25 else "moderate" if value > 0.10 else "low"
            else:
                value = population_stability_index(reference[feature], evaluation[feature])
                metric = "psi"
                status = "high" if value > PSI_HIGH_THRESHOLD else "moderate" if value > PSI_MODERATE_THRESHOLD else "low"
            rows.append(
                {
                    "origin_id": origin,
                    "evaluation_year": int(evaluation["signed_year"].iloc[0]),
                    "feature": feature,
                    "drift_metric": metric,
                    "drift_value": value,
                    "reference_missing_rate": float(reference[feature].isna().mean()),
                    "evaluation_missing_rate": float(evaluation[feature].isna().mean()),
                    "drift_status": status,
                }
            )
    return pd.DataFrame(rows)


def redundancy_stress(data: pd.DataFrame, raw_predictions: pd.DataFrame) -> pd.DataFrame:
    variants = {
        "full": FEATURES,
        "drop_opportunity": [f for f in FEATURES if f != INVOLVEMENT_FEATURES[0]],
        "drop_appearance": [f for f in FEATURES if f != INVOLVEMENT_FEATURES[1]],
    }
    rows = []
    for origin, train_end, validation_year, evaluation_year in ORIGINS:
        tuning, validation, final_fit, evaluation = split_origin(
            data, train_end, validation_year, evaluation_year
        )
        full = raw_predictions.loc[raw_predictions["origin_id"].eq(origin)].sort_values(
            "capology_extension_event_id"
        )
        for variant, features in variants.items():
            _, _, selected_c, _ = tune_c(tuning, validation, features)
            pre = fit_preprocessor(final_fit, features)
            model = LogisticRegression(
                C=selected_c,
                penalty="l2",
                solver="liblinear",
                max_iter=3000,
                random_state=RANDOM_SEED,
            ).fit(pre.transform(final_fit), pd.to_numeric(final_fit[TARGET]).to_numpy(int))
            estimate = model.predict_proba(pre.transform(evaluation))[:, 1]
            actual = pd.to_numeric(evaluation[TARGET]).to_numpy(int)
            rows.append(
                {
                    "origin_id": origin,
                    "evaluation_year": evaluation_year,
                    "variant": variant,
                    "selected_C": selected_c,
                    "rows": len(evaluation),
                    "brier": float(brier_score_loss(actual, estimate)),
                    "brier_change_vs_full": float(
                        brier_score_loss(actual, full["prediction"])
                        - brier_score_loss(actual, estimate)
                    ),
                    "prediction_correlation_with_full": float(
                        np.corrcoef(full["prediction"], estimate)[0, 1]
                    ),
                    "involvement_spearman_in_fit": float(
                        final_fit[list(INVOLVEMENT_FEATURES)].corr(method="spearman").iloc[0, 1]
                    ),
                }
            )
    return pd.DataFrame(rows)


def monotonicity_stress(model_cache: dict[str, dict[str, object]], selected_method: str) -> pd.DataFrame:
    rows = []
    for origin, cache in model_cache.items():
        evaluation = cache["evaluation"].copy()
        support = assess_support(evaluation, cache["profile"])
        evaluation = evaluation.loc[~support["support_status"].eq("refused")].copy()
        base_raw = cache["model"].predict_proba(
            cache["preprocessor"].transform(evaluation)
        )[:, 1]
        base = apply_calibrator(selected_method, cache["calibrators"][selected_method], base_raw)
        for feature in INVOLVEMENT_FEATURES:
            eligible = evaluation.loc[pd.to_numeric(evaluation[feature], errors="coerce").le(0.95)].copy()
            positions = evaluation.index.get_indexer(eligible.index)
            perturbed = eligible.copy()
            perturbed[feature] = (pd.to_numeric(perturbed[feature]) + 0.05).clip(upper=1.0)
            changed_raw = cache["model"].predict_proba(
                cache["preprocessor"].transform(perturbed)
            )[:, 1]
            changed = apply_calibrator(
                selected_method, cache["calibrators"][selected_method], changed_raw
            )
            delta = changed - base[positions]
            rows.append(
                {
                    "origin_id": origin,
                    "evaluation_year": int(cache["evaluation"]["signed_year"].iloc[0]),
                    "feature": feature,
                    "perturbation": 0.05,
                    "eligible_rows": len(delta),
                    "negative_delta_rows": int((delta < -1e-12).sum()),
                    "zero_delta_rows": int(np.isclose(delta, 0, atol=1e-12).sum()),
                    "mean_probability_delta": float(delta.mean()),
                    "minimum_probability_delta": float(delta.min()),
                    "maximum_probability_delta": float(delta.max()),
                }
            )
    return pd.DataFrame(rows)


def missingness_reliability(selected: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    additional_features = [feature for feature in FEATURES if feature not in selected.columns]
    evaluation = data.loc[
        data["signed_year"].between(2020, 2023),
        ["capology_extension_event_id", *additional_features],
    ]
    merged = selected.merge(evaluation, on="capology_extension_event_id", validate="one_to_one")
    rows = []
    for feature in FEATURES:
        for state, chunk in merged.groupby(merged[feature].isna().map({True: "missing", False: "present"})):
            events = int(chunk["actual"].sum())
            nonevents = len(chunk) - events
            rows.append(
                {
                    "feature": feature,
                    "state": state,
                    "rows": len(chunk),
                    "events": events,
                    "nonevents": nonevents,
                    "actual_rate": float(chunk["actual"].mean()),
                    "predicted_rate": float(chunk["prediction"].mean()),
                    "calibration_gap": float(abs(chunk["actual"].mean() - chunk["prediction"].mean())),
                    "brier": float(chunk["brier_loss"].mean()),
                    "roc_auc": float(roc_auc_score(chunk["actual"], chunk["prediction"])) if events and nonevents else np.nan,
                    "evidence_status": "supported" if len(chunk) >= 75 and events >= 15 and nonevents >= 15 else "insufficient_evidence",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for stale in ("independent_verification.csv", "independent_verification.json"):
        (OUTPUT / stale).unlink(missing_ok=True)
    run2_summary = json.loads((RUN2 / "run_summary.json").read_text(encoding="utf-8"))
    run2_verify = json.loads((RUN2 / "independent_verification.json").read_text(encoding="utf-8"))
    if (
        not run2_verify.get("all_checks_passed")
        or run2_summary.get("selected_development_candidate") != "B2_core_recent_involvement"
    ):
        raise RuntimeError("Run 2 did not authorize this reliability audit.")

    deployment_before = deployment_fingerprint()
    data = prepare_model_frame().loc[lambda x: x["run2_model_cohort"]].copy()
    predictions, calibration_fit, coefficients, support_profiles, model_cache = fit_origin_models(data)

    run2_raw = pd.read_csv(RUN2 / "out_of_time_predictions.csv")
    run2_raw = run2_raw.loc[
        run2_raw["protocol"].eq("player_disjoint")
        & run2_raw["candidate"].eq("B2_core_recent_involvement")
    ].sort_values(["origin_id", "capology_extension_event_id"])
    current_raw = predictions.loc[predictions["calibration_method"].eq("raw")].sort_values(
        ["origin_id", "capology_extension_event_id"]
    )
    raw_reproduction_error = float(
        np.max(np.abs(run2_raw["prediction"].to_numpy() - current_raw["prediction"].to_numpy()))
    )

    origin_metrics, method_comparison, calibration_selection, selected_method = calibration_evidence(predictions)
    selected = predictions.loc[predictions["calibration_method"].eq(selected_method)].copy()
    intervals, parameter_summary = parameter_intervals(data, selected_method)
    selected = selected.merge(
        intervals,
        on=["origin_id", "capology_extension_event_id"],
        validate="one_to_one",
    )
    bins = calibration_bins(selected)
    subgroups = subgroup_reliability(selected)
    drift = temporal_drift(model_cache)
    redundancy = redundancy_stress(data, current_raw)
    monotonicity = monotonicity_stress(model_cache, selected_method)
    missingness = missingness_reliability(selected, data)

    support_summary = selected.groupby(["origin_id", "evaluation_year", "support_status"], as_index=False).size().rename(columns={"size": "rows"})
    support_summary["share"] = support_summary["rows"] / support_summary.groupby("origin_id")["rows"].transform("sum")
    reasons = (
        selected.assign(reason=selected["support_reasons"].fillna("").str.split(";"))
        .explode("reason")
        .loc[lambda x: x["reason"].ne("")]
        .groupby(["origin_id", "support_status", "reason"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
    )

    pooled = probability_metrics(selected["actual"], selected["prediction"])
    by_origin = origin_metrics.loc[origin_metrics["calibration_method"].eq(selected_method)].copy()
    chronology = selected.groupby("origin_id").apply(
        lambda x: float(x["baseline_brier_loss"].mean() - x["brier_loss"].mean()),
        include_groups=False,
    )
    reliable_origins = int(
        (
            by_origin["adaptive_ece"].le(MAXIMUM_ORIGIN_ECE)
            & by_origin["calibration_gap"].le(MAXIMUM_ORIGIN_CALIBRATION_GAP)
        ).sum()
    )
    positive_skill_origins = int(chronology.gt(0).sum())
    refusal_rate = float(selected["support_status"].eq("refused").mean())
    widths = selected["parameter_interval_width_80"]
    extreme_share = float(
        selected.loc[~selected["support_status"].eq("refused"), "prediction"]
        .pipe(lambda x: x.lt(0.02) | x.gt(0.98))
        .mean()
    )
    core_groups = subgroups.loc[subgroups["group_type"].isin(REQUIRED_SUBGROUP_TYPES)]
    core_supported = bool(core_groups["reliability_status"].eq("supported").all())
    involvement_coefficients = coefficients.loc[
        coefficients["encoded_feature"].isin(INVOLVEMENT_FEATURES)
    ]
    involvement_signs = bool(
        len(involvement_coefficients) == len(ORIGINS) * len(INVOLVEMENT_FEATURES)
        and involvement_coefficients["coefficient"].gt(0).all()
    )
    latest_psi = float(
        drift.loc[
            drift["origin_id"].eq("origin_2023")
            & drift["feature"].isin(INVOLVEMENT_FEATURES),
            "drift_value",
        ].max()
    )
    target_rate_range = float(by_origin["actual_rate"].max() - by_origin["actual_rate"].min())
    monotonic = bool(monotonicity["negative_delta_rows"].eq(0).all())
    parameter_complete = bool(
        selected["successful_parameter_refits"].ge(
            int(PARAMETER_BOOTSTRAP_REPETITIONS * 0.95)
        ).all()
    )
    gates = {
        "pooled_ece": pooled["adaptive_ece"] <= MAXIMUM_POOLED_ECE,
        "pooled_calibration_gap": pooled["calibration_gap"] <= MAXIMUM_POOLED_CALIBRATION_GAP,
        "pooled_calibration_slope": MINIMUM_CALIBRATION_SLOPE <= pooled["calibration_slope"] <= MAXIMUM_CALIBRATION_SLOPE,
        "reliable_origins": reliable_origins >= MINIMUM_RELIABLE_ORIGINS,
        "positive_skill_origins": positive_skill_origins >= MINIMUM_POSITIVE_SKILL_ORIGINS,
        "refusal_rate": refusal_rate <= MAXIMUM_REFUSAL_RATE,
        "mean_parameter_width": float(widths.mean()) <= MAXIMUM_MEAN_PARAMETER_WIDTH_80,
        "p90_parameter_width": float(widths.quantile(0.90)) <= MAXIMUM_P90_PARAMETER_WIDTH_80,
        "extreme_probability_share": extreme_share <= MAXIMUM_EXTREME_PROBABILITY_SHARE,
        "core_subgroups_supported": core_supported,
        "latest_involvement_drift": latest_psi <= MAXIMUM_LATEST_INVOLVEMENT_PSI,
        "target_rate_stability": target_rate_range <= MAXIMUM_TARGET_RATE_RANGE,
        "involvement_coefficients_positive": involvement_signs,
        "monotonicity_stress": monotonic,
        "parameter_bootstrap_complete": parameter_complete,
    }
    gate_passed = all(gates.values())
    correlation = float(
        data[list(INVOLVEMENT_FEATURES)].corr(method="spearman").iloc[0, 1]
    )
    explanation_policy = (
        "group_only_recent_involvement"
        if abs(correlation) > REDUNDANCY_SPEARMAN_THRESHOLD
        else "individual_feature_contributions_permitted"
    )
    decision = pd.DataFrame(
        [
            {
                "reliability_contract_version": CONTRACT_VERSION,
                "model_contract_version": MODEL_CONTRACT_VERSION,
                "target_contract_version": TARGET_CONTRACT_VERSION,
                "selected_calibration": selected_method,
                "pooled_brier": pooled["brier"],
                "pooled_log_loss": pooled["log_loss"],
                "pooled_roc_auc": pooled["roc_auc"],
                "pooled_average_precision": pooled["average_precision"],
                "pooled_ece": pooled["adaptive_ece"],
                "pooled_calibration_gap": pooled["calibration_gap"],
                "pooled_calibration_intercept": pooled["calibration_intercept"],
                "pooled_calibration_slope": pooled["calibration_slope"],
                "reliable_origins": reliable_origins,
                "positive_skill_origins": positive_skill_origins,
                "refused_profile_rate": refusal_rate,
                "mean_parameter_width_80": float(widths.mean()),
                "p90_parameter_width_80": float(widths.quantile(0.90)),
                "extreme_probability_share": extreme_share,
                "supported_core_subgroups": int(core_groups["reliability_status"].eq("supported").sum()),
                "total_core_subgroups": len(core_groups),
                "latest_involvement_max_psi": latest_psi,
                "target_rate_range": target_rate_range,
                "involvement_spearman": correlation,
                "explanation_policy": explanation_policy,
                **{f"gate__{key}": value for key, value in gates.items()},
                "run3_gate_passed": gate_passed,
                "run3_decision": (
                    "freeze_candidate_pending_later_untouched_temporal_validation"
                    if gate_passed
                    else "blocked_reliability_repair"
                ),
                "deployment_status": "not_deployed",
            }
        ]
    )

    deployment_after = deployment_fingerprint()
    deployment_manifest = pd.DataFrame(
        {
            "artifact": path,
            "before_sha256": digest,
            "after_sha256": deployment_after.get(path, "MISSING"),
            "unchanged": deployment_after.get(path) == digest,
        }
        for path, digest in deployment_before.items()
    )
    checks = []
    def add(name: str, actual: object, expected: object, passed: bool, note: str) -> None:
        checks.append({"check": name, "actual": actual, "expected": expected, "passed": bool(passed), "note": note})
    add("run2_raw_predictions_reproduced", raw_reproduction_error, "<=1e-12", raw_reproduction_error <= 1e-12, "Run 3 starts from the exact selected Run 2 candidate.")
    add("three_calibration_methods", sorted(predictions["calibration_method"].unique()), sorted(CALIBRATION_METHODS), set(predictions["calibration_method"]) == set(CALIBRATION_METHODS), "No post-result calibration method was added.")
    add("evaluation_rows_exact", len(selected), 878, len(selected) == 878, "Run 3 uses the repaired Run 2 development evaluations.")
    add("player_disjoint_all_origins", sum(len(set(cache["final_fit"]["canonical_player_id"]) & set(cache["evaluation"]["canonical_player_id"])) for cache in model_cache.values()), 0, all(not (set(cache["final_fit"]["canonical_player_id"]) & set(cache["evaluation"]["canonical_player_id"])) for cache in model_cache.values()), "Evaluation players never enter model fitting.")
    add("support_states_complete", sorted(selected["support_status"].unique()), "subset of supported/limited/refused", set(selected["support_status"]).issubset({"supported", "limited", "refused"}), "Every profile has an explicit support state.")
    add("refusals_explained", int(selected.loc[selected["support_status"].eq("refused"), "support_reasons"].fillna("").str.len().gt(0).all()), 1, selected.loc[selected["support_status"].eq("refused"), "support_reasons"].fillna("").str.len().gt(0).all(), "No probability refusal is unexplained.")
    add("parameter_intervals_ordered", int(selected["parameter_interval_low_80"].le(selected["parameter_interval_high_80"]).all()), 1, selected["parameter_interval_low_80"].le(selected["parameter_interval_high_80"]).all(), "Bootstrap bounds are ordered.")
    add("subgroup_intervals_ordered", int(subgroups["brier_skill_ci_low_99"].le(subgroups["brier_skill_ci_high_99"]).all()), 1, subgroups["brier_skill_ci_low_99"].le(subgroups["brier_skill_ci_high_99"]).all(), "Every subgroup has a player-cluster skill interval.")
    add("no_2024_outcomes", int(selected["evaluation_year"].max()), 2023, selected["evaluation_year"].max() == 2023, "The later holdout remains closed.")
    add("no_model_binary", len(list(OUTPUT.rglob("*.joblib"))), 0, not list(OUTPUT.rglob("*.joblib")), "Run 3 writes no deployable model.")
    add("deployment_unchanged", int(deployment_before == deployment_after), 1, deployment_before == deployment_after, "Current deployment files retain their pre-run hashes.")
    build_checks = pd.DataFrame(checks)

    write_csv(predictions, OUTPUT / "calibration_predictions.csv")
    write_csv(calibration_fit, OUTPUT / "calibration_fit_predictions.csv")
    write_csv(origin_metrics, OUTPUT / "origin_calibration_metrics.csv")
    write_csv(method_comparison, OUTPUT / "calibration_method_comparison.csv")
    write_csv(calibration_selection, OUTPUT / "calibration_selection.csv")
    write_csv(selected, OUTPUT / "selected_predictions.csv")
    write_csv(bins, OUTPUT / "calibration_bins.csv")
    write_csv(parameter_summary, OUTPUT / "parameter_uncertainty_summary.csv")
    write_json(support_profiles, OUTPUT / "ood_support_profiles.json")
    write_csv(support_summary, OUTPUT / "ood_support_summary.csv")
    write_csv(reasons, OUTPUT / "ood_reason_summary.csv")
    write_csv(subgroups, OUTPUT / "subgroup_reliability.csv")
    write_csv(drift, OUTPUT / "temporal_drift.csv")
    write_csv(coefficients, OUTPUT / "coefficient_stability.csv")
    write_csv(redundancy, OUTPUT / "redundancy_stress_test.csv")
    write_csv(monotonicity, OUTPUT / "monotonicity_stress_test.csv")
    write_csv(missingness, OUTPUT / "missingness_reliability.csv")
    write_csv(decision, OUTPUT / "reliability_decision.csv")
    write_csv(deployment_manifest, OUTPUT / "deployment_unchanged_manifest.csv")
    write_csv(build_checks, OUTPUT / "build_checks.csv")
    write_json(contract_payload(), OUTPUT / "preregistration.json")

    source_paths = [
        RUN2 / "run_summary.json",
        RUN2 / "selection_decision.csv",
        RUN2 / "out_of_time_predictions.csv",
        RUN2 / "independent_verification.json",
        ROOT / "models" / "engine_v2" / "contribution_reliability_contract.py",
        ROOT / "models" / "engine_v2" / "contribution_model_development_contract.py",
        ROOT / "models" / "engine_v2" / "complete_contribution_target_contract.py",
        ROOT / "models" / "engine_v2" / "feature_contract.py",
        ROOT / "models" / "mixed_type_preprocessor.py",
        ROOT / "models" / "run_engine_v2_contribution_model_rerun.py",
        ROOT / "models" / "run_engine_v2_contribution_reliability_audit.py",
        FREEZE,
    ]
    source_manifest = pd.DataFrame(
        {
            "source": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in source_paths
    )
    write_csv(source_manifest, OUTPUT / "source_manifest.csv")

    summary = {
        "stage": "engine_v2_repaired_contribution_reliability_run3",
        "reliability_contract_version": CONTRACT_VERSION,
        "selected_calibration": selected_method,
        "evaluation_rows": len(selected),
        "run3_decision": decision.iloc[0]["run3_decision"],
        "run3_gate_passed": bool(gate_passed),
        "explanation_policy": explanation_policy,
        "parameter_bootstrap_repetitions": PARAMETER_BOOTSTRAP_REPETITIONS,
        "subgroup_bootstrap_repetitions": SUBGROUP_BOOTSTRAP_REPETITIONS,
        "final_holdout_evaluated": False,
        "model_binaries_written": 0,
        "deployment_changed": False,
        "checks_passed": int(build_checks["passed"].sum()),
        "checks_total": len(build_checks),
        "all_checks_passed": bool(build_checks["passed"].all()),
    }
    write_json(summary, OUTPUT / "run_summary.json")
    readme = f"""# Engine V2 repaired contribution reliability — Run 3

Run 3 audits the fixed Run 2 compact candidate. It searches no new feature or
model family, opens no 2024+ outcome, writes no model binary, and changes no
deployment file.

## Result

- Selected calibration: `{selected_method}`
- Pooled Brier: {pooled['brier']:.6f}
- Pooled ECE: {pooled['adaptive_ece']:.6f}
- Calibration gap: {pooled['calibration_gap']:.6f}
- Calibration slope: {pooled['calibration_slope']:.6f}
- Reliable origins: {reliable_origins}/4
- Positive-skill origins: {positive_skill_origins}/4
- Refused profiles: {refusal_rate:.2%}
- Mean / P90 80% parameter-interval width: {widths.mean():.4f} / {widths.quantile(0.90):.4f}
- Supported broad league/position groups: {int(core_groups['reliability_status'].eq('supported').sum())}/{len(core_groups)}
- Run 3 decision: `{decision.iloc[0]['run3_decision']}`

The two recent-involvement inputs have pooled Spearman correlation
{correlation:.3f}. The explanation policy is therefore
`{explanation_policy}`: they may be described together as recent involvement,
not as independent causal effects or separately stable importance values.

The complete tables retain calibration alternatives, bins, parameter
uncertainty, OOD/refusal reasons, 99% player-cluster subgroup intervals,
temporal drift, coefficient stability, redundancy ablations, monotonicity
stress tests, missingness behavior, hashes, and the final gate.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")

    manifest_rows = []
    for path in sorted(OUTPUT.iterdir()):
        if path.is_file() and path.name not in {
            "output_manifest.csv",
            "independent_verification.csv",
            "independent_verification.json",
        }:
            manifest_rows.append(
                {"output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            )
    write_csv(pd.DataFrame(manifest_rows), OUTPUT / "output_manifest.csv")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nReliability decision")
    print(decision.to_string(index=False))
    if not build_checks["passed"].all():
        raise RuntimeError("Run 3 build checks failed.")


if __name__ == "__main__":
    main()

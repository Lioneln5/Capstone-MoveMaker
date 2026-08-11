"""Calibrate and logically reconcile the frozen market-value downside model.

The existing 11-feature M1 model remains frozen.  Calibration strategies are
fit on the prior validation season, selected on its later chronological 40%,
refit on the full validation season, and evaluated on the next season.  The
policy may select only strategies that guarantee nested probabilities:
P(50% decline) <= P(25% decline) <= P(10% decline).
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
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for dependency_dir in [ROOT / "models", ROOT / "scripts"]:
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

import run_market_value_downside_model as downside
from train_compatibility_models import sha256_file


MODEL_DIR = ROOT / "Data" / "processed" / "market_value_downside_model"
MODEL_VERIFICATION_PATH = MODEL_DIR / "independent_verification.json"
OUTPUT_DIR = ROOT / "Data" / "processed" / "market_value_calibration_experiment"
M1 = "M1_market_profile"
THRESHOLD_ENDPOINTS = ["downside_10pct", "downside_25pct", "downside_50pct"]
THRESHOLD_LOG_RATIOS = {
    "downside_10pct": math.log(0.90),
    "downside_25pct": math.log(0.75),
    "downside_50pct": math.log(0.50),
}
STRATEGIES = [
    "raw_independent", "raw_projected", "platt_independent", "platt_projected",
    "isotonic_independent", "isotonic_projected", "gaussian_distributional",
    "empirical_distributional",
]
POLICY_CANDIDATES = [
    "raw_projected", "platt_projected", "isotonic_projected",
    "gaussian_distributional", "empirical_distributional",
]
WINDOWS = downside.WINDOWS
BOOTSTRAP_REPETITIONS = 5000
MIN_SUBGROUP_ROWS = 75
MIN_SUBGROUP_ORIGINS = 2
PROTECTED_DIRS = [
    MODEL_DIR,
    ROOT / "Data" / "processed" / "financial_module_feasibility",
    ROOT / "Data" / "processed" / "retention_diagnostic",
    ROOT / "Data" / "processed" / "compatibility_targets",
]


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").fillna("").str.strip().str.lower().isin({"true", "1", "yes"})


def stable_seed(*parts: str) -> int:
    token = "|".join(parts).encode("utf-8")
    return (downside.RANDOM_SEED + int(hashlib.sha256(token).hexdigest()[:8], 16)) % (2**32 - 1)


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


def project_row_decreasing(values: np.ndarray) -> np.ndarray:
    blocks: list[dict[str, Any]] = []
    for index, value in enumerate(values.astype(float)):
        blocks.append({"mean": value, "weight": 1, "indices": [index]})
        while len(blocks) >= 2 and blocks[-2]["mean"] < blocks[-1]["mean"]:
            right = blocks.pop()
            left = blocks.pop()
            weight = left["weight"] + right["weight"]
            blocks.append({
                "mean": (left["mean"] * left["weight"] + right["mean"] * right["weight"]) / weight,
                "weight": weight,
                "indices": left["indices"] + right["indices"],
            })
    result = np.empty(len(values), dtype=float)
    for block in blocks:
        result[block["indices"]] = block["mean"]
    return np.clip(result, 1e-6, 1 - 1e-6)


def project_probabilities(probabilities: pd.DataFrame) -> pd.DataFrame:
    values = probabilities[THRESHOLD_ENDPOINTS].to_numpy(float)
    projected = np.vstack([project_row_decreasing(row) for row in values])
    return pd.DataFrame(projected, columns=THRESHOLD_ENDPOINTS, index=probabilities.index)


def fit_platt(probability: pd.Series, actual: pd.Series) -> LogisticRegression | None:
    if actual.nunique() < 2:
        return None
    x = np.log(np.clip(probability.to_numpy(float), 1e-6, 1 - 1e-6) / np.clip(1 - probability.to_numpy(float), 1e-6, 1))
    model = LogisticRegression(C=1e6, max_iter=3000, random_state=downside.RANDOM_SEED)
    model.fit(x.reshape(-1, 1), actual.to_numpy(int))
    return model


def apply_platt(model: LogisticRegression | None, probability: pd.Series) -> np.ndarray:
    p = np.clip(probability.to_numpy(float), 1e-6, 1 - 1e-6)
    if model is None:
        return p
    x = np.log(p / (1 - p))
    return np.clip(model.predict_proba(x.reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)


def fit_isotonic(probability: pd.Series, actual: pd.Series) -> IsotonicRegression | None:
    if actual.nunique() < 2 or probability.nunique() < 2:
        return None
    return IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(
        probability.to_numpy(float), actual.to_numpy(int)
    )


def apply_isotonic(model: IsotonicRegression | None, probability: pd.Series) -> np.ndarray:
    p = probability.to_numpy(float)
    return np.clip(model.predict(p) if model is not None else p, 1e-6, 1 - 1e-6)


def fit_strategy(strategy: str, calibration: pd.DataFrame) -> dict[str, Any]:
    fitted: dict[str, Any] = {"strategy": strategy}
    if strategy.startswith("platt"):
        fitted["models"] = {
            endpoint: fit_platt(calibration[f"raw_{endpoint}"], calibration[f"actual_{endpoint}"])
            for endpoint in THRESHOLD_ENDPOINTS
        }
    elif strategy.startswith("isotonic"):
        fitted["models"] = {
            endpoint: fit_isotonic(calibration[f"raw_{endpoint}"], calibration[f"actual_{endpoint}"])
            for endpoint in THRESHOLD_ENDPOINTS
        }
    elif strategy.endswith("distributional"):
        residual = calibration["actual_value_log_ratio_24m"] - calibration["raw_value_log_ratio_24m"]
        fitted["residual_mean"] = float(residual.mean())
        fitted["residual_std"] = max(float(residual.std(ddof=1)), 1e-6)
        fitted["residuals"] = np.sort(residual.to_numpy(float))
    return fitted


def apply_strategy(fitted: dict[str, Any], score: pd.DataFrame) -> pd.DataFrame:
    strategy = fitted["strategy"]
    if strategy.startswith("raw"):
        result = score[[f"raw_{endpoint}" for endpoint in THRESHOLD_ENDPOINTS]].copy()
        result.columns = THRESHOLD_ENDPOINTS
    elif strategy.startswith("platt"):
        result = pd.DataFrame({
            endpoint: apply_platt(fitted["models"][endpoint], score[f"raw_{endpoint}"])
            for endpoint in THRESHOLD_ENDPOINTS
        }, index=score.index)
    elif strategy.startswith("isotonic"):
        result = pd.DataFrame({
            endpoint: apply_isotonic(fitted["models"][endpoint], score[f"raw_{endpoint}"])
            for endpoint in THRESHOLD_ENDPOINTS
        }, index=score.index)
    else:
        predictions = score["raw_value_log_ratio_24m"].to_numpy(float)
        if strategy == "gaussian_distributional":
            mean = fitted["residual_mean"]
            scale = fitted["residual_std"]
            result = pd.DataFrame({
                endpoint: norm.cdf((threshold - predictions - mean) / scale)
                for endpoint, threshold in THRESHOLD_LOG_RATIOS.items()
            }, index=score.index)
        else:
            residuals = fitted["residuals"]
            result = pd.DataFrame({
                endpoint: (np.searchsorted(residuals, threshold - predictions, side="right") + 0.5) / (len(residuals) + 1)
                for endpoint, threshold in THRESHOLD_LOG_RATIOS.items()
            }, index=score.index)
    if strategy.endswith("projected") or strategy.endswith("distributional"):
        result = project_probabilities(result)
    return result.clip(1e-6, 1 - 1e-6)


def strategy_score(probability: pd.DataFrame, actual: pd.DataFrame) -> tuple[float, float]:
    briers = []
    losses = []
    for endpoint in THRESHOLD_ENDPOINTS:
        y = actual[f"actual_{endpoint}"].to_numpy(int)
        p = probability[endpoint].to_numpy(float)
        briers.append(brier_score_loss(y, p))
        losses.append(log_loss(y, p, labels=[0, 1]))
    return float(np.mean(briers)), float(np.mean(losses))


def wide_predictions(long: pd.DataFrame) -> pd.DataFrame:
    keys = ["origin_id", "transfer_event_id", "player_id", "transfer_date"]
    raw = long.pivot(index=keys, columns="endpoint", values="prediction").reset_index()
    actual = long.pivot(index=keys, columns="endpoint", values="actual").reset_index()
    raw = raw.rename(columns={endpoint: f"raw_{endpoint}" for endpoint in raw.columns if endpoint not in keys})
    actual = actual.rename(columns={endpoint: f"actual_{endpoint}" for endpoint in actual.columns if endpoint not in keys})
    return raw.merge(actual, on=keys, validate="one_to_one")


def reconstruct_validation_predictions(
    frame: pd.DataFrame, feature_sets: dict[str, list[str]], origin_results: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    features = feature_sets[M1]
    for endpoint, spec in downside.ENDPOINTS.items():
        target, kind = spec["target"], spec["kind"]
        for origin_id, train_end, validation_year, _ in downside.ORIGINS:
            train = frame.loc[frame["season_start_year"].le(train_end)]
            validation = frame.loc[frame["season_start_year"].eq(validation_year)]
            selected = origin_results.loc[
                origin_results["endpoint"].eq(endpoint)
                & origin_results["origin_id"].eq(origin_id)
                & origin_results["model_variant"].eq(M1)
            ].iloc[0]
            family = selected["selected_family"]
            parameters = json.loads(selected["selected_hyperparameters"])
            seed = downside.stable_seed(endpoint, origin_id, family, json.dumps(parameters, sort_keys=True))
            prediction, _ = downside.fit_candidate(
                train, validation, features, target, kind, family, parameters, seed
            )
            if kind == "classification":
                prediction = np.clip(prediction, 1e-6, 1 - 1e-6)
            for row, predicted in zip(validation.itertuples(index=False), prediction):
                rows.append({
                    "endpoint": endpoint, "origin_id": origin_id,
                    "transfer_event_id": row.transfer_event_id, "player_id": row.player_id,
                    "transfer_date": row.transfer_date, "actual": getattr(row, target),
                    "prediction": float(predicted),
                })
    return pd.DataFrame(rows)


def run_strategies(
    validation_long: pd.DataFrame, evaluation_long: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation = wide_predictions(validation_long)
    evaluation = wide_predictions(evaluation_long)
    selection_rows = []
    prediction_rows = []
    for origin_id in sorted(validation["origin_id"].unique()):
        val = validation.loc[validation["origin_id"].eq(origin_id)].sort_values(
            ["transfer_date", "transfer_event_id"]
        ).reset_index(drop=True)
        evaluate = evaluation.loc[evaluation["origin_id"].eq(origin_id)].copy()
        cut = int(math.floor(0.60 * len(val)))
        calibration = val.iloc[:cut]
        selector = val.iloc[cut:]
        candidates = []
        for strategy in STRATEGIES:
            fitted = fit_strategy(strategy, calibration)
            probability = apply_strategy(fitted, selector)
            brier, cross_entropy = strategy_score(probability, selector)
            eligible = strategy in POLICY_CANDIDATES
            selection_rows.append({
                "origin_id": origin_id, "strategy": strategy,
                "calibration_rows": len(calibration), "selector_rows": len(selector),
                "selector_mean_brier": brier, "selector_mean_log_loss": cross_entropy,
                "policy_eligible": eligible,
            })
            if eligible:
                candidates.append(((brier, cross_entropy, strategy), strategy))
        selected_strategy = min(candidates, key=lambda item: item[0])[1]
        for row in selection_rows:
            if row["origin_id"] == origin_id:
                row["selected_for_policy"] = row["strategy"] == selected_strategy

        fitted_all = {strategy: fit_strategy(strategy, val) for strategy in STRATEGIES}
        probabilities = {
            strategy: apply_strategy(fitted, evaluate)
            for strategy, fitted in fitted_all.items()
        }
        probabilities["policy_selected"] = probabilities[selected_strategy].copy()
        for strategy, probability in probabilities.items():
            for endpoint in THRESHOLD_ENDPOINTS:
                for index, row in evaluate.iterrows():
                    prediction_rows.append({
                        "strategy": strategy, "selected_origin_strategy": selected_strategy,
                        "endpoint": endpoint, "origin_id": origin_id,
                        "evaluation_year": downside.ORIGINS[[item[0] for item in downside.ORIGINS].index(origin_id)][3],
                        "transfer_event_id": row["transfer_event_id"], "player_id": row["player_id"],
                        "transfer_date": row["transfer_date"],
                        "actual": int(row[f"actual_{endpoint}"]),
                        "raw_probability": float(row[f"raw_{endpoint}"]),
                        "probability": float(probability.loc[index, endpoint]),
                    })
    return pd.DataFrame(prediction_rows), pd.DataFrame(selection_rows)


def expected_calibration_error(actual: pd.Series, probability: pd.Series, bins: int = 10) -> float:
    work = pd.DataFrame({"actual": actual.astype(float), "probability": probability.astype(float)})
    work["bin"] = pd.qcut(work["probability"].rank(method="first"), bins, labels=False, duplicates="drop")
    grouped = work.groupby("bin").agg(n=("actual", "size"), actual=("actual", "mean"), probability=("probability", "mean"))
    return float((grouped["n"] / len(work) * (grouped["actual"] - grouped["probability"]).abs()).sum())


def calibration_slope_intercept(actual: pd.Series, probability: pd.Series) -> tuple[float, float]:
    if actual.nunique() < 2:
        return np.nan, np.nan
    p = np.clip(probability.to_numpy(float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, max_iter=3000, random_state=downside.RANDOM_SEED)
    model.fit(logit, actual.to_numpy(int))
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def build_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy, endpoint, origin_id), group in predictions.groupby(
        ["strategy", "endpoint", "origin_id"], sort=True
    ):
        intercept, slope = calibration_slope_intercept(group["actual"], group["probability"])
        rows.append({
            "strategy": strategy, "endpoint": endpoint, "origin_id": origin_id,
            "evaluation_year": int(group["evaluation_year"].iloc[0]), "n": len(group),
            "actual_rate": group["actual"].mean(), "predicted_rate": group["probability"].mean(),
            "brier": brier_score_loss(group["actual"], group["probability"]),
            "log_loss": log_loss(group["actual"], group["probability"], labels=[0, 1]),
            "roc_auc": roc_auc_score(group["actual"], group["probability"]),
            "average_precision": average_precision_score(group["actual"], group["probability"]),
            "ece_10bin": expected_calibration_error(group["actual"], group["probability"]),
            "calibration_intercept": intercept, "calibration_slope": slope,
        })
    return pd.DataFrame(rows)


def cluster_bootstrap(pairs: pd.DataFrame, seed: int) -> dict[str, Any]:
    difference = np.square(pairs["actual"] - pairs["raw_probability"]) - np.square(
        pairs["actual"] - pairs["candidate_probability"]
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
        "cluster_count": len(clusters), "cluster_ci_lower_95": np.quantile(draws, 0.025),
        "cluster_ci_upper_95": np.quantile(draws, 0.975),
        "cluster_probability_candidate_better": np.mean(draws > 0),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS, "bootstrap_seed": seed,
    }


def build_comparisons(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baseline = predictions.loc[predictions["strategy"].eq("raw_independent")]
    for strategy in [item for item in list(STRATEGIES) + ["policy_selected"] if item != "raw_independent"]:
        candidate = predictions.loc[predictions["strategy"].eq(strategy)]
        keys = ["endpoint", "origin_id", "evaluation_year", "transfer_event_id", "player_id"]
        paired_all = baseline[keys + ["actual", "probability"]].merge(
            candidate[keys + ["probability"]], on=keys, suffixes=("_raw", "_candidate"),
            validate="one_to_one",
        ).rename(columns={"probability_raw": "raw_probability", "probability_candidate": "candidate_probability"})
        for endpoint in THRESHOLD_ENDPOINTS:
            for window, years in WINDOWS.items():
                pairs = paired_all.loc[
                    paired_all["endpoint"].eq(endpoint) & paired_all["evaluation_year"].isin(years)
                ].copy()
                raw_brier = brier_score_loss(pairs["actual"], pairs["raw_probability"])
                candidate_brier = brier_score_loss(pairs["actual"], pairs["candidate_probability"])
                raw_log = log_loss(pairs["actual"], pairs["raw_probability"], labels=[0, 1])
                candidate_log = log_loss(pairs["actual"], pairs["candidate_probability"], labels=[0, 1])
                raw_auc = np.average([
                    roc_auc_score(group["actual"], group["raw_probability"])
                    for _, group in pairs.groupby("origin_id")
                ], weights=[len(group) for _, group in pairs.groupby("origin_id")])
                candidate_auc = np.average([
                    roc_auc_score(group["actual"], group["candidate_probability"])
                    for _, group in pairs.groupby("origin_id")
                ], weights=[len(group) for _, group in pairs.groupby("origin_id")])
                origin_improvements = [
                    brier_score_loss(group["actual"], group["raw_probability"])
                    - brier_score_loss(group["actual"], group["candidate_probability"])
                    for _, group in pairs.groupby("origin_id")
                ]
                bootstrap = cluster_bootstrap(pairs, stable_seed(strategy, endpoint, window))
                rows.append({
                    "strategy": strategy, "endpoint": endpoint, "window": window,
                    "evaluation_years": ";".join(map(str, years)), "pooled_n": len(pairs),
                    "unique_players": pairs["player_id"].nunique(),
                    "raw_brier": raw_brier, "candidate_brier": candidate_brier,
                    "brier_improvement": raw_brier - candidate_brier,
                    "raw_log_loss": raw_log, "candidate_log_loss": candidate_log,
                    "log_loss_improvement": raw_log - candidate_log,
                    "raw_weighted_origin_auc": raw_auc,
                    "candidate_weighted_origin_auc": candidate_auc,
                    "auc_change": candidate_auc - raw_auc,
                    "origin_wins": sum(value > 0 for value in origin_improvements),
                    "origin_count": len(origin_improvements), **bootstrap,
                })
    return pd.DataFrame(rows)


def build_consistency(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy, origin_id), group in predictions.groupby(["strategy", "origin_id"]):
        wide = group.pivot(index="transfer_event_id", columns="endpoint", values="probability")
        v25 = wide["downside_25pct"].gt(wide["downside_10pct"] + 1e-12)
        v50 = wide["downside_50pct"].gt(wide["downside_25pct"] + 1e-12)
        rows.append({
            "strategy": strategy, "origin_id": origin_id, "rows": len(wide),
            "p25_gt_p10_rows": int(v25.sum()), "p50_gt_p25_rows": int(v50.sum()),
            "any_violation_rows": int((v25 | v50).sum()),
            "any_violation_rate": float((v25 | v50).mean()),
        })
    return pd.DataFrame(rows)


def build_risk_bands(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected = predictions.loc[
        predictions["strategy"].isin(["raw_independent", "policy_selected"])
    ].copy()
    selected["risk_band"] = selected.groupby(["strategy", "endpoint", "origin_id"])[
        "probability"
    ].transform(
        lambda values: pd.qcut(
            values.rank(method="first"), 5, labels=False, duplicates="drop"
        ) + 1
    )
    for (strategy, endpoint, band), group in selected.groupby(
        ["strategy", "endpoint", "risk_band"], sort=True
    ):
        rows.append(
            {
                "strategy": strategy,
                "endpoint": endpoint,
                "risk_band": int(band),
                "risk_band_label": (
                    "lowest_predicted_risk"
                    if band == 1
                    else ("highest_predicted_risk" if band == 5 else f"band_{int(band)}")
                ),
                "rows": len(group),
                "unique_players": group["player_id"].nunique(),
                "mean_probability": group["probability"].mean(),
                "observed_rate": group["actual"].mean(),
            }
        )
    return pd.DataFrame(rows)


def add_subgroups(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[["transfer_event_id", "age_at_transfer", "historical_position_group", "destination_competition_id", "valuation_pre_eur", "canonical_transfer_fee_status", "season_start_year"]].copy()
    result["age_band"] = pd.cut(result["age_at_transfer"], [-np.inf, 20, 24, 28, np.inf], labels=["20_or_younger", "21_to_24", "25_to_28", "29_or_older"])
    result["pre_value_band"] = pd.cut(pd.to_numeric(result["valuation_pre_eur"], errors="coerce"), [-np.inf, 1e6, 5e6, 15e6, np.inf], labels=["under_1m", "1m_to_5m", "5m_to_15m", "15m_plus"], right=False)
    result["fee_evidence"] = np.where(result["canonical_transfer_fee_status"].eq("positive_reported"), "positive_reported", "zero_or_unclassified")
    return result


def build_subgroup_stability(predictions: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    selected = predictions.loc[
        predictions["endpoint"].eq("downside_25pct")
        & predictions["strategy"].isin(["raw_independent", "policy_selected"])
    ]
    keys = ["origin_id", "evaluation_year", "transfer_event_id", "player_id"]
    raw = selected.loc[selected["strategy"].eq("raw_independent"), keys + ["actual", "probability"]].rename(columns={"probability": "raw_probability"})
    policy = selected.loc[selected["strategy"].eq("policy_selected"), keys + ["probability"]].rename(columns={"probability": "candidate_probability"})
    paired = raw.merge(policy, on=keys, validate="one_to_one").merge(add_subgroups(frame), on="transfer_event_id", validate="many_to_one")
    dimensions = ["historical_position_group", "age_band", "destination_competition_id", "pre_value_band", "fee_evidence", "evaluation_year"]
    rows = []
    for dimension in dimensions:
        for category, group in paired.groupby(dimension, dropna=False, observed=True):
            adequate = len(group) >= MIN_SUBGROUP_ROWS and group["origin_id"].nunique() >= MIN_SUBGROUP_ORIGINS
            raw_brier = brier_score_loss(group["actual"], group["raw_probability"])
            policy_brier = brier_score_loss(group["actual"], group["candidate_probability"])
            row = {
                "dimension": dimension, "category": str(category), "rows": len(group),
                "unique_players": group["player_id"].nunique(), "origin_count": group["origin_id"].nunique(),
                "adequate_for_inference": adequate, "actual_rate": group["actual"].mean(),
                "raw_brier": raw_brier, "policy_brier": policy_brier,
                "brier_improvement": raw_brier - policy_brier,
            }
            if adequate:
                bootstrap = cluster_bootstrap(group, stable_seed("subgroup", dimension, str(category)))
                row.update(bootstrap)
                row["stability_label"] = "supported_positive" if bootstrap["cluster_ci_lower_95"] > 0 else ("supported_negative" if bootstrap["cluster_ci_upper_95"] < 0 else "uncertain")
            else:
                row.update({"cluster_count": np.nan, "cluster_ci_lower_95": np.nan, "cluster_ci_upper_95": np.nan, "cluster_probability_candidate_better": np.nan, "bootstrap_repetitions": np.nan, "bootstrap_seed": np.nan, "stability_label": "descriptive_only"})
            rows.append(row)
    return pd.DataFrame(rows)


def build_decision(comparisons: pd.DataFrame, consistency: pd.DataFrame, metrics: pd.DataFrame, subgroups: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pooled = comparisons.loc[comparisons["strategy"].eq("policy_selected") & comparisons["window"].eq("pooled_four")].set_index("endpoint")
    for endpoint in THRESHOLD_ENDPOINTS:
        result = pooled.loc[endpoint]
        policy_violations = int(consistency.loc[consistency["strategy"].eq("policy_selected"), "any_violation_rows"].sum())
        raw_metric = metrics.loc[metrics["strategy"].eq("raw_independent") & metrics["endpoint"].eq(endpoint)]
        policy_metric = metrics.loc[metrics["strategy"].eq("policy_selected") & metrics["endpoint"].eq(endpoint)]
        raw_ece = np.average(raw_metric["ece_10bin"], weights=raw_metric["n"])
        policy_ece = np.average(policy_metric["ece_10bin"], weights=policy_metric["n"])
        supported_negative = int(subgroups.loc[subgroups["adequate_for_inference"] & subgroups["stability_label"].eq("supported_negative")].shape[0]) if endpoint == "downside_25pct" else np.nan
        replace = (
            result["cluster_ci_lower_95"] > 0
            and result["origin_wins"] >= 3
            and result["auc_change"] >= -0.005
            and policy_violations == 0
            and policy_ece < raw_ece
            and (endpoint != "downside_25pct" or supported_negative == 0)
        )
        rows.append({
            "endpoint": endpoint, "policy_brier_improvement": result["brier_improvement"],
            "policy_ci_lower_95": result["cluster_ci_lower_95"], "policy_ci_upper_95": result["cluster_ci_upper_95"],
            "policy_origin_wins": int(result["origin_wins"]), "policy_auc_change": result["auc_change"],
            "raw_weighted_ece": raw_ece, "policy_weighted_ece": policy_ece,
            "policy_monotonic_violations": policy_violations,
            "adequate_supported_negative_subgroups": supported_negative,
            "replace_raw_probability_model": bool(replace),
            "recommendation": "replace_with_selected_calibrated_policy" if replace else "retain_raw_model_for_ranking_and_use_monotonic_risk_bands",
        })
    return pd.DataFrame(rows)


def build_checks(validation: pd.DataFrame, predictions: pd.DataFrame, selection: pd.DataFrame, consistency: pd.DataFrame) -> pd.DataFrame:
    checks = []
    def add(check: str, passed: bool, observed: Any, expected: Any, note: str) -> None:
        checks.append({"check": check, "passed": bool(passed), "observed": observed, "expected": expected, "note": note})
    key = ["strategy", "endpoint", "origin_id", "transfer_event_id"]
    add("validation_rows_present", len(validation) > 0, len(validation), ">0", "Frozen base model reconstructed validation forecasts.")
    add("prediction_key_unique", not predictions.duplicated(key).any(), int(predictions.duplicated(key).sum()), 0, "One forecast per strategy/endpoint/origin/event.")
    add("probability_bounds", predictions["probability"].between(0, 1).all(), f"{predictions['probability'].min()}..{predictions['probability'].max()}", "0..1", "All forecasts are probabilities.")
    add("strategies_complete", set(predictions["strategy"]) == set(STRATEGIES + ["policy_selected"]), ";".join(sorted(set(predictions["strategy"]))), ";".join(sorted(STRATEGIES + ["policy_selected"])), "All predeclared strategies are evaluated.")
    selected = selection.loc[as_bool(selection["selected_for_policy"])]
    add("one_strategy_per_origin", selected.groupby("origin_id").size().eq(1).all() and len(selected) == 4, selected.groupby("origin_id").size().to_dict(), "one each", "Policy selection is unique.")
    add("selected_strategies_consistent", set(selected["strategy"]).issubset(POLICY_CANDIDATES), ";".join(sorted(set(selected["strategy"]))), ";".join(POLICY_CANDIDATES), "Policy candidates guarantee nesting.")
    policy_violations = int(consistency.loc[consistency["strategy"].eq("policy_selected"), "any_violation_rows"].sum())
    add("policy_zero_monotonic_violations", policy_violations == 0, policy_violations, 0, "Selected probabilities obey nested thresholds.")
    alignment = predictions.groupby(["strategy", "endpoint", "origin_id"]).size().groupby(level=[1, 2]).nunique().eq(1).all()
    add("paired_evaluation_rows", alignment, alignment, True, "All strategies score identical rows.")
    return pd.DataFrame(checks)


def write_readme(decision: pd.DataFrame, selection: pd.DataFrame, consistency: pd.DataFrame) -> None:
    lines = [
        "# Market-value downside calibration and ordinal-consistency experiment", "",
        "The frozen 11-feature model is unchanged. Calibration is learned only from the prior validation season and evaluated on the next season.", "",
        "## Selected strategies", "",
    ]
    for row in selection.loc[as_bool(selection["selected_for_policy"])].itertuples(index=False):
        lines.append(f"- `{row.origin_id}`: `{row.strategy}` (selector mean Brier {row.selector_mean_brier:.6f}).")
    raw_violations = int(consistency.loc[consistency["strategy"].eq("raw_independent"), "any_violation_rows"].sum())
    policy_violations = int(consistency.loc[consistency["strategy"].eq("policy_selected"), "any_violation_rows"].sum())
    lines += ["", "## Decision", "", f"Raw independent classifiers contain {raw_violations} threshold-order violations; the selected policy contains {policy_violations}.", ""]
    for row in decision.itertuples(index=False):
        lines.append(
            f"- `{row.endpoint}`: Brier improvement {row.policy_brier_improvement:.6f}, 95% CI [{row.policy_ci_lower_95:.6f}, {row.policy_ci_upper_95:.6f}], {row.policy_origin_wins}/4 origin wins, AUC change {row.policy_auc_change:.6f}, ECE {row.raw_weighted_ece:.6f} -> {row.policy_weighted_ece:.6f}; `{row.recommendation}`."
        )
    lines += ["", "Calibration can improve probability accuracy but cannot create new ranking information. If the replacement gate fails, retain the frozen model for ranking and communicate ordered risk bands rather than exact probabilities.", ""]
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_VERIFICATION_PATH.open("r", encoding="utf-8") as handle:
        verification = json.load(handle)
    if verification.get("status") != "pass":
        raise RuntimeError("Frozen downside model verification is not passing.")
    protected_before = protected_tree_sha256()
    frame, feature_sets, _ = downside.build_model_frame()
    origin_results = pd.read_csv(MODEL_DIR / "market_value_origin_results.csv", low_memory=False)
    existing = pd.read_csv(MODEL_DIR / "market_value_predictions.csv", low_memory=False)
    evaluation_long = existing.loc[
        existing["model_variant"].eq(M1) & existing["endpoint"].isin(list(downside.ENDPOINTS))
    ].copy()
    validation_long = reconstruct_validation_predictions(frame, feature_sets, origin_results)
    predictions, selection = run_strategies(validation_long, evaluation_long)
    metrics = build_metrics(predictions)
    comparisons = build_comparisons(predictions)
    consistency = build_consistency(predictions)
    risk_bands = build_risk_bands(predictions)
    subgroups = build_subgroup_stability(predictions, frame)
    decision = build_decision(comparisons, consistency, metrics, subgroups)
    checks = build_checks(validation_long, predictions, selection, consistency)
    if not checks["passed"].all():
        raise RuntimeError(f"Calibration checks failed: {checks.loc[~checks['passed'], 'check'].tolist()}")

    tables = {
        "calibration_validation_predictions.csv": validation_long,
        "calibration_strategy_selection.csv": selection,
        "calibrated_predictions.csv": predictions,
        "calibration_origin_metrics.csv": metrics,
        "calibration_comparison_results.csv": comparisons,
        "probability_consistency_audit.csv": consistency,
        "calibrated_risk_bands.csv": risk_bands,
        "calibration_subgroup_stability.csv": subgroups,
        "calibration_decision_summary.csv": decision,
        "build_checks.csv": checks,
    }
    for name, table in tables.items():
        table.to_csv(OUTPUT_DIR / name, index=False)
    source_paths = [
        MODEL_DIR / "market_value_predictions.csv", MODEL_DIR / "market_value_origin_results.csv",
        MODEL_DIR / "market_value_downside_targets.csv", MODEL_VERIFICATION_PATH,
        downside.MASTER_PATH, downside.AUDIT_PATH,
    ]
    sources = pd.DataFrame([
        {"source": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in source_paths
    ])
    sources.to_csv(OUTPUT_DIR / "source_manifest.csv", index=False)
    tables["source_manifest.csv"] = sources
    write_readme(decision, selection, consistency)
    if protected_before != protected_tree_sha256():
        raise RuntimeError("A protected verified output changed during calibration.")
    run_summary = {
        "evaluation_rows": int(predictions.loc[predictions["strategy"].eq("policy_selected"), "transfer_event_id"].nunique()),
        "origins": sorted(predictions["origin_id"].unique().tolist()),
        "strategies": STRATEGIES + ["policy_selected"],
        "selected_strategies": selection.loc[as_bool(selection["selected_for_policy"])].set_index("origin_id")["strategy"].to_dict(),
        "raw_monotonic_violations": int(consistency.loc[consistency["strategy"].eq("raw_independent"), "any_violation_rows"].sum()),
        "policy_monotonic_violations": int(consistency.loc[consistency["strategy"].eq("policy_selected"), "any_violation_rows"].sum()),
        "replacement_decisions": decision.set_index("endpoint")["replace_raw_probability_model"].astype(bool).to_dict(),
        "protected_outputs_unchanged": True, "build_checks": len(checks), "build_checks_passed": int(checks["passed"].sum()),
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

"""Rebuild public-value downside development evidence without exposed 2024+ data.

This run evaluates one predeclared, proposal-free compact candidate at 12 and
24 months.  It uses player-disjoint rolling origins ending in 2023, calibrates
only on preceding-year validation predictions, records profile support and
subgroup uncertainty, and writes no model binary.
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

from models.engine_v2.public_value_downside_repair_contract import (  # noqa: E402
    BOOTSTRAP_REPETITIONS,
    CALIBRATION_BRIER_NONINFERIORITY,
    CALIBRATION_ECE_NEAR_TIE,
    CALIBRATION_LOG_LOSS_NONINFERIORITY,
    CALIBRATION_METHODS,
    CALIBRATION_SIMPLICITY,
    CANDIDATE_FEATURES,
    CONTRACT_VERSION,
    ENDPOINTS,
    LOGISTIC_C_GRID,
    MAXIMUM_ABSOLUTE_CALIBRATION_GAP,
    MAXIMUM_REFUSAL_RATE,
    MAXIMUM_SUBGROUP_CALIBRATION_GAP,
    MINIMUM_ORIGIN_WINS,
    MINIMUM_POOLED_AUC,
    MINIMUM_SUBGROUP_EVENTS,
    MINIMUM_SUBGROUP_NONEVENTS,
    MINIMUM_SUBGROUP_ORIGINS,
    MINIMUM_SUBGROUP_ROWS,
    MINIMUM_SUBGROUP_AUC,
    OOD_LIMITED_PERCENTILE,
    OOD_REFUSAL_PERCENTILE,
    ORIGINS,
    PARAMETER_BOOTSTRAP_REPETITIONS,
    PARAMETER_INTERVAL_ALPHA,
    PRIMARY_CANDIDATE,
    RANDOM_SEED,
    SUBGROUP_INTERVAL_ALPHA,
    contract_payload,
)
from models.engine_v2.feature_contract import (  # noqa: E402
    ADVANCED_PERFORMANCE,
    COMMERCIAL_SCENARIO,
    RAW_SCORING_RATES,
)
from mixed_type_preprocessor import fit_preprocessor  # noqa: E402
import run_extension_value_preservation_diagnostic as value_source  # noqa: E402


OUTPUT = ROOT / "Data" / "processed" / "engine_v2_public_value_downside_repair"
DEPLOYMENT = ROOT / "Data" / "processed" / "deployment_models"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"
FEATURES = list(CANDIDATE_FEATURES[PRIMARY_CANDIDATE])
HARD_REQUIRED_FEATURES = tuple(FEATURES)


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
    mutable_verification_reports = {
        "Data/processed/deployment_models/live_request_integrity.csv",
        "Data/processed/deployment_models/live_request_integrity.json",
    }
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in sorted(DEPLOYMENT.rglob("*"))
        if path.is_file() and path.relative_to(ROOT).as_posix() not in mutable_verification_reports
    }


def prepare_frame(endpoint: str, *, include_exposed: bool = False) -> pd.DataFrame:
    frame = value_source.endpoint_frame(value_source.load_frame(), endpoint).copy()
    frame["signed_year"] = pd.to_numeric(frame["signed_year"], errors="coerce").astype("Int64")
    frame["age_band"] = pd.cut(
        pd.to_numeric(frame["age_at_signing"], errors="coerce"),
        [-np.inf, 21, 25, 29, 33, np.inf],
        labels=["<=21", "22-25", "26-29", "30-33", "34+"],
    )
    frame = frame.loc[
        frame["canonical_player_id"].notna()
        & frame["capology_extension_event_id"].notna()
        & frame[list(HARD_REQUIRED_FEATURES)].notna().all(axis=1)
    ].copy()
    if not include_exposed:
        frame = frame.loc[frame["signed_year"].le(2023)].copy()
    return frame.sort_values(["signed_year", "capology_extension_event_id"])


def split_origin(frame: pd.DataFrame, spec: tuple[str, int, int, int]):
    _, train_end, validation_year, evaluation_year = spec
    evaluation = frame.loc[frame["signed_year"].eq(evaluation_year)].sort_values(
        "capology_extension_event_id"
    ).copy()
    evaluation_players = set(evaluation["canonical_player_id"].dropna())
    base_train = frame.loc[
        frame["signed_year"].le(train_end)
        & ~frame["canonical_player_id"].isin(evaluation_players)
    ].copy()
    validation = frame.loc[
        frame["signed_year"].eq(validation_year)
        & ~frame["canonical_player_id"].isin(evaluation_players)
    ].copy()
    validation_players = set(validation["canonical_player_id"].dropna())
    tuning = base_train.loc[~base_train["canonical_player_id"].isin(validation_players)].copy()
    final_fit = pd.concat([base_train, validation], ignore_index=True)
    if min(len(tuning), len(validation), len(final_fit), len(evaluation)) < 30:
        raise RuntimeError(f"Insufficient split for {spec[0]}")
    return tuning, validation, final_fit, evaluation


def logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def adaptive_ece(actual: np.ndarray, prediction: np.ndarray, bins: int = 10) -> float:
    ordered = pd.DataFrame({"actual": actual, "prediction": prediction}).sort_values("prediction")
    groups = np.array_split(np.arange(len(ordered)), min(bins, len(ordered)))
    return float(sum(
        len(indices) * abs(ordered.iloc[indices]["actual"].mean() - ordered.iloc[indices]["prediction"].mean())
        for indices in groups
    ) / len(ordered))


def probability_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, int)
    prediction = np.clip(np.asarray(prediction, float), 1e-6, 1 - 1e-6)
    calibration = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000).fit(
        logit(prediction).reshape(-1, 1), actual
    )
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


def tune_c(tuning: pd.DataFrame, validation: pd.DataFrame, features: list[str], target: str):
    pre = fit_preprocessor(tuning, features)
    x_train, x_validation = pre.transform(tuning), pre.transform(validation)
    y_train = pd.to_numeric(tuning[target]).to_numpy(int)
    y_validation = pd.to_numeric(validation[target]).to_numpy(int)
    trials = []
    for c_value in LOGISTIC_C_GRID:
        model = LogisticRegression(
            C=c_value, penalty="l2", solver="liblinear", max_iter=3000,
            random_state=RANDOM_SEED,
        ).fit(x_train, y_train)
        prediction = np.clip(model.predict_proba(x_validation)[:, 1], 1e-6, 1 - 1e-6)
        trials.append((
            float(brier_score_loss(y_validation, prediction)),
            float(log_loss(y_validation, prediction, labels=[0, 1])),
            c_value,
            prediction,
        ))
    return min(trials, key=lambda item: (item[0], item[1], item[2]))


def fit_calibrator(method: str, probability: np.ndarray, actual: np.ndarray):
    if method == "raw":
        return None
    if method == "platt":
        return LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000).fit(
            logit(probability).reshape(-1, 1), actual
        )
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(
            probability, actual
        )
    raise KeyError(method)


def apply_calibrator(method: str, calibrator, probability: np.ndarray) -> np.ndarray:
    raw = np.clip(np.asarray(probability, float), 1e-6, 1 - 1e-6)
    if method == "raw":
        return raw
    if method == "platt":
        return np.clip(calibrator.predict_proba(logit(raw).reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)
    return np.clip(calibrator.predict(raw), 1e-6, 1 - 1e-6)


def build_support_profile(reference: pd.DataFrame) -> dict[str, object]:
    profile: dict[str, object] = {"categorical": {}, "numeric": {}, "rows": len(reference)}
    distances = pd.DataFrame(index=reference.index)
    for feature in FEATURES:
        if feature in {"canonical_position", "league"}:
            profile["categorical"][feature] = sorted(reference[feature].astype(str).unique())
            continue
        values = pd.to_numeric(reference[feature], errors="coerce")
        median = float(values.median())
        q1, q3 = float(values.quantile(.25)), float(values.quantile(.75))
        scale = max((q3 - q1) / 1.349, 1e-8)
        profile["numeric"][feature] = {
            "median": median, "scale": scale, "minimum": float(values.min()),
            "maximum": float(values.max()), "q01": float(values.quantile(.01)),
            "q99": float(values.quantile(.99)),
        }
        distances[feature] = (values - median).abs() / scale
    maximum = distances.max(axis=1)
    profile["limited_distance"] = float(maximum.quantile(OOD_LIMITED_PERCENTILE))
    profile["refusal_distance"] = float(maximum.quantile(OOD_REFUSAL_PERCENTILE))
    return profile


def assess_support(frame: pd.DataFrame, profile: dict[str, object]) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        refused, limited, distances = [], [], []
        missing = [feature for feature in HARD_REQUIRED_FEATURES if pd.isna(row[feature])]
        if missing:
            refused.append("MISSING_REQUIRED_FEATURE:" + "|".join(missing))
        for feature in ("canonical_position", "league"):
            if pd.notna(row[feature]) and str(row[feature]) not in profile["categorical"][feature]:
                refused.append("UNSEEN_CATEGORY:" + feature)
        if not missing:
            for feature, spec in profile["numeric"].items():
                current = float(row[feature])
                distances.append(abs(current - spec["median"]) / spec["scale"])
                if current < spec["minimum"] or current > spec["maximum"]:
                    refused.append("OUTSIDE_OBSERVED_RANGE:" + feature)
                elif current < spec["q01"] or current > spec["q99"]:
                    limited.append("OUTSIDE_TRAINING_P01_P99:" + feature)
        distance = max(distances) if distances else np.nan
        if pd.notna(distance) and distance > profile["refusal_distance"]:
            refused.append("ROBUST_DISTANCE_ABOVE_TRAINING_P99")
        elif pd.notna(distance) and distance > profile["limited_distance"]:
            limited.append("ROBUST_DISTANCE_ABOVE_TRAINING_P95")
        rows.append({
            "support_status": "refused" if refused else "limited" if limited else "supported",
            "support_reasons": ";".join(refused + limited),
            "support_distance": distance,
        })
    return pd.DataFrame(rows, index=frame.index)


def fit_endpoint(endpoint: str):
    target = ENDPOINTS[endpoint]["target"]
    frame = prepare_frame(endpoint)
    candidate_rows, calibrated_rows, coefficient_rows, cohort_rows = [], [], [], []
    caches = {}
    for spec in ORIGINS:
        origin, train_end, validation_year, evaluation_year = spec
        tuning, validation, final_fit, evaluation = split_origin(frame, spec)
        cohort_rows.append({
            "endpoint": endpoint, "origin_id": origin, "tuning_rows": len(tuning),
            "validation_rows": len(validation), "final_fit_rows": len(final_fit),
            "evaluation_rows": len(evaluation), "evaluation_players": evaluation["canonical_player_id"].nunique(),
            "identity_overlap": len(set(final_fit["canonical_player_id"]) & set(evaluation["canonical_player_id"])),
            "tuning_identity_overlap": len(set(tuning["canonical_player_id"]) & set(evaluation["canonical_player_id"])),
            "calibration_identity_overlap": len(set(validation["canonical_player_id"]) & set(evaluation["canonical_player_id"])),
            "evaluation_year": evaluation_year,
        })
        evaluation_actual = pd.to_numeric(evaluation[target]).to_numpy(int)
        primary_validation_raw = None
        selected_c = None
        primary_pre = primary_model = None
        for candidate, features in CANDIDATE_FEATURES.items():
            if not features:
                estimate = np.repeat(float(pd.to_numeric(final_fit[target]).mean()), len(evaluation))
                current_c = np.nan
            else:
                _, _, current_c, validation_raw = tune_c(tuning, validation, features, target)
                pre = fit_preprocessor(final_fit, features)
                model = LogisticRegression(
                    C=current_c, penalty="l2", solver="liblinear", max_iter=3000,
                    random_state=RANDOM_SEED,
                ).fit(pre.transform(final_fit), pd.to_numeric(final_fit[target]).to_numpy(int))
                estimate = np.clip(model.predict_proba(pre.transform(evaluation))[:, 1], 1e-6, 1 - 1e-6)
                if candidate == PRIMARY_CANDIDATE:
                    primary_validation_raw = validation_raw
                    selected_c, primary_pre, primary_model = current_c, pre, model
            for index, (_, row) in enumerate(evaluation.iterrows()):
                candidate_rows.append({
                    "endpoint": endpoint, "origin_id": origin, "evaluation_year": evaluation_year,
                    "capology_extension_event_id": row["capology_extension_event_id"],
                    "canonical_player_id": row["canonical_player_id"], "league": row["league"],
                    "canonical_position": row["canonical_position"], "age_band": row["age_band"],
                    "candidate": candidate, "selected_C": current_c, "actual": int(evaluation_actual[index]),
                    "prediction": float(estimate[index]), "brier_loss": float((evaluation_actual[index] - estimate[index]) ** 2),
                })
        validation_actual = pd.to_numeric(validation[target]).to_numpy(int)
        calibrators = {method: fit_calibrator(method, primary_validation_raw, validation_actual) for method in CALIBRATION_METHODS}
        raw = np.clip(primary_model.predict_proba(primary_pre.transform(evaluation))[:, 1], 1e-6, 1 - 1e-6)
        profile = build_support_profile(final_fit)
        support = assess_support(evaluation, profile)
        baseline = float(pd.to_numeric(final_fit[target]).mean())
        for method in CALIBRATION_METHODS:
            estimate = apply_calibrator(method, calibrators[method], raw)
            for index, (_, row) in enumerate(evaluation.iterrows()):
                calibrated_rows.append({
                    "endpoint": endpoint, "origin_id": origin, "evaluation_year": evaluation_year,
                    "calibration_method": method, "capology_extension_event_id": row["capology_extension_event_id"],
                    "canonical_player_id": row["canonical_player_id"], "league": row["league"],
                    "canonical_position": row["canonical_position"], "age_band": row["age_band"],
                    "actual": int(evaluation_actual[index]), "prediction": float(estimate[index]),
                    "raw_prediction": float(raw[index]), "baseline_prediction": baseline,
                    "brier_loss": float((evaluation_actual[index] - estimate[index]) ** 2),
                    "baseline_brier_loss": float((evaluation_actual[index] - baseline) ** 2),
                    **support.iloc[index].to_dict(),
                })
        for encoded, raw_feature, coefficient in zip(primary_pre.encoded_names, primary_pre.encoded_raw_features, primary_model.coef_.reshape(-1)):
            coefficient_rows.append({
                "endpoint": endpoint, "origin_id": origin, "encoded_feature": encoded,
                "raw_feature": raw_feature, "coefficient": float(coefficient),
            })
        caches[origin] = {
            "spec": spec, "tuning": tuning, "validation": validation, "final_fit": final_fit,
            "evaluation": evaluation, "selected_C": selected_c, "calibrators": calibrators,
            "preprocessor": primary_pre, "model": primary_model, "profile": profile,
        }
    return frame, pd.DataFrame(candidate_rows), pd.DataFrame(calibrated_rows), pd.DataFrame(coefficient_rows), pd.DataFrame(cohort_rows), caches


def candidate_evidence(predictions: pd.DataFrame):
    metric_rows = []
    for (endpoint, origin, candidate), chunk in predictions.groupby(["endpoint", "origin_id", "candidate"]):
        metric_rows.append({"endpoint": endpoint, "origin_id": origin, "candidate": candidate, "rows": len(chunk), **probability_metrics(chunk.actual, chunk.prediction)})
    metrics = pd.DataFrame(metric_rows)
    comparisons = []
    for endpoint, endpoint_rows in predictions.groupby("endpoint"):
        pivot = endpoint_rows.pivot_table(
            index=["origin_id", "capology_extension_event_id", "canonical_player_id"],
            columns="candidate", values="brier_loss",
        ).reset_index()
        for baseline in ("B0_chronology_prevalence", "B1_market_profile"):
            pivot["improvement"] = pivot[baseline] - pivot[PRIMARY_CANDIDATE]
            grouped = pivot.groupby("canonical_player_id")["improvement"].agg(["sum", "size"])
            rng = np.random.default_rng(stable_seed(endpoint, baseline, "candidate_bootstrap"))
            sums, sizes, draws = grouped["sum"].to_numpy(float), grouped["size"].to_numpy(float), []
            for start in range(0, BOOTSTRAP_REPETITIONS, 250):
                count = min(250, BOOTSTRAP_REPETITIONS - start)
                indices = rng.integers(0, len(grouped), size=(count, len(grouped)))
                draws.extend((sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)).tolist())
            origin_wins = sum(
                chunk[baseline].mean() > chunk[PRIMARY_CANDIDATE].mean()
                for _, chunk in pivot.groupby("origin_id")
            )
            comparisons.append({
                "endpoint": endpoint, "candidate": PRIMARY_CANDIDATE, "baseline": baseline,
                "rows": len(pivot), "player_clusters": len(grouped),
                "brier_improvement": float(pivot.improvement.mean()),
                "brier_improvement_ci_low_95": float(np.quantile(draws, .025)),
                "brier_improvement_ci_high_95": float(np.quantile(draws, .975)),
                "origin_wins": int(origin_wins),
            })
    return metrics, pd.DataFrame(comparisons)


def calibration_evidence(predictions: pd.DataFrame):
    rows, decisions, selected = [], [], {}
    for endpoint, endpoint_rows in predictions.groupby("endpoint"):
        raw_metrics = probability_metrics(
            endpoint_rows.loc[endpoint_rows.calibration_method.eq("raw"), "actual"],
            endpoint_rows.loc[endpoint_rows.calibration_method.eq("raw"), "prediction"],
        )
        options = []
        for method, chunk in endpoint_rows.groupby("calibration_method"):
            current = probability_metrics(chunk.actual, chunk.prediction)
            row = {"endpoint": endpoint, "calibration_method": method, "rows": len(chunk), **current,
                   "brier_change_vs_raw": current["brier"] - raw_metrics["brier"],
                   "log_loss_change_vs_raw": current["log_loss"] - raw_metrics["log_loss"]}
            rows.append(row); options.append(row)
        options = pd.DataFrame(options)
        eligible = options.loc[
            options.brier.le(raw_metrics["brier"] + CALIBRATION_BRIER_NONINFERIORITY)
            & options.log_loss.le(raw_metrics["log_loss"] + CALIBRATION_LOG_LOSS_NONINFERIORITY)
        ].copy()
        best_ece = eligible.adaptive_ece.min()
        eligible = eligible.loc[eligible.adaptive_ece.le(best_ece + CALIBRATION_ECE_NEAR_TIE)].copy()
        eligible["simplicity"] = eligible.calibration_method.map(CALIBRATION_SIMPLICITY)
        method = eligible.sort_values(["simplicity", "brier"]).iloc[0].calibration_method
        selected[endpoint] = method
        for option in options.itertuples(index=False):
            decisions.append({
                "endpoint": endpoint, "calibration_method": option.calibration_method,
                "within_brier_noninferiority": option.brier <= raw_metrics["brier"] + CALIBRATION_BRIER_NONINFERIORITY,
                "within_log_loss_noninferiority": option.log_loss <= raw_metrics["log_loss"] + CALIBRATION_LOG_LOSS_NONINFERIORITY,
                "within_ece_near_tie": option.adaptive_ece <= best_ece + CALIBRATION_ECE_NEAR_TIE,
                "selected": option.calibration_method == method,
            })
    return pd.DataFrame(rows), pd.DataFrame(decisions), selected


def cluster_bootstrap(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    clusters = frame["canonical_player_id"].astype(str)
    unique = clusters.unique()
    sampled = rng.choice(unique, size=len(unique), replace=True)
    return pd.concat([frame.loc[clusters.eq(cluster)] for cluster in sampled], ignore_index=True)


def parameter_interval_task(endpoint: str, frame: pd.DataFrame, cache: dict, selected_method: str):
    origin = cache["spec"][0]
    target = ENDPOINTS[endpoint]["target"]
    evaluation = cache["evaluation"]
    rng = np.random.default_rng(stable_seed(endpoint, origin, selected_method, "parameter"))
    draws = []
    for _ in range(PARAMETER_BOOTSTRAP_REPETITIONS):
        tuning = cluster_bootstrap(cache["tuning"], rng)
        validation = cluster_bootstrap(cache["validation"], rng)
        if pd.to_numeric(tuning[target]).nunique() < 2 or pd.to_numeric(validation[target]).nunique() < 2:
            continue
        _, _, c_value, validation_raw = tune_c(tuning, validation, FEATURES, target)
        calibrator = fit_calibrator(selected_method, validation_raw, pd.to_numeric(validation[target]).to_numpy(int))
        final_fit = pd.concat([tuning, validation], ignore_index=True)
        pre = fit_preprocessor(final_fit, FEATURES)
        model = LogisticRegression(C=c_value, penalty="l2", solver="liblinear", max_iter=3000, random_state=RANDOM_SEED).fit(
            pre.transform(final_fit), pd.to_numeric(final_fit[target]).to_numpy(int)
        )
        raw = model.predict_proba(pre.transform(evaluation))[:, 1]
        draws.append(apply_calibrator(selected_method, calibrator, raw))
    matrix = np.asarray(draws)
    if len(matrix) < int(PARAMETER_BOOTSTRAP_REPETITIONS * .95):
        raise RuntimeError(f"Too few bootstrap refits for {endpoint}/{origin}: {len(matrix)}")
    low = np.quantile(matrix, PARAMETER_INTERVAL_ALPHA / 2, axis=0)
    high = np.quantile(matrix, 1 - PARAMETER_INTERVAL_ALPHA / 2, axis=0)
    result = pd.DataFrame({
        "endpoint": endpoint, "origin_id": origin,
        "capology_extension_event_id": evaluation.capology_extension_event_id.to_numpy(),
        "parameter_interval_low_80": low, "parameter_interval_high_80": high,
        "parameter_interval_width_80": high - low, "successful_parameter_refits": len(matrix),
    })
    return result


def parameter_intervals(frames: dict[str, pd.DataFrame], caches: dict[str, dict], selected: dict[str, str]):
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(parameter_interval_task, endpoint, frames[endpoint], cache, selected[endpoint]): (endpoint, origin)
            for endpoint, endpoint_cache in caches.items() for origin, cache in endpoint_cache.items()
        }
        for future in as_completed(futures):
            endpoint, origin = futures[future]
            results.append(future.result())
            print(f"Completed {endpoint}/{origin} parameter bootstrap.", flush=True)
    return pd.concat(results, ignore_index=True)


def clustered_skill_interval(chunk: pd.DataFrame, seed: tuple[str, ...]):
    grouped = chunk.assign(improvement=chunk.baseline_brier_loss - chunk.brier_loss).groupby("canonical_player_id").improvement.agg(["sum", "size"])
    rng = np.random.default_rng(stable_seed(*seed))
    sums, sizes, draws = grouped["sum"].to_numpy(float), grouped["size"].to_numpy(float), []
    for start in range(0, BOOTSTRAP_REPETITIONS, 250):
        count = min(250, BOOTSTRAP_REPETITIONS - start)
        indices = rng.integers(0, len(grouped), size=(count, len(grouped)))
        draws.extend((sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)).tolist())
    return float(np.quantile(draws, SUBGROUP_INTERVAL_ALPHA / 2)), float(np.quantile(draws, 1 - SUBGROUP_INTERVAL_ALPHA / 2))


def subgroup_evidence(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint, endpoint_rows in selected.groupby("endpoint"):
        for group_type, column in (("league", "league"), ("position", "canonical_position"), ("age", "age_band"), ("support", "support_status")):
            population = endpoint_rows if group_type == "support" else endpoint_rows.loc[~endpoint_rows.support_status.eq("refused")]
            for group, chunk in population.groupby(column, dropna=False):
                events, nonevents = int(chunk.actual.sum()), int(len(chunk) - chunk.actual.sum())
                sufficient = len(chunk) >= MINIMUM_SUBGROUP_ROWS and events >= MINIMUM_SUBGROUP_EVENTS and nonevents >= MINIMUM_SUBGROUP_NONEVENTS and chunk.origin_id.nunique() >= MINIMUM_SUBGROUP_ORIGINS
                auc = float(roc_auc_score(chunk.actual, chunk.prediction)) if events and nonevents else np.nan
                low, high = clustered_skill_interval(chunk, (endpoint, group_type, str(group), "subgroup"))
                gap = float(abs(chunk.actual.mean() - chunk.prediction.mean()))
                supported = sufficient and auc >= MINIMUM_SUBGROUP_AUC and gap <= MAXIMUM_SUBGROUP_CALIBRATION_GAP and low > 0
                rows.append({
                    "endpoint": endpoint, "group_type": group_type, "group": str(group), "rows": len(chunk),
                    "player_clusters": chunk.canonical_player_id.nunique(), "events": events, "nonevents": nonevents,
                    "origins": chunk.origin_id.nunique(), "actual_rate": float(chunk.actual.mean()),
                    "predicted_rate": float(chunk.prediction.mean()), "calibration_gap": gap,
                    "brier": float(chunk.brier_loss.mean()), "chronology_brier": float(chunk.baseline_brier_loss.mean()),
                    "brier_skill": float((chunk.baseline_brier_loss - chunk.brier_loss).mean()),
                    "brier_skill_ci_low_99": low, "brier_skill_ci_high_99": high, "roc_auc": auc,
                    "mean_parameter_width_80": float(chunk.parameter_interval_width_80.mean()),
                    "reliability_status": "supported" if supported else "limited_reliability" if sufficient else "insufficient_evidence",
                })
    return pd.DataFrame(rows)


def temporal_drift(caches: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for endpoint, endpoint_cache in caches.items():
        for origin, cache in endpoint_cache.items():
            reference, evaluation = cache["final_fit"], cache["evaluation"]
            for feature in FEATURES:
                if feature in {"canonical_position", "league"}:
                    ref = reference[feature].astype(str).value_counts(normalize=True)
                    ev = evaluation[feature].astype(str).value_counts(normalize=True)
                    categories = ref.index.union(ev.index)
                    score = float(.5 * (ref.reindex(categories, fill_value=0) - ev.reindex(categories, fill_value=0)).abs().sum())
                    metric = "total_variation"
                else:
                    ref = pd.to_numeric(reference[feature], errors="coerce").dropna()
                    ev = pd.to_numeric(evaluation[feature], errors="coerce").dropna()
                    edges = np.unique(ref.quantile(np.linspace(0, 1, 11)).to_numpy(float))
                    if len(edges) < 3:
                        score = 0.0
                    else:
                        edges[0], edges[-1] = -np.inf, np.inf
                        a = (pd.cut(ref, edges, include_lowest=True).value_counts(sort=False) / len(ref)).clip(lower=1e-6)
                        b = (pd.cut(ev, edges, include_lowest=True).value_counts(sort=False) / len(ev)).clip(lower=1e-6)
                        score = float(((b - a) * np.log(b / a)).sum())
                    metric = "psi"
                rows.append({"endpoint": endpoint, "origin_id": origin, "evaluation_year": cache["spec"][3], "feature": feature, "drift_metric": metric, "drift_value": score, "drift_status": "high" if score > .25 else "moderate" if score > .10 else "low"})
    return pd.DataFrame(rows)


def exposure_audit() -> pd.DataFrame:
    rows = []
    for endpoint in ENDPOINTS:
        all_rows = prepare_frame(endpoint, include_exposed=True)
        development_players = set(all_rows.loc[all_rows.signed_year.le(2023), "canonical_player_id"])
        exposed = all_rows.loc[all_rows.signed_year.ge(2024)]
        exposed_2024 = all_rows.loc[all_rows.signed_year.eq(2024)]
        disjoint_2024 = exposed_2024.loc[~exposed_2024.canonical_player_id.isin(development_players)]
        rows.append({
            "endpoint": endpoint, "first_exposure_date": "2026-08-22",
            "exposure_source": "frozen_V1_out_of_sample_verifier",
            "exposure_content": "labels_predictions_and_aggregate_performance",
            "exposed_2024plus_complete_feature_rows": len(exposed),
            "exposed_2024_complete_feature_rows": len(exposed_2024),
            "exposed_2024_player_disjoint_rows": len(disjoint_2024),
            "smallest_2024_player_disjoint_league_rows": int(disjoint_2024.groupby("league").size().min()),
            "classification": "previously_exposed_restricted_V1_confirmation_only",
            "permitted_for_v2_selection_or_promotion": False,
        })
    return pd.DataFrame(rows)


def redundancy_audit() -> pd.DataFrame:
    rows = []
    pair = [
        "pre365_same_club_all_competition_opportunity_share",
        "pre365_same_club_all_competition_appearance_rate",
    ]
    for endpoint in ENDPOINTS:
        frame = prepare_frame(endpoint)
        correlation = float(frame[pair].corr(method="spearman").iloc[0, 1])
        rows.append({
            "endpoint": endpoint,
            "feature_1": pair[0],
            "feature_2": pair[1],
            "spearman": correlation,
            "high_redundancy": abs(correlation) > .90,
            "required_explanation_policy": "group_only_recent_involvement",
        })
    return pd.DataFrame(rows)


def write_manifests() -> None:
    source_paths = [
        value_source.SOURCE,
        value_source.UPSTREAM_VERIFY,
        ROOT / "models" / "engine_v2" / "feature_contract.py",
        ROOT / "models" / "engine_v2" / "public_value_downside_repair_contract.py",
        ROOT / "models" / "run_engine_v2_public_value_downside_repair.py",
        ROOT / "scripts" / "verify_engine_v2_public_value_downside_repair.py",
        ROOT / "scripts" / "verify_deployment_out_of_sample.py",
        ROOT / "Data" / "processed" / "deployment_models" / "out_of_sample_verification.csv",
        FREEZE,
    ]
    source = pd.DataFrame([
        {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}
        for path in source_paths
    ])
    write_csv(source, OUTPUT / "source_manifest.csv")
    files = [path for path in sorted(OUTPUT.iterdir()) if path.is_file() and path.name != "output_manifest.csv"]
    output = pd.DataFrame([
        {"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
        for path in files
    ])
    write_csv(output, OUTPUT / "output_manifest.csv")


def finalize_existing_outputs() -> None:
    """Finalize a completed evidence run without repeating bootstrap refits.

    This exists for recovery when a post-computation integrity check changes;
    it never recomputes, edits, or selects model evidence.
    """
    decisions = pd.read_csv(OUTPUT / "selection_decision.csv")
    write_json(contract_payload(), OUTPUT / "preregistration.json")
    write_csv(redundancy_audit(), OUTPUT / "feature_redundancy.csv")
    checks = pd.read_csv(OUTPUT / "build_checks.csv")
    mutable_verification_reports = {
        "Data/processed/deployment_models/live_request_integrity.csv",
        "Data/processed/deployment_models/live_request_integrity.json",
    }
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    changed = [
        item["path"] for item in freeze["files"]
        if item["path"] not in mutable_verification_reports
        and sha256(ROOT / item["path"]) != item["sha256"]
    ]
    mask = checks["check"].eq("frozen_v1_unchanged")
    if mask.sum() != 1:
        raise RuntimeError("Existing build checks do not contain one V1 freeze row.")
    checks.loc[mask, "passed"] = not changed
    checks.loc[mask, "notes"] = "Frozen V1 model artifacts unchanged; mutable verification reports excluded."
    deployment_manifest = pd.read_csv(OUTPUT / "deployment_unchanged_manifest.csv")
    deployment_ok = bool(
        deployment_manifest.loc[
            ~deployment_manifest["artifact"].isin(mutable_verification_reports),
            "unchanged",
        ].astype(bool).all()
    )
    deployment_mask = checks["check"].eq("deployment_unchanged")
    if deployment_mask.sum() != 1:
        raise RuntimeError("Existing build checks do not contain one deployment row.")
    checks.loc[deployment_mask, "passed"] = deployment_ok
    checks.loc[deployment_mask, "notes"] = "Predictive deployment artifacts unchanged; mutable verification reports excluded."
    write_csv(checks, OUTPUT / "build_checks.csv")
    summary = {
        "contract_version": CONTRACT_VERSION,
        "all_build_checks_passed": bool(checks.passed.all()),
        "endpoints": json.loads(decisions.to_json(orient="records")),
        "known_2024plus_exposure": True,
        "overall_status": "development_evidence_only_no_pristine_final_holdout",
    }
    write_json(summary, OUTPUT / "run_summary.json")
    readme = [
        "# Engine V2 public-value downside cleanup", "",
        "This run removes proposal-time contract fields, raw goals/assists, and advanced event statistics; evaluates only a compact player/market/recent-involvement profile; and excludes every extension signed in 2024 or later.", "",
        "## Results", "",
    ]
    for row in decisions.itertuples(index=False):
        readme.append(
            f"- **{row.endpoint}:** AUC {row.pooled_roc_auc:.3f}, Brier {row.pooled_brier:.4f}, "
            f"chronology improvement {row.chronology_brier_improvement:+.4f}, calibration gap {row.pooled_calibration_gap:.3f}; "
            f"decision `{row.decision}`."
        )
    readme += [
        "", "## Interpretation boundary", "",
        "The output estimates whether Transfermarkt's public market-value estimate falls by at least 25% after an incumbent-club extension. It does not estimate accounting impairment, realized sale proceeds, transfer ROI, or the wisdom of extending the player.", "",
        "The 12-month endpoint is primary because it matures faster. The 24-month endpoint is secondary and provisional. Both available 2024+ cohorts were previously scored and their aggregate performance inspected for frozen V1, so neither can serve as an untouched final test for this repair.", "",
        "No model or calibrator was serialized or deployed.", "",
    ]
    (OUTPUT / "README.md").write_text("\n".join(readme), encoding="utf-8")
    write_manifests()
    if not checks.passed.all():
        raise RuntimeError(checks.loc[~checks.passed].to_string(index=False))
    print(json.dumps(summary, indent=2, default=str))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT.iterdir():
        if path.is_file():
            path.unlink()
    deployment_before = deployment_fingerprint()
    frames, candidate_parts, calibrated_parts, coefficients, cohorts, caches = {}, [], [], [], [], {}
    for endpoint in ENDPOINTS:
        frame, candidate, calibrated, coefficient, cohort, cache = fit_endpoint(endpoint)
        frames[endpoint], caches[endpoint] = frame, cache
        candidate_parts.append(candidate); calibrated_parts.append(calibrated)
        coefficients.append(coefficient); cohorts.append(cohort)
    candidate_predictions = pd.concat(candidate_parts, ignore_index=True)
    calibrated_predictions = pd.concat(calibrated_parts, ignore_index=True)
    coefficient_frame = pd.concat(coefficients, ignore_index=True)
    cohort_frame = pd.concat(cohorts, ignore_index=True)
    candidate_metrics, comparisons = candidate_evidence(candidate_predictions)
    calibration_metrics, calibration_selection, selected_methods = calibration_evidence(calibrated_predictions)
    intervals = parameter_intervals(frames, caches, selected_methods)
    selected = pd.concat([
        calibrated_predictions.loc[(calibrated_predictions.endpoint.eq(endpoint)) & (calibrated_predictions.calibration_method.eq(method))]
        for endpoint, method in selected_methods.items()
    ], ignore_index=True).merge(intervals, on=["endpoint", "origin_id", "capology_extension_event_id"], validate="one_to_one")
    subgroups = subgroup_evidence(selected)
    drift = temporal_drift(caches)
    exposure = exposure_audit()
    redundancy = redundancy_audit()

    decision_rows = []
    for endpoint in ENDPOINTS:
        endpoint_selected = selected.loc[selected.endpoint.eq(endpoint)]
        pooled = probability_metrics(endpoint_selected.actual, endpoint_selected.prediction)
        comparison = comparisons.loc[comparisons.endpoint.eq(endpoint)].set_index("baseline")
        origin_skill = endpoint_selected.groupby("origin_id").apply(lambda x: float((x.baseline_brier_loss - x.brier_loss).mean()), include_groups=False)
        core = subgroups.loc[subgroups.endpoint.eq(endpoint) & subgroups.group_type.isin(["league", "position"])]
        gates = {
            "chronology_ci_positive": comparison.loc["B0_chronology_prevalence", "brier_improvement_ci_low_95"] > 0,
            "chronology_origin_wins": comparison.loc["B0_chronology_prevalence", "origin_wins"] >= MINIMUM_ORIGIN_WINS,
            "market_profile_increment_positive": comparison.loc["B1_market_profile", "brier_improvement"] > 0,
            "pooled_auc": pooled["roc_auc"] >= MINIMUM_POOLED_AUC,
            "pooled_calibration_gap": pooled["calibration_gap"] <= MAXIMUM_ABSOLUTE_CALIBRATION_GAP,
            "positive_skill_origins": int(origin_skill.gt(0).sum()) >= MINIMUM_ORIGIN_WINS,
            "refusal_rate": float(endpoint_selected.support_status.eq("refused").mean()) <= MAXIMUM_REFUSAL_RATE,
            "all_broad_subgroups_supported": bool(core.reliability_status.eq("supported").all()),
        }
        development_passed = all(value for key, value in gates.items() if key != "all_broad_subgroups_supported")
        reliability_passed = all(gates.values())
        decision_rows.append({
            "endpoint": endpoint, "product_role": ENDPOINTS[endpoint]["product_role"],
            "selected_candidate": PRIMARY_CANDIDATE, "selected_calibration": selected_methods[endpoint],
            **{f"pooled_{key}": value for key, value in pooled.items()},
            "chronology_brier_improvement": comparison.loc["B0_chronology_prevalence", "brier_improvement"],
            "market_profile_brier_improvement": comparison.loc["B1_market_profile", "brier_improvement"],
            "positive_skill_origins": int(origin_skill.gt(0).sum()),
            "refusal_rate": float(endpoint_selected.support_status.eq("refused").mean()),
            "supported_broad_subgroups": int(core.reliability_status.eq("supported").sum()),
            "total_broad_subgroups": len(core),
            **{f"gate__{key}": value for key, value in gates.items()},
            "development_gate_passed": development_passed,
            "reliability_gate_passed": reliability_passed,
            "final_release_authorized": False,
            "decision": "freeze_development_candidate_waiting_pristine_later_test" if reliability_passed else "retain_research_candidate_reliability_repair_required",
            "deployment_status": "not_deployed",
        })
    decisions = pd.DataFrame(decision_rows)
    deployment_after = deployment_fingerprint()
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    # The live-request integrity CSV/JSON are mutable verification reports, not
    # fitted V1 model artifacts. Product-boundary repairs legitimately refresh
    # them, so the model freeze check excludes those two reports.
    mutable_verification_reports = {
        "Data/processed/deployment_models/live_request_integrity.csv",
        "Data/processed/deployment_models/live_request_integrity.json",
    }
    freeze_changed = [
        item["path"] for item in freeze["files"]
        if item["path"] not in mutable_verification_reports
        and sha256(ROOT / item["path"]) != item["sha256"]
    ]
    forbidden = set(RAW_SCORING_RATES + ADVANCED_PERFORMANCE + COMMERCIAL_SCENARIO)
    checks = pd.DataFrame([
        ("two_endpoints_only", set(decisions.endpoint) == set(ENDPOINTS), "12m and 24m 25% downside only"),
        ("no_exposed_evaluation_year", int(selected.evaluation_year.max()) == 2023, "all evaluation years <=2023"),
        ("player_disjoint_every_stage", cohort_frame[["identity_overlap", "tuning_identity_overlap", "calibration_identity_overlap"]].eq(0).all().all(), "zero identity overlap in fitting, tuning, and calibration"),
        ("proposal_free_features", not (set(FEATURES) & forbidden), "no scoring, advanced, salary, or proposal fields"),
        ("complete_required_inputs", all(frame[FEATURES].notna().all().all() for frame in frames.values()), "all required features observed"),
        ("known_exposure_explicit", exposure.classification.str.contains("previously_exposed").all(), "both 2024+ cohorts restricted"),
        ("no_final_release", not decisions.final_release_authorized.any(), "no pristine final test exists"),
        ("no_model_binary", not list(OUTPUT.rglob("*.joblib")), "evidence only"),
        ("deployment_unchanged", deployment_before == deployment_after, "deployment hashes unchanged"),
        ("frozen_v1_unchanged", not freeze_changed, "frozen V1 hashes unchanged"),
    ], columns=["check", "passed", "notes"])

    write_json(contract_payload(), OUTPUT / "preregistration.json")
    write_csv(cohort_frame, OUTPUT / "cohort_by_origin.csv")
    write_csv(candidate_predictions, OUTPUT / "candidate_predictions.csv")
    write_csv(candidate_metrics, OUTPUT / "candidate_metrics.csv")
    write_csv(comparisons, OUTPUT / "candidate_comparisons.csv")
    write_csv(calibrated_predictions, OUTPUT / "calibration_predictions.csv")
    write_csv(calibration_metrics, OUTPUT / "calibration_method_metrics.csv")
    write_csv(calibration_selection, OUTPUT / "calibration_selection.csv")
    write_csv(selected, OUTPUT / "selected_predictions.csv")
    write_csv(subgroups, OUTPUT / "subgroup_reliability.csv")
    write_csv(coefficient_frame, OUTPUT / "coefficient_stability.csv")
    write_csv(drift, OUTPUT / "temporal_drift.csv")
    write_csv(exposure, OUTPUT / "known_exposure_audit.csv")
    write_csv(redundancy, OUTPUT / "feature_redundancy.csv")
    write_csv(decisions, OUTPUT / "selection_decision.csv")
    write_csv(checks, OUTPUT / "build_checks.csv")
    write_csv(pd.DataFrame({"artifact": list(deployment_before), "before_sha256": list(deployment_before.values()), "after_sha256": [deployment_after.get(path) for path in deployment_before], "unchanged": [deployment_after.get(path) == digest for path, digest in deployment_before.items()]}), OUTPUT / "deployment_unchanged_manifest.csv")
    write_json({
        "contract_version": CONTRACT_VERSION, "all_build_checks_passed": bool(checks.passed.all()),
        "endpoints": json.loads(decisions.to_json(orient="records")), "known_2024plus_exposure": True,
        "overall_status": "development_evidence_only_no_pristine_final_holdout",
    }, OUTPUT / "run_summary.json")
    readme = [
        "# Engine V2 public-value downside cleanup", "",
        "This run removes proposal-time contract fields, raw goals/assists, and advanced event statistics; evaluates only a compact player/market/recent-involvement profile; and excludes every extension signed in 2024 or later.", "",
        "## Results", "",
    ]
    for row in decisions.itertuples(index=False):
        readme.append(
            f"- **{row.endpoint}:** AUC {row.pooled_roc_auc:.3f}, Brier {row.pooled_brier:.4f}, "
            f"chronology improvement {row.chronology_brier_improvement:+.4f}, calibration gap {row.pooled_calibration_gap:.3f}; "
            f"decision `{row.decision}`."
        )
    readme += [
        "", "## Interpretation boundary", "",
        "The output estimates whether Transfermarkt's public market-value estimate falls by at least 25% after an incumbent-club extension. It does not estimate accounting impairment, realized sale proceeds, transfer ROI, or the wisdom of extending the player.", "",
        "The 12-month endpoint is primary because it matures faster. The 24-month endpoint is secondary and provisional. Both available 2024+ cohorts were previously scored and their aggregate performance inspected for frozen V1, so neither can serve as an untouched final test for this repair.", "",
        "No model or calibrator was serialized or deployed.", "",
    ]
    (OUTPUT / "README.md").write_text("\n".join(readme), encoding="utf-8")
    write_manifests()
    if not checks.passed.all():
        raise RuntimeError(checks.loc[~checks.passed].to_string(index=False))
    print(json.dumps({"contract_version": CONTRACT_VERSION, "decisions": decision_rows}, indent=2, default=str), flush=True)


if __name__ == "__main__":
    if "--finalize-existing" in sys.argv:
        finalize_existing_outputs()
    else:
        main()

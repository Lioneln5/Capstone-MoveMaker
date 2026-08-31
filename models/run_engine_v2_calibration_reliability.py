"""Engine V2 Phase 3: calibration, uncertainty, subgroup, and OOD audit.

The runner consumes only Phase-2-approved feature specifications.  It writes
research evidence, not deployable model artifacts.
"""

from __future__ import annotations

import hashlib
import json
import sys
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

from models.engine_v2.feature_contract import (  # noqa: E402
    V2_CORE_FEATURES,
    V2_HISTORY_FEATURES,
)
from models.engine_v2.reliability_contract import (  # noqa: E402
    CALIBRATION_BRIER_NONINFERIORITY,
    CALIBRATION_METHODS,
    MAX_ACCEPTABLE_ECE,
    MAX_ACCEPTABLE_PARAMETER_INTERVAL_WIDTH,
    MAX_ACCEPTABLE_REFUSAL_RATE,
    MAX_SUPPORTED_CALIBRATION_GAP,
    MIN_SUPPORTED_SUBGROUP_SHARE,
    MIN_SUBGROUP_EVENTS,
    MIN_SUBGROUP_NONEVENTS,
    MIN_SUBGROUP_ORIGINS,
    MIN_SUBGROUP_ROWS,
    MIN_SUPPORTED_AUC,
    OOD_LIMITED_PERCENTILE,
    OOD_REFUSAL_PERCENTILE,
    REQUIRED_SUPPORTED_GROUP_TYPES,
)
from mixed_type_preprocessor import fit_preprocessor  # noqa: E402
import run_engine_v2_feature_specification as phase2  # noqa: E402

OUTPUT = ROOT / "Data" / "processed" / "engine_v2_calibration_reliability"
PHASE2 = ROOT / "Data" / "processed" / "engine_v2_feature_specification"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"
RANDOM_SEED = 20260830
BOOTSTRAP_REPETITIONS = 250
C_GRID = (0.01, 0.1, 1.0, 10.0)

ENDPOINT_FEATURES = {
    "sustained_meaningful_contribution": V2_CORE_FEATURES,
    "any_outbound_24m": V2_HISTORY_FEATURES,
    "downside_25pct_24m": V2_CORE_FEATURES,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: str) -> int:
    return (RANDOM_SEED + int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)) % (2**32 - 1)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_calibrator(method: str, probability: np.ndarray, actual: np.ndarray):
    if method == "raw":
        return None
    if method == "platt":
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000, random_state=RANDOM_SEED)
        model.fit(logit(probability).reshape(-1, 1), actual)
        return model
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(probability, actual)
    raise KeyError(method)


def apply_calibrator(method: str, calibrator, probability: np.ndarray) -> np.ndarray:
    if method == "raw":
        return np.clip(probability, 1e-6, 1 - 1e-6)
    if method == "platt":
        return np.clip(calibrator.predict_proba(logit(probability).reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)
    return np.clip(calibrator.predict(probability), 1e-6, 1 - 1e-6)


def adaptive_ece(actual: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    data = pd.DataFrame({"actual": actual, "probability": probability}).sort_values("probability")
    groups = np.array_split(np.arange(len(data)), min(bins, len(data)))
    return float(
        sum(
            len(indices)
            * abs(data.iloc[indices]["actual"].mean() - data.iloc[indices]["probability"].mean())
            for indices in groups
        )
        / len(data)
    )


def probability_metrics(actual: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    calibration = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000, random_state=RANDOM_SEED)
    calibration.fit(logit(probability).reshape(-1, 1), actual)
    return {
        "brier": float(brier_score_loss(actual, probability)),
        "log_loss": float(log_loss(actual, probability, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(actual, probability)),
        "average_precision": float(average_precision_score(actual, probability)),
        "adaptive_ece": adaptive_ece(actual, probability),
        "calibration_gap": float(abs(actual.mean() - probability.mean())),
        "calibration_intercept": float(calibration.intercept_[0]),
        "calibration_slope": float(calibration.coef_[0][0]),
        "actual_rate": float(actual.mean()),
        "predicted_rate": float(probability.mean()),
    }


def tune_base_model(train: pd.DataFrame, validation: pd.DataFrame, features: list[str], target: str):
    pre = fit_preprocessor(train, features)
    x_train, x_validation = pre.transform(train), pre.transform(validation)
    y_train = pd.to_numeric(train[target]).to_numpy(int)
    y_validation = pd.to_numeric(validation[target]).to_numpy(int)
    candidates = []
    for c_value in C_GRID:
        model = LogisticRegression(C=c_value, penalty="l2", solver="liblinear", max_iter=3000, random_state=RANDOM_SEED)
        model.fit(x_train, y_train)
        estimate = np.clip(model.predict_proba(x_validation)[:, 1], 1e-6, 1 - 1e-6)
        candidates.append((brier_score_loss(y_validation, estimate), c_value, model, estimate))
    _, c_value, model, validation_probability = min(candidates, key=lambda item: item[0])
    return pre, model, c_value, validation_probability


def support_profile(reference: pd.DataFrame, features: list[str]) -> tuple[dict, pd.DataFrame]:
    categorical = [feature for feature in features if feature in {"canonical_position", "league"}]
    numeric = [feature for feature in features if feature not in categorical]
    complete = reference[features].notna().all(axis=1)
    clean = reference.loc[complete, features].copy()
    profile = {"categorical": {}, "numeric": {}, "complete_reference_rows": int(len(clean))}
    distances = pd.DataFrame(index=clean.index)
    for feature in categorical:
        counts = clean[feature].astype(str).value_counts()
        profile["categorical"][feature] = {str(k): int(v) for k, v in counts.items()}
    for feature in numeric:
        values = pd.to_numeric(clean[feature], errors="coerce")
        median = float(values.median())
        q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
        scale = max((q3 - q1) / 1.349, 1e-8)
        profile["numeric"][feature] = {
            "median": median, "robust_scale": scale,
            "min": float(values.min()), "max": float(values.max()),
            "q01": float(values.quantile(0.01)), "q99": float(values.quantile(0.99)),
        }
        distances[feature] = (values - median).abs() / scale
    max_distance = distances.max(axis=1)
    profile["distance_limited_threshold"] = float(max_distance.quantile(OOD_LIMITED_PERCENTILE))
    profile["distance_refusal_threshold"] = float(max_distance.quantile(OOD_REFUSAL_PERCENTILE))
    return profile, clean


def assess_support(frame: pd.DataFrame, features: list[str], profile: dict) -> pd.DataFrame:
    rows = []
    categorical = [feature for feature in features if feature in {"canonical_position", "league"}]
    numeric = [feature for feature in features if feature not in categorical]
    for _, row in frame.iterrows():
        reasons = []
        missing = [feature for feature in features if pd.isna(row[feature])]
        if missing:
            reasons.append("MISSING_REQUIRED_FEATURE:" + "|".join(missing))
        unseen = [feature for feature in categorical if not pd.isna(row[feature]) and str(row[feature]) not in profile["categorical"][feature]]
        if unseen:
            reasons.append("UNSEEN_CATEGORY:" + "|".join(unseen))
        distances = []
        outside_1_99 = []
        outside_observed = []
        if not missing:
            for feature in numeric:
                value = float(row[feature])
                spec = profile["numeric"][feature]
                distances.append(abs(value - spec["median"]) / spec["robust_scale"])
                if value < spec["min"] or value > spec["max"]:
                    outside_observed.append(feature)
                elif value < spec["q01"] or value > spec["q99"]:
                    outside_1_99.append(feature)
        distance = max(distances) if distances else np.nan
        if outside_observed:
            reasons.append("OUTSIDE_OBSERVED_RANGE:" + "|".join(outside_observed))
        if pd.notna(distance) and distance > profile["distance_refusal_threshold"]:
            reasons.append("MAX_ROBUST_FEATURE_DISTANCE_ABOVE_P99")
        refused = bool(missing or unseen or outside_observed or (pd.notna(distance) and distance > profile["distance_refusal_threshold"]))
        limited = bool(outside_1_99 or (pd.notna(distance) and distance > profile["distance_limited_threshold"]))
        status = "refused" if refused else "limited" if limited else "supported"
        rows.append({"support_status": status, "support_reasons": ";".join(reasons), "support_distance": distance})
    return pd.DataFrame(rows, index=frame.index)


def bootstrap_intervals(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    features: list[str],
    target: str,
    c_value: float,
    method: str,
    calibrator,
    pre,
    endpoint: str,
    origin: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    x_train = pre.transform(train)
    x_evaluation = pre.transform(evaluation)
    y_train = pd.to_numeric(train[target]).to_numpy(int)
    cluster_series = train["canonical_player_id"].astype(str)
    clusters = cluster_series.unique()
    row_indices = {cluster: np.flatnonzero(cluster_series.to_numpy() == cluster) for cluster in clusters}
    rng = np.random.default_rng(stable_seed(endpoint, origin, "parameter_bootstrap"))
    estimates = []
    for _ in range(BOOTSTRAP_REPETITIONS):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        indices = np.concatenate([row_indices[cluster] for cluster in sampled])
        if len(np.unique(y_train[indices])) < 2:
            continue
        model = LogisticRegression(C=c_value, penalty="l2", solver="liblinear", max_iter=3000, random_state=RANDOM_SEED)
        model.fit(x_train[indices], y_train[indices])
        raw = np.clip(model.predict_proba(x_evaluation)[:, 1], 1e-6, 1 - 1e-6)
        estimates.append(apply_calibrator(method, calibrator, raw))
    matrix = np.asarray(estimates)
    return np.quantile(matrix, 0.10, axis=0), np.quantile(matrix, 0.90, axis=0), len(matrix)


def fit_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    frames = phase2.load_endpoint_frames()
    prediction_rows, metric_rows, support_rows, support_profiles = [], [], [], []
    for endpoint, data in frames.items():
        target = phase2.ENDPOINT_META[endpoint]["target"]
        features = list(ENDPOINT_FEATURES[endpoint])
        for origin, train_end, validation_year, evaluation_year in phase2.ORIGINS:
            train = data.loc[data["signed_year"].le(train_end)].copy()
            validation = data.loc[data["signed_year"].eq(validation_year)].copy()
            evaluation = data.loc[data["signed_year"].eq(evaluation_year)].sort_values("capology_extension_event_id").copy()
            pre, model, c_value, validation_raw = tune_base_model(train, validation, features, target)
            evaluation_raw = np.clip(model.predict_proba(pre.transform(evaluation))[:, 1], 1e-6, 1 - 1e-6)
            actual_validation = pd.to_numeric(validation[target]).to_numpy(int)
            actual_evaluation = pd.to_numeric(evaluation[target]).to_numpy(int)
            baseline_probability = float(pd.to_numeric(train[target]).mean())
            calibrators = {method: fit_calibrator(method, validation_raw, actual_validation) for method in CALIBRATION_METHODS}
            reference = pd.concat([train, validation], ignore_index=True)
            profile, _ = support_profile(reference, features)
            profile_record = {"endpoint": endpoint, "origin_id": origin, "evaluation_year": evaluation_year, "features": features, **profile}
            support_profiles.append(profile_record)
            support = assess_support(evaluation, features, profile)
            for method in CALIBRATION_METHODS:
                estimate = apply_calibrator(method, calibrators[method], evaluation_raw)
                metrics = probability_metrics(actual_evaluation, estimate)
                metric_rows.append({
                    "endpoint": endpoint, "origin_id": origin, "evaluation_year": evaluation_year,
                    "calibration_method": method, "train_rows": len(train), "calibration_rows": len(validation),
                    "evaluation_rows": len(evaluation), "selected_C": c_value, **metrics,
                })
                for i, (_, row) in enumerate(evaluation.iterrows()):
                    prediction_rows.append({
                        "endpoint": endpoint, "origin_id": origin, "evaluation_year": evaluation_year,
                        "calibration_method": method, "capology_extension_event_id": row["capology_extension_event_id"],
                        "canonical_player_id": row["canonical_player_id"], "canonical_position": row["canonical_position"],
                        "league": row["league"], "age_band": row["age_band"],
                        "actual": int(actual_evaluation[i]), "prediction": float(estimate[i]),
                        "baseline_prediction": baseline_probability,
                        "support_status": support.iloc[i]["support_status"],
                        "support_reasons": support.iloc[i]["support_reasons"],
                        "support_distance": support.iloc[i]["support_distance"],
                        "brier_loss": float((actual_evaluation[i] - estimate[i]) ** 2),
                        "baseline_brier_loss": float((actual_evaluation[i] - baseline_probability) ** 2),
                    })
            for status, chunk in support.groupby("support_status"):
                support_rows.append({
                    "endpoint": endpoint, "origin_id": origin, "evaluation_year": evaluation_year,
                    "support_status": status, "rows": len(chunk), "share": len(chunk) / len(evaluation),
                })
    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows), pd.DataFrame(support_rows), support_profiles


def select_calibration(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons, decisions = [], []
    for endpoint, data in predictions.groupby("endpoint"):
        methods = {}
        for method, chunk in data.groupby("calibration_method"):
            metrics = probability_metrics(chunk["actual"].to_numpy(int), chunk["prediction"].to_numpy(float))
            methods[method] = metrics
        raw = methods["raw"]
        for method in CALIBRATION_METHODS:
            current = methods[method]
            by_origin = []
            if method != "raw":
                for _, origin in data.groupby("origin_id"):
                    raw_origin = origin.loc[origin["calibration_method"].eq("raw"), "brier_loss"].mean()
                    method_origin = origin.loc[origin["calibration_method"].eq(method), "brier_loss"].mean()
                    by_origin.append(raw_origin - method_origin)
            comparisons.append({
                "endpoint": endpoint, "calibration_method": method,
                "pooled_brier": current["brier"], "pooled_log_loss": current["log_loss"],
                "pooled_ece": current["adaptive_ece"], "calibration_gap": current["calibration_gap"],
                "brier_improvement_vs_raw": raw["brier"] - current["brier"],
                "ece_improvement_vs_raw": raw["adaptive_ece"] - current["adaptive_ece"],
                "origin_wins_vs_raw": int(sum(value > 0 for value in by_origin)) if by_origin else 0,
            })
        table = pd.DataFrame(comparisons).loc[lambda x: x["endpoint"].eq(endpoint)].copy()
        eligible = table.loc[
            table["pooled_brier"].le(raw["brier"] + CALIBRATION_BRIER_NONINFERIORITY)
            & table["pooled_log_loss"].le(raw["log_loss"] + 0.01)
        ].copy()
        eligible["simplicity"] = eligible["calibration_method"].map({"raw": 0, "platt": 1, "isotonic": 2})
        # Lowest ECE wins; near-ties (0.005) favor the simpler method.
        best_ece = eligible["pooled_ece"].min()
        chosen = eligible.loc[eligible["pooled_ece"].le(best_ece + 0.005)].sort_values("simplicity").iloc[0]
        decisions.append({
            "endpoint": endpoint, "selected_calibration": chosen.calibration_method,
            "pooled_brier": chosen.pooled_brier, "pooled_log_loss": chosen.pooled_log_loss,
            "pooled_ece": chosen.pooled_ece, "calibration_gap": chosen.calibration_gap,
            "brier_improvement_vs_raw": chosen.brier_improvement_vs_raw,
            "ece_improvement_vs_raw": chosen.ece_improvement_vs_raw,
        })
    return pd.DataFrame(comparisons), pd.DataFrame(decisions)


def attach_uncertainty(predictions: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    frames = phase2.load_endpoint_frames()
    selected = predictions.merge(decisions[["endpoint", "selected_calibration"]], on="endpoint")
    selected = selected.loc[selected["calibration_method"].eq(selected["selected_calibration"])].copy()
    output = []
    for endpoint, data in frames.items():
        target = phase2.ENDPOINT_META[endpoint]["target"]
        features = list(ENDPOINT_FEATURES[endpoint])
        method = decisions.loc[decisions["endpoint"].eq(endpoint), "selected_calibration"].iloc[0]
        for origin, train_end, validation_year, evaluation_year in phase2.ORIGINS:
            train = data.loc[data["signed_year"].le(train_end)].copy()
            validation = data.loc[data["signed_year"].eq(validation_year)].copy()
            evaluation = data.loc[data["signed_year"].eq(evaluation_year)].sort_values("capology_extension_event_id").copy()
            pre, model, c_value, validation_raw = tune_base_model(train, validation, features, target)
            calibrator = fit_calibrator(method, validation_raw, pd.to_numeric(validation[target]).to_numpy(int))
            low, high, successful = bootstrap_intervals(train, evaluation, features, target, c_value, method, calibrator, pre, endpoint, origin)
            chunk = selected.loc[selected["endpoint"].eq(endpoint) & selected["origin_id"].eq(origin)].sort_values("capology_extension_event_id").copy()
            if len(chunk) != len(evaluation):
                raise RuntimeError("Uncertainty/prediction row mismatch")
            chunk["parameter_interval_low_80"] = low
            chunk["parameter_interval_high_80"] = high
            chunk["parameter_interval_width_80"] = high - low
            chunk["successful_bootstrap_fits"] = successful
            output.append(chunk)
    return pd.concat(output, ignore_index=True)


def subgroup_reliability(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint, data in selected.groupby("endpoint"):
        for group_type, column in [("position", "canonical_position"), ("league", "league"), ("age", "age_band"), ("support", "support_status")]:
            for group, chunk in data.groupby(column, dropna=False):
                n = len(chunk)
                events = int(chunk["actual"].sum())
                nonevents = n - events
                origins = chunk["origin_id"].nunique()
                auc = float(roc_auc_score(chunk["actual"], chunk["prediction"])) if events and nonevents else np.nan
                brier = float(chunk["brier_loss"].mean())
                baseline_brier = float(chunk["baseline_brier_loss"].mean())
                gap = float(abs(chunk["actual"].mean() - chunk["prediction"].mean()))
                sufficient = n >= MIN_SUBGROUP_ROWS and events >= MIN_SUBGROUP_EVENTS and nonevents >= MIN_SUBGROUP_NONEVENTS and origins >= MIN_SUBGROUP_ORIGINS
                metrics_ok = bool(sufficient and auc >= MIN_SUPPORTED_AUC and gap <= MAX_SUPPORTED_CALIBRATION_GAP and brier < baseline_brier)
                status = "supported" if metrics_ok else "limited_reliability" if sufficient else "insufficient_evidence"
                rows.append({
                    "endpoint": endpoint, "group_type": group_type, "group": str(group),
                    "rows": n, "events": events, "nonevents": nonevents, "origins": origins,
                    "actual_rate": chunk["actual"].mean(), "predicted_rate": chunk["prediction"].mean(),
                    "calibration_gap": gap, "brier": brier, "baseline_brier": baseline_brier,
                    "brier_skill": baseline_brier - brier, "roc_auc": auc,
                    "mean_parameter_interval_width_80": chunk["parameter_interval_width_80"].mean(),
                    "reliability_status": status,
                })
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for stale in ["independent_verification.csv", "independent_verification.json"]:
        (OUTPUT / stale).unlink(missing_ok=True)

    phase2_verify = json.loads((PHASE2 / "independent_verification.json").read_text(encoding="utf-8"))
    if not phase2_verify.get("all_checks_passed"):
        raise RuntimeError("Phase 2 has not passed independent verification")

    predictions, origin_metrics, support_summary, support_profiles = fit_all()
    calibration_comparison, calibration_decisions = select_calibration(predictions)
    selected = attach_uncertainty(predictions, calibration_decisions)
    subgroup = subgroup_reliability(selected)

    # Retain all three out-of-time probability streams locally so the
    # independent verifier can reconstruct the calibration comparison rather
    # than trusting aggregate metrics emitted by this runner.  Repository
    # publication rules exclude regenerable prediction tables from Git.
    predictions.to_csv(OUTPUT / "calibration_predictions.csv", index=False, lineterminator="\n")

    prediction_columns = [
        "endpoint", "origin_id", "evaluation_year", "capology_extension_event_id", "canonical_player_id",
        "actual", "prediction", "baseline_prediction", "support_status", "support_reasons", "support_distance",
        "parameter_interval_low_80", "parameter_interval_high_80", "parameter_interval_width_80",
        "successful_bootstrap_fits", "canonical_position", "league", "age_band",
    ]
    selected[prediction_columns].to_csv(OUTPUT / "selected_predictions.csv", index=False, lineterminator="\n")
    origin_metrics.to_csv(OUTPUT / "origin_calibration_metrics.csv", index=False, lineterminator="\n")
    calibration_comparison.to_csv(OUTPUT / "calibration_method_comparison.csv", index=False, lineterminator="\n")
    calibration_decisions.to_csv(OUTPUT / "calibration_decisions.csv", index=False, lineterminator="\n")
    support_summary.to_csv(OUTPUT / "ood_support_summary.csv", index=False, lineterminator="\n")
    subgroup.to_csv(OUTPUT / "subgroup_reliability.csv", index=False, lineterminator="\n")
    write_json(OUTPUT / "ood_support_profiles.json", support_profiles)

    endpoint_rows = []
    for row in calibration_decisions.itertuples(index=False):
        chunk = selected.loc[selected["endpoint"].eq(row.endpoint)]
        refused_rate = float(chunk["support_status"].eq("refused").mean())
        width = float(chunk["parameter_interval_width_80"].mean())
        endpoint_groups = subgroup.loc[subgroup["endpoint"].eq(row.endpoint)].copy()
        reliable_groups = endpoint_groups["reliability_status"]
        supported_share = float(reliable_groups.eq("supported").mean())
        core_groups_supported = bool(
            endpoint_groups.loc[
                endpoint_groups["group_type"].isin(REQUIRED_SUPPORTED_GROUP_TYPES),
                "reliability_status",
            ].eq("supported").all()
        )
        gate = bool(
            row.pooled_ece <= MAX_ACCEPTABLE_ECE
            and refused_rate <= MAX_ACCEPTABLE_REFUSAL_RATE
            and width <= MAX_ACCEPTABLE_PARAMETER_INTERVAL_WIDTH
            and supported_share >= MIN_SUPPORTED_SUBGROUP_SHARE
            and core_groups_supported
        )
        endpoint_rows.append({
            "endpoint": row.endpoint, "selected_calibration": row.selected_calibration,
            "pooled_brier": row.pooled_brier, "pooled_ece": row.pooled_ece,
            "calibration_gap": row.calibration_gap, "refused_profile_rate": refused_rate,
            "mean_parameter_interval_width_80": width,
            "supported_subgroups": int(reliable_groups.eq("supported").sum()),
            "supported_subgroup_share": supported_share,
            "core_position_league_groups_supported": core_groups_supported,
            "limited_subgroups": int(reliable_groups.eq("limited_reliability").sum()),
            "insufficient_subgroups": int(reliable_groups.eq("insufficient_evidence").sum()),
            "phase3_gate_passed": gate,
            "phase3_decision": "advance_to_versioned_artifact_build" if gate else "blocked_reliability_repair",
            "deployment_status": "not_deployed",
        })
    endpoint_decisions = pd.DataFrame(endpoint_rows)
    endpoint_decisions.to_csv(OUTPUT / "phase3_decisions.csv", index=False, lineterminator="\n")

    source_paths = [
        PHASE2 / "phase2_decisions.csv", PHASE2 / "candidate_comparisons.csv",
        PHASE2 / "independent_verification.json", FREEZE,
    ]
    pd.DataFrame([
        {"source": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in source_paths
    ]).to_csv(OUTPUT / "source_manifest.csv", index=False, lineterminator="\n")

    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    freeze_ok = all(sha256(ROOT / item["path"]) == item["sha256"] for item in freeze["files"])
    checks = pd.DataFrame([
        ("three_phase2_candidates_only", set(endpoint_decisions["endpoint"]) == set(ENDPOINT_FEATURES), "No V1 auxiliary endpoint slipped back in."),
        ("calibration_methods_complete", set(calibration_comparison["calibration_method"]) == set(CALIBRATION_METHODS), "Raw, Platt, and isotonic tested."),
        ("four_rolling_origins", set(selected["evaluation_year"]) == {2020, 2021, 2022, 2023}, "No random split."),
        ("probabilities_bounded", selected["prediction"].between(0, 1).all(), "All selected probabilities are valid."),
        ("uncertainty_intervals_ordered", (selected["parameter_interval_low_80"] <= selected["parameter_interval_high_80"]).all(), "Bootstrap bounds are ordered."),
        ("support_status_exhaustive", set(selected["support_status"]).issubset({"supported", "limited", "refused"}), "Every evaluation profile has a declared support status."),
        ("refused_profiles_have_reasons", selected.loc[selected["support_status"].eq("refused"), "support_reasons"].fillna("").str.len().gt(0).all(), "No unexplained refusal."),
        ("subgroup_status_exhaustive", set(subgroup["reliability_status"]).issubset({"supported", "limited_reliability", "insufficient_evidence"}), "Every subgroup has a declared status."),
        ("no_model_artifacts_written", not list(OUTPUT.rglob("*.joblib")), "Phase 3 still cannot deploy."),
        ("v1_artifacts_remain_frozen", freeze_ok, "All Phase-0 hashes remain byte-identical."),
    ], columns=["check", "passed", "notes"])
    checks.to_csv(OUTPUT / "build_checks.csv", index=False, lineterminator="\n")

    summary = {
        "phase": "engine_v2_phase_3_calibration_reliability",
        "status": "complete_research_only_not_deployed",
        "endpoint_decisions": endpoint_decisions.set_index("endpoint")["phase3_decision"].to_dict(),
        "selected_calibration": endpoint_decisions.set_index("endpoint")["selected_calibration"].to_dict(),
        "evaluation_rows": int(len(selected)),
        "parameter_bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "checks_passed": int(checks["passed"].sum()), "checks_total": len(checks),
        "all_checks_passed": bool(checks["passed"].all()), "v1_artifacts_modified": False,
    }
    write_json(OUTPUT / "run_summary.json", summary)

    lines = [
        "# Engine V2 Phase 3 — calibration and reliability", "",
        "This package evaluates calibration, parameter uncertainty, subgroup reliability, and out-of-distribution refusal for the three Phase-2 candidates. No artifact was serialized or deployed.", "",
        "## Endpoint decisions", "",
    ]
    for row in endpoint_decisions.itertuples(index=False):
        lines.append(
            f"- `{row.endpoint}`: calibration `{row.selected_calibration}`; Brier {row.pooled_brier:.4f}; ECE {row.pooled_ece:.4f}; "
            f"refused-profile rate {row.refused_profile_rate:.1%}; mean 80% parameter-interval width {row.mean_parameter_interval_width_80:.3f}; "
            f"supported subgroup views {row.supported_subgroup_share:.1%}; broad position/league gate "
            f"{'passed' if row.core_position_league_groups_supported else 'failed'}; "
            f"decision `{row.phase3_decision}`."
        )
    lines += [
        "", "## Interpretation", "",
        "- `supported` means the profile lies inside the past-data density envelope; it does not mean the outcome is certain.",
        "- `limited` means the profile is near the historical edge and must display an extrapolation warning and model-fit uncertainty.",
        "- `refused` means a required feature is missing, a category is unseen, a numeric value is outside the observed range, or the profile's maximum robust feature deviation lies beyond the learned 99th-percentile threshold. V2 must not emit a probability in that state.",
        "- Subgroup reliability is evaluated separately from individual profile support. A profile can be in-domain while its league/position/age subgroup remains statistically limited.",
        "", "Passing Phase 3 permits only a separately reviewed, versioned artifact build. It does not restore public scoring.", "",
    ]
    (OUTPUT / "README.md").write_text("\n".join(lines), encoding="utf-8")

    manifest_rows = []
    for path in sorted(OUTPUT.iterdir()):
        if path.is_file() and path.name != "output_manifest.csv":
            manifest_rows.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest_rows).to_csv(OUTPUT / "output_manifest.csv", index=False, lineterminator="\n")
    print(json.dumps(summary, indent=2))
    if not summary["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

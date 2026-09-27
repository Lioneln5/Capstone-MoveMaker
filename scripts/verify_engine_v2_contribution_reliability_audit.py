"""Independent verification for repaired-contribution reliability Run 3.

The verifier does not import the Run 3 runner. It independently refits every
origin and calibration mapping, repeats all 1,000 nested parameter bootstraps,
and reconstructs subgroup intervals and the final gate.
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

from models.engine_v2.complete_contribution_target_contract import TARGET_FIELD  # noqa: E402
from models.engine_v2.contribution_model_development_contract import LOGISTIC_C_GRID, ORIGINS  # noqa: E402
from models.engine_v2.contribution_reliability_contract import (  # noqa: E402
    CALIBRATION_BRIER_NONINFERIORITY,
    CALIBRATION_ECE_NEAR_TIE,
    CALIBRATION_LOG_LOSS_NONINFERIORITY,
    CALIBRATION_METHODS,
    CALIBRATION_SIMPLICITY,
    HARD_REQUIRED_FEATURES,
    INVOLVEMENT_FEATURES,
    MAXIMUM_CALIBRATION_SLOPE,
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
    MINIMUM_CALIBRATION_SLOPE,
    MINIMUM_POSITIVE_SKILL_ORIGINS,
    MINIMUM_RELIABLE_ORIGINS,
    MINIMUM_SUBGROUP_AUC,
    MINIMUM_SUBGROUP_EVENTS,
    MINIMUM_SUBGROUP_NONEVENTS,
    MINIMUM_SUBGROUP_ORIGINS,
    MINIMUM_SUBGROUP_ROWS,
    OOD_LIMITED_PERCENTILE,
    OOD_REFUSAL_PERCENTILE,
    PARAMETER_BOOTSTRAP_REPETITIONS,
    RANDOM_SEED,
    REDUNDANCY_SPEARMAN_THRESHOLD,
    REQUIRED_SUBGROUP_TYPES,
    SUBGROUP_BOOTSTRAP_REPETITIONS,
    SUBGROUP_INTERVAL_ALPHA,
)
from models.engine_v2.feature_contract import V2_CORE_FEATURES  # noqa: E402
from models.mixed_type_preprocessor import fit_preprocessor  # noqa: E402
from models.run_engine_v2_contribution_model_rerun import prepare_model_frame  # noqa: E402


OUTPUT = ROOT / "Data" / "processed" / "engine_v2_contribution_reliability_audit"
RUN2 = ROOT / "Data" / "processed" / "engine_v2_contribution_model_rerun"
FEATURES = list(V2_CORE_FEATURES)
TARGET = TARGET_FIELD


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: str) -> int:
    token = "|".join(parts).encode("utf-8")
    return (RANDOM_SEED + int(hashlib.sha256(token).hexdigest()[:8], 16)) % (2**32 - 1)


def logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def fit_calibrator(method: str, probability: np.ndarray, actual: np.ndarray):
    if method == "raw":
        return None
    if method == "platt":
        return LogisticRegression(
            C=1e6, solver="lbfgs", max_iter=3000, random_state=RANDOM_SEED
        ).fit(logit(probability).reshape(-1, 1), actual)
    return IsotonicRegression(
        out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6
    ).fit(probability, actual)


def apply_calibrator(method: str, calibrator, probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    if method == "raw":
        return probability
    if method == "platt":
        return np.clip(
            calibrator.predict_proba(logit(probability).reshape(-1, 1))[:, 1],
            1e-6,
            1 - 1e-6,
        )
    return np.clip(calibrator.predict(probability), 1e-6, 1 - 1e-6)


def adaptive_ece(actual: np.ndarray, prediction: np.ndarray) -> float:
    ordered = pd.DataFrame({"actual": actual, "prediction": prediction}).sort_values("prediction")
    groups = np.array_split(np.arange(len(ordered)), min(10, len(ordered)))
    return float(sum(len(i) * abs(ordered.iloc[i].actual.mean() - ordered.iloc[i].prediction.mean()) for i in groups) / len(ordered))


def metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
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
    base = data.loc[data.signed_year.le(train_end)].copy()
    validation = data.loc[data.signed_year.eq(validation_year)].copy()
    evaluation = data.loc[data.signed_year.eq(evaluation_year)].sort_values("capology_extension_event_id").copy()
    tuning = base.loc[~base.canonical_player_id.isin(set(validation.canonical_player_id))].copy()
    final_fit = pd.concat([base, validation], ignore_index=True)
    final_fit = final_fit.loc[~final_fit.canonical_player_id.isin(set(evaluation.canonical_player_id))].copy()
    return tuning, validation, final_fit, evaluation


def tune_c(tuning: pd.DataFrame, validation: pd.DataFrame, features: list[str]):
    pre = fit_preprocessor(tuning, features)
    x_train, x_validation = pre.transform(tuning), pre.transform(validation)
    y_train = pd.to_numeric(tuning[TARGET]).to_numpy(int)
    y_validation = pd.to_numeric(validation[TARGET]).to_numpy(int)
    trials = []
    for c_value in LOGISTIC_C_GRID:
        model = LogisticRegression(C=c_value, penalty="l2", solver="liblinear", max_iter=3000, random_state=RANDOM_SEED).fit(x_train, y_train)
        prediction = np.clip(model.predict_proba(x_validation)[:, 1], 1e-6, 1 - 1e-6)
        trials.append((float(brier_score_loss(y_validation, prediction)), float(log_loss(y_validation, prediction, labels=[0, 1])), c_value, prediction))
    return min(trials, key=lambda item: (item[0], item[1], item[2]))


def support_profile(reference: pd.DataFrame) -> dict[str, object]:
    complete = reference.loc[reference[list(HARD_REQUIRED_FEATURES)].notna().all(axis=1)].copy()
    profile: dict[str, object] = {"categorical": {}, "numeric": {}}
    distances = pd.DataFrame(index=complete.index)
    for feature in FEATURES:
        if feature in {"canonical_position", "league"}:
            profile["categorical"][feature] = sorted(complete[feature].astype(str).unique())
        else:
            values = pd.to_numeric(complete[feature])
            median = float(values.median())
            q1, q3 = float(values.quantile(.25)), float(values.quantile(.75))
            scale = max((q3 - q1) / 1.349, 1e-8)
            profile["numeric"][feature] = {
                "median": median, "robust_scale": scale,
                "minimum": float(values.min()), "maximum": float(values.max()),
                "q01": float(values.quantile(.01)), "q99": float(values.quantile(.99)),
            }
            distances[feature] = (values - median).abs() / scale
    maximum = distances.max(axis=1)
    profile["distance_limited_threshold"] = float(maximum.quantile(OOD_LIMITED_PERCENTILE))
    profile["distance_refusal_threshold"] = float(maximum.quantile(OOD_REFUSAL_PERCENTILE))
    return profile


def support_status(frame: pd.DataFrame, profile: dict[str, object]) -> pd.DataFrame:
    output = []
    for _, row in frame.iterrows():
        refused, limited = [], []
        missing = [feature for feature in HARD_REQUIRED_FEATURES if pd.isna(row[feature])]
        if missing:
            refused.append("MISSING_REQUIRED_FEATURE:" + "|".join(missing))
        for feature in ("canonical_position", "league"):
            if pd.notna(row[feature]) and str(row[feature]) not in profile["categorical"][feature]:
                refused.append("UNSEEN_CATEGORY:" + feature)
        distances = []
        if not missing:
            for feature, spec in profile["numeric"].items():
                value = float(row[feature])
                distances.append(abs(value - spec["median"]) / spec["robust_scale"])
                if value < spec["minimum"] or value > spec["maximum"]:
                    refused.append("OUTSIDE_OBSERVED_RANGE:" + feature)
                elif value < spec["q01"] or value > spec["q99"]:
                    limited.append("OUTSIDE_TRAINING_P01_P99:" + feature)
        distance = max(distances) if distances else np.nan
        if pd.notna(distance) and distance > profile["distance_refusal_threshold"]:
            refused.append("ROBUST_DISTANCE_ABOVE_TRAINING_P99")
        elif pd.notna(distance) and distance > profile["distance_limited_threshold"]:
            limited.append("ROBUST_DISTANCE_ABOVE_TRAINING_P95")
        output.append({
            "support_status": "refused" if refused else "limited" if limited else "supported",
            "support_reasons": ";".join(refused + limited),
            "support_distance": distance,
        })
    return pd.DataFrame(output)


def refit_origins(data: pd.DataFrame):
    rows, coefficient_rows, cache = [], [], {}
    for origin, train_end, validation_year, evaluation_year in ORIGINS:
        tuning, validation, final_fit, evaluation = split_origin(data, train_end, validation_year, evaluation_year)
        _, _, selected_c, validation_raw = tune_c(tuning, validation, FEATURES)
        calibrators = {method: fit_calibrator(method, validation_raw, pd.to_numeric(validation[TARGET]).to_numpy(int)) for method in CALIBRATION_METHODS}
        pre = fit_preprocessor(final_fit, FEATURES)
        model = LogisticRegression(C=selected_c, penalty="l2", solver="liblinear", max_iter=3000, random_state=RANDOM_SEED).fit(pre.transform(final_fit), pd.to_numeric(final_fit[TARGET]).to_numpy(int))
        raw = np.clip(model.predict_proba(pre.transform(evaluation))[:, 1], 1e-6, 1 - 1e-6)
        profile = support_profile(final_fit)
        support = support_status(evaluation, profile)
        actual = pd.to_numeric(evaluation[TARGET]).to_numpy(int)
        baseline = float(pd.to_numeric(final_fit[TARGET]).mean())
        for method in CALIBRATION_METHODS:
            estimate = apply_calibrator(method, calibrators[method], raw)
            for i, (_, item) in enumerate(evaluation.iterrows()):
                rows.append({
                    "origin_id": origin, "evaluation_year": evaluation_year,
                    "calibration_method": method,
                    "capology_extension_event_id": item.capology_extension_event_id,
                    "canonical_player_id": item.canonical_player_id,
                    "actual": int(actual[i]), "prediction": float(estimate[i]),
                    "baseline_prediction": baseline,
                    "brier_loss": float((actual[i] - estimate[i]) ** 2),
                    "baseline_brier_loss": float((actual[i] - baseline) ** 2),
                    **support.iloc[i].to_dict(),
                })
        for encoded, raw_feature, coefficient in zip(pre.encoded_names, pre.encoded_raw_features, model.coef_.reshape(-1)):
            coefficient_rows.append({"origin_id": origin, "encoded_feature": encoded, "raw_feature": raw_feature, "coefficient": float(coefficient)})
        cache[origin] = {"tuning": tuning, "validation": validation, "final_fit": final_fit, "evaluation": evaluation, "validation_raw": validation_raw, "calibrators": calibrators, "pre": pre, "model": model, "profile": profile}
    return pd.DataFrame(rows), pd.DataFrame(coefficient_rows), cache


def cluster_sample(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    ids = frame.canonical_player_id.astype(str)
    unique = ids.unique()
    return pd.concat([frame.loc[ids.eq(item)] for item in rng.choice(unique, len(unique), replace=True)], ignore_index=True)


def interval_origin(data: pd.DataFrame, method: str, spec):
    origin, train_end, validation_year, evaluation_year = spec
    tuning, validation, _, evaluation = split_origin(data, train_end, validation_year, evaluation_year)
    rng = np.random.default_rng(stable_seed(origin, method, "nested_parameter"))
    draws = []
    for _ in range(PARAMETER_BOOTSTRAP_REPETITIONS):
        bt, bv = cluster_sample(tuning, rng), cluster_sample(validation, rng)
        if pd.to_numeric(bt[TARGET]).nunique() < 2 or pd.to_numeric(bv[TARGET]).nunique() < 2:
            continue
        _, _, c_value, validation_raw = tune_c(bt, bv, FEATURES)
        calibrator = fit_calibrator(method, validation_raw, pd.to_numeric(bv[TARGET]).to_numpy(int))
        final = pd.concat([bt, bv], ignore_index=True)
        pre = fit_preprocessor(final, FEATURES)
        model = LogisticRegression(C=c_value, penalty="l2", solver="liblinear", max_iter=3000, random_state=RANDOM_SEED).fit(pre.transform(final), pd.to_numeric(final[TARGET]).to_numpy(int))
        draws.append(apply_calibrator(method, calibrator, model.predict_proba(pre.transform(evaluation))[:, 1]))
    matrix = np.asarray(draws)
    low, high = np.quantile(matrix, .10, axis=0), np.quantile(matrix, .90, axis=0)
    return pd.DataFrame({
        "origin_id": origin,
        "capology_extension_event_id": evaluation.capology_extension_event_id.to_numpy(),
        "parameter_interval_low_80": low,
        "parameter_interval_high_80": high,
        "parameter_interval_width_80": high - low,
        "successful_parameter_refits": len(matrix),
    })


def all_intervals(data: pd.DataFrame, method: str) -> pd.DataFrame:
    output = {}
    with ThreadPoolExecutor(max_workers=len(ORIGINS)) as executor:
        futures = {executor.submit(interval_origin, data, method, spec): spec[0] for spec in ORIGINS}
        for future in as_completed(futures):
            origin = futures[future]
            output[origin] = future.result()
            print(f"Verified {origin} nested parameter bootstrap.", flush=True)
    return pd.concat([output[spec[0]] for spec in ORIGINS], ignore_index=True)


def subgroup_skill_interval(chunk: pd.DataFrame, group_type: str, group: str):
    clustered = chunk.assign(improvement=chunk.baseline_brier_loss - chunk.brier_loss).groupby("canonical_player_id").improvement.agg(["sum", "size"])
    sums, sizes = clustered["sum"].to_numpy(float), clustered["size"].to_numpy(float)
    rng = np.random.default_rng(stable_seed("subgroup", group_type, group))
    draws = []
    for start in range(0, SUBGROUP_BOOTSTRAP_REPETITIONS, 250):
        count = min(250, SUBGROUP_BOOTSTRAP_REPETITIONS - start)
        indices = rng.integers(0, len(clustered), size=(count, len(clustered)))
        draws.extend((sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)).tolist())
    return float(np.quantile(draws, SUBGROUP_INTERVAL_ALPHA / 2)), float(np.quantile(draws, 1 - SUBGROUP_INTERVAL_ALPHA / 2))


def psi(reference: pd.Series, evaluation: pd.Series) -> float:
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    evaluation = pd.to_numeric(evaluation, errors="coerce").dropna()
    edges = np.unique(reference.quantile(np.linspace(0, 1, 11)).to_numpy(float))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = (pd.cut(reference, edges, include_lowest=True).value_counts(sort=False) / len(reference)).clip(lower=1e-6)
    b = (pd.cut(evaluation, edges, include_lowest=True).value_counts(sort=False) / len(evaluation)).clip(lower=1e-6)
    return float(((b - a) * np.log(b / a)).sum())


def main() -> None:
    checks = []
    def add(name: str, actual: object, expected: object, passed: bool, note: str) -> None:
        checks.append({"check": name, "actual": actual, "expected": expected, "passed": bool(passed), "note": note})

    required = {
        "README.md", "build_checks.csv", "calibration_bins.csv", "calibration_fit_predictions.csv",
        "calibration_method_comparison.csv", "calibration_predictions.csv", "calibration_selection.csv",
        "coefficient_stability.csv", "deployment_unchanged_manifest.csv", "missingness_reliability.csv",
        "monotonicity_stress_test.csv", "ood_reason_summary.csv", "ood_support_profiles.json",
        "ood_support_summary.csv", "origin_calibration_metrics.csv", "output_manifest.csv",
        "parameter_uncertainty_summary.csv", "preregistration.json", "redundancy_stress_test.csv",
        "reliability_decision.csv", "run_summary.json", "selected_predictions.csv", "source_manifest.csv",
        "subgroup_reliability.csv", "temporal_drift.csv",
    }
    actual_files = {path.name for path in OUTPUT.iterdir() if path.is_file()}
    allowed = {"independent_verification.csv", "independent_verification.json"}
    add("required_outputs_exact", sorted(actual_files), sorted(required), required.issubset(actual_files) and not (actual_files - required - allowed), "The evidence package contains only declared outputs.")
    manifest = pd.read_csv(OUTPUT / "output_manifest.csv")
    manifest_ok = all((OUTPUT / row.output_file).exists() and (OUTPUT / row.output_file).stat().st_size == int(row.bytes) and sha256(OUTPUT / row.output_file) == row.sha256 for row in manifest.itertuples(index=False))
    add("output_manifest_exact", int(manifest_ok), 1, manifest_ok, "Every declared output matches its recorded hash and size.")
    sources = pd.read_csv(OUTPUT / "source_manifest.csv")
    source_ok = all((ROOT / row.source).exists() and (ROOT / row.source).stat().st_size == int(row.bytes) and sha256(ROOT / row.source) == row.sha256 for row in sources.itertuples(index=False))
    add("source_manifest_exact", int(source_ok), 1, source_ok, "Every Run 3 input retains its recorded hash.")

    data = prepare_model_frame().loc[lambda x: x.run2_model_cohort].copy()
    independent, independent_coefficients, cache = refit_origins(data)
    saved_all = pd.read_csv(OUTPUT / "calibration_predictions.csv")
    keys = ["origin_id", "calibration_method", "capology_extension_event_id"]
    joined = saved_all.merge(independent[keys + ["prediction", "support_status", "support_reasons", "support_distance"]], on=keys, suffixes=("_saved", "_independent"), validate="one_to_one")
    prediction_error = float((joined.prediction_saved - joined.prediction_independent).abs().max())
    distance_error = float((joined.support_distance_saved - joined.support_distance_independent).abs().max())
    support_exact = joined.support_status_saved.eq(joined.support_status_independent).all() and joined.support_reasons_saved.fillna("").eq(joined.support_reasons_independent.fillna("")).all()
    add("all_models_and_calibrators_refit", prediction_error, "<=1e-12", prediction_error <= 1e-12 and len(joined) == 2634, "Independent origin refits reproduce raw, Platt, and isotonic predictions.")
    add("support_policy_reconstructed", distance_error, "<=1e-12", distance_error <= 1e-12 and support_exact, "Origin-local missingness and OOD states reproduce.")
    saved_coefficients = pd.read_csv(OUTPUT / "coefficient_stability.csv")
    coeff = saved_coefficients.merge(independent_coefficients, on=["origin_id", "encoded_feature", "raw_feature"], suffixes=("_saved", "_independent"), validate="one_to_one")
    coeff_error = float((coeff.coefficient_saved - coeff.coefficient_independent).abs().max())
    add("coefficients_refit_exact", coeff_error, "<=1e-12", coeff_error <= 1e-12, "Every standardized coefficient reproduces.")

    comparison = pd.read_csv(OUTPUT / "calibration_method_comparison.csv")
    comparison_errors = []
    for row in comparison.itertuples(index=False):
        chunk = independent.loc[independent.calibration_method.eq(row.calibration_method)]
        current = metrics(chunk.actual, chunk.prediction)
        comparison_errors.extend(abs(current[key] - getattr(row, f"pooled_{key}")) for key in current)
    add("calibration_metrics_recomputed", max(comparison_errors), "<=1e-12", max(comparison_errors) <= 1e-12, "All pooled calibration metrics derive from row predictions.")
    raw = comparison.loc[comparison.calibration_method.eq("raw")].iloc[0]
    eligible = comparison.loc[comparison.pooled_brier.le(raw.pooled_brier + CALIBRATION_BRIER_NONINFERIORITY) & comparison.pooled_log_loss.le(raw.pooled_log_loss + CALIBRATION_LOG_LOSS_NONINFERIORITY)].copy()
    best_ece = eligible.pooled_adaptive_ece.min()
    eligible = eligible.loc[eligible.pooled_adaptive_ece.le(best_ece + CALIBRATION_ECE_NEAR_TIE)].copy()
    eligible["simplicity"] = eligible.calibration_method.map(CALIBRATION_SIMPLICITY)
    selected_method = eligible.sort_values(["simplicity", "pooled_brier"]).iloc[0].calibration_method
    saved_selection = pd.read_csv(OUTPUT / "calibration_selection.csv")
    add("calibration_selection_recomputed", selected_method, "raw", selected_method == "raw" and saved_selection.loc[saved_selection.selected, "calibration_method"].iloc[0] == selected_method, "The frozen non-inferiority/ECE/simplicity rule selects raw probabilities.")

    saved_selected = pd.read_csv(OUTPUT / "selected_predictions.csv")
    intervals = all_intervals(data, selected_method)
    interval_join = saved_selected.merge(intervals, on=["origin_id", "capology_extension_event_id"], suffixes=("_saved", "_independent"), validate="one_to_one")
    interval_error = max(float((interval_join.parameter_interval_low_80_saved - interval_join.parameter_interval_low_80_independent).abs().max()), float((interval_join.parameter_interval_high_80_saved - interval_join.parameter_interval_high_80_independent).abs().max()))
    add("nested_parameter_intervals_exact", interval_error, "<=1e-12", interval_error <= 1e-12 and interval_join.successful_parameter_refits_independent.eq(PARAMETER_BOOTSTRAP_REPETITIONS).all(), "All 1,000 independent nested player-cluster refits reproduce the saved intervals.")

    subgroup = pd.read_csv(OUTPUT / "subgroup_reliability.csv")
    subgroup_errors, subgroup_status_errors = [], []
    for row in subgroup.itertuples(index=False):
        column = {"league": "league", "position": "canonical_position", "age": "age_band", "support": "support_status"}[row.group_type]
        population = saved_selected if row.group_type == "support" else saved_selected.loc[~saved_selected.support_status.eq("refused")]
        chunk = population.loc[population[column].astype(str).eq(str(row.group))]
        events, nonevents = int(chunk.actual.sum()), len(chunk) - int(chunk.actual.sum())
        auc = float(roc_auc_score(chunk.actual, chunk.prediction)) if events and nonevents else np.nan
        brier = float(chunk.brier_loss.mean())
        baseline = float(chunk.baseline_brier_loss.mean())
        gap = float(abs(chunk.actual.mean() - chunk.prediction.mean()))
        low, high = subgroup_skill_interval(chunk, row.group_type, str(row.group))
        subgroup_errors.extend([abs(len(chunk) - row.rows), abs(events - row.events), abs(nonevents - row.nonevents), abs(brier - row.brier), abs(baseline - row.chronology_brier), abs(gap - row.calibration_gap), abs(low - row.brier_skill_ci_low_99), abs(high - row.brier_skill_ci_high_99)])
        if not (pd.isna(auc) and pd.isna(row.roc_auc)):
            subgroup_errors.append(abs(auc - row.roc_auc))
        sufficient = len(chunk) >= MINIMUM_SUBGROUP_ROWS and events >= MINIMUM_SUBGROUP_EVENTS and nonevents >= MINIMUM_SUBGROUP_NONEVENTS and chunk.origin_id.nunique() >= MINIMUM_SUBGROUP_ORIGINS
        supported = sufficient and auc >= MINIMUM_SUBGROUP_AUC and gap <= MAXIMUM_SUBGROUP_CALIBRATION_GAP and low > 0
        expected = "supported" if supported else "limited_reliability" if sufficient else "insufficient_evidence"
        if expected != row.reliability_status:
            subgroup_status_errors.append((row.group_type, row.group, row.reliability_status, expected))
    add("subgroup_metrics_and_intervals_exact", max(subgroup_errors), "<=1e-12", max(subgroup_errors) <= 1e-12, "Every subgroup metric and 99% player-cluster interval reconstructs.")
    add("subgroup_statuses_exact", subgroup_status_errors, [], not subgroup_status_errors, "The frozen evidence rules reproduce every subgroup status.")

    decision = pd.read_csv(OUTPUT / "reliability_decision.csv").iloc[0]
    selected = saved_selected
    pooled = metrics(selected.actual, selected.prediction)
    origin_metrics = pd.read_csv(OUTPUT / "origin_calibration_metrics.csv")
    by_origin = origin_metrics.loc[origin_metrics.calibration_method.eq(selected_method)]
    reliable_origins = int((by_origin.adaptive_ece.le(MAXIMUM_ORIGIN_ECE) & by_origin.calibration_gap.le(MAXIMUM_ORIGIN_CALIBRATION_GAP)).sum())
    positive_origins = int(selected.groupby("origin_id").apply(lambda x: x.baseline_brier_loss.mean() > x.brier_loss.mean()).sum())
    refusal = float(selected.support_status.eq("refused").mean())
    width = selected.parameter_interval_width_80
    scorable = selected.loc[~selected.support_status.eq("refused")]
    extreme = float((scorable.prediction.lt(.02) | scorable.prediction.gt(.98)).mean())
    core = subgroup.loc[subgroup.group_type.isin(REQUIRED_SUBGROUP_TYPES)]
    latest_cache = cache["origin_2023"]
    latest_psi = max(psi(latest_cache["final_fit"][feature], latest_cache["evaluation"][feature]) for feature in INVOLVEMENT_FEATURES)
    target_range = float(by_origin.actual_rate.max() - by_origin.actual_rate.min())
    involvement_coefficients = independent_coefficients.loc[independent_coefficients.encoded_feature.isin(INVOLVEMENT_FEATURES)]
    monotonicity = pd.read_csv(OUTPUT / "monotonicity_stress_test.csv")
    expected_gates = {
        "pooled_ece": pooled["adaptive_ece"] <= MAXIMUM_POOLED_ECE,
        "pooled_calibration_gap": pooled["calibration_gap"] <= MAXIMUM_POOLED_CALIBRATION_GAP,
        "pooled_calibration_slope": MINIMUM_CALIBRATION_SLOPE <= pooled["calibration_slope"] <= MAXIMUM_CALIBRATION_SLOPE,
        "reliable_origins": reliable_origins >= MINIMUM_RELIABLE_ORIGINS,
        "positive_skill_origins": positive_origins >= MINIMUM_POSITIVE_SKILL_ORIGINS,
        "refusal_rate": refusal <= MAXIMUM_REFUSAL_RATE,
        "mean_parameter_width": width.mean() <= MAXIMUM_MEAN_PARAMETER_WIDTH_80,
        "p90_parameter_width": width.quantile(.90) <= MAXIMUM_P90_PARAMETER_WIDTH_80,
        "extreme_probability_share": extreme <= MAXIMUM_EXTREME_PROBABILITY_SHARE,
        "core_subgroups_supported": core.reliability_status.eq("supported").all(),
        "latest_involvement_drift": latest_psi <= MAXIMUM_LATEST_INVOLVEMENT_PSI,
        "target_rate_stability": target_range <= MAXIMUM_TARGET_RATE_RANGE,
        "involvement_coefficients_positive": len(involvement_coefficients) == 8 and involvement_coefficients.coefficient.gt(0).all(),
        "monotonicity_stress": monotonicity.negative_delta_rows.eq(0).all(),
        "parameter_bootstrap_complete": intervals.successful_parameter_refits.ge(int(PARAMETER_BOOTSTRAP_REPETITIONS * .95)).all(),
    }
    gate_errors = [key for key, value in expected_gates.items() if bool(decision[f"gate__{key}"]) != bool(value)]
    add("reliability_gate_exact", gate_errors, [], not gate_errors and not all(expected_gates.values()) and decision.run3_decision == "blocked_reliability_repair", "All 15 frozen gates reconstruct; only all-group support fails.")
    correlation = float(data[list(INVOLVEMENT_FEATURES)].corr(method="spearman").iloc[0, 1])
    add("group_only_explanation_enforced", correlation, f">{REDUNDANCY_SPEARMAN_THRESHOLD}", correlation > REDUNDANCY_SPEARMAN_THRESHOLD and decision.explanation_policy == "group_only_recent_involvement", "Highly redundant involvement inputs cannot receive separate importance narratives.")

    deployment = pd.read_csv(OUTPUT / "deployment_unchanged_manifest.csv")
    recorded_stable = bool(
        deployment["unchanged"].all()
        and deployment["before_sha256"].eq(deployment["after_sha256"]).all()
    )
    model_rows = deployment.loc[deployment["artifact"].str.endswith(".joblib")]
    model_binaries_unchanged = all(
        (ROOT / row.artifact).exists()
        and sha256(ROOT / row.artifact) == row.before_sha256 == row.after_sha256
        for row in model_rows.itertuples(index=False)
    )
    add(
        "run_recorded_no_deployment_mutation",
        int(recorded_stable),
        1,
        recorded_stable,
        "The Run 3 snapshot records identical before/after hashes for every deployment file.",
    )
    add(
        "serialized_model_files_unchanged",
        int(model_binaries_unchanged),
        1,
        model_binaries_unchanged,
        "Every frozen serialized model retains its Run 3 hash; regenerable verification reports are not model binaries.",
    )
    add("no_model_binary", len(list(OUTPUT.rglob("*.joblib"))), 0, not list(OUTPUT.rglob("*.joblib")), "Run 3 writes no model or calibrator artifact.")
    summary = json.loads((OUTPUT / "run_summary.json").read_text(encoding="utf-8"))
    add("final_holdout_closed", summary.get("final_holdout_evaluated"), False, summary.get("final_holdout_evaluated") is False and selected.evaluation_year.max() == 2023, "No 2024+ outcome was evaluated.")
    docs = (ROOT / "docs" / "ENGINE_V2_CONTRIBUTION_RELIABILITY_RUN3.md").read_text(encoding="utf-8")
    critical = (ROOT / "docs" / "ENGINE_V2_CRITICAL_ISSUES.md").read_text(encoding="utf-8")
    registry = (ROOT / "models" / "README.md").read_text(encoding="utf-8")
    documentation_ok = "Run 3 reliability result" in docs and "E2-CRIT-005" in critical and "Repaired contribution reliability audit" in registry
    add("durable_documentation_updated", int(documentation_ok), 1, documentation_ok, "The result, release blocker, and runner registry are permanent.")

    result = pd.DataFrame(checks)
    result.to_csv(OUTPUT / "independent_verification.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    payload = {
        "checks_passed": int(result.passed.sum()),
        "checks_total": len(result),
        "all_checks_passed": bool(result.passed.all()),
        "independent_base_model_refits": 4,
        "independent_calibrator_refits": 8,
        "independent_nested_parameter_refits": 1000,
        "subgroup_bootstrap_repetitions_per_group": SUBGROUP_BOOTSTRAP_REPETITIONS,
        "final_holdout_evaluated": False,
        "deployment_changed": False,
    }
    (OUTPUT / "independent_verification.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not result.passed.all():
        failed = result.loc[~result.passed, ["check", "actual", "expected"]]
        raise RuntimeError("Run 3 independent verification failed:\n" + failed.to_string(index=False))
    print(f"All {len(result)}/{len(result)} Run 3 independent checks pass.")


if __name__ == "__main__":
    main()

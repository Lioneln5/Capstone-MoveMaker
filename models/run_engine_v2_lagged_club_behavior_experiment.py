"""Test time-safe club movement behavior on decomposed Engine V2 outcomes.

Development-only rolling-origin experiment.  The 2024+ final holdout remains
sealed; no model binary, API file, or deployment artifact is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "models")]

from models.engine_v2.feature_contract import V2_CORE_FEATURES  # noqa: E402
from models.engine_v2.movement_target_contract import (  # noqa: E402
    CONTRACT_VERSION,
    MODELED_STATE_ORDER,
    add_movement_targets,
)
from mixed_type_preprocessor import fit_preprocessor  # noqa: E402
import run_engine_v2_calibration_reliability as phase3  # noqa: E402
import run_engine_v2_candidate_contract as phase4  # noqa: E402
import run_engine_v2_feature_specification as phase2  # noqa: E402
from scripts.build_engine_v2_lagged_club_behavior import (  # noqa: E402
    EXTENSION_FEATURES,
    GENERAL_FEATURES,
    MODEL_FEATURES,
    ROLE_FEATURES,
)


DEFAULT_OUTPUT = ROOT / "Data" / "processed" / "engine_v2_lagged_club_behavior_experiment"
PROFILE_DIR = ROOT / "Data" / "processed" / "engine_v2_lagged_club_behavior"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"
AGE_CURVE = "age_distance_from_27_squared"
BASE_FEATURES = list(V2_CORE_FEATURES) + [AGE_CURVE]
STATE_ORDER = list(MODELED_STATE_ORDER)
RANDOM_SEED = 20260901
C_GRID = (0.01, 0.1, 1.0, 10.0)
BOOTSTRAPS = 5000
FAMILY_CONFIDENCE = 1 - 0.05 / 4

# Frozen before outcome evaluation.  C3 deliberately excludes the two
# position-specific prior-extension features so it can isolate same-club
# historical decisions without making broad-position availability a condition.
EXTENSION_CORE = EXTENSION_FEATURES[:4]
VARIANT_BLOCKS = {
    "C1_general_club_movement": GENERAL_FEATURES,
    "C2_role_conditioned_movement": ROLE_FEATURES,
    "C3_mature_extension_history": EXTENSION_CORE,
    "C4_combined_behavior": MODEL_FEATURES,
}
BINARY_TARGETS = {
    "temporary_first_24m": "target_temporary_first_24m",
    "permanent_first_24m": "target_permanent_first_24m",
    "permanent_relationship_ended_24m": "target_permanent_relationship_ended_24m",
    "strict_meaningful_stay_24m": "target_strict_meaningful_stay_24m",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_seed(*parts: str) -> int:
    token = "|".join(parts).encode()
    return (RANDOM_SEED + int(hashlib.sha256(token).hexdigest()[:8], 16)) % (2**32 - 1)


def safe_auc(actual: pd.Series | np.ndarray, prediction: pd.Series | np.ndarray) -> float:
    actual = pd.Series(actual)
    return float(roc_auc_score(actual, prediction)) if actual.nunique() == 2 else np.nan


def adaptive_ece(actual: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    frame = pd.DataFrame({"actual": actual, "prediction": probability}).sort_values("prediction")
    groups = np.array_split(np.arange(len(frame)), min(bins, len(frame)))
    return float(sum(len(group) * abs(frame.iloc[group].actual.mean() - frame.iloc[group].prediction.mean()) for group in groups) / len(frame))


def load_data() -> pd.DataFrame:
    data = phase4.add_age_curve(phase2.load_endpoint_frames()["any_outbound_24m"])
    data = add_movement_targets(data)
    profiles = pd.read_csv(
        PROFILE_DIR / "lagged_club_behavior_features.csv",
        usecols=["capology_extension_event_id"] + MODEL_FEATURES,
        low_memory=False,
    )
    return data.merge(profiles, on="capology_extension_event_id", how="left", validate="one_to_one")


def target_frame(data: pd.DataFrame, endpoint: str, target: str) -> pd.DataFrame:
    frame = data.loc[data[target].notna()].copy()
    if endpoint in {"temporary_first_24m", "permanent_first_24m"}:
        frame = frame.loc[frame["movement_state_3"].notna()].copy()
    return frame


def fit_binary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[object, LogisticRegression, float]:
    tune_pre = fit_preprocessor(train, features)
    x_train = tune_pre.transform(train)
    x_validation = tune_pre.transform(validation)
    y_train = pd.to_numeric(train[target]).to_numpy(int)
    y_validation = pd.to_numeric(validation[target]).to_numpy(int)
    candidates = []
    for c_value in C_GRID:
        model = LogisticRegression(C=c_value, penalty="l2", solver="liblinear", max_iter=4000, random_state=RANDOM_SEED)
        model.fit(x_train, y_train)
        probability = np.clip(model.predict_proba(x_validation)[:, 1], 1e-6, 1 - 1e-6)
        candidates.append((float(np.square(y_validation - probability).mean()), c_value))
    _, selected_c = min(candidates, key=lambda item: item[0])
    fit = pd.concat([train, validation], ignore_index=True)
    pre = fit_preprocessor(fit, features)
    model = LogisticRegression(C=selected_c, penalty="l2", solver="liblinear", max_iter=4000, random_state=RANDOM_SEED)
    model.fit(pre.transform(fit), pd.to_numeric(fit[target]).to_numpy(int))
    return pre, model, float(selected_c)


def fit_multinomial(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
) -> tuple[object, LogisticRegression, float]:
    tune_pre = fit_preprocessor(train, features)
    x_train = tune_pre.transform(train)
    x_validation = tune_pre.transform(validation)
    y_train = train["movement_state_3"].astype(str).to_numpy()
    y_validation = validation["movement_state_3"].astype(str).to_numpy()
    onehot = np.column_stack([y_validation == state for state in STATE_ORDER]).astype(float)
    candidates = []
    for c_value in C_GRID:
        model = LogisticRegression(C=c_value, penalty="l2", solver="lbfgs", max_iter=6000, random_state=RANDOM_SEED)
        model.fit(x_train, y_train)
        raw = np.clip(model.predict_proba(x_validation), 1e-6, 1 - 1e-6)
        index = {label: idx for idx, label in enumerate(model.classes_)}
        probability = np.column_stack([raw[:, index[state]] for state in STATE_ORDER])
        probability /= probability.sum(axis=1, keepdims=True)
        candidates.append((float(np.square(onehot - probability).sum(axis=1).mean()), c_value))
    _, selected_c = min(candidates, key=lambda item: item[0])
    fit = pd.concat([train, validation], ignore_index=True)
    pre = fit_preprocessor(fit, features)
    model = LogisticRegression(C=selected_c, penalty="l2", solver="lbfgs", max_iter=6000, random_state=RANDOM_SEED)
    model.fit(pre.transform(fit), fit["movement_state_3"].astype(str).to_numpy())
    return pre, model, float(selected_c)


def ordered_multinomial_probability(model: LogisticRegression, matrix: np.ndarray) -> np.ndarray:
    raw = np.clip(model.predict_proba(matrix), 1e-6, 1 - 1e-6)
    index = {label: idx for idx, label in enumerate(model.classes_)}
    probability = np.column_stack([raw[:, index[state]] for state in STATE_ORDER])
    return probability / probability.sum(axis=1, keepdims=True)


def coefficient_rows(
    endpoint: str,
    variant: str,
    origin: str,
    pre: object,
    model: LogisticRegression,
    added_features: list[str],
) -> list[dict[str, object]]:
    rows = []
    class_labels = ["event"] if model.coef_.shape[0] == 1 else list(model.classes_)
    for class_index, class_label in enumerate(class_labels):
        coefficients = model.coef_[class_index]
        for encoded, raw, coefficient in zip(pre.encoded_names, pre.encoded_raw_features, coefficients):
            if raw in added_features:
                rows.append({
                    "endpoint": endpoint, "variant": variant, "origin_id": origin,
                    "class_label": class_label, "raw_feature": raw,
                    "encoded_feature": encoded, "standardized_log_odds_coefficient": float(coefficient),
                })
    return rows


def run_binary(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions, origin_metrics, coverage, coefficients = [], [], [], []
    for endpoint, target in BINARY_TARGETS.items():
        endpoint_data = target_frame(data, endpoint, target)
        for variant, added in VARIANT_BLOCKS.items():
            features = BASE_FEATURES + added
            for origin, train_end, validation_year, evaluation_year in phase2.ORIGINS:
                train_all = endpoint_data.loc[endpoint_data["signed_year"].le(train_end)].copy()
                validation_all = endpoint_data.loc[endpoint_data["signed_year"].eq(validation_year)].copy()
                evaluation_all = endpoint_data.loc[endpoint_data["signed_year"].eq(evaluation_year)].sort_values("capology_extension_event_id").copy()
                train = train_all.loc[train_all[added].notna().all(axis=1)].copy()
                validation = validation_all.loc[validation_all[added].notna().all(axis=1)].copy()
                evaluation = evaluation_all.loc[evaluation_all[added].notna().all(axis=1)].copy()
                for split_name, split in [("train", train), ("validation", validation), ("evaluation", evaluation)]:
                    if len(split) < 30 or split[target].nunique() < 2:
                        raise RuntimeError(f"Insufficient {split_name} split for {endpoint}/{variant}/{origin}")

                base_pre, base_model, base_c = fit_binary(train, validation, BASE_FEATURES, target)
                cand_pre, cand_model, cand_c = fit_binary(train, validation, features, target)
                base_probability = np.clip(base_model.predict_proba(base_pre.transform(evaluation))[:, 1], 1e-6, 1 - 1e-6)
                candidate_probability = np.clip(cand_model.predict_proba(cand_pre.transform(evaluation))[:, 1], 1e-6, 1 - 1e-6)
                actual = pd.to_numeric(evaluation[target]).to_numpy(int)
                fit = pd.concat([train, validation], ignore_index=True)
                chronology = float(pd.to_numeric(fit[target]).mean())
                profile, _ = phase3.support_profile(fit, features)
                support = phase3.assess_support(evaluation, features, profile).reset_index(drop=True)
                refused = int(support["support_status"].eq("refused").sum())
                unavailable = len(evaluation_all) - len(evaluation)
                coverage.append({
                    "model_type": "binary", "endpoint": endpoint, "variant": variant,
                    "origin_id": origin, "evaluation_year": evaluation_year,
                    "evaluation_rows_total": len(evaluation_all), "feature_eligible_rows": len(evaluation),
                    "feature_unavailable_rows": unavailable, "support_refused_rows": refused,
                    "operational_refused_rows": unavailable + refused,
                    "operational_refusal_rate": (unavailable + refused) / len(evaluation_all),
                })
                delta = np.square(actual - base_probability) - np.square(actual - candidate_probability)
                supported = ~support["support_status"].eq("refused").to_numpy()
                origin_metrics.append({
                    "endpoint": endpoint, "variant": variant, "origin_id": origin,
                    "evaluation_year": evaluation_year, "train_rows_matched": len(train),
                    "validation_rows_matched": len(validation), "evaluation_rows_matched": len(evaluation),
                    "baseline_selected_C": base_c, "candidate_selected_C": cand_c,
                    "baseline_brier": float(np.square(actual - base_probability).mean()),
                    "candidate_brier": float(np.square(actual - candidate_probability).mean()),
                    "brier_improvement": float(delta.mean()),
                    "supported_rows": int(supported.sum()),
                    "supported_brier_improvement": float(delta[supported].mean()) if supported.any() else np.nan,
                    "candidate_auc": safe_auc(actual, candidate_probability),
                    "candidate_ece": adaptive_ece(actual, candidate_probability),
                })
                coefficients.extend(coefficient_rows(endpoint, variant, origin, cand_pre, cand_model, added))
                evaluation = evaluation.reset_index(drop=True)
                for index, row in evaluation.iterrows():
                    predictions.append({
                        "endpoint": endpoint, "variant": variant, "origin_id": origin,
                        "evaluation_year": evaluation_year,
                        "capology_extension_event_id": row["capology_extension_event_id"],
                        "canonical_player_id": row["canonical_player_id"],
                        "league": row["league"], "canonical_position": row["canonical_position"],
                        "actual": int(actual[index]), "candidate_prediction": float(candidate_probability[index]),
                        "matched_baseline_prediction": float(base_probability[index]),
                        "chronology_prediction": chronology,
                        "candidate_brier_loss": float((actual[index] - candidate_probability[index]) ** 2),
                        "matched_baseline_brier_loss": float((actual[index] - base_probability[index]) ** 2),
                        "loss_improvement": float(delta[index]),
                        "support_status": support.iloc[index]["support_status"],
                        "support_reasons": support.iloc[index]["support_reasons"],
                    })
    return pd.DataFrame(predictions), pd.DataFrame(origin_metrics), pd.DataFrame(coverage), pd.DataFrame(coefficients)


def run_multinomial(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    modelable = data.loc[data["movement_state_3"].notna()].copy()
    predictions, origin_metrics, coverage, coefficients = [], [], [], []
    for variant, added in VARIANT_BLOCKS.items():
        features = BASE_FEATURES + added
        for origin, train_end, validation_year, evaluation_year in phase2.ORIGINS:
            train_all = modelable.loc[modelable["signed_year"].le(train_end)].copy()
            validation_all = modelable.loc[modelable["signed_year"].eq(validation_year)].copy()
            evaluation_all = modelable.loc[modelable["signed_year"].eq(evaluation_year)].sort_values("capology_extension_event_id").copy()
            train = train_all.loc[train_all[added].notna().all(axis=1)].copy()
            validation = validation_all.loc[validation_all[added].notna().all(axis=1)].copy()
            evaluation = evaluation_all.loc[evaluation_all[added].notna().all(axis=1)].copy()
            if any(split["movement_state_3"].nunique() != 3 or len(split) < 30 for split in [train, validation, evaluation]):
                raise RuntimeError(f"Insufficient multinomial split for {variant}/{origin}")

            base_pre, base_model, base_c = fit_multinomial(train, validation, BASE_FEATURES)
            cand_pre, cand_model, cand_c = fit_multinomial(train, validation, features)
            base_probability = ordered_multinomial_probability(base_model, base_pre.transform(evaluation))
            candidate_probability = ordered_multinomial_probability(cand_model, cand_pre.transform(evaluation))
            actual_state = evaluation["movement_state_3"].astype(str).to_numpy()
            onehot = np.column_stack([actual_state == state for state in STATE_ORDER]).astype(float)
            fit = pd.concat([train, validation], ignore_index=True)
            prevalence = fit["movement_state_3"].value_counts(normalize=True)
            chronology = np.array([prevalence.get(state, 0.0) for state in STATE_ORDER], dtype=float)
            chronology /= chronology.sum()
            baseline_loss = np.square(onehot - base_probability).sum(axis=1)
            candidate_loss = np.square(onehot - candidate_probability).sum(axis=1)
            delta = baseline_loss - candidate_loss
            profile, _ = phase3.support_profile(fit, features)
            support = phase3.assess_support(evaluation, features, profile).reset_index(drop=True)
            refused = int(support["support_status"].eq("refused").sum())
            unavailable = len(evaluation_all) - len(evaluation)
            coverage.append({
                "model_type": "multinomial", "endpoint": "movement_state_3", "variant": variant,
                "origin_id": origin, "evaluation_year": evaluation_year,
                "evaluation_rows_total": len(evaluation_all), "feature_eligible_rows": len(evaluation),
                "feature_unavailable_rows": unavailable, "support_refused_rows": refused,
                "operational_refused_rows": unavailable + refused,
                "operational_refusal_rate": (unavailable + refused) / len(evaluation_all),
            })
            actual_index = np.array([STATE_ORDER.index(state) for state in actual_state], dtype=int)
            supported = ~support["support_status"].eq("refused").to_numpy()
            origin_metrics.append({
                "variant": variant, "origin_id": origin, "evaluation_year": evaluation_year,
                "train_rows_matched": len(train), "validation_rows_matched": len(validation),
                "evaluation_rows_matched": len(evaluation), "baseline_selected_C": base_c,
                "candidate_selected_C": cand_c, "baseline_multiclass_brier": float(baseline_loss.mean()),
                "candidate_multiclass_brier": float(candidate_loss.mean()),
                "brier_improvement": float(delta.mean()), "supported_rows": int(supported.sum()),
                "supported_brier_improvement": float(delta[supported].mean()) if supported.any() else np.nan,
                "candidate_log_loss": float(-np.log(candidate_probability[np.arange(len(actual_index)), actual_index]).mean()),
                "baseline_log_loss": float(-np.log(base_probability[np.arange(len(actual_index)), actual_index]).mean()),
                "candidate_macro_ovr_auc": float(roc_auc_score(onehot, candidate_probability, average="macro", multi_class="ovr")),
            })
            coefficients.extend(coefficient_rows("movement_state_3", variant, origin, cand_pre, cand_model, added))
            evaluation = evaluation.reset_index(drop=True)
            for index, row in evaluation.iterrows():
                record = {
                    "variant": variant, "origin_id": origin, "evaluation_year": evaluation_year,
                    "capology_extension_event_id": row["capology_extension_event_id"],
                    "canonical_player_id": row["canonical_player_id"], "league": row["league"],
                    "canonical_position": row["canonical_position"], "actual_state": actual_state[index],
                    "candidate_multiclass_brier_loss": float(candidate_loss[index]),
                    "matched_baseline_multiclass_brier_loss": float(baseline_loss[index]),
                    "loss_improvement": float(delta[index]),
                    "support_status": support.iloc[index]["support_status"],
                    "support_reasons": support.iloc[index]["support_reasons"],
                }
                for class_index, state in enumerate(STATE_ORDER):
                    record[f"candidate_probability_{state}"] = float(candidate_probability[index, class_index])
                    record[f"matched_baseline_probability_{state}"] = float(base_probability[index, class_index])
                    record[f"chronology_probability_{state}"] = float(chronology[class_index])
                predictions.append(record)
    return pd.DataFrame(predictions), pd.DataFrame(origin_metrics), pd.DataFrame(coverage), pd.DataFrame(coefficients)


def cluster_interval(frame: pd.DataFrame, confidence: float, *labels: str) -> tuple[float, float]:
    clusters = frame.groupby("canonical_player_id")["loss_improvement"].agg(["sum", "size"])
    sums = clusters["sum"].to_numpy(float)
    sizes = clusters["size"].to_numpy(float)
    rng = np.random.default_rng(stable_seed(*labels, str(confidence)))
    draws = []
    for _ in range(BOOTSTRAPS // 100):
        index = rng.integers(0, len(clusters), size=(100, len(clusters)))
        draws.extend((sums[index].sum(axis=1) / sizes[index].sum(axis=1)).tolist())
    alpha = 1 - confidence
    return float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def summarize_binary(
    predictions: pd.DataFrame,
    origins: pd.DataFrame,
    coverage: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons, leagues = [], []
    for (endpoint, variant), frame in predictions.groupby(["endpoint", "variant"], sort=False):
        ci95 = cluster_interval(frame, 0.95, endpoint, variant)
        ci_adj = cluster_interval(frame, FAMILY_CONFIDENCE, endpoint, variant, "adjusted")
        origin = origins.loc[origins["endpoint"].eq(endpoint) & origins["variant"].eq(variant)]
        cover = coverage.loc[coverage["endpoint"].eq(endpoint) & coverage["variant"].eq(variant)]
        supported = frame.loc[~frame["support_status"].eq("refused")]
        improvement = float(frame["loss_improvement"].mean())
        supported_improvement = float(supported["loss_improvement"].mean()) if len(supported) else np.nan
        refusal = float(cover["operational_refused_rows"].sum() / cover["evaluation_rows_total"].sum())
        if ci_adj[0] > 0 and int(origin["brier_improvement"].gt(0).sum()) >= 3 and refusal <= 0.15 and supported_improvement > 0:
            evidence = "promising_for_combined_development_testing"
        elif ci_adj[1] < 0:
            evidence = "repeated_harm"
        else:
            evidence = "inconclusive"
        comparisons.append({
            "endpoint": endpoint, "variant": variant, "evaluation_rows_matched": len(frame),
            "feature_coverage_rate": float(cover["feature_eligible_rows"].sum() / cover["evaluation_rows_total"].sum()),
            "operational_refusal_rate": refusal,
            "matched_baseline_brier": float(frame["matched_baseline_brier_loss"].mean()),
            "candidate_brier": float(frame["candidate_brier_loss"].mean()),
            "pooled_brier_improvement": improvement, "supported_brier_improvement": supported_improvement,
            "ci_low_95": ci95[0], "ci_high_95": ci95[1],
            "ci_low_adjusted_98_75": ci_adj[0], "ci_high_adjusted_98_75": ci_adj[1],
            "origin_wins": int(origin["brier_improvement"].gt(0).sum()),
            "candidate_auc": safe_auc(frame["actual"], frame["candidate_prediction"]),
            "matched_baseline_auc": safe_auc(frame["actual"], frame["matched_baseline_prediction"]),
            "candidate_average_precision": float(average_precision_score(frame["actual"], frame["candidate_prediction"])),
            "candidate_ece": adaptive_ece(frame["actual"].to_numpy(), frame["candidate_prediction"].to_numpy()),
            "evidence_classification": evidence, "deployment_status": "not_deployed",
        })
        for league, chunk in frame.groupby("league", dropna=False):
            leagues.append({
                "endpoint": endpoint, "variant": variant, "league": str(league), "rows": len(chunk),
                "events": int(chunk["actual"].sum()),
                "brier_improvement": float(chunk["loss_improvement"].mean()),
                "candidate_auc": safe_auc(chunk["actual"], chunk["candidate_prediction"]),
                "matched_baseline_auc": safe_auc(chunk["actual"], chunk["matched_baseline_prediction"]),
            })
    return pd.DataFrame(comparisons), pd.DataFrame(leagues)


def summarize_multinomial(
    predictions: pd.DataFrame,
    origins: pd.DataFrame,
    coverage: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparisons, classes, leagues = [], [], []
    for variant, frame in predictions.groupby("variant", sort=False):
        ci95 = cluster_interval(frame, 0.95, "multinomial", variant)
        ci_adj = cluster_interval(frame, FAMILY_CONFIDENCE, "multinomial", variant, "adjusted")
        origin = origins.loc[origins["variant"].eq(variant)]
        cover = coverage.loc[coverage["variant"].eq(variant)]
        supported = frame.loc[~frame["support_status"].eq("refused")]
        refusal = float(cover["operational_refused_rows"].sum() / cover["evaluation_rows_total"].sum())
        supported_improvement = float(supported["loss_improvement"].mean()) if len(supported) else np.nan
        if ci_adj[0] > 0 and int(origin["brier_improvement"].gt(0).sum()) >= 3 and refusal <= 0.15 and supported_improvement > 0:
            evidence = "promising_for_combined_development_testing"
        elif ci_adj[1] < 0:
            evidence = "repeated_harm"
        else:
            evidence = "inconclusive"
        candidate_loss = frame["candidate_multiclass_brier_loss"].mean()
        baseline_loss = frame["matched_baseline_multiclass_brier_loss"].mean()
        comparisons.append({
            "variant": variant, "evaluation_rows_matched": len(frame),
            "feature_coverage_rate": float(cover["feature_eligible_rows"].sum() / cover["evaluation_rows_total"].sum()),
            "operational_refusal_rate": refusal, "matched_baseline_multiclass_brier": baseline_loss,
            "candidate_multiclass_brier": candidate_loss, "pooled_brier_improvement": baseline_loss - candidate_loss,
            "supported_brier_improvement": supported_improvement,
            "ci_low_95": ci95[0], "ci_high_95": ci95[1],
            "ci_low_adjusted_98_75": ci_adj[0], "ci_high_adjusted_98_75": ci_adj[1],
            "origin_wins": int(origin["brier_improvement"].gt(0).sum()),
            "evidence_classification": evidence, "deployment_status": "not_deployed",
        })
        onehot = np.column_stack([frame["actual_state"].eq(state) for state in STATE_ORDER]).astype(float)
        for index, state in enumerate(STATE_ORDER):
            cand = frame[f"candidate_probability_{state}"].to_numpy(float)
            base = frame[f"matched_baseline_probability_{state}"].to_numpy(float)
            actual = onehot[:, index]
            classes.append({
                "variant": variant, "state": state, "rows": len(frame), "event_rate": float(actual.mean()),
                "candidate_binary_brier": float(np.square(actual - cand).mean()),
                "matched_baseline_binary_brier": float(np.square(actual - base).mean()),
                "brier_improvement": float(np.square(actual - base).mean() - np.square(actual - cand).mean()),
                "candidate_auc": safe_auc(actual, cand), "matched_baseline_auc": safe_auc(actual, base),
            })
        for league, chunk in frame.groupby("league", dropna=False):
            leagues.append({
                "variant": variant, "league": str(league), "rows": len(chunk),
                "brier_improvement": float(chunk["loss_improvement"].mean()),
            })
    return pd.DataFrame(comparisons), pd.DataFrame(classes), pd.DataFrame(leagues)


def build_checks(
    binary_predictions: pd.DataFrame,
    multi_predictions: pd.DataFrame,
    binary_comparison: pd.DataFrame,
    multi_comparison: pd.DataFrame,
    before: dict[str, str],
    after: dict[str, str],
) -> pd.DataFrame:
    rows = []

    def add(check: str, passed: bool, observed: object, expected: object, note: str) -> None:
        rows.append({"check": check, "passed": bool(passed), "observed": observed, "expected": expected, "note": note})

    add("four_binary_endpoints", binary_predictions["endpoint"].nunique() == 4, binary_predictions["endpoint"].nunique(), 4, "All decomposed binary questions were tested.")
    add("four_predeclared_variants", binary_predictions["variant"].nunique() == 4, binary_predictions["variant"].nunique(), 4, "No post-outcome variant was added.")
    add("four_rolling_origins", binary_predictions["origin_id"].nunique() == 4, binary_predictions["origin_id"].nunique(), 4, "Only the established development origins were used.")
    add("holdout_sealed", binary_predictions["evaluation_year"].max() == 2023 and multi_predictions["evaluation_year"].max() == 2023, max(binary_predictions["evaluation_year"].max(), multi_predictions["evaluation_year"].max()), 2023, "No 2024+ target was evaluated.")
    binary_formula = np.square(binary_predictions["actual"] - binary_predictions["matched_baseline_prediction"]) - np.square(binary_predictions["actual"] - binary_predictions["candidate_prediction"])
    add("binary_loss_formula", np.allclose(binary_formula, binary_predictions["loss_improvement"], atol=1e-14), float(np.max(np.abs(binary_formula - binary_predictions["loss_improvement"]))), "<=1e-14", "Paired Brier deltas reconstruct.")
    state_sum = multi_predictions[[f"candidate_probability_{state}" for state in STATE_ORDER]].sum(axis=1)
    add("multinomial_probability_sum", np.allclose(state_sum, 1, atol=1e-12), float(np.max(np.abs(state_sum - 1))), "<=1e-12", "Named candidate probabilities are coherent.")
    add("all_results_research_only", binary_comparison["deployment_status"].eq("not_deployed").all() and multi_comparison["deployment_status"].eq("not_deployed").all(), True, True, "No diagnostic result is treated as a deployment authorization.")
    add("v1_frozen", before == after, sum(before[key] == after[key] for key in before), len(before), "All frozen V1 files remain byte-identical.")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")

    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    before = {item["path"]: sha256(ROOT / item["path"]) for item in freeze["files"]}
    data = load_data()
    binary_predictions, binary_origins, binary_coverage, binary_coefficients = run_binary(data)
    multi_predictions, multi_origins, multi_coverage, multi_coefficients = run_multinomial(data)
    binary_comparison, binary_leagues = summarize_binary(binary_predictions, binary_origins, binary_coverage)
    multi_comparison, multi_classes, multi_leagues = summarize_multinomial(multi_predictions, multi_origins, multi_coverage)
    after = {path: sha256(ROOT / path) for path in before}
    checks = build_checks(binary_predictions, multi_predictions, binary_comparison, multi_comparison, before, after)

    outputs = {
        "binary_predictions.csv": binary_predictions,
        "binary_origin_metrics.csv": binary_origins,
        "binary_coverage_by_origin.csv": binary_coverage,
        "binary_candidate_comparison.csv": binary_comparison,
        "binary_league_metrics.csv": binary_leagues,
        "movement_state_predictions.csv": multi_predictions,
        "movement_state_origin_metrics.csv": multi_origins,
        "movement_state_coverage_by_origin.csv": multi_coverage,
        "movement_state_candidate_comparison.csv": multi_comparison,
        "movement_state_class_metrics.csv": multi_classes,
        "movement_state_league_metrics.csv": multi_leagues,
        "added_feature_coefficients.csv": pd.concat([binary_coefficients, multi_coefficients], ignore_index=True),
        "build_checks.csv": checks,
    }
    for name, frame in outputs.items():
        frame.to_csv(output / name, index=False)

    preregistration = {
        "contract_version": CONTRACT_VERSION,
        "development_origins": [list(origin) for origin in phase2.ORIGINS],
        "baseline_features": BASE_FEATURES,
        "variant_blocks": VARIANT_BLOCKS,
        "binary_endpoints": BINARY_TARGETS,
        "multinomial_state_order": STATE_ORDER,
        "primary_loss": "paired Brier loss versus baseline retrained on identical rows",
        "familywise_interval": "98.75% player-cluster bootstrap interval across four variants within each endpoint family",
        "support_policy": "score for development audit, but count unsupported rows as operational refusals",
        "future_holdout_policy": "2024+ target outcomes remain sealed",
        "promotion_policy": "No candidate can be deployed from this diagnostic; promising blocks proceed only to combined development tests.",
    }
    write_json(output / "preregistration.json", preregistration)
    sources = [
        PROFILE_DIR / "lagged_club_behavior_features.csv",
        PROFILE_DIR / "movement_target_contract.json",
        ROOT / "models" / "engine_v2" / "movement_target_contract.py",
        FREEZE,
    ]
    pd.DataFrame([
        {"source": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sources
    ]).to_csv(output / "source_manifest.csv", index=False)
    summary = {
        "contract_version": CONTRACT_VERSION,
        "development_evaluation_years": [2020, 2021, 2022, 2023],
        "binary_endpoints": list(BINARY_TARGETS),
        "variants": list(VARIANT_BLOCKS),
        "binary_evidence_counts": binary_comparison["evidence_classification"].value_counts().to_dict(),
        "multinomial_evidence_counts": multi_comparison["evidence_classification"].value_counts().to_dict(),
        "final_holdout_opened": False,
        "deployment_changed": False,
        "model_binaries_written": False,
        "v1_unchanged": before == after,
        "checks_passed": int(checks["passed"].sum()),
        "checks_total": len(checks),
    }
    write_json(output / "run_summary.json", summary)
    readme = """# Engine V2 lagged club-behavior rolling-origin experiment

This development diagnostic asks whether a club's strictly prior loan, sale,
turnover, and mature extension history adds signal beyond the compact Phase-4
player/context baseline. Four predeclared blocks are evaluated separately on
temporary-first movement, permanent-first movement, eventual permanent
separation, strict meaningful stay, and the coherent three-state outcome.

Every candidate is paired with a baseline retrained on identical rows. Support
failures remain visible as operational refusals. The 2024+ final holdout was not
opened; no model was serialized or deployed. See `preregistration.json` for the
locked comparison rules and the candidate-comparison tables for results.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    manifest = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "output_manifest.csv":
            manifest.append({"output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest).to_csv(output / "output_manifest.csv", index=False)
    if not checks["passed"].all():
        raise RuntimeError(checks.loc[~checks["passed"], ["check", "observed", "expected"]].to_dict("records"))
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nBinary comparisons\n", binary_comparison.to_string(index=False))
    print("\nMultinomial comparisons\n", multi_comparison.to_string(index=False))


if __name__ == "__main__":
    main()

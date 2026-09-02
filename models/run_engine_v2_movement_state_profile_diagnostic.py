"""Diagnose movement-state decomposition and underused football profiles.

Research-only: no final holdout, deployment artifact, API, or V1 mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "models")]

from models.engine_v2.feature_contract import V2_CORE_FEATURES  # noqa: E402
from mixed_type_preprocessor import fit_preprocessor  # noqa: E402
import run_engine_v2_candidate_contract as phase4  # noqa: E402
import run_engine_v2_feature_specification as phase2  # noqa: E402


OUTPUT = ROOT / "Data" / "processed" / "engine_v2_movement_state_profile_diagnostic"
CONTEXT = ROOT / "Data" / "processed" / "capology_club_financial_context"
USAGE = ROOT / "Data" / "processed" / "data_usage_audit_2026_09_02"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"
RANDOM_SEED = 20260901
C_GRID = (0.01, 0.1, 1.0, 10.0)
BOOTSTRAPS = 3000
AGE_CURVE = "age_distance_from_27_squared"
BASE_FEATURES = list(V2_CORE_FEATURES) + [AGE_CURVE]
CLUB_CONTEXT = [
    "payroll_log_eur", "payroll_league_percentile",
    "transfer_spend_to_payroll", "transfer_income_to_payroll",
    "squad_average_age", "squad_foreign_share",
]
STATE_ORDER = ["no_outbound_24m", "temporary_first_24m", "permanent_first_24m"]
BINARY_TARGETS = {
    "any_outbound_24m": "any_outbound_event_by_24m",
    "permanent_relationship_ended_24m": "permanent_outbound_event_by_24m",
    "temporary_first_24m": "target_temporary_first_24m",
    "strict_meaningful_stay_24m": "target_strict_meaningful_stay_24m",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: str) -> int:
    token = "|".join(parts).encode()
    return (RANDOM_SEED + int(hashlib.sha256(token).hexdigest()[:8], 16)) % (2**32 - 1)


def adaptive_ece(actual: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    frame = pd.DataFrame({"actual": actual, "prediction": probability}).sort_values("prediction")
    groups = np.array_split(np.arange(len(frame)), min(bins, len(frame)))
    return float(sum(len(group) * abs(frame.iloc[group].actual.mean() - frame.iloc[group].prediction.mean()) for group in groups) / len(frame))


def tune_binary(train: pd.DataFrame, validation: pd.DataFrame, features: list[str], target: str):
    pre = fit_preprocessor(train, features)
    x_train, x_validation = pre.transform(train), pre.transform(validation)
    y_train = pd.to_numeric(train[target]).to_numpy(int)
    y_validation = pd.to_numeric(validation[target]).to_numpy(int)
    candidates = []
    for c_value in C_GRID:
        model = LogisticRegression(C=c_value, penalty="l2", solver="liblinear", max_iter=3000, random_state=RANDOM_SEED).fit(x_train, y_train)
        probability = np.clip(model.predict_proba(x_validation)[:, 1], 1e-6, 1 - 1e-6)
        candidates.append((brier_score_loss(y_validation, probability), c_value, model))
    _, c_value, model = min(candidates, key=lambda item: item[0])
    return pre, model, float(c_value)


def tune_multinomial(train: pd.DataFrame, validation: pd.DataFrame, features: list[str]):
    pre = fit_preprocessor(train, features)
    x_train, x_validation = pre.transform(train), pre.transform(validation)
    y_train = train.movement_state_3.astype(str).to_numpy()
    y_validation = validation.movement_state_3.astype(str).to_numpy()
    candidates = []
    for c_value in C_GRID:
        model = LogisticRegression(C=c_value, penalty="l2", solver="lbfgs", max_iter=5000, random_state=RANDOM_SEED).fit(x_train, y_train)
        probability = np.clip(model.predict_proba(x_validation), 1e-6, 1 - 1e-6)
        probability /= probability.sum(axis=1, keepdims=True)
        candidates.append((log_loss(y_validation, probability, labels=model.classes_), c_value, model))
    _, c_value, model = min(candidates, key=lambda item: item[0])
    return pre, model, float(c_value)


def add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    result = phase4.add_age_curve(frame)
    any24 = result.any_outbound_event_by_24m.eq(1)
    first_type = result.first_any_outbound_type.astype("string")
    is_transfer = first_type.eq("transfer").fillna(False).to_numpy(bool)
    is_loan = first_type.eq("loan").fillna(False).to_numpy(bool)
    is_loan_return = first_type.eq("loan_return").fillna(False).to_numpy(bool)
    any_array = any24.to_numpy(bool)
    result["movement_state_raw"] = np.select(
        [~any_array, any_array & is_transfer, any_array & is_loan, any_array & is_loan_return],
        ["no_outbound_24m", "permanent_first_24m", "loan_first_24m", "loan_return_first_24m"],
        default="unresolved_first_type_24m",
    )
    result["movement_state_3"] = result.movement_state_raw.replace({
        "loan_first_24m": "temporary_first_24m",
        "loan_return_first_24m": "temporary_first_24m",
        "unresolved_first_type_24m": np.nan,
    })
    result["target_temporary_first_24m"] = result.movement_state_3.eq("temporary_first_24m").astype(int)
    result["target_strict_meaningful_stay_24m"] = (
        result.any_outbound_event_by_24m.eq(0)
        & result.target_sustained_meaningful_contribution.eq(1)
    ).astype(int)
    return result


def load_data() -> pd.DataFrame:
    data = add_targets(phase2.load_endpoint_frames()["any_outbound_24m"])
    audit = pd.read_csv(CONTEXT / "extension_prior_season_feature_coverage.csv", usecols=["capology_extension_event_id", "canonical_club_season_id", "missing_reason"])
    panel = pd.read_csv(CONTEXT / "canonical_club_season_financial_context.csv", usecols=["canonical_club_season_id"] + CLUB_CONTEXT)
    bridge = audit.merge(panel, on="canonical_club_season_id", how="left", validate="many_to_one")
    return data.merge(bridge, on="capology_extension_event_id", how="left", validate="one_to_one")


def target_audit(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    development = data.loc[data.signed_year.between(2020, 2023)].copy()
    definitions = pd.DataFrame([
        {"field": "no_outbound_24m", "rule": "any_outbound_event_by_24m == 0", "interpretation": "No recorded interruption from the extension club within 730 days."},
        {"field": "permanent_first_24m", "rule": "any_outbound_event_by_24m == 1 and first_any_outbound_type == transfer", "interpretation": "First recorded interruption is a permanent move."},
        {"field": "loan_first_24m", "rule": "any_outbound_event_by_24m == 1 and first_any_outbound_type == loan", "interpretation": "First recorded interruption is an outbound loan."},
        {"field": "loan_return_first_24m", "rule": "any_outbound_event_by_24m == 1 and first_any_outbound_type == loan_return", "interpretation": "Temporary-contract movement label; retained separately in the raw audit and grouped only for the three-state diagnostic."},
        {"field": "unresolved_first_type_24m", "rule": "any_outbound_event_by_24m == 1 and first_any_outbound_type missing", "interpretation": "Excluded from three-state model; never silently assigned."},
        {"field": "strict_meaningful_stay_24m", "rule": "no outbound within 24m and sustained >=25% same-club opportunity in both Year 1 and Year 2", "interpretation": "Joint sporting-use and uninterrupted-stay outcome; separate from movement-state probabilities."},
    ])
    counts = []
    for dimensions in [[], ["signed_year"], ["league"], ["signed_year", "league"]]:
        grouped = [((), development)] if not dimensions else development.groupby(dimensions, dropna=False)
        for key, chunk in grouped:
            key = key if isinstance(key, tuple) else (key,)
            base = {"scope": "pooled" if not dimensions else "+".join(dimensions)}
            base.update({dimension: value for dimension, value in zip(dimensions, key)})
            for state, n in chunk.movement_state_raw.value_counts().items():
                counts.append({**base, "movement_state_raw": state, "rows": int(n), "share": float(n / len(chunk)), "denominator": len(chunk)})
    interaction = pd.crosstab(
        development.movement_state_raw,
        development.target_sustained_meaningful_contribution,
        margins=True,
    ).reset_index().rename(columns={0: "not_sustained", 1: "sustained"})
    interaction["sustained_rate"] = interaction.sustained / (interaction.sustained + interaction.not_sustained)
    return definitions, pd.DataFrame(counts), interaction


def cluster_ci(frame: pd.DataFrame, loss_column: str, baseline_column: str, label: str) -> tuple[float, float]:
    temp = frame.assign(improvement=frame[baseline_column] - frame[loss_column])
    clusters = temp.groupby("canonical_player_id").improvement.agg(["sum", "size"])
    sums, sizes = clusters["sum"].to_numpy(float), clusters["size"].to_numpy(float)
    rng = np.random.default_rng(stable_seed(label))
    draws = []
    for _ in range(BOOTSTRAPS // 100):
        indices = rng.integers(0, len(clusters), size=(100, len(clusters)))
        draws.extend((sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)).tolist())
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def run_binary(data: pd.DataFrame):
    predictions, metrics = [], []
    for endpoint, target in BINARY_TARGETS.items():
        endpoint_data = data.copy()
        if endpoint == "permanent_relationship_ended_24m":
            endpoint_data = endpoint_data.loc[endpoint_data.permanent_outbound_horizon_observable_24m.eq(True)].copy()
        if endpoint == "temporary_first_24m":
            endpoint_data = endpoint_data.loc[endpoint_data.movement_state_3.notna()].copy()
        for origin, train_end, validation_year, evaluation_year in phase2.ORIGINS:
            train = endpoint_data.loc[endpoint_data.signed_year.le(train_end)].copy()
            validation = endpoint_data.loc[endpoint_data.signed_year.eq(validation_year)].copy()
            evaluation = endpoint_data.loc[endpoint_data.signed_year.eq(evaluation_year)].sort_values("capology_extension_event_id").copy()
            pre, model, c_value = tune_binary(train, validation, BASE_FEATURES, target)
            probability = np.clip(model.predict_proba(pre.transform(evaluation))[:, 1], 1e-6, 1 - 1e-6)
            actual = pd.to_numeric(evaluation[target]).to_numpy(int)
            baseline = float(pd.to_numeric(pd.concat([train, validation])[target]).mean())
            metrics.append({
                "endpoint": endpoint, "origin_id": origin, "evaluation_year": evaluation_year,
                "train_rows": len(train), "validation_rows": len(validation), "evaluation_rows": len(evaluation),
                "selected_C": c_value, "event_rate": actual.mean(), "brier": brier_score_loss(actual, probability),
                "chronology_brier": brier_score_loss(actual, np.repeat(baseline, len(actual))),
                "brier_skill": brier_score_loss(actual, np.repeat(baseline, len(actual))) - brier_score_loss(actual, probability),
                "auc": roc_auc_score(actual, probability), "average_precision": average_precision_score(actual, probability),
                "ece": adaptive_ece(actual, probability),
            })
            for i, (_, row) in enumerate(evaluation.iterrows()):
                predictions.append({
                    "endpoint": endpoint, "origin_id": origin, "evaluation_year": evaluation_year,
                    "capology_extension_event_id": row.capology_extension_event_id,
                    "canonical_player_id": row.canonical_player_id, "league": row.league,
                    "canonical_position": row.canonical_position, "actual": int(actual[i]),
                    "prediction": float(probability[i]), "chronology_prediction": baseline,
                    "brier_loss": float((actual[i] - probability[i]) ** 2),
                    "chronology_brier_loss": float((actual[i] - baseline) ** 2),
                })
    predictions = pd.DataFrame(predictions)
    pooled = []
    for endpoint, chunk in predictions.groupby("endpoint"):
        ci = cluster_ci(chunk, "brier_loss", "chronology_brier_loss", endpoint)
        origin = pd.DataFrame(metrics).loc[lambda x: x.endpoint.eq(endpoint)]
        pooled.append({
            "endpoint": endpoint, "evaluation_rows": len(chunk), "event_rate": chunk.actual.mean(),
            "pooled_brier": chunk.brier_loss.mean(), "chronology_brier": chunk.chronology_brier_loss.mean(),
            "brier_skill": chunk.chronology_brier_loss.mean() - chunk.brier_loss.mean(),
            "player_cluster_ci_low_95": ci[0], "player_cluster_ci_high_95": ci[1],
            "origin_wins": int(origin.brier_skill.gt(0).sum()),
            "pooled_auc": roc_auc_score(chunk.actual, chunk.prediction),
            "pooled_average_precision": average_precision_score(chunk.actual, chunk.prediction),
            "pooled_ece": adaptive_ece(chunk.actual.to_numpy(), chunk.prediction.to_numpy()),
        })
    return predictions, pd.DataFrame(metrics), pd.DataFrame(pooled)


def run_multinomial(data: pd.DataFrame):
    modelable = data.loc[data.movement_state_3.notna()].copy()
    rows, metrics = [], []
    for origin, train_end, validation_year, evaluation_year in phase2.ORIGINS:
        train = modelable.loc[modelable.signed_year.le(train_end)].copy()
        validation = modelable.loc[modelable.signed_year.eq(validation_year)].copy()
        evaluation = modelable.loc[modelable.signed_year.eq(evaluation_year)].sort_values("capology_extension_event_id").copy()
        pre, model, c_value = tune_multinomial(train, validation, BASE_FEATURES)
        probability = np.clip(model.predict_proba(pre.transform(evaluation)), 1e-6, 1 - 1e-6)
        probability /= probability.sum(axis=1, keepdims=True)
        class_index = {label: i for i, label in enumerate(model.classes_)}
        probability = np.column_stack([probability[:, class_index[label]] for label in STATE_ORDER])
        actual = evaluation.movement_state_3.astype(str).to_numpy()
        reference = pd.concat([train, validation]).movement_state_3.value_counts(normalize=True)
        baseline = np.array([reference.get(label, 0) for label in STATE_ORDER], dtype=float)
        baseline /= baseline.sum()
        onehot = np.column_stack([actual == label for label in STATE_ORDER]).astype(int)
        multiclass_brier = np.square(onehot - probability).sum(axis=1)
        baseline_brier = np.square(onehot - baseline).sum(axis=1)
        actual_index = np.array([STATE_ORDER.index(label) for label in actual], dtype=int)
        model_log_loss = float(-np.log(probability[np.arange(len(actual)), actual_index]).mean())
        chronology_log_loss = float(-np.log(np.tile(baseline, (len(actual), 1))[np.arange(len(actual)), actual_index]).mean())
        metrics.append({
            "origin_id": origin, "evaluation_year": evaluation_year, "train_rows": len(train),
            "validation_rows": len(validation), "evaluation_rows": len(evaluation), "selected_C": c_value,
            "multiclass_log_loss": model_log_loss,
            "chronology_log_loss": chronology_log_loss,
            "multiclass_brier": multiclass_brier.mean(), "chronology_brier": baseline_brier.mean(),
            "brier_skill": baseline_brier.mean() - multiclass_brier.mean(),
            "macro_ovr_auc": roc_auc_score(onehot, probability, average="macro", multi_class="ovr"),
        })
        for i, (_, row) in enumerate(evaluation.iterrows()):
            record = {
                "origin_id": origin, "evaluation_year": evaluation_year,
                "capology_extension_event_id": row.capology_extension_event_id,
                "canonical_player_id": row.canonical_player_id, "league": row.league,
                "canonical_position": row.canonical_position, "actual_state": actual[i],
                "multiclass_brier_loss": multiclass_brier[i], "chronology_brier_loss": baseline_brier[i],
            }
            for j, label in enumerate(STATE_ORDER):
                record[f"probability_{label}"] = probability[i, j]
                record[f"chronology_{label}"] = baseline[j]
            rows.append(record)
    predictions = pd.DataFrame(rows)
    metrics = pd.DataFrame(metrics)
    ci = cluster_ci(predictions, "multiclass_brier_loss", "chronology_brier_loss", "movement_state_multinomial")
    pooled = pd.DataFrame([{
        "evaluation_rows": len(predictions),
        "multiclass_brier": predictions.multiclass_brier_loss.mean(),
        "chronology_brier": predictions.chronology_brier_loss.mean(),
        "brier_skill": predictions.chronology_brier_loss.mean() - predictions.multiclass_brier_loss.mean(),
        "player_cluster_ci_low_95": ci[0], "player_cluster_ci_high_95": ci[1],
        "origin_wins": int(metrics.brier_skill.gt(0).sum()),
        "multiclass_log_loss": float(np.mean([
            -math.log(getattr(row, f"probability_{row.actual_state}")) for row in predictions.itertuples()
        ])),
        "chronology_log_loss": float(np.mean([
            -math.log(getattr(row, f"chronology_{row.actual_state}")) for row in predictions.itertuples()
        ])),
        "macro_ovr_auc": float(np.mean([
            roc_auc_score(predictions.actual_state.eq(label), predictions[f"probability_{label}"])
            for label in STATE_ORDER
        ])),
    }])
    class_rows = []
    for scope, column in [("pooled", None), ("league", "league"), ("position", "canonical_position")]:
        grouped = [("ALL", predictions)] if column is None else predictions.groupby(column, dropna=False)
        for group, chunk in grouped:
            for label in STATE_ORDER:
                actual_binary = chunk.actual_state.eq(label).astype(int)
                probability_label = chunk[f"probability_{label}"]
                chronology_label = chunk[f"chronology_{label}"]
                class_rows.append({
                    "scope": scope, "group": str(group), "state": label, "rows": len(chunk),
                    "events": int(actual_binary.sum()), "event_rate": actual_binary.mean(),
                    "auc": roc_auc_score(actual_binary, probability_label) if actual_binary.nunique() == 2 else np.nan,
                    "brier": brier_score_loss(actual_binary, probability_label),
                    "chronology_brier": brier_score_loss(actual_binary, chronology_label),
                    "brier_skill": brier_score_loss(actual_binary, chronology_label) - brier_score_loss(actual_binary, probability_label),
                    "calibration_gap": abs(actual_binary.mean() - probability_label.mean()),
                })
    return predictions, metrics, pooled, pd.DataFrame(class_rows)


def binary_subgroups(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint, endpoint_data in predictions.groupby("endpoint"):
        for scope, column in [("league", "league"), ("position", "canonical_position")]:
            for group, chunk in endpoint_data.groupby(column, dropna=False):
                rows.append({
                    "endpoint": endpoint, "scope": scope, "group": str(group), "rows": len(chunk),
                    "events": int(chunk.actual.sum()), "event_rate": chunk.actual.mean(),
                    "auc": roc_auc_score(chunk.actual, chunk.prediction) if chunk.actual.nunique() == 2 else np.nan,
                    "brier": chunk.brier_loss.mean(), "chronology_brier": chunk.chronology_brier_loss.mean(),
                    "brier_skill": chunk.chronology_brier_loss.mean() - chunk.brier_loss.mean(),
                    "calibration_gap": abs(chunk.actual.mean() - chunk.prediction.mean()),
                })
    return pd.DataFrame(rows)


def run_context_decomposition(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint, target in BINARY_TARGETS.items():
        endpoint_data = data.copy()
        if endpoint == "permanent_relationship_ended_24m":
            endpoint_data = endpoint_data.loc[endpoint_data.permanent_outbound_horizon_observable_24m.eq(True)].copy()
        if endpoint == "temporary_first_24m":
            endpoint_data = endpoint_data.loc[endpoint_data.movement_state_3.notna()].copy()
        endpoint_data = endpoint_data.loc[endpoint_data[CLUB_CONTEXT].notna().all(axis=1)].copy()
        pred_rows = []
        for origin, train_end, validation_year, evaluation_year in phase2.ORIGINS:
            train = endpoint_data.loc[endpoint_data.signed_year.le(train_end)].copy()
            validation = endpoint_data.loc[endpoint_data.signed_year.eq(validation_year)].copy()
            evaluation = endpoint_data.loc[endpoint_data.signed_year.eq(evaluation_year)].sort_values("capology_extension_event_id").copy()
            base_pre, base_model, _ = tune_binary(train, validation, BASE_FEATURES, target)
            context_pre, context_model, _ = tune_binary(train, validation, BASE_FEATURES + CLUB_CONTEXT, target)
            base = np.clip(base_model.predict_proba(base_pre.transform(evaluation))[:, 1], 1e-6, 1 - 1e-6)
            context = np.clip(context_model.predict_proba(context_pre.transform(evaluation))[:, 1], 1e-6, 1 - 1e-6)
            actual = pd.to_numeric(evaluation[target]).to_numpy(int)
            for i, (_, row) in enumerate(evaluation.iterrows()):
                pred_rows.append({
                    "origin_id": origin, "canonical_player_id": row.canonical_player_id, "actual": actual[i],
                    "baseline_loss": (actual[i] - base[i]) ** 2, "context_loss": (actual[i] - context[i]) ** 2,
                })
        pred = pd.DataFrame(pred_rows)
        pred["improvement"] = pred.baseline_loss - pred.context_loss
        clusters = pred.groupby("canonical_player_id").improvement.agg(["sum", "size"])
        rng = np.random.default_rng(stable_seed("context", endpoint))
        draws = []
        sums, sizes = clusters["sum"].to_numpy(), clusters["size"].to_numpy()
        for _ in range(BOOTSTRAPS // 100):
            indices = rng.integers(0, len(clusters), size=(100, len(clusters)))
            draws.extend((sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)).tolist())
        origin_delta = pred.groupby("origin_id").improvement.mean()
        rows.append({
            "endpoint": endpoint, "matched_rows": len(pred), "context_coverage_rate": len(pred) / len(data.loc[data.signed_year.between(2020, 2023)]),
            "baseline_brier": pred.baseline_loss.mean(), "context_brier": pred.context_loss.mean(),
            "brier_improvement": pred.improvement.mean(), "ci_low_95": np.quantile(draws, .025),
            "ci_high_95": np.quantile(draws, .975), "origin_wins": int(origin_delta.gt(0).sum()),
        })
    return pd.DataFrame(rows)


def profile_register() -> pd.DataFrame:
    rows = [
        ("P01", "player_role_trajectory", "Recent role and selection trajectory", "start share; bench share; captaincy; positional entropy; role changes; 30/90/365-day trends", "game_lineups 3.18M rows; development prior365 lineup coverage 88.3%, >=10 rows 75.2%", "constructed and tested; did not clear adjusted evidence gate", "strictly dated match records", "loan risk; meaningful stay", "P0_COMPLETE"),
        ("P02", "squad_competition", "Player-relative squad depth", "same-position minutes/start rank; competitor count; age/value/wage gap; incoming positional competition", "appearances + lineups + valuations + salaries; verified event-grain profiles built", "constructed and tested; did not clear adjusted evidence gate", "squad reconstructed as of signing date", "loan risk; meaningful stay", "P0_COMPLETE"),
        ("P03", "club_movement_behavior", "Lagged club loan/sale/extension behavior", "prior loan-out rate; permanent-sale rate; extension-to-loan/sale rate; age/position-conditioned turnover", "1.106M canonical transfer events + 2,805 extension events", "constructed and tested; small temporary-movement hints did not clear adjusted gate", "events strictly before each signing", "loan and permanent risks", "P0_COMPLETE"),
        ("P04", "manager_tactical_stability", "Manager and formation stability", "manager tenure; changes in 365d; formation entropy; position-system fit; matches under current manager", "88,958 games; development manager coverage 95.1% and formation/current-manager-role coverage 87.9%", "constructed and tested; no predeclared block passed", "pre-signing games only", "meaningful stay; permanent risk", "P0_COMPLETE"),
        ("P05", "relative_financial_status", "Player status inside club economics", "wage percentile; same-position wage rank; wage-to-minutes mismatch; value/payroll share; proposed tier jump", "salary panels + 1,074 canonical payroll/transfer club-seasons", "club-level financial blocks constructed and rejected as continuity repairs; proposal-tier movement remains untested", "lagged payroll; proposal metrics must be scenario-labeled", "all decision modules", "P1_PARTIAL"),
        ("P06", "market_trajectory", "Player and club-relative value trajectory", "6/12/24m momentum; volatility; staleness; peak drawdown; squad value rank; value-to-role mismatch", "1.408M canonical dated valuations", "partially engineered; compact V2 uses only current log value", "observations dated before signing", "permanent risk; financial downside", "P1"),
        ("P07", "injury_availability", "Historical availability burden", "days unavailable; spell count; recurrence; recency; burden per playing opportunity; injury-type groups", "15.6K thesis rows previously tested; 143K ID-linked injuries dormant; development any-prior coverage 70.4% in audited source", "larger source not integrated", "injury start/end before signing; open spells handled explicitly", "meaningful stay; wage exposure", "P1"),
        ("P08", "club_competitive_trajectory", "Club sporting level and transition", "points/game trend; rank; promotion/relegation; European load; goal-difference trend", "58,247 team-season rows + 177,916 club-games; standings coverage 97.3%", "older context research only", "last completed season or pre-signing games", "all movement states", "P1"),
        ("P09", "role_normalized_style", "Position/role-normalized player archetype", "within-role percentiles for progression, creation, defending, aerials, carrying; multi-season trend", "17,790 accepted advanced FBref rows x85 fields; old role-standardized styles exist", "research-only; current V2 uses none", "completed seasons before signing; position-specific definitions", "meaningful stay; value downside", "P1"),
        ("P10", "cohesion_network", "Dated teammate and lineup continuity", "co-start network; retained teammate share; stable-unit minutes; same-position competition", "1.257M career teammate edges are not time-safe; dated lineups available", "unsafe aggregate table dormant; dated reconstruction absent", "must reconstruct from dated matches, never career snapshot", "meaningful stay", "P1"),
        ("P11", "promotion_transition", "Promoted-club context", "prior second-tier payroll/transfer level; squad churn; promotion flag; top-flight experience", "current prior Big-Five club context missing for 12.2% of continuity development evaluation rows", "not constructed", "requires comparable second-tier historical sources", "league reliability; all states", "P0"),
        ("P12", "statsbomb_microstyle", "Event-level technical/tactical profile", "pressure, reception, pass/carry zones, shot quality, defensive actions, spatial role", "14.87M local events but selective modern Big-Five coverage; >=3 reviewed prior matches only 0.18% of development extensions", "case-study only", "reviewed identity and representative coverage required", "case study, not global engine", "P3"),
        ("P13", "contract_scenario", "Proposal-conditioned contract profile", "term, age at expiry, wage change, commitment/value ratio, options", "Capology extension and salary data", "commercial fields exist but excluded from unconditional V2 risk", "must be explicitly user-proposed scenario, not silently historical", "decision support, not player-intrinsic risk", "P1"),
    ]
    columns = ["profile_id", "domain", "proposed_profile", "candidate_metrics", "local_evidence", "current_use", "timing_rule", "relevant_outcomes", "priority"]
    return pd.DataFrame(rows, columns=columns)


def frozen_hashes() -> dict[str, str]:
    freeze = json.loads(FREEZE.read_text())
    return {item["path"]: sha256(ROOT / item["path"]) for item in freeze["files"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    frozen_before = frozen_hashes()
    data = load_data()
    definitions, counts, interaction = target_audit(data)
    binary_predictions, binary_origin, binary_pooled = run_binary(data)
    binary_subgroup = binary_subgroups(binary_predictions)
    multi_predictions, multi_origin, multi_pooled, multi_class = run_multinomial(data)
    context = run_context_decomposition(data)
    profiles = profile_register()
    frozen_after = frozen_hashes()

    development = data.loc[data.signed_year.between(2020, 2023)]
    pooled_counts = counts.loc[counts.scope.eq("pooled")].set_index("movement_state_raw").rows.to_dict()
    checks = [
        {"check": "development_rows", "passed": len(development) == 1560, "observed": len(development), "expected": 1560},
        {"check": "raw_states_exhaustive", "passed": sum(pooled_counts.values()) == 1560, "observed": sum(pooled_counts.values()), "expected": 1560},
        {"check": "unresolved_not_modeled", "passed": not multi_predictions.capology_extension_event_id.isin(development.loc[development.movement_state_3.isna(), "capology_extension_event_id"]).any(), "observed": int(development.movement_state_3.isna().sum()), "expected": "excluded"},
        {"check": "three_state_probabilities_sum_to_one", "passed": np.allclose(multi_predictions[[f"probability_{x}" for x in STATE_ORDER]].sum(axis=1), 1, atol=1e-12), "observed": float(np.max(np.abs(multi_predictions[[f"probability_{x}" for x in STATE_ORDER]].sum(axis=1) - 1))), "expected": "<=1e-12"},
        {"check": "four_origins_binary", "passed": binary_origin.origin_id.nunique() == 4, "observed": binary_origin.origin_id.nunique(), "expected": 4},
        {"check": "four_origins_multinomial", "passed": multi_origin.origin_id.nunique() == 4, "observed": multi_origin.origin_id.nunique(), "expected": 4},
        {"check": "no_holdout_after_2023", "passed": max(binary_predictions.evaluation_year.max(), multi_predictions.evaluation_year.max()) == 2023, "observed": max(binary_predictions.evaluation_year.max(), multi_predictions.evaluation_year.max()), "expected": 2023},
        {"check": "profile_register_records_completed_p0_work", "passed": profiles.priority.eq("P0_COMPLETE").sum() >= 4, "observed": int(profiles.priority.eq("P0_COMPLETE").sum()), "expected": ">=4"},
        {"check": "frozen_v1_unchanged", "passed": frozen_before == frozen_after, "observed": sum(frozen_before[k] == frozen_after[k] for k in frozen_before), "expected": len(frozen_before)},
    ]
    definitions.to_csv(output / "state_definition_audit.csv", index=False)
    counts.to_csv(output / "movement_state_counts.csv", index=False)
    interaction.to_csv(output / "movement_contribution_interaction.csv", index=False)
    binary_predictions.to_csv(output / "binary_endpoint_predictions.csv", index=False)
    binary_origin.to_csv(output / "binary_endpoint_origin_metrics.csv", index=False)
    binary_pooled.to_csv(output / "binary_endpoint_pooled_metrics.csv", index=False)
    binary_subgroup.to_csv(output / "binary_endpoint_subgroup_metrics.csv", index=False)
    multi_predictions.to_csv(output / "movement_state_predictions.csv", index=False)
    multi_origin.to_csv(output / "movement_state_origin_metrics.csv", index=False)
    multi_pooled.to_csv(output / "movement_state_pooled_metrics.csv", index=False)
    multi_class.to_csv(output / "movement_state_class_metrics.csv", index=False)
    context.to_csv(output / "club_context_by_decomposed_endpoint.csv", index=False)
    profiles.to_csv(output / "profile_opportunity_register.csv", index=False)
    pd.DataFrame(checks).to_csv(output / "build_checks.csv", index=False)
    source_paths = [
        CONTEXT / "canonical_club_season_financial_context.csv",
        CONTEXT / "extension_prior_season_feature_coverage.csv",
        USAGE / "extension_source_overlap_summary.csv",
        USAGE / "selected_model_feature_usage.csv",
        FREEZE,
    ]
    pd.DataFrame([{"source": p.relative_to(ROOT).as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size} for p in source_paths]).to_csv(output / "source_manifest.csv", index=False)
    summary = {
        "development_rows": len(development), "movement_state_raw_counts": pooled_counts,
        "binary_endpoint_results": binary_pooled.to_dict("records"),
        "multinomial_pooled_result": multi_pooled.iloc[0].to_dict(),
        "multinomial_origin_results": multi_origin.to_dict("records"),
        "profile_opportunities": len(profiles), "completed_p0_profiles": profiles.loc[profiles.priority.eq("P0_COMPLETE"), "profile_id"].tolist(),
        "final_holdout_opened": False, "deployment_changed": False,
        "all_build_checks_passed": bool(pd.DataFrame(checks).passed.all()),
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "README.md").write_text(
        "# Engine V2 movement-state and profile diagnostic\n\n"
        "Development-only decomposition of 24-month movement into no outbound, temporary-first, and permanent-first states, plus a separate strict meaningful-stay endpoint. Loan returns and unresolved movement types remain visible in the raw audit. The output also records high-value football profiles that can be constructed from existing local data. No 2024+ final outcome was evaluated and nothing was deployed.\n"
    )
    manifest = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "output_manifest.csv":
            manifest.append({"output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest).to_csv(output / "output_manifest.csv", index=False)
    print(json.dumps(summary, indent=2))
    print(binary_pooled.to_string(index=False))
    print(context.to_string(index=False))


if __name__ == "__main__":
    main()

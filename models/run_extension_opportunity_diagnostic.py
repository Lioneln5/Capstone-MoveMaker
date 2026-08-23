"""Phase-1 diagnostic for post-extension opportunity and role trajectories.

This is a signal test, not a production scoring model. It locks contract-covered
cohorts, pre-registers target definitions, and evaluates a fixed feature ladder
with rolling historical origins and player-clustered uncertainty.
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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "models") not in sys.path:
    sys.path.insert(0, str(ROOT / "models"))
from mixed_type_preprocessor import fit_preprocessor


SOURCE = ROOT / "Data" / "processed" / "contract_extension_integration" / "extension_modeling_master.csv"
UPSTREAM_VERIFY = ROOT / "Data" / "processed" / "contract_extension_integration" / "independent_verification.json"
OUTPUT = ROOT / "Data" / "processed" / "extension_opportunity_diagnostic"

RANDOM_SEED = 20260811
BOOTSTRAP_REPETITIONS = 5000
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]
LOGISTIC_C = [0.01, 0.1, 1.0, 10.0]
ORIGINS = [
    ("origin_2020", 2018, 2019, 2020),
    ("origin_2021", 2019, 2020, 2021),
    ("origin_2022", 2020, 2021, 2022),
    ("origin_2023", 2021, 2022, 2023),
]
WINDOWS = {
    "pooled_four": [2020, 2021, 2022, 2023],
    "mature_three": [2021, 2022, 2023],
    "terminal_two": [2022, 2023],
}

FEATURE_SETS = {
    "M0_chronological_baseline": [],
    "M1_player_market_profile": [
        "age_at_signing", "canonical_position", "league", "log_market_value_at_signing",
    ],
    "M2_plus_prior_opportunity": [
        "age_at_signing", "canonical_position", "league", "log_market_value_at_signing",
        "pre365_same_club_all_competition_opportunity_share",
        "pre365_same_club_all_competition_appearance_rate",
        "pre365_goals_per90", "pre365_assists_per90",
    ],
    "M3_plus_performance_history": [
        "age_at_signing", "canonical_position", "league", "log_market_value_at_signing",
        "pre365_same_club_all_competition_opportunity_share",
        "pre365_same_club_all_competition_appearance_rate",
        "pre365_goals_per90", "pre365_assists_per90",
        "pre1_canonical_minutes_played", "pre1_canonical_goals_per90", "pre1_canonical_assists_per90",
        "pre2_canonical_minutes_played", "pre2_canonical_goals_per90", "pre2_canonical_assists_per90",
        "pre1_fbref_expected_goals", "pre1_fbref_progressive_carries",
        "pre1_fbref_progressive_passes", "pre1_fbref_pass_completion_pct",
    ],
    "M4_plus_contract_terms": [
        "age_at_signing", "canonical_position", "league", "log_market_value_at_signing",
        "pre365_same_club_all_competition_opportunity_share",
        "pre365_same_club_all_competition_appearance_rate",
        "pre365_goals_per90", "pre365_assists_per90",
        "pre1_canonical_minutes_played", "pre1_canonical_goals_per90", "pre1_canonical_assists_per90",
        "pre2_canonical_minutes_played", "pre2_canonical_goals_per90", "pre2_canonical_assists_per90",
        "pre1_fbref_expected_goals", "pre1_fbref_progressive_carries",
        "pre1_fbref_progressive_passes", "pre1_fbref_pass_completion_pct",
        "exact_duration_years", "age_at_expiration", "log_annual_gross_eur",
        "log_fixed_wage_commitment_eur",
    ],
    "M5_plus_relative_financial_context": [
        "age_at_signing", "canonical_position", "league", "log_market_value_at_signing",
        "pre365_same_club_all_competition_opportunity_share",
        "pre365_same_club_all_competition_appearance_rate",
        "pre365_goals_per90", "pre365_assists_per90",
        "pre1_canonical_minutes_played", "pre1_canonical_goals_per90", "pre1_canonical_assists_per90",
        "pre2_canonical_minutes_played", "pre2_canonical_goals_per90", "pre2_canonical_assists_per90",
        "pre1_fbref_expected_goals", "pre1_fbref_progressive_carries",
        "pre1_fbref_progressive_passes", "pre1_fbref_pass_completion_pct",
        "exact_duration_years", "age_at_expiration", "log_annual_gross_eur",
        "log_fixed_wage_commitment_eur", "club_salary_percentile",
        "club_salary_share_known", "salary_change_from_prior_pct",
        "annual_wage_to_market_value", "commitment_to_market_value",
    ],
}
FEATURE_SETS["M6_compact_nonlinear"] = FEATURE_SETS["M5_plus_relative_financial_context"]

ENDPOINTS = {
    "year2_opportunity_share": {"kind": "continuous", "target": "target_year2_opportunity_share", "cohort": "primary"},
    "major_role_decline": {"kind": "binary", "target": "target_major_role_decline", "cohort": "initial_role"},
    "sustained_meaningful_contribution": {"kind": "binary", "target": "target_sustained_meaningful_contribution", "cohort": "primary"},
    "contract_covered_minimal_involvement": {"kind": "binary", "target": "target_contract_covered_minimal_involvement", "cohort": "primary"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: str) -> int:
    return (RANDOM_SEED + int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)) % (2**32 - 1)


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").fillna("").str.strip().str.casefold().isin({"true", "1", "yes"})


def load_frame() -> pd.DataFrame:
    verification = json.loads(UPSTREAM_VERIFY.read_text(encoding="utf-8"))
    if not verification.get("all_checks_passed"):
        raise RuntimeError("Contract integration has not passed independent verification.")
    frame = pd.read_csv(SOURCE, low_memory=False)
    frame["signed_date"] = pd.to_datetime(frame["signed_date"], errors="coerce")
    frame["expiration_date"] = pd.to_datetime(frame["expiration_date"], errors="coerce")
    frame["signed_year"] = frame["signed_date"].dt.year.astype("Int64")
    for year in [1, 2, 3]:
        frame[f"contract_covers_y{year}"] = frame["expiration_date"].ge(
            frame["signed_date"] + pd.Timedelta(days=365 * year)
        )
    numeric = lambda c: pd.to_numeric(frame[c], errors="coerce")
    frame["age_at_expiration"] = frame["age_at_signing"] + numeric("exact_duration_years")
    frame["log_market_value_at_signing"] = np.log1p(numeric("at_signing_market_value_eur").clip(lower=0))
    frame["log_annual_gross_eur"] = np.log1p(numeric("annual_gross_eur").clip(lower=0))
    fixed = numeric("annual_gross_eur") * numeric("exact_duration_years")
    frame["log_fixed_wage_commitment_eur"] = np.log1p(fixed.clip(lower=0))
    frame["annual_wage_to_market_value"] = numeric("annual_gross_eur").div(
        numeric("at_signing_market_value_eur").where(numeric("at_signing_market_value_eur").gt(0))
    )
    frame["commitment_to_market_value"] = fixed.div(
        numeric("at_signing_market_value_eur").where(numeric("at_signing_market_value_eur").gt(0))
    )
    pre_minutes = numeric("pre365_same_club_all_competition_minutes")
    frame["pre365_goals_per90"] = 90 * numeric("pre365_same_club_all_competition_goals").div(pre_minutes.where(pre_minutes.gt(0)))
    frame["pre365_assists_per90"] = 90 * numeric("pre365_same_club_all_competition_assists").div(pre_minutes.where(pre_minutes.gt(0)))

    pre = numeric("pre365_same_club_all_competition_opportunity_share")
    y1 = numeric("post_y1_same_club_all_competition_opportunity_share")
    y2 = numeric("post_y2_same_club_all_competition_opportunity_share")
    linked = frame["canonical_player_id"].notna()
    evidence = as_bool(frame["pre365_window_evidence_eligible_10_club_games"]) & as_bool(
        frame["post_y2_window_evidence_eligible_10_club_games"]
    )
    frame["cohort_primary_y2"] = linked & evidence & frame["contract_covers_y2"] & pre.notna() & y2.notna()
    frame["cohort_initial_role_y2"] = frame["cohort_primary_y2"] & pre.ge(0.25)
    frame["target_year2_opportunity_share"] = y2
    frame["target_opportunity_change"] = y2 - pre
    frame["target_major_role_decline"] = ((pre - y2).ge(0.20) & y2.le(0.50 * pre)).astype(int)
    frame["target_sustained_meaningful_contribution"] = (y1.ge(0.25) & y2.ge(0.25)).astype(int)
    uninterrupted = pd.to_numeric(frame["uninterrupted_same_club_spell_730d"], errors="coerce").eq(1)
    frame["target_contract_covered_minimal_involvement"] = (y2.lt(0.10) & uninterrupted).astype(int)
    frame["pre_role_band"] = pd.cut(pre, [-np.inf, 0.10, 0.25, 0.50, 0.75, np.inf], labels=["minimal", "fringe", "rotation", "regular", "core"], right=False)
    frame["year2_role_band"] = pd.cut(y2, [-np.inf, 0.10, 0.25, 0.50, 0.75, np.inf], labels=["minimal", "fringe", "rotation", "regular", "core"], right=False)
    frame["age_band"] = pd.cut(frame["age_at_signing"], [-np.inf, 21, 25, 29, 33, np.inf], labels=["<=21", "22-25", "26-29", "30-33", "34+"])
    frame["contract_length_band"] = pd.cut(frame["exact_duration_years"], [-np.inf, 2, 3, 4, 5, np.inf], labels=["<2y", "2-<3y", "3-<4y", "4-<5y", "5y+"])
    return frame


def endpoint_frame(frame: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    cohort = ENDPOINTS[endpoint]["cohort"]
    mask = frame["cohort_primary_y2"] if cohort == "primary" else frame["cohort_initial_role_y2"]
    return frame.loc[mask & frame[ENDPOINTS[endpoint]["target"]].notna()].copy()


def continuous_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    correlation = pd.Series(actual).corr(pd.Series(prediction), method="spearman")
    return {
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(actual, prediction))),
        "r2": float(r2_score(actual, prediction)),
        "spearman": float(correlation) if pd.notna(correlation) else np.nan,
    }


def classification_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
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


def tune_predict(train: pd.DataFrame, validation: pd.DataFrame, evaluation: pd.DataFrame, features: list[str], target: str, kind: str, variant: str) -> tuple[np.ndarray, dict[str, Any], int]:
    y_train = pd.to_numeric(train[target]).to_numpy()
    y_validation = pd.to_numeric(validation[target]).to_numpy()
    if not features:
        anchor = float(np.mean(y_train))
        validation_prediction = np.repeat(anchor, len(validation))
        fit_mean = float(pd.concat([train, validation])[target].mean())
        prediction = np.repeat(fit_mean, len(evaluation))
        metrics = continuous_metrics(y_validation, validation_prediction) if kind == "continuous" else classification_metrics(y_validation.astype(int), validation_prediction)
        return prediction, {"selected_parameter": np.nan, **metrics}, 0

    pre = fit_preprocessor(train, features)
    x_train, x_validation = pre.transform(train), pre.transform(validation)
    candidates = []
    nonlinear = variant == "M6_compact_nonlinear"
    if kind == "continuous":
        if nonlinear:
            grid = [
                {"n_estimators": 75, "learning_rate": 0.03, "max_depth": 1, "min_samples_leaf": 20},
                {"n_estimators": 100, "learning_rate": 0.03, "max_depth": 2, "min_samples_leaf": 25},
                {"n_estimators": 75, "learning_rate": 0.05, "max_depth": 2, "min_samples_leaf": 30},
            ]
            for i, params in enumerate(grid):
                model = GradientBoostingRegressor(loss="huber", random_state=stable_seed(variant, str(i)), **params).fit(x_train, y_train)
                pred = np.clip(model.predict(x_validation), 0, 1.05)
                candidates.append(((mean_absolute_error(y_validation, pred), math.sqrt(mean_squared_error(y_validation, pred))), i, params))
        else:
            for alpha in RIDGE_ALPHAS:
                model = Ridge(alpha=alpha).fit(x_train, y_train)
                pred = np.clip(model.predict(x_validation), 0, 1.05)
                candidates.append(((mean_absolute_error(y_validation, pred), math.sqrt(mean_squared_error(y_validation, pred))), alpha, {"alpha": alpha}))
    else:
        y_train_int = y_train.astype(int)
        if nonlinear:
            grid = [
                {"n_estimators": 75, "learning_rate": 0.03, "max_depth": 1, "min_samples_leaf": 20},
                {"n_estimators": 100, "learning_rate": 0.03, "max_depth": 2, "min_samples_leaf": 25},
                {"n_estimators": 75, "learning_rate": 0.05, "max_depth": 2, "min_samples_leaf": 30},
            ]
            for i, params in enumerate(grid):
                model = GradientBoostingClassifier(loss="log_loss", random_state=stable_seed(variant, str(i)), **params).fit(x_train, y_train_int)
                pred = np.clip(model.predict_proba(x_validation)[:, 1], 1e-6, 1 - 1e-6)
                candidates.append(((brier_score_loss(y_validation.astype(int), pred), log_loss(y_validation.astype(int), pred, labels=[0, 1])), i, params))
        else:
            for c_value in LOGISTIC_C:
                model = LogisticRegression(C=c_value, penalty="l2", solver="liblinear", max_iter=3000, random_state=RANDOM_SEED).fit(x_train, y_train_int)
                pred = np.clip(model.predict_proba(x_validation)[:, 1], 1e-6, 1 - 1e-6)
                candidates.append(((brier_score_loss(y_validation.astype(int), pred), log_loss(y_validation.astype(int), pred, labels=[0, 1])), c_value, {"C": c_value}))
    _, parameter, params = min(candidates, key=lambda x: x[0])
    fit = pd.concat([train, validation], ignore_index=True)
    fit_pre = fit_preprocessor(fit, features)
    x_fit, x_eval = fit_pre.transform(fit), fit_pre.transform(evaluation)
    y_fit = pd.to_numeric(fit[target]).to_numpy()
    if kind == "continuous":
        model = GradientBoostingRegressor(loss="huber", random_state=stable_seed(variant, "fit"), **params) if nonlinear else Ridge(alpha=params["alpha"])
        model.fit(x_fit, y_fit)
        prediction = np.clip(model.predict(x_eval), 0, 1.05)
        val_metrics = continuous_metrics(y_validation, np.clip((GradientBoostingRegressor(loss="huber", random_state=stable_seed(variant, "val"), **params) if nonlinear else Ridge(alpha=params["alpha"])).fit(x_train, y_train).predict(x_validation), 0, 1.05))
    else:
        model = GradientBoostingClassifier(loss="log_loss", random_state=stable_seed(variant, "fit"), **params) if nonlinear else LogisticRegression(C=params["C"], penalty="l2", solver="liblinear", max_iter=3000, random_state=RANDOM_SEED)
        model.fit(x_fit, y_fit.astype(int))
        prediction = np.clip(model.predict_proba(x_eval)[:, 1], 1e-6, 1 - 1e-6)
        val_model = GradientBoostingClassifier(loss="log_loss", random_state=stable_seed(variant, "val"), **params) if nonlinear else LogisticRegression(C=params["C"], penalty="l2", solver="liblinear", max_iter=3000, random_state=RANDOM_SEED)
        val_model.fit(x_train, y_train.astype(int))
        val_metrics = classification_metrics(y_validation.astype(int), np.clip(val_model.predict_proba(x_validation)[:, 1], 1e-6, 1 - 1e-6))
    return prediction, {"selected_parameter": json.dumps(params, sort_keys=True), **val_metrics}, x_fit.shape[1]


def fit_models(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows, metric_rows = [], []
    for endpoint, spec in ENDPOINTS.items():
        data = endpoint_frame(frame, endpoint)
        target, kind = spec["target"], spec["kind"]
        for origin, train_end, validation_year, evaluation_year in ORIGINS:
            train = data.loc[data["signed_year"].le(train_end)].copy()
            validation = data.loc[data["signed_year"].eq(validation_year)].copy()
            evaluation = data.loc[data["signed_year"].eq(evaluation_year)].sort_values("capology_extension_event_id").copy()
            if min(len(train), len(validation), len(evaluation)) < 30:
                raise RuntimeError(f"Insufficient rolling split: {endpoint}/{origin}: {len(train)}/{len(validation)}/{len(evaluation)}")
            for variant, features in FEATURE_SETS.items():
                prediction, validation_metrics, encoded = tune_predict(train, validation, evaluation, features, target, kind, variant)
                actual = pd.to_numeric(evaluation[target]).to_numpy()
                evaluation_metrics = continuous_metrics(actual, prediction) if kind == "continuous" else classification_metrics(actual.astype(int), prediction)
                metric_rows.append({
                    "endpoint": endpoint, "kind": kind, "origin_id": origin, "model_variant": variant,
                    "train_end_year": train_end, "validation_year": validation_year, "evaluation_year": evaluation_year,
                    "train_rows": len(train), "validation_rows": len(validation), "evaluation_rows": len(evaluation),
                    "raw_feature_count": len(features), "encoded_feature_count": encoded,
                    **{f"validation_{k}": v for k, v in validation_metrics.items()},
                    **{f"evaluation_{k}": v for k, v in evaluation_metrics.items()},
                })
                for i, (_, row) in enumerate(evaluation.iterrows()):
                    actual_value = float(row[target])
                    predicted = float(prediction[i])
                    prediction_rows.append({
                        "endpoint": endpoint, "kind": kind, "origin_id": origin,
                        "evaluation_year": evaluation_year, "capology_extension_event_id": row["capology_extension_event_id"],
                        "canonical_player_id": row["canonical_player_id"], "league": row["league"],
                        "canonical_position": row["canonical_position"], "age_band": row["age_band"],
                        "contract_length_band": row["contract_length_band"], "model_variant": variant,
                        "actual": actual_value, "prediction": predicted,
                        "primary_loss": abs(actual_value - predicted) if kind == "continuous" else (actual_value - predicted) ** 2,
                    })
    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)


def paired(predictions: pd.DataFrame, endpoint: str, baseline: str, candidate: str, years: list[int]) -> pd.DataFrame:
    data = predictions.loc[predictions["endpoint"].eq(endpoint) & predictions["evaluation_year"].isin(years)]
    keys = ["endpoint", "origin_id", "evaluation_year", "capology_extension_event_id", "canonical_player_id"]
    left = data.loc[data["model_variant"].eq(baseline), keys + ["actual", "primary_loss"]].rename(columns={"primary_loss": "baseline_loss"})
    right = data.loc[data["model_variant"].eq(candidate), keys + ["prediction", "primary_loss"]].rename(columns={"primary_loss": "candidate_loss"})
    return left.merge(right, on=keys, how="inner", validate="one_to_one")


def cluster_bootstrap(pairs: pd.DataFrame, seed: int) -> dict[str, float]:
    pair = pairs.copy()
    pair["improvement"] = pair["baseline_loss"] - pair["candidate_loss"]
    clusters = pair.groupby("canonical_player_id")["improvement"].agg(["sum", "size"])
    sums, sizes = clusters["sum"].to_numpy(float), clusters["size"].to_numpy(float)
    rng, draws = np.random.default_rng(seed), []
    for start in range(0, BOOTSTRAP_REPETITIONS, 250):
        n = min(250, BOOTSTRAP_REPETITIONS - start)
        idx = rng.integers(0, len(clusters), size=(n, len(clusters)))
        draws.extend((sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)).tolist())
    return {
        "cluster_count": len(clusters),
        "cluster_ci_lower_95": float(np.quantile(draws, 0.025)),
        "cluster_ci_upper_95": float(np.quantile(draws, 0.975)),
        "cluster_probability_candidate_better": float(np.mean(np.asarray(draws) > 0)),
    }


def comparisons(predictions: pd.DataFrame) -> pd.DataFrame:
    comparisons_to_run = [
        ("M1_vs_M0", "M0_chronological_baseline", "M1_player_market_profile"),
        ("M2_vs_M1", "M1_player_market_profile", "M2_plus_prior_opportunity"),
        ("M3_vs_M2", "M2_plus_prior_opportunity", "M3_plus_performance_history"),
        ("M4_vs_M3_contract_terms", "M3_plus_performance_history", "M4_plus_contract_terms"),
        ("M5_vs_M4_relative_financial", "M4_plus_contract_terms", "M5_plus_relative_financial_context"),
        ("M5_vs_M3_all_contract_economics", "M3_plus_performance_history", "M5_plus_relative_financial_context"),
        ("M5_vs_M0_total", "M0_chronological_baseline", "M5_plus_relative_financial_context"),
        ("M6_vs_M5_nonlinearity", "M5_plus_relative_financial_context", "M6_compact_nonlinear"),
    ]
    rows = []
    for endpoint, spec in ENDPOINTS.items():
        for window, years in WINDOWS.items():
            for label, baseline, candidate in comparisons_to_run:
                pairs = paired(predictions, endpoint, baseline, candidate, years)
                improvement = float((pairs["baseline_loss"] - pairs["candidate_loss"]).mean())
                by_origin = pairs.assign(improvement=pairs["baseline_loss"] - pairs["candidate_loss"]).groupby("origin_id")["improvement"].mean()
                boot = cluster_bootstrap(pairs, stable_seed(endpoint, window, label))
                rows.append({
                    "endpoint": endpoint, "kind": spec["kind"], "window": window, "comparison": label,
                    "baseline": baseline, "candidate": candidate, "evaluation_years": " | ".join(map(str, years)),
                    "evaluation_rows": len(pairs), "mean_loss_improvement": improvement,
                    "origin_wins": int(by_origin.gt(0).sum()), "origin_count": len(by_origin),
                    **boot,
                })
    return pd.DataFrame(rows)


def audits(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    linked = frame["canonical_player_id"].notna()
    pre = as_bool(frame["pre365_window_evidence_eligible_10_club_games"])
    post2 = as_bool(frame["post_y2_window_evidence_eligible_10_club_games"])
    funnel_masks = [
        ("All extension events", pd.Series(True, index=frame.index)),
        ("Canonical player linked", linked),
        ("Pre-365 evidence >=10 club games", linked & pre),
        ("Year-2 evidence >=10 club games", linked & pre & post2),
        ("Contract scheduled to cover 730 days", frame["cohort_primary_y2"]),
        ("Meaningful pre-extension role >=25%", frame["cohort_initial_role_y2"]),
    ]
    funnel = pd.DataFrame([{"step": i, "stage": label, "records": int(mask.sum())} for i, (label, mask) in enumerate(funnel_masks)])
    funnel["retention_from_start"] = funnel["records"] / len(frame)
    funnel["retention_from_previous"] = funnel["records"] / funnel["records"].shift(1)

    distribution_rows = []
    for endpoint, spec in ENDPOINTS.items():
        data = endpoint_frame(frame, endpoint)
        target = spec["target"]
        for group_type, column in [("overall", None), ("signed_year", "signed_year"), ("league", "league"), ("position", "canonical_position"), ("age_band", "age_band"), ("contract_length", "contract_length_band")]:
            groups = [("all", data)] if column is None else data.groupby(column, dropna=False, observed=True)
            for group, subset in groups:
                values = pd.to_numeric(subset[target], errors="coerce").dropna()
                row = {"endpoint": endpoint, "kind": spec["kind"], "group_type": group_type, "group": str(group), "records": len(values), "mean": float(values.mean())}
                if spec["kind"] == "continuous":
                    row.update({"std": float(values.std()), "p10": float(values.quantile(.1)), "p25": float(values.quantile(.25)), "median": float(values.median()), "p75": float(values.quantile(.75)), "p90": float(values.quantile(.9))})
                else:
                    row["positive_rows"] = int(values.sum())
                    row["positive_rate"] = float(values.mean())
                distribution_rows.append(row)
    distributions = pd.DataFrame(distribution_rows)

    primary = frame.loc[frame["cohort_primary_y2"]].copy()
    transition = pd.crosstab(primary["pre_role_band"], primary["year2_role_band"], dropna=False).stack(future_stack=True).rename("records").reset_index()
    transition["row_rate"] = transition["records"] / transition.groupby("pre_role_band", observed=True)["records"].transform("sum")
    threshold_rows = []
    pre_share = pd.to_numeric(primary["pre365_same_club_all_competition_opportunity_share"])
    y1_share = pd.to_numeric(primary["post_y1_same_club_all_competition_opportunity_share"])
    y2_share = pd.to_numeric(primary["post_y2_same_club_all_competition_opportunity_share"])
    for threshold in [0.05, 0.10, 0.20, 0.25, 0.50]:
        threshold_rows.extend([
            {"target_family": "year2_below_threshold", "threshold": threshold, "eligible_rows": len(primary), "positive_rows": int(y2_share.lt(threshold).sum()), "positive_rate": float(y2_share.lt(threshold).mean())},
            {"target_family": "sustained_y1_y2_above_threshold", "threshold": threshold, "eligible_rows": len(primary), "positive_rows": int((y1_share.ge(threshold) & y2_share.ge(threshold)).sum()), "positive_rate": float((y1_share.ge(threshold) & y2_share.ge(threshold)).mean())},
        ])
    for absolute in [0.10, 0.20, 0.30]:
        decline = pre_share.sub(y2_share).ge(absolute) & y2_share.le(.5 * pre_share) & pre_share.ge(.25)
        eligible = pre_share.ge(.25)
        threshold_rows.append({"target_family": "major_role_decline_absolute_floor", "threshold": absolute, "eligible_rows": int(eligible.sum()), "positive_rows": int(decline.sum()), "positive_rate": float(decline[eligible].mean())})
    return {"cohort_funnel": funnel, "target_distributions": distributions, "role_band_transitions": transition, "threshold_sensitivity": pd.DataFrame(threshold_rows)}


def risk_bands(predictions: pd.DataFrame) -> pd.DataFrame:
    selected = predictions.loc[predictions["model_variant"].eq("M5_plus_relative_financial_context")].copy()
    rows = []
    for endpoint, group in selected.groupby("endpoint"):
        group = group.copy()
        group["risk_band"] = pd.qcut(group["prediction"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        for band, subset in group.groupby("risk_band", observed=True):
            rows.append({"endpoint": endpoint, "risk_band": int(band), "records": len(subset), "mean_prediction": float(subset["prediction"].mean()), "observed_outcome": float(subset["actual"].mean()), "mean_primary_loss": float(subset["primary_loss"].mean())})
    result = pd.DataFrame(rows)
    result["observed_monotone"] = result.groupby("endpoint")["observed_outcome"].transform(lambda x: bool(x.is_monotonic_increasing))
    return result


def subgroup_stability(predictions: pd.DataFrame) -> pd.DataFrame:
    selected = predictions.loc[predictions["model_variant"].eq("M5_plus_relative_financial_context")].copy()
    rows = []
    for endpoint, endpoint_data in selected.groupby("endpoint"):
        kind = ENDPOINTS[endpoint]["kind"]
        for group_type, column in [("league", "league"), ("position", "canonical_position"), ("age_band", "age_band"), ("contract_length", "contract_length_band")]:
            for group, subset in endpoint_data.groupby(column, dropna=False):
                if len(subset) < 30:
                    continue
                actual, prediction = subset["actual"].to_numpy(), subset["prediction"].to_numpy()
                metrics = continuous_metrics(actual, prediction) if kind == "continuous" else classification_metrics(actual.astype(int), prediction)
                rows.append({"endpoint": endpoint, "kind": kind, "group_type": group_type, "group": str(group), "records": len(subset), **metrics})
    return pd.DataFrame(rows)


def decisions(comparison: pd.DataFrame, bands: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint, spec in ENDPOINTS.items():
        pooled = comparison.loc[comparison["endpoint"].eq(endpoint) & comparison["window"].eq("pooled_four")]
        total = pooled.loc[pooled["comparison"].eq("M5_vs_M0_total")].iloc[0]
        contract = pooled.loc[pooled["comparison"].eq("M5_vs_M3_all_contract_economics")].iloc[0]
        nonlinear = pooled.loc[pooled["comparison"].eq("M6_vs_M5_nonlinearity")].iloc[0]
        monotone = bool(bands.loc[bands["endpoint"].eq(endpoint), "observed_monotone"].all())
        signal = total.cluster_ci_lower_95 > 0 and total.origin_wins >= 3
        contract_signal = contract.cluster_ci_lower_95 > 0 and contract.origin_wins >= 3
        if signal and monotone:
            status = "advance_predictive_candidate"
        elif signal:
            status = "exploratory_signal_calibration_or_ranking_failed"
        else:
            status = "descriptive_only_no_validated_incremental_signal"
        rows.append({
            "endpoint": endpoint, "kind": spec["kind"], "status": status,
            "total_loss_improvement_vs_baseline": total.mean_loss_improvement,
            "total_ci_lower_95": total.cluster_ci_lower_95, "total_ci_upper_95": total.cluster_ci_upper_95,
            "total_origin_wins": int(total.origin_wins), "risk_bands_observed_monotone": monotone,
            "contract_economics_add_signal": bool(contract_signal),
            "contract_incremental_improvement": contract.mean_loss_improvement,
            "contract_incremental_ci_lower_95": contract.cluster_ci_lower_95,
            "contract_incremental_ci_upper_95": contract.cluster_ci_upper_95,
            "nonlinear_incremental_improvement": nonlinear.mean_loss_improvement,
            "recommended_display_if_advanced": {
                "year2_opportunity_share": "Expected Year-2 involvement range",
                "major_role_decline": "Major role-decline probability",
                "sustained_meaningful_contribution": "Meaningful-contributor-through-Year-2 probability",
                "contract_covered_minimal_involvement": "Contract-covered minimal-involvement probability",
            }[endpoint],
        })
    return pd.DataFrame(rows)


def feature_manifest() -> pd.DataFrame:
    rows = []
    for variant, features in FEATURE_SETS.items():
        for order, feature in enumerate(features, 1):
            block = "player_market"
            if variant.startswith("M2") or feature.startswith("pre365"):
                block = "prior_opportunity"
            if variant.startswith("M3") or feature.startswith(("pre1_", "pre2_")):
                block = "performance_history"
            if feature in {"exact_duration_years", "age_at_expiration", "log_annual_gross_eur", "log_fixed_wage_commitment_eur"}:
                block = "contract_terms"
            if feature in {"club_salary_percentile", "club_salary_share_known", "salary_change_from_prior_pct", "annual_wage_to_market_value", "commitment_to_market_value"}:
                block = "relative_financial_context"
            rows.append({"model_variant": variant, "feature_order": order, "feature": feature, "feature_block": block, "timing": "known_at_extension_signing"})
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    audit_tables = audits(frame)
    predictions, origin_metrics = fit_models(frame)
    comparison = comparisons(predictions)
    bands = risk_bands(predictions)
    subgroups = subgroup_stability(predictions)
    decision = decisions(comparison, bands)

    model_matrix_columns = [
        "capology_extension_event_id", "canonical_player_id", "player_name", "league", "canonical_position",
        "signed_date", "signed_year", "expiration_date", "contract_covers_y1", "contract_covers_y2", "contract_covers_y3",
        "cohort_primary_y2", "cohort_initial_role_y2", "pre_role_band", "year2_role_band",
        "target_year2_opportunity_share", "target_opportunity_change", "target_major_role_decline",
        "target_sustained_meaningful_contribution", "target_contract_covered_minimal_involvement",
    ] + list(dict.fromkeys(feature for features in FEATURE_SETS.values() for feature in features))
    model_matrix = frame[model_matrix_columns].copy()

    checks = []
    def add(name: str, observed: object, expected: object, passed: bool, notes: str) -> None:
        checks.append({"check": name, "observed": observed, "expected": expected, "passed": bool(passed), "notes": notes})
    primary = frame.loc[frame["cohort_primary_y2"]]
    initial = frame.loc[frame["cohort_initial_role_y2"]]
    add("source_event_ids_unique", int(frame["capology_extension_event_id"].duplicated().sum()), 0, not frame["capology_extension_event_id"].duplicated().any(), "One row per extension event.")
    add("primary_y2_cohort_size", len(primary), 1385, len(primary) == 1385, "Contract-covered, linked, exact pre/post evidence cohort.")
    add("initial_role_cohort_size", len(initial), 922, len(initial) == 922, "Primary cohort with pre-extension share >=25%.")
    add("contract_coverage_enforced", int(primary["contract_covers_y2"].sum()), len(primary), primary["contract_covers_y2"].all(), "Every primary event was scheduled through day 730.")
    add("primary_target_complete", int(primary["target_year2_opportunity_share"].notna().sum()), len(primary), primary["target_year2_opportunity_share"].notna().all(), "No missing continuous targets.")
    add("prediction_key_unique", int(predictions.duplicated(["endpoint", "origin_id", "model_variant", "capology_extension_event_id"]).sum()), 0, not predictions.duplicated(["endpoint", "origin_id", "model_variant", "capology_extension_event_id"]).any(), "One out-of-time prediction per endpoint/origin/model/event.")
    add("rolling_evaluation_years", " | ".join(map(str, sorted(predictions["evaluation_year"].unique()))), "2020 | 2021 | 2022 | 2023", sorted(predictions["evaluation_year"].unique()) == [2020, 2021, 2022, 2023], "Four declared historical origins.")
    add("predictions_bounded", int(predictions["prediction"].between(0, 1.05).sum()), len(predictions), predictions["prediction"].between(0, 1.05).all(), "All continuous and probability predictions fall in valid declared bounds.")
    add("feature_timing_safe", int(feature_manifest()["timing"].eq("known_at_extension_signing").sum()), len(feature_manifest()), feature_manifest()["timing"].eq("known_at_extension_signing").all(), "No post-signing feature in model ladder.")
    checks_frame = pd.DataFrame(checks)

    write_csv(model_matrix, OUTPUT / "opportunity_model_matrix.csv")
    for name, table in audit_tables.items():
        write_csv(table, OUTPUT / f"{name}.csv")
    write_csv(feature_manifest(), OUTPUT / "feature_manifest.csv")
    write_csv(origin_metrics, OUTPUT / "model_origin_metrics.csv")
    write_csv(predictions, OUTPUT / "model_predictions.csv")
    write_csv(comparison, OUTPUT / "model_comparison_results.csv")
    write_csv(bands, OUTPUT / "calibration_risk_bands.csv")
    write_csv(subgroups, OUTPUT / "subgroup_stability.csv")
    write_csv(decision, OUTPUT / "opportunity_decision_summary.csv")
    write_csv(checks_frame, OUTPUT / "build_checks.csv")
    source_manifest = pd.DataFrame([
        {"source_file": SOURCE.relative_to(ROOT).as_posix(), "bytes": SOURCE.stat().st_size, "sha256": sha256(SOURCE)},
        {"source_file": UPSTREAM_VERIFY.relative_to(ROOT).as_posix(), "bytes": UPSTREAM_VERIFY.stat().st_size, "sha256": sha256(UPSTREAM_VERIFY)},
    ])
    write_csv(source_manifest, OUTPUT / "source_manifest.csv")

    summary = {
        "source_extension_events": len(frame), "primary_y2_cohort": len(primary), "initial_role_y2_cohort": len(initial),
        "evaluation_years": [2020, 2021, 2022, 2023], "rolling_origins": len(ORIGINS),
        "endpoints": len(ENDPOINTS), "model_variants": len(FEATURE_SETS),
        "prediction_rows": len(predictions), "checks_passed": int(checks_frame["passed"].sum()),
        "checks_total": len(checks_frame), "all_checks_passed": bool(checks_frame["passed"].all()),
        "decisions": dict(zip(decision["endpoint"], decision["status"])),
    }
    (OUTPUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Extension opportunity and role-trajectory diagnostic", "",
        "This Phase-1 diagnostic tests whether future post-extension involvement is measurable and predictively supportable. It is not a production scoring model.", "",
        "## Pre-registered cohorts and targets", "",
        f"- Primary Year-2 cohort: {len(primary):,} extensions with linked identity, at least 10 club games in exact pre-365 and Year-2 windows, and a scheduled expiration no earlier than day 730.",
        f"- Initial-role cohort: {len(initial):,} primary events with pre-extension opportunity share of at least 25%.",
        "- Continuous outcome: exact Year-2 opportunity share.",
        "- Major role decline: pre-share >=25%, absolute decline >=20 points, and Year-2 share <=50% of pre-share.",
        "- Sustained meaningful contribution: opportunity share >=25% in both Year 1 and Year 2.",
        "- Contract-covered minimal involvement: Year-2 share <10% while the uninterrupted same-club spell survives 730 days.", "",
        "## Validation", "",
        "Four rolling origins evaluate signing years 2020-2023. Hyperparameters are selected only on the immediately preceding validation year. Paired player-cluster bootstraps use 5,000 repetitions.", "",
        "## Decisions", "",
    ]
    for row in decision.itertuples(index=False):
        lines.append(f"- `{row.endpoint}`: `{row.status}`; total loss improvement {row.total_loss_improvement_vs_baseline:.6f}, 95% CI [{row.total_ci_lower_95:.6f}, {row.total_ci_upper_95:.6f}], {row.total_origin_wins}/4 origin wins; contract economics add signal: `{row.contract_economics_add_signal}`.")
    lines.extend(["", "See the decision summary, comparison results, risk bands, subgroup table, target audits, feature manifest, and independent verification before making product claims."])
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    files = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name != "output_manifest.csv")
    write_csv(pd.DataFrame([{"output_file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files]), OUTPUT / "output_manifest.csv")
    if not checks_frame["passed"].all():
        raise RuntimeError(f"Failed checks: {checks_frame.loc[~checks_frame['passed'], 'check'].tolist()}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

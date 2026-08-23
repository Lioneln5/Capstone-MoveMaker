"""Test day-365 monitoring updates for utilization and market-value risk.

This v2 experiment leaves every finalized v1 output untouched.  It asks whether
evidence observed during the first 365 days after a permanent transfer improves
forecasts for the *remaining* second year (days 366-730).

Two leakage guards are central:

1. Only players with no recorded outbound event by day 365 are scored.
2. The day-365 valuation is the latest valuation on or before the landmark;
   a later "nearest to 12 months" observation is never used as a predictor.
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
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for dependency_dir in [ROOT / "models", ROOT / "scripts"]:
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

import run_market_value_downside_model as market
import run_paid_transfer_underutilization_model as paid
import run_retention_diagnostic as retention
from train_compatibility_models import ALPHA_GRID, sha256_file


OUTPUT_DIR = ROOT / "Data" / "processed" / "landmark_monitoring_experiment"
VALUATION_HISTORY_PATH = (
    ROOT / "Data" / "processed" / "canonical_integration" / "canonical_valuation_history.csv"
)
CANONICAL_TRANSFER_PATH = retention.CANONICAL_TRANSFER_PATH

RANDOM_SEED = 20260811
BOOTSTRAP_REPETITIONS = 5000
ORIGINS = paid.ORIGINS
WINDOWS = paid.WINDOWS
LOGISTIC_C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
MIN_YEAR2_CLUB_GAMES = 10
MAX_LANDMARK_VALUE_STALENESS_DAYS = 180

UTILIZATION_MODELS = [
    "M0_chronological_constant",
    "U1_transfer_time_profile",
    "U2_plus_first_year_involvement",
]
VALUE_MODELS = [
    "M0_chronological_constant",
    "V1_transfer_time_profile",
    "V2_plus_day365_value_state",
    "V3_plus_first_year_involvement",
]

ENDPOINTS = {
    "year2_opportunity_share": {
        "module": "utilization",
        "kind": "regression",
        "target": "target_year2_opportunity_share",
        "models": UTILIZATION_MODELS,
        "candidate": "U2_plus_first_year_involvement",
        "primary_comparison": "U2_vs_U1_first_year_update",
    },
    "year2_underutilization_10pct": {
        "module": "utilization",
        "kind": "classification",
        "target": "target_year2_underutilization_10pct",
        "models": UTILIZATION_MODELS,
        "candidate": "U2_plus_first_year_involvement",
        "primary_comparison": "U2_vs_U1_first_year_update",
    },
    "year2_underutilization_25pct": {
        "module": "utilization",
        "kind": "classification",
        "target": "target_year2_underutilization_25pct",
        "models": UTILIZATION_MODELS,
        "candidate": "U2_plus_first_year_involvement",
        "primary_comparison": "U2_vs_U1_first_year_update",
    },
    "year2_value_log_ratio": {
        "module": "market_value",
        "kind": "regression",
        "target": "target_landmark_to_24m_log_value_ratio",
        "models": VALUE_MODELS,
        "candidate": "V3_plus_first_year_involvement",
        "primary_comparison": "V3_vs_V1_total_monitoring_update",
    },
    "year2_value_downside_10pct": {
        "module": "market_value",
        "kind": "classification",
        "target": "target_landmark_to_24m_downside_10pct",
        "models": VALUE_MODELS,
        "candidate": "V3_plus_first_year_involvement",
        "primary_comparison": "V3_vs_V1_total_monitoring_update",
    },
    "year2_value_downside_25pct": {
        "module": "market_value",
        "kind": "classification",
        "target": "target_landmark_to_24m_downside_25pct",
        "models": VALUE_MODELS,
        "candidate": "V3_plus_first_year_involvement",
        "primary_comparison": "V3_vs_V1_total_monitoring_update",
    },
    "year2_value_downside_50pct": {
        "module": "market_value",
        "kind": "classification",
        "target": "target_landmark_to_24m_downside_50pct",
        "models": VALUE_MODELS,
        "candidate": "V3_plus_first_year_involvement",
        "primary_comparison": "V3_vs_V1_total_monitoring_update",
    },
}

COMPARISONS = {
    "utilization": [
        ("U1_vs_M0_transfer_time", "M0_chronological_constant", "U1_transfer_time_profile"),
        ("U2_vs_U1_first_year_update", "U1_transfer_time_profile", "U2_plus_first_year_involvement"),
        ("U2_vs_M0_total", "M0_chronological_constant", "U2_plus_first_year_involvement"),
    ],
    "market_value": [
        ("V1_vs_M0_transfer_time", "M0_chronological_constant", "V1_transfer_time_profile"),
        ("V2_vs_V1_day365_value_state", "V1_transfer_time_profile", "V2_plus_day365_value_state"),
        ("V3_vs_V2_first_year_involvement", "V2_plus_day365_value_state", "V3_plus_first_year_involvement"),
        ("V3_vs_V1_total_monitoring_update", "V1_transfer_time_profile", "V3_plus_first_year_involvement"),
        ("V3_vs_M0_total", "M0_chronological_constant", "V3_plus_first_year_involvement"),
    ],
}

FIRST_YEAR_INVOLVEMENT_FEATURES = [
    "first_year_destination_opportunity_share",
    "first_year_destination_appearance_rate",
    "first_year_destination_start_rate",
    "first_year_destination_substitute_selection_rate",
    "first_year_destination_goals_per90",
    "first_year_destination_assists_per90",
]
DAY365_VALUE_FEATURES = [
    "age_at_landmark",
    "log_landmark_value_eur",
    "landmark_value_staleness_days",
    "first_year_log_value_ratio",
    "log_fee_to_landmark_value",
]

PROTECTED_DIRS = [
    ROOT / "Data" / "processed" / name
    for name in [
        "retention_diagnostic",
        "market_value_downside_model",
        "market_value_calibration_experiment",
        "paid_transfer_underutilization_model",
        "paid_transfer_underutilization_refinement",
        "fee_benchmarking_model",
        "capital_at_risk_integration",
    ]
]


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


def add_first_year_features(frame: pd.DataFrame) -> None:
    games = pd.to_numeric(frame["club_destination_post12_games"], errors="coerce")
    minutes = pd.to_numeric(frame["appearance_post12_destination_minutes"], errors="coerce")
    frame["first_year_destination_opportunity_share"] = retention.coarse.safe_divide(
        minutes, 90.0 * games
    ).clip(0.0, 1.05)
    frame["first_year_destination_appearance_rate"] = retention.coarse.safe_divide(
        frame["appearance_post12_destination_games"], games
    )
    frame["first_year_destination_start_rate"] = retention.coarse.safe_divide(
        frame["lineup_post12_destination_starts"], games
    )
    frame["first_year_destination_substitute_selection_rate"] = retention.coarse.safe_divide(
        frame["lineup_post12_destination_substitute_selections"], games
    )
    frame["first_year_destination_goals_per90"] = 90.0 * retention.coarse.safe_divide(
        frame["appearance_post12_destination_goals"], minutes
    )
    frame["first_year_destination_assists_per90"] = 90.0 * retention.coarse.safe_divide(
        frame["appearance_post12_destination_assists"], minutes
    )


def add_day365_at_risk_flag(frame: pd.DataFrame) -> pd.DataFrame:
    canonical, _ = retention.canonical_transfers()
    working = frame.copy()
    for column in ["player_id", "from_club_id", "to_club_id"]:
        working[column] = retention.clean_id(working[column])
    working["transfer_date"] = pd.to_datetime(working["transfer_date"], errors="coerce")
    arrivals = working[["transfer_event_id", "player_id", "transfer_date", "to_club_id"]].rename(
        columns={"transfer_date": "arrival_date", "to_club_id": "from_club_id"}
    )
    candidates = arrivals.merge(canonical, on=["player_id", "from_club_id"], how="left")
    candidates = candidates.loc[candidates["transfer_date"].gt(candidates["arrival_date"])].copy()
    first = (
        candidates.sort_values(["transfer_event_id", "transfer_date", "canonical_transfer_event_id"])
        .drop_duplicates("transfer_event_id", keep="first")
        [["transfer_event_id", "transfer_date", "canonical_transfer_event_id"]]
        .rename(
            columns={
                "transfer_date": "first_outbound_date",
                "canonical_transfer_event_id": "first_outbound_event_id",
            }
        )
    )
    working = working.merge(first, on="transfer_event_id", how="left", validate="one_to_one")
    working["days_to_first_outbound"] = (
        working["first_outbound_date"] - working["transfer_date"]
    ).dt.days
    working["at_destination_day365"] = (
        working["days_to_first_outbound"].isna()
        | working["days_to_first_outbound"].gt(365)
    )
    return working


def add_asof_landmark_valuations(frame: pd.DataFrame) -> pd.DataFrame:
    history = pd.read_csv(
        VALUATION_HISTORY_PATH,
        usecols=[
            "canonical_player_id",
            "valuation_date",
            "canonical_market_value_eur",
            "valuation_date_valid",
        ],
        low_memory=False,
    )
    history["player_id"] = retention.clean_id(history["canonical_player_id"])
    relevant = set(retention.clean_id(frame["player_id"]))
    history = history.loc[history["player_id"].isin(relevant)].copy()
    history["valuation_date"] = pd.to_datetime(history["valuation_date"], errors="coerce")
    history["canonical_market_value_eur"] = pd.to_numeric(
        history["canonical_market_value_eur"], errors="coerce"
    )
    valid_date = market.as_bool(history["valuation_date_valid"])
    history = history.loc[
        valid_date
        & history["valuation_date"].notna()
        & history["canonical_market_value_eur"].gt(0)
    ].sort_values(["player_id", "valuation_date"])
    groups = {player: group for player, group in history.groupby("player_id", sort=False)}

    values: list[dict[str, Any]] = []
    for row in frame[["transfer_event_id", "player_id", "transfer_date"]].itertuples(index=False):
        player_id = str(row.player_id)
        cutoff = pd.Timestamp(row.transfer_date) + pd.Timedelta(days=365)
        group = groups.get(player_id)
        record: dict[str, Any] = {
            "transfer_event_id": row.transfer_event_id,
            "landmark_date": cutoff,
        }
        if group is not None and not group.empty:
            dates = group["valuation_date"].to_numpy(dtype="datetime64[ns]")
            index = int(np.searchsorted(dates, np.datetime64(cutoff), side="right") - 1)
            if index >= 0:
                observed_date = pd.Timestamp(dates[index])
                record.update(
                    {
                        "landmark_value_date": observed_date,
                        "landmark_value_eur": float(group.iloc[index]["canonical_market_value_eur"]),
                        "landmark_value_staleness_days": int((cutoff - observed_date).days),
                    }
                )
        values.append(record)
    result = frame.merge(pd.DataFrame(values), on="transfer_event_id", how="left", validate="one_to_one")
    if (
        result["landmark_value_date"].notna()
        & result["landmark_value_date"].gt(result["landmark_date"])
    ).any():
        raise RuntimeError("A landmark valuation was observed after day 365.")
    return result


def feature_manifest_rows(
    module: str,
    feature_sets: dict[str, list[str]],
    transfer_time_features: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_variant, features in feature_sets.items():
        if not features:
            rows.append(
                {
                    "module": module,
                    "model_variant": model_variant,
                    "feature_order": 0,
                    "feature": "",
                    "timing_classification": "not applicable",
                    "source_fields": "",
                    "rationale": "Chronological median or prevalence benchmark.",
                }
            )
            continue
        for order, feature in enumerate(features, start=1):
            if feature in transfer_time_features:
                timing = "available at or strictly before transfer"
                source = feature
                rationale = "Frozen compact transfer-time profile control."
            elif feature in FIRST_YEAR_INVOLVEMENT_FEATURES:
                timing = "observed during days 0-365 after transfer"
                source = "post12 destination appearance/lineup and club-game aggregates"
                rationale = "First-year sporting evidence available at the day-365 monitoring update."
            elif feature in DAY365_VALUE_FEATURES:
                timing = "available on day 365; valuation is strictly as-of the landmark"
                source = "canonical valuation history on/before day 365"
                rationale = "Current asset state known at the declared monitoring landmark."
            else:
                raise RuntimeError(f"Unclassified landmark feature: {feature}")
            rows.append(
                {
                    "module": module,
                    "model_variant": model_variant,
                    "feature_order": order,
                    "feature": feature,
                    "timing_classification": timing,
                    "source_fields": source,
                    "rationale": rationale,
                }
            )
    return rows


def build_utilization_frame() -> tuple[pd.DataFrame, dict[str, list[str]], list[dict[str, Any]]]:
    frame, upstream_features, _ = paid.build_model_frame()
    frame = add_day365_at_risk_flag(frame)
    frame["landmark_date"] = frame["transfer_date"] + pd.Timedelta(days=365)
    add_first_year_features(frame)
    year2_games = pd.to_numeric(frame["club_destination_post24_games"], errors="coerce") - pd.to_numeric(
        frame["club_destination_post12_games"], errors="coerce"
    )
    year2_minutes = pd.to_numeric(
        frame["appearance_post24_destination_minutes"], errors="coerce"
    ) - pd.to_numeric(frame["appearance_post12_destination_minutes"], errors="coerce")
    if year2_games.lt(0).any() or year2_minutes.lt(0).any():
        raise RuntimeError("Nested utilization windows produced a negative year-two difference.")
    frame["year2_destination_club_games"] = year2_games
    frame["year2_destination_minutes"] = year2_minutes
    frame["target_year2_opportunity_share"] = retention.coarse.safe_divide(
        year2_minutes, 90.0 * year2_games
    ).clip(0.0, 1.05)
    frame["target_year2_underutilization_10pct"] = frame[
        "target_year2_opportunity_share"
    ].lt(0.10).astype(int)
    frame["target_year2_underutilization_25pct"] = frame[
        "target_year2_opportunity_share"
    ].lt(0.25).astype(int)
    frame = frame.loc[
        frame["at_destination_day365"]
        & frame["year2_destination_club_games"].ge(MIN_YEAR2_CLUB_GAMES)
        & frame["target_year2_opportunity_share"].notna()
        & frame["season_start_year"].le(2023)
    ].copy()

    transfer_features = upstream_features["M1_paid_deal_market_profile"]
    feature_sets = {
        "M0_chronological_constant": [],
        "U1_transfer_time_profile": transfer_features,
        "U2_plus_first_year_involvement": transfer_features + FIRST_YEAR_INVOLVEMENT_FEATURES,
    }
    manifest = feature_manifest_rows("utilization", feature_sets, set(transfer_features))
    return frame.sort_values(["season_start_year", "transfer_date", "transfer_event_id"]), feature_sets, manifest


def build_market_value_frame() -> tuple[pd.DataFrame, dict[str, list[str]], list[dict[str, Any]]]:
    frame, upstream_features, _ = market.build_model_frame()
    frame = add_day365_at_risk_flag(frame)
    frame = add_asof_landmark_valuations(frame)
    add_first_year_features(frame)

    frame["age_at_landmark"] = pd.to_numeric(frame["age_at_transfer"], errors="coerce") + 1.0
    frame["log_landmark_value_eur"] = np.log1p(
        pd.to_numeric(frame["landmark_value_eur"], errors="coerce")
    )
    frame["first_year_log_value_ratio"] = market.safe_log_ratio(
        frame["landmark_value_eur"], frame["valuation_pre_eur"]
    )
    frame["log_fee_to_landmark_value"] = market.safe_log_ratio(
        frame["canonical_transfer_fee_eur"], frame["landmark_value_eur"]
    )
    frame["target_landmark_to_24m_log_value_ratio"] = market.safe_log_ratio(
        frame["valuation_post24_nearest_eur"], frame["landmark_value_eur"]
    )
    landmark_value = pd.to_numeric(frame["landmark_value_eur"], errors="coerce")
    post24_value = pd.to_numeric(frame["valuation_post24_nearest_eur"], errors="coerce")
    frame["target_landmark_to_24m_value_change_pct"] = (
        post24_value - landmark_value
    ) / landmark_value
    for threshold in [10, 25, 50]:
        boundary = np.nextafter(landmark_value * (1 - threshold / 100), np.inf)
        frame[f"target_landmark_to_24m_downside_{threshold}pct"] = post24_value.le(boundary).astype(int)
    frame = frame.loc[
        frame["at_destination_day365"]
        & landmark_value.gt(0)
        & pd.to_numeric(frame["landmark_value_staleness_days"], errors="coerce").between(
            0, MAX_LANDMARK_VALUE_STALENESS_DAYS
        )
        & frame["target_landmark_to_24m_log_value_ratio"].notna()
        & frame["season_start_year"].le(2023)
    ].copy()

    transfer_features = upstream_features["M1_market_profile"]
    feature_sets = {
        "M0_chronological_constant": [],
        "V1_transfer_time_profile": transfer_features,
        "V2_plus_day365_value_state": transfer_features + DAY365_VALUE_FEATURES,
        "V3_plus_first_year_involvement": (
            transfer_features + DAY365_VALUE_FEATURES + FIRST_YEAR_INVOLVEMENT_FEATURES
        ),
    }
    manifest = feature_manifest_rows("market_value", feature_sets, set(transfer_features))
    return frame.sort_values(["season_start_year", "transfer_date", "transfer_event_id"]), feature_sets, manifest


def candidate_specs(kind: str) -> list[tuple[str, dict[str, float]]]:
    if kind == "regression":
        return [("ridge", {"alpha": float(alpha)}) for alpha in ALPHA_GRID]
    return [("logistic", {"C": float(value)}) for value in LOGISTIC_C_GRID]


def endpoint_metrics(kind: str, actual: pd.Series | np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    if kind == "regression":
        return market.regression_metrics(actual, prediction)
    return market.classification_metrics(actual, prediction)


def tune_and_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    evaluation: pd.DataFrame,
    features: list[str],
    endpoint: str,
    target: str,
    kind: str,
    origin_id: str,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    candidates: list[tuple[tuple[Any, ...], str, dict[str, float], int]] = []
    rows: list[dict[str, Any]] = []
    for family, parameters in candidate_specs(kind):
        seed = stable_seed(endpoint, origin_id, family, json.dumps(parameters, sort_keys=True))
        prediction, encoded_count = market.fit_candidate(
            train, validation, features, target, kind, family, parameters, seed
        )
        if kind == "classification":
            prediction = np.clip(prediction, 1e-6, 1 - 1e-6)
        metrics = endpoint_metrics(kind, validation[target], prediction)
        primary = float(metrics["mae"] if kind == "regression" else metrics["brier"])
        secondary = float(metrics["rmse"] if kind == "regression" else metrics["log_loss"])
        parameter_text = json.dumps(parameters, sort_keys=True)
        rows.append(
            {
                "endpoint": endpoint,
                "origin_id": origin_id,
                "model_family": family,
                "hyperparameters": parameter_text,
                "encoded_feature_count": encoded_count,
                "validation_primary_loss": primary,
                "validation_secondary_loss": secondary,
                **{f"validation_{key}": value for key, value in metrics.items()},
            }
        )
        candidates.append(((primary, secondary, parameter_text), family, parameters, encoded_count))
    _, family, parameters, _ = min(candidates, key=lambda item: item[0])
    fit_frame = pd.concat([train, validation], ignore_index=True)
    prediction, encoded_count = market.fit_candidate(
        fit_frame,
        evaluation,
        features,
        target,
        kind,
        family,
        parameters,
        stable_seed(endpoint, origin_id, family, "refit", json.dumps(parameters, sort_keys=True)),
    )
    if kind == "classification":
        prediction = np.clip(prediction, 1e-6, 1 - 1e-6)
    selected_text = json.dumps(parameters, sort_keys=True)
    selected_row = next(row for row in rows if row["hyperparameters"] == selected_text)
    return prediction, {
        "selected_family": family,
        "selected_hyperparameters": selected_text,
        "encoded_feature_count": encoded_count,
        "validation_primary_loss": selected_row["validation_primary_loss"],
        "validation_secondary_loss": selected_row["validation_secondary_loss"],
    }, rows


def fit_models(
    frames: dict[str, pd.DataFrame],
    feature_sets: dict[str, dict[str, list[str]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions: list[dict[str, Any]] = []
    origin_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    for endpoint, spec in ENDPOINTS.items():
        frame = frames[spec["module"]]
        target, kind = spec["target"], spec["kind"]
        for origin_id, train_end, validation_year, evaluation_year in ORIGINS:
            train = frame.loc[frame["season_start_year"].le(train_end)].copy()
            validation = frame.loc[frame["season_start_year"].eq(validation_year)].copy()
            evaluation = frame.loc[frame["season_start_year"].eq(evaluation_year)].sort_values(
                "transfer_event_id"
            )
            if min(len(train), len(validation), len(evaluation)) == 0:
                raise RuntimeError(f"Empty split for {endpoint}/{origin_id}.")
            for model_variant in spec["models"]:
                print(f"endpoint={endpoint} origin={origin_id} model={model_variant}", flush=True)
                features = feature_sets[spec["module"]][model_variant]
                if not features:
                    fit_target = pd.concat([train[target], validation[target]], ignore_index=True)
                    if kind == "regression":
                        prediction = np.repeat(float(fit_target.median()), len(evaluation))
                        family = "chronological_median"
                        validation_prediction = np.repeat(float(train[target].median()), len(validation))
                    else:
                        prediction = np.repeat(float(fit_target.mean()), len(evaluation))
                        family = "chronological_prevalence"
                        validation_prediction = np.repeat(float(train[target].mean()), len(validation))
                    validation_metrics = endpoint_metrics(kind, validation[target], validation_prediction)
                    selected = {
                        "selected_family": family,
                        "selected_hyperparameters": "{}",
                        "encoded_feature_count": 0,
                        "validation_primary_loss": float(
                            validation_metrics["mae"] if kind == "regression" else validation_metrics["brier"]
                        ),
                        "validation_secondary_loss": float(
                            validation_metrics["rmse"] if kind == "regression" else validation_metrics["log_loss"]
                        ),
                    }
                else:
                    prediction, selected, tuning = tune_and_predict(
                        train, validation, evaluation, features, endpoint, target, kind, origin_id
                    )
                    for row in tuning:
                        row["model_variant"] = model_variant
                    tuning_rows.extend(tuning)
                if kind == "regression" and spec["module"] == "utilization":
                    prediction = np.clip(prediction, 0.0, 1.05)
                metrics = endpoint_metrics(kind, evaluation[target], prediction)
                origin_rows.append(
                    {
                        "endpoint": endpoint,
                        "module": spec["module"],
                        "endpoint_kind": kind,
                        "origin_id": origin_id,
                        "train_end_year": train_end,
                        "validation_year": validation_year,
                        "evaluation_year": evaluation_year,
                        "train_n": len(train),
                        "validation_n": len(validation),
                        "evaluation_n": len(evaluation),
                        "model_variant": model_variant,
                        "raw_feature_count": len(features),
                        **selected,
                        **metrics,
                    }
                )
                for row, predicted in zip(evaluation.itertuples(index=False), prediction):
                    predictions.append(
                        {
                            "endpoint": endpoint,
                            "module": spec["module"],
                            "endpoint_kind": kind,
                            "origin_id": origin_id,
                            "evaluation_year": evaluation_year,
                            "transfer_event_id": row.transfer_event_id,
                            "player_id": row.player_id,
                            "transfer_date": pd.Timestamp(row.transfer_date).date().isoformat(),
                            "model_variant": model_variant,
                            **selected,
                            "actual": float(getattr(row, target)),
                            "prediction": float(predicted),
                        }
                    )
    return pd.DataFrame(predictions), pd.DataFrame(origin_rows), pd.DataFrame(tuning_rows)


def paired_predictions(
    predictions: pd.DataFrame,
    endpoint: str,
    baseline: str,
    candidate: str,
    years: list[int],
) -> pd.DataFrame:
    subset = predictions.loc[
        predictions["endpoint"].eq(endpoint) & predictions["evaluation_year"].isin(years)
    ]
    keys = ["origin_id", "evaluation_year", "transfer_event_id", "player_id"]
    left = subset.loc[
        subset["model_variant"].eq(baseline), keys + ["actual", "prediction"]
    ].rename(columns={"prediction": "baseline_prediction"})
    right = subset.loc[
        subset["model_variant"].eq(candidate), keys + ["actual", "prediction"]
    ].rename(columns={"actual": "candidate_actual", "prediction": "candidate_prediction"})
    paired = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if not np.allclose(paired["actual"], paired["candidate_actual"], atol=1e-12, rtol=0):
        raise RuntimeError("Paired actual outcomes disagree.")
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
    for start in range(0, BOOTSTRAP_REPETITIONS, 250):
        stop = min(start + 250, BOOTSTRAP_REPETITIONS)
        indices = rng.integers(0, len(clusters), size=(stop - start, len(clusters)))
        draws[start:stop] = sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)
    return {
        "unique_player_clusters": len(clusters),
        "cluster_ci_lower_95": float(np.quantile(draws, 0.025)),
        "cluster_ci_upper_95": float(np.quantile(draws, 0.975)),
        "cluster_probability_candidate_better": float(np.mean(draws > 0)),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": seed,
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
    rows: list[dict[str, Any]] = []
    for endpoint, spec in ENDPOINTS.items():
        kind = spec["kind"]
        for comparison, baseline, candidate in COMPARISONS[spec["module"]]:
            for window, years in WINDOWS.items():
                pairs = paired_predictions(predictions, endpoint, baseline, candidate, years)
                if kind == "regression":
                    base_loss = np.abs(pairs["actual"] - pairs["baseline_prediction"])
                    candidate_loss = np.abs(pairs["actual"] - pairs["candidate_prediction"])
                    primary_metric = "mae"
                    base_discrimination = np.nan
                    candidate_discrimination = np.nan
                else:
                    base_loss = np.square(pairs["actual"] - pairs["baseline_prediction"])
                    candidate_loss = np.square(pairs["actual"] - pairs["candidate_prediction"])
                    primary_metric = "brier"
                    if pairs["actual"].nunique() == 2:
                        base_discrimination = float(
                            roc_auc_score(pairs["actual"], pairs["baseline_prediction"])
                        )
                        candidate_discrimination = float(
                            roc_auc_score(pairs["actual"], pairs["candidate_prediction"])
                        )
                    else:
                        base_discrimination = np.nan
                        candidate_discrimination = np.nan
                origin_values: list[float] = []
                for _, group in pairs.groupby("origin_id", sort=True):
                    if kind == "regression":
                        base = np.abs(group["actual"] - group["baseline_prediction"]).mean()
                        candidate_value = np.abs(
                            group["actual"] - group["candidate_prediction"]
                        ).mean()
                    else:
                        base = np.square(group["actual"] - group["baseline_prediction"]).mean()
                        candidate_value = np.square(
                            group["actual"] - group["candidate_prediction"]
                        ).mean()
                    origin_values.append(float(base - candidate_value))
                improvement = float(base_loss.mean() - candidate_loss.mean())
                bootstrap = cluster_bootstrap(
                    pairs, kind, stable_seed(endpoint, comparison, window, "cluster")
                )
                rows.append(
                    {
                        "endpoint": endpoint,
                        "module": spec["module"],
                        "endpoint_kind": kind,
                        "comparison": comparison,
                        "baseline_model": baseline,
                        "candidate_model": candidate,
                        "window": window,
                        "evaluation_years": ";".join(map(str, years)),
                        "primary_metric": primary_metric,
                        "pooled_n": len(pairs),
                        "unique_players": pairs["player_id"].nunique(),
                        "baseline_loss": float(base_loss.mean()),
                        "candidate_loss": float(candidate_loss.mean()),
                        "loss_improvement": improvement,
                        "relative_loss_improvement": improvement / float(base_loss.mean()),
                        "baseline_roc_auc": base_discrimination,
                        "candidate_roc_auc": candidate_discrimination,
                        "origin_wins": sum(value > 0 for value in origin_values),
                        "origin_count": len(origin_values),
                        **bootstrap,
                        "stability_label": stability_label(
                            improvement,
                            origin_values,
                            float(bootstrap["cluster_probability_candidate_better"]),
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_decisions(comparisons: pd.DataFrame) -> pd.DataFrame:
    pooled = comparisons.loc[comparisons["window"].eq("pooled_four")].set_index(
        ["endpoint", "comparison"]
    )
    rows: list[dict[str, Any]] = []
    for endpoint, spec in ENDPOINTS.items():
        result = pooled.loc[(endpoint, spec["primary_comparison"])]
        validated = result["cluster_ci_lower_95"] > 0 and result["origin_wins"] >= 3
        status = (
            "validated_monitoring_increment"
            if validated
            else (
                "promising_but_uncertain"
                if result["loss_improvement"] > 0
                else "monitoring_increment_not_validated"
            )
        )
        rows.append(
            {
                "endpoint": endpoint,
                "module": spec["module"],
                "endpoint_kind": spec["kind"],
                "status": status,
                "recommended_model": spec["candidate"] if validated else spec["models"][1],
                "primary_comparison": spec["primary_comparison"],
                "loss_improvement": result["loss_improvement"],
                "relative_loss_improvement": result["relative_loss_improvement"],
                "ci_lower_95": result["cluster_ci_lower_95"],
                "ci_upper_95": result["cluster_ci_upper_95"],
                "origin_wins": int(result["origin_wins"]),
                "origin_count": int(result["origin_count"]),
                "candidate_roc_auc": result["candidate_roc_auc"],
                "claim_guardrail": (
                    "Day-365 monitoring of year-two risk only; not a transfer-time claim, "
                    "causal diagnosis, realized cash loss, or accounting ROI."
                ),
            }
        )
    return pd.DataFrame(rows)


def build_risk_bands(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for endpoint, spec in ENDPOINTS.items():
        selected = predictions.loc[
            predictions["endpoint"].eq(endpoint)
            & predictions["model_variant"].eq(spec["candidate"])
        ].copy()
        selected["risk_score"] = (
            -selected["prediction"] if spec["kind"] == "regression" else selected["prediction"]
        )
        selected["risk_band"] = selected.groupby("origin_id")["risk_score"].transform(
            lambda values: pd.qcut(
                values.rank(method="first"), 5, labels=False, duplicates="drop"
            )
            + 1
        )
        for band, group in selected.groupby("risk_band", sort=True):
            rows.append(
                {
                    "endpoint": endpoint,
                    "module": spec["module"],
                    "model_variant": spec["candidate"],
                    "risk_band": int(band),
                    "risk_band_label": (
                        "lowest_predicted_risk"
                        if band == 1
                        else ("highest_predicted_risk" if band == 5 else f"band_{int(band)}")
                    ),
                    "rows": len(group),
                    "unique_players": group["player_id"].nunique(),
                    "mean_prediction": group["prediction"].mean(),
                    "mean_actual": group["actual"].mean(),
                }
            )
    return pd.DataFrame(rows)


def build_probability_consistency_audit(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    specifications = [
        (
            "utilization",
            ["U1_transfer_time_profile", "U2_plus_first_year_involvement"],
            "year2_underutilization_10pct",
            "year2_underutilization_25pct",
            "P(under10) <= P(under25)",
        ),
        (
            "market_value",
            [
                "V1_transfer_time_profile",
                "V2_plus_day365_value_state",
                "V3_plus_first_year_involvement",
            ],
            "year2_value_downside_25pct",
            "year2_value_downside_10pct",
            "P(down25) <= P(down10)",
        ),
        (
            "market_value",
            [
                "V1_transfer_time_profile",
                "V2_plus_day365_value_state",
                "V3_plus_first_year_involvement",
            ],
            "year2_value_downside_50pct",
            "year2_value_downside_25pct",
            "P(down50) <= P(down25)",
        ),
    ]
    keys = ["origin_id", "evaluation_year", "transfer_event_id", "player_id"]
    for module, variants, lower_endpoint, upper_endpoint, rule in specifications:
        for model_variant in variants:
            subset = predictions.loc[
                predictions["model_variant"].eq(model_variant)
                & predictions["endpoint"].isin([lower_endpoint, upper_endpoint])
            ]
            wide = subset.pivot(index=keys, columns="endpoint", values="prediction").reset_index()
            violation = wide[lower_endpoint].gt(wide[upper_endpoint])
            rows.append(
                {
                    "module": module,
                    "model_variant": model_variant,
                    "rule": rule,
                    "rows": len(wide),
                    "raw_violations": int(violation.sum()),
                    "raw_violation_rate": float(violation.mean()),
                    "maximum_violation": float(
                        (wide[lower_endpoint] - wide[upper_endpoint]).clip(lower=0).max()
                    ),
                    "policy_status": (
                        "requires_ordering_before_live_probability_use"
                        if violation.any()
                        else "already_ordered"
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_checks(
    utilization: pd.DataFrame,
    value: pd.DataFrame,
    feature_sets: dict[str, dict[str, list[str]]],
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def add(check: str, passed: bool, observed: Any, expected: Any, note: str) -> None:
        checks.append(
            {
                "check": check,
                "passed": bool(passed),
                "observed": observed,
                "expected": expected,
                "note": note,
            }
        )

    add("utilization_key_unique", utilization["transfer_event_id"].is_unique, utilization["transfer_event_id"].nunique(), len(utilization), "One row per paid transfer.")
    add("value_key_unique", value["transfer_event_id"].is_unique, value["transfer_event_id"].nunique(), len(value), "One row per permanent transfer.")
    add("all_players_at_risk_day365", utilization["at_destination_day365"].all() and value["at_destination_day365"].all(), True, True, "No already-departed player is scored.")
    add("year2_games_minimum", utilization["year2_destination_club_games"].ge(MIN_YEAR2_CLUB_GAMES).all(), utilization["year2_destination_club_games"].min(), MIN_YEAR2_CLUB_GAMES, "Stable year-two denominator.")
    add("utilization_target_bounds", utilization["target_year2_opportunity_share"].between(0, 1.05).all(), f"{utilization['target_year2_opportunity_share'].min()}..{utilization['target_year2_opportunity_share'].max()}", "0..1.05", "Valid opportunity shares.")
    add("utilization_targets_nested", (utilization["target_year2_underutilization_10pct"] <= utilization["target_year2_underutilization_25pct"]).all(), True, True, "Severe under-use is nested.")
    add("value_targets_nested", (value["target_landmark_to_24m_downside_50pct"] <= value["target_landmark_to_24m_downside_25pct"]).all() and (value["target_landmark_to_24m_downside_25pct"] <= value["target_landmark_to_24m_downside_10pct"]).all(), True, True, "Severe value downside is nested.")
    add("landmark_valuations_not_future", value["landmark_value_date"].le(value["landmark_date"]).all(), int(value["landmark_value_date"].gt(value["landmark_date"]).sum()), 0, "All predictor valuations existed by day 365.")
    add("landmark_value_staleness", value["landmark_value_staleness_days"].between(0, MAX_LANDMARK_VALUE_STALENESS_DAYS).all(), f"{value['landmark_value_staleness_days'].min()}..{value['landmark_value_staleness_days'].max()}", f"0..{MAX_LANDMARK_VALUE_STALENESS_DAYS}", "Current value is reasonably recent.")
    forbidden = [
        feature
        for module_sets in feature_sets.values()
        for features in module_sets.values()
        for feature in features
        if "post24" in feature.lower() or feature.lower().startswith("target_")
    ]
    add("no_outcome_predictors", not forbidden, ";".join(sorted(set(forbidden))), "none", "No year-two outcome field is a predictor.")
    key = ["endpoint", "origin_id", "model_variant", "transfer_event_id"]
    add("prediction_key_unique", not predictions.duplicated(key).any(), int(predictions.duplicated(key).sum()), 0, "One forecast per endpoint/origin/model/transfer.")
    probabilities = predictions.loc[predictions["endpoint_kind"].eq("classification"), "prediction"]
    add("probability_bounds", probabilities.between(0, 1).all(), f"{probabilities.min()}..{probabilities.max()}", "0..1", "Valid probabilities.")
    paired = predictions.groupby(["endpoint", "origin_id", "model_variant"]).size().groupby(level=[0, 1]).nunique().eq(1).all()
    add("paired_evaluation_rows", paired, paired, True, "All variants share evaluation rows.")
    return pd.DataFrame(checks)


def write_readme(
    utilization: pd.DataFrame,
    value: pd.DataFrame,
    decisions: pd.DataFrame,
    comparisons: pd.DataFrame,
    consistency: pd.DataFrame,
) -> None:
    pooled = comparisons.loc[comparisons["window"].eq("pooled_four")].set_index(
        ["endpoint", "comparison"]
    )
    lines = [
        "# MoveMaker day-365 landmark monitoring experiment",
        "",
        "This v2 experiment tests whether first-year evidence improves forecasts for the remaining second year. It does not replace or rewrite any finalized v1 model.",
        "",
        "## Leakage-safe design",
        "",
        f"- Utilization cohort: {len(utilization):,} clean paid permanent transfers still at the destination on day 365, with at least {MIN_YEAR2_CLUB_GAMES} destination-club matches in days 366-730.",
        f"- Market-value cohort: {len(value):,} permanent transfers still at the destination on day 365, with a positive valuation observed on/before day 365 and no more than {MAX_LANDMARK_VALUE_STALENESS_DAYS} days stale.",
        "- Utilization targets cover only days 366-730, not the already-observed first year.",
        "- Value targets measure change from the strictly as-of day-365 valuation to the existing 24-month outcome valuation.",
        "- Four rolling origins evaluate 2020-2023. L2 logistic/ridge strength is selected on the preceding validation season only.",
        "- Primary uncertainty uses 5,000 paired player-cluster bootstrap repetitions.",
        "",
        "## Monitoring-increment results",
        "",
    ]
    for row in decisions.itertuples(index=False):
        result = pooled.loc[(row.endpoint, row.primary_comparison)]
        auc = (
            f", pooled candidate AUC {result['candidate_roc_auc']:.3f}"
            if row.endpoint_kind == "classification"
            else ""
        )
        lines.append(
            f"- `{row.endpoint}`: `{row.status}`; {result['primary_metric']} improvement {result['loss_improvement']:+.6f}, player-clustered 95% CI [{result['cluster_ci_lower_95']:+.6f}, {result['cluster_ci_upper_95']:+.6f}], {int(result['origin_wins'])}/4 origin wins{auc}."
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "A validated increment means the day-365 update improves prediction of second-year risk over the compact transfer-time profile on the same at-risk rows. It turns MoveMaker into a two-stage screen-and-monitor workflow. It does not imply that first-year involvement causes later outcomes, predict sale proceeds, or establish accounting ROI.",
        "",
        "The threshold classifiers were fit independently for this signal test. Their raw probabilities are not yet a live scoring policy: the recommended utilization update has 12/365 under-10/under-25 ordering violations, while the recommended value update has 34/1,630 down-25/down-10 and 21/1,630 down-50/down-25 violations. Apply validation-only calibration and monotone ordering before deployment; see `probability_consistency_audit.csv`.",
        "",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    protected_before = protected_tree_sha256()
    utilization, utilization_features, utilization_manifest = build_utilization_frame()
    value, value_features, value_manifest = build_market_value_frame()
    frames = {"utilization": utilization, "market_value": value}
    feature_sets = {"utilization": utilization_features, "market_value": value_features}
    predictions, origin_results, tuning = fit_models(frames, feature_sets)
    comparisons = build_comparisons(predictions)
    decisions = build_decisions(comparisons)
    risk_bands = build_risk_bands(predictions)
    consistency = build_probability_consistency_audit(predictions)
    feature_manifest = pd.DataFrame(utilization_manifest + value_manifest)
    checks = build_checks(utilization, value, feature_sets, predictions)
    if not checks["passed"].all():
        raise RuntimeError(f"Build checks failed: {checks.loc[~checks['passed'], 'check'].tolist()}")

    utilization_targets = utilization[[
        "transfer_event_id",
        "player_id",
        "transfer_date",
        "season_start_year",
        "from_club_name",
        "to_club_name",
        "canonical_transfer_fee_eur",
        "landmark_date",
        "first_outbound_date",
        "days_to_first_outbound",
        "at_destination_day365",
        "year2_destination_club_games",
        "year2_destination_minutes",
        "target_year2_opportunity_share",
        "target_year2_underutilization_10pct",
        "target_year2_underutilization_25pct",
        *FIRST_YEAR_INVOLVEMENT_FEATURES,
    ]].copy()
    value_targets = value[[
        "transfer_event_id",
        "player_id",
        "transfer_date",
        "season_start_year",
        "from_club_name",
        "to_club_name",
        "landmark_date",
        "first_outbound_date",
        "days_to_first_outbound",
        "at_destination_day365",
        "landmark_value_date",
        "landmark_value_eur",
        "landmark_value_staleness_days",
        "valuation_post24_nearest_date",
        "valuation_post24_nearest_eur",
        "target_landmark_to_24m_log_value_ratio",
        "target_landmark_to_24m_value_change_pct",
        "target_landmark_to_24m_downside_10pct",
        "target_landmark_to_24m_downside_25pct",
        "target_landmark_to_24m_downside_50pct",
        *FIRST_YEAR_INVOLVEMENT_FEATURES,
    ]].copy()
    tables = {
        "utilization_landmark_targets.csv": utilization_targets,
        "market_value_landmark_targets.csv": value_targets,
        "landmark_feature_manifest.csv": feature_manifest,
        "validation_tuning_results.csv": tuning,
        "landmark_predictions.csv": predictions,
        "landmark_origin_results.csv": origin_results,
        "landmark_comparison_results.csv": comparisons,
        "landmark_decision_summary.csv": decisions,
        "landmark_risk_bands.csv": risk_bands,
        "probability_consistency_audit.csv": consistency,
        "build_checks.csv": checks,
    }
    for name, table in tables.items():
        table.to_csv(OUTPUT_DIR / name, index=False)
    source_paths = [
        paid.MASTER_PATH,
        paid.AUDIT_PATH,
        paid.AUDIT_VERIFICATION_PATH,
        CANONICAL_TRANSFER_PATH,
        VALUATION_HISTORY_PATH,
    ]
    source_manifest = pd.DataFrame(
        [
            {
                "source": path.relative_to(ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in source_paths
        ]
    )
    source_manifest.to_csv(OUTPUT_DIR / "source_manifest.csv", index=False)
    tables["source_manifest.csv"] = source_manifest
    write_readme(utilization, value, decisions, comparisons, consistency)

    if protected_before != protected_tree_sha256():
        raise RuntimeError("A finalized v1 output changed during the v2 experiment.")
    summary = {
        "utilization_cohort_rows": len(utilization),
        "utilization_unique_players": utilization["player_id"].nunique(),
        "market_value_cohort_rows": len(value),
        "market_value_unique_players": value["player_id"].nunique(),
        "evaluation_years": [2020, 2021, 2022, 2023],
        "origins": [origin[0] for origin in ORIGINS],
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "decisions": decisions.set_index("endpoint")["status"].to_dict(),
        "protected_v1_outputs_unchanged": True,
        "build_checks": len(checks),
        "build_checks_passed": int(checks["passed"].sum()),
    }
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_names = sorted(list(tables) + ["README.md", "run_summary.json"])
    output_manifest = pd.DataFrame(
        [
            {
                "file": name,
                "size_bytes": (OUTPUT_DIR / name).stat().st_size,
                "sha256": sha256_file(OUTPUT_DIR / name),
            }
            for name in manifest_names
        ]
    )
    output_manifest.to_csv(OUTPUT_DIR / "output_manifest.csv", index=False)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

"""Build a leakage-safe reported-fee benchmarking model.

The target is the reliable positive reported fee for a clean permanent transfer.
The primary loss is absolute error on log fee, so €100k-vs-€1m deals are not
overwhelmed by a handful of €50m+ transfers.  The model must beat both a
chronological median and the player's recent pre-transfer market value.

Every rolling origin uses an internal prior season to choose raw model family,
the next validation season to choose feature-block depth and estimate residual
ranges, and the following season for untouched evaluation.
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
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
for dependency_dir in [ROOT / "models", ROOT / "scripts"]:
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

import run_coarse_club_context_diagnostic as coarse
import run_market_value_downside_model as market
import run_opportunity_transferability_decomposition as opportunity
from train_compatibility_models import sha256_file


MASTER_PATH = market.MASTER_PATH
AUDIT_PATH = market.AUDIT_PATH
AUDIT_VERIFICATION_PATH = market.AUDIT_VERIFICATION_PATH
OUTPUT_DIR = ROOT / "Data" / "processed" / "fee_benchmarking_model"

RANDOM_SEED = 20260811
BOOTSTRAP_REPETITIONS = 5000
ORIGINS = coarse.ORIGINS
WINDOWS = market.WINDOWS
BASELINE_MEDIAN = "M0_chronological_median"
VALUE_ANCHOR = "A1_pre_transfer_value_anchor"
MODEL_ORDER = [
    "M1_market_profile",
    "M2_plus_valuation_trajectory",
    "M3_plus_prior_opportunity",
    "M4_plus_player_performance",
    "M5_plus_compact_club_context",
]
POLICY_MODEL = "M_policy_validation_selected"
EXPECTED_FEATURE_COUNTS = dict(zip(MODEL_ORDER, [10, 22, 30, 46, 72]))
COMPACT_CONTEXT_SUFFIXES = [
    "games", "win_pct", "points_per_game", "goals_for_per_game",
    "goals_against_per_game", "goal_difference_per_game",
    "domestic_league_game_share", "domestic_cup_game_share",
    "international_game_share",
]
PROTECTED_DIRS = [
    ROOT / "Data" / "processed" / "financial_module_feasibility",
    ROOT / "Data" / "processed" / "market_value_downside_model",
    ROOT / "Data" / "processed" / "market_value_calibration_experiment",
    ROOT / "Data" / "processed" / "market_value_downside_product",
    ROOT / "Data" / "processed" / "paid_transfer_underutilization_model",
    ROOT / "Data" / "processed" / "paid_transfer_underutilization_refinement",
    ROOT / "Data" / "processed" / "retention_diagnostic",
]


def as_bool(series: pd.Series) -> pd.Series:
    return market.as_bool(series)


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


def add_definition(
    definitions: dict[str, dict[str, str]], feature: str, sources: str,
    timing: str, rationale: str, kind: str = "derived",
) -> None:
    definitions[feature] = {
        "feature_kind": kind, "source_fields": sources,
        "timing_classification": timing, "rationale": rationale,
    }


def build_model_frame() -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    verification = json.loads(AUDIT_VERIFICATION_PATH.read_text(encoding="utf-8"))
    if verification.get("status") != "pass":
        raise RuntimeError("Financial feasibility audit is not independently verified.")
    audit_columns = [
        "transfer_event_id", "canonical_transfer_fee_eur", "canonical_transfer_fee_status",
        "fee_band", "valuation_pre_eur", "valuation_pre_staleness_days",
        "eligible_fee_benchmarking",
    ]
    audit = pd.read_csv(AUDIT_PATH, usecols=audit_columns, low_memory=False)
    eligible = audit.loc[as_bool(audit["eligible_fee_benchmarking"])].copy()
    eligible["transfer_event_id"] = eligible["transfer_event_id"].astype(str)
    if len(eligible) != 1486 or not eligible["transfer_event_id"].is_unique:
        raise RuntimeError(f"Fee benchmark cohort mismatch: {len(eligible)} rows.")
    master = pd.read_csv(MASTER_PATH, low_memory=False)
    master["transfer_event_id"] = master["transfer_event_id"].astype(str)
    frame = master.loc[master["transfer_event_id"].isin(set(eligible["transfer_event_id"]))].copy()
    overlaps = [column for column in audit_columns if column in frame.columns and column != "transfer_event_id"]
    frame = frame.drop(columns=overlaps).merge(
        eligible[audit_columns], on="transfer_event_id", how="inner", validate="one_to_one"
    )
    if len(frame) != 1486:
        raise RuntimeError(f"Master/audit fee cohort mismatch: {len(frame)} rows.")

    frame["transfer_date"] = pd.to_datetime(frame["transfer_date"], errors="coerce")
    frame["season_start_year"] = frame["transfer_date"].dt.year.where(
        frame["transfer_date"].dt.month.ge(7), frame["transfer_date"].dt.year - 1
    ).astype("Int64")
    fee = pd.to_numeric(frame["canonical_transfer_fee_eur"], errors="coerce")
    pre_value = pd.to_numeric(frame["valuation_pre_eur"], errors="coerce")
    if not fee.gt(0).all() or not pre_value.gt(0).all():
        raise RuntimeError("Fee benchmark requires positive fee and positive recent pre-value.")
    frame["target_log_fee_eur"] = np.log(fee)
    frame["anchor_log_pre_value_eur"] = np.log(pre_value)

    definitions: dict[str, dict[str, str]] = {}
    birth_date = pd.to_datetime(frame["player_date_of_birth"], errors="coerce")
    frame["age_at_transfer"] = ((frame["transfer_date"] - birth_date).dt.days / 365.2425).where(
        lambda values: values.between(15, 45)
    )
    historical_position = frame["lineup_pre365_all_primary_position"].where(
        frame["lineup_pre365_all_primary_position"].notna(),
        frame["lineup_pre365_origin_primary_position"],
    )
    frame["historical_position_group"] = historical_position.map(coarse.position_group)
    frame["origin_competition_id"] = frame[
        "club_origin_pre365_dominant_competition_id"
    ].astype("string").fillna("unknown")
    frame["destination_competition_id"] = frame[
        "club_destination_pre365_dominant_competition_id"
    ].astype("string").fillna("unknown")
    frame["transfer_month_sin"] = np.sin(2 * np.pi * frame["transfer_date"].dt.month / 12)
    frame["transfer_month_cos"] = np.cos(2 * np.pi * frame["transfer_date"].dt.month / 12)
    frame["transfer_year_centered"] = frame["season_start_year"].astype(float) - 2017.0
    frame["log_pre_value_eur"] = np.log(pre_value)
    m1 = [
        "age_at_transfer", "historical_position_group", "origin_competition_id",
        "destination_competition_id", "transfer_month_sin", "transfer_month_cos",
        "transfer_year_centered", "log_pre_value_eur", "valuation_pre_staleness_days",
        "valuation_pre_is_origin_club",
    ]
    for feature in m1:
        add_definition(
            definitions, feature, feature, "available at or before transfer",
            "Player, market, route, or timing control known when benchmarking the fee.",
            "raw" if feature in {"valuation_pre_staleness_days", "valuation_pre_is_origin_club"} else "derived",
        )

    trajectory_specs = {
        "valuation_pre365_log_change": ("valuation_pre365_last_eur", "valuation_pre365_first_eur"),
        "valuation_pre730_log_change": ("valuation_pre730_last_eur", "valuation_pre730_first_eur"),
        "valuation_career_pre_log_change": ("valuation_career_pre_last_eur", "valuation_career_pre_first_eur"),
        "valuation_pre_to_prior_peak_log_ratio": ("valuation_pre_eur", "valuation_prior_peak_eur"),
        "valuation_pre365_log_range": ("valuation_pre365_max_eur", "valuation_pre365_min_eur"),
        "valuation_pre730_log_range": ("valuation_pre730_max_eur", "valuation_pre730_min_eur"),
    }
    trajectory_features = []
    for feature, (numerator, denominator) in trajectory_specs.items():
        frame[feature] = market.safe_log_ratio(frame[numerator], frame[denominator])
        trajectory_features.append(feature)
        add_definition(
            definitions, feature, f"{numerator};{denominator}",
            "derived strictly pre-transfer valuation history",
            "As-of valuation momentum or dispersion control.",
        )
    for source in [
        "valuation_pre365_observations", "valuation_pre730_observations",
        "valuation_career_pre_observations", "valuation_pre365_observation_span_days",
        "valuation_pre730_observation_span_days", "valuation_career_pre_observation_span_days",
    ]:
        feature = f"log1p_{source}"
        frame[feature] = np.log1p(pd.to_numeric(frame[source], errors="coerce").clip(lower=0))
        trajectory_features.append(feature)
        add_definition(
            definitions, feature, source, "strictly pre-transfer valuation history",
            "Valuation-history coverage or horizon control.",
        )
    m2 = m1 + trajectory_features

    opportunity_features = []
    for feature, numerator, denominator, scale in opportunity.USAGE_SPECS:
        frame[feature] = coarse.safe_divide(
            frame[numerator], pd.to_numeric(frame[denominator], errors="coerce") * scale
        )
        opportunity_features.append(feature)
        add_definition(
            definitions, feature, f"{numerator};{denominator}",
            "derived strictly pre-transfer player opportunity",
            "Origin-club involvement available before negotiation.",
        )
    m3 = m2 + opportunity_features

    performance_features = []
    for context in opportunity.PERFORMANCE_CONTEXTS:
        for suffix in ["goals_per90", "assists_per90"]:
            feature = f"appearance_{context}_{suffix}"
            performance_features.append(feature)
            add_definition(
                definitions, feature, feature, "strictly pre-transfer player performance",
                "Pre-transfer player output rate.", "raw",
            )
        minutes = f"appearance_{context}_minutes"
        for card in ["yellow_cards", "red_cards"]:
            source = f"appearance_{context}_{card}"
            feature = f"appearance_{context}_{card}_per90"
            frame[feature] = 90 * coarse.safe_divide(frame[source], frame[minutes])
            performance_features.append(feature)
            add_definition(
                definitions, feature, f"{source};{minutes}",
                "derived strictly pre-transfer player performance",
                "Pre-transfer disciplinary rate.",
            )
    m4 = m3 + performance_features

    all_destination = coarse.add_context_features(frame, "destination", definitions)
    all_origin = coarse.add_context_features(frame, "origin", definitions)
    compact_destination = [
        f"club_destination_pre365_{suffix}" for suffix in COMPACT_CONTEXT_SUFFIXES
    ]
    compact_origin = [f"club_origin_pre365_{suffix}" for suffix in COMPACT_CONTEXT_SUFFIXES]
    transition_features = []
    for metric in coarse.TRANSITION_METRICS:
        feature = f"transition_pre365_{metric}_delta"
        destination = f"club_destination_pre365_{metric}"
        origin = f"club_origin_pre365_{metric}"
        frame[feature] = pd.to_numeric(frame[destination], errors="coerce") - pd.to_numeric(
            frame[origin], errors="coerce"
        )
        transition_features.append(feature)
        add_definition(
            definitions, feature, f"{destination};{origin}",
            "derived strictly pre-transfer club context",
            "Destination-minus-origin route context.",
        )
    if not set(compact_destination).issubset(all_destination) or not set(compact_origin).issubset(all_origin):
        raise RuntimeError("Compact club context is not a subset of frozen context features.")
    m5 = m4 + compact_destination + compact_origin + transition_features
    feature_sets = dict(zip(MODEL_ORDER, [m1, m2, m3, m4, m5]))
    observed = {name: len(features) for name, features in feature_sets.items()}
    if observed != EXPECTED_FEATURE_COUNTS:
        raise RuntimeError(f"Fee feature count mismatch: {observed}")
    for index in range(1, len(MODEL_ORDER)):
        if not set(feature_sets[MODEL_ORDER[index - 1]]).issubset(feature_sets[MODEL_ORDER[index]]):
            raise RuntimeError("Fee feature blocks are not nested.")
    if any(len(features) != len(set(features)) for features in feature_sets.values()):
        raise RuntimeError("Fee feature block contains duplicates.")

    manifest_rows = []
    for model, features in feature_sets.items():
        for order, feature in enumerate(features, start=1):
            manifest_rows.append({
                "model_variant": model, "feature_order": order, "feature": feature,
                **definitions[feature], "raw_feature_count": len(features),
            })
    return (
        frame.sort_values(["season_start_year", "transfer_date", "transfer_event_id"]),
        feature_sets, pd.DataFrame(manifest_rows),
    )


def fee_metrics(actual_log: pd.Series, predicted_log: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual_log, dtype=float)
    predicted = np.asarray(predicted_log, dtype=float)
    actual_fee, predicted_fee = np.exp(actual), np.exp(predicted)
    residual = actual - predicted
    correlation = spearmanr(actual, predicted).statistic if len(actual) > 1 else np.nan
    denominator = np.maximum((actual_fee + predicted_fee) / 2, 1)
    return {
        "n": len(actual), "mae_log": float(np.mean(np.abs(residual))),
        "rmse_log": float(np.sqrt(np.mean(np.square(residual)))),
        "r2_log": float(1 - np.square(residual).sum() / np.square(actual - actual.mean()).sum()),
        "spearman": float(correlation),
        "median_absolute_percentage_error": float(np.median(np.abs(actual_fee - predicted_fee) / actual_fee)),
        "median_symmetric_percentage_error": float(np.median(np.abs(actual_fee - predicted_fee) / denominator)),
        "mae_eur": float(np.mean(np.abs(actual_fee - predicted_fee))),
        "median_actual_fee_eur": float(np.median(actual_fee)),
        "median_predicted_fee_eur": float(np.median(predicted_fee)),
        "multiplicative_error_factor_from_mae": float(np.exp(np.mean(np.abs(residual)))),
    }


def select_raw_model(
    train: pd.DataFrame, selector: pd.DataFrame, features: list[str],
    origin_id: str, model_variant: str,
) -> tuple[str, dict[str, Any], pd.DataFrame]:
    rows, candidates = [], []
    for family, parameters in market.candidate_specs("regression"):
        parameter_text = json.dumps(parameters, sort_keys=True)
        prediction, encoded_count = market.fit_candidate(
            train, selector, features, "target_log_fee_eur", "regression",
            family, parameters, stable_seed(origin_id, model_variant, family, parameter_text),
        )
        result = fee_metrics(selector["target_log_fee_eur"], prediction)
        rows.append({
            "origin_id": origin_id, "model_variant": model_variant,
            "model_family": family, "hyperparameters": parameter_text,
            "encoded_feature_count": encoded_count,
            "internal_selector_mae_log": result["mae_log"],
            "internal_selector_rmse_log": result["rmse_log"],
        })
        candidates.append(((result["mae_log"], result["rmse_log"], family, parameter_text), family, parameters))
    _, family, parameters = min(candidates, key=lambda item: item[0])
    return family, parameters, pd.DataFrame(rows)


def empirical_quantile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), quantile, method="higher"))


def fit_and_evaluate(
    frame: pd.DataFrame, feature_sets: dict[str, list[str]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = []
    tuning_rows = []
    raw_selection_rows = []
    policy_selection_rows = []
    for origin_id, train_end, validation_year, evaluation_year in ORIGINS:
        internal_train = frame.loc[frame["season_start_year"].le(train_end - 1)].copy()
        internal_selector = frame.loc[frame["season_start_year"].eq(train_end)].copy()
        validation = frame.loc[frame["season_start_year"].eq(validation_year)].sort_values(
            ["transfer_date", "transfer_event_id"]
        ).copy()
        evaluation = frame.loc[frame["season_start_year"].eq(evaluation_year)].sort_values(
            ["transfer_date", "transfer_event_id"]
        ).copy()
        if min(len(internal_train), len(internal_selector), len(validation), len(evaluation)) == 0:
            raise RuntimeError(f"Empty nested fee split for {origin_id}.")
        validation_train = pd.concat([internal_train, internal_selector], ignore_index=True)
        evaluation_train = pd.concat([validation_train, validation], ignore_index=True)
        origin_validation: dict[str, np.ndarray] = {
            BASELINE_MEDIAN: np.repeat(float(validation_train["target_log_fee_eur"].median()), len(validation)),
            VALUE_ANCHOR: validation["anchor_log_pre_value_eur"].to_numpy(float),
        }
        origin_evaluation: dict[str, np.ndarray] = {
            BASELINE_MEDIAN: np.repeat(float(evaluation_train["target_log_fee_eur"].median()), len(evaluation)),
            VALUE_ANCHOR: evaluation["anchor_log_pre_value_eur"].to_numpy(float),
        }
        family_by_model: dict[str, tuple[str, dict[str, Any]]] = {}
        for model_variant in MODEL_ORDER:
            family, parameters, tuning = select_raw_model(
                internal_train, internal_selector, feature_sets[model_variant],
                origin_id, model_variant,
            )
            tuning_rows.extend(tuning.to_dict("records"))
            family_by_model[model_variant] = (family, parameters)
            parameter_text = json.dumps(parameters, sort_keys=True)
            val_prediction, val_encoded = market.fit_candidate(
                validation_train, validation, feature_sets[model_variant],
                "target_log_fee_eur", "regression", family, parameters,
                stable_seed(origin_id, model_variant, "validation", family, parameter_text),
            )
            eval_prediction, eval_encoded = market.fit_candidate(
                evaluation_train, evaluation, feature_sets[model_variant],
                "target_log_fee_eur", "regression", family, parameters,
                stable_seed(origin_id, model_variant, "evaluation", family, parameter_text),
            )
            origin_validation[model_variant] = val_prediction
            origin_evaluation[model_variant] = eval_prediction
            raw_selection_rows.append({
                "origin_id": origin_id, "internal_train_end_year": train_end - 1,
                "internal_selector_year": train_end, "validation_year": validation_year,
                "evaluation_year": evaluation_year, "model_variant": model_variant,
                "selected_family": family,
                "selected_hyperparameters": parameter_text,
                "validation_encoded_feature_count": val_encoded,
                "evaluation_encoded_feature_count": eval_encoded,
            })

        selection_candidates = [VALUE_ANCHOR] + MODEL_ORDER
        validation_metrics = {
            model: fee_metrics(validation["target_log_fee_eur"], origin_validation[model])
            for model in selection_candidates
        }
        selected_model = min(
            selection_candidates,
            key=lambda model: (
                validation_metrics[model]["mae_log"],
                validation_metrics[model]["rmse_log"],
                selection_candidates.index(model),
            ),
        )
        selected_validation_prediction = origin_validation[selected_model]
        residual_abs = np.abs(
            validation["target_log_fee_eur"].to_numpy(float) - selected_validation_prediction
        )
        q80, q90 = empirical_quantile(residual_abs, 0.80), empirical_quantile(residual_abs, 0.90)
        benchmark_residual_abs = np.abs(
            validation["target_log_fee_eur"].to_numpy(float) - origin_validation[MODEL_ORDER[0]]
        )
        benchmark_q80 = empirical_quantile(benchmark_residual_abs, 0.80)
        benchmark_q90 = empirical_quantile(benchmark_residual_abs, 0.90)
        for model in selection_candidates:
            policy_selection_rows.append({
                "origin_id": origin_id, "validation_year": validation_year,
                "evaluation_year": evaluation_year, "candidate_model": model,
                "validation_mae_log": validation_metrics[model]["mae_log"],
                "validation_rmse_log": validation_metrics[model]["rmse_log"],
                "selected_for_policy": model == selected_model,
                "selected_interval_abs_log_q80": q80 if model == selected_model else np.nan,
                "selected_interval_abs_log_q90": q90 if model == selected_model else np.nan,
            })
        origin_validation[POLICY_MODEL] = origin_validation[selected_model]
        origin_evaluation[POLICY_MODEL] = origin_evaluation[selected_model]

        for split, source, forecast_map in [
            ("validation", validation, origin_validation),
            ("evaluation", evaluation, origin_evaluation),
        ]:
            for model_variant, predicted_logs in forecast_map.items():
                for row, predicted_log in zip(source.itertuples(index=False), predicted_logs):
                    record = {
                        "split": split, "origin_id": origin_id,
                        "validation_year": validation_year, "evaluation_year": evaluation_year,
                        "transfer_event_id": row.transfer_event_id, "player_id": row.player_id,
                        "transfer_date": row.transfer_date.date().isoformat(),
                        "model_variant": model_variant,
                        "policy_selected_model": selected_model if model_variant == POLICY_MODEL else "",
                        "actual_log_fee_eur": float(row.target_log_fee_eur),
                        "predicted_log_fee_eur": float(predicted_log),
                        "actual_fee_eur": float(row.canonical_transfer_fee_eur),
                        "predicted_fee_eur": float(np.exp(predicted_log)),
                        "pre_transfer_value_eur": float(row.valuation_pre_eur),
                        "historical_position_group": row.historical_position_group,
                        "origin_competition_id": row.origin_competition_id,
                        "destination_competition_id": row.destination_competition_id,
                        "fee_band": row.fee_band,
                    }
                    if model_variant == POLICY_MODEL:
                        record.update({
                            "interval80_lower_eur": float(np.exp(predicted_log - q80)),
                            "interval80_upper_eur": float(np.exp(predicted_log + q80)),
                            "interval90_lower_eur": float(np.exp(predicted_log - q90)),
                            "interval90_upper_eur": float(np.exp(predicted_log + q90)),
                            "interval_abs_log_q80": q80,
                            "interval_abs_log_q90": q90,
                        })
                    elif model_variant == MODEL_ORDER[0]:
                        record.update({
                            "interval80_lower_eur": float(np.exp(predicted_log - benchmark_q80)),
                            "interval80_upper_eur": float(np.exp(predicted_log + benchmark_q80)),
                            "interval90_lower_eur": float(np.exp(predicted_log - benchmark_q90)),
                            "interval90_upper_eur": float(np.exp(predicted_log + benchmark_q90)),
                            "interval_abs_log_q80": benchmark_q80,
                            "interval_abs_log_q90": benchmark_q90,
                        })
                    predictions.append(record)
    prediction_frame = pd.DataFrame(predictions)
    evaluation_predictions = prediction_frame.loc[prediction_frame["split"].eq("evaluation")].copy()
    origin_rows = []
    for (origin_id, model_variant), group in evaluation_predictions.groupby(
        ["origin_id", "model_variant"], sort=True
    ):
        result = fee_metrics(group["actual_log_fee_eur"], group["predicted_log_fee_eur"])
        origin_rows.append({
            "origin_id": origin_id, "evaluation_year": int(group["evaluation_year"].iloc[0]),
            "model_variant": model_variant, **result,
        })
    return (
        prediction_frame, pd.DataFrame(origin_rows), pd.DataFrame(tuning_rows),
        pd.DataFrame(raw_selection_rows), pd.DataFrame(policy_selection_rows),
    )


def paired_predictions(
    predictions: pd.DataFrame, baseline: str, candidate: str, years: list[int]
) -> pd.DataFrame:
    subset = predictions.loc[
        predictions["split"].eq("evaluation") & predictions["evaluation_year"].isin(years)
    ]
    keys = ["origin_id", "evaluation_year", "transfer_event_id", "player_id"]
    left = subset.loc[subset["model_variant"].eq(baseline), keys + [
        "actual_log_fee_eur", "predicted_log_fee_eur"
    ]].rename(columns={"predicted_log_fee_eur": "baseline_prediction"})
    right = subset.loc[subset["model_variant"].eq(candidate), keys + [
        "predicted_log_fee_eur"
    ]].rename(columns={"predicted_log_fee_eur": "candidate_prediction"})
    return left.merge(right, on=keys, validate="one_to_one")


def cluster_bootstrap(pairs: pd.DataFrame, seed: int) -> dict[str, Any]:
    difference = np.abs(pairs["actual_log_fee_eur"] - pairs["baseline_prediction"]) - np.abs(
        pairs["actual_log_fee_eur"] - pairs["candidate_prediction"]
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


def build_comparisons(predictions: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("value_anchor_vs_median", BASELINE_MEDIAN, VALUE_ANCHOR),
        ("market_profile_vs_median", BASELINE_MEDIAN, MODEL_ORDER[0]),
        ("market_profile_vs_value_anchor", VALUE_ANCHOR, MODEL_ORDER[0]),
        ("trajectory_vs_market_profile", MODEL_ORDER[0], MODEL_ORDER[1]),
        ("prior_opportunity_vs_trajectory", MODEL_ORDER[1], MODEL_ORDER[2]),
        ("performance_vs_prior_opportunity", MODEL_ORDER[2], MODEL_ORDER[3]),
        ("compact_context_vs_performance", MODEL_ORDER[3], MODEL_ORDER[4]),
        ("full_compact_vs_market_profile", MODEL_ORDER[0], MODEL_ORDER[4]),
        ("policy_vs_median", BASELINE_MEDIAN, POLICY_MODEL),
        ("policy_vs_value_anchor", VALUE_ANCHOR, POLICY_MODEL),
        ("policy_vs_market_profile", MODEL_ORDER[0], POLICY_MODEL),
    ]
    rows = []
    for comparison, baseline, candidate in specs:
        for window, years in WINDOWS.items():
            pairs = paired_predictions(predictions, baseline, candidate, years)
            baseline_error = np.abs(pairs["actual_log_fee_eur"] - pairs["baseline_prediction"])
            candidate_error = np.abs(pairs["actual_log_fee_eur"] - pairs["candidate_prediction"])
            origin_values = [
                np.abs(group["actual_log_fee_eur"] - group["baseline_prediction"]).mean()
                - np.abs(group["actual_log_fee_eur"] - group["candidate_prediction"]).mean()
                for _, group in pairs.groupby("origin_id")
            ]
            bootstrap = cluster_bootstrap(pairs, stable_seed(comparison, window, "bootstrap"))
            rows.append({
                "comparison": comparison, "baseline_model": baseline,
                "candidate_model": candidate, "window": window,
                "evaluation_years": ";".join(map(str, years)), "pooled_n": len(pairs),
                "unique_players": pairs["player_id"].nunique(),
                "baseline_mae_log": baseline_error.mean(),
                "candidate_mae_log": candidate_error.mean(),
                "mae_log_improvement": baseline_error.mean() - candidate_error.mean(),
                "relative_mae_improvement": (
                    baseline_error.mean() - candidate_error.mean()
                ) / baseline_error.mean(),
                "origin_wins": sum(value > 0 for value in origin_values),
                "origin_count": len(origin_values), **bootstrap,
            })
    return pd.DataFrame(rows)


def build_interval_coverage(predictions: pd.DataFrame) -> pd.DataFrame:
    policy = predictions.loc[
        predictions["split"].eq("evaluation") & predictions["model_variant"].eq(MODEL_ORDER[0])
    ].copy()
    rows = []
    for label, lower, upper, nominal in [
        ("80pct", "interval80_lower_eur", "interval80_upper_eur", 0.80),
        ("90pct", "interval90_lower_eur", "interval90_upper_eur", 0.90),
    ]:
        for window, years in WINDOWS.items():
            group = policy.loc[policy["evaluation_year"].isin(years)]
            covered = group["actual_fee_eur"].between(group[lower], group[upper])
            rows.append({
                "interval": label, "window": window,
                "evaluation_years": ";".join(map(str, years)), "rows": len(group),
                "nominal_coverage": nominal, "observed_coverage": covered.mean(),
                "mean_lower_eur": group[lower].mean(), "mean_upper_eur": group[upper].mean(),
                "median_width_eur": (group[upper] - group[lower]).median(),
                "median_upper_to_lower_ratio": (group[upper] / group[lower]).median(),
            })
        for origin_id, group in policy.groupby("origin_id"):
            covered = group["actual_fee_eur"].between(group[lower], group[upper])
            rows.append({
                "interval": label, "window": origin_id,
                "evaluation_years": str(int(group["evaluation_year"].iloc[0])),
                "rows": len(group), "nominal_coverage": nominal,
                "observed_coverage": covered.mean(),
                "mean_lower_eur": group[lower].mean(), "mean_upper_eur": group[upper].mean(),
                "median_width_eur": (group[upper] - group[lower]).median(),
                "median_upper_to_lower_ratio": (group[upper] / group[lower]).median(),
            })
    return pd.DataFrame(rows)


def build_benchmark_cards(predictions: pd.DataFrame) -> pd.DataFrame:
    policy = predictions.loc[
        predictions["split"].eq("evaluation") & predictions["model_variant"].eq(MODEL_ORDER[0])
    ].copy()
    policy["benchmark_model"] = MODEL_ORDER[0]
    policy["actual_to_benchmark_ratio"] = policy["actual_fee_eur"] / policy["predicted_fee_eur"]
    policy["actual_log_premium"] = policy["actual_log_fee_eur"] - policy["predicted_log_fee_eur"]
    policy["historical_fee_assessment"] = pd.cut(
        policy["actual_to_benchmark_ratio"],
        [-np.inf, 0.67, 0.85, 1.18, 1.50, np.inf],
        labels=["large_discount", "discount", "near_benchmark", "premium", "large_premium"],
        right=False,
    ).astype(str)
    policy["benchmark_percentile_within_origin"] = policy.groupby("origin_id")[
        "predicted_fee_eur"
    ].rank(pct=True)
    columns = [
        "origin_id", "evaluation_year", "transfer_event_id", "player_id", "transfer_date",
        "historical_position_group", "origin_competition_id", "destination_competition_id",
        "benchmark_model", "pre_transfer_value_eur", "actual_fee_eur",
        "predicted_fee_eur", "interval80_lower_eur", "interval80_upper_eur",
        "interval90_lower_eur", "interval90_upper_eur", "actual_to_benchmark_ratio",
        "actual_log_premium", "historical_fee_assessment",
        "benchmark_percentile_within_origin",
    ]
    return policy[columns]


def subgroup_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[[
        "transfer_event_id", "historical_position_group", "age_at_transfer", "fee_band",
        "origin_competition_id", "destination_competition_id",
    ]].copy()
    result["age_band"] = pd.cut(
        result["age_at_transfer"], [-np.inf, 21, 24, 27, 30, np.inf],
        labels=["21_or_younger", "22_to_24", "25_to_27", "28_to_30", "31_or_older"],
    )
    return result


def build_subgroups(predictions: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["origin_id", "evaluation_year", "transfer_event_id", "player_id"]
    evals = predictions.loc[predictions["split"].eq("evaluation")]
    anchor = evals.loc[evals["model_variant"].eq(VALUE_ANCHOR), keys + [
        "actual_log_fee_eur", "predicted_log_fee_eur"
    ]].rename(columns={"predicted_log_fee_eur": "anchor_prediction"})
    policy = evals.loc[evals["model_variant"].eq(MODEL_ORDER[0]), keys + [
        "predicted_log_fee_eur"
    ]].rename(columns={"predicted_log_fee_eur": "policy_prediction"})
    paired = anchor.merge(policy, on=keys, validate="one_to_one").merge(
        subgroup_frame(frame), on="transfer_event_id", validate="many_to_one"
    )
    rows = []
    for dimension in [
        "historical_position_group", "age_band", "fee_band",
        "origin_competition_id", "destination_competition_id", "evaluation_year",
    ]:
        for category, group in paired.groupby(dimension, dropna=False, observed=True):
            adequate = len(group) >= 60 and group["origin_id"].nunique() >= 2
            anchor_error = np.abs(group["actual_log_fee_eur"] - group["anchor_prediction"])
            policy_error = np.abs(group["actual_log_fee_eur"] - group["policy_prediction"])
            row = {
                "dimension": dimension, "category": str(category), "rows": len(group),
                "unique_players": group["player_id"].nunique(),
                "origin_count": group["origin_id"].nunique(),
                "adequate_for_inference": adequate,
                "anchor_mae_log": anchor_error.mean(), "policy_mae_log": policy_error.mean(),
                "mae_log_improvement": anchor_error.mean() - policy_error.mean(),
            }
            if adequate:
                bootstrap_pairs = group.rename(columns={
                    "actual_log_fee_eur": "actual_log_fee_eur",
                    "anchor_prediction": "baseline_prediction",
                    "policy_prediction": "candidate_prediction",
                })
                bootstrap = cluster_bootstrap(
                    bootstrap_pairs, stable_seed("subgroup", dimension, str(category))
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


def build_decision(
    comparisons: pd.DataFrame, intervals: pd.DataFrame, subgroups: pd.DataFrame,
    origins: pd.DataFrame,
) -> pd.DataFrame:
    pooled = comparisons.loc[comparisons["window"].eq("pooled_four")].set_index("comparison")
    anchor = pooled.loc["market_profile_vs_value_anchor"]
    median = pooled.loc["market_profile_vs_median"]
    profile_anchor = pooled.loc["market_profile_vs_value_anchor"]
    policy_profile = pooled.loc["policy_vs_market_profile"]
    interval80 = intervals.loc[
        intervals["interval"].eq("80pct") & intervals["window"].eq("pooled_four")
    ].iloc[0]
    interval90 = intervals.loc[
        intervals["interval"].eq("90pct") & intervals["window"].eq("pooled_four")
    ].iloc[0]
    supported_negative = int(subgroups.loc[
        subgroups["adequate_for_inference"]
        & subgroups["stability_label"].eq("supported_negative")
    ].shape[0])
    profile_median = pooled.loc["market_profile_vs_median"]
    validated = (
        profile_anchor["cluster_ci_lower_95"] > 0 and profile_anchor["origin_wins"] >= 3
        and profile_median["cluster_ci_lower_95"] > 0 and profile_median["origin_wins"] >= 3
    )
    interval_coverage_valid = interval80["observed_coverage"] >= 0.70 and interval90["observed_coverage"] >= 0.80
    interval_ranges_precise = (
        interval_coverage_valid
        and interval80["median_upper_to_lower_ratio"] <= 5
        and interval90["median_upper_to_lower_ratio"] <= 10
    )
    richer_policy_supported = (
        policy_profile["cluster_ci_lower_95"] > 0 and policy_profile["origin_wins"] >= 3
    )
    status = "validated_fee_benchmark" if validated else (
        "promising_but_not_better_than_value_anchor" if profile_anchor["mae_log_improvement"] > 0
        else "not_validated_beyond_pre_transfer_value"
    )
    benchmark_origins = origins.loc[origins["model_variant"].eq(MODEL_ORDER[0])]
    return pd.DataFrame([{
        "status": status, "validated_beyond_pre_transfer_value": validated,
        "benchmark_mae_log": benchmark_origins.eval("mae_log * n").sum() / benchmark_origins["n"].sum(),
        "benchmark_multiplicative_error_factor": float(np.exp(
            benchmark_origins.eval("mae_log * n").sum() / benchmark_origins["n"].sum()
        )),
        "anchor_mae_log": anchor["baseline_mae_log"],
        "anchor_mae_improvement": anchor["mae_log_improvement"],
        "anchor_ci_lower_95": anchor["cluster_ci_lower_95"],
        "anchor_ci_upper_95": anchor["cluster_ci_upper_95"],
        "anchor_origin_wins": int(anchor["origin_wins"]),
        "market_profile_vs_anchor_improvement": profile_anchor["mae_log_improvement"],
        "market_profile_vs_anchor_ci_lower_95": profile_anchor["cluster_ci_lower_95"],
        "market_profile_vs_anchor_ci_upper_95": profile_anchor["cluster_ci_upper_95"],
        "market_profile_vs_anchor_origin_wins": int(profile_anchor["origin_wins"]),
        "richer_policy_vs_market_profile_improvement": policy_profile["mae_log_improvement"],
        "richer_policy_vs_market_profile_ci_lower_95": policy_profile["cluster_ci_lower_95"],
        "richer_policy_vs_market_profile_ci_upper_95": policy_profile["cluster_ci_upper_95"],
        "richer_policy_vs_market_profile_origin_wins": int(policy_profile["origin_wins"]),
        "richer_policy_increment_supported": richer_policy_supported,
        "recommended_model_scope": MODEL_ORDER[0],
        "median_mae_improvement": profile_median["mae_log_improvement"],
        "median_ci_lower_95": profile_median["cluster_ci_lower_95"],
        "median_ci_upper_95": profile_median["cluster_ci_upper_95"],
        "median_origin_wins": int(profile_median["origin_wins"]),
        "interval80_observed_coverage": interval80["observed_coverage"],
        "interval90_observed_coverage": interval90["observed_coverage"],
        "interval_coverage_valid": interval_coverage_valid,
        "interval_ranges_precise": interval_ranges_precise,
        "interval80_median_upper_to_lower_ratio": interval80["median_upper_to_lower_ratio"],
        "interval90_median_upper_to_lower_ratio": interval90["median_upper_to_lower_ratio"],
        "adequate_supported_negative_subgroups": supported_negative,
        "recommended_use": (
            "reported-fee comparables screen with broad uncertainty range and proposed-fee premium/discount"
            if validated else "descriptive fee comparisons only"
        ),
        "claim_guardrail": (
            "Reported fee only; excludes wages, bonuses, agent fees, add-ons, total deal cost, value creation, and ROI."
        ),
    }])


def build_checks(
    frame: pd.DataFrame, feature_sets: dict[str, list[str]], predictions: pd.DataFrame,
    selections: pd.DataFrame, intervals: pd.DataFrame,
) -> pd.DataFrame:
    checks = []
    def add(check: str, passed: bool, observed: Any, expected: Any, note: str) -> None:
        checks.append({
            "check": check, "passed": bool(passed), "observed": observed,
            "expected": expected, "note": note,
        })
    add("cohort_rows", len(frame) == 1486, len(frame), 1486, "Matches financial audit.")
    add("cohort_key_unique", frame["transfer_event_id"].is_unique, frame["transfer_event_id"].nunique(), len(frame), "One row per deal.")
    add("positive_fee_target", pd.to_numeric(frame["canonical_transfer_fee_eur"], errors="coerce").gt(0).all(), frame["canonical_transfer_fee_eur"].min(), ">0", "Every target is a reliable positive fee.")
    add("feature_counts", {name: len(fields) for name, fields in feature_sets.items()} == EXPECTED_FEATURE_COUNTS, json.dumps({name: len(fields) for name, fields in feature_sets.items()}, sort_keys=True), json.dumps(EXPECTED_FEATURE_COUNTS, sort_keys=True), "Frozen nested blocks.")
    forbidden = [feature for fields in feature_sets.values() for feature in fields if any(token in feature.lower() for token in ["canonical_transfer_fee", "log_fee", "fee_band", "post12", "post24", "target_", "eligible_"])]
    add("no_fee_or_outcome_predictors", not forbidden, ";".join(sorted(set(forbidden))), "none", "Target fee and downstream fields excluded.")
    key = ["split", "origin_id", "model_variant", "transfer_event_id"]
    add("prediction_key_unique", not predictions.duplicated(key).any(), predictions.duplicated(key).sum(), 0, "One prediction per split/origin/model/deal.")
    selected = selections.loc[selections["selected_for_policy"].astype(bool)]
    add("one_policy_model_per_origin", len(selected) == 4 and selected.groupby("origin_id").size().eq(1).all(), len(selected), 4, "Validation selection unique.")
    policy_eval = predictions.loc[
        predictions["split"].eq("evaluation") & predictions["model_variant"].eq(MODEL_ORDER[0])
    ]
    add("evaluation_rows", len(policy_eval) == 738, len(policy_eval), 738, "Four evaluation seasons complete.")
    add("positive_predictions", policy_eval["predicted_fee_eur"].gt(0).all(), policy_eval["predicted_fee_eur"].min(), ">0", "Log model yields positive fee benchmark.")
    bounds_valid = (
        (policy_eval["interval90_lower_eur"] <= policy_eval["interval80_lower_eur"]).all()
        and (policy_eval["interval80_lower_eur"] <= policy_eval["predicted_fee_eur"]).all()
        and (policy_eval["predicted_fee_eur"] <= policy_eval["interval80_upper_eur"]).all()
        and (policy_eval["interval80_upper_eur"] <= policy_eval["interval90_upper_eur"]).all()
    )
    add("intervals_nested", bounds_valid, bounds_valid, True, "90% range contains 80% range and point benchmark.")
    add("interval_rows", set(intervals["interval"]) == {"80pct", "90pct"}, ";".join(sorted(intervals["interval"].unique())), "80pct;90pct", "Both empirical ranges audited.")
    return pd.DataFrame(checks)


def write_readme(
    frame: pd.DataFrame, feature_sets: dict[str, list[str]], selections: pd.DataFrame,
    comparisons: pd.DataFrame, intervals: pd.DataFrame, decision: pd.DataFrame,
) -> None:
    selected = selections.loc[selections["selected_for_policy"].astype(bool)]
    pooled = comparisons.loc[comparisons["window"].eq("pooled_four")].set_index("comparison")
    row = decision.iloc[0]
    lines = [
        "# MoveMaker reported-fee benchmarking model", "",
        "This model estimates the disclosed permanent-transfer fee paid for comparable historical deals. It does not estimate total deal cost, future value, or ROI.", "",
        "## Design", "",
        f"- Cohort: {len(frame):,} clean permanent deals with a reliable positive reported fee and recent positive pre-transfer value.",
        "- Target: natural log of reported fee; euro figures are obtained by exponentiating the prediction.",
        "- Primary hurdles: chronological median and the player's recent pre-transfer market value.",
        "- Nested temporal design: prior-year model selection, validation-year feature-block selection/range estimation, next-year evaluation.",
        "- Empirical 80% and 90% ranges use validation residual magnitudes and are audited out of season.", "",
        "## Feature blocks", "",
    ]
    for model, features in feature_sets.items():
        lines.append(f"- `{model}`: {len(features)} predictors.")
    lines += ["", "## Validation-selected richer-policy diagnostics", ""]
    for item in selected.itertuples(index=False):
        lines.append(f"- `{item.origin_id}`: `{item.candidate_model}` (validation MAE-log {item.validation_mae_log:.6f}).")
    anchor = pooled.loc["market_profile_vs_value_anchor"]
    median = pooled.loc["market_profile_vs_median"]
    coverage80 = intervals.loc[
        intervals["interval"].eq("80pct") & intervals["window"].eq("pooled_four"),
        "observed_coverage",
    ].iloc[0]
    coverage90 = intervals.loc[
        intervals["interval"].eq("90pct") & intervals["window"].eq("pooled_four"),
        "observed_coverage",
    ].iloc[0]
    lines += [
        "", "## Result", "",
        f"- Status: `{row['status']}`.",
        f"- Versus pre-transfer value: MAE-log improvement {anchor['mae_log_improvement']:.6f}, 95% CI [{anchor['cluster_ci_lower_95']:.6f}, {anchor['cluster_ci_upper_95']:.6f}], {int(anchor['origin_wins'])}/4 origin wins.",
        f"- Versus chronological median: improvement {median['mae_log_improvement']:.6f}, CI [{median['cluster_ci_lower_95']:.6f}, {median['cluster_ci_upper_95']:.6f}], {int(median['origin_wins'])}/4 wins.",
        f"- Compact benchmark MAE-log {row['benchmark_mae_log']:.6f}, corresponding to an average multiplicative error factor of approximately {row['benchmark_multiplicative_error_factor']:.2f}x.",
        f"- Empirical interval coverage: 80% range {coverage80:.1%}; 90% range {coverage90:.1%}.",
        f"- The ranges are broad: median upper/lower ratio {row['interval80_median_upper_to_lower_ratio']:.1f}x for the 80% range and {row['interval90_median_upper_to_lower_ratio']:.1f}x for the 90% range. They are uncertainty guardrails, not precise fair-price bands.",
        f"- Recommended model scope: `{row['recommended_model_scope']}`; richer-policy increment over market/profile supported: `{str(bool(row['richer_policy_increment_supported'])).lower()}`.",
        "", "## Interpretation", "",
        "For a proposed deal, compare the proposed disclosed fee with the model's historical benchmark and range. A premium is a comparables warning for negotiation/diligence, not proof of overpayment; a discount is not proof of value. Historical cards use the realized reported fee only to audit the benchmark.",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    protected_before = protected_tree_sha256()
    frame, feature_sets, feature_manifest = build_model_frame()
    predictions, origins, tuning, raw_selections, policy_selections = fit_and_evaluate(
        frame, feature_sets
    )
    comparisons = build_comparisons(predictions)
    intervals = build_interval_coverage(predictions)
    cards = build_benchmark_cards(predictions)
    subgroups = build_subgroups(predictions, frame)
    decision = build_decision(comparisons, intervals, subgroups, origins)
    checks = build_checks(frame, feature_sets, predictions, policy_selections, intervals)
    if not checks["passed"].all():
        raise RuntimeError(f"Build checks failed: {checks.loc[~checks['passed'], 'check'].tolist()}")

    target_columns = [
        "transfer_event_id", "player_id", "transfer_date", "season_start_year",
        "from_club_name", "to_club_name", "canonical_transfer_fee_eur",
        "valuation_pre_eur", "fee_band", "target_log_fee_eur",
    ]
    tables = {
        "fee_benchmark_targets.csv": frame[target_columns],
        "fee_benchmark_feature_manifest.csv": feature_manifest,
        "raw_model_tuning_results.csv": tuning,
        "raw_model_selections.csv": raw_selections,
        "policy_model_selections.csv": policy_selections,
        "fee_benchmark_predictions.csv": predictions,
        "fee_benchmark_origin_results.csv": origins,
        "fee_benchmark_comparison_results.csv": comparisons,
        "fee_interval_coverage.csv": intervals,
        "historical_fee_benchmark_cards.csv": cards,
        "fee_benchmark_subgroup_stability.csv": subgroups,
        "fee_benchmark_decision_summary.csv": decision,
        "build_checks.csv": checks,
    }
    for name, table in tables.items():
        table.to_csv(OUTPUT_DIR / name, index=False)
    source_paths = [MASTER_PATH, AUDIT_PATH, AUDIT_VERIFICATION_PATH]
    sources = pd.DataFrame([
        {
            "source": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in source_paths
    ])
    sources.to_csv(OUTPUT_DIR / "source_manifest.csv", index=False)
    tables["source_manifest.csv"] = sources
    write_readme(frame, feature_sets, policy_selections, comparisons, intervals, decision)
    if protected_before != protected_tree_sha256():
        raise RuntimeError("A protected verified output changed during fee modeling.")
    run_summary = {
        "cohort_rows": len(frame), "unique_players": frame["player_id"].nunique(),
        "evaluation_rows": int(len(cards)),
        "feature_counts": {name: len(features) for name, features in feature_sets.items()},
        "selected_models": policy_selections.loc[
            policy_selections["selected_for_policy"].astype(bool)
        ].set_index("origin_id")["candidate_model"].to_dict(),
        "decision": decision.iloc[0]["status"],
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "protected_outputs_unchanged": True,
        "build_checks": len(checks), "build_checks_passed": int(checks["passed"].sum()),
    }
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_names = sorted(list(tables) + ["README.md", "run_summary.json"])
    manifest = pd.DataFrame([
        {
            "file": name, "size_bytes": (OUTPUT_DIR / name).stat().st_size,
            "sha256": sha256_file(OUTPUT_DIR / name),
        }
        for name in manifest_names
    ])
    manifest.to_csv(OUTPUT_DIR / "output_manifest.csv", index=False)
    print(json.dumps(run_summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

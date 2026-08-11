"""Model 24-month paid-transfer opportunity and under-utilization risk.

The cohort is restricted to clean permanent arrivals with a reliable positive
reported inbound fee and at least 20 observable destination-club matches in the
first 24 months.  The continuous target is the player's share of destination
match-minute capacity.  Binary targets flag shares below 10% and 25%.

All model selection is chronological.  Feature block and model family are
selected on the preceding validation season; the following season is untouched
evaluation data.  Ordered binary probabilities are obtained with a predeclared
Euclidean projection so P(under 10%) never exceeds P(under 25%).
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
OUTPUT_DIR = ROOT / "Data" / "processed" / "paid_transfer_underutilization_model"

RANDOM_SEED = 20260811
BOOTSTRAP_REPETITIONS = 5000
ORIGINS = coarse.ORIGINS
WINDOWS = market.WINDOWS
FIXED_MODEL_ORDER = [
    "M0_chronological_constant",
    "M1_paid_deal_market_profile",
    "M2_plus_prior_opportunity",
    "M3_plus_player_performance",
    "M4_plus_destination_context",
    "M5_plus_origin_transition_context",
]
POLICY_MODEL = "M_policy_validation_selected"
ORDERED_POLICY_MODEL = "M_policy_ordered"
MODEL_ORDER = FIXED_MODEL_ORDER + [POLICY_MODEL]
EXPECTED_FEATURE_COUNTS = dict(zip(FIXED_MODEL_ORDER, [0, 11, 19, 35, 79, 139]))
ENDPOINTS = {
    "opportunity_share_24m": {"kind": "regression", "target": "target_opportunity_share_24m"},
    "underutilization_10pct": {"kind": "classification", "target": "target_underutilization_10pct"},
    "underutilization_25pct": {"kind": "classification", "target": "target_underutilization_25pct"},
}
COMPARISONS = [
    ("M1_vs_M0_paid_deal_market_profile", FIXED_MODEL_ORDER[0], FIXED_MODEL_ORDER[1]),
    ("M2_vs_M1_prior_opportunity", FIXED_MODEL_ORDER[1], FIXED_MODEL_ORDER[2]),
    ("M3_vs_M2_player_performance", FIXED_MODEL_ORDER[2], FIXED_MODEL_ORDER[3]),
    ("M4_vs_M3_destination_context", FIXED_MODEL_ORDER[3], FIXED_MODEL_ORDER[4]),
    ("M5_vs_M4_origin_transition", FIXED_MODEL_ORDER[4], FIXED_MODEL_ORDER[5]),
    ("M5_vs_M1_all_incremental_history_context", FIXED_MODEL_ORDER[1], FIXED_MODEL_ORDER[5]),
    ("M5_vs_M0_total", FIXED_MODEL_ORDER[0], FIXED_MODEL_ORDER[5]),
    ("policy_vs_M0_total", FIXED_MODEL_ORDER[0], POLICY_MODEL),
    ("policy_vs_M1_market_profile", FIXED_MODEL_ORDER[1], POLICY_MODEL),
]
PROTECTED_DIRS = market.PROTECTED_DIRS + [
    ROOT / "Data" / "processed" / "market_value_downside_model",
    ROOT / "Data" / "processed" / "market_value_calibration_experiment",
    ROOT / "Data" / "processed" / "market_value_downside_product",
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
    definitions: dict[str, dict[str, str]],
    feature: str,
    source_fields: str,
    timing: str,
    rationale: str,
    kind: str = "derived",
) -> None:
    definitions[feature] = {
        "feature_kind": kind,
        "source_fields": source_fields,
        "timing_classification": timing,
        "rationale": rationale,
    }


def build_model_frame() -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    verification = json.loads(AUDIT_VERIFICATION_PATH.read_text(encoding="utf-8"))
    if verification.get("status") != "pass":
        raise RuntimeError("The upstream financial feasibility audit is not independently verified.")

    audit_columns = [
        "transfer_event_id", "canonical_transfer_fee_eur", "canonical_transfer_fee_status",
        "fee_band", "valuation_pre_eur", "valuation_pre_staleness_days",
        "club_destination_post24_games", "appearance_post24_destination_minutes",
        "destination_opportunity_share_24m", "eligible_paid_transfer_underutilization_24m",
    ]
    audit = pd.read_csv(AUDIT_PATH, usecols=audit_columns, low_memory=False)
    eligible = audit.loc[as_bool(audit["eligible_paid_transfer_underutilization_24m"])].copy()
    eligible["transfer_event_id"] = eligible["transfer_event_id"].astype(str)
    if len(eligible) != 1074 or not eligible["transfer_event_id"].is_unique:
        raise RuntimeError(f"Paid under-utilization audit cohort mismatch: {len(eligible)} rows.")

    master = pd.read_csv(MASTER_PATH, low_memory=False)
    master["transfer_event_id"] = master["transfer_event_id"].astype(str)
    frame = master.loc[master["transfer_event_id"].isin(set(eligible["transfer_event_id"]))].copy()
    overlap = [column for column in audit_columns if column in frame.columns and column != "transfer_event_id"]
    frame = frame.drop(columns=overlap)
    frame = frame.merge(eligible[audit_columns], on="transfer_event_id", how="inner", validate="one_to_one")
    if len(frame) != 1074:
        raise RuntimeError(f"Comprehensive-master cohort mismatch: {len(frame)} rows.")

    frame["transfer_date"] = pd.to_datetime(frame["transfer_date"], errors="coerce")
    frame["season_start_year"] = frame["transfer_date"].dt.year.where(
        frame["transfer_date"].dt.month.ge(7), frame["transfer_date"].dt.year - 1
    ).astype("Int64")
    target = pd.to_numeric(frame["destination_opportunity_share_24m"], errors="coerce")
    frame["target_opportunity_share_24m"] = target
    frame["target_underutilization_10pct"] = target.lt(0.10).astype(int)
    frame["target_underutilization_25pct"] = target.lt(0.25).astype(int)
    if target.isna().any() or not target.between(0, 1.05).all():
        raise RuntimeError("Opportunity target is incomplete or outside 0..1.05.")

    definitions: dict[str, dict[str, str]] = {}
    transfer_date = frame["transfer_date"]
    birth_date = pd.to_datetime(frame["player_date_of_birth"], errors="coerce")
    frame["age_at_transfer"] = ((transfer_date - birth_date).dt.days / 365.2425).where(
        lambda values: values.between(15, 45)
    )
    historical_position = frame["lineup_pre365_all_primary_position"].where(
        frame["lineup_pre365_all_primary_position"].notna(),
        frame["lineup_pre365_origin_primary_position"],
    )
    frame["historical_position_group"] = historical_position.map(coarse.position_group)
    frame["destination_competition_id"] = frame[
        "club_destination_pre365_dominant_competition_id"
    ].astype("string").fillna("unknown")
    frame["transfer_month_sin"] = np.sin(2 * np.pi * transfer_date.dt.month / 12)
    frame["transfer_month_cos"] = np.cos(2 * np.pi * transfer_date.dt.month / 12)
    frame["transfer_year_centered"] = frame["season_start_year"].astype(float) - 2017.0
    fee = pd.to_numeric(frame["canonical_transfer_fee_eur"], errors="coerce")
    pre_value = pd.to_numeric(frame["valuation_pre_eur"], errors="coerce")
    frame["log_inbound_fee_eur"] = np.log1p(fee.clip(lower=0))
    frame["log_pre_value_eur"] = np.log1p(pre_value.where(pre_value.gt(0)))
    frame["log_fee_to_pre_value"] = market.safe_log_ratio(fee, pre_value)
    frame["pre_value_available"] = pre_value.gt(0).astype(float)
    m1 = [
        "age_at_transfer", "historical_position_group", "destination_competition_id",
        "transfer_month_sin", "transfer_month_cos", "transfer_year_centered",
        "log_inbound_fee_eur", "log_pre_value_eur", "log_fee_to_pre_value",
        "valuation_pre_staleness_days", "pre_value_available",
    ]
    for feature in m1:
        add_definition(
            definitions, feature, feature, "available at transfer",
            "Paid-deal market, player-profile, destination, or calendar control known at signing.",
            "raw" if feature == "valuation_pre_staleness_days" else "derived",
        )

    usage_features: list[str] = []
    for feature, numerator, denominator, scale in opportunity.USAGE_SPECS:
        frame[feature] = coarse.safe_divide(
            frame[numerator], pd.to_numeric(frame[denominator], errors="coerce") * scale
        )
        usage_features.append(feature)
        add_definition(
            definitions, feature, f"{numerator};{denominator}",
            "derived strictly pre-transfer player opportunity",
            "Normalized origin-club involvement before the transfer.",
        )
    m2 = m1 + usage_features

    performance_features: list[str] = []
    for context in opportunity.PERFORMANCE_CONTEXTS:
        for suffix in ["goals_per90", "assists_per90"]:
            feature = f"appearance_{context}_{suffix}"
            performance_features.append(feature)
            add_definition(
                definitions, feature, feature, "strictly pre-transfer player performance",
                "Player output rate before signing.", "raw",
            )
        minutes = f"appearance_{context}_minutes"
        for card in ["yellow_cards", "red_cards"]:
            source = f"appearance_{context}_{card}"
            feature = f"appearance_{context}_{card}_per90"
            frame[feature] = 90.0 * coarse.safe_divide(frame[source], frame[minutes])
            performance_features.append(feature)
            add_definition(
                definitions, feature, f"{source};{minutes}",
                "derived strictly pre-transfer player performance",
                "Player disciplinary rate before signing.",
            )
    m3 = m2 + performance_features

    destination_features = coarse.add_context_features(frame, "destination", definitions)
    m4 = m3 + destination_features
    origin_features = coarse.add_context_features(frame, "origin", definitions)
    transition_features: list[str] = []
    for horizon in [365, 730]:
        for metric_name in coarse.TRANSITION_METRICS:
            destination = f"club_destination_pre{horizon}_{metric_name}"
            origin = f"club_origin_pre{horizon}_{metric_name}"
            feature = f"transition_pre{horizon}_{metric_name}_delta"
            frame[feature] = pd.to_numeric(frame[destination], errors="coerce") - pd.to_numeric(
                frame[origin], errors="coerce"
            )
            transition_features.append(feature)
            add_definition(
                definitions, feature, f"{destination};{origin}",
                "derived strictly pre-transfer club context",
                "Destination-minus-origin environment contrast.",
            )
    m5 = m4 + origin_features + transition_features
    feature_sets = dict(zip(FIXED_MODEL_ORDER, [[], m1, m2, m3, m4, m5]))
    counts = {name: len(features) for name, features in feature_sets.items()}
    if counts != EXPECTED_FEATURE_COUNTS:
        raise RuntimeError(f"Feature count mismatch: {counts}")
    for index in range(1, len(FIXED_MODEL_ORDER)):
        if not set(feature_sets[FIXED_MODEL_ORDER[index - 1]]).issubset(
            feature_sets[FIXED_MODEL_ORDER[index]]
        ):
            raise RuntimeError("Feature sets are not nested.")
    if any(len(features) != len(set(features)) for features in feature_sets.values()):
        raise RuntimeError("A feature set contains duplicate fields.")

    manifest_rows: list[dict[str, Any]] = [{
        "feature_set": FIXED_MODEL_ORDER[0], "feature_order": 0, "feature": "",
        "feature_kind": "no_predictors", "source_fields": "",
        "timing_classification": "not applicable",
        "rationale": "Chronological median or prevalence benchmark.", "raw_feature_count": 0,
    }]
    for feature_set in FIXED_MODEL_ORDER[1:]:
        for order, feature in enumerate(feature_sets[feature_set], start=1):
            manifest_rows.append({
                "feature_set": feature_set, "feature_order": order, "feature": feature,
                **definitions[feature], "raw_feature_count": len(feature_sets[feature_set]),
            })
    return (
        frame.sort_values(["season_start_year", "transfer_date", "transfer_event_id"]),
        feature_sets,
        pd.DataFrame(manifest_rows),
    )


def tune_and_fit(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    evaluation: pd.DataFrame,
    features: list[str],
    endpoint: str,
    target: str,
    kind: str,
    origin_id: str,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    candidates: list[tuple[tuple[Any, ...], str, dict[str, Any], int]] = []
    for family, parameters in market.candidate_specs(kind):
        seed = stable_seed(endpoint, origin_id, family, json.dumps(parameters, sort_keys=True))
        prediction, encoded_count = market.fit_candidate(
            train, validation, features, target, kind, family, parameters, seed
        )
        result = (
            market.regression_metrics(validation[target], np.clip(prediction, 0, 1.05))
            if kind == "regression"
            else market.classification_metrics(validation[target], prediction)
        )
        primary = float(result["mae"] if kind == "regression" else result["brier"])
        secondary = float(result["rmse"] if kind == "regression" else result["log_loss"])
        parameter_text = json.dumps(parameters, sort_keys=True)
        rows.append({
            "endpoint": endpoint, "origin_id": origin_id, "model_family": family,
            "hyperparameters": parameter_text, "encoded_feature_count": encoded_count,
            "validation_primary_loss": primary, "validation_secondary_loss": secondary,
            **{f"validation_{key}": value for key, value in result.items()},
        })
        candidates.append(((primary, secondary, family, parameter_text), family, parameters, encoded_count))
    _, family, parameters, _ = min(candidates, key=lambda item: item[0])
    fit_frame = pd.concat([train, validation], ignore_index=True)
    prediction, encoded_count = market.fit_candidate(
        fit_frame, evaluation, features, target, kind, family, parameters,
        stable_seed(endpoint, origin_id, family, json.dumps(parameters, sort_keys=True), "refit"),
    )
    selected_row = next(
        row for row in rows
        if row["model_family"] == family
        and row["hyperparameters"] == json.dumps(parameters, sort_keys=True)
    )
    selected = {
        "selected_family": family,
        "selected_hyperparameters": json.dumps(parameters, sort_keys=True),
        "encoded_feature_count": encoded_count,
        "validation_primary_loss": selected_row["validation_primary_loss"],
        "validation_secondary_loss": selected_row["validation_secondary_loss"],
    }
    return prediction, selected, rows


def fit_models(
    frame: pd.DataFrame, feature_sets: dict[str, list[str]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions: list[dict[str, Any]] = []
    origins: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for endpoint, spec in ENDPOINTS.items():
        kind, target = spec["kind"], spec["target"]
        for origin_id, train_end, validation_year, evaluation_year in ORIGINS:
            train = frame.loc[frame["season_start_year"].le(train_end)].copy()
            validation = frame.loc[frame["season_start_year"].eq(validation_year)].copy()
            evaluation = frame.loc[frame["season_start_year"].eq(evaluation_year)].copy()
            if min(len(train), len(validation), len(evaluation)) == 0:
                raise RuntimeError(f"Empty chronological split for {endpoint}/{origin_id}.")
            fit_frame = pd.concat([train, validation], ignore_index=True)
            evaluation = evaluation.sort_values("transfer_event_id")
            origin_candidates: list[dict[str, Any]] = []
            for model_variant in FIXED_MODEL_ORDER:
                if model_variant == FIXED_MODEL_ORDER[0]:
                    if kind == "regression":
                        validation_prediction = np.repeat(float(train[target].median()), len(validation))
                        prediction = np.repeat(float(fit_frame[target].median()), len(evaluation))
                        family = "chronological_median"
                    else:
                        validation_prediction = np.repeat(float(train[target].mean()), len(validation))
                        prediction = np.repeat(float(fit_frame[target].mean()), len(evaluation))
                        family = "chronological_prevalence"
                    validation_result = (
                        market.regression_metrics(validation[target], validation_prediction)
                        if kind == "regression"
                        else market.classification_metrics(validation[target], validation_prediction)
                    )
                    selected = {
                        "selected_family": family, "selected_hyperparameters": "{}",
                        "encoded_feature_count": 0,
                        "validation_primary_loss": float(
                            validation_result["mae"] if kind == "regression" else validation_result["brier"]
                        ),
                        "validation_secondary_loss": float(
                            validation_result["rmse"] if kind == "regression" else validation_result["log_loss"]
                        ),
                    }
                else:
                    prediction, selected, tuning = tune_and_fit(
                        train, validation, evaluation, feature_sets[model_variant], endpoint,
                        target, kind, origin_id,
                    )
                    for row in tuning:
                        row["model_variant"] = model_variant
                    tuning_rows.extend(tuning)
                prediction = np.clip(prediction, 0, 1.05) if kind == "regression" else np.clip(
                    prediction, 1e-6, 1 - 1e-6
                )
                result = (
                    market.regression_metrics(evaluation[target], prediction)
                    if kind == "regression"
                    else market.classification_metrics(evaluation[target], prediction)
                )
                origin_record = {
                    "endpoint": endpoint, "endpoint_kind": kind, "origin_id": origin_id,
                    "train_end_year": train_end, "validation_year": validation_year,
                    "evaluation_year": evaluation_year, "train_n": len(train),
                    "validation_n": len(validation), "evaluation_n": len(evaluation),
                    "model_variant": model_variant,
                    "raw_feature_count": len(feature_sets[model_variant]), **selected, **result,
                }
                origins.append(origin_record)
                origin_candidates.append(origin_record)
                for row, predicted in zip(evaluation.itertuples(index=False), prediction):
                    predictions.append({
                        "endpoint": endpoint, "endpoint_kind": kind, "origin_id": origin_id,
                        "evaluation_year": evaluation_year,
                        "transfer_event_id": row.transfer_event_id, "player_id": row.player_id,
                        "transfer_date": row.transfer_date.date().isoformat(),
                        "model_variant": model_variant, **selected,
                        "actual": float(getattr(row, target)), "prediction": float(predicted),
                        "target_opportunity_share_24m": float(row.target_opportunity_share_24m),
                        "canonical_transfer_fee_eur": float(row.canonical_transfer_fee_eur),
                        "fee_band": row.fee_band,
                        "historical_position_group": row.historical_position_group,
                        "destination_competition_id": row.destination_competition_id,
                        "age_at_transfer": row.age_at_transfer,
                    })
            winner = min(
                origin_candidates,
                key=lambda row: (
                    row["validation_primary_loss"], row["validation_secondary_loss"],
                    FIXED_MODEL_ORDER.index(row["model_variant"]),
                ),
            )
            selections.append({
                "endpoint": endpoint, "origin_id": origin_id,
                "validation_year": validation_year, "evaluation_year": evaluation_year,
                "selected_feature_set": winner["model_variant"],
                "selected_family": winner["selected_family"],
                "selected_hyperparameters": winner["selected_hyperparameters"],
                "validation_primary_loss": winner["validation_primary_loss"],
            })

    prediction_frame = pd.DataFrame(predictions)
    origin_frame = pd.DataFrame(origins)
    for selection in selections:
        mask = (
            prediction_frame["endpoint"].eq(selection["endpoint"])
            & prediction_frame["origin_id"].eq(selection["origin_id"])
            & prediction_frame["model_variant"].eq(selection["selected_feature_set"])
        )
        copied = prediction_frame.loc[mask].copy()
        copied["model_variant"] = POLICY_MODEL
        copied["policy_selected_feature_set"] = selection["selected_feature_set"]
        prediction_frame = pd.concat([prediction_frame, copied], ignore_index=True)

        origin_mask = (
            origin_frame["endpoint"].eq(selection["endpoint"])
            & origin_frame["origin_id"].eq(selection["origin_id"])
            & origin_frame["model_variant"].eq(selection["selected_feature_set"])
        )
        copied_origin = origin_frame.loc[origin_mask].copy()
        copied_origin["model_variant"] = POLICY_MODEL
        copied_origin["policy_selected_feature_set"] = selection["selected_feature_set"]
        origin_frame = pd.concat([origin_frame, copied_origin], ignore_index=True)
    return prediction_frame, origin_frame, pd.DataFrame(tuning_rows), pd.DataFrame(selections)


def add_ordered_policy(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy = predictions.loc[
        predictions["model_variant"].eq(POLICY_MODEL)
        & predictions["endpoint"].isin(["underutilization_10pct", "underutilization_25pct"])
    ].copy()
    keys = ["origin_id", "evaluation_year", "transfer_event_id", "player_id"]
    wide = policy.pivot(index=keys, columns="endpoint", values="prediction").reset_index()
    p10 = wide["underutilization_10pct"].to_numpy(float)
    p25 = wide["underutilization_25pct"].to_numpy(float)
    violations = p10 > p25
    midpoint = (p10 + p25) / 2
    wide["ordered_probability_under10"] = np.where(violations, midpoint, p10)
    wide["ordered_probability_under25"] = np.where(violations, midpoint, p25)
    wide["raw_probability_under10"] = p10
    wide["raw_probability_under25"] = p25
    wide["raw_order_violation"] = violations

    ordered_rows = []
    for endpoint, column in [
        ("underutilization_10pct", "ordered_probability_under10"),
        ("underutilization_25pct", "ordered_probability_under25"),
    ]:
        source = policy.loc[policy["endpoint"].eq(endpoint)].merge(
            wide[keys + [column]], on=keys, how="left", validate="one_to_one"
        )
        source["model_variant"] = ORDERED_POLICY_MODEL
        source["prediction"] = source[column]
        ordered_rows.append(source.drop(columns=column))
    return pd.concat([predictions] + ordered_rows, ignore_index=True), wide


def paired_predictions(
    predictions: pd.DataFrame, endpoint: str, baseline: str, candidate: str, years: list[int]
) -> pd.DataFrame:
    subset = predictions.loc[
        predictions["endpoint"].eq(endpoint) & predictions["evaluation_year"].isin(years)
    ]
    keys = ["origin_id", "evaluation_year", "transfer_event_id", "player_id"]
    left = subset.loc[subset["model_variant"].eq(baseline), keys + ["actual", "prediction"]].rename(
        columns={"prediction": "baseline_prediction"}
    )
    right = subset.loc[subset["model_variant"].eq(candidate), keys + ["actual", "prediction"]].rename(
        columns={"actual": "candidate_actual", "prediction": "candidate_prediction"}
    )
    pairs = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if not np.allclose(pairs["actual"], pairs["candidate_actual"], atol=1e-12, rtol=0):
        raise RuntimeError("Paired actual outcomes disagree.")
    return pairs.drop(columns="candidate_actual")


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
    rows = []
    comparison_specs = list(COMPARISONS)
    comparison_specs += [
        ("ordered_policy_vs_M0_total", FIXED_MODEL_ORDER[0], ORDERED_POLICY_MODEL),
        ("ordered_policy_vs_raw_policy", POLICY_MODEL, ORDERED_POLICY_MODEL),
    ]
    for endpoint, spec in ENDPOINTS.items():
        kind = spec["kind"]
        for comparison, baseline, candidate in comparison_specs:
            if candidate == ORDERED_POLICY_MODEL and kind != "classification":
                continue
            for window, years in WINDOWS.items():
                pairs = paired_predictions(predictions, endpoint, baseline, candidate, years)
                if pairs.empty:
                    continue
                if kind == "regression":
                    base_loss = np.abs(pairs["actual"] - pairs["baseline_prediction"])
                    candidate_loss = np.abs(pairs["actual"] - pairs["candidate_prediction"])
                    metric_name = "mae"
                else:
                    base_loss = np.square(pairs["actual"] - pairs["baseline_prediction"])
                    candidate_loss = np.square(pairs["actual"] - pairs["candidate_prediction"])
                    metric_name = "brier"
                origin_values = []
                for _, group in pairs.groupby("origin_id", sort=True):
                    if kind == "regression":
                        base = np.abs(group["actual"] - group["baseline_prediction"]).mean()
                        candidate_value = np.abs(group["actual"] - group["candidate_prediction"]).mean()
                    else:
                        base = np.square(group["actual"] - group["baseline_prediction"]).mean()
                        candidate_value = np.square(group["actual"] - group["candidate_prediction"]).mean()
                    origin_values.append(float(base - candidate_value))
                improvement = float(base_loss.mean() - candidate_loss.mean())
                bootstrap = cluster_bootstrap(
                    pairs, kind, stable_seed(endpoint, comparison, window, "cluster")
                )
                rows.append({
                    "endpoint": endpoint, "endpoint_kind": kind, "comparison": comparison,
                    "baseline_model": baseline, "candidate_model": candidate,
                    "window": window, "evaluation_years": ";".join(map(str, years)),
                    "primary_metric": metric_name, "pooled_n": len(pairs),
                    "unique_players": pairs["player_id"].nunique(),
                    "baseline_loss": float(base_loss.mean()),
                    "candidate_loss": float(candidate_loss.mean()),
                    "loss_improvement": improvement,
                    "relative_loss_improvement": improvement / float(base_loss.mean()),
                    "origin_wins": sum(value > 0 for value in origin_values),
                    "origin_count": len(origin_values), **bootstrap,
                    "stability_label": stability_label(
                        improvement, origin_values,
                        float(bootstrap["cluster_probability_candidate_better"]),
                    ),
                })
    return pd.DataFrame(rows)


def recommended_variant(endpoint: str) -> str:
    return POLICY_MODEL if ENDPOINTS[endpoint]["kind"] == "regression" else ORDERED_POLICY_MODEL


def build_risk_bands(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint, spec in ENDPOINTS.items():
        subset = predictions.loc[
            predictions["endpoint"].eq(endpoint)
            & predictions["model_variant"].eq(recommended_variant(endpoint))
        ].copy()
        subset["risk_score"] = -subset["prediction"] if spec["kind"] == "regression" else subset["prediction"]
        subset["risk_band"] = subset.groupby("origin_id")["risk_score"].transform(
            lambda values: pd.qcut(values.rank(method="first"), 5, labels=False, duplicates="drop") + 1
        )
        for band, group in subset.groupby("risk_band", sort=True):
            rows.append({
                "endpoint": endpoint, "recommended_model": recommended_variant(endpoint),
                "risk_band": int(band),
                "risk_band_label": "lowest_predicted_risk" if band == 1 else (
                    "highest_predicted_risk" if band == 5 else f"band_{int(band)}"
                ),
                "rows": len(group), "unique_players": group["player_id"].nunique(),
                "mean_prediction": group["prediction"].mean(),
                "mean_actual": group["actual"].mean(),
                "mean_actual_opportunity_share": group["target_opportunity_share_24m"].mean(),
                "reported_fee_eur": group["canonical_transfer_fee_eur"].sum(),
            })
    return pd.DataFrame(rows)


def build_subgroups(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint in ["underutilization_10pct", "underutilization_25pct"]:
        subset = predictions.loc[
            predictions["endpoint"].eq(endpoint)
            & predictions["model_variant"].eq(ORDERED_POLICY_MODEL)
        ].copy()
        subset["age_band"] = pd.cut(
            pd.to_numeric(subset["age_at_transfer"], errors="coerce"),
            bins=[-np.inf, 21, 24, 27, 30, np.inf],
            labels=["<=21", "22-24", "25-27", "28-30", "31+"],
        ).astype("string").fillna("unknown")
        for dimension in [
            "historical_position_group", "age_band", "fee_band", "destination_competition_id",
            "evaluation_year",
        ]:
            for value, group in subset.groupby(dimension, dropna=False):
                rows.append({
                    "endpoint": endpoint, "dimension": dimension, "group": str(value),
                    "rows": len(group), "unique_players": group["player_id"].nunique(),
                    "adequate_rows": len(group) >= 30,
                    "mean_probability": group["prediction"].mean(),
                    "actual_rate": group["actual"].mean(),
                    "brier": np.square(group["actual"] - group["prediction"]).mean(),
                    "mean_opportunity_share": group["target_opportunity_share_24m"].mean(),
                })
    return pd.DataFrame(rows)


def build_decisions(comparisons: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pooled = comparisons.loc[comparisons["window"].eq("pooled_four")].set_index(
        ["endpoint", "comparison"]
    )
    for endpoint, spec in ENDPOINTS.items():
        comparison = "policy_vs_M0_total" if spec["kind"] == "regression" else "ordered_policy_vs_M0_total"
        total = pooled.loc[(endpoint, comparison)]
        market_row = pooled.loc[(endpoint, "M1_vs_M0_paid_deal_market_profile")]
        validated = total["cluster_ci_lower_95"] > 0 and total["origin_wins"] >= 3
        status = (
            "validated_predictive_signal" if validated
            else ("promising_but_uncertain" if total["loss_improvement"] > 0 else "not_validated")
        )
        selected = selections.loc[selections["endpoint"].eq(endpoint)]
        rows.append({
            "endpoint": endpoint, "endpoint_kind": spec["kind"], "status": status,
            "recommended_model": recommended_variant(endpoint),
            "total_loss_improvement": total["loss_improvement"],
            "total_relative_loss_improvement": total["relative_loss_improvement"],
            "total_ci_lower_95": total["cluster_ci_lower_95"],
            "total_ci_upper_95": total["cluster_ci_upper_95"],
            "total_origin_wins": int(total["origin_wins"]),
            "market_profile_improvement": market_row["loss_improvement"],
            "market_profile_ci_lower_95": market_row["cluster_ci_lower_95"],
            "selected_feature_set_counts": json.dumps(
                selected["selected_feature_set"].value_counts().to_dict(), sort_keys=True
            ),
            "claim_guardrail": (
                "Capital-utilization screening only; not tactical quality, causation, realized loss, or ROI."
            ),
        })
    return pd.DataFrame(rows)


def build_checks(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    predictions: pd.DataFrame,
    ordered: pd.DataFrame,
) -> pd.DataFrame:
    checks = []
    def add(check: str, passed: bool, observed: Any, expected: Any, note: str) -> None:
        checks.append({
            "check": check, "passed": bool(passed), "observed": observed,
            "expected": expected, "note": note,
        })
    add("cohort_rows", len(frame) == 1074, len(frame), 1074, "Matches audited paid-transfer cohort.")
    add("cohort_key_unique", frame["transfer_event_id"].is_unique, frame["transfer_event_id"].nunique(), len(frame), "One row per transfer.")
    add("continuous_target_complete", frame["target_opportunity_share_24m"].notna().all(), frame["target_opportunity_share_24m"].notna().sum(), len(frame), "All targets observed.")
    add("continuous_target_bounds", frame["target_opportunity_share_24m"].between(0, 1.05).all(), f"{frame['target_opportunity_share_24m'].min()}..{frame['target_opportunity_share_24m'].max()}", "0..1.05", "Club-capacity share bounds.")
    add("under10_count", int(frame["target_underutilization_10pct"].sum()) == 355, int(frame["target_underutilization_10pct"].sum()), 355, "Matches audit.")
    add("under25_count", int(frame["target_underutilization_25pct"].sum()) == 553, int(frame["target_underutilization_25pct"].sum()), 553, "Matches audit.")
    add("targets_nested", (frame["target_underutilization_10pct"] <= frame["target_underutilization_25pct"]).all(), True, True, "Severe class is nested.")
    forbidden = [
        feature for features in feature_sets.values() for feature in features
        if any(token in feature.lower() for token in ["post12", "post24", "target_", "eligible_"])
    ]
    add("no_posttransfer_predictors", not forbidden, ";".join(sorted(set(forbidden))), "none", "No outcome fields are predictors.")
    key = ["endpoint", "origin_id", "model_variant", "transfer_event_id"]
    add("prediction_key_unique", not predictions.duplicated(key).any(), predictions.duplicated(key).sum(), 0, "One forecast per key.")
    class_predictions = predictions.loc[predictions["endpoint_kind"].eq("classification"), "prediction"]
    add("probability_bounds", class_predictions.between(0, 1).all(), f"{class_predictions.min()}..{class_predictions.max()}", "0..1", "Valid probabilities.")
    add("ordered_probabilities", (ordered["ordered_probability_under10"] <= ordered["ordered_probability_under25"]).all(), int((ordered["ordered_probability_under10"] > ordered["ordered_probability_under25"]).sum()), 0, "Logical nesting enforced.")
    add("ordered_rows", len(ordered) == 584, len(ordered), 584, "One policy row per evaluated transfer.")
    return pd.DataFrame(checks)


def write_readme(
    frame: pd.DataFrame,
    comparisons: pd.DataFrame,
    decisions: pd.DataFrame,
    risk_bands: pd.DataFrame,
    ordered: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> None:
    pooled = comparisons.loc[comparisons["window"].eq("pooled_four")].set_index(
        ["endpoint", "comparison"]
    )
    lines = [
        "# MoveMaker paid-transfer under-utilization model", "",
        "This experiment estimates 24-month sporting opportunity after a paid permanent transfer. It is a capital-utilization screen, not an accounting ROI or tactical-quality model.", "",
        "## Design", "",
        f"- Cohort: {len(frame):,} clean paid permanent arrivals with at least 20 observable destination matches.",
        "- Continuous target: share of destination-club match-minute capacity in the first 24 months.",
        "- Binary targets: under 10% (severe) and under 25% (meaningful) of capacity.",
        "- Four rolling origins evaluate the 2020-2023 seasons. Family, hyperparameters, and feature block are selected using only the preceding validation season.",
        "- Binary policy probabilities are projected to guarantee P(under 10%) <= P(under 25%).",
        "- Primary uncertainty uses 5,000 paired player-cluster bootstrap repetitions.", "",
        "## Feature blocks", "",
    ]
    for model in FIXED_MODEL_ORDER:
        lines.append(f"- `{model}`: {len(feature_sets[model])} predictors.")
    lines += ["", "## Primary results", ""]
    for row in decisions.itertuples(index=False):
        comparison = "policy_vs_M0_total" if row.endpoint_kind == "regression" else "ordered_policy_vs_M0_total"
        total = pooled.loc[(row.endpoint, comparison)]
        lines.append(
            f"- `{row.endpoint}`: `{row.status}`; {total['primary_metric']} improvement {total['loss_improvement']:.6f}, player-clustered 95% CI [{total['cluster_ci_lower_95']:.6f}, {total['cluster_ci_upper_95']:.6f}], {int(total['origin_wins'])}/4 origin wins."
        )
    under10 = risk_bands.loc[risk_bands["endpoint"].eq("underutilization_10pct")].set_index("risk_band")
    under25 = risk_bands.loc[risk_bands["endpoint"].eq("underutilization_25pct")].set_index("risk_band")
    lines += [
        "", "## Risk separation", "",
        f"- Severe under-utilization: lowest predicted-risk quintile {under10.loc[1, 'mean_actual']:.1%} observed vs highest {under10.loc[5, 'mean_actual']:.1%}.",
        f"- Meaningful under-utilization: lowest predicted-risk quintile {under25.loc[1, 'mean_actual']:.1%} observed vs highest {under25.loc[5, 'mean_actual']:.1%}.",
        f"- Raw policy threshold violations: {int(ordered['raw_order_violation'].sum())}/{len(ordered)}; ordered policy violations: 0.",
        "", "## Guardrails", "",
        "A positive result supports pre-transfer screening of likely under-use. It does not identify why the under-use occurred, prove player-club incompatibility, value the lost minutes in euros, measure performance quality, or estimate ROI. Absolute probabilities remain candidates for a separate calibration audit before live use.",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    protected_before = protected_tree_sha256()
    frame, feature_sets, feature_manifest = build_model_frame()
    predictions, origin_results, tuning, selections = fit_models(frame, feature_sets)
    predictions, ordered = add_ordered_policy(predictions)
    comparisons = build_comparisons(predictions)
    risk_bands = build_risk_bands(predictions)
    subgroups = build_subgroups(predictions)
    decisions = build_decisions(comparisons, selections)
    checks = build_checks(frame, feature_sets, predictions, ordered)
    if not checks["passed"].all():
        raise RuntimeError(f"Build checks failed: {checks.loc[~checks['passed'], 'check'].tolist()}")

    target_columns = [
        "transfer_event_id", "player_id", "transfer_date", "season_start_year",
        "from_club_name", "to_club_name", "canonical_transfer_fee_eur", "fee_band",
        "club_destination_post24_games", "appearance_post24_destination_minutes",
        "target_opportunity_share_24m", "target_underutilization_10pct",
        "target_underutilization_25pct",
    ]
    tables = {
        "underutilization_targets.csv": frame[target_columns],
        "underutilization_feature_manifest.csv": feature_manifest,
        "validation_candidate_results.csv": tuning,
        "feature_block_selections.csv": selections,
        "underutilization_predictions.csv": predictions,
        "underutilization_origin_results.csv": origin_results,
        "underutilization_comparison_results.csv": comparisons,
        "ordered_policy_predictions.csv": ordered,
        "underutilization_risk_bands.csv": risk_bands,
        "underutilization_subgroup_performance.csv": subgroups,
        "underutilization_decision_summary.csv": decisions,
        "build_checks.csv": checks,
    }
    for name, table in tables.items():
        table.to_csv(OUTPUT_DIR / name, index=False)
    source_manifest = pd.DataFrame([
        {
            "source": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in [MASTER_PATH, AUDIT_PATH, AUDIT_VERIFICATION_PATH]
    ])
    source_manifest.to_csv(OUTPUT_DIR / "source_manifest.csv", index=False)
    tables["source_manifest.csv"] = source_manifest
    write_readme(frame, comparisons, decisions, risk_bands, ordered, feature_sets)

    if protected_before != protected_tree_sha256():
        raise RuntimeError("A protected verified output changed during modeling.")
    run_summary = {
        "cohort_rows": len(frame), "unique_players": frame["player_id"].nunique(),
        "date_min": frame["transfer_date"].min().date().isoformat(),
        "date_max": frame["transfer_date"].max().date().isoformat(),
        "target_counts": {
            "under10": int(frame["target_underutilization_10pct"].sum()),
            "under25": int(frame["target_underutilization_25pct"].sum()),
        },
        "feature_counts": {name: len(features) for name, features in feature_sets.items()},
        "evaluation_rows": int(len(ordered)), "origins": [origin[0] for origin in ORIGINS],
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "raw_probability_order_violations": int(ordered["raw_order_violation"].sum()),
        "ordered_probability_order_violations": int(
            (ordered["ordered_probability_under10"] > ordered["ordered_probability_under25"]).sum()
        ),
        "decisions": decisions.set_index("endpoint")["status"].to_dict(),
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

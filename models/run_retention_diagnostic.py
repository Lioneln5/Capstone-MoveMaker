"""Model sustained sporting involvement and destination-spell retention.

This diagnostic deliberately separates three questions:

1. At transfer time, can pre-transfer evidence predict meaningful involvement
   during months 13-24 at the destination?
2. At transfer time, can it predict an uninterrupted two-year spell at the
   destination?
3. Among players still at the destination after one year, does first-year
   involvement materially improve prediction of second-year involvement?

The primary cohort is restricted to canonically classified permanent arrivals.
Loans and loan returns are reported in the audit but are not mixed into the
retention models.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for dependency_dir in [ROOT / "models", ROOT / "scripts"]:
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

import run_coarse_club_context_diagnostic as coarse
import run_opportunity_transferability_decomposition as opportunity
from train_compatibility_models import fit_preprocessor, sha256_file


MASTER_PATH = coarse.MASTER_PATH
CANONICAL_TRANSFER_PATH = (
    ROOT / "Data" / "processed" / "canonical_integration" / "canonical_transfer_history.csv"
)
OUTPUT_DIR = ROOT / "Data" / "processed" / "retention_diagnostic"

AS_OF_DATE = pd.Timestamp("2026-08-10")
SPORTING_THRESHOLD = 0.10
RANDOM_SEED = 20260810
BOOTSTRAP_REPETITIONS = 5000
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]

ORIGINS = coarse.ORIGINS
WINDOWS = {
    "pooled_four": [2020, 2021, 2022, 2023],
    "mature_three": [2021, 2022, 2023],
    "terminal_two": [2022, 2023],
}

AT_TRANSFER_MODELS = [
    "M0_prevalence_only",
    "M1_market_profile",
    "M2_plus_prior_opportunity",
    "M3_plus_player_performance",
    "M4_plus_destination_context",
    "M5_plus_origin_transition_context",
]
LANDMARK_MODELS = [
    "M0_prevalence_only",
    "M5_plus_origin_transition_context",
    "M6_plus_first_year_involvement",
]

TARGET_SPECS = {
    "sporting_retention_y2_10pct": {
        "stage": "at_transfer",
        "target": "sporting_retention_y2_10pct",
        "models": AT_TRANSFER_MODELS,
        "description": (
            "At least 10% of destination club match-minute capacity during days 366-730."
        ),
    },
    "destination_spell_retained_730d": {
        "stage": "at_transfer",
        "target": "destination_spell_retained_730d",
        "models": AT_TRANSFER_MODELS,
        "description": "No later outbound event from the destination within 730 days.",
    },
    "landmark_sporting_retention_y2_10pct": {
        "stage": "landmark_365d",
        "target": "sporting_retention_y2_10pct",
        "models": LANDMARK_MODELS,
        "description": (
            "Year-two sporting retention among players with no outbound event in the first 365 days."
        ),
    },
}

COMPARISONS = {
    "sporting_retention_y2_10pct": [
        ("M1_vs_M0_market_profile", "M0_prevalence_only", "M1_market_profile"),
        ("M2_vs_M1_prior_opportunity", "M1_market_profile", "M2_plus_prior_opportunity"),
        ("M3_vs_M2_player_performance", "M2_plus_prior_opportunity", "M3_plus_player_performance"),
        ("M4_vs_M3_destination_context", "M3_plus_player_performance", "M4_plus_destination_context"),
        ("M5_vs_M4_origin_transition", "M4_plus_destination_context", "M5_plus_origin_transition_context"),
        ("M5_vs_M0_all_transfer_time", "M0_prevalence_only", "M5_plus_origin_transition_context"),
    ],
    "destination_spell_retained_730d": [
        ("M1_vs_M0_market_profile", "M0_prevalence_only", "M1_market_profile"),
        ("M2_vs_M1_prior_opportunity", "M1_market_profile", "M2_plus_prior_opportunity"),
        ("M3_vs_M2_player_performance", "M2_plus_prior_opportunity", "M3_plus_player_performance"),
        ("M4_vs_M3_destination_context", "M3_plus_player_performance", "M4_plus_destination_context"),
        ("M5_vs_M4_origin_transition", "M4_plus_destination_context", "M5_plus_origin_transition_context"),
        ("M5_vs_M0_all_transfer_time", "M0_prevalence_only", "M5_plus_origin_transition_context"),
    ],
    "landmark_sporting_retention_y2_10pct": [
        ("M5_vs_M0_all_transfer_time", "M0_prevalence_only", "M5_plus_origin_transition_context"),
        (
            "M6_vs_M5_first_year_involvement",
            "M5_plus_origin_transition_context",
            "M6_plus_first_year_involvement",
        ),
        ("M6_vs_M0_landmark_total", "M0_prevalence_only", "M6_plus_first_year_involvement"),
    ],
}

FIRST_YEAR_FEATURES = [
    "first_year_destination_opportunity_share",
    "first_year_destination_appearance_rate",
    "first_year_destination_start_rate",
    "first_year_destination_substitute_selection_rate",
    "first_year_destination_goals_per90",
    "first_year_destination_assists_per90",
]

PROTECTED_DIRS = coarse.PROTECTED_DIRS + [
    coarse.OUTPUT_DIR,
    opportunity.OUTPUT_DIR,
]


def stable_seed(*parts: str) -> int:
    token = "|".join(parts).encode("utf-8")
    return (RANDOM_SEED + int(hashlib.sha256(token).hexdigest()[:8], 16)) % (2**32 - 1)


def protected_tree_sha256() -> str:
    digest = hashlib.sha256()
    for directory in sorted(set(PROTECTED_DIRS)):
        if not directory.exists():
            continue
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def clean_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype(str)


def canonical_transfers() -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = [
        "canonical_transfer_event_id",
        "canonical_player_id",
        "transfer_date",
        "from_club_id",
        "to_club_id",
        "canonical_transfer_type",
        "transfer_type_conflict",
        "event_conflict",
        "event_resolution_status",
    ]
    frame = pd.read_csv(CANONICAL_TRANSFER_PATH, usecols=columns, low_memory=False)
    frame = frame.rename(columns={"canonical_player_id": "player_id"})
    for column in ["player_id", "from_club_id", "to_club_id"]:
        frame[column] = clean_id(frame[column])
    raw_date = frame["transfer_date"].copy()
    frame["transfer_date"] = pd.to_datetime(frame["transfer_date"], errors="coerce")
    audit = {
        "canonical_rows": len(frame),
        "canonical_invalid_or_missing_dates": int(frame["transfer_date"].isna().sum()),
        "canonical_nonblank_invalid_dates": int(
            (frame["transfer_date"].isna() & raw_date.fillna("").astype(str).str.strip().ne("")).sum()
        ),
    }
    natural_key = ["player_id", "transfer_date", "from_club_id", "to_club_id"]
    if frame.duplicated(natural_key, keep=False).any():
        raise RuntimeError("Canonical transfer natural key is not unique.")
    return frame, audit


def build_retention_frame() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    master = pd.read_csv(MASTER_PATH, low_memory=False)
    broad = coarse.cohort_from_master(master)
    coarse.assert_preflight(broad)
    frame, opportunity_feature_sets, base_manifest = opportunity.build_model_frame(broad)

    canonical, canonical_audit = canonical_transfers()
    for column in ["player_id", "from_club_id", "to_club_id"]:
        frame[column] = clean_id(frame[column])
    natural_key = ["player_id", "transfer_date", "from_club_id", "to_club_id"]
    frame = frame.merge(canonical, on=natural_key, how="left", validate="many_to_one")
    canonical_match_count = int(frame["canonical_transfer_event_id"].notna().sum())
    if canonical_match_count != len(frame):
        raise RuntimeError("Not every broad-cohort arrival mapped to a canonical transfer event.")

    type_counts = frame["canonical_transfer_type"].fillna("blank_unclassified").value_counts()
    audit_rows: list[dict[str, Any]] = [
        {"area": "cohort", "metric": "broad_context_cohort_rows", "value": len(frame)},
        {"area": "cohort", "metric": "canonical_arrival_matches", "value": canonical_match_count},
    ]
    for transfer_type, count in type_counts.items():
        audit_rows.append(
            {"area": "arrival_type", "metric": str(transfer_type), "value": int(count)}
        )
    for key, value in canonical_audit.items():
        audit_rows.append({"area": "canonical_source", "metric": key, "value": value})

    conflict = coarse.as_bool(frame["event_conflict"]) | coarse.as_bool(frame["transfer_type_conflict"])
    permanent = frame.loc[frame["canonical_transfer_type"].eq("transfer") & ~conflict].copy()
    audit_rows.extend(
        [
            {"area": "cohort", "metric": "permanent_arrivals_before_conflict_filter", "value": int(frame["canonical_transfer_type"].eq("transfer").sum())},
            {"area": "cohort", "metric": "arrival_event_conflicts_excluded", "value": int((frame["canonical_transfer_type"].eq("transfer") & conflict).sum())},
            {"area": "cohort", "metric": "primary_permanent_arrivals", "value": len(permanent)},
        ]
    )

    year2_minutes = pd.to_numeric(
        permanent["appearance_post24_destination_minutes"], errors="coerce"
    ) - pd.to_numeric(permanent["appearance_post12_destination_minutes"], errors="coerce")
    year2_club_games = pd.to_numeric(
        permanent["club_destination_post24_games"], errors="coerce"
    ) - pd.to_numeric(permanent["club_destination_post12_games"], errors="coerce")
    if year2_minutes.lt(0).any() or year2_club_games.lt(0).any():
        raise RuntimeError("Nested sporting windows produced a negative year-two difference.")
    permanent["year2_destination_minutes"] = year2_minutes
    permanent["year2_destination_club_games"] = year2_club_games
    permanent["year2_destination_opportunity_share"] = coarse.safe_divide(
        year2_minutes, 90.0 * year2_club_games
    ).clip(lower=0.0, upper=1.05)
    permanent["sporting_retention_y2_10pct"] = permanent[
        "year2_destination_opportunity_share"
    ].ge(SPORTING_THRESHOLD).where(permanent["year2_destination_opportunity_share"].notna())

    # A spell ends at the first later outbound event from the original destination,
    # regardless of whether that event is a permanent transfer, loan, or loan return.
    # This is intentionally an uninterrupted-club-spell label, not contract termination.
    outbound = canonical.loc[canonical["transfer_date"].le(AS_OF_DATE)].copy()
    arrivals = permanent[["transfer_event_id", "player_id", "transfer_date", "to_club_id"]].rename(
        columns={"transfer_date": "arrival_date", "to_club_id": "from_club_id"}
    )
    candidates = arrivals.merge(
        outbound,
        on=["player_id", "from_club_id"],
        how="left",
        suffixes=("_arrival", "_outbound"),
    )
    candidates = candidates.loc[candidates["transfer_date"].gt(candidates["arrival_date"])].copy()
    first_outbound = (
        candidates.sort_values(["transfer_event_id", "transfer_date", "canonical_transfer_event_id"])
        .drop_duplicates("transfer_event_id", keep="first")
        [[
            "transfer_event_id",
            "canonical_transfer_event_id",
            "transfer_date",
            "to_club_id",
            "canonical_transfer_type",
        ]]
        .rename(
            columns={
                "canonical_transfer_event_id": "first_outbound_event_id",
                "transfer_date": "first_outbound_date",
                "to_club_id": "first_outbound_to_club_id",
                "canonical_transfer_type": "first_outbound_transfer_type",
            }
        )
    )
    permanent = permanent.merge(first_outbound, on="transfer_event_id", how="left", validate="one_to_one")
    permanent["days_to_first_outbound"] = (
        permanent["first_outbound_date"] - permanent["transfer_date"]
    ).dt.days
    permanent["destination_spell_retained_365d"] = (
        permanent["days_to_first_outbound"].isna()
        | permanent["days_to_first_outbound"].gt(365)
    ).astype(int)
    permanent["destination_spell_retained_730d"] = (
        permanent["days_to_first_outbound"].isna()
        | permanent["days_to_first_outbound"].gt(730)
    ).astype(int)
    if (permanent["transfer_date"] + pd.to_timedelta(730, unit="D")).gt(AS_OF_DATE).any():
        raise RuntimeError("At least one primary arrival lacks a complete 730-day administrative horizon.")

    first_year_games = pd.to_numeric(permanent["club_destination_post12_games"], errors="coerce")
    first_year_minutes = pd.to_numeric(
        permanent["appearance_post12_destination_minutes"], errors="coerce"
    )
    permanent["first_year_destination_opportunity_share"] = coarse.safe_divide(
        first_year_minutes, 90.0 * first_year_games
    ).clip(0.0, 1.05)
    permanent["first_year_destination_appearance_rate"] = coarse.safe_divide(
        permanent["appearance_post12_destination_games"], first_year_games
    )
    permanent["first_year_destination_start_rate"] = coarse.safe_divide(
        permanent["lineup_post12_destination_starts"], first_year_games
    )
    permanent["first_year_destination_substitute_selection_rate"] = coarse.safe_divide(
        permanent["lineup_post12_destination_substitute_selections"], first_year_games
    )
    permanent["first_year_destination_goals_per90"] = 90.0 * coarse.safe_divide(
        permanent["appearance_post12_destination_goals"], first_year_minutes
    )
    permanent["first_year_destination_assists_per90"] = 90.0 * coarse.safe_divide(
        permanent["appearance_post12_destination_assists"], first_year_minutes
    )

    feature_sets = {
        "M0_prevalence_only": [],
        "M1_market_profile": opportunity_feature_sets["M1_market_profile"],
        "M2_plus_prior_opportunity": opportunity_feature_sets["M2_plus_prior_opportunity"],
        "M3_plus_player_performance": opportunity_feature_sets["M3_plus_player_performance"],
        "M4_plus_destination_context": opportunity_feature_sets["M4_plus_destination_context"],
        "M5_plus_origin_transition_context": opportunity_feature_sets["M5_plus_origin_transition_context"],
    }
    feature_sets["M6_plus_first_year_involvement"] = (
        feature_sets["M5_plus_origin_transition_context"] + FIRST_YEAR_FEATURES
    )
    for model, features in feature_sets.items():
        if len(features) != len(set(features)):
            raise RuntimeError(f"Duplicate feature in {model}.")
        missing = sorted(set(features) - set(permanent.columns))
        if missing:
            raise RuntimeError(f"Missing retention feature(s) for {model}: {missing}")

    manifest = base_manifest.rename(columns={"feature_set": "source_feature_set"}).copy()
    alias = {
        "M0_mean_only": "M0_prevalence_only",
        "M1_market_profile": "M1_market_profile",
        "M2_plus_prior_opportunity": "M2_plus_prior_opportunity",
        "M3_plus_player_performance": "M3_plus_player_performance",
        "M4_plus_destination_context": "M4_plus_destination_context",
        "M5_plus_origin_transition_context": "M5_plus_origin_transition_context",
    }
    manifest["model_variant"] = manifest["source_feature_set"].map(alias)
    manifest = manifest.loc[manifest["model_variant"].notna()].copy()
    first_year_rows = []
    for order, feature in enumerate(FIRST_YEAR_FEATURES, start=1):
        first_year_rows.append(
            {
                "source_feature_set": "retention_landmark",
                "feature_order": len(feature_sets["M5_plus_origin_transition_context"]) + order,
                "feature": feature,
                "feature_kind": "derived_landmark",
                "source_fields": feature,
                "timing_classification": "observed during days 0-365 after transfer",
                "rationale": "Available only to the declared 365-day landmark model.",
                "raw_feature_count": len(feature_sets["M6_plus_first_year_involvement"]),
                "model_variant": "M6_plus_first_year_involvement",
            }
        )
    manifest = pd.concat([manifest, pd.DataFrame(first_year_rows)], ignore_index=True)

    threshold_rows = []
    observed_share = permanent["year2_destination_opportunity_share"].dropna()
    for threshold in [0.05, 0.10, 0.25, 0.50]:
        threshold_rows.append(
            {
                "threshold": threshold,
                "eligible_rows": len(observed_share),
                "positive_rows": int(observed_share.ge(threshold).sum()),
                "positive_rate": float(observed_share.ge(threshold).mean()),
            }
        )
    threshold_sensitivity = pd.DataFrame(threshold_rows)

    audit_rows.extend(
        [
            {"area": "target", "metric": "sporting_target_observed_rows", "value": int(permanent["sporting_retention_y2_10pct"].notna().sum())},
            {"area": "target", "metric": "sporting_target_positive_rows", "value": int(permanent["sporting_retention_y2_10pct"].astype(float).fillna(0.0).sum())},
            {"area": "target", "metric": "sporting_target_positive_rate", "value": float(permanent["sporting_retention_y2_10pct"].astype(float).mean())},
            {"area": "target", "metric": "spell_retained_365d_rows", "value": int(permanent["destination_spell_retained_365d"].sum())},
            {"area": "target", "metric": "spell_retained_365d_rate", "value": float(permanent["destination_spell_retained_365d"].mean())},
            {"area": "target", "metric": "spell_retained_730d_rows", "value": int(permanent["destination_spell_retained_730d"].sum())},
            {"area": "target", "metric": "spell_retained_730d_rate", "value": float(permanent["destination_spell_retained_730d"].mean())},
            {"area": "cohort", "metric": "landmark_365d_at_risk_rows", "value": int((permanent["destination_spell_retained_365d"].eq(1) & permanent["sporting_retention_y2_10pct"].notna()).sum())},
        ]
    )
    return permanent, pd.DataFrame(audit_rows), threshold_sensitivity, manifest, feature_sets


def classification_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=int)
    predicted = np.clip(np.asarray(predicted, dtype=float), 1e-6, 1.0 - 1e-6)
    result: dict[str, float | int] = {
        "n": len(actual),
        "actual_rate": float(actual.mean()),
        "predicted_rate": float(predicted.mean()),
        "brier": float(brier_score_loss(actual, predicted)),
        "log_loss": float(log_loss(actual, predicted, labels=[0, 1])),
        "accuracy_at_0_5": float(np.mean((predicted >= 0.5) == actual)),
    }
    if len(np.unique(actual)) == 2:
        result["roc_auc"] = float(roc_auc_score(actual, predicted))
        result["average_precision"] = float(average_precision_score(actual, predicted))
    else:
        result["roc_auc"] = float("nan")
        result["average_precision"] = float("nan")
    return result


def tune_and_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    evaluation: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[np.ndarray, dict[str, Any], int]:
    y_train = train[target].astype(int).to_numpy()
    y_validation = validation[target].astype(int).to_numpy()
    if not features:
        validation_prediction = np.repeat(float(y_train.mean()), len(validation))
        fit_rate = float(pd.concat([train, validation], ignore_index=True)[target].astype(int).mean())
        evaluation_prediction = np.repeat(fit_rate, len(evaluation))
        return evaluation_prediction, {
            "selected_c": None,
            **classification_metrics(y_validation, validation_prediction),
        }, 0

    preprocessor = fit_preprocessor(train, features)
    x_train = preprocessor.transform(train)
    x_validation = preprocessor.transform(validation)
    candidates = []
    for c_value in C_GRID:
        model = LogisticRegression(
            C=c_value,
            penalty="l2",
            solver="liblinear",
            max_iter=3000,
            random_state=RANDOM_SEED,
        )
        model.fit(x_train, y_train)
        prediction = np.clip(model.predict_proba(x_validation)[:, 1], 1e-6, 1 - 1e-6)
        result = classification_metrics(y_validation, prediction)
        key = (float(result["brier"]), float(result["log_loss"]), c_value)
        candidates.append((key, c_value, result))
    _, selected_c, selected_metrics = min(candidates, key=lambda item: item[0])

    fit_frame = pd.concat([train, validation], ignore_index=True)
    fit_preprocessor_obj = fit_preprocessor(fit_frame, features)
    x_fit = fit_preprocessor_obj.transform(fit_frame)
    x_evaluation = fit_preprocessor_obj.transform(evaluation)
    model = LogisticRegression(
        C=selected_c,
        penalty="l2",
        solver="liblinear",
        max_iter=3000,
        random_state=RANDOM_SEED,
    )
    model.fit(x_fit, fit_frame[target].astype(int).to_numpy())
    prediction = np.clip(model.predict_proba(x_evaluation)[:, 1], 1e-6, 1 - 1e-6)
    return prediction, {"selected_c": selected_c, **selected_metrics}, x_fit.shape[1]


def endpoint_frame(frame: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    spec = TARGET_SPECS[endpoint]
    result = frame.loc[frame[spec["target"]].notna()].copy()
    if spec["stage"] == "landmark_365d":
        result = result.loc[result["destination_spell_retained_365d"].eq(1)].copy()
    return result


def fit_models(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions: list[dict[str, Any]] = []
    origin_results: list[dict[str, Any]] = []
    earliest_year = int(frame["season_start_year"].min())
    for endpoint, spec in TARGET_SPECS.items():
        data = endpoint_frame(frame, endpoint)
        target = spec["target"]
        for origin_id, train_end, validation_year, evaluation_year in ORIGINS:
            train = data.loc[data["season_start_year"].le(train_end)].copy()
            validation = data.loc[data["season_start_year"].eq(validation_year)].copy()
            evaluation = data.loc[data["season_start_year"].eq(evaluation_year)].copy()
            if min(len(train), len(validation), len(evaluation)) == 0:
                raise RuntimeError(f"Empty split for {endpoint}/{origin_id}.")
            evaluation = evaluation.sort_values("transfer_event_id").reset_index(drop=True)
            for model_variant in spec["models"]:
                print(f"endpoint={endpoint} origin={origin_id} model={model_variant}", flush=True)
                probability, validation_metrics, encoded_count = tune_and_predict(
                    train, validation, evaluation, feature_sets[model_variant], target
                )
                evaluation_metrics = classification_metrics(
                    evaluation[target].astype(int).to_numpy(), probability
                )
                origin_results.append(
                    {
                        "endpoint": endpoint,
                        "analysis_stage": spec["stage"],
                        "origin_id": origin_id,
                        "model_variant": model_variant,
                        "model_family": "l2_logistic_regression",
                        "train_start_year": earliest_year,
                        "train_end_year": train_end,
                        "validation_year": validation_year,
                        "evaluation_year": evaluation_year,
                        "train_rows": len(train),
                        "validation_rows": len(validation),
                        "evaluation_rows": len(evaluation),
                        "raw_feature_count": len(feature_sets[model_variant]),
                        "encoded_feature_count": encoded_count,
                        "selected_c": validation_metrics["selected_c"],
                        **{f"validation_{key}": value for key, value in validation_metrics.items() if key != "selected_c"},
                        **{f"evaluation_{key}": value for key, value in evaluation_metrics.items()},
                    }
                )
                for index, (_, row) in enumerate(evaluation.iterrows()):
                    actual = int(row[target])
                    prob = float(probability[index])
                    predictions.append(
                        {
                            "endpoint": endpoint,
                            "analysis_stage": spec["stage"],
                            "origin_id": origin_id,
                            "evaluation_year": evaluation_year,
                            "transfer_event_id": row["transfer_event_id"],
                            "player_id": row["player_id"],
                            "transfer_date": row["transfer_date"].date().isoformat(),
                            "model_variant": model_variant,
                            "actual": actual,
                            "predicted_probability": prob,
                            "squared_error": (actual - prob) ** 2,
                        }
                    )
    return pd.DataFrame(predictions), pd.DataFrame(origin_results)


def paired_predictions(
    predictions: pd.DataFrame,
    endpoint: str,
    baseline: str,
    candidate: str,
    years: list[int],
) -> pd.DataFrame:
    selected = predictions.loc[
        predictions["endpoint"].eq(endpoint) & predictions["evaluation_year"].isin(years)
    ]
    keys = ["endpoint", "origin_id", "evaluation_year", "transfer_event_id", "player_id"]
    left = selected.loc[selected["model_variant"].eq(baseline), keys + ["actual", "predicted_probability"]].rename(
        columns={"predicted_probability": "baseline_probability"}
    )
    right = selected.loc[selected["model_variant"].eq(candidate), keys + ["actual", "predicted_probability"]].rename(
        columns={"actual": "candidate_actual", "predicted_probability": "candidate_probability"}
    )
    paired = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if not np.array_equal(paired["actual"].to_numpy(), paired["candidate_actual"].to_numpy()):
        raise RuntimeError("Actual values disagree in paired comparison.")
    return paired.drop(columns="candidate_actual")


def bootstrap_improvement(paired: pd.DataFrame, seed: int) -> dict[str, Any]:
    working = paired.copy()
    working["brier_improvement"] = (
        np.square(working["actual"] - working["baseline_probability"])
        - np.square(working["actual"] - working["candidate_probability"])
    )
    clusters = working.groupby("player_id", sort=True)["brier_improvement"].agg(["sum", "size"])
    sums = clusters["sum"].to_numpy(dtype=float)
    sizes = clusters["size"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(clusters), size=(BOOTSTRAP_REPETITIONS, len(clusters)))
    values = sums[draw].sum(axis=1) / sizes[draw].sum(axis=1)
    return {
        "unique_player_clusters": len(clusters),
        "cluster_ci_lower_95": float(np.quantile(values, 0.025)),
        "cluster_ci_upper_95": float(np.quantile(values, 0.975)),
        "cluster_probability_candidate_better": float(np.mean(values > 0)),
        "bootstrap_seed": seed,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
    }


def stability_label(improvement: float, origin_values: np.ndarray, ci_low: float, ci_high: float) -> str:
    wins = int(np.sum(origin_values > 0))
    if improvement > 0 and ci_low > 0 and wins >= max(3, len(origin_values) - 1):
        return "stable_positive"
    if improvement < 0 and ci_high < 0 and wins <= 1:
        return "stable_negative"
    if improvement > 0 and wins >= math.ceil(len(origin_values) / 2):
        return "mixed_positive"
    if improvement < 0 and wins < math.ceil(len(origin_values) / 2):
        return "mixed_negative"
    return "mixed"


def summarize_comparisons(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint, comparisons in COMPARISONS.items():
        for comparison, baseline, candidate in comparisons:
            for window, years in WINDOWS.items():
                paired = paired_predictions(predictions, endpoint, baseline, candidate, years)
                actual = paired["actual"].to_numpy(dtype=int)
                base = paired["baseline_probability"].to_numpy(dtype=float)
                cand = paired["candidate_probability"].to_numpy(dtype=float)
                base_metrics = classification_metrics(actual, base)
                candidate_metrics = classification_metrics(actual, cand)
                improvement = float(base_metrics["brier"] - candidate_metrics["brier"])
                origin_improvements = []
                origin_baseline_aucs = []
                origin_candidate_aucs = []
                origin_sizes = []
                for _, group in paired.groupby("origin_id", sort=True):
                    origin_improvements.append(
                        float(
                            np.mean(np.square(group["actual"] - group["baseline_probability"]))
                            - np.mean(np.square(group["actual"] - group["candidate_probability"]))
                        )
                    )
                    group_actual = group["actual"].to_numpy(dtype=int)
                    origin_baseline_aucs.append(
                        float(roc_auc_score(group_actual, group["baseline_probability"]))
                    )
                    origin_candidate_aucs.append(
                        float(roc_auc_score(group_actual, group["candidate_probability"]))
                    )
                    origin_sizes.append(len(group))
                origin_values = np.asarray(origin_improvements)
                origin_weights = np.asarray(origin_sizes, dtype=float)
                bootstrap = bootstrap_improvement(
                    paired, stable_seed(endpoint, comparison, window, "player_cluster")
                )
                rows.append(
                    {
                        "endpoint": endpoint,
                        "analysis_stage": TARGET_SPECS[endpoint]["stage"],
                        "comparison": comparison,
                        "baseline_model": baseline,
                        "candidate_model": candidate,
                        "window": window,
                        "evaluation_years": ";".join(str(year) for year in years),
                        "pooled_n": len(paired),
                        "pooled_actual_rate": float(actual.mean()),
                        "baseline_brier": base_metrics["brier"],
                        "candidate_brier": candidate_metrics["brier"],
                        "brier_improvement": improvement,
                        "relative_brier_improvement": improvement / float(base_metrics["brier"]),
                        "baseline_roc_auc": base_metrics["roc_auc"],
                        "candidate_roc_auc": candidate_metrics["roc_auc"],
                        "roc_auc_change": float(candidate_metrics["roc_auc"] - base_metrics["roc_auc"]),
                        "weighted_origin_baseline_roc_auc": float(
                            np.average(origin_baseline_aucs, weights=origin_weights)
                        ),
                        "weighted_origin_candidate_roc_auc": float(
                            np.average(origin_candidate_aucs, weights=origin_weights)
                        ),
                        "weighted_origin_roc_auc_change": float(
                            np.average(origin_candidate_aucs, weights=origin_weights)
                            - np.average(origin_baseline_aucs, weights=origin_weights)
                        ),
                        "baseline_log_loss": base_metrics["log_loss"],
                        "candidate_log_loss": candidate_metrics["log_loss"],
                        "origins_candidate_better": int(np.sum(origin_values > 0)),
                        "origin_win_rate": float(np.mean(origin_values > 0)),
                        "worst_origin_improvement": float(origin_values.min()),
                        "best_origin_improvement": float(origin_values.max()),
                        **bootstrap,
                        "stability_label": stability_label(
                            improvement,
                            origin_values,
                            float(bootstrap["cluster_ci_lower_95"]),
                            float(bootstrap["cluster_ci_upper_95"]),
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_survival_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(frame)
    for day in [90, 180, 365, 545, 730]:
        retained = frame["days_to_first_outbound"].isna() | frame["days_to_first_outbound"].gt(day)
        rows.append(
            {
                "day": day,
                "at_risk_arrivals": total,
                "outbound_by_day": int((~retained).sum()),
                "uninterrupted_spell_rows": int(retained.sum()),
                "uninterrupted_spell_rate": float(retained.mean()),
            }
        )
    return pd.DataFrame(rows)


def make_summary(
    frame: pd.DataFrame,
    audit: pd.DataFrame,
    predictions: pd.DataFrame,
    comparisons: pd.DataFrame,
    protected_unchanged: bool,
) -> dict[str, Any]:
    headline: dict[str, Any] = {}
    headline_comparisons = {
        "sporting_at_transfer_all_features_vs_prevalence": (
            "sporting_retention_y2_10pct",
            "M5_vs_M0_all_transfer_time",
        ),
        "spell_at_transfer_all_features_vs_prevalence": (
            "destination_spell_retained_730d",
            "M5_vs_M0_all_transfer_time",
        ),
        "landmark_first_year_increment": (
            "landmark_sporting_retention_y2_10pct",
            "M6_vs_M5_first_year_involvement",
        ),
    }
    for key, (endpoint, comparison) in headline_comparisons.items():
        row = comparisons.loc[
            comparisons["endpoint"].eq(endpoint)
            & comparisons["comparison"].eq(comparison)
            & comparisons["window"].eq("pooled_four")
        ].iloc[0]
        headline[key] = {
            "pooled_n": int(row["pooled_n"]),
            "brier_improvement": float(row["brier_improvement"]),
            "relative_brier_improvement": float(row["relative_brier_improvement"]),
            "weighted_origin_candidate_roc_auc": float(
                row["weighted_origin_candidate_roc_auc"]
            ),
            "cluster_ci_lower_95": float(row["cluster_ci_lower_95"]),
            "cluster_ci_upper_95": float(row["cluster_ci_upper_95"]),
            "stability_label": row["stability_label"],
        }
    return {
        "as_of_date": AS_OF_DATE.date().isoformat(),
        "sporting_threshold": SPORTING_THRESHOLD,
        "primary_permanent_arrivals": len(frame),
        "sporting_target_observed_rows": int(frame["sporting_retention_y2_10pct"].notna().sum()),
        "sporting_target_positive_rate": float(frame["sporting_retention_y2_10pct"].astype(float).mean()),
        "spell_retained_365d_rate": float(frame["destination_spell_retained_365d"].mean()),
        "spell_retained_730d_rate": float(frame["destination_spell_retained_730d"].mean()),
        "landmark_rows": len(endpoint_frame(frame, "landmark_sporting_retention_y2_10pct")),
        "prediction_rows": len(predictions),
        "audit_rows": len(audit),
        "headline": headline,
        "master_sha256": sha256_file(MASTER_PATH),
        "canonical_transfer_sha256": sha256_file(CANONICAL_TRANSFER_PATH),
        "protected_outputs_unchanged": protected_unchanged,
    }


def write_readme(summary: dict[str, Any]) -> None:
    h = summary["headline"]
    sport = h["sporting_at_transfer_all_features_vs_prevalence"]
    spell = h["spell_at_transfer_all_features_vs_prevalence"]
    landmark = h["landmark_first_year_increment"]
    text = f"""# MoveMaker retention diagnostic

This is a first predictability test for retention, not a claim that exact contract duration is solved.

## Frozen questions

- **Sporting retention:** among canonically classified permanent arrivals, did the player earn at least {SPORTING_THRESHOLD:.0%} of destination club match-minute capacity during days 366-730?
- **Destination-spell retention:** was there no later outbound event from the original destination within 730 days?
- **365-day landmark update:** among players with no outbound event during the first 365 days, does observed first-year involvement improve prediction of second-year sporting retention?

An outbound loan ends the uninterrupted destination spell even though it does not necessarily end the player's contract. Therefore `destination_spell_retained_730d` must not be described as contract retention.

## Cohorts

- Permanent arrivals after conflict exclusions: {summary['primary_permanent_arrivals']:,}
- Sporting-label rows: {summary['sporting_target_observed_rows']:,}; positive rate: {summary['sporting_target_positive_rate']:.1%}
- Uninterrupted spell at 365 days: {summary['spell_retained_365d_rate']:.1%}
- Uninterrupted spell at 730 days: {summary['spell_retained_730d_rate']:.1%}
- Players in the 365-day landmark analysis: {summary['landmark_rows']:,}

## Pooled rolling-origin headlines (evaluation years 2020-2023)

- At-transfer sporting model, all pre-transfer blocks vs prevalence: Brier improvement {sport['brier_improvement']:+.6f}, 95% player-cluster CI [{sport['cluster_ci_lower_95']:+.6f}, {sport['cluster_ci_upper_95']:+.6f}], weighted-origin candidate AUC {sport['weighted_origin_candidate_roc_auc']:.3f}, `{sport['stability_label']}`.
- At-transfer two-year spell model, all pre-transfer blocks vs prevalence: Brier improvement {spell['brier_improvement']:+.6f}, 95% player-cluster CI [{spell['cluster_ci_lower_95']:+.6f}, {spell['cluster_ci_upper_95']:+.6f}], weighted-origin candidate AUC {spell['weighted_origin_candidate_roc_auc']:.3f}, `{spell['stability_label']}`.
- First-year involvement added at the 365-day landmark: Brier improvement {landmark['brier_improvement']:+.6f}, 95% player-cluster CI [{landmark['cluster_ci_lower_95']:+.6f}, {landmark['cluster_ci_upper_95']:+.6f}], weighted-origin candidate AUC {landmark['weighted_origin_candidate_roc_auc']:.3f}, `{landmark['stability_label']}`.

Positive Brier improvement means the candidate probabilities are better calibrated and more accurate than the named baseline. See `retention_comparison_results.csv` for every nested feature increment and evaluation window.

## Guardrails

- Only permanent arrivals are modeled; loan and loan-return counts remain visible in the cohort audit.
- All transfer-time predictors are available at or strictly before the move.
- First-year fields appear only in the declared 365-day landmark model.
- The 730-day administrative horizon is complete as of {summary['as_of_date']} for every primary arrival.
- Exact uncensored tenure is not regressed as an ordinary numeric outcome. Fixed-horizon retention avoids treating right-censored spells as known durations.
- Protected existing outputs were unchanged: `{summary['protected_outputs_unchanged']}`.
"""
    (OUTPUT_DIR / "README.md").write_text(text, encoding="utf-8")


def write_manifest() -> None:
    rows = []
    for path in sorted(item for item in OUTPUT_DIR.iterdir() if item.is_file() and item.name != "output_manifest.csv"):
        rows.append(
            {
                "file": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "output_manifest.csv", index=False)


def run() -> None:
    master_before = sha256_file(MASTER_PATH)
    canonical_before = sha256_file(CANONICAL_TRANSFER_PATH)
    protected_before = protected_tree_sha256()

    frame, audit, thresholds, feature_manifest, feature_sets = build_retention_frame()
    predictions, origin_results = fit_models(frame, feature_sets)
    comparisons = summarize_comparisons(predictions)
    survival = build_survival_summary(frame)

    if sha256_file(MASTER_PATH) != master_before:
        raise RuntimeError("Comprehensive master changed during the retention diagnostic.")
    if sha256_file(CANONICAL_TRANSFER_PATH) != canonical_before:
        raise RuntimeError("Canonical transfer history changed during the retention diagnostic.")
    protected_unchanged = protected_tree_sha256() == protected_before
    if not protected_unchanged:
        raise RuntimeError("A protected existing output tree changed during the retention diagnostic.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_columns = [
        "transfer_event_id",
        "canonical_transfer_event_id",
        "player_id",
        "transfer_date",
        "from_club_id",
        "to_club_id",
        "canonical_transfer_type",
        "season_start_year",
        "year2_destination_minutes",
        "year2_destination_club_games",
        "year2_destination_opportunity_share",
        "sporting_retention_y2_10pct",
        "first_outbound_event_id",
        "first_outbound_date",
        "first_outbound_to_club_id",
        "first_outbound_transfer_type",
        "days_to_first_outbound",
        "destination_spell_retained_365d",
        "destination_spell_retained_730d",
    ] + FIRST_YEAR_FEATURES
    frame[target_columns].sort_values(["transfer_date", "transfer_event_id"]).to_csv(
        OUTPUT_DIR / "retention_targets.csv", index=False
    )
    audit.to_csv(OUTPUT_DIR / "retention_cohort_audit.csv", index=False)
    thresholds.to_csv(OUTPUT_DIR / "sporting_threshold_sensitivity.csv", index=False)
    feature_manifest.to_csv(OUTPUT_DIR / "retention_feature_manifest.csv", index=False)
    survival.to_csv(OUTPUT_DIR / "destination_spell_survival_summary.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "retention_predictions.csv", index=False)
    origin_results.to_csv(OUTPUT_DIR / "retention_origin_results.csv", index=False)
    comparisons.to_csv(OUTPUT_DIR / "retention_comparison_results.csv", index=False)

    summary = make_summary(frame, audit, predictions, comparisons, protected_unchanged)
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_readme(summary)
    write_manifest()
    print(f"Wrote {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    run()

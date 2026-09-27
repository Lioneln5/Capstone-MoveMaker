"""Run 2: rolling-origin development modeling for the repaired target.

The run is intentionally narrow: one predeclared linear candidate ladder,
four established development origins, and no 2024+ extension evaluation.
No model binary or deployment artifact is written.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "models") not in sys.path:
    sys.path.insert(0, str(ROOT / "models"))

from models.engine_v2.complete_contribution_target_contract import (  # noqa: E402
    AVAILABLE_FIELD,
    CONTRACT_VERSION as TARGET_CONTRACT_VERSION,
    TARGET_FIELD,
    add_complete_contribution_target,
    add_original_league_schedule_coverage,
)
from models.engine_v2.contribution_model_development_contract import (  # noqa: E402
    BOOTSTRAP_REPETITIONS,
    CANDIDATE_FEATURES,
    CONTRACT_VERSION,
    LOGISTIC_C_GRID,
    MAXIMUM_ABSOLUTE_CALIBRATION_GAP,
    MINIMUM_ORIGIN_WINS,
    MINIMUM_POOLED_AUC,
    OPTIONAL_EXTENSION_CANDIDATE,
    ORIGINS,
    PRIMARY_CANDIDATE,
    PRIMARY_COMPARISONS,
    RANDOM_SEED,
    contract_payload,
)
from models.engine_v2.feature_contract import (  # noqa: E402
    ADVANCED_PERFORMANCE,
    COMMERCIAL_SCENARIO,
    RAW_SCORING_RATES,
)
from mixed_type_preprocessor import fit_preprocessor  # noqa: E402


MASTER = ROOT / "Data" / "processed" / "contract_extension_integration" / "extension_modeling_master.csv"
GAMES = ROOT / "Data" / "processed" / "transfermarkt_clean" / "tables" / "games_clean.csv"
RUN1 = ROOT / "Data" / "processed" / "engine_v2_contribution_target_repair"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"
OUTPUT = ROOT / "Data" / "processed" / "engine_v2_contribution_model_rerun"
DEPLOYMENT = ROOT / "Data" / "processed" / "deployment_models"
PROTOCOLS = ("player_disjoint", "ordinary_temporal_sensitivity")


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


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().str.casefold().isin(
        {"true", "1", "yes"}
    )


def deployment_fingerprint() -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in sorted(DEPLOYMENT.rglob("*"))
        if path.is_file()
    }


def prepare_model_frame() -> pd.DataFrame:
    run1_verification = json.loads(
        (RUN1 / "independent_verification.json").read_text(encoding="utf-8")
    )
    if not run1_verification.get("all_checks_passed"):
        raise RuntimeError("Run 1 has not passed independent verification.")

    frame = pd.read_csv(MASTER, low_memory=False)
    games = pd.read_csv(GAMES, low_memory=False)
    frame["signed_date"] = pd.to_datetime(frame["signed_date"], errors="coerce")
    frame["expiration_date"] = pd.to_datetime(frame["expiration_date"], errors="coerce")
    frame["signed_year"] = frame["signed_date"].dt.year.astype("Int64")
    frame = frame.loc[frame["signed_year"].le(2023)].copy()
    frame = add_original_league_schedule_coverage(frame, games)
    frame = add_complete_contribution_target(frame)

    numeric = lambda column: pd.to_numeric(frame[column], errors="coerce")
    frame["log_market_value_at_signing"] = np.log1p(
        numeric("at_signing_market_value_eur").clip(lower=0)
    )
    frame["age_band"] = pd.cut(
        numeric("age_at_signing"),
        [-np.inf, 21, 25, 29, 33, np.inf],
        labels=["<=21", "22-25", "26-29", "30-33", "34+"],
    )
    frame["contract_covers_y2"] = frame["expiration_date"].ge(
        frame["signed_date"] + pd.Timedelta(days=730)
    )
    pre_share = numeric("pre365_same_club_all_competition_opportunity_share")
    frame["run2_model_cohort"] = (
        frame["canonical_player_id"].notna()
        & frame["contract_covers_y2"]
        & as_bool(frame["pre365_window_evidence_eligible_10_club_games"])
        & pre_share.notna()
        & frame[AVAILABLE_FIELD]
        & pd.to_numeric(frame[TARGET_FIELD], errors="coerce").notna()
    )
    return frame


def adaptive_ece(actual: np.ndarray, prediction: np.ndarray, bins: int = 10) -> float:
    ordered = pd.DataFrame({"actual": actual, "prediction": prediction}).sort_values(
        "prediction"
    )
    groups = np.array_split(np.arange(len(ordered)), min(bins, len(ordered)))
    return float(
        sum(
            len(indices)
            * abs(
                ordered.iloc[indices]["actual"].mean()
                - ordered.iloc[indices]["prediction"].mean()
            )
            for indices in groups
        )
        / len(ordered)
    )


def metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=int)
    prediction = np.clip(np.asarray(prediction, dtype=float), 1e-6, 1 - 1e-6)
    result = {
        "brier": float(brier_score_loss(actual, prediction)),
        "log_loss": float(log_loss(actual, prediction, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(actual, prediction)),
        "average_precision": float(average_precision_score(actual, prediction)),
        "adaptive_ece": adaptive_ece(actual, prediction),
        "actual_rate": float(actual.mean()),
        "predicted_rate": float(prediction.mean()),
        "calibration_gap": float(abs(actual.mean() - prediction.mean())),
    }
    return result


def fit_candidate(
    tuning_train: pd.DataFrame,
    validation: pd.DataFrame,
    final_fit: pd.DataFrame,
    evaluation: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, float | None, object | None, object | None, pd.DataFrame]:
    if not features:
        probability = float(pd.to_numeric(final_fit[TARGET_FIELD]).mean())
        return (
            np.repeat(np.clip(probability, 1e-6, 1 - 1e-6), len(evaluation)),
            None,
            None,
            None,
            pd.DataFrame(),
        )

    tune_pre = fit_preprocessor(tuning_train, features)
    x_train = tune_pre.transform(tuning_train)
    x_validation = tune_pre.transform(validation)
    y_train = pd.to_numeric(tuning_train[TARGET_FIELD]).to_numpy(int)
    y_validation = pd.to_numeric(validation[TARGET_FIELD]).to_numpy(int)
    candidates: list[tuple[float, float, float]] = []
    for c_value in LOGISTIC_C_GRID:
        model = LogisticRegression(
            C=c_value,
            penalty="l2",
            solver="liblinear",
            max_iter=3000,
            random_state=RANDOM_SEED,
        ).fit(x_train, y_train)
        estimate = np.clip(model.predict_proba(x_validation)[:, 1], 1e-6, 1 - 1e-6)
        candidates.append(
            (
                float(brier_score_loss(y_validation, estimate)),
                float(log_loss(y_validation, estimate, labels=[0, 1])),
                c_value,
            )
        )
    _, _, selected_c = min(candidates)

    pre = fit_preprocessor(final_fit, features)
    model = LogisticRegression(
        C=selected_c,
        penalty="l2",
        solver="liblinear",
        max_iter=3000,
        random_state=RANDOM_SEED,
    ).fit(pre.transform(final_fit), pd.to_numeric(final_fit[TARGET_FIELD]).to_numpy(int))
    prediction = np.clip(
        model.predict_proba(pre.transform(evaluation))[:, 1], 1e-6, 1 - 1e-6
    )
    coefficient_frame = pd.DataFrame(
        {
            "encoded_feature": pre.encoded_names,
            "raw_feature": pre.encoded_raw_features,
            "coefficient": model.coef_.reshape(-1),
        }
    )
    return prediction, selected_c, pre, model, coefficient_frame


def run_models(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    cohort_rows: list[dict[str, object]] = []
    coefficient_rows: list[pd.DataFrame] = []
    missingness_rows: list[dict[str, object]] = []

    for protocol in PROTOCOLS:
        for origin, train_end, validation_year, evaluation_year in ORIGINS:
            base_train = data.loc[data["signed_year"].le(train_end)].copy()
            validation = data.loc[data["signed_year"].eq(validation_year)].copy()
            evaluation = data.loc[data["signed_year"].eq(evaluation_year)].sort_values(
                "capology_extension_event_id"
            ).copy()

            tuning_train = base_train
            final_fit = pd.concat([base_train, validation], ignore_index=True)
            removed_for_validation = 0
            removed_for_evaluation = 0
            if protocol == "player_disjoint":
                validation_players = set(validation["canonical_player_id"].dropna())
                tuning_keep = ~base_train["canonical_player_id"].isin(validation_players)
                removed_for_validation = int((~tuning_keep).sum())
                tuning_train = base_train.loc[tuning_keep].copy()

                evaluation_players = set(evaluation["canonical_player_id"].dropna())
                fit_keep = ~final_fit["canonical_player_id"].isin(evaluation_players)
                removed_for_evaluation = int((~fit_keep).sum())
                final_fit = final_fit.loc[fit_keep].copy()

            if min(len(tuning_train), len(validation), len(final_fit), len(evaluation)) < 30:
                raise RuntimeError(f"Insufficient split for {protocol}/{origin}.")
            if pd.to_numeric(tuning_train[TARGET_FIELD]).nunique() != 2:
                raise RuntimeError(f"One-class tuning fold for {protocol}/{origin}.")

            validation_overlap = len(
                set(tuning_train["canonical_player_id"].dropna())
                & set(validation["canonical_player_id"].dropna())
            )
            evaluation_overlap = len(
                set(final_fit["canonical_player_id"].dropna())
                & set(evaluation["canonical_player_id"].dropna())
            )
            cohort_rows.append(
                {
                    "protocol": protocol,
                    "origin_id": origin,
                    "train_end_year": train_end,
                    "validation_year": validation_year,
                    "evaluation_year": evaluation_year,
                    "base_train_rows": len(base_train),
                    "tuning_train_rows": len(tuning_train),
                    "validation_rows": len(validation),
                    "final_fit_rows": len(final_fit),
                    "evaluation_rows": len(evaluation),
                    "evaluation_events": int(pd.to_numeric(evaluation[TARGET_FIELD]).sum()),
                    "evaluation_event_rate": float(pd.to_numeric(evaluation[TARGET_FIELD]).mean()),
                    "rows_removed_for_validation_player_disjointness": removed_for_validation,
                    "rows_removed_for_evaluation_player_disjointness": removed_for_evaluation,
                    "tuning_validation_player_overlap": validation_overlap,
                    "fit_evaluation_player_overlap": evaluation_overlap,
                }
            )

            for feature in sorted({item for values in CANDIDATE_FEATURES.values() for item in values}):
                for frame_name, sample in (
                    ("tuning_train", tuning_train),
                    ("validation", validation),
                    ("final_fit", final_fit),
                    ("evaluation", evaluation),
                ):
                    missingness_rows.append(
                        {
                            "protocol": protocol,
                            "origin_id": origin,
                            "sample": frame_name,
                            "feature": feature,
                            "rows": len(sample),
                            "missing_rows": int(sample[feature].isna().sum()),
                            "missing_rate": float(sample[feature].isna().mean()),
                        }
                    )

            actual = pd.to_numeric(evaluation[TARGET_FIELD]).to_numpy(int)
            for candidate, features in CANDIDATE_FEATURES.items():
                prediction, selected_c, pre, _, coefficients = fit_candidate(
                    tuning_train,
                    validation,
                    final_fit,
                    evaluation,
                    list(features),
                )
                result = metrics(actual, prediction)
                metric_rows.append(
                    {
                        "protocol": protocol,
                        "origin_id": origin,
                        "train_end_year": train_end,
                        "validation_year": validation_year,
                        "evaluation_year": evaluation_year,
                        "candidate": candidate,
                        "raw_feature_count": len(features),
                        "encoded_feature_count": 0 if pre is None else len(pre.encoded_names),
                        "selected_C": selected_c,
                        "train_rows": len(tuning_train),
                        "validation_rows": len(validation),
                        "fit_rows": len(final_fit),
                        "evaluation_rows": len(evaluation),
                        **result,
                    }
                )
                if not coefficients.empty:
                    coefficients.insert(0, "candidate", candidate)
                    coefficients.insert(0, "origin_id", origin)
                    coefficients.insert(0, "protocol", protocol)
                    coefficient_rows.append(coefficients)
                for i, (_, row) in enumerate(evaluation.iterrows()):
                    prediction_rows.append(
                        {
                            "protocol": protocol,
                            "origin_id": origin,
                            "evaluation_year": evaluation_year,
                            "candidate": candidate,
                            "capology_extension_event_id": row["capology_extension_event_id"],
                            "canonical_player_id": row["canonical_player_id"],
                            "league": row["league"],
                            "canonical_position": row["canonical_position"],
                            "age_band": row["age_band"],
                            "actual": int(actual[i]),
                            "prediction": float(prediction[i]),
                            "brier_loss": float((actual[i] - prediction[i]) ** 2),
                        }
                    )

    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(metric_rows),
        pd.DataFrame(cohort_rows),
        pd.concat(coefficient_rows, ignore_index=True),
        pd.DataFrame(missingness_rows),
    )


def clustered_comparisons(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = [
        "protocol",
        "origin_id",
        "evaluation_year",
        "capology_extension_event_id",
        "canonical_player_id",
    ]
    for protocol in PROTOCOLS:
        sample = predictions.loc[predictions["protocol"].eq(protocol)]
        for baseline, candidate in PRIMARY_COMPARISONS:
            left = sample.loc[
                sample["candidate"].eq(baseline), keys + ["brier_loss"]
            ].rename(columns={"brier_loss": "baseline_loss"})
            right = sample.loc[
                sample["candidate"].eq(candidate), keys + ["brier_loss"]
            ].rename(columns={"brier_loss": "candidate_loss"})
            paired = left.merge(right, on=keys, validate="one_to_one")
            paired["improvement"] = paired["baseline_loss"] - paired["candidate_loss"]
            clusters = paired.groupby("canonical_player_id")["improvement"].agg(["sum", "size"])
            sums = clusters["sum"].to_numpy(float)
            sizes = clusters["size"].to_numpy(float)
            rng = np.random.default_rng(stable_seed(protocol, baseline, candidate))
            draws: list[float] = []
            for start in range(0, BOOTSTRAP_REPETITIONS, 250):
                count = min(250, BOOTSTRAP_REPETITIONS - start)
                indices = rng.integers(0, len(clusters), size=(count, len(clusters)))
                draws.extend((sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)).tolist())
            draw_array = np.asarray(draws)
            origin_deltas = paired.groupby("origin_id")["improvement"].mean()
            candidate_predictions = sample.loc[sample["candidate"].eq(candidate)]
            pooled = metrics(
                candidate_predictions["actual"].to_numpy(int),
                candidate_predictions["prediction"].to_numpy(float),
            )
            rows.append(
                {
                    "protocol": protocol,
                    "baseline": baseline,
                    "candidate": candidate,
                    "evaluation_rows": len(paired),
                    "player_clusters": len(clusters),
                    "pooled_brier_improvement": float(paired["improvement"].mean()),
                    "origin_wins": int(origin_deltas.gt(0).sum()),
                    "origin_count": len(origin_deltas),
                    "ci_lower_95": float(np.quantile(draw_array, 0.025)),
                    "ci_upper_95": float(np.quantile(draw_array, 0.975)),
                    "probability_candidate_better": float(np.mean(draw_array > 0)),
                    "candidate_brier": pooled["brier"],
                    "candidate_roc_auc": pooled["roc_auc"],
                    "candidate_average_precision": pooled["average_precision"],
                    "candidate_adaptive_ece": pooled["adaptive_ece"],
                    "candidate_calibration_gap": pooled["calibration_gap"],
                }
            )
    return pd.DataFrame(rows)


def selection_decision(comparisons: pd.DataFrame) -> pd.DataFrame:
    strict = comparisons.loc[comparisons["protocol"].eq("player_disjoint")]
    versus_baseline = strict.loc[
        strict["baseline"].eq("B0_chronology_prevalence")
        & strict["candidate"].eq(PRIMARY_CANDIDATE)
    ].iloc[0]
    versus_profile = strict.loc[
        strict["baseline"].eq("B1_market_profile")
        & strict["candidate"].eq(PRIMARY_CANDIDATE)
    ].iloc[0]
    extension = strict.loc[
        strict["baseline"].eq(PRIMARY_CANDIDATE)
        & strict["candidate"].eq(OPTIONAL_EXTENSION_CANDIDATE)
    ].iloc[0]
    primary_gates = {
        "ci_lower_vs_chronology_above_zero": versus_baseline.ci_lower_95 > 0,
        "origin_wins_vs_chronology_at_least_3": versus_baseline.origin_wins >= MINIMUM_ORIGIN_WINS,
        "improvement_vs_market_profile_positive": versus_profile.pooled_brier_improvement > 0,
        "pooled_auc_at_least_0_70": versus_baseline.candidate_roc_auc >= MINIMUM_POOLED_AUC,
        "calibration_gap_at_most_0_05": versus_baseline.candidate_calibration_gap <= MAXIMUM_ABSOLUTE_CALIBRATION_GAP,
    }
    extension_gates = {
        "ci_lower_vs_primary_above_zero": extension.ci_lower_95 > 0,
        "origin_wins_vs_primary_at_least_3": extension.origin_wins >= MINIMUM_ORIGIN_WINS,
    }
    primary_passes = all(primary_gates.values())
    extension_replaces = primary_passes and all(extension_gates.values())
    selected = OPTIONAL_EXTENSION_CANDIDATE if extension_replaces else PRIMARY_CANDIDATE
    decision = "advance_to_reliability_audit" if primary_passes else "blocked_no_valid_signal"
    return pd.DataFrame(
        [
            {
                "target_contract_version": TARGET_CONTRACT_VERSION,
                "model_contract_version": CONTRACT_VERSION,
                "primary_candidate": PRIMARY_CANDIDATE,
                "optional_extension_candidate": OPTIONAL_EXTENSION_CANDIDATE,
                "selected_development_candidate": selected if primary_passes else "NONE",
                "primary_signal_gate_passed": primary_passes,
                "optional_prior_volume_replaced_primary": extension_replaces,
                **{f"primary_gate__{key}": value for key, value in primary_gates.items()},
                **{f"extension_gate__{key}": value for key, value in extension_gates.items()},
                "run2_decision": decision,
                "deployment_status": "not_deployed",
            }
        ]
    )


def subgroup_metrics(
    predictions: pd.DataFrame, selected_candidate: str
) -> pd.DataFrame:
    selected = predictions.loc[
        predictions["protocol"].eq("player_disjoint")
        & predictions["candidate"].eq(selected_candidate)
    ].copy()
    baseline = predictions.loc[
        predictions["protocol"].eq("player_disjoint")
        & predictions["candidate"].eq("B0_chronology_prevalence"),
        ["origin_id", "capology_extension_event_id", "brier_loss"],
    ].rename(columns={"brier_loss": "baseline_brier_loss"})
    selected = selected.merge(
        baseline,
        on=["origin_id", "capology_extension_event_id"],
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    for group_type, column in (
        ("league", "league"),
        ("position", "canonical_position"),
        ("age", "age_band"),
    ):
        for group, chunk in selected.groupby(column, dropna=False):
            events = int(chunk["actual"].sum())
            nonevents = len(chunk) - events
            sufficient = (
                len(chunk) >= 50
                and events >= 10
                and nonevents >= 10
                and chunk["origin_id"].nunique() >= 3
            )
            auc = (
                float(roc_auc_score(chunk["actual"], chunk["prediction"]))
                if events and nonevents
                else np.nan
            )
            brier = float(chunk["brier_loss"].mean())
            baseline_brier = float(chunk["baseline_brier_loss"].mean())
            gap = float(abs(chunk["actual"].mean() - chunk["prediction"].mean()))
            supported = sufficient and auc >= 0.55 and gap <= 0.10 and brier < baseline_brier
            status = "supported" if supported else "limited_reliability" if sufficient else "insufficient_evidence"
            rows.append(
                {
                    "candidate": selected_candidate,
                    "group_type": group_type,
                    "group": str(group),
                    "rows": len(chunk),
                    "events": events,
                    "nonevents": nonevents,
                    "origins": chunk["origin_id"].nunique(),
                    "actual_rate": float(chunk["actual"].mean()),
                    "predicted_rate": float(chunk["prediction"].mean()),
                    "calibration_gap": gap,
                    "brier": brier,
                    "chronology_baseline_brier": baseline_brier,
                    "brier_skill": baseline_brier - brier,
                    "roc_auc": auc,
                    "evidence_status": status,
                }
            )
    return pd.DataFrame(rows)


def protocol_sensitivity(origin_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for protocol in PROTOCOLS:
        for candidate in CANDIDATE_FEATURES:
            chunk = origin_metrics.loc[
                origin_metrics["protocol"].eq(protocol)
                & origin_metrics["candidate"].eq(candidate)
            ]
            rows.append(
                {
                    "protocol": protocol,
                    "candidate": candidate,
                    "evaluation_rows": int(chunk["evaluation_rows"].sum()),
                    "mean_origin_brier": float(
                        np.average(chunk["brier"], weights=chunk["evaluation_rows"])
                    ),
                    "mean_origin_auc": float(
                        np.average(chunk["roc_auc"], weights=chunk["evaluation_rows"])
                    ),
                    "mean_origin_calibration_gap": float(
                        np.average(chunk["calibration_gap"], weights=chunk["evaluation_rows"])
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for stale in ("independent_verification.csv", "independent_verification.json"):
        (OUTPUT / stale).unlink(missing_ok=True)

    deployment_before = deployment_fingerprint()
    data = prepare_model_frame()
    model_data = data.loc[data["run2_model_cohort"]].copy()
    predictions, origin_metrics, cohorts, coefficients, missingness = run_models(model_data)
    comparisons = clustered_comparisons(predictions)
    decision = selection_decision(comparisons)
    selected = decision.iloc[0]["selected_development_candidate"]
    subgroup_candidate = selected if selected != "NONE" else PRIMARY_CANDIDATE
    subgroups = subgroup_metrics(predictions, subgroup_candidate)
    protocol_table = protocol_sensitivity(origin_metrics)
    deployment_after = deployment_fingerprint()

    checks: list[dict[str, object]] = []

    def add(check: str, actual: object, expected: object, passed: bool, note: str) -> None:
        checks.append(
            {"check": check, "actual": actual, "expected": expected, "passed": bool(passed), "note": note}
        )

    strict_cohorts = cohorts.loc[cohorts["protocol"].eq("player_disjoint")]
    strict_predictions = predictions.loc[predictions["protocol"].eq("player_disjoint")]
    prohibited = set(RAW_SCORING_RATES + ADVANCED_PERFORMANCE + COMMERCIAL_SCENARIO)
    candidate_union = {feature for values in CANDIDATE_FEATURES.values() for feature in values}
    add("target_contract_exact", TARGET_CONTRACT_VERSION, "engine_v2_sustained_realized_contribution_v2_2026-09-07", TARGET_CONTRACT_VERSION == "engine_v2_sustained_realized_contribution_v2_2026-09-07", "Run 2 cannot silently change the repaired target.")
    add("four_rolling_origins", strict_cohorts["origin_id"].nunique(), 4, strict_cohorts["origin_id"].nunique() == 4, "Established 2020-2023 development origins only.")
    add("evaluation_rows_exact", int(strict_cohorts["evaluation_rows"].sum()), 878, int(strict_cohorts["evaluation_rows"].sum()) == 878, "Run 2 uses the complete Run 1 reference cohort.")
    add("latest_evaluation_year", int(strict_cohorts["evaluation_year"].max()), 2023, strict_cohorts["evaluation_year"].max() == 2023, "No 2024+ extension outcome is evaluated.")
    add("player_disjoint_validation", int(strict_cohorts["tuning_validation_player_overlap"].max()), 0, strict_cohorts["tuning_validation_player_overlap"].eq(0).all(), "Validation players are absent from tuning training rows.")
    add("player_disjoint_evaluation", int(strict_cohorts["fit_evaluation_player_overlap"].max()), 0, strict_cohorts["fit_evaluation_player_overlap"].eq(0).all(), "Evaluation players are absent from all fitting rows.")
    add("four_predeclared_candidates", strict_predictions["candidate"].nunique(), 4, strict_predictions["candidate"].nunique() == 4, "No candidate was added after results.")
    add("prediction_keys_unique", int(strict_predictions.duplicated(["origin_id", "candidate", "capology_extension_event_id"]).sum()), 0, not strict_predictions.duplicated(["origin_id", "candidate", "capology_extension_event_id"]).any(), "One prediction per event, origin, and candidate.")
    add("probabilities_bounded", int(predictions["prediction"].between(0, 1, inclusive="both").all()), 1, predictions["prediction"].between(0, 1, inclusive="both").all(), "All probabilities lie in [0,1].")
    reconstructed_loss = np.square(predictions["actual"] - predictions["prediction"])
    add("brier_losses_exact", float(np.max(np.abs(reconstructed_loss - predictions["brier_loss"]))), "<=1e-14", np.allclose(reconstructed_loss, predictions["brier_loss"], atol=1e-14), "Row-level loss reconstructs exactly.")
    add("prohibited_features_absent", sorted(prohibited & candidate_union), [], not (prohibited & candidate_union), "No scoring, advanced-event, salary, or proposed-term field enters Run 2.")
    add("comparison_rows_matched", sorted(comparisons["evaluation_rows"].unique().tolist()), [878], comparisons["evaluation_rows"].eq(878).all(), "Every paired comparison uses identical evaluation events.")
    add("deployment_unchanged", int(deployment_before == deployment_after), 1, deployment_before == deployment_after, "All current deployment files retain their pre-run hashes.")
    add("no_model_binary_written", len(list(OUTPUT.rglob("*.joblib"))), 0, not list(OUTPUT.rglob("*.joblib")), "Run 2 cannot serialize a candidate model.")
    add("decision_research_only", decision.iloc[0]["deployment_status"], "not_deployed", decision.iloc[0]["deployment_status"] == "not_deployed", "Even a passing development signal only advances to reliability audit.")
    checks_frame = pd.DataFrame(checks)

    write_csv(predictions, OUTPUT / "out_of_time_predictions.csv")
    write_csv(origin_metrics, OUTPUT / "origin_metrics.csv")
    write_csv(cohorts, OUTPUT / "cohort_by_origin.csv")
    write_csv(comparisons, OUTPUT / "candidate_comparisons.csv")
    write_csv(decision, OUTPUT / "selection_decision.csv")
    write_csv(subgroups, OUTPUT / "subgroup_metrics.csv")
    write_csv(protocol_table, OUTPUT / "player_disjoint_sensitivity.csv")
    write_csv(coefficients, OUTPUT / "coefficient_stability.csv")
    write_csv(missingness, OUTPUT / "feature_missingness_by_origin.csv")
    write_csv(checks_frame, OUTPUT / "build_checks.csv")
    write_json(contract_payload(), OUTPUT / "preregistration.json")

    deployment_manifest = pd.DataFrame(
        {
            "artifact": path,
            "before_sha256": digest,
            "after_sha256": deployment_after.get(path, "MISSING"),
            "unchanged": deployment_after.get(path) == digest,
        }
        for path, digest in deployment_before.items()
    )
    write_csv(deployment_manifest, OUTPUT / "deployment_unchanged_manifest.csv")

    source_paths = [
        MASTER,
        GAMES,
        RUN1 / "target_contract.json",
        RUN1 / "independent_verification.json",
        ROOT / "models" / "engine_v2" / "complete_contribution_target_contract.py",
        ROOT / "models" / "engine_v2" / "contribution_model_development_contract.py",
        ROOT / "models" / "engine_v2" / "feature_contract.py",
        ROOT / "models" / "mixed_type_preprocessor.py",
        ROOT / "models" / "run_engine_v2_contribution_model_rerun.py",
        ROOT / "scripts" / "verify_engine_v2_contribution_model_rerun.py",
        FREEZE,
    ]
    source_manifest = pd.DataFrame(
        {
            "source": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in source_paths
    )
    write_csv(source_manifest, OUTPUT / "source_manifest.csv")

    summary = {
        "stage": "engine_v2_repaired_contribution_model_run2",
        "target_contract_version": TARGET_CONTRACT_VERSION,
        "model_contract_version": CONTRACT_VERSION,
        "development_evaluation_years": [2020, 2021, 2022, 2023],
        "evaluation_rows": int(strict_cohorts["evaluation_rows"].sum()),
        "player_disjoint_primary": True,
        "selected_development_candidate": selected,
        "run2_decision": decision.iloc[0]["run2_decision"],
        "models_fitted_in_memory": True,
        "model_binaries_written": 0,
        "final_holdout_evaluated": False,
        "deployment_changed": False,
        "checks_passed": int(checks_frame["passed"].sum()),
        "checks_total": len(checks_frame),
        "all_checks_passed": bool(checks_frame["passed"].all()),
    }
    write_json(summary, OUTPUT / "run_summary.json")

    strict_comparisons = comparisons.loc[
        comparisons["protocol"].eq("player_disjoint")
    ].set_index(["baseline", "candidate"])
    primary = strict_comparisons.loc[("B0_chronology_prevalence", PRIMARY_CANDIDATE)]
    profile = strict_comparisons.loc[("B1_market_profile", PRIMARY_CANDIDATE)]
    extension = strict_comparisons.loc[(PRIMARY_CANDIDATE, OPTIONAL_EXTENSION_CANDIDATE)]
    supported_groups = int(subgroups["evidence_status"].eq("supported").sum())
    total_groups = len(subgroups)
    readme = f"""# Engine V2 repaired contribution model — Run 2

Run 2 asks whether the repaired **sustained realized extension-club
contribution** target has a repeatable pre-extension signal. It does not ask
whether a club should extend a player and it does not measure tactical role
when available.

## Design

- Four expanding rolling origins evaluate extensions signed in 2020-2023.
- Evaluation players are removed from all model-fitting rows. Validation
  players are also removed from the training rows used to choose regularization.
- Brier score is the primary loss. Logistic `C` is chosen only on the preceding
  validation year.
- The fixed candidate ladder contains chronology prevalence, basic market
  profile, recent involvement, and optional prior-volume history.
- Goals/assists, advanced-event statistics, wages, proposed contract terms,
  manager/tactical fields, and post-extension predictors are excluded.

## Result

The predeclared primary candidate `{PRIMARY_CANDIDATE}` has player-disjoint
pooled Brier improvement {primary.pooled_brier_improvement:+.6f} versus the
chronology baseline, with a 95% player-cluster interval
[{primary.ci_lower_95:+.6f}, {primary.ci_upper_95:+.6f}] and
{int(primary.origin_wins)}/4 origin wins. Its pooled ROC AUC is
{primary.candidate_roc_auc:.3f}, average precision
{primary.candidate_average_precision:.3f}, adaptive ECE
{primary.candidate_adaptive_ece:.3f}, and calibration gap
{primary.candidate_calibration_gap:.3f}.

Recent involvement adds {profile.pooled_brier_improvement:+.6f} Brier skill
over market profile alone. Adding prior-volume history changes Brier skill by
{extension.pooled_brier_improvement:+.6f} versus the primary candidate, with
a 95% interval [{extension.ci_lower_95:+.6f}, {extension.ci_upper_95:+.6f}].

The frozen decision is `{decision.iloc[0]['run2_decision']}` and the selected
development candidate is `{selected}`. This means only that it may enter the
next calibration/OOD/subgroup reliability audit. It is not a deployable model.

The preliminary subgroup table marks {supported_groups}/{total_groups} views
as supported under the existing minimum-size, discrimination, calibration,
and Brier-skill rules. Those results are diagnostic, not a completed release
gate.

## Boundary

Run 2 fits models only in memory and writes no `.joblib` file. The 2024+
extension cohort remains closed, and every file under the current deployment
directory retains its exact pre-run hash. Role/squad, manager/tactical, club
behavior, nonlinear families, and calibration methods were not searched here;
they require separately frozen follow-up tests if the compact signal survives.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")

    output_files = sorted(
        path
        for path in OUTPUT.iterdir()
        if path.is_file()
        and path.name not in {"output_manifest.csv", "independent_verification.csv", "independent_verification.json"}
    )
    write_csv(
        pd.DataFrame(
            {
                "output_file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in output_files
        ),
        OUTPUT / "output_manifest.csv",
    )

    if not checks_frame["passed"].all():
        failed = checks_frame.loc[
            ~checks_frame["passed"], ["check", "actual", "expected"]
        ]
        raise RuntimeError("Run 2 build checks failed:\n" + failed.to_string(index=False))
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nPlayer-disjoint comparisons\n", comparisons.loc[comparisons["protocol"].eq("player_disjoint")].to_string(index=False))


if __name__ == "__main__":
    main()

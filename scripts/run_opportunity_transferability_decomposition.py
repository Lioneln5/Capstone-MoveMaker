"""Run the frozen opportunity-transferability and context decomposition."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_coarse_club_context_diagnostic as coarse
from train_compatibility_models import (
    ALPHA_GRID,
    apply_bounds,
    fit_preprocessor,
    fit_ridge,
    metrics,
    predict_ridge,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = coarse.MASTER_PATH
SUMMARY_PATH = coarse.SUMMARY_PATH
COARSE_OUTPUT_DIR = coarse.OUTPUT_DIR
OUTPUT_DIR = ROOT / "Data" / "processed" / "opportunity_transferability_decomposition"
TARGET = "destination_opportunity_share_post24"
RANDOM_SEED = 20260810
BOOTSTRAP_REPETITIONS = 5000

MODEL_ORDER = [
    "M0_mean_only",
    "M1_market_profile",
    "M2_plus_prior_opportunity",
    "M3_plus_player_performance",
    "M4_plus_destination_context",
    "M5_plus_origin_transition_context",
]
EXPECTED_COUNTS = dict(zip(MODEL_ORDER, [0, 5, 13, 29, 73, 133]))
ORIGINS = coarse.ORIGINS
WINDOWS = coarse.WINDOWS
COMPARISONS = [
    ("M1_vs_M0_market_profile", MODEL_ORDER[0], MODEL_ORDER[1]),
    ("M2_vs_M1_prior_opportunity", MODEL_ORDER[1], MODEL_ORDER[2]),
    ("M3_vs_M2_player_performance", MODEL_ORDER[2], MODEL_ORDER[3]),
    ("M4_vs_M3_destination_context", MODEL_ORDER[3], MODEL_ORDER[4]),
    ("M5_vs_M4_origin_transition_context", MODEL_ORDER[4], MODEL_ORDER[5]),
    ("M5_vs_M0_total", MODEL_ORDER[0], MODEL_ORDER[5]),
]
PROTECTED_DIRS = coarse.PROTECTED_DIRS + [COARSE_OUTPUT_DIR]

USAGE_SPECS = [
    (
        "origin_opportunity_share_pre365",
        "appearance_pre365_origin_minutes",
        "club_origin_pre365_games",
        90.0,
    ),
    (
        "origin_opportunity_share_pre730",
        "appearance_pre730_origin_minutes",
        "club_origin_pre730_games",
        90.0,
    ),
    (
        "origin_appearance_rate_pre365",
        "appearance_pre365_origin_games",
        "club_origin_pre365_games",
        1.0,
    ),
    (
        "origin_appearance_rate_pre730",
        "appearance_pre730_origin_games",
        "club_origin_pre730_games",
        1.0,
    ),
    (
        "origin_start_rate_pre365",
        "lineup_pre365_origin_starts",
        "club_origin_pre365_games",
        1.0,
    ),
    (
        "origin_start_rate_pre730",
        "lineup_pre730_origin_starts",
        "club_origin_pre730_games",
        1.0,
    ),
    (
        "origin_substitute_selection_rate_pre365",
        "lineup_pre365_origin_substitute_selections",
        "club_origin_pre365_games",
        1.0,
    ),
    (
        "origin_substitute_selection_rate_pre730",
        "lineup_pre730_origin_substitute_selections",
        "club_origin_pre730_games",
        1.0,
    ),
]
PERFORMANCE_CONTEXTS = ["pre365_all", "pre365_origin", "pre730_all", "pre730_origin"]


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


def cohort_from_master(master: pd.DataFrame) -> pd.DataFrame:
    cohort = coarse.cohort_from_master(master)
    cohort[TARGET] = pd.to_numeric(cohort[coarse.TARGET], errors="coerce")
    return cohort


def audit_cohort(cohort: pd.DataFrame) -> None:
    coarse.assert_preflight(cohort)
    target = pd.to_numeric(cohort[TARGET], errors="coerce")
    if target.isna().any() or not target.between(0, 1.05).all():
        raise RuntimeError("Destination opportunity target is incomplete or out of bounds.")

    coarse_predictions = pd.read_csv(
        COARSE_OUTPUT_DIR / "coarse_club_context_predictions.csv", low_memory=False
    )
    coarse_actual = (
        coarse_predictions.loc[
            coarse_predictions["model_variant"].eq(coarse.MODEL_ORDER[0]),
            ["transfer_event_id", "actual"],
        ]
        .drop_duplicates("transfer_event_id")
        .copy()
    )
    matched = cohort[["transfer_event_id", TARGET]].merge(
        coarse_actual, on="transfer_event_id", how="inner", validate="one_to_one"
    )
    if len(matched) != coarse_actual["transfer_event_id"].nunique() or not np.allclose(
        matched[TARGET], matched["actual"], atol=1e-12, rtol=0
    ):
        raise RuntimeError("Destination target disagrees with the verified coarse diagnostic.")


def build_model_frame(
    cohort: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    frame = cohort.copy()
    definitions: dict[str, dict[str, str]] = {}

    transfer_date = pd.to_datetime(frame["transfer_date"], errors="coerce")
    birth_date = pd.to_datetime(frame["player_date_of_birth"], errors="coerce")
    age = (transfer_date - birth_date).dt.days / 365.2425
    frame["age_at_transfer"] = age.where(age.between(15, 45))
    definitions["age_at_transfer"] = {
        "feature_kind": "derived",
        "source_fields": "player_date_of_birth;transfer_date",
        "timing_classification": "derived at transfer",
        "rationale": "Age known on the transfer date; implausible ages become missing.",
    }
    historical_position = frame["lineup_pre365_all_primary_position"].where(
        frame["lineup_pre365_all_primary_position"].notna(),
        frame["lineup_pre365_origin_primary_position"],
    )
    frame["historical_position_group"] = historical_position.map(coarse.position_group)
    definitions["historical_position_group"] = {
        "feature_kind": "derived_categorical",
        "source_fields": "lineup_pre365_all_primary_position;lineup_pre365_origin_primary_position",
        "timing_classification": "derived strictly pre-transfer player history",
        "rationale": "Frozen broad role from historically observed lineup position.",
    }
    for feature in ["transfer_fee", "transfer_fee_status", "market_value_in_eur"]:
        definitions[feature] = {
            "feature_kind": "raw",
            "source_fields": feature,
            "timing_classification": "available at transfer",
            "rationale": "Frozen market/profile control.",
        }

    m0: list[str] = []
    m1 = [
        "age_at_transfer",
        "historical_position_group",
        "transfer_fee",
        "transfer_fee_status",
        "market_value_in_eur",
    ]

    usage_features: list[str] = []
    for feature, numerator, denominator, minutes_scale in USAGE_SPECS:
        scaled_denominator = pd.to_numeric(frame[denominator], errors="coerce") * minutes_scale
        frame[feature] = coarse.safe_divide(frame[numerator], scaled_denominator)
        definitions[feature] = {
            "feature_kind": "derived",
            "source_fields": f"{numerator};{denominator}",
            "timing_classification": "derived strictly pre-transfer opportunity",
            "rationale": "Frozen normalized origin-club usage measure.",
        }
        usage_features.append(feature)
    m2 = m1 + usage_features

    performance_features: list[str] = []
    for context in PERFORMANCE_CONTEXTS:
        goals_feature = f"appearance_{context}_goals_per90"
        assists_feature = f"appearance_{context}_assists_per90"
        for feature, label in [(goals_feature, "goals"), (assists_feature, "assists")]:
            definitions[feature] = {
                "feature_kind": "raw",
                "source_fields": feature,
                "timing_classification": "strictly pre-transfer player performance",
                "rationale": f"Frozen {label}-per-90 performance rate.",
            }
            performance_features.append(feature)
        minutes = f"appearance_{context}_minutes"
        for card in ["yellow_cards", "red_cards"]:
            source = f"appearance_{context}_{card}"
            feature = f"appearance_{context}_{card}_per90"
            frame[feature] = 90.0 * coarse.safe_divide(frame[source], frame[minutes])
            definitions[feature] = {
                "feature_kind": "derived",
                "source_fields": f"{source};{minutes}",
                "timing_classification": "derived strictly pre-transfer player performance",
                "rationale": f"Frozen {card.replace('_', ' ')}-per-90 performance rate.",
            }
            performance_features.append(feature)
    m3 = m2 + performance_features

    destination_context = coarse.add_context_features(frame, "destination", definitions)
    m4 = m3 + destination_context
    origin_context = coarse.add_context_features(frame, "origin", definitions)
    transition_features: list[str] = []
    for horizon in [365, 730]:
        for metric_name in coarse.TRANSITION_METRICS:
            destination_feature = f"club_destination_pre{horizon}_{metric_name}"
            origin_feature = f"club_origin_pre{horizon}_{metric_name}"
            feature = f"transition_pre{horizon}_{metric_name}_delta"
            frame[feature] = pd.to_numeric(frame[destination_feature], errors="coerce") - pd.to_numeric(
                frame[origin_feature], errors="coerce"
            )
            definitions[feature] = {
                "feature_kind": "derived",
                "source_fields": (
                    f"{definitions[destination_feature]['source_fields']};"
                    f"{definitions[origin_feature]['source_fields']}"
                ),
                "timing_classification": "derived strictly pre-transfer transition context",
                "rationale": "Frozen destination-minus-origin transition contrast.",
            }
            transition_features.append(feature)
    m5 = m4 + origin_context + transition_features

    feature_sets = dict(zip(MODEL_ORDER, [m0, m1, m2, m3, m4, m5]))
    observed_counts = {name: len(features) for name, features in feature_sets.items()}
    if observed_counts != EXPECTED_COUNTS:
        raise RuntimeError(f"Feature count mismatch: observed={observed_counts}; expected={EXPECTED_COUNTS}")
    for index in range(1, len(MODEL_ORDER)):
        if not set(feature_sets[MODEL_ORDER[index - 1]]).issubset(feature_sets[MODEL_ORDER[index]]):
            raise RuntimeError("Feature sets are not nested.")
    for name, features in feature_sets.items():
        if len(features) != len(set(features)):
            raise RuntimeError(f"Duplicate feature in {name}.")
        missing = sorted(set(features) - set(frame.columns))
        if missing:
            raise RuntimeError(f"Missing features in {name}: {missing}")

    manifest_rows: list[dict[str, Any]] = [
        {
            "feature_set": MODEL_ORDER[0],
            "feature_order": 0,
            "feature": "",
            "feature_kind": "no_predictors",
            "source_fields": "",
            "timing_classification": "not applicable",
            "rationale": "Mean-only benchmark with no predictors.",
            "raw_feature_count": 0,
        }
    ]
    for feature_set in MODEL_ORDER[1:]:
        features = feature_sets[feature_set]
        for order, feature in enumerate(features, start=1):
            manifest_rows.append(
                {
                    "feature_set": feature_set,
                    "feature_order": order,
                    "feature": feature,
                    **definitions[feature],
                    "raw_feature_count": len(features),
                }
            )
    return frame, feature_sets, pd.DataFrame(manifest_rows)


def audit_usage_measures(frame: pd.DataFrame) -> None:
    for feature, _, denominator, _ in USAGE_SPECS:
        if not pd.to_numeric(frame[denominator], errors="coerce").gt(0).all():
            raise RuntimeError(f"Nonpositive usage denominator: {denominator}")
        values = pd.to_numeric(frame[feature], errors="coerce")
        if values.isna().any() or not values.between(0, 1.05).all():
            raise RuntimeError(f"Usage measure is incomplete or out of bounds: {feature}")
    expectations = {
        "origin_opportunity_share_pre365": (367, 1.0277777777777777),
        "origin_opportunity_share_pre730": (192, 1.0044444444444445),
    }
    for feature, (zero_count, maximum) in expectations.items():
        values = pd.to_numeric(frame[feature], errors="coerce")
        if int(values.eq(0).sum()) != zero_count or not np.isclose(values.max(), maximum, atol=1e-12, rtol=0):
            raise RuntimeError(f"Usage expectation mismatch for {feature}.")


def tune_ridge(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
) -> tuple[float, dict[str, float]]:
    preprocessor = fit_preprocessor(train, features)
    x_train = preprocessor.transform(train)
    x_validation = preprocessor.transform(validation)
    y_train = train[TARGET].to_numpy(dtype=float)
    y_validation = validation[TARGET].to_numpy(dtype=float)
    candidates: list[tuple[tuple[float, float, str], float, dict[str, float]]] = []
    for alpha in ALPHA_GRID:
        intercept, beta = fit_ridge(x_train, y_train, alpha)
        prediction = apply_bounds(predict_ridge(x_validation, intercept, beta), 0.0, 1.05)
        result = metrics(y_validation, prediction)
        key = (float(result["mae"]), float(result["rmse"]), json.dumps({"alpha": alpha}, sort_keys=True))
        candidates.append((key, float(alpha), result))
    _, selected_alpha, selected_metrics = min(candidates, key=lambda item: item[0])
    return selected_alpha, selected_metrics


def fit_and_score(
    fit_frame: pd.DataFrame,
    evaluation: pd.DataFrame,
    features: list[str],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    preprocessor = fit_preprocessor(fit_frame, features)
    x_fit = preprocessor.transform(fit_frame)
    x_evaluation = preprocessor.transform(evaluation)
    intercept, beta = fit_ridge(x_fit, fit_frame[TARGET].to_numpy(dtype=float), alpha)
    raw = predict_ridge(x_evaluation, intercept, beta)
    bounded = apply_bounds(raw, 0.0, 1.05)
    return raw, bounded, x_fit.shape[1]


def paired_frame(
    predictions: pd.DataFrame,
    baseline: str,
    candidate: str,
    evaluation_years: list[int],
) -> pd.DataFrame:
    selected = predictions.loc[predictions["evaluation_year"].isin(evaluation_years)].copy()
    keys = ["origin_id", "evaluation_year", "transfer_event_id", "player_id"]
    left = selected.loc[selected["model_variant"].eq(baseline), keys + ["actual", "bounded_prediction"]].rename(
        columns={"bounded_prediction": "baseline_prediction"}
    )
    right = selected.loc[selected["model_variant"].eq(candidate), keys + ["actual", "bounded_prediction"]].rename(
        columns={"actual": "candidate_actual", "bounded_prediction": "candidate_prediction"}
    )
    pairs = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if not np.allclose(pairs["actual"], pairs["candidate_actual"], atol=1e-12, rtol=0):
        raise RuntimeError("Actual outcomes disagree in a paired comparison.")
    return pairs.drop(columns="candidate_actual")


def player_cluster_bootstrap(pairs: pd.DataFrame, seed: int) -> dict[str, float | int]:
    difference = np.abs(pairs["actual"] - pairs["baseline_prediction"]) - np.abs(
        pairs["actual"] - pairs["candidate_prediction"]
    )
    working = pd.DataFrame({"player_id": pairs["player_id"], "difference": difference})
    clusters = working.groupby("player_id", sort=True)["difference"].agg(["sum", "size"])
    sums = clusters["sum"].to_numpy(dtype=float)
    sizes = clusters["size"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(clusters), size=(BOOTSTRAP_REPETITIONS, len(clusters)))
    draws = sums[draw].sum(axis=1) / sizes[draw].sum(axis=1)
    return {
        "cluster_count": len(clusters),
        "cluster_bootstrap_ci_lower_95": float(np.quantile(draws, 0.025)),
        "cluster_bootstrap_ci_upper_95": float(np.quantile(draws, 0.975)),
        "cluster_bootstrap_probability_candidate_better": float(np.mean(draws > 0)),
    }


def row_bootstrap(pairs: pd.DataFrame, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    total = np.zeros(BOOTSTRAP_REPETITIONS, dtype=float)
    for _, group in pairs.groupby("origin_id", sort=True):
        difference = (
            np.abs(group["actual"].to_numpy(dtype=float) - group["baseline_prediction"].to_numpy(dtype=float))
            - np.abs(group["actual"].to_numpy(dtype=float) - group["candidate_prediction"].to_numpy(dtype=float))
        )
        draw = rng.integers(0, len(group), size=(BOOTSTRAP_REPETITIONS, len(group)))
        total += difference[draw].sum(axis=1)
    draws = total / len(pairs)
    return {
        "row_bootstrap_ci_lower_95": float(np.quantile(draws, 0.025)),
        "row_bootstrap_ci_upper_95": float(np.quantile(draws, 0.975)),
        "row_bootstrap_probability_candidate_better": float(np.mean(draws > 0)),
    }


def stability_label(improvement: float, origin_values: np.ndarray, probability: float) -> str:
    wins = int(np.sum(origin_values > 0))
    count = len(origin_values)
    if improvement > 0 and wins >= max(3, count - 1) and probability >= 0.90:
        return "stable_positive"
    if improvement > 0 and wins >= math.ceil(count / 2):
        return "mixed_positive"
    if improvement <= 0 and wins <= 1:
        return "unstable_negative"
    return "mixed"


def summarize_comparison(
    predictions: pd.DataFrame,
    comparison: str,
    baseline: str,
    candidate: str,
    window: str,
    years: list[int],
) -> dict[str, Any]:
    pairs = paired_frame(predictions, baseline, candidate, years)
    baseline_error = np.abs(pairs["actual"] - pairs["baseline_prediction"])
    candidate_error = np.abs(pairs["actual"] - pairs["candidate_prediction"])
    baseline_mae = float(baseline_error.mean())
    candidate_mae = float(candidate_error.mean())
    improvement = baseline_mae - candidate_mae
    origin_values = []
    for _, group in pairs.groupby("origin_id", sort=True):
        origin_values.append(
            float(
                np.abs(group["actual"] - group["baseline_prediction"]).mean()
                - np.abs(group["actual"] - group["candidate_prediction"]).mean()
            )
        )
    origin_array = np.asarray(origin_values, dtype=float)
    cluster_seed = stable_seed(comparison, window, "player_cluster")
    row_seed = stable_seed(comparison, window, "row")
    cluster = player_cluster_bootstrap(pairs, cluster_seed)
    row = row_bootstrap(pairs, row_seed)
    return {
        "comparison": comparison,
        "baseline_model": baseline,
        "candidate_model": candidate,
        "window": window,
        "evaluation_years": ";".join(str(year) for year in years),
        "pooled_n": len(pairs),
        "unique_players": pairs["player_id"].nunique(),
        "origin_count": len(origin_array),
        "pooled_baseline_mae": baseline_mae,
        "pooled_candidate_mae": candidate_mae,
        "pooled_mae_improvement": improvement,
        "pooled_relative_mae_improvement": improvement / baseline_mae,
        **cluster,
        **row,
        "origin_win_rate": float(np.mean(origin_array > 0)),
        "origins_candidate_better": int(np.sum(origin_array > 0)),
        "origin_improvement_std": float(np.std(origin_array, ddof=1)) if len(origin_array) > 1 else 0.0,
        "worst_origin_improvement": float(origin_array.min()),
        "best_origin_improvement": float(origin_array.max()),
        "stability_label": stability_label(
            improvement,
            origin_array,
            float(cluster["cluster_bootstrap_probability_candidate_better"]),
        ),
        "cluster_bootstrap_seed": cluster_seed,
        "row_bootstrap_seed": row_seed,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
    }


def correlation(x: pd.Series, y: pd.Series, method: str) -> float:
    return float(pd.to_numeric(x, errors="coerce").corr(pd.to_numeric(y, errors="coerce"), method=method))


def persistence_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = predictions.loc[predictions["model_variant"].eq(MODEL_ORDER[0])].copy()
    specifications: list[tuple[str, list[int]]] = list(WINDOWS.items())
    specifications += [(origin_id, [evaluation_year]) for origin_id, _, _, evaluation_year in ORIGINS]
    output = []
    for window, years in specifications:
        selected = rows.loc[rows["evaluation_year"].isin(years)].copy()
        origin_share = selected["origin_opportunity_share_pre365"].clip(0, 1.05)
        actual = selected["actual"]
        output.append(
            {
                "window": window,
                "evaluation_years": ";".join(str(year) for year in years),
                "rows": len(selected),
                "unique_players": selected["player_id"].nunique(),
                "pearson_correlation": correlation(origin_share, actual, "pearson"),
                "spearman_correlation": correlation(origin_share, actual, "spearman"),
                "naive_origin_share_mae": float(np.mean(np.abs(actual - origin_share))),
                "mean_origin_opportunity_share": float(origin_share.mean()),
                "mean_destination_opportunity_share": float(actual.mean()),
            }
        )
    return pd.DataFrame(output)


def scope_angle(results: pd.DataFrame) -> tuple[str, str]:
    def row(comparison: str, window: str) -> pd.Series:
        return results.loc[results["comparison"].eq(comparison) & results["window"].eq(window)].iloc[0]

    persistence = row("M2_vs_M1_prior_opportunity", "pooled_four")
    persistence_mature = row("M2_vs_M1_prior_opportunity", "mature_three")
    destination = row("M4_vs_M3_destination_context", "pooled_four")
    transition = row("M5_vs_M4_origin_transition_context", "pooled_four")
    persistence_supported = (
        persistence["pooled_mae_improvement"] > 0
        and persistence["cluster_bootstrap_ci_lower_95"] > 0
        and persistence_mature["pooled_mae_improvement"] > 0
    )
    context_supported = any(
        candidate["pooled_mae_improvement"] > 0
        and candidate["cluster_bootstrap_ci_lower_95"] > 0
        for candidate in [destination, transition]
    )
    if persistence_supported and context_supported:
        return "both", "The evidence supports opportunity persistence plus a smaller contextual adjustment."
    if persistence_supported:
        return "opportunity persistence", "The evidence supports opportunity persistence but not a stable contextual adjustment."
    if context_supported:
        return "contextual adjustment", "The evidence supports contextual adjustment but not stable opportunity persistence beyond market profile."
    return "neither", "Neither opportunity persistence nor contextual adjustment adds stable incremental value under this design."


def write_readme(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    results: pd.DataFrame,
    persistence: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    master_hash: str,
    protected_unchanged: bool,
) -> None:
    def result(comparison: str, window: str = "pooled_four") -> pd.Series:
        return results.loc[results["comparison"].eq(comparison) & results["window"].eq(window)].iloc[0]

    angle, statement = scope_angle(results)
    full_pearson_365 = correlation(frame["origin_opportunity_share_pre365"], frame[TARGET], "pearson")
    full_spearman_365 = correlation(frame["origin_opportunity_share_pre365"], frame[TARGET], "spearman")
    full_pearson_730 = correlation(frame["origin_opportunity_share_pre730"], frame[TARGET], "pearson")
    full_spearman_730 = correlation(frame["origin_opportunity_share_pre730"], frame[TARGET], "spearman")
    lines = [
        "# Opportunity transferability and context decomposition",
        "",
        "## Cohort and pre-model audit",
        "",
        f"The frozen cohort contains {len(frame):,} transfers and {frame['player_id'].nunique():,} players. The pooled evaluation contains {predictions.loc[predictions['model_variant'].eq(MODEL_ORDER[0]), 'transfer_event_id'].nunique():,} transfers and {predictions.loc[predictions['model_variant'].eq(MODEL_ORDER[0]), 'player_id'].nunique():,} players.",
        "",
        f"Origin pre365 opportunity: mean {frame['origin_opportunity_share_pre365'].mean():.6f}, median {frame['origin_opportunity_share_pre365'].median():.6f}, zero rows {int(frame['origin_opportunity_share_pre365'].eq(0).sum())}, maximum {frame['origin_opportunity_share_pre365'].max():.16f}.",
        f"Origin pre730 opportunity: mean {frame['origin_opportunity_share_pre730'].mean():.6f}, median {frame['origin_opportunity_share_pre730'].median():.6f}, zero rows {int(frame['origin_opportunity_share_pre730'].eq(0).sum())}, maximum {frame['origin_opportunity_share_pre730'].max():.16f}.",
        f"Full-cohort descriptive correlations with destination opportunity: pre365 Pearson {full_pearson_365:.6f}, Spearman {full_spearman_365:.6f}; pre730 Pearson {full_pearson_730:.6f}, Spearman {full_spearman_730:.6f}. These were not used for model selection.",
        "",
        "## Frozen models",
        "",
    ]
    for model in MODEL_ORDER:
        lines.append(f"- `{model}`: {len(feature_sets[model])} predictors.")
    lines += ["", "## Plain-language block results", ""]
    descriptions = [
        ("M1_vs_M0_market_profile", "Market/profile over mean"),
        ("M2_vs_M1_prior_opportunity", "Prior opportunity beyond market/profile"),
        ("M3_vs_M2_player_performance", "Performance beyond prior opportunity"),
        ("M4_vs_M3_destination_context", "Destination context beyond player controls"),
        ("M5_vs_M4_origin_transition_context", "Origin/transition beyond destination context"),
    ]
    for index, (comparison, description) in enumerate(descriptions, start=1):
        pooled = result(comparison)
        mature = result(comparison, "mature_three")
        lines.append(
            f"{index}. {description}: pooled MAE improvement {pooled['pooled_mae_improvement']:.6f}, player-clustered 95% CI [{pooled['cluster_bootstrap_ci_lower_95']:.6f}, {pooled['cluster_bootstrap_ci_upper_95']:.6f}], `{pooled['stability_label']}`; mature-three {mature['pooled_mae_improvement']:.6f}, CI [{mature['cluster_bootstrap_ci_lower_95']:.6f}, {mature['cluster_bootstrap_ci_upper_95']:.6f}], `{mature['stability_label']}`."
        )
    lines += [
        "",
        f"The defensible revised angle is **{angle}**. {statement}",
        "",
        "Persistence is not compatibility, and contextual improvement is predictive rather than causal. `opportunity_persistence_summary.csv` contains the evaluation-only persistence diagnostics.",
        "",
        "## Integrity",
        "",
        f"- Comprehensive master SHA-256: `{master_hash}`.",
        f"- Protected existing outputs unchanged during the run: `{str(protected_unchanged).lower()}`.",
        "- Independent verification is recorded separately after the verifier runs.",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> None:
    protected_before = protected_tree_sha256()
    master_hash_before = sha256_file(MASTER_PATH)
    expected_hash = json.loads(SUMMARY_PATH.read_text(encoding="utf-8")).get("master_sha256")
    if master_hash_before != expected_hash:
        raise RuntimeError("Master hash does not match its verified summary.")

    master = pd.read_csv(MASTER_PATH, low_memory=False)
    cohort = cohort_from_master(master)
    audit_cohort(cohort)
    frame, feature_sets, manifest = build_model_frame(cohort)
    audit_usage_measures(frame)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUTPUT_DIR / "opportunity_transferability_feature_manifest.csv", index=False)

    prediction_rows: list[dict[str, Any]] = []
    origin_rows: list[dict[str, Any]] = []
    earliest_season = int(frame["season_start_year"].min())
    for origin_id, train_end, validation_year, evaluation_year in ORIGINS:
        train = frame.loc[frame["season_start_year"].le(train_end)].copy()
        validation = frame.loc[frame["season_start_year"].eq(validation_year)].copy()
        evaluation = frame.loc[frame["season_start_year"].eq(evaluation_year)].copy()
        if min(len(train), len(validation), len(evaluation)) == 0:
            raise RuntimeError(f"Empty split for {origin_id}.")
        evaluation = evaluation.sort_values("transfer_event_id").reset_index(drop=True)
        fit_frame = pd.concat([train, validation], ignore_index=True)
        for model_variant in MODEL_ORDER:
            print(f"origin={origin_id} model={model_variant}", flush=True)
            features = feature_sets[model_variant]
            if model_variant == MODEL_ORDER[0]:
                validation_prediction = np.full(len(validation), train[TARGET].mean(), dtype=float)
                validation_metrics = metrics(validation[TARGET].to_numpy(dtype=float), validation_prediction)
                raw_prediction = np.full(len(evaluation), fit_frame[TARGET].mean(), dtype=float)
                bounded_prediction = apply_bounds(raw_prediction, 0.0, 1.05)
                alpha: float | None = None
                encoded_count = 0
                family = "mean_only"
            else:
                alpha, validation_metrics = tune_ridge(train, validation, features)
                raw_prediction, bounded_prediction, encoded_count = fit_and_score(
                    fit_frame, evaluation, features, alpha
                )
                family = "ridge"
            evaluation_metrics = metrics(evaluation[TARGET].to_numpy(dtype=float), bounded_prediction)
            origin_rows.append(
                {
                    "origin_id": origin_id,
                    "model_variant": model_variant,
                    "model_family": family,
                    "train_start_year": earliest_season,
                    "train_end_year": train_end,
                    "validation_year": validation_year,
                    "evaluation_year": evaluation_year,
                    "train_rows": len(train),
                    "train_unique_players": train["player_id"].nunique(),
                    "validation_rows": len(validation),
                    "validation_unique_players": validation["player_id"].nunique(),
                    "evaluation_rows": len(evaluation),
                    "evaluation_unique_players": evaluation["player_id"].nunique(),
                    "raw_feature_count": len(features),
                    "encoded_feature_count": encoded_count,
                    "selected_alpha": alpha,
                    "validation_mae": validation_metrics["mae"],
                    "validation_rmse": validation_metrics["rmse"],
                    "evaluation_mae": evaluation_metrics["mae"],
                    "evaluation_rmse": evaluation_metrics["rmse"],
                    "evaluation_r2": evaluation_metrics["r2"],
                    "alpha_selection_year": None if alpha is None else validation_year,
                    "preprocessor_train_end_year": None if alpha is None else train_end,
                    "preprocessor_refit_end_year": None if alpha is None else validation_year,
                }
            )
            for row_index, (_, row) in enumerate(evaluation.iterrows()):
                prediction_rows.append(
                    {
                        "origin_id": origin_id,
                        "evaluation_year": evaluation_year,
                        "transfer_event_id": row["transfer_event_id"],
                        "player_id": row["player_id"],
                        "transfer_date": row["transfer_date"].date().isoformat(),
                        "model_variant": model_variant,
                        "model_family": family,
                        "selected_alpha": alpha,
                        "actual": float(row[TARGET]),
                        "origin_opportunity_share_pre365": float(row["origin_opportunity_share_pre365"]),
                        "raw_prediction": float(raw_prediction[row_index]),
                        "bounded_prediction": float(bounded_prediction[row_index]),
                        "absolute_error": float(abs(row[TARGET] - bounded_prediction[row_index])),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    for origin_id, _, _, _ in ORIGINS:
        expected_ids: list[str] | None = None
        for model in MODEL_ORDER:
            ids = predictions.loc[
                predictions["origin_id"].eq(origin_id) & predictions["model_variant"].eq(model),
                "transfer_event_id",
            ].astype(str).tolist()
            if expected_ids is None:
                expected_ids = ids
            elif ids != expected_ids:
                raise RuntimeError(f"Evaluation rows differ across models for {origin_id}.")

    result_rows = []
    for comparison, baseline, candidate in COMPARISONS:
        for window, years in WINDOWS.items():
            result_rows.append(
                summarize_comparison(predictions, comparison, baseline, candidate, window, years)
            )
    results = pd.DataFrame(result_rows)
    origins = pd.DataFrame(origin_rows)
    persistence = persistence_summary(predictions)

    master_hash_after = sha256_file(MASTER_PATH)
    protected_after = protected_tree_sha256()
    protected_unchanged = protected_before == protected_after
    if master_hash_before != master_hash_after:
        raise RuntimeError("Master changed during the run.")
    if not protected_unchanged:
        raise RuntimeError("A protected existing output changed during the run.")

    origins["master_sha256_before"] = master_hash_before
    origins["master_sha256_after"] = master_hash_after
    origins["protected_tree_sha256_before"] = protected_before
    origins["protected_tree_sha256_after"] = protected_after
    origins["protected_outputs_unchanged"] = protected_unchanged

    predictions.to_csv(OUTPUT_DIR / "opportunity_transferability_predictions.csv", index=False)
    origins.to_csv(OUTPUT_DIR / "opportunity_transferability_origin_results.csv", index=False)
    results.to_csv(OUTPUT_DIR / "opportunity_transferability_results.csv", index=False)
    persistence.to_csv(OUTPUT_DIR / "opportunity_persistence_summary.csv", index=False)
    write_readme(
        frame,
        predictions,
        results,
        persistence,
        feature_sets,
        master_hash_after,
        protected_unchanged,
    )
    print(f"Wrote {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    run()

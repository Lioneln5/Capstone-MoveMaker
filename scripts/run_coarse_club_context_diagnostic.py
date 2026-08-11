"""Run the bounded large-n Transfermarkt club-context/compatibility diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
MASTER_PATH = ROOT / "Data" / "processed" / "transfermarkt_merged" / "transfermarkt_comprehensive_master.csv"
SUMMARY_PATH = ROOT / "Data" / "processed" / "transfermarkt_merged" / "transfermarkt_comprehensive_summary.json"
OUTPUT_DIR = ROOT / "Data" / "processed" / "coarse_club_context_diagnostic"

TARGET = "target_destination_minutes_opportunity_share_24m"
RANDOM_SEED = 20260810
BOOTSTRAP_REPETITIONS = 5000
MODEL_ORDER = [
    "M0_player_history_baseline",
    "M1_plus_destination_context",
    "M2_plus_origin_transition_context",
    "M3_plus_coarse_fit",
]
ORIGINS = [
    ("origin_2019", 2018, 2019, 2020),
    ("origin_2020", 2019, 2020, 2021),
    ("origin_2021", 2020, 2021, 2022),
    ("origin_2022", 2021, 2022, 2023),
]
WINDOWS = {
    "pooled_four": [2020, 2021, 2022, 2023],
    "mature_three": [2021, 2022, 2023],
    "terminal_two": [2022, 2023],
}
COMPARISONS = [
    ("M1_vs_M0_destination_context", MODEL_ORDER[0], MODEL_ORDER[1]),
    ("M2_vs_M1_origin_transition_context", MODEL_ORDER[1], MODEL_ORDER[2]),
    ("M3_vs_M2_coarse_fit", MODEL_ORDER[2], MODEL_ORDER[3]),
    ("M3_vs_M0_total_contextual", MODEL_ORDER[0], MODEL_ORDER[3]),
]
PROTECTED_DIRS = [
    ROOT / "Data" / "processed" / "compatibility_features",
    ROOT / "Data" / "processed" / "compatibility_model_ablations",
    ROOT / "Data" / "processed" / "compatibility_models",
    ROOT / "Data" / "processed" / "compatibility_targets",
    ROOT / "Data" / "processed" / "rolling_origin_stability",
    ROOT / "Data" / "processed" / "subgroup_stability",
    ROOT / "Data" / "processed" / "player_club_case_studies",
]


VALUATION_FEATURES = [
    "valuation_pre_eur",
    "valuation_pre_staleness_days",
    "valuation_pre_is_origin_club",
    "valuation_pre_within_90d",
    "valuation_pre_within_180d",
    "valuation_pre_within_365d",
    "valuation_pre365_observations",
    "valuation_pre730_observations",
    "valuation_career_pre_observations",
    "valuation_career_pre_first_eur",
    "valuation_career_pre_last_eur",
    "valuation_career_pre_mean_eur",
    "valuation_career_pre_min_eur",
    "valuation_career_pre_max_eur",
    "valuation_career_pre_change_eur",
    "valuation_career_pre_change_pct",
    "valuation_career_pre_observation_span_days",
    "valuation_pre730_first_eur",
    "valuation_pre730_last_eur",
    "valuation_pre730_mean_eur",
    "valuation_pre730_min_eur",
    "valuation_pre730_max_eur",
    "valuation_pre730_change_eur",
    "valuation_pre730_change_pct",
    "valuation_pre730_observation_span_days",
    "valuation_pre365_first_eur",
    "valuation_pre365_last_eur",
    "valuation_pre365_mean_eur",
    "valuation_pre365_min_eur",
    "valuation_pre365_max_eur",
    "valuation_pre365_change_eur",
    "valuation_pre365_change_pct",
    "valuation_pre365_observation_span_days",
    "has_pretransfer_valuation_365d",
]

APPEARANCE_PREFIXES = [
    "appearance_career_pre",
    "appearance_pre365_all",
    "appearance_pre365_origin",
    "appearance_pre730_all",
    "appearance_pre730_origin",
]
APPEARANCE_SUFFIXES = [
    "has_records",
    "records",
    "games",
    "minutes",
    "goals",
    "assists",
    "yellow_cards",
    "red_cards",
    "clubs",
    "competitions",
    "goals_per90",
    "assists_per90",
]
LINEUP_PREFIXES = [
    "lineup_career_pre",
    "lineup_pre365_all",
    "lineup_pre365_origin",
    "lineup_pre730_all",
    "lineup_pre730_origin",
]
LINEUP_SUFFIXES = [
    "has_records",
    "records",
    "games",
    "starts",
    "substitute_selections",
    "captain_selections",
    "start_pct",
    "distinct_positions",
]
EVENT_PREFIXES = [
    "event_career_pre",
    "event_pre365_all",
    "event_pre365_origin",
    "event_pre730_all",
    "event_pre730_origin",
]
EVENT_SUFFIXES = [
    "has_records",
    "games",
    "primary_events",
    "goal_events",
    "card_events",
    "shootout_events",
    "substitution_out_events",
    "substitution_in_events",
    "assist_events",
]
CONTEXT_RAW_SUFFIXES = [
    "games",
    "wins",
    "draws",
    "losses",
    "goals_for",
    "goals_against",
    "goal_difference",
    "points",
    "win_pct",
    "points_per_game",
    "home_games",
    "away_games",
    "competitions",
    "domestic_league_games",
    "domestic_cup_games",
    "international_games",
]
CONTEXT_DERIVED_SUFFIXES = [
    "goals_for_per_game",
    "goals_against_per_game",
    "goal_difference_per_game",
    "domestic_league_game_share",
    "domestic_cup_game_share",
    "international_game_share",
]
TRANSITION_METRICS = [
    "points_per_game",
    "win_pct",
    "goals_for_per_game",
    "goals_against_per_game",
    "goal_difference_per_game",
    "domestic_league_game_share",
    "domestic_cup_game_share",
    "international_game_share",
]
POSITION_GROUPS = ["goalkeeper", "defender", "midfielder", "attacker", "unknown"]


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    return num.div(den.where(den.gt(0)))


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


def position_group(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    if "goal" in text:
        return "goalkeeper"
    if any(token in text for token in ["back", "defender", "sweeper"]):
        return "defender"
    if "midfield" in text:
        return "midfielder"
    if any(token in text for token in ["winger", "forward", "striker", "attack"]):
        return "attacker"
    return "unknown"


def cohort_from_master(master: pd.DataFrame) -> pd.DataFrame:
    mask = (
        as_bool(master["eligible_sporting_24m_provisional"])
        & as_bool(master["sporting_post24_horizon_observable"])
        & pd.to_numeric(master["club_origin_pre365_games"], errors="coerce").ge(10)
        & pd.to_numeric(master["club_destination_pre365_games"], errors="coerce").ge(10)
    )
    cohort = master.loc[mask].copy()
    cohort["transfer_date"] = pd.to_datetime(cohort["transfer_date"], errors="coerce")
    cohort["season_start_year"] = cohort["transfer_date"].dt.year.where(
        cohort["transfer_date"].dt.month.ge(7),
        cohort["transfer_date"].dt.year - 1,
    ).astype("Int64")
    denominator = 90.0 * pd.to_numeric(cohort["club_destination_post24_games"], errors="coerce")
    cohort[TARGET] = safe_divide(cohort["appearance_post24_destination_minutes"], denominator)
    cohort = cohort.sort_values(["season_start_year", "transfer_date", "transfer_event_id"]).reset_index(drop=True)
    return cohort


def assert_preflight(cohort: pd.DataFrame) -> None:
    observed = {
        "rows": len(cohort),
        "players": cohort["player_id"].nunique(),
        "date_min": cohort["transfer_date"].min().date().isoformat(),
        "date_max": cohort["transfer_date"].max().date().isoformat(),
    }
    expected = {
        "rows": 3134,
        "players": 1072,
        "date_min": "2012-12-31",
        "date_max": "2024-07-05",
    }
    if observed != expected:
        raise RuntimeError(f"Audited cohort mismatch: observed={observed}; expected={expected}")
    denominator = pd.to_numeric(cohort["club_destination_post24_games"], errors="coerce")
    target = pd.to_numeric(cohort[TARGET], errors="coerce")
    if not denominator.gt(0).all():
        raise RuntimeError("Target denominator is nonpositive or missing for at least one cohort row.")
    if target.isna().any() or not target.between(0, 1.05).all():
        raise RuntimeError("Normalized opportunity target is missing or outside [0, 1.05].")
    zero_minutes = pd.to_numeric(cohort["appearance_post24_destination_minutes"], errors="coerce").eq(0)
    if not target.loc[zero_minutes].eq(0).all():
        raise RuntimeError("Zero-minute outcomes were not retained as genuine zero shares.")


def expected_history_features() -> list[str]:
    appearance = [f"{prefix}_{suffix}" for prefix in APPEARANCE_PREFIXES for suffix in APPEARANCE_SUFFIXES]
    lineup = [f"{prefix}_{suffix}" for prefix in LINEUP_PREFIXES for suffix in LINEUP_SUFFIXES]
    events = [f"{prefix}_{suffix}" for prefix in EVENT_PREFIXES for suffix in EVENT_SUFFIXES]
    return VALUATION_FEATURES + appearance + lineup + events


def add_context_features(
    frame: pd.DataFrame,
    club: str,
    definitions: dict[str, dict[str, str]],
) -> list[str]:
    features: list[str] = []
    for horizon in [365, 730]:
        prefix = f"club_{club}_pre{horizon}"
        for suffix in CONTEXT_RAW_SUFFIXES:
            feature = f"{prefix}_{suffix}"
            features.append(feature)
            definitions.setdefault(
                feature,
                {
                    "feature_kind": "raw",
                    "source_fields": feature,
                    "timing_classification": "strictly pre-transfer club context",
                    "rationale": f"{club} club pre{horizon} context main effect.",
                },
            )
        games = frame[f"{prefix}_games"]
        derived_inputs = {
            "goals_for_per_game": f"{prefix}_goals_for",
            "goals_against_per_game": f"{prefix}_goals_against",
            "goal_difference_per_game": f"{prefix}_goal_difference",
            "domestic_league_game_share": f"{prefix}_domestic_league_games",
            "domestic_cup_game_share": f"{prefix}_domestic_cup_games",
            "international_game_share": f"{prefix}_international_games",
        }
        for suffix, numerator in derived_inputs.items():
            feature = f"{prefix}_{suffix}"
            frame[feature] = safe_divide(frame[numerator], games)
            features.append(feature)
            definitions[feature] = {
                "feature_kind": "derived",
                "source_fields": f"{numerator};{prefix}_games",
                "timing_classification": "derived strictly pre-transfer club context",
                "rationale": f"Normalized {club} club pre{horizon} context main effect.",
            }
    return features


def fallback_rate(
    frame: pd.DataFrame,
    name: str,
    pre365: pd.Series,
    pre730: pd.Series,
    sources: list[str],
    definitions: dict[str, dict[str, str]],
) -> tuple[str, str]:
    feature = f"player_{name}_for_fit"
    indicator = f"player_{name}_fallback_pre730"
    pre365_numeric = pd.to_numeric(pre365, errors="coerce")
    pre730_numeric = pd.to_numeric(pre730, errors="coerce")
    frame[feature] = pre365_numeric.where(pre365_numeric.notna(), pre730_numeric)
    frame[indicator] = (pre365_numeric.isna() & pre730_numeric.notna()).astype(float)
    definitions[feature] = {
        "feature_kind": "derived",
        "source_fields": ";".join(sources),
        "timing_classification": "derived strictly pre-transfer player history",
        "rationale": "Pre365 player rate with declared pre730 fallback used by fixed fit terms.",
    }
    definitions[indicator] = {
        "feature_kind": "derived",
        "source_fields": ";".join(sources),
        "timing_classification": "derived strictly pre-transfer coverage",
        "rationale": "Records use of the declared pre730 fallback without dropping rows.",
    }
    return feature, indicator


def build_model_frame(
    cohort: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    frame = cohort.copy()
    definitions: dict[str, dict[str, str]] = {}

    required_raw = [
        "transfer_fee",
        "transfer_fee_status",
        "market_value_in_eur",
        "player_date_of_birth",
        "lineup_pre365_all_primary_position",
        "lineup_pre365_origin_primary_position",
    ] + expected_history_features()
    missing = sorted(set(required_raw) - set(frame.columns))
    if missing:
        raise RuntimeError(f"Required baseline fields are missing: {missing}")

    transfer_date = pd.to_datetime(frame["transfer_date"], errors="coerce")
    birth_date = pd.to_datetime(frame["player_date_of_birth"], errors="coerce")
    age = (transfer_date - birth_date).dt.days / 365.2425
    frame["age_at_transfer"] = age.where(age.between(15, 45))
    definitions["age_at_transfer"] = {
        "feature_kind": "derived",
        "source_fields": "player_date_of_birth;transfer_date",
        "timing_classification": "derived at transfer",
        "rationale": "Age known on the transfer date; invalid ages become missing.",
    }

    historical_position = frame["lineup_pre365_all_primary_position"].where(
        frame["lineup_pre365_all_primary_position"].notna(),
        frame["lineup_pre365_origin_primary_position"],
    )
    frame["historical_position_group"] = historical_position.map(position_group)
    definitions["historical_position_group"] = {
        "feature_kind": "derived_categorical",
        "source_fields": "lineup_pre365_all_primary_position;lineup_pre365_origin_primary_position",
        "timing_classification": "derived strictly pre-transfer player history",
        "rationale": "Frozen broad role derived from historically observed lineup position.",
    }

    baseline_raw = ["transfer_fee", "transfer_fee_status", "market_value_in_eur"] + expected_history_features()
    for feature in baseline_raw:
        definitions[feature] = {
            "feature_kind": "raw",
            "source_fields": feature,
            "timing_classification": (
                "available at transfer" if feature in {"transfer_fee", "transfer_fee_status", "market_value_in_eur"}
                else "strictly pre-transfer player history"
            ),
            "rationale": "Common transfer-time/player-history control included in all models.",
        }

    start_rate, start_fallback = fallback_rate(
        frame,
        "start_pct",
        frame["lineup_pre365_all_start_pct"],
        frame["lineup_pre730_all_start_pct"],
        ["lineup_pre365_all_start_pct", "lineup_pre730_all_start_pct"],
        definitions,
    )
    minutes_365 = safe_divide(frame["appearance_pre365_all_minutes"], frame["appearance_pre365_all_games"])
    minutes_730 = safe_divide(frame["appearance_pre730_all_minutes"], frame["appearance_pre730_all_games"])
    minutes_rate, minutes_fallback = fallback_rate(
        frame,
        "minutes_per_appearance",
        minutes_365,
        minutes_730,
        [
            "appearance_pre365_all_minutes",
            "appearance_pre365_all_games",
            "appearance_pre730_all_minutes",
            "appearance_pre730_all_games",
        ],
        definitions,
    )
    goals_rate, goals_fallback = fallback_rate(
        frame,
        "goals_per90",
        frame["appearance_pre365_all_goals_per90"],
        frame["appearance_pre730_all_goals_per90"],
        ["appearance_pre365_all_goals_per90", "appearance_pre730_all_goals_per90"],
        definitions,
    )
    assists_rate, assists_fallback = fallback_rate(
        frame,
        "assists_per90",
        frame["appearance_pre365_all_assists_per90"],
        frame["appearance_pre730_all_assists_per90"],
        ["appearance_pre365_all_assists_per90", "appearance_pre730_all_assists_per90"],
        definitions,
    )

    m0 = list(
        dict.fromkeys(
            ["age_at_transfer", "historical_position_group"]
            + baseline_raw
            + [
                start_rate,
                start_fallback,
                minutes_rate,
                minutes_fallback,
                goals_rate,
                goals_fallback,
                assists_rate,
                assists_fallback,
            ]
        )
    )

    destination = add_context_features(frame, "destination", definitions)
    m1 = m0 + destination
    origin = add_context_features(frame, "origin", definitions)
    transition: list[str] = []
    for horizon in [365, 730]:
        for metric_name in TRANSITION_METRICS:
            destination_feature = f"club_destination_pre{horizon}_{metric_name}"
            origin_feature = f"club_origin_pre{horizon}_{metric_name}"
            feature = f"transition_pre{horizon}_{metric_name}_delta"
            frame[feature] = pd.to_numeric(frame[destination_feature], errors="coerce") - pd.to_numeric(
                frame[origin_feature], errors="coerce"
            )
            transition.append(feature)
            definitions[feature] = {
                "feature_kind": "derived",
                "source_fields": (
                    f"{definitions[destination_feature]['source_fields']};"
                    f"{definitions[origin_feature]['source_fields']}"
                ),
                "timing_classification": "derived strictly pre-transfer transition context",
                "rationale": "Predeclared destination-minus-origin transition contrast.",
            }
    m2 = m1 + origin + transition

    fit_features: list[str] = []
    fit_specs = [
        ("fit_start_pct_x_destination_ppg", start_rate, "club_destination_pre365_points_per_game"),
        ("fit_minutes_per_appearance_x_destination_ppg", minutes_rate, "club_destination_pre365_points_per_game"),
        ("fit_goals_per90_x_destination_goals_for_per_game", goals_rate, "club_destination_pre365_goals_for_per_game"),
        ("fit_assists_per90_x_destination_goals_for_per_game", assists_rate, "club_destination_pre365_goals_for_per_game"),
        ("fit_start_pct_x_transition_ppg", start_rate, "transition_pre365_points_per_game_delta"),
        ("fit_minutes_per_appearance_x_transition_ppg", minutes_rate, "transition_pre365_points_per_game_delta"),
    ]
    for feature, player_feature, context_feature in fit_specs:
        frame[feature] = pd.to_numeric(frame[player_feature], errors="coerce") * pd.to_numeric(
            frame[context_feature], errors="coerce"
        )
        fit_features.append(feature)
        definitions[feature] = {
            "feature_kind": "derived_interaction",
            "source_fields": f"{definitions[player_feature]['source_fields']};{definitions[context_feature]['source_fields']}",
            "timing_classification": "derived strictly pre-transfer fit interaction",
            "rationale": "Fixed player-rate by destination/transition context interaction.",
        }

    position_contexts = [
        "club_destination_pre365_points_per_game",
        "club_destination_pre365_goals_for_per_game",
        "club_destination_pre365_goals_against_per_game",
        "transition_pre365_points_per_game_delta",
    ]
    position_sources = definitions["historical_position_group"]["source_fields"]
    for group in POSITION_GROUPS:
        indicator = frame["historical_position_group"].eq(group).astype(float)
        for context_feature in position_contexts:
            short_context = context_feature.removeprefix("club_destination_pre365_").removeprefix("transition_pre365_")
            feature = f"fit_position_{group}_x_{short_context}"
            frame[feature] = indicator * pd.to_numeric(frame[context_feature], errors="coerce")
            fit_features.append(feature)
            definitions[feature] = {
                "feature_kind": "derived_interaction",
                "source_fields": f"{position_sources};{definitions[context_feature]['source_fields']}",
                "timing_classification": "derived strictly pre-transfer fit interaction",
                "rationale": "Fixed historical-position by destination/transition context interaction.",
            }

    m3 = m2 + fit_features
    feature_sets = {
        MODEL_ORDER[0]: m0,
        MODEL_ORDER[1]: m1,
        MODEL_ORDER[2]: m2,
        MODEL_ORDER[3]: m3,
    }
    for name, features in feature_sets.items():
        if len(features) != len(set(features)):
            raise RuntimeError(f"Duplicate feature in {name}.")
        absent = sorted(set(features) - set(frame.columns))
        if absent:
            raise RuntimeError(f"Modeled features absent from frame for {name}: {absent}")

    manifest_rows: list[dict[str, Any]] = []
    for feature_set, features in feature_sets.items():
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


def paired_prediction_frame(
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
        raise RuntimeError("Actual outcomes do not align for a paired comparison.")
    return pairs.drop(columns="candidate_actual")


def row_bootstrap(pairs: pd.DataFrame, seed: int) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    total_n = len(pairs)
    draw_sum = np.zeros(BOOTSTRAP_REPETITIONS, dtype=float)
    for _, group in pairs.groupby("origin_id", sort=True):
        difference = (
            np.abs(group["actual"].to_numpy(dtype=float) - group["baseline_prediction"].to_numpy(dtype=float))
            - np.abs(group["actual"].to_numpy(dtype=float) - group["candidate_prediction"].to_numpy(dtype=float))
        )
        draw = rng.integers(0, len(group), size=(BOOTSTRAP_REPETITIONS, len(group)))
        draw_sum += difference[draw].sum(axis=1)
    draws = draw_sum / total_n
    return {
        "row_bootstrap_ci_lower_95": float(np.quantile(draws, 0.025)),
        "row_bootstrap_ci_upper_95": float(np.quantile(draws, 0.975)),
        "row_bootstrap_probability_candidate_better": float(np.mean(draws > 0)),
    }


def player_cluster_bootstrap(pairs: pd.DataFrame, seed: int) -> dict[str, float | int]:
    working = pairs.copy()
    working["absolute_error_improvement"] = (
        np.abs(working["actual"] - working["baseline_prediction"])
        - np.abs(working["actual"] - working["candidate_prediction"])
    )
    clusters = working.groupby("player_id", sort=True)["absolute_error_improvement"].agg(["sum", "size"])
    sums = clusters["sum"].to_numpy(dtype=float)
    sizes = clusters["size"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(clusters), size=(BOOTSTRAP_REPETITIONS, len(clusters)))
    draws = sums[draw].sum(axis=1) / sizes[draw].sum(axis=1)
    return {
        "cluster_count": int(len(clusters)),
        "cluster_bootstrap_ci_lower_95": float(np.quantile(draws, 0.025)),
        "cluster_bootstrap_ci_upper_95": float(np.quantile(draws, 0.975)),
        "cluster_bootstrap_probability_candidate_better": float(np.mean(draws > 0)),
    }


def stability_label(improvement: float, origin_improvements: np.ndarray, probability: float) -> str:
    origin_count = len(origin_improvements)
    origins_better = int(np.sum(origin_improvements > 0))
    if improvement > 0 and origins_better >= max(3, origin_count - 1) and probability >= 0.90:
        return "stable_positive"
    if improvement > 0 and origins_better >= math.ceil(origin_count / 2):
        return "mixed_positive"
    if improvement <= 0 and origins_better <= 1:
        return "unstable_negative"
    return "mixed"


def summarize_comparison(
    predictions: pd.DataFrame,
    comparison: str,
    baseline: str,
    candidate: str,
    window: str,
    evaluation_years: list[int],
) -> dict[str, Any]:
    pairs = paired_prediction_frame(predictions, baseline, candidate, evaluation_years)
    baseline_error = np.abs(pairs["actual"] - pairs["baseline_prediction"])
    candidate_error = np.abs(pairs["actual"] - pairs["candidate_prediction"])
    baseline_mae = float(baseline_error.mean())
    candidate_mae = float(candidate_error.mean())
    improvement = baseline_mae - candidate_mae
    origin_improvements = []
    for _, group in pairs.groupby("origin_id", sort=True):
        base = np.abs(group["actual"] - group["baseline_prediction"]).mean()
        cand = np.abs(group["actual"] - group["candidate_prediction"]).mean()
        origin_improvements.append(float(base - cand))
    origin_array = np.asarray(origin_improvements, dtype=float)
    cluster_seed = stable_seed(comparison, window, "player_cluster")
    row_seed = stable_seed(comparison, window, "row")
    cluster = player_cluster_bootstrap(pairs, cluster_seed)
    row = row_bootstrap(pairs, row_seed)
    return {
        "comparison": comparison,
        "baseline_model": baseline,
        "candidate_model": candidate,
        "window": window,
        "evaluation_years": ";".join(str(year) for year in evaluation_years),
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


def evidence_statement(results: pd.DataFrame) -> tuple[str, str]:
    pooled = results.loc[
        results["comparison"].eq("M3_vs_M2_coarse_fit") & results["window"].eq("pooled_four")
    ].iloc[0]
    mature = results.loc[
        results["comparison"].eq("M3_vs_M2_coarse_fit") & results["window"].eq("mature_three")
    ].iloc[0]
    if (
        pooled["stability_label"] == "stable_positive"
        and mature["pooled_mae_improvement"] > 0
        and mature["cluster_bootstrap_ci_lower_95"] > 0
    ):
        return "raises", "The stable incremental coarse-fit effect raises confidence that a compatibility-shaped signal exists."
    if (
        pooled["cluster_bootstrap_ci_upper_95"] <= 0
        and mature["cluster_bootstrap_ci_upper_95"] <= 0
    ):
        return "lowers", "The consistently nonpositive coarse-fit effect lowers confidence in these coarse proxies, without disproving richer tactical compatibility."
    return (
        "leaves unchanged",
        "The coarse-fit effect is not stable enough to change confidence that a compatibility-shaped signal exists.",
    )


def write_readme(
    cohort: pd.DataFrame,
    predictions: pd.DataFrame,
    results: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    master_hash: str,
    protected_unchanged: bool,
) -> None:
    def result(comparison: str, window: str = "pooled_four") -> pd.Series:
        return results.loc[results["comparison"].eq(comparison) & results["window"].eq(window)].iloc[0]

    context = result("M1_vs_M0_destination_context")
    transition = result("M2_vs_M1_origin_transition_context")
    fit = result("M3_vs_M2_coarse_fit")
    fit_mature = result("M3_vs_M2_coarse_fit", "mature_three")
    direction, statement = evidence_statement(results)
    lines = [
        "# Coarse club-context and compatibility diagnostic",
        "",
        "## Cohort and target",
        "",
        f"The audited cohort contains {len(cohort):,} transfer rows and {cohort['player_id'].nunique():,} unique players. "
        f"The four pooled evaluation seasons contain {predictions.loc[predictions['model_variant'].eq(MODEL_ORDER[0]), 'transfer_event_id'].nunique():,} rows and "
        f"{predictions.loc[predictions['model_variant'].eq(MODEL_ORDER[0]), 'player_id'].nunique():,} unique players.",
        "",
        f"The sole target is `{TARGET}`: destination minutes in the 730 days after transfer divided by 90 times the destination club's observed games over the same horizon. All cohort rows have a positive denominator and a target in [0, 1.05].",
        "",
        "## Frozen models",
        "",
    ]
    for model in MODEL_ORDER:
        lines.append(f"- `{model}`: {len(feature_sets[model])} raw/derived predictors; ridge only.")
    lines += [
        "",
        "All preprocessing and alpha selection are fit within each chronological origin. The first training window uses every eligible season through 2018. Player-clustered paired bootstrap intervals are primary; row-bootstrap intervals are secondary.",
        "",
        "## Plain-language results",
        "",
        f"1. Destination context (M1 vs M0): MAE improvement {context['pooled_mae_improvement']:.6f}, player-clustered 95% CI [{context['cluster_bootstrap_ci_lower_95']:.6f}, {context['cluster_bootstrap_ci_upper_95']:.6f}], `{context['stability_label']}`.",
        f"2. Origin/transition context (M2 vs M1): MAE improvement {transition['pooled_mae_improvement']:.6f}, player-clustered 95% CI [{transition['cluster_bootstrap_ci_lower_95']:.6f}, {transition['cluster_bootstrap_ci_upper_95']:.6f}], `{transition['stability_label']}`.",
        f"3. Fixed player-by-context fit terms (M3 vs M2): pooled MAE improvement {fit['pooled_mae_improvement']:.6f}, player-clustered 95% CI [{fit['cluster_bootstrap_ci_lower_95']:.6f}, {fit['cluster_bootstrap_ci_upper_95']:.6f}], `{fit['stability_label']}`; mature-three improvement {fit_mature['pooled_mae_improvement']:.6f}, CI [{fit_mature['cluster_bootstrap_ci_lower_95']:.6f}, {fit_mature['cluster_bootstrap_ci_upper_95']:.6f}], `{fit_mature['stability_label']}`.",
        f"4. Stability is evaluated in pooled-four, mature-three, and terminal-two windows in `coarse_club_context_results.csv`.",
        f"5. This evidence **{direction}** confidence: {statement}",
        "",
        "The existing rich-cohort ridge opportunity result was positive (MAE improvement 0.044804, 95% CI [0.029543, 0.060442], pooled n 207). Only direction and stability should be compared: this diagnostic uses a 24-month target, a broader cohort, and coarser context proxies.",
        "",
        "## Integrity",
        "",
        f"- Comprehensive master SHA-256: `{master_hash}`.",
        f"- Protected existing output trees unchanged during the run: `{str(protected_unchanged).lower()}`.",
        "- Independent verification is recorded in `independent_verification.json` and `.csv` after the separate verifier runs.",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> None:
    protected_before = protected_tree_sha256()
    master_hash_before = sha256_file(MASTER_PATH)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if master_hash_before != summary.get("master_sha256"):
        raise RuntimeError("Comprehensive master hash does not match its verified summary.")

    master = pd.read_csv(MASTER_PATH, low_memory=False)
    cohort = cohort_from_master(master)
    assert_preflight(cohort)
    frame, feature_sets, manifest = build_model_frame(cohort)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUTPUT_DIR / "coarse_club_context_feature_manifest.csv", index=False)

    prediction_rows: list[dict[str, Any]] = []
    origin_rows: list[dict[str, Any]] = []
    earliest_season = int(frame["season_start_year"].min())
    for origin_id, train_end, validation_year, evaluation_year in ORIGINS:
        train = frame.loc[frame["season_start_year"].le(train_end)].copy()
        validation = frame.loc[frame["season_start_year"].eq(validation_year)].copy()
        evaluation = frame.loc[frame["season_start_year"].eq(evaluation_year)].copy()
        if min(len(train), len(validation), len(evaluation)) == 0:
            raise RuntimeError(f"Empty chronological split for {origin_id}.")
        evaluation = evaluation.sort_values("transfer_event_id").reset_index(drop=True)
        fit_frame = pd.concat([train, validation], ignore_index=True)
        for model_variant in MODEL_ORDER:
            print(f"origin={origin_id} model={model_variant}", flush=True)
            features = feature_sets[model_variant]
            alpha, validation_metrics = tune_ridge(train, validation, features)
            raw_prediction, bounded_prediction, encoded_count = fit_and_score(
                fit_frame, evaluation, features, alpha
            )
            evaluation_metrics = metrics(evaluation[TARGET].to_numpy(dtype=float), bounded_prediction)
            origin_rows.append(
                {
                    "origin_id": origin_id,
                    "model_variant": model_variant,
                    "model_family": "ridge",
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
                    "alpha_selection_year": validation_year,
                    "preprocessor_train_end_year": train_end,
                    "preprocessor_refit_end_year": validation_year,
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
                        "model_family": "ridge",
                        "selected_alpha": alpha,
                        "actual": float(row[TARGET]),
                        "raw_prediction": float(raw_prediction[row_index]),
                        "bounded_prediction": float(bounded_prediction[row_index]),
                        "absolute_error": float(abs(row[TARGET] - bounded_prediction[row_index])),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    for origin_id, _, _, _ in ORIGINS:
        expected_ids: list[str] | None = None
        for model_variant in MODEL_ORDER:
            ids = predictions.loc[
                predictions["origin_id"].eq(origin_id) & predictions["model_variant"].eq(model_variant),
                "transfer_event_id",
            ].astype(str).tolist()
            if expected_ids is None:
                expected_ids = ids
            elif ids != expected_ids:
                raise RuntimeError(f"Evaluation rows are not identical across models for {origin_id}.")

    result_rows = []
    for comparison, baseline, candidate in COMPARISONS:
        for window, years in WINDOWS.items():
            result_rows.append(
                summarize_comparison(predictions, comparison, baseline, candidate, window, years)
            )
    results = pd.DataFrame(result_rows)
    origins = pd.DataFrame(origin_rows)

    master_hash_after = sha256_file(MASTER_PATH)
    protected_after = protected_tree_sha256()
    protected_unchanged = protected_before == protected_after
    if master_hash_before != master_hash_after:
        raise RuntimeError("Comprehensive master changed during the diagnostic run.")
    if not protected_unchanged:
        raise RuntimeError("A protected existing output tree changed during the diagnostic run.")

    origins["master_sha256_before"] = master_hash_before
    origins["master_sha256_after"] = master_hash_after
    origins["protected_tree_sha256_before"] = protected_before
    origins["protected_tree_sha256_after"] = protected_after
    origins["protected_outputs_unchanged"] = protected_unchanged

    predictions.to_csv(OUTPUT_DIR / "coarse_club_context_predictions.csv", index=False)
    origins.to_csv(OUTPUT_DIR / "coarse_club_context_origin_results.csv", index=False)
    results.to_csv(OUTPUT_DIR / "coarse_club_context_results.csv", index=False)
    write_readme(cohort, predictions, results, feature_sets, master_hash_after, protected_unchanged)
    print(f"Wrote {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    run()

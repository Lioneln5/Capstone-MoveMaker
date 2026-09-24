"""Predeclared development contract for repaired contribution target Run 2."""

from __future__ import annotations

from typing import Final

from models.engine_v2.feature_contract import (
    CORE_PROFILE_FEATURES,
    V2_CORE_FEATURES,
    V2_HISTORY_FEATURES,
)


CONTRACT_VERSION: Final[str] = "engine_v2_contribution_model_run2_v1_2026-09-08"
ORIGINS: Final[tuple[tuple[str, int, int, int], ...]] = (
    ("origin_2020", 2018, 2019, 2020),
    ("origin_2021", 2019, 2020, 2021),
    ("origin_2022", 2020, 2021, 2022),
    ("origin_2023", 2021, 2022, 2023),
)
RANDOM_SEED: Final[int] = 20260908
LOGISTIC_C_GRID: Final[tuple[float, ...]] = (0.01, 0.1, 1.0, 10.0)
BOOTSTRAP_REPETITIONS: Final[int] = 5000

CANDIDATE_FEATURES: Final[dict[str, list[str]]] = {
    "B0_chronology_prevalence": [],
    "B1_market_profile": list(CORE_PROFILE_FEATURES),
    "B2_core_recent_involvement": list(V2_CORE_FEATURES),
    "B3_plus_prior_volume": list(V2_HISTORY_FEATURES),
}

PRIMARY_COMPARISONS: Final[tuple[tuple[str, str], ...]] = (
    ("B0_chronology_prevalence", "B1_market_profile"),
    ("B0_chronology_prevalence", "B2_core_recent_involvement"),
    ("B0_chronology_prevalence", "B3_plus_prior_volume"),
    ("B1_market_profile", "B2_core_recent_involvement"),
    ("B2_core_recent_involvement", "B3_plus_prior_volume"),
)

# The primary candidate is fixed before repaired-target performance is read.
# Prior volume may replace it only with clear incremental evidence.
PRIMARY_CANDIDATE: Final[str] = "B2_core_recent_involvement"
OPTIONAL_EXTENSION_CANDIDATE: Final[str] = "B3_plus_prior_volume"
MINIMUM_ORIGIN_WINS: Final[int] = 3
MINIMUM_POOLED_AUC: Final[float] = 0.70
MAXIMUM_ABSOLUTE_CALIBRATION_GAP: Final[float] = 0.05


def contract_payload() -> dict[str, object]:
    return {
        "contract_version": CONTRACT_VERSION,
        "target_contract_version": "engine_v2_sustained_realized_contribution_v2_2026-09-07",
        "target_semantics": "sustained realized extension-club contribution, not role when available",
        "development_origins": [list(origin) for origin in ORIGINS],
        "candidate_features": CANDIDATE_FEATURES,
        "primary_candidate": PRIMARY_CANDIDATE,
        "optional_extension_candidate": OPTIONAL_EXTENSION_CANDIDATE,
        "model_family": "L2 logistic regression",
        "hyperparameter_selection": "C selected by preceding-year validation Brier loss only",
        "c_grid": list(LOGISTIC_C_GRID),
        "primary_loss": "Brier score",
        "secondary_metrics": [
            "log loss",
            "ROC AUC",
            "average precision",
            "adaptive expected calibration error",
            "calibration gap",
        ],
        "uncertainty": f"{BOOTSTRAP_REPETITIONS} player-cluster bootstrap draws over pooled out-of-time loss deltas",
        "primary_evaluation": "expanding rolling origin, evaluation-player-disjoint from model fitting rows",
        "sensitivity_evaluation": "ordinary expanding rolling origin retaining prior events for repeat players",
        "advancement_gates": {
            "primary_candidate": [
                "95% player-cluster interval for Brier improvement versus chronology prevalence is above zero",
                f"wins at least {MINIMUM_ORIGIN_WINS} of 4 origins versus chronology prevalence",
                "pooled Brier improvement versus market profile is positive",
                f"pooled ROC AUC is at least {MINIMUM_POOLED_AUC:.2f}",
                f"absolute pooled calibration gap is at most {MAXIMUM_ABSOLUTE_CALIBRATION_GAP:.2f}",
            ],
            "optional_prior_volume_replacement": [
                "95% player-cluster interval for Brier improvement versus the primary candidate is above zero",
                f"wins at least {MINIMUM_ORIGIN_WINS} of 4 origins versus the primary candidate",
            ],
        },
        "prohibited_features": [
            "raw goals or assists rates",
            "advanced event statistics",
            "salary or wage fields",
            "proposed contract terms",
            "post-extension predictors",
        ],
        "excluded_from_run2_search": [
            "role/squad competition blocks",
            "manager/tactical blocks",
            "lagged club-behavior blocks",
            "nonlinear model families",
        ],
        "final_holdout_policy": "No extension signed in 2024 or later is evaluated.",
        "artifact_policy": "No fitted model binary is written; results are development evidence only.",
        "deployment_status": "research_only_not_deployed",
    }

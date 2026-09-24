"""Frozen development contract for the public-value downside cleanup.

The 2024+ public-value outcomes were previously inspected by the frozen V1
out-of-sample verifier.  They are therefore excluded from every selection,
calibration, and promotion decision in this repair.  The run produces
development evidence only; it does not serialize or deploy a model.
"""

from __future__ import annotations

from typing import Final

from models.engine_v2.feature_contract import CORE_PROFILE_FEATURES, V2_CORE_FEATURES


CONTRACT_VERSION: Final[str] = "engine_v2_public_value_downside_repair_v1_2026-09-09"
RANDOM_SEED: Final[int] = 20260909

ORIGINS: Final[tuple[tuple[str, int, int, int], ...]] = (
    ("origin_2020", 2018, 2019, 2020),
    ("origin_2021", 2019, 2020, 2021),
    ("origin_2022", 2020, 2021, 2022),
    ("origin_2023", 2021, 2022, 2023),
)

ENDPOINTS: Final[dict[str, dict[str, str]]] = {
    "downside_25pct_12m": {
        "target": "target_downside_25pct_12m",
        "horizon": "12m",
        "product_role": "primary",
    },
    "downside_25pct_24m": {
        "target": "target_downside_25pct_24m",
        "horizon": "24m",
        "product_role": "secondary_provisional",
    },
}

CANDIDATE_FEATURES: Final[dict[str, list[str]]] = {
    "B0_chronology_prevalence": [],
    "B1_market_profile": list(CORE_PROFILE_FEATURES),
    "B2_core_recent_involvement": list(V2_CORE_FEATURES),
}
PRIMARY_CANDIDATE: Final[str] = "B2_core_recent_involvement"
LOGISTIC_C_GRID: Final[tuple[float, ...]] = (0.01, 0.1, 1.0, 10.0)
CALIBRATION_METHODS: Final[tuple[str, ...]] = ("raw", "platt", "isotonic")
CALIBRATION_SIMPLICITY: Final[dict[str, int]] = {"raw": 0, "platt": 1, "isotonic": 2}

BOOTSTRAP_REPETITIONS: Final[int] = 5000
PARAMETER_BOOTSTRAP_REPETITIONS: Final[int] = 100
SUBGROUP_INTERVAL_ALPHA: Final[float] = 0.01
PARAMETER_INTERVAL_ALPHA: Final[float] = 0.20

MINIMUM_ORIGIN_WINS: Final[int] = 3
MINIMUM_POOLED_AUC: Final[float] = 0.70
MAXIMUM_ABSOLUTE_CALIBRATION_GAP: Final[float] = 0.05
CALIBRATION_BRIER_NONINFERIORITY: Final[float] = 0.002
CALIBRATION_LOG_LOSS_NONINFERIORITY: Final[float] = 0.01
CALIBRATION_ECE_NEAR_TIE: Final[float] = 0.005

MINIMUM_SUBGROUP_ROWS: Final[int] = 75
MINIMUM_SUBGROUP_EVENTS: Final[int] = 15
MINIMUM_SUBGROUP_NONEVENTS: Final[int] = 15
MINIMUM_SUBGROUP_ORIGINS: Final[int] = 3
MINIMUM_SUBGROUP_AUC: Final[float] = 0.60
MAXIMUM_SUBGROUP_CALIBRATION_GAP: Final[float] = 0.10

OOD_LIMITED_PERCENTILE: Final[float] = 0.95
OOD_REFUSAL_PERCENTILE: Final[float] = 0.99
MAXIMUM_REFUSAL_RATE: Final[float] = 0.10


def contract_payload() -> dict[str, object]:
    return {
        "contract_version": CONTRACT_VERSION,
        "scope": "incumbent-club extension public-value downside development evidence",
        "endpoints": ENDPOINTS,
        "primary_candidate": PRIMARY_CANDIDATE,
        "candidate_features": CANDIDATE_FEATURES,
        "model_family": "L2 logistic regression",
        "hyperparameter_selection": "C selected by preceding-year validation Brier loss",
        "development_origins": [list(item) for item in ORIGINS],
        "identity_policy": "evaluation-player IDs are excluded from hyperparameter training, calibration/validation, and final fitting rows in each origin",
        "calibration": {
            "methods": list(CALIBRATION_METHODS),
            "fit_data": "preceding-year validation predictions only",
            "selection": (
                "retain methods within raw +0.002 Brier and +0.01 log loss; "
                "choose lowest pooled ECE, with differences <=0.005 favoring simplicity"
            ),
        },
        "prohibited_features": [
            "raw goals or assists rates",
            "advanced event statistics",
            "salary or wage fields",
            "proposed contract terms",
            "post-extension predictors",
        ],
        "advancement_gates": [
            "95% player-cluster interval for Brier improvement versus chronology prevalence is above zero",
            f"wins at least {MINIMUM_ORIGIN_WINS} of four origins versus chronology prevalence",
            "pooled Brier improvement versus market profile is positive",
            f"pooled ROC AUC is at least {MINIMUM_POOLED_AUC:.2f}",
            f"absolute pooled calibration gap is at most {MAXIMUM_ABSOLUTE_CALIBRATION_GAP:.2f}",
        ],
        "exposure_policy": {
            "known_exposed_period": "signed_year >= 2024",
            "known_exposure": "V1 out-of-sample labels, predictions, and aggregate performance were inspected on 2026-08-22",
            "permitted_use": "restricted confirmation of the exact frozen V1 artifacts only",
            "prohibited_use": "feature selection, model selection, calibration, thresholding, refusal design, or V2 promotion",
        },
        "horizon_policy": {
            "12m": "primary because it matures sooner and supports larger later cohorts",
            "24m": "secondary and provisional because follow-up matures slowly and later subgroup support is small",
        },
        "uncertainty": {
            "pooled_loss_draws": BOOTSTRAP_REPETITIONS,
            "subgroup_interval": "99% player-cluster Brier-skill interval",
            "parameter_draws_per_origin": PARAMETER_BOOTSTRAP_REPETITIONS,
            "parameter_prediction_interval": "80% nested player-cluster refit interval",
        },
        "explanation_policy": "The two recent-involvement inputs are reported as one grouped contribution when absolute Spearman correlation exceeds 0.90; no independent causal importance claim is allowed.",
        "artifact_policy": "No fitted model or calibrator is serialized; deployment remains unchanged.",
        "release_status": "development_only_no_pristine_final_holdout",
    }

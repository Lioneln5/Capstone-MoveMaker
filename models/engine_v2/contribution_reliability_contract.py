"""Predeclared Run 3 reliability contract for repaired contribution."""

from __future__ import annotations

from typing import Final


CONTRACT_VERSION: Final[str] = "engine_v2_contribution_reliability_run3_v1_2026-09-08"
MODEL_CONTRACT_VERSION: Final[str] = "engine_v2_contribution_model_run2_v1_2026-09-08"
TARGET_CONTRACT_VERSION: Final[str] = "engine_v2_sustained_realized_contribution_v2_2026-09-07"

RANDOM_SEED: Final[int] = 20260908
CALIBRATION_METHODS: Final[tuple[str, ...]] = ("raw", "platt", "isotonic")
CALIBRATION_SIMPLICITY: Final[dict[str, int]] = {"raw": 0, "platt": 1, "isotonic": 2}
CALIBRATION_BRIER_NONINFERIORITY: Final[float] = 0.002
CALIBRATION_LOG_LOSS_NONINFERIORITY: Final[float] = 0.01
CALIBRATION_ECE_NEAR_TIE: Final[float] = 0.005

PARAMETER_BOOTSTRAP_REPETITIONS: Final[int] = 250
SUBGROUP_BOOTSTRAP_REPETITIONS: Final[int] = 5000
PARAMETER_INTERVAL_ALPHA: Final[float] = 0.20
SUBGROUP_INTERVAL_ALPHA: Final[float] = 0.01

OOD_LIMITED_PERCENTILE: Final[float] = 0.95
OOD_REFUSAL_PERCENTILE: Final[float] = 0.99
MAXIMUM_REFUSAL_RATE: Final[float] = 0.10

MINIMUM_SUBGROUP_ROWS: Final[int] = 75
MINIMUM_SUBGROUP_EVENTS: Final[int] = 15
MINIMUM_SUBGROUP_NONEVENTS: Final[int] = 15
MINIMUM_SUBGROUP_ORIGINS: Final[int] = 3
MINIMUM_SUBGROUP_AUC: Final[float] = 0.60
MAXIMUM_SUBGROUP_CALIBRATION_GAP: Final[float] = 0.10
REQUIRED_SUBGROUP_TYPES: Final[tuple[str, ...]] = ("league", "position")

MAXIMUM_POOLED_ECE: Final[float] = 0.08
MAXIMUM_POOLED_CALIBRATION_GAP: Final[float] = 0.05
MINIMUM_CALIBRATION_SLOPE: Final[float] = 0.75
MAXIMUM_CALIBRATION_SLOPE: Final[float] = 1.25
MAXIMUM_ORIGIN_ECE: Final[float] = 0.12
MAXIMUM_ORIGIN_CALIBRATION_GAP: Final[float] = 0.08
MINIMUM_RELIABLE_ORIGINS: Final[int] = 3
MINIMUM_POSITIVE_SKILL_ORIGINS: Final[int] = 3

MAXIMUM_MEAN_PARAMETER_WIDTH_80: Final[float] = 0.20
MAXIMUM_P90_PARAMETER_WIDTH_80: Final[float] = 0.30
MAXIMUM_EXTREME_PROBABILITY_SHARE: Final[float] = 0.05

PSI_MODERATE_THRESHOLD: Final[float] = 0.10
PSI_HIGH_THRESHOLD: Final[float] = 0.25
MAXIMUM_LATEST_INVOLVEMENT_PSI: Final[float] = 0.25
MAXIMUM_TARGET_RATE_RANGE: Final[float] = 0.15

REDUNDANCY_SPEARMAN_THRESHOLD: Final[float] = 0.90
INVOLVEMENT_FEATURES: Final[tuple[str, ...]] = (
    "pre365_same_club_all_competition_opportunity_share",
    "pre365_same_club_all_competition_appearance_rate",
)
HARD_REQUIRED_FEATURES: Final[tuple[str, ...]] = (
    "age_at_signing",
    "canonical_position",
    "league",
    "log_market_value_at_signing",
    *INVOLVEMENT_FEATURES,
)


def contract_payload() -> dict[str, object]:
    return {
        "contract_version": CONTRACT_VERSION,
        "model_contract_version": MODEL_CONTRACT_VERSION,
        "target_contract_version": TARGET_CONTRACT_VERSION,
        "candidate": "B2_core_recent_involvement",
        "scope": "probability reliability audit only; no feature or model-family search",
        "development_origins": [2020, 2021, 2022, 2023],
        "final_holdout_policy": "No extension signed in 2024 or later is evaluated.",
        "calibration": {
            "methods": list(CALIBRATION_METHODS),
            "fit_data": "preceding-year player-disjoint validation predictions only",
            "selection": (
                "retain methods within raw +0.002 Brier and +0.01 log loss; "
                "choose lowest pooled ECE, with differences <=0.005 favoring simplicity"
            ),
        },
        "uncertainty": {
            "parameter_draws": PARAMETER_BOOTSTRAP_REPETITIONS,
            "parameter_interval": "80% nested player-cluster refit interval",
            "subgroup_draws": SUBGROUP_BOOTSTRAP_REPETITIONS,
            "subgroup_interval": "99% player-cluster Brier-skill interval",
        },
        "profile_support": {
            "limited": "outside a training 1st-99th percentile or above the learned P95 robust-distance envelope",
            "refused": "missing required input, unseen category, outside observed range, or above the learned P99 robust-distance envelope",
            "maximum_refusal_rate": MAXIMUM_REFUSAL_RATE,
        },
        "subgroup_support": {
            "minimum_rows": MINIMUM_SUBGROUP_ROWS,
            "minimum_events": MINIMUM_SUBGROUP_EVENTS,
            "minimum_nonevents": MINIMUM_SUBGROUP_NONEVENTS,
            "minimum_origins": MINIMUM_SUBGROUP_ORIGINS,
            "minimum_auc": MINIMUM_SUBGROUP_AUC,
            "maximum_calibration_gap": MAXIMUM_SUBGROUP_CALIBRATION_GAP,
            "brier_skill_rule": "99% player-cluster interval lower bound above zero",
            "required_types": list(REQUIRED_SUBGROUP_TYPES),
        },
        "endpoint_gates": {
            "pooled_ece_maximum": MAXIMUM_POOLED_ECE,
            "pooled_calibration_gap_maximum": MAXIMUM_POOLED_CALIBRATION_GAP,
            "pooled_calibration_slope_range": [
                MINIMUM_CALIBRATION_SLOPE,
                MAXIMUM_CALIBRATION_SLOPE,
            ],
            "origin_reliability": (
                f"at least {MINIMUM_RELIABLE_ORIGINS}/4 origins with ECE <= "
                f"{MAXIMUM_ORIGIN_ECE} and calibration gap <= {MAXIMUM_ORIGIN_CALIBRATION_GAP}"
            ),
            "positive_skill_origins": MINIMUM_POSITIVE_SKILL_ORIGINS,
            "mean_parameter_width_80_maximum": MAXIMUM_MEAN_PARAMETER_WIDTH_80,
            "p90_parameter_width_80_maximum": MAXIMUM_P90_PARAMETER_WIDTH_80,
            "extreme_probability_share_maximum": MAXIMUM_EXTREME_PROBABILITY_SHARE,
            "latest_involvement_psi_maximum": MAXIMUM_LATEST_INVOLVEMENT_PSI,
            "target_rate_range_maximum": MAXIMUM_TARGET_RATE_RANGE,
            "all_broad_leagues_and_positions_supported": True,
            "involvement_coefficients_positive_in_all_origins": True,
        },
        "explanation_policy": (
            "If the two recent-involvement inputs have absolute Spearman correlation above "
            f"{REDUNDANCY_SPEARMAN_THRESHOLD}, expose only their combined group contribution; "
            "do not present independent causal or importance claims."
        ),
        "artifact_policy": "No fitted model or calibrator is serialized and deployment is unchanged.",
    }

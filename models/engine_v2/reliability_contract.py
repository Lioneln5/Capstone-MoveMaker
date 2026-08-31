"""Reliability thresholds for Engine V2 Phase-3 research candidates."""

from __future__ import annotations

MIN_SUBGROUP_ROWS = 50
MIN_SUBGROUP_EVENTS = 10
MIN_SUBGROUP_NONEVENTS = 10
MIN_SUBGROUP_ORIGINS = 3

MAX_SUPPORTED_CALIBRATION_GAP = 0.10
MIN_SUPPORTED_AUC = 0.55

# Learned independently inside each chronological origin. The support score is
# the maximum robust standardized deviation across required numeric features;
# it is deliberately simpler and more auditable than a density estimate.
OOD_LIMITED_PERCENTILE = 0.95
OOD_REFUSAL_PERCENTILE = 0.99

MAX_ACCEPTABLE_REFUSAL_RATE = 0.15
MAX_ACCEPTABLE_ECE = 0.08
MAX_ACCEPTABLE_PARAMETER_INTERVAL_WIDTH = 0.35
MIN_SUPPORTED_SUBGROUP_SHARE = 0.75
REQUIRED_SUPPORTED_GROUP_TYPES = ("position", "league")
CALIBRATION_BRIER_NONINFERIORITY = 0.002

CALIBRATION_METHODS = ("raw", "platt", "isotonic")

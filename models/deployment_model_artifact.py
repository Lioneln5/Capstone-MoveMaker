"""Shared, independently-importable container for a serialized Phase 1-4
deployment model. Must live in its own module (not inside a script that runs
as __main__) so joblib/pickle can resolve the class from any entry point
that loads an artifact -- scripts/build_deployment_models.py (which creates
them) and models/deployment_scorer.py (which loads them) are different
__main__ contexts, and pickle resolves classes by their defining module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import sklearn

from mixed_type_preprocessor import Preprocessor


@dataclass
class ModelArtifact:
    phase: str
    endpoint: str
    horizon: str | None
    kind: str
    unit: str  # "share_0_1" | "probability" | "log_ratio" | "log_eur" | "years"
    variant: str
    features: list[str]
    preprocessor: Preprocessor
    model: Any
    hyperparameter: dict[str, Any]
    interval_abs_residual_q80: float | None
    training_rows: int
    encoded_feature_count: int
    validation_metrics: dict[str, Any]
    deployment_cutoff_signed_year: int
    model_vintage: str
    trained_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_data_sha256: str = ""
    sklearn_version: str = sklearn.__version__

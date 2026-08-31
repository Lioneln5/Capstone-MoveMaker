"""Versioned, refusal-safe response contract for Engine V2 candidates.

This module deliberately contains no serving code. It defines the envelope a
future API must satisfy before a model result can be exposed. Unsupported
requests cannot retain a numeric value, and independently modeled endpoints
cannot be collapsed into a recommendation or overall score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ENGINE_VERSION = "2.0.0-candidate"
RESULT_STATUSES = {"supported", "limited", "refused", "unavailable"}
RESULT_KINDS = {"probability", "historical_benchmark"}


@dataclass(frozen=True)
class Interval:
    kind: Literal["model_fit_80", "historical_reference_80"]
    low: float
    high: float
    unit: str

    def validate(self) -> None:
        if self.low > self.high:
            raise ValueError("Interval lower bound exceeds upper bound")


@dataclass(frozen=True)
class ModuleResult:
    module_id: str
    endpoint_id: str
    result_kind: Literal["probability", "historical_benchmark"]
    status: Literal["supported", "limited", "refused", "unavailable"]
    value: float | None
    unit: str
    target_definition: str
    horizon: str
    calibration_method: str | None = None
    interval: Interval | None = None
    support_reasons: tuple[str, ...] = ()
    subgroup_reliability: dict[str, str] = field(default_factory=dict)
    data_vintage: dict[str, str] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()
    artifact_id: str | None = None

    def validate(self) -> None:
        if self.result_kind not in RESULT_KINDS:
            raise ValueError(f"Unknown result kind: {self.result_kind}")
        if self.status not in RESULT_STATUSES:
            raise ValueError(f"Unknown result status: {self.status}")
        if self.status in {"refused", "unavailable"}:
            if self.value is not None or self.interval is not None:
                raise ValueError("Refused/unavailable modules must not expose a value or interval")
            if not self.support_reasons:
                raise ValueError("Refused/unavailable modules require a machine-readable reason")
        elif self.value is None:
            raise ValueError("Supported/limited modules require a value")
        if self.result_kind == "probability" and self.value is not None and not 0 <= self.value <= 1:
            raise ValueError("Probability must lie in [0, 1]")
        if self.interval is not None:
            self.interval.validate()
            if self.value is not None and not self.interval.low <= self.value <= self.interval.high:
                raise ValueError("Point value must lie inside its interval")
        if self.result_kind == "probability" and self.calibration_method is None:
            raise ValueError("Probability results must identify their calibration method")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "module_id": self.module_id,
            "endpoint_id": self.endpoint_id,
            "result_kind": self.result_kind,
            "status": self.status,
            "value": self.value,
            "unit": self.unit,
            "target_definition": self.target_definition,
            "horizon": self.horizon,
            "calibration_method": self.calibration_method,
            "interval": None if self.interval is None else {
                "kind": self.interval.kind,
                "low": self.interval.low,
                "high": self.interval.high,
                "unit": self.interval.unit,
            },
            "support_reasons": list(self.support_reasons),
            "subgroup_reliability": self.subgroup_reliability,
            "data_vintage": self.data_vintage,
            "limitations": list(self.limitations),
            "artifact_id": self.artifact_id,
        }


@dataclass(frozen=True)
class ResultEnvelope:
    request_id: str
    engine_version: str
    scoring_as_of: str
    decision_scope: str
    modules: tuple[ModuleResult, ...]
    overall_score: None = None
    recommendation: None = None

    def validate(self) -> None:
        if self.engine_version != ENGINE_VERSION:
            raise ValueError(f"Unexpected engine version: {self.engine_version}")
        if self.decision_scope != "incumbent_club_extension_scenario":
            raise ValueError("Engine V2 candidate is extension-scenario only")
        if self.overall_score is not None or self.recommendation is not None:
            raise ValueError("The contract prohibits an overall score or automated recommendation")
        ids = [(item.module_id, item.endpoint_id) for item in self.modules]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate module/endpoint result")
        for item in self.modules:
            item.validate()

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "request_id": self.request_id,
            "engine_version": self.engine_version,
            "scoring_as_of": self.scoring_as_of,
            "decision_scope": self.decision_scope,
            "overall_score": None,
            "recommendation": None,
            "modules": [item.as_dict() for item in self.modules],
        }


def validate_payload(payload: dict[str, Any]) -> None:
    """Validate a serialized envelope without silently filling missing fields."""

    required = {
        "request_id", "engine_version", "scoring_as_of", "decision_scope",
        "overall_score", "recommendation", "modules",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Missing envelope fields: {sorted(missing)}")
    if payload["engine_version"] != ENGINE_VERSION:
        raise ValueError("Engine version mismatch")
    if payload["decision_scope"] != "incumbent_club_extension_scenario":
        raise ValueError("Invalid decision scope")
    if payload["overall_score"] is not None or payload["recommendation"] is not None:
        raise ValueError("Overall scores and recommendations are prohibited")
    if not isinstance(payload["modules"], list) or not payload["modules"]:
        raise ValueError("At least one module result is required")
    seen: set[tuple[str, str]] = set()
    for item in payload["modules"]:
        key = (item.get("module_id"), item.get("endpoint_id"))
        if key in seen:
            raise ValueError("Duplicate module/endpoint result")
        seen.add(key)
        status = item.get("status")
        kind = item.get("result_kind")
        value = item.get("value")
        interval = item.get("interval")
        if status not in RESULT_STATUSES or kind not in RESULT_KINDS:
            raise ValueError("Unknown module status or result kind")
        if status in {"refused", "unavailable"}:
            if value is not None or interval is not None:
                raise ValueError("Refused/unavailable module leaks a numeric result")
            if not item.get("support_reasons"):
                raise ValueError("Refused/unavailable module lacks a reason")
        elif value is None:
            raise ValueError("Supported/limited module lacks a value")
        if kind == "probability" and value is not None and not 0 <= value <= 1:
            raise ValueError("Probability outside [0, 1]")
        if interval is not None:
            if interval["low"] > interval["high"]:
                raise ValueError("Reversed interval")
            if value is not None and not interval["low"] <= value <= interval["high"]:
                raise ValueError("Value outside interval")

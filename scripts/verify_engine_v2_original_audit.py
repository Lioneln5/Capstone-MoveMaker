#!/usr/bin/env python3
"""Verify the original Engine V2 audit ledger and Ridge explainer repair."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
if str(MODELS) not in sys.path:
    sys.path.insert(0, str(MODELS))

from feature_contribution_explainer import raw_contributions  # noqa: E402


class IdentityPreprocessor:
    encoded_raw_features = ["test_feature"]

    @staticmethod
    def transform(frame):
        return frame[["test_feature"]].to_numpy(dtype=float)


def test_artifact(*, classifier_shape: bool):
    coef = np.array([[3.0]]) if classifier_shape else np.array([3.0])
    intercept = np.array([4.0]) if classifier_shape else 4.0
    return SimpleNamespace(
        features=["test_feature"],
        preprocessor=IdentityPreprocessor(),
        model=SimpleNamespace(coef_=coef, intercept_=intercept),
    )


def main() -> None:
    ledger_path = ROOT / "docs" / "ENGINE_V2_ORIGINAL_AUDIT.md"
    registry_path = ROOT / "models" / "README.md"
    project_readme_path = ROOT / "README.md"
    ledger = ledger_path.read_text(encoding="utf-8")
    registry = registry_path.read_text(encoding="utf-8")
    project_readme = project_readme_path.read_text(encoding="utf-8")

    rows = re.findall(
        r"^\|\s*(\d+)\s*\|.*?\|\s*\*\*(FIXED|CONTAINED|DEFERRED|OPEN)\*\*\s*\|",
        ledger,
        flags=re.MULTILINE,
    )
    issue_numbers = [int(number) for number, _ in rows]
    statuses = {int(number): status for number, status in rows}
    expected_statuses = {
        1: "FIXED", 2: "DEFERRED", 3: "FIXED", 4: "CONTAINED",
        5: "CONTAINED", 6: "CONTAINED", 7: "FIXED", 8: "CONTAINED",
        9: "CONTAINED", 10: "CONTAINED", 11: "CONTAINED", 12: "FIXED",
    }

    errors: list[str] = []
    if issue_numbers != list(range(1, 13)):
        errors.append(f"expected issue rows 1-12 exactly once, observed {issue_numbers}")
    if statuses != expected_statuses:
        errors.append(f"status ledger mismatch: {statuses}")
    if "ENGINE_V2_ORIGINAL_AUDIT.md" not in registry:
        errors.append("model registry does not link the original audit ledger")
    if "ENGINE_V2_ORIGINAL_AUDIT.md" not in project_readme:
        errors.append("project README does not link the original audit ledger")

    expected = ([('test_feature', 6.0)], 4.0, 10.0)
    for label, classifier_shape in (("ridge", False), ("logistic", True)):
        observed = raw_contributions(
            test_artifact(classifier_shape=classifier_shape),
            {"test_feature": 2.0},
        )
        if observed != expected:
            errors.append(f"{label} reconstruction mismatch: {observed} != {expected}")

    if errors:
        raise SystemExit("Original Engine V2 audit verification failed:\n- " + "\n- ".join(errors))

    print(
        "PASS: 12/12 original issues have explicit statuses; Ridge and "
        "LogisticRegression contribution shapes reconstruct exactly."
    )


if __name__ == "__main__":
    main()

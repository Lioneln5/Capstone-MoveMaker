"""Create mature-origin sensitivity summaries from completed rolling predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_rolling_origin_stability import (
    BOOTSTRAP_REPETITIONS,
    LOCKED_TARGETS,
    MODEL_FAMILIES,
    RANDOM_SEED,
    paired_rows,
    sha256_file,
    summarize_comparison,
)
from train_compatibility_models import paired_bootstrap


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "Data" / "processed" / "rolling_origin_stability"

SUBSETS = {
    "mature_three_origins": ["origin_2020", "origin_2021", "origin_2022"],
    "terminal_two_origins": ["origin_2021", "origin_2022"],
}


def sensitivity_signal(summary: dict[str, Any]) -> str:
    improvement = float(summary["pooled_mae_improvement"])
    win_rate = float(summary["origin_win_rate"])
    lower = float(summary["mae_improvement_ci_lower_95"])
    upper = float(summary["mae_improvement_ci_upper_95"])
    if lower > 0 and win_rate >= 2 / 3:
        return "supported_positive"
    if improvement > 0 and win_rate >= 2 / 3:
        return "directionally_positive"
    if improvement > 0:
        return "mixed_positive"
    if upper < 0 and win_rate <= 1 / 3:
        return "supported_negative"
    return "negative_or_mixed"


def origin_bootstraps(pairs: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    for origin_index, (rolling_origin, group) in enumerate(
        pairs.groupby("origin_id", sort=True)
    ):
        result = paired_bootstrap(
            group["actual"].to_numpy(dtype=float),
            group["baseline_prediction"].to_numpy(dtype=float),
            group["candidate_prediction"].to_numpy(dtype=float),
            seed + origin_index,
        )
        result["evaluation_n"] = result.pop("test_n")
        rows.append({"origin_id": rolling_origin, **result})
    return pd.DataFrame(rows)


def append_summary(
    rows: list[dict[str, Any]],
    pairs: pd.DataFrame,
    subset_name: str,
    analysis_type: str,
    target_key: str,
    comparison: str,
    seed: int,
    model_family: str | None = None,
) -> None:
    subset_pairs = pairs.loc[pairs["origin_id"].isin(SUBSETS[subset_name])].copy()
    origin_results = origin_bootstraps(subset_pairs, seed)
    summary = summarize_comparison(subset_pairs, origin_results, seed + 100)
    rows.append(
        {
            "analysis_type": analysis_type,
            "target_key": target_key,
            "model_family": model_family,
            "comparison": comparison,
            "origin_subset": subset_name,
            "included_origins": ";".join(SUBSETS[subset_name]),
            **summary,
            "sensitivity_signal": sensitivity_signal(summary),
        }
    )


def run() -> None:
    predictions = pd.read_csv(OUTPUT_DIR / "rolling_predictions.csv", low_memory=False)
    policy_predictions = pd.read_csv(
        OUTPUT_DIR / "rolling_policy_predictions.csv", low_memory=False
    )
    locked_predictions = pd.read_csv(
        OUTPUT_DIR / "rolling_locked_predictions.csv", low_memory=False
    )
    rows: list[dict[str, Any]] = []

    for target_index, target_key in enumerate(sorted(predictions["target_key"].unique())):
        for family_index, model_family in enumerate(MODEL_FAMILIES):
            group = predictions.loc[
                predictions["target_key"].eq(target_key)
                & predictions["model_family"].eq(model_family)
            ]
            baseline = group.loc[
                group["feature_set"].eq("player_history_baseline")
            ].sort_values(["origin_id", "canonical_performance_id"])
            full = group.loc[group["feature_set"].eq("full_explicit_fit")].sort_values(
                ["origin_id", "canonical_performance_id"]
            )
            pairs = paired_rows(baseline, full, "full_explicit_fit")
            for subset_index, subset_name in enumerate(SUBSETS):
                append_summary(
                    rows,
                    pairs,
                    subset_name,
                    "within_family_full_vs_baseline",
                    target_key,
                    "full_explicit_fit_vs_player_history_baseline",
                    RANDOM_SEED
                    + 10000
                    + target_index * 100
                    + family_index * 10
                    + subset_index,
                    model_family,
                )

    for target_index, target_key in enumerate(sorted(policy_predictions["target_key"].unique())):
        group = policy_predictions.loc[policy_predictions["target_key"].eq(target_key)]
        baseline = group.loc[
            group["selection_type"].eq("rolling_selected_baseline")
        ].sort_values(["origin_id", "canonical_performance_id"])
        full = group.loc[group["selection_type"].eq("rolling_selected_full")].sort_values(
            ["origin_id", "canonical_performance_id"]
        )
        pairs = paired_rows(baseline, full, "rolling_selected_full")
        for subset_index, subset_name in enumerate(SUBSETS):
            append_summary(
                rows,
                pairs,
                subset_name,
                "rolling_selection_policy",
                target_key,
                "rolling_selected_full_vs_rolling_selected_baseline",
                RANDOM_SEED + 11000 + target_index * 10 + subset_index,
            )

    for target_index, target_key in enumerate(LOCKED_TARGETS):
        group = locked_predictions.loc[locked_predictions["target_key"].eq(target_key)]
        baseline = group.loc[group["design_role"].eq("baseline")].sort_values(
            ["origin_id", "canonical_performance_id"]
        )
        candidate = group.loc[group["design_role"].eq("candidate")].sort_values(
            ["origin_id", "canonical_performance_id"]
        )
        pairs = paired_rows(baseline, candidate, "locked_current_candidate")
        for subset_index, subset_name in enumerate(SUBSETS):
            append_summary(
                rows,
                pairs,
                subset_name,
                "retrospective_locked_design",
                target_key,
                "locked_current_candidate_vs_locked_current_baseline",
                RANDOM_SEED + 12000 + target_index * 10 + subset_index,
            )

    sensitivity = pd.DataFrame(rows).sort_values(
        ["analysis_type", "target_key", "model_family", "origin_subset"],
        na_position="last",
    )
    sensitivity.to_csv(OUTPUT_DIR / "rolling_sensitivity_summary.csv", index=False)

    output_paths = sorted(
        path for path in OUTPUT_DIR.iterdir() if path.is_file() and path.name != "output_manifest.csv"
    )
    pd.DataFrame(
        [
            {"output_file": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in output_paths
        ]
    ).to_csv(OUTPUT_DIR / "output_manifest.csv", index=False)
    print(
        {
            "status": "pass",
            "sensitivity_rows": len(sensitivity),
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        }
    )


if __name__ == "__main__":
    run()

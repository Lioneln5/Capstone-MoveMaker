"""Independently recompute and verify subgroup-stability artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROLLING = ROOT / "Data" / "processed" / "rolling_origin_stability"
OUTPUT = ROOT / "Data" / "processed" / "subgroup_stability"
CROSSWALK = ROOT / "Data" / "processed" / "integration_crosswalks" / "competition_crosswalk.csv"

PREDICTIONS = ROLLING / "rolling_predictions.csv"
POLICY_PREDICTIONS = ROLLING / "rolling_policy_predictions.csv"
SEED = 20260810
REPETITIONS = 5000
MIN_POOLED_N = 15
MIN_ORIGIN_N = 5

CANDIDATES = {
    "opportunity": ("full_explicit_fit", "elastic_net"),
    "performance": ("full_explicit_fit", "gradient_boosted_trees"),
}
SUBSETS = {
    "all_four_origins": ["origin_2019", "origin_2020", "origin_2021", "origin_2022"],
    "mature_three_origins": ["origin_2020", "origin_2021", "origin_2022"],
    "terminal_two_origins": ["origin_2021", "origin_2022"],
}
DIMENSIONS = {
    "role": ("role_group", "role_name"),
    "competition": ("canonical_competition_id", "competition_name"),
}
ROLE_NAMES = {"DF": "Defender", "FW": "Forward", "MF": "Midfielder"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def bh(values: pd.Series) -> pd.Series:
    answer = pd.Series(np.nan, index=values.index, dtype=float)
    ordered = values.dropna().astype(float).sort_values()
    if ordered.empty:
        return answer
    adjusted = ordered.to_numpy() * len(ordered) / np.arange(1, len(ordered) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    answer.loc[ordered.index] = np.minimum(adjusted, 1.0)
    return answer


def origin_bootstrap(group: pd.DataFrame, seed: int) -> dict[str, float]:
    actual = group["actual"].to_numpy(float)
    baseline = group["baseline_prediction"].to_numpy(float)
    candidate = group["candidate_prediction"].to_numpy(float)
    baseline_abs = np.abs(actual - baseline)
    candidate_abs = np.abs(actual - candidate)
    baseline_sq = np.square(actual - baseline)
    candidate_sq = np.square(actual - candidate)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(group), size=(REPETITIONS, len(group)))
    mae_draws = np.mean(baseline_abs[indices] - candidate_abs[indices], axis=1)
    rmse_draws = np.sqrt(np.mean(baseline_sq[indices], axis=1)) - np.sqrt(
        np.mean(candidate_sq[indices], axis=1)
    )
    lower = (np.sum(mae_draws <= 0) + 1) / (REPETITIONS + 1)
    upper = (np.sum(mae_draws >= 0) + 1) / (REPETITIONS + 1)
    return {
        "n": len(group),
        "baseline_mae": np.mean(baseline_abs),
        "candidate_mae": np.mean(candidate_abs),
        "mae_improvement": np.mean(baseline_abs - candidate_abs),
        "relative_mae_improvement": np.mean(baseline_abs - candidate_abs) / np.mean(baseline_abs),
        "mae_improvement_ci_lower_95": np.quantile(mae_draws, 0.025),
        "mae_improvement_ci_upper_95": np.quantile(mae_draws, 0.975),
        "mae_probability_candidate_better": np.mean(mae_draws > 0),
        "bootstrap_p_two_sided": min(1.0, 2 * min(lower, upper)),
        "baseline_rmse": np.sqrt(np.mean(baseline_sq)),
        "candidate_rmse": np.sqrt(np.mean(candidate_sq)),
        "rmse_improvement": np.sqrt(np.mean(baseline_sq)) - np.sqrt(np.mean(candidate_sq)),
        "rmse_improvement_ci_lower_95": np.quantile(rmse_draws, 0.025),
        "rmse_improvement_ci_upper_95": np.quantile(rmse_draws, 0.975),
    }


def aggregate_bootstrap(group: pd.DataFrame, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    mae_sum = np.zeros(REPETITIONS)
    baseline_sq_sum = np.zeros(REPETITIONS)
    candidate_sq_sum = np.zeros(REPETITIONS)
    for _, origin in group.groupby("origin_id", sort=True):
        actual = origin["actual"].to_numpy(float)
        baseline = origin["baseline_prediction"].to_numpy(float)
        candidate = origin["candidate_prediction"].to_numpy(float)
        indices = rng.integers(0, len(origin), size=(REPETITIONS, len(origin)))
        mae_sum += (np.abs(actual - baseline) - np.abs(actual - candidate))[indices].sum(axis=1)
        baseline_sq_sum += np.square(actual - baseline)[indices].sum(axis=1)
        candidate_sq_sum += np.square(actual - candidate)[indices].sum(axis=1)
    actual = group["actual"].to_numpy(float)
    baseline = group["baseline_prediction"].to_numpy(float)
    candidate = group["candidate_prediction"].to_numpy(float)
    baseline_abs = np.abs(actual - baseline)
    candidate_abs = np.abs(actual - candidate)
    mae_draws = mae_sum / len(group)
    rmse_draws = np.sqrt(baseline_sq_sum / len(group)) - np.sqrt(candidate_sq_sum / len(group))
    lower = (np.sum(mae_draws <= 0) + 1) / (REPETITIONS + 1)
    upper = (np.sum(mae_draws >= 0) + 1) / (REPETITIONS + 1)
    return {
        "pooled_n": len(group),
        "pooled_baseline_mae": np.mean(baseline_abs),
        "pooled_candidate_mae": np.mean(candidate_abs),
        "pooled_mae_improvement": np.mean(baseline_abs - candidate_abs),
        "pooled_relative_mae_improvement": np.mean(baseline_abs - candidate_abs) / np.mean(baseline_abs),
        "mae_improvement_ci_lower_95": np.quantile(mae_draws, 0.025),
        "mae_improvement_ci_upper_95": np.quantile(mae_draws, 0.975),
        "mae_probability_candidate_better": np.mean(mae_draws > 0),
        "bootstrap_p_two_sided": min(1.0, 2 * min(lower, upper)),
        "pooled_baseline_rmse": np.sqrt(np.mean(np.square(actual - baseline))),
        "pooled_candidate_rmse": np.sqrt(np.mean(np.square(actual - candidate))),
        "pooled_rmse_improvement": np.sqrt(np.mean(np.square(actual - baseline))) - np.sqrt(np.mean(np.square(actual - candidate))),
        "rmse_improvement_ci_lower_95": np.quantile(rmse_draws, 0.025),
        "rmse_improvement_ci_upper_95": np.quantile(rmse_draws, 0.975),
    }


def heterogeneity(group: pd.DataFrame, column: str, seed: int) -> dict[str, Any]:
    values = group["mae_improvement"].to_numpy(float)
    labels_text = group[column].astype(str)
    categories = sorted(labels_text.unique())
    lookup = {name: index for index, name in enumerate(categories)}
    labels = labels_text.map(lookup).to_numpy(int)
    origin_text = group["origin_id"].astype(str)
    origin_rows = [np.flatnonzero(origin_text.eq(origin).to_numpy()) for origin in sorted(origin_text.unique())]
    counts = np.bincount(labels, minlength=len(categories))
    means = np.bincount(labels, weights=values, minlength=len(categories)) / counts
    overall = np.mean(values)
    observed = np.sum(counts * np.square(means - overall))
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(REPETITIONS):
        permuted = labels.copy()
        for rows in origin_rows:
            permuted[rows] = rng.permutation(permuted[rows])
        permuted_means = np.bincount(permuted, weights=values, minlength=len(categories)) / counts
        statistic = np.sum(counts * np.square(permuted_means - overall))
        extreme += statistic >= observed - 1e-15
    best = int(np.argmax(means))
    worst = int(np.argmin(means))
    return {
        "pooled_n": len(group),
        "subgroup_count": len(categories),
        "heterogeneity_statistic": observed,
        "heterogeneity_p_value": (extreme + 1) / (REPETITIONS + 1),
        "best_subgroup": categories[best],
        "best_subgroup_mae_improvement": means[best],
        "worst_subgroup": categories[worst],
        "worst_subgroup_mae_improvement": means[worst],
        "subgroup_improvement_range": means[best] - means[worst],
    }


def compare(expected: pd.DataFrame, observed: pd.DataFrame, keys: list[str], columns: list[str], tolerance: float = 1e-12) -> tuple[bool, str]:
    left = expected.sort_values(keys).reset_index(drop=True)
    right = observed.sort_values(keys).reset_index(drop=True)
    if len(left) != len(right):
        return False, f"row count {len(right)} versus {len(left)}"
    if not left[keys].astype(str).equals(right[keys].astype(str)):
        return False, "key mismatch"
    maximum = 0.0
    for column in columns:
        if pd.api.types.is_numeric_dtype(left[column]) or pd.api.types.is_numeric_dtype(right[column]):
            a = pd.to_numeric(left[column], errors="coerce").to_numpy(float)
            b = pd.to_numeric(right[column], errors="coerce").to_numpy(float)
            if not np.allclose(a, b, atol=tolerance, rtol=0, equal_nan=True):
                return False, f"numeric mismatch in {column}"
            finite = np.abs(a - b)[np.isfinite(a - b)]
            if finite.size:
                maximum = max(maximum, float(finite.max()))
        elif not left[column].fillna("").astype(str).equals(right[column].fillna("").astype(str)):
            return False, f"text mismatch in {column}"
    return True, f"{len(left)} rows; maximum numeric difference {maximum:.3g}"


def run() -> None:
    required = [
        "subgroup_candidate_definitions.csv", "subgroup_pair_predictions.csv",
        "subgroup_coverage.csv", "subgroup_origin_results.csv",
        "subgroup_aggregate_results.csv", "subgroup_primary_results.csv",
        "subgroup_heterogeneity.csv", "subgroup_decision_summary.csv",
        "subgroup_build_checks.csv", "subgroup_run_summary.json", "source_manifest.csv",
    ]
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any, expected: Any, detail: str) -> None:
        checks.append({"check": name, "severity": "blocking", "passed": bool(passed), "observed": observed, "expected": expected, "detail": detail})

    missing = [name for name in required if not (OUTPUT / name).exists()]
    add("required_outputs_exist", not missing, ";".join(missing) or "all present", "all present", "Every declared subgroup artifact exists.")
    if missing:
        raise RuntimeError(f"Missing subgroup outputs: {missing}")

    build_checks = pd.read_csv(OUTPUT / "subgroup_build_checks.csv")
    build_pass = build_checks["passed"].astype(str).str.lower().eq("true")
    add("build_checks_all_pass", bool(build_pass.all()), int((~build_pass).sum()), 0, "All construction-time invariants passed.")

    source_manifest = pd.read_csv(OUTPUT / "source_manifest.csv")
    bad_hashes = []
    for row in source_manifest.itertuples(index=False):
        path = ROOT / str(row.source_file)
        if not path.exists() or sha256(path) != str(row.sha256):
            bad_hashes.append(str(row.source_file))
    add("source_manifest_hashes_match", not bad_hashes, ";".join(bad_hashes) or "all match", "all match", "Verification uses exactly the recorded source files.")

    rolling = pd.read_csv(PREDICTIONS, low_memory=False)
    policy = pd.read_csv(POLICY_PREDICTIONS, low_memory=False)
    crosswalk = pd.read_csv(CROSSWALK, low_memory=False)
    competition_names = crosswalk.drop_duplicates("canonical_competition_id").set_index("canonical_competition_id")["canonical_name"].to_dict()
    pair_parts = []
    join_keys = ["origin_id", "target_key", "canonical_performance_id"]
    identity = ["evaluation_year", "canonical_player_name", "canonical_club_name", "role_group", "canonical_competition_id"]
    for target, (feature_set, family) in CANDIDATES.items():
        candidate = rolling.loc[rolling["target_key"].eq(target) & rolling["feature_set"].eq(feature_set) & rolling["model_family"].eq(family)].copy()
        references = {
            "same_family_baseline": rolling.loc[rolling["target_key"].eq(target) & rolling["feature_set"].eq("player_history_baseline") & rolling["model_family"].eq(family)].copy(),
            "rolling_selected_baseline": policy.loc[policy["target_key"].eq(target) & policy["selection_type"].eq("rolling_selected_baseline")].copy(),
        }
        for reference_type, baseline in references.items():
            left = candidate[join_keys + identity + ["actual", "prediction", "feature_set", "model_family", "selected_hyperparameters"]].rename(columns={"prediction": "candidate_prediction", "feature_set": "candidate_feature_set", "model_family": "candidate_model_family", "selected_hyperparameters": "candidate_hyperparameters"})
            right = baseline[join_keys + ["actual", "prediction", "feature_set", "model_family", "selected_hyperparameters"]].rename(columns={"actual": "baseline_actual", "prediction": "baseline_prediction", "feature_set": "baseline_feature_set", "model_family": "baseline_model_family", "selected_hyperparameters": "baseline_hyperparameters"})
            paired = left.merge(right, on=join_keys, validate="one_to_one")
            if not np.allclose(paired["actual"], paired["baseline_actual"], atol=1e-12, rtol=0):
                raise RuntimeError("Actual values differ during independent pairing.")
            paired = paired.drop(columns="baseline_actual")
            paired["reference_type"] = reference_type
            paired["candidate_absolute_error"] = np.abs(paired["actual"] - paired["candidate_prediction"])
            paired["baseline_absolute_error"] = np.abs(paired["actual"] - paired["baseline_prediction"])
            paired["mae_improvement"] = paired["baseline_absolute_error"] - paired["candidate_absolute_error"]
            paired["role_name"] = paired["role_group"].map(ROLE_NAMES)
            paired["competition_name"] = paired["canonical_competition_id"].map(competition_names)
            pair_parts.append(paired)
    expected_pairs = pd.concat(pair_parts, ignore_index=True)
    observed_pairs = pd.read_csv(OUTPUT / "subgroup_pair_predictions.csv", low_memory=False)
    pair_columns = [column for column in observed_pairs.columns if column not in join_keys + ["reference_type"]]
    ok, detail = compare(expected_pairs, observed_pairs, ["target_key", "reference_type", "origin_id", "canonical_performance_id"], pair_columns)
    add("paired_predictions_reconstructed", ok, detail, "758 exact paired rows", "Candidate and both benchmark predictions independently map back to rolling outputs.")

    coverage_rows = []
    for target in CANDIDATES:
        target_pairs = expected_pairs.loc[expected_pairs["target_key"].eq(target) & expected_pairs["reference_type"].eq("same_family_baseline")]
        for dimension, (value_column, name_column) in DIMENSIONS.items():
            for subgroup, group in target_pairs.groupby(value_column, sort=True):
                counts = group.groupby("origin_id").size()
                coverage_rows.append({"target_key": target, "subgroup_dimension": dimension, "subgroup_value": subgroup, "subgroup_name": group[name_column].iloc[0], "all_origin_n": len(group), "mature_origin_n": group["origin_id"].isin(SUBSETS["mature_three_origins"]).sum(), "terminal_origin_n": group["origin_id"].isin(SUBSETS["terminal_two_origins"]).sum(), "origins_present": counts.size, "minimum_origin_n": counts.min(), "maximum_origin_n": counts.max()})
    expected_coverage = pd.DataFrame(coverage_rows)
    observed_coverage = pd.read_csv(OUTPUT / "subgroup_coverage.csv")
    coverage_columns = [column for column in observed_coverage.columns if column not in ["target_key", "subgroup_dimension", "subgroup_value"]]
    ok, detail = compare(expected_coverage, observed_coverage, ["target_key", "subgroup_dimension", "subgroup_value"], coverage_columns)
    add("coverage_recomputed", ok, detail, "16 exact cells", "Role and competition sample sizes are independently counted.")

    origin_rows = []
    for (target, reference), target_pairs in expected_pairs.groupby(["target_key", "reference_type"], sort=True):
        for dimension, (value_column, name_column) in DIMENSIONS.items():
            for (subgroup, origin), group in target_pairs.groupby([value_column, "origin_id"], sort=True):
                result = origin_bootstrap(group, stable_seed(SEED, "origin", target, reference, dimension, subgroup, origin))
                origin_rows.append({"target_key": target, "reference_type": reference, "subgroup_dimension": dimension, "subgroup_value": subgroup, "subgroup_name": group[name_column].iloc[0], "origin_id": origin, "evaluation_year": group["evaluation_year"].iloc[0], "sample_quality": "adequate" if len(group) >= MIN_ORIGIN_N else "very_small", **result})
    expected_origins = pd.DataFrame(origin_rows)
    observed_origins = pd.read_csv(OUTPUT / "subgroup_origin_results.csv")
    origin_keys = ["target_key", "reference_type", "subgroup_dimension", "subgroup_value", "origin_id"]
    origin_columns = [column for column in expected_origins.columns if column not in origin_keys]
    ok, detail = compare(expected_origins, observed_origins, origin_keys, origin_columns)
    add("origin_bootstraps_recomputed", ok, detail, "128 exact results", "All origin-level metrics and 5,000-draw paired bootstraps reproduce.")

    aggregate_rows = []
    for (target, reference), target_pairs in expected_pairs.groupby(["target_key", "reference_type"], sort=True):
        for subset_name, origins in SUBSETS.items():
            subset = target_pairs.loc[target_pairs["origin_id"].isin(origins)]
            for dimension, (value_column, name_column) in DIMENSIONS.items():
                for subgroup, group in subset.groupby(value_column, sort=True):
                    counts = group.groupby("origin_id").size()
                    result = aggregate_bootstrap(group, stable_seed(SEED, "aggregate", target, reference, subset_name, dimension, subgroup))
                    origin_comparisons = expected_origins.loc[expected_origins["target_key"].eq(target) & expected_origins["reference_type"].eq(reference) & expected_origins["subgroup_dimension"].eq(dimension) & expected_origins["subgroup_value"].astype(str).eq(str(subgroup)) & expected_origins["origin_id"].isin(origins)]
                    adequate = len(group) >= MIN_POOLED_N and counts.min() >= MIN_ORIGIN_N and counts.size == len(origins)
                    aggregate_rows.append({"target_key": target, "reference_type": reference, "origin_subset": subset_name, "subgroup_dimension": dimension, "subgroup_value": subgroup, "subgroup_name": group[name_column].iloc[0], "origin_count": counts.size, "minimum_origin_n": counts.min(), "maximum_origin_n": counts.max(), "sample_quality": "adequate" if adequate else "small_cell", "origins_candidate_better": (origin_comparisons["mae_improvement"] > 0).sum(), "origin_win_rate": (origin_comparisons["mae_improvement"] > 0).mean(), **result})
    expected_aggregates = pd.DataFrame(aggregate_rows)
    observed_aggregates = pd.read_csv(OUTPUT / "subgroup_aggregate_results.csv")
    aggregate_keys = ["target_key", "reference_type", "origin_subset", "subgroup_dimension", "subgroup_value"]
    aggregate_columns = [column for column in expected_aggregates.columns if column not in aggregate_keys]
    ok, detail = compare(expected_aggregates, observed_aggregates, aggregate_keys, aggregate_columns)
    add("aggregate_bootstraps_recomputed", ok, detail, "96 exact results", "All pooled metrics, sample flags, and stratified bootstraps reproduce.")

    confirmatory = observed_aggregates["reference_type"].eq("same_family_baseline") & observed_aggregates["origin_subset"].eq("mature_three_origins") & observed_aggregates["sample_quality"].eq("adequate")
    expected_q = bh(observed_aggregates.loc[confirmatory, "bootstrap_p_two_sided"])
    q_ok = np.allclose(expected_q, observed_aggregates.loc[confirmatory, "fdr_q_value"], atol=1e-12, rtol=0, equal_nan=True)
    add("subgroup_fdr_recomputed", q_ok, int(confirmatory.sum()), 11, "Benjamini-Hochberg q-values reproduce across the declared confirmatory family.")

    heterogeneity_rows = []
    for (target, reference), target_pairs in expected_pairs.groupby(["target_key", "reference_type"], sort=True):
        for subset_name, origins in SUBSETS.items():
            subset = target_pairs.loc[target_pairs["origin_id"].isin(origins)]
            for dimension, (value_column, _) in DIMENSIONS.items():
                result = heterogeneity(subset, value_column, stable_seed(SEED, "heterogeneity", target, reference, subset_name, dimension))
                heterogeneity_rows.append({"target_key": target, "reference_type": reference, "origin_subset": subset_name, "subgroup_dimension": dimension, **result})
    expected_heterogeneity = pd.DataFrame(heterogeneity_rows)
    observed_heterogeneity = pd.read_csv(OUTPUT / "subgroup_heterogeneity.csv")
    heterogeneity_keys = ["target_key", "reference_type", "origin_subset", "subgroup_dimension"]
    heterogeneity_columns = [column for column in expected_heterogeneity.columns if column not in heterogeneity_keys]
    ok, detail = compare(expected_heterogeneity, observed_heterogeneity, heterogeneity_keys, heterogeneity_columns)
    add("heterogeneity_permutations_recomputed", ok, detail, "24 exact results", "All stratified 5,000-permutation heterogeneity tests reproduce.")
    heterogeneity_primary = observed_heterogeneity["reference_type"].eq("same_family_baseline") & observed_heterogeneity["origin_subset"].eq("mature_three_origins")
    expected_heterogeneity_q = bh(observed_heterogeneity.loc[heterogeneity_primary, "heterogeneity_p_value"])
    heterogeneity_q_ok = np.allclose(expected_heterogeneity_q, observed_heterogeneity.loc[heterogeneity_primary, "heterogeneity_fdr_q_value"], atol=1e-12, rtol=0)
    add("heterogeneity_fdr_recomputed", heterogeneity_q_ok, int(heterogeneity_primary.sum()), 4, "Heterogeneity q-values reproduce across the four primary tests.")

    primary = pd.read_csv(OUTPUT / "subgroup_primary_results.csv")
    expected_primary = observed_aggregates.loc[observed_aggregates["reference_type"].eq("same_family_baseline") & observed_aggregates["origin_subset"].eq("mature_three_origins")]
    primary_keys = ["target_key", "subgroup_dimension", "subgroup_value"]
    common_columns = [column for column in primary.columns if column in expected_primary.columns and column not in primary_keys]
    ok, detail = compare(expected_primary, primary, primary_keys, common_columns)
    add("primary_extract_reconciles", ok, detail, "16 exact rows", "The primary table is an exact mature same-family extract.")

    verification = pd.DataFrame(checks)
    verification.to_csv(OUTPUT / "independent_verification.csv", index=False)
    failures = int((~verification["passed"]).sum())
    summary = {"status": "pass" if failures == 0 else "fail", "checks": len(verification), "blocking_checks": len(verification), "blocking_failures": failures, "verified_pair_rows": len(observed_pairs), "verified_origin_results": len(observed_origins), "verified_aggregate_results": len(observed_aggregates), "verified_heterogeneity_results": len(observed_heterogeneity), "bootstrap_repetitions": REPETITIONS, "permutation_repetitions": REPETITIONS}
    (OUTPUT / "independent_verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_paths = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name != "output_manifest.csv")
    pd.DataFrame([{"output_file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size} for path in output_paths]).to_csv(OUTPUT / "output_manifest.csv", index=False)
    print(json.dumps(summary, indent=2))
    if failures:
        raise RuntimeError(f"Independent subgroup verification failed {failures} checks.")


if __name__ == "__main__":
    run()

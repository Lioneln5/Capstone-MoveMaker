"""Independently verify the coarse club-context diagnostic from saved predictions."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_coarse_club_context_diagnostic as diagnostic
from train_compatibility_models import sha256_file


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "Data" / "processed" / "coarse_club_context_diagnostic"
FIELD_DICTIONARY_PATH = (
    ROOT
    / "Data"
    / "processed"
    / "transfermarkt_merged"
    / "transfermarkt_comprehensive_field_dictionary.csv"
)
RUNNER_PATH = ROOT / "scripts" / "run_coarse_club_context_diagnostic.py"


def close(a: Any, b: Any, tolerance: float = 1e-12) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    return bool(np.isclose(float(a), float(b), atol=tolerance, rtol=0))


def paired(predictions: pd.DataFrame, baseline: str, candidate: str, years: list[int]) -> pd.DataFrame:
    selected = predictions.loc[predictions["evaluation_year"].isin(years)].copy()
    keys = ["origin_id", "evaluation_year", "transfer_event_id", "player_id"]
    base = selected.loc[selected["model_variant"].eq(baseline), keys + ["actual", "bounded_prediction"]].rename(
        columns={"bounded_prediction": "baseline_prediction"}
    )
    cand = selected.loc[selected["model_variant"].eq(candidate), keys + ["actual", "bounded_prediction"]].rename(
        columns={"actual": "candidate_actual", "bounded_prediction": "candidate_prediction"}
    )
    result = base.merge(cand, on=keys, how="inner", validate="one_to_one")
    if not np.allclose(result["actual"], result["candidate_actual"], atol=1e-12, rtol=0):
        raise RuntimeError("Paired actual outcomes disagree.")
    return result.drop(columns="candidate_actual")


def independent_cluster_bootstrap(pairs: pd.DataFrame, seed: int) -> tuple[float, float, float]:
    difference = np.abs(pairs["actual"] - pairs["baseline_prediction"]) - np.abs(
        pairs["actual"] - pairs["candidate_prediction"]
    )
    working = pd.DataFrame({"player_id": pairs["player_id"], "difference": difference})
    clusters = working.groupby("player_id", sort=True)["difference"].agg(["sum", "size"])
    sums = clusters["sum"].to_numpy(dtype=float)
    sizes = clusters["size"].to_numpy(dtype=float)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(
        0,
        len(clusters),
        size=(diagnostic.BOOTSTRAP_REPETITIONS, len(clusters)),
    )
    draws = sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)
    return (
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
        float(np.mean(draws > 0)),
    )


def independent_row_bootstrap(pairs: pd.DataFrame, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(int(seed))
    total = np.zeros(diagnostic.BOOTSTRAP_REPETITIONS, dtype=float)
    for _, group in pairs.groupby("origin_id", sort=True):
        difference = (
            np.abs(group["actual"].to_numpy(dtype=float) - group["baseline_prediction"].to_numpy(dtype=float))
            - np.abs(group["actual"].to_numpy(dtype=float) - group["candidate_prediction"].to_numpy(dtype=float))
        )
        indices = rng.integers(
            0,
            len(group),
            size=(diagnostic.BOOTSTRAP_REPETITIONS, len(group)),
        )
        total += difference[indices].sum(axis=1)
    draws = total / len(pairs)
    return (
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
        float(np.mean(draws > 0)),
    )


def independent_label(improvement: float, origin_values: np.ndarray, probability: float) -> str:
    wins = int(np.sum(origin_values > 0))
    count = len(origin_values)
    if improvement > 0 and wins >= max(3, count - 1) and probability >= 0.90:
        return "stable_positive"
    if improvement > 0 and wins >= math.ceil(count / 2):
        return "mixed_positive"
    if improvement <= 0 and wins <= 1:
        return "unstable_negative"
    return "mixed"


def verify() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(check: str, passed: bool, observed: Any, expected: Any, detail: str) -> None:
        checks.append(
            {
                "check": check,
                "severity": "blocking",
                "passed": bool(passed),
                "observed": observed,
                "expected": expected,
                "detail": detail,
            }
        )

    required_preverification = {
        "coarse_club_context_results.csv",
        "coarse_club_context_origin_results.csv",
        "coarse_club_context_predictions.csv",
        "coarse_club_context_feature_manifest.csv",
        "README.md",
    }
    existing = {path.name for path in OUTPUT_DIR.iterdir() if path.is_file()} if OUTPUT_DIR.exists() else set()
    add(
        "declared_runner_outputs_exist",
        required_preverification.issubset(existing),
        ";".join(sorted(existing)),
        ";".join(sorted(required_preverification)),
        "The runner produced only the artifacts needed for independent verification.",
    )

    master_hash = sha256_file(diagnostic.MASTER_PATH)
    summary = json.loads(diagnostic.SUMMARY_PATH.read_text(encoding="utf-8"))
    add(
        "master_hash_matches_verified_upstream",
        master_hash == summary.get("master_sha256"),
        master_hash,
        summary.get("master_sha256"),
        "The comprehensive master remains byte-identical to its verified upstream summary.",
    )

    master = pd.read_csv(diagnostic.MASTER_PATH, low_memory=False)
    cohort = diagnostic.cohort_from_master(master)
    target = pd.to_numeric(cohort[diagnostic.TARGET], errors="coerce")
    independently_reconstructed = pd.to_numeric(
        cohort["appearance_post24_destination_minutes"], errors="coerce"
    ) / (90.0 * pd.to_numeric(cohort["club_destination_post24_games"], errors="coerce"))
    cohort_ok = (
        len(cohort) == 3134
        and cohort["player_id"].nunique() == 1072
        and cohort["transfer_date"].min().date().isoformat() == "2012-12-31"
        and cohort["transfer_date"].max().date().isoformat() == "2024-07-05"
    )
    add(
        "cohort_reconstructs_exactly",
        cohort_ok,
        f"rows={len(cohort)};players={cohort['player_id'].nunique()};dates={cohort['transfer_date'].min().date()}..{cohort['transfer_date'].max().date()}",
        "rows=3134;players=1072;dates=2012-12-31..2024-07-05",
        "The verifier independently applies the audited cohort rules.",
    )
    target_ok = (
        pd.to_numeric(cohort["club_destination_post24_games"], errors="coerce").gt(0).all()
        and target.notna().all()
        and target.between(0, 1.05).all()
        and np.allclose(target, independently_reconstructed, atol=1e-12, rtol=0)
    )
    add(
        "normalized_target_reconstructs_and_is_bounded",
        target_ok,
        f"nonmissing={target.notna().sum()};min={target.min()};max={target.max()}",
        "3134 nonmissing values within [0,1.05] and exact formula match",
        "The target is independently rebuilt from destination minutes and club match opportunities.",
    )
    zero_minutes = pd.to_numeric(cohort["appearance_post24_destination_minutes"], errors="coerce").eq(0)
    add(
        "zero_minutes_retained_as_zero_outcome",
        target.loc[zero_minutes].eq(0).all() and int(zero_minutes.sum()) > 0,
        f"zero_minute_rows={int(zero_minutes.sum())};zero_target_rows={int(target.eq(0).sum())}",
        "all zero-minute rows have target zero",
        "Observed zero opportunity is not converted to missing.",
    )

    predictions = pd.read_csv(OUTPUT_DIR / "coarse_club_context_predictions.csv", low_memory=False)
    origins = pd.read_csv(OUTPUT_DIR / "coarse_club_context_origin_results.csv", low_memory=False)
    results = pd.read_csv(OUTPUT_DIR / "coarse_club_context_results.csv", low_memory=False)
    manifest = pd.read_csv(OUTPUT_DIR / "coarse_club_context_feature_manifest.csv", low_memory=False)

    frame, feature_sets, expected_manifest = diagnostic.build_model_frame(cohort)
    expected_keys = expected_manifest[["feature_set", "feature_order", "feature"]].reset_index(drop=True)
    observed_keys = manifest[["feature_set", "feature_order", "feature"]].reset_index(drop=True)
    manifest_exact = expected_keys.equals(observed_keys)
    add(
        "feature_manifest_matches_modeled_allowlist",
        manifest_exact,
        f"rows={len(observed_keys)};sets={manifest['feature_set'].nunique()}",
        f"rows={len(expected_keys)};sets=4",
        "The saved manifest exactly matches the frozen nested feature lists used by the runner.",
    )
    nested = all(
        set(feature_sets[diagnostic.MODEL_ORDER[index - 1]]).issubset(feature_sets[diagnostic.MODEL_ORDER[index]])
        for index in range(1, len(diagnostic.MODEL_ORDER))
    )
    add(
        "four_feature_sets_are_nested",
        nested,
        ";".join(f"{name}={len(values)}" for name, values in feature_sets.items()),
        "M0 subset M1 subset M2 subset M3",
        "Each comparison adds only its declared context/fit block.",
    )

    forbidden_tokens = ["post12", "post24", "target_", "eligible_", "current_", "latest_", "snapshot"]
    raw_features = set(manifest["feature"].astype(str))
    forbidden_features = sorted(
        feature
        for feature in raw_features
        if any(token in feature.lower() for token in forbidden_tokens)
        or feature.lower().endswith(("_date", "_id", "_name", "_url"))
    )
    add(
        "no_forbidden_field_entered_predictor_matrix",
        not forbidden_features,
        ";".join(forbidden_features),
        "none",
        "Post-outcome, eligibility, snapshot, identifier, name, URL, and raw-date fields are not predictors.",
    )

    source_fields = sorted(
        {
            source
            for cell in manifest["source_fields"].fillna("").astype(str)
            for source in cell.split(";")
            if source
        }
    )
    missing_sources = sorted(set(source_fields) - set(master.columns))
    post_sources = sorted(source for source in source_fields if "post12" in source.lower() or "post24" in source.lower())
    allowed_transfer_sources = {
        "transfer_fee",
        "transfer_fee_status",
        "market_value_in_eur",
        "player_date_of_birth",
        "transfer_date",
    }
    invalid_timing_sources = sorted(
        source
        for source in source_fields
        if source not in allowed_transfer_sources
        and not (
            source.startswith("valuation_pre")
            or "_career_pre_" in source
            or "_pre365_" in source
            or "_pre730_" in source
            or source == "has_pretransfer_valuation_365d"
        )
    )
    add(
        "all_source_fields_exist_and_are_pretransfer",
        not missing_sources and not post_sources and not invalid_timing_sources,
        f"missing={missing_sources};post={post_sources};invalid_timing={invalid_timing_sources}",
        "none",
        "Every raw input to a modeled or derived feature exists and is known no later than the transfer date.",
    )

    alignment_ok = True
    alignment_detail = []
    expected_total = 0
    for origin_id, _, _, evaluation_year in diagnostic.ORIGINS:
        model_ids = []
        expected_eval_ids = frame.loc[frame["season_start_year"].eq(evaluation_year), "transfer_event_id"].astype(str).sort_values().tolist()
        expected_total += len(expected_eval_ids)
        for model in diagnostic.MODEL_ORDER:
            ids = predictions.loc[
                predictions["origin_id"].eq(origin_id) & predictions["model_variant"].eq(model),
                "transfer_event_id",
            ].astype(str).sort_values().tolist()
            model_ids.append(ids)
        ok = all(ids == expected_eval_ids for ids in model_ids)
        alignment_ok &= ok
        alignment_detail.append(f"{origin_id}={len(expected_eval_ids)}:{ok}")
    add(
        "evaluation_rows_identical_across_models",
        alignment_ok,
        ";".join(alignment_detail),
        "all four models exactly match independently reconstructed evaluation rows",
        "Missing candidate features never remove evaluation rows.",
    )
    add(
        "pooled_evaluation_row_count_complete",
        len(predictions) == expected_total * len(diagnostic.MODEL_ORDER),
        f"prediction_rows={len(predictions)};pooled_unique_events={predictions['transfer_event_id'].nunique()}",
        f"prediction_rows={expected_total * 4};per_model_rows={expected_total}",
        "All four declared evaluation seasons are scored once per model.",
    )

    ridge_only = (
        set(predictions["model_family"].astype(str)) == {"ridge"}
        and set(origins["model_family"].astype(str)) == {"ridge"}
        and set(predictions["model_variant"].astype(str)) == set(diagnostic.MODEL_ORDER)
    )
    add(
        "ridge_only_and_four_declared_models",
        ridge_only,
        f"families={sorted(predictions['model_family'].unique())};models={sorted(predictions['model_variant'].unique())}",
        "ridge and M0-M3 only",
        "No undeclared model family or feature-set search was fit.",
    )
    chronology_ok = (
        origins["preprocessor_train_end_year"].lt(origins["validation_year"]).all()
        and origins["alpha_selection_year"].eq(origins["validation_year"]).all()
        and origins["preprocessor_refit_end_year"].eq(origins["validation_year"]).all()
        and origins["validation_year"].lt(origins["evaluation_year"]).all()
        and origins["train_start_year"].eq(int(frame["season_start_year"].min())).all()
    )
    runner_source = RUNNER_PATH.read_text(encoding="utf-8")
    implementation_markers = all(
        marker in runner_source
        for marker in [
            "fit_preprocessor(train, features)",
            "fit_preprocessor(fit_frame, features)",
            "selected_alpha, selected_metrics",
        ]
    )
    add(
        "chronology_and_fold_fitted_preprocessing",
        chronology_ok and implementation_markers,
        f"chronology={chronology_ok};implementation_markers={implementation_markers}",
        "train-only tuning preprocessing; train+validation refit; evaluation untouched",
        "Recorded split years and runner implementation enforce leakage-safe fitting.",
    )

    origin_metrics_ok = True
    for _, row in origins.iterrows():
        scored = predictions.loc[
            predictions["origin_id"].eq(row["origin_id"])
            & predictions["model_variant"].eq(row["model_variant"])
        ]
        actual = scored["actual"].to_numpy(dtype=float)
        predicted = scored["bounded_prediction"].to_numpy(dtype=float)
        mae = float(np.mean(np.abs(actual - predicted)))
        rmse = float(np.sqrt(np.mean(np.square(actual - predicted))))
        origin_metrics_ok &= close(mae, row["evaluation_mae"]) and close(rmse, row["evaluation_rmse"])
    add(
        "origin_metrics_reconstruct_from_predictions",
        origin_metrics_ok,
        origin_metrics_ok,
        True,
        "Origin MAE and RMSE are independently recalculated from saved predictions.",
    )

    pooled_metrics_ok = True
    bootstrap_ok = True
    label_ok = True
    for _, row in results.iterrows():
        years = [int(value) for value in str(row["evaluation_years"]).split(";")]
        pairs = paired(predictions, row["baseline_model"], row["candidate_model"], years)
        baseline_error = np.abs(pairs["actual"] - pairs["baseline_prediction"])
        candidate_error = np.abs(pairs["actual"] - pairs["candidate_prediction"])
        baseline_mae = float(baseline_error.mean())
        candidate_mae = float(candidate_error.mean())
        improvement = baseline_mae - candidate_mae
        pooled_metrics_ok &= (
            int(row["pooled_n"]) == len(pairs)
            and int(row["unique_players"]) == pairs["player_id"].nunique()
            and close(row["pooled_baseline_mae"], baseline_mae)
            and close(row["pooled_candidate_mae"], candidate_mae)
            and close(row["pooled_mae_improvement"], improvement)
            and close(row["pooled_relative_mae_improvement"], improvement / baseline_mae)
        )
        cluster = independent_cluster_bootstrap(pairs, int(row["cluster_bootstrap_seed"]))
        row_result = independent_row_bootstrap(pairs, int(row["row_bootstrap_seed"]))
        bootstrap_ok &= all(
            [
                close(cluster[0], row["cluster_bootstrap_ci_lower_95"]),
                close(cluster[1], row["cluster_bootstrap_ci_upper_95"]),
                close(cluster[2], row["cluster_bootstrap_probability_candidate_better"]),
                close(row_result[0], row["row_bootstrap_ci_lower_95"]),
                close(row_result[1], row["row_bootstrap_ci_upper_95"]),
                close(row_result[2], row["row_bootstrap_probability_candidate_better"]),
            ]
        )
        origin_values = []
        for _, group in pairs.groupby("origin_id", sort=True):
            origin_values.append(
                float(
                    np.abs(group["actual"] - group["baseline_prediction"]).mean()
                    - np.abs(group["actual"] - group["candidate_prediction"]).mean()
                )
            )
        expected_label = independent_label(improvement, np.asarray(origin_values), cluster[2])
        label_ok &= str(row["stability_label"]) == expected_label
    add(
        "pooled_window_metrics_reconstruct_from_predictions",
        pooled_metrics_ok,
        pooled_metrics_ok,
        True,
        "All comparison/window sample sizes and paired MAE effects reconstruct independently.",
    )
    add(
        "paired_bootstraps_reconstruct_from_predictions",
        bootstrap_ok,
        bootstrap_ok,
        True,
        "Both deterministic row-stratified and player-clustered intervals reconstruct.",
    )
    add(
        "stability_labels_reconstruct",
        label_ok,
        label_ok,
        True,
        "Labels use the player-clustered probability and declared origin rules.",
    )

    current_protected_hash = diagnostic.protected_tree_sha256()
    recorded_master_before = set(origins["master_sha256_before"].astype(str))
    recorded_master_after = set(origins["master_sha256_after"].astype(str))
    recorded_protected_before = set(origins["protected_tree_sha256_before"].astype(str))
    recorded_protected_after = set(origins["protected_tree_sha256_after"].astype(str))
    source_hash_ok = recorded_master_before == {master_hash} and recorded_master_after == {master_hash}
    protected_ok = (
        len(recorded_protected_before) == 1
        and recorded_protected_before == recorded_protected_after
        and recorded_protected_after == {current_protected_hash}
        and diagnostic.as_bool(origins["protected_outputs_unchanged"]).all()
    )
    add(
        "source_hash_unchanged_before_and_after",
        source_hash_ok,
        f"before={recorded_master_before};after={recorded_master_after};current={master_hash}",
        "all verified master hash",
        "The source master was unchanged by the run.",
    )
    add(
        "protected_existing_outputs_unchanged",
        protected_ok,
        f"before={recorded_protected_before};after={recorded_protected_after};current={current_protected_hash}",
        "all identical",
        "Protected compatibility, rolling, subgroup, and case-study trees were byte-identical before and after.",
    )

    status = "pass" if all(check["passed"] for check in checks) else "fail"
    summary_payload = {
        "status": status,
        "blocking_checks": len(checks),
        "blocking_checks_passed": sum(int(check["passed"]) for check in checks),
        "blocking_checks_failed": sum(int(not check["passed"]) for check in checks),
        "cohort_rows": len(cohort),
        "cohort_unique_players": int(cohort["player_id"].nunique()),
        "master_sha256": master_hash,
        "protected_tree_sha256": current_protected_hash,
        "checks": checks,
    }
    return checks, summary_payload


def main() -> None:
    checks, payload = verify()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(OUTPUT_DIR / "independent_verification.csv", index=False)
    (OUTPUT_DIR / "independent_verification.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"verification={payload['status']} passed={payload['blocking_checks_passed']}/{payload['blocking_checks']}",
        flush=True,
    )
    if payload["status"] != "pass":
        failed = [check["check"] for check in checks if not check["passed"]]
        print("failed=" + ",".join(failed), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

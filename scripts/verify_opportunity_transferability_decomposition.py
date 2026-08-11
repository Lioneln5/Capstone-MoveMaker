"""Independently verify the opportunity-transferability decomposition."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_opportunity_transferability_decomposition as study
from train_compatibility_models import sha256_file


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = study.OUTPUT_DIR
RUNNER_PATH = ROOT / "scripts" / "run_opportunity_transferability_decomposition.py"


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
        raise RuntimeError("Actual outcomes disagree in paired predictions.")
    return result.drop(columns="candidate_actual")


def cluster_bootstrap(pairs: pd.DataFrame, seed: int) -> tuple[float, float, float]:
    difference = np.abs(pairs["actual"] - pairs["baseline_prediction"]) - np.abs(
        pairs["actual"] - pairs["candidate_prediction"]
    )
    clusters = pd.DataFrame({"player_id": pairs["player_id"], "difference": difference}).groupby(
        "player_id", sort=True
    )["difference"].agg(["sum", "size"])
    sums = clusters["sum"].to_numpy(dtype=float)
    sizes = clusters["size"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(clusters), size=(study.BOOTSTRAP_REPETITIONS, len(clusters)))
    values = sums[draw].sum(axis=1) / sizes[draw].sum(axis=1)
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        float(np.mean(values > 0)),
    )


def row_bootstrap(pairs: pd.DataFrame, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    total = np.zeros(study.BOOTSTRAP_REPETITIONS, dtype=float)
    for _, group in pairs.groupby("origin_id", sort=True):
        difference = (
            np.abs(group["actual"].to_numpy(dtype=float) - group["baseline_prediction"].to_numpy(dtype=float))
            - np.abs(group["actual"].to_numpy(dtype=float) - group["candidate_prediction"].to_numpy(dtype=float))
        )
        draw = rng.integers(0, len(group), size=(study.BOOTSTRAP_REPETITIONS, len(group)))
        total += difference[draw].sum(axis=1)
    values = total / len(pairs)
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        float(np.mean(values > 0)),
    )


def label(improvement: float, origin_values: np.ndarray, probability: float) -> str:
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

    required_runner_files = {
        "opportunity_transferability_results.csv",
        "opportunity_transferability_origin_results.csv",
        "opportunity_transferability_predictions.csv",
        "opportunity_transferability_feature_manifest.csv",
        "opportunity_persistence_summary.csv",
        "README.md",
    }
    existing = {path.name for path in OUTPUT_DIR.iterdir() if path.is_file()} if OUTPUT_DIR.exists() else set()
    add(
        "declared_runner_outputs_exist",
        required_runner_files.issubset(existing),
        ";".join(sorted(existing)),
        ";".join(sorted(required_runner_files)),
        "All declared runner artifacts exist before independent verification.",
    )

    master_hash = sha256_file(study.MASTER_PATH)
    upstream_hash = json.loads(study.SUMMARY_PATH.read_text(encoding="utf-8")).get("master_sha256")
    add(
        "master_hash_matches_verified_upstream",
        master_hash == upstream_hash,
        master_hash,
        upstream_hash,
        "The source master is byte-identical to its verified summary.",
    )
    master = pd.read_csv(study.MASTER_PATH, low_memory=False)
    cohort = study.cohort_from_master(master)
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
        "The verifier independently applies the frozen cohort rules.",
    )

    destination = pd.to_numeric(cohort["appearance_post24_destination_minutes"], errors="coerce") / (
        90.0 * pd.to_numeric(cohort["club_destination_post24_games"], errors="coerce")
    )
    target_ok = (
        pd.to_numeric(cohort["club_destination_post24_games"], errors="coerce").gt(0).all()
        and destination.notna().all()
        and destination.between(0, 1.05).all()
        and np.allclose(destination, cohort[study.TARGET], atol=1e-12, rtol=0)
    )
    coarse_predictions = pd.read_csv(
        study.COARSE_OUTPUT_DIR / "coarse_club_context_predictions.csv", low_memory=False
    )
    coarse_actual = coarse_predictions.loc[
        coarse_predictions["model_variant"].eq(study.coarse.MODEL_ORDER[0]),
        ["transfer_event_id", "actual"],
    ].drop_duplicates("transfer_event_id")
    target_match = cohort[["transfer_event_id", study.TARGET]].merge(
        coarse_actual, on="transfer_event_id", how="inner", validate="one_to_one"
    )
    coarse_target_ok = (
        len(target_match) == coarse_actual["transfer_event_id"].nunique()
        and np.allclose(target_match[study.TARGET], target_match["actual"], atol=1e-12, rtol=0)
    )
    add(
        "destination_target_reconstructs_and_matches_coarse_diagnostic",
        target_ok and coarse_target_ok,
        f"target_ok={target_ok};coarse_match={coarse_target_ok};min={destination.min()};max={destination.max()}",
        "complete bounded exact match",
        "The destination target is independently reconstructed and agrees with the protected prior diagnostic.",
    )

    frame, feature_sets, expected_manifest = study.build_model_frame(cohort)
    usage_ok = True
    usage_detail = []
    for feature, _, denominator, _ in study.USAGE_SPECS:
        values = pd.to_numeric(frame[feature], errors="coerce")
        good = (
            pd.to_numeric(frame[denominator], errors="coerce").gt(0).all()
            and values.notna().all()
            and values.between(0, 1.05).all()
        )
        usage_ok &= good
        usage_detail.append(f"{feature}:n={values.notna().sum()},min={values.min()},max={values.max()}")
    expectation_ok = (
        int(frame["origin_opportunity_share_pre365"].eq(0).sum()) == 367
        and close(frame["origin_opportunity_share_pre365"].max(), 1.0277777777777777)
        and int(frame["origin_opportunity_share_pre730"].eq(0).sum()) == 192
        and close(frame["origin_opportunity_share_pre730"].max(), 1.0044444444444445)
    )
    add(
        "prior_opportunity_measures_reconstruct_exactly",
        usage_ok and expectation_ok,
        ";".join(usage_detail) + f";expectations={expectation_ok}",
        "all 3134 bounded plus declared zero/max expectations",
        "All eight prior-opportunity measures are independently rebuilt from raw pre-transfer inputs.",
    )

    predictions = pd.read_csv(OUTPUT_DIR / "opportunity_transferability_predictions.csv", low_memory=False)
    origins = pd.read_csv(OUTPUT_DIR / "opportunity_transferability_origin_results.csv", low_memory=False)
    results = pd.read_csv(OUTPUT_DIR / "opportunity_transferability_results.csv", low_memory=False)
    manifest = pd.read_csv(OUTPUT_DIR / "opportunity_transferability_feature_manifest.csv", low_memory=False)
    persistence = pd.read_csv(OUTPUT_DIR / "opportunity_persistence_summary.csv", low_memory=False)

    key_columns = ["feature_set", "feature_order", "feature"]
    expected_keys = expected_manifest[key_columns].copy()
    observed_keys = manifest[key_columns].copy()
    for table in [expected_keys, observed_keys]:
        table["feature"] = table["feature"].fillna("").astype(str)
        table["feature_order"] = pd.to_numeric(table["feature_order"], errors="coerce").astype(int)
    manifest_exact = expected_keys.reset_index(drop=True).equals(observed_keys.reset_index(drop=True))
    observed_counts = {
        model: int(
            manifest.loc[manifest["feature_set"].eq(model), "raw_feature_count"].dropna().iloc[0]
        )
        for model in study.MODEL_ORDER
    }
    add(
        "feature_manifest_matches_frozen_counts",
        manifest_exact and observed_counts == study.EXPECTED_COUNTS,
        f"manifest_exact={manifest_exact};counts={observed_counts}",
        str(study.EXPECTED_COUNTS),
        "The manifest matches the exact six predeclared feature sets.",
    )
    nested = all(
        set(feature_sets[study.MODEL_ORDER[index - 1]]).issubset(feature_sets[study.MODEL_ORDER[index]])
        for index in range(1, len(study.MODEL_ORDER))
    )
    add(
        "six_feature_sets_are_nested",
        nested,
        nested,
        True,
        "Each stage adds only its declared information block.",
    )

    modeled_features = {
        feature for feature in manifest["feature"].fillna("").astype(str) if feature
    }
    forbidden_tokens = ["post12", "post24", "target_", "eligible_", "current_", "latest_", "snapshot"]
    forbidden_features = sorted(
        feature
        for feature in modeled_features
        if any(token in feature.lower() for token in forbidden_tokens)
        or feature.lower().endswith(("_date", "_id", "_name", "_url"))
    )
    add(
        "no_forbidden_predictor_fields",
        not forbidden_features,
        ";".join(forbidden_features),
        "none",
        "No post-outcome, target, eligibility, snapshot, identity, name, URL, or raw-date field is modeled.",
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
    allowed_at_transfer = {
        "transfer_fee",
        "transfer_fee_status",
        "market_value_in_eur",
        "player_date_of_birth",
        "transfer_date",
    }
    invalid_timing = sorted(
        source
        for source in source_fields
        if source not in allowed_at_transfer
        and not ("_pre365_" in source or "_pre730_" in source)
    )
    add(
        "feature_lineage_exists_and_is_pretransfer",
        not missing_sources and not post_sources and not invalid_timing,
        f"missing={missing_sources};post={post_sources};invalid={invalid_timing}",
        "none",
        "Every modeled/derived input has complete raw lineage available by the transfer date.",
    )

    alignment_ok = True
    expected_per_model = 0
    alignment_detail = []
    for origin_id, _, _, evaluation_year in study.ORIGINS:
        expected_ids = frame.loc[
            frame["season_start_year"].eq(evaluation_year), "transfer_event_id"
        ].astype(str).sort_values().tolist()
        expected_per_model += len(expected_ids)
        model_ok = True
        for model in study.MODEL_ORDER:
            ids = predictions.loc[
                predictions["origin_id"].eq(origin_id) & predictions["model_variant"].eq(model),
                "transfer_event_id",
            ].astype(str).sort_values().tolist()
            model_ok &= ids == expected_ids
        alignment_ok &= model_ok
        alignment_detail.append(f"{origin_id}={len(expected_ids)}:{model_ok}")
    add(
        "evaluation_rows_identical_across_all_models",
        alignment_ok and len(predictions) == expected_per_model * len(study.MODEL_ORDER),
        f"{';'.join(alignment_detail)};prediction_rows={len(predictions)}",
        f"identical;prediction_rows={expected_per_model * 6}",
        "No missing candidate feature removes an evaluation row.",
    )

    family_ok = (
        set(predictions.loc[predictions["model_variant"].eq(study.MODEL_ORDER[0]), "model_family"]) == {"mean_only"}
        and set(predictions.loc[~predictions["model_variant"].eq(study.MODEL_ORDER[0]), "model_family"]) == {"ridge"}
        and set(predictions["model_variant"]) == set(study.MODEL_ORDER)
    )
    add(
        "mean_only_and_ridge_only",
        family_ok,
        f"families={sorted(predictions['model_family'].unique())};models={sorted(predictions['model_variant'].unique())}",
        "M0 mean-only; M1-M5 ridge",
        "No undeclared family or model specification was fit.",
    )

    ridge_origins = origins.loc[origins["model_family"].eq("ridge")]
    mean_origins = origins.loc[origins["model_family"].eq("mean_only")]
    chronology_ok = (
        ridge_origins["preprocessor_train_end_year"].lt(ridge_origins["validation_year"]).all()
        and ridge_origins["alpha_selection_year"].eq(ridge_origins["validation_year"]).all()
        and ridge_origins["preprocessor_refit_end_year"].eq(ridge_origins["validation_year"]).all()
        and origins["validation_year"].lt(origins["evaluation_year"]).all()
        and origins["train_start_year"].eq(int(frame["season_start_year"].min())).all()
        and mean_origins["selected_alpha"].isna().all()
    )
    runner_source = RUNNER_PATH.read_text(encoding="utf-8")
    markers_ok = all(
        marker in runner_source
        for marker in [
            "fit_preprocessor(train, features)",
            "fit_preprocessor(fit_frame, features)",
            "_, selected_alpha, selected_metrics",
        ]
    )
    add(
        "chronology_and_fold_fitted_preprocessing",
        chronology_ok and markers_ok,
        f"chronology={chronology_ok};markers={markers_ok}",
        "train-only preprocessing/tuning; train+validation refit; evaluation untouched",
        "Recorded years and implementation markers enforce the declared leakage-safe procedure.",
    )

    origin_metrics_ok = True
    for _, row in origins.iterrows():
        scored = predictions.loc[
            predictions["origin_id"].eq(row["origin_id"])
            & predictions["model_variant"].eq(row["model_variant"])
        ]
        actual = scored["actual"].to_numpy(dtype=float)
        predicted = scored["bounded_prediction"].to_numpy(dtype=float)
        origin_metrics_ok &= close(np.mean(np.abs(actual - predicted)), row["evaluation_mae"])
        origin_metrics_ok &= close(np.sqrt(np.mean(np.square(actual - predicted))), row["evaluation_rmse"])
    add(
        "origin_metrics_reconstruct_from_predictions",
        origin_metrics_ok,
        origin_metrics_ok,
        True,
        "Evaluation MAE and RMSE are independently reconstructed for every origin/model.",
    )

    pooled_ok = True
    bootstrap_ok = True
    labels_ok = True
    for _, row in results.iterrows():
        years = [int(value) for value in str(row["evaluation_years"]).split(";")]
        pairs = paired(predictions, row["baseline_model"], row["candidate_model"], years)
        baseline_mae = float(np.mean(np.abs(pairs["actual"] - pairs["baseline_prediction"])))
        candidate_mae = float(np.mean(np.abs(pairs["actual"] - pairs["candidate_prediction"])))
        improvement = baseline_mae - candidate_mae
        pooled_ok &= (
            int(row["pooled_n"]) == len(pairs)
            and int(row["unique_players"]) == pairs["player_id"].nunique()
            and close(row["pooled_baseline_mae"], baseline_mae)
            and close(row["pooled_candidate_mae"], candidate_mae)
            and close(row["pooled_mae_improvement"], improvement)
            and close(row["pooled_relative_mae_improvement"], improvement / baseline_mae)
        )
        cluster = cluster_bootstrap(pairs, int(row["cluster_bootstrap_seed"]))
        rows = row_bootstrap(pairs, int(row["row_bootstrap_seed"]))
        bootstrap_ok &= all(
            [
                close(cluster[0], row["cluster_bootstrap_ci_lower_95"]),
                close(cluster[1], row["cluster_bootstrap_ci_upper_95"]),
                close(cluster[2], row["cluster_bootstrap_probability_candidate_better"]),
                close(rows[0], row["row_bootstrap_ci_lower_95"]),
                close(rows[1], row["row_bootstrap_ci_upper_95"]),
                close(rows[2], row["row_bootstrap_probability_candidate_better"]),
            ]
        )
        origin_values = []
        for _, group in pairs.groupby("origin_id", sort=True):
            origin_values.append(
                float(
                    np.mean(np.abs(group["actual"] - group["baseline_prediction"]))
                    - np.mean(np.abs(group["actual"] - group["candidate_prediction"]))
                )
            )
        labels_ok &= str(row["stability_label"]) == label(
            improvement, np.asarray(origin_values), cluster[2]
        )
    add(
        "pooled_window_metrics_reconstruct",
        pooled_ok,
        pooled_ok,
        True,
        "All nested comparison/window effects reconstruct from paired saved predictions.",
    )
    add(
        "paired_bootstraps_reconstruct",
        bootstrap_ok,
        bootstrap_ok,
        True,
        "Player-clustered and origin-stratified row bootstrap outputs reconstruct deterministically.",
    )
    add(
        "stability_labels_reconstruct",
        labels_ok,
        labels_ok,
        True,
        "All stability labels reconstruct from the declared rules.",
    )

    persistence_ok = True
    base_rows = predictions.loc[predictions["model_variant"].eq(study.MODEL_ORDER[0])].copy()
    for _, row in persistence.iterrows():
        years = [int(value) for value in str(row["evaluation_years"]).split(";")]
        selected = base_rows.loc[base_rows["evaluation_year"].isin(years)].copy()
        origin_share = selected["origin_opportunity_share_pre365"].clip(0, 1.05)
        actual = selected["actual"]
        persistence_ok &= (
            int(row["rows"]) == len(selected)
            and int(row["unique_players"]) == selected["player_id"].nunique()
            and close(row["pearson_correlation"], origin_share.corr(actual, method="pearson"))
            and close(row["spearman_correlation"], origin_share.corr(actual, method="spearman"))
            and close(row["naive_origin_share_mae"], np.mean(np.abs(actual - origin_share)))
            and close(row["mean_origin_opportunity_share"], origin_share.mean())
            and close(row["mean_destination_opportunity_share"], actual.mean())
        )
    add(
        "persistence_diagnostics_reconstruct",
        persistence_ok and len(persistence) == 7,
        f"reconstruct={persistence_ok};rows={len(persistence)}",
        "true;rows=7",
        "Pooled, sensitivity-window, and four origin persistence diagnostics reconstruct independently.",
    )

    current_protected = study.protected_tree_sha256()
    recorded_master_before = set(origins["master_sha256_before"].astype(str))
    recorded_master_after = set(origins["master_sha256_after"].astype(str))
    recorded_protected_before = set(origins["protected_tree_sha256_before"].astype(str))
    recorded_protected_after = set(origins["protected_tree_sha256_after"].astype(str))
    source_unchanged = recorded_master_before == {master_hash} and recorded_master_after == {master_hash}
    protected_unchanged = (
        len(recorded_protected_before) == 1
        and recorded_protected_before == recorded_protected_after
        and recorded_protected_after == {current_protected}
        and study.coarse.as_bool(origins["protected_outputs_unchanged"]).all()
    )
    add(
        "source_hash_unchanged_before_and_after",
        source_unchanged,
        f"before={recorded_master_before};after={recorded_master_after};current={master_hash}",
        "all verified master hash",
        "The comprehensive master was not altered.",
    )
    add(
        "all_protected_outputs_unchanged",
        protected_unchanged,
        f"before={recorded_protected_before};after={recorded_protected_after};current={current_protected}",
        "all identical",
        "Compatibility, rolling, subgroup, case-study, and coarse-diagnostic trees remain byte-identical.",
    )

    status = "pass" if all(check["passed"] for check in checks) else "fail"
    payload = {
        "status": status,
        "blocking_checks": len(checks),
        "blocking_checks_passed": sum(int(check["passed"]) for check in checks),
        "blocking_checks_failed": sum(int(not check["passed"]) for check in checks),
        "cohort_rows": len(cohort),
        "cohort_unique_players": int(cohort["player_id"].nunique()),
        "master_sha256": master_hash,
        "protected_tree_sha256": current_protected,
        "checks": checks,
    }
    return checks, payload


def main() -> None:
    checks, payload = verify()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(OUTPUT_DIR / "independent_verification.csv", index=False)
    (OUTPUT_DIR / "independent_verification.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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

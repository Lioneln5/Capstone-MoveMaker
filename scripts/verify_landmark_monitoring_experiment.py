"""Independently verify the day-365 landmark monitoring artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "Data" / "processed" / "landmark_monitoring_experiment"
EXPECTED_OUTPUTS = {
    "README.md",
    "build_checks.csv",
    "landmark_comparison_results.csv",
    "landmark_decision_summary.csv",
    "landmark_feature_manifest.csv",
    "landmark_origin_results.csv",
    "landmark_predictions.csv",
    "landmark_risk_bands.csv",
    "market_value_landmark_targets.csv",
    "probability_consistency_audit.csv",
    "run_summary.json",
    "source_manifest.csv",
    "utilization_landmark_targets.csv",
    "validation_tuning_results.csv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    rows: list[dict[str, Any]] = []

    def add(check: str, passed: bool, observed: Any, expected: Any, note: str) -> None:
        rows.append(
            {
                "check": check,
                "passed": bool(passed),
                "observed": observed,
                "expected": expected,
                "note": note,
            }
        )

    output_manifest = pd.read_csv(OUTPUT_DIR / "output_manifest.csv")
    add(
        "output_manifest_file_set",
        set(output_manifest["file"]) == EXPECTED_OUTPUTS,
        len(set(output_manifest["file"])),
        len(EXPECTED_OUTPUTS),
        "The runner declared every expected primary artifact.",
    )
    output_hash_mismatches = []
    for row in output_manifest.itertuples(index=False):
        path = OUTPUT_DIR / row.file
        if not path.exists() or sha256_file(path) != row.sha256:
            output_hash_mismatches.append(row.file)
    add(
        "output_hashes",
        not output_hash_mismatches,
        ";".join(output_hash_mismatches),
        "none",
        "Every declared output matches its recorded SHA-256.",
    )

    source_manifest = pd.read_csv(OUTPUT_DIR / "source_manifest.csv")
    source_hash_mismatches = []
    for row in source_manifest.itertuples(index=False):
        path = ROOT / row.source
        if not path.exists() or sha256_file(path) != row.sha256:
            source_hash_mismatches.append(row.source)
    add(
        "source_hashes",
        not source_hash_mismatches,
        ";".join(source_hash_mismatches),
        "none",
        "The experiment still points to the exact audited source files it used.",
    )

    build_checks = pd.read_csv(OUTPUT_DIR / "build_checks.csv")
    build_pass = build_checks["passed"].astype(str).str.lower().eq("true")
    add(
        "runner_build_checks",
        build_pass.all(),
        int(build_pass.sum()),
        len(build_checks),
        "All runner-side leakage, cohort, and pairing checks passed.",
    )

    utilization = pd.read_csv(OUTPUT_DIR / "utilization_landmark_targets.csv", low_memory=False)
    value = pd.read_csv(OUTPUT_DIR / "market_value_landmark_targets.csv", low_memory=False)
    add(
        "utilization_cohort_exact",
        len(utilization) == 667 and utilization["player_id"].nunique() == 551,
        f"{len(utilization)} rows; {utilization['player_id'].nunique()} players",
        "667 rows; 551 players",
        "Exact frozen landmark utilization cohort.",
    )
    add(
        "market_value_cohort_exact",
        len(value) == 2632 and value["player_id"].nunique() == 1730,
        f"{len(value)} rows; {value['player_id'].nunique()} players",
        "2632 rows; 1730 players",
        "Exact frozen landmark market-value cohort.",
    )
    add(
        "cohort_keys_unique",
        utilization["transfer_event_id"].is_unique and value["transfer_event_id"].is_unique,
        f"{utilization['transfer_event_id'].nunique()};{value['transfer_event_id'].nunique()}",
        "667;2632",
        "One row per transfer in both target tables.",
    )

    for frame in [utilization, value]:
        frame["transfer_date"] = pd.to_datetime(frame["transfer_date"], errors="coerce")
        frame["landmark_date"] = pd.to_datetime(frame["landmark_date"], errors="coerce")
        frame["first_outbound_date"] = pd.to_datetime(frame["first_outbound_date"], errors="coerce")
    landmark_dates_valid = (
        (utilization["landmark_date"] - utilization["transfer_date"]).dt.days.eq(365).all()
        and (value["landmark_date"] - value["transfer_date"]).dt.days.eq(365).all()
    )
    add(
        "landmark_dates_exact",
        landmark_dates_valid,
        landmark_dates_valid,
        True,
        "Every monitoring date is exactly 365 days after the transfer.",
    )
    at_risk_valid = (
        utilization["at_destination_day365"].astype(str).str.lower().eq("true").all()
        and value["at_destination_day365"].astype(str).str.lower().eq("true").all()
        and (
            utilization["first_outbound_date"].isna()
            | utilization["first_outbound_date"].gt(utilization["landmark_date"])
        ).all()
        and (
            value["first_outbound_date"].isna()
            | value["first_outbound_date"].gt(value["landmark_date"])
        ).all()
    )
    add(
        "day365_at_risk_independent",
        at_risk_valid,
        at_risk_valid,
        True,
        "No player had a known outbound event on or before the scoring landmark.",
    )

    recomputed_share = utilization["year2_destination_minutes"] / (
        90.0 * utilization["year2_destination_club_games"]
    )
    utilization_targets_valid = (
        np.allclose(
            recomputed_share,
            utilization["target_year2_opportunity_share"],
            atol=1e-12,
            rtol=0,
        )
        and np.array_equal(
            recomputed_share.lt(0.10).astype(int),
            utilization["target_year2_underutilization_10pct"].astype(int),
        )
        and np.array_equal(
            recomputed_share.lt(0.25).astype(int),
            utilization["target_year2_underutilization_25pct"].astype(int),
        )
    )
    add(
        "utilization_targets_recomputed",
        utilization_targets_valid,
        utilization_targets_valid,
        True,
        "Continuous and threshold targets reproduce from year-two minutes and club games.",
    )

    value["landmark_value_date"] = pd.to_datetime(value["landmark_value_date"], errors="coerce")
    asof_valid = (
        value["landmark_value_date"].le(value["landmark_date"]).all()
        and value["landmark_value_staleness_days"].between(0, 180).all()
    )
    add(
        "valuation_asof_timing_independent",
        asof_valid,
        int(value["landmark_value_date"].gt(value["landmark_date"]).sum()),
        0,
        "Landmark value predictors never use a later valuation.",
    )
    recomputed_log = np.log(
        value["valuation_post24_nearest_eur"] / value["landmark_value_eur"]
    )
    recomputed_pct = (
        value["valuation_post24_nearest_eur"] - value["landmark_value_eur"]
    ) / value["landmark_value_eur"]
    value_targets_valid = np.allclose(
        recomputed_log,
        value["target_landmark_to_24m_log_value_ratio"],
        atol=1e-12,
        rtol=0,
    ) and np.allclose(
        recomputed_pct,
        value["target_landmark_to_24m_value_change_pct"],
        atol=1e-12,
        rtol=0,
    )
    for threshold in [10, 25, 50]:
        boundary = np.nextafter(value["landmark_value_eur"] * (1 - threshold / 100), np.inf)
        value_targets_valid = value_targets_valid and np.array_equal(
            value["valuation_post24_nearest_eur"].le(boundary).astype(int),
            value[f"target_landmark_to_24m_downside_{threshold}pct"].astype(int),
        )
    add(
        "market_value_targets_recomputed",
        value_targets_valid,
        value_targets_valid,
        True,
        "All day365-to-24m continuous and threshold targets reproduce exactly.",
    )

    features = pd.read_csv(OUTPUT_DIR / "landmark_feature_manifest.csv")
    feature_names = features["feature"].fillna("").astype(str).str.lower()
    forbidden = features.loc[
        feature_names.str.contains("post24", regex=False)
        | feature_names.str.startswith("target_"),
        "feature",
    ].tolist()
    add(
        "feature_manifest_no_outcomes",
        not forbidden,
        ";".join(map(str, forbidden)),
        "none",
        "No 24-month outcome or target appears in the predictor manifest.",
    )

    predictions = pd.read_csv(OUTPUT_DIR / "landmark_predictions.csv", low_memory=False)
    prediction_key = ["endpoint", "origin_id", "model_variant", "transfer_event_id"]
    probabilities = predictions.loc[
        predictions["endpoint_kind"].eq("classification"), "prediction"
    ]
    prediction_valid = (
        not predictions.duplicated(prediction_key).any()
        and probabilities.between(0, 1).all()
        and set(predictions["evaluation_year"]) == {2020, 2021, 2022, 2023}
    )
    consistency = pd.read_csv(OUTPUT_DIR / "probability_consistency_audit.csv")
    recommended_consistency = consistency.set_index(["model_variant", "rule"])
    consistency_valid = (
        recommended_consistency.loc[
            ("U2_plus_first_year_involvement", "P(under10) <= P(under25)"),
            "raw_violations",
        ]
        == 12
        and recommended_consistency.loc[
            ("V3_plus_first_year_involvement", "P(down25) <= P(down10)"),
            "raw_violations",
        ]
        == 34
        and recommended_consistency.loc[
            ("V3_plus_first_year_involvement", "P(down50) <= P(down25)"),
            "raw_violations",
        ]
        == 21
    )
    add(
        "prediction_integrity",
        prediction_valid and consistency_valid,
        f"{len(predictions)} rows; {int(predictions.duplicated(prediction_key).sum())} duplicates; consistency audited={consistency_valid}",
        "unique, bounded, evaluation years 2020-2023, raw ordering audited",
        "Prediction keys, probability bounds, chronology, and known raw threshold-order violations are intact.",
    )

    comparisons = pd.read_csv(OUTPUT_DIR / "landmark_comparison_results.csv")
    pooled = comparisons.loc[comparisons["window"].eq("pooled_four")]
    point_mismatches = []
    for comparison in pooled.itertuples(index=False):
        subset = predictions.loc[predictions["endpoint"].eq(comparison.endpoint)]
        keys = ["origin_id", "evaluation_year", "transfer_event_id", "player_id"]
        left = subset.loc[
            subset["model_variant"].eq(comparison.baseline_model),
            keys + ["actual", "prediction"],
        ].rename(columns={"prediction": "baseline"})
        right = subset.loc[
            subset["model_variant"].eq(comparison.candidate_model),
            keys + ["prediction"],
        ].rename(columns={"prediction": "candidate"})
        pair = left.merge(right, on=keys, how="inner", validate="one_to_one")
        if comparison.endpoint_kind == "regression":
            base = np.abs(pair["actual"] - pair["baseline"]).mean()
            candidate = np.abs(pair["actual"] - pair["candidate"]).mean()
        else:
            base = np.square(pair["actual"] - pair["baseline"]).mean()
            candidate = np.square(pair["actual"] - pair["candidate"]).mean()
        if (
            len(pair) != comparison.pooled_n
            or not np.isclose(base, comparison.baseline_loss, atol=1e-12)
            or not np.isclose(candidate, comparison.candidate_loss, atol=1e-12)
            or not np.isclose(base - candidate, comparison.loss_improvement, atol=1e-12)
        ):
            point_mismatches.append(f"{comparison.endpoint}:{comparison.comparison}")
        if comparison.endpoint_kind == "classification" and pair["actual"].nunique() == 2:
            auc = roc_auc_score(pair["actual"], pair["candidate"])
            if not np.isclose(auc, comparison.candidate_roc_auc, atol=1e-12):
                point_mismatches.append(f"{comparison.endpoint}:{comparison.comparison}:auc")
    add(
        "pooled_comparisons_recomputed",
        not point_mismatches,
        ";".join(point_mismatches),
        "none",
        "Every pooled MAE/Brier improvement and classification AUC reproduces from row predictions.",
    )

    decisions = pd.read_csv(OUTPUT_DIR / "landmark_decision_summary.csv")
    validated_expected = {
        "year2_opportunity_share",
        "year2_underutilization_10pct",
        "year2_underutilization_25pct",
        "year2_value_log_ratio",
        "year2_value_downside_10pct",
        "year2_value_downside_25pct",
    }
    validated_observed = set(
        decisions.loc[
            decisions["status"].eq("validated_monitoring_increment"), "endpoint"
        ]
    )
    uncertain = decisions.loc[
        decisions["endpoint"].eq("year2_value_downside_50pct"), "status"
    ].iloc[0]
    add(
        "decision_set_exact",
        validated_observed == validated_expected and uncertain == "promising_but_uncertain",
        f"validated={sorted(validated_observed)}; 50pct={uncertain}",
        "six validated; 50pct promising_but_uncertain",
        "Decision labels match the predeclared CI and origin-win rule.",
    )

    verification = pd.DataFrame(rows)
    verification.to_csv(OUTPUT_DIR / "independent_verification.csv", index=False)
    status = "pass" if verification["passed"].all() else "fail"
    summary = {
        "status": status,
        "checks": len(verification),
        "checks_passed": int(verification["passed"].sum()),
        "utilization_cohort_rows": len(utilization),
        "market_value_cohort_rows": len(value),
        "validated_monitoring_endpoints": sorted(validated_observed),
        "uncertain_endpoint": "year2_value_downside_50pct",
    }
    (OUTPUT_DIR / "independent_verification.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if status != "pass":
        failed = verification.loc[~verification["passed"], "check"].tolist()
        raise RuntimeError(f"Independent verification failed: {failed}")


if __name__ == "__main__":
    main()

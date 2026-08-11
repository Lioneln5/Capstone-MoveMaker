"""Integrate frozen MoveMaker components into a capital-at-risk matrix.

This is deliberately not a new raw-feature classifier.  It reconciles honest
out-of-season component forecasts, converts them to within-origin ranks, and
tests a predeclared geometric joint-priority score.  Reported fee and the fee
benchmark are attached as exposure/negotiation context but do not enter the
primary risk score.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
for dependency_dir in [ROOT / "models", ROOT / "scripts"]:
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

from train_compatibility_models import sha256_file


AUDIT_PATH = ROOT / "Data" / "processed" / "financial_module_feasibility" / "financial_event_audit.csv"
AUDIT_VERIFICATION_PATH = ROOT / "Data" / "processed" / "financial_module_feasibility" / "independent_verification.json"
MARKET_DIR = ROOT / "Data" / "processed" / "market_value_calibration_experiment"
UNDER_DIR = ROOT / "Data" / "processed" / "paid_transfer_underutilization_model"
REFINEMENT_DIR = ROOT / "Data" / "processed" / "paid_transfer_underutilization_refinement"
FEE_DIR = ROOT / "Data" / "processed" / "fee_benchmarking_model"
OUTPUT_DIR = ROOT / "Data" / "processed" / "capital_at_risk_integration"

RANDOM_SEED = 20260811
BOOTSTRAP_REPETITIONS = 5000
EVALUATION_YEARS = [2020, 2021, 2022, 2023]
WINDOWS = {
    "pooled_four": [2020, 2021, 2022, 2023],
    "mature_three": [2021, 2022, 2023],
    "terminal_two": [2022, 2023],
}
PROTECTED_DIRS = [
    ROOT / "Data" / "processed" / "financial_module_feasibility",
    MARKET_DIR, UNDER_DIR, REFINEMENT_DIR, FEE_DIR,
    ROOT / "Data" / "processed" / "market_value_downside_model",
    ROOT / "Data" / "processed" / "market_value_downside_product",
]


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().str.lower().isin({"true", "1", "yes"})


def stable_seed(*parts: str) -> int:
    token = "|".join(parts).encode("utf-8")
    return (RANDOM_SEED + int(hashlib.sha256(token).hexdigest()[:8], 16)) % (2**32 - 1)


def protected_tree_sha256() -> str:
    digest = hashlib.sha256()
    for directory in sorted(PROTECTED_DIRS):
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def require_verified(path: Path, label: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "pass":
        raise RuntimeError(f"{label} independent verification is not passing.")


def load_dual_cohort() -> pd.DataFrame:
    require_verified(AUDIT_VERIFICATION_PATH, "Financial audit")
    columns = [
        "transfer_event_id", "canonical_transfer_event_id", "player_id", "transfer_date",
        "season_start_year", "from_club_name", "to_club_name", "position_group",
        "destination_competition", "canonical_transfer_fee_eur", "fee_band",
        "valuation_pre_eur", "valuation_post24_nearest_eur", "market_value_change_24m_pct",
        "destination_opportunity_share_24m", "eligible_dual_capital_at_risk_24m",
    ]
    audit = pd.read_csv(AUDIT_PATH, usecols=columns, low_memory=False)
    cohort = audit.loc[as_bool(audit["eligible_dual_capital_at_risk_24m"])].copy()
    cohort["transfer_event_id"] = cohort["transfer_event_id"].astype(str)
    cohort["transfer_date"] = pd.to_datetime(cohort["transfer_date"], errors="coerce")
    cohort["target_value_downside_25pct"] = pd.to_numeric(
        cohort["market_value_change_24m_pct"], errors="coerce"
    ).le(-0.25).astype(int)
    opportunity_share = pd.to_numeric(
        cohort["destination_opportunity_share_24m"], errors="coerce"
    )
    cohort["target_underutilization_10pct"] = opportunity_share.lt(0.10).astype(int)
    cohort["target_underutilization_25pct"] = opportunity_share.lt(0.25).astype(int)
    cohort["target_dual_severe"] = (
        cohort["target_value_downside_25pct"]
        & cohort["target_underutilization_10pct"]
    ).astype(int)
    cohort["target_dual_meaningful"] = (
        cohort["target_value_downside_25pct"]
        & cohort["target_underutilization_25pct"]
    ).astype(int)
    if len(cohort) != 827 or cohort["target_dual_severe"].sum() != 110:
        raise RuntimeError(
            f"Strict dual cohort mismatch: rows={len(cohort)}, severe={cohort['target_dual_severe'].sum()}."
        )
    return cohort


def load_component_predictions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    require_verified(MARKET_DIR / "independent_verification.json", "Market calibration")
    require_verified(UNDER_DIR / "independent_verification.json", "Under-utilization model")
    require_verified(REFINEMENT_DIR / "independent_verification.json", "Under-utilization refinement")
    require_verified(FEE_DIR / "independent_verification.json", "Fee benchmark")

    market = pd.read_csv(MARKET_DIR / "calibrated_predictions.csv", low_memory=False)
    market = market.loc[
        market["strategy"].eq("policy_selected") & market["endpoint"].eq("downside_25pct"),
        ["origin_id", "evaluation_year", "transfer_event_id", "player_id", "probability"],
    ].rename(columns={"probability": "market_value_downside_probability_25pct"})

    severe = pd.read_csv(UNDER_DIR / "underutilization_predictions.csv", low_memory=False)
    severe = severe.loc[
        severe["model_variant"].eq("M_policy_ordered")
        & severe["endpoint"].eq("underutilization_10pct"),
        ["origin_id", "evaluation_year", "transfer_event_id", "player_id", "prediction"],
    ].rename(columns={"prediction": "severe_underutilization_risk_score"})

    meaningful = pd.read_csv(REFINEMENT_DIR / "classification_predictions.csv", low_memory=False)
    meaningful = meaningful.loc[
        meaningful["strategy"].eq("selected_compact_calibrated")
        & meaningful["endpoint"].eq("underutilization_25pct"),
        ["origin_id", "evaluation_year", "transfer_event_id", "player_id", "probability"],
    ].rename(columns={"probability": "meaningful_underutilization_probability_25pct"})

    fee = pd.read_csv(FEE_DIR / "fee_benchmark_predictions.csv", low_memory=False)
    fee = fee.loc[
        fee["split"].eq("evaluation") & fee["model_variant"].eq("M1_market_profile"),
        [
            "origin_id", "evaluation_year", "transfer_event_id", "player_id",
            "actual_fee_eur", "predicted_fee_eur", "interval80_lower_eur",
            "interval80_upper_eur", "interval90_lower_eur", "interval90_upper_eur",
        ],
    ].rename(columns={
        "actual_fee_eur": "reported_fee_eur",
        "predicted_fee_eur": "fee_benchmark_eur",
        "interval80_lower_eur": "fee_benchmark_80_lower_eur",
        "interval80_upper_eur": "fee_benchmark_80_upper_eur",
        "interval90_lower_eur": "fee_benchmark_90_lower_eur",
        "interval90_upper_eur": "fee_benchmark_90_upper_eur",
    })
    for frame in [market, severe, meaningful, fee]:
        frame["transfer_event_id"] = frame["transfer_event_id"].astype(str)
        if frame["transfer_event_id"].duplicated().any():
            raise RuntimeError("Component prediction IDs are not unique.")
    return market, severe, meaningful, fee


def tier_from_percentile(series: pd.Series) -> pd.Series:
    return pd.cut(
        series, bins=[0, 1 / 3, 2 / 3, 1.000001],
        labels=["low", "medium", "high"], include_lowest=True,
    ).astype(str)


def build_integration_frame(cohort: pd.DataFrame) -> pd.DataFrame:
    market, severe, meaningful, fee = load_component_predictions()
    keys = ["origin_id", "evaluation_year", "transfer_event_id", "player_id"]
    frame = cohort.merge(market, on=["transfer_event_id", "player_id"], how="inner")
    frame = frame.merge(severe, on=keys, how="inner", validate="one_to_one")
    frame = frame.merge(meaningful, on=keys, how="inner", validate="one_to_one")
    frame = frame.merge(fee, on=keys, how="inner", validate="one_to_one")
    if len(frame) != 446 or not frame["transfer_event_id"].is_unique:
        raise RuntimeError(f"Component evaluation overlap mismatch: {len(frame)} rows.")
    if frame["target_dual_severe"].sum() != 78:
        raise RuntimeError(f"Evaluation severe count mismatch: {frame['target_dual_severe'].sum()}.")
    if not np.allclose(frame["canonical_transfer_fee_eur"], frame["reported_fee_eur"]):
        raise RuntimeError("Audit and fee-model reported fees disagree.")

    rank_specs = {
        "value_downside_rank": "market_value_downside_probability_25pct",
        "severe_underutilization_rank": "severe_underutilization_risk_score",
        "meaningful_underutilization_rank": "meaningful_underutilization_probability_25pct",
    }
    for output, source in rank_specs.items():
        frame[output] = frame.groupby("origin_id")[source].rank(pct=True, method="average")
    frame["severe_joint_priority_score"] = np.sqrt(
        frame["value_downside_rank"] * frame["severe_underutilization_rank"]
    )
    frame["meaningful_joint_priority_score"] = np.sqrt(
        frame["value_downside_rank"] * frame["meaningful_underutilization_rank"]
    )
    frame["severe_joint_min_score"] = np.minimum(
        frame["value_downside_rank"], frame["severe_underutilization_rank"]
    )
    frame["meaningful_joint_min_score"] = np.minimum(
        frame["value_downside_rank"], frame["meaningful_underutilization_rank"]
    )
    frame["reported_fee_to_benchmark_ratio"] = (
        frame["reported_fee_eur"] / frame["fee_benchmark_eur"]
    )
    frame["fee_benchmark_assessment"] = pd.cut(
        frame["reported_fee_to_benchmark_ratio"],
        [-np.inf, 0.67, 0.85, 1.18, 1.50, np.inf],
        labels=["large_discount", "discount", "near_benchmark", "premium", "large_premium"],
        right=False,
    ).astype(str)
    frame["value_downside_tier"] = tier_from_percentile(frame["value_downside_rank"])
    frame["severe_underutilization_tier"] = tier_from_percentile(
        frame["severe_underutilization_rank"]
    )
    frame["meaningful_underutilization_tier"] = tier_from_percentile(
        frame["meaningful_underutilization_rank"]
    )
    frame["severe_high_high_flag"] = (
        frame["value_downside_rank"].ge(0.50)
        & frame["severe_underutilization_rank"].ge(0.50)
    )
    frame["meaningful_high_high_flag"] = (
        frame["value_downside_rank"].ge(0.50)
        & frame["meaningful_underutilization_rank"].ge(0.50)
    )
    frame["value_downside_exposure_index_eur"] = (
        frame["reported_fee_eur"] * frame["market_value_downside_probability_25pct"]
    )
    frame["severe_capital_priority_index_eur"] = (
        frame["reported_fee_eur"] * frame["severe_joint_priority_score"]
    )
    frame["meaningful_capital_priority_index_eur"] = (
        frame["reported_fee_eur"] * frame["meaningful_joint_priority_score"]
    )
    return frame.sort_values(["evaluation_year", "transfer_date", "transfer_event_id"])


def binary_auc(actual: np.ndarray, score: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=int)
    score = np.asarray(score, dtype=float)
    positives = int(actual.sum())
    negatives = len(actual) - positives
    if positives == 0 or negatives == 0:
        return np.nan
    ranks = rankdata(score, method="average")
    return float(
        (ranks[actual == 1].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def build_score_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("dual_severe", "target_dual_severe", "value_downside_component", "value_downside_rank"),
        ("dual_severe", "target_dual_severe", "severe_underutilization_component", "severe_underutilization_rank"),
        ("dual_severe", "target_dual_severe", "geometric_joint", "severe_joint_priority_score"),
        ("dual_severe", "target_dual_severe", "minimum_joint", "severe_joint_min_score"),
        ("dual_meaningful", "target_dual_meaningful", "value_downside_component", "value_downside_rank"),
        ("dual_meaningful", "target_dual_meaningful", "meaningful_underutilization_component", "meaningful_underutilization_rank"),
        ("dual_meaningful", "target_dual_meaningful", "geometric_joint", "meaningful_joint_priority_score"),
        ("dual_meaningful", "target_dual_meaningful", "minimum_joint", "meaningful_joint_min_score"),
    ]
    rows = []
    for target_name, actual_column, score_name, score_column in specs:
        for window, years in WINDOWS.items():
            group = frame.loc[frame["evaluation_year"].isin(years)]
            rows.append({
                "target": target_name, "score": score_name, "score_column": score_column,
                "window": window, "evaluation_years": ";".join(map(str, years)),
                "rows": len(group), "unique_players": group["player_id"].nunique(),
                "positives": int(group[actual_column].sum()),
                "positive_rate": group[actual_column].mean(),
                "roc_auc": binary_auc(group[actual_column], group[score_column]),
                "average_precision": average_precision_score(
                    group[actual_column], group[score_column]
                ),
            })
        for origin_id, group in frame.groupby("origin_id"):
            rows.append({
                "target": target_name, "score": score_name, "score_column": score_column,
                "window": origin_id,
                "evaluation_years": str(int(group["evaluation_year"].iloc[0])),
                "rows": len(group), "unique_players": group["player_id"].nunique(),
                "positives": int(group[actual_column].sum()),
                "positive_rate": group[actual_column].mean(),
                "roc_auc": binary_auc(group[actual_column], group[score_column]),
                "average_precision": average_precision_score(
                    group[actual_column], group[score_column]
                ),
            })
    return pd.DataFrame(rows)


def auc_cluster_bootstrap(
    frame: pd.DataFrame, actual_column: str, baseline_column: str,
    candidate_column: str, seed: int,
) -> dict[str, Any]:
    groups = [group.index.to_numpy() for _, group in frame.groupby("player_id", sort=True)]
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_REPETITIONS)
    for repetition in range(BOOTSTRAP_REPETITIONS):
        sampled = rng.integers(0, len(groups), len(groups))
        indices = np.concatenate([groups[index] for index in sampled])
        subset = frame.loc[indices]
        draws[repetition] = (
            binary_auc(subset[actual_column], subset[candidate_column])
            - binary_auc(subset[actual_column], subset[baseline_column])
        )
    return {
        "cluster_count": len(groups),
        "cluster_ci_lower_95": float(np.quantile(draws, 0.025)),
        "cluster_ci_upper_95": float(np.quantile(draws, 0.975)),
        "cluster_probability_candidate_better": float(np.mean(draws > 0)),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS, "bootstrap_seed": seed,
    }


def build_integration_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("dual_severe", "target_dual_severe", "geometric_vs_value", "value_downside_rank", "severe_joint_priority_score"),
        ("dual_severe", "target_dual_severe", "geometric_vs_underutilization", "severe_underutilization_rank", "severe_joint_priority_score"),
        ("dual_meaningful", "target_dual_meaningful", "geometric_vs_value", "value_downside_rank", "meaningful_joint_priority_score"),
        ("dual_meaningful", "target_dual_meaningful", "geometric_vs_underutilization", "meaningful_underutilization_rank", "meaningful_joint_priority_score"),
    ]
    rows = []
    for target_name, actual, comparison, baseline, candidate in specs:
        baseline_auc = binary_auc(frame[actual], frame[baseline])
        candidate_auc = binary_auc(frame[actual], frame[candidate])
        origin_changes = []
        for _, group in frame.groupby("origin_id"):
            origin_changes.append(
                binary_auc(group[actual], group[candidate])
                - binary_auc(group[actual], group[baseline])
            )
        bootstrap = auc_cluster_bootstrap(
            frame, actual, baseline, candidate,
            stable_seed(target_name, comparison, "auc_bootstrap"),
        )
        rows.append({
            "target": target_name, "comparison": comparison,
            "baseline_score_column": baseline, "candidate_score_column": candidate,
            "rows": len(frame), "unique_players": frame["player_id"].nunique(),
            "positives": int(frame[actual].sum()),
            "baseline_auc": baseline_auc, "candidate_auc": candidate_auc,
            "auc_improvement": candidate_auc - baseline_auc,
            "origin_wins": int(sum(value > 0 for value in origin_changes)),
            "origin_count": len(origin_changes), **bootstrap,
        })
    return pd.DataFrame(rows)


def build_risk_matrices(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix_rows = []
    quadrant_rows = []
    specs = [
        (
            "dual_severe", "target_dual_severe", "severe_underutilization_tier",
            "severe_underutilization_rank", "severe_high_high_flag",
        ),
        (
            "dual_meaningful", "target_dual_meaningful", "meaningful_underutilization_tier",
            "meaningful_underutilization_rank", "meaningful_high_high_flag",
        ),
    ]
    for target_name, actual, under_tier, under_rank, high_high in specs:
        grouped = frame.groupby(
            ["value_downside_tier", under_tier], observed=True, sort=False
        )
        for (value_tier, utilization_tier), group in grouped:
            matrix_rows.append({
                "target": target_name, "value_downside_tier": value_tier,
                "underutilization_tier": utilization_tier, "rows": len(group),
                "unique_players": group["player_id"].nunique(),
                "joint_event_count": int(group[actual].sum()),
                "joint_event_rate": group[actual].mean(),
                "value_downside_rate": group["target_value_downside_25pct"].mean(),
                "underutilization_10pct_rate": group["target_underutilization_10pct"].mean(),
                "underutilization_25pct_rate": group["target_underutilization_25pct"].mean(),
                "total_reported_fee_eur": group["reported_fee_eur"].sum(),
                "mean_reported_fee_eur": group["reported_fee_eur"].mean(),
            })
        value_high = frame["value_downside_rank"].ge(0.50)
        utilization_high = frame[under_rank].ge(0.50)
        labels = np.select(
            [
                ~value_high & ~utilization_high,
                ~value_high & utilization_high,
                value_high & ~utilization_high,
                value_high & utilization_high,
            ],
            ["low_low", "utilization_only_high", "value_only_high", "high_high"],
            default="unassigned",
        )
        working = frame.assign(quadrant=labels)
        for quadrant, group in working.groupby("quadrant"):
            quadrant_rows.append({
                "target": target_name, "quadrant": quadrant, "rows": len(group),
                "unique_players": group["player_id"].nunique(),
                "joint_event_count": int(group[actual].sum()),
                "joint_event_rate": group[actual].mean(),
                "total_reported_fee_eur": group["reported_fee_eur"].sum(),
                "share_of_evaluated_reported_fees": (
                    group["reported_fee_eur"].sum() / frame["reported_fee_eur"].sum()
                ),
            })
    return pd.DataFrame(matrix_rows), pd.DataFrame(quadrant_rows)


def build_priority_bands(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("dual_severe", "target_dual_severe", "severe_joint_priority_score"),
        ("dual_meaningful", "target_dual_meaningful", "meaningful_joint_priority_score"),
    ]
    for target_name, actual, score in specs:
        work = frame.copy()
        work["priority_quintile"] = work.groupby("origin_id")[score].transform(
            lambda values: pd.qcut(values.rank(method="first"), 5, labels=False) + 1
        )
        for quintile, group in work.groupby("priority_quintile"):
            rows.append({
                "target": target_name, "priority_quintile": int(quintile),
                "rows": len(group), "unique_players": group["player_id"].nunique(),
                "mean_priority_score": group[score].mean(),
                "joint_event_count": int(group[actual].sum()),
                "joint_event_rate": group[actual].mean(),
                "total_reported_fee_eur": group["reported_fee_eur"].sum(),
                "mean_reported_fee_eur": group["reported_fee_eur"].mean(),
            })
    return pd.DataFrame(rows)


def build_subgroups(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target_name, actual, score in [
        ("dual_severe", "target_dual_severe", "severe_joint_priority_score"),
        ("dual_meaningful", "target_dual_meaningful", "meaningful_joint_priority_score"),
    ]:
        for dimension in [
            "position_group", "fee_band", "destination_competition", "evaluation_year",
        ]:
            for category, group in frame.groupby(dimension, dropna=False):
                adequate = len(group) >= 50 and group[actual].sum() >= 8 and group[actual].nunique() == 2
                rows.append({
                    "target": target_name, "dimension": dimension, "category": str(category),
                    "rows": len(group), "unique_players": group["player_id"].nunique(),
                    "positives": int(group[actual].sum()),
                    "positive_rate": group[actual].mean(),
                    "adequate_for_auc": adequate,
                    "joint_score_auc": binary_auc(group[actual], group[score]) if adequate else np.nan,
                })
    return pd.DataFrame(rows)


def build_decision(
    frame: pd.DataFrame, comparisons: pd.DataFrame, quadrants: pd.DataFrame,
) -> pd.DataFrame:
    severe_value = comparisons.loc[
        comparisons["target"].eq("dual_severe")
        & comparisons["comparison"].eq("geometric_vs_value")
    ].iloc[0]
    severe_under = comparisons.loc[
        comparisons["target"].eq("dual_severe")
        & comparisons["comparison"].eq("geometric_vs_underutilization")
    ].iloc[0]
    meaningful_value = comparisons.loc[
        comparisons["target"].eq("dual_meaningful")
        & comparisons["comparison"].eq("geometric_vs_value")
    ].iloc[0]
    severe_validated = (
        severe_value["cluster_ci_lower_95"] > 0
        and severe_under["cluster_ci_lower_95"] > 0
        and severe_value["origin_wins"] >= 3
        and severe_under["origin_wins"] >= 3
    )
    meaningful_increment = meaningful_value["cluster_ci_lower_95"] > 0
    severe_quadrants = quadrants.loc[quadrants["target"].eq("dual_severe")].set_index("quadrant")
    high_high_rate = severe_quadrants.loc["high_high", "joint_event_rate"]
    overall_rate = frame["target_dual_severe"].mean()
    status = "validated_incremental_joint_score" if severe_validated else "useful_matrix_but_incremental_gain_uncertain"
    return pd.DataFrame([{
        "status": status, "evaluation_rows": len(frame),
        "evaluation_unique_players": frame["player_id"].nunique(),
        "dual_severe_events": int(frame["target_dual_severe"].sum()),
        "dual_severe_rate": overall_rate,
        "severe_geometric_auc": severe_value["candidate_auc"],
        "severe_auc_improvement_over_value": severe_value["auc_improvement"],
        "severe_improvement_over_value_ci_lower_95": severe_value["cluster_ci_lower_95"],
        "severe_improvement_over_value_ci_upper_95": severe_value["cluster_ci_upper_95"],
        "severe_improvement_over_value_origin_wins": int(severe_value["origin_wins"]),
        "severe_increment_validated": severe_validated,
        "meaningful_increment_over_value_validated": meaningful_increment,
        "severe_high_high_rows": int(severe_quadrants.loc["high_high", "rows"]),
        "severe_high_high_event_rate": high_high_rate,
        "severe_high_high_rate_ratio_vs_overall": high_high_rate / overall_rate,
        "recommended_use": (
            "component risk matrix and diligence prioritization; retain component outputs separately"
        ),
        "probability_claim_allowed": False,
        "fee_weighting_claim": (
            "prioritization index only; not expected loss, fair value, total cost, or ROI"
        ),
    }])


def build_checks(
    cohort: pd.DataFrame, frame: pd.DataFrame, comparisons: pd.DataFrame,
    matrices: pd.DataFrame,
) -> pd.DataFrame:
    checks = []
    def add(check: str, passed: bool, observed: Any, expected: Any, note: str) -> None:
        checks.append({
            "check": check, "passed": bool(passed), "observed": observed,
            "expected": expected, "note": note,
        })
    add("strict_cohort_rows", len(cohort) == 827, len(cohort), 827, "Matches financial audit overlap.")
    add("strict_cohort_severe_events", cohort["target_dual_severe"].sum() == 110, cohort["target_dual_severe"].sum(), 110, "Frozen dual definition exact.")
    add("evaluation_overlap_rows", len(frame) == 446, len(frame), 446, "All four component forecasts overlap.")
    add("evaluation_key_unique", frame["transfer_event_id"].is_unique, frame["transfer_event_id"].nunique(), len(frame), "One row per evaluated transfer.")
    add("evaluation_years", set(frame["evaluation_year"]) == set(EVALUATION_YEARS), sorted(frame["evaluation_year"].unique()), EVALUATION_YEARS, "Four rolling origins represented.")
    add("evaluation_severe_events", frame["target_dual_severe"].sum() == 78, frame["target_dual_severe"].sum(), 78, "Evaluation outcome count exact.")
    scores = frame[[
        "value_downside_rank", "severe_underutilization_rank",
        "meaningful_underutilization_rank", "severe_joint_priority_score",
        "meaningful_joint_priority_score",
    ]]
    add("rank_score_bounds", scores.ge(0).all().all() and scores.le(1).all().all(), f"{scores.min().min()}..{scores.max().max()}", "0..1", "All rank scores bounded.")
    severe_exact = np.allclose(
        frame["severe_joint_priority_score"],
        np.sqrt(frame["value_downside_rank"] * frame["severe_underutilization_rank"]),
    )
    add("geometric_score_exact", severe_exact, severe_exact, True, "Frozen formula exact.")
    add("reported_fee_reconciles", np.allclose(frame["reported_fee_eur"], frame["canonical_transfer_fee_eur"]), True, True, "Fee input agrees across modules.")
    add("comparison_rows", len(comparisons) == 4, len(comparisons), 4, "Four predeclared component comparisons.")
    add("matrix_rows", len(matrices) == 18, len(matrices), 18, "Two complete 3x3 matrices.")
    add("matrix_counts", matrices.groupby("target")["rows"].sum().eq(446).all(), matrices.groupby("target")["rows"].sum().to_dict(), "446 each", "Every deal assigned once per matrix.")
    add("no_joint_probability", not any("joint_probability" in column for column in frame.columns), ";".join(frame.columns), "none", "Unvalidated score is not labeled probability.")
    return pd.DataFrame(checks)


def write_readme(
    cohort: pd.DataFrame, frame: pd.DataFrame, metrics: pd.DataFrame,
    comparisons: pd.DataFrame, quadrants: pd.DataFrame, decision: pd.DataFrame,
) -> None:
    row = decision.iloc[0]
    severe_metrics = metrics.loc[
        metrics["target"].eq("dual_severe") & metrics["window"].eq("pooled_four")
    ].set_index("score")
    severe_value = comparisons.loc[
        comparisons["target"].eq("dual_severe")
        & comparisons["comparison"].eq("geometric_vs_value")
    ].iloc[0]
    severe_quadrants = quadrants.loc[quadrants["target"].eq("dual_severe")].set_index("quadrant")
    lines = [
        "# MoveMaker component-based capital-at-risk integration", "",
        "This layer combines frozen out-of-season component scores. It does not train a new raw-feature joint model and does not output expected financial loss or ROI.", "",
        "## Cohort", "",
        f"- Strict audited overlap: {len(cohort):,} paid transfers; {int(cohort['target_dual_severe'].sum())} suffered both >=25% estimated-value downside and <10% opportunity.",
        f"- Fully out-of-season component overlap: {len(frame):,} transfers and {frame['player_id'].nunique():,} players; {int(frame['target_dual_severe'].sum())} severe dual events ({frame['target_dual_severe'].mean():.1%}).", "",
        "## Frozen integration", "",
        "- Value component: calibrated probability of >=25% estimated market-value downside.",
        "- Severe opportunity component: ordered under-10% risk score, used only as a rank.",
        "- Meaningful opportunity component: calibrated under-25% probability.",
        "- Joint priority: geometric mean of within-origin component percentiles, forcing both components to matter.",
        "- Proposed/reported fee and fee-benchmark premium are displayed separately. They do not alter the primary joint risk score.", "",
        "## Result", "",
        f"- Severe value-only AUC: {severe_metrics.loc['value_downside_component', 'roc_auc']:.3f}.",
        f"- Severe under-use-only AUC: {severe_metrics.loc['severe_underutilization_component', 'roc_auc']:.3f}.",
        f"- Severe geometric joint AUC: {severe_metrics.loc['geometric_joint', 'roc_auc']:.3f}, improvement over value {severe_value['auc_improvement']:+.3f}, 95% CI [{severe_value['cluster_ci_lower_95']:+.3f}, {severe_value['cluster_ci_upper_95']:+.3f}], {int(severe_value['origin_wins'])}/4 origin wins.",
        f"- Median-cut high/high quadrant: {int(severe_quadrants.loc['high_high', 'rows'])} rows, {severe_quadrants.loc['high_high', 'joint_event_rate']:.1%} severe dual-event rate versus {frame['target_dual_severe'].mean():.1%} overall.",
        f"- Decision: `{row['status']}`.", "",
        "## Interpretation", "",
        "Use the matrix to distinguish value risk, opportunity risk, and deals exposed to both. The combined ranking is a triage aid: its incremental AUC gain is not strong enough to claim a calibrated joint probability. Keep the underlying component outputs visible and explain which dimension drives each flag.",
        "", "Any euro-weighted `capital_priority_index` is reported fee multiplied by a rank-based priority score. It is not expected loss, recoverable value, total deal cost, profit, or ROI.",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    protected_before = protected_tree_sha256()
    cohort = load_dual_cohort()
    frame = build_integration_frame(cohort)
    metrics = build_score_metrics(frame)
    comparisons = build_integration_comparisons(frame)
    matrices, quadrants = build_risk_matrices(frame)
    bands = build_priority_bands(frame)
    subgroups = build_subgroups(frame)
    decision = build_decision(frame, comparisons, quadrants)
    checks = build_checks(cohort, frame, comparisons, matrices)
    if not checks["passed"].all():
        raise RuntimeError(f"Build checks failed: {checks.loc[~checks['passed'], 'check'].tolist()}")

    target_columns = [
        "transfer_event_id", "canonical_transfer_event_id", "player_id", "transfer_date",
        "season_start_year", "canonical_transfer_fee_eur", "market_value_change_24m_pct",
        "destination_opportunity_share_24m", "target_value_downside_25pct",
        "target_underutilization_10pct", "target_underutilization_25pct",
        "target_dual_severe", "target_dual_meaningful",
    ]
    card_columns = [
        "origin_id", "evaluation_year", "transfer_event_id", "player_id", "transfer_date",
        "from_club_name", "to_club_name", "position_group", "destination_competition",
        "reported_fee_eur", "fee_benchmark_eur", "reported_fee_to_benchmark_ratio",
        "fee_benchmark_assessment", "market_value_downside_probability_25pct",
        "severe_underutilization_risk_score",
        "meaningful_underutilization_probability_25pct", "value_downside_rank",
        "severe_underutilization_rank", "meaningful_underutilization_rank",
        "severe_joint_priority_score", "meaningful_joint_priority_score",
        "severe_joint_min_score", "meaningful_joint_min_score",
        "value_downside_tier", "severe_underutilization_tier",
        "meaningful_underutilization_tier", "severe_high_high_flag",
        "meaningful_high_high_flag", "value_downside_exposure_index_eur",
        "severe_capital_priority_index_eur", "meaningful_capital_priority_index_eur",
        "target_value_downside_25pct", "target_underutilization_10pct",
        "target_underutilization_25pct", "target_dual_severe", "target_dual_meaningful",
    ]
    tables = {
        "dual_risk_targets.csv": cohort[target_columns],
        "component_integration_cards.csv": frame[card_columns],
        "component_score_metrics.csv": metrics,
        "integration_comparison_results.csv": comparisons,
        "risk_matrix_3x3.csv": matrices,
        "risk_quadrant_2x2.csv": quadrants,
        "priority_quintiles.csv": bands,
        "integration_subgroup_diagnostics.csv": subgroups,
        "integration_decision_summary.csv": decision,
        "build_checks.csv": checks,
    }
    for name, table in tables.items():
        table.to_csv(OUTPUT_DIR / name, index=False)
    source_paths = [
        AUDIT_PATH, AUDIT_VERIFICATION_PATH,
        MARKET_DIR / "calibrated_predictions.csv",
        MARKET_DIR / "independent_verification.json",
        UNDER_DIR / "underutilization_predictions.csv",
        UNDER_DIR / "independent_verification.json",
        REFINEMENT_DIR / "classification_predictions.csv",
        REFINEMENT_DIR / "independent_verification.json",
        FEE_DIR / "fee_benchmark_predictions.csv",
        FEE_DIR / "independent_verification.json",
    ]
    sources = pd.DataFrame([
        {
            "source": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in source_paths
    ])
    sources.to_csv(OUTPUT_DIR / "source_manifest.csv", index=False)
    tables["source_manifest.csv"] = sources
    write_readme(cohort, frame, metrics, comparisons, quadrants, decision)
    if protected_before != protected_tree_sha256():
        raise RuntimeError("A protected verified component changed during integration.")
    run_summary = {
        "strict_cohort_rows": len(cohort), "strict_dual_severe_events": int(cohort["target_dual_severe"].sum()),
        "evaluation_rows": len(frame), "evaluation_unique_players": frame["player_id"].nunique(),
        "evaluation_dual_severe_events": int(frame["target_dual_severe"].sum()),
        "decision": decision.iloc[0]["status"],
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "protected_outputs_unchanged": True,
        "build_checks": len(checks), "build_checks_passed": int(checks["passed"].sum()),
    }
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_names = sorted(list(tables) + ["README.md", "run_summary.json"])
    manifest = pd.DataFrame([
        {
            "file": name, "size_bytes": (OUTPUT_DIR / name).stat().st_size,
            "sha256": sha256_file(OUTPUT_DIR / name),
        }
        for name in manifest_names
    ])
    manifest.to_csv(OUTPUT_DIR / "output_manifest.csv", index=False)
    print(json.dumps(run_summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

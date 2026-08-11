"""Independently verify the component-based capital-at-risk integration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "Data" / "processed" / "financial_module_feasibility" / "financial_event_audit.csv"
MARKET_DIR = ROOT / "Data" / "processed" / "market_value_calibration_experiment"
UNDER_DIR = ROOT / "Data" / "processed" / "paid_transfer_underutilization_model"
REFINEMENT_DIR = ROOT / "Data" / "processed" / "paid_transfer_underutilization_refinement"
FEE_DIR = ROOT / "Data" / "processed" / "fee_benchmarking_model"
OUTPUT_DIR = ROOT / "Data" / "processed" / "capital_at_risk_integration"
EXPECTED_YEAR_COUNTS = {2020: 95, 2021: 135, 2022: 141, 2023: 75}


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().str.lower().isin({"true", "1", "yes"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left: Any, right: Any, atol: float = 1e-10) -> bool:
    return bool(np.isclose(float(left), float(right), atol=atol, rtol=0, equal_nan=True))


def binary_auc(actual: pd.Series, score: pd.Series) -> float:
    y = np.asarray(actual, dtype=int)
    values = np.asarray(score, dtype=float)
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return np.nan
    ranks = rankdata(values, method="average")
    return float(
        (ranks[y == 1].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def add(
    checks: list[dict[str, Any]], check: str, passed: bool, observed: Any,
    expected: Any, note: str,
) -> None:
    checks.append({
        "check": check, "passed": bool(passed), "observed": observed,
        "expected": expected, "note": note,
    })


def verify() -> None:
    checks: list[dict[str, Any]] = []
    required = [
        "dual_risk_targets.csv", "component_integration_cards.csv",
        "component_score_metrics.csv", "integration_comparison_results.csv",
        "risk_matrix_3x3.csv", "risk_quadrant_2x2.csv", "priority_quintiles.csv",
        "integration_subgroup_diagnostics.csv", "integration_decision_summary.csv",
        "build_checks.csv", "source_manifest.csv", "README.md", "run_summary.json",
        "output_manifest.csv",
    ]
    missing = [name for name in required if not (OUTPUT_DIR / name).exists()]
    add(checks, "required_outputs_exist", not missing, ";".join(missing), "none", "All promised integration artifacts exist.")
    if missing:
        raise RuntimeError(f"Missing outputs: {missing}")

    upstream_verifications = [
        ROOT / "Data" / "processed" / "financial_module_feasibility" / "independent_verification.json",
        MARKET_DIR / "independent_verification.json",
        UNDER_DIR / "independent_verification.json",
        REFINEMENT_DIR / "independent_verification.json",
        FEE_DIR / "independent_verification.json",
    ]
    upstream_statuses = [json.loads(path.read_text(encoding="utf-8")).get("status") for path in upstream_verifications]
    add(checks, "upstream_verifications_pass", all(value == "pass" for value in upstream_statuses), ";".join(map(str, upstream_statuses)), "pass;pass;pass;pass;pass", "Only independently passing component artifacts are integrated.")

    audit = pd.read_csv(AUDIT_PATH, low_memory=False)
    targets = pd.read_csv(OUTPUT_DIR / "dual_risk_targets.csv", low_memory=False)
    cards = pd.read_csv(OUTPUT_DIR / "component_integration_cards.csv", low_memory=False)
    metrics = pd.read_csv(OUTPUT_DIR / "component_score_metrics.csv")
    comparisons = pd.read_csv(OUTPUT_DIR / "integration_comparison_results.csv")
    matrix = pd.read_csv(OUTPUT_DIR / "risk_matrix_3x3.csv")
    quadrants = pd.read_csv(OUTPUT_DIR / "risk_quadrant_2x2.csv")
    quintiles = pd.read_csv(OUTPUT_DIR / "priority_quintiles.csv")
    subgroups = pd.read_csv(OUTPUT_DIR / "integration_subgroup_diagnostics.csv")
    decision = pd.read_csv(OUTPUT_DIR / "integration_decision_summary.csv")
    build_checks = pd.read_csv(OUTPUT_DIR / "build_checks.csv")
    sources = pd.read_csv(OUTPUT_DIR / "source_manifest.csv")
    manifest = pd.read_csv(OUTPUT_DIR / "output_manifest.csv")
    summary = json.loads((OUTPUT_DIR / "run_summary.json").read_text(encoding="utf-8"))

    for frame in [audit, targets, cards]:
        frame["transfer_event_id"] = frame["transfer_event_id"].astype(str)
    cohort = audit.loc[as_bool(audit["eligible_dual_capital_at_risk_24m"])].copy()
    cohort["transfer_event_id"] = cohort["transfer_event_id"].astype(str)
    value_down = pd.to_numeric(cohort["market_value_change_24m_pct"], errors="coerce").le(-0.25).astype(int)
    opportunity = pd.to_numeric(cohort["destination_opportunity_share_24m"], errors="coerce")
    under10 = opportunity.lt(0.10).astype(int)
    under25 = opportunity.lt(0.25).astype(int)
    severe = (value_down & under10).astype(int)
    meaningful = (value_down & under25).astype(int)
    add(checks, "strict_cohort_rows", len(cohort) == 827, len(cohort), 827, "Frozen dual-eligible cohort reproduced from the audit.")
    add(checks, "strict_severe_events", int(severe.sum()) == 110, int(severe.sum()), 110, "Severe joint outcome independently reproduced.")
    add(checks, "strict_meaningful_events", int(meaningful.sum()) == int(targets["target_dual_meaningful"].sum()), int(meaningful.sum()), int(targets["target_dual_meaningful"].sum()), "Meaningful joint outcome independently reproduced.")
    add(checks, "target_rows_and_keys", len(targets) == 827 and targets["transfer_event_id"].is_unique, f"{len(targets)};{targets['transfer_event_id'].nunique()}", "827;827", "One target row per audited deal.")
    add(checks, "target_ids_exact", set(targets["transfer_event_id"]) == set(cohort["transfer_event_id"]), len(set(targets["transfer_event_id"]) ^ set(cohort["transfer_event_id"])), 0, "No audited IDs added or lost.")
    target_match = targets.merge(
        cohort[["transfer_event_id"]].assign(
            expected_value=value_down.to_numpy(), expected_under10=under10.to_numpy(),
            expected_under25=under25.to_numpy(), expected_severe=severe.to_numpy(),
            expected_meaningful=meaningful.to_numpy(),
        ), on="transfer_event_id", validate="one_to_one",
    )
    target_exact = (
        target_match["target_value_downside_25pct"].eq(target_match["expected_value"]).all()
        and target_match["target_underutilization_10pct"].eq(target_match["expected_under10"]).all()
        and target_match["target_underutilization_25pct"].eq(target_match["expected_under25"]).all()
        and target_match["target_dual_severe"].eq(target_match["expected_severe"]).all()
        and target_match["target_dual_meaningful"].eq(target_match["expected_meaningful"]).all()
    )
    add(checks, "targets_recomputed_exact", target_exact, target_exact, True, "All five binary outcomes match the frozen thresholds.")

    add(checks, "card_rows_and_keys", len(cards) == 446 and cards["transfer_event_id"].is_unique, f"{len(cards)};{cards['transfer_event_id'].nunique()}", "446;446", "One fully out-of-season integration card per deal.")
    year_counts = cards.groupby("evaluation_year").size().to_dict()
    add(checks, "evaluation_year_counts", year_counts == EXPECTED_YEAR_COUNTS, json.dumps(year_counts, sort_keys=True), json.dumps(EXPECTED_YEAR_COUNTS, sort_keys=True), "Four rolling-origin evaluation counts exact.")
    add(checks, "evaluation_players", cards["player_id"].nunique() == 397, cards["player_id"].nunique(), 397, "Player-cluster count exact.")
    add(checks, "evaluation_event_counts", cards["target_dual_severe"].sum() == 78 and cards["target_dual_meaningful"].sum() == 128, f"{cards['target_dual_severe'].sum()};{cards['target_dual_meaningful'].sum()}", "78;128", "Evaluation outcome counts exact.")
    card_target_match = cards.merge(
        targets[["transfer_event_id", "target_value_downside_25pct", "target_underutilization_10pct", "target_underutilization_25pct", "target_dual_severe", "target_dual_meaningful"]],
        on="transfer_event_id", suffixes=("_card", "_target"), validate="one_to_one",
    )
    outcomes_exact = all(
        card_target_match[f"{column}_card"].eq(card_target_match[f"{column}_target"]).all()
        for column in ["target_value_downside_25pct", "target_underutilization_10pct", "target_underutilization_25pct", "target_dual_severe", "target_dual_meaningful"]
    )
    add(checks, "card_outcomes_exact", outcomes_exact, outcomes_exact, True, "Cards retain audited outcomes exactly.")

    market = pd.read_csv(MARKET_DIR / "calibrated_predictions.csv", low_memory=False)
    market = market.loc[market["strategy"].eq("policy_selected") & market["endpoint"].eq("downside_25pct"), ["transfer_event_id", "probability"]]
    under = pd.read_csv(UNDER_DIR / "underutilization_predictions.csv", low_memory=False)
    under = under.loc[under["model_variant"].eq("M_policy_ordered") & under["endpoint"].eq("underutilization_10pct"), ["transfer_event_id", "prediction"]]
    refined = pd.read_csv(REFINEMENT_DIR / "classification_predictions.csv", low_memory=False)
    refined = refined.loc[refined["strategy"].eq("selected_compact_calibrated") & refined["endpoint"].eq("underutilization_25pct"), ["transfer_event_id", "probability"]]
    fee = pd.read_csv(FEE_DIR / "fee_benchmark_predictions.csv", low_memory=False)
    fee = fee.loc[fee["split"].eq("evaluation") & fee["model_variant"].eq("M1_market_profile"), ["transfer_event_id", "actual_fee_eur", "predicted_fee_eur"]]
    for frame in [market, under, refined, fee]:
        frame["transfer_event_id"] = frame["transfer_event_id"].astype(str)
    component_match = cards.merge(market, on="transfer_event_id", validate="one_to_one").merge(under, on="transfer_event_id", validate="one_to_one").merge(refined, on="transfer_event_id", suffixes=("_market", "_meaningful"), validate="one_to_one").merge(fee, on="transfer_event_id", validate="one_to_one")
    component_exact = (
        np.allclose(component_match["market_value_downside_probability_25pct"], component_match["probability_market"], atol=1e-12, rtol=0)
        and np.allclose(component_match["severe_underutilization_risk_score"], component_match["prediction"], atol=1e-12, rtol=0)
        and np.allclose(component_match["meaningful_underutilization_probability_25pct"], component_match["probability_meaningful"], atol=1e-12, rtol=0)
        and np.allclose(component_match["reported_fee_eur"], component_match["actual_fee_eur"], atol=1e-6, rtol=0)
        and np.allclose(component_match["fee_benchmark_eur"], component_match["predicted_fee_eur"], atol=1e-6, rtol=0)
    )
    add(checks, "component_values_exact", component_exact, component_exact, True, "All frozen component values and fees match their verified sources.")

    rank_specs = {
        "value_downside_rank": "market_value_downside_probability_25pct",
        "severe_underutilization_rank": "severe_underutilization_risk_score",
        "meaningful_underutilization_rank": "meaningful_underutilization_probability_25pct",
    }
    ranks_exact = True
    for output, source in rank_specs.items():
        expected = cards.groupby("origin_id")[source].rank(pct=True, method="average")
        ranks_exact &= np.allclose(cards[output], expected, atol=1e-12, rtol=0)
    add(checks, "within_origin_ranks_exact", ranks_exact, ranks_exact, True, "Percentile normalization uses only each evaluation origin.")
    rank_columns = list(rank_specs)
    add(checks, "rank_bounds", cards[rank_columns].gt(0).all().all() and cards[rank_columns].le(1).all().all(), f"{cards[rank_columns].min().min()}..{cards[rank_columns].max().max()}", "(0,1]", "All component ranks are bounded.")
    severe_geo = np.sqrt(cards["value_downside_rank"] * cards["severe_underutilization_rank"])
    meaningful_geo = np.sqrt(cards["value_downside_rank"] * cards["meaningful_underutilization_rank"])
    add(checks, "geometric_scores_exact", np.allclose(cards["severe_joint_priority_score"], severe_geo, atol=1e-12, rtol=0) and np.allclose(cards["meaningful_joint_priority_score"], meaningful_geo, atol=1e-12, rtol=0), True, True, "Frozen geometric integration formula exact.")
    add(checks, "capital_indices_exact", np.allclose(cards["severe_capital_priority_index_eur"], cards["reported_fee_eur"] * severe_geo, atol=1e-6, rtol=0) and np.allclose(cards["meaningful_capital_priority_index_eur"], cards["reported_fee_eur"] * meaningful_geo, atol=1e-6, rtol=0), True, True, "Euro fields are explicitly fee-weighted rank indices.")
    forbidden = [column for column in cards if "joint_probability" in column or "expected_loss" in column or "roi" in column.lower()]
    add(checks, "no_unsupported_financial_claim_columns", not forbidden, ";".join(forbidden), "none", "No joint probability, expected loss, or ROI is emitted.")

    metric_ok = True
    for row in metrics.itertuples(index=False):
        years = [int(value) for value in str(row.evaluation_years).split(";")]
        group = cards.loc[cards["evaluation_year"].isin(years)]
        actual = "target_dual_severe" if row.target == "dual_severe" else "target_dual_meaningful"
        auc = binary_auc(group[actual], group[row.score_column])
        ap = average_precision_score(group[actual], group[row.score_column])
        metric_ok &= len(group) == row.rows and int(group[actual].sum()) == row.positives and close(auc, row.roc_auc) and close(ap, row.average_precision)
    add(checks, "all_score_metrics_recomputed", metric_ok, metric_ok, True, "Every pooled, window, and origin AUC/AP is exact.")

    comparison_ok = True
    for row in comparisons.itertuples(index=False):
        actual = "target_dual_severe" if row.target == "dual_severe" else "target_dual_meaningful"
        baseline_auc = binary_auc(cards[actual], cards[row.baseline_score_column])
        candidate_auc = binary_auc(cards[actual], cards[row.candidate_score_column])
        origin_wins = sum(
            binary_auc(group[actual], group[row.candidate_score_column])
            > binary_auc(group[actual], group[row.baseline_score_column])
            for _, group in cards.groupby("origin_id")
        )
        comparison_ok &= close(baseline_auc, row.baseline_auc) and close(candidate_auc, row.candidate_auc) and close(candidate_auc - baseline_auc, row.auc_improvement) and origin_wins == row.origin_wins
    add(checks, "comparison_point_estimates_recomputed", comparison_ok, comparison_ok, True, "All AUC changes and origin wins are exact.")
    add(checks, "comparison_bootstrap_complete", len(comparisons) == 4 and comparisons["bootstrap_repetitions"].eq(5000).all() and comparisons["cluster_count"].eq(397).all(), f"{len(comparisons)};{comparisons['bootstrap_repetitions'].min()};{comparisons['cluster_count'].min()}", "4;5000;397", "Four player-cluster bootstrap comparisons retained.")
    severe_value = comparisons.loc[comparisons["target"].eq("dual_severe") & comparisons["comparison"].eq("geometric_vs_value")].iloc[0]
    severe_under = comparisons.loc[comparisons["target"].eq("dual_severe") & comparisons["comparison"].eq("geometric_vs_underutilization")].iloc[0]
    add(checks, "severe_joint_beats_underuse_only", severe_under["cluster_ci_lower_95"] > 0 and severe_under["origin_wins"] == 4, f"{severe_under['cluster_ci_lower_95']};{severe_under['origin_wins']}", ">0;4", "Joint ranking improves over the weaker opportunity-only severe rank.")
    add(checks, "severe_joint_increment_vs_value_uncertain", severe_value["cluster_ci_lower_95"] <= 0 <= severe_value["cluster_ci_upper_95"], f"{severe_value['cluster_ci_lower_95']};{severe_value['cluster_ci_upper_95']}", "CI crosses zero", "Primary incremental claim remains unsupported.")

    matrix_ok = True
    for row in matrix.itertuples(index=False):
        under_tier = "severe_underutilization_tier" if row.target == "dual_severe" else "meaningful_underutilization_tier"
        actual = "target_dual_severe" if row.target == "dual_severe" else "target_dual_meaningful"
        group = cards.loc[cards["value_downside_tier"].eq(row.value_downside_tier) & cards[under_tier].eq(row.underutilization_tier)]
        matrix_ok &= len(group) == row.rows and int(group[actual].sum()) == row.joint_event_count and close(group[actual].mean(), row.joint_event_rate)
    add(checks, "matrix_cells_recomputed", matrix_ok and len(matrix) == 18, f"{matrix_ok};{len(matrix)}", "true;18", "Both complete 3x3 matrices reconcile.")
    quadrant_ok = True
    for row in quadrants.itertuples(index=False):
        under_rank = "severe_underutilization_rank" if row.target == "dual_severe" else "meaningful_underutilization_rank"
        actual = "target_dual_severe" if row.target == "dual_severe" else "target_dual_meaningful"
        value_high = cards["value_downside_rank"].ge(0.50)
        under_high = cards[under_rank].ge(0.50)
        masks = {
            "low_low": ~value_high & ~under_high,
            "utilization_only_high": ~value_high & under_high,
            "value_only_high": value_high & ~under_high,
            "high_high": value_high & under_high,
        }
        group = cards.loc[masks[row.quadrant]]
        quadrant_ok &= len(group) == row.rows and int(group[actual].sum()) == row.joint_event_count and close(group[actual].mean(), row.joint_event_rate)
    add(checks, "quadrants_recomputed", quadrant_ok and len(quadrants) == 8, f"{quadrant_ok};{len(quadrants)}", "true;8", "Both median-cut matrices reconcile.")
    severe_high = quadrants.loc[quadrants["target"].eq("dual_severe") & quadrants["quadrant"].eq("high_high")].iloc[0]
    overall_severe_rate = cards["target_dual_severe"].mean()
    add(checks, "severe_high_high_concentration", severe_high["rows"] == 59 and severe_high["joint_event_rate"] > overall_severe_rate, f"{severe_high['rows']};{severe_high['joint_event_rate']};{overall_severe_rate}", "59;high-high > overall", "High/high quadrant concentrates severe outcomes.")
    add(checks, "quintile_totals", len(quintiles) == 10 and quintiles.groupby("target")["rows"].sum().eq(446).all(), f"{len(quintiles)};{quintiles.groupby('target')['rows'].sum().to_dict()}", "10;446 each", "Every card assigned to one origin-relative priority quintile.")
    severe_quintiles = quintiles.loc[quintiles["target"].eq("dual_severe")].sort_values("priority_quintile")
    add(checks, "severe_quintiles_not_strictly_monotonic", not severe_quintiles["joint_event_rate"].is_monotonic_increasing, severe_quintiles["joint_event_rate"].tolist(), "not strictly monotonic", "Known ranking limitation explicitly verified.")
    add(checks, "subgroups_are_diagnostic", set(subgroups["adequate_for_auc"].astype(str).str.lower()).issubset({"true", "false"}) and len(subgroups) > 0, len(subgroups), ">0", "Subgroup AUCs are gated by minimum support.")

    decision_row = decision.iloc[0]
    add(checks, "decision_status_conservative", decision_row["status"] == "useful_matrix_but_incremental_gain_uncertain", decision_row["status"], "useful_matrix_but_incremental_gain_uncertain", "Decision follows the primary CI result.")
    add(checks, "decision_rejects_probability_claim", not as_bool(pd.Series([decision_row["probability_claim_allowed"]])).iloc[0], decision_row["probability_claim_allowed"], False, "Combined score is not presented as a probability.")
    add(checks, "decision_numbers_reconcile", decision_row["evaluation_rows"] == len(cards) and decision_row["dual_severe_events"] == cards["target_dual_severe"].sum() and close(decision_row["severe_geometric_auc"], binary_auc(cards["target_dual_severe"], cards["severe_joint_priority_score"])), True, True, "Decision summary matches evaluated cards.")
    add(checks, "builder_checks_pass", as_bool(build_checks["passed"]).all(), int(as_bool(build_checks["passed"]).sum()), len(build_checks), "All builder integrity checks pass.")

    source_ok = True
    for item in sources.itertuples(index=False):
        path = ROOT / item.source
        source_ok &= path.exists() and path.stat().st_size == item.size_bytes and sha256_file(path) == item.sha256
    add(checks, "source_manifest_hashes", source_ok, source_ok, True, "All verified upstream inputs hash-match.")
    output_ok = True
    for item in manifest.itertuples(index=False):
        path = OUTPUT_DIR / item.file
        output_ok &= path.exists() and path.stat().st_size == item.size_bytes and sha256_file(path) == item.sha256
    add(checks, "output_manifest_hashes", output_ok, output_ok, True, "All integration outputs hash-match.")
    summary_ok = (
        summary["strict_cohort_rows"] == 827
        and summary["evaluation_rows"] == 446
        and summary["evaluation_dual_severe_events"] == 78
        and summary["decision"] == "useful_matrix_but_incremental_gain_uncertain"
        and summary["protected_outputs_unchanged"] is True
    )
    add(checks, "run_summary_reconciles", summary_ok, json.dumps(summary, sort_keys=True), "827;446;78;conservative decision;protected", "Run summary and protection status exact.")

    result = pd.DataFrame(checks)
    result.to_csv(OUTPUT_DIR / "independent_verification.csv", index=False)
    payload = {
        "status": "pass" if result["passed"].all() else "fail",
        "checks": len(result), "passed": int(result["passed"].sum()),
        "failed": result.loc[~result["passed"], "check"].tolist(),
    }
    (OUTPUT_DIR / "independent_verification.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise RuntimeError(f"Independent verification failed: {payload['failed']}")


if __name__ == "__main__":
    verify()

"""Independent verification for Engine V2 Phase 3.

This verifier reconstructs calibration metrics, method selection, support
counts, subgroup labels, and endpoint gates from row-level predictions.  It
does not import or call the Phase-3 runner.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.engine_v2.reliability_contract import (  # noqa: E402
    CALIBRATION_BRIER_NONINFERIORITY,
    CALIBRATION_METHODS,
    MAX_ACCEPTABLE_ECE,
    MAX_ACCEPTABLE_PARAMETER_INTERVAL_WIDTH,
    MAX_ACCEPTABLE_REFUSAL_RATE,
    MAX_SUPPORTED_CALIBRATION_GAP,
    MIN_SUBGROUP_EVENTS,
    MIN_SUBGROUP_NONEVENTS,
    MIN_SUBGROUP_ORIGINS,
    MIN_SUBGROUP_ROWS,
    MIN_SUPPORTED_AUC,
    MIN_SUPPORTED_SUBGROUP_SHARE,
    REQUIRED_SUPPORTED_GROUP_TYPES,
)

OUTPUT = ROOT / "Data" / "processed" / "engine_v2_calibration_reliability"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"
ENDPOINTS = {
    "sustained_meaningful_contribution",
    "any_outbound_24m",
    "downside_25pct_24m",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adaptive_ece(actual: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    data = pd.DataFrame({"actual": actual, "probability": probability}).sort_values("probability")
    groups = np.array_split(np.arange(len(data)), min(bins, len(data)))
    total = 0.0
    for indices in groups:
        group = data.iloc[indices]
        total += len(group) * abs(group["actual"].mean() - group["probability"].mean())
    return float(total / len(data))


def metrics(actual: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "brier": float(brier_score_loss(actual, probability)),
        "log_loss": float(log_loss(actual, probability, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(actual, probability)),
        "average_precision": float(average_precision_score(actual, probability)),
        "adaptive_ece": adaptive_ece(actual, probability),
        "calibration_gap": float(abs(actual.mean() - probability.mean())),
    }


def main() -> None:
    results: list[dict] = []

    def add(name: str, passed: bool, observed: object, expected: object, notes: str) -> None:
        results.append({
            "check": name,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
            "notes": notes,
        })

    all_predictions = pd.read_csv(OUTPUT / "calibration_predictions.csv")
    selected = pd.read_csv(OUTPUT / "selected_predictions.csv")
    origin_metrics = pd.read_csv(OUTPUT / "origin_calibration_metrics.csv")
    comparisons = pd.read_csv(OUTPUT / "calibration_method_comparison.csv")
    calibration_decisions = pd.read_csv(OUTPUT / "calibration_decisions.csv")
    support_summary = pd.read_csv(OUTPUT / "ood_support_summary.csv")
    subgroup = pd.read_csv(OUTPUT / "subgroup_reliability.csv")
    endpoint_decisions = pd.read_csv(OUTPUT / "phase3_decisions.csv")
    sources = pd.read_csv(OUTPUT / "source_manifest.csv")
    profiles = json.loads((OUTPUT / "ood_support_profiles.json").read_text(encoding="utf-8"))
    summary = json.loads((OUTPUT / "run_summary.json").read_text(encoding="utf-8"))

    add("three_endpoints_exact", set(selected["endpoint"]) == ENDPOINTS, sorted(selected["endpoint"].unique()), sorted(ENDPOINTS), "Only Phase-2-approved predictive candidates are present.")
    add("three_methods_exact", set(all_predictions["calibration_method"]) == set(CALIBRATION_METHODS), sorted(all_predictions["calibration_method"].unique()), sorted(CALIBRATION_METHODS), "Raw, Platt, and isotonic streams are retained for audit.")
    add("rolling_years_exact", set(selected["evaluation_year"]) == {2020, 2021, 2022, 2023}, sorted(selected["evaluation_year"].unique()), [2020, 2021, 2022, 2023], "Evaluation remains chronological.")
    all_key = ["endpoint", "origin_id", "calibration_method", "capology_extension_event_id"]
    selected_key = ["endpoint", "origin_id", "capology_extension_event_id"]
    add("all_prediction_keys_unique", not all_predictions.duplicated(all_key).any(), int(all_predictions.duplicated(all_key).sum()), 0, "One prediction per method/event/origin.")
    add("selected_prediction_keys_unique", not selected.duplicated(selected_key).any(), int(selected.duplicated(selected_key).sum()), 0, "One selected probability per event/origin.")
    add("probabilities_bounded", all_predictions["prediction"].between(0, 1).all() and selected["prediction"].between(0, 1).all(), [float(all_predictions["prediction"].min()), float(all_predictions["prediction"].max())], "[0,1]", "All probability streams are valid.")
    add("actuals_binary", set(all_predictions["actual"]) == {0, 1}, sorted(all_predictions["actual"].unique()), [0, 1], "All endpoints are binary.")

    chosen_map = calibration_decisions.set_index("endpoint")["selected_calibration"].to_dict()
    expected_selected_rows = all_predictions.loc[
        all_predictions.apply(lambda row: row["calibration_method"] == chosen_map[row["endpoint"]], axis=1)
    ]
    add("selected_rows_reconcile", len(expected_selected_rows) == len(selected), len(selected), len(expected_selected_rows), "Selected table has exactly one chosen-method stream per endpoint.")
    selected_merge = selected.merge(
        expected_selected_rows[selected_key + ["prediction"]],
        on=selected_key,
        suffixes=("_saved", "_source"),
        validate="one_to_one",
    )
    max_selected_diff = float((selected_merge["prediction_saved"] - selected_merge["prediction_source"]).abs().max())
    add("selected_probabilities_exact", max_selected_diff < 1e-12, max_selected_diff, "<1e-12", "Chosen probabilities are copied without alteration.")

    origin_diffs = []
    for row in origin_metrics.itertuples(index=False):
        chunk = all_predictions.loc[
            all_predictions["endpoint"].eq(row.endpoint)
            & all_predictions["origin_id"].eq(row.origin_id)
            & all_predictions["calibration_method"].eq(row.calibration_method)
        ]
        current = metrics(chunk["actual"], chunk["prediction"])
        for column in ["brier", "log_loss", "roc_auc", "average_precision", "adaptive_ece", "calibration_gap"]:
            origin_diffs.append(abs(current[column] - getattr(row, column)))
    add("origin_metrics_recompute", max(origin_diffs) < 1e-12, max(origin_diffs), "<1e-12", "Saved per-origin metrics derive from row-level probabilities.")

    comparison_diffs = []
    win_diffs = []
    reconstructed: dict[tuple[str, str], dict[str, float]] = {}
    for endpoint, data in all_predictions.groupby("endpoint"):
        raw = data.loc[data["calibration_method"].eq("raw")]
        raw_metrics = metrics(raw["actual"], raw["prediction"])
        for method in CALIBRATION_METHODS:
            chunk = data.loc[data["calibration_method"].eq(method)]
            current = metrics(chunk["actual"], chunk["prediction"])
            reconstructed[(endpoint, method)] = current
            row = comparisons.loc[
                comparisons["endpoint"].eq(endpoint)
                & comparisons["calibration_method"].eq(method)
            ].iloc[0]
            expected = {
                "pooled_brier": current["brier"],
                "pooled_log_loss": current["log_loss"],
                "pooled_ece": current["adaptive_ece"],
                "calibration_gap": current["calibration_gap"],
                "brier_improvement_vs_raw": raw_metrics["brier"] - current["brier"],
                "ece_improvement_vs_raw": raw_metrics["adaptive_ece"] - current["adaptive_ece"],
            }
            comparison_diffs.extend(abs(float(row[column]) - value) for column, value in expected.items())
            wins = 0
            if method != "raw":
                for _, origin in data.groupby("origin_id"):
                    raw_loss = origin.loc[origin["calibration_method"].eq("raw"), "brier_loss"].mean()
                    method_loss = origin.loc[origin["calibration_method"].eq(method), "brier_loss"].mean()
                    wins += int(raw_loss > method_loss)
            win_diffs.append(abs(int(row["origin_wins_vs_raw"]) - wins))
    add("pooled_calibration_metrics_recompute", max(comparison_diffs) < 1e-12, max(comparison_diffs), "<1e-12", "Pooled method comparison is independently reconstructed.")
    add("origin_wins_recompute", max(win_diffs) == 0, max(win_diffs), 0, "Origin wins derive from saved row losses.")

    selection_mismatches = []
    for endpoint in sorted(ENDPOINTS):
        table = comparisons.loc[comparisons["endpoint"].eq(endpoint)].copy()
        raw = table.loc[table["calibration_method"].eq("raw")].iloc[0]
        eligible = table.loc[
            table["pooled_brier"].le(raw["pooled_brier"] + CALIBRATION_BRIER_NONINFERIORITY)
            & table["pooled_log_loss"].le(raw["pooled_log_loss"] + 0.01)
        ].copy()
        eligible["simplicity"] = eligible["calibration_method"].map({"raw": 0, "platt": 1, "isotonic": 2})
        best_ece = eligible["pooled_ece"].min()
        expected = eligible.loc[eligible["pooled_ece"].le(best_ece + 0.005)].sort_values("simplicity").iloc[0]["calibration_method"]
        if chosen_map[endpoint] != expected:
            selection_mismatches.append({"endpoint": endpoint, "saved": chosen_map[endpoint], "expected": expected})
    add("calibration_selection_rule_recomputes", not selection_mismatches, selection_mismatches, [], "Declared non-inferiority and simplicity rule reproduces every choice.")

    add("intervals_ordered", (selected["parameter_interval_low_80"] <= selected["prediction"]).all() and (selected["prediction"] <= selected["parameter_interval_high_80"]).all(), int(((selected["parameter_interval_low_80"] > selected["prediction"]) | (selected["prediction"] > selected["parameter_interval_high_80"])).sum()), 0, "Point estimate lies inside every 80% parameter interval.")
    width_diff = float((selected["parameter_interval_high_80"] - selected["parameter_interval_low_80"] - selected["parameter_interval_width_80"]).abs().max())
    add("interval_widths_recompute", width_diff < 1e-12, width_diff, "<1e-12", "Widths equal upper minus lower bounds.")
    add("all_bootstraps_complete", selected["successful_bootstrap_fits"].eq(250).all(), int(selected["successful_bootstrap_fits"].min()), 250, "Every player-cluster bootstrap completed.")

    add("support_statuses_exact", set(selected["support_status"]) == {"supported", "limited", "refused"}, sorted(selected["support_status"].unique()), ["limited", "refused", "supported"], "All three product support states occur and are explicit.")
    refused_reasons_ok = selected.loc[selected["support_status"].eq("refused"), "support_reasons"].fillna("").str.len().gt(0).all()
    add("refused_have_reasons", refused_reasons_ok, bool(refused_reasons_ok), True, "The engine can explain every refusal.")
    support_recomputed = selected.groupby(["endpoint", "origin_id", "evaluation_year", "support_status"], as_index=False).size().rename(columns={"size": "rows"})
    totals = support_recomputed.groupby(["endpoint", "origin_id"])["rows"].transform("sum")
    support_recomputed["share"] = support_recomputed["rows"] / totals
    support_join = support_summary.merge(support_recomputed, on=["endpoint", "origin_id", "evaluation_year", "support_status"], suffixes=("_saved", "_calc"), validate="one_to_one")
    support_diff = max(
        float((support_join["rows_saved"] - support_join["rows_calc"]).abs().max()),
        float((support_join["share_saved"] - support_join["share_calc"]).abs().max()),
    )
    add("support_summary_recomputes", support_diff < 1e-12 and len(support_join) == len(support_summary), support_diff, "<1e-12", "Support shares derive from selected row statuses.")
    profile_thresholds_ok = all(item["distance_limited_threshold"] < item["distance_refusal_threshold"] for item in profiles)
    add("ood_thresholds_ordered", profile_thresholds_ok, bool(profile_thresholds_ok), True, "The limited envelope is strictly inside the refusal envelope.")
    add("support_profiles_complete", len(profiles) == 12, len(profiles), 12, "Three endpoints by four origins have learned support profiles.")

    subgroup_diffs = []
    status_mismatches = []
    for row in subgroup.itertuples(index=False):
        column = {"position": "canonical_position", "league": "league", "age": "age_band", "support": "support_status"}[row.group_type]
        chunk = selected.loc[selected["endpoint"].eq(row.endpoint) & selected[column].astype(str).eq(str(row.group))]
        n = len(chunk)
        events = int(chunk["actual"].sum())
        nonevents = n - events
        origins = chunk["origin_id"].nunique()
        auc = float(roc_auc_score(chunk["actual"], chunk["prediction"])) if events and nonevents else np.nan
        brier = float(np.mean((chunk["actual"] - chunk["prediction"]) ** 2))
        baseline_brier = float(np.mean((chunk["actual"] - chunk["baseline_prediction"]) ** 2))
        gap = float(abs(chunk["actual"].mean() - chunk["prediction"].mean()))
        subgroup_diffs.extend([
            abs(n - row.rows), abs(events - row.events), abs(nonevents - row.nonevents), abs(origins - row.origins),
            abs(brier - row.brier), abs(baseline_brier - row.baseline_brier), abs(gap - row.calibration_gap),
        ])
        if not (pd.isna(auc) and pd.isna(row.roc_auc)):
            subgroup_diffs.append(abs(auc - row.roc_auc))
        sufficient = n >= MIN_SUBGROUP_ROWS and events >= MIN_SUBGROUP_EVENTS and nonevents >= MIN_SUBGROUP_NONEVENTS and origins >= MIN_SUBGROUP_ORIGINS
        supported = bool(sufficient and auc >= MIN_SUPPORTED_AUC and gap <= MAX_SUPPORTED_CALIBRATION_GAP and brier < baseline_brier)
        expected = "supported" if supported else "limited_reliability" if sufficient else "insufficient_evidence"
        if row.reliability_status != expected:
            status_mismatches.append({"endpoint": row.endpoint, "group_type": row.group_type, "group": row.group, "saved": row.reliability_status, "expected": expected})
    add("subgroup_metrics_recompute", max(subgroup_diffs) < 1e-12, max(subgroup_diffs), "<1e-12", "Every subgroup count and metric derives from selected predictions.")
    add("subgroup_status_rules_recompute", not status_mismatches, status_mismatches, [], "Evidence thresholds reproduce every supported/limited label.")

    gate_mismatches = []
    for row in endpoint_decisions.itertuples(index=False):
        chunk = selected.loc[selected["endpoint"].eq(row.endpoint)]
        selected_method = calibration_decisions.loc[calibration_decisions["endpoint"].eq(row.endpoint)].iloc[0]
        refused_rate = float(chunk["support_status"].eq("refused").mean())
        width = float(chunk["parameter_interval_width_80"].mean())
        endpoint_groups = subgroup.loc[subgroup["endpoint"].eq(row.endpoint)]
        supported_share = float(endpoint_groups["reliability_status"].eq("supported").mean())
        core_supported = bool(
            endpoint_groups.loc[
                endpoint_groups["group_type"].isin(REQUIRED_SUPPORTED_GROUP_TYPES),
                "reliability_status",
            ].eq("supported").all()
        )
        expected = bool(
            selected_method["pooled_ece"] <= MAX_ACCEPTABLE_ECE
            and refused_rate <= MAX_ACCEPTABLE_REFUSAL_RATE
            and width <= MAX_ACCEPTABLE_PARAMETER_INTERVAL_WIDTH
            and supported_share >= MIN_SUPPORTED_SUBGROUP_SHARE
            and core_supported
        )
        if abs(float(row.supported_subgroup_share) - supported_share) > 1e-12:
            gate_mismatches.append({"endpoint": row.endpoint, "field": "supported_subgroup_share", "saved": row.supported_subgroup_share, "expected": supported_share})
        if bool(row.core_position_league_groups_supported) != core_supported:
            gate_mismatches.append({"endpoint": row.endpoint, "field": "core_position_league_groups_supported", "saved": bool(row.core_position_league_groups_supported), "expected": core_supported})
        if bool(row.phase3_gate_passed) != expected:
            gate_mismatches.append({"endpoint": row.endpoint, "saved": bool(row.phase3_gate_passed), "expected": expected})
    add("phase3_gates_recompute", not gate_mismatches, gate_mismatches, [], "Calibration, refusal, uncertainty, and subgroup gates reproduce independently.")
    outbound = endpoint_decisions.loc[endpoint_decisions["endpoint"].eq("any_outbound_24m")].iloc[0]
    add("continuity_block_is_real", not bool(outbound["phase3_gate_passed"]) and outbound["refused_profile_rate"] > MAX_ACCEPTABLE_REFUSAL_RATE, float(outbound["refused_profile_rate"]), f">{MAX_ACCEPTABLE_REFUSAL_RATE}", "Continuity is blocked by missing/out-of-domain live inputs, not presentation preference.")
    add("nothing_deployed", set(endpoint_decisions["deployment_status"]) == {"not_deployed"}, sorted(endpoint_decisions["deployment_status"].unique()), ["not_deployed"], "Phase 3 cannot modify production.")
    add("no_phase3_joblib", not list(OUTPUT.rglob("*.joblib")), [str(path) for path in OUTPUT.rglob("*.joblib")], [], "No research candidate is loadable as a deployed artifact.")

    bad_sources = []
    for row in sources.itertuples(index=False):
        path = ROOT / row.source
        if not path.exists() or sha256(path) != row.sha256:
            bad_sources.append(row.source)
    add("source_manifest_integrity", not bad_sources, bad_sources, [], "All declared Phase-3 inputs retain their hashes.")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    bad_frozen = [item["path"] for item in freeze["files"] if not (ROOT / item["path"]).exists() or sha256(ROOT / item["path"]) != item["sha256"]]
    add("v1_hashes_unchanged", not bad_frozen, bad_frozen, [], "Every Phase-0 protected file remains byte-identical.")
    add("summary_reconciles", summary["evaluation_rows"] == len(selected) and summary["checks_total"] == 10 and summary["all_checks_passed"], summary, "row count and 10/10 runner checks", "Top-level runner summary agrees with evidence.")

    checks = pd.DataFrame(results)
    checks.to_csv(OUTPUT / "independent_verification.csv", index=False, lineterminator="\n")
    payload = {
        "checks_passed": int(checks["passed"].sum()),
        "checks_total": int(len(checks)),
        "all_checks_passed": bool(checks["passed"].all()),
        "verified_scope": "phase3_calibration_uncertainty_subgroup_support_ood_refusal_and_v1_isolation",
    }
    (OUTPUT / "independent_verification.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest_rows = []
    for path in sorted(OUTPUT.iterdir()):
        if path.is_file() and path.name != "output_manifest.csv":
            manifest_rows.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest_rows).to_csv(OUTPUT / "output_manifest.csv", index=False, lineterminator="\n")
    print(json.dumps(payload, indent=2))
    if not payload["all_checks_passed"]:
        failed = checks.loc[~checks["passed"], "check"].tolist()
        raise SystemExit(f"Failed checks: {failed}")


if __name__ == "__main__":
    main()

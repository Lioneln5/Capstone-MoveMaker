"""Build presentation-ready artifacts from the frozen calibrated downside model."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

import run_market_value_downside_model as downside
from train_compatibility_models import sha256_file


MODEL_DIR = ROOT / "Data" / "processed" / "market_value_downside_model"
CALIBRATION_DIR = ROOT / "Data" / "processed" / "market_value_calibration_experiment"
CALIBRATION_VERIFICATION = CALIBRATION_DIR / "independent_verification.json"
OUTPUT_DIR = ROOT / "Data" / "processed" / "market_value_downside_product"
PRIMARY_ENDPOINT = "downside_25pct"
THRESHOLD_ENDPOINTS = ["downside_10pct", "downside_25pct", "downside_50pct"]
PROTECTED_DIRS = [MODEL_DIR, CALIBRATION_DIR, ROOT / "Data" / "processed" / "financial_module_feasibility"]


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


def build_decision_cards(frame: pd.DataFrame) -> pd.DataFrame:
    predictions = pd.read_csv(CALIBRATION_DIR / "calibrated_predictions.csv", low_memory=False)
    policy = predictions.loc[predictions["strategy"].eq("policy_selected")].copy()
    keys = [
        "origin_id", "evaluation_year", "transfer_event_id", "player_id", "transfer_date",
        "selected_origin_strategy",
    ]
    probability = policy.pivot(index=keys, columns="endpoint", values="probability").reset_index()
    probability = probability.rename(columns={endpoint: f"probability_{endpoint}" for endpoint in THRESHOLD_ENDPOINTS if endpoint in probability.columns})
    raw = policy.pivot(index=keys, columns="endpoint", values="raw_probability").reset_index()
    raw = raw.rename(columns={endpoint: f"raw_probability_{endpoint}" for endpoint in THRESHOLD_ENDPOINTS if endpoint in raw.columns})
    actual = policy.pivot(index=keys, columns="endpoint", values="actual").reset_index()
    actual = actual.rename(columns={endpoint: f"actual_{endpoint}" for endpoint in THRESHOLD_ENDPOINTS if endpoint in actual.columns})
    cards = probability.merge(raw, on=keys, validate="one_to_one").merge(actual, on=keys, validate="one_to_one")

    context_columns = [
        "transfer_event_id", "player_name", "from_club_name", "to_club_name",
        "age_at_transfer", "historical_position_group", "destination_competition_id",
        "valuation_pre_eur", "valuation_post24_nearest_eur", "canonical_transfer_fee_eur",
        "canonical_transfer_fee_status", "target_value_change_24m_pct",
        "target_log_value_ratio_24m",
    ]
    cards = cards.merge(frame[context_columns], on="transfer_event_id", validate="one_to_one")
    cards["primary_risk_band"] = cards.groupby("origin_id")["probability_downside_25pct"].transform(
        lambda values: pd.qcut(values.rank(method="first"), 5, labels=False, duplicates="drop") + 1
    ).astype(int)
    labels = {1: "very_low", 2: "low", 3: "medium", 4: "high", 5: "very_high"}
    cards["primary_risk_band_label"] = cards["primary_risk_band"].map(labels)
    cards["primary_probability_calibration_adjustment"] = (
        cards["probability_downside_25pct"] - cards["raw_probability_downside_25pct"]
    )
    fee = pd.to_numeric(cards["canonical_transfer_fee_eur"], errors="coerce")
    cards["reported_fee_exposure_index_eur"] = (
        fee * cards["probability_downside_25pct"]
    ).where(fee.gt(0))
    cards["primary_forecast_at_50pct_cutoff"] = cards["probability_downside_25pct"].ge(0.50).astype(int)
    cards["primary_forecast_correct_at_50pct_cutoff"] = cards[
        "primary_forecast_at_50pct_cutoff"
    ].eq(cards["actual_downside_25pct"].astype(int))
    cards["output_guardrail"] = (
        "Estimated market-value downside screen; not sale proceeds, profit, causality, or ROI."
    )
    preferred = [
        "transfer_event_id", "player_id", "player_name", "transfer_date", "evaluation_year",
        "origin_id", "from_club_name", "to_club_name", "historical_position_group",
        "age_at_transfer", "destination_competition_id", "valuation_pre_eur",
        "canonical_transfer_fee_eur", "canonical_transfer_fee_status",
        "selected_origin_strategy", "probability_downside_10pct",
        "probability_downside_25pct", "probability_downside_50pct",
        "primary_risk_band", "primary_risk_band_label", "reported_fee_exposure_index_eur",
        "raw_probability_downside_25pct", "primary_probability_calibration_adjustment",
        "actual_downside_10pct", "actual_downside_25pct", "actual_downside_50pct",
        "valuation_post24_nearest_eur", "target_value_change_24m_pct",
        "primary_forecast_at_50pct_cutoff", "primary_forecast_correct_at_50pct_cutoff",
        "output_guardrail",
    ]
    return cards[preferred].sort_values(["evaluation_year", "probability_downside_25pct"], ascending=[True, False])


def add_profile_fields(cards: pd.DataFrame) -> pd.DataFrame:
    result = cards.copy()
    result["age_band"] = pd.cut(result["age_at_transfer"], [-np.inf, 20, 24, 28, np.inf], labels=["20_or_younger", "21_to_24", "25_to_28", "29_or_older"])
    result["pre_value_band"] = pd.cut(pd.to_numeric(result["valuation_pre_eur"], errors="coerce"), [-np.inf, 1e6, 5e6, 15e6, np.inf], labels=["under_1m", "1m_to_5m", "5m_to_15m", "15m_plus"], right=False)
    result["fee_evidence"] = np.where(pd.to_numeric(result["canonical_transfer_fee_eur"], errors="coerce").gt(0), "positive_reported", "zero_or_unclassified")
    return result


def build_driver_profiles(cards: pd.DataFrame) -> pd.DataFrame:
    work = add_profile_fields(cards)
    dimensions = [
        "historical_position_group", "age_band", "destination_competition_id",
        "pre_value_band", "fee_evidence", "evaluation_year",
    ]
    rows = []
    for dimension in dimensions:
        for category, group in work.groupby(dimension, dropna=False, observed=True):
            rows.append({
                "dimension": dimension, "category": str(category), "rows": len(group),
                "unique_players": group["player_id"].nunique(),
                "mean_probability_10pct": group["probability_downside_10pct"].mean(),
                "mean_probability_25pct": group["probability_downside_25pct"].mean(),
                "mean_probability_50pct": group["probability_downside_50pct"].mean(),
                "observed_25pct_rate": group["actual_downside_25pct"].mean(),
                "calibration_gap_25pct": group["actual_downside_25pct"].mean() - group["probability_downside_25pct"].mean(),
                "median_actual_value_change_pct": group["target_value_change_24m_pct"].median(),
                "interpretation": "Descriptive model behavior and outcome profile; not a causal driver estimate.",
            })
    return pd.DataFrame(rows)


def choose_cases(cards: pd.DataFrame, category: str, count: int = 3) -> pd.DataFrame:
    if category == "correct_warning":
        candidates = cards.loc[cards["actual_downside_25pct"].eq(1)].sort_values("probability_downside_25pct", ascending=False)
    elif category == "correct_low_risk":
        candidates = cards.loc[cards["actual_downside_25pct"].eq(0)].sort_values("probability_downside_25pct", ascending=True)
    elif category == "false_alarm":
        candidates = cards.loc[cards["actual_downside_25pct"].eq(0)].sort_values("probability_downside_25pct", ascending=False)
    elif category == "missed_downside":
        candidates = cards.loc[cards["actual_downside_25pct"].eq(1)].sort_values("probability_downside_25pct", ascending=True)
    else:
        raise ValueError(category)
    selected = []
    positions: set[str] = set()
    players: set[str] = set()
    for require_new_position in [True, False]:
        for _, row in candidates.iterrows():
            player = str(row["player_id"])
            position = str(row["historical_position_group"])
            if player in players or (require_new_position and position in positions):
                continue
            selected.append(row)
            players.add(player)
            positions.add(position)
            if len(selected) == count:
                result = pd.DataFrame(selected)
                result.insert(0, "case_category", category)
                return result
    raise RuntimeError(f"Could not select {count} cases for {category}.")


def build_case_portfolio(cards: pd.DataFrame) -> pd.DataFrame:
    portfolio = pd.concat(
        [choose_cases(cards, category) for category in ["correct_warning", "correct_low_risk", "false_alarm", "missed_downside"]],
        ignore_index=True,
    )
    narratives = {
        "correct_warning": "The calibrated model issued a high downside warning and the estimated market value later fell by at least 25%.",
        "correct_low_risk": "The model assigned low downside risk and the estimated market value avoided a 25% decline.",
        "false_alarm": "The model issued a high warning, but the estimated market value avoided a 25% decline; this is a conservative false positive.",
        "missed_downside": "The model assigned relatively low risk, but the estimated market value later fell by at least 25%; this exposes a model limitation.",
    }
    portfolio["case_interpretation"] = portfolio["case_category"].map(narratives)
    portfolio["case_rank_within_category"] = portfolio.groupby("case_category").cumcount() + 1
    return portfolio[
        ["case_category", "case_rank_within_category", "transfer_event_id", "player_name", "transfer_date",
         "from_club_name", "to_club_name", "historical_position_group", "age_at_transfer",
         "valuation_pre_eur", "canonical_transfer_fee_eur", "probability_downside_10pct",
         "probability_downside_25pct", "probability_downside_50pct", "primary_risk_band_label",
         "actual_downside_25pct", "target_value_change_24m_pct", "case_interpretation", "output_guardrail"]
    ]


def build_contract(feature_manifest: pd.DataFrame) -> pd.DataFrame:
    inputs = feature_manifest.loc[
        feature_manifest["feature_set"].eq("M1_market_profile") & feature_manifest["feature"].notna(),
        ["feature", "timing_classification", "rationale"],
    ].copy()
    rows = [
        {
            "field": row.feature, "field_role": "input", "data_type": "numeric_or_categorical",
            "definition": row.rationale, "timing_rule": row.timing_classification,
            "allowed_use": "Frozen pre-transfer risk predictor.", "forbidden_claim": "No causal interpretation.",
        }
        for row in inputs.itertuples(index=False)
    ]
    output_specs = [
        ("probability_downside_10pct", "Calibrated probability of estimated value declining at least 10% within 24 months."),
        ("probability_downside_25pct", "Calibrated probability of estimated value declining at least 25% within 24 months."),
        ("probability_downside_50pct", "Calibrated probability of estimated value declining at least 50% within 24 months."),
        ("primary_risk_band_label", "Within-origin quintile of the primary 25% downside probability."),
        ("reported_fee_exposure_index_eur", "Reported positive fee multiplied by 25% downside probability; an exposure index, not expected loss."),
    ]
    for field, definition in output_specs:
        rows.append({
            "field": field, "field_role": "output", "data_type": "numeric_or_label",
            "definition": definition, "timing_rule": "generated at scoring time",
            "allowed_use": "Screening, ranking, due-diligence prioritization, and portfolio monitoring.",
            "forbidden_claim": "Not sale proceeds, profit, realized loss, causal effect, or accounting ROI.",
        })
    return pd.DataFrame(rows)


def build_checks(cards: pd.DataFrame, cases: pd.DataFrame, contract: pd.DataFrame) -> pd.DataFrame:
    checks = []
    def add(check: str, passed: bool, observed: Any, expected: Any, note: str) -> None:
        checks.append({"check": check, "passed": bool(passed), "observed": observed, "expected": expected, "note": note})
    add("decision_card_rows", len(cards) == 2686, len(cards), 2686, "Every pooled out-of-season transfer has one card.")
    add("decision_card_key_unique", cards["transfer_event_id"].is_unique, cards["transfer_event_id"].nunique(), len(cards), "One card per event.")
    nested = (cards["probability_downside_50pct"] <= cards["probability_downside_25pct"] + 1e-12).all() and (cards["probability_downside_25pct"] <= cards["probability_downside_10pct"] + 1e-12).all()
    add("probabilities_nested", nested, nested, True, "Calibrated thresholds are logically ordered.")
    add("risk_bands_complete", set(cards["primary_risk_band"]) == {1, 2, 3, 4, 5}, ";".join(map(str, sorted(set(cards["primary_risk_band"])))), "1;2;3;4;5", "Five risk bands are present.")
    exposure = pd.to_numeric(cards["canonical_transfer_fee_eur"], errors="coerce") * cards["probability_downside_25pct"]
    observed = cards["reported_fee_exposure_index_eur"]
    mask = observed.notna()
    add("exposure_index_formula", np.allclose(observed[mask], exposure[mask], atol=1e-8), float(np.abs(observed[mask] - exposure[mask]).max()), 0, "Exposure index is formulaic and labeled.")
    add("case_portfolio_rows", len(cases) == 12, len(cases), 12, "Three cases in four outcome categories.")
    add("case_category_balance", cases.groupby("case_category").size().eq(3).all(), cases.groupby("case_category").size().to_dict(), "3 each", "Case types are balanced.")
    add("contract_input_count", int(contract["field_role"].eq("input").sum()) == 11, int(contract["field_role"].eq("input").sum()), 11, "Frozen M1 inputs only.")
    add("guardrail_complete", cards["output_guardrail"].notna().all(), int(cards["output_guardrail"].notna().sum()), len(cards), "Every card carries the claim guardrail.")
    return pd.DataFrame(checks)


def write_readme(cards: pd.DataFrame, cases: pd.DataFrame) -> None:
    low = cards.loc[cards["primary_risk_band"].eq(1), "actual_downside_25pct"].mean()
    high = cards.loc[cards["primary_risk_band"].eq(5), "actual_downside_25pct"].mean()
    text = f"""# MoveMaker calibrated market-value downside product layer

This directory converts the independently validated downside model into presentation-ready historical outputs without retraining or changing it.

## Product output

Each of the {len(cards):,} out-of-season historical decision cards contains calibrated and logically ordered probabilities for estimated-value declines of at least 10%, 25%, and 50%, plus a primary 25% risk band. The lowest-risk band has a {low:.1%} observed 25% decline rate and the highest-risk band has a {high:.1%} rate.

`reported_fee_exposure_index_eur` equals a positive reported fee multiplied by the calibrated 25% downside probability. It is a prioritization index, **not** expected loss, sale proceeds, profit, or ROI.

## Artifacts

- `historical_decision_cards.csv`: complete out-of-season scoring layer.
- `risk_driver_profiles.csv`: descriptive behavior by position, age, competition, pre-value, fee evidence, and season; not causal importance.
- `case_study_portfolio.csv`: {len(cases)} balanced correct-warning, correct-low-risk, false-alarm, and missed-downside examples.
- `scoring_contract.csv`: frozen inputs, outputs, timing rules, allowed uses, and forbidden claims.
- `model_card.json`: versioned evidence and limitations.

This is an acquisition screening and due-diligence prioritization tool. It does not establish tactical fit, realized financial return, or a causal effect of joining a club.
"""
    (OUTPUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CALIBRATION_VERIFICATION.open("r", encoding="utf-8") as handle:
        if json.load(handle).get("status") != "pass":
            raise RuntimeError("Calibration experiment verification is not passing.")
    protected_before = protected_tree_sha256()
    frame, _, feature_manifest = downside.build_model_frame()
    cards = build_decision_cards(frame)
    profiles = build_driver_profiles(cards)
    cases = build_case_portfolio(cards)
    contract = build_contract(feature_manifest)
    checks = build_checks(cards, cases, contract)
    if not checks["passed"].all():
        raise RuntimeError(f"Product checks failed: {checks.loc[~checks['passed'], 'check'].tolist()}")
    tables = {
        "historical_decision_cards.csv": cards,
        "risk_driver_profiles.csv": profiles,
        "case_study_portfolio.csv": cases,
        "scoring_contract.csv": contract,
        "build_checks.csv": checks,
    }
    for name, table in tables.items():
        table.to_csv(OUTPUT_DIR / name, index=False)
    decision = pd.read_csv(CALIBRATION_DIR / "calibration_decision_summary.csv")
    comparison = pd.read_csv(CALIBRATION_DIR / "calibration_comparison_results.csv")
    primary_comparison = comparison.loc[
        comparison["strategy"].eq("policy_selected")
        & comparison["endpoint"].eq(PRIMARY_ENDPOINT)
        & comparison["window"].eq("pooled_four")
    ].iloc[0]
    model_card = {
        "model_name": "MoveMaker calibrated 24-month market-value downside",
        "version": "1.0-historical-validation",
        "validated_population": "Clean permanent 2017+ transfers with recent positive pre-value and strict 24-month valuation outcome.",
        "validated_evaluation_rows": len(cards),
        "primary_endpoint": PRIMARY_ENDPOINT,
        "primary_raw_brier": float(primary_comparison["raw_brier"]),
        "primary_calibrated_brier": float(primary_comparison["candidate_brier"]),
        "primary_weighted_origin_auc": float(primary_comparison["candidate_weighted_origin_auc"]),
        "primary_low_risk_band_observed_rate": float(cards.loc[cards["primary_risk_band"].eq(1), "actual_downside_25pct"].mean()),
        "primary_high_risk_band_observed_rate": float(cards.loc[cards["primary_risk_band"].eq(5), "actual_downside_25pct"].mean()),
        "allowed_claim": "Estimated market-value downside screening and ranking.",
        "forbidden_claims": ["realized sale proceeds", "profit", "ROI", "causal club effect", "universal tactical compatibility"],
        "live_deployment_status": "historically validated; live scoring package not yet frozen",
    }
    (OUTPUT_DIR / "model_card.json").write_text(json.dumps(model_card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(cards, cases)
    if protected_before != protected_tree_sha256():
        raise RuntimeError("A protected verified output changed while building the product layer.")
    source_paths = [
        CALIBRATION_DIR / "calibrated_predictions.csv", CALIBRATION_DIR / "calibration_decision_summary.csv",
        CALIBRATION_VERIFICATION, MODEL_DIR / "market_value_feature_manifest.csv",
        downside.MASTER_PATH,
    ]
    source_manifest = pd.DataFrame([
        {"source": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in source_paths
    ])
    source_manifest.to_csv(OUTPUT_DIR / "source_manifest.csv", index=False)
    tables["source_manifest.csv"] = source_manifest
    run_summary = {
        "decision_cards": len(cards), "case_studies": len(cases),
        "primary_low_risk_observed_rate": model_card["primary_low_risk_band_observed_rate"],
        "primary_high_risk_observed_rate": model_card["primary_high_risk_band_observed_rate"],
        "protected_outputs_unchanged": True, "build_checks": len(checks),
        "build_checks_passed": int(checks["passed"].sum()),
    }
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(run_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_names = sorted(list(tables) + ["README.md", "model_card.json", "run_summary.json"])
    manifest = pd.DataFrame([
        {"file": name, "size_bytes": (OUTPUT_DIR / name).stat().st_size, "sha256": sha256_file(OUTPUT_DIR / name)}
        for name in manifest_names
    ])
    manifest.to_csv(OUTPUT_DIR / "output_manifest.csv", index=False)
    print(json.dumps(run_summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

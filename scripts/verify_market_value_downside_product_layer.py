"""Independently verify the calibrated downside product-layer artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "Data" / "processed" / "market_value_downside_model"
CALIBRATION_DIR = ROOT / "Data" / "processed" / "market_value_calibration_experiment"
OUTPUT_DIR = ROOT / "Data" / "processed" / "market_value_downside_product"


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").fillna("").str.strip().str.lower().isin({"true", "1", "yes"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add(checks: list[dict[str, Any]], check: str, passed: bool, observed: Any, expected: Any, note: str) -> None:
    checks.append({"check": check, "passed": bool(passed), "observed": observed, "expected": expected, "note": note})


def verify() -> None:
    required = [
        "README.md", "run_summary.json", "model_card.json", "historical_decision_cards.csv",
        "risk_driver_profiles.csv", "case_study_portfolio.csv", "scoring_contract.csv",
        "build_checks.csv", "source_manifest.csv", "output_manifest.csv",
    ]
    checks: list[dict[str, Any]] = []
    missing = [name for name in required if not (OUTPUT_DIR / name).exists()]
    add(checks, "required_outputs_exist", not missing, ";".join(missing), "", "Every declared artifact exists.")
    if missing:
        raise RuntimeError(f"Missing outputs: {missing}")

    cards = pd.read_csv(OUTPUT_DIR / "historical_decision_cards.csv", low_memory=False)
    profiles = pd.read_csv(OUTPUT_DIR / "risk_driver_profiles.csv", low_memory=False)
    cases = pd.read_csv(OUTPUT_DIR / "case_study_portfolio.csv", low_memory=False)
    contract = pd.read_csv(OUTPUT_DIR / "scoring_contract.csv", low_memory=False)
    build_checks = pd.read_csv(OUTPUT_DIR / "build_checks.csv")
    sources = pd.read_csv(OUTPUT_DIR / "source_manifest.csv")
    manifest = pd.read_csv(OUTPUT_DIR / "output_manifest.csv")
    model_card = json.loads((OUTPUT_DIR / "model_card.json").read_text(encoding="utf-8"))
    run = json.loads((OUTPUT_DIR / "run_summary.json").read_text(encoding="utf-8"))

    add(checks, "card_rows", len(cards) == 2686, len(cards), 2686, "Complete pooled out-of-season population.")
    add(checks, "card_key_unique", cards["transfer_event_id"].is_unique, cards["transfer_event_id"].nunique(), len(cards), "One card per historical event.")
    probabilities = cards[["probability_downside_10pct", "probability_downside_25pct", "probability_downside_50pct"]]
    add(checks, "probability_bounds", probabilities.apply(lambda column: column.between(0, 1).all()).all(), f"{probabilities.min().min()}..{probabilities.max().max()}", "0..1", "Every product probability is valid.")
    nested = (cards["probability_downside_50pct"] <= cards["probability_downside_25pct"] + 1e-12).all() and (cards["probability_downside_25pct"] <= cards["probability_downside_10pct"] + 1e-12).all()
    add(checks, "probabilities_nested", nested, nested, True, "Threshold probabilities are logically ordered.")

    calibrated = pd.read_csv(CALIBRATION_DIR / "calibrated_predictions.csv", low_memory=False)
    policy = calibrated.loc[calibrated["strategy"].eq("policy_selected")]
    source_wide = policy.pivot(index="transfer_event_id", columns="endpoint", values="probability")
    joined = cards.set_index("transfer_event_id").join(source_wide, how="left", validate="one_to_one")
    source_match = all(
        np.allclose(joined[f"probability_{endpoint}"], joined[endpoint], atol=1e-12)
        for endpoint in ["downside_10pct", "downside_25pct", "downside_50pct"]
    )
    add(checks, "calibrated_probabilities_exact", source_match, source_match, True, "Cards reproduce the independently verified calibrated outputs.")
    add(checks, "source_membership_exact", set(cards["transfer_event_id"].astype(str)) == set(policy["transfer_event_id"].astype(str)), len(cards), policy["transfer_event_id"].nunique(), "No evaluated event is added or omitted.")

    target_source = pd.read_csv(MODEL_DIR / "market_value_downside_targets.csv", low_memory=False)
    target_columns = ["transfer_event_id", "target_downside_10pct", "target_downside_25pct", "target_downside_50pct", "target_value_change_24m_pct"]
    target_pair = cards.merge(target_source[target_columns], on="transfer_event_id", validate="one_to_one")
    target_ok = all(
        target_pair[f"actual_downside_{threshold}pct"].astype(int).equals(target_pair[f"target_downside_{threshold}pct"].astype(int))
        for threshold in [10, 25, 50]
    ) and np.allclose(target_pair["target_value_change_24m_pct_x"], target_pair["target_value_change_24m_pct_y"], atol=1e-12)
    add(checks, "outcomes_exact", target_ok, target_ok, True, "Historical outcomes match the verified target layer.")

    recomputed_band = cards.groupby("origin_id")["probability_downside_25pct"].transform(
        lambda values: pd.qcut(values.rank(method="first"), 5, labels=False, duplicates="drop") + 1
    ).astype(int)
    add(checks, "risk_bands_recomputed", recomputed_band.equals(cards["primary_risk_band"].astype(int)), recomputed_band.value_counts().sort_index().to_dict(), cards["primary_risk_band"].value_counts().sort_index().to_dict(), "Bands are within-origin quintiles.")
    label_map = {1: "very_low", 2: "low", 3: "medium", 4: "high", 5: "very_high"}
    add(checks, "risk_band_labels", cards["primary_risk_band"].map(label_map).equals(cards["primary_risk_band_label"]), True, True, "Band names match numbers.")
    low_rate = cards.loc[cards["primary_risk_band"].eq(1), "actual_downside_25pct"].mean()
    high_rate = cards.loc[cards["primary_risk_band"].eq(5), "actual_downside_25pct"].mean()
    add(checks, "risk_separation", high_rate > low_rate, f"{low_rate:.6f}..{high_rate:.6f}", "high > low", "Product bands preserve strong observed separation.")

    fee = pd.to_numeric(cards["canonical_transfer_fee_eur"], errors="coerce")
    expected_exposure = fee * cards["probability_downside_25pct"]
    exposure_mask = cards["reported_fee_exposure_index_eur"].notna()
    exposure_ok = np.allclose(cards.loc[exposure_mask, "reported_fee_exposure_index_eur"], expected_exposure[exposure_mask], atol=1e-8) and (~exposure_mask | fee.gt(0)).all()
    add(checks, "exposure_index_exact", exposure_ok, exposure_ok, True, "Exposure index uses only positive reported fees and is formulaic.")
    guardrail_ok = cards["output_guardrail"].str.contains("not sale proceeds", case=False, na=False).all() and cards["output_guardrail"].str.contains("ROI", case=False, na=False).all()
    add(checks, "card_guardrails", guardrail_ok, guardrail_ok, True, "Every card forbids cash-return interpretation.")

    add(checks, "case_rows", len(cases) == 12, len(cases), 12, "Balanced twelve-case portfolio.")
    expected_categories = {"correct_warning", "correct_low_risk", "false_alarm", "missed_downside"}
    add(checks, "case_categories", set(cases["case_category"]) == expected_categories and cases.groupby("case_category").size().eq(3).all(), cases.groupby("case_category").size().to_dict(), "3 each", "Successes and failures are balanced.")
    add(checks, "case_membership", set(cases["transfer_event_id"]).issubset(set(cards["transfer_event_id"])), len(set(cases["transfer_event_id"]) - set(cards["transfer_event_id"])), 0, "Cases come only from evaluated cards.")
    case_rules = (
        cases.loc[cases["case_category"].eq("correct_warning"), "actual_downside_25pct"].eq(1).all()
        and cases.loc[cases["case_category"].eq("correct_low_risk"), "actual_downside_25pct"].eq(0).all()
        and cases.loc[cases["case_category"].eq("false_alarm"), "actual_downside_25pct"].eq(0).all()
        and cases.loc[cases["case_category"].eq("missed_downside"), "actual_downside_25pct"].eq(1).all()
    )
    add(checks, "case_outcome_rules", case_rules, case_rules, True, "Case labels match realized outcomes.")

    dimensions = {"historical_position_group", "age_band", "destination_competition_id", "pre_value_band", "fee_evidence", "evaluation_year"}
    add(checks, "profile_dimensions", set(profiles["dimension"]) == dimensions, ";".join(sorted(set(profiles["dimension"]))), ";".join(sorted(dimensions)), "Declared descriptive slices exist.")
    profile_totals = profiles.groupby("dimension")["rows"].sum()
    add(checks, "profile_totals", profile_totals.eq(len(cards)).all(), profile_totals.to_dict(), len(cards), "Each dimension partitions the full card population.")
    profile_guardrail = profiles["interpretation"].str.contains("not a causal", case=False, na=False).all()
    add(checks, "profile_guardrails", profile_guardrail, profile_guardrail, True, "Profiles are not mislabeled feature importance.")

    feature_manifest = pd.read_csv(MODEL_DIR / "market_value_feature_manifest.csv", low_memory=False)
    expected_inputs = set(feature_manifest.loc[feature_manifest["feature_set"].eq("M1_market_profile") & feature_manifest["feature"].notna(), "feature"])
    contract_inputs = set(contract.loc[contract["field_role"].eq("input"), "field"])
    add(checks, "contract_inputs_frozen", contract_inputs == expected_inputs and len(contract_inputs) == 11, ";".join(sorted(contract_inputs)), ";".join(sorted(expected_inputs)), "Contract exposes exactly the validated M1 fields.")
    add(checks, "contract_outputs", int(contract["field_role"].eq("output").sum()) == 5, int(contract["field_role"].eq("output").sum()), 5, "Probabilities, band, and exposure index are documented.")
    forbidden_claims = contract.loc[contract["field_role"].eq("output"), "forbidden_claim"].str.contains("ROI", case=False, na=False).all()
    add(checks, "contract_forbids_roi", forbidden_claims, forbidden_claims, True, "Every product output blocks ROI claims.")

    add(checks, "model_card_population", int(model_card["validated_evaluation_rows"]) == len(cards), model_card["validated_evaluation_rows"], len(cards), "Model card population is exact.")
    add(checks, "model_card_not_live", model_card["live_deployment_status"] == "historically validated; live scoring package not yet frozen", model_card["live_deployment_status"], "historically validated; live scoring package not yet frozen", "Historical validation is not mislabeled deployment.")
    add(checks, "model_card_band_rates", np.isclose(model_card["primary_low_risk_band_observed_rate"], low_rate) and np.isclose(model_card["primary_high_risk_band_observed_rate"], high_rate), f"{low_rate};{high_rate}", f"{model_card['primary_low_risk_band_observed_rate']};{model_card['primary_high_risk_band_observed_rate']}", "Headline separation recomputes.")
    add(checks, "runner_checks_pass", as_bool(build_checks["passed"]).all(), int(as_bool(build_checks["passed"]).sum()), len(build_checks), "All builder invariants passed.")

    source_ok = True
    for _, row in sources.iterrows():
        path = ROOT / row["source"]
        source_ok &= path.exists() and path.stat().st_size == int(row["size_bytes"])
        source_ok &= sha256_file(path) == row["sha256"]
    add(checks, "source_manifest_hashes", source_ok, source_ok, True, "Every frozen input matches.")
    manifest_ok = True
    for _, row in manifest.iterrows():
        path = OUTPUT_DIR / row["file"]
        manifest_ok &= path.exists() and path.stat().st_size == int(row["size_bytes"])
        manifest_ok &= sha256_file(path) == row["sha256"]
    add(checks, "output_manifest_hashes", manifest_ok, manifest_ok, True, "Every builder output matches.")
    add(checks, "protected_outputs_unchanged", run["protected_outputs_unchanged"] is True, run["protected_outputs_unchanged"], True, "Verified upstream outputs were not modified.")

    result = pd.DataFrame(checks)
    result.to_csv(OUTPUT_DIR / "independent_verification.csv", index=False)
    failures = result.loc[~result["passed"]]
    payload = {"status": "pass" if failures.empty else "fail", "checks": len(result), "passed": int(result["passed"].sum()), "failed": len(failures), "failed_checks": failures["check"].tolist()}
    (OUTPUT_DIR / "independent_verification.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    if not failures.empty:
        raise RuntimeError(f"Verification failed: {failures['check'].tolist()}")


if __name__ == "__main__":
    verify()

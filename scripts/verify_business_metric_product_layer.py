"""Independently verify the frozen business-metric product layer."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.business_metric_engine import build_extension_business_profile


OUTPUT = ROOT / "Data" / "processed" / "business_metric_product"
AUDIT = ROOT / "Data" / "processed" / "business_metric_translation_audit"
SCOPE = "incumbent_club_extension"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").fillna("").str.strip().str.casefold().isin({"true", "1", "yes"})


def close(left: Any, right: Any, tolerance: float = 1e-8) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    return bool(np.isclose(float(left), float(right), atol=tolerance, rtol=tolerance))


def lookup(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {metric["metric_key"]: metric for card in profile["cards"].values() for metric in card}


def add(checks: list[dict[str, Any]], name: str, observed: Any, expected: Any, passed: bool, notes: str) -> None:
    checks.append({"check": name, "observed": observed, "expected": expected, "passed": bool(passed), "notes": notes})


def main() -> None:
    checks: list[dict[str, Any]] = []
    audit_verification = json.loads((AUDIT / "independent_verification.json").read_text(encoding="utf-8"))
    add(checks, "audit_source_verified", int(audit_verification.get("all_checks_passed", False)), 1, bool(audit_verification.get("all_checks_passed")), "The product consumes only the verified translation audit.")

    profiles = pd.read_csv(AUDIT / "translated_historical_profiles.csv", low_memory=False)
    residual = np.log(profiles["fixed_wage_commitment_eur"] / profiles["fixed_commitment_peer_estimate_eur"]).dropna().sort_values().to_numpy()
    reference = json.loads((OUTPUT / "offer_aggressiveness_reference.json").read_text(encoding="utf-8"))
    saved_residual = np.asarray(reference["sorted_log_residuals"], dtype=float)
    reference_ok = len(saved_residual) == len(residual) == 1504 and np.allclose(saved_residual, residual, atol=1e-12, rtol=1e-12)
    add(checks, "offer_reference_recomputation", len(saved_residual), 1504, reference_ok, "Frozen reference exactly reproduces the out-of-time peer residual distribution.")
    add(checks, "offer_reference_sorted", int(np.all(np.diff(saved_residual) >= 0)), 1, np.all(np.diff(saved_residual) >= 0), "Reference distribution is sorted.")
    quantile_mismatches = sum(not close(value, np.quantile(residual, float(key)), 1e-12) for key, value in reference["quantiles"].items())
    add(checks, "offer_reference_quantiles", quantile_mismatches, 0, quantile_mismatches == 0, "Published reference quantiles reproduce.")

    reference_csv = pd.read_csv(OUTPUT / "offer_aggressiveness_reference.csv", low_memory=False)
    reference_csv_ok = (
        len(reference_csv) == 1504
        and np.allclose(reference_csv["log_commitment_residual"], residual)
        and np.allclose(reference_csv["percentile"], np.arange(1, 1505) / 1504)
        and np.allclose(reference_csv["commitment_multiple_vs_peer"], np.exp(residual))
    )
    add(checks, "offer_reference_table", int(reference_csv_ok), 1, reference_csv_ok, "Human-readable CDF table reproduces.")

    schema = json.loads((OUTPUT / "input_schema.json").read_text(encoding="utf-8"))
    required = {
        "profile_scope", "age_at_decision", "proposed_annual_fixed_wage_eur",
        "proposed_contract_years", "current_public_market_value_eur",
    }
    optional_backend = {
        "peer_fixed_commitment_estimate_eur", "sustained_contribution_probability_24m",
        "continuous_stay_probability_36m", "public_value_downside_25pct_probability_24m",
        "club_salary_percentile", "club_known_payroll_share",
    }
    schema_ok = required == set(schema["required"]) and optional_backend.issubset(schema["properties"]) and schema["additionalProperties"] is False
    add(checks, "input_schema_scope", int(schema_ok), 1, schema_ok, "Required proposal fields and optional backend outputs are explicit.")

    contract = json.loads((OUTPUT / "output_contract.json").read_text(encoding="utf-8"))
    cards = {item["card"] for item in contract["cards"]}
    expected_cards = {"capital_commitment", "offer_context", "contribution_exposure", "asset_value", "continuity", "salary_structure"}
    contract_ok = cards == expected_cards and contract["overall_score"]["status"] == "rejected" and contract["overall_score"]["required_value"] is None
    add(checks, "output_contract_scope", int(contract_ok), 1, contract_ok, "Six business cards are frozen and the overall score remains rejected.")

    example_inputs = json.loads((OUTPUT / "historical_example_inputs.json").read_text(encoding="utf-8"))
    example_outputs = json.loads((OUTPUT / "historical_example_profiles.json").read_text(encoding="utf-8"))
    add(checks, "historical_example_count", len(example_outputs), 12, len(example_inputs) == len(example_outputs) == 12, "Input/output examples are paired.")
    categories = {item["case_category"] for item in example_inputs}
    add(checks, "historical_example_categories", len(categories), 6, len(categories) == 6, "Six success/failure/divergence categories are represented.")

    formula_mismatches = 0
    percentile_mismatches = 0
    scenario_mismatches = 0
    evidence_mismatches = 0
    for input_item, output in zip(example_inputs, example_outputs):
        payload = input_item["payload"]
        metrics = lookup(output)
        wage = float(payload["proposed_annual_fixed_wage_eur"])
        years = float(payload["proposed_contract_years"])
        market = float(payload["current_public_market_value_eur"])
        commitment = wage * years
        formula_mismatches += int(not close(metrics["fixed_wage_commitment_eur"]["value"], commitment))
        formula_mismatches += int(not close(metrics["fixed_commitment_to_public_value"]["value"], commitment / market))
        formula_mismatches += int(not close(metrics["age_at_expiration"]["value"], float(payload["age_at_decision"]) + years))
        formula_mismatches += int(not close(metrics["scheduled_fixed_wages_after_36m_eur"]["value"], wage * max(years - 3, 0)))
        if payload.get("prior_annual_fixed_wage_eur") is not None:
            formula_mismatches += int(not close(metrics["proposed_wage_change_pct"]["value"], wage / float(payload["prior_annual_fixed_wage_eur"]) - 1))
        if payload.get("peer_fixed_commitment_estimate_eur") is not None:
            peer = float(payload["peer_fixed_commitment_estimate_eur"])
            residual_log = math.log(commitment / peer)
            expected_percentile = bisect.bisect_right(saved_residual.tolist(), residual_log + 1e-12) / len(saved_residual)
            percentile_mismatches += int(not close(metrics["offer_aggressiveness_percentile"]["value"], expected_percentile))
            formula_mismatches += int(not close(metrics["fixed_commitment_peer_difference_eur"]["value"], commitment - peer))
            lower = payload.get("peer_fixed_commitment_lower_eur")
            upper = payload.get("peer_fixed_commitment_upper_eur")
            if lower is None or upper is None:
                expected_position = "unavailable"
            elif commitment < float(lower):
                expected_position = "below_historical_80pct_range"
            elif commitment > float(upper):
                expected_position = "above_historical_80pct_range"
            else:
                expected_position = "within_historical_80pct_range"
            evidence_mismatches += int(metrics["fixed_commitment_peer_difference_eur"]["components"]["range_position"] != expected_position)
        if payload.get("sustained_contribution_probability_24m") is not None:
            expected_exposure = wage * min(years, 2) * (1 - float(payload["sustained_contribution_probability_24m"]))
            formula_mismatches += int(not close(metrics["low_contribution_wage_exposure_eur"]["value"], expected_exposure))
            components = metrics["low_contribution_wage_exposure_eur"]["components"]
            evidence_mismatches += int("low_contribution_probability_24m" not in components or "fixed_wages_through_24m_eur" not in components)
        value = metrics["public_value_downside_25pct_scenario"]["value"]
        if payload.get("public_value_downside_25pct_probability_24m") is not None:
            scenario_mismatches += int(not close(value["probability"], payload["public_value_downside_25pct_probability_24m"]))
            scenario_mismatches += int(not close(value["minimum_threshold_eur"], market * .25))
        continuity = metrics["contract_continuity_horizon"]["value"]
        if payload.get("continuous_stay_probability_36m") is not None:
            scenario_mismatches += int(not close(continuity["continuous_stay_probability_36m"], payload["continuous_stay_probability_36m"]))
            scenario_mismatches += int(not close(continuity["scheduled_fixed_wages_after_36m_eur"], wage * max(years - 3, 0)))
        evidence_mismatches += int(metrics["low_contribution_wage_exposure_eur"]["evidence_class"] != "probability_weighted_proxy")
        evidence_mismatches += int(metrics["public_value_downside_25pct_scenario"]["evidence_class"] != "paired_probability_scenario")
        evidence_mismatches += int(metrics["contract_continuity_horizon"]["evidence_class"] != "paired_probability_scenario")
        evidence_mismatches += int(output["overall_contract_score"] is not None)
    add(checks, "historical_formula_recomputation", formula_mismatches, 0, formula_mismatches == 0, "Deterministic, peer, and qualified exposure formulas reproduce independently.")
    add(checks, "historical_percentile_recomputation", percentile_mismatches, 0, percentile_mismatches == 0, "Frozen empirical CDF scores reproduce independently.")
    add(checks, "historical_scenario_recomputation", scenario_mismatches, 0, scenario_mismatches == 0, "Paired value and continuity scenarios reproduce without multiplication.")
    add(checks, "historical_evidence_classes", evidence_mismatches, 0, evidence_mismatches == 0, "Proxy, scenario, and rejected-composite classes remain distinct.")

    minimal = {
        "profile_scope": SCOPE,
        "age_at_decision": 27,
        "proposed_annual_fixed_wage_eur": 5_000_000,
        "proposed_contract_years": 4,
        "current_public_market_value_eur": 30_000_000,
    }
    minimal_result = build_extension_business_profile(minimal, saved_residual.tolist())
    minimal_metrics = lookup(minimal_result)
    unavailable_keys = {
        "proposed_wage_change_pct", "fixed_commitment_peer_difference_eur", "offer_aggressiveness_percentile",
        "low_contribution_wage_exposure_eur", "public_value_downside_25pct_scenario",
        "contract_continuity_horizon", "club_salary_percentile", "club_known_payroll_share",
    }
    missing_ok = all(minimal_metrics[key]["status"] == "unavailable" and minimal_metrics[key]["value"] is None for key in unavailable_keys)
    add(checks, "missing_data_behavior", int(missing_ok), 1, missing_ok, "Optional missing inputs fail closed as unavailable.")

    rejected_cases = 0
    invalid_payloads = [
        {**minimal, "profile_scope": "new_club_acquisition"},
        {**minimal, "profile_scope": "bettor_profile"},
        {**minimal, "proposed_annual_fixed_wage_eur": -1},
        {**minimal, "continuous_stay_probability_24m": .4, "continuous_stay_probability_36m": .6},
        {**minimal, "public_value_downside_25pct_probability_24m": .2, "public_value_downside_50pct_probability_24m": .4},
    ]
    for payload in invalid_payloads:
        try:
            build_extension_business_profile(payload, saved_residual.tolist())
        except ValueError:
            rejected_cases += 1
    add(checks, "invalid_input_rejection", rejected_cases, len(invalid_payloads), rejected_cases == len(invalid_payloads), "Scope, financial, horizon, and threshold violations fail closed.")

    salary_panel_result = build_extension_business_profile({
        **minimal,
        "club_known_annual_fixed_wages_excluding_player_eur": [1_000_000, 3_000_000, 5_000_000, 8_000_000],
    }, saved_residual.tolist())
    salary_metrics = lookup(salary_panel_result)
    salary_context_ok = (
        close(salary_metrics["club_salary_percentile"]["value"], 3.5 / 5)
        and close(salary_metrics["club_known_payroll_share"]["value"], 5_000_000 / 22_000_000)
        and salary_metrics["club_salary_percentile"]["components"]["context_source"] == "calculated_from_matched_panel_excluding_player"
    )
    add(checks, "salary_panel_recomputation", int(salary_context_ok), 1, salary_context_ok, "Proposed salary rank/share reproduce from a panel that explicitly excludes the player's prior wage.")

    registry = pd.read_csv(OUTPUT / "metric_registry.csv", low_memory=False)
    registry_map = dict(zip(registry["metric_key"], registry["engine_metric_key"]));
    registry_ok = registry_map.get("overall_contract_score") == "not_emitted" and all(key in set(registry["engine_metric_key"]) for key in [
        "fixed_wage_commitment_eur", "offer_aggressiveness_percentile", "low_contribution_wage_exposure_eur",
        "public_value_downside_25pct_scenario", "contract_continuity_horizon",
    ])
    add(checks, "metric_registry_scope", int(registry_ok), 1, registry_ok, "Audited metrics map to stable engine keys and the composite is not emitted.")

    source_manifest = pd.read_csv(OUTPUT / "source_manifest.csv", low_memory=False)
    source_failures = []
    for row in source_manifest.itertuples(index=False):
        path = ROOT / row.source_file
        if not path.exists() or sha256(path) != row.sha256:
            source_failures.append(row.source_file)
    add(checks, "source_hashes", len(source_failures), 0, not source_failures, f"Failures: {source_failures or 'none'}")

    output_manifest = pd.read_csv(OUTPUT / "output_manifest.csv", low_memory=False)
    output_failures = []
    for row in output_manifest.itertuples(index=False):
        if row.output_file in {"independent_verification.csv", "independent_verification.json"}:
            continue
        path = OUTPUT / row.output_file
        if not path.exists() or sha256(path) != row.sha256:
            output_failures.append(row.output_file)
    add(checks, "builder_output_hashes", len(output_failures), 0, not output_failures, f"Failures: {output_failures or 'none'}")
    build_checks = pd.read_csv(OUTPUT / "build_checks.csv", low_memory=False)
    builder_ok = as_bool(build_checks["passed"]).all()
    add(checks, "builder_checks", int(builder_ok), 1, builder_ok, "All product builder checks pass.")

    results = pd.DataFrame(checks)
    results.to_csv(OUTPUT / "independent_verification.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    payload = {
        "checks_passed": int(as_bool(results["passed"]).sum()),
        "checks_total": len(results),
        "all_checks_passed": bool(as_bool(results["passed"]).all()),
        "offer_reference_rows": len(saved_residual),
        "historical_examples": len(example_outputs),
        "metric_version": reference["metric_version"],
        "profile_scope": SCOPE,
        "overall_contract_score": "rejected",
    }
    (OUTPUT / "independent_verification.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    files = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name != "output_manifest.csv")
    manifest = pd.DataFrame([{"output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files])
    manifest.to_csv(OUTPUT / "output_manifest.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    if not as_bool(results["passed"]).all():
        failed = results.loc[~as_bool(results["passed"]), "check"].tolist()
        raise RuntimeError(f"Independent business metric product verification failed: {failed}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

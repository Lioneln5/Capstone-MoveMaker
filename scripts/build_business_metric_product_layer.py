"""Build the frozen MoveMaker business-metric product layer."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.business_metric_engine import METRIC_VERSION, SCOPE, build_extension_business_profile


OUTPUT = ROOT / "Data" / "processed" / "business_metric_product"
AUDIT = ROOT / "Data" / "processed" / "business_metric_translation_audit"
PHASE5 = ROOT / "Data" / "processed" / "integrated_contract_profile"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").fillna("").str.strip().str.casefold().isin({"true", "1", "yes"})


def optional(row: pd.Series, column: str) -> Any:
    value = row.get(column)
    return None if pd.isna(value) else value


def historical_payload(row: pd.Series) -> dict[str, Any]:
    return {
        "profile_scope": SCOPE,
        "player": row["player_name"],
        "current_club": row["club_name"],
        "decision_date": row["signed_date"],
        "age_at_decision": row["age_at_signing"],
        "proposed_annual_fixed_wage_eur": row["annual_gross_eur"],
        "proposed_contract_years": row["exact_duration_years"],
        "current_public_market_value_eur": row["at_signing_market_value_eur"],
        "prior_annual_fixed_wage_eur": optional(row, "prior_season_salary_annual_gross_eur"),
        "peer_fixed_commitment_estimate_eur": optional(row, "fixed_commitment_peer_estimate_eur"),
        "peer_fixed_commitment_lower_eur": optional(row, "fixed_commitment_peer_lower_eur"),
        "peer_fixed_commitment_upper_eur": optional(row, "fixed_commitment_peer_upper_eur"),
        "sustained_contribution_probability_24m": optional(row, "sustained_contribution_probability"),
        "continuous_stay_probability_24m": optional(row, "continuous_stay_probability_24m"),
        "continuous_stay_probability_36m": optional(row, "continuous_stay_probability_36m"),
        "public_value_downside_25pct_probability_12m": optional(row, "value_downside_25pct_probability_12m"),
        "public_value_downside_25pct_probability_24m": optional(row, "value_downside_25pct_probability_24m"),
        "public_value_downside_50pct_probability_24m": optional(row, "value_downside_50pct_probability_24m"),
        "club_salary_percentile": optional(row, "club_salary_percentile"),
        "club_known_payroll_share": optional(row, "club_salary_share_known"),
    }


def metric_lookup(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        metric["metric_key"]: metric
        for card in profile["cards"].values()
        for metric in card
    }


def input_schema() -> dict[str, Any]:
    probability = {"type": ["number", "null"], "minimum": 0, "maximum": 1}
    optional_positive = {"type": ["number", "null"], "exclusiveMinimum": 0}
    properties: dict[str, Any] = {
        "profile_scope": {"const": SCOPE},
        "player": {"type": ["string", "null"]},
        "current_club": {"type": ["string", "null"]},
        "decision_date": {"type": ["string", "null"], "format": "date"},
        "age_at_decision": {"type": "number", "minimum": 15, "maximum": 50},
        "proposed_annual_fixed_wage_eur": {"type": "number", "exclusiveMinimum": 0},
        "proposed_contract_years": {"type": "number", "exclusiveMinimum": 0, "maximum": 10.5},
        "current_public_market_value_eur": {"type": "number", "exclusiveMinimum": 0},
        "prior_annual_fixed_wage_eur": optional_positive,
        "peer_fixed_commitment_estimate_eur": optional_positive,
        "peer_fixed_commitment_lower_eur": optional_positive,
        "peer_fixed_commitment_upper_eur": optional_positive,
        "sustained_contribution_probability_24m": probability,
        "continuous_stay_probability_24m": probability,
        "continuous_stay_probability_36m": probability,
        "public_value_downside_25pct_probability_12m": probability,
        "public_value_downside_25pct_probability_24m": probability,
        "public_value_downside_50pct_probability_24m": probability,
        "club_salary_percentile": probability,
        "club_known_payroll_share": probability,
        "club_known_annual_fixed_wages_excluding_player_eur": {
            "type": ["array", "null"],
            "minItems": 1,
            "items": {"type": "number", "exclusiveMinimum": 0},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "movemaker.extension-business-input.v1",
        "title": "MoveMaker incumbent-extension business metric input",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "profile_scope", "age_at_decision", "proposed_annual_fixed_wage_eur",
            "proposed_contract_years", "current_public_market_value_eur",
        ],
        "properties": properties,
        "x_model_output_note": "Probability and peer fields are backend model outputs, not ordinary user-entered fields.",
    }


def output_contract() -> dict[str, Any]:
    return {
        "metric_version": METRIC_VERSION,
        "profile_scope": SCOPE,
        "product_status": "verified_translation_prototype_pending_decision_utility_review",
        "overall_score": {"status": "rejected", "required_value": None},
        "cards": [
            {
                "card": "capital_commitment",
                "headline_metrics": ["fixed_wage_commitment_eur", "fixed_commitment_to_public_value"],
                "detail_metrics": ["age_at_expiration", "proposed_wage_change_pct"],
            },
            {
                "card": "offer_context",
                "headline_metrics": ["fixed_commitment_peer_difference_eur", "offer_aggressiveness_percentile"],
                "detail_metrics": [],
            },
            {
                "card": "contribution_exposure",
                "headline_metrics": ["low_contribution_wage_exposure_eur"],
                "detail_metrics": ["fixed_wages_through_24m_eur", "low_contribution_probability_24m"],
            },
            {
                "card": "asset_value",
                "headline_metrics": ["public_value_downside_25pct_scenario_12m"],
                "detail_metrics": [
                    "public_value_downside_25pct_scenario",
                    "public_value_downside_50pct_scenario",
                ],
                "horizon_policy": {
                    "primary": "12m",
                    "optional_with_warning": "24m",
                },
            },
            {
                "card": "continuity",
                "headline_metrics": ["contract_continuity_horizon"],
                "detail_metrics": ["scheduled_fixed_wages_after_36m_eur"],
            },
            {
                "card": "salary_structure",
                "headline_metrics": [],
                "detail_metrics": ["club_salary_percentile", "club_known_payroll_share"],
            },
        ],
        "evidence_classes": [
            "exact_calculation", "exact_ratio_proxy", "historical_benchmark_translation",
            "historical_oos_reference", "probability_weighted_proxy",
            "paired_probability_scenario", "observed_context", "unavailable",
        ],
        "required_behavior": [
            "Return unavailable rather than imputing a missing optional input.",
            "Show probability-weighted exposure components alongside the euro proxy.",
            "Never multiply paired public-value or continuity scenarios into an expected-loss number.",
            "Never emit an overall contract score or approve/reject recommendation.",
        ],
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    audit_verification = json.loads((AUDIT / "independent_verification.json").read_text(encoding="utf-8"))
    phase5_verification = json.loads((PHASE5 / "independent_verification.json").read_text(encoding="utf-8"))
    if not audit_verification.get("all_checks_passed") or not phase5_verification.get("all_checks_passed"):
        raise RuntimeError("Required business audit or Phase-5 profile is not independently verified")

    profiles = pd.read_csv(AUDIT / "translated_historical_profiles.csv", low_memory=False)
    phase5_profiles = pd.read_csv(PHASE5 / "historical_integrated_profiles.csv", low_memory=False)
    profiles = profiles.merge(
        phase5_profiles[[
            "capology_extension_event_id", "fixed_commitment_peer_lower_eur", "fixed_commitment_peer_upper_eur",
            "value_downside_25pct_probability_12m",
        ]],
        on="capology_extension_event_id", how="left", validate="one_to_one",
    )
    cases = pd.read_csv(AUDIT / "historical_business_profile_cases.csv", low_memory=False)
    residual = np.log(profiles["fixed_wage_commitment_eur"] / profiles["fixed_commitment_peer_estimate_eur"])
    sorted_residuals = sorted(float(value) for value in residual.dropna())
    reference = {
        "reference_id": "offer_aggressiveness_oos_v1_2026-08-12",
        "metric_version": METRIC_VERSION,
        "profile_scope": SCOPE,
        "definition": "Empirical CDF of log(proposed fixed-wage commitment / selected Phase-4 peer estimate).",
        "reference_rows": len(sorted_residuals),
        "evaluation_years": sorted(int(value) for value in profiles.loc[residual.notna(), "evaluation_year"].unique()),
        "percentile_method": "bisect_right(sorted residuals, new residual + 1e-12) / reference_rows",
        "quantiles": {str(value): float(np.quantile(sorted_residuals, value)) for value in [.01, .05, .10, .25, .50, .75, .90, .95, .99]},
        "sorted_log_residuals": sorted_residuals,
        "boundary": "Historical offer aggressiveness only; not fair value, optimality, or overpayment.",
    }
    (OUTPUT / "offer_aggressiveness_reference.json").write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    reference_table = pd.DataFrame({
        "rank": np.arange(1, len(sorted_residuals) + 1),
        "percentile": np.arange(1, len(sorted_residuals) + 1) / len(sorted_residuals),
        "log_commitment_residual": sorted_residuals,
        "commitment_multiple_vs_peer": np.exp(sorted_residuals),
    })
    write_csv(reference_table, OUTPUT / "offer_aggressiveness_reference.csv")

    (OUTPUT / "input_schema.json").write_text(json.dumps(input_schema(), indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "output_contract.json").write_text(json.dumps(output_contract(), indent=2) + "\n", encoding="utf-8")

    sample_inputs: list[dict[str, Any]] = []
    sample_outputs: list[dict[str, Any]] = []
    for row in cases.itertuples(index=False):
        source = profiles.loc[profiles["capology_extension_event_id"].eq(row.capology_extension_event_id)].iloc[0]
        payload = historical_payload(source)
        result = build_extension_business_profile(payload, sorted_residuals)
        result["historical_case"] = {
            "capology_extension_event_id": row.capology_extension_event_id,
            "case_category": row.case_category,
            "case_interpretation": row.case_interpretation,
            "actual_low_contribution": optional(pd.Series(row._asdict()), "actual_low_contribution"),
            "actual_outbound_24m": optional(pd.Series(row._asdict()), "actual_outbound_24m"),
            "actual_value_downside_25pct_24m": optional(pd.Series(row._asdict()), "actual_value_downside_25pct_24m"),
        }
        sample_inputs.append({"case_category": row.case_category, "payload": payload})
        sample_outputs.append(result)
    (OUTPUT / "historical_example_inputs.json").write_text(json.dumps(sample_inputs, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "historical_example_profiles.json").write_text(json.dumps(sample_outputs, indent=2) + "\n", encoding="utf-8")

    checks: list[dict[str, Any]] = []
    def add(name: str, observed: Any, expected: Any, passed: bool, notes: str) -> None:
        checks.append({"check": name, "observed": observed, "expected": expected, "passed": bool(passed), "notes": notes})

    add("reference_rows", len(sorted_residuals), 1504, len(sorted_residuals) == 1504, "All financially eligible out-of-time peer residuals are frozen.")
    add("reference_sorted", int(np.all(np.diff(sorted_residuals) >= 0)), 1, np.all(np.diff(sorted_residuals) >= 0), "CDF reference is sorted ascending.")
    add("historical_example_count", len(sample_outputs), 12, len(sample_outputs) == 12, "Phase-5 successes, failures, and divergence cases are represented.")
    add("historical_example_categories", len({item["case_category"] for item in sample_inputs}), 6, len({item["case_category"] for item in sample_inputs}) == 6, "Six case categories are represented.")

    metric_mismatches = {key: 0 for key in [
        "fixed_wage_commitment_eur", "fixed_commitment_to_public_value", "proposed_wage_change_pct",
        "fixed_commitment_peer_difference_eur", "offer_aggressiveness_percentile",
        "fixed_commitment_peer_range_position",
        "low_contribution_wage_exposure_eur", "public_value_downside_25pct_scenario_12m",
        "public_value_downside_25pct_scenario",
        "scheduled_fixed_wages_after_36m_eur", "contract_continuity_horizon",
            "club_salary_percentile", "club_known_payroll_share",
            "club_known_annual_fixed_wages_excluding_player_eur",
    ]}
    evaluated = 0
    for _, row in profiles.iterrows():
        if pd.isna(row["annual_gross_eur"]) or pd.isna(row["exact_duration_years"]) or pd.isna(row["at_signing_market_value_eur"]):
            continue
        result = build_extension_business_profile(historical_payload(row), sorted_residuals)
        lookup = metric_lookup(result)
        evaluated += 1
        comparisons = {
            "fixed_wage_commitment_eur": row["fixed_wage_commitment_eur"],
            "fixed_commitment_to_public_value": row["fixed_commitment_to_market_value"],
            "proposed_wage_change_pct": row["prior_wage_change_pct"],
            "fixed_commitment_peer_difference_eur": row["fixed_commitment_peer_difference_eur"],
            "offer_aggressiveness_percentile": row["fixed_commitment_peer_percentile"],
            "low_contribution_wage_exposure_eur": row["risk_weighted_low_contribution_wage_exposure_eur"],
            "scheduled_fixed_wages_after_36m_eur": row["scheduled_fixed_wages_after_36m_eur"],
            "club_salary_percentile": row["club_salary_percentile"],
            "club_known_payroll_share": row["club_salary_share_known"],
        }
        for key, expected in comparisons.items():
            observed = lookup[key]["value"]
            if pd.isna(expected):
                metric_mismatches[key] += int(observed is not None)
            else:
                metric_mismatches[key] += int(observed is None or not np.isclose(float(observed), float(expected), atol=1e-8, rtol=1e-8))
        value_12m_metric = lookup["public_value_downside_25pct_scenario_12m"]
        if pd.isna(row["value_downside_25pct_probability_12m"]):
            metric_mismatches["public_value_downside_25pct_scenario_12m"] += int(value_12m_metric["value"] is not None)
        else:
            value_12m_ok = (
                np.isclose(value_12m_metric["value"]["probability"], row["value_downside_25pct_probability_12m"])
                and np.isclose(value_12m_metric["value"]["minimum_threshold_eur"], row["at_signing_market_value_eur"] * .25)
                and value_12m_metric["status"] == "primary_horizon_candidate"
            )
            metric_mismatches["public_value_downside_25pct_scenario_12m"] += int(not value_12m_ok)
        value_metric = lookup["public_value_downside_25pct_scenario"]
        if pd.isna(row["value_downside_25pct_probability_24m"]):
            metric_mismatches["public_value_downside_25pct_scenario"] += int(value_metric["value"] is not None)
        else:
            value_ok = (
                np.isclose(value_metric["value"]["probability"], row["value_downside_25pct_probability_24m"])
                and np.isclose(value_metric["value"]["minimum_threshold_eur"], row["public_value_downside_floor_25pct_eur"])
                and value_metric["status"] == "provisional_pending_later_validation"
            )
            metric_mismatches["public_value_downside_25pct_scenario"] += int(not value_ok)
        continuity_metric = lookup["contract_continuity_horizon"]
        if pd.isna(row["continuous_stay_probability_36m"]):
            metric_mismatches["contract_continuity_horizon"] += int(continuity_metric["value"] is not None)
        else:
            continuity_ok = (
                np.isclose(continuity_metric["value"]["continuous_stay_probability_36m"], row["continuous_stay_probability_36m"])
                and np.isclose(continuity_metric["value"]["scheduled_fixed_wages_after_36m_eur"], row["scheduled_fixed_wages_after_36m_eur"])
            )
            metric_mismatches["contract_continuity_horizon"] += int(not continuity_ok)
        observed_position = lookup["fixed_commitment_peer_difference_eur"]["components"].get("range_position")
        expected_position = row["fixed_commitment_peer_position"]
        metric_mismatches["fixed_commitment_peer_range_position"] += int(observed_position != expected_position)
    add("historical_profiles_evaluated", evaluated, 1476, evaluated == 1476, "Every profile with valid wage, term, and positive public value is replayed through the engine; 28 peer profiles lack public value and remain in the offer reference only.")
    for key, mismatches in metric_mismatches.items():
        add(f"metric_parity_{key}", mismatches, 0, mismatches == 0, "Engine output reproduces the frozen translated historical profile.")

    minimal_payload = {
        "profile_scope": SCOPE,
        "age_at_decision": 27,
        "proposed_annual_fixed_wage_eur": 5_000_000,
        "proposed_contract_years": 4,
        "current_public_market_value_eur": 30_000_000,
    }
    minimal = build_extension_business_profile(minimal_payload, sorted_residuals)
    minimal_lookup = metric_lookup(minimal)
    expected_unavailable = {
        "proposed_wage_change_pct", "fixed_commitment_peer_difference_eur", "offer_aggressiveness_percentile",
        "low_contribution_wage_exposure_eur", "public_value_downside_25pct_scenario_12m",
        "public_value_downside_25pct_scenario",
        "contract_continuity_horizon", "club_salary_percentile", "club_known_payroll_share",
    }
    unavailable_ok = all(minimal_lookup[key]["status"] == "unavailable" for key in expected_unavailable)
    add("missing_optional_inputs", int(unavailable_ok), 1, unavailable_ok, "Optional metrics return unavailable rather than zero or an imputed value.")

    scope_rejected = False
    try:
        build_extension_business_profile({**minimal_payload, "profile_scope": "new_club_acquisition"}, sorted_residuals)
    except ValueError:
        scope_rejected = True
    add("new_club_scope_rejected", int(scope_rejected), 1, scope_rejected, "Extension metrics refuse new-club acquisition mode.")
    order_rejected = False
    try:
        build_extension_business_profile({
            **minimal_payload,
            "public_value_downside_25pct_probability_24m": .2,
            "public_value_downside_50pct_probability_24m": .4,
        }, sorted_residuals)
    except ValueError:
        order_rejected = True
    add("probability_order_rejected", int(order_rejected), 1, order_rejected, "Incoherent value-threshold probabilities fail closed.")
    salary_panel_profile = build_extension_business_profile({
        **minimal_payload,
        "proposed_annual_fixed_wage_eur": 5_000_000,
        "club_known_annual_fixed_wages_excluding_player_eur": [1_000_000, 3_000_000, 5_000_000, 8_000_000],
    }, sorted_residuals)
    salary_lookup = metric_lookup(salary_panel_profile)
    expected_salary_percentile = 3.5 / 5
    expected_payroll_share = 5_000_000 / 22_000_000
    salary_panel_ok = (
        np.isclose(salary_lookup["club_salary_percentile"]["value"], expected_salary_percentile)
        and np.isclose(salary_lookup["club_known_payroll_share"]["value"], expected_payroll_share)
    )
    add("salary_panel_calculation", int(salary_panel_ok), 1, salary_panel_ok, "Proposed wage rank and known-payroll share calculate from a matched panel excluding the player without double-counting.")
    no_composite = all(output["overall_contract_score"] is None for output in sample_outputs)
    add("overall_score_absent", int(no_composite), 1, no_composite, "No historical product example exposes an overall score.")
    check_frame = pd.DataFrame(checks)
    write_csv(check_frame, OUTPUT / "build_checks.csv")

    registry = pd.read_csv(AUDIT / "business_metric_catalog.csv", low_memory=False)
    engine_key_map = {
        "fixed_wage_commitment_eur": "fixed_wage_commitment_eur",
        "fixed_commitment_to_market_value": "fixed_commitment_to_public_value",
        "prior_wage_change_pct": "proposed_wage_change_pct",
        "fixed_commitment_peer_difference_eur": "fixed_commitment_peer_difference_eur",
        "fixed_commitment_peer_percentile": "offer_aggressiveness_percentile",
        "risk_weighted_low_contribution_wage_exposure_eur": "low_contribution_wage_exposure_eur",
        "value_downside_25pct_display": "public_value_downside_25pct_scenario",
        "continuity_commitment_display": "contract_continuity_horizon",
        "club_salary_percentile": "club_salary_percentile",
        "club_salary_share_known": "club_known_payroll_share",
        "overall_contract_score": "not_emitted",
    }
    registry.insert(1, "engine_metric_key", registry["metric_key"].map(engine_key_map))
    write_csv(registry, OUTPUT / "metric_registry.csv")

    summary = {
        "metric_version": METRIC_VERSION,
        "profile_scope": SCOPE,
        "product_status": "verified_translation_prototype_pending_decision_utility_review",
        "offer_reference_rows": len(sorted_residuals),
        "historical_profiles_replayed": evaluated,
        "historical_examples": len(sample_outputs),
        "builder_checks_passed": int(as_bool(check_frame["passed"]).sum()),
        "builder_checks_total": len(check_frame),
        "all_builder_checks_passed": bool(as_bool(check_frame["passed"]).all()),
        "overall_contract_score": "rejected",
    }
    (OUTPUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    readme = f"""# MoveMaker business metric product-layer prototype

This package implements and verifies candidate business translations for incumbent-club extension profiles. Its formulas and response behavior are reproducible, but its card selection is not the finalized application design. A post-build review found that several valid calculations—especially salary percentile and commitment/public-value—may be too intuitive or insufficiently actionable to headline the product. It does not fit the Phase 1-4 models and it does not create an overall contract score.

## Implemented candidate outputs

- Fixed-wage commitment, commitment/public-value scale, age at expiration, and proposed wage change.
- Fixed-commitment difference from the Phase-4 peer estimate and an offer-aggressiveness percentile based on {len(sorted_residuals):,} frozen out-of-time residuals.
- Qualified low-contribution wage exposure: two-year fixed wages × modeled low-contribution probability.
- Primary paired 12-month public-value downside probability and euro threshold.
- Optional 24-month public-value downside scenarios with an enforced warning that available 2024+ outcomes were previously exposed in frozen V1 verification and that the repaired complete-feature, player-disjoint 2024 cohort contains only 26 outcomes, with as few as 3 in one Big Five league.
- Paired 36-month continuous-stay probability and scheduled post-36-month fixed wages.
- Matched known-panel club salary position and payroll share.

## Integration

Call `models.business_metric_engine.build_extension_business_profile(payload, sorted_reference_residuals)`. Load `sorted_offer_reference_residuals` from `offer_aggressiveness_reference.json`. The input JSON schema distinguishes required proposed-contract inputs from optional backend model outputs. The engine validates scope, probability bounds, horizon/threshold coherence, positive financial inputs, and missing-data behavior.

The personalized probability and Phase-4 peer inputs still require production model serialization. Until those scorers exist, use the 12 historical examples for the application layout and demonstration; do not pretend the engine can infer probabilities directly from the few visible user inputs.

Before application integration, apply the decision-utility standard documented in the repository README and scope-pivot journal: retain only outputs that reveal non-obvious information from the underlying data and can change a plausible club decision. Simple facts may remain as background contract context without being presented as MoveMaker's analytical contribution.

## Boundaries

The product is scoped to incumbent-club extensions. Low-contribution exposure is a qualified risk-weighted proxy, not expected accounting loss or wasted wages. Public-value and continuity outputs are paired scenarios, not multiplied losses. Offer percentile is historical aggressiveness, not fairness or optimality. No betting, new-club acquisition, ROI, or approve/reject claim is supported.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")

    sources = [
        AUDIT / "translated_historical_profiles.csv",
        AUDIT / "historical_business_profile_cases.csv",
        AUDIT / "business_metric_catalog.csv",
        AUDIT / "independent_verification.json",
        PHASE5 / "independent_verification.json",
        PHASE5 / "historical_integrated_profiles.csv",
        ROOT / "models" / "business_metric_engine.py",
    ]
    source_manifest = pd.DataFrame([{
        "source_file": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    } for path in sources])
    write_csv(source_manifest, OUTPUT / "source_manifest.csv")

    files = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name != "output_manifest.csv")
    manifest = pd.DataFrame([{
        "output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path),
    } for path in files])
    write_csv(manifest, OUTPUT / "output_manifest.csv")
    if not as_bool(check_frame["passed"]).all():
        failed = check_frame.loc[~as_bool(check_frame["passed"]), "check"].tolist()
        raise RuntimeError(f"Business metric product build checks failed: {failed}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

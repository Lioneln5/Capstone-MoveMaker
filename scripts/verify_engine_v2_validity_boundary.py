"""Independent verification for Engine V2 Phase 1."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.engine_v2.validity_boundary import EngineV2Request, evaluate_request  # noqa: E402

OUTPUT = ROOT / "Data" / "processed" / "engine_v2_validity_boundary"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    results: list[dict] = []

    def add(name: str, passed: bool, observed: object, expected: object, notes: str) -> None:
        results.append({"check": name, "passed": bool(passed), "observed": observed, "expected": expected, "notes": notes})

    cohort = json.loads((OUTPUT / "historical_cohort_boundary.json").read_text(encoding="utf-8"))
    claims = json.loads((OUTPUT / "claim_boundary.json").read_text(encoding="utf-8"))
    selection = pd.read_csv(OUTPUT / "candidate_selection_boundary.csv")
    catalog = pd.read_csv(OUTPUT / "refusal_reason_catalog.csv")
    audit = pd.read_csv(OUTPUT / "decision_time_feature_audit.csv")

    add("cohort_event_count", cohort["extension_events"] == 2805, cohort["extension_events"], 2805, "Recomputed build output reports the cleaned master count.")
    add("cohort_is_big_five", set(cohort["competition_counts"]) == {"GB1", "ES1", "IT1", "L1", "FR1"}, sorted(cohort["competition_counts"]), "five competition codes", "No outside-league population is implied.")
    unknown = selection.loc[selection["population"].eq("players considered but not extended")].iloc[0]
    add("candidate_denominator_unknown", pd.isna(unknown["observed_count"]) and unknown["status"] == "not_observable_in_sources", unknown["status"], "not_observable_in_sources", "The audit does not invent a comparison cohort.")
    add("claim_is_conditional", claims["supported_claim"] == "outcome_scenario_conditional_on_extension", claims["supported_claim"], "outcome_scenario_conditional_on_extension", "No extend/do-not-extend claim.")
    add("selection_bias_explicit", claims["candidate_selection_bias_status"] == "unresolved_material_limitation", claims["candidate_selection_bias_status"], "unresolved_material_limitation", "Limitation is machine readable.")
    add("refusal_catalog_complete", len(catalog) == 24 and catalog["refusal_code"].is_unique, len(catalog), 24, "Every contract refusal has a documented meaning.")
    add("decision_time_audit_complete", len(audit) == 12 and audit["input_group"].is_unique, len(audit), 12, "Predictor, scenario-input, and target roles are separated.")

    base = EngineV2Request(
        use_case="incumbent_club_extension", canonical_player_id=7, canonical_club_id=70,
        competition_id="ES1", decision_date=date(2023, 6, 30), player_identity_verified=True,
        club_identity_verified=True, date_of_birth=date(1998, 1, 1), broad_position="Midfield",
        incumbency_verified=True, incumbency_evidence_type="dated_club_appearance",
        incumbency_evidence_date=date(2023, 5, 20), sporting_data_cutoff=date(2023, 6, 30),
        pre365_club_games_observed=40, current_market_value_eur=25_000_000,
        market_value_as_of_date=date(2023, 6, 1), proposed_annual_fixed_wage_eur=4_000_000,
        proposed_contract_years=4, salary_panel_valid_through=date(2023, 6, 30),
    )
    eligible = evaluate_request(base, today=date(2026, 8, 30))
    add("covered_request_eligible", eligible.status == "eligible_for_research_scoring" and all(x.eligible for x in eligible.modules.values()), eligible.as_dict(), "all four modules eligible", "Positive control.")

    arbitrary = evaluate_request(replace(base, canonical_club_id=999, incumbency_verified=False), today=date(2026, 8, 30))
    add("arbitrary_club_refused", arbitrary.status == "refused" and "INCUMBENCY_UNVERIFIED" in arbitrary.global_refusal_codes, arbitrary.global_refusal_codes, "INCUMBENCY_UNVERIFIED", "Closes V1 warning-only loophole.")
    stale_incumbency = evaluate_request(replace(base, incumbency_evidence_date=date(2022, 1, 1)), today=date(2026, 8, 30))
    add("stale_incumbency_refused", stale_incumbency.status == "refused" and "INCUMBENCY_EVIDENCE_STALE" in stale_incumbency.global_refusal_codes, stale_incumbency.global_refusal_codes, "INCUMBENCY_EVIDENCE_STALE", "Ever appeared is not equivalent to current incumbent.")
    future = evaluate_request(replace(base, decision_date=date(2027, 1, 1)), today=date(2026, 8, 30))
    add("future_date_refused", future.status == "refused" and "FUTURE_DECISION_DATE" in future.global_refusal_codes, future.global_refusal_codes, "FUTURE_DECISION_DATE", "Future information boundary is undefined.")
    new_club = evaluate_request(replace(base, use_case="new_club_transfer"), today=date(2026, 8, 30))
    add("transfer_use_case_refused", new_club.status == "refused" and "UNSUPPORTED_USE_CASE" in new_club.global_refusal_codes, new_club.global_refusal_codes, "UNSUPPORTED_USE_CASE", "V2 cannot revert to transfer compatibility.")
    stale_sport = evaluate_request(replace(base, sporting_data_cutoff=date(2023, 5, 1)), today=date(2026, 8, 30))
    add("stale_sporting_modules_refused", not stale_sport.modules["future_role"].eligible and not stale_sport.modules["club_continuity"].eligible and stale_sport.modules["wage_benchmark"].eligible, stale_sport.as_dict(), "sporting off; wage on", "Module isolation.")
    low_games = evaluate_request(replace(base, pre365_club_games_observed=9), today=date(2026, 8, 30))
    add("low_game_evidence_refused", "INSUFFICIENT_PRE365_CLUB_GAME_EVIDENCE" in low_games.modules["future_role"].refusal_codes, low_games.modules["future_role"].refusal_codes, "INSUFFICIENT_PRE365_CLUB_GAME_EVIDENCE", "Matches historical evidence eligibility.")
    stale_value = evaluate_request(replace(base, market_value_as_of_date=date(2021, 1, 1)), today=date(2026, 8, 30))
    add("stale_value_module_refused", not stale_value.modules["public_value_downside"].eligible and stale_value.modules["future_role"].eligible, stale_value.as_dict(), "value off; role on", "No stale public-value estimate.")
    stale_salary = evaluate_request(replace(base, salary_panel_valid_through=date(2023, 6, 29)), today=date(2026, 8, 30))
    add("stale_salary_module_refused", not stale_salary.modules["wage_benchmark"].eligible and stale_salary.modules["future_role"].eligible, stale_salary.as_dict(), "wage off; role on", "No prior-season fallback in V2.")
    no_modules = evaluate_request(
        replace(
            base,
            sporting_data_cutoff=date(2023, 5, 1),
            current_market_value_eur=None,
            proposed_annual_fixed_wage_eur=None,
            salary_panel_valid_through=None,
        ),
        today=date(2026, 8, 30),
    )
    add("no_eligible_module_refuses_request", no_modules.status == "refused" and not any(item.eligible for item in no_modules.modules.values()), no_modules.as_dict(), "request refused", "A globally valid identity cannot produce an empty scored shell when every module lacks evidence.")
    short_term = evaluate_request(replace(base, proposed_contract_years=1), today=date(2026, 8, 30))
    add("short_term_refused", short_term.status == "refused" and "PROPOSED_TERM_OUTSIDE_SUPPORTED_RANGE" in short_term.global_refusal_codes, short_term.global_refusal_codes, "PROPOSED_TERM_OUTSIDE_SUPPORTED_RANGE", "The product's two-year outcome horizon must be contract-covered.")

    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    bad = [item["path"] for item in freeze["files"] if sha256(ROOT / item["path"]) != item["sha256"]]
    add("v1_hashes_unchanged", not bad, bad, [], "Phase 1 remains outside frozen V1.")

    manifest = pd.read_csv(OUTPUT / "output_manifest.csv")
    mismatches = []
    for row in manifest.itertuples(index=False):
        path = OUTPUT / row.file
        if not path.exists() or sha256(path) != row.sha256 or path.stat().st_size != row.bytes:
            mismatches.append(row.file)
    add("output_manifest_integrity", not mismatches, mismatches, [], "Every declared Phase-1 output matches its hash and byte size.")

    checks = pd.DataFrame(results)
    checks.to_csv(OUTPUT / "independent_verification.csv", index=False, encoding="utf-8", lineterminator="\n")
    payload = {
        "checks_passed": int(checks["passed"].sum()),
        "checks_total": len(checks),
        "all_checks_passed": bool(checks["passed"].all()),
        "verified_scope": "cohort_boundary_decision_time_contract_and_v1_isolation",
    }
    (OUTPUT / "independent_verification.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_rows = []
    for path in sorted(OUTPUT.iterdir()):
        if path.is_file() and path.name != "output_manifest.csv":
            manifest_rows.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest_rows).to_csv(
        OUTPUT / "output_manifest.csv", index=False, encoding="utf-8", lineterminator="\n"
    )
    print(json.dumps(payload, indent=2))
    if not payload["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

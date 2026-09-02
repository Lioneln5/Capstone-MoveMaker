"""Build the Phase-1 Engine V2 validity-boundary evidence package.

This is a cohort and request-contract audit only.  It does not fit, compare,
or deploy any predictive model and never writes inside the frozen V1 artifact
directory.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.engine_v2.validity_boundary import (  # noqa: E402
    EngineV2Request,
    PROHIBITED_CLAIMS,
    SUPPORTED_CLAIM,
    evaluate_request,
)

MASTER = ROOT / "Data" / "processed" / "contract_extension_integration" / "extension_modeling_master.csv"
INTEGRATION_SUMMARY = ROOT / "Data" / "processed" / "contract_extension_integration" / "run_summary.json"
CONTRACT_SUMMARY = ROOT / "Data" / "processed" / "capology_contracts" / "run_summary.json"
OUTPUT = ROOT / "Data" / "processed" / "engine_v2_validity_boundary"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def decision_time_audit() -> pd.DataFrame:
    rows = [
        ("player_identity", "canonical player ID and date of birth", "identity sources known on/before decision", "allow_if_verified", "Required; current/latest snapshot fields are not outcome predictors."),
        ("incumbent_club", "player-club relationship", "dated appearance, roster, contract, or salary-panel evidence", "hard_refusal_if_unverified", "An arbitrary player-club pairing is never scoreable in V2."),
        ("league", "Big-Five competition at decision", "dated club/competition evidence", "hard_refusal_outside_big_five", "Caller text alone is insufficient provenance."),
        ("broad_position", "Goalkeeper/Defender/Midfield/Attack", "as-of role evidence", "allow_if_resolved", "Historical retraining must not use a later current-position snapshot as though it were contemporaneous."),
        ("age", "age at decision", "verified date of birth", "allow_16_to_46_1", "Observed historical support, not a claim that boundary cases are equally reliable."),
        ("pre365_sporting", "same-club games, appearances, minutes, opportunity", "strictly before decision; source reaches decision date", "require_10_club_games", "No stale fallback and no partial-window score."),
        ("completed_season_performance", "prior completed same-club season", "season end on/before decision", "candidate_feature_only", "Role-aware specification is a later modeling phase."),
        ("current_market_value", "positive public valuation", "as-of date on/before decision and no more than 365 days old", "required_for_value_module", "Public estimate, not sale proceeds or accounting value."),
        ("proposed_fixed_wage", "club-entered scenario input", "known at decision", "financial_module_only", "Prohibited from sporting endpoints until a dedicated causal/ablation case survives."),
        ("proposed_contract_term", "club-entered scenario input", "known at decision", "allow_2_to_6_years", "Needed for the two-year product horizon and within broad historical support."),
        ("salary_panel", "club/player wage context", "panel valid through decision date", "required_for_wage_benchmark", "Older-season fallback is prohibited in V2 scoring."),
        ("post_extension_outcomes", "future minutes, departure, valuation", "after decision", "target_only_never_predictor", "Any use as a predictor is leakage."),
    ]
    return pd.DataFrame(rows, columns=["input_group", "field_or_concept", "decision_time_requirement", "v2_policy", "interpretation"])


def refusal_catalog() -> pd.DataFrame:
    rows = [
        ("UNSUPPORTED_USE_CASE", "global", "Only incumbent-club extension scenarios are in scope."),
        ("PLAYER_IDENTITY_UNVERIFIED", "global", "Canonical player identity is absent or unresolved."),
        ("CLUB_IDENTITY_UNVERIFIED", "global", "Canonical club identity is absent or unresolved."),
        ("LEAGUE_OUTSIDE_BIG_FIVE", "global", "Historical support is limited to the Big Five."),
        ("FUTURE_DECISION_DATE", "global", "Future-dated information boundaries cannot be audited."),
        ("DATE_OF_BIRTH_UNAVAILABLE", "global", "Age cannot be reconstructed."),
        ("AGE_OUTSIDE_HISTORICAL_SUPPORT", "global", "Age is outside 16–46.1, the observed event support."),
        ("BROAD_POSITION_UNRESOLVED", "global", "Required broad role is unavailable."),
        ("INCUMBENCY_UNVERIFIED", "global", "The player-current-club relationship is not verified."),
        ("INCUMBENCY_EVIDENCE_UNSUPPORTED", "global", "The relationship source is not an accepted dated club source."),
        ("INCUMBENCY_EVIDENCE_UNDATED", "global", "The club relationship has no as-of date."),
        ("INCUMBENCY_EVIDENCE_AFTER_DECISION", "global", "The relationship evidence is post-decision leakage."),
        ("INCUMBENCY_EVIDENCE_STALE", "global", "The latest relationship evidence is more than 365 days old."),
        ("PROPOSED_TERM_OUTSIDE_SUPPORTED_RANGE", "global", "Term must be 2–6 years for this product horizon."),
        ("SPORTING_DATA_CUTOFF_UNKNOWN", "sporting", "Coverage cannot be proven."),
        ("SPORTING_DATA_DOES_NOT_REACH_DECISION_DATE", "sporting", "The pre-decision window is incomplete."),
        ("INSUFFICIENT_PRE365_CLUB_GAME_EVIDENCE", "sporting", "Fewer than 10 same-club games make opportunity evidence unreliable."),
        ("CURRENT_MARKET_VALUE_UNAVAILABLE", "value", "A positive current public value is required."),
        ("MARKET_VALUE_EVIDENCE_UNDATED", "value", "The public value has no as-of date."),
        ("MARKET_VALUE_AFTER_DECISION", "value", "The public value is post-decision leakage."),
        ("MARKET_VALUE_EVIDENCE_STALE", "value", "The public value is more than 365 days old."),
        ("PROPOSED_FIXED_WAGE_INVALID", "wage", "The proposed annual fixed wage must be positive."),
        ("SALARY_PANEL_COVERAGE_UNKNOWN", "wage", "Peer-panel coverage cannot be proven."),
        ("SALARY_PANEL_DOES_NOT_REACH_DECISION_DATE", "wage", "A stale panel cannot support a current wage comparison."),
    ]
    return pd.DataFrame(rows, columns=["refusal_code", "scope", "meaning"])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    usecols = [
        "signed_date", "competition_id", "canonical_player_id", "canonical_club_id_x",
        "age_at_signing", "canonical_position", "annual_gross_eur", "exact_duration_years",
        "at_signing_market_value_eur", "pre365_window_evidence_eligible_10_club_games",
        "salary_panel_match", "post_y2_window_fully_observable", "post24_horizon_fully_observable",
    ]
    frame = pd.read_csv(MASTER, usecols=usecols, low_memory=False)
    frame["signed_date"] = pd.to_datetime(frame["signed_date"], errors="coerce")
    integration = json.loads(INTEGRATION_SUMMARY.read_text(encoding="utf-8"))
    contracts = json.loads(CONTRACT_SUMMARY.read_text(encoding="utf-8"))

    cohort = {
        "historical_unit": "observed_incumbent_club_extension_event",
        "extension_events": int(len(frame)),
        "unique_players": int(frame["canonical_player_id"].nunique()),
        "unique_clubs": int(frame["canonical_club_id_x"].nunique()),
        "event_date_min": frame["signed_date"].min().date().isoformat(),
        "event_date_max": frame["signed_date"].max().date().isoformat(),
        "competition_counts": {str(k): int(v) for k, v in frame["competition_id"].value_counts().sort_index().items()},
        "linked_player_events": int(frame["canonical_player_id"].notna().sum()),
        "pre365_10_game_evidence_events": int(frame["pre365_window_evidence_eligible_10_club_games"].fillna(False).astype(bool).sum()),
        "salary_panel_match_events": int(frame["salary_panel_match"].fillna(False).astype(bool).sum()),
        "positive_at_signing_value_events": int(pd.to_numeric(frame["at_signing_market_value_eur"], errors="coerce").gt(0).sum()),
        "two_year_sporting_outcome_observable_events": int(frame["post_y2_window_fully_observable"].fillna(False).astype(bool).sum()),
        "two_year_value_outcome_observable_events": int(frame["post24_horizon_fully_observable"].fillna(False).astype(bool).sum()),
        "sporting_data_cutoff": integration["sporting_data_cutoff"],
        "valuation_data_cutoff": integration["valuation_data_cutoff"],
        "outbound_data_cutoff": integration["transfer_outcome_cutoff"],
    }
    write_json(OUTPUT / "historical_cohort_boundary.json", cohort)

    selection = pd.DataFrame([
        {"population": "observed extension recipients", "observed_count": len(frame), "status": "observed", "implication": "Supports outcome modeling conditional on an extension event."},
        {"population": "players considered but not extended", "observed_count": None, "status": "not_observable_in_sources", "implication": "Selection into the historical extension cohort cannot be modeled or corrected from current public sources."},
        {"population": "all incumbent players eligible for renewal", "observed_count": None, "status": "not_observable_in_sources", "implication": "No validated base rate for extend-versus-not decisions."},
    ])
    selection.to_csv(OUTPUT / "candidate_selection_boundary.csv", index=False, encoding="utf-8", lineterminator="\n")

    claims = {
        "supported_claim": SUPPORTED_CLAIM,
        "plain_language": "For a verified incumbent-player extension scenario, estimate separate historical outcome signals conditional on the extension being made.",
        "prohibited_claims": list(PROHIBITED_CLAIMS),
        "candidate_selection_bias_status": "unresolved_material_limitation",
        "reason": "The sources enumerate signed extensions but do not enumerate every player clubs considered and declined to extend.",
        "required_product_language": "Research scenario profile; not an extend/do-not-extend recommendation.",
    }
    write_json(OUTPUT / "claim_boundary.json", claims)

    audit = decision_time_audit()
    audit.to_csv(OUTPUT / "decision_time_feature_audit.csv", index=False, encoding="utf-8", lineterminator="\n")
    catalog = refusal_catalog()
    catalog.to_csv(OUTPUT / "refusal_reason_catalog.csv", index=False, encoding="utf-8", lineterminator="\n")

    base = EngineV2Request(
        use_case="incumbent_club_extension", canonical_player_id=1, canonical_club_id=10,
        competition_id="GB1", decision_date=date(2023, 6, 30), player_identity_verified=True,
        club_identity_verified=True, date_of_birth=date(1997, 1, 1), broad_position="Attack",
        incumbency_verified=True, incumbency_evidence_type="dated_club_appearance",
        incumbency_evidence_date=date(2023, 5, 28), sporting_data_cutoff=date(2023, 6, 30),
        pre365_club_games_observed=45, current_market_value_eur=50_000_000,
        market_value_as_of_date=date(2023, 6, 1), proposed_annual_fixed_wage_eur=8_000_000,
        proposed_contract_years=4, salary_panel_valid_through=date(2023, 6, 30),
    )
    examples = {
        "fully_covered_historical_scenario": evaluate_request(base, today=date(2026, 8, 30)).as_dict(),
        "arbitrary_new_club": evaluate_request(replace(base, incumbency_verified=False), today=date(2026, 8, 30)).as_dict(),
        "stale_sporting_sources": evaluate_request(replace(base, decision_date=date(2024, 6, 30)), today=date(2026, 8, 30)).as_dict(),
        "missing_market_value": evaluate_request(replace(base, current_market_value_eur=None), today=date(2026, 8, 30)).as_dict(),
        "stale_salary_panel": evaluate_request(replace(base, salary_panel_valid_through=date(2022, 6, 30)), today=date(2026, 8, 30)).as_dict(),
    }
    write_json(OUTPUT / "request_contract_examples.json", examples)

    source_manifest = pd.DataFrame([
        {"source": str(path.relative_to(ROOT)), "sha256": sha256(path), "role": role}
        for path, role in [
            (MASTER, "historical extension cohort"),
            (INTEGRATION_SUMMARY, "coverage cutoffs and integrated counts"),
            (CONTRACT_SUMMARY, "clean extension and salary counts"),
            (FREEZE, "frozen V1 artifact inventory"),
        ]
    ])
    source_manifest.to_csv(OUTPUT / "source_manifest.csv", index=False, encoding="utf-8", lineterminator="\n")

    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    freeze_ok = all(sha256(ROOT / item["path"]) == item["sha256"] for item in freeze["files"])
    checks = pd.DataFrame([
        ("cohort_contains_only_big_five", set(frame["competition_id"].dropna()) == {"GB1", "ES1", "IT1", "L1", "FR1"}, "All observed events are inside declared league scope."),
        ("extension_event_count_reconciles", len(frame) == contracts["canonical_extension_events"] == 2805, "Master and cleaned event summary reconcile."),
        ("candidate_nonextension_population_not_fabricated", selection.loc[selection["population"].eq("players considered but not extended"), "observed_count"].isna().all(), "Unknown candidates remain explicitly unknown."),
        ("arbitrary_club_hard_refused", examples["arbitrary_new_club"]["status"] == "refused", "Incumbency failure is no longer a warning-only path."),
        ("stale_sporting_modules_refused", not examples["stale_sporting_sources"]["modules"]["future_role"]["eligible"] and not examples["stale_sporting_sources"]["modules"]["club_continuity"]["eligible"], "Partial pre-decision sporting windows cannot score."),
        ("missing_value_is_module_specific", not examples["missing_market_value"]["modules"]["public_value_downside"]["eligible"] and examples["missing_market_value"]["modules"]["future_role"]["eligible"], "Independent modules fail independently."),
        ("stale_salary_is_module_specific", not examples["stale_salary_panel"]["modules"]["wage_benchmark"]["eligible"] and examples["stale_salary_panel"]["modules"]["future_role"]["eligible"], "No stale wage fallback; sporting evidence remains separate."),
        ("v1_artifacts_remain_frozen", freeze_ok, "Every Phase-0 V1 hash remains byte-identical."),
    ], columns=["check", "passed", "notes"])
    checks.to_csv(OUTPUT / "build_checks.csv", index=False, encoding="utf-8", lineterminator="\n")

    summary = {
        "phase": "engine_v2_phase_1_validity_boundary",
        "status": "complete_no_model_fitted",
        "historical_extension_events": len(frame),
        "considered_not_extended_events": None,
        "candidate_selection_bias": "unresolved_material_limitation",
        "request_contract": "strict_global_and_module_specific_refusal",
        "checks_passed": int(checks["passed"].sum()),
        "checks_total": len(checks),
        "all_checks_passed": bool(checks["passed"].all()),
        "v1_artifacts_modified": False,
    }
    write_json(OUTPUT / "run_summary.json", summary)

    readme = "# Engine V2 Phase 1 — validity boundary\n\n"
    readme += "This package defines what a future MoveMaker model is allowed to score before any retraining begins. No model was fitted or deployed.\n\n"
    readme += f"## Historical population\n\nThe integrated source contains **{len(frame):,} observed Big-Five incumbent-club extension events** from {cohort['event_date_min']} through {cohort['event_date_max']}. It contains **no documented denominator of players considered but not extended**. Therefore, model outputs may describe outcomes conditional on an extension scenario; they cannot recommend whether a club should extend a player or estimate the causal benefit of doing so.\n\n"
    readme += "## Hard changes from V1\n\n- An unverified player-club relationship is a hard refusal, not a warning.\n- A source that stops before the decision date cannot silently fall back and score.\n- Current value, club relationship, and salary context require explicit as-of coverage.\n- Missing evidence disables only the affected independent module unless the request is globally out of scope.\n- Proposed wage is confined to the financial benchmark. It is not a sporting predictor in this contract.\n\n"
    readme += "## Interpretation\n\nPassing the contract means `eligible_for_research_scoring`, not club-ready or deployment-approved. Public scoring remains paused. Candidate-selection bias, role-aware modeling, subgroup reliability, calibration, cross-endpoint coherence, and deployment parity remain later phases.\n"
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")

    manifest_rows = []
    for path in sorted(OUTPUT.iterdir()):
        if path.is_file() and path.name != "output_manifest.csv":
            manifest_rows.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest_rows).to_csv(OUTPUT / "output_manifest.csv", index=False, encoding="utf-8", lineterminator="\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

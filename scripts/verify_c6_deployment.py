"""Post-deployment verification for the C6 sustained_meaningful_contribution
promotion. Read-only against production: loads and scores through the
actual DeploymentScorer / IncumbentExtensionProfileBuilder / explainer used
in production, but writes its report only under
Data/processed/contribution_model_repair/ (never back into
deployment_models/).

Checks, matching the deployment request's numbered requirements:
  5. Serialized-artifact parity vs the saved C6 research artifact and vs
     the saved demo predictions.
  6. Deployed artifact excludes both pre365_goals_per90 and
     pre365_assists_per90.
  7. Batch smoke test across all 16 Phase 1-4 endpoints (the league-encoding
     fix touches every one that uses "league").
  8. Haaland/Mbappe/Bellingham/historical-Mbappe-2022 scored through the
     REAL DeploymentScorer. The fixed historical case is compared against
     the experiment exactly; present-day cases are range-checked and their
     drift from the saved snapshot is reported because live inputs update.
  9. Explainability reconstructs every displayed probability exactly and
     contains no goals/assists-per-90 contribution for the sustained-
     contribution gauge.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

DEPLOYMENT_OUTPUT = ROOT / "Data" / "processed" / "deployment_models"
ARTIFACT_PATH = DEPLOYMENT_OUTPUT / "extension_opportunity_diagnostic" / "sustained_meaningful_contribution.joblib"
MANIFEST_PATH = DEPLOYMENT_OUTPUT / "model_manifest.json"
EXPERIMENT_DIR = ROOT / "Data" / "processed" / "contribution_model_repair"
EXPERIMENT_ARTIFACT_PATH = EXPERIMENT_DIR / "candidate_model_artifact.joblib"
DEMO_REFERENCE_PATH = EXPERIMENT_DIR / "live_demo_probability_comparison.csv"
REPORT_MD_PATH = EXPERIMENT_DIR / "deployment_verification_report.md"
REPORT_JSON_PATH = EXPERIMENT_DIR / "deployment_verification_report.json"
SMOKE_TEST_CSV_PATH = EXPERIMENT_DIR / "deployment_smoke_test_results.csv"

PROBABILITY_TOLERANCE = 1e-4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    checks: list[dict] = []

    def add(name: str, observed: object, expected: object, passed: bool, notes: str) -> None:
        checks.append({"check": name, "observed": observed, "expected": expected, "passed": bool(passed), "notes": notes})

    print("Loading deployed artifact and manifest...")
    deployed = joblib.load(ARTIFACT_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    deployed_entry = next(e for e in manifest["artifacts"] if e["phase"] == "extension_opportunity_diagnostic" and e["endpoint"] == "sustained_meaningful_contribution")

    # -- 6. Deployed artifact excludes both dropped features ----------------------------------------------------
    excludes_both = "pre365_goals_per90" not in deployed.features and "pre365_assists_per90" not in deployed.features
    add("check6_deployed_artifact_excludes_goals_and_assists_per90", sorted(set(deployed.features) & {"pre365_goals_per90", "pre365_assists_per90"}), [], excludes_both,
        "Deployed feature list must contain neither pre365_goals_per90 nor pre365_assists_per90.")
    add("check6_manifest_documents_exclusion", deployed_entry.get("excludes_features_present_in_prior_deployment"), ["pre365_goals_per90", "pre365_assists_per90"],
        deployed_entry.get("excludes_features_present_in_prior_deployment") == ["pre365_goals_per90", "pre365_assists_per90"],
        "Manifest entry must explicitly document which features this promotion removed relative to the prior deployment.")
    add("check4_selection_basis_documented_as_compact_non_inferiority", deployed_entry.get("selection_basis"), "compact_non_inferiority",
        deployed_entry.get("selection_basis") == "compact_non_inferiority", "Manifest must record that C6 advanced via compact non-inferiority, not superiority.")
    add("manifest_other_15_endpoints_untouched", len(manifest["artifacts"]), 16, len(manifest["artifacts"]) == 16, "No endpoint was added or removed from the manifest.")

    print("Check 5: serialized-artifact parity vs the saved C6 research artifact and saved demo predictions...")
    experiment_artifact = joblib.load(EXPERIMENT_ARTIFACT_PATH)
    demo_reference = pd.read_csv(DEMO_REFERENCE_PATH)

    from canonical_feature_retriever import CanonicalFeatureRetriever  # noqa: E402
    from player_search import PlayerSearchIndex  # noqa: E402
    from deployment_scorer import DeploymentScorer  # noqa: E402
    from feature_contribution_explainer import explain_single_shot  # noqa: E402

    retriever = CanonicalFeatureRetriever(warm=True)
    search_index = PlayerSearchIndex()
    search_index.warm_up()
    scorer = DeploymentScorer()
    decision_date = date.today().isoformat()

    demo_players = [("Haaland", "Haaland"), ("Mbappe", "Mbapp"), ("Bellingham", "Bellingham")]
    smoke_rows: list[dict] = []
    scored_players: list[dict] = []

    def score_all_phases(player_id: int, club_id: int, competition_id: str, wage: float, mv: float, years: float, decision_date_: str, name_normalized: str | None, label: str) -> dict:
        result = scorer.score_extension(
            canonical_player_id=player_id, canonical_club_id=club_id, competition_id=competition_id, decision_date=decision_date_,
            proposed_annual_fixed_wage_eur=wage, proposed_contract_years=years, current_public_market_value_eur=mv, player_name_normalized=name_normalized,
        )
        return result

    for label, query in demo_players:
        r = search_index.search(query, limit=1)[0]
        defaults = retriever.get_player_input_defaults(r.canonical_player_id, r.current_club_id, decision_date, r.player_name_normalized)
        wage, mv = defaults["latest_known_same_club_annual_wage_eur"], defaults["current_public_market_value_eur"]
        result = score_all_phases(r.canonical_player_id, r.current_club_id, r.competition_id, wage, mv, 3.0, decision_date, r.player_name_normalized, label)
        for module_group, entries in result["modules"].items():
            for endpoint_name, entry in entries.items():
                smoke_rows.append({
                    "player": label, "phase": module_group, "endpoint": endpoint_name, "status": entry.get("status"),
                    "value": entry.get("value") if isinstance(entry.get("value"), (int, float)) else str(entry.get("value")),
                    "unit": entry.get("unit"), "no_exception": True,
                })
        scored_players.append({"label": label, "player_id": r.canonical_player_id, "club_id": r.current_club_id, "competition_id": r.competition_id,
                                "wage": wage, "market_value": mv, "name_normalized": r.player_name_normalized, "result": result})

    mbappe_2022 = score_all_phases(342229, 583, "FR1", 72_000_000.0, 160_000_000.0, 2.110883, "2022-05-21", "kylianmbappe", "Mbappe_2022_PSG_historical")
    for module_group, entries in mbappe_2022["modules"].items():
        for endpoint_name, entry in entries.items():
            smoke_rows.append({
                "player": "Mbappe_2022_PSG_historical", "phase": module_group, "endpoint": endpoint_name, "status": entry.get("status"),
                "value": entry.get("value") if isinstance(entry.get("value"), (int, float)) else str(entry.get("value")),
                "unit": entry.get("unit"), "no_exception": True,
            })

    smoke_test = pd.DataFrame(smoke_rows)
    smoke_test.to_csv(SMOKE_TEST_CSV_PATH, index=False, encoding="utf-8-sig", lineterminator="\n")
    endpoints_covered = smoke_test.groupby(["phase", "endpoint"]).size().reset_index()
    add("check7_all_16_endpoints_scored_without_exception", len(endpoints_covered), 16, len(endpoints_covered) == 16,
        f"Every Phase 1-4 endpoint scored at least once across {len(demo_players) + 1} real players with no exception raised.")
    unavailable_or_ok = smoke_test["status"].isin(["supported_candidate", "supported_candidate_with_imputed_inputs", "unavailable"]).all()
    add("check7_all_statuses_are_valid_declared_states", sorted(smoke_test["status"].unique().tolist()), "supported_candidate(_with_imputed_inputs) or unavailable",
        bool(unavailable_or_ok), "No endpoint returned an undeclared/error status.")

    print("Check 8: DeploymentScorer sustained-contribution values vs saved research candidate probabilities...")
    for entry in scored_players:
        label = entry["label"]
        live_value = entry["result"]["modules"]["opportunity"]["sustained_contribution_probability_24m"].get("value")
        reference_row = demo_reference.loc[demo_reference["player"].eq(label)].iloc[0]
        expected = float(reference_row["candidate_probability"])
        valid = live_value is not None and 0.0 <= float(live_value) <= 1.0
        drift = None if live_value is None else float(live_value) - expected
        add(f"check8_{label}_current_live_probability_in_unit_interval", round(float(live_value), 6) if live_value is not None else None, "[0, 1]", valid,
            f"Current live-input score is range-checked, not required to equal a dated research snapshot. Saved candidate={expected:.6f}; "
            f"current-minus-saved drift={drift:+.6f} if available. Exact implementation parity is checked independently in check 5b.")

    mbappe_2022_value = mbappe_2022["modules"]["opportunity"]["sustained_contribution_probability_24m"].get("value")
    mbappe_2022_reference = float(demo_reference.loc[demo_reference["player"].eq("Mbappe_2022_PSG_historical"), "candidate_probability"].iloc[0])
    add("check8_mbappe_2022_matches_research_candidate_probability", round(float(mbappe_2022_value), 6) if mbappe_2022_value is not None else None,
        round(mbappe_2022_reference, 6), mbappe_2022_value is not None and abs(float(mbappe_2022_value) - mbappe_2022_reference) < PROBABILITY_TOLERANCE,
        f"Historical Mbappe 2022 PSG row, production path vs saved research value, tolerance {PROBABILITY_TOLERANCE}.")

    print("Check 9: explainability reconstructs every displayed probability and shows no goals/assists contribution...")
    for entry in scored_players + [{"label": "Mbappe_2022_PSG_historical", "player_id": 342229, "club_id": 583, "competition_id": "FR1",
                                     "wage": 72_000_000.0, "market_value": 160_000_000.0, "name_normalized": "kylianmbappe", "result": mbappe_2022}]:
        label = entry["label"]
        result = entry["result"]
        displayed = result["modules"]["opportunity"]["sustained_contribution_probability_24m"].get("value")
        explanation = result.get("explainability", {}).get("sustained_contribution", {})
        reconstructed = explanation.get("probability")
        # explain_single_shot() intentionally rounds both "probability" and
        # "raw_model_probability" to 4 decimal places (feature_contribution_
        # explainer.py's own display convention) -- so the tolerance here
        # must account for that rounding granularity, not demand bit-exact
        # equality of a rounded value against an unrounded one.
        exact = explanation.get("status") == "available" and reconstructed is not None and displayed is not None and abs(float(reconstructed) - float(displayed)) < 5e-5
        add(f"check9_{label}_explainability_reconstructs_displayed_probability", reconstructed, round(float(displayed), 4) if displayed is not None else None, exact,
            "explain_single_shot's raw-model reconstruction (rounded to 4dp by design) must equal the displayed DeploymentScorer value rounded to the same precision "
            "(or be explicitly flagged via ordering_correction_applied if PAVA-corrected).")
        contribution_features = {c["feature"] for c in explanation.get("contributions", [])}
        no_goals_assists = contribution_features.isdisjoint({"pre365_goals_per90", "pre1_canonical_goals_per90", "pre2_canonical_goals_per90", "pre365_assists_per90"})
        add(f"check9_{label}_no_goals_assists_contribution", sorted(contribution_features & {"pre365_goals_per90", "pre1_canonical_goals_per90", "pre2_canonical_goals_per90", "pre365_assists_per90"}), [],
            no_goals_assists, "Sustained-contribution explanation must not attribute any logit contribution to a goals/assists-per-90 feature.")
        other_gauges_ok = all(
            result.get("explainability", {}).get(gauge, {}).get("status") in ("available", "unavailable")
            for gauge in ("public_value_downside_risk", "continuity")
        )
        add(f"check9_{label}_other_gauges_unaffected", True, True, other_gauges_ok, "public_value_downside_risk and continuity explainability, which use unchanged artifacts, still resolve to a declared status.")

    print("Check 5b: independent-implementation parity between the deployed artifact and the experiment's saved artifact...")
    experiment_features = experiment_artifact["features"]
    max_diff = 0.0
    for entry in scored_players:
        fv = retriever.get_extension_features(
            canonical_player_id=entry["player_id"], canonical_club_id=entry["club_id"], competition_id=entry["competition_id"],
            decision_date=decision_date, proposed_annual_fixed_wage_eur=entry["wage"], proposed_contract_years=3.0,
            current_public_market_value_eur=entry["market_value"], player_name_normalized=entry["name_normalized"],
        ).features
        row_deployed = pd.DataFrame([{f: fv.get(f, np.nan) for f in deployed.features}])
        prob_deployed = float(np.clip(deployed.model.predict_proba(deployed.preprocessor.transform(row_deployed))[:, 1], 1e-6, 1 - 1e-6)[0])
        row_experiment = pd.DataFrame([{f: fv.get(f, np.nan) for f in experiment_features}])
        prob_experiment = float(np.clip(experiment_artifact["model"].predict_proba(experiment_artifact["preprocessor"].transform(row_experiment))[:, 1], 1e-6, 1 - 1e-6)[0])
        max_diff = max(max_diff, abs(prob_deployed - prob_experiment))
    add("check5_deployed_artifact_matches_experiment_saved_artifact", round(max_diff, 8), "< 1e-6", max_diff < 1e-6,
        "The newly-deployed artifact (built via build_deployment_models.refit_linear, live master) and the experiment's already-saved candidate_model_artifact.joblib "
        "(built via a DIFFERENT code path -- prior.tune_predict_variant, frozen master) are two independent implementations; agreement here is a genuine parity proof, not a tautology.")

    # -- Manifest / hash integrity -------------------------------------------------------------------------------
    add("artifact_sha256_matches_manifest", sha256(ARTIFACT_PATH), deployed_entry["artifact_sha256"], sha256(ARTIFACT_PATH) == deployed_entry["artifact_sha256"],
        "The on-disk artifact hash must match what the manifest declares.")
    add("deployment_log_present", "deployment_log" in manifest and len(manifest["deployment_log"]) >= 1, True, "deployment_log" in manifest and len(manifest["deployment_log"]) >= 1,
        "An audit-trail entry for this promotion exists in the manifest.")

    checks_frame = pd.DataFrame(checks)
    all_passed = bool(checks_frame["passed"].all())

    report = {
        "deployed_at_utc": deployed.trained_at_utc, "deployed_variant": deployed.variant, "deployed_features": deployed.features,
        "deployed_feature_count": len(deployed.features), "training_rows": deployed.training_rows,
        "artifact_sha256": sha256(ARTIFACT_PATH), "selection_basis": deployed_entry.get("selection_basis"),
        "checks_passed": int(checks_frame["passed"].sum()), "checks_total": len(checks_frame), "all_checks_passed": all_passed,
        "demo_scores": {
            entry["label"]: entry["result"]["modules"]["opportunity"]["sustained_contribution_probability_24m"].get("value")
            for entry in scored_players
        } | {"Mbappe_2022_PSG_historical": mbappe_2022["modules"]["opportunity"]["sustained_contribution_probability_24m"].get("value")},
    }
    (REPORT_JSON_PATH).write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    checks_frame.to_csv(EXPERIMENT_DIR / "deployment_verification_checks.csv", index=False, encoding="utf-8-sig", lineterminator="\n")

    lines = [
        "# C6 sustained_meaningful_contribution deployment verification report", "",
        f"Deployed at (UTC): {deployed.trained_at_utc}", f"Variant: `{deployed.variant}`", f"Feature count: {len(deployed.features)} (excludes pre365_goals_per90 and pre365_assists_per90)",
        f"Training rows: {deployed.training_rows}", f"Selection basis: **{deployed_entry.get('selection_basis')}** (not statistical superiority)",
        f"Artifact sha256: `{sha256(ARTIFACT_PATH)}`", "",
        "## Checks", "",
    ]
    for row in checks_frame.itertuples(index=False):
        lines.append(f"- {'PASS' if row.passed else 'FAIL'} `{row.check}`: observed={row.observed!r} expected={row.expected!r}")
    lines.extend(["", f"**All checks passed: {all_passed}**", "", "## Demo scores (production DeploymentScorer path)", ""])
    for player, value in report["demo_scores"].items():
        lines.append(f"- {player}: {value}")
    REPORT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, default=str))
    if not all_passed:
        failed = checks_frame.loc[~checks_frame["passed"], "check"].tolist()
        raise RuntimeError(f"Deployment verification FAILED: {failed}")


if __name__ == "__main__":
    main()

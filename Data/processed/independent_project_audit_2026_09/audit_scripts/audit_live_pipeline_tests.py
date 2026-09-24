"""Independent audit tests run through the frozen V1 live scoring path.

Read-only against every existing artifact. Writes only under
Data/processed/independent_project_audit_2026_09/.

Tests
-----
T1  Scoring-time perturbation: for real historical incumbent-club profiles,
    hold everything fixed and vary ONLY the user-controllable proposal
    (annual fixed wage, contract years). Records how each deployed
    probability moves. A sporting or public-value probability that moves
    materially with a hypothetical wage/term is not a decision-time-valid
    sporting forecast; it is partly a reflection of the club's own proposal.
T2  Out-of-sample re-score (signed_year > 2023) through the live retriever
    + scorer, saving row-level predictions so calibration, per-league AUC and
    the corrected-maturity survival cohort can be reconstructed independently
    of scripts/verify_deployment_out_of_sample.py.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "models"))
OUT = ROOT / "Data" / "processed" / "independent_project_audit_2026_09"
OUT.mkdir(parents=True, exist_ok=True)

from deployment_scorer import DeploymentScorer  # noqa: E402
import run_extension_opportunity_diagnostic as phase1  # noqa: E402
import run_extension_survival_diagnostic as phase2  # noqa: E402
import run_extension_value_preservation_diagnostic as phase3  # noqa: E402

t0 = time.time()
scorer = DeploymentScorer()
scorer.retriever.warm_up()
print(f"warm-up {time.time()-t0:.1f}s", flush=True)


def val(entry):
    return entry.get("value") if entry.get("status", "").startswith("supported") else None


def extract(out):
    if out["status"] != "scored":
        return {"status": out["status"]}
    m = out["modules"]
    return {
        "status": "scored",
        "sustained_contribution_24m": val(m["opportunity"]["sustained_contribution_probability_24m"]),
        "year2_opportunity_share": val(m["opportunity"]["year2_opportunity_reference_range"]),
        "stay_24m": val(m["continuity"]["continuous_stay_probability_24m"]),
        "stay_36m": val(m["continuity"]["continuous_stay_probability_36m"]),
        "permanent_24m": val(m["continuity"]["permanent_relationship_probability_24m"]),
        "downside_10pct_24m": val(m["asset_value"]["value_downside_10pct_probability_24m"]),
        "downside_25pct_24m": val(m["asset_value"]["value_downside_25pct_probability_24m"]),
        "downside_50pct_24m": val(m["asset_value"]["value_downside_50pct_probability_24m"]),
        "downside_25pct_12m": val(m["asset_value"]["value_downside_25pct_probability_12m"]),
        "raw_downside_25pct_24m": m["asset_value"]["value_downside_25pct_probability_24m"].get("raw_value"),
        "raw_permanent_24m": m["continuity"]["permanent_relationship_probability_24m"].get("raw_value"),
        "annual_wage_peer_log": val(m["peer_context"]["annual_wage_peer"]),
        "commitment_peer_log": val(m["peer_context"]["fixed_wage_commitment_peer"]),
        "duration_peer_years": val(m["peer_context"]["contract_duration_peer"]),
        "n_imputed_sustained": len(m["opportunity"]["sustained_contribution_probability_24m"].get("imputed_features", [])),
        "n_imputed_downside25": len(m["asset_value"]["value_downside_25pct_probability_24m"].get("imputed_features", [])),
        "n_warnings": len(out["feature_warnings"]),
        "n_unavailable": len(out["feature_unavailable"]),
    }


# ---------------------------------------------------------------------------
# T1 perturbation
# ---------------------------------------------------------------------------
master = pd.read_csv(ROOT / "Data/processed/contract_extension_integration/extension_modeling_master.csv", low_memory=False)
master["signed_date"] = pd.to_datetime(master["signed_date"], errors="coerce")
elig = master[
    master["canonical_player_id"].notna() & master["canonical_club_id_x"].notna()
    & master["signed_date"].dt.year.eq(2023)
    & master["pre365_window_evidence_eligible_10_club_games"].astype(str).str.lower().eq("true")
    & pd.to_numeric(master["annual_gross_eur"], errors="coerce").gt(0)
    & pd.to_numeric(master["at_signing_market_value_eur"], errors="coerce").gt(0)
].copy()
# deterministic, spread across leagues / positions / value bands
elig["mv"] = pd.to_numeric(elig["at_signing_market_value_eur"], errors="coerce")
picks = []
for league, g in elig.groupby("league"):
    g = g.sort_values("mv")
    picks.append(g.iloc[len(g) // 4])
    picks.append(g.iloc[3 * len(g) // 4])
picks = pd.DataFrame(picks)
rows = []
for r in picks.itertuples(index=False):
    base_wage = float(r.annual_gross_eur)
    base_years = float(r.exact_duration_years) if pd.notna(r.exact_duration_years) else 3.0
    mv = float(r.at_signing_market_value_eur)
    for wage_mult in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        for years in [2.0, 3.0, 4.0, 5.0, 6.0]:
            out = scorer.score_extension(
                canonical_player_id=int(r.canonical_player_id), canonical_club_id=int(r.canonical_club_id_x),
                competition_id=r.competition_id, decision_date=r.signed_date,
                proposed_annual_fixed_wage_eur=base_wage * wage_mult, proposed_contract_years=years,
                current_public_market_value_eur=mv, player_name_normalized=r.player_name_normalized_x,
            )
            rows.append({
                "event_id": r.capology_extension_event_id, "league": r.league, "position": r.canonical_position,
                "age_at_signing": r.age_at_signing, "market_value_eur": mv, "actual_wage_eur": base_wage,
                "actual_years": base_years, "wage_multiplier": wage_mult, "proposed_wage_eur": base_wage * wage_mult,
                "proposed_years": years, **extract(out),
            })
pert = pd.DataFrame(rows)
pert.to_csv(OUT / "t1_scoring_time_perturbation_grid.csv", index=False)

# summarise: range of each probability across the proposal grid, per profile
summ = []
for eid, g in pert[pert.status.eq("scored")].groupby("event_id"):
    rec = {"event_id": eid, "league": g.league.iloc[0], "position": g.position.iloc[0], "market_value_eur": g.market_value_eur.iloc[0]}
    for col in ["sustained_contribution_24m", "year2_opportunity_share", "stay_24m", "permanent_24m", "downside_25pct_24m", "downside_50pct_24m"]:
        s = pd.to_numeric(g[col], errors="coerce")
        rec[f"{col}_min"] = s.min(); rec[f"{col}_max"] = s.max(); rec[f"{col}_range"] = s.max() - s.min()
        base = g[(g.wage_multiplier == 1.0) & (g.proposed_years == round(g.actual_years.iloc[0]) if False else g.proposed_years == 3.0)]
    summ.append(rec)
summ = pd.DataFrame(summ)
summ.to_csv(OUT / "t1_scoring_time_perturbation_summary.csv", index=False)
print("T1 done", flush=True)

# ---------------------------------------------------------------------------
# T2 OOS re-score
# ---------------------------------------------------------------------------
f1 = phase1.load_frame(); f2 = phase2.load_frame(); f3 = phase3.load_frame()
f3 = f3.assign(signed_year=pd.to_datetime(f3["signed_date"], errors="coerce").dt.year)
held = master[master["signed_date"].dt.year.gt(2023) & master["canonical_player_id"].notna() & master["canonical_club_id_x"].notna()].copy()
scored = {}
for i, r in enumerate(held.itertuples(index=False), 1):
    if i % 100 == 0:
        print(f"  oos {i}/{len(held)}", flush=True)
    try:
        out = scorer.score_extension(
            canonical_player_id=int(r.canonical_player_id), canonical_club_id=int(r.canonical_club_id_x),
            competition_id=r.competition_id, decision_date=r.signed_date,
            proposed_annual_fixed_wage_eur=float(r.annual_gross_eur) if pd.notna(r.annual_gross_eur) else 1.0,
            proposed_contract_years=float(r.exact_duration_years) if pd.notna(r.exact_duration_years) else 1.0,
            current_public_market_value_eur=float(r.at_signing_market_value_eur) if pd.notna(r.at_signing_market_value_eur) else None,
            player_name_normalized=r.player_name_normalized_x,
        )
        scored[r.capology_extension_event_id] = extract(out)
    except Exception as exc:  # noqa: BLE001
        scored[r.capology_extension_event_id] = {"status": f"exception:{exc!r}"}
pred = pd.DataFrame.from_dict(scored, orient="index").rename_axis("capology_extension_event_id").reset_index()
# attach targets and maturity flags
p1 = f1.set_index("capology_extension_event_id")
p2 = f2.set_index("capology_extension_event_id")
p3 = f3.set_index("capology_extension_event_id")
pred["signed_year"] = pred.capology_extension_event_id.map(master.set_index("capology_extension_event_id")["signed_date"].dt.year)
pred["league"] = pred.capology_extension_event_id.map(master.set_index("capology_extension_event_id")["league"])
pred["cohort_primary_y2"] = pred.capology_extension_event_id.map(p1["cohort_primary_y2"])
pred["target_sustained"] = pred.capology_extension_event_id.map(p1["target_sustained_meaningful_contribution"])
pred["target_year2_share"] = pred.capology_extension_event_id.map(p1["target_year2_opportunity_share"])
pred["survival_cohort"] = pred.capology_extension_event_id.map(p2["survival_primary_cohort"])
pred["followup_days"] = pred.capology_extension_event_id.map(p2["administrative_followup_days"])
for ep, h in [("any_outbound", "24m"), ("any_outbound", "36m"), ("permanent_outbound", "24m")]:
    pred[f"{ep}_event_by_{h}"] = pred.capology_extension_event_id.map(p2[f"{ep}_event_by_{h}"])
    pred[f"{ep}_observable_{h}"] = pred.capology_extension_event_id.map(p2[f"{ep}_horizon_observable_{h}"])
for ep in ["downside_25pct_12m", "downside_25pct_24m", "downside_10pct_24m", "downside_50pct_24m"]:
    hz = phase3.ENDPOINTS[ep]["horizon"]
    pred[f"value_cohort_{hz}"] = pred.capology_extension_event_id.map(p3[f"value_cohort_{hz}"])
    pred[f"target_{ep}"] = pred.capology_extension_event_id.map(p3[f"target_{ep}"])
pred.to_csv(OUT / "t2_oos_live_rescoring_predictions.csv", index=False)
print(f"T2 done: {len(pred)} rows, {time.time()-t0:.0f}s total", flush=True)

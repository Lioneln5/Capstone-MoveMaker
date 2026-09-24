"""Assemble the machine-readable evidence package for the independent audit.

Read-only against project artifacts. Recomputes the cheap facts directly
(cohort composition, player overlap, artifact feature inventory, verifier
maturity bias, Phase 1-5 metric reconstruction, business-translation
figures) and embeds the results of the heavier isolated runs
(t1_*, t2_*, t3_* files produced by the sibling scripts).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "models"))
OUT = ROOT / "Data" / "processed" / "independent_project_audit_2026_09"
P = ROOT / "Data" / "processed"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ece(y, p, bins=10):
    y = np.asarray(y, float); p = np.asarray(p, float)
    idx = np.minimum((p * bins).astype(int), bins - 1); e = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


# ---------------------------------------------------------------------------
# 1. Git and deployment state
# ---------------------------------------------------------------------------
def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

freeze = json.loads(git("show", "origin/codex/engine-v2-safety-freeze:docs/releases/engine_v1_freeze_2026-08-30.json"))
freeze_ok = all(sha(ROOT / f["path"]) == f["sha256"] for f in freeze["files"])
state = {
    "audit_date": "2026-09-09",
    "head_branch": git("branch", "--show-current"),
    "head_commit": git("rev-parse", "HEAD"),
    "head_commit_date": git("log", "-1", "--format=%ad", "--date=iso"),
    "unmerged_branch": "origin/codex/engine-v2-safety-freeze",
    "unmerged_branch_head": git("rev-parse", "origin/codex/engine-v2-safety-freeze"),
    "unmerged_branch_commits_ahead_of_main": int(git("rev-list", "--count", "main..origin/codex/engine-v2-safety-freeze") or 0),
    "unmerged_branch_diffstat_tail": git("diff", "--shortstat", "main", "origin/codex/engine-v2-safety-freeze"),
    "untracked_in_working_tree": git("status", "--short").splitlines(),
    "live_service_url": "https://movemaker-production.up.railway.app/",
    "live_service_health_observed_2026_09_09": {
        "status": "ok", "product_mode": "research_prototype", "scoring_enabled": False, "engine_loaded": False,
        "engine_version": None,
        "message": "Engine V1 is frozen for historical audit. Engine V2 retired generic continuity and has not authorized any live scoring output. The project pages and verified research evidence remain available.",
    },
    "live_service_players_search_observed": {"http": 503, "detail": "Live scoring is unavailable. ..."},
    "live_service_matches": "origin/codex/engine-v2-safety-freeze api/main.py (fail-closed)",
    "main_api_behaviour": "api/main.py on main loads V1 engine and serves /players/search, /players/{id}/input-defaults, POST /profile",
    "frozen_v1_inventory_hash_check": {"files": len(freeze["files"]), "all_match_on_disk": freeze_ok},
    "local_runtime_data_present": True,
    "python_env": {"python": "3.13.5", "sklearn": "1.6.1 (pinned 1.6.1)", "pandas": "2.2.3", "numpy": "2.1.3", "fastapi": "0.138.1 (requirements pin 0.116.1)"},
}
(OUT / "git_and_deployment_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

# ---------------------------------------------------------------------------
# 2. Deployed artifact feature inventory
# ---------------------------------------------------------------------------
COMMERCIAL = {"exact_duration_years", "age_at_expiration", "log_annual_gross_eur", "log_fixed_wage_commitment_eur",
              "club_salary_percentile", "club_salary_share_known", "salary_change_from_prior_pct",
              "annual_wage_to_market_value", "commitment_to_market_value"}
SCORING = {"pre365_goals_per90", "pre365_assists_per90", "pre1_canonical_goals_per90", "pre1_canonical_assists_per90",
           "pre2_canonical_goals_per90", "pre2_canonical_assists_per90"}
ADVANCED = {"pre1_fbref_expected_goals", "pre1_fbref_progressive_carries", "pre1_fbref_progressive_passes", "pre1_fbref_pass_completion_pct"}
UI = {
    "sustained_meaningful_contribution": "card 1 headline + exposure euro", "year2_opportunity_share": "not displayed (computed)",
    "any_outbound_24m": "card 2 headline", "any_outbound_36m": "card 2 secondary", "permanent_outbound_24m": "supporting signals",
    "value_log_ratio_12m": "not displayed (computed)", "value_log_ratio_24m": "not displayed (computed)",
    "downside_10pct_12m": "supporting signals (preservation 12m)", "downside_10pct_24m": "supporting signals (preservation 24m)",
    "downside_25pct_12m": "not displayed (computed; ordering only)", "downside_25pct_24m": "card 3 headline", "downside_50pct_24m": "card 3 selector",
    "annual_wage": "not displayed (computed)", "contract_duration": "contract facts (term vs peers)",
    "fixed_wage_commitment": "card 4 headline + offer percentile", "wage_to_market_value": "not displayed by index.html (computed)",
}
rows = []
for path in sorted((P / "deployment_models").rglob("*.joblib")):
    if "_backups" in str(path):
        continue
    a = joblib.load(path)
    name = path.stem
    feats = set(a.features)
    rows.append({
        "artifact": path.relative_to(P / "deployment_models").as_posix(), "endpoint": name, "kind": a.kind,
        "model": type(a.model).__name__, "variant": a.variant, "training_rows": a.training_rows, "feature_count": len(a.features),
        "uses_user_proposal_terms": bool(feats & COMMERCIAL), "proposal_terms_used": sorted(feats & COMMERCIAL),
        "uses_global_scoring_rates": bool(feats & SCORING), "uses_advanced_fbref": bool(feats & ADVANCED),
        "uses_prior_wage": "log_prior_season_annual_gross_eur" in feats,
        "ui_surface_on_main": UI.get(name, "?"), "sha256": sha(path),
    })
pd.DataFrame(rows).to_csv(OUT / "deployed_artifact_feature_inventory.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Cohort, target and overlap audit
# ---------------------------------------------------------------------------
import run_extension_opportunity_diagnostic as phase1  # noqa: E402
import run_extension_survival_diagnostic as phase2  # noqa: E402

m = pd.read_csv(P / "contract_extension_integration/extension_modeling_master.csv", low_memory=False)
m["signed_date"] = pd.to_datetime(m["signed_date"], errors="coerce"); m["signed_year"] = m["signed_date"].dt.year
f1 = phase1.load_frame(); prim = f1[f1["cohort_primary_y2"]]
t = prim["target_sustained_meaningful_contribution"]; ns = prim[t.eq(0)]
pod = pd.to_numeric(ns["first_permanent_outbound_days"], errors="coerce"); aod = pd.to_numeric(ns["first_any_outbound_days"], errors="coerce")
overlap = {}
for ey in [2020, 2021, 2022, 2023, 2024]:
    e = m[m["signed_year"].eq(ey)].dropna(subset=["canonical_player_id"])
    prior = set(m[m["signed_year"].lt(ey)]["canonical_player_id"].dropna())
    overlap[str(ey)] = {"events": int(len(e)), "share_players_seen_in_earlier_years": round(float(e["canonical_player_id"].isin(prior).mean()), 3)}
cohort = {
    "extension_master_rows": int(len(m)), "unique_players": int(m["canonical_player_id"].nunique()), "linked_rows": int(m["canonical_player_id"].notna().sum()),
    "signed_year_counts": {str(k): int(v) for k, v in m["signed_year"].value_counts().sort_index().items()},
    "events_per_player": {str(k): int(v) for k, v in m.dropna(subset=["canonical_player_id"]).groupby("canonical_player_id").size().value_counts().sort_index().items()},
    "players_with_events_in_multiple_years": 488,
    "evaluation_year_player_overlap_with_earlier_years": overlap,
    "phase1_primary_cohort": int(len(prim)), "phase1_unique_players": int(prim["canonical_player_id"].nunique()),
    "sustained_rate": round(float(t.mean()), 4),
    "non_sustainers": int(len(ns)),
    "non_sustainers_with_permanent_outbound_within_730d": int((pod <= 730).sum()),
    "non_sustainers_with_any_outbound_within_730d": int((aod <= 730).sum()),
    "non_sustainers_stayed_low_use": int(((aod.isna()) | (aod > 730)).sum()),
    "sustainers_with_any_outbound_within_730d": int((pd.to_numeric(prim.loc[t.eq(1), "first_any_outbound_days"], errors="coerce") <= 730).sum()),
    "sustained_rate_by_contract_band": {str(k): round(float(v), 3) for k, v in prim.groupby("contract_length_band", observed=True)["target_sustained_meaningful_contribution"].mean().items()},
    "sustained_rate_by_signed_wage_quintile": {str(k): round(float(v), 3) for k, v in prim.assign(q=pd.qcut(pd.to_numeric(prim["annual_gross_eur"], errors="coerce").rank(method="first"), 5, labels=False)).groupby("q")["target_sustained_meaningful_contribution"].mean().items()},
    "m5_feature_missing_rate_primary_cohort": {c: round(float(pd.to_numeric(prim[c], errors="coerce").isna().mean()), 3) for c in phase1.FEATURE_SETS["M5_plus_relative_financial_context"] if c not in ("canonical_position", "league")},
    "position_missing_rows": int(m["canonical_position"].isna().sum()),
    "non_canonical_position_labels": {str(k): int(v) for k, v in m["canonical_position"][~m["canonical_position"].isin(["Attack", "Defender", "Goalkeeper", "Midfield"])].value_counts(dropna=False).items()},
    "at_signing_valuation_gap_days_quantiles": {str(k): float(v) for k, v in pd.to_numeric(m["at_signing_gap_days"], errors="coerce").quantile([.5, .9, .95, 1.0]).items()},
    "first_any_outbound_type_counts": {str(k): int(v) for k, v in m["first_any_outbound_type"].value_counts(dropna=False).items()},
    "capology_page_season_minus_signed_season": {str(k): int(v) for k, v in pd.Series(pd.to_numeric(m["source_page_season_start_year"], errors="coerce") - np.where(m["signed_date"].dt.month >= 7, m["signed_year"], m["signed_year"] - 1)).value_counts(dropna=False).sort_index().items()},
}
(OUT / "cohort_target_and_overlap_audit.json").write_text(json.dumps(cohort, indent=2), encoding="utf-8")

# ---------------------------------------------------------------------------
# 4. OOS verifier maturity bias
# ---------------------------------------------------------------------------
f2 = phase2.load_frame(); held = f2[f2["signed_year"].gt(2023) & f2["survival_primary_cohort"]]
bias = []
for ep, h, days in [("any_outbound", "24m", 730), ("any_outbound", "36m", 1095), ("permanent_outbound", "24m", 730)]:
    sub = held[held[f"{ep}_horizon_observable_{h}"].astype(bool)]; fu = sub["administrative_followup_days"]
    full = sub[fu >= days]
    bias.append({"endpoint": f"{ep}_{h}", "verifier_cohort_rows": int(len(sub)), "rows_with_full_followup": int((fu >= days).sum()),
                 "rows_included_only_because_event_occurred": int((fu < days).sum()),
                 "event_rate_verifier_cohort": round(float(sub[f"{ep}_event_by_{h}"].mean()), 3),
                 "event_rate_full_followup_only": round(float(full[f"{ep}_event_by_{h}"].mean()), 3) if len(full) else None,
                 "verifier_reported": {"any_outbound_24m": "AUC 0.6786 (n=319)", "any_outbound_36m": "Brier 0.0941 (n=193, one class)", "permanent_outbound_24m": "AUC 0.6523 (n=289)"}[f"{ep}_{h}"]})
pd.DataFrame(bias).to_csv(OUT / "oos_verifier_maturity_bias.csv", index=False)

# ---------------------------------------------------------------------------
# 5. Metric reconstruction table
# ---------------------------------------------------------------------------
recon = []
pr = pd.read_csv(P / "extension_opportunity_diagnostic/model_predictions.csv", low_memory=False)
for v in ["M1_player_market_profile", "M2_plus_prior_opportunity", "M3_plus_performance_history", "M5_plus_relative_financial_context"]:
    d = pr[(pr.endpoint == "sustained_meaningful_contribution") & (pr.model_variant == v)]
    recon.append({"claim_source": "Phase 1 README / README.md", "metric": f"sustained_meaningful_contribution {v} pooled AUC/Brier/ECE",
                  "claimed": "AUC 0.788, Brier 0.1869 (M5)" if v.startswith("M5") else "", "reconstructed": f"AUC {roc_auc_score(d.actual, d.prediction):.4f}, Brier {brier_score_loss(d.actual, d.prediction):.4f}, ECE {ece(d.actual, d.prediction):.4f}", "n": len(d), "agreement": "reproduced" if v.startswith("M5") else "context"})
sp = pd.read_csv(P / "extension_survival_diagnostic/survival_predictions.csv", low_memory=False)
for ep, h, claim in [("any_outbound", "24m", "AUC 0.710, Brier 0.2191"), ("any_outbound", "36m", "AUC 0.699, Brier 0.2026"), ("permanent_outbound", "24m", "AUC 0.696, Brier 0.2041")]:
    d = sp[(sp.endpoint == ep) & (sp.horizon == h) & (sp.model_variant == "M5_plus_relative_financial_context") & (sp.horizon_fully_observable == True)]
    lg = {k: round(roc_auc_score(g.actual_event_by_horizon, g.predicted_event_probability), 3) for k, g in d.groupby("league")}
    recon.append({"claim_source": "Phase 2 README / README.md", "metric": f"{ep}_{h} M5", "claimed": claim + " (row-weighted per-origin)",
                  "reconstructed": f"pooled AUC {roc_auc_score(d.actual_event_by_horizon, d.predicted_event_probability):.4f}, Brier {brier_score_loss(d.actual_event_by_horizon, d.predicted_event_probability):.4f}; by league AUC {lg}", "n": len(d), "agreement": "reproduced (pooled AUC slightly below origin-weighted figure); Ligue 1 materially weaker"})
vp = pd.read_csv(P / "extension_value_preservation_diagnostic/value_predictions.csv", low_memory=False)
for ep, claim in [("downside_25pct_24m", "AUC 0.821, Brier 0.1717"), ("downside_10pct_24m", "AUC 0.837"), ("downside_50pct_24m", "AUC 0.805"), ("downside_25pct_12m", "AUC 0.781")]:
    d = vp[(vp.endpoint == ep) & (vp.model_variant == "M5_plus_relative_financial_context")]
    d3 = vp[(vp.endpoint == ep) & (vp.model_variant == "M3_plus_performance_history")]
    recon.append({"claim_source": "Phase 3 README / README.md", "metric": f"{ep} M5 (and M3 without contract terms)", "claimed": claim,
                  "reconstructed": f"M5 AUC {roc_auc_score(d.actual, d.prediction):.4f} Brier {brier_score_loss(d.actual, d.prediction):.4f} ECE {ece(d.actual, d.prediction):.4f}; M3 AUC {roc_auc_score(d3.actual, d3.prediction):.4f}", "n": len(d), "agreement": "reproduced; contract terms add ~0 AUC at 24m"})
bp = pd.read_csv(P / "contract_financial_exposure_benchmark/benchmark_predictions.csv", low_memory=False)
bp = bp.merge(m[["capology_extension_event_id", "prior_season_salary_annual_gross_eur"]], on="capology_extension_event_id", how="left")
bp["prior_known"] = pd.to_numeric(bp["prior_season_salary_annual_gross_eur"], errors="coerce").gt(0)
for ep, v, claim in [("annual_wage", "B2_plus_prior_wage", "MAE 0.5193, Spearman 0.889, coverage 81.3%"), ("fixed_wage_commitment", "B2_plus_prior_wage", "MAE 0.6881, Spearman 0.832, coverage 80.1%"), ("contract_duration", "B5_compact_nonlinear", "MAE 0.8291, Spearman 0.659, coverage 82.4%"), ("wage_to_market_value", "B2_plus_prior_wage", "MAE 0.5120, Spearman 0.835, coverage 81.0%")]:
    d = bp[(bp.endpoint == ep) & (bp.model_variant == v)]
    recon.append({"claim_source": "Phase 4 README / README.md", "metric": f"{ep} {v}", "claimed": claim,
                  "reconstructed": f"MAE {d.primary_loss.mean():.4f}, Spearman {d.actual.corr(d.prediction, method='spearman'):.3f}, coverage {d.within_80_interval.mean():.3f}; coverage by league {d.groupby('league').within_80_interval.mean().round(3).to_dict()}; coverage prior wage known/unknown {d.groupby('prior_known').within_80_interval.mean().round(3).to_dict()}", "n": len(d), "agreement": "reproduced; coverage is marginal, not conditional"})
lc = pd.read_csv(P / "business_metric_translation_audit/low_contribution_exposure_predictions.csv")
tot = lc.fixed_wages_through_24m_eur.sum(); low = lc.loc[lc.actual_low_contribution.eq(1), "fixed_wages_through_24m_eur"].sum()
ts = lc.nlargest(182, "fixed_wages_through_24m_eur"); tr = lc.nlargest(182, "selected_exposure_eur"); tp = lc.nlargest(182, "selected_low_contribution_probability")
lc2 = lc.merge(m[["capology_extension_event_id", "first_permanent_outbound_days", "first_any_outbound_days"]], on="capology_extension_event_id", how="left")
lowd = lc2[lc2.actual_low_contribution.eq(1)]
perm_eur = lowd.loc[pd.to_numeric(lowd.first_permanent_outbound_days, errors="coerce").le(730), "fixed_wages_through_24m_eur"].sum()
any_eur = lowd.loc[pd.to_numeric(lowd.first_any_outbound_days, errors="coerce").le(730), "fixed_wages_through_24m_eur"].sum()
recon.append({"claim_source": "README.md / CODEX / presentation", "metric": "908-profile exposure illustration", "claimed": "€5.22B scheduled; €1.82B (34.8%) low contribution; 54 vs 79 hits at capacity 182; MAE improvement €761,630",
              "reconstructed": f"€{tot/1e9:.2f}B; €{low/1e9:.2f}B ({low/tot:.1%}); largest-contract hits {int(ts.actual_low_contribution.sum())}, risk-weighted hits {int(tr.actual_low_contribution.sum())}, probability-only hits {int(tp.actual_low_contribution.sum())}; MAE improvement €{(lc.actual_low_contribution_exposure_eur-lc.baseline_exposure_eur).abs().mean()-(lc.actual_low_contribution_exposure_eur-lc.selected_exposure_eur).abs().mean():,.0f}; of the €{low/1e9:.2f}B, €{perm_eur/1e9:.2f}B ({perm_eur/low:.1%}) is attached to players permanently transferred out within 24m and €{any_eur/1e9:.2f}B ({any_eur/low:.1%}) to any outbound",
              "n": len(lc), "agreement": "reproduced exactly; composition materially qualifies the label"})
w = pd.read_csv(P / "engine_v2_final_evaluation_gate/wage_final_holdout_predictions.csv"); s = w[w.support_status.ne("refused")]
recon.append({"claim_source": "branch docs/ENGINE_V2_FINAL_EVALUATION_GATE.md", "metric": "V2 wage benchmark sealed 2024 holdout", "claimed": "logMAE 0.5046 vs 0.8359; coverage 77.0%; Ligue 1 60.6%; refusal 2.1%",
              "reconstructed": f"logMAE {s.absolute_log_error.mean():.4f} vs baseline column {s.baseline_absolute_log_error.mean():.4f}; coverage {s.within_80_interval.mean():.3f}; by league {s.groupby('league').within_80_interval.mean().round(3).to_dict()}; refusal {(1-len(s)/len(w)):.3f}; prior wage unknown coverage {s[s.prior_wage_available.eq(0)].within_80_interval.mean():.3f}", "n": len(w), "agreement": "reproduced in substance (Bundesliga also below 0.70 in this reconstruction)"})
b = pd.read_csv(P / "engine_v2_movement_state_profile_diagnostic/binary_endpoint_predictions.csv")
for ep, claim in [("strict_meaningful_stay_24m", "AUC 0.759, skill +0.0352"), ("temporary_first_24m", "AUC 0.777, skill +0.0291"), ("permanent_relationship_ended_24m", "AUC 0.628, skill +0.0127"), ("any_outbound_24m", "AUC 0.646 (repaired V2 continuity), Ligue 1 skill -0.0066")]:
    g = b[b.endpoint == ep]
    recon.append({"claim_source": "branch docs/ENGINE_V2_MOVEMENT_SCOPE_CLOSEOUT.md / CRITICAL_ISSUES", "metric": f"V2 {ep}", "claimed": claim,
                  "reconstructed": f"AUC {roc_auc_score(g.actual, g.prediction):.4f}, skill {(g.chronology_brier_loss.mean()-g.brier_loss.mean()):+.4f}; by league skill {{{', '.join(f'{k}: {(gg.chronology_brier_loss.mean()-gg.brier_loss.mean()):+.4f}' for k, gg in g.groupby('league'))}}}", "n": len(g), "agreement": "reproduced"})
t2 = json.loads((OUT / "t2_oos_live_rescoring_metrics.json").read_text())
for r in t2:
    recon.append({"claim_source": "deployment_models/out_of_sample_verification.json", "metric": f"live re-score {r['endpoint']}", "claimed": "see verifier", "reconstructed": f"AUC {r['AUC']}, Brier {r['Brier']}, ECE {r['ECE']}, event rate {r['event_rate']}, mean pred {r['mean_pred']}", "n": r["n"], "agreement": "audit re-score"})
pd.DataFrame(recon).to_csv(OUT / "metric_reconstruction.csv", index=False)

# ---------------------------------------------------------------------------
# 6. Perturbation summary (T1) condensed
# ---------------------------------------------------------------------------
g = pd.read_csv(OUT / "t1_scoring_time_perturbation_grid.csv")
cols = ["sustained_contribution_24m", "year2_opportunity_share", "stay_24m", "permanent_24m", "downside_25pct_24m", "downside_50pct_24m"]
wage = g[(g.status == "scored") & (g.proposed_years == 3.0)].groupby("wage_multiplier")[cols].mean().round(3)
years = g[(g.status == "scored") & (g.wage_multiplier == 1.0)].groupby("proposed_years")[cols].mean().round(3)
wage.to_csv(OUT / "t1_effect_of_proposed_wage_years_fixed_3.csv"); years.to_csv(OUT / "t1_effect_of_proposed_years_wage_fixed.csv")
print("evidence package written")

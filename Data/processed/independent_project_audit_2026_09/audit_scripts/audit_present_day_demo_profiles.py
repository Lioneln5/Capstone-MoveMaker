"""Present-day (decision_date = today) profiles for the three demo players
through the frozen V1 application path (IncumbentExtensionProfileBuilder),
exactly as main's api/main.py would serve them. Read-only; writes only under
the audit directory. Records warnings, unavailable inputs, card statuses,
data-density flags and the explainability top features so the audit can
describe what a user of main's interface actually sees today.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "models"))
OUT = ROOT / "Data" / "processed" / "independent_project_audit_2026_09"

from incumbent_extension_profile import IncumbentExtensionProfileBuilder  # noqa: E402
from player_search import PlayerSearchIndex  # noqa: E402

builder = IncumbentExtensionProfileBuilder()
builder.scorer.retriever.warm_up()
index = PlayerSearchIndex()
index.warm_up()
today = date.today().isoformat()
results = {}
for query in ["bellingham", "haaland", "mbapp"]:
    hits = index.search(query, limit=5)
    hit = next((h for h in hits if h.current_club_id is not None), None)
    if hit is None:
        results[query] = {"search": [h.as_dict() for h in hits], "note": "no resolved club"}
        continue
    defaults = builder.scorer.retriever.get_player_input_defaults(
        canonical_player_id=hit.canonical_player_id, canonical_club_id=hit.current_club_id,
        decision_date=today, player_name_normalized=hit.player_name_normalized,
    )
    wage = defaults.get("latest_known_same_club_annual_wage_eur") or 10_000_000.0
    profile = builder.build_profile(
        canonical_player_id=hit.canonical_player_id, canonical_club_id=hit.current_club_id,
        competition_id=hit.competition_id, decision_date=today,
        proposed_annual_fixed_wage_eur=float(wage), proposed_contract_years=3.0,
        current_public_market_value_eur=defaults.get("current_public_market_value_eur"),
        player_display_name=hit.display_name, player_name_normalized=hit.player_name_normalized,
        current_club_display_name=hit.current_club_name,
    )
    summary = {
        "search_hit": hit.as_dict(), "input_defaults": defaults, "proposed_wage_used": wage, "proposed_years": 3.0,
        "status": profile.get("status"), "model_vintage": profile.get("model_vintage"),
        "incumbency_verified": profile.get("incumbency_verified"),
        "data_density_warning": profile.get("data_density_warning"),
        "market_value_training_percentile": profile.get("market_value_training_percentile"),
        "upstream_feature_warnings": profile.get("upstream_feature_warnings"),
        "upstream_feature_unavailable": profile.get("upstream_feature_unavailable"),
        "availability": profile.get("availability"),
        "cards": {
            card: [{"metric": m["metric_key"], "status": m["status"], "value": m["value"]} for m in metrics]
            for card, metrics in (profile.get("cards") or {}).items()
        },
        "explainability": {
            k: {"status": v.get("status"), "probability": v.get("probability"), "ordering_correction_applied": v.get("ordering_correction_applied"),
                "top": [(c["label"], c["logit_contribution"], c["favorable"]) for c in v.get("contributions", [])[:6]]}
            for k, v in (profile.get("explainability") or {}).items()
        },
    }
    results[query] = summary
    print(query, "->", hit.display_name, hit.current_club_name, hit.competition_id, "| status", profile.get("status"), flush=True)

# out-of-scope and misuse probes through the same builder
probe = {}
b = index.search("bellingham", limit=3)[0]
try:
    probe["arbitrary_club"] = builder.build_profile(
        canonical_player_id=b.canonical_player_id, canonical_club_id=985, competition_id="GB1", decision_date=today,
        proposed_annual_fixed_wage_eur=20_000_000.0, proposed_contract_years=4.0, current_public_market_value_eur=150_000_000.0,
        player_name_normalized=b.player_name_normalized,
    )
    probe["arbitrary_club"] = {k: probe["arbitrary_club"].get(k) for k in ["status", "incumbency_verified", "refusal_reason", "upstream_feature_warnings"]}
except Exception as exc:  # noqa: BLE001
    probe["arbitrary_club"] = {"exception": repr(exc)}
try:
    r = builder.build_profile(
        canonical_player_id=b.canonical_player_id, canonical_club_id=b.current_club_id, competition_id="POR1", decision_date=today,
        proposed_annual_fixed_wage_eur=20_000_000.0, proposed_contract_years=4.0, current_public_market_value_eur=150_000_000.0,
    )
    probe["out_of_scope_league"] = {k: r.get(k) for k in ["status", "refusal_reason"]}
except Exception as exc:  # noqa: BLE001
    probe["out_of_scope_league"] = {"exception": repr(exc)}
results["_probes"] = probe
(OUT / "t3_present_day_demo_profiles.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
print("written", flush=True)

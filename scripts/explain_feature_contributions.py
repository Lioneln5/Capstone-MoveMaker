"""CLI: exact per-feature contribution breakdown for the two headline gauges,
for one player, via the command line.

The actual decomposition logic lives in
models/feature_contribution_explainer.py, shared with the live API's
"why this score" panel (models/deployment_scorer.py's
explain_headline_scores(), surfaced in Data/app-html/index.html) -- this
script is just a thin, human-readable presentation of the same numbers, so
there is exactly one place the math can drift.

Usage:
    python3 scripts/explain_feature_contributions.py [player query] [wage] [years]
    python3 scripts/explain_feature_contributions.py bellingham 15000000 3
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

import joblib  # noqa: E402

from canonical_feature_retriever import CanonicalFeatureRetriever  # noqa: E402
from feature_contribution_explainer import explain_hazard, explain_single_shot  # noqa: E402
from player_search import PlayerSearchIndex  # noqa: E402

DEPLOYMENT_OUTPUT = ROOT / "Data" / "processed" / "deployment_models"


def load_artifact(phase: str, endpoint: str, horizon: str | None) -> object:
    name = endpoint if (horizon is None or str(horizon) in endpoint) else f"{endpoint}_{horizon}"
    path = DEPLOYMENT_OUTPUT / phase / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(path)
    return joblib.load(path)


def print_breakdown(title: str, subtitle: str, result: dict) -> None:
    print("=" * 78)
    print(title)
    print(subtitle)
    print("=" * 78)
    print(f"intercept (avg. training-set player, logit units): {result['intercept_logit']:+.4f}")
    if "player_offset_logit" in result:
        print(f"this player's constant monthly offset from that baseline: {result['player_offset_logit']:+.4f}")
        lo, hi = result["month_shape_logit_range"]
        print(f"(model's own month-shape term ranges {lo:+.3f} to {hi:+.3f} logits across the horizon -- not player-specific)")
    print(f"probability shown on the gauge: {result['probability']:.4f}")
    if result.get("ordering_correction_applied"):
        print(f"[note] {result['correction_note']}")
    print()
    print("Top features by |contribution to the logit| (favorable=True means it helps this gauge's good outcome):")
    for c in result["contributions"]:
        arrow = "favorable" if c["favorable"] else "unfavorable"
        print(f"  {c['logit_contribution']:+.4f}  ({arrow:11s})  {c['label']}")
    print()


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "bellingham"
    proposed_wage = float(sys.argv[2]) if len(sys.argv) > 2 else 15_000_000.0
    proposed_years = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0

    index = PlayerSearchIndex()
    index.warm_up()
    hits = index.search(query, limit=1)
    if not hits:
        raise SystemExit(f"No player matched {query!r}")
    hit = hits[0]
    print(f"Player: {hit.display_name}  (club_id={hit.current_club_id}, competition_id={hit.competition_id})\n")

    retriever = CanonicalFeatureRetriever()
    retriever.warm_up()
    decision_date = pd.Timestamp.today().normalize()
    result = retriever.get_extension_features(
        canonical_player_id=hit.canonical_player_id,
        canonical_club_id=hit.current_club_id,
        competition_id=hit.competition_id,
        decision_date=decision_date,
        proposed_annual_fixed_wage_eur=proposed_wage,
        proposed_contract_years=proposed_years,
        player_name_normalized=hit.player_name_normalized,
    )
    if not result.in_scope:
        raise SystemExit(f"Out of scope: {result.refusal_reason}")
    for w in result.warnings:
        print(f"[warning] {w}")
    if result.unavailable:
        print(f"[unavailable] {json.dumps(result.unavailable)}")
    print()

    fv = result.features
    downside_artifact = load_artifact("extension_value_preservation_diagnostic", "downside_25pct_24m", "24m")
    downside = explain_single_shot(downside_artifact, fv)
    print_breakdown(
        "PUBLIC-VALUE DOWNSIDE RISK  (Phase 3: downside_25pct_24m)",
        "P(public market value falls >=25% within 24 months) -- higher = riskier",
        downside,
    )

    hazard_artifact = load_artifact("extension_survival_diagnostic", "any_outbound", "24m")
    interval_bounds = json.loads((DEPLOYMENT_OUTPUT / "extension_survival_diagnostic" / "interval_bounds.json").read_text())
    continuity = explain_hazard(hazard_artifact, fv, interval_bounds, horizon_days=730)
    print_breakdown(
        "CONTINUITY / 24-MONTH RETENTION  (Phase 2 hazard: any_outbound, 24m)",
        "1 - P(any departure -- loan or permanent -- within 24 months) -- higher = more retained",
        continuity,
    )


if __name__ == "__main__":
    main()

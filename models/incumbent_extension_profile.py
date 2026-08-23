"""The final connecting layer: canonical inputs -> live model scores ->
typed, guarded business profile.

This module owns nothing of its own -- it is pure wiring between two
independently verified pieces:
  - models/deployment_scorer.py     (canonical features -> Phase 1-4 probabilities/ranges)
  - models/business_metric_engine.py (probabilities/ranges -> six typed, warned business cards)

Per the August-12 product-scope correction, business_metric_engine.py is a
calculation/response layer that was previously only ever exercised with
manually-typed probabilities (the Jude Bellingham example was a formatting
demo, not a real forecast). This module is what turns that into an actual
forecast: it is the first place in the repository where a live-scored
probability reaches the engine, and it is why models/deployment_scorer.py's
parity/refusal verification (scripts/verify_deployment_models.py) had to
pass before this file was written.

The frozen offer-aggressiveness reference distribution (1,504 out-of-time
peer log-commitment residuals, serialized during the August-12 product
translation audit) is loaded from disk, not recomputed -- it must stay
exactly the same reference every scoring call uses, matching the audit's
own frozen contract.
"""

from __future__ import annotations

import bisect
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from business_metric_engine import SCOPE, build_extension_business_profile  # noqa: E402
from deployment_scorer import DeploymentScorer  # noqa: E402
import run_contract_financial_exposure_benchmark as phase4  # noqa: E402

OFFER_REFERENCE_PATH = ROOT / "Data" / "processed" / "business_metric_product" / "offer_aggressiveness_reference.json"

# A player whose market value sits above this percentile of the Phase-4 peer
# model's own training data gets an explicit low-confidence flag. This is
# not a tunable "make the warning appear less often" knob -- it marks the
# point past which the peer benchmark is extrapolating on genuinely thin
# data (see the Jude Bellingham case: >=95th percentile means fewer than
# ~130 of ~2,600 training extensions are anywhere near this comparable).
DATA_DENSITY_WARNING_PERCENTILE = 0.95


def _load_market_value_training_reference() -> list[float]:
    """Sorted at_signing_market_value_eur from the same cohort Phase 4's
    fixed-wage-commitment peer model was trained on (benchmark_commitment_
    cohort) -- reused directly, not a separately-chosen sample, so this
    reflects exactly the data density the deployed peer model actually has."""
    frame = phase4.load_frame()
    values = pd.to_numeric(frame.loc[frame["benchmark_commitment_cohort"], "at_signing_market_value_eur"], errors="coerce").dropna()
    return sorted(values.tolist())


def _training_data_percentile(value: float, sorted_reference: list[float]) -> float:
    return bisect.bisect_right(sorted_reference, value) / len(sorted_reference)


def _load_offer_reference() -> list[float]:
    payload = json.loads(OFFER_REFERENCE_PATH.read_text(encoding="utf-8"))
    residuals = payload["sorted_log_residuals"]
    if payload.get("reference_rows") != len(residuals):
        raise RuntimeError(
            f"{OFFER_REFERENCE_PATH} declares reference_rows={payload.get('reference_rows')} "
            f"but contains {len(residuals)} residuals; frozen reference is corrupted."
        )
    return residuals


def _module_value(card: dict[str, Any]) -> float | None:
    """A scored module dict has a numeric 'value' only when its status
    starts with 'supported'; 'unavailable' modules never do. Never coerce
    the latter to zero -- pass None straight through, exactly as
    business_metric_engine.py's own _optional_number expects."""
    status = card.get("status", "")
    return card.get("value") if status.startswith("supported") else None


class IncumbentExtensionProfileBuilder:
    """Construct once per process (loads and caches the frozen offer
    reference and the deployment scorer's artifacts); call build_profile
    per request."""

    def __init__(self) -> None:
        self.scorer = DeploymentScorer()
        self.sorted_offer_reference_residuals = _load_offer_reference()
        self.sorted_market_value_training_reference = _load_market_value_training_reference()

    def build_profile(
        self,
        canonical_player_id: int,
        canonical_club_id: int,
        competition_id: str,
        decision_date: str,
        proposed_annual_fixed_wage_eur: float,
        proposed_contract_years: float,
        current_public_market_value_eur: float | None = None,
        player_name_normalized: str | None = None,
        proposed_transfer_fee_eur: float | None = None,
        player_display_name: str | None = None,
        current_club_display_name: str | None = None,
    ) -> dict[str, Any]:
        scored = self.scorer.score_extension(
            canonical_player_id=canonical_player_id, canonical_club_id=canonical_club_id,
            competition_id=competition_id, decision_date=decision_date,
            proposed_annual_fixed_wage_eur=proposed_annual_fixed_wage_eur,
            proposed_contract_years=proposed_contract_years,
            current_public_market_value_eur=current_public_market_value_eur,
            player_name_normalized=player_name_normalized,
        )

        if scored["status"] == "refused":
            return {
                "profile_scope": SCOPE,
                "status": "refused",
                "refusal_reason": scored["refusal_reason"],
                "cards": None,
                "overall_contract_score": None,
                "overall_contract_score_status": "rejected_show_separate_business_metrics",
            }

        opportunity = scored["modules"]["opportunity"]
        continuity = scored["modules"]["continuity"]
        asset_value = scored["modules"]["asset_value"]
        peer = scored["modules"]["peer_context"]
        context = scored["context"]

        commitment_peer = peer["fixed_wage_commitment_peer"]
        commitment_range = commitment_peer.get("reference_range") if commitment_peer.get("status", "").startswith("supported") else None
        duration_peer = peer["contract_duration_peer"]
        duration_range = duration_peer.get("reference_range") if duration_peer.get("status", "").startswith("supported") else None
        wage_value_peer = peer["wage_to_market_value_peer"]
        wage_value_range = wage_value_peer.get("reference_range") if wage_value_peer.get("status", "").startswith("supported") else None

        # current_public_market_value_eur: prefer the caller's own live
        # figure (already round-tripped through the scorer's context if they
        # supplied one); fall back to the retriever's canonical valuation
        # lookup only when the caller didn't supply one at all.
        market_value = current_public_market_value_eur if current_public_market_value_eur is not None else context.get("current_public_market_value_eur")
        if market_value is None:
            return {
                "profile_scope": SCOPE,
                "status": "refused",
                "refusal_reason": "current_public_market_value_eur is required and no canonical valuation was found on or before decision_date",
                "cards": None,
                "overall_contract_score": None,
                "overall_contract_score_status": "rejected_show_separate_business_metrics",
            }
        if context.get("age_at_decision") is None:
            return {
                "profile_scope": SCOPE,
                "status": "refused",
                "refusal_reason": "age_at_decision is unavailable (missing date of birth); business_metric_engine.py requires it",
                "cards": None,
                "overall_contract_score": None,
                "overall_contract_score_status": "rejected_show_separate_business_metrics",
            }

        # Data-density check: is this player's market value in a region
        # the peer models actually have enough training support for, or is
        # the peer estimate (and, to a lesser extent, every other model that
        # also uses log_market_value_at_signing as a feature) extrapolating?
        market_value_training_percentile = _training_data_percentile(market_value, self.sorted_market_value_training_reference)
        data_density_warning: str | None = None
        if market_value_training_percentile >= DATA_DENSITY_WARNING_PERCENTILE:
            comparable_count = len(self.sorted_market_value_training_reference) - bisect.bisect_left(self.sorted_market_value_training_reference, market_value)
            data_density_warning = (
                f"This player's market value (EUR {market_value:,.0f}) is at the {market_value_training_percentile:.1%} percentile of the "
                f"peer-benchmark training data (only {comparable_count} of {len(self.sorted_market_value_training_reference)} historical "
                "extensions had a comparable-or-higher market value). Every model in this profile is extrapolating in this range, not "
                "interpolating -- treat all estimates, and the peer-commitment benchmark especially, as directional only."
            )

        payload: dict[str, Any] = {
            "profile_scope": SCOPE,
            "player": player_display_name,
            "current_club": current_club_display_name,
            "decision_date": scored["decision_date"],
            "age_at_decision": context["age_at_decision"],
            "proposed_annual_fixed_wage_eur": proposed_annual_fixed_wage_eur,
            "proposed_contract_years": proposed_contract_years,
            "proposed_transfer_fee": proposed_transfer_fee_eur,
            "current_public_market_value_eur": market_value,
            "prior_annual_fixed_wage_eur": context.get("prior_annual_fixed_wage_eur"),
            "peer_fixed_commitment_estimate_eur": commitment_range["point_multiplier"] if commitment_range else None,
            "peer_fixed_commitment_lower_eur": commitment_range["lower_multiplier"] if commitment_range else None,
            "peer_fixed_commitment_upper_eur": commitment_range["upper_multiplier"] if commitment_range else None,
            "sustained_contribution_probability_24m": _module_value(opportunity["sustained_contribution_probability_24m"]),
            "continuous_stay_probability_24m": _module_value(continuity["continuous_stay_probability_24m"]),
            "continuous_stay_probability_36m": _module_value(continuity["continuous_stay_probability_36m"]),
            "permanent_relationship_probability_24m": _module_value(continuity["permanent_relationship_probability_24m"]),
            "public_value_downside_10pct_probability_12m": _module_value(asset_value["value_downside_10pct_probability_12m"]),
            "public_value_downside_10pct_probability_24m": _module_value(asset_value["value_downside_10pct_probability_24m"]),
            "public_value_downside_25pct_probability_12m": _module_value(asset_value["value_downside_25pct_probability_12m"]),
            "public_value_downside_25pct_probability_24m": _module_value(asset_value["value_downside_25pct_probability_24m"]),
            "public_value_downside_50pct_probability_24m": _module_value(asset_value["value_downside_50pct_probability_24m"]),
            "club_salary_percentile": context.get("club_salary_percentile"),
            "club_known_payroll_share": context.get("club_known_payroll_share"),
            # Duration peer is in "years" units directly -- no exp() transform.
            "peer_contract_duration_estimate_years": _module_value(duration_peer),
            "peer_contract_duration_lower_years": duration_range["lower"] if duration_range else None,
            "peer_contract_duration_upper_years": duration_range["upper"] if duration_range else None,
            # Wage-to-value peer is fit in log-ratio space; the *_multiplier
            # fields are already the exp()-transformed real ratio.
            "peer_wage_to_market_value_estimate": wage_value_range["point_multiplier"] if wage_value_range else None,
            "peer_wage_to_market_value_lower": wage_value_range["lower_multiplier"] if wage_value_range else None,
            "peer_wage_to_market_value_upper": wage_value_range["upper_multiplier"] if wage_value_range else None,
        }

        profile = build_extension_business_profile(payload, self.sorted_offer_reference_residuals)
        profile["status"] = "scored"
        profile["incumbency_verified"] = scored["incumbency_verified"]
        profile["model_vintage"] = scored["model_vintage"]
        profile["data_density_warning"] = data_density_warning
        profile["market_value_training_percentile"] = round(market_value_training_percentile, 4)
        # Flat, easy-to-consume echo of what was actually used to score this
        # profile -- saves every caller (this session's frontend included)
        # from having to dig a specific value back out of nested card
        # components just to know, e.g., what market value was auto-filled.
        profile["resolved_inputs"] = {
            "current_public_market_value_eur": market_value,
            "age_at_decision": context["age_at_decision"],
            "prior_annual_fixed_wage_eur": context.get("prior_annual_fixed_wage_eur"),
        }
        if data_density_warning:
            for metric in profile["cards"].get("offer_context", []):
                if metric["metric_key"] in ("fixed_commitment_peer_difference_eur", "offer_aggressiveness_percentile") and metric["status"] != "unavailable":
                    metric["warning"] = metric["warning"] + " " + data_density_warning
        # Preserve upstream model-layer warnings (imputed inputs, stale
        # valuation, unmatched salary panel, ordering violations) alongside
        # the engine's own per-metric warnings -- neither layer's warnings
        # substitute for the other's.
        profile["upstream_feature_warnings"] = scored["feature_warnings"]
        profile["upstream_feature_unavailable"] = scored["feature_unavailable"]
        # Exact per-feature logit-contribution breakdown for the two
        # headline gauges ("why this score" panel) -- see
        # models/feature_contribution_explainer.py. Already computed inside
        # score_extension() using the same (possibly PAVA-corrected)
        # displayed values as the cards above, so it can never silently
        # disagree with what's on screen.
        profile["explainability"] = scored["explainability"]
        return profile


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test the full canonical-features -> scorer -> business-profile pipeline.")
    parser.add_argument("--player-id", type=int, required=True)
    parser.add_argument("--club-id", type=int, required=True)
    parser.add_argument("--competition-id", type=str, required=True)
    parser.add_argument("--decision-date", type=str, required=True)
    parser.add_argument("--wage", type=float, required=True)
    parser.add_argument("--years", type=float, required=True)
    parser.add_argument("--market-value", type=float, default=None)
    parser.add_argument("--player-name", type=str, default=None, help="player_name_normalized, for prior-wage/salary-panel matching")
    args = parser.parse_args()

    builder = IncumbentExtensionProfileBuilder()
    result = builder.build_profile(
        canonical_player_id=args.player_id, canonical_club_id=args.club_id,
        competition_id=args.competition_id, decision_date=args.decision_date,
        proposed_annual_fixed_wage_eur=args.wage, proposed_contract_years=args.years,
        current_public_market_value_eur=args.market_value, player_name_normalized=args.player_name,
    )
    print(json.dumps(result, indent=2, default=str))

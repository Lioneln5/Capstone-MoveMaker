"""Live scorer: canonical features -> frozen Phase 1-4 artifacts -> typed output.

This is the piece that finally connects the two things built earlier:
  - models/canonical_feature_retriever.py (arbitrary player/club/date -> features)
  - scripts/build_deployment_models.py's serialized artifacts (fitted, frozen models)

It does not fit anything and does not compute business-metric arithmetic --
models/business_metric_engine.py still owns fixed-wage commitment, peer
euro-difference, offer-aggressiveness percentile, and every downstream
formatting/warning rule. This module's only job is turning a proposed
incumbent-club extension into the four modeled probability/range inputs
that engine expects (sustained_contribution_probability_24m,
continuous_stay_probability_24m/36m, public_value_downside_*_probability_24m,
peer_fixed_commitment_estimate_eur, etc.) -- and to do that scoring exactly
the way each Phase 1-4 diagnostic validated, never inventing a new formula.

Missingness policy (mirrors the project's "never silently substitute zero"
rule at the layer where it actually applies):
  - A handful of CORE features (age, position, league, market value) define
    who is being scored. If any of these are unavailable, every model that
    needs them is marked unavailable outright -- we do not let the
    preprocessor's trained median silently stand in for "this specific
    player's age."
  - Secondary predictors (pre1/pre2 season stats, advanced FBref features,
    club salary context) were deliberately trained with missing-indicator
    columns so the model could use "this wasn't observed" as a signal --
    that is a validated, evaluated part of Phase 1-4, not a shortcut. Those
    are passed through as NaN (the trained preprocessor imputes them the
    same way it did during training) and listed in `imputed_features` on
    the result so the caller can decide how much to trust a given score.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from canonical_feature_retriever import CanonicalFeatureRetriever, FeatureResult  # noqa: E402
from feature_contribution_explainer import explain_hazard, explain_single_shot  # noqa: E402
from run_integrated_contract_profile import decreasing_projection  # noqa: E402

DEPLOYMENT_OUTPUT = ROOT / "Data" / "processed" / "deployment_models"
MANIFEST_PATH = DEPLOYMENT_OUTPUT / "model_manifest.json"

CORE_FEATURES = {"age_at_signing", "canonical_position", "league", "log_market_value_at_signing"}


class DeploymentScorer:
    """Construct once per process; loads the manifest lazily and caches
    joblib artifacts and the feature retriever's own lazily-loaded tables."""

    def __init__(self) -> None:
        if not MANIFEST_PATH.exists():
            raise FileNotFoundError(f"{MANIFEST_PATH} not found; run scripts/build_deployment_models.py first.")
        self.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self._artifact_cache: dict[tuple[str, str, str | None], Any] = {}
        self.retriever = CanonicalFeatureRetriever()
        interval_bounds_path = DEPLOYMENT_OUTPUT / "extension_survival_diagnostic" / "interval_bounds.json"
        self.interval_bounds: list[int] = json.loads(interval_bounds_path.read_text(encoding="utf-8"))

    def _artifact_path(self, phase: str, endpoint: str, horizon: str | None) -> Path:
        name = endpoint if (horizon is None or str(horizon) in endpoint) else f"{endpoint}_{horizon}"
        return DEPLOYMENT_OUTPUT / phase / f"{name}.joblib"

    def _load(self, phase: str, endpoint: str, horizon: str | None = None):
        key = (phase, endpoint, horizon)
        if key not in self._artifact_cache:
            path = self._artifact_path(phase, endpoint, horizon)
            if not path.exists():
                raise FileNotFoundError(f"No serialized artifact at {path}; check model_manifest.json for what was built.")
            self._artifact_cache[key] = joblib.load(path)
        return self._artifact_cache[key]

    # -- generic single-row scoring for the non-hazard model families --------
    def _feature_frame(self, artifact, feature_values: dict[str, Any]) -> tuple[pd.DataFrame, list[str], bool]:
        row: dict[str, Any] = {}
        imputed: list[str] = []
        core_missing = False
        for feat in artifact.features:
            value = feature_values.get(feat)
            if value is None:
                row[feat] = np.nan
                imputed.append(feat)
                if feat in CORE_FEATURES:
                    core_missing = True
            else:
                row[feat] = value
        return pd.DataFrame([row]), imputed, core_missing

    def _score_ridge_or_logistic(self, phase: str, endpoint: str, feature_values: dict[str, Any], horizon: str | None = None) -> dict[str, Any]:
        artifact = self._load(phase, endpoint, horizon)
        frame, imputed, core_missing = self._feature_frame(artifact, feature_values)
        if core_missing:
            return {"status": "unavailable", "reason": "one or more core features (age/position/league/market value) unavailable", "imputed_features": imputed}
        x = artifact.preprocessor.transform(frame)
        if artifact.kind == "binary":
            value = float(np.clip(artifact.model.predict_proba(x)[0, 1], 1e-6, 1 - 1e-6))
        else:
            raw = float(artifact.model.predict(x)[0])
            value = float(np.clip(raw, 0, 1.05)) if artifact.unit == "share_0_1" else raw
        result: dict[str, Any] = {
            "status": "supported_candidate" if imputed == [] else "supported_candidate_with_imputed_inputs",
            "value": value, "unit": artifact.unit, "imputed_features": imputed,
        }
        if artifact.interval_abs_residual_q80 is not None:
            width = artifact.interval_abs_residual_q80
            result["reference_range"] = self._unit_range(value, width, artifact.unit)
        return result

    @staticmethod
    def _unit_range(point: float, half_width: float, unit: str) -> dict[str, float]:
        lower, upper = point - half_width, point + half_width
        if unit == "share_0_1":
            return {"lower": max(0.0, lower), "upper": min(1.05, upper)}
        if unit in ("log_eur", "log_ratio"):
            return {"lower_multiplier": math.exp(lower), "upper_multiplier": math.exp(upper), "point_multiplier": math.exp(point)}
        return {"lower": lower, "upper": upper}

    # -- Phase 2: discrete-time hazard, needs the expanded-interval trick ----
    def _score_hazard(self, endpoint: str, horizon: str, feature_values: dict[str, Any]) -> dict[str, Any]:
        artifact = self._load("extension_survival_diagnostic", endpoint, horizon)
        base_features = [f for f in artifact.features if f != "hazard_interval"]
        row: dict[str, Any] = {}
        imputed: list[str] = []
        core_missing = False
        for feat in base_features:
            value = feature_values.get(feat)
            if value is None:
                row[feat] = np.nan
                imputed.append(feat)
                if feat in CORE_FEATURES:
                    core_missing = True
            else:
                row[feat] = value
        if core_missing:
            return {"status": "unavailable", "reason": "one or more core features unavailable", "imputed_features": imputed}

        horizon_days = {"24m": 730, "36m": 1095}[horizon]
        n_periods = len(self.interval_bounds) - 1
        expanded = pd.DataFrame([{**row, "hazard_interval": f"period_{i}"} for i in range(1, n_periods + 1)])
        hazard = np.clip(artifact.model.predict_proba(artifact.preprocessor.transform(expanded))[:, 1], 1e-6, 1 - 1e-6)
        cumulative_event = 1.0 - np.prod(1.0 - hazard)
        horizon_index = self.interval_bounds[1:].index(horizon_days)
        cumulative_event_at_horizon = 1.0 - np.prod(1.0 - hazard[: horizon_index + 1])
        stay_probability = 1.0 - float(cumulative_event_at_horizon)
        return {
            "status": "supported_candidate" if not imputed else "supported_candidate_with_imputed_inputs",
            "value": stay_probability, "unit": "probability", "imputed_features": imputed,
        }

    # -- "why this score" breakdowns for the predictive headline cards -------
    def explain_headline_scores(
        self, feature_values: dict[str, Any],
        displayed_contribution_24m: float | None,
        displayed_downside_25pct_24m: float | None,
        displayed_continuous_stay_24m: float | None,
    ) -> dict[str, Any]:
        """Exact per-feature logit-contribution breakdown for Sustained
        Contribution, Public-Value Downside Risk, and Continuity
        (24m) cards -- see
        models/feature_contribution_explainer.py for the mechanism. Called
        from score_extension() with the CORRECTED (post-_enforce_ordering)
        displayed values so a rare PAVA correction on downside_25pct_24m is
        surfaced honestly rather than silently producing a breakdown that
        doesn't sum to the number on screen. Returns "unavailable" for a
        gauge whose core features aren't present -- never fabricates a
        breakdown for a score that itself couldn't be computed."""
        result: dict[str, Any] = {}

        contribution_artifact = self._load(
            "extension_opportunity_diagnostic", "sustained_meaningful_contribution"
        )
        if displayed_contribution_24m is None:
            result["sustained_contribution"] = {"status": "unavailable"}
        else:
            breakdown = explain_single_shot(
                contribution_artifact,
                feature_values,
                displayed_probability=displayed_contribution_24m,
                favorable_when_positive=True,
            )
            result["sustained_contribution"] = {"status": "available", **breakdown}

        downside_artifact = self._load("extension_value_preservation_diagnostic", "downside_25pct_24m", "24m")
        if displayed_downside_25pct_24m is None:
            result["public_value_downside_risk"] = {"status": "unavailable"}
        else:
            breakdown = explain_single_shot(downside_artifact, feature_values, displayed_probability=displayed_downside_25pct_24m)
            result["public_value_downside_risk"] = {"status": "available", **breakdown}

        hazard_artifact = self._load("extension_survival_diagnostic", "any_outbound", "24m")
        if displayed_continuous_stay_24m is None:
            result["continuity"] = {"status": "unavailable"}
        else:
            breakdown = explain_hazard(hazard_artifact, feature_values, self.interval_bounds, horizon_days=730)
            result["continuity"] = {"status": "available", **breakdown}
        return result

    # -- public entry point ---------------------------------------------------
    def score_extension(
        self,
        canonical_player_id: int,
        canonical_club_id: int,
        competition_id: str,
        decision_date: str | pd.Timestamp,
        proposed_annual_fixed_wage_eur: float,
        proposed_contract_years: float,
        current_public_market_value_eur: float | None = None,
        player_name_normalized: str | None = None,
    ) -> dict[str, Any]:
        features: FeatureResult = self.retriever.get_extension_features(
            canonical_player_id=canonical_player_id, canonical_club_id=canonical_club_id,
            competition_id=competition_id, decision_date=decision_date,
            proposed_annual_fixed_wage_eur=proposed_annual_fixed_wage_eur,
            proposed_contract_years=proposed_contract_years,
            current_public_market_value_eur=current_public_market_value_eur,
            player_name_normalized=player_name_normalized,
        )
        if not features.in_scope:
            return {
                "profile_scope": "incumbent_club_extension",
                "model_vintage": self.manifest["model_vintage"],
                "decision_date": features.decision_date.date().isoformat() if features.decision_date is not None else None,
                "status": "refused",
                "refusal_reason": features.refusal_reason,
                "modules": None,
            }
        fv = features.features

        opportunity = {
            "year2_opportunity_reference_range": self._score_ridge_or_logistic("extension_opportunity_diagnostic", "year2_opportunity_share", fv),
            "sustained_contribution_probability_24m": self._score_ridge_or_logistic("extension_opportunity_diagnostic", "sustained_meaningful_contribution", fv),
        }
        continuity = {
            "continuous_stay_probability_24m": self._score_hazard("any_outbound", "24m", fv),
            "continuous_stay_probability_36m": self._score_hazard("any_outbound", "36m", fv),
            "permanent_relationship_probability_24m": self._score_hazard("permanent_outbound", "24m", fv),
        }
        asset_value = {
            "value_multiplier_reference_range_12m": self._score_ridge_or_logistic("extension_value_preservation_diagnostic", "value_log_ratio_12m", fv, "12m"),
            "value_multiplier_reference_range_24m": self._score_ridge_or_logistic("extension_value_preservation_diagnostic", "value_log_ratio_24m", fv, "24m"),
            "value_downside_10pct_probability_12m": self._score_ridge_or_logistic("extension_value_preservation_diagnostic", "downside_10pct_12m", fv, "12m"),
            "value_downside_10pct_probability_24m": self._score_ridge_or_logistic("extension_value_preservation_diagnostic", "downside_10pct_24m", fv, "24m"),
            "value_downside_25pct_probability_12m": self._score_ridge_or_logistic("extension_value_preservation_diagnostic", "downside_25pct_12m", fv, "12m"),
            "value_downside_25pct_probability_24m": self._score_ridge_or_logistic("extension_value_preservation_diagnostic", "downside_25pct_24m", fv, "24m"),
            "value_downside_50pct_probability_24m": self._score_ridge_or_logistic("extension_value_preservation_diagnostic", "downside_50pct_24m", fv, "24m"),
        }
        peer_context = {
            "annual_wage_peer": self._score_ridge_or_logistic("contract_financial_exposure_benchmark", "annual_wage", fv),
            "contract_duration_peer": self._score_ridge_or_logistic("contract_financial_exposure_benchmark", "contract_duration", fv),
            "fixed_wage_commitment_peer": self._score_ridge_or_logistic("contract_financial_exposure_benchmark", "fixed_wage_commitment", fv),
            "wage_to_market_value_peer": self._score_ridge_or_logistic("contract_financial_exposure_benchmark", "wage_to_market_value", fv),
        }
        self._enforce_ordering(asset_value, continuity)

        def _value(entry: dict[str, Any]) -> float | None:
            return entry.get("value") if entry.get("status", "").startswith("supported") else None

        explainability = self.explain_headline_scores(
            fv,
            displayed_contribution_24m=_value(opportunity["sustained_contribution_probability_24m"]),
            displayed_downside_25pct_24m=_value(asset_value["value_downside_25pct_probability_24m"]),
            displayed_continuous_stay_24m=_value(continuity["continuous_stay_probability_24m"]),
        )

        # Raw pass-through facts a downstream layer (business_metric_engine.py)
        # needs but that aren't themselves model outputs -- exposed here so a
        # caller never has to re-query the retriever a second time and risk
        # the two queries silently disagreeing.
        prior_wage = None
        if fv.get("prior_wage_available") and "log_prior_season_annual_gross_eur" in fv:
            prior_wage = math.expm1(fv["log_prior_season_annual_gross_eur"])
        context = {
            "age_at_decision": fv.get("age_at_signing"),
            "current_public_market_value_eur": fv.get("at_signing_market_value_eur"),
            "prior_annual_fixed_wage_eur": prior_wage,
            "club_salary_percentile": fv.get("club_salary_percentile"),
            "club_known_payroll_share": fv.get("club_salary_share_known"),
        }

        return {
            "profile_scope": "incumbent_club_extension",
            "model_vintage": self.manifest["model_vintage"],
            "decision_date": features.decision_date.date().isoformat() if features.decision_date is not None else None,
            "status": "scored",
            "incumbency_verified": features.incumbency_verified,
            "feature_unavailable": features.unavailable,
            "feature_warnings": features.warnings,
            "context": context,
            "modules": {
                "opportunity": opportunity, "continuity": continuity,
                "asset_value": asset_value, "peer_context": peer_context,
            },
            "explainability": explainability,
        }

    @staticmethod
    def _enforce_ordering(asset_value: dict[str, Any], continuity: dict[str, Any]) -> None:
        """Apply the same isotonic (PAVA) display-ordering correction Phase 5
        already validated (models/run_integrated_contract_profile.py's
        decreasing_projection) -- the model decision record describes this as removing all raw
        pairwise violations (50/1367 at 12m, 134/1367 at 24m in the Phase-5
        cohort) with mean Brier score changing by only -0.00006 to -0.00020,
        i.e. coherence gained essentially for free. This was originally left
        as a warning-only check, but downstream consumers (business_metric_
        engine.py) correctly refuse to display an internally-contradictory
        pair of probabilities, so an un-applied raw violation is not a
        theoretical concern -- ~5% of a real batch run hit it. The raw,
        pre-correction values are preserved under raw_value on each entry so
        the correction is never hidden, only fixed for display.

        Continuity's 24m/36m pair is NOT run through this: both horizons are
        read off the SAME fitted discrete-time hazard curve at different
        cumulative-product lengths, which makes stay_36m <= stay_24m true by
        construction (multiplying in more (1-hazard) terms, each < 1, cannot
        increase the product) -- there is no raw violation to correct."""
        def value(entry: dict[str, Any]) -> float | None:
            return entry.get("value") if entry.get("status", "").startswith("supported") else None

        def apply_downside_ordering(horizon_label: str, keys_least_to_most_severe: list[str]) -> None:
            """Jointly project a FULL ordered threshold sequence (e.g. [10%,
            25%, 50%] at 24m), matching Phase 5's own per-row methodology
            exactly (decreasing_projection applied to the whole available
            row at once, not pairwise) -- only when every threshold in the
            sequence is available, matching Phase 5's "only project when the
            complete row is present" gate. As of Tier 2 (2026-08-12) this
            replaces an earlier version that only handled the 25%/50% pair;
            deploying the 10% threshold made a proper 2-3 point projection
            necessary, not just a pairwise one."""
            entries = [asset_value[k] for k in keys_least_to_most_severe if k in asset_value]
            if len(entries) < 2:
                return
            values = [value(e) for e in entries]
            if any(v is None for v in values):
                return
            if all(values[i] >= values[i + 1] - 1e-9 for i in range(len(values) - 1)):
                return  # already coherent
            ordered = decreasing_projection(values)
            for entry, raw, corrected in zip(entries, values, ordered):
                entry["raw_value"] = raw
                entry["value"] = corrected
            asset_value[f"threshold_ordering_correction_applied_{horizon_label}"] = (
                f"raw downside probabilities at {horizon_label} violated non-increasing severity order "
                f"({[round(v, 4) for v in values]}); applied Phase-5's validated isotonic projection -> {[round(v, 4) for v in ordered]}."
            )

        apply_downside_ordering("12m", ["value_downside_10pct_probability_12m", "value_downside_25pct_probability_12m"])
        apply_downside_ordering("24m", ["value_downside_10pct_probability_24m", "value_downside_25pct_probability_24m", "value_downside_50pct_probability_24m"])

        s24 = value(continuity["continuous_stay_probability_24m"])
        s36 = value(continuity["continuous_stay_probability_36m"])
        if s24 is not None and s36 is not None and s36 > s24 + 1e-9:
            # Should be unreachable given the hazard-curve construction above;
            # kept as a defensive check that fails loudly rather than silently
            # feeding an incoherent pair into business_metric_engine.py.
            raise RuntimeError(
                f"continuous_stay_probability_36m ({s36:.6f}) exceeded continuous_stay_probability_24m ({s24:.6f}); "
                "this should be impossible given the shared hazard-curve construction -- investigate before trusting this profile."
            )

        # continuous_stay_probability_24m (any_outbound model) vs
        # permanent_relationship_probability_24m (permanent_outbound model)
        # are NOT read off the same hazard curve like 24m/36m above -- they
        # are two INDEPENDENTLY FIT logistic models (different hazard_event
        # targets), so nothing guarantees permanent >= continuous_stay in
        # their raw predictions even though it must hold for the true
        # probabilities (a permanent departure is a subset of any
        # departure). scripts/verify_live_request_integrity.py's Tier-1
        # check found this violated in 23/58 (40%) of a real sample -- not a
        # rare edge case, a routine occurrence -- so this gets the same PAVA
        # correction as the asset-value threshold pair, not a "should be
        # unreachable" assumption.
        # Deliberately a ONE-SIDED clamp on `permanent` only, NOT symmetric
        # PAVA pooling like the asset-value pair above: continuous_stay_24m
        # already carries a hard, hazard-curve-guaranteed relationship with
        # continuous_stay_36m (checked just above). If this correction also
        # nudged continuous_stay_24m downward, it could retroactively break
        # that already-validated 24m>=36m relationship without re-checking
        # it. Adjusting only the newer, less-load-bearing permanent-
        # relationship figure avoids that risk entirely.
        permanent_entry = continuity["permanent_relationship_probability_24m"]
        permanent = value(permanent_entry)
        if s24 is not None and permanent is not None and permanent < s24 - 1e-9:
            permanent_entry["raw_value"] = permanent
            permanent_entry["value"] = s24
            continuity["permanent_relationship_ordering_correction_applied"] = (
                f"raw permanent-relationship probability ({permanent:.4f}) was below the continuous-stay probability ({s24:.4f}), which "
                "is logically impossible (a permanent departure is a subset of any departure) -- clamped up to match continuous-stay "
                "rather than applying two-sided pooling, since continuous_stay_24m carries its own separate, already-validated "
                "relationship with continuous_stay_36m that must not be disturbed here."
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test the deployment scorer against a real historical player.")
    parser.add_argument("--player-id", type=int, required=True)
    parser.add_argument("--club-id", type=int, required=True)
    parser.add_argument("--competition-id", type=str, required=True)
    parser.add_argument("--decision-date", type=str, required=True)
    parser.add_argument("--wage", type=float, required=True)
    parser.add_argument("--years", type=float, required=True)
    parser.add_argument("--market-value", type=float, default=None)
    args = parser.parse_args()

    scorer = DeploymentScorer()
    output = scorer.score_extension(
        canonical_player_id=args.player_id, canonical_club_id=args.club_id,
        competition_id=args.competition_id, decision_date=args.decision_date,
        proposed_annual_fixed_wage_eur=args.wage, proposed_contract_years=args.years,
        current_public_market_value_eur=args.market_value,
    )
    print(json.dumps(output, indent=2, default=str))

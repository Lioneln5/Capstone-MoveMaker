"""Exact per-feature logit-contribution breakdown for the deployed
LogisticRegression/Ridge artifacts.

Shared by:
  - scripts/explain_feature_contributions.py   (CLI / ad-hoc analysis)
  - models/deployment_scorer.py's explain_headline_scores()  (live API,
    "why this score" panel in Data/app-html/index.html)

Both a LogisticRegression and a Ridge model, fit on
mixed_type_preprocessor.Preprocessor's standardized (z-scored + one-hot)
feature matrix, are exactly linear in that space:

    logit_or_value = intercept_ + sum_j( coef_j * standardized_feature_j )

so this is a closed-form decomposition, not a SHAP-style approximation --
it reconstructs the model's own output bit-for-bit (see the docstring in
scripts/explain_feature_contributions.py for the full mechanism writeup,
including the discrete-time hazard model's month-invariant player offset).

Deliberately does NOT cover contract_duration (the one GradientBoosting
artifact in the deployed engine, models/deployment_scorer.py's
peer_context.contract_duration_peer) -- a tree ensemble has no coefficients,
so this exact technique doesn't apply there; see the 2026-08-12 chat record
for why GBM was tried and rejected everywhere else in the engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Plain-language labels for the features that actually show up among the
# top contributors for the two headline gauges (downside_25pct_24m,
# any_outbound hazard). Anything not listed here falls back to a
# title-cased, underscore-stripped version of the raw name -- so a new
# feature added to either model later never renders as a blank or crashes,
# just as a slightly less polished label until someone adds it here.
FEATURE_LABELS: dict[str, str] = {
    "log_market_value_at_signing": "Market value",
    "age_at_signing": "Age (at signing)",
    "age_at_expiration": "Age (at contract end)",
    "annual_wage_to_market_value": "Proposed wage / market value",
    "log_annual_gross_eur": "Current annual wage",
    "log_fixed_wage_commitment_eur": "Total fixed-wage commitment",
    "salary_change_from_prior_pct": "Wage change vs. prior season",
    "canonical_position": "Position",
    "pre365_same_club_all_competition_appearance_rate": "Recent appearance rate (same club)",
    "pre2_canonical_assists_per90": "Assists per 90 (2 seasons ago)",
    "pre2_canonical_goals_per90": "Goals per 90 (2 seasons ago)",
    "pre2_canonical_minutes_played": "Minutes played (2 seasons ago)",
    "pre365_goals_per90": "Goals per 90 (last 365 days)",
    "pre365_assists_per90": "Assists per 90 (last 365 days)",
    # NOT a completeness metric -- canonical_feature_retriever.py's
    # _club_salary_context() defines this as proposed_wage / (known club
    # payroll + proposed_wage), i.e. the player's own wage as a share of
    # the known club wage bill. Mislabeled until a 2026-08-16 audit traced
    # the exact formula back to the retriever and corrected it here.
    "club_salary_share_known": "Wage share of known club payroll",
    "club_salary_percentile": "Position in club's wage structure",
    "exact_duration_years": "Proposed contract length",
    "league": "League",
    "pre1_fbref_expected_goals": "Expected goals, xG (prior season)",
    "pre1_fbref_progressive_carries": "Progressive carries (prior season)",
    "pre1_fbref_progressive_passes": "Progressive passes (prior season)",
    "pre1_fbref_pass_completion_pct": "Pass completion % (prior season)",
}


def label_for(raw_feature: str) -> str:
    return FEATURE_LABELS.get(raw_feature, raw_feature.replace("_", " ").replace("pre365", "recent-365d").strip().capitalize())


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def full_row(fv: dict, features: list[str], extra: dict | None = None) -> dict:
    """Fills every feature the artifact expects, defaulting to NaN for
    anything the retriever marked unavailable -- mirrors
    DeploymentScorer._feature_frame/_score_hazard's `feature_values.get(feat)`
    exactly, so this reconstruction can never silently diverge from what the
    live scorer actually used."""
    row = {f: fv.get(f, np.nan) for f in features}
    if extra:
        row.update(extra)
    return row


def raw_contributions(
    artifact, fv: dict, drop_features: tuple[str, ...] = (), extra: dict | None = None,
) -> tuple[list[tuple[str, float]], float, float]:
    """Returns (list of (raw_feature_name, logit_contribution) descending by
    |contribution|, intercept, total logit)."""
    frame = pd.DataFrame([full_row(fv, artifact.features, extra)])
    x = artifact.preprocessor.transform(frame)[0]
    # LogisticRegression exposes ``coef_`` as (1, n_features) and
    # ``intercept_`` as (1,), while Ridge exposes ``coef_`` as
    # (n_features,) and may expose a scalar ``intercept_``.  Flattening both
    # representations keeps the shared explainer faithful to either linear
    # artifact instead of assuming classifier-only shapes.
    coef = np.ravel(np.asarray(artifact.model.coef_, dtype=float))
    intercept = float(np.ravel(np.asarray(artifact.model.intercept_, dtype=float))[0])
    encoded_raw = artifact.preprocessor.encoded_raw_features

    per_raw: dict[str, float] = {}
    for raw, xi, ci in zip(encoded_raw, x, coef):
        if raw in drop_features:
            continue
        per_raw[raw] = per_raw.get(raw, 0.0) + float(xi * ci)

    logit = intercept + sum(per_raw.values())
    ranked = sorted(per_raw.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return ranked, intercept, logit


def explain_single_shot(
    artifact,
    fv: dict,
    displayed_probability: float | None = None,
    top_n: int = 8,
    *,
    favorable_when_positive: bool = False,
) -> dict:
    """For a single-shot LogisticRegression endpoint (e.g. downside_25pct_24m):
    the raw model probability is sigmoid(logit). `displayed_probability` is
    the CALLER's already-corrected card value (deployment_scorer.py's PAVA
    ordering correction can move this away from the raw model output on a
    minority of profiles) -- when it's supplied and differs from the raw
    reconstruction, that's surfaced explicitly rather than silently
    reporting a breakdown that doesn't sum to the number on screen."""
    ranked, intercept, logit = raw_contributions(artifact, fv)
    raw_probability = float(sigmoid(logit))
    result: dict = {
        "intercept_logit": round(intercept, 4),
        "raw_model_probability": round(raw_probability, 4),
        "contributions": [
            {
                "feature": raw,
                "label": label_for(raw),
                "logit_contribution": round(contrib, 4),
                # Risk endpoints improve when their logit contribution is
                # negative; positive-outcome endpoints (such as sustained
                # contribution) improve when it is positive.  The caller
                # declares which target it is explaining so the UI never
                # silently labels a risk-increasing factor as favourable.
                "favorable": contrib > 0 if favorable_when_positive else contrib < 0,
            }
            for raw, contrib in ranked[:top_n]
        ],
    }
    if displayed_probability is not None:
        result["probability"] = round(float(displayed_probability), 4)
        if abs(displayed_probability - raw_probability) > 1e-4:
            result["ordering_correction_applied"] = True
            result["correction_note"] = (
                f"The raw model estimate here was {raw_probability:.1%}. A small consistency correction (the same "
                "isotonic projection validated in Phase 5) adjusted the displayed value to keep it non-contradictory "
                "with the other loss-severity thresholds also scored for this player."
            )
        else:
            result["ordering_correction_applied"] = False
    else:
        result["probability"] = round(raw_probability, 4)
        result["ordering_correction_applied"] = False
    return result


def explain_hazard(artifact, fv: dict, interval_bounds: list[int], horizon_days: int, top_n: int = 8) -> dict:
    """For the Phase 2 discrete-time hazard endpoint behind the Continuity
    gauge: the displayed probability is 1 - prod(1 - monthly_hazard), never
    PAVA-corrected (see deployment_scorer.py's _enforce_ordering docstring),
    so the reconstruction here always exactly equals the displayed value.
    The player's real features contribute an IDENTICAL logit offset in every
    one of the monthly hazard evaluations (only the hazard_interval period
    dummy changes month to month) -- that constant offset is what's reported
    per-feature; the month-shape term itself is not player-specific."""
    n_periods = len(interval_bounds) - 1
    horizon_index = interval_bounds[1:].index(horizon_days)

    ranked, intercept, _ = raw_contributions(artifact, fv, drop_features=("hazard_interval",), extra={"hazard_interval": "period_1"})
    player_offset = sum(c for _, c in ranked)

    expanded = pd.DataFrame([full_row(fv, artifact.features, {"hazard_interval": f"period_{i}"}) for i in range(1, n_periods + 1)])
    hazard = np.clip(artifact.model.predict_proba(artifact.preprocessor.transform(expanded))[:, 1], 1e-6, 1 - 1e-6)
    stay_probability = float(1.0 - (1.0 - np.prod(1.0 - hazard[: horizon_index + 1])))
    month_shape_logit = np.log(hazard / (1 - hazard)) - intercept - player_offset

    return {
        "intercept_logit": round(intercept, 4),
        "player_offset_logit": round(player_offset, 4),
        "month_shape_logit_range": [round(float(month_shape_logit.min()), 4), round(float(month_shape_logit.max()), 4)],
        "probability": round(stay_probability, 4),
        "ordering_correction_applied": False,
        "contributions": [
            {
                "feature": raw,
                "label": label_for(raw),
                "logit_contribution": round(contrib, 4),
                # Positive contribution here raises monthly departure hazard,
                # i.e. LOWERS retention -- so "favorable" (green, good-for-
                # continuity) is the negative-contribution direction, same
                # sign convention as explain_single_shot's risk gauge.
                "favorable": contrib < 0,
            }
            for raw, contrib in ranked[:top_n]
        ],
    }

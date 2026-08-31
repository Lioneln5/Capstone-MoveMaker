"""Engine V2 Phase 4: repair continuity and freeze the candidate result contract.

This phase remains research-only. It does not serialize a model, change the
frozen V1 API, or claim that an extension should be accepted or rejected.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, r2_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "models") not in sys.path:
    sys.path.insert(0, str(ROOT / "models"))

from models.engine_v2.feature_contract import V2_CORE_FEATURES, V2_HISTORY_FEATURES  # noqa: E402
from models.engine_v2.result_contract import (  # noqa: E402
    ENGINE_VERSION, Interval, ModuleResult, ResultEnvelope, validate_payload,
)
from models.engine_v2.reliability_contract import (  # noqa: E402
    MAX_ACCEPTABLE_ECE, MAX_ACCEPTABLE_REFUSAL_RATE, MAX_SUPPORTED_CALIBRATION_GAP,
    MIN_SUBGROUP_EVENTS, MIN_SUBGROUP_NONEVENTS, MIN_SUBGROUP_ORIGINS,
    MIN_SUBGROUP_ROWS, MIN_SUPPORTED_AUC,
)
from mixed_type_preprocessor import fit_preprocessor  # noqa: E402
import run_contract_financial_exposure_benchmark as wage_source  # noqa: E402
import run_engine_v2_calibration_reliability as phase3  # noqa: E402
import run_engine_v2_feature_specification as phase2  # noqa: E402


OUTPUT = ROOT / "Data" / "processed" / "engine_v2_candidate_contract"
PHASE3 = ROOT / "Data" / "processed" / "engine_v2_calibration_reliability"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"
RANDOM_SEED = 20260830
BOOTSTRAPS = 2000
C_GRID = (0.01, 0.1, 1.0, 10.0)

AGE_CURVE = "age_distance_from_27_squared"
CONTINUITY_VARIANTS = {
    "chronology_prevalence": [],
    "phase2_prior_volume": list(V2_HISTORY_FEATURES),
    "core_available": list(V2_CORE_FEATURES),
    "core_plus_age_curve": list(V2_CORE_FEATURES) + [AGE_CURVE],
}
CONTINUITY_SELECTED = "core_plus_age_curve"
CONTINUITY_TARGET = "any_outbound_event_by_24m"
WAGE_FEATURES = list(wage_source.PRIOR_WAGE)
WAGE_TARGET = "target_log_annual_wage_eur"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: str) -> int:
    return (RANDOM_SEED + int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)) % (2**32 - 1)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def adaptive_ece(actual: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    data = pd.DataFrame({"actual": actual, "probability": probability}).sort_values("probability")
    groups = np.array_split(np.arange(len(data)), min(bins, len(data)))
    return float(sum(len(i) * abs(data.iloc[i].actual.mean() - data.iloc[i].probability.mean()) for i in groups) / len(data))


def add_age_curve(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    age = pd.to_numeric(result["age_at_signing"], errors="coerce")
    result[AGE_CURVE] = (age - 27.0) ** 2
    return result


def tune_probability(train: pd.DataFrame, validation: pd.DataFrame, features: list[str], target: str):
    if not features:
        return None, None, np.nan
    pre = fit_preprocessor(train, features)
    x_train, x_validation = pre.transform(train), pre.transform(validation)
    y_train = pd.to_numeric(train[target]).to_numpy(int)
    y_validation = pd.to_numeric(validation[target]).to_numpy(int)
    candidates = []
    for c_value in C_GRID:
        model = LogisticRegression(C=c_value, penalty="l2", solver="liblinear", max_iter=3000, random_state=RANDOM_SEED)
        model.fit(x_train, y_train)
        prediction = np.clip(model.predict_proba(x_validation)[:, 1], 1e-6, 1 - 1e-6)
        candidates.append((brier_score_loss(y_validation, prediction), c_value, model))
    _, c_value, model = min(candidates, key=lambda item: item[0])
    return pre, model, c_value


def continuity_audit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    data = add_age_curve(phase2.load_endpoint_frames()["any_outbound_24m"])
    prediction_rows: list[dict] = []
    metric_rows: list[dict] = []
    selected_support_profiles = []
    selected_interval_rows = []
    for origin, train_end, validation_year, evaluation_year in phase2.ORIGINS:
        train = data.loc[data.signed_year.le(train_end)].copy()
        validation = data.loc[data.signed_year.eq(validation_year)].copy()
        evaluation = data.loc[data.signed_year.eq(evaluation_year)].sort_values("capology_extension_event_id").copy()
        y_eval = pd.to_numeric(evaluation[CONTINUITY_TARGET]).to_numpy(int)
        baseline = float(pd.to_numeric(pd.concat([train, validation])[CONTINUITY_TARGET]).mean())
        for variant, features in CONTINUITY_VARIANTS.items():
            if not features:
                estimate = np.repeat(baseline, len(evaluation))
                c_value = np.nan
            else:
                pre, model, c_value = tune_probability(train, validation, features, CONTINUITY_TARGET)
                estimate = np.clip(model.predict_proba(pre.transform(evaluation))[:, 1], 1e-6, 1 - 1e-6)
            metric_rows.append({
                "origin_id": origin, "evaluation_year": evaluation_year, "variant": variant,
                "train_rows": len(train), "validation_rows": len(validation), "evaluation_rows": len(evaluation),
                "selected_C": c_value, "brier": brier_score_loss(y_eval, estimate),
                "roc_auc": roc_auc_score(y_eval, estimate), "average_precision": average_precision_score(y_eval, estimate),
                "actual_rate": y_eval.mean(), "predicted_rate": estimate.mean(),
            })
            for i, (_, row) in enumerate(evaluation.iterrows()):
                prediction_rows.append({
                    "origin_id": origin, "evaluation_year": evaluation_year,
                    "capology_extension_event_id": row.capology_extension_event_id,
                    "canonical_player_id": row.canonical_player_id, "canonical_position": row.canonical_position,
                    "league": row.league, "age_band": row.age_band, "variant": variant,
                    "actual": int(y_eval[i]), "prediction": float(estimate[i]),
                    "brier_loss": float((y_eval[i] - estimate[i]) ** 2), "baseline_prediction": baseline,
                    "baseline_brier_loss": float((y_eval[i] - baseline) ** 2),
                })
        features = CONTINUITY_VARIANTS[CONTINUITY_SELECTED]
        pre, model, c_value = tune_probability(train, validation, features, CONTINUITY_TARGET)
        profile, _ = phase3.support_profile(pd.concat([train, validation], ignore_index=True), features)
        support = phase3.assess_support(evaluation, features, profile)
        selected_support_profiles.append({"origin_id": origin, "evaluation_year": evaluation_year, "features": features, **profile})
        low, high, successful = phase3.bootstrap_intervals(
            train, evaluation, features, CONTINUITY_TARGET, c_value, "raw", None, pre,
            "any_outbound_24m_phase4", origin,
        )
        selected = pd.DataFrame(prediction_rows).loc[
            lambda x: x.origin_id.eq(origin) & x.variant.eq(CONTINUITY_SELECTED)
        ].sort_values("capology_extension_event_id").copy()
        for i, row in selected.reset_index(drop=True).iterrows():
            selected_interval_rows.append({
                **row.to_dict(), "support_status": support.iloc[i].support_status,
                "support_reasons": support.iloc[i].support_reasons,
                "support_distance": support.iloc[i].support_distance,
                "parameter_interval_low_80": float(low[i]), "parameter_interval_high_80": float(high[i]),
                "parameter_interval_width_80": float(high[i] - low[i]), "successful_bootstrap_fits": successful,
            })
    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)
    selected = pd.DataFrame(selected_interval_rows)

    comparisons = []
    keys = ["origin_id", "evaluation_year", "capology_extension_event_id", "canonical_player_id"]
    for candidate in ["phase2_prior_volume", "core_available", "core_plus_age_curve"]:
        left = predictions.loc[predictions.variant.eq("chronology_prevalence"), keys + ["brier_loss"]].rename(columns={"brier_loss": "baseline_loss"})
        right = predictions.loc[predictions.variant.eq(candidate), keys + ["brier_loss"]].rename(columns={"brier_loss": "candidate_loss"})
        pair = left.merge(right, on=keys, validate="one_to_one")
        pair["improvement"] = pair.baseline_loss - pair.candidate_loss
        clusters = pair.groupby("canonical_player_id").improvement.agg(["sum", "size"])
        sums, sizes = clusters["sum"].to_numpy(float), clusters["size"].to_numpy(float)
        rng = np.random.default_rng(stable_seed("continuity", candidate))
        draws = []
        for _ in range(BOOTSTRAPS // 100):
            idx = rng.integers(0, len(clusters), size=(100, len(clusters)))
            draws.extend((sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)).tolist())
        draws = np.asarray(draws)
        origin_delta = pair.groupby("origin_id").improvement.mean()
        comparisons.append({
            "candidate": candidate, "evaluation_rows": len(pair), "player_clusters": len(clusters),
            "pooled_brier": float(right.candidate_loss.mean()),
            "pooled_brier_improvement_vs_chronology": float(pair.improvement.mean()),
            "ci_low_95": float(np.quantile(draws, .025)), "ci_high_95": float(np.quantile(draws, .975)),
            "origin_wins": int((origin_delta > 0).sum()),
            "pooled_roc_auc": float(roc_auc_score(predictions.loc[predictions.variant.eq(candidate), "actual"], predictions.loc[predictions.variant.eq(candidate), "prediction"])),
            "refused_rate": float(selected.support_status.eq("refused").mean()) if candidate == CONTINUITY_SELECTED else np.nan,
        })
    comparisons = pd.DataFrame(comparisons)

    subgroup_rows = []
    for group_type, column in [("position", "canonical_position"), ("league", "league"), ("age", "age_band")]:
        for group, chunk in selected.groupby(column, dropna=False):
            n, events = len(chunk), int(chunk.actual.sum())
            nonevents, origins = n - events, chunk.origin_id.nunique()
            auc = roc_auc_score(chunk.actual, chunk.prediction) if events and nonevents else np.nan
            brier = float(chunk.brier_loss.mean())
            baseline_brier = float(chunk.baseline_brier_loss.mean())
            gap = float(abs(chunk.actual.mean() - chunk.prediction.mean()))
            sufficient = n >= MIN_SUBGROUP_ROWS and events >= MIN_SUBGROUP_EVENTS and nonevents >= MIN_SUBGROUP_NONEVENTS and origins >= MIN_SUBGROUP_ORIGINS
            supported = bool(sufficient and auc >= MIN_SUPPORTED_AUC and gap <= MAX_SUPPORTED_CALIBRATION_GAP and brier < baseline_brier)
            subgroup_rows.append({
                "group_type": group_type, "group": str(group), "rows": n, "events": events,
                "nonevents": nonevents, "origins": origins, "roc_auc": auc, "brier": brier,
                "baseline_brier": baseline_brier, "brier_skill": baseline_brier - brier,
                "calibration_gap": gap,
                "reliability_status": "supported" if supported else "limited_reliability" if sufficient else "insufficient_evidence",
            })
    subgroup = pd.DataFrame(subgroup_rows)
    selected_ece = adaptive_ece(selected.actual.to_numpy(), selected.prediction.to_numpy())
    league_supported = subgroup.loc[subgroup.group_type.eq("league"), "reliability_status"].eq("supported").all()
    decision = {
        "selected_candidate": CONTINUITY_SELECTED,
        "pooled_brier": float(selected.brier_loss.mean()),
        "pooled_auc": float(roc_auc_score(selected.actual, selected.prediction)),
        "pooled_ece": selected_ece,
        "refused_rate": float(selected.support_status.eq("refused").mean()),
        "mean_parameter_interval_width_80": float(selected.parameter_interval_width_80.mean()),
        "all_big_five_leagues_supported": bool(league_supported),
        "limited_leagues": subgroup.loc[(subgroup.group_type.eq("league")) & ~subgroup.reliability_status.eq("supported"), "group"].tolist(),
        "phase4_decision": "blocked_big_five_deployment_pending_new_context_or_untouched_holdout" if not league_supported else "advance_to_final_holdout",
        "deployment_status": "not_deployed",
    }
    return predictions, comparisons, selected, subgroup, {"decision": decision, "support_profiles": selected_support_profiles}


def wage_support(reference: pd.DataFrame, evaluation: pd.DataFrame) -> pd.DataFrame:
    categorical = ["canonical_position", "league"]
    required_numeric = ["age_at_signing", "log_market_value_at_signing", "prior_wage_available"]
    optional_prior = "log_prior_season_annual_gross_eur"
    profiles = {}
    for feature in required_numeric + [optional_prior]:
        values = pd.to_numeric(reference[feature], errors="coerce").dropna()
        profiles[feature] = {"min": values.min(), "max": values.max(), "q01": values.quantile(.01), "q99": values.quantile(.99)}
    categories = {feature: set(reference[feature].dropna().astype(str)) for feature in categorical}
    rows = []
    for _, row in evaluation.iterrows():
        reasons, limited = [], False
        for feature in categorical:
            if pd.isna(row[feature]):
                reasons.append(f"MISSING_REQUIRED_FEATURE:{feature}")
            elif str(row[feature]) not in categories[feature]:
                reasons.append(f"UNSEEN_CATEGORY:{feature}")
        for feature in required_numeric:
            value = pd.to_numeric(pd.Series([row[feature]]), errors="coerce").iloc[0]
            if pd.isna(value):
                reasons.append(f"MISSING_REQUIRED_FEATURE:{feature}")
                continue
            spec = profiles[feature]
            if value < spec["min"] or value > spec["max"]:
                reasons.append(f"OUTSIDE_OBSERVED_RANGE:{feature}")
            elif value < spec["q01"] or value > spec["q99"]:
                limited = True
        prior_available = float(row["prior_wage_available"]) == 1.0 if pd.notna(row["prior_wage_available"]) else False
        prior_value = row[optional_prior]
        if prior_available and pd.isna(prior_value):
            reasons.append(f"MISSING_REQUIRED_FEATURE:{optional_prior}")
        elif prior_available:
            spec = profiles[optional_prior]
            if prior_value < spec["min"] or prior_value > spec["max"]:
                reasons.append(f"OUTSIDE_OBSERVED_RANGE:{optional_prior}")
            elif prior_value < spec["q01"] or prior_value > spec["q99"]:
                limited = True
        status = "refused" if reasons else "limited" if limited else "supported"
        rows.append({"support_status": status, "support_reasons": ";".join(reasons)})
    return pd.DataFrame(rows, index=evaluation.index)


def wage_benchmark_audit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    frame = wage_source.load_frame().loc[lambda x: x.benchmark_wage_cohort].copy()
    prediction_rows, metric_rows = [], []
    for origin, train_end, validation_year, evaluation_year in wage_source.ORIGINS:
        train = frame.loc[frame.signed_year.le(train_end)].copy()
        validation = frame.loc[frame.signed_year.eq(validation_year)].copy()
        evaluation = frame.loc[frame.signed_year.eq(evaluation_year)].sort_values("capology_extension_event_id").copy()
        support = wage_support(pd.concat([train, validation], ignore_index=True), evaluation)
        candidate, tuning, encoded, half_width = wage_source.tune_predict(train, validation, evaluation, "annual_wage", "B2_plus_prior_wage", WAGE_FEATURES)
        baseline, _, _, baseline_half_width = wage_source.tune_predict(train, validation, evaluation, "annual_wage", "B0_chronological_median", [])
        actual = pd.to_numeric(evaluation[WAGE_TARGET]).to_numpy(float)
        lower, upper = candidate - half_width, candidate + half_width
        metric_rows.append({
            "origin_id": origin, "evaluation_year": evaluation_year, "train_rows": len(train),
            "validation_rows": len(validation), "evaluation_rows": len(evaluation), "encoded_features": encoded,
            "log_mae": mean_absolute_error(actual, candidate), "baseline_log_mae": mean_absolute_error(actual, baseline),
            "mae_improvement_vs_chronology": mean_absolute_error(actual, baseline) - mean_absolute_error(actual, candidate),
            "r2": r2_score(actual, candidate), "spearman": pd.Series(actual).corr(pd.Series(candidate), method="spearman"),
            "interval_80_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
            "refused_rate": float(support.support_status.eq("refused").mean()),
            "validation_interval_abs_residual_q80": half_width,
            "baseline_interval_abs_residual_q80": baseline_half_width,
            "selected_parameter": tuning["selected_parameter"],
        })
        for i, (_, row) in enumerate(evaluation.iterrows()):
            prediction_rows.append({
                "origin_id": origin, "evaluation_year": evaluation_year,
                "capology_extension_event_id": row.capology_extension_event_id,
                "canonical_player_id": row.canonical_player_id, "canonical_position": row.canonical_position,
                "league": row.league, "age_band": row.age_band, "market_value_band": row.market_value_band,
                "prior_wage_available": row.prior_wage_available,
                "actual_log_eur": actual[i], "prediction_log_eur": candidate[i],
                "prediction_eur": math.exp(candidate[i]), "interval_lower_eur": math.exp(lower[i]),
                "interval_upper_eur": math.exp(upper[i]), "within_80_interval": bool(lower[i] <= actual[i] <= upper[i]),
                "absolute_log_error": abs(actual[i] - candidate[i]), "baseline_absolute_log_error": abs(actual[i] - baseline[i]),
                "support_status": support.iloc[i].support_status, "support_reasons": support.iloc[i].support_reasons,
            })
    predictions, metrics = pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)
    subgroup_rows = []
    for group_type, column in [("position", "canonical_position"), ("league", "league"), ("age", "age_band"), ("prior_wage", "prior_wage_available")]:
        for group, chunk in predictions.groupby(column, dropna=False):
            sufficient = len(chunk) >= MIN_SUBGROUP_ROWS and chunk.origin_id.nunique() >= MIN_SUBGROUP_ORIGINS
            improvement = float((chunk.baseline_absolute_log_error - chunk.absolute_log_error).mean())
            coverage = float(chunk.within_80_interval.mean())
            supported = bool(sufficient and improvement > 0 and .70 <= coverage <= .90)
            subgroup_rows.append({
                "group_type": group_type, "group": str(group), "rows": len(chunk), "origins": chunk.origin_id.nunique(),
                "log_mae": chunk.absolute_log_error.mean(), "baseline_log_mae": chunk.baseline_absolute_log_error.mean(),
                "mae_improvement_vs_chronology": improvement, "interval_80_coverage": coverage,
                "reliability_status": "supported" if supported else "limited_reliability" if sufficient else "insufficient_evidence",
            })
    subgroup = pd.DataFrame(subgroup_rows)
    broad_supported = subgroup.loc[subgroup.group_type.isin(["position", "league"]), "reliability_status"].eq("supported").all()
    decision = {
        "selected_candidate": "B2_plus_prior_wage",
        "pooled_log_mae": float(predictions.absolute_log_error.mean()),
        "pooled_baseline_log_mae": float(predictions.baseline_absolute_log_error.mean()),
        "pooled_mae_improvement_vs_chronology": float((predictions.baseline_absolute_log_error - predictions.absolute_log_error).mean()),
        "pooled_interval_80_coverage": float(predictions.within_80_interval.mean()),
        "pooled_refused_rate": float(predictions.support_status.eq("refused").mean()),
        "all_position_league_groups_supported": bool(broad_supported),
        "limited_groups": subgroup.loc[~subgroup.reliability_status.eq("supported"), ["group_type", "group"]].to_dict("records"),
        "phase4_decision": "advance_to_final_holdout" if broad_supported else "blocked_subgroup_repair",
        "deployment_status": "not_deployed",
    }
    return predictions, metrics, subgroup, decision


def relationship_audit() -> pd.DataFrame:
    frames = phase2.load_endpoint_frames()
    role = frames["sustained_meaningful_contribution"][["capology_extension_event_id", "target_sustained_meaningful_contribution"]]
    continuity = frames["any_outbound_24m"][["capology_extension_event_id", "any_outbound_event_by_24m"]]
    overlap = role.merge(continuity, on="capology_extension_event_id", validate="one_to_one")
    rows = []
    for sustained in [0, 1]:
        for outbound in [0, 1]:
            count = int(((overlap.target_sustained_meaningful_contribution == sustained) & (overlap.any_outbound_event_by_24m == outbound)).sum())
            rows.append({
                "sustained_meaningful_contribution": sustained, "any_outbound_24m": outbound,
                "rows": count, "share_of_overlap": count / len(overlap),
            })
    return pd.DataFrame(rows)


def contract_examples() -> list[dict]:
    examples = [
        ResultEnvelope(
            request_id="contract-example-supported", engine_version=ENGINE_VERSION,
            scoring_as_of="2026-08-30", decision_scope="incumbent_club_extension_scenario",
            modules=(
                ModuleResult(
                    module_id="future_role", endpoint_id="sustained_meaningful_contribution",
                    result_kind="probability", status="supported", value=.72, unit="probability",
                    target_definition="at_least_25pct_same_club_opportunity_share_in_each_year",
                    horizon="year_1_and_year_2", calibration_method="raw",
                    interval=Interval("model_fit_80", .64, .79, "probability"),
                    subgroup_reliability={"league": "supported", "position": "supported"},
                    data_vintage={"evaluation_cutoff": "2023"},
                    limitations=("Historical conditional estimate; not a recommendation.",),
                    artifact_id="not_built_phase4",
                ),
                ModuleResult(
                    module_id="club_continuity", endpoint_id="any_outbound_24m",
                    result_kind="probability", status="refused", value=None, unit="probability",
                    target_definition="any_recorded_loan_or_permanent_outbound_move", horizon="24m",
                    calibration_method="raw", support_reasons=("SUBGROUP_RELIABILITY_LIMITED:Ligue 1",),
                    data_vintage={"evaluation_cutoff": "2023"},
                    limitations=("No personalized Ligue 1 continuity result is authorized by Phase 4.",),
                ),
                ModuleResult(
                    module_id="wage_benchmark", endpoint_id="annual_wage_peer_benchmark",
                    result_kind="historical_benchmark", status="limited", value=8_000_000,
                    unit="EUR_per_year", target_definition="historical_annual_fixed_wage", horizon="proposal",
                    interval=Interval("historical_reference_80", 4_200_000, 14_900_000, "EUR_per_year"),
                    support_reasons=("PROFILE_NEAR_HISTORICAL_EDGE",),
                    subgroup_reliability={"league": "supported", "position": "supported"},
                    data_vintage={"evaluation_cutoff": "2023"},
                    limitations=("Peer context, not fair value or an optimal wage.",), artifact_id="not_built_phase4",
                ),
            ),
        ).as_dict(),
    ]
    for payload in examples:
        validate_payload(payload)
    return examples


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for stale in ["independent_verification.csv", "independent_verification.json"]:
        (OUTPUT / stale).unlink(missing_ok=True)
    phase3_verify = json.loads((PHASE3 / "independent_verification.json").read_text(encoding="utf-8"))
    if not phase3_verify.get("all_checks_passed"):
        raise RuntimeError("Phase 3 has not passed independent verification")

    continuity_all, continuity_comparison, continuity_selected, continuity_subgroup, continuity_meta = continuity_audit()
    wage_predictions, wage_metrics, wage_subgroup, wage_decision = wage_benchmark_audit()
    relationships = relationship_audit()
    examples = contract_examples()

    continuity_all.to_csv(OUTPUT / "continuity_candidate_predictions.csv", index=False, lineterminator="\n")
    continuity_comparison.to_csv(OUTPUT / "continuity_candidate_comparison.csv", index=False, lineterminator="\n")
    continuity_selected.to_csv(OUTPUT / "continuity_selected_predictions.csv", index=False, lineterminator="\n")
    continuity_subgroup.to_csv(OUTPUT / "continuity_subgroup_reliability.csv", index=False, lineterminator="\n")
    write_json(OUTPUT / "continuity_support_profiles.json", continuity_meta["support_profiles"])
    wage_predictions.to_csv(OUTPUT / "wage_benchmark_predictions.csv", index=False, lineterminator="\n")
    wage_metrics.to_csv(OUTPUT / "wage_benchmark_origin_metrics.csv", index=False, lineterminator="\n")
    wage_subgroup.to_csv(OUTPUT / "wage_benchmark_subgroup_reliability.csv", index=False, lineterminator="\n")
    relationships.to_csv(OUTPUT / "target_relationship_audit.csv", index=False, lineterminator="\n")
    write_json(OUTPUT / "result_contract_examples.json", examples)
    write_json(OUTPUT / "phase4_decisions.json", {
        "continuity": continuity_meta["decision"], "wage_benchmark": wage_decision,
        "future_role": {"phase4_decision": "specification_frozen_pending_final_holdout", "calibration": "raw"},
        "public_value_downside": {"phase4_decision": "specification_frozen_pending_final_holdout", "calibration": "platt"},
        "overall_engine": {
            "decision": "not_ready_for_deployment",
            "reason": "Continuity lacks supported Big-Five coverage; no final untouched temporal holdout has been evaluated.",
        },
    })
    write_json(OUTPUT / "final_holdout_plan.json", {
        "status": "reserved_not_evaluated",
        "selection_and_calibration_evidence_through": 2023,
        "earliest_eligible_extension_year": 2024,
        "requirements": [
            "Outcome horizon must be completely observable as of scoring date.",
            "No feature, calibration, support threshold, or subgroup rule may be changed after opening the holdout.",
            "Player identity clusters must not cross fit and test partitions.",
            "Every module must pass its pooled, calibration, refusal, and subgroup gates independently.",
        ],
    })

    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    freeze_ok = all(sha256(ROOT / item["path"]) == item["sha256"] for item in freeze["files"])
    simultaneous = relationships.loc[(relationships.sustained_meaningful_contribution == 1) & (relationships.any_outbound_24m == 1), "rows"].iloc[0]
    checks = pd.DataFrame([
        ("continuity_four_variants", set(continuity_all.variant) == set(CONTINUITY_VARIANTS), "Declared repair candidates only."),
        ("continuity_four_rolling_origins", set(continuity_selected.evaluation_year) == {2020, 2021, 2022, 2023}, "Chronological evaluation retained."),
        ("continuity_age_curve_improves_chronology", continuity_comparison.loc[continuity_comparison.candidate.eq(CONTINUITY_SELECTED), "ci_low_95"].iloc[0] > 0, "Player-cluster CI excludes zero."),
        ("continuity_coverage_repaired", continuity_meta["decision"]["refused_rate"] <= MAX_ACCEPTABLE_REFUSAL_RATE, "Missing historical volume no longer drives refusal."),
        ("continuity_big_five_failure_not_hidden", not continuity_meta["decision"]["all_big_five_leagues_supported"], "Ligue 1 limitation remains explicit."),
        ("wage_benchmark_improves_chronology", wage_decision["pooled_mae_improvement_vs_chronology"] > 0, "Historical salary context adds signal."),
        ("wage_interval_coverage_reasonable", .70 <= wage_decision["pooled_interval_80_coverage"] <= .90, "Reference range is empirically audited."),
        ("wage_refusal_within_limit", wage_decision["pooled_refused_rate"] <= MAX_ACCEPTABLE_REFUSAL_RATE, "Benchmark is operationally available."),
        ("targets_not_forced_mutually_exclusive", simultaneous > 0, "Observed labels prove cards cannot be normalized as mutually exclusive."),
        ("contract_examples_validate", all(validate_payload(item) is None for item in examples), "Serialized examples satisfy the V2 contract."),
        ("no_overall_scores", all(item["overall_score"] is None for item in examples), "No universal score reintroduced."),
        ("no_model_artifacts_written", not list(OUTPUT.rglob("*.joblib")), "Phase 4 does not serialize candidates."),
        ("v1_artifacts_remain_frozen", freeze_ok, "All Phase-0 hashes remain byte-identical."),
    ], columns=["check", "passed", "notes"])
    checks.to_csv(OUTPUT / "build_checks.csv", index=False, lineterminator="\n")

    source_paths = [
        PHASE3 / "phase3_decisions.csv", PHASE3 / "independent_verification.json",
        ROOT / "Data" / "processed" / "contract_financial_exposure_benchmark" / "independent_verification.json",
        FREEZE,
    ]
    pd.DataFrame([{"source": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in source_paths]).to_csv(
        OUTPUT / "source_manifest.csv", index=False, lineterminator="\n"
    )
    summary = {
        "phase": "engine_v2_phase_4_candidate_contract", "status": "complete_research_only_not_deployed",
        "engine_version": ENGINE_VERSION, "continuity_decision": continuity_meta["decision"]["phase4_decision"],
        "wage_benchmark_decision": wage_decision["phase4_decision"],
        "overall_engine_decision": "not_ready_for_deployment",
        "checks_passed": int(checks.passed.sum()), "checks_total": len(checks),
        "all_checks_passed": bool(checks.passed.all()), "v1_artifacts_modified": False,
    }
    write_json(OUTPUT / "run_summary.json", summary)
    lines = [
        "# Engine V2 Phase 4 — candidate contract", "",
        "Phase 4 repairs continuity's input contract, audits the annual-wage peer benchmark, and freezes a refusal-safe response schema. It does not serialize or deploy a model.", "",
        "## Decisions", "",
        f"- Continuity: `{continuity_meta['decision']['phase4_decision']}`. The selected compact age-curve candidate has Brier {continuity_meta['decision']['pooled_brier']:.4f}, AUC {continuity_meta['decision']['pooled_auc']:.3f}, ECE {continuity_meta['decision']['pooled_ece']:.3f}, and refusal rate {continuity_meta['decision']['refused_rate']:.1%}. Its observable feature contract repairs the prior 60.7% refusal failure, but {', '.join(continuity_meta['decision']['limited_leagues'])} still fails the predeclared subgroup-skill gate.",
        f"- Wage benchmark: `{wage_decision['phase4_decision']}`. The historical peer estimate improves log-MAE over a chronology-only median by {wage_decision['pooled_mae_improvement_vs_chronology']:.3f}; its nominal 80% range covered {wage_decision['pooled_interval_80_coverage']:.1%} of unseen observations and refused {wage_decision['pooled_refused_rate']:.1%} of profiles.",
        "- Overall engine: `not_ready_for_deployment`. Continuity does not meet Big-Five subgroup coverage and the final later-season temporal holdout remains sealed.", "",
        "## Contract rules", "",
        "- Each module carries its own value, interval, support state, subgroup status, target, horizon, calibration, data vintage, artifact identity, and limitations.",
        "- Refused or unavailable modules must return no numeric value and no interval.",
        "- The engine cannot return an overall score or automated extend/do-not-extend recommendation.",
        f"- Contribution and outbound are not mutually exclusive in the labels: {int(simultaneous)} overlapping historical events had both sustained contribution and an outbound event. Their probabilities must not be added or normalized.",
        "- The wage output is historical peer context, not fair value, an optimal offer, or proof of savings.", "",
        "## Holdout", "",
        "The earliest candidate final holdout is extension year 2024, only after each endpoint's full outcome horizon is observable. Phase 4 does not open or inspect that holdout.", "",
    ]
    (OUTPUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    manifest_rows = []
    for path in sorted(OUTPUT.iterdir()):
        if path.is_file() and path.name != "output_manifest.csv":
            manifest_rows.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest_rows).to_csv(OUTPUT / "output_manifest.csv", index=False, lineterminator="\n")
    print(json.dumps(summary, indent=2))
    if not summary["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

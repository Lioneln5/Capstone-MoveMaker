"""Build and evaluate the Engine V2 Phase-2 feature specification.

This phase does not promote or deploy a model.  It evaluates compact,
decision-time-valid candidates for the three predictive headline modules and
audits the frozen V1 feature lists against the V2 contract.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "models") not in sys.path:
    sys.path.insert(0, str(ROOT / "models"))

from models.engine_v2.feature_contract import (  # noqa: E402
    ADVANCED_PERFORMANCE,
    COMMERCIAL_SCENARIO,
    CORE_PROFILE_FEATURES,
    HEADLINE_ENDPOINTS,
    HISTORICAL_SALARY_CONTEXT,
    RAW_SCORING_RATES,
    V2_CORE_FEATURES,
    V2_HISTORY_FEATURES,
    feature_group,
)
from mixed_type_preprocessor import fit_preprocessor  # noqa: E402
import run_extension_opportunity_diagnostic as opportunity  # noqa: E402
import run_extension_survival_diagnostic as survival  # noqa: E402
import run_extension_value_preservation_diagnostic as value  # noqa: E402

OUTPUT = ROOT / "Data" / "processed" / "engine_v2_feature_specification"
DEPLOYMENT = ROOT / "Data" / "processed" / "deployment_models"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"
ROLE_EVIDENCE = ROOT / "Data" / "processed" / "role_contextualization_diagnostic"
ORIGINS = opportunity.ORIGINS
RANDOM_SEED = 20260830
BOOTSTRAPS = 3000
C_GRID = (0.01, 0.1, 1.0, 10.0)

VARIANTS = {
    "B0_intercept_only": [],
    "B1_market_profile": CORE_PROFILE_FEATURES,
    "V2_core_involvement": V2_CORE_FEATURES,
    "V2_plus_prior_volume": V2_HISTORY_FEATURES,
    "historical_M5_reference": opportunity.FEATURE_SETS["M5_plus_relative_financial_context"],
}

ENDPOINT_META = {
    "sustained_meaningful_contribution": {
        "module": "future_role", "target": "target_sustained_meaningful_contribution",
    },
    "any_outbound_24m": {
        "module": "club_continuity", "target": "any_outbound_event_by_24m",
    },
    "downside_25pct_24m": {
        "module": "public_value_downside", "target": "target_downside_25pct_24m",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_seed(*parts: str) -> int:
    return (RANDOM_SEED + int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)) % (2**32 - 1)


def load_endpoint_frames() -> dict[str, pd.DataFrame]:
    role = opportunity.load_frame()
    role = opportunity.endpoint_frame(role, "sustained_meaningful_contribution")

    stay = survival.load_frame()
    stay = stay.loc[
        stay["survival_primary_cohort"]
        & stay["any_outbound_horizon_observable_24m"]
    ].copy()

    downside = value.load_frame()
    downside = value.endpoint_frame(downside, "downside_25pct_24m")
    return {
        "sustained_meaningful_contribution": role,
        "any_outbound_24m": stay,
        "downside_25pct_24m": downside,
    }


def tune_and_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    evaluation: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[np.ndarray, float, int]:
    if not features:
        # A chronology-safe intercept baseline: the evaluation year receives
        # only the prevalence observed through the preceding validation year.
        fit = pd.concat([train, validation], ignore_index=True)
        prevalence = float(pd.to_numeric(fit[target]).mean())
        return np.repeat(np.clip(prevalence, 1e-6, 1 - 1e-6), len(evaluation)), np.nan, 0
    pre = fit_preprocessor(train, features)
    x_train, x_validation = pre.transform(train), pre.transform(validation)
    y_train = pd.to_numeric(train[target]).to_numpy(int)
    y_validation = pd.to_numeric(validation[target]).to_numpy(int)
    candidates = []
    for c_value in C_GRID:
        model = LogisticRegression(
            C=c_value, penalty="l2", solver="liblinear", max_iter=3000,
            random_state=RANDOM_SEED,
        ).fit(x_train, y_train)
        estimate = np.clip(model.predict_proba(x_validation)[:, 1], 1e-6, 1 - 1e-6)
        candidates.append((brier_score_loss(y_validation, estimate), c_value))
    _, c_value = min(candidates)
    fit = pd.concat([train, validation], ignore_index=True)
    fit_pre = fit_preprocessor(fit, features)
    model = LogisticRegression(
        C=c_value, penalty="l2", solver="liblinear", max_iter=3000,
        random_state=RANDOM_SEED,
    ).fit(fit_pre.transform(fit), pd.to_numeric(fit[target]).to_numpy(int))
    prediction = np.clip(model.predict_proba(fit_pre.transform(evaluation))[:, 1], 1e-6, 1 - 1e-6)
    return prediction, c_value, fit_pre.transform(fit).shape[1]


def evaluate_candidates(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions: list[dict] = []
    metrics: list[dict] = []
    for endpoint, data in frames.items():
        target = ENDPOINT_META[endpoint]["target"]
        for origin, train_end, validation_year, evaluation_year in ORIGINS:
            train = data.loc[data["signed_year"].le(train_end)].copy()
            validation = data.loc[data["signed_year"].eq(validation_year)].copy()
            evaluation = data.loc[data["signed_year"].eq(evaluation_year)].sort_values("capology_extension_event_id").copy()
            if min(len(train), len(validation), len(evaluation)) < 30:
                raise RuntimeError(f"Insufficient split for {endpoint}/{origin}")
            for variant, features in VARIANTS.items():
                estimate, c_value, encoded = tune_and_predict(train, validation, evaluation, list(features), target)
                actual = pd.to_numeric(evaluation[target]).to_numpy(int)
                metrics.append({
                    "endpoint": endpoint, "module": ENDPOINT_META[endpoint]["module"],
                    "origin_id": origin, "evaluation_year": evaluation_year,
                    "variant": variant, "train_rows": len(train), "validation_rows": len(validation),
                    "evaluation_rows": len(evaluation), "raw_feature_count": len(features),
                    "encoded_feature_count": encoded, "selected_C": c_value,
                    "brier": brier_score_loss(actual, estimate),
                    "roc_auc": roc_auc_score(actual, estimate),
                    "average_precision": average_precision_score(actual, estimate),
                    "actual_rate": actual.mean(), "predicted_rate": estimate.mean(),
                })
                for i, (_, row) in enumerate(evaluation.iterrows()):
                    predictions.append({
                        "endpoint": endpoint, "module": ENDPOINT_META[endpoint]["module"],
                        "origin_id": origin, "evaluation_year": evaluation_year,
                        "capology_extension_event_id": row["capology_extension_event_id"],
                        "canonical_player_id": row["canonical_player_id"],
                        "league": row["league"], "canonical_position": row["canonical_position"],
                        "age_band": row["age_band"], "variant": variant,
                        "actual": int(actual[i]), "prediction": float(estimate[i]),
                        "brier_loss": float((actual[i] - estimate[i]) ** 2),
                    })
    return pd.DataFrame(predictions), pd.DataFrame(metrics)


def clustered_comparisons(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["endpoint", "origin_id", "evaluation_year", "capology_extension_event_id", "canonical_player_id"]
    for endpoint in ENDPOINT_META:
        data = predictions.loc[predictions["endpoint"].eq(endpoint)]
        for candidate in ["B1_market_profile", "V2_core_involvement", "V2_plus_prior_volume"]:
            baselines = ["B0_intercept_only", "historical_M5_reference"]
            if candidate != "B1_market_profile":
                baselines.append("B1_market_profile")
            for baseline in baselines:
                left = data.loc[data["variant"].eq(baseline), keys + ["brier_loss"]].rename(columns={"brier_loss": "baseline_loss"})
                right = data.loc[data["variant"].eq(candidate), keys + ["brier_loss"]].rename(columns={"brier_loss": "candidate_loss"})
                pair = left.merge(right, on=keys, validate="one_to_one")
                pair["improvement"] = pair["baseline_loss"] - pair["candidate_loss"]
                clusters = pair.groupby("canonical_player_id")["improvement"].agg(["sum", "size"])
                sums, sizes = clusters["sum"].to_numpy(float), clusters["size"].to_numpy(float)
                rng = np.random.default_rng(stable_seed(endpoint, candidate, baseline))
                draws = []
                for start in range(0, BOOTSTRAPS, 250):
                    count = min(250, BOOTSTRAPS - start)
                    idx = rng.integers(0, len(clusters), size=(count, len(clusters)))
                    draws.extend((sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)).tolist())
                draws = np.asarray(draws)
                origin_delta = pair.groupby("origin_id")["improvement"].mean()
                rows.append({
                    "endpoint": endpoint, "baseline": baseline, "candidate": candidate,
                    "evaluation_rows": len(pair), "player_clusters": len(clusters),
                    "pooled_brier_improvement": pair["improvement"].mean(),
                    "origin_wins": int(origin_delta.gt(0).sum()), "origin_count": len(origin_delta),
                    "bootstrap_ci_lower_95": np.quantile(draws, 0.025),
                    "bootstrap_ci_upper_95": np.quantile(draws, 0.975),
                    "probability_candidate_better": np.mean(draws > 0),
                })
    return pd.DataFrame(rows)


def subgroup_audit(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint in ENDPOINT_META:
        data = predictions.loc[predictions["endpoint"].eq(endpoint)]
        for group_type, column in [("position", "canonical_position"), ("league", "league"), ("age", "age_band")]:
            for group, chunk in data.groupby(column, dropna=False):
                for variant, sample in chunk.groupby("variant"):
                    rows.append({
                        "endpoint": endpoint, "group_type": group_type, "group": str(group),
                        "variant": variant, "n": len(sample), "actual_rate": sample["actual"].mean(),
                        "brier": sample["brier_loss"].mean(),
                    })
    return pd.DataFrame(rows)


def artifact_inventory() -> pd.DataFrame:
    rows = []
    for path in sorted(DEPLOYMENT.rglob("*.joblib")):
        if "_backups" in path.parts:
            continue
        artifact = joblib.load(path)
        endpoint = artifact.endpoint + (f"_{artifact.horizon}" if artifact.horizon else "")
        module = (
            "future_role" if artifact.endpoint in {"year2_opportunity_share", "sustained_meaningful_contribution"}
            else "club_continuity" if artifact.endpoint in {"any_outbound", "permanent_outbound"}
            else "public_value_downside" if artifact.endpoint.startswith(("value_log_ratio", "downside_"))
            else "wage_benchmark"
        )
        contract = next(item for item in HEADLINE_ENDPOINTS if item.module == module)
        for feature in artifact.features:
            group = feature_group(feature)
            rows.append({
                "artifact": str(path.relative_to(ROOT)), "v1_endpoint": endpoint,
                "module": module, "feature": feature, "feature_group": group,
                "v2_allowed_for_module": group in contract.allowed_feature_groups,
                "v2_violation": group in contract.prohibited_feature_groups,
            })
    return pd.DataFrame(rows)


def endpoint_contract_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "module": item.module, "endpoint": item.endpoint, "horizon": item.horizon,
            "target": item.target, "public_status": item.public_status, "model_form": item.model_form,
            "allowed_feature_groups": " | ".join(item.allowed_feature_groups),
            "prohibited_feature_groups": " | ".join(item.prohibited_feature_groups),
            "interpretation": item.interpretation,
        }
        for item in HEADLINE_ENDPOINTS
    ])


def feature_policy_frame() -> pd.DataFrame:
    rows = []
    for variant, features in VARIANTS.items():
        for feature in features:
            rows.append({"variant": variant, "feature": feature, "feature_group": feature_group(feature)})
    return pd.DataFrame(rows)


def choose_candidates(comparisons: pd.DataFrame, subgroup: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint in ENDPOINT_META:
        versus_intercept = comparisons.loc[
            comparisons["endpoint"].eq(endpoint)
            & comparisons["baseline"].eq("B0_intercept_only")
            & comparisons["candidate"].isin(["V2_core_involvement", "V2_plus_prior_volume"])
        ].copy()
        # Prefer the valid candidate with the largest pooled gain over a
        # chronology-only baseline.  The invalid historical full ladder is
        # retained as a transparent performance reference, never as a gate.
        best = versus_intercept.sort_values(
            ["pooled_brier_improvement", "candidate"], ascending=[False, True]
        ).iloc[0]
        versus_market = comparisons.loc[
            comparisons["endpoint"].eq(endpoint)
            & comparisons["baseline"].eq("B1_market_profile")
            & comparisons["candidate"].eq(best.candidate)
        ].iloc[0]
        versus_history = comparisons.loc[
            comparisons["endpoint"].eq(endpoint)
            & comparisons["baseline"].eq("historical_M5_reference")
            & comparisons["candidate"].eq(best.candidate)
        ].iloc[0]
        validated_signal = bool(best.bootstrap_ci_lower_95 > 0 and best.origin_wins >= 3)
        involvement_increment = bool(versus_market.pooled_brier_improvement > 0 and versus_market.origin_wins >= 2)
        rows.append({
            "endpoint": endpoint, "module": ENDPOINT_META[endpoint]["module"],
            "selected_phase2_candidate": best.candidate,
            "brier_improvement_vs_intercept": best.pooled_brier_improvement,
            "intercept_ci_lower_95": best.bootstrap_ci_lower_95,
            "intercept_ci_upper_95": best.bootstrap_ci_upper_95,
            "origin_wins_vs_intercept": int(best.origin_wins),
            "brier_improvement_vs_market_profile": versus_market.pooled_brier_improvement,
            "origin_wins_vs_market_profile": int(versus_market.origin_wins),
            "brier_improvement_vs_historical_m5": versus_history.pooled_brier_improvement,
            "historical_m5_ci_lower_95": versus_history.bootstrap_ci_lower_95,
            "historical_m5_ci_upper_95": versus_history.bootstrap_ci_upper_95,
            "validated_signal_gate": validated_signal,
            "recent_involvement_increment": involvement_increment,
            "phase3_must_audit_subgroups": True,
            "phase2_decision": "advance_to_phase3_candidate" if validated_signal else "blocked_no_valid_signal",
            "deployment_status": "not_deployed",
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for stale_verification in ["independent_verification.csv", "independent_verification.json"]:
        (OUTPUT / stale_verification).unlink(missing_ok=True)
    frames = load_endpoint_frames()
    predictions, metrics = evaluate_candidates(frames)
    comparisons = clustered_comparisons(predictions)
    subgroup = subgroup_audit(predictions)
    inventory = artifact_inventory()
    decisions = choose_candidates(comparisons, subgroup)

    endpoint_contract_frame().to_csv(OUTPUT / "endpoint_contract.csv", index=False, lineterminator="\n")
    feature_policy_frame().to_csv(OUTPUT / "candidate_feature_manifest.csv", index=False, lineterminator="\n")
    inventory.to_csv(OUTPUT / "v1_feature_policy_audit.csv", index=False, lineterminator="\n")
    metrics.to_csv(OUTPUT / "origin_metrics.csv", index=False, lineterminator="\n")
    predictions.to_csv(OUTPUT / "out_of_time_predictions.csv", index=False, lineterminator="\n")
    comparisons.to_csv(OUTPUT / "candidate_comparisons.csv", index=False, lineterminator="\n")
    subgroup.to_csv(OUTPUT / "subgroup_audit.csv", index=False, lineterminator="\n")
    decisions.to_csv(OUTPUT / "phase2_decisions.csv", index=False, lineterminator="\n")

    role = pd.read_csv(ROLE_EVIDENCE / "pooled_model_comparison.csv")
    role = role.loc[
        role["candidate"].eq("R1_remove_scoring")
        & role["window"].eq("pooled_four")
        & role["endpoint"].isin(["year2_opportunity_share", "any_outbound_24m", "downside_25pct_24m"])
    ].copy()
    role.to_csv(OUTPUT / "prior_role_ablation_evidence.csv", index=False, lineterminator="\n")

    source_paths = [
        opportunity.SOURCE,
        ROLE_EVIDENCE / "pooled_model_comparison.csv",
        ROOT / "Data" / "processed" / "extension_opportunity_diagnostic" / "model_comparison_results.csv",
        ROOT / "Data" / "processed" / "extension_survival_diagnostic" / "model_comparison_results.csv",
        ROOT / "Data" / "processed" / "extension_value_preservation_diagnostic" / "model_comparison_results.csv",
        FREEZE,
    ]
    pd.DataFrame([
        {"source": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in source_paths
    ]).to_csv(OUTPUT / "source_manifest.csv", index=False, lineterminator="\n")

    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    freeze_ok = all(sha256(ROOT / item["path"]) == item["sha256"] for item in freeze["files"])
    checks = pd.DataFrame([
        ("four_headline_modules_declared", len(HEADLINE_ENDPOINTS) == 4, "One separate contract per module."),
        ("three_predictive_candidates_evaluated", set(decisions["endpoint"]) == set(ENDPOINT_META), "Sporting/value modules evaluated chronologically."),
        ("candidate_features_have_no_scoring_rates", not set(RAW_SCORING_RATES) & set(V2_HISTORY_FEATURES), "No global goals/assists rate."),
        ("candidate_features_have_no_advanced_stats", not set(ADVANCED_PERFORMANCE) & set(V2_HISTORY_FEATURES), "No unsupported incremental event block."),
        ("candidate_features_have_no_commercial_terms", not set(COMMERCIAL_SCENARIO) & set(V2_HISTORY_FEATURES), "User proposals cannot alter sporting/value probabilities."),
        ("v1_policy_violations_are_visible", inventory["v2_violation"].any(), "Phase 2 does not silently bless V1 feature lists."),
        ("all_evaluations_are_rolling_2020_to_2023", set(metrics["evaluation_year"]) == {2020, 2021, 2022, 2023}, "No random split."),
        ("no_candidate_artifact_written", not list(OUTPUT.rglob("*.joblib")), "Passing Phase 2 cannot promote a model."),
        ("v1_artifacts_remain_frozen", freeze_ok, "Every frozen V1 hash remains byte-identical."),
    ], columns=["check", "passed", "notes"])
    checks.to_csv(OUTPUT / "build_checks.csv", index=False, lineterminator="\n")

    summary = {
        "phase": "engine_v2_phase_2_feature_specification",
        "status": "complete_research_candidates_not_deployed",
        "evaluated_endpoints": len(decisions),
        "rolling_origins": 4,
        "out_of_time_prediction_rows": len(predictions),
        "v1_artifacts_audited": inventory["artifact"].nunique(),
        "v1_feature_policy_violation_rows": int(inventory["v2_violation"].sum()),
        "v1_artifacts_with_policy_violations": int(inventory.loc[inventory["v2_violation"], "artifact"].nunique()),
        "v1_artifacts_with_raw_scoring_rates": int(inventory.loc[inventory["feature_group"].eq("raw_scoring_rates"), "artifact"].nunique()),
        "candidate_decisions": decisions.set_index("endpoint")["phase2_decision"].to_dict(),
        "checks_passed": int(checks["passed"].sum()),
        "checks_total": len(checks),
        "all_checks_passed": bool(checks["passed"].all()),
        "v1_artifacts_modified": False,
    }
    write_json(OUTPUT / "run_summary.json", summary)

    lines = [
        "# Engine V2 Phase 2 — endpoint and feature specification", "",
        "This package converts the Phase-1 information boundary into model specifications and tests compact candidates. It does not deploy or serialize a model.", "",
        "## Decisions", "",
        "- Keep four modules separate: future role, 24-month club continuity, 24-month public-value downside, and historical wage benchmarking.",
        "- Use a single headline threshold/horizon per predictive module rather than exposing V1's full endpoint grid.",
        "- Exclude raw goals/assists rates. The prior role diagnostic found removal non-inferior for the headline endpoints, while role-aware variants did not reliably fix demo-player behavior.",
        "- Exclude advanced event statistics. Earlier block tests did not establish incremental value beyond simpler information.",
        "- Exclude proposed wage, term, and their derivatives from sporting and public-value models. Historical predictiveness can reflect club selection and negotiation; it does not make a user-controlled proposal a valid causal input.",
        "- Keep proposed wage and term in the financial scenario layer only. The wage peer model must predict a historical benchmark without seeing the proposal it is asked to benchmark.", "",
        "## Candidate results", "",
    ]
    for row in decisions.itertuples(index=False):
        lines.append(
            f"- `{row.endpoint}`: `{row.selected_phase2_candidate}`; Brier improvement "
            f"{row.brier_improvement_vs_intercept:+.6f} versus the chronology-only baseline, "
            f"95% cluster interval [{row.intercept_ci_lower_95:+.6f}, {row.intercept_ci_upper_95:+.6f}], "
            f"historical-M5 comparison {row.brier_improvement_vs_historical_m5:+.6f}, "
            f"decision `{row.phase2_decision}`."
        )
    lines += ["", "These are model-specification results, not evidence that public scoring can resume. Phase 3 must address calibration, uncertainty, subgroup/OOD refusals, and candidate artifacts under the Phase-1 request contract.", ""]
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

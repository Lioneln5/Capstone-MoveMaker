"""Rolling-origin continuity experiment for Capology club financial context.

This is a development-only repair experiment for Engine V2. It reconstructs
the locked Phase-4 continuity baseline, evaluates five predeclared feature
blocks on the same four historical origins, and never opens the future final
holdout or writes a deployable model artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "models")]

from models.engine_v2.feature_contract import V2_CORE_FEATURES  # noqa: E402
from models.engine_v2.reliability_contract import (  # noqa: E402
    MAX_ACCEPTABLE_ECE,
    MAX_ACCEPTABLE_REFUSAL_RATE,
    MAX_SUPPORTED_CALIBRATION_GAP,
    MIN_SUBGROUP_EVENTS,
    MIN_SUBGROUP_NONEVENTS,
    MIN_SUBGROUP_ORIGINS,
    MIN_SUBGROUP_ROWS,
    MIN_SUPPORTED_AUC,
)
from mixed_type_preprocessor import fit_preprocessor  # noqa: E402
import run_engine_v2_calibration_reliability as phase3  # noqa: E402
import run_engine_v2_candidate_contract as phase4  # noqa: E402
import run_engine_v2_feature_specification as phase2  # noqa: E402


OUTPUT = ROOT / "Data" / "processed" / "engine_v2_club_financial_context_experiment"
CONTEXT = ROOT / "Data" / "processed" / "capology_club_financial_context"
PHASE4 = ROOT / "Data" / "processed" / "engine_v2_candidate_contract"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"
TARGET = "any_outbound_event_by_24m"
AGE_CURVE = "age_distance_from_27_squared"
BASELINE_FEATURES = list(V2_CORE_FEATURES) + [AGE_CURVE]
RANDOM_SEED = 20260901
BOOTSTRAPS = 5000
C_GRID = (0.01, 0.1, 1.0, 10.0)
TOP_CAPACITY_SHARE = 0.20

# Frozen before outcome evaluation. R5 is deliberately diagnostic-only because
# its source gate covers less than 85% of the development evaluation cohort.
VARIANT_BLOCKS = {
    "R1_resource_level": ["payroll_log_eur", "payroll_league_percentile"],
    "R2_recruitment_intensity": ["transfer_spend_to_payroll", "transfer_income_to_payroll"],
    "R3_squad_context": ["squad_average_age", "squad_foreign_share"],
    "R4_combined_core_context": [
        "payroll_log_eur", "payroll_league_percentile",
        "transfer_spend_to_payroll", "transfer_income_to_payroll",
        "squad_average_age", "squad_foreign_share",
    ],
    "R5_wage_structure_diagnostic": ["wage_structure_gini", "wage_structure_top5_share"],
}
DEPLOYMENT_ELIGIBLE = {
    "R1_resource_level": True,
    "R2_recruitment_intensity": True,
    "R3_squad_context": True,
    "R4_combined_core_context": True,
    "R5_wage_structure_diagnostic": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: str) -> int:
    token = "|".join(parts).encode()
    return (RANDOM_SEED + int(hashlib.sha256(token).hexdigest()[:8], 16)) % (2**32 - 1)


def adaptive_ece(actual: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    frame = pd.DataFrame({"actual": actual, "prediction": probability}).sort_values("prediction")
    groups = np.array_split(np.arange(len(frame)), min(bins, len(frame)))
    return float(sum(len(group) * abs(frame.iloc[group].actual.mean() - frame.iloc[group].prediction.mean()) for group in groups) / len(frame))


def tune(train: pd.DataFrame, validation: pd.DataFrame, features: list[str]):
    pre = fit_preprocessor(train, features)
    x_train, x_validation = pre.transform(train), pre.transform(validation)
    y_train = pd.to_numeric(train[TARGET]).to_numpy(int)
    y_validation = pd.to_numeric(validation[TARGET]).to_numpy(int)
    candidates = []
    for c_value in C_GRID:
        model = LogisticRegression(
            C=c_value, penalty="l2", solver="liblinear", max_iter=3000,
            random_state=RANDOM_SEED,
        ).fit(x_train, y_train)
        prediction = np.clip(model.predict_proba(x_validation)[:, 1], 1e-6, 1 - 1e-6)
        candidates.append((brier_score_loss(y_validation, prediction), c_value, model))
    _, c_value, model = min(candidates, key=lambda item: item[0])
    return pre, model, float(c_value)


def load_data() -> pd.DataFrame:
    data = phase4.add_age_curve(phase2.load_endpoint_frames()["any_outbound_24m"])
    audit = pd.read_csv(CONTEXT / "extension_prior_season_feature_coverage.csv", low_memory=False)
    panel = pd.read_csv(CONTEXT / "canonical_club_season_financial_context.csv", low_memory=False)
    feature_columns = sorted({feature for block in VARIANT_BLOCKS.values() for feature in block})
    bridge = audit[["capology_extension_event_id", "canonical_club_season_id", "missing_reason"]].merge(
        panel[["canonical_club_season_id"] + feature_columns],
        on="canonical_club_season_id", how="left", validate="many_to_one",
    )
    return data.merge(bridge, on="capology_extension_event_id", how="left", validate="one_to_one")


def frozen_hashes() -> dict[str, str]:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    return {item["path"]: sha256(ROOT / item["path"]) for item in freeze["files"]}


def reconstruct_locked_baseline(data: pd.DataFrame) -> dict:
    generated = []
    for origin, train_end, validation_year, evaluation_year in phase2.ORIGINS:
        train = data.loc[data.signed_year.le(train_end)].copy()
        validation = data.loc[data.signed_year.eq(validation_year)].copy()
        evaluation = data.loc[data.signed_year.eq(evaluation_year)].sort_values("capology_extension_event_id").copy()
        pre, model, _ = tune(train, validation, BASELINE_FEATURES)
        prediction = np.clip(model.predict_proba(pre.transform(evaluation))[:, 1], 1e-6, 1 - 1e-6)
        generated.extend(zip(evaluation.capology_extension_event_id, prediction))
    generated = pd.DataFrame(generated, columns=["capology_extension_event_id", "reconstructed_prediction"])
    locked = pd.read_csv(PHASE4 / "continuity_selected_predictions.csv", usecols=["capology_extension_event_id", "prediction"])
    pair = locked.merge(generated, on="capology_extension_event_id", validate="one_to_one")
    return {
        "rows": int(len(pair)),
        "max_absolute_prediction_difference": float((pair.prediction - pair.reconstructed_prediction).abs().max()),
        "exact_within_1e_12": bool(np.allclose(pair.prediction, pair.reconstructed_prediction, atol=1e-12, rtol=0)),
    }


def cluster_interval(frame: pd.DataFrame, confidence: float, label: str) -> tuple[float, float]:
    clusters = frame.groupby("canonical_player_id").loss_improvement.agg(["sum", "size"])
    sums, sizes = clusters["sum"].to_numpy(float), clusters["size"].to_numpy(float)
    rng = np.random.default_rng(stable_seed(label, str(confidence)))
    draws = []
    for _ in range(BOOTSTRAPS // 100):
        indices = rng.integers(0, len(clusters), size=(100, len(clusters)))
        draws.extend((sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)).tolist())
    alpha = 1 - confidence
    return float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def safe_auc(actual: pd.Series, prediction: pd.Series) -> float:
    return float(roc_auc_score(actual, prediction)) if actual.nunique() == 2 else np.nan


def run_variants(data: pd.DataFrame):
    prediction_rows, origin_rows, coverage_rows, support_profiles = [], [], [], []
    for variant, added_features in VARIANT_BLOCKS.items():
        candidate_features = BASELINE_FEATURES + added_features
        for origin, train_end, validation_year, evaluation_year in phase2.ORIGINS:
            train_all = data.loc[data.signed_year.le(train_end)].copy()
            validation_all = data.loc[data.signed_year.eq(validation_year)].copy()
            evaluation_all = data.loc[data.signed_year.eq(evaluation_year)].sort_values("capology_extension_event_id").copy()
            train = train_all.loc[train_all[added_features].notna().all(axis=1)].copy()
            validation = validation_all.loc[validation_all[added_features].notna().all(axis=1)].copy()
            evaluation = evaluation_all.loc[evaluation_all[added_features].notna().all(axis=1)].copy()
            if min(train[TARGET].nunique(), validation[TARGET].nunique(), evaluation[TARGET].nunique()) < 2:
                raise RuntimeError(f"Single-class split for {variant} {origin}")

            base_pre, base_model, base_c = tune(train, validation, BASELINE_FEATURES)
            cand_pre, cand_model, cand_c = tune(train, validation, candidate_features)
            base_prediction = np.clip(base_model.predict_proba(base_pre.transform(evaluation))[:, 1], 1e-6, 1 - 1e-6)
            cand_prediction = np.clip(cand_model.predict_proba(cand_pre.transform(evaluation))[:, 1], 1e-6, 1 - 1e-6)
            actual = pd.to_numeric(evaluation[TARGET]).to_numpy(int)
            chronology = float(pd.to_numeric(pd.concat([train, validation], ignore_index=True)[TARGET]).mean())
            profile, _ = phase3.support_profile(pd.concat([train, validation], ignore_index=True), candidate_features)
            support = phase3.assess_support(evaluation, candidate_features, profile).reset_index(drop=True)
            support_profiles.append({"variant": variant, "origin_id": origin, "features": candidate_features, **profile})

            unavailable = len(evaluation_all) - len(evaluation)
            refused = unavailable + int(support.support_status.eq("refused").sum())
            coverage_rows.append({
                "variant": variant, "origin_id": origin, "evaluation_year": evaluation_year,
                "evaluation_rows_total": len(evaluation_all), "feature_eligible_rows": len(evaluation),
                "feature_unavailable_rows": unavailable, "feature_coverage_rate": len(evaluation) / len(evaluation_all),
                "support_refused_rows_among_eligible": int(support.support_status.eq("refused").sum()),
                "operational_refused_rows": refused, "operational_refusal_rate": refused / len(evaluation_all),
            })
            origin_rows.append({
                "variant": variant, "origin_id": origin, "evaluation_year": evaluation_year,
                "train_rows_matched": len(train), "validation_rows_matched": len(validation),
                "evaluation_rows_matched": len(evaluation), "baseline_selected_C": base_c,
                "candidate_selected_C": cand_c, "matched_baseline_brier": brier_score_loss(actual, base_prediction),
                "candidate_brier": brier_score_loss(actual, cand_prediction),
                "brier_improvement": brier_score_loss(actual, base_prediction) - brier_score_loss(actual, cand_prediction),
                "matched_baseline_auc": safe_auc(pd.Series(actual), pd.Series(base_prediction)),
                "candidate_auc": safe_auc(pd.Series(actual), pd.Series(cand_prediction)),
                "candidate_ece": adaptive_ece(actual, cand_prediction),
            })
            for i, (_, row) in enumerate(evaluation.reset_index(drop=True).iterrows()):
                prediction_rows.append({
                    "variant": variant, "origin_id": origin, "evaluation_year": evaluation_year,
                    "capology_extension_event_id": row.capology_extension_event_id,
                    "canonical_player_id": row.canonical_player_id, "canonical_position": row.canonical_position,
                    "league": row.league, "age_band": row.age_band, "actual": int(actual[i]),
                    "candidate_prediction": float(cand_prediction[i]), "matched_baseline_prediction": float(base_prediction[i]),
                    "chronology_prediction": chronology,
                    "candidate_brier_loss": float((actual[i] - cand_prediction[i]) ** 2),
                    "matched_baseline_brier_loss": float((actual[i] - base_prediction[i]) ** 2),
                    "chronology_brier_loss": float((actual[i] - chronology) ** 2),
                    "loss_improvement": float((actual[i] - base_prediction[i]) ** 2 - (actual[i] - cand_prediction[i]) ** 2),
                    "support_status": support.iloc[i].support_status,
                    "support_reasons": support.iloc[i].support_reasons,
                    "support_distance": support.iloc[i].support_distance,
                })
    return pd.DataFrame(prediction_rows), pd.DataFrame(origin_rows), pd.DataFrame(coverage_rows), support_profiles


def summarize(predictions: pd.DataFrame, origins: pd.DataFrame, coverage: pd.DataFrame):
    comparisons, subgroups, capacity = [], [], []
    bonferroni_confidence = 1 - 0.05 / len(VARIANT_BLOCKS)
    for variant, frame in predictions.groupby("variant", sort=False):
        ci95 = cluster_interval(frame, 0.95, variant)
        ci_adjusted = cluster_interval(frame, bonferroni_confidence, variant)
        origin = origins.loc[origins.variant.eq(variant)]
        cover = coverage.loc[coverage.variant.eq(variant)]
        overall_refusal = cover.operational_refused_rows.sum() / cover.evaluation_rows_total.sum()
        candidate_brier = float(frame.candidate_brier_loss.mean())
        base_brier = float(frame.matched_baseline_brier_loss.mean())
        comparisons.append({
            "variant": variant, "deployment_eligible_by_design": DEPLOYMENT_ELIGIBLE[variant],
            "matched_evaluation_rows": len(frame), "player_clusters": frame.canonical_player_id.nunique(),
            "feature_coverage_rate": cover.feature_eligible_rows.sum() / cover.evaluation_rows_total.sum(),
            "operational_refusal_rate": overall_refusal,
            "matched_baseline_brier": base_brier, "candidate_brier": candidate_brier,
            "pooled_brier_improvement": base_brier - candidate_brier,
            "ci_low_95": ci95[0], "ci_high_95": ci95[1],
            "ci_low_bonferroni_99": ci_adjusted[0], "ci_high_bonferroni_99": ci_adjusted[1],
            "origin_wins": int(origin.brier_improvement.gt(0).sum()),
            "candidate_auc": safe_auc(frame.actual, frame.candidate_prediction),
            "matched_baseline_auc": safe_auc(frame.actual, frame.matched_baseline_prediction),
            "candidate_average_precision": float(average_precision_score(frame.actual, frame.candidate_prediction)),
            "candidate_ece": adaptive_ece(frame.actual.to_numpy(), frame.candidate_prediction.to_numpy()),
        })
        for group_type, column in [("league", "league"), ("position", "canonical_position"), ("age", "age_band")]:
            for group, chunk in frame.groupby(column, dropna=False):
                rows, events = len(chunk), int(chunk.actual.sum())
                nonevents, n_origins = rows - events, chunk.origin_id.nunique()
                candidate_brier = float(chunk.candidate_brier_loss.mean())
                base_brier = float(chunk.matched_baseline_brier_loss.mean())
                chronology_brier = float(chunk.chronology_brier_loss.mean())
                auc = safe_auc(chunk.actual, chunk.candidate_prediction)
                gap = float(abs(chunk.actual.mean() - chunk.candidate_prediction.mean()))
                sufficient = rows >= MIN_SUBGROUP_ROWS and events >= MIN_SUBGROUP_EVENTS and nonevents >= MIN_SUBGROUP_NONEVENTS and n_origins >= MIN_SUBGROUP_ORIGINS
                supported = bool(sufficient and auc >= MIN_SUPPORTED_AUC and gap <= MAX_SUPPORTED_CALIBRATION_GAP and candidate_brier < chronology_brier)
                subgroups.append({
                    "variant": variant, "group_type": group_type, "group": str(group), "rows": rows,
                    "events": events, "nonevents": nonevents, "origins": n_origins,
                    "candidate_auc": auc, "candidate_brier": candidate_brier,
                    "matched_baseline_brier": base_brier, "chronology_brier": chronology_brier,
                    "brier_improvement_vs_matched_baseline": base_brier - candidate_brier,
                    "brier_skill_vs_chronology": chronology_brier - candidate_brier,
                    "calibration_gap": gap,
                    "reliability_status": "supported" if supported else "limited_reliability" if sufficient else "insufficient_evidence",
                })
        for origin_id, chunk in frame.groupby("origin_id"):
            capacity_n = max(1, int(math.ceil(len(chunk) * TOP_CAPACITY_SHARE)))
            cand = chunk.nlargest(capacity_n, "candidate_prediction")
            base = chunk.nlargest(capacity_n, "matched_baseline_prediction")
            capacity.append({
                "variant": variant, "origin_id": origin_id, "eligible_rows": len(chunk),
                "review_capacity_rows": capacity_n, "candidate_events_identified": int(cand.actual.sum()),
                "matched_baseline_events_identified": int(base.actual.sum()),
                "event_difference": int(cand.actual.sum() - base.actual.sum()),
                "candidate_hit_rate": float(cand.actual.mean()), "matched_baseline_hit_rate": float(base.actual.mean()),
            })
    comparisons = pd.DataFrame(comparisons)
    subgroups = pd.DataFrame(subgroups)
    decisions = []
    for _, row in comparisons.iterrows():
        league = subgroups.loc[(subgroups.variant.eq(row.variant)) & subgroups.group_type.eq("league")]
        gates = {
            "deployment_eligible_by_design": bool(row.deployment_eligible_by_design),
            "positive_bonferroni_cluster_ci": bool(row.ci_low_bonferroni_99 > 0),
            "at_least_3_of_4_origin_wins": bool(row.origin_wins >= 3),
            "refusal_at_or_below_15pct": bool(row.operational_refusal_rate <= MAX_ACCEPTABLE_REFUSAL_RATE),
            "ece_at_or_below_0_08": bool(row.candidate_ece <= MAX_ACCEPTABLE_ECE),
            "all_big_five_leagues_supported": bool(len(league) == 5 and league.reliability_status.eq("supported").all()),
        }
        advance = all(gates.values())
        decisions.append({
            "variant": row.variant, **gates, "advance_gate_passed": advance,
            "decision": "advance_to_future_final_holdout_only" if advance else "reject_as_continuity_repair_candidate",
            "deployment_status": "not_deployed",
        })
    return comparisons, subgroups, pd.DataFrame(capacity), pd.DataFrame(decisions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    frozen_before = frozen_hashes()
    data = load_data()
    reconstruction = reconstruct_locked_baseline(data)
    predictions, origins, coverage, support_profiles = run_variants(data)
    comparisons, subgroups, capacity, decisions = summarize(predictions, origins, coverage)
    frozen_after = frozen_hashes()

    checks = [
        {"check": "locked_phase4_baseline_reconstructed", "passed": reconstruction["exact_within_1e_12"], "observed": reconstruction["max_absolute_prediction_difference"], "expected": "<=1e-12"},
        {"check": "four_rolling_origins", "passed": origins.origin_id.nunique() == 4, "observed": origins.origin_id.nunique(), "expected": 4},
        {"check": "five_predeclared_variants", "passed": origins.variant.nunique() == 5, "observed": origins.variant.nunique(), "expected": 5},
        {"check": "no_evaluation_after_2023", "passed": predictions.evaluation_year.max() == 2023, "observed": predictions.evaluation_year.max(), "expected": 2023},
        {"check": "no_missing_context_scored", "passed": predictions.candidate_prediction.notna().all(), "observed": int(predictions.candidate_prediction.isna().sum()), "expected": 0},
        {"check": "matched_pairing_complete", "passed": predictions.matched_baseline_prediction.notna().all(), "observed": int(predictions.matched_baseline_prediction.isna().sum()), "expected": 0},
        {"check": "frozen_v1_byte_identical", "passed": frozen_before == frozen_after, "observed": sum(frozen_before[k] == frozen_after[k] for k in frozen_before), "expected": len(frozen_before)},
        {"check": "no_candidate_passes_silently", "passed": decisions.deployment_status.eq("not_deployed").all(), "observed": decisions.deployment_status.unique().tolist(), "expected": ["not_deployed"]},
    ]
    pd.DataFrame(checks).to_csv(output / "build_checks.csv", index=False)
    predictions.to_csv(output / "paired_predictions.csv", index=False)
    origins.to_csv(output / "origin_metrics.csv", index=False)
    coverage.to_csv(output / "coverage_by_origin.csv", index=False)
    comparisons.to_csv(output / "candidate_comparison.csv", index=False)
    subgroups.to_csv(output / "subgroup_metrics.csv", index=False)
    capacity.to_csv(output / "decision_capacity_lift.csv", index=False)
    decisions.to_csv(output / "selection_decision.csv", index=False)
    (output / "support_profiles.json").write_text(json.dumps(support_profiles, indent=2, sort_keys=True) + "\n")
    preregistration = {
        "endpoint": "any_outbound_24m", "target": TARGET,
        "origins": [list(item) for item in phase2.ORIGINS], "baseline_features": BASELINE_FEATURES,
        "variant_blocks": VARIANT_BLOCKS, "candidate_family_size": len(VARIANT_BLOCKS),
        "familywise_interval": "Bonferroni 99% player-cluster bootstrap interval",
        "missing_context_policy": "unavailable; never imputed and never scored",
        "comparison_policy": "baseline retrained on the identical eligible train/validation/evaluation rows",
        "final_holdout_policy": "2024 and later are not evaluated",
        "advance_gate": "positive adjusted CI; >=3/4 origin wins; <=15% refusal; ECE<=0.08; all Big Five leagues supported",
    }
    (output / "preregistration.json").write_text(json.dumps(preregistration, indent=2, sort_keys=True) + "\n")
    sources = [
        CONTEXT / "canonical_club_season_financial_context.csv",
        CONTEXT / "extension_prior_season_feature_coverage.csv",
        PHASE4 / "continuity_selected_predictions.csv",
        FREEZE,
    ]
    pd.DataFrame([{"source": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size} for path in sources]).to_csv(output / "source_manifest.csv", index=False)
    advanced = decisions.loc[decisions.advance_gate_passed, "variant"].tolist()
    summary = {
        "development_evaluation_years": [2020, 2021, 2022, 2023],
        "locked_baseline_reconstruction": reconstruction,
        "variants_evaluated": len(VARIANT_BLOCKS), "advanced_variants": advanced,
        "final_holdout_opened": False, "deployment_changed": False,
        "all_build_checks_passed": bool(pd.DataFrame(checks).passed.all()),
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    readme = """# Engine V2 club financial context rolling-origin experiment

Development-only continuity repair test. Five feature blocks were evaluated on
the four already-designated 2020-2023 rolling origins against the locked
Phase-4 baseline. Every comparison retrains that baseline on the candidate's
identical eligible rows. Missing club context is unavailable, not imputed.

The 2024+ final holdout was not opened. No model was serialized or deployed.
See `selection_decision.csv` for the mechanical gate and
`candidate_comparison.csv` / `subgroup_metrics.csv` for the evidence.
"""
    (output / "README.md").write_text(readme)
    files = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "output_manifest.csv":
            files.append({"output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(files).to_csv(output / "output_manifest.csv", index=False)
    print(json.dumps(summary, indent=2))
    print(comparisons.to_string(index=False))
    print(decisions.to_string(index=False))


if __name__ == "__main__":
    main()

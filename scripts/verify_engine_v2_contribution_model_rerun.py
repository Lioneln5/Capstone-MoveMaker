"""Independent verification for repaired contribution model Run 2."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "models") not in sys.path:
    sys.path.insert(0, str(ROOT / "models"))

from models.engine_v2.complete_contribution_target_contract import (  # noqa: E402
    AVAILABLE_FIELD,
    TARGET_FIELD,
    add_complete_contribution_target,
    add_original_league_schedule_coverage,
)
from models.engine_v2.contribution_model_development_contract import (  # noqa: E402
    BOOTSTRAP_REPETITIONS,
    CANDIDATE_FEATURES,
    LOGISTIC_C_GRID,
    OPTIONAL_EXTENSION_CANDIDATE,
    ORIGINS,
    PRIMARY_CANDIDATE,
    PRIMARY_COMPARISONS,
    RANDOM_SEED,
)
from mixed_type_preprocessor import fit_preprocessor  # noqa: E402


MASTER = ROOT / "Data" / "processed" / "contract_extension_integration" / "extension_modeling_master.csv"
GAMES = ROOT / "Data" / "processed" / "transfermarkt_clean" / "tables" / "games_clean.csv"
OUTPUT = ROOT / "Data" / "processed" / "engine_v2_contribution_model_rerun"
REGENERABLE_DEPLOYMENT_REPORTS = {
    "Data/processed/deployment_models/live_request_integrity.csv",
    "Data/processed/deployment_models/live_request_integrity.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().str.casefold().isin(
        {"true", "1", "yes"}
    )


def stable_seed(*parts: str) -> int:
    token = "|".join(parts).encode("utf-8")
    return (RANDOM_SEED + int(hashlib.sha256(token).hexdigest()[:8], 16)) % (2**32 - 1)


def prepare_frame() -> pd.DataFrame:
    frame = pd.read_csv(MASTER, low_memory=False)
    games = pd.read_csv(GAMES, low_memory=False)
    frame["signed_date"] = pd.to_datetime(frame["signed_date"], errors="coerce")
    frame["expiration_date"] = pd.to_datetime(frame["expiration_date"], errors="coerce")
    frame["signed_year"] = frame["signed_date"].dt.year.astype("Int64")
    frame = frame.loc[frame["signed_year"].le(2023)].copy()
    frame = add_complete_contribution_target(
        add_original_league_schedule_coverage(frame, games)
    )
    market = pd.to_numeric(frame["at_signing_market_value_eur"], errors="coerce")
    frame["log_market_value_at_signing"] = np.log1p(market.clip(lower=0))
    frame["age_band"] = pd.cut(
        pd.to_numeric(frame["age_at_signing"], errors="coerce"),
        [-np.inf, 21, 25, 29, 33, np.inf],
        labels=["<=21", "22-25", "26-29", "30-33", "34+"],
    )
    covers = frame["expiration_date"].ge(frame["signed_date"] + pd.Timedelta(days=730))
    pre_share = pd.to_numeric(
        frame["pre365_same_club_all_competition_opportunity_share"], errors="coerce"
    )
    mask = (
        frame["canonical_player_id"].notna()
        & covers
        & as_bool(frame["pre365_window_evidence_eligible_10_club_games"])
        & pre_share.notna()
        & frame[AVAILABLE_FIELD]
        & pd.to_numeric(frame[TARGET_FIELD], errors="coerce").notna()
    )
    return frame.loc[mask].copy()


def independent_prediction(
    tuning_train: pd.DataFrame,
    validation: pd.DataFrame,
    final_fit: pd.DataFrame,
    evaluation: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, float | None]:
    if not features:
        probability = float(pd.to_numeric(final_fit[TARGET_FIELD]).mean())
        return np.repeat(np.clip(probability, 1e-6, 1 - 1e-6), len(evaluation)), None

    tuning_pre = fit_preprocessor(tuning_train, features)
    x_train = tuning_pre.transform(tuning_train)
    x_validation = tuning_pre.transform(validation)
    y_train = pd.to_numeric(tuning_train[TARGET_FIELD]).to_numpy(int)
    y_validation = pd.to_numeric(validation[TARGET_FIELD]).to_numpy(int)
    trials: list[tuple[float, float, float]] = []
    for c_value in LOGISTIC_C_GRID:
        model = LogisticRegression(
            C=c_value,
            penalty="l2",
            solver="liblinear",
            max_iter=3000,
            random_state=RANDOM_SEED,
        ).fit(x_train, y_train)
        probability = np.clip(model.predict_proba(x_validation)[:, 1], 1e-6, 1 - 1e-6)
        trials.append(
            (
                float(brier_score_loss(y_validation, probability)),
                float(log_loss(y_validation, probability, labels=[0, 1])),
                c_value,
            )
        )
    _, _, selected_c = min(trials)
    pre = fit_preprocessor(final_fit, features)
    model = LogisticRegression(
        C=selected_c,
        penalty="l2",
        solver="liblinear",
        max_iter=3000,
        random_state=RANDOM_SEED,
    ).fit(pre.transform(final_fit), pd.to_numeric(final_fit[TARGET_FIELD]).to_numpy(int))
    prediction = np.clip(
        model.predict_proba(pre.transform(evaluation))[:, 1], 1e-6, 1 - 1e-6
    )
    return prediction, selected_c


def main() -> None:
    checks: list[dict[str, object]] = []

    def add(check: str, actual: object, expected: object, passed: bool, note: str) -> None:
        checks.append(
            {"check": check, "actual": actual, "expected": expected, "passed": bool(passed), "note": note}
        )

    required = {
        "README.md",
        "build_checks.csv",
        "candidate_comparisons.csv",
        "coefficient_stability.csv",
        "cohort_by_origin.csv",
        "deployment_unchanged_manifest.csv",
        "feature_missingness_by_origin.csv",
        "origin_metrics.csv",
        "out_of_time_predictions.csv",
        "output_manifest.csv",
        "player_disjoint_sensitivity.csv",
        "preregistration.json",
        "run_summary.json",
        "selection_decision.csv",
        "source_manifest.csv",
        "subgroup_metrics.csv",
    }
    actual_files = {path.name for path in OUTPUT.iterdir() if path.is_file()}
    allowed_extra = {"independent_verification.csv", "independent_verification.json"}
    add("required_outputs_exact", sorted(actual_files), sorted(required), required.issubset(actual_files) and not (actual_files - required - allowed_extra), "Evidence package contains only declared outputs.")

    manifest = pd.read_csv(OUTPUT / "output_manifest.csv")
    manifest_ok = True
    for item in manifest.itertuples(index=False):
        path = OUTPUT / item.output_file
        manifest_ok &= path.exists() and path.stat().st_size == int(item.bytes) and sha256(path) == item.sha256
    add("output_manifest_exact", int(manifest_ok), 1, manifest_ok, "Every evidence output matches its recorded hash and byte count.")

    sources = pd.read_csv(OUTPUT / "source_manifest.csv")
    source_ok = True
    for item in sources.itertuples(index=False):
        path = ROOT / item.source
        source_ok &= path.exists() and path.stat().st_size == int(item.bytes) and sha256(path) == item.sha256
    add("source_manifest_exact", int(source_ok), 1, source_ok, "Every model and data input matches the Run 2 source record.")

    data = prepare_frame()
    evaluation = data.loc[data["signed_year"].between(2020, 2023)]
    add("repaired_evaluation_cohort_exact", len(evaluation), 878, len(evaluation) == 878, "Independent cohort reconstruction matches Run 1.")
    add("repaired_positive_count_exact", int(pd.to_numeric(evaluation[TARGET_FIELD]).sum()), 438, int(pd.to_numeric(evaluation[TARGET_FIELD]).sum()) == 438, "Independent target reconstruction has 438 positives.")

    saved_predictions = pd.read_csv(OUTPUT / "out_of_time_predictions.csv")
    saved_metrics = pd.read_csv(OUTPUT / "origin_metrics.csv")
    maximum_prediction_error = 0.0
    selected_c_exact = True
    player_disjoint_exact = True
    reconstructed_rows = 0
    for protocol in ("player_disjoint", "ordinary_temporal_sensitivity"):
        for origin, train_end, validation_year, evaluation_year in ORIGINS:
            base_train = data.loc[data["signed_year"].le(train_end)].copy()
            validation = data.loc[data["signed_year"].eq(validation_year)].copy()
            evaluation = data.loc[data["signed_year"].eq(evaluation_year)].sort_values("capology_extension_event_id").copy()
            tuning_train = base_train
            final_fit = pd.concat([base_train, validation], ignore_index=True)
            if protocol == "player_disjoint":
                tuning_train = tuning_train.loc[
                    ~tuning_train["canonical_player_id"].isin(set(validation["canonical_player_id"]))
                ].copy()
                final_fit = final_fit.loc[
                    ~final_fit["canonical_player_id"].isin(set(evaluation["canonical_player_id"]))
                ].copy()
                player_disjoint_exact &= not (
                    set(tuning_train["canonical_player_id"])
                    & set(validation["canonical_player_id"])
                )
                player_disjoint_exact &= not (
                    set(final_fit["canonical_player_id"])
                    & set(evaluation["canonical_player_id"])
                )

            for candidate, features in CANDIDATE_FEATURES.items():
                prediction, selected_c = independent_prediction(
                    tuning_train, validation, final_fit, evaluation, list(features)
                )
                saved = saved_predictions.loc[
                    saved_predictions["protocol"].eq(protocol)
                    & saved_predictions["origin_id"].eq(origin)
                    & saved_predictions["candidate"].eq(candidate)
                ].sort_values("capology_extension_event_id")
                if len(saved) != len(evaluation):
                    maximum_prediction_error = np.inf
                    continue
                maximum_prediction_error = max(
                    maximum_prediction_error,
                    float(np.max(np.abs(prediction - saved["prediction"].to_numpy(float)))),
                )
                metric_row = saved_metrics.loc[
                    saved_metrics["protocol"].eq(protocol)
                    & saved_metrics["origin_id"].eq(origin)
                    & saved_metrics["candidate"].eq(candidate)
                ].iloc[0]
                if selected_c is None:
                    selected_c_exact &= pd.isna(metric_row.selected_C)
                else:
                    selected_c_exact &= np.isclose(selected_c, float(metric_row.selected_C))
                reconstructed_rows += len(saved)
    add("all_predictions_refit_exact", maximum_prediction_error, "<=1e-12", maximum_prediction_error <= 1e-12, "Independent refits reproduce both protocols and every candidate.")
    add("selected_hyperparameters_exact", int(selected_c_exact), 1, selected_c_exact, "Every preceding-year validation choice reproduces.")
    add("player_disjointness_independent", int(player_disjoint_exact), 1, player_disjoint_exact, "No evaluation identity appears in fitting rows and no validation identity appears in tuning rows.")
    add("prediction_row_count_exact", reconstructed_rows, 7024, reconstructed_rows == 7024, "878 events × four candidates × two protocols are reconstructed.")

    loss = np.square(saved_predictions["actual"] - saved_predictions["prediction"])
    add("saved_brier_loss_exact", float(np.max(np.abs(loss - saved_predictions["brier_loss"]))), "<=1e-14", np.allclose(loss, saved_predictions["brier_loss"], atol=1e-14), "Every saved row-level loss reconstructs.")

    saved_comparisons = pd.read_csv(OUTPUT / "candidate_comparisons.csv")
    comparison_exact = True
    for row in saved_comparisons.itertuples(index=False):
        sample = saved_predictions.loc[saved_predictions["protocol"].eq(row.protocol)]
        keys = ["origin_id", "evaluation_year", "capology_extension_event_id", "canonical_player_id"]
        left = sample.loc[sample["candidate"].eq(row.baseline), keys + ["brier_loss"]].rename(columns={"brier_loss": "baseline_loss"})
        right = sample.loc[sample["candidate"].eq(row.candidate), keys + ["brier_loss"]].rename(columns={"brier_loss": "candidate_loss"})
        paired = left.merge(right, on=keys, validate="one_to_one")
        paired["improvement"] = paired["baseline_loss"] - paired["candidate_loss"]
        clusters = paired.groupby("canonical_player_id")["improvement"].agg(["sum", "size"])
        sums, sizes = clusters["sum"].to_numpy(float), clusters["size"].to_numpy(float)
        rng = np.random.default_rng(stable_seed(row.protocol, row.baseline, row.candidate))
        draws: list[float] = []
        for start in range(0, BOOTSTRAP_REPETITIONS, 250):
            count = min(250, BOOTSTRAP_REPETITIONS - start)
            indices = rng.integers(0, len(clusters), size=(count, len(clusters)))
            draws.extend((sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)).tolist())
        origin_wins = int(paired.groupby("origin_id")["improvement"].mean().gt(0).sum())
        comparison_exact &= np.isclose(paired["improvement"].mean(), row.pooled_brier_improvement, atol=1e-14)
        comparison_exact &= np.isclose(np.quantile(draws, 0.025), row.ci_lower_95, atol=1e-14)
        comparison_exact &= np.isclose(np.quantile(draws, 0.975), row.ci_upper_95, atol=1e-14)
        comparison_exact &= origin_wins == row.origin_wins
    add("clustered_comparisons_exact", int(comparison_exact), 1, comparison_exact, "All pooled deltas, origin wins, and player-cluster intervals reconstruct.")

    strict = saved_comparisons.loc[saved_comparisons["protocol"].eq("player_disjoint")]
    primary = strict.loc[
        strict["baseline"].eq("B0_chronology_prevalence")
        & strict["candidate"].eq(PRIMARY_CANDIDATE)
    ].iloc[0]
    profile = strict.loc[
        strict["baseline"].eq("B1_market_profile")
        & strict["candidate"].eq(PRIMARY_CANDIDATE)
    ].iloc[0]
    extension = strict.loc[
        strict["baseline"].eq(PRIMARY_CANDIDATE)
        & strict["candidate"].eq(OPTIONAL_EXTENSION_CANDIDATE)
    ].iloc[0]
    primary_passes = bool(
        primary.ci_lower_95 > 0
        and primary.origin_wins >= 3
        and profile.pooled_brier_improvement > 0
        and primary.candidate_roc_auc >= 0.70
        and primary.candidate_calibration_gap <= 0.05
    )
    extension_replaces = bool(
        primary_passes and extension.ci_lower_95 > 0 and extension.origin_wins >= 3
    )
    decision = pd.read_csv(OUTPUT / "selection_decision.csv").iloc[0]
    add("primary_gate_independent", int(primary_passes), 1, primary_passes and bool(decision.primary_signal_gate_passed), "The compact core passes every predeclared development gate.")
    add("prior_volume_not_selected", int(extension_replaces), 0, not extension_replaces and decision.selected_development_candidate == PRIMARY_CANDIDATE, "Sparse prior-volume fields do not establish incremental Brier skill.")
    add("decision_research_only", decision.deployment_status, "not_deployed", decision.deployment_status == "not_deployed", "Passing Run 2 cannot authorize deployment.")

    subgroups = pd.read_csv(OUTPUT / "subgroup_metrics.csv")
    broad_supported = subgroups.loc[subgroups["group_type"].isin(["league", "position"]), "evidence_status"].eq("supported").all()
    add("broad_league_position_diagnostic", int(broad_supported), 1, broad_supported, "All five leagues and four broad positions beat chronology with sufficient evidence in the preliminary audit.")
    age_34 = subgroups.loc[(subgroups["group_type"].eq("age")) & (subgroups["group"].eq("34+"))].iloc[0]
    add("small_age_group_not_overclaimed", age_34.evidence_status, "insufficient_evidence", age_34.evidence_status == "insufficient_evidence", "The 42-row age-34+ view is explicitly limited.")

    deployment = pd.read_csv(OUTPUT / "deployment_unchanged_manifest.csv")
    deployment_exact = True
    for item in deployment.itertuples(index=False):
        if item.artifact in REGENERABLE_DEPLOYMENT_REPORTS:
            continue
        path = ROOT / item.artifact
        deployment_exact &= path.exists() and sha256(path) == item.before_sha256 == item.after_sha256
    add("deployment_model_files_unchanged", int(deployment_exact), 1, deployment_exact, "Every protected deployment artifact remains at its Run 2 pre-run hash; the two regenerable integrity reports are excluded from artifact immutability.")
    add("no_serialized_candidate", len(list(OUTPUT.rglob("*.joblib"))), 0, not list(OUTPUT.rglob("*.joblib")), "Run 2 writes no model binary.")

    summary = json.loads((OUTPUT / "run_summary.json").read_text(encoding="utf-8"))
    add("final_holdout_not_opened", summary.get("final_holdout_evaluated"), False, summary.get("final_holdout_evaluated") is False, "No 2024+ extension outcome is evaluated.")
    add("development_only_decision", summary.get("run2_decision"), "advance_to_reliability_audit", summary.get("run2_decision") == "advance_to_reliability_audit", "Run 2 advances only to a new reliability audit.")

    docs = (ROOT / "docs" / "ENGINE_V2_CONTRIBUTION_MODEL_RUN2.md").read_text(encoding="utf-8")
    audit = (ROOT / "docs" / "ENGINE_V2_CRITICAL_ISSUES.md").read_text(encoding="utf-8")
    registry = (ROOT / "models" / "README.md").read_text(encoding="utf-8")
    doc_ok = (
        "Run 2 development result" in docs
        and "MODEL RERUN 2 COMPLETE" in audit
        and "Repaired contribution model rerun" in registry
    )
    add("durable_documentation_updated", int(doc_ok), 1, doc_ok, "Run 2 result and its non-deployment boundary are registered.")

    result = pd.DataFrame(checks)
    result.to_csv(OUTPUT / "independent_verification.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    verification = {
        "checks_passed": int(result["passed"].sum()),
        "checks_total": len(result),
        "all_checks_passed": bool(result["passed"].all()),
        "independent_model_refits": 24,
        "final_holdout_evaluated": False,
        "deployment_changed": False,
    }
    (OUTPUT / "independent_verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not result["passed"].all():
        failed = result.loc[~result["passed"], ["check", "actual", "expected"]]
        raise RuntimeError("Run 2 independent verification failed:\n" + failed.to_string(index=False))
    print(f"All {len(result)}/{len(result)} Run 2 independent checks pass.")


if __name__ == "__main__":
    main()

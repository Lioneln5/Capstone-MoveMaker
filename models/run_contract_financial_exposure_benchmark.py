"""Phase-4 financial exposure calculations and historical peer benchmarks.

The module keeps exact input-driven calculations separate from empirical
benchmarks. It never treats unavailable taxes, agent fees, signing fees,
clauses, options, or unreported bonuses as zero.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "models") not in sys.path:
    sys.path.insert(0, str(ROOT / "models"))
from mixed_type_preprocessor import fit_preprocessor
import run_extension_opportunity_diagnostic as phase1


SOURCE = phase1.SOURCE
UPSTREAM_VERIFY = ROOT / "Data" / "processed" / "contract_extension_integration" / "independent_verification.json"
OUTPUT = ROOT / "Data" / "processed" / "contract_financial_exposure_benchmark"
RANDOM_SEED = 20260811
BOOTSTRAP_REPETITIONS = 5000
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]
ORIGINS = phase1.ORIGINS
WINDOWS = phase1.WINDOWS

PROFILE = ["age_at_signing", "canonical_position", "league", "log_market_value_at_signing"]
PRIOR_WAGE = PROFILE + ["log_prior_season_annual_gross_eur", "prior_wage_available"]
PRIOR_OPPORTUNITY = PRIOR_WAGE + [
    "pre365_same_club_all_competition_opportunity_share",
    "pre365_same_club_all_competition_appearance_rate",
    "pre365_goals_per90", "pre365_assists_per90",
]
SPORTING = PRIOR_OPPORTUNITY + [
    "pre1_canonical_minutes_played", "pre1_canonical_goals_per90", "pre1_canonical_assists_per90",
    "pre2_canonical_minutes_played", "pre2_canonical_goals_per90", "pre2_canonical_assists_per90",
    "pre1_fbref_expected_goals", "pre1_fbref_progressive_carries",
    "pre1_fbref_progressive_passes", "pre1_fbref_pass_completion_pct",
]
FEATURE_SETS = {
    "B0_chronological_median": [],
    "B1_player_market_profile": PROFILE,
    "B2_plus_prior_wage": PRIOR_WAGE,
    "B3_plus_prior_opportunity": PRIOR_OPPORTUNITY,
    "B4_plus_sporting_history": SPORTING,
    "B5_compact_nonlinear": SPORTING,
}

ENDPOINTS = {
    "annual_wage": {"target": "target_log_annual_wage_eur", "cohort": "benchmark_wage_cohort", "unit": "log_eur", "display": "Expected annual fixed-wage peer range"},
    "contract_duration": {"target": "target_contract_duration_years", "cohort": "benchmark_duration_cohort", "unit": "years", "display": "Expected contract-duration peer range"},
    "fixed_wage_commitment": {"target": "target_log_fixed_wage_commitment_eur", "cohort": "benchmark_commitment_cohort", "unit": "log_eur", "display": "Expected fixed-wage commitment peer range"},
    "wage_to_market_value": {"target": "target_log_wage_to_market_value", "cohort": "benchmark_wage_value_cohort", "unit": "log_ratio", "display": "Annual wage-to-market-value peer range"},
}

COMPARISONS = [
    ("B1_vs_B0_profile", "B0_chronological_median", "B1_player_market_profile"),
    ("B2_vs_B1_prior_wage", "B1_player_market_profile", "B2_plus_prior_wage"),
    ("B3_vs_B2_prior_opportunity", "B2_plus_prior_wage", "B3_plus_prior_opportunity"),
    ("B4_vs_B3_sporting_history", "B3_plus_prior_opportunity", "B4_plus_sporting_history"),
    ("B4_vs_B2_all_sporting", "B2_plus_prior_wage", "B4_plus_sporting_history"),
    ("B4_vs_B0_total", "B0_chronological_median", "B4_plus_sporting_history"),
    ("B5_vs_B4_nonlinearity", "B4_plus_sporting_history", "B5_compact_nonlinear"),
    ("B2_vs_B0_parsimonious_total", "B0_chronological_median", "B2_plus_prior_wage"),
    ("B5_vs_B0_nonlinear_total", "B0_chronological_median", "B5_compact_nonlinear"),
]

SELECTED_MODELS = {
    "annual_wage": "B2_plus_prior_wage",
    "contract_duration": "B5_compact_nonlinear",
    "fixed_wage_commitment": "B2_plus_prior_wage",
    "wage_to_market_value": "B2_plus_prior_wage",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: str) -> int:
    return (RANDOM_SEED + int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)) % (2**32 - 1)


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").fillna("").str.strip().str.casefold().isin({"true", "1", "yes"})


def load_frame() -> pd.DataFrame:
    upstream = json.loads(UPSTREAM_VERIFY.read_text(encoding="utf-8"))
    if not upstream.get("all_checks_passed"):
        raise RuntimeError("Contract extension integration has not passed independent verification.")
    frame = phase1.load_frame()
    frame["signed_date"] = pd.to_datetime(frame["signed_date"], errors="coerce")
    frame["signed_year"] = frame["signed_date"].dt.year.astype("Int64")
    wage = pd.to_numeric(frame["annual_gross_eur"], errors="coerce")
    duration = pd.to_numeric(frame["exact_duration_years"], errors="coerce")
    market = pd.to_numeric(frame["at_signing_market_value_eur"], errors="coerce")
    prior = pd.to_numeric(frame["prior_season_salary_annual_gross_eur"], errors="coerce")
    linked = frame["canonical_player_id"].notna() & frame["signed_date"].notna()
    commitment = wage * duration
    frame["log_prior_season_annual_gross_eur"] = np.log1p(prior.clip(lower=0))
    frame["prior_wage_available"] = prior.gt(0).astype(float)
    frame["target_log_annual_wage_eur"] = np.log(wage.where(wage.gt(0)))
    frame["target_contract_duration_years"] = duration
    frame["target_fixed_wage_commitment_eur"] = commitment
    frame["target_log_fixed_wage_commitment_eur"] = np.log(commitment.where(commitment.gt(0)))
    frame["target_wage_to_market_value"] = wage.div(market.where(market.gt(0)))
    frame["target_log_wage_to_market_value"] = np.log(frame["target_wage_to_market_value"].where(frame["target_wage_to_market_value"].gt(0)))
    frame["age_at_expiration"] = pd.to_numeric(frame["age_at_signing"], errors="coerce") + duration
    frame["reported_bonus_adjusted_annual_gross_eur"] = wage + pd.to_numeric(frame["bonus_gross_eur"], errors="coerce")
    frame["benchmark_wage_cohort"] = linked & wage.gt(0)
    frame["benchmark_duration_cohort"] = linked & duration.gt(0)
    frame["benchmark_commitment_cohort"] = linked & commitment.gt(0)
    frame["benchmark_wage_value_cohort"] = linked & wage.gt(0) & market.gt(0)
    frame["age_band"] = pd.cut(frame["age_at_signing"], [-np.inf, 21, 25, 29, 33, np.inf], labels=["<=21", "22-25", "26-29", "30-33", "34+"])
    frame["market_value_band"] = pd.cut(market, [-np.inf, 1e6, 5e6, 15e6, 40e6, np.inf], labels=["<1m", "1-5m", "5-15m", "15-40m", "40m+"])
    return frame


def regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    correlation = np.nan if np.ptp(actual) == 0 or np.ptp(prediction) == 0 else pd.Series(actual).corr(pd.Series(prediction), method="spearman")
    return {
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(actual, prediction))),
        "r2": float(r2_score(actual, prediction)),
        "spearman": float(correlation) if pd.notna(correlation) else np.nan,
        "median_absolute_error": float(np.median(np.abs(actual - prediction))),
    }


def tune_predict(train: pd.DataFrame, validation: pd.DataFrame, evaluation: pd.DataFrame, endpoint: str, variant: str, features: list[str]) -> tuple[np.ndarray, dict[str, Any], int, float]:
    target = ENDPOINTS[endpoint]["target"]
    y_train = pd.to_numeric(train[target]).to_numpy()
    y_validation = pd.to_numeric(validation[target]).to_numpy()
    if not features:
        anchor = float(np.median(y_train))
        validation_estimate = np.repeat(anchor, len(validation))
        prediction = np.repeat(float(pd.to_numeric(pd.concat([train, validation])[target]).median()), len(evaluation))
        interval_width = float(np.quantile(np.abs(y_validation - validation_estimate), .80))
        return prediction, {"selected_parameter": np.nan, **regression_metrics(y_validation, validation_estimate)}, 0, interval_width

    pre = fit_preprocessor(train, features)
    x_train, x_validation = pre.transform(train), pre.transform(validation)
    nonlinear = variant == "B5_compact_nonlinear"
    candidates = []
    if nonlinear:
        grid = [
            {"n_estimators": 75, "learning_rate": .03, "max_depth": 1, "min_samples_leaf": 20},
            {"n_estimators": 100, "learning_rate": .03, "max_depth": 2, "min_samples_leaf": 25},
            {"n_estimators": 75, "learning_rate": .05, "max_depth": 2, "min_samples_leaf": 30},
        ]
        for i, params in enumerate(grid):
            model = GradientBoostingRegressor(loss="huber", random_state=stable_seed(endpoint, variant, str(i)), **params).fit(x_train, y_train)
            estimate = model.predict(x_validation)
            candidates.append(((mean_absolute_error(y_validation, estimate), math.sqrt(mean_squared_error(y_validation, estimate))), params, estimate))
    else:
        for alpha in RIDGE_ALPHAS:
            model = Ridge(alpha=alpha).fit(x_train, y_train)
            estimate = model.predict(x_validation)
            candidates.append(((mean_absolute_error(y_validation, estimate), math.sqrt(mean_squared_error(y_validation, estimate))), {"alpha": alpha}, estimate))
    _, params, validation_estimate = min(candidates, key=lambda item: item[0])
    interval_width = float(np.quantile(np.abs(y_validation - validation_estimate), .80))
    fit = pd.concat([train, validation], ignore_index=True)
    fit_pre = fit_preprocessor(fit, features)
    x_fit, x_eval = fit_pre.transform(fit), fit_pre.transform(evaluation)
    y_fit = pd.to_numeric(fit[target]).to_numpy()
    model = GradientBoostingRegressor(loss="huber", random_state=stable_seed(endpoint, variant, "fit"), **params) if nonlinear else Ridge(alpha=params["alpha"])
    prediction = model.fit(x_fit, y_fit).predict(x_eval)
    return prediction, {"selected_parameter": json.dumps(params, sort_keys=True), **regression_metrics(y_validation, validation_estimate)}, x_fit.shape[1], interval_width


def fit_models(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows, metric_rows = [], []
    for endpoint, spec in ENDPOINTS.items():
        data = frame.loc[frame[spec["cohort"]]].copy()
        target = spec["target"]
        for origin, train_end, validation_year, evaluation_year in ORIGINS:
            train = data.loc[data["signed_year"].le(train_end)].copy()
            validation = data.loc[data["signed_year"].eq(validation_year)].copy()
            evaluation = data.loc[data["signed_year"].eq(evaluation_year)].sort_values("capology_extension_event_id").copy()
            if min(len(train), len(validation), len(evaluation)) < 30:
                raise RuntimeError(f"Insufficient split {endpoint}/{origin}: {len(train)}/{len(validation)}/{len(evaluation)}")
            for variant, features in FEATURE_SETS.items():
                estimate, validation_metrics, encoded, interval_width = tune_predict(train, validation, evaluation, endpoint, variant, features)
                actual = pd.to_numeric(evaluation[target]).to_numpy()
                lower, upper = estimate - interval_width, estimate + interval_width
                evaluation_metrics = regression_metrics(actual, estimate)
                evaluation_metrics["interval_80_coverage"] = float(np.mean((actual >= lower) & (actual <= upper)))
                evaluation_metrics["interval_80_mean_width"] = float(np.mean(upper - lower))
                metric_rows.append({
                    "endpoint": endpoint, "unit": spec["unit"], "origin_id": origin, "model_variant": variant,
                    "train_end_year": train_end, "validation_year": validation_year, "evaluation_year": evaluation_year,
                    "train_rows": len(train), "validation_rows": len(validation), "evaluation_rows": len(evaluation),
                    "raw_feature_count": len(features), "encoded_feature_count": encoded,
                    "validation_interval_abs_residual_q80": interval_width,
                    **{f"validation_{key}": value for key, value in validation_metrics.items()},
                    **{f"evaluation_{key}": value for key, value in evaluation_metrics.items()},
                })
                for i, (_, row) in enumerate(evaluation.iterrows()):
                    value, prediction = float(actual[i]), float(estimate[i])
                    premium = math.exp(value - prediction) - 1 if spec["unit"].startswith("log") else value - prediction
                    prediction_rows.append({
                        "endpoint": endpoint, "unit": spec["unit"], "origin_id": origin,
                        "evaluation_year": evaluation_year, "capology_extension_event_id": row["capology_extension_event_id"],
                        "canonical_player_id": row["canonical_player_id"], "league": row["league"],
                        "canonical_position": row["canonical_position"], "age_band": row["age_band"],
                        "market_value_band": row["market_value_band"], "model_variant": variant,
                        "actual": value, "prediction": prediction, "interval_lower": float(lower[i]),
                        "interval_upper": float(upper[i]), "within_80_interval": bool(lower[i] <= value <= upper[i]),
                        "observed_premium_vs_benchmark": premium, "primary_loss": abs(value - prediction),
                    })
    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)


def paired(predictions: pd.DataFrame, endpoint: str, baseline: str, candidate: str, years: list[int]) -> pd.DataFrame:
    subset = predictions.loc[predictions["endpoint"].eq(endpoint) & predictions["evaluation_year"].isin(years)]
    keys = ["origin_id", "evaluation_year", "capology_extension_event_id", "canonical_player_id"]
    left = subset.loc[subset["model_variant"].eq(baseline), keys + ["primary_loss"]].rename(columns={"primary_loss": "baseline_loss"})
    right = subset.loc[subset["model_variant"].eq(candidate), keys + ["primary_loss"]].rename(columns={"primary_loss": "candidate_loss"})
    return left.merge(right, on=keys, validate="one_to_one")


def cluster_bootstrap(pairs: pd.DataFrame, seed: int) -> dict[str, float]:
    data = pairs.assign(improvement=pairs["baseline_loss"] - pairs["candidate_loss"])
    clusters = data.groupby("canonical_player_id")["improvement"].agg(["sum", "size"])
    sums, sizes = clusters["sum"].to_numpy(float), clusters["size"].to_numpy(float)
    rng, draws = np.random.default_rng(seed), []
    for start in range(0, BOOTSTRAP_REPETITIONS, 250):
        count = min(250, BOOTSTRAP_REPETITIONS - start)
        indices = rng.integers(0, len(clusters), size=(count, len(clusters)))
        draws.extend((sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)).tolist())
    values = np.asarray(draws)
    return {
        "cluster_count": len(clusters), "cluster_ci_lower_95": float(np.quantile(values, .025)),
        "cluster_ci_upper_95": float(np.quantile(values, .975)),
        "cluster_probability_candidate_better": float(np.mean(values > 0)),
    }


def comparisons(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint in ENDPOINTS:
        for window, years in WINDOWS.items():
            for name, baseline, candidate in COMPARISONS:
                pairs = paired(predictions, endpoint, baseline, candidate, years)
                delta = pairs["baseline_loss"] - pairs["candidate_loss"]
                origin_delta = pairs.assign(delta=delta).groupby("evaluation_year")["delta"].mean()
                rows.append({
                    "endpoint": endpoint, "window": window, "comparison": name,
                    "baseline": baseline, "candidate": candidate,
                    "evaluation_years": " | ".join(map(str, sorted(pairs["evaluation_year"].unique()))),
                    "evaluation_rows": len(pairs), "mean_mae_improvement": float(delta.mean()),
                    "origin_wins": int(origin_delta.gt(0).sum()), "origin_count": len(origin_delta),
                    **cluster_bootstrap(pairs, stable_seed(endpoint, window, name)),
                })
    return pd.DataFrame(rows)


def benchmark_bands(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint in ENDPOINTS:
        selected_model = SELECTED_MODELS[endpoint]
        data = predictions.loc[predictions["endpoint"].eq(endpoint) & predictions["model_variant"].eq(selected_model)].copy()
        data["benchmark_band"] = pd.qcut(data["prediction"].rank(method="first"), 5, labels=False) + 1
        grouped = data.groupby("benchmark_band", observed=True).agg(
            records=("actual", "size"), mean_prediction=("prediction", "mean"),
            observed_mean=("actual", "mean"), mean_absolute_error=("primary_loss", "mean"),
        ).reset_index()
        grouped.insert(0, "endpoint", endpoint)
        grouped.insert(1, "selected_model", selected_model)
        grouped["observed_monotone"] = bool(grouped["observed_mean"].is_monotonic_increasing)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def subgroup_stability(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint, selected_model in SELECTED_MODELS.items():
        data = predictions.loc[predictions["endpoint"].eq(endpoint) & predictions["model_variant"].eq(selected_model)]
        for group_type, column in [("league", "league"), ("position", "canonical_position"), ("age_band", "age_band"), ("market_value", "market_value_band")]:
            for group, subset in data.groupby(column, observed=True, dropna=False):
                if len(subset) < 30:
                    continue
                score = regression_metrics(subset["actual"].to_numpy(), subset["prediction"].to_numpy())
                rows.append({"endpoint": endpoint, "selected_model": selected_model, "group_type": group_type, "group": str(group), "records": len(subset), **score})
    return pd.DataFrame(rows)


def deterministic_metric_catalog() -> pd.DataFrame:
    rows = [
        ("age_at_expiration", "deterministic_formula", "age_at_signing + proposed_contract_years", "years", "age_at_signing; proposed_contract_years", "supported", "Does not incorporate unilateral options unless explicitly entered."),
        ("fixed_wage_commitment", "deterministic_formula", "proposed_annual_gross_fixed_wage * proposed_contract_years", "EUR", "proposed_annual_gross_fixed_wage; proposed_contract_years", "supported_proxy", "Estimated fixed commitment; termination, renewal, taxes, and special clauses are outside scope."),
        ("reported_bonus_adjusted_annual_pay", "deterministic_formula", "annual_gross_fixed_wage + reported_annual_bonus", "EUR/year", "annual_gross_fixed_wage; reported_annual_bonus", "optional_sparse", "Bonus is missing, not zero, for most pre-2022 records; do not impute as guaranteed."),
        ("wage_to_market_value", "deterministic_formula", "proposed_annual_gross_fixed_wage / current_market_value", "ratio", "proposed_annual_gross_fixed_wage; current_market_value", "supported", "Public market value is an estimate, not sale proceeds."),
        ("fixed_commitment_to_market_value", "deterministic_formula", "fixed_wage_commitment / current_market_value", "ratio", "fixed_wage_commitment; current_market_value", "supported_proxy", "Shows commitment scale relative to public asset value."),
        ("estimated_acquisition_commitment", "deterministic_formula", "proposed_transfer_fee + fixed_wage_commitment", "EUR", "proposed_transfer_fee; fixed_wage_commitment", "supported_proxy", "Excludes agent, signing, tax, solidarity, contingent, and financing costs."),
        ("transfer_premium", "deterministic_formula", "(proposed_transfer_fee - current_market_value) / current_market_value", "ratio", "proposed_transfer_fee; current_market_value", "supported", "Benchmark against a public value estimate, not intrinsic fair value."),
        ("annual_wage_peer_premium", "historical_peer_benchmark", "proposed_annual_wage / expected_peer_annual_wage - 1", "ratio", "proposed_annual_wage; Phase-4 annual_wage benchmark", "candidate_if_validated", "Historical comparator, not proof of overpayment."),
        ("contract_duration_peer_delta", "historical_peer_benchmark", "proposed_contract_years - expected_peer_contract_years", "years", "proposed_contract_years; Phase-4 duration benchmark", "candidate_if_validated", "Historical comparator, not an optimal term recommendation."),
        ("fixed_commitment_peer_premium", "historical_peer_benchmark", "proposed_fixed_commitment / expected_peer_fixed_commitment - 1", "ratio", "proposed_fixed_commitment; Phase-4 commitment benchmark", "candidate_if_validated", "Historical comparator, not a total-cost estimate."),
        ("club_salary_percentile", "historical_observed_context", "percentile rank among known club salaries", "percentile", "club salary panel; player salary", "supported_when_matched", "Known-salary panel may omit players and bonuses."),
        ("club_known_payroll_share", "historical_observed_context", "player annual gross wage / total known club annual gross wages", "ratio", "club salary panel", "supported_when_matched", "Denominator is known-panel payroll, not audited total payroll."),
    ]
    return pd.DataFrame(rows, columns=["metric", "metric_class", "formula", "unit", "required_inputs", "status", "limitation"])


def limitation_catalog() -> pd.DataFrame:
    return pd.DataFrame([
        ("employer_taxes", "unavailable", "Excluded from all commitment metrics."),
        ("agent_and_intermediary_fees", "unavailable", "Excluded from all commitment metrics."),
        ("signing_on_fees", "unavailable", "Excluded from all commitment metrics."),
        ("release_clauses", "unavailable", "Cannot quantify downside from clause terms."),
        ("club_extension_options", "unavailable", "Cannot price flexibility from unilateral options."),
        ("performance_based_pay", "partial", "Reported bonuses exist for 759 events and are sparse before 2022; never assume missing means zero."),
        ("contract_termination_or_renewal", "not_modeled_in_formula", "Fixed commitment assumes entered term for scale comparison; actual obligation may differ."),
        ("market_value", "public_estimate", "Transfermarkt-style market value is not realized sale price or accounting carrying value."),
        ("currency_and_tax_treatment", "normalized_estimate", "Gross estimates are normalized to EUR; local tax and FX terms are not modeled."),
    ], columns=["item", "availability", "treatment"])


def peer_summary(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.loc[frame["benchmark_commitment_cohort"]].copy()
    rows = []
    for keys, group in data.groupby(["league", "canonical_position", "market_value_band"], observed=True, dropna=False):
        if len(group) < 20:
            continue
        wage = pd.to_numeric(group["annual_gross_eur"], errors="coerce")
        duration = pd.to_numeric(group["exact_duration_years"], errors="coerce")
        commitment = pd.to_numeric(group["target_fixed_wage_commitment_eur"], errors="coerce")
        rows.append({
            "league": keys[0], "position": keys[1], "market_value_band": str(keys[2]), "records": len(group),
            "annual_wage_p25_eur": wage.quantile(.25), "annual_wage_median_eur": wage.median(), "annual_wage_p75_eur": wage.quantile(.75),
            "contract_years_p25": duration.quantile(.25), "contract_years_median": duration.median(), "contract_years_p75": duration.quantile(.75),
            "fixed_commitment_p25_eur": commitment.quantile(.25), "fixed_commitment_median_eur": commitment.median(), "fixed_commitment_p75_eur": commitment.quantile(.75),
        })
    return pd.DataFrame(rows)


def feature_manifest() -> pd.DataFrame:
    rows = []
    for variant, features in FEATURE_SETS.items():
        for feature in features:
            rows.append({"model_variant": variant, "feature": feature, "timing": "known_before_current_extension_terms", "target_leakage_role": "predictor"})
    prohibited = [
        "annual_gross_eur", "exact_duration_years", "age_at_expiration", "log_fixed_wage_commitment_eur",
        "club_salary_percentile", "club_salary_share_known", "salary_change_from_prior_pct",
    ]
    for feature in prohibited:
        rows.append({"model_variant": "prohibited_current_term_fields", "feature": feature, "timing": "known_current_extension_term_or_derived_from_target", "target_leakage_role": "prohibited"})
    return pd.DataFrame(rows)


def decisions(comparison: pd.DataFrame, bands: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint, spec in ENDPOINTS.items():
        selected_model = SELECTED_MODELS[endpoint]
        total_name = "B5_vs_B0_nonlinear_total" if selected_model == "B5_compact_nonlinear" else "B2_vs_B0_parsimonious_total"
        total = comparison.loc[comparison["endpoint"].eq(endpoint) & comparison["window"].eq("pooled_four") & comparison["comparison"].eq(total_name)].iloc[0]
        prior = comparison.loc[comparison["endpoint"].eq(endpoint) & comparison["window"].eq("pooled_four") & comparison["comparison"].eq("B2_vs_B1_prior_wage")].iloc[0]
        sporting = comparison.loc[comparison["endpoint"].eq(endpoint) & comparison["window"].eq("pooled_four") & comparison["comparison"].eq("B4_vs_B2_all_sporting")].iloc[0]
        nonlinear = comparison.loc[comparison["endpoint"].eq(endpoint) & comparison["window"].eq("pooled_four") & comparison["comparison"].eq("B5_vs_B4_nonlinearity")].iloc[0]
        endpoint_metrics = metrics.loc[metrics["endpoint"].eq(endpoint) & metrics["model_variant"].eq(selected_model)]
        mae = float(np.average(endpoint_metrics["evaluation_mae"], weights=endpoint_metrics["evaluation_rows"]))
        r2 = float(np.average(endpoint_metrics["evaluation_r2"], weights=endpoint_metrics["evaluation_rows"]))
        spearman = float(np.average(endpoint_metrics["evaluation_spearman"], weights=endpoint_metrics["evaluation_rows"]))
        median_absolute_error = float(np.average(endpoint_metrics["evaluation_median_absolute_error"], weights=endpoint_metrics["evaluation_rows"]))
        coverage = float(np.average(endpoint_metrics["evaluation_interval_80_coverage"], weights=endpoint_metrics["evaluation_rows"]))
        interval_width = float(np.average(endpoint_metrics["evaluation_interval_80_mean_width"], weights=endpoint_metrics["evaluation_rows"]))
        monotone = bool(bands.loc[bands["endpoint"].eq(endpoint), "observed_monotone"].iloc[0])
        supported = total["cluster_ci_lower_95"] > 0 and total["origin_wins"] >= 3 and monotone and .70 <= coverage <= .90
        rows.append({
            "endpoint": endpoint, "status": "advance_historical_peer_benchmark" if supported else "descriptive_peer_context_only",
            "selected_model": selected_model,
            "recommended_display": spec["display"] if supported else "none", "weighted_origin_mae": mae,
            "weighted_origin_median_absolute_error": median_absolute_error,
            "weighted_origin_r2": r2, "weighted_origin_spearman": spearman,
            "empirical_80_interval_coverage": coverage, "empirical_80_interval_mean_width": interval_width,
            "total_mae_improvement_vs_median": total["mean_mae_improvement"],
            "total_ci_lower_95": total["cluster_ci_lower_95"], "total_ci_upper_95": total["cluster_ci_upper_95"],
            "total_origin_wins": total["origin_wins"], "benchmark_bands_monotone": monotone,
            "prior_wage_adds_signal": bool(prior["cluster_ci_lower_95"] > 0 and prior["origin_wins"] >= 3),
            "prior_wage_increment": prior["mean_mae_improvement"],
            "sporting_history_adds_signal": bool(sporting["cluster_ci_lower_95"] > 0 and sporting["origin_wins"] >= 3),
            "sporting_increment": sporting["mean_mae_improvement"],
            "nonlinear_adds_signal": bool(nonlinear["cluster_ci_lower_95"] > 0 and nonlinear["origin_wins"] >= 3),
            "nonlinear_increment": nonlinear["mean_mae_improvement"],
        })
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    predictions, metrics = fit_models(frame)
    comparison = comparisons(predictions)
    bands = benchmark_bands(predictions)
    subgroups = subgroup_stability(predictions)
    decision = decisions(comparison, bands, metrics)
    catalog = deterministic_metric_catalog()
    limitations = limitation_catalog()
    peers = peer_summary(frame)

    audit_rows = []
    for endpoint, spec in ENDPOINTS.items():
        cohort = frame[spec["cohort"]]
        audit_rows.append({"endpoint": endpoint, "eligible_rows": int(cohort.sum()), "ineligible_rows": int((~cohort).sum()), "target_nonmissing": int(frame.loc[cohort, spec["target"]].notna().sum())})
    coverage = pd.DataFrame(audit_rows)
    bonus_by_year = frame.assign(bonus_flag=as_bool(frame["bonus_available"])).groupby("signed_year", observed=True).agg(records=("capology_extension_event_id", "size"), bonus_available=("bonus_flag", "sum")).reset_index()

    checks = []
    def add(name: str, observed: object, expected: object, passed: bool, notes: str) -> None:
        checks.append({"check": name, "observed": observed, "expected": expected, "passed": bool(passed), "notes": notes})
    add("source_event_ids_unique", int(frame["capology_extension_event_id"].duplicated().sum()), 0, not frame["capology_extension_event_id"].duplicated().any(), "One row per extension event.")
    expected_counts = {"annual_wage": 2265, "contract_duration": 2358, "fixed_wage_commitment": 2265, "wage_to_market_value": 2222}
    for endpoint, expected in expected_counts.items():
        observed = int(frame[ENDPOINTS[endpoint]["cohort"]].sum())
        add(f"{endpoint}_cohort", observed, expected, observed == expected, "Linked positive target cohort.")
    commitment_ok = np.allclose(frame.loc[frame["benchmark_commitment_cohort"], "target_fixed_wage_commitment_eur"], pd.to_numeric(frame.loc[frame["benchmark_commitment_cohort"], "annual_gross_eur"]) * pd.to_numeric(frame.loc[frame["benchmark_commitment_cohort"], "exact_duration_years"]))
    add("fixed_commitment_formula", int(commitment_ok), 1, commitment_ok, "Fixed commitment equals annual gross fixed wage times exact years.")
    leakage = feature_manifest()
    predictors = leakage.loc[leakage["target_leakage_role"].eq("predictor"), "feature"]
    prohibited = set(leakage.loc[leakage["target_leakage_role"].eq("prohibited"), "feature"])
    leakage_ok = not set(predictors).intersection(prohibited)
    add("target_leakage_exclusion", int(leakage_ok), 1, leakage_ok, "Current terms and their derivatives are not benchmark predictors.")
    add("prediction_key_unique", int(predictions.duplicated(["endpoint", "origin_id", "model_variant", "capology_extension_event_id"]).sum()), 0, not predictions.duplicated(["endpoint", "origin_id", "model_variant", "capology_extension_event_id"]).any(), "One prediction per endpoint/origin/model/event.")
    interval_ok = predictions["interval_lower"].le(predictions["prediction"]).all() and predictions["prediction"].le(predictions["interval_upper"]).all()
    add("interval_ordering", int(interval_ok), 1, interval_ok, "Every empirical interval contains its point benchmark.")
    add("decision_rows", len(decision), 4, len(decision) == 4, "One decision per peer benchmark.")
    checks_frame = pd.DataFrame(checks)

    model_columns = [
        "capology_extension_event_id", "canonical_player_id", "player_name", "league", "canonical_position",
        "signed_date", "signed_year", "age_at_signing", "age_at_expiration", "at_signing_market_value_eur",
        "annual_gross_eur", "bonus_available", "bonus_gross_eur", "reported_bonus_adjusted_annual_gross_eur",
        "exact_duration_years", "target_fixed_wage_commitment_eur", "target_wage_to_market_value",
        "prior_season_salary_annual_gross_eur", "club_salary_percentile", "club_salary_share_known",
        "salary_change_from_prior_pct", *[spec["cohort"] for spec in ENDPOINTS.values()],
        *[spec["target"] for spec in ENDPOINTS.values()],
    ]
    write_csv(frame[list(dict.fromkeys(model_columns))], OUTPUT / "financial_exposure_model_matrix.csv")
    write_csv(coverage, OUTPUT / "benchmark_cohort_audit.csv")
    write_csv(bonus_by_year, OUTPUT / "bonus_coverage_by_year.csv")
    write_csv(catalog, OUTPUT / "deterministic_metric_catalog.csv")
    write_csv(limitations, OUTPUT / "financial_scope_limitations.csv")
    write_csv(peers, OUTPUT / "historical_peer_summary.csv")
    write_csv(feature_manifest(), OUTPUT / "feature_manifest.csv")
    write_csv(metrics, OUTPUT / "model_origin_metrics.csv")
    write_csv(predictions, OUTPUT / "benchmark_predictions.csv")
    write_csv(comparison, OUTPUT / "model_comparison_results.csv")
    write_csv(bands, OUTPUT / "benchmark_bands.csv")
    write_csv(subgroups, OUTPUT / "subgroup_stability.csv")
    write_csv(decision, OUTPUT / "benchmark_decision_summary.csv")
    write_csv(checks_frame, OUTPUT / "build_checks.csv")
    write_csv(pd.DataFrame([
        {"source_file": SOURCE.relative_to(ROOT).as_posix(), "bytes": SOURCE.stat().st_size, "sha256": sha256(SOURCE)},
        {"source_file": UPSTREAM_VERIFY.relative_to(ROOT).as_posix(), "bytes": UPSTREAM_VERIFY.stat().st_size, "sha256": sha256(UPSTREAM_VERIFY)},
    ]), OUTPUT / "source_manifest.csv")

    summary = {
        "source_extension_events": len(frame), "wage_cohort": 2265, "duration_cohort": 2358,
        "commitment_cohort": 2265, "wage_value_cohort": 2222,
        "bonus_available_events": int(as_bool(frame["bonus_available"]).sum()),
        "prediction_rows": len(predictions), "checks_passed": int(checks_frame["passed"].sum()),
        "checks_total": len(checks_frame), "all_checks_passed": bool(checks_frame["passed"].all()),
        "decisions": dict(zip(decision["endpoint"], decision["status"])),
    }
    (OUTPUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Contract financial exposure and peer benchmark", "",
        "Phase 4 separates exact input-driven calculations from historical extension benchmarks. It is not a full cost, fair-value, or optimal-contract engine.", "",
        "## Deterministic layer", "",
        "Supported calculations include age at expiration, fixed wage commitment, wage-to-market-value, commitment-to-market-value, transfer premium, and a proposed fee-plus-fixed-wage commitment proxy. See the metric catalog for exact formulas and required inputs.",
        "Bonuses are optional and sparse. Missing bonuses, taxes, agent fees, signing fees, clauses, and options are never converted to zero.", "",
        "## Historical benchmark decisions", "",
    ]
    for row in decision.itertuples(index=False):
        error_text = (
            f"median multiplicative error factor {math.exp(row.weighted_origin_median_absolute_error):.2f}x"
            if ENDPOINTS[row.endpoint]["unit"].startswith("log")
            else f"median absolute error {row.weighted_origin_median_absolute_error:.2f} years"
        )
        lines.append(
            f"- `{row.endpoint}`: `{row.status}` using `{row.selected_model}`; MAE {row.weighted_origin_mae:.4f}, R² {row.weighted_origin_r2:.3f}, Spearman {row.weighted_origin_spearman:.3f}; {error_text}; "
            f"improvement vs median {row.total_mae_improvement_vs_median:.4f} (95% CI [{row.total_ci_lower_95:.4f}, {row.total_ci_upper_95:.4f}]); empirical 80% interval coverage {row.empirical_80_interval_coverage:.1%}."
        )
    lines.extend(["", "Peer premiums describe how unusual an offer is relative to historical extensions. They do not prove overpayment or recommend an optimal contract."])
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    files = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name != "output_manifest.csv")
    write_csv(pd.DataFrame([{"output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files]), OUTPUT / "output_manifest.csv")
    if not checks_frame["passed"].all():
        raise RuntimeError(f"Failed build checks: {checks_frame.loc[~checks_frame['passed'], 'check'].tolist()}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

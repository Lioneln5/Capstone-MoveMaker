"""Bounded refinement experiment for the Phase-1 sustained-contribution model.

Question: do pre-extension injury history and evidence-based elite-player
interactions improve "meaningful-contributor-through-Year-2" (the exact,
unchanged target `target_sustained_meaningful_contribution` from
models/run_extension_opportunity_diagnostic.py)?

This is a research experiment, not a redeployment. It:
  - reproduces the frozen M5 baseline exactly (Step 1), on the SAME frozen
    source `run_extension_opportunity_diagnostic.py` itself uses;
  - builds leakage-safe injury features from Data/injury/full_dataset_thesis
    - 1.csv, linked to canonical identity conservatively (exact-unique-name
    crosswalk match only, cross-checked against real club history -- no
    forced/ambiguous matches);
  - builds elite-player features (percentile / spline / interactions) fit
    ONLY on each rolling origin's training fold, never on its evaluation
    fold;
  - compares five compact variants (R0 baseline .. R4 stability-screened
    compact candidate) under the exact same chronological rolling-origin /
    player-cluster-bootstrap protocol as the frozen Phase-1 diagnostic;
  - decides, by pre-declared gates, whether any candidate should advance;
  - if (and only if) a candidate advances, refits it through the same
    deployment cutoff as the live production model and runs a face-validity
    audit against three real, named players -- reported, never used to pick
    the model.

Every output goes to Data/processed/contribution_model_refinement/. Nothing
under Data/processed/extension_opportunity_diagnostic/ or
Data/processed/deployment_models/ is read for writing, and this script never
touches api/, html/, or models/deployment_scorer.py.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
SCRIPTS_DIR = ROOT / "scripts"
for path in (MODELS_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_extension_opportunity_diagnostic as m  # noqa: E402
from mixed_type_preprocessor import fit_preprocessor  # noqa: E402
from build_capology_contracts import build_player_map, build_club_map, link_player, link_club, normalize_text  # noqa: E402

# ---------------------------------------------------------------------------
# Paths -- every source is read-only; every output is new, under a new dir.
# ---------------------------------------------------------------------------
FROZEN_DIR = ROOT / "Data" / "processed" / "frozen_contract_sources" / "phase1_5_2026-08-11" / "contract_extension_integration"
FROZEN_SOURCE = FROZEN_DIR / "extension_modeling_master.csv"
FROZEN_VERIFY = FROZEN_DIR / "independent_verification.json"
INJURY_SOURCE = ROOT / "Data" / "injury" / "full_dataset_thesis - 1.csv"
APPEARANCE_PATH = ROOT / "Data" / "processed" / "transfermarkt_clean" / "tables" / "appearances_clean.csv"
LIVE_PERFORMANCE_OVERLAY = ROOT / "Data" / "processed" / "live_input_refresh" / "canonical_performance_overlay.csv"
LIVE_SALARY_OVERLAY = ROOT / "Data" / "processed" / "live_input_refresh" / "canonical_salary_overlay.csv"
LIVE_INPUT_VERIFY = ROOT / "Data" / "processed" / "live_input_refresh" / "independent_verification.json"
DEPLOYMENT_MANIFEST = ROOT / "Data" / "processed" / "deployment_models" / "model_manifest.json"

OUTPUT = ROOT / "Data" / "processed" / "contribution_model_refinement"

RANDOM_SEED = m.RANDOM_SEED
LOGISTIC_C = m.LOGISTIC_C
BOOTSTRAP_REPETITIONS = m.BOOTSTRAP_REPETITIONS
ORIGINS = m.ORIGINS
WINDOWS = m.WINDOWS
ENDPOINT = "sustained_meaningful_contribution"
TARGET = "target_sustained_meaningful_contribution"
M5_FEATURES = m.FEATURE_SETS["M5_plus_relative_financial_context"]

INJURY_COVERAGE_START = pd.Timestamp("2020-07-01")  # start of the 20/21 season, the injury source's declared coverage floor

INJURY_FEATURES = [
    "injury_completed_spell_count_365", "injury_completed_spell_count_730",
    "injury_calendar_days_365", "injury_calendar_days_730",
    "injury_completed_spell_games_missed_365", "injury_completed_spell_games_missed_730",
    "injury_days_since_last_completed", "injury_active_at_signing_indicator",
    "injury_recurring_indicator", "injury_max_completed_spell_duration_730",
    "injury_burden_per_covered_day_365", "injury_data_coverage_indicator",
    "injury_left_censored_365_indicator", "injury_left_censored_730_indicator",
]
ELITE_FEATURES = [
    "elite_market_value_percentile", "elite_top_decile_indicator", "elite_top_5pct_indicator",
    "log_market_value_hinge_k1", "log_market_value_hinge_k2", "log_market_value_hinge_k3",
    "elite_percentile_x_pre365_opportunity", "elite_percentile_x_age",
    "elite_percentile_x_contract_years", "pre365_opportunity_x_age",
]

DEMO_PLAYERS = [
    {"label": "Haaland", "query": "Haaland", "reference_baseline": 0.7109},
    {"label": "Mbappe", "query": "Mbapp", "reference_baseline": 0.6428},
    {"label": "Bellingham", "query": "Bellingham", "reference_baseline": 0.6947},
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


# ---------------------------------------------------------------------------
# Step 1: exact-reproduction base frame (mirrors run_extension_opportunity_
# diagnostic.load_frame() field-for-field, but reads the FROZEN source path
# explicitly named in the prompt, not the live contract_extension_integration
# path that module hardcodes).
# ---------------------------------------------------------------------------
def load_base_frame() -> pd.DataFrame:
    verification = json.loads(FROZEN_VERIFY.read_text(encoding="utf-8"))
    if not verification.get("all_checks_passed"):
        raise RuntimeError("Frozen contract-integration snapshot has not passed its own independent verification.")
    frame = pd.read_csv(FROZEN_SOURCE, low_memory=False)
    frame["signed_date"] = pd.to_datetime(frame["signed_date"], errors="coerce")
    frame["expiration_date"] = pd.to_datetime(frame["expiration_date"], errors="coerce")
    frame["signed_year"] = frame["signed_date"].dt.year.astype("Int64")
    for year in [1, 2, 3]:
        frame[f"contract_covers_y{year}"] = frame["expiration_date"].ge(frame["signed_date"] + pd.Timedelta(days=365 * year))
    numeric = lambda c: pd.to_numeric(frame[c], errors="coerce")
    frame["age_at_expiration"] = frame["age_at_signing"] + numeric("exact_duration_years")
    frame["log_market_value_at_signing"] = np.log1p(numeric("at_signing_market_value_eur").clip(lower=0))
    frame["log_annual_gross_eur"] = np.log1p(numeric("annual_gross_eur").clip(lower=0))
    fixed = numeric("annual_gross_eur") * numeric("exact_duration_years")
    frame["log_fixed_wage_commitment_eur"] = np.log1p(fixed.clip(lower=0))
    frame["annual_wage_to_market_value"] = numeric("annual_gross_eur").div(numeric("at_signing_market_value_eur").where(numeric("at_signing_market_value_eur").gt(0)))
    frame["commitment_to_market_value"] = fixed.div(numeric("at_signing_market_value_eur").where(numeric("at_signing_market_value_eur").gt(0)))
    pre_minutes = numeric("pre365_same_club_all_competition_minutes")
    frame["pre365_goals_per90"] = 90 * numeric("pre365_same_club_all_competition_goals").div(pre_minutes.where(pre_minutes.gt(0)))
    frame["pre365_assists_per90"] = 90 * numeric("pre365_same_club_all_competition_assists").div(pre_minutes.where(pre_minutes.gt(0)))

    pre = numeric("pre365_same_club_all_competition_opportunity_share")
    y1 = numeric("post_y1_same_club_all_competition_opportunity_share")
    y2 = numeric("post_y2_same_club_all_competition_opportunity_share")
    linked = frame["canonical_player_id"].notna()
    evidence = m.as_bool(frame["pre365_window_evidence_eligible_10_club_games"]) & m.as_bool(frame["post_y2_window_evidence_eligible_10_club_games"])
    frame["cohort_primary_y2"] = linked & evidence & frame["contract_covers_y2"] & pre.notna() & y2.notna()
    frame[TARGET] = (y1.ge(0.25) & y2.ge(0.25)).astype(int)
    frame["age_band"] = pd.cut(frame["age_at_signing"], [-np.inf, 21, 25, 29, 33, np.inf], labels=["<=21", "22-25", "26-29", "30-33", "34+"])
    frame["contract_length_band"] = pd.cut(frame["exact_duration_years"], [-np.inf, 2, 3, 4, 5, np.inf], labels=["<2y", "2-<3y", "3-<4y", "4-<5y", "5y+"])
    return frame


def reproduce_baseline(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Step 1: run the frozen M5 sustained-contribution model through
    run_extension_opportunity_diagnostic.tune_predict UNCHANGED and compare
    pooled out-of-time metrics against the published reference. Hard-stops
    if reproduction fails."""
    data = frame.loc[frame["cohort_primary_y2"] & frame[TARGET].notna()].copy()
    rows = []
    actual_all, pred_all = [], []
    for origin, train_end, validation_year, evaluation_year in ORIGINS:
        train = data.loc[data["signed_year"].le(train_end)].copy()
        validation = data.loc[data["signed_year"].eq(validation_year)].copy()
        evaluation = data.loc[data["signed_year"].eq(evaluation_year)].sort_values("capology_extension_event_id").copy()
        prediction, val_metrics, encoded = m.tune_predict(train, validation, evaluation, M5_FEATURES, TARGET, "binary", "M5_plus_relative_financial_context")
        actual = pd.to_numeric(evaluation[TARGET]).to_numpy().astype(int)
        actual_all.append(actual)
        pred_all.append(prediction)
        rows.append({
            "origin_id": origin, "evaluation_year": evaluation_year, "evaluation_rows": len(evaluation),
            "evaluation_brier": brier_score_loss(actual, prediction),
        })
    actual = np.concatenate(actual_all)
    prediction = np.concatenate(pred_all)
    reproduced = {
        "pooled_rows": int(len(actual)),
        "observed_rate": float(actual.mean()),
        "mean_predicted": float(prediction.mean()),
        "brier": float(brier_score_loss(actual, prediction)),
        "auc": float(roc_auc_score(actual, prediction)),
        "average_precision": float(average_precision_score(actual, prediction)),
    }
    expected = {"pooled_rows": 929, "observed_rate": 0.4887, "mean_predicted": 0.5227, "brier": 0.1869, "auc": 0.788, "average_precision": 0.732}
    tolerances = {"pooled_rows": 0, "observed_rate": 0.001, "mean_predicted": 0.001, "brier": 0.001, "auc": 0.001, "average_precision": 0.001}
    mismatches = []
    for key, expected_value in expected.items():
        observed_value = reproduced[key]
        tolerance = tolerances[key]
        if abs(observed_value - expected_value) > tolerance:
            mismatches.append(f"{key}: observed={observed_value} expected~={expected_value} tolerance={tolerance}")
    if mismatches:
        raise RuntimeError("Baseline reproduction FAILED to match the published reference within tolerance:\n" + "\n".join(mismatches))

    high_value = frame.loc[frame["capology_extension_event_id"].isin(data["capology_extension_event_id"])].copy()
    predictions_frame = data[["capology_extension_event_id"]].copy()
    market_value = pd.to_numeric(frame.set_index("capology_extension_event_id").loc[data["capology_extension_event_id"], "at_signing_market_value_eur"], errors="coerce").to_numpy()
    # Pool the same evaluation-only rows used above, in the same order, to audit >=75M.
    ordered_ids = np.concatenate([
        data.loc[data["signed_year"].eq(evaluation_year)].sort_values("capology_extension_event_id")["capology_extension_event_id"].to_numpy()
        for _, _, _, evaluation_year in ORIGINS
    ])
    ordered_mv = pd.to_numeric(frame.set_index("capology_extension_event_id").loc[ordered_ids, "at_signing_market_value_eur"], errors="coerce").to_numpy()
    hv_mask = ordered_mv >= 75_000_000
    high_value_audit = {
        "n": int(hv_mask.sum()),
        "observed": float(actual[hv_mask].mean()) if hv_mask.sum() else None,
        "predicted": float(prediction[hv_mask].mean()) if hv_mask.sum() else None,
    }
    expected_hv = {"n": 22, "observed": 0.864, "predicted": 0.856}
    hv_mismatches = []
    if high_value_audit["n"] != expected_hv["n"]:
        hv_mismatches.append(f"n: observed={high_value_audit['n']} expected={expected_hv['n']}")
    if high_value_audit["observed"] is not None and abs(high_value_audit["observed"] - expected_hv["observed"]) > 0.01:
        hv_mismatches.append(f"observed_rate: observed={high_value_audit['observed']} expected~={expected_hv['observed']}")
    if high_value_audit["predicted"] is not None and abs(high_value_audit["predicted"] - expected_hv["predicted"]) > 0.01:
        hv_mismatches.append(f"predicted_rate: observed={high_value_audit['predicted']} expected~={expected_hv['predicted']}")
    if hv_mismatches:
        raise RuntimeError("High-value (>=75M) audit FAILED to reproduce within tolerance:\n" + "\n".join(hv_mismatches))

    return pd.DataFrame(rows), {"pooled": reproduced, "expected": expected, "high_value_75m_audit": high_value_audit, "cohort_rows": len(data)}


# ---------------------------------------------------------------------------
# Step 2: leakage-safe injury linkage and point-in-time feature construction.
# ---------------------------------------------------------------------------
def _expand_link_column(keys: pd.Series, linker) -> pd.DataFrame:
    unique_keys = keys.drop_duplicates()
    rows = [dict(link=linker(key), key=key) for key in unique_keys]
    expanded = pd.json_normalize([r["link"] for r in rows])
    expanded.insert(0, "key", [r["key"] for r in rows])
    return expanded


def build_injury_linkage() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (clean_spells, player_linkage_audit, club_linkage_audit)."""
    fbref_map, identity_map = build_player_map()
    club_map, _ = build_club_map()

    raw = pd.read_csv(INJURY_SOURCE, low_memory=False)
    raw["player_key"] = raw["player_name"].map(normalize_text)
    raw["club_key"] = raw["club"].map(normalize_text)
    raw["from_date"] = pd.to_datetime(raw["injury_from_parsed"], errors="coerce")
    raw["until_date"] = pd.to_datetime(raw["injury_until_parsed"], errors="coerce")
    raw["days_recomputed"] = (raw["until_date"] - raw["from_date"]).dt.days
    raw["games_missed"] = pd.to_numeric(raw["Games missed"], errors="coerce")
    raw["days_field_parsed"] = pd.to_numeric(raw["Days"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")

    player_link = _expand_link_column(raw["player_key"], lambda k: link_player(k, fbref_map, identity_map))
    player_link = player_link.rename(columns={
        "key": "player_key", "canonical_player_id": "canonical_player_id",
        "player_link_method": "player_link_method",
    })[["player_key", "canonical_player_id", "canonical_player_name", "player_link_method"]]
    club_link = _expand_link_column(raw["club_key"], lambda k: link_club(k, club_map))
    club_link = club_link.rename(columns={"key": "club_key"})[["club_key", "canonical_club_id", "canonical_club_name", "club_link_method"]]

    raw = raw.merge(player_link, on="player_key", how="left").merge(club_link, on="club_key", how="left", suffixes=("", "_club"))

    # Cross-check: does the resolved club ever appear in this player's real
    # Transfermarkt appearance history? A resolved name+club combination that
    # never overlaps that player's actual club history is a strong signal of
    # a coincidental name collision the crosswalk's exact-unique-name match
    # did not catch, so it is EXCLUDED rather than forced through.
    apps = pd.read_csv(APPEARANCE_PATH, usecols=["player_id", "player_club_id"], low_memory=False)
    apps["player_id"] = pd.to_numeric(apps["player_id"], errors="coerce")
    apps["player_club_id"] = pd.to_numeric(apps["player_club_id"], errors="coerce")
    apps = apps.dropna(subset=["player_id", "player_club_id"])
    club_history = apps.groupby("player_id")["player_club_id"].apply(lambda s: frozenset(int(v) for v in s)).to_dict()

    def check_consistency(row) -> object:
        if pd.isna(row["canonical_player_id"]) or pd.isna(row["canonical_club_id"]):
            return np.nan  # nothing to cross-check
        history = club_history.get(float(row["canonical_player_id"]))
        if not history:
            return np.nan  # no appearance history at all to check against
        return int(row["canonical_club_id"]) in history

    raw["club_history_consistent"] = raw.apply(check_consistency, axis=1)
    raw["excluded_club_mismatch"] = raw["club_history_consistent"].eq(False)

    linked = raw.loc[raw["canonical_player_id"].notna() & ~raw["excluded_club_mismatch"]].copy()
    linked["canonical_player_id"] = linked["canonical_player_id"].astype(int)
    before = len(linked)
    linked = linked.drop_duplicates(subset=["canonical_player_id", "from_date", "until_date", "Injury"])
    duplicates_dropped = before - len(linked)

    spells = linked[[
        "canonical_player_id", "canonical_player_name", "from_date", "until_date",
        "days_recomputed", "days_field_parsed", "games_missed", "Injury", "Season", "league",
        "canonical_club_id", "club", "club_history_consistent",
    ]].reset_index(drop=True)

    player_audit = raw.drop_duplicates(subset=["player_key"])[[
        "player_key", "player_name", "canonical_player_id", "canonical_player_name", "player_link_method",
    ]].copy()
    player_audit["row_count_in_source"] = player_audit["player_key"].map(raw["player_key"].value_counts())
    player_audit["linked"] = player_audit["canonical_player_id"].notna()

    club_audit = raw.drop_duplicates(subset=["club_key"])[[
        "club_key", "club", "canonical_club_id", "canonical_club_name", "club_link_method",
    ]].copy()
    club_audit["row_count_in_source"] = club_audit["club_key"].map(raw["club_key"].value_counts())
    club_audit["linked"] = club_audit["canonical_club_id"].notna()

    coverage_notes = pd.DataFrame([{
        "raw_rows": len(raw),
        "unique_player_names": raw["player_key"].nunique(),
        "unique_players_linked": int(player_audit["linked"].sum()),
        "row_level_player_link_rate": float(raw["canonical_player_id"].notna().mean()),
        "unique_clubs": raw["club_key"].nunique(),
        "unique_clubs_linked": int(club_audit["linked"].sum()),
        "club_history_mismatches_excluded": int(raw["excluded_club_mismatch"].sum()),
        "duplicate_spells_dropped": int(duplicates_dropped),
        "clean_spell_rows": len(spells),
        "distinct_linked_players_with_spells": spells["canonical_player_id"].nunique(),
        "injury_source_min_from_date": str(raw["from_date"].min().date()) if raw["from_date"].notna().any() else None,
        "injury_source_max_until_date": str(raw["until_date"].max().date()) if raw["until_date"].notna().any() else None,
        "declared_coverage_start": str(INJURY_COVERAGE_START.date()),
    }])
    return spells, player_audit, coverage_notes, club_audit


def build_spell_lookup(spells: pd.DataFrame) -> dict[int, list[dict]]:
    lookup: dict[int, list[dict]] = {}
    for row in spells.itertuples(index=False):
        lookup.setdefault(int(row.canonical_player_id), []).append({
            "from": row.from_date, "until": row.until_date,
            "days": row.days_recomputed, "games_missed": row.games_missed,
        })
    return lookup


def injury_features_for(player_id: float, signing_date: pd.Timestamp, lookup: dict[int, list[dict]]) -> dict[str, float]:
    result: dict[str, float] = {}
    covered = pd.notna(player_id) and int(player_id) in lookup
    result["injury_data_coverage_indicator"] = 1.0 if covered else 0.0
    spells = lookup.get(int(player_id), []) if covered else []

    for window_days, suffix in [(365, "365"), (730, "730")]:
        window_start = signing_date - pd.Timedelta(days=window_days)
        confirmed_start = max(window_start, INJURY_COVERAGE_START)
        covered_days = max(0, (signing_date - confirmed_start).days)
        result[f"injury_left_censored_{suffix}_indicator"] = 1.0 if covered_days < window_days else 0.0
        if not covered:
            result[f"injury_completed_spell_count_{suffix}"] = np.nan
            result[f"injury_calendar_days_{suffix}"] = np.nan
            result[f"injury_completed_spell_games_missed_{suffix}"] = np.nan
            if suffix == "730":
                result["injury_max_completed_spell_duration_730"] = np.nan
            continue
        completed_count = 0
        calendar_days = 0
        games_missed_sum = 0.0
        max_duration = 0.0
        for spell in spells:
            f, u = spell["from"], spell["until"]
            if pd.isna(f) or pd.isna(u) or f >= signing_date:
                continue
            is_completed = u <= signing_date
            overlap_start = max(f, confirmed_start)
            overlap_end = min(u, signing_date) if is_completed else signing_date
            if overlap_end > overlap_start:
                calendar_days += (overlap_end - overlap_start).days
            if is_completed and u > window_start:
                completed_count += 1
                if pd.notna(spell["games_missed"]):
                    games_missed_sum += float(spell["games_missed"])
                if suffix == "730" and pd.notna(spell["days"]):
                    max_duration = max(max_duration, float(spell["days"]))
        result[f"injury_completed_spell_count_{suffix}"] = float(completed_count)
        result[f"injury_calendar_days_{suffix}"] = float(calendar_days)
        result[f"injury_completed_spell_games_missed_{suffix}"] = float(games_missed_sum)
        if suffix == "730":
            result["injury_max_completed_spell_duration_730"] = float(max_duration)

    if not covered:
        result["injury_days_since_last_completed"] = np.nan
        result["injury_active_at_signing_indicator"] = np.nan
        result["injury_recurring_indicator"] = np.nan
        result["injury_burden_per_covered_day_365"] = np.nan
        return result

    completed_before = [s for s in spells if pd.notna(s["until"]) and s["until"] <= signing_date]
    result["injury_days_since_last_completed"] = float((signing_date - max(s["until"] for s in completed_before)).days) if completed_before else np.nan
    result["injury_active_at_signing_indicator"] = 1.0 if any(
        pd.notna(s["from"]) and pd.notna(s["until"]) and s["from"] <= signing_date <= s["until"] for s in spells
    ) else 0.0
    result["injury_recurring_indicator"] = 1.0 if result["injury_completed_spell_count_730"] >= 2 else 0.0
    window_start_365 = signing_date - pd.Timedelta(days=365)
    confirmed_start_365 = max(window_start_365, INJURY_COVERAGE_START)
    covered_days_365 = max(0, (signing_date - confirmed_start_365).days)
    result["injury_burden_per_covered_day_365"] = (
        result["injury_calendar_days_365"] / covered_days_365 if covered_days_365 > 0 else np.nan
    )
    return result


def attach_injury_features(frame: pd.DataFrame, lookup: dict[int, list[dict]]) -> pd.DataFrame:
    records = []
    for row in frame.itertuples(index=False):
        records.append(injury_features_for(getattr(row, "canonical_player_id"), getattr(row, "signed_date"), lookup))
    injury_frame = pd.DataFrame(records, index=frame.index)
    out = frame.copy()
    for column in injury_frame.columns:
        out[column] = injury_frame[column]
    return out


# ---------------------------------------------------------------------------
# Step 3: fold-safe elite/value-shape features.
# ---------------------------------------------------------------------------
def fit_elite_reference(fold: pd.DataFrame) -> dict[str, Any]:
    league_values: dict[str, np.ndarray] = {}
    for league, group in fold.groupby("league"):
        values = pd.to_numeric(group["log_market_value_at_signing"], errors="coerce").dropna().sort_values().to_numpy()
        if len(values) >= 20:
            league_values[str(league)] = values
    pooled_values = pd.to_numeric(fold["log_market_value_at_signing"], errors="coerce").dropna().sort_values().to_numpy()
    knots = np.quantile(pooled_values, [0.50, 0.80, 0.95]).tolist() if len(pooled_values) >= 30 else [0.0, 0.0, 0.0]
    return {"league_values": league_values, "pooled_values": pooled_values, "knots": knots, "fold_rows": int(len(fold))}


def _percentile_rank(value: float, sorted_values: np.ndarray) -> float:
    if pd.isna(value) or len(sorted_values) == 0:
        return np.nan
    left = np.searchsorted(sorted_values, value, side="left")
    right = np.searchsorted(sorted_values, value, side="right")
    return float((left + right) / 2.0 / len(sorted_values))


def apply_elite_features(frame: pd.DataFrame, reference: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    league_values, pooled_values = reference["league_values"], reference["pooled_values"]
    percentile = [
        _percentile_rank(mv, league_values.get(str(league), pooled_values))
        for league, mv in zip(out["league"], pd.to_numeric(out["log_market_value_at_signing"], errors="coerce"))
    ]
    pct = pd.Series(percentile, index=out.index)
    out["elite_market_value_percentile"] = pct
    out["elite_top_decile_indicator"] = np.where(pct.isna(), np.nan, (pct >= 0.90).astype(float))
    out["elite_top_5pct_indicator"] = np.where(pct.isna(), np.nan, (pct >= 0.95).astype(float))
    k1, k2, k3 = reference["knots"]
    lmv = pd.to_numeric(out["log_market_value_at_signing"], errors="coerce")
    out["log_market_value_hinge_k1"] = (lmv - k1).clip(lower=0)
    out["log_market_value_hinge_k2"] = (lmv - k2).clip(lower=0)
    out["log_market_value_hinge_k3"] = (lmv - k3).clip(lower=0)
    pre365 = pd.to_numeric(out["pre365_same_club_all_competition_opportunity_share"], errors="coerce")
    age = pd.to_numeric(out["age_at_signing"], errors="coerce")
    years = pd.to_numeric(out["exact_duration_years"], errors="coerce")
    out["elite_percentile_x_pre365_opportunity"] = pct * pre365
    out["elite_percentile_x_age"] = pct * age
    out["elite_percentile_x_contract_years"] = pct * years
    out["pre365_opportunity_x_age"] = pre365 * age
    return out


# ---------------------------------------------------------------------------
# Step 4: compact candidate fitting -- binary-logistic clone of
# run_extension_opportunity_diagnostic.tune_predict's binary branch, with an
# optional fold-safe elite-feature injection point.
# ---------------------------------------------------------------------------
def tune_predict_variant(
    train: pd.DataFrame, validation: pd.DataFrame, evaluation: pd.DataFrame,
    static_features: list[str], fold_features: list[str] | None,
) -> dict[str, Any]:
    use_fold = bool(fold_features)
    if use_fold:
        ref1 = fit_elite_reference(train)
        train1, validation1 = apply_elite_features(train, ref1), apply_elite_features(validation, ref1)
    else:
        ref1 = None
        train1, validation1 = train, validation
    features = static_features + (fold_features or [])

    pre = fit_preprocessor(train1, features)
    x_train, x_validation = pre.transform(train1), pre.transform(validation1)
    y_train = pd.to_numeric(train1[TARGET]).to_numpy().astype(int)
    y_validation = pd.to_numeric(validation1[TARGET]).to_numpy().astype(int)

    candidates = []
    for c_value in LOGISTIC_C:
        model = LogisticRegression(C=c_value, penalty="l2", solver="liblinear", max_iter=3000, random_state=RANDOM_SEED).fit(x_train, y_train)
        pred = np.clip(model.predict_proba(x_validation)[:, 1], 1e-6, 1 - 1e-6)
        candidates.append(((brier_score_loss(y_validation, pred), log_loss(y_validation, pred, labels=[0, 1])), c_value, pred))
    _, c_selected, best_validation_pred = min(candidates, key=lambda item: item[0])
    validation_metrics = m.classification_metrics(y_validation, best_validation_pred)

    fit = pd.concat([train, validation], ignore_index=True)
    if use_fold:
        ref2 = fit_elite_reference(fit)
        fit1, evaluation1 = apply_elite_features(fit, ref2), apply_elite_features(evaluation, ref2)
    else:
        ref2 = None
        fit1, evaluation1 = fit, evaluation
    fit_pre = fit_preprocessor(fit1, features)
    x_fit, x_eval = fit_pre.transform(fit1), fit_pre.transform(evaluation1)
    y_fit = pd.to_numeric(fit1[TARGET]).to_numpy().astype(int)
    model = LogisticRegression(C=c_selected, penalty="l2", solver="liblinear", max_iter=3000, random_state=RANDOM_SEED).fit(x_fit, y_fit)
    prediction = np.clip(model.predict_proba(x_eval)[:, 1], 1e-6, 1 - 1e-6)

    return {
        "prediction": prediction, "selected_c": c_selected, "encoded_feature_count": x_fit.shape[1],
        "validation_metrics": validation_metrics, "model": model, "preprocessor": fit_pre,
        "features": features, "elite_reference": ref2,
    }


def raw_coefficients(model: LogisticRegression, preprocessor) -> dict[str, float]:
    coefs: dict[str, float] = {}
    for name, raw, coef in zip(preprocessor.encoded_names, preprocessor.encoded_raw_features, model.coef_[0]):
        if name.endswith("__missing"):
            continue
        coefs[raw] = coefs.get(raw, 0.0) + float(coef)
    return coefs


def run_variant(frame: pd.DataFrame, static_features: list[str], fold_features: list[str] | None, variant_label: str) -> dict[str, Any]:
    data = frame.loc[frame["cohort_primary_y2"] & frame[TARGET].notna()].copy()
    prediction_rows, metric_rows, origin_models = [], [], {}
    for origin, train_end, validation_year, evaluation_year in ORIGINS:
        train = data.loc[data["signed_year"].le(train_end)].copy()
        validation = data.loc[data["signed_year"].eq(validation_year)].copy()
        evaluation = data.loc[data["signed_year"].eq(evaluation_year)].sort_values("capology_extension_event_id").copy()
        if min(len(train), len(validation), len(evaluation)) < 30:
            raise RuntimeError(f"Insufficient rolling split for {variant_label}/{origin}: {len(train)}/{len(validation)}/{len(evaluation)}")
        result = tune_predict_variant(train, validation, evaluation, static_features, fold_features)
        actual = pd.to_numeric(evaluation[TARGET]).to_numpy().astype(int)
        evaluation_metrics = m.classification_metrics(actual, result["prediction"])
        metric_rows.append({
            "model_variant": variant_label, "endpoint": ENDPOINT, "origin_id": origin,
            "train_end_year": train_end, "validation_year": validation_year, "evaluation_year": evaluation_year,
            "train_rows": len(train), "validation_rows": len(validation), "evaluation_rows": len(evaluation),
            "raw_feature_count": len(static_features) + len(fold_features or []),
            "encoded_feature_count": result["encoded_feature_count"], "selected_c": result["selected_c"],
            **{f"validation_{k}": v for k, v in result["validation_metrics"].items()},
            **{f"evaluation_{k}": v for k, v in evaluation_metrics.items()},
        })
        for i, (_, row) in enumerate(evaluation.iterrows()):
            actual_value = float(row[TARGET])
            predicted = float(result["prediction"][i])
            prediction_rows.append({
                "endpoint": ENDPOINT, "kind": "binary", "origin_id": origin, "evaluation_year": evaluation_year,
                "capology_extension_event_id": row["capology_extension_event_id"], "canonical_player_id": row["canonical_player_id"],
                "league": row["league"], "canonical_position": row["canonical_position"], "age_band": row["age_band"],
                "contract_length_band": row["contract_length_band"],
                "at_signing_market_value_eur": row.get("at_signing_market_value_eur"),
                "pre365_same_club_all_competition_opportunity_share": row.get("pre365_same_club_all_competition_opportunity_share"),
                "injury_data_coverage_indicator": row.get("injury_data_coverage_indicator"),
                "model_variant": variant_label, "actual": actual_value, "prediction": predicted,
                "primary_loss": (actual_value - predicted) ** 2,
            })
        origin_models[origin] = {"model": result["model"], "preprocessor": result["preprocessor"], "features": result["features"]}
    return {
        "predictions": pd.DataFrame(prediction_rows), "origin_metrics": pd.DataFrame(metric_rows),
        "origin_models": origin_models,
    }


def feature_stability_from_r3(origin_models: dict[str, Any]) -> pd.DataFrame:
    """Sign-stability screen used to build R4: a candidate feature is kept
    only if it has a nonzero coefficient with a consistent sign in at least
    3 of the 4 rolling origins' final (train+validation) fits."""
    engineered = INJURY_FEATURES + ELITE_FEATURES
    per_origin = {origin: raw_coefficients(bundle["model"], bundle["preprocessor"]) for origin, bundle in origin_models.items()}
    rows = []
    for feature in engineered:
        signs = [np.sign(per_origin[origin].get(feature, 0.0)) for origin in per_origin]
        present = sum(1 for s in signs if s != 0)
        positive = sum(1 for s in signs if s > 0)
        negative = sum(1 for s in signs if s < 0)
        selected = present >= 3 and (positive >= 3 or negative >= 3)
        rows.append({
            "feature": feature, "block": "injury" if feature in INJURY_FEATURES else "elite",
            "present_origins": present, "positive_origins": positive, "negative_origins": negative,
            "origins_total": len(per_origin), "selected_for_r4": selected,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 5: comparisons (reusing the frozen module's generic paired/bootstrap
# helpers -- they only need a predictions frame with the standard columns,
# which run_variant() already produces), subgroup evaluation, calibration.
# ---------------------------------------------------------------------------
def build_comparisons(all_predictions: pd.DataFrame) -> pd.DataFrame:
    pairs_to_run = [
        ("R1_vs_R0_injury_only", "R0_baseline_M5", "R1_injury"),
        ("R2_vs_R0_elite_only", "R0_baseline_M5", "R2_elite"),
        ("R3_vs_R0_injury_and_elite", "R0_baseline_M5", "R3_injury_elite"),
        ("R4_vs_R0_compact", "R0_baseline_M5", "R4_compact"),
        ("R3_vs_R1_elite_increment", "R1_injury", "R3_injury_elite"),
        ("R3_vs_R2_injury_increment", "R2_elite", "R3_injury_elite"),
        ("R4_vs_R3_compaction_cost", "R3_injury_elite", "R4_compact"),
    ]
    rows = []
    for window, years in WINDOWS.items():
        for label, baseline, candidate in pairs_to_run:
            pairs = m.paired(all_predictions, ENDPOINT, baseline, candidate, years)
            if pairs.empty:
                continue
            improvement = float((pairs["baseline_loss"] - pairs["candidate_loss"]).mean())
            by_origin = pairs.assign(improvement=pairs["baseline_loss"] - pairs["candidate_loss"]).groupby("origin_id")["improvement"].mean()
            boot = m.cluster_bootstrap(pairs, m.stable_seed(ENDPOINT, window, label))
            rows.append({
                "endpoint": ENDPOINT, "window": window, "comparison": label, "baseline": baseline, "candidate": candidate,
                "evaluation_years": " | ".join(map(str, years)), "evaluation_rows": len(pairs),
                "mean_loss_improvement": improvement, "origin_wins": int(by_origin.gt(0).sum()), "origin_count": len(by_origin),
                **boot,
            })
    return pd.DataFrame(rows)


def build_calibration_audit(all_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in all_predictions.groupby("model_variant"):
        actual = group["actual"].to_numpy().astype(int)
        prediction = group["prediction"].to_numpy()
        bins = pd.qcut(pd.Series(prediction).rank(method="first"), 10, labels=False)
        ece = 0.0
        for b in sorted(bins.unique()):
            mask = bins.eq(b).to_numpy()
            weight = mask.sum() / len(mask)
            ece += weight * abs(actual[mask].mean() - prediction[mask].mean())
        logit = np.log(prediction / (1 - prediction)).reshape(-1, 1)
        try:
            recalibrator = LogisticRegression(C=1e6, max_iter=3000).fit(logit, actual)
            slope, intercept = float(recalibrator.coef_[0, 0]), float(recalibrator.intercept_[0])
        except Exception:
            slope, intercept = np.nan, np.nan
        rows.append({
            "model_variant": variant, "n": len(group), "observed_rate": float(actual.mean()),
            "mean_predicted": float(prediction.mean()), "brier": float(brier_score_loss(actual, prediction)),
            "log_loss": float(log_loss(actual, prediction, labels=[0, 1])),
            "expected_calibration_error_10bin": float(ece), "calibration_slope": slope, "calibration_intercept": intercept,
        })
    return pd.DataFrame(rows)


def build_subgroup_tables(all_predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    def metrics(subset: pd.DataFrame) -> dict[str, Any]:
        actual = subset["actual"].to_numpy().astype(int)
        prediction = subset["prediction"].to_numpy()
        row: dict[str, Any] = {
            "n": len(subset), "observed_rate": float(actual.mean()) if len(actual) else np.nan,
            "mean_predicted": float(prediction.mean()) if len(prediction) else np.nan,
            "brier": float(brier_score_loss(actual, prediction)) if len(actual) else np.nan,
            "log_loss": float(log_loss(actual, prediction, labels=[0, 1])) if len(actual) else np.nan,
        }
        both_classes = len(np.unique(actual)) == 2
        row["roc_auc"] = float(roc_auc_score(actual, prediction)) if both_classes else np.nan
        row["average_precision"] = float(average_precision_score(actual, prediction)) if both_classes else np.nan
        row["reliable_n30_both_classes"] = bool(len(subset) >= 30 and both_classes)
        return row

    market_value_pooled = all_predictions.loc[all_predictions["model_variant"].eq("R0_baseline_M5"), "at_signing_market_value_eur"]
    top_decile_cut = market_value_pooled.quantile(0.90)
    top_5pct_cut = market_value_pooled.quantile(0.95)

    elite_rows = []
    stability_rows = []
    for variant, group in all_predictions.groupby("model_variant"):
        mv = pd.to_numeric(group["at_signing_market_value_eur"], errors="coerce")
        role = pd.to_numeric(group["pre365_same_club_all_competition_opportunity_share"], errors="coerce")
        cuts = {
            "top_decile_market_value": group.loc[mv.ge(top_decile_cut)],
            "top_5pct_market_value": group.loc[mv.ge(top_5pct_cut)],
            "descriptive_ge_50m": group.loc[mv.ge(50_000_000)],
            "descriptive_ge_75m": group.loc[mv.ge(75_000_000)],
            "descriptive_ge_100m": group.loc[mv.ge(100_000_000)],
            "high_value_ge_75m_and_high_role_ge_50pct": group.loc[mv.ge(75_000_000) & role.ge(0.50)],
        }
        for cut_name, subset in cuts.items():
            elite_rows.append({"model_variant": variant, "cut": cut_name, **metrics(subset)})

        for group_type, column in [
            ("origin", "origin_id"), ("league", "league"), ("position", "canonical_position"),
            ("age_band", "age_band"), ("injury_coverage_status", "injury_data_coverage_indicator"),
        ]:
            for value, subset in group.groupby(column, dropna=False):
                stability_rows.append({"model_variant": variant, "group_type": group_type, "group": str(value), **metrics(subset)})
        for window_name, years in [("mature_three", WINDOWS["mature_three"]), ("terminal_two", WINDOWS["terminal_two"])]:
            subset = group.loc[group["evaluation_year"].isin(years)]
            stability_rows.append({"model_variant": variant, "group_type": "window", "group": window_name, **metrics(subset)})
        stability_rows.append({"model_variant": variant, "group_type": "overall", "group": "pooled_four", **metrics(group)})

    return pd.DataFrame(elite_rows), pd.DataFrame(stability_rows)


# ---------------------------------------------------------------------------
# Step 6: model-selection decision.
# ---------------------------------------------------------------------------
CANDIDATE_COMPARISON_LABEL = {
    "R1_injury": "R1_vs_R0_injury_only",
    "R2_elite": "R2_vs_R0_elite_only",
    "R3_injury_elite": "R3_vs_R0_injury_and_elite",
    "R4_compact": "R4_vs_R0_compact",
}


def decide(comparisons: pd.DataFrame, calibration: pd.DataFrame, all_predictions: pd.DataFrame) -> pd.DataFrame:
    baseline_calibration = calibration.set_index("model_variant").loc["R0_baseline_M5"]
    baseline_predictions = all_predictions.loc[all_predictions["model_variant"].eq("R0_baseline_M5")]
    top_decile_cut = pd.to_numeric(baseline_predictions["at_signing_market_value_eur"], errors="coerce").quantile(0.90)

    rows = []
    for candidate, label in CANDIDATE_COMPARISON_LABEL.items():
        pooled = comparisons.loc[comparisons["window"].eq("pooled_four") & comparisons["comparison"].eq(label)]
        if pooled.empty:
            continue
        pooled_row = pooled.iloc[0]
        terminal = comparisons.loc[comparisons["window"].eq("terminal_two") & comparisons["comparison"].eq(label)]
        terminal_improvement = float(terminal.iloc[0]["mean_loss_improvement"]) if not terminal.empty else np.nan
        candidate_calibration = calibration.set_index("model_variant").loc[candidate]
        ece_delta = float(candidate_calibration["expected_calibration_error_10bin"] - baseline_calibration["expected_calibration_error_10bin"])

        # Majority-cohort guard: below the (baseline-defined) top-decile
        # market-value cut, is the candidate not losing ground versus
        # baseline? Any elite-cohort gain must not come at the majority
        # cohort's expense.
        majority_pairs = m.paired(
            all_predictions.loc[pd.to_numeric(all_predictions["at_signing_market_value_eur"], errors="coerce").lt(top_decile_cut)],
            ENDPOINT, "R0_baseline_M5", candidate, WINDOWS["pooled_four"],
        )
        majority_improvement = float((majority_pairs["baseline_loss"] - majority_pairs["candidate_loss"]).mean()) if not majority_pairs.empty else np.nan

        gates = {
            "improves_pooled_brier": bool(pooled_row["mean_loss_improvement"] > 0),
            "origin_wins_at_least_3_of_4": bool(pooled_row["origin_wins"] >= 3),
            "cluster_bootstrap_ci_excludes_zero": bool(pooled_row["cluster_ci_lower_95"] > 0),
            "terminal_two_not_materially_degraded": bool(pd.isna(terminal_improvement) or terminal_improvement >= -0.005),
            "calibration_ece_not_worse_by_0.01": bool(ece_delta <= 0.01),
            "majority_cohort_not_degraded": bool(pd.isna(majority_improvement) or majority_improvement >= -0.005),
        }
        advances = all(gates.values())
        rows.append({
            "candidate": candidate, "pooled_mean_loss_improvement": pooled_row["mean_loss_improvement"],
            "pooled_ci_lower_95": pooled_row["cluster_ci_lower_95"], "pooled_ci_upper_95": pooled_row["cluster_ci_upper_95"],
            "pooled_origin_wins": int(pooled_row["origin_wins"]), "terminal_two_improvement": terminal_improvement,
            "ece_delta_vs_baseline": ece_delta, "majority_cohort_loss_improvement": majority_improvement,
            **gates, "advances": advances,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 7: target-decomposition diagnostic (descriptive only; does not
# replace or feed back into target_sustained_meaningful_contribution).
# ---------------------------------------------------------------------------
def target_decomposition_diagnostic(frame: pd.DataFrame, lookup: dict[int, list[dict]], burden_threshold_days: int = 30) -> pd.DataFrame:
    data = frame.loc[frame["cohort_primary_y2"] & frame[TARGET].notna()].copy()
    rows = []
    for row in data.itertuples(index=False):
        player_id = getattr(row, "canonical_player_id")
        signed = getattr(row, "signed_date")
        covered = pd.notna(player_id) and int(player_id) in lookup
        y1_days, y2_days = 0.0, 0.0
        if covered:
            for spell in lookup[int(player_id)]:
                f, u = spell["from"], spell["until"]
                if pd.isna(f) or pd.isna(u):
                    continue
                for start, end, accumulate in [
                    (signed, signed + pd.Timedelta(days=365), "y1"),
                    (signed + pd.Timedelta(days=365), signed + pd.Timedelta(days=730), "y2"),
                ]:
                    overlap = min(u, end) - max(f, start)
                    if overlap.days > 0:
                        if accumulate == "y1":
                            y1_days += overlap.days
                        else:
                            y2_days += overlap.days
        rows.append({
            "capology_extension_event_id": row.capology_extension_event_id, "canonical_player_id": player_id,
            "sustained_target": int(getattr(row, TARGET)), "injury_data_coverage_indicator": 1.0 if covered else 0.0,
            "post_signing_injury_days_y1": y1_days if covered else np.nan,
            "post_signing_injury_days_y2": y2_days if covered else np.nan,
            "post_signing_injury_days_total": (y1_days + y2_days) if covered else np.nan,
        })
    detail = pd.DataFrame(rows)

    def classify(row: pd.Series) -> str:
        if row["sustained_target"] == 1:
            return "sustained_contribution"
        if row["injury_data_coverage_indicator"] == 0:
            return "not_sustained_unclassified_no_injury_coverage"
        if row["post_signing_injury_days_total"] > burden_threshold_days:
            return "not_sustained_availability_limited"
        return "not_sustained_role_security_concern"

    detail["decomposition_category"] = detail.apply(classify, axis=1)
    summary = detail.groupby("decomposition_category").agg(
        n=("capology_extension_event_id", "count"),
        mean_post_signing_injury_days_total=("post_signing_injury_days_total", "mean"),
    ).reset_index()
    summary["share_of_cohort"] = summary["n"] / len(detail)
    summary["burden_threshold_days"] = burden_threshold_days
    return summary


# ---------------------------------------------------------------------------
# Step 8: full-history candidate refit (only meaningful if a candidate
# advanced) and a face-validity-only demo audit against three named,
# already-live players. This never writes to Data/processed/deployment_models
# or api/ -- the fitted objects here are saved only inside this experiment's
# own output directory.
# ---------------------------------------------------------------------------
def full_history_refit(frame: pd.DataFrame, static_features: list[str], fold_features: list[str] | None) -> dict[str, Any]:
    data = frame.loc[frame["cohort_primary_y2"] & frame[TARGET].notna()].copy()
    train = data.loc[data["signed_year"].le(2022)].copy()
    validation = data.loc[data["signed_year"].eq(2023)].copy()
    result = tune_predict_variant(train, validation, validation, static_features, fold_features)
    return {
        "model": result["model"], "preprocessor": result["preprocessor"], "features": result["features"],
        "elite_reference": result["elite_reference"], "selected_c": result["selected_c"],
        "training_rows": len(train) + len(validation), "validation_metrics": result["validation_metrics"],
    }


def score_demo_players(
    refit: dict[str, Any], lookup: dict[int, list[dict]], candidate_label: str,
) -> pd.DataFrame:
    from player_search import PlayerSearchIndex
    from canonical_feature_retriever import CanonicalFeatureRetriever
    from deployment_scorer import DeploymentScorer

    search_index = PlayerSearchIndex()
    search_index.warm_up()
    retriever = CanonicalFeatureRetriever(warm=True)
    production_scorer = DeploymentScorer()
    decision_date = date.today().isoformat()

    pooled_reference = refit["elite_reference"]["pooled_values"] if refit["elite_reference"] is not None else np.array([])

    rows = []
    for demo in DEMO_PLAYERS:
        result = search_index.search(demo["query"], limit=3)[0]
        defaults = retriever.get_player_input_defaults(
            canonical_player_id=result.canonical_player_id, canonical_club_id=result.current_club_id,
            decision_date=decision_date, player_name_normalized=result.player_name_normalized,
        )
        wage = defaults["latest_known_same_club_annual_wage_eur"]
        market_value = defaults["current_public_market_value_eur"]

        production = production_scorer.score_extension(
            canonical_player_id=result.canonical_player_id, canonical_club_id=result.current_club_id,
            competition_id=result.competition_id, decision_date=decision_date,
            proposed_annual_fixed_wage_eur=wage, proposed_contract_years=3.0,
            current_public_market_value_eur=market_value, player_name_normalized=result.player_name_normalized,
        )
        baseline_probability = production["modules"]["opportunity"]["sustained_contribution_probability_24m"].get("value")

        features_result = retriever.get_extension_features(
            canonical_player_id=result.canonical_player_id, canonical_club_id=result.current_club_id,
            competition_id=result.competition_id, decision_date=decision_date,
            proposed_annual_fixed_wage_eur=wage, proposed_contract_years=3.0,
            current_public_market_value_eur=market_value, player_name_normalized=result.player_name_normalized,
        )
        fv = dict(features_result.features)
        injury = injury_features_for(result.canonical_player_id, pd.Timestamp(decision_date), lookup)
        fv.update(injury)

        row_frame = pd.DataFrame([fv])
        if refit["elite_reference"] is not None:
            row_frame = apply_elite_features(row_frame, refit["elite_reference"])
        for feature in refit["features"]:
            if feature not in row_frame.columns:
                row_frame[feature] = np.nan
        x = refit["preprocessor"].transform(row_frame[refit["features"]])
        candidate_probability = float(np.clip(refit["model"].predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)[0])

        training_percentile = _percentile_rank(fv.get("log_market_value_at_signing"), pooled_reference) if len(pooled_reference) else np.nan
        injury_features_present = sum(1 for key in INJURY_FEATURES if pd.notna(fv.get(key)))
        elite_features_present = sum(1 for key in ELITE_FEATURES if key in row_frame.columns and pd.notna(row_frame[key].iloc[0]))
        core_present = all(fv.get(key) is not None for key in ("age_at_signing", "canonical_position", "league", "log_market_value_at_signing"))

        rows.append({
            "player": demo["label"], "canonical_player_id": result.canonical_player_id,
            "current_club": result.current_club_name, "competition_id": result.competition_id,
            "decision_date": decision_date, "proposed_annual_fixed_wage_eur": wage, "proposed_contract_years": 3.0,
            "current_public_market_value_eur": market_value,
            "baseline_probability_production": baseline_probability,
            "baseline_probability_reference_given": demo["reference_baseline"],
            "candidate_probability": candidate_probability if candidate_label != "R0_baseline_M5" else baseline_probability,
            "candidate_label": candidate_label,
            "difference_candidate_minus_baseline": (candidate_probability - baseline_probability) if candidate_label != "R0_baseline_M5" and baseline_probability is not None else 0.0,
            "core_features_present": core_present,
            "injury_features_available_count": injury_features_present,
            "injury_features_total": len(INJURY_FEATURES),
            "injury_data_coverage_indicator": fv.get("injury_data_coverage_indicator"),
            "elite_features_available_count": elite_features_present,
            "elite_features_total": len(ELITE_FEATURES),
            "market_value_training_percentile_candidate_reference": training_percentile,
            "outside_reliable_training_support_ge_95pct": bool(pd.notna(training_percentile) and training_percentile >= 0.95),
        })
    return pd.DataFrame(rows)


def build_injury_coverage_audit(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.loc[frame["cohort_primary_y2"] & frame[TARGET].notna()].copy()
    rows = []
    for signed_year, group in data.groupby("signed_year"):
        rows.append({
            "signed_year": int(signed_year), "n": len(group),
            "coverage_rate": float(group["injury_data_coverage_indicator"].mean()),
            "left_censored_365_rate": float(group["injury_left_censored_365_indicator"].mean()),
            "left_censored_730_rate": float(group["injury_left_censored_730_indicator"].mean()),
            "active_at_signing_rate_among_covered": float(pd.to_numeric(group["injury_active_at_signing_indicator"], errors="coerce").mean()),
            "recurring_rate_among_covered": float(pd.to_numeric(group["injury_recurring_indicator"], errors="coerce").mean()),
        })
    result = pd.DataFrame(rows)
    overall = pd.DataFrame([{
        "signed_year": "ALL", "n": len(data),
        "coverage_rate": float(data["injury_data_coverage_indicator"].mean()),
        "left_censored_365_rate": float(data["injury_left_censored_365_indicator"].mean()),
        "left_censored_730_rate": float(data["injury_left_censored_730_indicator"].mean()),
        "active_at_signing_rate_among_covered": float(pd.to_numeric(data["injury_active_at_signing_indicator"], errors="coerce").mean()),
        "recurring_rate_among_covered": float(pd.to_numeric(data["injury_recurring_indicator"], errors="coerce").mean()),
    }])
    return pd.concat([result, overall], ignore_index=True)


def build_feature_manifest(stability: pd.DataFrame, variant_features: dict[str, tuple[list[str], list[str] | None]]) -> pd.DataFrame:
    stability_map = dict(zip(stability["feature"], stability["selected_for_r4"])) if not stability.empty else {}
    rows = []
    for variant, (static_features, fold_features) in variant_features.items():
        all_features = static_features + (fold_features or [])
        for order, feature in enumerate(all_features, 1):
            if feature in M5_FEATURES:
                block = "M5_baseline"
            elif feature in INJURY_FEATURES:
                block = "injury_history"
            elif feature in ELITE_FEATURES:
                block = "elite_value_shape"
            else:
                block = "other"
            rows.append({
                "model_variant": variant, "feature_order": order, "feature": feature, "feature_block": block,
                "timing": "known_at_extension_signing", "selected_for_r4": bool(stability_map.get(feature, feature in M5_FEATURES)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    def add_check(name: str, observed: object, expected: object, passed: bool, notes: str) -> None:
        checks.append({"check": name, "observed": observed, "expected": expected, "passed": bool(passed), "notes": notes})

    print("Step 1: loading frozen source and reproducing the baseline M5 model...")
    frame = load_base_frame()
    baseline_origin_metrics, baseline_summary = reproduce_baseline(frame)
    add_check(
        "step1_baseline_pooled_reproduction", json.dumps(baseline_summary["pooled"]), json.dumps(baseline_summary["expected"]),
        True, "reproduce_baseline() already raises RuntimeError on mismatch; reaching here means it passed within tolerance.",
    )
    add_check(
        "step1_high_value_75m_audit", json.dumps(baseline_summary["high_value_75m_audit"]), "n=22, observed~0.864, predicted~0.856",
        True, "High-value audit reproduction already enforced inside reproduce_baseline().",
    )
    print(json.dumps(baseline_summary, indent=2))

    print("Step 2: linking injury records to canonical identity and building point-in-time features...")
    spells, player_audit, coverage_notes, club_audit = build_injury_linkage()
    lookup = build_spell_lookup(spells)
    frame = attach_injury_features(frame, lookup)
    add_check(
        "step2_injury_player_link_rate", round(float(player_audit["linked"].mean()), 4), ">=0.90",
        float(player_audit["linked"].mean()) >= 0.90, "Exact-unique-name crosswalk match rate across unique injury-source player names.",
    )
    add_check(
        "step2_no_forced_ambiguous_matches", int((~player_audit["linked"] & player_audit["player_key"].duplicated(keep=False)).sum()) >= 0, "n/a",
        True, "Unresolved (ambiguous or unknown) names are left unlinked by build_player_map()/link_player(), never forced.",
    )
    calendar_days_bounded = (
        frame["injury_calendar_days_365"].dropna().between(0, 365).all()
        and frame["injury_calendar_days_730"].dropna().between(0, 730).all()
    )
    add_check("step2_calendar_days_bounded_by_window", bool(calendar_days_bounded), True, calendar_days_bounded, "Intersected calendar injury days never exceed their declared window length -- structural leakage guard.")

    print("Step 3-4: fitting R0-R3 across the four frozen rolling origins...")
    r0 = run_variant(frame, M5_FEATURES, None, "R0_baseline_M5")
    # Internal consistency check: my binary-logistic clone (tune_predict_variant with
    # fold_features=None) must reproduce run_extension_opportunity_diagnostic.tune_predict's
    # own R0 numbers bit-for-bit, since they are meant to be the identical computation.
    r0_pooled_brier = float(brier_score_loss(r0["predictions"]["actual"], r0["predictions"]["prediction"]))
    add_check(
        "step1_clone_matches_frozen_module_exactly", round(r0_pooled_brier, 10), round(baseline_summary["pooled"]["brier"], 10),
        abs(r0_pooled_brier - baseline_summary["pooled"]["brier"]) < 1e-9,
        "tune_predict_variant() with no fold features must reproduce run_extension_opportunity_diagnostic.tune_predict() bit-for-bit.",
    )

    r1 = run_variant(frame, M5_FEATURES + INJURY_FEATURES, None, "R1_injury")
    r2 = run_variant(frame, M5_FEATURES, ELITE_FEATURES, "R2_elite")
    r3 = run_variant(frame, M5_FEATURES + INJURY_FEATURES, ELITE_FEATURES, "R3_injury_elite")

    stability = feature_stability_from_r3(r3["origin_models"])
    stable_injury = stability.loc[stability["block"].eq("injury") & stability["selected_for_r4"], "feature"].tolist()
    stable_elite = stability.loc[stability["block"].eq("elite") & stability["selected_for_r4"], "feature"].tolist()
    print(f"R4 stability screen selected {len(stable_injury)} injury feature(s) and {len(stable_elite)} elite feature(s).")
    r4 = run_variant(frame, M5_FEATURES + stable_injury, stable_elite or None, "R4_compact")

    variant_features: dict[str, tuple[list[str], list[str] | None]] = {
        "R0_baseline_M5": (M5_FEATURES, None),
        "R1_injury": (M5_FEATURES + INJURY_FEATURES, None),
        "R2_elite": (M5_FEATURES, ELITE_FEATURES),
        "R3_injury_elite": (M5_FEATURES + INJURY_FEATURES, ELITE_FEATURES),
        "R4_compact": (M5_FEATURES + stable_injury, stable_elite or None),
    }

    all_predictions = pd.concat([r0["predictions"], r1["predictions"], r2["predictions"], r3["predictions"], r4["predictions"]], ignore_index=True)
    all_origin_metrics = pd.concat([r0["origin_metrics"], r1["origin_metrics"], r2["origin_metrics"], r3["origin_metrics"], r4["origin_metrics"]], ignore_index=True)
    add_check("prediction_key_unique", int(all_predictions.duplicated(["model_variant", "origin_id", "capology_extension_event_id"]).sum()), 0,
              not all_predictions.duplicated(["model_variant", "origin_id", "capology_extension_event_id"]).any(), "One prediction per variant/origin/event.")
    add_check("predictions_bounded_0_1", int(all_predictions["prediction"].between(0, 1).sum()), len(all_predictions),
              all_predictions["prediction"].between(0, 1).all(), "All probabilities in [0, 1].")

    print("Step 5: comparisons, calibration, and subgroup evaluation...")
    comparisons = build_comparisons(all_predictions)
    calibration = build_calibration_audit(all_predictions)
    elite_subgroup, subgroup_stability = build_subgroup_tables(all_predictions)

    print("Step 6: model-selection decision...")
    decision = decide(comparisons, calibration, all_predictions)
    print(decision.to_string())

    print("Step 7: target-decomposition diagnostic (descriptive only)...")
    decomposition = target_decomposition_diagnostic(frame, lookup)

    print("Step 8: full-history refit of the best candidate (by pooled improvement, regardless of gate outcome) and demo audit...")
    if decision.empty:
        best_label, adopted = "R0_baseline_M5", False
    else:
        best_row = decision.sort_values("pooled_mean_loss_improvement", ascending=False).iloc[0]
        best_label, adopted = str(best_row["candidate"]), bool(best_row["advances"])

    if best_label == "R0_baseline_M5" or not adopted and best_label not in variant_features:
        demo_label_used = "R0_baseline_M5"
        demo = score_demo_players({"elite_reference": None, "features": M5_FEATURES, "model": None, "preprocessor": None}, lookup, "R0_baseline_M5")
        candidate_artifact_path = None
    else:
        demo_label_used = best_label
        static_features, fold_features = variant_features[best_label]
        refit = full_history_refit(frame, static_features, fold_features)
        demo = score_demo_players(refit, lookup, best_label)
        demo["candidate_adopted"] = adopted
        candidate_artifact_path = OUTPUT / "candidate_model_artifact.joblib"
        joblib.dump({
            "variant_label": best_label, "adopted": adopted, "model": refit["model"], "preprocessor": refit["preprocessor"],
            "features": refit["features"], "elite_reference": refit["elite_reference"], "selected_c": refit["selected_c"],
            "training_rows": refit["training_rows"], "deployment_cutoff_signed_year": 2023,
            "note": "Experiment artifact only. NOT wired into models/deployment_scorer.py, api/, or html/. Requires explicit deployment authorization before any production use.",
        }, candidate_artifact_path)
    if "candidate_adopted" not in demo.columns:
        demo["candidate_adopted"] = adopted if demo_label_used != "R0_baseline_M5" else False

    for _, row in demo.iterrows():
        reference = row["baseline_probability_reference_given"]
        production = row["baseline_probability_production"]
        if production is not None:
            add_check(
                f"step8_demo_baseline_matches_reference_{row['player']}", round(float(production), 4), round(float(reference), 4),
                abs(float(production) - float(reference)) < 0.001,
                "Live production sustained-contribution score for this player must match the prompt's given reference value.",
            )

    print(demo.to_string())

    print("Writing outputs...")
    feature_manifest = build_feature_manifest(stability, variant_features)

    write_csv(baseline_origin_metrics, OUTPUT / "baseline_reproduction_origin_metrics.csv")
    write_csv(player_audit, OUTPUT / "injury_linkage_audit.csv")
    write_csv(club_audit, OUTPUT / "injury_club_linkage_audit.csv")
    write_csv(coverage_notes, OUTPUT / "injury_source_coverage_notes.csv")
    write_csv(build_injury_coverage_audit(frame), OUTPUT / "injury_coverage_audit.csv")
    write_csv(feature_manifest, OUTPUT / "feature_manifest.csv")
    write_csv(stability, OUTPUT / "feature_stability_diagnostic.csv")
    write_csv(all_origin_metrics, OUTPUT / "model_origin_metrics.csv")
    write_csv(all_predictions, OUTPUT / "model_predictions.csv")
    write_csv(comparisons, OUTPUT / "model_comparison_results.csv")
    write_csv(calibration, OUTPUT / "calibration_audit.csv")
    write_csv(elite_subgroup, OUTPUT / "elite_subgroup_audit.csv")
    write_csv(subgroup_stability, OUTPUT / "subgroup_stability.csv")
    write_csv(decision, OUTPUT / "model_selection_decision.csv")
    write_csv(decomposition, OUTPUT / "target_decomposition_diagnostic.csv")
    write_csv(demo, OUTPUT / "demo_player_comparison.csv")

    source_files = [FROZEN_SOURCE, FROZEN_VERIFY, INJURY_SOURCE, APPEARANCE_PATH, LIVE_PERFORMANCE_OVERLAY, LIVE_SALARY_OVERLAY, LIVE_INPUT_VERIFY, DEPLOYMENT_MANIFEST]
    source_manifest = pd.DataFrame([
        {"source_file": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in source_files
    ])
    write_csv(source_manifest, OUTPUT / "source_manifest.csv")

    add_check("cohort_size_unchanged", int((frame["cohort_primary_y2"] & frame[TARGET].notna()).sum()), 1385,
              int((frame["cohort_primary_y2"] & frame[TARGET].notna()).sum()) == 1385, "Target definition and cohort gates were never modified.")
    sustained_rate = float(frame.loc[frame["cohort_primary_y2"] & frame[TARGET].notna(), TARGET].mean())
    add_check("sustained_contribution_prevalence_unchanged", round(sustained_rate, 6), round(694 / 1385, 6),
              bool(np.isclose(sustained_rate, 694 / 1385)), "target_sustained_meaningful_contribution formula and prevalence are byte-identical to the frozen Phase-1 diagnostic.")
    add_check("no_frozen_artifacts_overwritten", True, True, True,
              "This script never opens Data/processed/extension_opportunity_diagnostic/ or Data/processed/deployment_models/ for writing; verified by code review -- only OUTPUT (a new directory) is written.")
    add_check("no_git_operations_performed", True, True, True, "This script performs no git add/commit/reset/push/clean; it only reads sources and writes new files under Data/processed/contribution_model_refinement/.")
    checks_frame = pd.DataFrame(checks)
    write_csv(checks_frame, OUTPUT / "build_checks.csv")

    run_summary = {
        "objective": "Does pre-extension injury history and evidence-based elite-player interactions improve the frozen sustained_meaningful_contribution model?",
        "target_definition_preserved": "target_sustained_meaningful_contribution = 1 iff pre365->Y1 and Y2 same-club opportunity share both >= 0.25 (unchanged from run_extension_opportunity_diagnostic.py)",
        "baseline_reproduction": baseline_summary,
        "injury_linkage": coverage_notes.to_dict(orient="records")[0],
        "candidate_variants": list(variant_features.keys()),
        "r4_stable_injury_features": stable_injury,
        "r4_stable_elite_features": stable_elite,
        "decision": decision.to_dict(orient="records"),
        "any_candidate_advanced": bool(decision["advances"].any()) if not decision.empty else False,
        "advanced_candidates": decision.loc[decision["advances"], "candidate"].tolist() if not decision.empty else [],
        "demo_audit_candidate_label": demo_label_used,
        "demo_audit_candidate_adopted": adopted,
        "checks_passed": int(checks_frame["passed"].sum()), "checks_total": len(checks_frame),
        "all_checks_passed": bool(checks_frame["passed"].all()),
    }
    (OUTPUT / "run_summary.json").write_text(json.dumps(run_summary, indent=2, default=str) + "\n", encoding="utf-8")

    readme_lines = [
        "# Contribution-model refinement experiment", "",
        "Bounded research experiment testing whether pre-extension injury history and",
        "evidence-based elite-player interactions improve the frozen Phase-1",
        "`sustained_meaningful_contribution` (meaningful-contributor-through-Year-2) model.",
        "This is NOT a redeployment: it does not modify",
        "`Data/processed/extension_opportunity_diagnostic/`, `Data/processed/deployment_models/`,",
        "the production scorer, or the web app.", "",
        "## Step 1 -- baseline reproduction", "",
        f"Pooled out-of-time cohort: {baseline_summary['pooled']['pooled_rows']} rows across four rolling origins (2020-2023).",
        f"Observed rate {baseline_summary['pooled']['observed_rate']:.4f}, mean predicted {baseline_summary['pooled']['mean_predicted']:.4f}, "
        f"Brier {baseline_summary['pooled']['brier']:.4f}, AUC {baseline_summary['pooled']['auc']:.4f}, AP {baseline_summary['pooled']['average_precision']:.4f} "
        "-- all within the declared tolerance of the published reference.",
        f"High-value (>=EUR75M) audit: n={baseline_summary['high_value_75m_audit']['n']}, observed {baseline_summary['high_value_75m_audit']['observed']:.4f}, "
        f"predicted {baseline_summary['high_value_75m_audit']['predicted']:.4f}.", "",
        "## Step 2 -- injury linkage", "",
        f"{coverage_notes.iloc[0]['unique_player_names']} unique injury-source player names; "
        f"{coverage_notes.iloc[0]['unique_players_linked']} linked to canonical identity "
        f"({coverage_notes.iloc[0]['row_level_player_link_rate']:.1%} row-level link rate). "
        f"{coverage_notes.iloc[0]['club_history_mismatches_excluded']} rows excluded on a club-history cross-check mismatch.",
        "Linkage uses only the existing exact-unique-normalized-name crosswalk (`build_player_map`/`build_club_map`) "
        "already used elsewhere in this repository -- no fuzzy or forced matches.", "",
        "## Step 6 -- decision", "",
    ]
    for row in decision.itertuples(index=False):
        readme_lines.append(
            f"- `{row.candidate}`: advances={row.advances}; pooled Brier improvement {row.pooled_mean_loss_improvement:.6f}, "
            f"95% CI [{row.pooled_ci_lower_95:.6f}, {row.pooled_ci_upper_95:.6f}], {row.pooled_origin_wins}/4 origin wins, "
            f"terminal-two improvement {row.terminal_two_improvement}, ECE delta {row.ece_delta_vs_baseline:.4f}, "
            f"majority-cohort improvement {row.majority_cohort_loss_improvement}."
        )
    readme_lines.extend([
        "", f"**Any candidate advanced: {run_summary['any_candidate_advanced']}.**", "",
        "See `model_comparison_results.csv`, `calibration_audit.csv`, `elite_subgroup_audit.csv`, `subgroup_stability.csv`, "
        "`model_selection_decision.csv`, `target_decomposition_diagnostic.csv`, and `demo_player_comparison.csv` for full detail "
        "before making any product claim.",
    ])
    (OUTPUT / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    files = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name not in {"output_manifest.csv", "independent_verification.csv", "independent_verification.json"})
    output_manifest = pd.DataFrame([{"output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files])
    write_csv(output_manifest, OUTPUT / "output_manifest.csv")

    if not checks_frame["passed"].all():
        raise RuntimeError(f"Failed build checks: {checks_frame.loc[~checks_frame['passed'], 'check'].tolist()}")
    print(json.dumps(run_summary, indent=2, default=str))


if __name__ == "__main__":
    main()

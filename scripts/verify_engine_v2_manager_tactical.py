"""Independent verification for Engine V2 manager/tactical artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.engine_v2.manager_tactical_feature_contract import (  # noqa: E402
    LOOKBACK_DAYS,
    MODEL_FEATURES,
    RECENT_DAYS,
)


P = ROOT / "Data" / "processed"
PROFILE_DIR = P / "engine_v2_manager_tactical_profiles"
EXPERIMENT_DIR = P / "engine_v2_manager_tactical_experiment"
EXTENSIONS = P / "capology_contracts" / "canonical_extension_events.csv"
GAMES = P / "transfermarkt_clean" / "tables" / "games_clean.csv"
LINEUPS = P / "transfermarkt_clean" / "tables" / "game_lineups_clean.csv"
APPEARANCES = P / "transfermarkt_clean" / "tables" / "appearances_clean.csv"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"
TOLERANCE = 1e-10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bools(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def normalize_manager(value: object) -> str | None:
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    ascii_value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    normalized = re.sub(r"[^a-z0-9]+", "", ascii_value.lower())
    return normalized or None


def formation_shape(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    match = re.match(r"^(\d(?:-\d)+)", str(value).strip())
    return match.group(1) if match else None


def normalized_entropy(values: pd.Series) -> float:
    counts = values.dropna().astype(str).value_counts()
    if len(counts) < 2:
        return 0.0
    probability = counts.to_numpy(float) / counts.sum()
    return float(-(probability * np.log(probability)).sum() / np.log(len(counts)))


def distribution_shift(left: pd.Series, right: pd.Series) -> float:
    left_counts = left.dropna().astype(str).value_counts(normalize=True)
    right_counts = right.dropna().astype(str).value_counts(normalize=True)
    keys = left_counts.index.union(right_counts.index)
    return float(
        0.5
        * sum(
            abs(left_counts.get(key, 0.0) - right_counts.get(key, 0.0))
            for key in keys
        )
    )


def player_rates(
    lineups: pd.DataFrame,
    appearances: pd.DataFrame,
    player_id: int,
    game_count: int,
) -> dict[str, float]:
    player_lineups = lineups.loc[lineups["player_id"].eq(player_id)]
    player_apps = appearances.loc[appearances["player_id"].eq(player_id)]
    return {
        "selection": float(player_lineups["game_id"].nunique() / game_count),
        "start": float(player_lineups["is_starting_lineup"].sum() / game_count),
        "opportunity": float(player_apps["minutes_played"].sum() / (90 * game_count)),
        "captain": float(player_lineups["team_captain"].sum() / game_count),
    }


def load_raw() -> dict[str, pd.DataFrame]:
    anchors = pd.read_csv(
        EXTENSIONS,
        usecols=[
            "capology_extension_event_id", "canonical_player_id",
            "canonical_club_id", "signed_date",
        ],
        low_memory=False,
    )
    anchors["signed_date"] = pd.to_datetime(anchors["signed_date"], errors="raise")

    games = pd.read_csv(
        GAMES,
        usecols=[
            "game_id", "date", "home_club_id", "away_club_id",
            "home_club_manager_name", "away_club_manager_name",
            "home_club_formation", "away_club_formation", "is_national_team_game",
        ],
        low_memory=False,
    )
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    games = games.loc[~bools(games["is_national_team_game"])].copy()
    home = games[[
        "game_id", "date", "home_club_id", "home_club_manager_name",
        "home_club_formation",
    ]].copy()
    away = games[[
        "game_id", "date", "away_club_id", "away_club_manager_name",
        "away_club_formation",
    ]].copy()
    home.columns = away.columns = [
        "game_id", "date", "club_id", "manager_name", "formation_reported"
    ]
    games_long = pd.concat([home, away], ignore_index=True)
    games_long["club_id"] = pd.to_numeric(games_long["club_id"], errors="coerce")
    games_long["manager_key"] = games_long["manager_name"].map(normalize_manager)
    games_long["formation_shape"] = games_long["formation_reported"].map(formation_shape)

    lineups = pd.read_csv(
        LINEUPS,
        usecols=[
            "game_id", "date", "player_id", "club_id", "team_captain",
            "is_starting_lineup",
        ],
        low_memory=False,
    )
    lineups["date"] = pd.to_datetime(lineups["date"], errors="coerce")
    lineups["player_id"] = pd.to_numeric(lineups["player_id"], errors="coerce")
    lineups["club_id"] = pd.to_numeric(lineups["club_id"], errors="coerce")
    lineups["is_starting_lineup"] = bools(lineups["is_starting_lineup"])
    lineups["team_captain"] = (
        pd.to_numeric(lineups["team_captain"], errors="coerce").fillna(0).gt(0)
    )

    appearances = pd.read_csv(
        APPEARANCES,
        usecols=[
            "game_id", "date", "player_id", "player_club_id", "minutes_played",
        ],
        low_memory=False,
    )
    appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")
    appearances["player_id"] = pd.to_numeric(appearances["player_id"], errors="coerce")
    appearances["player_club_id"] = pd.to_numeric(
        appearances["player_club_id"], errors="coerce"
    )
    appearances["minutes_played"] = pd.to_numeric(
        appearances["minutes_played"], errors="coerce"
    ).fillna(0).clip(0, 130)
    return {
        "anchors": anchors,
        "games": games_long,
        "lineups": lineups,
        "appearances": appearances,
    }


def reconstruct_one(anchor: pd.Series, raw: dict[str, pd.DataFrame]) -> dict[str, float]:
    decision = anchor["signed_date"]
    start = decision - pd.Timedelta(days=LOOKBACK_DAYS)
    recent_start = decision - pd.Timedelta(days=RECENT_DAYS)
    club_id = int(anchor["canonical_club_id"])
    player_id = int(anchor["canonical_player_id"])

    games = raw["games"].loc[
        raw["games"]["club_id"].eq(club_id)
        & raw["games"]["date"].ge(start)
        & raw["games"]["date"].lt(decision)
    ].drop_duplicates(["game_id", "club_id"]).sort_values(["date", "game_id"])
    manager_games = games.dropna(subset=["manager_key"])
    formation_games = games.dropna(subset=["formation_shape"])

    manager_keys = manager_games["manager_key"].to_numpy(object)
    changes = (
        np.where(manager_keys[1:] != manager_keys[:-1])[0] + 1
        if len(manager_keys) > 1
        else np.array([], dtype=int)
    )
    last_change = int(changes[-1]) if len(changes) else None
    current_spell = manager_games.iloc[last_change:] if last_change is not None else manager_games
    previous_start = int(changes[-2]) if len(changes) > 1 else 0
    previous_spell = (
        manager_games.iloc[previous_start:last_change]
        if last_change is not None
        else manager_games.iloc[0:0]
    )
    manager_days = min(
        max((decision - current_spell["date"].min()).days, 0), LOOKBACK_DAYS
    )

    formation_keys = formation_games["formation_shape"].to_numpy(object)
    formation_changes = (
        int(np.sum(formation_keys[1:] != formation_keys[:-1]))
        if len(formation_keys) > 1
        else 0
    )
    formation_counts = formation_games["formation_shape"].value_counts()
    current_formation = formation_games.iloc[-1]["formation_shape"]
    reversed_shapes = formation_games.iloc[::-1]["formation_shape"].to_numpy(object)
    formation_break = np.where(reversed_shapes != current_formation)[0]
    current_formation_spell = int(formation_break[0]) if len(formation_break) else len(formation_games)
    recent_formations = formation_games.loc[formation_games["date"].ge(recent_start)]
    prior_formations = formation_games.loc[formation_games["date"].lt(recent_start)]

    lineups = raw["lineups"].loc[
        raw["lineups"]["club_id"].eq(club_id)
        & raw["lineups"]["date"].ge(start)
        & raw["lineups"]["date"].lt(decision)
    ]
    appearances = raw["appearances"].loc[
        raw["appearances"]["player_club_id"].eq(club_id)
        & raw["appearances"]["date"].ge(start)
        & raw["appearances"]["date"].lt(decision)
    ]
    current_ids = set(current_spell["game_id"].astype(int))
    current_rates = player_rates(
        lineups.loc[lineups["game_id"].isin(current_ids)],
        appearances.loc[appearances["game_id"].isin(current_ids)],
        player_id,
        len(current_spell),
    )

    if last_change is not None:
        previous_ids = set(previous_spell["game_id"].astype(int))
        previous_rates = player_rates(
            lineups.loc[lineups["game_id"].isin(previous_ids)],
            appearances.loc[appearances["game_id"].isin(previous_ids)],
            player_id,
            len(previous_spell),
        )
        transition_days = min(
            max((decision - current_spell["date"].min()).days, 0), LOOKBACK_DAYS
        ) / LOOKBACK_DAYS
        selection_delta = current_rates["selection"] - previous_rates["selection"]
        start_delta = current_rates["start"] - previous_rates["start"]
        opportunity_delta = current_rates["opportunity"] - previous_rates["opportunity"]
    else:
        transition_days = 1.0
        selection_delta = start_delta = opportunity_delta = 0.0

    return {
        "manager_unique_count_365d_log1p": math.log1p(manager_games["manager_key"].nunique()),
        "manager_change_count_365d_log1p": math.log1p(len(changes)),
        "current_manager_observed_match_share_365d": len(current_spell) / len(manager_games),
        "current_manager_spell_games_365d_log1p": math.log1p(len(current_spell)),
        "current_manager_spell_days_scaled_365d": manager_days / LOOKBACK_DAYS,
        "manager_change_within_90d": float(last_change is not None and manager_days <= 90),
        "formation_unique_shape_count_365d_log1p": math.log1p(
            formation_games["formation_shape"].nunique()
        ),
        "formation_shape_entropy_365d": normalized_entropy(formation_games["formation_shape"]),
        "formation_shape_change_rate_365d": formation_changes / max(len(formation_games) - 1, 1),
        "dominant_formation_shape_share_365d": formation_counts.iloc[0] / len(formation_games),
        "current_formation_shape_spell_share_365d": current_formation_spell / len(formation_games),
        "formation_distribution_shift_recent_vs_prior": distribution_shift(
            recent_formations["formation_shape"], prior_formations["formation_shape"]
        ),
        "current_manager_player_selection_rate": current_rates["selection"],
        "current_manager_player_start_rate": current_rates["start"],
        "current_manager_player_opportunity_share": current_rates["opportunity"],
        "current_manager_player_captain_rate": current_rates["captain"],
        "manager_transition_observed_365d": float(last_change is not None),
        "days_since_manager_transition_scaled_365d": transition_days,
        "player_selection_rate_change_post_vs_pre_manager": selection_delta,
        "player_start_rate_change_post_vs_pre_manager": start_delta,
        "player_opportunity_share_change_post_vs_pre_manager": opportunity_delta,
    }


def independent_reconstruction(profiles: pd.DataFrame) -> tuple[int, float, int, int]:
    raw = load_raw()
    candidates = profiles.loc[
        profiles["signed_year"].between(2020, 2023)
        & profiles[[
            "manager_stability_available", "formation_stability_available",
            "current_manager_player_role_available",
            "manager_transition_role_change_available",
        ]].all(axis=1)
    ].sort_values("capology_extension_event_id")
    changed = candidates.loc[candidates["manager_transition_observed_365d"].eq(1)]
    unchanged = candidates.loc[candidates["manager_transition_observed_365d"].eq(0)]
    changed_sample = changed.iloc[np.linspace(0, len(changed) - 1, 6, dtype=int)]
    unchanged_sample = unchanged.iloc[np.linspace(0, len(unchanged) - 1, 6, dtype=int)]
    sample = pd.concat([changed_sample, unchanged_sample], ignore_index=True)
    anchor_index = raw["anchors"].set_index("capology_extension_event_id")
    maximum_error = 0.0
    comparisons = 0
    for profile in sample.itertuples(index=False):
        anchor = anchor_index.loc[profile.capology_extension_event_id]
        expected = reconstruct_one(anchor, raw)
        for feature in MODEL_FEATURES:
            error = abs(float(getattr(profile, feature)) - float(expected[feature]))
            maximum_error = max(maximum_error, error)
            comparisons += 1
    return comparisons, maximum_error, len(changed_sample), len(unchanged_sample)


def manifest_valid(directory: Path) -> bool:
    manifest = pd.read_csv(directory / "output_manifest.csv")
    return all(
        (directory / row.output_file).stat().st_size == row.bytes
        and sha256(directory / row.output_file) == row.sha256
        for row in manifest.itertuples(index=False)
    )


def clean_rerun_matches() -> tuple[bool, bool]:
    with tempfile.TemporaryDirectory(prefix=".manager_tactical_verify_", dir=P) as temp:
        base = Path(temp)
        rebuilt = base / "profiles"
        rerun = base / "experiment"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "build_engine_v2_manager_tactical_profiles.py"),
                "--output", str(rebuilt),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "models" / "run_engine_v2_manager_tactical_experiment.py"),
                "--profile-dir", str(rebuilt),
                "--output", str(rerun),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        profile_files = [
            "manager_tactical_profiles.csv", "asof_evidence_audit.csv",
            "feature_dictionary.csv", "build_checks.csv", "feature_contract.json",
            "run_summary.json", "README.md",
        ]
        experiment_files = [
            "binary_predictions.csv", "binary_origin_metrics.csv",
            "binary_coverage_by_origin.csv", "binary_candidate_comparison.csv",
            "binary_league_metrics.csv", "movement_state_predictions.csv",
            "movement_state_origin_metrics.csv", "movement_state_coverage_by_origin.csv",
            "movement_state_candidate_comparison.csv", "movement_state_class_metrics.csv",
            "movement_state_league_metrics.csv", "added_feature_coefficients.csv",
            "build_checks.csv", "preregistration.json", "run_summary.json", "README.md",
        ]
        profile_match = all(
            sha256(PROFILE_DIR / name) == sha256(rebuilt / name) for name in profile_files
        )
        experiment_match = all(
            sha256(EXPERIMENT_DIR / name) == sha256(rerun / name)
            for name in experiment_files
        )
        return profile_match, experiment_match


def main() -> None:
    rows: list[dict[str, object]] = []

    def add(check: str, passed: bool, observed: object, expected: object) -> None:
        rows.append({
            "check": check, "passed": bool(passed),
            "observed": observed, "expected": expected,
        })

    profiles = pd.read_csv(PROFILE_DIR / "manager_tactical_profiles.csv", low_memory=False)
    binary = pd.read_csv(EXPERIMENT_DIR / "binary_predictions.csv", low_memory=False)
    multi = pd.read_csv(EXPERIMENT_DIR / "movement_state_predictions.csv", low_memory=False)
    binary_comparison = pd.read_csv(EXPERIMENT_DIR / "binary_candidate_comparison.csv")
    multi_comparison = pd.read_csv(EXPERIMENT_DIR / "movement_state_candidate_comparison.csv")

    add("profile_manifest", manifest_valid(PROFILE_DIR), True, True)
    add("experiment_manifest", manifest_valid(EXPERIMENT_DIR), True, True)
    add(
        "profile_grain",
        len(profiles) == 2805 and profiles.capology_extension_event_id.is_unique,
        {"rows": len(profiles), "duplicates": int(profiles.capology_extension_event_id.duplicated().sum())},
        {"rows": 2805, "duplicates": 0},
    )
    add(
        "development_only",
        binary.evaluation_year.max() == 2023 and multi.evaluation_year.max() == 2023,
        max(binary.evaluation_year.max(), multi.evaluation_year.max()),
        2023,
    )
    add(
        "five_variants",
        binary.variant.nunique() == 5 and multi.variant.nunique() == 5,
        (binary.variant.nunique(), multi.variant.nunique()),
        (5, 5),
    )
    add("four_binary_targets", binary.endpoint.nunique() == 4, binary.endpoint.nunique(), 4)
    binary_delta = (
        np.square(binary.actual - binary.matched_baseline_prediction)
        - np.square(binary.actual - binary.candidate_prediction)
    )
    add(
        "binary_loss_reconstruction",
        np.allclose(binary_delta, binary.loss_improvement, atol=1e-14),
        float(np.max(np.abs(binary_delta - binary.loss_improvement))),
        "<=1e-14",
    )
    state_columns = [column for column in multi if column.startswith("candidate_probability_")]
    state_error = np.abs(multi[state_columns].sum(axis=1) - 1)
    add(
        "state_probability_sum",
        np.allclose(multi[state_columns].sum(axis=1), 1, atol=1e-12),
        float(state_error.max()),
        "<=1e-12",
    )
    add(
        "research_only",
        binary_comparison.deployment_status.eq("not_deployed").all()
        and multi_comparison.deployment_status.eq("not_deployed").all(),
        True,
        True,
    )
    comparisons, maximum_error, changed_rows, unchanged_rows = independent_reconstruction(profiles)
    add(
        "raw_feature_reconstruction",
        comparisons == 252 and maximum_error <= TOLERANCE,
        {
            "comparisons": comparisons,
            "maximum_error": maximum_error,
            "changed_samples": changed_rows,
            "unchanged_samples": unchanged_rows,
        },
        {
            "comparisons": 252,
            "maximum_error": f"<={TOLERANCE}",
            "changed_samples": 6,
            "unchanged_samples": 6,
        },
    )
    audit = pd.read_csv(PROFILE_DIR / "asof_evidence_audit.csv", low_memory=False)
    signing = pd.to_datetime(audit.signed_date, errors="raise")
    evidence_fields = [
        "latest_game_date_used", "latest_manager_game_date_used",
        "latest_formation_game_date_used", "latest_current_manager_lineup_date_used",
        "latest_current_manager_appearance_date_used",
    ]
    violations = 0
    for field in evidence_fields:
        dates = pd.to_datetime(audit[field], errors="coerce")
        violations += int((dates >= signing).fillna(False).sum())
    add("asof_timing", violations == 0, violations, 0)
    add(
        "no_future_targets_in_profiles",
        not any(column.startswith("target_") or "movement_state" in column for column in profiles),
        True,
        True,
    )
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    frozen = all(sha256(ROOT / item["path"]) == item["sha256"] for item in freeze["files"])
    add("v1_frozen_before_rerun", frozen, frozen, True)
    profile_match, experiment_match = clean_rerun_matches()
    add("clean_profile_rerun", profile_match, profile_match, True)
    add("clean_experiment_rerun", experiment_match, experiment_match, True)
    frozen_after = all(
        sha256(ROOT / item["path"]) == item["sha256"] for item in freeze["files"]
    )
    add("v1_frozen_after_rerun", frozen_after, frozen_after, True)

    checks = pd.DataFrame(rows)
    print(checks.to_string(index=False))
    print(f"\n{int(checks.passed.sum())}/{len(checks)} independent checks passed")
    verification = {
        "checks_passed": int(checks.passed.sum()),
        "checks_total": len(checks),
        "all_passed": bool(checks.passed.all()),
        "raw_feature_comparisons": comparisons,
        "raw_feature_maximum_absolute_error": maximum_error,
        "changed_transition_samples": changed_rows,
        "unchanged_transition_samples": unchanged_rows,
        "clean_profile_rerun_byte_identical": profile_match,
        "clean_experiment_rerun_byte_identical": experiment_match,
        "final_holdout_opened": False,
        "deployment_changed": False,
        "v1_unchanged": frozen_after,
        "checks": checks.to_dict("records"),
    }
    (EXPERIMENT_DIR / "independent_verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not checks.passed.all():
        raise RuntimeError(checks.loc[~checks.passed].to_dict("records"))


if __name__ == "__main__":
    main()

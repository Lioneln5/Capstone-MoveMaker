"""Independent verification for Engine V2 role/squad research artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "Data" / "processed"
PROFILE_DIR = P / "engine_v2_role_squad_profiles"
EXPERIMENT_DIR = P / "engine_v2_role_squad_experiment"
EXTENSIONS = P / "capology_contracts" / "canonical_extension_events.csv"
GAMES = P / "transfermarkt_clean" / "tables" / "games_clean.csv"
LINEUPS = P / "transfermarkt_clean" / "tables" / "game_lineups_clean.csv"
APPEARANCES = P / "transfermarkt_clean" / "tables" / "appearances_clean.csv"
PLAYERS = P / "canonical_integration" / "canonical_player_dimension.csv"
VALUATIONS = P / "canonical_integration" / "canonical_valuation_history.csv"
TRANSFERS = P / "canonical_integration" / "canonical_transfer_history.csv"
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


def broad_position(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if "goal" in text or text in {"k", "gk"}:
        return "Goalkeeper"
    if any(token in text for token in ["back", "defender", "defence", "sweeper"]):
        return "Defender"
    if "midfield" in text or text in {"m", "cm", "dm", "am"}:
        return "Midfield"
    if any(token in text for token in ["winger", "forward", "striker", "attack"]):
        return "Attack"
    return None


def at_or_below(values: list[float], target: float) -> float:
    return float(np.mean(np.asarray(values, dtype=float) <= target))


def load_raw() -> dict[str, pd.DataFrame]:
    anchors = pd.read_csv(EXTENSIONS, low_memory=False)
    anchors["signed_date"] = pd.to_datetime(anchors["signed_date"], errors="raise")
    anchors["canonical_position_check"] = anchors["canonical_position"].map(broad_position)

    games = pd.read_csv(
        GAMES,
        usecols=["game_id", "date", "home_club_id", "away_club_id", "is_national_team_game"],
        low_memory=False,
    )
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    games = games.loc[~bools(games["is_national_team_game"])].copy()

    lineups = pd.read_csv(
        LINEUPS,
        usecols=["game_id", "player_id", "club_id", "position", "is_starting_lineup"],
        low_memory=False,
    )
    lineups["is_starting_lineup"] = bools(lineups["is_starting_lineup"])
    lineups["player_id"] = pd.to_numeric(lineups["player_id"], errors="coerce")
    lineups["club_id"] = pd.to_numeric(lineups["club_id"], errors="coerce")

    appearances = pd.read_csv(
        APPEARANCES,
        usecols=["game_id", "player_id", "player_club_id", "minutes_played"],
        low_memory=False,
    )
    appearances["player_id"] = pd.to_numeric(appearances["player_id"], errors="coerce")
    appearances["player_club_id"] = pd.to_numeric(appearances["player_club_id"], errors="coerce")
    appearances["minutes_played"] = pd.to_numeric(
        appearances["minutes_played"], errors="coerce"
    ).fillna(0).clip(0, 130)

    players = pd.read_csv(
        PLAYERS,
        usecols=["canonical_player_id", "canonical_sub_position", "canonical_position"],
        low_memory=False,
    )
    players["position_check"] = players["canonical_sub_position"].map(broad_position)
    players["position_check"] = players["position_check"].fillna(
        players["canonical_position"].map(broad_position)
    )
    position_map = players.set_index("canonical_player_id")["position_check"]
    lineups["position_check"] = lineups["position"].map(broad_position)
    lineups["position_check"] = lineups["position_check"].fillna(lineups["player_id"].map(position_map))

    valuations = pd.read_csv(
        VALUATIONS,
        usecols=["canonical_player_id", "valuation_date", "canonical_market_value_eur", "valuation_date_valid"],
        low_memory=False,
    )
    valuations["valuation_date"] = pd.to_datetime(valuations["valuation_date"], errors="coerce")
    valuations["canonical_market_value_eur"] = pd.to_numeric(
        valuations["canonical_market_value_eur"], errors="coerce"
    )
    valuations = valuations.loc[
        bools(valuations["valuation_date_valid"])
        & valuations["canonical_market_value_eur"].gt(0)
    ].copy()

    transfers = pd.read_csv(
        TRANSFERS,
        usecols=[
            "canonical_player_id", "transfer_date", "to_club_id",
            "canonical_transfer_type", "event_conflict", "transfer_type_conflict",
        ],
        low_memory=False,
    )
    transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"], errors="coerce")
    transfers["to_club_id"] = pd.to_numeric(transfers["to_club_id"], errors="coerce")
    transfers["position_check"] = transfers["canonical_player_id"].map(position_map)
    transfers = transfers.loc[
        ~bools(transfers["event_conflict"])
        & ~bools(transfers["transfer_type_conflict"])
        & transfers["canonical_transfer_type"].isin(["transfer", "loan", "loan_return"])
    ].copy()
    return {
        "anchors": anchors,
        "games": games,
        "lineups": lineups,
        "appearances": appearances,
        "valuations": valuations,
        "transfers": transfers,
    }


def independent_reconstruction(profiles: pd.DataFrame) -> tuple[int, float]:
    raw = load_raw()
    candidates = profiles.loc[
        profiles["signed_year"].between(2020, 2023)
        & profiles[[
            "role_trajectory_available", "squad_onfield_available",
            "squad_value_incoming_available",
        ]].all(axis=1)
    ].sort_values("capology_extension_event_id")
    sample = candidates.iloc[np.linspace(0, len(candidates) - 1, 12, dtype=int)]
    maximum_error = 0.0
    comparisons = 0

    anchor_index = raw["anchors"].set_index("capology_extension_event_id")
    for profile in sample.itertuples(index=False):
        anchor = anchor_index.loc[profile.capology_extension_event_id]
        decision = anchor["signed_date"]
        start = decision - pd.Timedelta(days=365)
        split = decision - pd.Timedelta(days=180)
        club = int(anchor["canonical_club_id"])
        player = int(anchor["canonical_player_id"])
        position = anchor["canonical_position_check"]

        games = raw["games"].loc[
            raw["games"]["date"].ge(start)
            & raw["games"]["date"].lt(decision)
            & (
                raw["games"]["home_club_id"].eq(club)
                | raw["games"]["away_club_id"].eq(club)
            )
        ]
        game_ids = set(games["game_id"])
        recent_ids = set(games.loc[games["date"].ge(split), "game_id"])
        prior_ids = game_ids - recent_ids
        lineups = raw["lineups"].loc[
            raw["lineups"]["club_id"].eq(club) & raw["lineups"]["game_id"].isin(game_ids)
        ]
        apps = raw["appearances"].loc[
            raw["appearances"]["player_club_id"].eq(club)
            & raw["appearances"]["game_id"].isin(game_ids)
        ]
        player_lineups = lineups.loc[lineups["player_id"].eq(player)]
        start_rate = player_lineups["is_starting_lineup"].sum() / len(game_ids)
        selection_rate = player_lineups["game_id"].nunique() / len(game_ids)
        for observed, expected in [
            (profile.role_start_rate_365d, start_rate),
            (profile.role_lineup_selection_rate_365d, selection_rate),
        ]:
            maximum_error = max(maximum_error, abs(float(observed) - float(expected)))
            comparisons += 1

        def opportunity(ids: set[int]) -> float:
            minutes = apps.loc[apps["game_id"].isin(ids) & apps["player_id"].eq(player), "minutes_played"].sum()
            return float(minutes / (90 * len(ids)))

        recent_minutes = opportunity(recent_ids)
        prior_minutes = opportunity(prior_ids)
        maximum_error = max(
            maximum_error,
            abs(profile.role_opportunity_share_change_recent_vs_prior - (recent_minutes - prior_minutes)),
        )
        comparisons += 1

        aggregate = lineups.groupby("player_id").agg(
            selections=("game_id", "nunique"), starts=("is_starting_lineup", "sum")
        )
        dominant = (
            lineups.dropna(subset=["position_check"])
            .groupby(["player_id", "position_check"]).size().rename("n").reset_index()
            .sort_values(["player_id", "n", "position_check"], ascending=[True, False, True])
            .drop_duplicates("player_id").set_index("player_id")["position_check"]
        )
        aggregate["position_check"] = dominant
        aggregate["minutes"] = apps.groupby("player_id")["minutes_played"].sum()
        aggregate["minutes"] = aggregate["minutes"].fillna(0.0)
        active = aggregate.loc[
            aggregate["selections"].ge(3) & aggregate["position_check"].eq(position)
        ]
        competitors = [int(pid) for pid in active.index if int(pid) != player]
        comparison = sorted(set(competitors + [player]))
        starts = [float(aggregate["starts"].get(pid, 0.0)) / len(game_ids) for pid in comparison]
        minutes = [float(aggregate["minutes"].get(pid, 0.0)) for pid in comparison]
        target_start = float(aggregate["starts"].get(player, 0.0)) / len(game_ids)
        target_minutes = float(aggregate["minutes"].get(player, 0.0))
        expected_values = [
            (profile.squad_same_position_competitor_count, len(competitors)),
            (profile.player_same_position_start_percentile, at_or_below(starts, target_start)),
            (profile.player_share_of_position_minutes, target_minutes / sum(minutes)),
        ]
        for observed, expected in expected_values:
            maximum_error = max(maximum_error, abs(float(observed) - float(expected)))
            comparisons += 1

        player_values = raw["valuations"].loc[
            raw["valuations"]["canonical_player_id"].eq(player)
            & raw["valuations"]["valuation_date"].le(decision)
        ].sort_values("valuation_date")
        latest = player_values.iloc[-1]
        expected_value = float(latest["canonical_market_value_eur"])
        maximum_error = max(maximum_error, abs(profile.player_predecision_value_eur - expected_value))
        comparisons += 1

        incoming = raw["transfers"].loc[
            raw["transfers"]["to_club_id"].eq(club)
            & raw["transfers"]["transfer_date"].ge(split)
            & raw["transfers"]["transfer_date"].lt(decision)
            & raw["transfers"]["position_check"].eq(position)
            & ~raw["transfers"]["canonical_player_id"].eq(player)
        ].drop_duplicates("canonical_player_id")
        expected_incoming = int(incoming["canonical_player_id"].nunique())
        maximum_error = max(
            maximum_error,
            abs(profile.incoming_same_position_players_180d - expected_incoming),
        )
        comparisons += 1
    return comparisons, maximum_error


def manifest_valid(directory: Path) -> bool:
    manifest = pd.read_csv(directory / "output_manifest.csv")
    return all(
        (directory / row.output_file).stat().st_size == row.bytes
        and sha256(directory / row.output_file) == row.sha256
        for row in manifest.itertuples(index=False)
    )


def clean_rerun_matches() -> tuple[bool, bool]:
    with tempfile.TemporaryDirectory(prefix=".role_squad_verify_", dir=P) as temp:
        base = Path(temp)
        rebuilt = base / "profiles"
        rerun = base / "experiment"
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_engine_v2_role_squad_profiles.py"), "--output", str(rebuilt)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "models" / "run_engine_v2_role_squad_experiment.py"),
                "--profile-dir", str(rebuilt),
                "--output", str(rerun),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        profile_files = [
            "role_squad_profiles.csv", "asof_evidence_audit.csv", "feature_dictionary.csv",
            "build_checks.csv", "feature_contract.json", "run_summary.json", "README.md",
        ]
        experiment_files = [
            "binary_predictions.csv", "binary_origin_metrics.csv", "binary_coverage_by_origin.csv",
            "binary_candidate_comparison.csv", "binary_league_metrics.csv",
            "movement_state_predictions.csv", "movement_state_origin_metrics.csv",
            "movement_state_coverage_by_origin.csv", "movement_state_candidate_comparison.csv",
            "movement_state_class_metrics.csv", "movement_state_league_metrics.csv",
            "added_feature_coefficients.csv", "build_checks.csv", "preregistration.json",
            "run_summary.json", "README.md",
        ]
        profile_match = all(
            sha256(PROFILE_DIR / name) == sha256(rebuilt / name) for name in profile_files
        )
        experiment_match = all(
            sha256(EXPERIMENT_DIR / name) == sha256(rerun / name) for name in experiment_files
        )
        return profile_match, experiment_match


def main() -> None:
    rows: list[dict[str, object]] = []

    def add(check: str, passed: bool, observed: object, expected: object) -> None:
        rows.append(
            {"check": check, "passed": bool(passed), "observed": observed, "expected": expected}
        )

    profiles = pd.read_csv(PROFILE_DIR / "role_squad_profiles.csv", low_memory=False)
    binary = pd.read_csv(EXPERIMENT_DIR / "binary_predictions.csv", low_memory=False)
    multi = pd.read_csv(EXPERIMENT_DIR / "movement_state_predictions.csv", low_memory=False)
    binary_comparison = pd.read_csv(EXPERIMENT_DIR / "binary_candidate_comparison.csv")
    multi_comparison = pd.read_csv(EXPERIMENT_DIR / "movement_state_candidate_comparison.csv")

    add("profile_manifest", manifest_valid(PROFILE_DIR), True, True)
    add("experiment_manifest", manifest_valid(EXPERIMENT_DIR), True, True)
    add("profile_grain", len(profiles) == 2805 and profiles.capology_extension_event_id.is_unique, len(profiles), 2805)
    add("development_only", binary.evaluation_year.max() == 2023 and multi.evaluation_year.max() == 2023, max(binary.evaluation_year.max(), multi.evaluation_year.max()), 2023)
    add("five_variants", binary.variant.nunique() == 5 and multi.variant.nunique() == 5, (binary.variant.nunique(), multi.variant.nunique()), (5, 5))
    add("four_binary_targets", binary.endpoint.nunique() == 4, binary.endpoint.nunique(), 4)
    binary_delta = np.square(binary.actual - binary.matched_baseline_prediction) - np.square(binary.actual - binary.candidate_prediction)
    add("binary_loss_reconstruction", np.allclose(binary_delta, binary.loss_improvement, atol=1e-14), float(np.max(np.abs(binary_delta - binary.loss_improvement))), "<=1e-14")
    state_columns = [column for column in multi if column.startswith("candidate_probability_")]
    add("state_probability_sum", np.allclose(multi[state_columns].sum(axis=1), 1, atol=1e-12), float(np.max(np.abs(multi[state_columns].sum(axis=1) - 1))), "<=1e-12")
    add("research_only", binary_comparison.deployment_status.eq("not_deployed").all() and multi_comparison.deployment_status.eq("not_deployed").all(), True, True)
    comparisons, maximum_error = independent_reconstruction(profiles)
    add("raw_feature_reconstruction", comparisons == 96 and maximum_error <= TOLERANCE, {"comparisons": comparisons, "maximum_error": maximum_error}, {"comparisons": 96, "maximum_error": f"<={TOLERANCE}"})
    signing = pd.to_datetime(profiles.signed_date)
    value_date = pd.to_datetime(profiles.player_predecision_value_date, errors="coerce")
    add("valuation_timing", (value_date.dropna() <= signing.loc[value_date.notna()]).all(), int((value_date > signing).fillna(False).sum()), 0)
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    frozen = all(sha256(ROOT / item["path"]) == item["sha256"] for item in freeze["files"])
    add("v1_frozen_before_rerun", frozen, frozen, True)
    profile_match, experiment_match = clean_rerun_matches()
    add("clean_profile_rerun", profile_match, profile_match, True)
    add("clean_experiment_rerun", experiment_match, experiment_match, True)
    frozen_after = all(sha256(ROOT / item["path"]) == item["sha256"] for item in freeze["files"])
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

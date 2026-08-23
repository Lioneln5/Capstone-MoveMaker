#!/usr/bin/env python3
"""Verify the external data tree required by the deployed MoveMaker API."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


REQUIRED: dict[str, tuple[str, ...]] = {
    "processed/canonical_integration/canonical_player_dimension.csv": (
        "canonical_player_id", "canonical_name", "canonical_position",
    ),
    "processed/canonical_integration/canonical_club_dimension.csv": (
        "canonical_club_id", "canonical_club_name",
    ),
    "processed/canonical_integration/canonical_valuation_history.csv": (
        "canonical_player_id",
    ),
    "processed/transfermarkt_clean/tables/appearances_clean.csv": (
        "game_id", "player_id", "player_club_id", "date", "competition_id",
    ),
    "processed/transfermarkt_clean/tables/games_clean.csv": (
        "game_id", "is_national_team_game",
    ),
    "processed/canonical_performance/canonical_player_team_season_performance.csv": (
        "canonical_player_id", "canonical_club_id", "season_start_year",
    ),
    "processed/canonical_performance/fbref_advanced_performance_features.csv": (
        "canonical_performance_id",
    ),
    "processed/capology_contracts/canonical_salary_panel.csv": (
        "canonical_player_id",
    ),
    "processed/contract_extension_integration/extension_modeling_master.csv": (
        "canonical_player_id", "signed_date",
    ),
}

OPTIONAL: dict[str, tuple[str, ...]] = {
    "processed/live_input_refresh/canonical_performance_overlay.csv": (
        "canonical_player_id",
    ),
    "processed/live_input_refresh/canonical_salary_overlay.csv": (
        "canonical_player_id",
    ),
}


def validate_file(root: Path, relative: str, columns: tuple[str, ...]) -> list[str]:
    path = root / relative
    if not path.is_file():
        return [f"missing: {path}"]
    try:
        observed = set(pd.read_csv(path, nrows=0).columns)
    except Exception as exc:  # pragma: no cover - deployment diagnostic
        return [f"unreadable: {path}: {exc}"]
    missing = sorted(set(columns) - observed)
    return [f"missing columns in {path}: {', '.join(missing)}"] if missing else []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("MOVEMAKER_DATA_ROOT", "Data")),
        help="Directory containing processed/ (Railway uses /data).",
    )
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()

    errors: list[str] = []
    for relative, columns in REQUIRED.items():
        errors.extend(validate_file(root, relative, columns))

    for relative, columns in OPTIONAL.items():
        path = root / relative
        if path.exists():
            errors.extend(validate_file(root, relative, columns))
        else:
            print(f"OPTIONAL MISSING: {path}")

    if errors:
        print("Railway runtime-data verification FAILED:")
        for error in errors:
            print(f"- {error}")
        return 1

    total_bytes = sum((root / relative).stat().st_size for relative in REQUIRED)
    print(f"Railway runtime-data verification PASSED: {len(REQUIRED)} required files, {total_bytes / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

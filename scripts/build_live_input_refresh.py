"""Build append-only live-scoring overlays from post-2023 source snapshots.

This deliberately does not rebuild or mutate the frozen Phase 1-5 research
datasets.  It gives ``canonical_feature_retriever.py`` fresher completed-
season performance and salary-panel context for present-day API requests.
"""

from __future__ import annotations

import json
import sys
import unicodedata
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_capology_contracts import (  # noqa: E402
    build_club_map,
    build_player_map,
    collapse_salaries,
    read_source,
)


FBREF_SOURCE = ROOT / "Data" / "2024-2026"
LEAGUES = ROOT / "Data" / "leagues"
OUTPUT = ROOT / "Data" / "processed" / "live_input_refresh"
PERFORMANCE_OUTPUT = OUTPUT / "canonical_performance_overlay.csv"
SALARY_OUTPUT = OUTPUT / "canonical_salary_overlay.csv"

COMPETITIONS = {
    "eng Premier League": ("Premier League", "GB1"),
    "es La Liga": ("La Liga", "ES1"),
    "it Serie A": ("Serie A", "IT1"),
    "de Bundesliga": ("Bundesliga", "L1"),
    "fr Ligue 1": ("Ligue 1", "FR1"),
}


def normalize(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def build_performance_overlay() -> tuple[pd.DataFrame, dict]:
    fbref_map, identity_map = build_player_map()
    club_map, _ = build_club_map()
    frames: list[pd.DataFrame] = []

    for season_start in (2024, 2025):
        path = FBREF_SOURCE / f"players_data-{season_start}_{season_start + 1}.csv"
        raw = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
        out = pd.DataFrame(index=raw.index)
        out["player_name"] = raw["Player"].astype("string").str.strip()
        out["player_name_normalized"] = out["player_name"].map(normalize)
        out["squad_name"] = raw["Squad"].astype("string").str.strip()
        out["squad_name_normalized"] = out["squad_name"].map(normalize)
        out["canonical_competition_id"] = raw["Comp"].map(lambda value: COMPETITIONS.get(str(value), (None, None))[1])
        out["season_start_year"] = season_start

        player_links = out["player_name_normalized"].map(
            lambda name: fbref_map.get(name) or identity_map.get(name) or {}
        )
        out["canonical_player_id"] = player_links.map(lambda link: link.get("canonical_player_id"))
        club_links = out["squad_name_normalized"].map(lambda name: club_map.get(name, {}))
        out["canonical_club_id"] = club_links.map(lambda link: link.get("canonical_club_id"))

        minutes = numeric(raw, "Min")
        matches = numeric(raw, "MP")
        starts = numeric(raw, "Starts")
        goals = numeric(raw, "Gls")
        assists = numeric(raw, "Ast")
        out["canonical_matches_played"] = matches
        out["canonical_minutes_played"] = minutes
        out["canonical_goals"] = goals
        out["canonical_assists"] = assists
        out["canonical_yellow_cards"] = numeric(raw, "CrdY")
        out["canonical_red_cards"] = numeric(raw, "CrdR")
        out["canonical_goals_per90"] = 90 * goals.div(minutes.where(minutes.gt(0)))
        out["canonical_assists_per90"] = 90 * assists.div(minutes.where(minutes.gt(0)))
        out["current_starts"] = starts
        out["current_start_pct"] = starts.div(matches.where(matches.gt(0)))
        out["fbref_expected_goals"] = numeric(raw, "xG")
        out["fbref_non_penalty_expected_goals"] = numeric(raw, "npxG")
        out["fbref_progressive_carries"] = numeric(raw, "PrgC")
        out["fbref_progressive_passes"] = numeric(raw, "PrgP")
        out["fbref_pass_completion_pct"] = numeric(raw, "Cmp%")
        out["advanced_features_available"] = out[[
            "fbref_expected_goals", "fbref_progressive_carries",
            "fbref_progressive_passes", "fbref_pass_completion_pct",
        ]].notna().all(axis=1)
        out["data_coverage_tier"] = np.where(out["advanced_features_available"], "live_fbref_advanced", "live_fbref_basic")
        out["basic_model_eligible_450_minutes"] = minutes.ge(450)
        out["canonical_performance_id"] = [
            f"live:{int(pid)}:{int(cid)}:{comp}:{season_start}"
            if pd.notna(pid) and pd.notna(cid) and comp
            else None
            for pid, cid, comp in zip(
                out["canonical_player_id"], out["canonical_club_id"], out["canonical_competition_id"]
            )
        ]
        out["source_file"] = path.name
        frames.append(out)

    combined = pd.concat(frames, ignore_index=True)
    eligible = combined.loc[
        combined["canonical_player_id"].notna()
        & combined["canonical_club_id"].notna()
        & combined["canonical_competition_id"].notna()
    ].copy()
    key = ["canonical_player_id", "canonical_club_id", "canonical_competition_id", "season_start_year"]
    duplicates = int(eligible.duplicated(key).sum())
    if duplicates:
        eligible = eligible.sort_values(key + ["canonical_minutes_played"]).drop_duplicates(key, keep="last")
    eligible = eligible.sort_values(key).reset_index(drop=True)
    summary = {
        "source_rows": int(len(combined)),
        "linked_rows": int(len(eligible)),
        "player_link_rate": float(combined["canonical_player_id"].notna().mean()),
        "club_link_rate": float(combined["canonical_club_id"].notna().mean()),
        "advanced_rows": int(eligible["advanced_features_available"].sum()),
        "duplicate_keys_collapsed": duplicates,
    }
    return eligible, summary


def build_salary_overlay() -> tuple[pd.DataFrame, dict]:
    fbref_map, identity_map = build_player_map()
    club_map, _ = build_club_map()
    files = sorted(
        path for path in LEAGUES.glob("*/sallaries/*.csv")
        if int(re.search(r"(20\d{2})_(20\d{2})", path.name).group(1)) >= 2024
    )
    source = pd.concat(
        [read_source(path, "salary", fbref_map, identity_map, club_map) for path in files],
        ignore_index=True,
    )
    salary = collapse_salaries(source)
    summary = {
        "source_files": len(files),
        "source_rows": int(len(source)),
        "canonical_rows": int(len(salary)),
        "player_link_rate": float(salary["canonical_player_id"].notna().mean()),
        "club_link_rate": float(salary["canonical_club_id"].notna().mean()),
        "seasons": sorted(salary["source_page_season"].dropna().unique().tolist()),
    }
    return salary, summary


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    performance, performance_summary = build_performance_overlay()
    salary, salary_summary = build_salary_overlay()
    performance.to_csv(PERFORMANCE_OUTPUT, index=False, encoding="utf-8-sig", lineterminator="\n")
    salary.to_csv(SALARY_OUTPUT, index=False, encoding="utf-8-sig", lineterminator="\n")
    summary = {
        "scope": "append_only_live_scoring_refresh",
        "frozen_research_artifacts_modified": False,
        "performance": performance_summary,
        "salary": salary_summary,
    }
    (OUTPUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

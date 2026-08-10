from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "Data" / "processed"
OUTPUT = PROCESSED / "integration_crosswalks"
TM = PROCESSED / "transfermarkt_clean"
DL = PROCESSED / "football_datalake_clean" / "tables"
FBREF = PROCESSED / "fbref_2017_2024_clean" / "fbref_player_squad_season_clean.csv"
CHUNK = 250_000


def text_id(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if re.fullmatch(r"-?\d+\.0", text) else text


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_ids(path: Path, columns: list[str]) -> set[str]:
    values = set()
    for chunk in pd.read_csv(path, usecols=columns, dtype="string", chunksize=CHUNK, low_memory=False):
        for column in columns:
            values.update(text_id(value) for value in chunk[column] if text_id(value))
    return values


def add_check(checks, name, passed, observed, expected, detail):
    checks.append(
        {
            "check": name,
            "status": "pass" if passed else "fail",
            "observed": observed,
            "expected": expected,
            "detail": detail,
        }
    )


def main():
    required = [
        "player_identity_crosswalk.csv", "fbref_player_crosswalk.csv", "fbref_player_candidate_review.csv",
        "club_identity_crosswalk.csv", "fbref_club_season_crosswalk.csv", "fbref_club_crosswalk.csv",
        "competition_crosswalk.csv", "season_crosswalk.csv", "crosswalk_summary.csv",
        "crosswalk_checks.csv", "crosswalk_dictionary.csv", "source_manifest.csv", "README.md",
    ]
    checks = []
    missing = [name for name in required if not (OUTPUT / name).exists()]
    add_check(checks, "required_outputs", not missing, len(required) - len(missing), len(required), f"missing={missing}")
    if missing:
        raise SystemExit(json.dumps(checks, indent=2))

    players = pd.read_csv(OUTPUT / "player_identity_crosswalk.csv", dtype="string", low_memory=False)
    fb_players = pd.read_csv(OUTPUT / "fbref_player_crosswalk.csv", dtype="string", low_memory=False)
    player_review = pd.read_csv(OUTPUT / "fbref_player_candidate_review.csv", dtype="string", low_memory=False)
    clubs = pd.read_csv(OUTPUT / "club_identity_crosswalk.csv", dtype="string", low_memory=False)
    fb_club_seasons = pd.read_csv(OUTPUT / "fbref_club_season_crosswalk.csv", dtype="string", low_memory=False)
    fb_clubs = pd.read_csv(OUTPUT / "fbref_club_crosswalk.csv", dtype="string", low_memory=False)
    competitions = pd.read_csv(OUTPUT / "competition_crosswalk.csv", dtype="string", low_memory=False)
    seasons = pd.read_csv(OUTPUT / "season_crosswalk.csv", dtype="string", low_memory=False)

    current_player_ids = collect_ids(TM / "player_identity_union.csv", ["player_id"]) | collect_ids(TM / "tables" / "players_clean.csv", ["player_id"])
    datalake_player_ids = collect_ids(DL / "player_profiles_clean.csv", ["player_id"])
    expected_player_ids = current_player_ids | datalake_player_ids
    output_player_ids = set(players["canonical_player_id"].map(text_id))
    add_check(checks, "native_player_union", output_player_ids == expected_player_ids, len(output_player_ids), len(expected_player_ids), "Canonical player IDs equal the independent union of native profile IDs.")
    add_check(checks, "native_player_unique", players["canonical_player_id"].is_unique, players["canonical_player_id"].nunique(), len(players), "One row per native Transfermarkt player ID.")

    fb = pd.read_csv(FBREF, usecols=["player_identity_key", "squad_name_normalized", "competition_id", "season"], dtype="string")
    expected_fb_player_rows = fb["player_identity_key"].nunique(dropna=False)
    add_check(checks, "fbref_player_cardinality", len(fb_players) == expected_fb_player_rows, len(fb_players), expected_fb_player_rows, "One row per FBref player identity key.")
    add_check(checks, "fbref_player_unique", fb_players["fbref_player_identity_key"].is_unique, fb_players["fbref_player_identity_key"].nunique(), len(fb_players), "No duplicated FBref identity keys.")
    accepted_players = fb_players[fb_players["match_status"].eq("accepted")]
    add_check(checks, "accepted_player_references", set(accepted_players["canonical_player_id"].map(text_id)).issubset(output_player_ids), len(accepted_players), len(accepted_players), "Every accepted link resolves to the native player identity map.")
    add_check(checks, "player_review_flags", (fb_players["manual_review_required"].str.lower().eq("false") == fb_players["match_status"].eq("accepted")).all(), int(fb_players["manual_review_required"].str.lower().eq("true").sum()), int(fb_players["match_status"].ne("accepted").sum()), "Only accepted rows bypass manual review.")
    add_check(checks, "accepted_player_many_to_one", not accepted_players["canonical_player_id"].duplicated().any(), int(accepted_players["canonical_player_id"].duplicated().sum()), 0, "No two FBref identity keys were silently collapsed onto one canonical player.")

    fuzzy_multi = accepted_players[accepted_players["match_method"].eq("fuzzy_name_birth_year_multiple_club_seasons")].copy()
    fuzzy_multi["score"] = pd.to_numeric(fuzzy_multi["top_name_similarity"], errors="coerce")
    fuzzy_multi["agreements"] = pd.to_numeric(fuzzy_multi["club_season_agreements"], errors="coerce")
    fuzzy_multi_ok = ((fuzzy_multi["score"] >= 0.75) & (fuzzy_multi["agreements"] >= 2)).all()
    add_check(checks, "fuzzy_multi_evidence_threshold", fuzzy_multi_ok, len(fuzzy_multi), len(fuzzy_multi), "Accepted lower-similarity names require at least two club-season agreements.")
    fuzzy_high = accepted_players[accepted_players["match_method"].eq("high_similarity_birth_year_club_season")].copy()
    fuzzy_high["score"] = pd.to_numeric(fuzzy_high["top_name_similarity"], errors="coerce")
    fuzzy_high["second"] = pd.to_numeric(fuzzy_high["second_name_similarity"], errors="coerce")
    fuzzy_high["agreements"] = pd.to_numeric(fuzzy_high["club_season_agreements"], errors="coerce")
    fuzzy_high_ok = ((fuzzy_high["score"] >= 0.92) & ((fuzzy_high["score"] - fuzzy_high["second"]) >= 0.03 - 1e-9) & (fuzzy_high["agreements"] >= 1)).all()
    add_check(checks, "fuzzy_high_evidence_threshold", fuzzy_high_ok, len(fuzzy_high), len(fuzzy_high), "High-similarity links retain the score margin and club-season evidence rule.")
    review_keys = set(player_review["fbref_player_identity_key"])
    unresolved_keys = set(fb_players.loc[fb_players["match_status"].ne("accepted") & fb_players["candidate_count"].astype(float).gt(0), "fbref_player_identity_key"])
    add_check(checks, "review_candidates_retained", unresolved_keys.issubset(review_keys), len(unresolved_keys & review_keys), len(unresolved_keys), "Every unresolved identity with candidates has ranked review evidence.")

    expected_club_ids = collect_ids(TM / "club_dimension_historical.csv", ["club_id"])
    club_sources = [
        (DL / "team_details_clean.csv", ["club_id"]),
        (DL / "team_competitions_seasons_clean.csv", ["club_id"]),
        (DL / "player_performances_clean.csv", ["team_id"]),
        (DL / "transfer_history_clean.csv", ["from_team_id", "to_team_id"]),
        (DL / "team_children_clean.csv", ["parent_team_id", "child_team_id"]),
        (DL / "player_profiles_clean.csv", ["current_club_id", "on_loan_from_club_id"]),
    ]
    for path, columns in club_sources:
        expected_club_ids |= collect_ids(path, columns)
    output_club_ids = set(clubs["canonical_club_id"].map(text_id))
    add_check(checks, "native_club_union", output_club_ids == expected_club_ids, len(output_club_ids), len(expected_club_ids), "Canonical club IDs equal the independent union across all native club-role fields.")
    add_check(checks, "native_club_unique", clubs["canonical_club_id"].is_unique, clubs["canonical_club_id"].nunique(), len(clubs), "One row per native Transfermarkt club ID.")

    expected_club_seasons = len(fb[["squad_name_normalized", "competition_id", "season"]].drop_duplicates())
    add_check(checks, "fbref_club_season_cardinality", len(fb_club_seasons) == expected_club_seasons, len(fb_club_seasons), expected_club_seasons, "One row per FBref squad-competition-season.")
    add_check(checks, "fbref_club_season_unique", fb_club_seasons["fbref_club_season_key"].is_unique, fb_club_seasons["fbref_club_season_key"].nunique(), len(fb_club_seasons), "No duplicated squad-season keys.")
    accepted_clubs = fb_club_seasons[fb_club_seasons["match_status"].eq("accepted")]
    add_check(checks, "accepted_club_references", set(accepted_clubs["canonical_club_id"].map(text_id)).issubset(output_club_ids), len(accepted_clubs), len(accepted_clubs), "Every accepted club-season link resolves to the native club identity map.")
    roster = accepted_clubs[accepted_clubs["match_method"].eq("season_competition_roster_evidence")].copy()
    roster["top"] = pd.to_numeric(roster["top_roster_overlap"], errors="coerce")
    roster["second"] = pd.to_numeric(roster["second_roster_overlap"], errors="coerce")
    roster_ok = ((roster["top"] >= 3) & ((roster["top"] - roster["second"]) >= 2)).all()
    add_check(checks, "club_roster_evidence_threshold", roster_ok, len(roster), len(roster), "Roster-only club links retain minimum overlap and winning-margin rules.")
    add_check(checks, "stable_fbref_club_unique", fb_clubs["fbref_squad_name_normalized"].is_unique, fb_clubs["fbref_squad_name_normalized"].nunique(), len(fb_clubs), "One stable row per normalized FBref squad name.")

    fb_comp_ids = set(fb["competition_id"].dropna())
    canonical_comp_ids = set(competitions["canonical_competition_id"].dropna())
    add_check(checks, "fbref_competition_coverage", fb_comp_ids.issubset(canonical_comp_ids), len(fb_comp_ids & canonical_comp_ids), len(fb_comp_ids), "All FBref competition codes map through the shared Transfermarkt code.")
    fb_seasons = seasons[seasons["source_system"].eq("fbref")]
    add_check(checks, "fbref_season_coverage", fb_seasons["match_status"].eq("accepted").all(), int(fb_seasons["match_status"].eq("accepted").sum()), len(fb_seasons), "All FBref season labels parse to start/end years.")
    nonblank_seasons = seasons[seasons["source_season_value"].notna() & seasons["source_season_value"].ne("")]
    add_check(checks, "nonblank_season_coverage", nonblank_seasons["match_status"].eq("accepted").all(), int(nonblank_seasons["match_status"].eq("accepted").sum()), len(nonblank_seasons), "Every nonblank source season representation maps to a canonical key.")

    source_manifest = pd.read_csv(OUTPUT / "source_manifest.csv", dtype="string")
    hashes_match = True
    for row in source_manifest.itertuples(index=False):
        path = ROOT / str(row.source_file)
        hashes_match &= path.exists() and sha256(path) == row.sha256
    add_check(checks, "source_manifest_hashes", hashes_match, int(hashes_match), 1, "Independent SHA-256 recomputation matches every recorded source.")

    result = pd.DataFrame(checks)
    result.to_csv(OUTPUT / "independent_verification.csv", index=False, encoding="utf-8-sig")
    output_files = sorted(path for path in OUTPUT.glob("*.csv") if path.name not in {"output_manifest.csv"})
    manifest = pd.DataFrame(
        [{"output_file": path.name, "file_bytes": path.stat().st_size, "sha256": sha256(path)} for path in output_files]
    )
    manifest.to_csv(OUTPUT / "output_manifest.csv", index=False, encoding="utf-8-sig")
    summary = {
        "status": "pass" if result["status"].eq("pass").all() else "fail",
        "checks": len(result),
        "passed": int(result["status"].eq("pass").sum()),
        "failed": int(result["status"].eq("fail").sum()),
    }
    (OUTPUT / "independent_verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["failed"]:
        print(result[result["status"].eq("fail")].to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

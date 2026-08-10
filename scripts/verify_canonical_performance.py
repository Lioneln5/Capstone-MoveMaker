from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "Data" / "processed"
OUTPUT = PROCESSED / "canonical_performance"
CANONICAL = PROCESSED / "canonical_integration"
XW = PROCESSED / "integration_crosswalks"

CORE = OUTPUT / "canonical_player_team_season_performance.csv"
MODEL = OUTPUT / "modeling_scope_2017_2024.csv"
ADVANCED = OUTPUT / "fbref_advanced_performance_features.csv"
DIFFERENCES = OUTPUT / "performance_difference_review.csv"
UNRESOLVED = OUTPUT / "unresolved_fbref_performance_review.csv"
CHUNK_SIZE = 75_000


def clean_id(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].lstrip("-").isdigit():
        return text[:-2]
    return text


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype("string").str.casefold().isin(["true", "1", "yes"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add(checks: list[dict[str, object]], name: str, passed: bool, observed: object, expected: object, detail: str) -> None:
    checks.append({
        "check": name,
        "severity": "blocking",
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
        "detail": detail,
    })


def numeric_equal(left: pd.Series, right: pd.Series, tolerance: float = 1e-9) -> pd.Series:
    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    both_missing = left_num.isna() & right_num.isna()
    both_present_equal = left_num.notna() & right_num.notna() & np.isclose(
        left_num.fillna(0), right_num.fillna(0), rtol=0, atol=tolerance
    )
    return both_missing | both_present_equal


def expected_metric(frame: pd.DataFrame, candidates: list[tuple[str, str]]) -> tuple[pd.Series, pd.Series]:
    value = pd.Series(np.nan, index=frame.index, dtype="float64")
    source = pd.Series("", index=frame.index, dtype="string")
    for column, label in candidates:
        candidate = pd.to_numeric(frame[column], errors="coerce")
        mask = value.isna() & candidate.notna()
        value.loc[mask] = candidate.loc[mask]
        source.loc[mask] = label
    return value, source


def main() -> None:
    required = [
        "canonical_player_team_season_performance.csv", "modeling_scope_2017_2024.csv",
        "fbref_advanced_performance_features.csv", "performance_difference_review.csv",
        "unresolved_fbref_performance_review.csv", "coverage_by_season.csv",
        "coverage_by_competition.csv", "source_agreement_summary.csv", "performance_summary.csv",
        "performance_summary.json", "performance_checks.csv", "performance_table_dictionary.csv",
        "performance_field_dictionary.csv", "source_manifest.csv", "output_manifest.csv", "README.md",
    ]
    missing = [name for name in required if not (OUTPUT / name).exists()]
    checks: list[dict[str, object]] = []
    add(checks, "required_outputs", not missing, len(required) - len(missing), len(required), f"missing={missing}")
    if missing:
        result = pd.DataFrame(checks)
        result.to_csv(OUTPUT / "independent_verification.csv", index=False, encoding="utf-8", lineterminator="\n")
        raise SystemExit(json.dumps(checks, indent=2))

    summary = json.loads((OUTPUT / "performance_summary.json").read_text(encoding="utf-8"))
    expected_rows = int(summary["canonical_performance_rows"])
    expected_model_rows = int(summary["modeling_scope_2017_2024_rows"])
    expected_difference_rows = int(summary["difference_review_rows"])
    expected_fbref_rows = int(summary["accepted_fbref_rows"])
    expected_unresolved_rows = int(summary["unresolved_fbref_rows"])

    player_ids = set(pd.read_csv(
        CANONICAL / "canonical_player_dimension.csv", usecols=["canonical_player_id"], dtype=str,
        keep_default_na=False,
    )["canonical_player_id"].map(clean_id))
    club_ids = set(pd.read_csv(
        CANONICAL / "canonical_club_dimension.csv", usecols=["canonical_club_id"], dtype=str,
        keep_default_na=False,
    )["canonical_club_id"].map(clean_id))
    competition_ids = set(pd.read_csv(
        XW / "competition_crosswalk.csv", usecols=["canonical_competition_id"], dtype=str,
        keep_default_na=False,
    )["canonical_competition_id"].map(clean_id))

    usecols = [
        "canonical_performance_id", "canonical_player_id", "canonical_club_id",
        "canonical_competition_id", "season_start_year", "season_end_year",
        "present_in_fbref", "present_in_current_appearances", "present_in_current_lineups",
        "present_in_supplemental_performances", "source_system_count",
        "basic_model_eligible_450_minutes", "canonical_matches_played", "canonical_minutes_played",
        "canonical_goals", "canonical_assists", "canonical_yellow_cards", "canonical_red_cards",
        "canonical_goals_per90", "canonical_assists_per90", "canonical_matches_source",
        "canonical_minutes_source", "canonical_goals_source", "canonical_assists_source",
        "canonical_cards_source", "fbref_matches_played", "fbref_total_minutes", "fbref_goals",
        "fbref_assists", "current_matches_played", "current_minutes_played", "current_goals",
        "current_assists", "current_yellow_cards", "current_red_cards", "current_lineup_games",
        "datalake_nb_on_pitch", "datalake_minutes_played", "datalake_goals", "datalake_assists",
        "datalake_yellow_cards", "datalake_red_cards", "difference_metric_count",
        "any_basic_cross_source_difference", "matches_agreement_status", "minutes_agreement_status",
        "goals_agreement_status", "assists_agreement_status", "yellow_cards_agreement_status",
        "red_cards_agreement_status",
    ]

    seen_ids: set[str] = set()
    expected_model_ids: set[str] = set()
    expected_difference_ids: set[str] = set()
    observed_rows = 0
    duplicate_ids = 0
    malformed_ids = 0
    invalid_seasons = 0
    missing_player_refs: set[str] = set()
    missing_club_refs: set[str] = set()
    missing_comp_refs: set[str] = set()
    empty_source_rows = 0
    source_count_errors = 0
    precedence_errors = {metric: 0 for metric in ["matches", "minutes", "goals", "assists", "yellow_cards", "red_cards"]}
    source_label_errors = {metric: 0 for metric in ["matches", "minutes", "goals", "assists", "cards"]}
    per90_errors = 0
    eligibility_errors = 0
    difference_flag_errors = 0
    source_presence_counts = {name: 0 for name in ["fbref", "current_appearances", "current_lineups", "datalake"]}

    for frame in pd.read_csv(CORE, usecols=usecols, dtype=str, keep_default_na=False, chunksize=CHUNK_SIZE):
        observed_rows += len(frame)
        ids = frame["canonical_performance_id"].tolist()
        duplicate_ids += sum(value in seen_ids for value in ids)
        seen_ids.update(ids)
        player = frame["canonical_player_id"].map(clean_id)
        club = frame["canonical_club_id"].map(clean_id)
        competition = frame["canonical_competition_id"].map(clean_id)
        start = pd.to_numeric(frame["season_start_year"], errors="coerce")
        end = pd.to_numeric(frame["season_end_year"], errors="coerce")
        expected_id = [
            f"pts:{player_id}:{club_id}:{competition_id}:{int(season)}" if pd.notna(season) else ""
            for player_id, club_id, competition_id, season in zip(player, club, competition, start)
        ]
        malformed_ids += int((frame["canonical_performance_id"] != pd.Series(expected_id, index=frame.index)).sum())
        invalid_seasons += int((start.isna() | end.ne(start + 1)).sum())
        missing_player_refs.update(set(player) - player_ids)
        missing_club_refs.update(set(club) - club_ids)
        missing_comp_refs.update(set(competition) - competition_ids)

        fbref = bool_series(frame["present_in_fbref"])
        current_appearances = bool_series(frame["present_in_current_appearances"])
        current_lineups = bool_series(frame["present_in_current_lineups"])
        datalake = bool_series(frame["present_in_supplemental_performances"])
        source_presence_counts["fbref"] += int(fbref.sum())
        source_presence_counts["current_appearances"] += int(current_appearances.sum())
        source_presence_counts["current_lineups"] += int(current_lineups.sum())
        source_presence_counts["datalake"] += int(datalake.sum())
        empty_source_rows += int((~(fbref | current_appearances | current_lineups | datalake)).sum())
        expected_source_count = fbref.astype(int) + (current_appearances | current_lineups).astype(int) + datalake.astype(int)
        source_count_errors += int((pd.to_numeric(frame["source_system_count"], errors="coerce") != expected_source_count).sum())

        expected_values: dict[str, pd.Series] = {}
        expected_sources: dict[str, pd.Series] = {}
        expected_values["matches"], expected_sources["matches"] = expected_metric(frame, [
            ("fbref_matches_played", "fbref"),
            ("current_matches_played", "current_transfermarkt_appearances"),
            ("datalake_nb_on_pitch", "supplemental_datalake"),
            ("current_lineup_games", "current_transfermarkt_lineups"),
        ])
        expected_values["minutes"], expected_sources["minutes"] = expected_metric(frame, [
            ("fbref_total_minutes", "fbref"),
            ("current_minutes_played", "current_transfermarkt_appearances"),
            ("datalake_minutes_played", "supplemental_datalake"),
        ])
        expected_values["goals"], expected_sources["goals"] = expected_metric(frame, [
            ("fbref_goals", "fbref"),
            ("current_goals", "current_transfermarkt_appearances"),
            ("datalake_goals", "supplemental_datalake"),
        ])
        expected_values["assists"], expected_sources["assists"] = expected_metric(frame, [
            ("fbref_assists", "fbref"),
            ("current_assists", "current_transfermarkt_appearances"),
            ("datalake_assists", "supplemental_datalake"),
        ])
        expected_values["yellow_cards"], yellow_source = expected_metric(frame, [
            ("current_yellow_cards", "current_transfermarkt_appearances"),
            ("datalake_yellow_cards", "supplemental_datalake"),
        ])
        expected_values["red_cards"], red_source = expected_metric(frame, [
            ("current_red_cards", "current_transfermarkt_appearances"),
            ("datalake_red_cards", "supplemental_datalake"),
        ])
        canonical_metric_columns = {
            "matches": "canonical_matches_played",
            "minutes": "canonical_minutes_played",
            "goals": "canonical_goals",
            "assists": "canonical_assists",
            "yellow_cards": "canonical_yellow_cards",
            "red_cards": "canonical_red_cards",
        }
        for metric in ["matches", "minutes", "goals", "assists", "yellow_cards", "red_cards"]:
            precedence_errors[metric] += int((~numeric_equal(frame[canonical_metric_columns[metric]], expected_values[metric])).sum())
        for metric in ["matches", "minutes", "goals", "assists"]:
            source_label_errors[metric] += int((frame[f"canonical_{metric}_source"].astype("string") != expected_sources[metric]).sum())
        expected_cards_source = yellow_source.where(yellow_source.ne(""), red_source)
        source_label_errors["cards"] += int((frame["canonical_cards_source"].astype("string") != expected_cards_source).sum())

        minutes = pd.to_numeric(frame["canonical_minutes_played"], errors="coerce")
        goals = pd.to_numeric(frame["canonical_goals"], errors="coerce")
        assists = pd.to_numeric(frame["canonical_assists"], errors="coerce")
        expected_goals_per90 = pd.Series(np.where(minutes.gt(0), goals * 90 / minutes, np.nan), index=frame.index)
        expected_assists_per90 = pd.Series(np.where(minutes.gt(0), assists * 90 / minutes, np.nan), index=frame.index)
        per90_errors += int((~numeric_equal(frame["canonical_goals_per90"], expected_goals_per90, 1e-5)).sum())
        per90_errors += int((~numeric_equal(frame["canonical_assists_per90"], expected_assists_per90, 1e-5)).sum())
        expected_eligible = minutes.ge(450) & pd.to_numeric(frame["canonical_matches_played"], errors="coerce").ge(5)
        eligibility_errors += int((bool_series(frame["basic_model_eligible_450_minutes"]) != expected_eligible).sum())

        status_columns = [f"{metric}_agreement_status" for metric in [
            "matches", "minutes", "goals", "assists", "yellow_cards", "red_cards"
        ]]
        expected_difference_count = frame[status_columns].eq("sources_differ").sum(axis=1)
        difference_flag = bool_series(frame["any_basic_cross_source_difference"])
        difference_flag_errors += int((pd.to_numeric(frame["difference_metric_count"], errors="coerce") != expected_difference_count).sum())
        difference_flag_errors += int((difference_flag != expected_difference_count.gt(0)).sum())

        model_mask = start.between(2017, 2023)
        expected_model_ids.update(frame.loc[model_mask, "canonical_performance_id"])
        expected_difference_ids.update(frame.loc[difference_flag, "canonical_performance_id"])

    add(checks, "core_row_count", observed_rows == expected_rows, observed_rows, expected_rows, "Independent chunk scan matches build summary.")
    add(checks, "core_unique_ids", duplicate_ids == 0 and len(seen_ids) == observed_rows, duplicate_ids, 0, "Canonical performance IDs are unique.")
    add(checks, "canonical_id_structure", malformed_ids == 0, malformed_ids, 0, "ID text exactly encodes player, club, competition, and season start.")
    add(checks, "season_consistency", invalid_seasons == 0, invalid_seasons, 0, "Every season end equals season start plus one.")
    add(checks, "player_foreign_keys", not missing_player_refs, len(missing_player_refs), 0, "Every player resolves to canonical player dimension.")
    add(checks, "club_foreign_keys", not missing_club_refs, len(missing_club_refs), 0, "Every club resolves to canonical club dimension.")
    add(checks, "competition_foreign_keys", not missing_comp_refs, len(missing_comp_refs), 0, "Every competition resolves to canonical competition crosswalk.")
    add(checks, "source_presence", empty_source_rows == 0 and source_count_errors == 0, {"empty": empty_source_rows, "count_errors": source_count_errors}, {"empty": 0, "count_errors": 0}, "Every spine row has evidence and its system count reconciles.")
    add(checks, "source_presence_counts", source_presence_counts == {
        "fbref": expected_fbref_rows,
        "current_appearances": int(summary["current_appearance_keys"]),
        "current_lineups": int(summary["current_lineup_keys"]),
        "datalake": int(summary["supplemental_performance_keys"]),
    }, source_presence_counts, {
        "fbref": expected_fbref_rows,
        "current_appearances": int(summary["current_appearance_keys"]),
        "current_lineups": int(summary["current_lineup_keys"]),
        "datalake": int(summary["supplemental_performance_keys"]),
    }, "Source-presence counts reconcile to independently aggregated facts.")
    add(checks, "metric_precedence", sum(precedence_errors.values()) == 0, precedence_errors, {key: 0 for key in precedence_errors}, "Canonical metrics follow the documented source precedence exactly.")
    add(checks, "source_labels", sum(source_label_errors.values()) == 0, source_label_errors, {key: 0 for key in source_label_errors}, "Selected-source labels match selected values.")
    add(checks, "per90_recalculation", per90_errors == 0, per90_errors, 0, "Canonical per-90 values recompute from selected counts and minutes.")
    add(checks, "model_eligibility_rule", eligibility_errors == 0, eligibility_errors, 0, "The convenience screen is exactly 450 minutes and five matches.")
    add(checks, "difference_flags", difference_flag_errors == 0, difference_flag_errors, 0, "Difference count and boolean exactly match per-metric statuses.")

    model_ids: set[str] = set()
    model_rows = 0
    model_invalid_seasons = 0
    for frame in pd.read_csv(MODEL, usecols=["canonical_performance_id", "season_start_year"], dtype=str, keep_default_na=False, chunksize=CHUNK_SIZE):
        model_rows += len(frame)
        model_ids.update(frame["canonical_performance_id"])
        season = pd.to_numeric(frame["season_start_year"], errors="coerce")
        model_invalid_seasons += int((~season.between(2017, 2023)).sum())
    add(checks, "modeling_view_exact", model_rows == expected_model_rows and model_ids == expected_model_ids and model_invalid_seasons == 0, {"rows": model_rows, "ids": len(model_ids), "invalid_seasons": model_invalid_seasons}, {"rows": expected_model_rows, "ids": len(expected_model_ids), "invalid_seasons": 0}, "Model view is the exact 2017-18 through 2023-24 subset.")

    review = pd.read_csv(DIFFERENCES, usecols=["canonical_performance_id"], dtype=str, keep_default_na=False)
    review_ids = set(review["canonical_performance_id"])
    add(checks, "difference_review_exact", len(review) == expected_difference_rows and review_ids == expected_difference_ids, {"rows": len(review), "ids": len(review_ids)}, {"rows": expected_difference_rows, "ids": len(expected_difference_ids)}, "Review queue exactly contains all rows with one or more differing comparable metrics.")

    advanced = pd.read_csv(ADVANCED, usecols=["canonical_performance_id"], dtype=str, keep_default_na=False)
    advanced_ids = set(advanced["canonical_performance_id"])
    core_fbref_ids = set()
    for frame in pd.read_csv(CORE, usecols=["canonical_performance_id", "present_in_fbref"], dtype=str, keep_default_na=False, chunksize=CHUNK_SIZE):
        core_fbref_ids.update(frame.loc[bool_series(frame["present_in_fbref"]), "canonical_performance_id"])
    add(checks, "fbref_advanced_exact", len(advanced) == expected_fbref_rows and advanced_ids == core_fbref_ids, {"rows": len(advanced), "ids": len(advanced_ids)}, {"rows": expected_fbref_rows, "ids": len(core_fbref_ids)}, "Advanced companion exactly covers accepted FBref spine rows.")

    unresolved = pd.read_csv(UNRESOLVED, dtype=str, keep_default_na=False)
    add(checks, "unresolved_fbref_accounting", len(unresolved) == expected_unresolved_rows and len(unresolved) + len(advanced) == 18_243, {"unresolved": len(unresolved), "accepted": len(advanced)}, {"unresolved": expected_unresolved_rows, "source_total": 18_243}, "All FBref source rows are accepted or retained for review.")

    source_manifest = pd.read_csv(OUTPUT / "source_manifest.csv", dtype=str, keep_default_na=False)
    bad_source_hashes = []
    for row in source_manifest.itertuples(index=False):
        path = ROOT / Path(row.source_file)
        if not path.exists() or sha256(path) != row.sha256:
            bad_source_hashes.append(row.source_file)
    add(checks, "source_manifest_hashes", not bad_source_hashes, len(bad_source_hashes), 0, f"mismatches={bad_source_hashes}")

    result = pd.DataFrame(checks)
    result.to_csv(OUTPUT / "independent_verification.csv", index=False, encoding="utf-8", lineterminator="\n")
    failures = int((~result["passed"].astype(bool)).sum())
    verification_summary = {
        "status": "pass" if failures == 0 else "fail",
        "checks": len(result),
        "failures": failures,
        "canonical_rows_scanned": observed_rows,
        "modeling_rows_scanned": model_rows,
    }
    (OUTPUT / "independent_verification.json").write_text(
        json.dumps(verification_summary, indent=2), encoding="utf-8"
    )

    output_paths = [
        path for path in OUTPUT.iterdir()
        if path.is_file() and path.name != "output_manifest.csv"
    ]
    manifest = pd.DataFrame([
        {
            "output_file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "file_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(output_paths)
    ])
    manifest.to_csv(OUTPUT / "output_manifest.csv", index=False, encoding="utf-8", lineterminator="\n")
    print(json.dumps(verification_summary, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

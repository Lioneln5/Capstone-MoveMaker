"""Build the low-risk Transfermarkt transfer core from cleaned inputs.

The output grain is exactly one row per cleaned transfer event. This stage only
adds many-to-one identity attributes from the historical player and club union
dimensions. Time-dependent facts (valuations, appearances, games, lineups, and
events) are deliberately excluded until their as-of or aggregation rules are
defined.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "Data" / "processed" / "transfermarkt_clean"
TABLES = CLEAN / "tables"
OUTPUT = ROOT / "Data" / "processed" / "transfermarkt_merged"

TRANSFERS_PATH = TABLES / "transfers_clean.csv"
PLAYERS_PATH = CLEAN / "player_identity_union.csv"
CLUBS_PATH = CLEAN / "club_dimension_historical.csv"

CORE_PATH = OUTPUT / "transfer_core.csv"
CHECKS_PATH = OUTPUT / "transfer_core_join_checks.csv"
DICTIONARY_PATH = OUTPUT / "transfer_core_field_dictionary.csv"
SUMMARY_PATH = OUTPUT / "transfer_core_summary.json"
README_PATH = OUTPUT / "README.md"


def clean_id(series: pd.Series) -> pd.Series:
    """Preserve identifiers as trimmed strings without spreadsheet decimals."""

    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transfers = pd.read_csv(TRANSFERS_PATH, dtype=str, keep_default_na=False)
    players = pd.read_csv(PLAYERS_PATH, dtype=str, keep_default_na=False)
    clubs = pd.read_csv(CLUBS_PATH, dtype=str, keep_default_na=False)

    for column in ["player_id", "from_club_id", "to_club_id"]:
        transfers[column] = clean_id(transfers[column])
    players["player_id"] = clean_id(players["player_id"])
    clubs["club_id"] = clean_id(clubs["club_id"])
    return transfers, players, clubs


def prepare_player_dimension(players: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "player_id": "player_id",
        "canonical_name": "player_canonical_name",
        "canonical_name_normalized": "player_canonical_name_normalized",
        "canonical_name_source": "player_canonical_name_source",
        "present_in_players_snapshot": "player_present_in_snapshot",
        "first_observed_date": "player_first_observed_date",
        "last_observed_date": "player_last_observed_date",
        "observation_count": "player_observation_count",
        "normalized_name_variant_count": "player_name_variant_count",
        "source_table_count": "player_source_table_count",
        "date_of_birth": "player_date_of_birth",
        "birth_year": "player_birth_year",
        "country_of_birth": "player_country_of_birth",
        "country_of_citizenship": "player_country_of_citizenship",
        "position": "player_snapshot_position",
        "sub_position": "player_snapshot_sub_position",
        "foot": "player_preferred_foot",
        "height_in_cm": "player_height_in_cm",
        "player_code": "player_code",
        "snapshot_url": "player_snapshot_url",
    }
    return players[list(columns)].rename(columns=columns)


def prepare_club_dimension(clubs: pd.DataFrame, role: str) -> pd.DataFrame:
    columns = {
        "club_id": f"{role}_club_id",
        "canonical_name": f"{role}_club_canonical_name",
        "canonical_name_normalized": f"{role}_club_canonical_name_normalized",
        "canonical_name_source": f"{role}_club_canonical_name_source",
        "present_in_clubs_snapshot": f"{role}_club_present_in_snapshot",
        "first_observed_date": f"{role}_club_first_observed_date",
        "last_observed_date": f"{role}_club_last_observed_date",
        "observation_count": f"{role}_club_observation_count",
        "normalized_name_variant_count": f"{role}_club_name_variant_count",
        "source_table_count": f"{role}_club_source_table_count",
    }
    return clubs[list(columns)].rename(columns=columns)


def merge_core(
    transfers: pd.DataFrame, players: pd.DataFrame, clubs: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    player_dimension = prepare_player_dimension(players)
    from_club_dimension = prepare_club_dimension(clubs, "from")
    to_club_dimension = prepare_club_dimension(clubs, "to")

    if not player_dimension["player_id"].is_unique:
        raise ValueError("player_identity_union.player_id must be unique")
    if not clubs["club_id"].is_unique:
        raise ValueError("club_dimension_historical.club_id must be unique")

    core = transfers.merge(
        player_dimension,
        on="player_id",
        how="left",
        validate="many_to_one",
        indicator="_player_merge",
    )
    core = core.merge(
        from_club_dimension,
        on="from_club_id",
        how="left",
        validate="many_to_one",
        indicator="_from_club_merge",
    )
    core = core.merge(
        to_club_dimension,
        on="to_club_id",
        how="left",
        validate="many_to_one",
        indicator="_to_club_merge",
    )

    merge_counts = {
        "player_dimension_rows_matched": int(core["_player_merge"].eq("both").sum()),
        "from_club_dimension_rows_matched": int(core["_from_club_merge"].eq("both").sum()),
        "to_club_dimension_rows_matched": int(core["_to_club_merge"].eq("both").sum()),
    }
    core["player_identity_joined"] = core["_player_merge"].eq("both")
    core["from_club_identity_joined"] = core["_from_club_merge"].eq("both")
    core["to_club_identity_joined"] = core["_to_club_merge"].eq("both")
    core["from_club_canonical_name_available"] = core["from_club_canonical_name"].ne("") & core[
        "from_club_canonical_name"
    ].notna()
    core["to_club_canonical_name_available"] = core["to_club_canonical_name"].ne("") & core[
        "to_club_canonical_name"
    ].notna()
    core["is_same_club_id"] = core["from_club_id"].eq(core["to_club_id"])
    core = core.drop(columns=["_player_merge", "_from_club_merge", "_to_club_merge"])
    return core, merge_counts


def build_checks(
    transfers: pd.DataFrame,
    players: pd.DataFrame,
    clubs: pd.DataFrame,
    core: pd.DataFrame,
    merge_counts: dict[str, int],
) -> pd.DataFrame:
    checks: list[dict[str, object]] = []

    def add(check: str, actual: object, expected: object, passed: bool, notes: str) -> None:
        checks.append(
            {
                "check": check,
                "actual": actual,
                "expected": expected,
                "status": "OK" if passed else "FAIL",
                "notes": notes,
            }
        )

    input_rows = len(transfers)
    add("Output row preservation", len(core), input_rows, len(core) == input_rows, "Left joins must retain one row per transfer.")
    add(
        "Transfer event ID uniqueness",
        core["transfer_event_id"].nunique(),
        len(core),
        core["transfer_event_id"].is_unique,
        "Primary transfer event key remains unique after every merge.",
    )
    natural_key = ["player_id", "transfer_date", "from_club_id", "to_club_id"]
    duplicate_natural_keys = int(core.duplicated(natural_key, keep=False).sum())
    add("Transfer natural-key duplicates", duplicate_natural_keys, 0, duplicate_natural_keys == 0, "No join fanout or duplicate source transfers.")
    add("Player dimension key uniqueness", int(players["player_id"].duplicated().sum()), 0, players["player_id"].is_unique, "Required for a many-to-one join.")
    add("Club dimension key uniqueness", int(clubs["club_id"].duplicated().sum()), 0, clubs["club_id"].is_unique, "Required for two many-to-one joins.")

    for label, actual in merge_counts.items():
        add(label.replace("_", " ").title(), actual, input_rows, actual == input_rows, "Historical union dimension should cover every transfer fact ID.")

    transfer_columns_preserved = all(core[column].astype("string").fillna("").equals(transfers[column].astype("string").fillna("")) for column in transfers.columns)
    add("Source transfer fields preserved", transfer_columns_preserved, True, transfer_columns_preserved, "Every cleaned transfer column remains unchanged and in original row order.")
    add(
        "Player identity join flags",
        int(core["player_identity_joined"].sum()),
        input_rows,
        bool(core["player_identity_joined"].all()),
        "A historical player-dimension row exists for every transfer.",
    )
    add(
        "Origin club identity join flags",
        int(core["from_club_identity_joined"].sum()),
        input_rows,
        bool(core["from_club_identity_joined"].all()),
        "A historical club-dimension row exists for every origin club ID.",
    )
    add(
        "Destination club identity join flags",
        int(core["to_club_identity_joined"].sum()),
        input_rows,
        bool(core["to_club_identity_joined"].all()),
        "A historical club-dimension row exists for every destination club ID.",
    )
    return pd.DataFrame(checks)


def field_definition(column: str, transfer_columns: set[str]) -> tuple[str, str, str]:
    if column in transfer_columns:
        return "transfer_fact", "transfers_clean.csv", "Preserved cleaned transfer fact field; see the Transfermarkt cleaned-column dictionary."
    quality_flags = {
        "player_identity_joined": "True when player_id matched a row in the historical player union.",
        "from_club_identity_joined": "True when from_club_id matched a row in the historical club union.",
        "to_club_identity_joined": "True when to_club_id matched a row in the historical club union.",
        "from_club_canonical_name_available": "True when the origin club has a resolved historical canonical name.",
        "to_club_canonical_name_available": "True when the destination club has a resolved historical canonical name.",
        "is_same_club_id": "True when origin and destination club IDs are identical.",
    }
    if column in quality_flags:
        return "derived_quality_flag", "derived", quality_flags[column]
    if column.startswith("player_"):
        note = "Historical player identity attribute. Snapshot-labelled fields are descriptive references, not historical as-of features."
        return "player_identity", "player_identity_union.csv", note
    if column.startswith("from_club_"):
        return "origin_club_identity", "club_dimension_historical.csv", "Historical union identity attribute for the origin club."
    if column.startswith("to_club_"):
        return "destination_club_identity", "club_dimension_historical.csv", "Historical union identity attribute for the destination club."
    return "derived_quality_flag", "derived", "Join-availability or merge-quality flag."


def build_dictionary(core: pd.DataFrame, transfer_columns: set[str]) -> pd.DataFrame:
    rows = []
    for position, column in enumerate(core.columns, start=1):
        group, source, definition = field_definition(column, transfer_columns)
        usage = "reference_or_quality_control" if group != "transfer_fact" else "preserved_source_fact"
        leakage_rule = "Review before modeling" if "snapshot" in column else "No new temporal fact added in this merge"
        rows.append(
            {
                "column_order": position,
                "column": column,
                "group": group,
                "source": source,
                "definition": definition,
                "intended_usage": usage,
                "leakage_rule": leakage_rule,
            }
        )
    return pd.DataFrame(rows)


def write_readme() -> None:
    README_PATH.write_text(
        "# Transfermarkt transfer core\n\n"
        "Generated by `scripts/build_transfermarkt_transfer_core.py`.\n\n"
        "## Grain\n\n"
        "One row per cleaned Transfermarkt transfer event (`transfer_event_id`).\n\n"
        "## Included merges\n\n"
        "- `transfers_clean.csv` as the preserved fact-table spine.\n"
        "- `player_identity_union.csv` on `player_id` (many-to-one).\n"
        "- `club_dimension_historical.csv` twice, on `from_club_id` and `to_club_id` (many-to-one).\n\n"
        "## Deliberately excluded\n\n"
        "Valuations, appearances, games, club games, lineups, events, competitions, countries, and national-team snapshots are not merged here. They require temporal, aggregation, or explicit analytical rules.\n\n"
        "## Guardrails\n\n"
        "All joins are left joins with many-to-one validation. The build fails if row count changes, transfer keys duplicate, dimension keys duplicate, a historical identity row is unmatched, or any cleaned transfer fact changes.\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    transfers, players, clubs = read_inputs()
    core, merge_counts = merge_core(transfers, players, clubs)
    checks = build_checks(transfers, players, clubs, core, merge_counts)

    failed = checks[checks["status"].eq("FAIL")]
    if not failed.empty:
        raise RuntimeError("Transfer-core validation failed:\n" + failed.to_string(index=False))

    core.to_csv(CORE_PATH, index=False, encoding="utf-8", lineterminator="\n")
    checks.to_csv(CHECKS_PATH, index=False, encoding="utf-8", lineterminator="\n")
    dictionary = build_dictionary(core, set(transfers.columns))
    dictionary.to_csv(DICTIONARY_PATH, index=False, encoding="utf-8", lineterminator="\n")
    write_readme()

    summary = {
        "grain": "One cleaned player transfer event.",
        "primary_key": "transfer_event_id",
        "natural_key": ["player_id", "transfer_date", "from_club_id", "to_club_id"],
        "input_transfer_rows": len(transfers),
        "output_rows": len(core),
        "output_columns": len(core.columns),
        "unique_players": int(core["player_id"].nunique()),
        "unique_origin_clubs": int(core["from_club_id"].nunique()),
        "unique_destination_clubs": int(core["to_club_id"].nunique()),
        "same_club_id_rows": int(core["is_same_club_id"].sum()),
        "origin_canonical_name_available_rows": int(core["from_club_canonical_name_available"].sum()),
        "destination_canonical_name_available_rows": int(core["to_club_canonical_name_available"].sum()),
        "checks_passed": int(checks["status"].eq("OK").sum()),
        "checks_failed": int(checks["status"].eq("FAIL").sum()),
        "inputs": {
            str(TRANSFERS_PATH.relative_to(ROOT)): sha256_file(TRANSFERS_PATH),
            str(PLAYERS_PATH.relative_to(ROOT)): sha256_file(PLAYERS_PATH),
            str(CLUBS_PATH.relative_to(ROOT)): sha256_file(CLUBS_PATH),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary["output_sha256"] = sha256_file(CORE_PATH)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {CORE_PATH.relative_to(ROOT)}: {len(core):,} rows x {len(core.columns)} columns")
    print(f"All {len(checks)} validation checks passed")
    print(f"Output SHA-256: {summary['output_sha256']}")


if __name__ == "__main__":
    main()

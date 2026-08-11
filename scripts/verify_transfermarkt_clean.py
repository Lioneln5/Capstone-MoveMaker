"""Independent verification for the generated Transfermarkt cleaning layer."""

from __future__ import annotations

import hashlib
import json
from itertools import zip_longest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "Data" / "processed" / "transfermarkt_clean"
OUTPUTS = ROOT / "outputs" / "transfermarkt_cleaning"
CHUNK_SIZE = 250_000

KEYS = {
    "appearances": ["appearance_id"],
    "club_games": ["game_id", "club_id"],
    "clubs": ["club_id"],
    "competitions": ["competition_id"],
    "countries": ["country_id"],
    "game_events": ["game_event_id"],
    "game_lineups": ["game_lineups_id"],
    "games": ["game_id"],
    "national_teams": ["national_team_id"],
    "player_valuations": ["player_id", "date"],
    "players": ["player_id"],
    "transfers": ["player_id", "transfer_date", "from_club_id", "to_club_id"],
}

NUMERIC_KEY_COLUMNS = {
    "game_id", "club_id", "player_id", "from_club_id", "to_club_id",
    "national_team_id",
}
DATE_KEY_COLUMNS = {"date", "transfer_date"}

CLUB_ID_COLUMNS = {
    "clubs": ["club_id"],
    "players": ["current_club_id"],
    "transfers": ["from_club_id", "to_club_id"],
    "player_valuations": ["current_club_id"],
    "games": ["home_club_id", "away_club_id"],
    "club_games": ["club_id", "opponent_id"],
    "appearances": ["player_club_id", "player_current_club_id"],
    "game_lineups": ["club_id"],
    "game_events": ["club_id"],
}

PLAYER_ID_COLUMNS = {
    "players": ["player_id"],
    "transfers": ["player_id"],
    "player_valuations": ["player_id"],
    "appearances": ["player_id"],
    "game_lineups": ["player_id"],
    "game_events": ["player_id", "player_in_id", "player_assist_id"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonicalize_key(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for column in columns:
        values = frame[column].astype("string").str.strip()
        values = values.mask(values.eq(""))
        if column in NUMERIC_KEY_COLUMNS:
            values = pd.to_numeric(values, errors="coerce").astype("Int64").astype("string")
        elif column in DATE_KEY_COLUMNS:
            values = pd.to_datetime(values, errors="coerce", format="mixed").dt.strftime("%Y-%m-%d").astype("string")
        result[column] = values
    return result


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(PROCESSED / "source_file_manifest.csv", encoding="utf-8-sig")
    checks = pd.read_csv(PROCESSED / "transfermarkt_cleaning_checks.csv", encoding="utf-8-sig")
    results: list[dict] = []
    club_ids: set[int] = set()
    player_ids: set[int] = set()

    def record(check: str, observed, expected, passed: bool, notes: str) -> None:
        results.append(
            {
                "check": check,
                "observed": observed,
                "expected": expected,
                "passed": bool(passed),
                "notes": notes,
            }
        )

    for row in manifest.itertuples(index=False):
        source = ROOT / row.source_relative_path
        backup = ROOT / row.backup_relative_path
        clean = ROOT / row.clean_relative_path
        source_hash = sha256(source)
        backup_hash = sha256(backup)
        clean_hash = sha256(clean)
        record(
            f"{row.table}: source hash unchanged",
            source_hash,
            row.source_sha256,
            source_hash == row.source_sha256,
            "Recomputed independently from the raw source.",
        )
        record(
            f"{row.table}: backup equals source",
            backup_hash,
            source_hash,
            backup_hash == source_hash,
            "Exact binary source preservation.",
        )
        record(
            f"{row.table}: clean hash matches manifest",
            clean_hash,
            row.clean_sha256,
            clean_hash == row.clean_sha256,
            "Detects post-cleaning modification.",
        )

        raw_reader = pd.read_csv(
            source,
            usecols=KEYS[row.table],
            dtype="string",
            keep_default_na=False,
            chunksize=CHUNK_SIZE,
            encoding="utf-8-sig",
        )
        clean_reader = pd.read_csv(
            clean,
            usecols=KEYS[row.table] + ["source_row_number"],
            dtype="string",
            keep_default_na=False,
            chunksize=CHUNK_SIZE,
            encoding="utf-8-sig",
        )
        raw_rows = 0
        clean_rows = 0
        key_mismatches = 0
        source_row_mismatches = 0
        for raw_chunk, clean_chunk in zip_longest(raw_reader, clean_reader):
            if raw_chunk is None or clean_chunk is None:
                key_mismatches += 1
                continue
            raw_key = canonicalize_key(raw_chunk.reset_index(drop=True), KEYS[row.table])
            clean_key = canonicalize_key(clean_chunk.reset_index(drop=True), KEYS[row.table])
            key_mismatches += int((raw_key.ne(clean_key) & ~(raw_key.isna() & clean_key.isna())).any(axis=1).sum())
            expected_rows = pd.Series(
                range(raw_rows + 2, raw_rows + 2 + len(raw_chunk)), dtype="Int64"
            )
            actual_rows = pd.to_numeric(clean_chunk["source_row_number"], errors="coerce").astype("Int64").reset_index(drop=True)
            source_row_mismatches += int(actual_rows.ne(expected_rows).sum())
            raw_rows += len(raw_chunk)
            clean_rows += len(clean_chunk)
        record(
            f"{row.table}: row count preserved",
            clean_rows,
            raw_rows,
            clean_rows == raw_rows == int(row.source_row_count),
            "Parsed independently in chunks.",
        )
        record(
            f"{row.table}: natural keys preserved in source order",
            key_mismatches,
            0,
            key_mismatches == 0,
            "Compares canonical raw and clean natural-key values row by row.",
        )
        record(
            f"{row.table}: source row provenance",
            source_row_mismatches,
            0,
            source_row_mismatches == 0,
            "source_row_number must point back to the physical CSV row.",
        )

        selected = sorted(set(CLUB_ID_COLUMNS.get(row.table, [])) | set(PLAYER_ID_COLUMNS.get(row.table, [])))
        if selected:
            for chunk in pd.read_csv(clean, usecols=selected, chunksize=CHUNK_SIZE, encoding="utf-8-sig"):
                for column in CLUB_ID_COLUMNS.get(row.table, []):
                    club_ids.update(int(value) for value in pd.to_numeric(chunk[column], errors="coerce").dropna().unique())
                for column in PLAYER_ID_COLUMNS.get(row.table, []):
                    player_ids.update(int(value) for value in pd.to_numeric(chunk[column], errors="coerce").dropna().unique())

    blocking_failed = int((checks["severity"].eq("blocking") & ~checks["passed"].astype(bool)).sum())
    record(
        "Generated blocking checks",
        blocking_failed,
        0,
        blocking_failed == 0,
        "All declared blocking source, key, type, sign, and reconciliation checks must pass.",
    )

    club_union = pd.read_csv(PROCESSED / "club_dimension_historical.csv", usecols=["club_id"], encoding="utf-8-sig")
    player_union = pd.read_csv(PROCESSED / "player_identity_union.csv", usecols=["player_id"], encoding="utf-8-sig")
    club_union_ids = set(club_union["club_id"].dropna().astype(int))
    player_union_ids = set(player_union["player_id"].dropna().astype(int))
    record(
        "Historical club union covers every observed club ID",
        len(club_ids - club_union_ids),
        0,
        club_ids <= club_union_ids,
        "No historical fact row is lost because clubs.csv is a limited snapshot.",
    )
    record(
        "Historical player union covers every observed player ID",
        len(player_ids - player_union_ids),
        0,
        player_ids <= player_union_ids,
        "No historical fact row is lost because players.csv is a limited snapshot.",
    )

    event_minutes = pd.read_csv(
        PROCESSED / "tables" / "game_events_clean.csv",
        usecols=["minute", "minute_raw", "event_minute_available"],
        encoding="utf-8-sig",
    )
    sentinel = event_minutes["minute_raw"].eq(-1)
    sentinel_errors = int(
        (
            sentinel
            & (
                event_minutes["minute"].notna()
                | event_minutes["event_minute_available"].fillna(True).astype(bool)
            )
        ).sum()
    )
    record(
        "Event minute -1 sentinel is reversible",
        sentinel_errors,
        0,
        sentinel_errors == 0,
        "minute_raw preserves -1 while analytical minute is missing and availability is false.",
    )

    verification = pd.DataFrame(results)
    verification.to_csv(OUTPUTS / "independent_verification.csv", index=False, encoding="utf-8-sig")
    summary = {
        "checks": len(verification),
        "passed": int(verification["passed"].sum()),
        "failed": int((~verification["passed"]).sum()),
        "all_passed": bool(verification["passed"].all()),
        "raw_rows_reconciled": int(manifest["source_row_count"].sum()),
        "observed_club_ids": len(club_ids),
        "observed_player_ids": len(player_ids),
        "event_minute_sentinel_rows": int(sentinel.sum()),
    }
    (OUTPUTS / "independent_verification.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if not summary["all_passed"]:
        failed = verification.loc[~verification["passed"], ["check", "observed"]].to_dict("records")
        raise RuntimeError(f"Independent verification failures: {failed}")


if __name__ == "__main__":
    main()

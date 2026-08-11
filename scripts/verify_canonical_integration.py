from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "Data" / "processed"
OUTPUT = PROCESSED / "canonical_integration"
TM = PROCESSED / "transfermarkt_clean" / "tables"
DL = PROCESSED / "football_datalake_clean" / "tables"
XW = PROCESSED / "integration_crosswalks"
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


def collect_ids(path: Path, column: str) -> set[str]:
    values = set()
    for chunk in pd.read_csv(path, usecols=[column], dtype="string", chunksize=CHUNK, low_memory=False):
        values.update(text_id(value) for value in chunk[column] if text_id(value))
    return values


def add(checks, name, passed, observed, expected, detail):
    checks.append(
        {
            "check": name,
            "status": "pass" if bool(passed) else "fail",
            "observed": observed,
            "expected": expected,
            "detail": detail,
        }
    )


def main():
    required = [
        "canonical_player_dimension.csv", "player_snapshot_evidence.csv", "canonical_club_dimension.csv",
        "club_snapshot_evidence.csv", "valuation_source_evidence.csv", "canonical_valuation_history.csv",
        "canonical_latest_valuation.csv", "valuation_conflict_review.csv", "transfer_source_evidence.csv", "canonical_transfer_history.csv",
        "transfer_conflict_review.csv", "integration_summary.csv", "integration_checks.csv",
        "integration_table_dictionary.csv", "integration_field_dictionary.csv", "source_manifest.csv", "README.md",
    ]
    checks = []
    missing = [name for name in required if not (OUTPUT / name).exists()]
    add(checks, "required_outputs", not missing, len(required) - len(missing), len(required), f"missing={missing}")
    if missing:
        raise SystemExit(json.dumps(checks, indent=2))

    players = pd.read_csv(OUTPUT / "canonical_player_dimension.csv", dtype="string", low_memory=False)
    clubs = pd.read_csv(OUTPUT / "canonical_club_dimension.csv", dtype="string", low_memory=False)
    valuation_evidence = pd.read_csv(OUTPUT / "valuation_source_evidence.csv", dtype="string", low_memory=False)
    valuation_history = pd.read_csv(OUTPUT / "canonical_valuation_history.csv", dtype="string", low_memory=False)
    latest = pd.read_csv(OUTPUT / "canonical_latest_valuation.csv", dtype="string", low_memory=False)
    valuation_review = pd.read_csv(OUTPUT / "valuation_conflict_review.csv", dtype="string", low_memory=False)
    transfer_evidence = pd.read_csv(OUTPUT / "transfer_source_evidence.csv", dtype="string", low_memory=False)
    transfer_history = pd.read_csv(OUTPUT / "canonical_transfer_history.csv", dtype="string", low_memory=False)
    transfer_review = pd.read_csv(OUTPUT / "transfer_conflict_review.csv", dtype="string", low_memory=False)

    xwalk_player_ids = collect_ids(XW / "player_identity_crosswalk.csv", "canonical_player_id")
    fact_player_ids = (
        collect_ids(TM / "player_valuations_clean.csv", "player_id")
        | collect_ids(DL / "player_market_value_clean.csv", "player_id")
        | collect_ids(TM / "transfers_clean.csv", "player_id")
        | collect_ids(DL / "transfer_history_clean.csv", "player_id")
    )
    expected_player_ids = xwalk_player_ids | fact_player_ids
    output_player_ids = set(players["canonical_player_id"].map(text_id))
    add(checks, "canonical_player_union", output_player_ids == expected_player_ids, len(output_player_ids), len(expected_player_ids), "Player dimension equals identity-crosswalk plus valuation/transfer ID union.")
    add(checks, "canonical_player_unique", players["canonical_player_id"].is_unique, players["canonical_player_id"].nunique(), len(players), "One row per canonical player ID.")
    expected_fact_only = fact_player_ids - xwalk_player_ids
    observed_fact_only = set(players.loc[players["identity_record_type"].eq("fact_only_stub"), "canonical_player_id"].map(text_id))
    add(checks, "fact_only_stub_accounting", observed_fact_only == expected_fact_only, len(observed_fact_only), len(expected_fact_only), "Only IDs absent from the identity crosswalk are labeled fact-only stubs.")

    expected_club_ids = collect_ids(XW / "club_identity_crosswalk.csv", "canonical_club_id")
    output_club_ids = set(clubs["canonical_club_id"].map(text_id))
    add(checks, "canonical_club_union", output_club_ids == expected_club_ids, len(output_club_ids), len(expected_club_ids), "Club dimension exactly retains the comprehensive club identity crosswalk.")
    add(checks, "canonical_club_unique", clubs["canonical_club_id"].is_unique, clubs["canonical_club_id"].nunique(), len(clubs), "One row per canonical club ID.")

    valuation_source_counts = valuation_evidence["source_dataset"].value_counts().to_dict()
    add(checks, "valuation_source_rows", valuation_source_counts == {"supplemental_datalake": 901429, "current_transfermarkt": 507815}, valuation_source_counts, {"supplemental_datalake": 901429, "current_transfermarkt": 507815}, "Every cleaned valuation source row is retained.")
    add(checks, "valuation_source_record_unique", not valuation_evidence.duplicated(["source_dataset", "source_record_id"]).any(), int(valuation_evidence.duplicated(["source_dataset", "source_record_id"]).sum()), 0, "Source record provenance is unique within each source.")
    add(checks, "valuation_spine_unique", valuation_history["canonical_valuation_id"].is_unique, valuation_history["canonical_valuation_id"].nunique(), len(valuation_history), "One row per canonical player-date key.")
    valuation_evidence_sum = int(pd.to_numeric(valuation_history["evidence_count"], errors="coerce").sum())
    add(checks, "valuation_evidence_accounting", valuation_evidence_sum == len(valuation_evidence), valuation_evidence_sum, len(valuation_evidence), "Spine evidence counts reconcile to the row-preserving evidence table.")
    valuation_overlap = int(valuation_history["cross_source_overlap"].str.lower().eq("true").sum())
    valuation_conflicts = int(valuation_history["value_conflict"].str.lower().eq("true").sum())
    add(checks, "valuation_overlap_reconciliation", valuation_overlap == 410, valuation_overlap, 410, "Cross-source player-date overlap matches the source audit.")
    add(checks, "valuation_conflict_reconciliation", valuation_conflicts == 187, valuation_conflicts, 187, "Conflicting player-date values match the source audit.")
    conflict_value_present = valuation_history.loc[valuation_history["value_conflict"].str.lower().eq("true"), "canonical_market_value_eur"].notna().sum()
    add(checks, "valuation_conflicts_unresolved", conflict_value_present == 0, int(conflict_value_present), 0, "No canonical value is invented for conflicting evidence.")
    expected_valuation_review = valuation_history[
        valuation_history["value_conflict"].str.lower().eq("true")
        | valuation_history["within_source_duplicate"].str.lower().eq("true")
        | valuation_history["natural_key_missing"].str.lower().eq("true")
    ]
    add(checks, "valuation_review_queue", set(valuation_review["canonical_valuation_id"]) == set(expected_valuation_review["canonical_valuation_id"]) and len(valuation_review) == len(expected_valuation_review), len(valuation_review), len(expected_valuation_review), "Valuation review exactly contains conflicts, within-source duplicates, and incomplete keys.")
    add(checks, "valuation_player_references", set(valuation_evidence["canonical_player_id"].map(text_id)) - {""} <= output_player_ids, len(set(valuation_evidence["canonical_player_id"].map(text_id)) - {""}), len(set(valuation_evidence["canonical_player_id"].map(text_id)) - {""}), "Every valuation player resolves to the canonical dimension.")

    add(checks, "latest_player_unique", latest["canonical_player_id"].is_unique, latest["canonical_player_id"].nunique(), len(latest), "One latest row per player.")
    max_dates = valuation_history[valuation_history["valuation_date_valid"].str.lower().eq("true")].groupby("canonical_player_id")["valuation_date"].max()
    latest_dates = latest.set_index("canonical_player_id")["valuation_date"]
    latest_ok = latest_dates.index.equals(max_dates.index) and latest_dates.equals(max_dates.loc[latest_dates.index])
    add(checks, "latest_derived_from_history", latest_ok, len(latest_dates), len(max_dates), "Latest valuations use each player's maximum canonical history date.")

    transfer_source_counts = transfer_evidence["source_dataset"].value_counts().to_dict()
    add(checks, "transfer_source_rows", transfer_source_counts == {"supplemental_datalake": 1101440, "current_transfermarkt": 35139}, transfer_source_counts, {"supplemental_datalake": 1101440, "current_transfermarkt": 35139}, "Every cleaned transfer source row is retained.")
    add(checks, "transfer_source_record_unique", not transfer_evidence.duplicated(["source_dataset", "source_record_id"]).any(), int(transfer_evidence.duplicated(["source_dataset", "source_record_id"]).sum()), 0, "Source record provenance is unique within each source.")
    add(checks, "transfer_spine_unique", transfer_history["canonical_transfer_event_id"].is_unique, transfer_history["canonical_transfer_event_id"].nunique(), len(transfer_history), "One row per canonical complete event; incomplete records retain source-specific IDs.")
    transfer_evidence_sum = int(pd.to_numeric(transfer_history["evidence_count"], errors="coerce").sum())
    add(checks, "transfer_evidence_accounting", transfer_evidence_sum == len(transfer_evidence), transfer_evidence_sum, len(transfer_evidence), "Spine evidence counts reconcile to the row-preserving evidence table.")

    canonical_overlap = int(transfer_history["cross_source_overlap"].str.lower().eq("true").sum())
    key_columns = ["canonical_player_id", "transfer_date", "from_club_id", "to_club_id"]
    current_keys = set(map(tuple, transfer_evidence.loc[transfer_evidence["source_dataset"].eq("current_transfermarkt"), key_columns].fillna("").to_numpy()))
    datalake_keys = list(map(tuple, transfer_evidence.loc[transfer_evidence["source_dataset"].eq("supplemental_datalake"), key_columns].fillna("").to_numpy()))
    shared_source_rows = sum(key in current_keys for key in datalake_keys)
    duplicate_shared_rows = shared_source_rows - canonical_overlap
    add(checks, "transfer_overlap_accounting", shared_source_rows == 30180 and canonical_overlap == 30178 and duplicate_shared_rows == 2, {"shared_datalake_rows": shared_source_rows, "canonical_shared_events": canonical_overlap, "duplicate_shared_rows": duplicate_shared_rows}, {"shared_datalake_rows": 30180, "canonical_shared_events": 30178, "duplicate_shared_rows": 2}, "The source audit counted rows; two duplicated datalake rows collapse into their canonical shared events.")

    fee_conflicts = int(transfer_history["transfer_fee_conflict"].str.lower().eq("true").sum())
    market_conflicts = int(transfer_history["market_value_conflict"].str.lower().eq("true").sum())
    add(checks, "transfer_fee_conflict_reconciliation", fee_conflicts == 38, fee_conflicts, 38, "Fee conflicts match the source audit.")
    add(checks, "transfer_value_conflict_reconciliation", market_conflicts == 175, market_conflicts, 175, "Market-value-at-transfer conflicts match the source audit.")
    fee_conflict_values = transfer_history.loc[transfer_history["transfer_fee_conflict"].str.lower().eq("true"), "canonical_transfer_fee_eur"].notna().sum()
    add(checks, "transfer_fee_conflicts_unresolved", fee_conflict_values == 0, int(fee_conflict_values), 0, "Conflicting transfer fees do not receive an invented canonical value.")
    add(checks, "zero_fee_semantics", not transfer_history["canonical_transfer_fee_status"].str.contains("free", case=False, na=False).any(), int(transfer_history["canonical_transfer_fee_status"].str.contains("free", case=False, na=False).sum()), 0, "Literal zeros remain unclassified rather than inferred free transfers.")

    expected_review = transfer_history[
        transfer_history["event_conflict"].str.lower().eq("true")
        | transfer_history["within_source_duplicate"].str.lower().eq("true")
        | transfer_history["natural_key_missing"].str.lower().eq("true")
    ]
    add(checks, "transfer_review_queue", set(transfer_review["canonical_transfer_event_id"]) == set(expected_review["canonical_transfer_event_id"]), len(transfer_review), len(expected_review), "Conflict review exactly contains conflicts, within-source duplicates, and incomplete keys.")
    add(checks, "transfer_player_references", set(transfer_evidence["canonical_player_id"].map(text_id)) - {""} <= output_player_ids, len(set(transfer_evidence["canonical_player_id"].map(text_id)) - {""}), len(set(transfer_evidence["canonical_player_id"].map(text_id)) - {""}), "Every transfer player resolves to the canonical dimension.")
    add(checks, "transfer_from_club_references", set(transfer_evidence["from_club_id"].map(text_id)) - {""} <= output_club_ids, len(set(transfer_evidence["from_club_id"].map(text_id)) - {""}), len(set(transfer_evidence["from_club_id"].map(text_id)) - {""}), "Every origin club resolves to the canonical club dimension.")
    add(checks, "transfer_to_club_references", set(transfer_evidence["to_club_id"].map(text_id)) - {""} <= output_club_ids, len(set(transfer_evidence["to_club_id"].map(text_id)) - {""}), len(set(transfer_evidence["to_club_id"].map(text_id)) - {""}), "Every destination club resolves to the canonical club dimension.")

    source_manifest = pd.read_csv(OUTPUT / "source_manifest.csv", dtype="string")
    source_hashes_ok = True
    for row in source_manifest.itertuples(index=False):
        path = ROOT / str(row.source_file)
        source_hashes_ok &= path.exists() and sha256(path) == row.sha256
    add(checks, "source_manifest_hashes", source_hashes_ok, int(source_hashes_ok), 1, "Independent SHA-256 recomputation matches every recorded source.")

    result = pd.DataFrame(checks)
    result.to_csv(OUTPUT / "independent_verification.csv", index=False, encoding="utf-8-sig")
    output_files = sorted(path for path in OUTPUT.glob("*.csv") if path.name != "output_manifest.csv")
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

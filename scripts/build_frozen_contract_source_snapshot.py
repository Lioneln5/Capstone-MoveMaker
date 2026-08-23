"""Reconstruct the exact 2,425-event source used by contract Phases 1-5.

The live Capology and integration layers are append-only and now include the
2024-25 extension pages.  The completed Phase 1-5 evaluations must continue to
resolve to the exact upstream bytes they originally used, so this script
rebuilds the pre-2024-25 source into a versioned archive and asserts the hashes
recorded by the frozen model artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import build_capology_contracts as capology
import build_contract_extension_integration as integration
import verify_contract_extension_integration as integration_verifier


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_ROOT = ROOT / "Data" / "processed" / "frozen_contract_sources" / "phase1_5_2026-08-11"
SNAPSHOT_CONTRACTS = SNAPSHOT_ROOT / "capology_contracts"
SNAPSHOT_INTEGRATION = SNAPSHOT_ROOT / "contract_extension_integration"

EXPECTED_FILES = 29
EXPECTED_RAW_ROWS = 2_431
EXPECTED_EVENTS = 2_425
EXPECTED_MASTER_SHA256 = "28a7e369d37bdba0eafa25a7b2e11977d836075c1e7db58a4dce62fe015d9127"
EXPECTED_VERIFICATION_SHA256 = "7942f72a5ab25c85909c741662da11dce657ac189519273e6fd1ce93cbbaf499"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    extension_files = sorted(
        path
        for path in capology.LEAGUES.glob("*/extensions/*.csv")
        if "2024_2025" not in path.name
    )
    if len(extension_files) != EXPECTED_FILES:
        raise RuntimeError(f"Expected {EXPECTED_FILES} pre-2024-25 extension files; found {len(extension_files)}")

    fbref_map, identity_map = capology.build_player_map()
    club_map, _ = capology.build_club_map()
    source = pd.concat(
        [capology.read_source(path, "extension", fbref_map, identity_map, club_map) for path in extension_files],
        ignore_index=True,
    )
    if len(source) != EXPECTED_RAW_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_RAW_ROWS} raw rows; found {len(source)}")

    conflicts = source.groupby("capology_player_url_suffix")["player_name_normalized"].nunique()
    source["capology_suffix_name_conflict"] = source["capology_player_url_suffix"].map(conflicts).gt(1)
    canonical = capology.collapse_extensions(source)
    if len(canonical) != EXPECTED_EVENTS:
        raise RuntimeError(f"Expected {EXPECTED_EVENTS} canonical events; found {len(canonical)}")

    SNAPSHOT_CONTRACTS.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_INTEGRATION.mkdir(parents=True, exist_ok=True)
    canonical_path = SNAPSHOT_CONTRACTS / "canonical_extension_events.csv"
    capology.write_csv(canonical, canonical_path)

    integration.EXTENSION_PATH = canonical_path
    integration.OUTPUT = SNAPSHOT_INTEGRATION
    integration.main()

    integration_verifier.CONTRACTS = SNAPSHOT_CONTRACTS
    integration_verifier.OUTPUT = SNAPSHOT_INTEGRATION
    integration_verifier.main()

    master_path = SNAPSHOT_INTEGRATION / "extension_modeling_master.csv"
    verification_path = SNAPSHOT_INTEGRATION / "independent_verification.json"
    master_hash = sha256(master_path)
    verification_hash = sha256(verification_path)
    if master_hash != EXPECTED_MASTER_SHA256:
        raise RuntimeError(f"Frozen master hash mismatch: {master_hash}")
    if verification_hash != EXPECTED_VERIFICATION_SHA256:
        raise RuntimeError(f"Frozen verification hash mismatch: {verification_hash}")

    summary = {
        "snapshot": "phase1_5_2026-08-11",
        "extension_files": len(extension_files),
        "raw_extension_rows": len(source),
        "canonical_extension_events": len(canonical),
        "extension_modeling_master_sha256": master_hash,
        "independent_verification_sha256": verification_hash,
        "purpose": "Immutable source resolution for the already-completed Phase 1-5 model evaluations.",
    }
    (SNAPSHOT_ROOT / "snapshot_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (SNAPSHOT_ROOT / "README.md").write_text(
        "# Frozen contract source for Phases 1-5\n\n"
        "This archive reconstructs the exact 2,425-event integration source used by the completed "
        "Phase 1-5 contract diagnostics before the append-only 2024-25 Capology update. The live "
        "canonical and integration paths contain 2,805 events. Do not train new work from this "
        "archive; it exists only to keep the historical model artifacts independently verifiable.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

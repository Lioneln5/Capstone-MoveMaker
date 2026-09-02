#!/usr/bin/env python3
"""Verify that every frozen Engine V1 deployment artifact is byte-identical."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = {item["path"]: item["sha256"] for item in manifest["files"]}
    artifact_root = ROOT / manifest["artifact_root"]
    actual_paths = {
        path.relative_to(ROOT).as_posix()
        for path in artifact_root.rglob("*")
        if path.is_file() and "_backups" not in path.parts
    }

    errors: list[str] = []
    expected_paths = set(expected)
    for missing in sorted(expected_paths - actual_paths):
        errors.append(f"missing frozen file: {missing}")
    for extra in sorted(actual_paths - expected_paths):
        errors.append(f"unexpected file in frozen artifact root: {extra}")
    for relative_path in sorted(expected_paths & actual_paths):
        observed = sha256(ROOT / relative_path)
        if observed != expected[relative_path]:
            errors.append(
                f"hash mismatch: {relative_path} expected={expected[relative_path]} "
                f"observed={observed}"
            )

    if len(expected) != manifest["file_count"]:
        errors.append(
            f"manifest file_count={manifest['file_count']} but lists {len(expected)} files"
        )

    if errors:
        raise SystemExit("Engine V1 freeze verification failed:\n- " + "\n- ".join(errors))

    print(
        f"PASS: {len(expected)} Engine V1 files are byte-identical to "
        f"{manifest['release_id']} ({manifest['source_commit']})."
    )


if __name__ == "__main__":
    main()

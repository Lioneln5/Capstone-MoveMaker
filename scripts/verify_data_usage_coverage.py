"""Independently check the data-usage audit without fitting/scoring models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def is_raw_or_external_source(relative_path: str) -> bool:
    prefixes = (
        "Data/leagues/",
        "Data/2017-2024/",
        "Data/2024-2026/",
        "Data/injury/",
        "Data/Open-Data-Master/",
        "Data/TransferMarkt/",
        "Data/source_backups/",
        "football-datasets-main/datalake/",
    )
    return relative_path.startswith(prefixes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    out = Path(parser.parse_args().audit).resolve()
    if not out.is_relative_to(ROOT / "Data/processed"):
        raise SystemExit("Audit must be inside Data/processed")
    checks = []

    def check(name, ok, detail):
        checks.append(dict(check=name, passed=bool(ok), detail=str(detail)))

    inv = pd.read_csv(out / "file_inventory.csv")
    tables = pd.read_csv(out / "table_inventory.csv")
    schemas = pd.read_csv(out / "table_schemas.csv")
    profiles = pd.read_csv(out / "source_column_completeness.csv")
    overlap = pd.read_csv(out / "extension_source_overlap_rows.csv")
    summary = json.loads((out / "run_summary.json").read_text())
    check("inventory_unique", inv.path.is_unique, len(inv))
    source_inventory = inv.loc[inv.path.map(is_raw_or_external_source)]
    check("raw_external_sources_still_unchanged", all((ROOT / r.path).stat().st_size == r.bytes and (ROOT / r.path).stat().st_mtime_ns == r.mtime_ns for r in source_inventory.itertuples()), f"{len(source_inventory)} acquired files checked by size and nanosecond mtime")
    expected_csv = set(inv.loc[inv.path.str.endswith((".csv", ".csv.gz")), "path"])
    check("all_csv_files_counted", set(tables.path) == expected_csv, len(tables))
    check("csv_parse_errors_explicit_and_none", not tables.status.str.startswith("ERROR").any(), summary["csv_parse_errors"])
    counts = schemas.groupby("path").size()
    check("schema_counts_match", all(counts[r.path] == r.columns for r in tables.itertuples()), "Header columns reconcile")
    table_counts = tables.set_index("path").rows
    check("profile_rows_match_inventory", all(table_counts[r.path] == r.rows for r in profiles.itertuples()), len(profiles))
    check("profile_nonnull_bounds", profiles.nonnull.between(0, profiles.rows).all() and (profiles.nonnull + profiles.missing).eq(profiles.rows).all(), "All profiled fields")
    for rel in ["Data/processed/contract_extension_integration/extension_modeling_master.csv", "Data/processed/capology_contracts/canonical_salary_panel.csv", "Data/processed/football_datalake_clean/tables/team_competitions_seasons_clean.csv"]:
        with (ROOT / rel).open(newline="", encoding="utf-8-sig") as f:
            independent_rows = sum(1 for _ in csv.reader(f)) - 1
        check("independent_csv_reader_count_" + Path(rel).stem, independent_rows == table_counts[rel], independent_rows)
    master = pd.read_csv(ROOT / "Data/processed/contract_extension_integration/extension_modeling_master.csv", low_memory=False)
    check("overlap_one_row_per_extension", len(overlap) == len(master) and overlap.event_id.is_unique and set(overlap.event_id) == set(master.capology_extension_event_id), len(overlap))
    check("manager_and_formation_subset_club_games", overlap.prior365_manager_recorded_games.le(overlap.prior365_club_games).all() and overlap.prior365_formation_recorded_games.le(overlap.prior365_club_games).all(), "Cannot exceed available matches")
    sm = pd.read_csv(out / "extension_source_overlap_summary.csv")
    for r in sm.itertuples():
        part = overlap if r.period == "all_extension_years_inventory" else overlap[overlap.signed_year.le(2023)]
        part = part if r.league == "ALL" else part[part.league.eq(r.league)]
        if "_at_least_" in r.metric:
            field, threshold = r.metric.rsplit("_at_least_", 1)
            count = int(part[field].ge(int(threshold)).sum())
        else:
            count = int(part[r.metric].gt(0).sum())
        if count != r.events_with_any or len(part) != r.denominator:
            raise AssertionError(f"Bad aggregate: {r}")
    check("all_overlap_aggregates_reconstructed", True, len(sm))
    # Independently reconstruct club-manager availability with a merge,
    # rather than the builder's per-club dictionary/window iteration.
    games = pd.read_csv(ROOT / "Data/processed/transfermarkt_clean/tables/games_clean.csv", low_memory=False)
    games = games[~games.is_national_team_game.astype(str).str.lower().eq("true")]
    game_parts = []
    for side in ["home", "away"]:
        g = games[["date", f"{side}_club_id", f"{side}_club_manager_name"]].copy()
        g.columns = ["game_date", "cid", "manager"]
        game_parts.append(g)
    g = pd.concat(game_parts)
    e = master[["capology_extension_event_id", "canonical_player_id", "canonical_club_id_x", "signed_date"]].dropna(subset=["canonical_player_id", "canonical_club_id_x"])
    merged = e.merge(g, left_on="canonical_club_id_x", right_on="cid", how="inner")
    delta = (pd.to_datetime(merged.signed_date) - pd.to_datetime(merged.game_date)).dt.days
    actual = merged[delta.between(1, 365) & merged.manager.notna()].groupby("capology_extension_event_id").size()
    expected = overlap.set_index("event_id").prior365_manager_recorded_games
    actual = actual.reindex(expected.index, fill_value=0)
    check("independent_manager_window_reconstruction", actual.eq(expected).all(), int(actual.gt(0).sum()))
    sb = ROOT / "Data/Open-Data-Master/open-data-master/open-data-master/data"
    payloads = pd.read_csv(out / "statsbomb_payload_inventory.csv")
    check("all_sb_payload_paths_inventoried", set(payloads.path) == {p.relative_to(ROOT).as_posix() for d in ["events", "lineups", "three-sixty"] for p in (sb / d).glob("*.json")}, len(payloads))
    catalog = {str(m["match_id"]) for p in (sb / "matches").rglob("*.json") for m in json.loads(p.read_text())}
    check("sb_catalog_reconstructed", len(catalog) == len(pd.read_csv(out / "statsbomb_catalog_audit.csv")), len(catalog))
    check("sb_orphan_classification", all((str(r.match_id) in catalog) == r.cataloged for r in payloads.itertuples()), "File presence separate from usable match metadata")
    invalid = payloads[payloads.parse_status.ne("valid_array")]
    for r in invalid.itertuples():
        try:
            obj = json.loads((ROOT / r.path).read_bytes())
            fails = not isinstance(obj, list)
        except (ValueError, UnicodeDecodeError):
            fails = True
        check("invalid_sb_file_confirmed_" + Path(r.path).stem, fails, r.path)
    sb_summary = json.loads((out / "statsbomb_summary.json").read_text())
    check("sb_counts_reconcile", all(len(payloads[payloads.kind.eq(r["kind"])]) == r["files"] and payloads.loc[payloads.kind.eq(r["kind"]), "records"].sum() == r["records"] for r in sb_summary["payloads"]), "Raw payload totals; invalid payload excluded from record count")
    usage = pd.read_csv(out / "selected_model_feature_usage.csv")
    v1 = usage[usage.version.str.startswith("V1")]
    check("sixteen_frozen_endpoints", len(v1[["endpoint", "horizon"]].drop_duplicates()) == 16, len(v1[["endpoint", "horizon"]].drop_duplicates()))
    check("no_statsbomb_or_injury_selected_features", not usage.feature.str.contains("statsbomb|injury", case=False).any(), "Selected V1 artifacts and V2 candidate lists, not all experiments")
    cap = pd.read_csv(out / "capology_file_usage.csv")
    check("all_raw_capology_csvs_classified", set(cap.path) == {p.relative_to(ROOT).as_posix() for p in (ROOT / "Data/leagues").glob("*/*/*.csv")}, len(cap))
    expected_capology_partition = {
        "club_financial_context_input": 110,
        "base_manifest_input": 69,
        "live_salary_overlay_input": 15,
        "no_base_or_live_overlay_ingestion_found": 10,
    }
    check("capology_partition", cap.status.value_counts().to_dict() == expected_capology_partition, cap.status.value_counts().to_dict())
    check("club_financial_context_sources_complete", set(cap.loc[cap.status.eq("club_financial_context_input"), "source_type"]) == {"Payrolls", "TransferWindows"}, cap.loc[cap.status.eq("club_financial_context_input"), "source_type"].value_counts().to_dict())
    layers = pd.read_csv(out / "data_layer_usage_register.csv")
    check("every_inventory_family_classified", set(layers.family) == set(inv.family), len(layers))
    freeze = json.loads((ROOT / "docs/releases/engine_v1_freeze_2026-08-30.json").read_text())
    check("frozen_V1_sha256_unchanged", all(hashlib.sha256((ROOT / f["path"]).read_bytes()).hexdigest() == f["sha256"] for f in freeze["files"]), len(freeze["files"]))
    pd.DataFrame(checks).to_csv(out / "independent_verification.csv", index=False)
    result = {"checks": len(checks), "passed": sum(c["passed"] for c in checks), "failed": [c for c in checks if not c["passed"]],
              "limits": "Checks inventory integrity and independently reconstructs selected counts/windows; does not prove semantic validity of every field or upstream completeness."}
    (out / "independent_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

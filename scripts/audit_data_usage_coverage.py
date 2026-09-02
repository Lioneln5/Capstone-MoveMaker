"""Read-only source inventory and coverage/usage audit; no training or scoring.

Writes only to its own --output directory (new by default; explicit resume).
All source files remain untouched.
Static references are evidence of a mention, NOT proof a column was fit.
CSV counts use a real parser. StatsBomb JSON counts parse every local payload.
No validation/holdout outcome rates or new model metrics are calculated.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SB = ROOT / "Data/Open-Data-Master/open-data-master/open-data-master/data"
P = ROOT / "Data/processed"
sys.path[:0] = [str(ROOT), str(ROOT / "models")]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False) + "\n")


def write(out, name, rows):
    pd.DataFrame(rows).to_csv(out / name, index=False)


def read(path, **kwargs):
    return pd.read_csv(path, low_memory=False, **kwargs)


def family(path):
    r = path.relative_to(ROOT).parts
    if r[0] != "Data":
        return "/".join(r[:3]) if r[0] == "football-datasets-main" else r[0]
    if r[1] in {"processed", "source_backups"}:
        return "/".join(r[:3]) if len(r) > 3 else "/".join(r[:2])
    return "/".join(r[:2])


def is_raw_or_external_source(relative_path: str) -> bool:
    """Limit immutability checks to acquired inputs, not generated evidence."""
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


def inventory(out):
    rows = []
    roots = [ROOT / x for x in ["Data", "football-datasets-main", "outputs", "output", "git"]]
    for base in roots:
        for path in sorted(base.rglob("*")):
            if path.is_symlink() or not path.is_file() or path.is_relative_to(out):
                continue
            st = path.stat()
            rows.append(dict(path=path.relative_to(ROOT).as_posix(), family=family(path),
                             bytes=st.st_size, mtime_ns=st.st_mtime_ns, suffix="".join(path.suffixes)))
    write(out, "file_inventory.csv", rows)
    df = pd.DataFrame(rows)
    write(out, "folder_inventory.csv", df.groupby("family").agg(files=("path", "size"), bytes=("bytes", "sum")).reset_index())
    return rows


def tables(out, inventory_rows):
    result, schemas = [], []
    for i, item in enumerate(inventory_rows):
        path = ROOT / item["path"]
        if not (path.name.endswith(".csv") or path.name.endswith(".csv.gz")):
            continue
        row = dict(path=item["path"], family=item["family"], bytes=item["bytes"])
        try:
            columns = pd.read_csv(path, nrows=0).columns.tolist()
            n = sum(len(chunk) for chunk in pd.read_csv(path, usecols=[0], chunksize=100000, dtype=str)) if columns else 0
            row.update(rows=n, columns=len(columns), status="parsed_all_rows_first_column")
            schemas.extend(dict(path=item["path"], column=c, ordinal=k) for k, c in enumerate(columns))
        except Exception as e:
            row.update(rows=None, columns=None, status=f"ERROR: {type(e).__name__}: {e}")
        result.append(row)
        if len(result) % 100 == 0:
            print(f"Counted {len(result)} CSV tables", flush=True)
    write(out, "table_inventory.csv", result)
    write(out, "table_schemas.csv", schemas)
    return result


def references(out, inventory_rows):
    # Includes current code and notebooks, excludes copied git object database,
    # old reports and our own audit so mentions cannot bootstrap evidence of use.
    files = [p for d in ["scripts", "models", "api", "Notebook", "html"]
             for p in (ROOT / d).rglob("*")
             if p.suffix in {".py", ".ipynb", ".mjs", ".js", ".html"}
             and p.name not in {Path(__file__).name, "verify_data_usage_coverage.py"}]
    families = sorted({r["family"] for r in inventory_rows if r["family"].startswith("Data/")})
    needles = {f: f.split("/")[-1] for f in families if f != "Data/processed"}
    rows, column_rows = [], []
    for file in files:
        text = file.read_text(errors="replace")
        kind = "verification" if file.name.startswith("verify") else "runtime" if file.parent.name in {"api", "html"} or file.name in {"canonical_feature_retriever.py", "deployment_scorer.py", "player_search.py", "incumbent_extension_profile.py", "business_metric_engine.py"} else "research_or_build"
        for number, line in enumerate(text.splitlines(), 1):
            for f, token in needles.items():
                if token in line:
                    rows.append(dict(family=f, file=file.relative_to(ROOT).as_posix(), line=number, kind=kind, text=line.strip()[:400]))
        if file.suffix == ".py":
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) < 150:
                    column_rows.append(dict(token=node.value, file=file.relative_to(ROOT).as_posix(), line=node.lineno, kind=kind))
    write(out, "folder_reference_evidence.csv", rows)
    write(out, "code_string_evidence.csv", column_rows)


def statsbomb(out):
    catalog = {}
    for path in sorted((SB / "matches").rglob("*.json")):
        for m in json.loads(path.read_bytes()):
            mid = str(m["match_id"])
            catalog[mid] = dict(match_id=mid, date=m["match_date"], competition=m["competition"]["competition_name"],
                                competition_id=m["competition"]["competition_id"], country=m["competition"].get("country_name"),
                                season=m["season"]["season_name"], home_id=m["home_team"]["home_team_id"], away_id=m["away_team"]["away_team_id"])
    comps = json.loads((SB / "competitions.json").read_bytes())
    gender = {(str(c["competition_id"]), c["season_name"]): c.get("competition_gender") for c in comps}
    files, players = [], []
    for directory in ["lineups", "events", "three-sixty"]:
        paths = sorted((SB / directory).glob("*.json"))
        for i, path in enumerate(paths):
            raw = path.read_bytes()
            try:
                data = json.loads(raw)
                if not isinstance(data, list):
                    raise ValueError(f"Expected array, got {type(data).__name__}")
            except (ValueError, UnicodeDecodeError) as exc:
                files.append(dict(path=path.relative_to(ROOT).as_posix(), kind=directory, match_id=path.stem,
                                  records=0, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), cataloged=path.stem in catalog,
                                  parse_status=f"INVALID: {type(exc).__name__}: {exc}"))
                continue
            row = dict(path=path.relative_to(ROOT).as_posix(), kind=directory, match_id=path.stem,
                       records=len(data), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), cataloged=path.stem in catalog, parse_status="valid_array")
            if directory == "lineups":
                row["player_entries"] = sum(len(t.get("lineup", [])) for t in data)
                for team in data:
                    for p in team.get("lineup", []):
                        players.append(dict(match_id=path.stem, team_id=team["team_id"], player_id=p["player_id"],
                                            player_name=p.get("player_name"), positions=len(p.get("positions", []))))
            files.append(row)
            if (i + 1) % 700 == 0:
                print(f"StatsBomb parsed {directory}: {i + 1}/{len(paths)}", flush=True)
    payloads = pd.DataFrame(files)
    write(out, "statsbomb_payload_inventory.csv", files)
    match_rows = []
    for mid, c in catalog.items():
        match_rows.append({**c, "gender": gender.get((str(c["competition_id"]), c["season"])),
                           **{f"has_{k}": (SB / k / (mid + ".json")).exists() for k in ["events", "lineups", "three-sixty"]}})
    matches = pd.DataFrame(match_rows)
    write(out, "statsbomb_catalog_audit.csv", matches)
    write(out, "statsbomb_competition_coverage.csv", matches.groupby(["competition", "season", "gender"], dropna=False).agg(
        matches=("match_id", "size"), first_date=("date", "min"), last_date=("date", "max"), events=("has_events", "sum"), lineups=("has_lineups", "sum"), three_sixty=("has_three-sixty", "sum")).reset_index())
    dump(out / "statsbomb_summary.json", dict(cataloged_matches=len(matches), competition_seasons=len(comps), unique_lineup_players=len({r["player_id"] for r in players}),
        payloads=payloads.groupby("kind").agg(files=("path", "size"), records=("records", "sum"), bytes=("bytes", "sum"), cataloged=("cataloged", "sum")).reset_index().to_dict("records")))
    return matches, pd.DataFrame(players)


def profile_sources(out, table_rows):
    # Full column completeness for the tables feeding or potentially feeding the
    # engine, not every repeated prediction/diagnostic output in the repository.
    paths = [r["path"] for r in table_rows if (
        r["path"].startswith(("Data/processed/football_datalake_clean/tables/", "Data/processed/transfermarkt_clean/tables/", "Data/2017-2024/", "Data/2024-2026/", "Data/injury/"))
        or r["path"] in {"Data/processed/capology_contracts/canonical_salary_panel.csv", "Data/processed/capology_contracts/canonical_extension_events.csv", "Data/processed/contract_extension_integration/extension_modeling_master.csv", "Data/processed/canonical_performance/canonical_player_team_season_performance.csv", "Data/processed/canonical_performance/fbref_advanced_performance_features.csv", "Data/processed/live_input_refresh/canonical_performance_overlay.csv", "Data/processed/live_input_refresh/canonical_salary_overlay.csv"})]
    profiles, groups = [], []
    for rel in paths:
        n, missing = 0, Counter()
        values = defaultdict(Counter)
        extrema = {}
        for chunk in pd.read_csv(ROOT / rel, chunksize=60000, low_memory=False):
            n += len(chunk)
            for col in chunk.columns:
                missing[col] += int(chunk[col].isna().sum())
                if col.lower() in {"league", "competition_id", "canonical_competition_id", "source_page_season", "season_name", "season_start_year", "data_coverage_tier", "player_link_method", "club_link_method", "transfer_type_normalized"}:
                    values[col].update(chunk[col].fillna("<missing>").astype(str))
                if col in {"date", "transfer_date", "signed_date", "from_date", "end_date", "valuation_date", "injury_from_parsed", "injury_until_parsed"}:
                    x = chunk[col].dropna().astype(str)
                    if len(x):
                        old = extrema.get(col, (x.min(), x.max()))
                        extrema[col] = (min(old[0], x.min()), max(old[1], x.max()))
        for col, count in missing.items():
            profiles.append(dict(path=rel, column=col, rows=n, nonnull=n-count, missing=count,
                                 nonnull_fraction=(n-count)/n if n else None, lexical_min=extrema.get(col, (None, None))[0], lexical_max=extrema.get(col, (None, None))[1]))
        for col, counter in values.items():
            groups.extend(dict(path=rel, column=col, value=k, rows=v) for k, v in counter.items())
        print(f"Profiled {rel}: {n:,} rows", flush=True)
    write(out, "source_column_completeness.csv", profiles)
    write(out, "source_group_coverage.csv", groups)


def model_usage(out):
    import joblib
    from models.engine_v2.feature_contract import V2_CORE_FEATURES
    import run_contract_financial_exposure_benchmark as wage
    manifest = json.loads((P / "deployment_models/model_manifest.json").read_text())
    freeze = json.loads((ROOT / "docs/releases/engine_v1_freeze_2026-08-30.json").read_text())
    checks = []
    for f in freeze["files"]:
        checks.append(dict(path=f["path"], expected_sha256=f["sha256"], actual_sha256=sha(ROOT / f["path"])))
    if not all(x["expected_sha256"] == x["actual_sha256"] for x in checks):
        raise RuntimeError("V1 freeze mismatch")
    write(out, "v1_freeze_check.csv", checks)
    rows = []
    for a in manifest["artifacts"]:
        artifact = joblib.load(ROOT / a["artifact_path"])
        if len(artifact.features) != a["feature_count"]:
            raise RuntimeError("Manifest feature count mismatch")
        rows.extend(dict(version="V1_frozen_not_enabled_by_default", endpoint=a["endpoint"], horizon=a.get("horizon"), variant=a["variant"], feature=f, evidence=a["artifact_path"]) for f in artifact.features)
    choices = {
        "sustained_meaningful_contribution": ("V2_candidate_only_not_deployed", "V2_core_involvement", V2_CORE_FEATURES),
        "meaningful_retention_24m": ("V2_development_candidate_not_deployed", "core_plus_age_curve", V2_CORE_FEATURES + ["age_distance_from_27_squared"]),
        "temporary_displacement_risk_24m": ("V2_development_candidate_not_deployed", "core_plus_age_curve", V2_CORE_FEATURES + ["age_distance_from_27_squared"]),
        "permanent_separation_risk_24m": ("V2_research_only_unavailable", "core_plus_age_curve", V2_CORE_FEATURES + ["age_distance_from_27_squared"]),
        "downside_25pct_24m": ("V2_candidate_only_not_deployed", "V2_core_involvement", V2_CORE_FEATURES),
        "annual_wage": ("V2_candidate_only_not_deployed", "B2_plus_prior_wage", wage.FEATURE_SETS["B2_plus_prior_wage"]),
    }
    for endpoint, (version, variant, features) in choices.items():
        rows.extend(dict(version=version, endpoint=endpoint, horizon="", variant=variant, feature=f, evidence="models/engine_v2/feature_contract.py; models/run_engine_v2_movement_state_profile_diagnostic.py; models/run_engine_v2_candidate_contract.py") for f in features)
    write(out, "selected_model_feature_usage.csv", rows)


def lineage_and_register(out, inv):
    """Manifest-backed lineage plus explicit, reviewed layer classifications.

    A manifest path may include an output or code dependency, so edges remain
    evidence, not an automatically inferred training dependency graph.
    """
    edges = []
    for manifest in sorted(P.rglob("source*manifest*.csv")):
        if manifest.is_relative_to(out):
            continue
        f = read(manifest).fillna("")
        cols = [c for c in f if c in {"source", "source_file", "source_path", "source_relative_path", "path", "file", "relative_path"}]
        for index, r in f.iterrows():
            for col in cols:
                value = str(r[col])
                candidate = ROOT / value
                edges.append(dict(manifest=manifest.relative_to(ROOT).as_posix(), row=int(index)+2, field=col,
                                  source=value, root_relative_path_exists=candidate.is_file()))
    write(out, "lineage_manifest_evidence.csv", edges)
    special = {
        "canonical_integration": ("core_integration_and_V1_runtime", "Player/club dimensions and valuation history load in V1; transfer history builds extension outcomes. Snapshot evidence and reviews are not all predictors."),
        "canonical_performance": ("core_integration_partial_model_use", "Basic/advanced seasonal features join extension master and V1 retriever; V2 selected candidates no longer use prior seasonal volume or advanced stats."),
        "capology_contracts": ("core_integration_and_V1_runtime", "19770 base salary rows and 2805 extension events; source season inventory differs from newer raw additions."),
        "contract_extension_integration": ("current_V2_training_spine_and_V1_history", "Master is source for current V2 development. Many columns are targets, identifiers, or unused context, not predictors."),
        "transfermarkt_clean": ("core_integration_partial_runtime", "Appearances/games directly feed exact windows and V1; lineups feed seasonal starts. Manager/formation columns are retained but not current model inputs."),
        "football_datalake_clean": ("mixed_integrated_and_dormant", "Valuations/transfers/profiles/performance integrated; injury spells, career nationals, lifetime teammate edges, season standings/manager fields and hierarchy mostly outside extension modeling."),
        "fbref_2017_2024_clean": ("integrated_and_legacy_research", "Cleaned seasonal data feeds crosswalks/performance and legacy compatibility. V2 selected feature lists exclude advanced stats."),
        "integration_crosswalks": ("identity_infrastructure", "Accepted source links support canonical integration. Reading a table for entity aliases does not use its substantive metrics."),
        "live_input_refresh": ("V1_runtime_overlay_only", "Newer performance and salary overlays, not appended to frozen historical research. V2 has no deployed inference path."),
        "deployment_models": ("frozen_V1_artifacts", "16 fitted artifacts retained; API scoring disabled by default. Backups retained. No V2 artifacts promoted."),
        "frozen_contract_sources": ("frozen_research_copy", "Intentional reproducibility snapshot, not independent additional observations."),
        "statsbomb_case_studies": ("case_study_processed_not_engine", "642 selected matches; workbook and historical case availability evidence. Crosswalks review-only; no selected V1/V2 model feature input."),
        "compatibility_features": ("legacy_engineered_research_not_current_engine", "Broad lagged player features, role-season standardized styles, prior-club teammate context and preference interactions; not ported into extension V2."),
        "compatibility_targets": ("legacy_transfer_targets", "Transfer-scope target/cohort matrices; not current extension inputs."),
        "transfermarkt_merged": ("legacy_transfer_spine", "Broad 35139-transfer master and transfer_core used by old experiments/notebook; not current extension master."),
        "contribution_model_refinement": ("superseded_research_evidence", "Thesis injury and refinement research; later repair corrects methodological issues. No blanket inference that all results are valid."),
        "contribution_model_repair": ("historical_repair_and_injury_diagnostics", "C6 repair selected for frozen V1; corrected thesis-injury diagnostics remain separate, not selected injury model inputs."),
        "role_contextualization_diagnostic": ("nondeployed_candidate_research", "Role-aware/removal comparisons and candidate joblib files; not deployed."),
        "business_metric_product": ("V1_business_reference_data", "Offer comparison reference consumed by historical profile builder; not an independent predictive source."),
        "integrated_contract_profile": ("historical_product_research", "Profile experiments/reference outputs; distinguish from runtime business reference files."),
        "engine_v2_validity_boundary": ("V2_current_research_governance", "Phase 1 boundary checks; not a new raw data source."),
        "engine_v2_feature_specification": ("V2_current_research_evidence", "Phase 2 feature candidates and policy audit, not production artifacts."),
        "engine_v2_calibration_reliability": ("V2_current_research_evidence", "Phase 3 calibration/support evidence."),
        "engine_v2_candidate_contract": ("V2_current_research_evidence", "Phase 4 continuity/wage candidates and result contract evidence."),
        "engine_v2_final_evaluation_gate": ("V2_release_gate_evidence", "Phase 5 blocks promotion; prior outcomes not re-evaluated by this audit."),
        "capology_club_financial_context": ("V2_club_financial_context_features", "Canonical lagged payroll, transfer-window and squad context built from the newly added Capology club-season files."),
        "capology_club_financial_context_audit": ("V2_club_financial_context_verification", "Independent parsing, identity, timing and coverage evidence for the club-season context build."),
        "engine_v2_club_financial_context_experiment": ("V2_rejected_mechanism_evidence", "Club financial context did not repair continuity and remains non-deployed."),
        "engine_v2_movement_state_profile_diagnostic": ("V2_movement_decomposition_evidence", "Separates meaningful retention, temporary displacement and permanent separation without opening the final holdout."),
        "engine_v2_lagged_club_behavior": ("V2_research_feature_infrastructure", "Decision-time lagged club movement and extension behavior; row-level profiles remain local."),
        "engine_v2_lagged_club_behavior_experiment": ("V2_rejected_mechanism_evidence", "Lagged club behavior did not clear the evidence gate."),
        "engine_v2_role_squad_profiles": ("V2_research_feature_infrastructure", "Decision-time player role trajectory and player-relative squad competition; row-level profiles remain local."),
        "engine_v2_role_squad_experiment": ("V2_rejected_mechanism_evidence", "Role and squad blocks did not clear the evidence gate."),
        "engine_v2_manager_tactical_profiles": ("V2_research_feature_infrastructure", "Decision-time manager, formation and player-manager context; row-level profiles remain local."),
        "engine_v2_manager_tactical_experiment": ("V2_rejected_mechanism_evidence", "Manager/tactical blocks did not clear the evidence gate."),
        "engine_v2_movement_scope_closeout": ("V2_product_scope_contract", "Retires generic continuity and freezes replacement statuses without deploying numeric outputs."),
    }
    rows = []
    for f in sorted({r["family"] for r in inv}):
        name = f.split("/")[-1]
        if f.startswith("Data/processed/"):
            status, note = special.get(name, ("historical_research_or_verification_outputs", "Generated experiment outputs, not independent raw observations. See source-manifest and static-code evidence for exact consumers."))
        elif f.startswith("Data/source_backups"):
            status, note = "preservation_backup", "Intentional copies; not unused unique observations and not a deletion recommendation."
        elif f == "Data/Open-Data-Master":
            status, note = "raw_StatsBomb_partially_extracted", "Full local payload audit; most outside selected 2017-2024 case-study scope. Local completeness does not imply current upstream completeness."
        elif f == "Data/2024-2026":
            status, note = "raw_live_overlay_input_plus_unused_light_exports", "Full files read by build_live_input_refresh; light files not read by that builder."
        elif f in {"Data/2017-2024", "Data/TransferMarkt", "football-datasets-main/datalake/transfermarkt"}:
            status, note = "raw_integrated_selectively", "See domain-level report and table schema/column coverage; not all fields reach models."
        elif f == "Data/leagues":
            status, note = "raw_contract_plus_club_financial_inputs", "50 salary, 44 extension, 55 payroll and 55 transfer-window files. The 110 club-season files feed the separately verified financial-context builder; later extension pages remain outside the base/overlay path."
        elif f == "Data/injury":
            status, note = "thesis_injury_research_only", "Separate 15603-row source used in refinement/repair; not the supplemental 143195-row injury source."
        elif f == "Data/processed":
            status, note = "legacy_root_level_matrices", "Old transfer cohort/player features/crosswalk plus notebook model_dataset and model_dataset_with_injuries; no selected current model loading those notebook matrices."
        elif f in {"outputs", "output", "Data/app-html", "Data/slides"}:
            status, note = "reports_presentations_or_exports", "Not independent training data; preserve source/provenance distinction."
        elif f == "git":
            status, note = "local_git_metadata_copy", "Objects/refs, not a football dataset; not read as .git by the current checkout. No removal performed."
        else:
            status, note = "support_or_filesystem_metadata", "README, workflow, source docs, or filesystem metadata; see inventory."
        subset = [r for r in inv if r["family"] == f]
        rows.append(dict(family=f, status=status, files=len(subset), bytes=sum(r["bytes"] for r in subset), note=note))
    write(out, "data_layer_usage_register.csv", rows)
    cap_manifest = read(P / "capology_contracts/source_manifest.csv")
    base_paths = set(cap_manifest.source_relative_path)
    raw_rows = []
    for path in sorted((ROOT / "Data/leagues").glob("*/*/*.csv")):
        import re
        season = re.search(r"(20\d{2})_(20\d{2})", path.name)
        start = int(season.group(1)) if season else None
        rel = path.relative_to(ROOT).as_posix()
        source_type = path.parent.name
        if rel in base_paths:
            status = "base_manifest_input"
        elif source_type == "sallaries" and start is not None and start >= 2024:
            status = "live_salary_overlay_input"
        elif source_type in {"Payrolls", "TransferWindows"}:
            status = "club_financial_context_input"
        else:
            status = "no_base_or_live_overlay_ingestion_found"
        raw_rows.append(dict(path=rel, season_start=start, source_type=path.parent.name, status=status))
    write(out, "capology_file_usage.csv", raw_rows)
    dump(out / "capology_rebuild_preflight.json", dict(
        base_manifest_files=len(base_paths),
        observed_salary_files=sum(r["source_type"] == "sallaries" for r in raw_rows),
        observed_extension_files=sum(r["source_type"] == "extensions" for r in raw_rows),
        observed_payroll_files=sum(r["source_type"] == "Payrolls" for r in raw_rows),
        observed_transfer_window_files=sum(r["source_type"] == "TransferWindows" for r in raw_rows),
        result="Legacy salary/extension rebuilding remains version-gated. Payroll and transfer-window files have a separate verified club-financial-context ingestion path.",
    ))


def ids(series):
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def extension_overlap(out, sb_matches, sb_players):
    ext = read(P / "contract_extension_integration/extension_modeling_master.csv")
    ext["pid"] = ids(ext.canonical_player_id)
    ext["cid"] = ids(ext.canonical_club_id_x)
    ext["decision"] = pd.to_datetime(ext.signed_date)
    # Follows existing completed-season convention; not proof of publication.
    ext["completed_season"] = ext.decision.dt.year - np.where(ext.decision.dt.month.ge(7) | ((ext.decision.dt.month.eq(6)) & (ext.decision.dt.day.eq(30))), 1, 2)
    inj = read(P / "football_datalake_clean/tables/player_injuries_clean.csv", usecols=["player_id", "from_date", "end_date"])
    inj["pid"] = ids(inj.player_id)
    inj["date"] = pd.to_datetime(inj.from_date, errors="coerce")
    inj_groups = {k: g.date.dropna().sort_values().to_numpy() for k, g in inj.groupby("pid")}
    games = read(P / "transfermarkt_clean/tables/games_clean.csv")
    games = games.loc[~games.is_national_team_game.astype(str).str.lower().eq("true")].copy()
    games["date"] = pd.to_datetime(games.date, errors="coerce")
    home = games[["game_id", "date", "home_club_id", "home_club_manager_name", "home_club_formation"]].copy()
    away = games[["game_id", "date", "away_club_id", "away_club_manager_name", "away_club_formation"]].copy()
    home.columns = away.columns = ["game_id", "date", "cid", "manager", "formation"]
    cg = pd.concat([home, away]); cg["cid"] = ids(cg.cid)
    clubs = {k: g for k, g in cg.groupby("cid")}
    lineups = read(P / "transfermarkt_clean/tables/game_lineups_clean.csv", usecols=["date", "player_id", "club_id", "type", "position"])
    lineups["pid"] = ids(lineups.player_id); lineups["cid"] = ids(lineups.club_id)
    # Restrict before constructing groups; retain all actual lineup rows for count.
    lineups = lineups.loc[lineups.pid.isin(ext.pid.dropna())].copy()
    lineups["date"] = pd.to_datetime(lineups.date, errors="coerce")
    lup = {k: g.date.dropna().sort_values().to_numpy() for k, g in lineups.groupby(["pid", "cid"])}
    salary = read(P / "capology_contracts/canonical_salary_panel.csv", usecols=["canonical_club_id", "source_page_season_start_year", "annual_gross_eur"])
    salary["cid"] = ids(salary.canonical_club_id)
    sal = salary.groupby(["cid", "source_page_season_start_year"]).annual_gross_eur.count().to_dict()
    standings = read(P / "football_datalake_clean/tables/team_competitions_seasons_clean.csv")
    standings["cid"] = ids(standings.club_id)
    st = standings.groupby(["cid", "season_start_year"]).size().to_dict()
    teammate = read(P / "football_datalake_clean/tables/player_teammates_played_with_clean.csv", usecols=["player_id"])
    tmids = set(ids(teammate.player_id).dropna())
    national = read(P / "football_datalake_clean/tables/player_national_performances_clean.csv", usecols=["player_id"])
    ntids = set(ids(national.player_id).dropna())
    # Candidate identity links remain UNREVIEWED. Existing crosswalks cover
    # the selected case-study sample, not all raw players: this is exploratory
    # overlap under those candidate links, not a maximum attainable overlap.
    pc = read(P / "statsbomb_case_studies/statsbomb_player_crosswalk_candidates.csv")
    cc = read(P / "statsbomb_case_studies/statsbomb_club_crosswalk_candidates.csv")
    pc = pc.loc[pc.transfermarkt_candidate_count.eq(1)].dropna(subset=["transfermarkt_player_id"])
    cc = cc.loc[cc.transfermarkt_candidate_count.eq(1)].dropna(subset=["transfermarkt_club_id"])
    players = sb_players.merge(pc[["statsbomb_player_id", "transfermarkt_player_id"]], left_on="player_id", right_on="statsbomb_player_id", how="inner")
    players = players.merge(cc[["team_id", "transfermarkt_club_id"]], on="team_id", how="inner")
    players = players.merge(sb_matches[["match_id", "date", "gender", "competition"]], on="match_id", how="inner")
    players = players.loc[players.gender.eq("male") & players.positions.gt(0)].copy()
    players["pid"] = ids(players.transfermarkt_player_id); players["cid"] = ids(players.transfermarkt_club_id)
    players["date"] = pd.to_datetime(players.date)
    sbgroups = {k: g.date.drop_duplicates().sort_values().to_numpy() for k, g in players.groupby(["pid", "cid"])}
    counts = []
    def in_window(lookup, key, start, stop):
        dates = lookup.get(key)
        return 0 if dates is None else int(np.searchsorted(dates, stop.to_datetime64()) - np.searchsorted(dates, start.to_datetime64()))
    for r in ext.itertuples():
        valid = pd.notna(r.pid) and pd.notna(r.cid)
        start = r.decision - pd.Timedelta(days=365)
        club = clubs.get(r.cid) if valid else None
        recent = club.loc[club.date.ge(start) & club.date.lt(r.decision)] if club is not None else pd.DataFrame()
        count = dict(event_id=r.capology_extension_event_id, league=r.league, signed_year=r.decision.year, linked=valid,
                     prior365_injury_start_records=in_window(inj_groups, r.pid, start, r.decision) if valid else 0,
                     any_prior_injury_start_record=in_window(inj_groups, r.pid, pd.Timestamp("1900-01-01"), r.decision) > 0 if valid else False,
                     prior365_lineup_rows=in_window(lup, (r.pid, r.cid), start, r.decision) if valid else 0,
                     prior365_club_games=len(recent), prior365_manager_recorded_games=int(recent.manager.notna().sum()) if len(recent) else 0,
                     prior365_formation_recorded_games=int(recent.formation.notna().sum()) if len(recent) else 0,
                     previous_completed_season_known_wage_players=int(sal.get((r.cid, r.completed_season), 0)) if valid else 0,
                     previous_completed_club_season_standings_rows=int(st.get((r.cid, r.completed_season), 0)) if valid else 0,
                     lifetime_teammate_record_present=bool(r.pid in tmids) if valid else False,
                     career_national_record_present=bool(r.pid in ntids) if valid else False,
                     unreviewed_sb_prior365_same_club_matches=in_window(sbgroups, (r.pid, r.cid), start, r.decision) if valid else 0)
        counts.append(count)
    write(out, "extension_source_overlap_rows.csv", counts)
    counts = pd.DataFrame(counts)
    summaries = []
    for period, data in [("all_extension_years_inventory", counts), ("development_through_2023", counts[counts.signed_year.le(2023)])]:
        for league, group in [("ALL", data), *list(data.groupby("league"))]:
            for c in counts.columns[3:]:
                v = group[c]
                summaries.append(dict(period=period, league=league, metric=c, denominator=len(group),
                                      events_with_any=int(v.gt(0).sum()), fraction_with_any=float(v.gt(0).mean()), total_records=int(v.sum())))
            for c, threshold in [("prior365_manager_recorded_games", 10), ("prior365_lineup_rows", 10), ("previous_completed_season_known_wage_players", 15), ("unreviewed_sb_prior365_same_club_matches", 3)]:
                summaries.append(dict(period=period, league=league, metric=f"{c}_at_least_{threshold}", denominator=len(group), events_with_any=int(group[c].ge(threshold).sum()), fraction_with_any=float(group[c].ge(threshold).mean()), total_records=None))
    write(out, "extension_source_overlap_summary.csv", summaries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume-after-csv", action="store_true", help="Reuse completed inventory/CSV audit after a JSON scan interruption, verifying input metadata first.")
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if out.exists() and not args.resume_after_csv:
        raise SystemExit("Output must be a NEW directory; no overwrites or deletions.")
    if not out.is_relative_to(P):
        raise SystemExit("Output must be inside Data/processed.")
    if args.resume_after_csv:
        inv = read(out / "file_inventory.csv").to_dict("records")
        tab = read(out / "table_inventory.csv").to_dict("records")
        if not (out / "source_column_completeness.csv").exists():
            raise SystemExit("CSV profiling has not completed; cannot resume here.")
        if any(is_raw_or_external_source(r["path"]) and ((ROOT / r["path"]).stat().st_mtime_ns != r["mtime_ns"] or (ROOT / r["path"]).stat().st_size != r["bytes"]) for r in inv):
            raise SystemExit("Raw/external input metadata changed since inventory; do not reuse old counts.")
    else:
        out.mkdir(parents=True)
        print("Inventorying local holdings", flush=True)
        inv = inventory(out)
        references(out, inv)
        model_usage(out)
        tab = tables(out, inv)
        profile_sources(out, tab)
    matches, players = statsbomb(out)
    extension_overlap(out, matches, players)
    lineage_and_register(out, inv)
    changed = [r["path"] for r in inv if is_raw_or_external_source(r["path"]) and ((ROOT / r["path"]).stat().st_size != r["bytes"] or (ROOT / r["path"]).stat().st_mtime_ns != r["mtime_ns"])]
    freeze = read(out / "v1_freeze_check.csv")
    hash_changed = [r.path for r in freeze.itertuples() if sha(ROOT / r.path) != r.actual_sha256]
    errors = [r for r in tab if r["status"].startswith("ERROR")]
    sb_inventory = read(out / "statsbomb_payload_inventory.csv")
    summary = dict(audit_utc=datetime.now(timezone.utc).isoformat(), file_count=len(inv), logical_bytes=sum(r["bytes"] for r in inv), csv_tables=len(tab), csv_parse_errors=errors,
                   invalid_statsbomb_payloads=sb_inventory.loc[sb_inventory.parse_status.ne("valid_array"), ["path", "parse_status"]].to_dict("records"),
                   changed_source_files_size_mtime=changed, changed_v1_files_sha256=hash_changed,
                   method="Full filesystem inventory; full CSV row parse; full StatsBomb payload parse/hash; selected source column profiles; static code evidence; artifact feature inspection; no training/scoring.",
                   limits=["Inventory is local, not proof of completeness against a current upstream source.", "Static references do not prove runtime execution or predictive contribution.", "Raw/external source preservation checked by size/mtime and frozen V1 by SHA256; generated evidence may be refreshed after inventory.", "Extension overlaps are availability, not outcome/performance metrics or confirmed temporal validity.", "StatsBomb identity overlap is unreviewed candidate matching, not approved linkage.", "Injury-record presence is positive evidence only; absence cannot establish health or reporting coverage.", "Standings overlap is any competition in previous completed season, not confirmed same-league or publication-safe data."])
    dump(out / "run_summary.json", summary)
    if changed or hash_changed or errors:
        print("Audit completed with issues; inspect run_summary.json", flush=True)
    else:
        print("Audit completed; source size/mtime unchanged and frozen V1 hashes unchanged", flush=True)


if __name__ == "__main__":
    main()

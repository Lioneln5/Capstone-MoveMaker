"""Build audited canonical Capology salary and contract-extension tables.

Raw CSVs under ``Data/leagues`` are immutable.  This script parses their HTML
display cells, standardizes money and dates, makes exact source backups, applies
only conservative identity links, deduplicates repeated observations, and
writes machine-readable provenance and validation artifacts.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
LEAGUES = DATA / "leagues"
OUTPUT = DATA / "processed" / "capology_contracts"
BACKUP = DATA / "source_backups" / "capology_contracts_original"
CROSSWALKS = DATA / "processed" / "integration_crosswalks"

LEAGUE_META = {
    "bundesliga": ("Bundesliga", "L1"),
    "la liga": ("La Liga", "ES1"),
    "ligue1": ("Ligue 1", "FR1"),
    "premier league": ("Premier League", "GB1"),
    "serie a": ("Serie A", "IT1"),
}

# Reviewed aliases absent from the accepted FBref club crosswalk.  These are
# exact normalized aliases, not fuzzy matches.
CLUB_ALIASES = {
    "acmilan": 5,
    "arminiabielefeld": 10,
    # Reviewed Capology club-level payroll/transfer-window aliases.  Each ID
    # was confirmed against the source slug, source club code, Transfermarkt
    # club URL/ID, and historical top-flight competition membership.  Do not
    # infer these from club_identity_crosswalk.name_aliases: that legacy field
    # contains unrelated movement-history names and is not identity evidence.
    "bastia": 595,  # SC Bastia; not AC Bastia 1924 (28904)
    "carpi": 4102,  # AC Carpi / Carpi FC 1909
    "cesena": 1429,  # Cesena FC / AC Cesena
    "cordoba": 993,  # Cordoba CF; not unrelated canonical 'Cordoba' (82232)
    "evian": 14171,  # Thonon Evian Grand Geneve FC
    "gazelecajaccio": 3558,  # GFC Ajaccio
    "holsteinkiel": 269,
    "hullcity": 3008,
    "ingolstadt": 4795,  # FC Ingolstadt 04
    "middlesbrough": 641,
    "nancy": 1159,  # AS Nancy-Lorraine
    "palermo": 458,
    "pescara": 2921,  # Delfino Pescara 1936
    "qpr": 1039,  # Queens Park Rangers
    "sportinggijon": 2448,
    "sunderland": 289,
    "borussiadortmund": 16,
    "chievoverona": 862,
    "eintrachtfrankfurt": 24,
    "intermilan": 46,
    "manchesterunited": 985,
    "nottinghamforest": 703,
    "sheffieldunited": 350,
    "bayerleverkusen": 15,
    "darmstadt": 105,
    "furth": 65,
    "hamburg": 41,
    "hannover": 42,
    "herthaberlin": 44,
    "leipzig": 23826,
    "mainz": 39,
    "monchengladbach": 18,
    "paderborn": 127,
    "deportivo": 897,
    "realbetis": 150,
    "clermont": 3524,
    "psg": 583,
    "stetienne": 618,
    "cardiff": 603,
    "leeds": 399,
    "leicester": 1003,
    "newcastle": 762,
    "norwich": 1123,
    "swansea": 2288,
    "westbromwich": 984,
    "wolverhampton": 543,
    # Reviewed 2024-25 promoted-club aliases.  The canonical identity layer
    # uses the fuller Transfermarkt names shown in the comments.
    "como": 1047,  # Como 1907
    "ipswichtown": 677,
    "stpauli": 35,  # FC St. Pauli
}

EXPECTED_SALARY_FILES = 35
EXPECTED_EXTENSION_FILES = 34
EXPECTED_SALARY_ROWS = 19_770
EXPECTED_EXTENSION_ROWS = 2_811
EXPECTED_EXTENSION_EVENTS = 2_805

EUR_FIELDS = [
    "weekly_gross_eur",
    "annual_gross_eur",
    "weekly_net_eur",
    "annual_net_eur",
    "bonus_gross_eur",
    "bonus_net_eur",
    "total_gross_eur",
    "total_net_eur",
    "adjusted_total_gross_eur",
    "adjusted_total_net_eur",
]
EXTENSION_EUR_FIELDS = EUR_FIELDS + ["contract_total_gross_eur", "contract_total_net_eur"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.casefold().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", "", text)


def extract_display(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def extract_href(value: object) -> str | None:
    if pd.isna(value):
        return None
    match = re.search(r"href=['\"]([^'\"]+)['\"]", str(value))
    return match.group(1) if match else None


def extract_player_suffix(value: object) -> str | None:
    href = extract_href(value)
    if not href:
        return None
    match = re.search(r"/player/[^/]+-([0-9]+)/?", href)
    return match.group(1) if match else None


def extract_club_slug(value: object) -> str | None:
    href = extract_href(value)
    if not href:
        return None
    match = re.search(r"/club/([^/]+)/", href)
    return match.group(1) if match else None


def parse_eur(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text in {"", "-", "—", "nan", "None"}:
        return np.nan
    text = re.sub(r"[^0-9,.-]", "", text).replace(",", "")
    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_bool(value: object) -> bool | None:
    if pd.isna(value):
        return None
    return str(value).strip().casefold() in {"true", "1", "yes"}


def season_from_path(path: Path) -> tuple[str, int, int]:
    match = re.search(r"(20\d{2})_(20\d{2})", path.name)
    if not match:
        raise ValueError(f"No season in filename: {path}")
    start, end = map(int, match.groups())
    return f"{start}-{end}", start, end


def stable_hash(*values: object) -> str:
    return hashlib.sha256("|".join("" if pd.isna(v) else str(v) for v in values).encode()).hexdigest()


def join_unique(values: pd.Series) -> str:
    return " | ".join(sorted({str(value) for value in values if pd.notna(value) and str(value) != ""}))


def copy_and_manifest(files: list[Path]) -> pd.DataFrame:
    rows: list[dict] = []
    for source in files:
        relative = source.relative_to(LEAGUES)
        target = BACKUP / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source_hash = sha256(source)
        if target.exists():
            if sha256(target) != source_hash:
                raise RuntimeError(f"Existing backup differs from source: {target}")
        else:
            shutil.copy2(source, target)
        rows.append(
            {
                "source_file": source.name,
                "source_relative_path": source.relative_to(ROOT).as_posix(),
                "source_type": "salary" if path_is_salary(source) else "extension",
                "source_bytes": source.stat().st_size,
                "source_sha256": source_hash,
                "backup_relative_path": target.relative_to(ROOT).as_posix(),
                "backup_bytes": target.stat().st_size,
                "backup_sha256": sha256(target),
                "hashes_match": source_hash == sha256(target),
                "source_rows": len(pd.read_csv(source)),
                "source_columns": len(pd.read_csv(source, nrows=0).columns),
            }
        )
    return pd.DataFrame(rows)


def path_is_salary(path: Path) -> bool:
    return path.parent.name == "sallaries"


def build_player_map() -> tuple[dict[str, dict], dict[str, dict]]:
    fbref = pd.read_csv(CROSSWALKS / "fbref_player_crosswalk.csv")
    accepted = fbref.loc[fbref["match_status"].eq("accepted")].copy()
    accepted["key"] = accepted["fbref_player_name"].map(normalize_text)
    counts = accepted.groupby("key")["canonical_player_id"].nunique()
    accepted = accepted.loc[accepted["key"].map(counts).eq(1)]
    fbref_map: dict[str, dict] = {}
    for key, group in accepted.groupby("key"):
        row = group.iloc[0]
        fbref_map[key] = {
            "canonical_player_id": int(row["canonical_player_id"]),
            "canonical_player_name": row["canonical_player_name"],
            "player_link_method": "fbref_accepted_exact_unique_name",
        }
    # FBref's enhanced normalization repairs a small set of source-specific
    # diacritic encodings (for example Polish ł). It remains an exact,
    # one-canonical-ID match rather than a fuzzy comparison.
    accepted["enhanced_key"] = accepted["fbref_name_enhanced_normalized"].fillna("").astype(str)
    enhanced_counts = accepted.groupby("enhanced_key")["canonical_player_id"].nunique()
    for key, group in accepted.loc[accepted["enhanced_key"].map(enhanced_counts).eq(1)].groupby("enhanced_key"):
        if not key or key in fbref_map:
            continue
        row = group.iloc[0]
        fbref_map[key] = {
            "canonical_player_id": int(row["canonical_player_id"]),
            "canonical_player_name": row["canonical_player_name"],
            "player_link_method": "fbref_accepted_enhanced_exact_unique_name",
        }

    identity_all = pd.read_csv(CROSSWALKS / "player_identity_crosswalk.csv")
    identity_all["key"] = identity_all["canonical_name"].map(normalize_text)
    counts = identity_all.groupby("key")["canonical_player_id"].nunique()
    identity = identity_all.loc[identity_all["key"].map(counts).eq(1)].copy()
    identity_map: dict[str, dict] = {}
    for key, group in identity.groupby("key"):
        row = group.iloc[0]
        identity_map[key] = {
            "canonical_player_id": int(row["canonical_player_id"]),
            "canonical_player_name": row["canonical_name"],
            "date_of_birth": row.get("date_of_birth"),
            "canonical_position": row.get("position"),
            "canonical_sub_position": row.get("sub_position"),
            "player_link_method": "canonical_identity_exact_unique_name",
        }
    identity_by_id = {
        int(row.canonical_player_id): {
            "date_of_birth": row.date_of_birth,
            "canonical_position": row.position,
            "canonical_sub_position": row.sub_position,
        }
        for row in identity_all.itertuples(index=False)
        if pd.notna(row.canonical_player_id)
    }
    for linked in fbref_map.values():
        linked.update(identity_by_id.get(linked["canonical_player_id"], {}))
    return fbref_map, identity_map


def build_club_map() -> tuple[dict[str, dict], dict[int, str]]:
    clubs = pd.read_csv(CROSSWALKS / "fbref_club_crosswalk.csv")
    accepted = clubs.loc[clubs["match_status"].eq("accepted")].copy()
    accepted["key"] = accepted["fbref_squad_name"].map(normalize_text)
    counts = accepted.groupby("key")["canonical_club_id"].nunique()
    accepted = accepted.loc[accepted["key"].map(counts).eq(1)]
    mapping = {}
    id_names = {}
    for key, group in accepted.groupby("key"):
        row = group.iloc[0]
        club_id = int(row["canonical_club_id"])
        mapping[key] = {
            "canonical_club_id": club_id,
            "canonical_club_name": row["canonical_club_name"],
            "club_link_method": "fbref_accepted_exact_alias",
        }
        id_names[club_id] = str(row["canonical_club_name"])
    identity_path = CROSSWALKS / "club_identity_crosswalk.csv"
    if identity_path.exists():
        identities = pd.read_csv(identity_path)
        for row in identities.itertuples(index=False):
            if pd.notna(getattr(row, "canonical_club_id", np.nan)):
                name = getattr(row, "canonical_name", None)
                if pd.notna(name):
                    id_names[int(row.canonical_club_id)] = str(name)
    for alias, club_id in CLUB_ALIASES.items():
        mapping[alias] = {
            "canonical_club_id": club_id,
            "canonical_club_name": id_names.get(club_id),
            "club_link_method": "reviewed_exact_capology_alias",
        }
    return mapping, id_names


def link_player(name_normalized: str, fbref_map: dict, identity_map: dict) -> dict:
    if name_normalized in fbref_map:
        return dict(fbref_map[name_normalized])
    if name_normalized in identity_map:
        return dict(identity_map[name_normalized])
    return {
        "canonical_player_id": np.nan,
        "canonical_player_name": None,
        "date_of_birth": None,
        "canonical_position": None,
        "canonical_sub_position": None,
        "player_link_method": "unresolved",
    }


def link_club(club_normalized: str, club_map: dict) -> dict:
    return dict(
        club_map.get(
            club_normalized,
            {"canonical_club_id": np.nan, "canonical_club_name": None, "club_link_method": "unresolved"},
        )
    )


def read_source(path: Path, source_type: str, fbref_map: dict, identity_map: dict, club_map: dict) -> pd.DataFrame:
    raw = pd.read_csv(path)
    season, start, end = season_from_path(path)
    league_key = path.parents[1].name
    league, competition_id = LEAGUE_META[league_key]
    rows: list[dict] = []
    money_fields = EUR_FIELDS if source_type == "salary" else EXTENSION_EUR_FIELDS
    suffix_names: dict[str, set[str]] = {}
    for value in raw["name"]:
        suffix = extract_player_suffix(value)
        suffix_names.setdefault(str(suffix), set()).add(normalize_text(extract_display(value)))
    for index, row in raw.iterrows():
        player_name = extract_display(row["name"])
        player_norm = normalize_text(player_name)
        club_name = extract_display(row["club"])
        club_norm = normalize_text(club_name)
        suffix = extract_player_suffix(row["name"])
        record = {
            "source_file": path.name,
            "source_relative_path": path.relative_to(ROOT).as_posix(),
            "source_row_number": index + 2,
            "source_page_season": season,
            "source_page_season_start_year": start,
            "source_page_season_end_year": end,
            "league": league,
            "competition_id": competition_id,
            "player_name": player_name,
            "player_name_normalized": player_norm,
            "capology_player_url": extract_href(row["name"]),
            "capology_player_url_suffix": suffix,
            "capology_suffix_name_conflict_in_file": len(suffix_names.get(str(suffix), set())) > 1,
            "club_name": club_name,
            "club_name_normalized": club_norm,
            "capology_club_slug": extract_club_slug(row["club"]),
            "capology_club_url": extract_href(row["club"]),
        }
        record.update(link_player(player_norm, fbref_map, identity_map))
        record.update(link_club(club_norm, club_map))
        zero_fields = []
        for field in money_fields:
            value = parse_eur(row.get(field))
            if pd.notna(value) and value == 0:
                zero_fields.append(field)
                value = np.nan
            record[field] = value
        record["reported_zero_salary"] = bool(zero_fields)
        record["zero_salary_fields"] = " | ".join(zero_fields)
        record["bonus_available"] = pd.notna(record.get("bonus_gross_eur"))
        if source_type == "salary":
            verified = str(row.get("verified", ""))
            match = re.search(r"verified-([a-z]+)", verified)
            record.update(
                {
                    "verification_badge": match.group(1) if match else None,
                    "status": extract_display(row.get("status")),
                    "position": row.get("position"),
                    "age": pd.to_numeric(row.get("age"), errors="coerce"),
                    "country": row.get("country"),
                    "active": parse_bool(row.get("active")),
                    "loan": parse_bool(row.get("loan")),
                }
            )
        else:
            signed = pd.to_datetime(row.get("signed"), errors="coerce")
            expiration = pd.to_datetime(row.get("expiration"), errors="coerce")
            exact_days = (expiration - signed).days if pd.notna(signed) and pd.notna(expiration) else np.nan
            actual_season = signed.year if pd.notna(signed) and signed.month >= 7 else (signed.year - 1 if pd.notna(signed) else np.nan)
            if pd.notna(signed):
                # A season beginning in year Y is conservatively considered
                # complete on June 30 of Y+1.  January-May signings therefore
                # cannot use that ongoing season as a predictor. June 30 is
                # accepted as the standard season-completion boundary.
                last_completed_season = (
                    signed.year - 1
                    if signed.month >= 7 or (signed.month == 6 and signed.day == 30)
                    else signed.year - 2
                )
                first_full_post_season = (
                    signed.year
                    if signed.month <= 6 or (signed.month == 7 and signed.day == 1)
                    else signed.year + 1
                )
            else:
                last_completed_season = np.nan
                first_full_post_season = np.nan
            record.update(
                {
                    "signed_date": signed,
                    "expiration_date": expiration,
                    "reported_years": pd.to_numeric(row.get("years"), errors="coerce"),
                    "exact_duration_days": exact_days,
                    "exact_duration_years": exact_days / 365.25 if pd.notna(exact_days) else np.nan,
                    "event_season_start_year": actual_season,
                    "event_season": f"{int(actual_season)}-{int(actual_season)+1}" if pd.notna(actual_season) else None,
                    "pre_completed_season_start_year": last_completed_season,
                    "first_full_post_season_start_year": first_full_post_season,
                }
            )
            date_of_birth = pd.to_datetime(record.get("date_of_birth"), errors="coerce")
            record["age_at_signing"] = (
                (signed - date_of_birth).days / 365.25 if pd.notna(signed) and pd.notna(date_of_birth) else np.nan
            )
            record["reported_minus_exact_years"] = record["reported_years"] - record["exact_duration_years"]
            record["contract_total_recomputed_eur"] = (
                record["total_gross_eur"] * record["reported_years"]
                if pd.notna(record["total_gross_eur"]) and pd.notna(record["reported_years"])
                else np.nan
            )
            record["contract_total_arithmetic_difference_eur"] = (
                record["contract_total_gross_eur"] - record["contract_total_recomputed_eur"]
                if pd.notna(record["contract_total_gross_eur"]) and pd.notna(record["contract_total_recomputed_eur"])
                else np.nan
            )
        rows.append(record)
    return pd.DataFrame(rows)


def collapse_salaries(source: pd.DataFrame) -> pd.DataFrame:
    source = source.copy()
    source["entity_key"] = np.where(
        source["canonical_player_id"].notna(),
        "id:" + source["canonical_player_id"].astype("Int64").astype(str),
        "name:" + source["player_name_normalized"],
    )
    source["club_key"] = np.where(
        source["canonical_club_id"].notna(),
        "id:" + source["canonical_club_id"].astype("Int64").astype(str),
        "name:" + source["club_name_normalized"],
    )
    key = ["entity_key", "club_key", "source_page_season"]
    source["selection_score"] = (
        source["active"].fillna(False).astype(int) * 4
        + (~source["loan"].fillna(False)).astype(int) * 2
        + source["annual_gross_eur"].notna().astype(int)
    )
    source = source.sort_values(key + ["selection_score", "source_file", "source_row_number"], ascending=[True, True, True, False, True, True])
    selected = source.groupby(key, sort=False, dropna=False).head(1).copy()
    aggregates = source.groupby(key, dropna=False).agg(
        source_row_count=("source_file", "size"),
        source_files=("source_file", join_unique),
        source_rows=("source_row_number", lambda x: " | ".join(str(int(v)) for v in sorted(x))),
        observed_statuses=("status", join_unique),
        observed_capology_suffixes=("capology_player_url_suffix", join_unique),
        annual_gross_eur_min=("annual_gross_eur", "min"),
        annual_gross_eur_max=("annual_gross_eur", "max"),
    ).reset_index()
    selected = selected.merge(aggregates, on=key, how="left", validate="one_to_one")
    selected["duplicate_source_rows_collapsed"] = selected["source_row_count"] > 1
    selected["annual_gross_conflict"] = (
        selected["annual_gross_eur_min"].notna()
        & selected["annual_gross_eur_max"].notna()
        & ~np.isclose(selected["annual_gross_eur_min"], selected["annual_gross_eur_max"])
    )
    selected["capology_salary_id"] = [stable_hash(a, b, c) for a, b, c in selected[key].itertuples(index=False, name=None)]
    selected["salary_natural_key"] = selected[key].astype(str).agg("|".join, axis=1)
    selected = selected.drop(columns=["selection_score"])
    first = ["capology_salary_id", "salary_natural_key"]
    return selected[first + [column for column in selected.columns if column not in first]].sort_values(
        ["source_page_season_start_year", "league", "club_name", "player_name"]
    ).reset_index(drop=True)


def collapse_extensions(source: pd.DataFrame) -> pd.DataFrame:
    source = source.copy()
    source["entity_key"] = np.where(
        source["canonical_player_id"].notna(),
        "id:" + source["canonical_player_id"].astype("Int64").astype(str),
        "name:" + source["player_name_normalized"],
    )
    source["club_key"] = np.where(
        source["canonical_club_id"].notna(),
        "id:" + source["canonical_club_id"].astype("Int64").astype(str),
        "name:" + source["club_name_normalized"],
    )
    source["signed_key"] = source["signed_date"].dt.strftime("%Y-%m-%d")
    key = ["entity_key", "club_key", "signed_key"]
    source["page_distance"] = (source["source_page_season_start_year"] - source["event_season_start_year"]).abs()
    source = source.sort_values(key + ["page_distance", "source_page_season_start_year", "source_file", "source_row_number"])
    selected = source.groupby(key, sort=False, dropna=False).head(1).copy()
    aggregates = source.groupby(key, dropna=False).agg(
        source_row_count=("source_file", "size"),
        source_files=("source_file", join_unique),
        source_rows=("source_row_number", lambda x: " | ".join(str(int(v)) for v in sorted(x))),
        source_page_seasons=("source_page_season", join_unique),
        observed_capology_suffixes=("capology_player_url_suffix", join_unique),
        reported_years_min=("reported_years", "min"),
        reported_years_max=("reported_years", "max"),
    ).reset_index()
    selected = selected.merge(aggregates, on=key, how="left", validate="one_to_one")
    selected["duplicate_event_rows_collapsed"] = selected["source_row_count"] > 1
    selected["reported_years_conflict"] = ~np.isclose(selected["reported_years_min"], selected["reported_years_max"])
    selected["event_outside_source_page_season"] = selected["page_distance"] > 0
    selected["capology_extension_event_id"] = [stable_hash(a, b, c) for a, b, c in selected[key].itertuples(index=False, name=None)]
    selected["extension_natural_key"] = selected[key].astype(str).agg("|".join, axis=1)
    selected = selected.drop(columns=["page_distance"])
    first = ["capology_extension_event_id", "extension_natural_key"]
    return selected[first + [column for column in selected.columns if column not in first]].sort_values(
        ["signed_date", "league", "club_name", "player_name"]
    ).reset_index(drop=True)


def add_check(checks: list[dict], check: str, observed: object, expected: object, passed: bool, notes: str) -> None:
    checks.append({"check": check, "observed": observed, "expected": expected, "passed": bool(passed), "notes": notes})


def field_dictionary(salary: pd.DataFrame, extension: pd.DataFrame) -> pd.DataFrame:
    definitions = {
        "capology_salary_id": "Deterministic hash of the canonical player-club-season key.",
        "capology_extension_event_id": "Deterministic hash of canonical player, club, and exact signing date.",
        "canonical_player_id": "Conservatively linked canonical Transfermarkt player identity; missing when unresolved.",
        "player_link_method": "Exact unique-name linkage method, or unresolved. No fuzzy links are forced.",
        "canonical_club_id": "Canonical Transfermarkt club identity from an accepted or reviewed exact alias.",
        "capology_player_url_suffix": "Numeric suffix parsed from Capology URL; provenance only and not a reliable identity key.",
        "source_row_count": "Number of raw rows represented by the canonical row.",
        "reported_zero_salary": "True where Capology literally reported zero; zero is converted to missing as unavailable salary.",
        "bonus_available": "True when bonus compensation was reported; missing bonus never means zero bonus.",
        "exact_duration_years": "Contract duration from expiration minus signing date divided by 365.25.",
        "reported_years": "Capology-reported season-count duration, retained separately from exact elapsed time.",
        "pre_completed_season_start_year": "Leakage-safe last completed season helper: signing calendar year minus one.",
        "first_full_post_season_start_year": "First full season beginning after the extension signing date.",
        "age_at_signing": "Exact age at signing from the canonical date of birth, where player identity is resolved.",
    }
    rows = []
    for dataset, frame in (("salary", salary), ("extension", extension)):
        for column in frame.columns:
            rows.append(
                {
                    "dataset": dataset,
                    "field": column,
                    "dtype": str(frame[column].dtype),
                    "non_null_rows": int(frame[column].notna().sum()),
                    "null_rows": int(frame[column].isna().sum()),
                    "definition": definitions.get(column, column.replace("_", " ").capitalize() + "."),
                }
            )
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d", lineterminator="\n")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    salary_files = sorted(LEAGUES.glob("*/sallaries/*.csv"))
    extension_files = sorted(LEAGUES.glob("*/extensions/*.csv"))
    files = salary_files + extension_files
    if len(salary_files) != EXPECTED_SALARY_FILES or len(extension_files) != EXPECTED_EXTENSION_FILES:
        raise RuntimeError(
            f"Expected {EXPECTED_SALARY_FILES} salary and {EXPECTED_EXTENSION_FILES} extension files; "
            f"found {len(salary_files)} and {len(extension_files)}"
        )
    manifest = copy_and_manifest(files)
    fbref_map, identity_map = build_player_map()
    club_map, _ = build_club_map()

    salary_source = pd.concat(
        [read_source(path, "salary", fbref_map, identity_map, club_map) for path in salary_files], ignore_index=True
    )
    extension_source = pd.concat(
        [read_source(path, "extension", fbref_map, identity_map, club_map) for path in extension_files], ignore_index=True
    )

    # A URL suffix is globally unsafe if it appears against multiple names.
    for frame in (salary_source, extension_source):
        conflicts = frame.groupby("capology_player_url_suffix")["player_name_normalized"].nunique()
        frame["capology_suffix_name_conflict"] = frame["capology_player_url_suffix"].map(conflicts).gt(1)

    salary = collapse_salaries(salary_source)
    extension = collapse_extensions(extension_source)

    unresolved_players = pd.concat(
        [
            salary_source.loc[salary_source["player_link_method"].eq("unresolved"), ["player_name", "player_name_normalized", "league", "source_file"]].assign(source_type="salary"),
            extension_source.loc[extension_source["player_link_method"].eq("unresolved"), ["player_name", "player_name_normalized", "league", "source_file"]].assign(source_type="extension"),
        ],
        ignore_index=True,
    )
    unresolved_players = (
        unresolved_players.groupby(["source_type", "player_name_normalized"], dropna=False)
        .agg(player_names=("player_name", join_unique), leagues=("league", join_unique), source_files=("source_file", join_unique), raw_rows=("source_file", "size"))
        .reset_index()
    )
    unresolved_clubs = pd.concat(
        [
            salary_source.loc[salary_source["club_link_method"].eq("unresolved"), ["club_name", "club_name_normalized", "league", "source_file"]].assign(source_type="salary"),
            extension_source.loc[extension_source["club_link_method"].eq("unresolved"), ["club_name", "club_name_normalized", "league", "source_file"]].assign(source_type="extension"),
        ],
        ignore_index=True,
    )
    if len(unresolved_clubs):
        unresolved_clubs = unresolved_clubs.groupby(["source_type", "club_name_normalized"], dropna=False).agg(
            club_names=("club_name", join_unique), leagues=("league", join_unique), source_files=("source_file", join_unique), raw_rows=("source_file", "size")
        ).reset_index()
    else:
        unresolved_clubs = pd.DataFrame(columns=["source_type", "club_name_normalized", "club_names", "leagues", "source_files", "raw_rows"])

    salary_summary = salary.groupby(["league", "source_page_season"], as_index=False).agg(
        canonical_rows=("capology_salary_id", "size"),
        linked_players=("canonical_player_id", "count"),
        linked_clubs=("canonical_club_id", "count"),
        annual_gross_available=("annual_gross_eur", "count"),
        bonus_available=("bonus_available", "sum"),
        duplicate_rows_collapsed=("duplicate_source_rows_collapsed", "sum"),
    )
    extension_summary = extension.groupby(["league", "event_season"], dropna=False, as_index=False).agg(
        canonical_events=("capology_extension_event_id", "size"),
        linked_players=("canonical_player_id", "count"),
        linked_clubs=("canonical_club_id", "count"),
        annual_gross_available=("annual_gross_eur", "count"),
        bonus_available=("bonus_available", "sum"),
        duplicate_events_collapsed=("duplicate_event_rows_collapsed", "sum"),
    )

    actions = pd.DataFrame(
        [
            ["Preserve raw sources", f"{len(files)} CSV files", len(files), "Copied byte-for-byte into a relative-path backup and verified SHA-256 hashes."],
            ["Parse HTML display cells", "All raw rows", len(salary_source) + len(extension_source), "Extracted names, club names, hrefs, slugs, and URL suffixes without changing raw files."],
            ["Standardize EUR money", "EUR fields", int(salary_source[EUR_FIELDS].notna().sum().sum() + extension_source[EXTENSION_EUR_FIELDS].notna().sum().sum()), "Removed currency symbols and separators; retained EUR as canonical currency."],
            ["Convert reported salary zero to missing", "Compensation fields", int(salary_source["reported_zero_salary"].sum() + extension_source["reported_zero_salary"].sum()), "Capology zero salaries represent unavailable estimates, not defensible evidence of free labor."],
            ["Preserve absent bonuses as missing", "Bonus fields", int((~salary_source["bonus_available"]).sum() + (~extension_source["bonus_available"]).sum()), "A dash or blank remains missing and is never recoded to zero."],
            ["Conservative player linkage", "All raw rows", int(salary_source["canonical_player_id"].notna().sum() + extension_source["canonical_player_id"].notna().sum()), "Accepted exact unique FBref-name links, then exact unique canonical-name links; unresolved names remain unresolved."],
            ["Reviewed club linkage", "All raw rows", int(salary_source["canonical_club_id"].notna().sum() + extension_source["canonical_club_id"].notna().sum()), "Used accepted FBref aliases plus a documented reviewed exact Capology alias table."],
            ["Collapse same player-club-season salary rows", "Salary source", len(salary_source) - len(salary), "Selected active, non-loan, salary-available record deterministically and retained source-row counts and conflicts."],
            ["Collapse repeated extension events", "Extension source", len(extension_source) - len(extension), "Selected the page closest to the actual signing season and retained all contributing files and reported-years conflicts."],
            ["Compute exact duration", "Extension events", len(extension), "Computed expiration minus signing date; retained Capology's reported season count separately."],
        ],
        columns=["action", "scope", "affected_rows_or_cells", "details"],
    )

    checks: list[dict] = []
    add_check(checks, "salary_source_file_count", len(salary_files), EXPECTED_SALARY_FILES, len(salary_files) == EXPECTED_SALARY_FILES, "Five leagues by seven salary seasons.")
    add_check(checks, "extension_source_file_count", len(extension_files), EXPECTED_EXTENSION_FILES, len(extension_files) == EXPECTED_EXTENSION_FILES, "Five new 2024-25 pages added; no 2017-18 pages and Premier League 2022-23 remains unavailable.")
    add_check(checks, "salary_source_rows", len(salary_source), EXPECTED_SALARY_ROWS, len(salary_source) == EXPECTED_SALARY_ROWS, "Raw salary rows parsed.")
    add_check(checks, "extension_source_rows", len(extension_source), EXPECTED_EXTENSION_ROWS, len(extension_source) == EXPECTED_EXTENSION_ROWS, "Raw extension rows parsed, including 380 from the five 2024-25 pages.")
    add_check(checks, "extension_canonical_events", len(extension), EXPECTED_EXTENSION_EVENTS, len(extension) == EXPECTED_EXTENSION_EVENTS, "Six repeated adjacent-page events collapsed.")
    add_check(checks, "source_backup_hashes", int(manifest["hashes_match"].sum()), len(manifest), manifest["hashes_match"].all(), "Every exact backup matches its source.")
    add_check(checks, "salary_key_unique", int(salary["capology_salary_id"].duplicated().sum()), 0, not salary["capology_salary_id"].duplicated().any(), "One row per canonical player-club-season key.")
    add_check(checks, "extension_key_unique", int(extension["capology_extension_event_id"].duplicated().sum()), 0, not extension["capology_extension_event_id"].duplicated().any(), "One row per player-club-signing-date event.")
    add_check(checks, "extension_positive_duration", int((extension["exact_duration_days"] > 0).sum()), len(extension), extension["exact_duration_days"].gt(0).all(), "All expiration dates follow signing dates.")
    add_check(checks, "extension_club_link_complete", int(extension["canonical_club_id"].notna().sum()), len(extension), extension["canonical_club_id"].notna().all(), "Every extension club has an accepted or reviewed exact mapping.")
    add_check(checks, "extension_player_link_rate", round(extension["canonical_player_id"].notna().mean(), 4), ">=0.95", extension["canonical_player_id"].notna().mean() >= 0.95, "Unresolved names are preserved rather than forced.")
    ext_arithmetic = extension["contract_total_arithmetic_difference_eur"].dropna().abs().le(100)
    add_check(checks, "extension_contract_total_arithmetic", int(ext_arithmetic.sum()), len(ext_arithmetic), bool(ext_arithmetic.all()), "Reported contract total equals total annual gross times reported years within EUR 100.")
    zero_numeric = int((salary[EUR_FIELDS] == 0).sum().sum() + (extension[EXTENSION_EUR_FIELDS] == 0).sum().sum())
    add_check(checks, "no_zero_salary_values_retained", zero_numeric, 0, zero_numeric == 0, "Literal zero compensation values were flagged and converted to missing.")
    checks_frame = pd.DataFrame(checks)

    write_csv(salary, OUTPUT / "canonical_salary_panel.csv")
    write_csv(extension, OUTPUT / "canonical_extension_events.csv")
    write_csv(unresolved_players, OUTPUT / "unresolved_player_matches.csv")
    write_csv(unresolved_clubs, OUTPUT / "unresolved_club_matches.csv")
    write_csv(salary_summary, OUTPUT / "salary_summary.csv")
    write_csv(extension_summary, OUTPUT / "extension_summary.csv")
    write_csv(actions, OUTPUT / "capology_cleaning_actions.csv")
    write_csv(field_dictionary(salary, extension), OUTPUT / "field_dictionary.csv")
    write_csv(manifest, OUTPUT / "source_manifest.csv")
    write_csv(checks_frame, OUTPUT / "build_checks.csv")

    summary = {
        "salary_source_files": len(salary_files),
        "extension_source_files": len(extension_files),
        "salary_source_rows": len(salary_source),
        "canonical_salary_rows": len(salary),
        "salary_duplicate_rows_collapsed": len(salary_source) - len(salary),
        "extension_source_rows": len(extension_source),
        "canonical_extension_events": len(extension),
        "extension_duplicate_rows_collapsed": len(extension_source) - len(extension),
        "extension_player_links": int(extension["canonical_player_id"].notna().sum()),
        "extension_player_link_rate": round(float(extension["canonical_player_id"].notna().mean()), 6),
        "extension_club_links": int(extension["canonical_club_id"].notna().sum()),
        "extension_club_link_rate": round(float(extension["canonical_club_id"].notna().mean()), 6),
        "salary_annual_gross_coverage": round(float(salary["annual_gross_eur"].notna().mean()), 6),
        "extension_annual_gross_coverage": round(float(extension["annual_gross_eur"].notna().mean()), 6),
        "extension_bonus_coverage": round(float(extension["bonus_available"].mean()), 6),
        "checks_passed": int(checks_frame["passed"].sum()),
        "checks_total": len(checks_frame),
        "all_checks_passed": bool(checks_frame["passed"].all()),
    }
    (OUTPUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    readme = f"""# Canonical Capology contracts data

Generated by `scripts/build_capology_contracts.py` from immutable CSVs under `Data/leagues`.

## Outputs

- `canonical_salary_panel.csv`: {len(salary):,} player-club-season salary observations from {len(salary_source):,} raw rows.
- `canonical_extension_events.csv`: {len(extension):,} unique extension events from {len(extension_source):,} raw rows.
- `unresolved_player_matches.csv` and `unresolved_club_matches.csv`: explicit review queues; no uncertain identity is forced.
- `salary_summary.csv` and `extension_summary.csv`: coverage by league and season/event season.
- `field_dictionary.csv`, `capology_cleaning_actions.csv`, `source_manifest.csv`, and `build_checks.csv`: definitions and audit trail.
- `independent_verification.csv` and `.json`: written by the separate verifier.

## Material rules and limitations

1. Every source is copied byte-for-byte under `{BACKUP.relative_to(ROOT).as_posix()}` and hash checked. The misspelled raw folder name `sallaries` is deliberately left unchanged.
2. EUR is the canonical currency. Capology's redundant converted GBP/USD fields remain available in raw files but are not repeated here.
3. Literal zero salary is flagged and converted to missing because it denotes an unavailable estimate. Missing bonuses remain missing and never become zero.
4. A salary row represents a player at one club in one season. Mid-season movement can therefore produce multiple club rows for one player-season. Same player-club-season duplicates are selected deterministically with provenance retained.
5. Extension events use exact signing and expiration dates. `exact_duration_years` is the modeling duration; `reported_years` is retained because Capology usually counts contract seasons rather than elapsed years.
6. Capology URL suffixes are provenance only: suffixes are reused across different names and are not safe identity keys.
7. Player linkage is exact and conservative. Extension linkage is {summary['extension_player_links']:,}/{len(extension):,} ({summary['extension_player_link_rate']:.1%}); unresolved identities remain visible. Club linkage is complete for extension events through accepted and reviewed exact aliases.
8. Extension pages cover 2018-19 through 2024-25 for all five leagues except the unavailable Premier League 2022-23 page; 2017-18 pages are also unavailable. Bonus coverage is sparse before 2022, so bonus-inclusive analyses must use explicit availability filters.
9. These tables are a cleaned contract-data layer, not a causal model. Performance, retention, and market-value outcomes must be joined later with leakage-safe timing.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")

    output_files = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name != "output_manifest.csv")
    output_manifest = pd.DataFrame(
        [{"output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in output_files]
    )
    write_csv(output_manifest, OUTPUT / "output_manifest.csv")

    if not checks_frame["passed"].all():
        failed = checks_frame.loc[~checks_frame["passed"], "check"].tolist()
        raise RuntimeError(f"Build completed with failed checks: {failed}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

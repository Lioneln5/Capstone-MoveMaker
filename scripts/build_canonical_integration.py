from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "Data" / "processed"
TM = PROCESSED / "transfermarkt_clean" / "tables"
TM_ROOT = PROCESSED / "transfermarkt_clean"
DL = PROCESSED / "football_datalake_clean" / "tables"
XW = PROCESSED / "integration_crosswalks"
OUTPUT = PROCESSED / "canonical_integration"
CHUNK = 250_000

SPECIAL_TRANSLITERATION = str.maketrans(
    {"ð": "d", "ł": "l", "ø": "o", "æ": "ae", "œ": "oe", "ß": "ss", "þ": "th", "đ": "d", "ı": "i"}
)


def text_id(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if re.fullmatch(r"-?\d+\.0", text) else text


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalized_text(value: object) -> str:
    text = clean_text(value).lower().translate(SPECIAL_TRANSLITERATION)
    text = "".join(char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "", text)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype("string").str.lower().eq("true")


def choose(preferred: pd.Series, fallback: pd.Series) -> pd.Series:
    preferred = preferred.astype("string").fillna("").str.strip()
    fallback = fallback.astype("string").fillna("").str.strip()
    return preferred.mask(preferred.eq(""), fallback)


def conflict(left: pd.Series, right: pd.Series, normalizer=None) -> pd.Series:
    left = left.astype("string").fillna("").str.strip()
    right = right.astype("string").fillna("").str.strip()
    if normalizer:
        left_compare = left.map(normalizer)
        right_compare = right.map(normalizer)
    else:
        left_compare, right_compare = left, right
    return left.ne("") & right.ne("") & left_compare.ne(right_compare)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, filename: str) -> Path:
    path = OUTPUT / filename
    frame.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    return path


def canonical_season_from_date(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series, errors="coerce")
    start_year = dates.dt.year - (dates.dt.month < 7).astype("Int64")
    end_year = start_year + 1
    result = start_year.astype("string") + "-" + end_year.astype("string")
    return result.mask(dates.isna(), pd.NA)


def build_valuation_tables():
    current = pd.read_csv(
        TM / "player_valuations_clean.csv",
        usecols=[
            "source_row_number", "player_id", "date", "market_value_in_eur", "current_club_name",
            "current_club_id", "player_club_domestic_competition_id", "has_positive_market_value",
        ],
        dtype="string",
        low_memory=False,
    )
    current["canonical_player_id"] = current["player_id"].map(text_id)
    current["valuation_date"] = current["date"].str.strip()
    current["market_value_eur"] = numeric(current["market_value_in_eur"])
    current_evidence = pd.DataFrame(
        {
            "source_dataset": "current_transfermarkt",
            "source_table": "player_valuations_clean",
            "source_record_id": "current_valuation:" + current["source_row_number"].astype("string"),
            "canonical_player_id": current["canonical_player_id"],
            "valuation_date": current["valuation_date"],
            "market_value_eur": current["market_value_eur"],
            "source_current_club_id": current["current_club_id"].map(text_id),
            "source_current_club_name": current["current_club_name"],
            "source_competition_id": current["player_club_domestic_competition_id"],
            "source_value_status": np.where(current["market_value_eur"].gt(0), "positive_reported", np.where(current["market_value_eur"].eq(0), "reported_zero", "missing")),
            "source_natural_key_missing": current["canonical_player_id"].eq("") | current["valuation_date"].fillna("").eq(""),
            "source_natural_key_duplicate": False,
        }
    )

    datalake = pd.read_csv(
        DL / "player_market_value_clean.csv",
        usecols=[
            "source_record_id", "player_id", "date_unix", "value", "valuation_status",
            "natural_key_missing", "natural_key_duplicate",
        ],
        dtype="string",
        low_memory=False,
    )
    datalake_evidence = pd.DataFrame(
        {
            "source_dataset": "supplemental_datalake",
            "source_table": "player_market_value_clean",
            "source_record_id": datalake["source_record_id"],
            "canonical_player_id": datalake["player_id"].map(text_id),
            "valuation_date": datalake["date_unix"].str.strip(),
            "market_value_eur": numeric(datalake["value"]),
            "source_current_club_id": "",
            "source_current_club_name": "",
            "source_competition_id": "",
            "source_value_status": datalake["valuation_status"],
            "source_natural_key_missing": bool_series(datalake["natural_key_missing"]),
            "source_natural_key_duplicate": bool_series(datalake["natural_key_duplicate"]),
        }
    )

    evidence = pd.concat([current_evidence, datalake_evidence], ignore_index=True)
    evidence["canonical_valuation_id"] = "valuation:" + evidence["canonical_player_id"] + ":" + evidence["valuation_date"].fillna("")
    incomplete = evidence["source_natural_key_missing"] | evidence["canonical_player_id"].eq("") | evidence["valuation_date"].fillna("").eq("")
    evidence.loc[incomplete, "canonical_valuation_id"] = "valuation_incomplete:" + evidence.loc[incomplete, "source_dataset"] + ":" + evidence.loc[incomplete, "source_record_id"]
    evidence["exact_key_value_duplicate"] = evidence.duplicated(
        ["canonical_valuation_id", "market_value_eur"], keep=False
    )
    evidence["cross_source_exact_match"] = evidence.groupby(
        ["canonical_valuation_id", "market_value_eur"], dropna=False
    )["source_dataset"].transform("nunique").gt(1)

    evidence["is_current"] = evidence["source_dataset"].eq("current_transfermarkt").astype(int)
    evidence["is_datalake"] = evidence["source_dataset"].eq("supplemental_datalake").astype(int)
    grouped = evidence.groupby("canonical_valuation_id", sort=False, dropna=False)
    spine = grouped.agg(
        canonical_player_id=("canonical_player_id", "first"),
        valuation_date=("valuation_date", "first"),
        evidence_count=("source_record_id", "size"),
        source_count=("source_dataset", "nunique"),
        current_evidence_count=("is_current", "sum"),
        datalake_evidence_count=("is_datalake", "sum"),
        distinct_value_count=("market_value_eur", "nunique"),
        minimum_reported_value_eur=("market_value_eur", "min"),
        maximum_reported_value_eur=("market_value_eur", "max"),
        natural_key_missing=("source_natural_key_missing", "max"),
    ).reset_index()

    for source_name, prefix in [("current_transfermarkt", "current"), ("supplemental_datalake", "datalake")]:
        subset = evidence[evidence["source_dataset"].eq(source_name)]
        source_group = subset.groupby("canonical_valuation_id", sort=False).agg(
            **{
                f"{prefix}_value_distinct_count": ("market_value_eur", "nunique"),
                f"{prefix}_minimum_value_eur": ("market_value_eur", "min"),
                f"{prefix}_maximum_value_eur": ("market_value_eur", "max"),
            }
        ).reset_index()
        spine = spine.merge(source_group, on="canonical_valuation_id", how="left")

    spine["canonical_market_value_eur"] = spine["minimum_reported_value_eur"].where(spine["distinct_value_count"].eq(1))
    spine["value_conflict"] = spine["distinct_value_count"].gt(1)
    spine["value_difference_eur"] = (spine["maximum_reported_value_eur"] - spine["minimum_reported_value_eur"]).where(spine["value_conflict"])
    spine["cross_source_overlap"] = spine["source_count"].gt(1)
    spine["within_source_duplicate"] = spine["current_evidence_count"].gt(1) | spine["datalake_evidence_count"].gt(1)
    spine["value_resolution_status"] = np.select(
        [
            spine["natural_key_missing"].astype(bool),
            spine["distinct_value_count"].eq(0),
            spine["value_conflict"],
            spine["cross_source_overlap"],
            spine["within_source_duplicate"],
        ],
        ["incomplete_key", "missing_value", "unresolved_value_conflict", "cross_source_consensus", "within_source_duplicate_consensus"],
        default="single_source_reported",
    )

    date_values = pd.to_datetime(spine["valuation_date"], errors="coerce")
    spine["valuation_date_valid"] = date_values.notna()
    spine = spine.sort_values(["canonical_player_id", "valuation_date", "canonical_valuation_id"])
    valid = spine[spine["valuation_date_valid"] & spine["canonical_player_id"].ne("")].copy()
    latest_idx = valid.groupby("canonical_player_id")["valuation_date"].idxmax()
    latest = valid.loc[latest_idx].copy().sort_values("canonical_player_id")
    latest.insert(0, "canonical_latest_valuation_id", "latest:" + latest["canonical_player_id"])
    latest["latest_value_available"] = latest["canonical_market_value_eur"].notna()
    latest["derived_from_canonical_history"] = True

    conflict_review = spine[
        spine["value_conflict"] | spine["within_source_duplicate"] | spine["natural_key_missing"].astype(bool)
    ].copy().sort_values(["value_resolution_status", "valuation_date", "canonical_player_id"])

    evidence = evidence.drop(columns=["is_current", "is_datalake"])
    return evidence, spine, latest, conflict_review


def build_transfer_tables():
    current = pd.read_csv(TM / "transfers_clean.csv", dtype="string", low_memory=False)
    current_player = current["player_id"].map(text_id)
    current_date = current["transfer_date"].str.strip()
    current_from = current["from_club_id"].map(text_id)
    current_to = current["to_club_id"].map(text_id)
    current_key_missing = current_player.eq("") | current_date.fillna("").eq("") | current_from.eq("") | current_to.eq("")
    current_evidence = pd.DataFrame(
        {
            "source_dataset": "current_transfermarkt",
            "source_table": "transfers_clean",
            "source_record_id": "current_transfer:" + current["transfer_event_id"].astype("string"),
            "canonical_player_id": current_player,
            "transfer_date": current_date,
            "from_club_id": current_from,
            "to_club_id": current_to,
            "source_from_club_name": current["from_club_name"],
            "source_to_club_name": current["to_club_name"],
            "source_transfer_type": "",
            "source_transfer_fee_eur": numeric(current["transfer_fee"]),
            "source_market_value_eur": numeric(current["market_value_in_eur"]),
            "source_transfer_season": current["transfer_season"],
            "canonical_season_key": canonical_season_from_date(current_date),
            "source_transfer_fee_status": current["transfer_fee_status"],
            "source_market_value_status": np.where(numeric(current["market_value_in_eur"]).notna(), "reported", "missing"),
            "source_natural_key_missing": current_key_missing,
            "source_natural_key_duplicate": False,
        }
    )

    datalake = pd.read_csv(DL / "transfer_history_clean.csv", dtype="string", low_memory=False)
    datalake_evidence = pd.DataFrame(
        {
            "source_dataset": "supplemental_datalake",
            "source_table": "transfer_history_clean",
            "source_record_id": datalake["source_record_id"],
            "canonical_player_id": datalake["player_id"].map(text_id),
            "transfer_date": datalake["transfer_date"].str.strip(),
            "from_club_id": datalake["from_team_id"].map(text_id),
            "to_club_id": datalake["to_team_id"].map(text_id),
            "source_from_club_name": datalake["from_team_name"],
            "source_to_club_name": datalake["to_team_name"],
            "source_transfer_type": datalake["transfer_type_normalized"],
            "source_transfer_fee_eur": numeric(datalake["transfer_fee"]),
            "source_market_value_eur": numeric(datalake["value_at_transfer"]),
            "source_transfer_season": datalake["season_name"],
            "canonical_season_key": canonical_season_from_date(datalake["transfer_date"]),
            "source_transfer_fee_status": datalake["transfer_fee_status"],
            "source_market_value_status": datalake["value_at_transfer_status"],
            "source_natural_key_missing": bool_series(datalake["natural_key_missing"]),
            "source_natural_key_duplicate": bool_series(datalake["natural_key_duplicate"]),
        }
    )
    evidence = pd.concat([current_evidence, datalake_evidence], ignore_index=True)
    complete = ~(
        evidence["source_natural_key_missing"]
        | evidence["canonical_player_id"].eq("")
        | evidence["transfer_date"].fillna("").eq("")
        | evidence["from_club_id"].eq("")
        | evidence["to_club_id"].eq("")
    )
    evidence["canonical_transfer_event_id"] = (
        "transfer:" + evidence["canonical_player_id"] + ":" + evidence["transfer_date"].fillna("")
        + ":" + evidence["from_club_id"] + ":" + evidence["to_club_id"]
    )
    evidence.loc[~complete, "canonical_transfer_event_id"] = (
        "transfer_incomplete:" + evidence.loc[~complete, "source_dataset"] + ":" + evidence.loc[~complete, "source_record_id"]
    )
    evidence["source_natural_key_missing"] = ~complete
    evidence["exact_key_fee_value_duplicate"] = evidence.duplicated(
        ["canonical_transfer_event_id", "source_transfer_fee_eur", "source_market_value_eur"], keep=False
    )
    evidence["cross_source_exact_fee_value"] = evidence.groupby(
        ["canonical_transfer_event_id", "source_transfer_fee_eur", "source_market_value_eur"], dropna=False
    )["source_dataset"].transform("nunique").gt(1)
    evidence["is_current"] = evidence["source_dataset"].eq("current_transfermarkt").astype(int)
    evidence["is_datalake"] = evidence["source_dataset"].eq("supplemental_datalake").astype(int)
    evidence["transfer_type_for_group"] = evidence["source_transfer_type"].replace("", pd.NA)

    grouped = evidence.groupby("canonical_transfer_event_id", sort=False, dropna=False)
    spine = grouped.agg(
        canonical_player_id=("canonical_player_id", "first"),
        transfer_date=("transfer_date", "first"),
        canonical_season_key=("canonical_season_key", "first"),
        from_club_id=("from_club_id", "first"),
        to_club_id=("to_club_id", "first"),
        source_from_club_name_sample=("source_from_club_name", "first"),
        source_to_club_name_sample=("source_to_club_name", "first"),
        evidence_count=("source_record_id", "size"),
        source_count=("source_dataset", "nunique"),
        current_evidence_count=("is_current", "sum"),
        datalake_evidence_count=("is_datalake", "sum"),
        transfer_type_distinct_count=("transfer_type_for_group", "nunique"),
        canonical_transfer_type=("transfer_type_for_group", "first"),
        transfer_fee_distinct_count=("source_transfer_fee_eur", "nunique"),
        minimum_reported_transfer_fee_eur=("source_transfer_fee_eur", "min"),
        maximum_reported_transfer_fee_eur=("source_transfer_fee_eur", "max"),
        market_value_distinct_count=("source_market_value_eur", "nunique"),
        minimum_reported_market_value_eur=("source_market_value_eur", "min"),
        maximum_reported_market_value_eur=("source_market_value_eur", "max"),
        natural_key_missing=("source_natural_key_missing", "max"),
    ).reset_index()

    for source_name, prefix in [("current_transfermarkt", "current"), ("supplemental_datalake", "datalake")]:
        subset = evidence[evidence["source_dataset"].eq(source_name)]
        source_group = subset.groupby("canonical_transfer_event_id", sort=False).agg(
            **{
                f"{prefix}_fee_distinct_count": ("source_transfer_fee_eur", "nunique"),
                f"{prefix}_minimum_fee_eur": ("source_transfer_fee_eur", "min"),
                f"{prefix}_maximum_fee_eur": ("source_transfer_fee_eur", "max"),
                f"{prefix}_market_value_distinct_count": ("source_market_value_eur", "nunique"),
                f"{prefix}_minimum_market_value_eur": ("source_market_value_eur", "min"),
                f"{prefix}_maximum_market_value_eur": ("source_market_value_eur", "max"),
            }
        ).reset_index()
        spine = spine.merge(source_group, on="canonical_transfer_event_id", how="left")

    spine["canonical_transfer_fee_eur"] = spine["minimum_reported_transfer_fee_eur"].where(spine["transfer_fee_distinct_count"].eq(1))
    spine["transfer_fee_conflict"] = spine["transfer_fee_distinct_count"].gt(1)
    spine["transfer_fee_difference_eur"] = (spine["maximum_reported_transfer_fee_eur"] - spine["minimum_reported_transfer_fee_eur"]).where(spine["transfer_fee_conflict"])
    spine["canonical_market_value_eur"] = spine["minimum_reported_market_value_eur"].where(spine["market_value_distinct_count"].eq(1))
    spine["market_value_conflict"] = spine["market_value_distinct_count"].gt(1)
    spine["market_value_difference_eur"] = (spine["maximum_reported_market_value_eur"] - spine["minimum_reported_market_value_eur"]).where(spine["market_value_conflict"])
    spine["transfer_type_conflict"] = spine["transfer_type_distinct_count"].gt(1)
    spine.loc[spine["transfer_type_conflict"], "canonical_transfer_type"] = ""
    spine["cross_source_overlap"] = spine["source_count"].gt(1)
    spine["within_source_duplicate"] = spine["current_evidence_count"].gt(1) | spine["datalake_evidence_count"].gt(1)
    spine["duplicate_evidence_rows_collapsed"] = (spine["current_evidence_count"] - 1).clip(lower=0) + (spine["datalake_evidence_count"] - 1).clip(lower=0)
    spine["canonical_transfer_fee_status"] = np.select(
        [
            spine["transfer_fee_conflict"].fillna(False).to_numpy(dtype=bool),
            spine["transfer_fee_distinct_count"].eq(0).fillna(False).to_numpy(dtype=bool),
            spine["canonical_transfer_fee_eur"].eq(0).fillna(False).to_numpy(dtype=bool),
            spine["canonical_transfer_fee_eur"].gt(0).fillna(False).to_numpy(dtype=bool),
        ],
        ["unresolved_fee_conflict", "missing", "unclassified_zero", "positive_reported"],
        default="missing",
    )
    spine["event_conflict"] = spine[["transfer_fee_conflict", "market_value_conflict", "transfer_type_conflict"]].any(axis=1)
    spine["event_resolution_status"] = np.select(
        [
            spine["natural_key_missing"].fillna(False).to_numpy(dtype=bool),
            spine["event_conflict"].fillna(False).to_numpy(dtype=bool),
            spine["within_source_duplicate"].fillna(False).to_numpy(dtype=bool),
            spine["cross_source_overlap"].fillna(False).to_numpy(dtype=bool),
        ],
        ["incomplete_key_retained", "unresolved_source_conflict", "within_source_duplicates_collapsed", "cross_source_evidence_reconciled"],
        default="single_source_event",
    )

    club_names = pd.read_csv(XW / "club_identity_crosswalk.csv", usecols=["canonical_club_id", "canonical_name"], dtype="string")
    club_name_map = dict(zip(club_names["canonical_club_id"].map(text_id), club_names["canonical_name"].fillna("")))
    spine["canonical_from_club_name"] = spine["from_club_id"].map(club_name_map)
    spine["canonical_to_club_name"] = spine["to_club_id"].map(club_name_map)
    spine["canonical_from_club_name"] = choose(spine["canonical_from_club_name"], spine["source_from_club_name_sample"])
    spine["canonical_to_club_name"] = choose(spine["canonical_to_club_name"], spine["source_to_club_name_sample"])

    conflict_review = spine[
        spine["event_conflict"] | spine["within_source_duplicate"] | spine["natural_key_missing"].astype(bool)
    ].copy()
    spine = spine.sort_values(["transfer_date", "canonical_player_id", "canonical_transfer_event_id"])
    conflict_review = conflict_review.sort_values(["event_resolution_status", "transfer_date", "canonical_player_id"])
    evidence = evidence.drop(columns=["is_current", "is_datalake", "transfer_type_for_group"])
    return evidence, spine, conflict_review


def build_player_dimension(valuation_evidence, transfer_evidence):
    identity = pd.read_csv(XW / "player_identity_crosswalk.csv", dtype="string", low_memory=False)
    identity["canonical_player_id"] = identity["canonical_player_id"].map(text_id)
    identity["present_in_identity_crosswalk"] = True
    all_ids = set(identity["canonical_player_id"])
    valuation_ids = set(valuation_evidence["canonical_player_id"]) - {""}
    transfer_ids = set(transfer_evidence["canonical_player_id"]) - {""}
    missing = sorted((valuation_ids | transfer_ids) - all_ids, key=lambda value: (not value.lstrip("-").isdigit(), int(value) if value.lstrip("-").isdigit() else value))
    if missing:
        stubs = pd.DataFrame({"canonical_player_id": missing})
        stubs["canonical_player_key"] = "tm_player:" + stubs["canonical_player_id"]
        stubs["present_in_identity_crosswalk"] = False
        identity = pd.concat([identity, stubs], ignore_index=True)

    current = pd.read_csv(
        TM / "players_clean.csv",
        usecols=[
            "player_id", "name", "date_of_birth", "country_of_birth", "country_of_citizenship",
            "position", "sub_position", "foot", "height_in_cm", "current_club_id", "current_club_name",
            "contract_expiration_date", "agent_name", "last_season",
        ],
        dtype="string",
        low_memory=False,
    ).drop_duplicates("player_id")
    current["canonical_player_id"] = current["player_id"].map(text_id)
    current = current.rename(columns={column: f"current_{column}" for column in current.columns if column not in {"canonical_player_id"}})

    datalake = pd.read_csv(
        DL / "player_profiles_clean.csv",
        usecols=[
            "source_record_id", "player_id", "player_name_display", "date_of_birth", "country_of_birth", "citizenship",
            "position", "main_position", "foot", "height", "current_club_id", "current_club_name",
            "contract_expires", "player_agent_name", "joined", "on_loan_from_club_id",
        ],
        dtype="string",
        low_memory=False,
    ).drop_duplicates("player_id")
    datalake["canonical_player_id"] = datalake["player_id"].map(text_id)
    datalake = datalake.rename(columns={column: f"datalake_{column}" for column in datalake.columns if column not in {"canonical_player_id"}})

    dimension = identity.merge(current, on="canonical_player_id", how="left").merge(datalake, on="canonical_player_id", how="left")
    accepted_fbref = pd.read_csv(XW / "fbref_player_crosswalk.csv", dtype="string")
    accepted_fbref = accepted_fbref[accepted_fbref["match_status"].eq("accepted")][
        ["canonical_player_id", "fbref_player_identity_key", "fbref_player_name", "match_method"]
    ].rename(columns={"match_method": "fbref_match_method"})
    dimension = dimension.merge(accepted_fbref, on="canonical_player_id", how="left")

    dimension["present_in_current_player_profile"] = dimension["current_player_id"].notna()
    dimension["present_in_datalake_player_profile"] = dimension["datalake_player_id"].notna()
    dimension["present_in_current_valuations"] = dimension["canonical_player_id"].isin(set(valuation_evidence.loc[valuation_evidence["source_dataset"].eq("current_transfermarkt"), "canonical_player_id"]))
    dimension["present_in_datalake_valuations"] = dimension["canonical_player_id"].isin(set(valuation_evidence.loc[valuation_evidence["source_dataset"].eq("supplemental_datalake"), "canonical_player_id"]))
    dimension["present_in_current_transfers"] = dimension["canonical_player_id"].isin(set(transfer_evidence.loc[transfer_evidence["source_dataset"].eq("current_transfermarkt"), "canonical_player_id"]))
    dimension["present_in_datalake_transfers"] = dimension["canonical_player_id"].isin(set(transfer_evidence.loc[transfer_evidence["source_dataset"].eq("supplemental_datalake"), "canonical_player_id"]))
    dimension["canonical_name"] = choose(dimension["canonical_name"], choose(dimension["current_name"], dimension["datalake_player_name_display"]))
    dimension["canonical_name_normalized"] = dimension["canonical_name"].map(normalized_text)
    dimension["present_in_identity_crosswalk"] = dimension["present_in_identity_crosswalk"].fillna(False).astype(bool)
    dimension["identity_record_type"] = np.select(
        [
            ~dimension["present_in_identity_crosswalk"],
            dimension["canonical_name"].fillna("").eq(""),
        ],
        ["fact_only_stub", "unnamed_identity"],
        default="named_identity",
    )
    dimension["current_date_of_birth"] = dimension["current_date_of_birth"].fillna("")
    dimension["datalake_date_of_birth"] = dimension["datalake_date_of_birth"].fillna("")
    dimension["canonical_date_of_birth"] = choose(dimension["current_date_of_birth"], choose(dimension["date_of_birth"], dimension["datalake_date_of_birth"]))
    dimension["date_of_birth_conflict"] = conflict(dimension["current_date_of_birth"], dimension["datalake_date_of_birth"])
    dimension["canonical_birth_year"] = dimension["canonical_date_of_birth"].str.extract(r"((?:18|19|20)\d{2})", expand=False)

    dimension["canonical_country_of_birth"] = choose(dimension["current_country_of_birth"], dimension["datalake_country_of_birth"])
    dimension["country_of_birth_conflict"] = conflict(dimension["current_country_of_birth"], dimension["datalake_country_of_birth"], normalized_text)
    dimension["canonical_citizenship"] = choose(dimension["current_country_of_citizenship"], dimension["datalake_citizenship"])
    dimension["citizenship_conflict"] = conflict(dimension["current_country_of_citizenship"], dimension["datalake_citizenship"], normalized_text)
    dimension["canonical_foot"] = choose(dimension["current_foot"], dimension["datalake_foot"])
    dimension["foot_conflict"] = conflict(dimension["current_foot"], dimension["datalake_foot"], normalized_text)

    current_height = numeric(dimension["current_height_in_cm"]).where(numeric(dimension["current_height_in_cm"]).gt(0))
    datalake_height = numeric(dimension["datalake_height"]).where(numeric(dimension["datalake_height"]).gt(0))
    dimension["current_height_cm"] = current_height
    dimension["datalake_height_cm"] = datalake_height
    dimension["canonical_height_cm"] = current_height.combine_first(datalake_height)
    dimension["height_conflict"] = current_height.notna() & datalake_height.notna() & current_height.ne(datalake_height)
    dimension["canonical_position"] = choose(dimension["current_position"], dimension["datalake_position"])
    dimension["canonical_sub_position"] = choose(dimension["current_sub_position"], dimension["datalake_main_position"])
    dimension["stable_attribute_conflict"] = dimension[
        ["date_of_birth_conflict", "country_of_birth_conflict", "citizenship_conflict", "foot_conflict", "height_conflict"]
    ].any(axis=1)
    dimension["stable_attribute_resolution_status"] = np.select(
        [dimension["identity_record_type"].ne("named_identity"), dimension["stable_attribute_conflict"]],
        ["stub_missing_profile", "source_attribute_conflict"],
        default="resolved_by_source_precedence",
    )
    dimension["snapshot_fields_excluded_from_stable_dimension"] = True

    keep = [
        "canonical_player_id", "canonical_player_key", "identity_record_type", "present_in_identity_crosswalk", "canonical_name",
        "canonical_name_normalized", "name_aliases", "canonical_date_of_birth", "canonical_birth_year",
        "canonical_country_of_birth", "canonical_citizenship", "canonical_position", "canonical_sub_position",
        "canonical_foot", "canonical_height_cm", "current_date_of_birth", "datalake_date_of_birth",
        "date_of_birth_conflict", "current_country_of_birth", "datalake_country_of_birth", "country_of_birth_conflict",
        "current_country_of_citizenship", "datalake_citizenship", "citizenship_conflict", "current_foot",
        "datalake_foot", "foot_conflict", "current_height_cm", "datalake_height_cm", "height_conflict",
        "stable_attribute_conflict", "stable_attribute_resolution_status", "present_in_current_player_profile",
        "present_in_datalake_player_profile", "present_in_current_valuations", "present_in_datalake_valuations",
        "present_in_current_transfers", "present_in_datalake_transfers", "fbref_player_identity_key",
        "fbref_player_name", "fbref_match_method", "snapshot_fields_excluded_from_stable_dimension",
    ]
    dimension = dimension[keep].sort_values("canonical_player_id", key=lambda values: values.map(lambda value: (not str(value).lstrip("-").isdigit(), int(value) if str(value).lstrip("-").isdigit() else str(value))))

    current_snapshots = pd.DataFrame(
        {
            "canonical_player_id": current["canonical_player_id"],
            "source_dataset": "current_transfermarkt",
            "source_record_id": "current_player:" + current["canonical_player_id"],
            "snapshot_as_of_date": "",
            "source_season_reference": current["current_last_season"],
            "source_current_club_id": current["current_current_club_id"].map(text_id),
            "source_current_club_name": current["current_current_club_name"],
            "contract_expiration_date": current["current_contract_expiration_date"],
            "agent_name": current["current_agent_name"],
            "joined_date": "",
            "on_loan_from_club_id": "",
            "snapshot_leakage_risk": True,
        }
    )
    datalake_snapshots = pd.DataFrame(
        {
            "canonical_player_id": datalake["canonical_player_id"],
            "source_dataset": "supplemental_datalake",
            "source_record_id": datalake["datalake_source_record_id"],
            "snapshot_as_of_date": "",
            "source_season_reference": "",
            "source_current_club_id": datalake["datalake_current_club_id"].map(text_id),
            "source_current_club_name": datalake["datalake_current_club_name"],
            "contract_expiration_date": datalake["datalake_contract_expires"],
            "agent_name": datalake["datalake_player_agent_name"],
            "joined_date": datalake["datalake_joined"],
            "on_loan_from_club_id": datalake["datalake_on_loan_from_club_id"].map(text_id),
            "snapshot_leakage_risk": True,
        }
    )
    snapshots = pd.concat([current_snapshots, datalake_snapshots], ignore_index=True)
    return dimension, snapshots


def build_club_dimension():
    dimension = pd.read_csv(XW / "club_identity_crosswalk.csv", dtype="string", low_memory=False)
    dimension["canonical_club_id"] = dimension["canonical_club_id"].map(text_id)
    current = pd.read_csv(TM / "clubs_clean.csv", dtype="string", low_memory=False).drop_duplicates("club_id")
    current["canonical_club_id"] = current["club_id"].map(text_id)
    competitions = pd.read_csv(TM / "competitions_clean.csv", usecols=["competition_id", "country_name"], dtype="string")
    competition_country = dict(zip(competitions["competition_id"], competitions["country_name"]))
    current["current_country_name"] = current["domestic_competition_id"].map(competition_country)
    current = current.rename(columns={column: f"current_{column}" for column in current.columns if column != "canonical_club_id"})

    datalake = pd.read_csv(DL / "team_details_clean.csv", dtype="string", low_memory=False).drop_duplicates("club_id")
    datalake["canonical_club_id"] = datalake["club_id"].map(text_id)
    datalake = datalake.rename(columns={column: f"datalake_{column}" for column in datalake.columns if column != "canonical_club_id"})
    dimension = dimension.merge(current, on="canonical_club_id", how="left").merge(datalake, on="canonical_club_id", how="left")

    fbref = pd.read_csv(XW / "fbref_club_crosswalk.csv", dtype="string")
    fbref = fbref[fbref["match_status"].eq("accepted")]
    fbref_agg = fbref.groupby("canonical_club_id", sort=False).agg(
        fbref_squad_names=("fbref_squad_name", lambda values: " | ".join(sorted(set(values.dropna())))),
        fbref_squad_identity_count=("fbref_squad_name_normalized", "nunique"),
    ).reset_index()
    dimension = dimension.merge(fbref_agg, on="canonical_club_id", how="left")

    dimension["current_club_name"] = dimension["current_name"].fillna("")
    dimension["datalake_club_name"] = dimension["datalake_club_name_display"].fillna("")
    dimension["canonical_club_name"] = choose(dimension["canonical_name"], choose(dimension["datalake_club_name"], dimension["current_club_name"]))
    dimension["canonical_club_name_normalized"] = dimension["canonical_club_name"].map(normalized_text)
    dimension["current_country_name"] = dimension["current_current_country_name"].fillna("")
    dimension["datalake_country_name"] = dimension["datalake_country_name"].fillna("")
    dimension["canonical_country_name"] = choose(dimension["datalake_country_name"], dimension["current_country_name"])
    dimension["country_conflict"] = conflict(dimension["current_country_name"], dimension["datalake_country_name"], normalized_text)
    dimension["identity_record_type"] = np.where(dimension["canonical_club_name"].fillna("").ne(""), "named_identity", "club_id_stub")
    dimension["identity_resolution_status"] = np.select(
        [dimension["identity_record_type"].eq("club_id_stub"), dimension["country_conflict"]],
        ["stub_missing_name", "country_source_conflict"],
        default="resolved_by_shared_transfermarkt_id",
    )
    keep = [
        "canonical_club_id", "canonical_club_key", "identity_record_type", "canonical_club_name",
        "canonical_club_name_normalized", "name_aliases", "canonical_country_name", "current_country_name",
        "datalake_country_name", "country_conflict", "present_in_current_transfermarkt", "present_in_datalake",
        "fbref_squad_names", "fbref_squad_identity_count", "identity_resolution_status",
    ]
    output_dimension = dimension[keep].sort_values("canonical_club_id", key=lambda values: values.map(lambda value: (not str(value).lstrip("-").isdigit(), int(value) if str(value).lstrip("-").isdigit() else str(value))))

    current_snapshots = pd.DataFrame(
        {
            "canonical_club_id": current["canonical_club_id"],
            "source_dataset": "current_transfermarkt",
            "source_record_id": "current_club:" + current["canonical_club_id"],
            "snapshot_as_of_date": "",
            "source_season_reference": current["current_last_season"],
            "source_domestic_competition_id": current["current_domestic_competition_id"],
            "source_club_name": current["current_name"],
            "total_market_value": current["current_total_market_value"],
            "squad_size": current["current_squad_size"],
            "stadium_name": current["current_stadium_name"],
            "coach_name": current["current_coach_name"],
            "snapshot_leakage_risk": True,
        }
    )
    datalake_snapshots = pd.DataFrame(
        {
            "canonical_club_id": datalake["canonical_club_id"],
            "source_dataset": "supplemental_datalake",
            "source_record_id": datalake["datalake_source_record_id"],
            "snapshot_as_of_date": datalake["datalake__last_modified_at"],
            "source_season_reference": datalake["datalake_season_id"],
            "source_domestic_competition_id": datalake["datalake_competition_id"],
            "source_club_name": datalake["datalake_club_name_display"],
            "total_market_value": "",
            "squad_size": "",
            "stadium_name": "",
            "coach_name": "",
            "snapshot_leakage_risk": True,
        }
    )
    snapshots = pd.concat([current_snapshots, datalake_snapshots], ignore_index=True)
    return output_dimension, snapshots


def make_field_dictionary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for table_name, frame in tables.items():
        for order, column in enumerate(frame.columns, start=1):
            if column.startswith("canonical_") and column.endswith("_id"):
                definition = "Canonical integration identifier used for downstream joins."
            elif column.startswith("source_"):
                definition = "Source-specific evidence retained without overwriting other sources."
            elif "conflict" in column:
                definition = "Conflict diagnostic; unresolved conflicts are retained for review."
            elif "status" in column:
                definition = "Documented integration or resolution state."
            elif "count" in column:
                definition = "Row, source, or distinct-value count used for reconciliation."
            elif "present_in" in column:
                definition = "Source-presence flag for coverage auditing."
            else:
                definition = "Canonical integration field; see the table grain and README policy."
            rows.append({"table": table_name, "column_order": order, "column": column, "definition": definition})
    return pd.DataFrame(rows)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print("Integrating valuation history", flush=True)
    valuation_evidence, valuation_history, latest_valuation, valuation_conflicts = build_valuation_tables()
    print("Integrating transfer history", flush=True)
    transfer_evidence, transfer_history, transfer_conflicts = build_transfer_tables()
    print("Building canonical player and club dimensions", flush=True)
    player_dimension, player_snapshots = build_player_dimension(valuation_evidence, transfer_evidence)
    club_dimension, club_snapshots = build_club_dimension()

    outputs = {
        "canonical_player_dimension.csv": player_dimension,
        "player_snapshot_evidence.csv": player_snapshots,
        "canonical_club_dimension.csv": club_dimension,
        "club_snapshot_evidence.csv": club_snapshots,
        "valuation_source_evidence.csv": valuation_evidence,
        "canonical_valuation_history.csv": valuation_history,
        "canonical_latest_valuation.csv": latest_valuation,
        "valuation_conflict_review.csv": valuation_conflicts,
        "transfer_source_evidence.csv": transfer_evidence,
        "canonical_transfer_history.csv": transfer_history,
        "transfer_conflict_review.csv": transfer_conflicts,
    }
    for filename, frame in outputs.items():
        print(f"Writing {filename}: {len(frame):,} rows", flush=True)
        write_csv(frame, filename)

    table_dictionary = pd.DataFrame(
        [
            {"table": "canonical_player_dimension", "grain": "One row per canonical Transfermarkt player ID, including flagged fact-only stubs.", "merge_policy": "Stable identity attributes only; current club, contracts, and agents remain snapshot evidence."},
            {"table": "player_snapshot_evidence", "grain": "One source snapshot per profiled player and source.", "merge_policy": "Never use as historical predictors without an explicit as-of cutoff."},
            {"table": "canonical_club_dimension", "grain": "One row per canonical Transfermarkt club ID.", "merge_policy": "Shared native club ID; country conflicts flagged."},
            {"table": "club_snapshot_evidence", "grain": "One source snapshot per club and source.", "merge_policy": "Snapshot fields remain source-specific and leakage-sensitive."},
            {"table": "valuation_source_evidence", "grain": "One received valuation source row.", "merge_policy": "Row-preserving evidence with canonical player-date key."},
            {"table": "canonical_valuation_history", "grain": "One canonical player-date valuation key.", "merge_policy": "Exact values collapse; conflicting values remain source-specific and canonical value is blank."},
            {"table": "canonical_latest_valuation", "grain": "One maximum-date valuation row per player.", "merge_policy": "Derived from canonical history; not an independent fact source."},
            {"table": "valuation_conflict_review", "grain": "One canonical valuation key requiring conflict or duplicate review.", "merge_policy": "Review queue; conflicting values remain unresolved."},
            {"table": "transfer_source_evidence", "grain": "One received transfer source row.", "merge_policy": "Row-preserving evidence with canonical event key."},
            {"table": "canonical_transfer_history", "grain": "One complete player-date-origin-destination event; incomplete keys remain source-row-specific.", "merge_policy": "Exact/consistent evidence collapses; fee, value, or type conflicts stay unresolved."},
            {"table": "transfer_conflict_review", "grain": "One canonical event requiring review or documenting within-source duplicates.", "merge_policy": "Review queue; never silently overwrite conflicting evidence."},
        ]
    )
    write_csv(table_dictionary, "integration_table_dictionary.csv")
    write_csv(make_field_dictionary({name.replace(".csv", ""): frame for name, frame in outputs.items()}), "integration_field_dictionary.csv")

    summary_rows = [
        {"area": "players", "metric": "canonical_player_ids", "value": len(player_dimension)},
        {"area": "players", "metric": "fact_only_stub_ids", "value": int(player_dimension["identity_record_type"].eq("fact_only_stub").sum())},
        {"area": "players", "metric": "unnamed_existing_identity_ids", "value": int(player_dimension["identity_record_type"].eq("unnamed_identity").sum())},
        {"area": "players", "metric": "stable_attribute_conflicts", "value": int(player_dimension["stable_attribute_conflict"].sum())},
        {"area": "clubs", "metric": "canonical_club_ids", "value": len(club_dimension)},
        {"area": "clubs", "metric": "club_id_stubs", "value": int(club_dimension["identity_record_type"].eq("club_id_stub").sum())},
        {"area": "valuations", "metric": "source_evidence_rows", "value": len(valuation_evidence)},
        {"area": "valuations", "metric": "canonical_player_date_rows", "value": len(valuation_history)},
        {"area": "valuations", "metric": "cross_source_overlap_keys", "value": int(valuation_history["cross_source_overlap"].sum())},
        {"area": "valuations", "metric": "unresolved_value_conflicts", "value": int(valuation_history["value_conflict"].sum())},
        {"area": "valuations", "metric": "latest_player_rows", "value": len(latest_valuation)},
        {"area": "transfers", "metric": "source_evidence_rows", "value": len(transfer_evidence)},
        {"area": "transfers", "metric": "canonical_event_rows", "value": len(transfer_history)},
        {"area": "transfers", "metric": "cross_source_overlap_events", "value": int(transfer_history["cross_source_overlap"].sum())},
        {"area": "transfers", "metric": "unresolved_fee_conflicts", "value": int(transfer_history["transfer_fee_conflict"].sum())},
        {"area": "transfers", "metric": "unresolved_market_value_conflicts", "value": int(transfer_history["market_value_conflict"].sum())},
        {"area": "transfers", "metric": "within_source_duplicate_events", "value": int(transfer_history["within_source_duplicate"].sum())},
        {"area": "transfers", "metric": "incomplete_key_events", "value": int(transfer_history["natural_key_missing"].sum())},
        {"area": "transfers", "metric": "zero_fee_events_unclassified", "value": int(transfer_history["canonical_transfer_fee_status"].eq("unclassified_zero").sum())},
    ]
    summary = pd.DataFrame(summary_rows)
    write_csv(summary, "integration_summary.csv")

    player_ids = set(player_dimension["canonical_player_id"])
    club_ids = set(club_dimension["canonical_club_id"])
    checks = pd.DataFrame(
        [
            {"check": "canonical_player_ids_unique", "passed": player_dimension["canonical_player_id"].is_unique, "severity": "blocking", "observed": player_dimension["canonical_player_id"].nunique(), "expected": len(player_dimension)},
            {"check": "canonical_club_ids_unique", "passed": club_dimension["canonical_club_id"].is_unique, "severity": "blocking", "observed": club_dimension["canonical_club_id"].nunique(), "expected": len(club_dimension)},
            {"check": "valuation_evidence_row_preserved", "passed": len(valuation_evidence) == 507815 + 901429, "severity": "blocking", "observed": len(valuation_evidence), "expected": 507815 + 901429},
            {"check": "valuation_spine_unique", "passed": valuation_history["canonical_valuation_id"].is_unique, "severity": "blocking", "observed": valuation_history["canonical_valuation_id"].nunique(), "expected": len(valuation_history)},
            {"check": "valuation_player_references", "passed": set(valuation_evidence["canonical_player_id"]) - {""} <= player_ids, "severity": "blocking", "observed": len(set(valuation_evidence["canonical_player_id"]) - {""}), "expected": len(set(valuation_evidence["canonical_player_id"]) - {""})},
            {"check": "valuation_conflicts_not_resolved", "passed": valuation_history.loc[valuation_history["value_conflict"], "canonical_market_value_eur"].isna().all(), "severity": "blocking", "observed": int(valuation_history.loc[valuation_history["value_conflict"], "canonical_market_value_eur"].notna().sum()), "expected": 0},
            {"check": "latest_valuation_one_per_player", "passed": latest_valuation["canonical_player_id"].is_unique, "severity": "blocking", "observed": latest_valuation["canonical_player_id"].nunique(), "expected": len(latest_valuation)},
            {"check": "transfer_evidence_row_preserved", "passed": len(transfer_evidence) == 35139 + 1101440, "severity": "blocking", "observed": len(transfer_evidence), "expected": 35139 + 1101440},
            {"check": "transfer_spine_unique", "passed": transfer_history["canonical_transfer_event_id"].is_unique, "severity": "blocking", "observed": transfer_history["canonical_transfer_event_id"].nunique(), "expected": len(transfer_history)},
            {"check": "transfer_player_references", "passed": set(transfer_evidence["canonical_player_id"]) - {""} <= player_ids, "severity": "blocking", "observed": len(set(transfer_evidence["canonical_player_id"]) - {""}), "expected": len(set(transfer_evidence["canonical_player_id"]) - {""})},
            {"check": "transfer_from_club_references", "passed": set(transfer_evidence["from_club_id"]) - {""} <= club_ids, "severity": "blocking", "observed": len(set(transfer_evidence["from_club_id"]) - {""}), "expected": len(set(transfer_evidence["from_club_id"]) - {""})},
            {"check": "transfer_to_club_references", "passed": set(transfer_evidence["to_club_id"]) - {""} <= club_ids, "severity": "blocking", "observed": len(set(transfer_evidence["to_club_id"]) - {""}), "expected": len(set(transfer_evidence["to_club_id"]) - {""})},
            {"check": "transfer_fee_conflicts_not_resolved", "passed": transfer_history.loc[transfer_history["transfer_fee_conflict"], "canonical_transfer_fee_eur"].isna().all(), "severity": "blocking", "observed": int(transfer_history.loc[transfer_history["transfer_fee_conflict"], "canonical_transfer_fee_eur"].notna().sum()), "expected": 0},
            {"check": "zero_fees_not_called_free", "passed": ~transfer_history["canonical_transfer_fee_status"].astype("string").str.contains("free", case=False, na=False).any(), "severity": "blocking", "observed": int(transfer_history["canonical_transfer_fee_status"].astype("string").str.contains("free", case=False, na=False).sum()), "expected": 0},
        ]
    )
    write_csv(checks, "integration_checks.csv")

    input_paths = [
        XW / "player_identity_crosswalk.csv", XW / "club_identity_crosswalk.csv", XW / "fbref_player_crosswalk.csv",
        XW / "fbref_club_crosswalk.csv", TM / "players_clean.csv", TM / "clubs_clean.csv",
        TM / "competitions_clean.csv", TM / "player_valuations_clean.csv", TM / "transfers_clean.csv",
        DL / "player_profiles_clean.csv", DL / "team_details_clean.csv", DL / "player_market_value_clean.csv",
        DL / "transfer_history_clean.csv",
    ]
    manifest = pd.DataFrame(
        [{"source_file": str(path.relative_to(ROOT)).replace("\\", "/"), "file_bytes": path.stat().st_size, "sha256": sha256(path)} for path in input_paths]
    )
    write_csv(manifest, "source_manifest.csv")

    status = "pass" if checks.loc[checks["severity"].eq("blocking"), "passed"].all() else "fail"
    result = {row["metric"] if row["area"] == "players" else f"{row['area']}_{row['metric']}": int(row["value"]) for row in summary_rows}
    result.update({"status": status, "blocking_checks": int(checks["severity"].eq("blocking").sum()), "blocking_failures": int((checks["severity"].eq("blocking") & ~checks["passed"]).sum())})
    (OUTPUT / "integration_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUTPUT / "README.md").write_text(
        f"""# MoveMaker canonical integration

This layer integrates the cleaned current Transfermarkt collection, the supplemental Transfermarkt datalake, and accepted FBref identity crosswalks.

## Policies

- Transfermarkt numeric player and club IDs remain the canonical native identifiers.
- {int(player_dimension['identity_record_type'].eq('fact_only_stub').sum()):,} fact-only player IDs are retained as explicit stubs; no names or biographies are invented.
- Stable player and club attributes use documented source precedence with source-specific conflict flags.
- Current club, contract, agent, roster, and other latest-state fields remain in snapshot evidence tables and are not treated as timeless attributes.
- Valuation evidence is row-preserving. One canonical player-date spine row is created; conflicting values remain unresolved and the canonical value is blank.
- Transfer evidence is row-preserving. One canonical event spine is created for complete natural keys; incomplete keys remain source-row-specific.
- Literal zero fees remain `unclassified_zero`; they are never inferred to be free transfers.
- Player performance, injuries, national performance, teammates, hierarchy, and standings are not merged here.

## Result

- Canonical players: {len(player_dimension):,}
- Canonical clubs: {len(club_dimension):,}
- Valuation evidence: {len(valuation_evidence):,} rows → {len(valuation_history):,} canonical player-date rows
- Transfer evidence: {len(transfer_evidence):,} rows → {len(transfer_history):,} canonical event rows
- Blocking checks: {int(checks['severity'].eq('blocking').sum()):,}; failures: {int((checks['severity'].eq('blocking') & ~checks['passed']).sum()):,}
""",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

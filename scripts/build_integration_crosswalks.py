from __future__ import annotations

import csv
import difflib
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "Data" / "processed"
TM = PROCESSED / "transfermarkt_clean"
DL = PROCESSED / "football_datalake_clean" / "tables"
FBREF = PROCESSED / "fbref_2017_2024_clean" / "fbref_player_squad_season_clean.csv"
OUTPUT = PROCESSED / "integration_crosswalks"
CHUNK = 250_000

SPECIAL_TRANSLITERATION = str.maketrans(
    {"ð": "d", "ł": "l", "ø": "o", "æ": "ae", "œ": "oe", "ß": "ss", "þ": "th", "đ": "d", "ı": "i"}
)


def text_id(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"-?\d+\.0", text):
        text = text[:-2]
    return text


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def enhanced_normalize(value: object) -> str:
    text = clean_text(value).lower().translate(SPECIAL_TRANSLITERATION)
    text = "".join(char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "", text)


def token_signature(value: object) -> str:
    text = clean_text(value).lower().translate(SPECIAL_TRANSLITERATION)
    text = "".join(char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char))
    tokens = sorted(re.findall(r"[a-z0-9]+", text))
    return "|".join(tokens)


def year_text(value: object) -> str:
    match = re.search(r"(18|19|20)\d{2}", clean_text(value))
    return match.group(0) if match else ""


def first_nonblank(*values: object) -> str:
    for value in values:
        text = clean_text(value)
        if text:
            return text
    return ""


def join_sorted(values, separator: str = " | ") -> str:
    return separator.join(sorted({clean_text(value) for value in values if clean_text(value)}))


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


def build_player_identity():
    old_union_path = TM / "player_identity_union.csv"
    old_players_path = TM / "tables" / "players_clean.csv"
    new_profiles_path = DL / "player_profiles_clean.csv"
    old_alias_path = TM / "player_name_observations.csv"

    old_union = pd.read_csv(
        old_union_path,
        usecols=[
            "player_id", "canonical_name", "canonical_name_normalized", "date_of_birth", "birth_year",
            "country_of_birth", "country_of_citizenship", "position", "sub_position",
            "present_in_players_snapshot",
        ],
        dtype="string",
        low_memory=False,
    )
    old_union["player_id"] = old_union["player_id"].map(text_id)
    old_union["birth_year"] = old_union["birth_year"].map(year_text)

    old_players = pd.read_csv(
        old_players_path,
        usecols=["player_id", "name", "date_of_birth", "birth_year", "country_of_citizenship"],
        dtype="string",
        low_memory=False,
    )
    old_players["player_id"] = old_players["player_id"].map(text_id)
    old_players["birth_year"] = old_players["birth_year"].map(year_text)
    old_players = old_players.drop_duplicates("player_id").set_index("player_id")

    new_profiles = pd.read_csv(
        new_profiles_path,
        usecols=[
            "player_id", "player_name_display", "player_name_display_normalized", "date_of_birth",
            "country_of_birth", "citizenship", "position", "main_position",
        ],
        dtype="string",
        low_memory=False,
    )
    new_profiles["player_id"] = new_profiles["player_id"].map(text_id)
    new_profiles["birth_year"] = new_profiles["date_of_birth"].map(year_text)
    new_profiles = new_profiles.drop_duplicates("player_id").set_index("player_id")

    old_union = old_union.drop_duplicates("player_id").set_index("player_id")
    all_ids = sorted(set(old_union.index) | set(new_profiles.index) | set(old_players.index), key=lambda x: (not x.lstrip("-").isdigit(), int(x) if x.lstrip("-").isdigit() else x))

    identity_rows = []
    alias_names: dict[str, set[str]] = defaultdict(set)
    canonical_birth_year: dict[str, str] = {}
    dob_conflict: dict[str, bool] = {}

    old_aliases = pd.read_csv(old_alias_path, usecols=["player_id", "observed_name"], dtype="string")
    for row in old_aliases.itertuples(index=False):
        pid = text_id(row.player_id)
        name = clean_text(row.observed_name)
        if pid and name:
            alias_names[pid].add(name)

    for player_id in all_ids:
        old = old_union.loc[player_id] if player_id in old_union.index else None
        snapshot = old_players.loc[player_id] if player_id in old_players.index else None
        new = new_profiles.loc[player_id] if player_id in new_profiles.index else None

        old_name = clean_text(old.get("canonical_name")) if old is not None else ""
        snapshot_name = clean_text(snapshot.get("name")) if snapshot is not None else ""
        new_name = clean_text(new.get("player_name_display")) if new is not None else ""
        for name in [old_name, snapshot_name, new_name]:
            if name:
                alias_names[player_id].add(name)

        old_dob = first_nonblank(snapshot.get("date_of_birth") if snapshot is not None else "", old.get("date_of_birth") if old is not None else "")
        new_dob = clean_text(new.get("date_of_birth")) if new is not None else ""
        years = {year_text(value) for value in [old_dob, new_dob, snapshot.get("birth_year") if snapshot is not None else "", old.get("birth_year") if old is not None else ""] if year_text(value)}
        birth_year = first_nonblank(year_text(old_dob), year_text(new_dob), *(sorted(years)))
        conflict = len(years) > 1 or bool(old_dob and new_dob and old_dob != new_dob)
        canonical_birth_year[player_id] = birth_year
        dob_conflict[player_id] = conflict

        canonical_name = first_nonblank(snapshot_name, new_name, old_name, *(sorted(alias_names[player_id], key=lambda x: (len(x), x))))
        old_country = clean_text(old.get("country_of_citizenship")) if old is not None else ""
        snapshot_country = clean_text(snapshot.get("country_of_citizenship")) if snapshot is not None else ""
        new_country = clean_text(new.get("citizenship")) if new is not None else ""
        alias_normalized = {enhanced_normalize(name) for name in alias_names[player_id] if enhanced_normalize(name)}
        identity_rows.append(
            {
                "canonical_player_id": player_id,
                "canonical_player_key": f"tm_player:{player_id}",
                "current_transfermarkt_player_id": player_id if player_id in old_union.index or player_id in old_players.index else "",
                "datalake_transfermarkt_player_id": player_id if player_id in new_profiles.index else "",
                "present_in_current_transfermarkt": player_id in old_union.index or player_id in old_players.index,
                "present_in_datalake_profiles": player_id in new_profiles.index,
                "canonical_name": canonical_name,
                "canonical_name_normalized": enhanced_normalize(canonical_name),
                "name_alias_count": len(alias_normalized),
                "name_aliases": join_sorted(alias_names[player_id]),
                "date_of_birth": first_nonblank(old_dob, new_dob),
                "birth_year": birth_year,
                "date_of_birth_conflict": conflict,
                "country_of_citizenship": first_nonblank(snapshot_country, old_country, new_country),
                "position": first_nonblank(old.get("position") if old is not None else "", new.get("position") if new is not None else ""),
                "sub_position": first_nonblank(old.get("sub_position") if old is not None else "", new.get("main_position") if new is not None else ""),
                "identity_rule": "shared Transfermarkt player_id",
            }
        )

    identity = pd.DataFrame(identity_rows)
    return identity, alias_names, canonical_birth_year, dob_conflict


def build_fbref_player_candidates(identity, alias_names, canonical_birth_year, dob_conflict):
    fb = pd.read_csv(FBREF, dtype="string", low_memory=False)
    fb["birth_year"] = fb["birth_year"].map(year_text)
    fb["enhanced_name"] = fb["player_name"].map(enhanced_normalize)
    fb["token_signature"] = fb["player_name"].map(token_signature)

    alias_norms: dict[str, set[str]] = defaultdict(set)
    alias_tokens: dict[str, set[str]] = defaultdict(set)
    exact_lookup: dict[tuple[str, str], set[str]] = defaultdict(set)
    token_lookup: dict[tuple[str, str], set[str]] = defaultdict(set)
    players_by_year: dict[str, set[str]] = defaultdict(set)
    identity_name = identity.set_index("canonical_player_id")["canonical_name"].to_dict()

    for player_id, names in alias_names.items():
        birth_year = canonical_birth_year.get(player_id, "")
        if not birth_year:
            continue
        players_by_year[birth_year].add(player_id)
        for name in names:
            normalized = enhanced_normalize(name)
            signature = token_signature(name)
            if normalized:
                alias_norms[player_id].add(normalized)
                exact_lookup[(normalized, birth_year)].add(player_id)
            if signature:
                alias_tokens[player_id].add(signature)
                token_lookup[(signature, birth_year)].add(player_id)

    grouped = []
    for key, group in fb.groupby("player_identity_key", sort=True, dropna=False):
        first = group.iloc[0]
        grouped.append(
            {
                "fbref_player_identity_key": clean_text(key),
                "fbref_player_name": clean_text(first["player_name"]),
                "fbref_name_normalized": clean_text(first["player_name_normalized"]),
                "fbref_name_enhanced_normalized": clean_text(first["enhanced_name"]),
                "fbref_token_signature": clean_text(first["token_signature"]),
                "birth_year": clean_text(first["birth_year"]),
                "seasonal_rows": len(group),
                "seasons": join_sorted(group["season"]),
                "squads": join_sorted(group["squad_name"]),
                "competitions": join_sorted(group["competition_name"]),
            }
        )
    players = pd.DataFrame(grouped)

    initial = {}
    candidate_scores: dict[str, dict[str, float]] = defaultdict(dict)
    candidate_sources: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in players.itertuples(index=False):
        key = row.fbref_player_identity_key
        exact = sorted(exact_lookup.get((row.fbref_name_enhanced_normalized, row.birth_year), set()))
        token = sorted(token_lookup.get((row.fbref_token_signature, row.birth_year), set()))
        for player_id in exact:
            candidate_scores[key][player_id] = 1.0
            candidate_sources[key][player_id].add("exact_enhanced_name_birth_year")
        for player_id in token:
            candidate_scores[key][player_id] = max(candidate_scores[key].get(player_id, 0.0), 1.0)
            candidate_sources[key][player_id].add("exact_token_set_birth_year")

        accepted_id = ""
        status = "unresolved"
        method = "candidate_review_required"
        if len(exact) == 1 and not dob_conflict.get(exact[0], False):
            accepted_id, status, method = exact[0], "accepted", "exact_enhanced_name_birth_year"
        elif not exact and len(token) == 1 and not dob_conflict.get(token[0], False):
            accepted_id, status, method = token[0], "accepted", "exact_token_set_birth_year"
        elif len(set(exact) | set(token)) > 1:
            status = "ambiguous"

        if not accepted_id:
            for player_id in players_by_year.get(row.birth_year, set()):
                aliases = alias_norms.get(player_id, set())
                score = max(
                    (difflib.SequenceMatcher(None, row.fbref_name_enhanced_normalized, alias).ratio() for alias in aliases),
                    default=0.0,
                )
                if score >= 0.60:
                    candidate_scores[key][player_id] = max(candidate_scores[key].get(player_id, 0.0), score)
                    candidate_sources[key][player_id].add("fuzzy_name_same_birth_year")

        ranked = sorted(candidate_scores[key].items(), key=lambda item: (-item[1], item[0]))[:5]
        initial[key] = {
            "accepted_id": accepted_id,
            "status": status,
            "method": method,
            "ranked": ranked,
            "candidate_ids": sorted(set(exact) | set(token) | {item[0] for item in ranked}),
            "exact_ids": set(exact),
            "token_ids": set(token),
        }

    return fb, players, initial, candidate_scores, candidate_sources, identity_name


def accumulate_club_aliases():
    alias_counts: dict[str, Counter] = defaultdict(Counter)
    source_ids: dict[str, set[str]] = defaultdict(set)
    preferred_old = {}
    preferred_new = {}
    countries = {}

    def add_frame(frame, id_col, name_col, source):
        for club_id, name in zip(frame[id_col], frame[name_col]):
            club_id = text_id(club_id)
            name = clean_text(name)
            if not club_id:
                continue
            source_ids[source].add(club_id)
            if name:
                alias_counts[club_id][name] += 1

    old_dim = pd.read_csv(TM / "club_dimension_historical.csv", dtype="string", low_memory=False)
    add_frame(old_dim, "club_id", "canonical_name", "current_transfermarkt")
    preferred_old = {text_id(row.club_id): clean_text(row.canonical_name) for row in old_dim.itertuples(index=False)}

    old_alias = pd.read_csv(TM / "club_name_observations.csv", usecols=["club_id", "observed_name", "observation_count"], dtype="string")
    for row in old_alias.itertuples(index=False):
        club_id = text_id(row.club_id)
        name = clean_text(row.observed_name)
        if club_id:
            source_ids["current_transfermarkt"].add(club_id)
        if club_id and name:
            count = int(float(clean_text(row.observation_count) or 1))
            alias_counts[club_id][name] += max(count, 1)

    details = pd.read_csv(DL / "team_details_clean.csv", dtype="string", low_memory=False)
    add_frame(details, "club_id", "club_name_display", "datalake")
    preferred_new = {text_id(row.club_id): clean_text(row.club_name_display) for row in details.itertuples(index=False)}
    countries = {text_id(row.club_id): clean_text(row.country_name) for row in details.itertuples(index=False)}

    sources = [
        (DL / "team_competitions_seasons_clean.csv", [("club_id", "team_name")]),
        (DL / "player_performances_clean.csv", [("team_id", "team_name")]),
        (DL / "transfer_history_clean.csv", [("from_team_id", "from_team_name"), ("to_team_id", "to_team_name")]),
        (DL / "team_children_clean.csv", [("parent_team_id", "parent_team_name"), ("child_team_id", "child_team_name")]),
        (DL / "player_profiles_clean.csv", [("current_club_id", "current_club_name"), ("on_loan_from_club_id", "on_loan_from_club_name")]),
    ]
    for path, pairs in sources:
        usecols = sorted({column for pair in pairs for column in pair})
        for chunk in pd.read_csv(path, usecols=usecols, dtype="string", chunksize=CHUNK, low_memory=False):
            for id_col, name_col in pairs:
                add_frame(chunk, id_col, name_col, "datalake")

    all_ids = sorted(set(alias_counts) | source_ids["current_transfermarkt"] | source_ids["datalake"], key=lambda x: (not x.lstrip("-").isdigit(), int(x) if x.lstrip("-").isdigit() else x))
    rows = []
    alias_norm_to_ids: dict[str, set[str]] = defaultdict(set)
    canonical_names = {}
    for club_id in all_ids:
        most_common = alias_counts[club_id].most_common()
        fallback = most_common[0][0] if most_common else ""
        canonical_name = first_nonblank(preferred_new.get(club_id, ""), preferred_old.get(club_id, ""), fallback)
        canonical_names[club_id] = canonical_name
        aliases = sorted(alias_counts[club_id])
        normalized_aliases = {enhanced_normalize(name) for name in aliases if enhanced_normalize(name)}
        for normalized in normalized_aliases:
            alias_norm_to_ids[normalized].add(club_id)
        rows.append(
            {
                "canonical_club_id": club_id,
                "canonical_club_key": f"tm_club:{club_id}",
                "current_transfermarkt_club_id": club_id if club_id in source_ids["current_transfermarkt"] else "",
                "datalake_transfermarkt_club_id": club_id if club_id in source_ids["datalake"] else "",
                "present_in_current_transfermarkt": club_id in source_ids["current_transfermarkt"],
                "present_in_datalake": club_id in source_ids["datalake"],
                "canonical_name": canonical_name,
                "canonical_name_normalized": enhanced_normalize(canonical_name),
                "country_name": countries.get(club_id, ""),
                "normalized_alias_count": len(normalized_aliases),
                "name_aliases": join_sorted(aliases),
                "identity_rule": "shared Transfermarkt club_id",
            }
        )
    return pd.DataFrame(rows), alias_norm_to_ids, canonical_names


def build_membership(candidate_player_ids: set[str]):
    membership: dict[str, set[tuple[str, str, str]]] = defaultdict(set)

    perf_path = DL / "player_performances_clean.csv"
    for chunk in pd.read_csv(
        perf_path,
        usecols=["player_id", "season_start_year", "competition_id", "team_id"],
        dtype="string",
        chunksize=CHUNK,
        low_memory=False,
    ):
        chunk["player_id"] = chunk["player_id"].map(text_id)
        chunk = chunk[chunk["player_id"].isin(candidate_player_ids)]
        for row in chunk.itertuples(index=False):
            season = year_text(row.season_start_year)
            competition = clean_text(row.competition_id)
            club_id = text_id(row.team_id)
            if season and competition and club_id:
                membership[row.player_id].add((season, competition, club_id))

    games = pd.read_csv(
        TM / "tables" / "games_clean.csv",
        usecols=["game_id", "competition_id", "season"],
        dtype="string",
        low_memory=False,
    )
    games["game_id"] = games["game_id"].map(text_id)
    games["season"] = games["season"].map(year_text)
    games = games.drop_duplicates("game_id").set_index("game_id")
    game_comp = games["competition_id"].to_dict()
    game_season = games["season"].to_dict()

    for chunk in pd.read_csv(
        TM / "tables" / "appearances_clean.csv",
        usecols=["game_id", "player_id", "player_club_id"],
        dtype="string",
        chunksize=CHUNK,
        low_memory=False,
    ):
        chunk["player_id"] = chunk["player_id"].map(text_id)
        chunk = chunk[chunk["player_id"].isin(candidate_player_ids)]
        for row in chunk.itertuples(index=False):
            game_id = text_id(row.game_id)
            season = game_season.get(game_id, "")
            competition = clean_text(game_comp.get(game_id, ""))
            club_id = text_id(row.player_club_id)
            if season and competition and club_id:
                membership[row.player_id].add((season, competition, club_id))
    return membership


def resolve_club_seasons(fb, player_links, alias_norm_to_ids, membership):
    rows = []
    for (squad_norm, season, competition_id), group in fb.groupby(
        ["squad_name_normalized", "season", "competition_id"], sort=True, dropna=False
    ):
        first = group.iloc[0]
        squad_name = clean_text(first["squad_name"])
        season_start = year_text(first["season_start_year"] or season)
        competition = clean_text(competition_id)
        name_candidates = sorted(alias_norm_to_ids.get(enhanced_normalize(squad_name), set()))

        roster_counter = Counter()
        accepted_player_ids = set()
        for key in group["player_identity_key"].dropna().unique():
            player_id = player_links.get(clean_text(key), "")
            if not player_id:
                continue
            accepted_player_ids.add(player_id)
            for member_season, member_comp, club_id in membership.get(player_id, set()):
                if member_season == season_start and member_comp == competition:
                    roster_counter[club_id] += 1

        ranked_roster = sorted(roster_counter.items(), key=lambda item: (-item[1], item[0]))
        top_id, top_overlap = ranked_roster[0] if ranked_roster else ("", 0)
        second_overlap = ranked_roster[1][1] if len(ranked_roster) > 1 else 0
        name_overlap = roster_counter.get(name_candidates[0], 0) if len(name_candidates) == 1 else 0
        status, method, accepted_id = "unmatched", "no_safe_match", ""

        if len(name_candidates) == 1:
            if top_id and top_id != name_candidates[0] and top_overlap >= 3 and name_overlap == 0:
                status, method = "review_conflict", "exact_name_conflicts_with_roster"
            else:
                status, method, accepted_id = "accepted", "exact_normalized_club_name", name_candidates[0]
        elif len(name_candidates) > 1:
            eligible = [(club_id, roster_counter.get(club_id, 0)) for club_id in name_candidates]
            eligible.sort(key=lambda item: (-item[1], item[0]))
            if eligible and eligible[0][1] >= 2 and (len(eligible) == 1 or eligible[0][1] > eligible[1][1]):
                status, method, accepted_id = "accepted", "ambiguous_name_resolved_by_roster", eligible[0][0]
            else:
                status, method = "ambiguous", "multiple_exact_name_candidates"
        elif top_overlap >= 3 and top_overlap - second_overlap >= 2 and top_overlap / max(len(accepted_player_ids), 1) >= 0.20:
            status, method, accepted_id = "accepted", "season_competition_roster_evidence", top_id
        elif top_overlap:
            status, method = "review_candidate", "insufficient_roster_evidence"

        rows.append(
            {
                "fbref_club_season_key": f"{squad_norm}|{competition}|{season}",
                "fbref_squad_name": squad_name,
                "fbref_squad_name_normalized": clean_text(squad_norm),
                "fbref_competition_id": competition,
                "fbref_competition_name": clean_text(first["competition_name"]),
                "fbref_season": clean_text(season),
                "season_start_year": season_start,
                "season_end_year": year_text(first["season_end_year"]),
                "fbref_player_rows": len(group),
                "accepted_player_ids_available": len(accepted_player_ids),
                "exact_name_candidate_count": len(name_candidates),
                "exact_name_candidate_ids": join_sorted(name_candidates),
                "top_roster_candidate_id": top_id,
                "top_roster_overlap": top_overlap,
                "second_roster_overlap": second_overlap,
                "canonical_club_id": accepted_id,
                "match_status": status,
                "match_method": method,
                "manual_review_required": status != "accepted",
            }
        )

    result = pd.DataFrame(rows)
    # If a squad name resolves consistently in at least one season, use that stable identity for otherwise unresolved seasons.
    for squad_norm, index in result.groupby("fbref_squad_name_normalized").groups.items():
        group = result.loc[index]
        accepted_ids = set(group.loc[group["match_status"].eq("accepted"), "canonical_club_id"]) - {""}
        if len(accepted_ids) == 1:
            stable_id = next(iter(accepted_ids))
            fill_mask = result.index.isin(index) & result["match_status"].isin(["unmatched", "review_candidate"])
            result.loc[fill_mask, "canonical_club_id"] = stable_id
            result.loc[fill_mask, "match_status"] = "accepted"
            result.loc[fill_mask, "match_method"] = "stable_squad_name_across_seasons"
            result.loc[fill_mask, "manual_review_required"] = False
    return result


def resolve_players(players, initial, candidate_scores, candidate_sources, club_seasons, membership, identity_name):
    club_map = {
        (row.fbref_squad_name_normalized, row.fbref_season, row.fbref_competition_id): row.canonical_club_id
        for row in club_seasons.itertuples(index=False)
        if row.match_status == "accepted" and clean_text(row.canonical_club_id)
    }
    fb_full = pd.read_csv(
        FBREF,
        usecols=["player_identity_key", "squad_name_normalized", "season", "season_start_year", "competition_id"],
        dtype="string",
        low_memory=False,
    )
    fb_groups = {clean_text(key): group for key, group in fb_full.groupby("player_identity_key", dropna=False)}
    output_rows = []
    review_rows = []

    for row in players.itertuples(index=False):
        key = row.fbref_player_identity_key
        state = initial[key]
        evidence = {}
        candidate_ids = state["candidate_ids"]
        for player_id in candidate_ids:
            agreements = 0
            comparable = 0
            for fb_row in fb_groups.get(key, pd.DataFrame()).itertuples(index=False):
                club_id = club_map.get((clean_text(fb_row.squad_name_normalized), clean_text(fb_row.season), clean_text(fb_row.competition_id)), "")
                if not club_id:
                    continue
                comparable += 1
                season_start = year_text(fb_row.season_start_year or fb_row.season)
                if (season_start, clean_text(fb_row.competition_id), club_id) in membership.get(player_id, set()):
                    agreements += 1
            evidence[player_id] = (agreements, comparable)

        accepted_id = state["accepted_id"]
        status = state["status"]
        method = state["method"]
        ranked = sorted(
            candidate_ids,
            key=lambda player_id: (-evidence.get(player_id, (0, 0))[0], -candidate_scores[key].get(player_id, 0.0), player_id),
        )
        top_id = ranked[0] if ranked else ""
        top_agreements = evidence.get(top_id, (0, 0))[0] if top_id else 0
        second_agreements = evidence.get(ranked[1], (0, 0))[0] if len(ranked) > 1 else 0
        score_ranked = sorted([(pid, candidate_scores[key].get(pid, 0.0)) for pid in candidate_ids], key=lambda item: (-item[1], item[0]))
        top_score = score_ranked[0][1] if score_ranked else 0.0
        second_score = score_ranked[1][1] if len(score_ranked) > 1 else 0.0

        if not accepted_id and top_id:
            exact_or_token = top_id in state["exact_ids"] or top_id in state["token_ids"]
            if exact_or_token and top_agreements >= 1 and top_agreements > second_agreements:
                accepted_id, status, method = top_id, "accepted", "exact_name_birth_year_resolved_by_club_season"
            elif top_score >= 0.92 and top_score - second_score >= 0.03 and top_agreements >= 1:
                accepted_id, status, method = top_id, "accepted", "high_similarity_birth_year_club_season"
            elif top_score >= 0.75 and top_agreements >= 2 and top_agreements > second_agreements:
                accepted_id, status, method = top_id, "accepted", "fuzzy_name_birth_year_multiple_club_seasons"
            elif candidate_ids:
                status, method = ("ambiguous", "candidate_scores_or_evidence_tied") if len(candidate_ids) > 1 else ("review_candidate", "insufficient_evidence")
            else:
                status, method = "unmatched", "no_same_birth_year_name_candidate"
        elif not accepted_id:
            status, method = "unmatched", "no_same_birth_year_name_candidate"

        for rank, player_id in enumerate(ranked[:5], start=1):
            agreements, comparable = evidence.get(player_id, (0, 0))
            review_rows.append(
                {
                    "fbref_player_identity_key": key,
                    "fbref_player_name": row.fbref_player_name,
                    "birth_year": row.birth_year,
                    "candidate_rank": rank,
                    "candidate_player_id": player_id,
                    "candidate_player_name": identity_name.get(player_id, ""),
                    "name_similarity": round(candidate_scores[key].get(player_id, 0.0), 6),
                    "candidate_sources": join_sorted(candidate_sources[key].get(player_id, set())),
                    "club_season_agreements": agreements,
                    "club_seasons_comparable": comparable,
                    "selected": player_id == accepted_id,
                    "final_status": status,
                    "manual_review_required": status != "accepted",
                }
            )

        output_rows.append(
            {
                **row._asdict(),
                "canonical_player_id": accepted_id,
                "canonical_player_name": identity_name.get(accepted_id, "") if accepted_id else "",
                "match_status": status,
                "match_method": method,
                "candidate_count": len(candidate_ids),
                "candidate_player_ids": join_sorted(candidate_ids),
                "top_name_similarity": round(top_score, 6),
                "second_name_similarity": round(second_score, 6),
                "club_season_agreements": evidence.get(accepted_id, (0, 0))[0] if accepted_id else top_agreements,
                "club_seasons_comparable": evidence.get(accepted_id, (0, 0))[1] if accepted_id else (evidence.get(top_id, (0, 0))[1] if top_id else 0),
                "manual_review_required": status != "accepted",
            }
        )
    return pd.DataFrame(output_rows), pd.DataFrame(review_rows)


def build_stable_club_crosswalk(club_seasons, canonical_names):
    rows = []
    for squad_norm, group in club_seasons.groupby("fbref_squad_name_normalized", sort=True):
        accepted = group[group["match_status"].eq("accepted")]
        accepted_ids = sorted(set(accepted["canonical_club_id"]) - {""})
        conflicts = group["match_status"].isin(["review_conflict", "ambiguous"]).sum()
        if len(accepted_ids) == 1 and conflicts == 0:
            status = "accepted"
            club_id = accepted_ids[0]
        elif len(accepted_ids) > 1 or conflicts:
            status = "ambiguous"
            club_id = ""
        else:
            status = "unmatched"
            club_id = ""
        candidate_ids = {text_id(value) for value in group["canonical_club_id"] if text_id(value)}
        for value in group["exact_name_candidate_ids"]:
            candidate_ids.update(text_id(item) for item in clean_text(value).split(" | ") if text_id(item))
        rows.append(
            {
                "fbref_squad_name": join_sorted(group["fbref_squad_name"]),
                "fbref_squad_name_normalized": squad_norm,
                "canonical_club_id": club_id,
                "canonical_club_name": canonical_names.get(club_id, ""),
                "match_status": status,
                "season_rows": len(group),
                "accepted_season_rows": int(group["match_status"].eq("accepted").sum()),
                "candidate_club_ids": join_sorted(candidate_ids),
                "seasons": join_sorted(group["fbref_season"]),
                "competition_ids": join_sorted(group["fbref_competition_id"]),
                "match_methods": join_sorted(group["match_method"]),
                "manual_review_required": status != "accepted",
            }
        )
    return pd.DataFrame(rows)


def build_competition_crosswalk():
    observations: dict[str, Counter] = defaultdict(Counter)
    flags: dict[str, set[str]] = defaultdict(set)
    metadata = {}

    current = pd.read_csv(TM / "tables" / "competitions_clean.csv", dtype="string", low_memory=False)
    for row in current.itertuples(index=False):
        competition_id = clean_text(row.competition_id)
        if not competition_id:
            continue
        observations[competition_id][clean_text(row.name)] += 1
        flags[competition_id].add("current_transfermarkt")
        metadata[competition_id] = {
            "country_name": clean_text(row.country_name),
            "competition_type": clean_text(row.type),
            "competition_sub_type": clean_text(row.sub_type),
        }

    fb = pd.read_csv(FBREF, usecols=["competition_id", "competition_name"], dtype="string")
    for row in fb.drop_duplicates().itertuples(index=False):
        competition_id = clean_text(row.competition_id)
        if competition_id:
            observations[competition_id][clean_text(row.competition_name)] += 1
            flags[competition_id].add("fbref")

    datalake_sources = [
        (DL / "player_performances_clean.csv", "competition_id", "competition_name"),
        (DL / "team_competitions_seasons_clean.csv", "competition_id", "competition_name"),
        (DL / "team_details_clean.csv", "competition_id", "competition_name"),
    ]
    for path, id_col, name_col in datalake_sources:
        for chunk in pd.read_csv(path, usecols=[id_col, name_col], dtype="string", chunksize=CHUNK, low_memory=False):
            for row in chunk.drop_duplicates().itertuples(index=False, name=None):
                competition_id, name = clean_text(row[0]), clean_text(row[1])
                if competition_id:
                    observations[competition_id][name] += 1
                    flags[competition_id].add("datalake")

    rows = []
    fb_names = fb.drop_duplicates("competition_id").set_index("competition_id")["competition_name"].to_dict()
    current_names = current.drop_duplicates("competition_id").set_index("competition_id")["name"].to_dict()
    for competition_id in sorted(observations):
        aliases = [name for name in observations[competition_id] if name]
        canonical_name = first_nonblank(fb_names.get(competition_id, ""), current_names.get(competition_id, ""), observations[competition_id].most_common(1)[0][0] if observations[competition_id] else "")
        meta = metadata.get(competition_id, {})
        rows.append(
            {
                "canonical_competition_id": competition_id,
                "canonical_competition_key": f"tm_competition:{competition_id}",
                "current_transfermarkt_competition_id": competition_id if "current_transfermarkt" in flags[competition_id] else "",
                "datalake_competition_id": competition_id if "datalake" in flags[competition_id] else "",
                "fbref_competition_id": competition_id if "fbref" in flags[competition_id] else "",
                "present_in_current_transfermarkt": "current_transfermarkt" in flags[competition_id],
                "present_in_datalake": "datalake" in flags[competition_id],
                "present_in_fbref": "fbref" in flags[competition_id],
                "canonical_name": canonical_name,
                "competition_name_aliases": join_sorted(aliases),
                "country_name": meta.get("country_name", ""),
                "competition_type": meta.get("competition_type", ""),
                "competition_sub_type": meta.get("competition_sub_type", ""),
                "match_status": "accepted",
                "match_method": "shared Transfermarkt competition code",
            }
        )
    return pd.DataFrame(rows)


def season_from_years(start, end=""):
    start_year = year_text(start)
    end_year = year_text(end)
    if start_year and not end_year:
        end_year = str(int(start_year) + 1)
    return start_year, end_year, f"{start_year}-{end_year}" if start_year and end_year else ""


def build_season_crosswalk():
    rows = []

    def add(source_system, source_table, source_field, raw_label, start, end, method):
        raw = text_id(raw_label)
        start_year, end_year, canonical = season_from_years(start, end)
        rows.append(
            {
                "source_system": source_system,
                "source_table": source_table,
                "source_field": source_field,
                "source_season_value": raw,
                "canonical_season_key": canonical,
                "season_start_year": start_year,
                "season_end_year": end_year,
                "match_status": "accepted" if canonical else "unmatched",
                "match_method": method if canonical else "unparseable_season",
            }
        )

    fb = pd.read_csv(FBREF, usecols=["season", "season_start_year", "season_end_year"], dtype="string").drop_duplicates()
    for row in fb.itertuples(index=False):
        add("fbref", "fbref_player_squad_season_clean", "season", row.season, row.season_start_year, row.season_end_year, "supplied_start_and_end_year")

    games = pd.read_csv(TM / "tables" / "games_clean.csv", usecols=["season"], dtype="string").drop_duplicates()
    for value in games["season"]:
        add("current_transfermarkt", "games_clean", "season", value, value, "", "numeric_start_year")

    datalake_specs = [
        ("player_performances_clean.csv", "season_name", "season_start_year", "season_end_year"),
        ("transfer_history_clean.csv", "season_name", "season_start_year", "season_end_year"),
        ("player_injuries_clean.csv", "season_name", "season_start_year", "season_end_year"),
        ("team_competitions_seasons_clean.csv", "season_season", "season_start_year", "season_end_year"),
        ("team_competitions_seasons_clean.csv", "season_id", "season_start_year", "season_end_year"),
    ]
    for filename, raw_col, start_col, end_col in datalake_specs:
        observed = set()
        for chunk in pd.read_csv(DL / filename, usecols=[raw_col, start_col, end_col], dtype="string", chunksize=CHUNK, low_memory=False):
            observed.update(tuple(clean_text(value) for value in row) for row in chunk.drop_duplicates().itertuples(index=False, name=None))
        for raw, start, end in sorted(observed):
            add("datalake", filename.replace(".csv", ""), raw_col, raw, start, end, "clean_layer_parsed_years")

    result = pd.DataFrame(rows).drop_duplicates().sort_values(["source_system", "source_table", "source_field", "season_start_year", "source_season_value"])
    return result


def add_dictionary_rows():
    descriptions = {
        "player_identity_crosswalk.csv": "One row per native Transfermarkt player ID across current and datalake collections.",
        "fbref_player_crosswalk.csv": "One row per FBref name-plus-birth-year identity mapped to a canonical Transfermarkt player ID when evidence is sufficient.",
        "fbref_player_candidate_review.csv": "Ranked candidate evidence for player identities, including accepted and unresolved cases.",
        "club_identity_crosswalk.csv": "One row per native Transfermarkt club ID across all observed current and datalake tables.",
        "fbref_club_season_crosswalk.csv": "Season-aware mapping from an FBref squad to a canonical Transfermarkt club ID.",
        "fbref_club_crosswalk.csv": "Stable roll-up of the season-aware FBref squad mapping.",
        "competition_crosswalk.csv": "Competition codes reconciled across current Transfermarkt, datalake Transfermarkt, and FBref.",
        "season_crosswalk.csv": "Every observed season representation mapped to a canonical start/end season key.",
    }
    return pd.DataFrame([{"file": filename, "grain": grain} for filename, grain in descriptions.items()])


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("Building native player identities", flush=True)
    player_identity, player_aliases, player_years, dob_conflict = build_player_identity()
    fb, player_groups, initial, candidate_scores, candidate_sources, player_names = build_fbref_player_candidates(
        player_identity, player_aliases, player_years, dob_conflict
    )

    print("Building native club identities and aliases", flush=True)
    club_identity, club_alias_lookup, club_names = accumulate_club_aliases()

    candidate_ids = {state["accepted_id"] for state in initial.values() if state["accepted_id"]}
    for state in initial.values():
        candidate_ids.update(state["candidate_ids"])
    print(f"Building season membership evidence for {len(candidate_ids):,} candidate players", flush=True)
    membership = build_membership(candidate_ids)

    initial_player_links = {key: state["accepted_id"] for key, state in initial.items() if state["accepted_id"]}
    club_seasons = resolve_club_seasons(fb, initial_player_links, club_alias_lookup, membership)
    player_crosswalk, player_review = resolve_players(
        player_groups, initial, candidate_scores, candidate_sources, club_seasons, membership, player_names
    )
    final_player_links = dict(
        player_crosswalk.loc[player_crosswalk["match_status"].eq("accepted"), ["fbref_player_identity_key", "canonical_player_id"]].itertuples(index=False, name=None)
    )
    club_seasons = resolve_club_seasons(fb, final_player_links, club_alias_lookup, membership)
    player_crosswalk, player_review = resolve_players(
        player_groups, initial, candidate_scores, candidate_sources, club_seasons, membership, player_names
    )
    final_player_links = dict(
        player_crosswalk.loc[player_crosswalk["match_status"].eq("accepted"), ["fbref_player_identity_key", "canonical_player_id"]].itertuples(index=False, name=None)
    )
    club_seasons = resolve_club_seasons(fb, final_player_links, club_alias_lookup, membership)
    club_crosswalk = build_stable_club_crosswalk(club_seasons, club_names)

    print("Building competition and season crosswalks", flush=True)
    competition_crosswalk = build_competition_crosswalk()
    season_crosswalk = build_season_crosswalk()

    player_identity = player_identity.sort_values("canonical_player_id", key=lambda series: series.map(lambda x: (not str(x).lstrip("-").isdigit(), int(x) if str(x).lstrip("-").isdigit() else str(x))))
    player_crosswalk = player_crosswalk.sort_values(["match_status", "fbref_player_name", "birth_year"])
    player_review = player_review.sort_values(["manual_review_required", "fbref_player_name", "candidate_rank"], ascending=[False, True, True])
    club_identity = club_identity.sort_values("canonical_club_id", key=lambda series: series.map(lambda x: (not str(x).lstrip("-").isdigit(), int(x) if str(x).lstrip("-").isdigit() else str(x))))
    club_seasons = club_seasons.merge(
        club_identity[["canonical_club_id", "canonical_name"]].rename(columns={"canonical_name": "canonical_club_name"}),
        on="canonical_club_id",
        how="left",
    )

    outputs = {
        "player_identity_crosswalk.csv": player_identity,
        "fbref_player_crosswalk.csv": player_crosswalk,
        "fbref_player_candidate_review.csv": player_review,
        "club_identity_crosswalk.csv": club_identity,
        "fbref_club_season_crosswalk.csv": club_seasons,
        "fbref_club_crosswalk.csv": club_crosswalk,
        "competition_crosswalk.csv": competition_crosswalk,
        "season_crosswalk.csv": season_crosswalk,
    }
    for filename, frame in outputs.items():
        write_csv(frame, filename)

    summary_rows = []
    summary_rows.extend(
        [
            {"crosswalk": "native_player_identity", "metric": "canonical_ids", "value": len(player_identity)},
            {"crosswalk": "native_player_identity", "metric": "shared_current_and_datalake", "value": int((player_identity["present_in_current_transfermarkt"] & player_identity["present_in_datalake_profiles"]).sum())},
            {"crosswalk": "fbref_player", "metric": "identities", "value": len(player_crosswalk)},
            {"crosswalk": "fbref_player", "metric": "accepted", "value": int(player_crosswalk["match_status"].eq("accepted").sum())},
            {"crosswalk": "fbref_player", "metric": "manual_review", "value": int(player_crosswalk["manual_review_required"].sum())},
            {"crosswalk": "native_club_identity", "metric": "canonical_ids", "value": len(club_identity)},
            {"crosswalk": "fbref_club_season", "metric": "rows", "value": len(club_seasons)},
            {"crosswalk": "fbref_club_season", "metric": "accepted", "value": int(club_seasons["match_status"].eq("accepted").sum())},
            {"crosswalk": "fbref_club", "metric": "squad_names", "value": len(club_crosswalk)},
            {"crosswalk": "fbref_club", "metric": "accepted", "value": int(club_crosswalk["match_status"].eq("accepted").sum())},
            {"crosswalk": "competition", "metric": "canonical_ids", "value": len(competition_crosswalk)},
            {"crosswalk": "competition", "metric": "fbref_codes", "value": int(competition_crosswalk["present_in_fbref"].sum())},
            {"crosswalk": "season", "metric": "source_representations", "value": len(season_crosswalk)},
            {"crosswalk": "season", "metric": "unmatched", "value": int(season_crosswalk["match_status"].ne("accepted").sum())},
        ]
    )
    summary = pd.DataFrame(summary_rows)
    write_csv(summary, "crosswalk_summary.csv")

    canonical_player_ids = set(player_identity["canonical_player_id"].map(text_id))
    canonical_club_ids = set(club_identity["canonical_club_id"].map(text_id))
    fb_comp_ids = set(fb["competition_id"].dropna().map(clean_text))
    comp_ids = set(competition_crosswalk["canonical_competition_id"].map(clean_text))
    checks = pd.DataFrame(
        [
            {"check": "native_player_ids_unique", "passed": player_identity["canonical_player_id"].is_unique, "severity": "blocking", "observed": len(player_identity), "expected": player_identity["canonical_player_id"].nunique()},
            {"check": "fbref_player_keys_unique", "passed": player_crosswalk["fbref_player_identity_key"].is_unique, "severity": "blocking", "observed": len(player_crosswalk), "expected": player_crosswalk["fbref_player_identity_key"].nunique()},
            {"check": "accepted_player_ids_exist", "passed": set(player_crosswalk.loc[player_crosswalk["match_status"].eq("accepted"), "canonical_player_id"]).issubset(canonical_player_ids), "severity": "blocking", "observed": int(player_crosswalk["match_status"].eq("accepted").sum()), "expected": int(player_crosswalk["match_status"].eq("accepted").sum())},
            {"check": "native_club_ids_unique", "passed": club_identity["canonical_club_id"].is_unique, "severity": "blocking", "observed": len(club_identity), "expected": club_identity["canonical_club_id"].nunique()},
            {"check": "fbref_club_season_keys_unique", "passed": club_seasons["fbref_club_season_key"].is_unique, "severity": "blocking", "observed": len(club_seasons), "expected": club_seasons["fbref_club_season_key"].nunique()},
            {"check": "accepted_club_ids_exist", "passed": set(club_seasons.loc[club_seasons["match_status"].eq("accepted"), "canonical_club_id"]).issubset(canonical_club_ids), "severity": "blocking", "observed": int(club_seasons["match_status"].eq("accepted").sum()), "expected": int(club_seasons["match_status"].eq("accepted").sum())},
            {"check": "fbref_competitions_covered", "passed": fb_comp_ids.issubset(comp_ids), "severity": "blocking", "observed": len(fb_comp_ids & comp_ids), "expected": len(fb_comp_ids)},
            {"check": "fbref_seasons_parse", "passed": season_crosswalk.loc[season_crosswalk["source_system"].eq("fbref"), "match_status"].eq("accepted").all(), "severity": "blocking", "observed": int(season_crosswalk.loc[season_crosswalk["source_system"].eq("fbref"), "match_status"].eq("accepted").sum()), "expected": int(season_crosswalk["source_system"].eq("fbref").sum())},
            {"check": "unresolved_players_retained_for_review", "passed": True, "severity": "control", "observed": int(player_crosswalk["manual_review_required"].sum()), "expected": int(player_crosswalk["manual_review_required"].sum())},
            {"check": "unresolved_clubs_retained_for_review", "passed": True, "severity": "control", "observed": int(club_seasons["manual_review_required"].sum()), "expected": int(club_seasons["manual_review_required"].sum())},
        ]
    )
    write_csv(checks, "crosswalk_checks.csv")
    write_csv(add_dictionary_rows(), "crosswalk_dictionary.csv")

    input_paths = [
        TM / "player_identity_union.csv", TM / "player_name_observations.csv", TM / "club_dimension_historical.csv",
        TM / "club_name_observations.csv", TM / "tables" / "players_clean.csv", TM / "tables" / "appearances_clean.csv",
        TM / "tables" / "games_clean.csv", TM / "tables" / "competitions_clean.csv", FBREF,
        DL / "player_profiles_clean.csv", DL / "player_performances_clean.csv", DL / "team_details_clean.csv",
        DL / "team_competitions_seasons_clean.csv", DL / "transfer_history_clean.csv", DL / "team_children_clean.csv",
        DL / "player_injuries_clean.csv",
    ]
    manifest = pd.DataFrame(
        [{"source_file": str(path.relative_to(ROOT)).replace("\\", "/"), "file_bytes": path.stat().st_size, "sha256": sha256(path)} for path in input_paths]
    )
    write_csv(manifest, "source_manifest.csv")

    readme = f"""# MoveMaker integration crosswalks

Generated from the cleaned FBref, current Transfermarkt, and supplemental Transfermarkt datalake layers.

## Identity policy

- Native Transfermarkt player, club, and competition IDs are authoritative across the two Transfermarkt collections.
- FBref players are auto-accepted only with deterministic name-plus-birth-year evidence or strong season/club evidence.
- FBref clubs are resolved at squad-competition-season grain first, then rolled up to a stable squad mapping.
- Ambiguous and unmatched records remain in the outputs with `manual_review_required = True`.
- Crosswalks do not overwrite or deduplicate source facts.

## Coverage

- Player identities: {len(player_identity):,} canonical Transfermarkt IDs.
- FBref player identities accepted: {int(player_crosswalk['match_status'].eq('accepted').sum()):,} of {len(player_crosswalk):,}.
- Club identities: {len(club_identity):,} canonical Transfermarkt IDs.
- FBref club-season mappings accepted: {int(club_seasons['match_status'].eq('accepted').sum()):,} of {len(club_seasons):,}.
- Competitions: {len(competition_crosswalk):,} canonical codes; all {len(fb_comp_ids):,} FBref codes covered.
- Season representations: {len(season_crosswalk):,}.

See `crosswalk_dictionary.csv`, `crosswalk_checks.csv`, and `crosswalk_summary.csv` for the audit trail.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")

    result = {
        "status": "pass" if checks.loc[checks["severity"].eq("blocking"), "passed"].all() else "fail",
        "player_identity_rows": len(player_identity),
        "fbref_player_rows": len(player_crosswalk),
        "fbref_players_accepted": int(player_crosswalk["match_status"].eq("accepted").sum()),
        "fbref_players_manual_review": int(player_crosswalk["manual_review_required"].sum()),
        "club_identity_rows": len(club_identity),
        "fbref_club_season_rows": len(club_seasons),
        "fbref_club_seasons_accepted": int(club_seasons["match_status"].eq("accepted").sum()),
        "fbref_club_rows": len(club_crosswalk),
        "fbref_clubs_accepted": int(club_crosswalk["match_status"].eq("accepted").sum()),
        "competition_rows": len(competition_crosswalk),
        "season_rows": len(season_crosswalk),
        "blocking_failures": int((checks["severity"].eq("blocking") & ~checks["passed"]).sum()),
    }
    (OUTPUT / "crosswalk_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

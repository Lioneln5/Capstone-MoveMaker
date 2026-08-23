"""Player search / autocomplete for the web app front end.

Nobody using the app should ever type or see a canonical_player_id. This
module is the missing piece that turns a free-text name search into a
resolved (canonical_player_id, canonical_club_id, competition_id) triple --
with the club and league auto-derived from the player's most recent
Transfermarkt appearance, never freely chosen by the user. That's a
deliberate scope guardrail, not just convenience: letting a user pick an
arbitrary club for a player is exactly the new-club-acquisition misuse that
the product's scope boundary exists to prevent. The engine's own incumbency
check (see canonical_feature_retriever.py) is a second, independent guard
behind this one, not a replacement for it.

"Current club" is deliberately NOT read from canonical_player_dimension's
snapshot fields (those are explicitly documented elsewhere as a leakage-risk
current/latest snapshot, not something usable for arbitrary historical
scoring) -- it's derived fresh from the most recent dated appearance, which
is safe because this module is only ever used for right-now, present-day
search, never for reconstructing a player's club as of some past date.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("MOVEMAKER_DATA_ROOT", ROOT / "Data")).expanduser().resolve()
PROCESSED = DATA_ROOT / "processed"
PLAYER_DIM_PATH = PROCESSED / "canonical_integration" / "canonical_player_dimension.csv"
CLUB_DIM_PATH = PROCESSED / "canonical_integration" / "canonical_club_dimension.csv"
APPEARANCE_PATH = PROCESSED / "transfermarkt_clean" / "tables" / "appearances_clean.csv"
GAMES_PATH = PROCESSED / "transfermarkt_clean" / "tables" / "games_clean.csv"


@dataclass
class PlayerSearchResult:
    canonical_player_id: int
    display_name: str
    player_name_normalized: str
    date_of_birth: str | None
    position: str | None
    current_club_id: int | None
    current_club_name: str | None
    competition_id: str | None
    last_appearance_date: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "canonical_player_id": self.canonical_player_id,
            "display_name": self.display_name,
            "player_name_normalized": self.player_name_normalized,
            "date_of_birth": self.date_of_birth,
            "position": self.position,
            "current_club_id": self.current_club_id,
            "current_club_name": self.current_club_name,
            "competition_id": self.competition_id,
            "last_appearance_date": self.last_appearance_date,
        }


class PlayerSearchIndex:
    """Loads and caches the (small, name-only) lookup tables once per
    process. The appearances scan for "most recent club" is done lazily,
    once, and cached -- NOT repeated per search query, since query volume
    from a search box is much higher than from the scoring endpoint."""

    def __init__(self) -> None:
        players = pd.read_csv(
            PLAYER_DIM_PATH,
            usecols=["canonical_player_id", "canonical_name", "canonical_name_normalized", "name_aliases", "canonical_date_of_birth", "canonical_position"],
            low_memory=False,
        )
        players["canonical_player_id"] = pd.to_numeric(players["canonical_player_id"], errors="coerce").astype("Int64")
        players["canonical_date_of_birth"] = pd.to_datetime(players["canonical_date_of_birth"], errors="coerce")
        players["search_blob"] = (
            players["canonical_name"].astype("string").fillna("")
            + " | " + players["canonical_name_normalized"].astype("string").fillna("")
            + " | " + players["name_aliases"].astype("string").fillna("")
        ).str.casefold()
        self._players = players.dropna(subset=["canonical_player_id"])

        clubs = pd.read_csv(CLUB_DIM_PATH, usecols=["canonical_club_id", "canonical_club_name"], low_memory=False)
        clubs["canonical_club_id"] = pd.to_numeric(clubs["canonical_club_id"], errors="coerce").astype("Int64")
        self._clubs = clubs.dropna(subset=["canonical_club_id"]).set_index("canonical_club_id")["canonical_club_name"]

        self._most_recent_club: pd.DataFrame | None = None  # lazy

    def warm_up(self) -> None:
        """Eagerly build the most-recent-club lookup (a groupby over the
        full appearances table) so the first real search doesn't pay that
        cost -- call once at process startup alongside CanonicalFeature
        Retriever.warm_up()."""
        self._most_recent_club_lookup()

    def _most_recent_club_lookup(self) -> pd.DataFrame:
        """Most recent CLUB appearance -- national-team caps must be excluded
        or "current club" resolves to a country, not an employer. This
        mirrors canonical_feature_retriever.py's _load_games() filter
        exactly (join to games_clean.csv's is_national_team_game), rather
        than a competition_id heuristic, which would need to enumerate every
        international competition code and silently miss ones it doesn't
        know about."""
        if self._most_recent_club is None:
            apps = pd.read_csv(
                APPEARANCE_PATH, usecols=["game_id", "player_id", "player_club_id", "date", "competition_id"], low_memory=False,
            )
            apps["player_id"] = pd.to_numeric(apps["player_id"], errors="coerce").astype("Int64")
            apps["player_club_id"] = pd.to_numeric(apps["player_club_id"], errors="coerce").astype("Int64")
            apps["date"] = pd.to_datetime(apps["date"], errors="coerce")
            apps = apps.dropna(subset=["player_id", "player_club_id", "date"])

            games = pd.read_csv(GAMES_PATH, usecols=["game_id", "is_national_team_game"], low_memory=False)
            is_national = games["is_national_team_game"].astype("string").fillna("").str.strip().str.casefold().isin({"true", "1", "yes"})
            national_game_ids = set(games.loc[is_national, "game_id"])
            apps = apps.loc[~apps["game_id"].isin(national_game_ids)]

            latest_idx = apps.groupby("player_id")["date"].idxmax()
            self._most_recent_club = apps.loc[latest_idx].set_index("player_id")
        return self._most_recent_club

    def search(self, query: str, limit: int = 10) -> list[PlayerSearchResult]:
        query = query.strip().casefold()
        if len(query) < 2:
            return []
        matches = self._players.loc[self._players["search_blob"].str.contains(query, regex=False, na=False)]
        if matches.empty:
            return []
        # Prefer names that START with the query (typeahead relevance),
        # then fall back to substring matches elsewhere in the name/aliases.
        starts_with = matches["canonical_name_normalized"].astype("string").fillna("").str.startswith(query.replace(" ", ""))
        matches = pd.concat([matches.loc[starts_with], matches.loc[~starts_with]]).head(limit * 3)  # overfetch; some may lack any appearance

        recent_club = self._most_recent_club_lookup()
        results: list[PlayerSearchResult] = []
        for row in matches.itertuples(index=False):
            pid = int(row.canonical_player_id)
            club_row = recent_club.loc[pid] if pid in recent_club.index else None
            club_id = int(club_row["player_club_id"]) if club_row is not None else None
            results.append(PlayerSearchResult(
                canonical_player_id=pid,
                display_name=row.canonical_name,
                player_name_normalized=row.canonical_name_normalized,
                date_of_birth=row.canonical_date_of_birth.date().isoformat() if pd.notna(row.canonical_date_of_birth) else None,
                position=row.canonical_position if pd.notna(row.canonical_position) else None,
                current_club_id=club_id,
                current_club_name=self._clubs.get(club_id) if club_id is not None else None,
                competition_id=club_row["competition_id"] if club_row is not None else None,
                last_appearance_date=club_row["date"].date().isoformat() if club_row is not None else None,
            ))
            if len(results) >= limit:
                break
        return results


if __name__ == "__main__":
    import json
    import sys

    index = PlayerSearchIndex()
    query = sys.argv[1] if len(sys.argv) > 1 else "bellingham"
    print(json.dumps([r.as_dict() for r in index.search(query)], indent=2, default=str))

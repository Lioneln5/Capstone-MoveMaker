"""HTTP API for the MoveMaker incumbent-club extension profile.

Stack-agnostic on purpose: we didn't know today whether the front end's
backend would end up Python or something else, so this stands alone as a
plain HTTP service any frontend language can call, rather than assuming a
same-language integration. If the frontend backend does turn out to be
Python, it can still import models/incumbent_extension_profile.py directly
instead of going over HTTP -- this doesn't preclude that, it just doesn't
require it.

The root URL serves the browser tool, with two scoring endpoints matching the
proposed input flow exactly:
  GET  /players/search?q=...   -- typeahead; returns players with their
                                   CURRENT club/league already resolved.
                                   The frontend must never let a user pick
                                   an arbitrary club -- only ever one of
                                   these resolved (player, club, league)
                                   triples. That's a scope guardrail, not
                                   just convenience.
  POST /profile                -- score a proposed extension for one of
                                   those resolved triples.

Run with:
    python3 -m uvicorn api.main:app --reload --port 8000
from the repository root.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
HTML_DIR = ROOT / "html"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from incumbent_extension_profile import IncumbentExtensionProfileBuilder  # noqa: E402
from player_search import PlayerSearchIndex  # noqa: E402

app = FastAPI(title="MoveMaker Incumbent-Extension API", version="1.0.0")
app.mount("/assets", StaticFiles(directory=HTML_DIR / "assets"), name="assets")

# Permissive for local development while the front end is being wired up
# today. Tighten to the deployed frontend's real origin before shipping.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["GET", "POST"], allow_headers=["*"],
)

# Constructed once at process start -- both load and cache several CSVs;
# must not be rebuilt per-request.
_search_index: PlayerSearchIndex | None = None
_profile_builder: IncumbentExtensionProfileBuilder | None = None


@app.on_event("startup")
def _load_engine() -> None:
    import time
    global _search_index, _profile_builder
    t0 = time.time()
    print("MoveMaker API: loading engine and warming caches...", flush=True)
    _search_index = PlayerSearchIndex()
    _profile_builder = IncumbentExtensionProfileBuilder()
    # Eagerly parse every source table now, at boot, not on whichever
    # request happens to arrive first. Measured cost is ~44s (pandas CSV
    # parsing, dominated by the ~1.9M-row appearances table); every request
    # after that is ~0.08-0.15s. Without this, the first real user pays the
    # 44s tax instead of the deployment process.
    _profile_builder.scorer.retriever.warm_up()
    _search_index.warm_up()
    print(f"MoveMaker API: ready in {time.time() - t0:.1f}s. Requests should now take well under a second.", flush=True)


@app.get("/", include_in_schema=False)
def tool_page() -> FileResponse:
    """Serve the actual MoveMaker interface at the same address as the API."""
    return FileResponse(HTML_DIR / "index.html")


@app.get("/{page_name}.html", include_in_schema=False)
def supporting_page(page_name: str) -> FileResponse:
    allowed = {"index", "about", "how-it-works", "scope", "contact"}
    if page_name not in allowed:
        raise HTTPException(status_code=404, detail="Page not found")
    return FileResponse(HTML_DIR / f"{page_name}.html")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "engine_loaded": _profile_builder is not None}


@app.get("/players/search")
def search_players(q: str, limit: int = 10) -> list[dict[str, Any]]:
    if _search_index is None:
        raise HTTPException(status_code=503, detail="Search index not loaded yet.")
    if len(q.strip()) < 2:
        return []
    return [r.as_dict() for r in _search_index.search(q, limit=min(limit, 25))]


@app.get("/players/{canonical_player_id}/input-defaults")
def player_input_defaults(
    canonical_player_id: int,
    canonical_club_id: int,
    player_name_normalized: str | None = None,
) -> dict[str, Any]:
    if _profile_builder is None:
        raise HTTPException(status_code=503, detail="Engine not loaded yet.")
    return _profile_builder.scorer.retriever.get_player_input_defaults(
        canonical_player_id=canonical_player_id,
        canonical_club_id=canonical_club_id,
        decision_date=date.today().isoformat(),
        player_name_normalized=player_name_normalized,
    )


class ProfileRequest(BaseModel):
    canonical_player_id: int
    canonical_club_id: int
    competition_id: str
    proposed_annual_fixed_wage_eur: float = Field(gt=0)
    proposed_contract_years: float = Field(gt=0, le=10.5)
    current_public_market_value_eur: float | None = Field(default=None, gt=0)
    player_display_name: str | None = None
    player_name_normalized: str | None = None  # from the /players/search result; enables prior-wage/salary-panel matching
    current_club_display_name: str | None = None
    # Deliberately no proposed_transfer_fee_eur in the MVP request shape --
    # see the product-scope discussion: exposing acquisition-fee arithmetic
    # in the extension-mode UI risks conflating the two documented scopes
    # keeps explicitly separate. Add back only as a clearly separate,
    # clearly labeled "acquisition-only" field if that's ever wanted.


@app.post("/profile")
def build_profile(request: ProfileRequest) -> dict[str, Any]:
    if _profile_builder is None:
        raise HTTPException(status_code=503, detail="Engine not loaded yet.")
    try:
        return _profile_builder.build_profile(
            canonical_player_id=request.canonical_player_id,
            canonical_club_id=request.canonical_club_id,
            competition_id=request.competition_id,
            decision_date=date.today().isoformat(),
            proposed_annual_fixed_wage_eur=request.proposed_annual_fixed_wage_eur,
            proposed_contract_years=request.proposed_contract_years,
            current_public_market_value_eur=request.current_public_market_value_eur,
            player_display_name=request.player_display_name,
            player_name_normalized=request.player_name_normalized,
            current_club_display_name=request.current_club_display_name,
        )
    except ValueError as exc:
        # Guarded-input validation errors from business_metric_engine.py
        # (e.g. an internally inconsistent payload) surface as 422s, not 500s.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

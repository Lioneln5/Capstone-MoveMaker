"""Fail-closed HTTP service for the MoveMaker research record.

The root URL serves the explanatory site and ``/health`` reports the current
product status. Frozen V1 request schemas and endpoint paths are retained only
for interface lineage; every scoring route refuses before loading data or
models. No environment variable can enable scoring.

Run with:
    python3 -m uvicorn api.main:app --reload --port 8000
from the repository root.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Final

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
HTML_DIR = ROOT / "html"
HISTORICAL_PROFILES = (
    ROOT
    / "Data"
    / "processed"
    / "business_metric_product"
    / "historical_example_profiles.json"
)

app = FastAPI(title="MoveMaker Incumbent-Extension API", version="1.0.0")
app.mount("/assets", StaticFiles(directory=HTML_DIR / "assets"), name="assets")


# V1 remains available as frozen code and artifacts for offline regression
# verification, but it is no longer an HTTP scoring product.  There is
# deliberately no environment-variable override: Engine V2 retired generic
# continuity, and no V2 output has completed final temporal validation.
SCORING_ENABLED: Final[bool] = False
SCORING_PAUSED_REASON = (
    "Engine V1 is frozen for historical audit. Engine V2 retired generic "
    "continuity and has not authorized any live scoring output. The project "
    "pages and verified research evidence remain available."
)
SCORING_PAUSED_DETAIL = f"Live scoring is unavailable. {SCORING_PAUSED_REASON}"

# The service exposes static research pages, saved historical exhibit data,
# status, and refusing scoring routes. The exhibit is a serialized record;
# serving it never loads or invokes a model.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["GET", "POST"], allow_headers=["*"],
)

# Explicit ``None`` sentinels let health and verification prove that frozen
# scoring components were never loaded.
_search_index: Any | None = None
_profile_builder: Any | None = None


@app.on_event("startup")
def _load_engine() -> None:
    print(
        "MoveMaker API: research-record mode; HTTP scoring is disabled and "
        "frozen V1 artifacts will not be loaded.",
        flush=True,
    )


@app.get("/", include_in_schema=False)
def case_study_page() -> FileResponse:
    """Serve the MoveMaker case-study landing page."""
    return FileResponse(HTML_DIR / "index.html")


@app.get("/{page_name}.html", include_in_schema=False)
def supporting_page(page_name: str) -> FileResponse:
    allowed = {"index", "prototype", "about", "how-it-works", "scope", "contact"}
    if page_name not in allowed:
        raise HTTPException(status_code=404, detail="Page not found")
    return FileResponse(HTML_DIR / f"{page_name}.html")


@app.get("/historical-example-profiles.json", include_in_schema=False)
def historical_example_profiles() -> FileResponse:
    """Serve frozen historical exhibits without invoking a scoring engine."""
    if not HISTORICAL_PROFILES.is_file():
        raise HTTPException(status_code=404, detail="Historical exhibits unavailable")
    return FileResponse(HISTORICAL_PROFILES, media_type="application/json")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "product_mode": "research_prototype",
        "scoring_enabled": SCORING_ENABLED,
        "engine_loaded": _profile_builder is not None,
        "engine_version": "v1_frozen" if SCORING_ENABLED else None,
        "message": None if SCORING_ENABLED else SCORING_PAUSED_REASON,
    }


def _require_scoring_enabled() -> None:
    if not SCORING_ENABLED:
        raise HTTPException(status_code=503, detail=SCORING_PAUSED_DETAIL)


@app.get("/players/search")
def search_players(q: str, limit: int = 10) -> list[dict[str, Any]]:
    _require_scoring_enabled()
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
    _require_scoring_enabled()
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
    _require_scoring_enabled()
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

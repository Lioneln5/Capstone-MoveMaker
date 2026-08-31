#!/usr/bin/env python3
"""Keep Railway healthy while its first runtime-data volume is populated.

An empty Railway volume creates a first-deployment dependency cycle: the API
cannot start without its large serving tables, while Railway's volume file CLI
requires an active deployment. This launcher serves a minimal /health endpoint
until every deployment-parity table exists, then replaces itself with uvicorn.
On normal deployments where the volume is already populated, it immediately
starts the real API.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


DATA_ROOT = Path(os.environ.get("MOVEMAKER_DATA_ROOT", "Data")).expanduser().resolve()
PORT = int(os.environ.get("PORT", "8000"))


def environment_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


SCORING_ENABLED = environment_flag("MOVEMAKER_ENABLE_SCORING", default=False)

# The two live-input overlays are technically optional fallbacks in the
# retriever, but required here so the public deployment matches the current
# local demo rather than starting halfway through the initial upload.
RUNTIME_FILES = (
    "processed/canonical_integration/canonical_player_dimension.csv",
    "processed/canonical_integration/canonical_club_dimension.csv",
    "processed/canonical_integration/canonical_valuation_history.csv",
    "processed/transfermarkt_clean/tables/appearances_clean.csv",
    "processed/transfermarkt_clean/tables/games_clean.csv",
    "processed/canonical_performance/canonical_player_team_season_performance.csv",
    "processed/canonical_performance/fbref_advanced_performance_features.csv",
    "processed/capology_contracts/canonical_salary_panel.csv",
    "processed/contract_extension_integration/extension_modeling_master.csv",
    "processed/live_input_refresh/canonical_performance_overlay.csv",
    "processed/live_input_refresh/canonical_salary_overlay.csv",
)


def missing_files() -> list[str]:
    return [relative for relative in RUNTIME_FILES if not (DATA_ROOT / relative).is_file()]


class BootstrapHealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        missing = missing_files()
        status = 200 if self.path == "/health" else 503
        payload = json.dumps(
            {
                "status": "awaiting_runtime_data",
                "engine_loaded": False,
                "missing_file_count": len(missing),
            }
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


def start_api() -> None:
    argv = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
    ]
    print("MoveMaker bootstrap: runtime data complete; starting API.", flush=True)
    os.execvpe(sys.executable, argv, os.environ.copy())


def main() -> None:
    # A paused research-prototype deployment must not wait for the large
    # serving tables. The real API can serve the site and a healthy, explicit
    # scoring-disabled state without mounting or parsing any runtime data.
    if not SCORING_ENABLED:
        print(
            "MoveMaker bootstrap: live scoring disabled; starting the "
            "research-prototype site without runtime data.",
            flush=True,
        )
        start_api()
        return

    missing = missing_files()
    if not missing:
        start_api()
        return

    print(
        f"MoveMaker bootstrap: waiting for {len(missing)} runtime files under {DATA_ROOT}.",
        flush=True,
    )
    server = ThreadingHTTPServer(("0.0.0.0", PORT), BootstrapHealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        while missing_files():
            time.sleep(5)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    start_api()


if __name__ == "__main__":
    main()

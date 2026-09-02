#!/usr/bin/env python3
"""Start the fail-closed MoveMaker research site on Railway.

Generic continuity is retired and no Engine V2 output is authorized for live
scoring, so the hosted service deliberately starts without runtime data.
Frozen V1 artifacts remain available to offline verifiers only.
"""

from __future__ import annotations

import os
import sys


PORT = int(os.environ.get("PORT", "8000"))


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
    print(
        "MoveMaker bootstrap: starting the fail-closed research site without runtime data.",
        flush=True,
    )
    start_api()


if __name__ == "__main__":
    main()

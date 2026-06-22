#!/usr/bin/env python3
"""
serve_site.py — minimal static web server for local preview of site/.

Why this exists: the Claude Code desktop preview window can't render plain
local HTML files past a certain complexity (no real origin, no proper MIME /
relative-URL resolution). Serving site/ over real HTTP fixes that. This server
is for LOCAL use only; GitHub Pages hosts the same site/ directory with its own
static stack (see .github/workflows/deploy-pages.yml).

Serves on 0.0.0.0 so the preview window (and other devices on the LAN) can reach
it. Sends no-cache headers so edits show up on refresh. stdlib only.
Anchored to the repo root so it runs from any CWD (Rule 1).

Usage:
    python utils/python/serve_site.py [--port 8000] [--dir site] [--host 0.0.0.0]
    PORT=9000 python utils/python/serve_site.py
"""
from __future__ import annotations

import argparse
import contextlib
import os
import socket
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPO_ROOT / "site"


class NoCacheHandler(SimpleHTTPRequestHandler):
    """Static handler that disables caching (so preview reflects live edits)."""

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        # concise one-line access log
        print(f"  {self.address_string()} - {fmt % args}")


def _lan_ip() -> str | None:
    with contextlib.suppress(OSError):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        with contextlib.closing(s):
            s.connect(("8.8.8.8", 80))  # no packets sent; just picks the route
            return s.getsockname()[0]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Local static server for site/ preview.")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    ap.add_argument("--dir", default=str(DEFAULT_DIR),
                    help="directory to serve (default: <repo>/site)")
    args = ap.parse_args()

    serve_dir = Path(args.dir).resolve()
    if not serve_dir.is_dir():
        print(f"!! not a directory: {serve_dir}")
        return 1

    handler = partial(NoCacheHandler, directory=str(serve_dir))
    httpd = ThreadingHTTPServer((args.host, args.port), handler)

    lan = _lan_ip()
    print(f"Serving {serve_dir}")
    print(f"  local:   http://localhost:{args.port}/")
    if lan:
        print(f"  network: http://{lan}:{args.port}/   (use this in the preview window)")
    print(f"  bind:    {args.host}:{args.port}   (Ctrl+C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

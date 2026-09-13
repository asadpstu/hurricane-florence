#!/usr/bin/env python3
"""Serve the generated Hurricane Florence dashboard V2 with a local HTTP server."""
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve output/frontend_v2 locally.")
    p.add_argument("--root", type=Path, default=Path("output/frontend_v2"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8008)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    root = a.root.resolve()
    if not (root / "dashboard/index.html").exists():
        raise FileNotFoundError(
            f"Dashboard not built: {root / 'dashboard/index.html'}\n"
            "Run src/frontend/build_dashboard_v2.py first."
        )
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    server = ThreadingHTTPServer((a.host, a.port), handler)
    print("=" * 90)
    print("HURRICANE FLORENCE DASHBOARD V2")
    print("=" * 90)
    print(f"Serving: {root}")
    print(f"Open   : http://{a.host}:{a.port}/dashboard/")
    print("Stop   : Ctrl-C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

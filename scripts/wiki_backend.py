"""A tiny Flask backend that mocks Serper's scrape API from the local cache.

At startup it loads every page cached by ``cache_web_links.py`` into memory
(~90 MB for the FRAMES corpus), keyed by URL. The ``/`` endpoint mirrors
Serper's scrape API — POST ``{"url": ...}`` returns ``{"text": ...}`` — so
fetch_url can read pages from disk for free instead of paying per call.

Nothing is intercepted automatically: a script opts in by pointing fetch_url's
scrape backend at this server. ``tools.py`` exposes the address as a constant
(``LOCAL_SCRAPE_URL``, which matches the default host/port below) and threads it
through ``build_tools``:

    from tools import build_tools, LOCAL_SCRAPE_URL
    fetch = build_tools(["fetch_url"], scrape_url=LOCAL_SCRAPE_URL)[0]["fetch_url"]

A cache miss returns 404 with an empty body, which fetch_url maps to its usual
"no content" result. The lookup is an in-memory dict, so each request is O(1).

Run:
    python3 scripts/wiki_backend.py            # serves on 127.0.0.1:5000
    python3 scripts/wiki_backend.py --port 8000

If you change the host/port, update LOCAL_SCRAPE_URL in tools.py to match.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flask import Flask, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _normalize(url: str) -> str:
    """Canonical key for lookups: trimmed, scheme- and host-insensitive enough to
    survive http/https and the mobile (en.m.) Wikipedia host."""
    url = url.strip().rstrip("/")
    url = url.replace("https://", "").replace("http://", "")
    url = url.replace("en.m.wikipedia.org", "en.wikipedia.org")
    return url


def load_documents(cache_dir: Path) -> dict[str, str]:
    """Load every cached page into a {normalized_url: text} dict."""
    docs: dict[str, str] = {}
    for path in cache_dir.glob("*.json"):
        rec = json.loads(path.read_text())
        url, text = rec.get("url"), rec.get("text")
        if url and text:
            docs[_normalize(url)] = text
    return docs


def create_app(cache_dir: Path) -> Flask:
    app = Flask(__name__)
    docs = load_documents(cache_dir)
    print(f"loaded {len(docs)} documents from {cache_dir}", file=sys.stderr)

    @app.post("/")
    def scrape():
        """Mock of https://scrape.serper.dev — body {"url": ...} -> {"text": ...}."""
        body = request.get_json(silent=True) or {}
        url = body.get("url", "")
        text = docs.get(_normalize(url))
        if text is None:
            return jsonify({"text": "", "error": "url not in cache", "url": url}), 404
        return jsonify({"text": text})

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "documents": len(docs)})

    return app


def main(args: argparse.Namespace) -> None:
    app = create_app(args.cache_dir)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, default=Path("Wikipedia_cache"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    main(ap.parse_args())

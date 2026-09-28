"""Pre-download every Wikipedia page referenced by the dataset to disk.

Later experiments fetch the ground-truth documents over and over via Serper's
scrape API, which costs money on every run. This script does that fetch ONCE:
it collects every unique URL across all questions in the dataset and saves each
page's full text under ``Wikipedia_cache/``. Experiments can then read the text
straight from disk (see ``cached_text`` below) instead of hitting the API.

The cache is keyed by a hash of the URL, so lookup is a pure function of the URL
with no manifest needed. Re-running is safe and cheap: URLs already on disk are
skipped, so an interrupted run just resumes.

    export SERPER_API_KEY=...
    python3 scripts/cache_web_links.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import build_tools

try:  # load API keys from the project .env if python-dotenv is available
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


def cache_path(url: str, cache_dir: Path) -> Path:
    """Deterministic on-disk location for a URL's cached text.

    Experiments can import this to read the cache: ``cache_path(url, dir)`` gives
    the same JSON file this script writes. The file is ``{"url", "text"}``.
    """
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{digest}.json"


def cached_text(url: str, cache_dir: Path) -> str | None:
    """Return a URL's cached page text, or None if it was never cached."""
    path = cache_path(url, cache_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text())["text"]


def split_links(entry: str) -> list[str]:
    """A wiki_links element is usually one URL, but a few entries pack several
    URLs into one comma-joined string. Split only on a comma that is followed by
    http(s):// — so genuine path commas ('Tulsa,_Oklahoma') stay intact."""
    parts = re.split(r"\s*,\s*(?=https?://)", entry.strip())
    return [p.strip().rstrip(",").strip() for p in parts if p.strip().rstrip(",").strip()]


def collect_urls(dataset: Path) -> list[str]:
    """Every unique wiki_link across all questions, in first-seen order."""
    rows = json.loads(dataset.read_text())
    seen: dict[str, None] = {}
    for row in rows:
        for entry in row.get("wiki_links", []):
            for url in split_links(entry):
                seen.setdefault(url, None)
    return list(seen)


def download_url(url: str, fetch, cache_dir: Path) -> tuple[str, str]:
    """Fetch one URL's full text and write it to the cache. Returns (url, status)."""
    path = cache_path(url, cache_dir)
    if path.exists():
        return url, "cached"
    try:
        text = fetch(url)
    except Exception as exc:
        return url, f"error: {exc}"
    if not text or text.startswith(("[fetch_url", "No readable content")):
        return url, "empty"
    path.write_text(json.dumps({"url": url, "text": text}, ensure_ascii=False))
    return url, "ok"


def main(args: argparse.Namespace) -> None:
    urls = collect_urls(args.dataset)
    print(f"{len(urls)} unique URLs in {args.dataset}")

    fetch = build_tools(["fetch_url"])[0]["fetch_url"]  # full-text page fetch
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_url, url, fetch, args.cache_dir) for url in urls]
        for future in tqdm(as_completed(futures), total=len(futures), desc="urls", unit="url"):
            url, status = future.result()
            key = status.split(":", 1)[0]
            counts[key] = counts.get(key, 0) + 1
            if key not in ("ok", "cached"):
                tqdm.write(f"  {status}  {url}")

    print(f"\ndone -> {args.cache_dir}")
    print("breakdown:", counts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path,
                    default=Path("datasets/frames_guaranteed_answerable.json"))
    ap.add_argument("--cache-dir", type=Path, default=Path("Wikipedia_cache"))
    ap.add_argument("--workers", type=int, default=5, help="URLs fetched concurrently.")
    main(ap.parse_args())

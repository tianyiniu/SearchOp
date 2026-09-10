"""(1) Precompute and cache the document context for every qwen_unanswerable question.

For each question in frames_qwen_unanswerable.json (the debate step's input) we
fetch its ground-truth wiki_links, pack them into one structured context, and —
when the documents exceed Qwen's budget — compress them with query-aware
iterative summarization using GPT-5.4 (the strongest available summarizer, so the
compression is not the bottleneck). Each compressed context is then verified: GPT
is asked to answer the question from the compressed context alone, and if it
cannot, the context is rebuilt once at a larger per-document budget. Contexts that
still fail are cached with status 'verify_failed' and excluded downstream.

The packed context is cached under doc_context_cache/, keyed by question id +
summarizer config, so the debate step reuses the EXACT same context instead of
re-summarizing (which is non-deterministic). Point --dataset at another split to
cache it too.

Documents are fetched from the local Wikipedia cache server (scripts/wiki_backend.py)
by default, so this is free; pass --scrape-url to hit Serper instead.

Prereqs:
  - vLLM serving Qwen/Qwen3-14B on port 7472.
  - The local Wikipedia cache server running and populated by cache_web_links.py:

      export SERPER_API_KEY=...
      python3 scripts/cache_web_links.py          # one-time: download gold pages
      python3 scripts/wiki_backend.py &           # serve them on 127.0.0.1:5000
      python3 scripts/cache_doc_summary.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from doc_qa import SummarizerConfig, build_and_cache_context
from tools import LOCAL_SCRAPE_URL, build_tools

try:  # load API keys from the project .env if python-dotenv is available
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


def main(args: argparse.Namespace) -> None:
    rows = json.loads(Path(args.dataset).read_text())
    if args.limit is not None:
        rows = rows[: args.limit]

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    fetch = build_tools(["fetch_url"], scrape_url=args.scrape_url)[0]["fetch_url"]
    cfg = SummarizerConfig(
        model=args.model,
        summarizer_kind=args.summarizer,
        summarizer_model=args.summarizer_model,
        context_window=args.context_window,
        summary_tokens=args.summary_tokens,
        temperature=args.temperature,
        verify_compression=not args.no_verify,
    )
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"summarizer config: {cfg}\n-> cache key signature: {cfg.signature()}")

    counts: Counter = Counter()
    n_summarized = n_retried = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(build_and_cache_context, client, row, fetch, cfg,
                               args.cache_dir, None, args.overwrite)
                   for row in rows]
        for future in tqdm(as_completed(futures), total=len(futures), desc="questions", unit="q"):
            rec = future.result()
            counts[rec["status"]] += 1
            if rec.get("summarized"):
                n_summarized += 1
            if (rec.get("verify_attempts") or 0) > 1:
                n_retried += 1
            if rec["status"] in ("fetch_failed", "verify_failed"):
                tqdm.write(f"  {rec['status']}: {rec.get('id')}")

    print("\nstatus breakdown:", dict(counts))
    print(f"summarized (over budget): {n_summarized} / {len(rows)}")
    print(f"re-compressed after a failed verification: {n_retried}")
    print(f"cache -> {args.cache_dir}  |  config signature: {cfg.signature()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path("datasets/frames_qwen_unanswerable.json"))
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--base-url", default="http://localhost:7472/v1", help="Local vLLM endpoint.")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--scrape-url", default=LOCAL_SCRAPE_URL,
                    help="fetch_url backend: local Wikipedia cache server (default) or Serper scrape URL.")
    ap.add_argument("--context-window", type=int, default=32768)
    ap.add_argument("--summary-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.3, help="Summarizer temperature.")
    ap.add_argument("--summarizer", choices=["gpt", "qwen"], default="gpt",
                    help="Who compresses over-budget documents (default GPT, for quality).")
    ap.add_argument("--summarizer-model", default="gpt-5.4",
                    help="Model used when --summarizer gpt.")
    ap.add_argument("--no-verify", action="store_true",
                    help="Skip the post-compression check that the answer survived.")
    ap.add_argument("--workers", type=int, default=5, help="Questions processed concurrently.")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N questions.")
    ap.add_argument("--overwrite", action="store_true", help="Re-summarize even if already cached.")
    ap.add_argument("--cache-dir", type=Path, default=Path("doc_context_cache"),
                    help="Where packed contexts are cached.")
    main(ap.parse_args())

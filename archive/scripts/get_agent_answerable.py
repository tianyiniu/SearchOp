"""Step 2: split 'guaranteed_answerable' into qwen_answerable / qwen_unanswerable.

Given the FRAMES guaranteed_answerable split (Step 1), hand Qwen3-14B the ground-
truth documents directly — no web search — and see whether it can answer. This
isolates the *aggregation* step from retrieval: every question here is known to be
answerable from its documents, so a failure is Qwen failing to synthesize them.

Labels come from k independent samples at temperature > 0, so a label is a claim
about robustness rather than one lucky or unlucky decode:

  - qwen_answerable   : Qwen answers correctly in ALL k samples.
  - qwen_unanswerable : Qwen fails in ALL k samples -> the place to explore
                        aggregation/debate schemas in later work.
  - qwen_borderline   : sometimes correct. Excluded from both splits (noise-
                        sensitive), written out so downstream can inspect it.

Context management. Documents are fetched, packed, and — when they exceed Qwen's
budget — compressed with query-aware iterative summarization by GPT-5.4 (the
strongest available summarizer, so the compression itself is not the bottleneck).
Every compressed context is then VERIFIED: GPT is asked to answer the question
from the compressed context alone, and if it cannot, the compression dropped the
load-bearing facts. Those questions are re-compressed once at a larger budget and,
if they still fail, excluded entirely — a question whose context no longer
contains its answer cannot measure aggregation. Contexts are cached under
doc_context_cache/ keyed by question id + summarizer config, so every downstream
script evaluates on the EXACT same context this script labelled against.

    export OPENAI_API_KEY=...   # GPT-5.4 summarizer + verifier, gpt-5.4-mini judge
    python3 scripts/wiki_backend.py &         # local doc cache (see cache_web_links.py)
    # vLLM serving Qwen/Qwen3-14B on port 7472
    python3 scripts/get_agent_answerable.py --limit 50
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from doc_qa import (SummarizerConfig, answer_with_docs, build_and_cache_context,
                    usable_context)
from llm_judge import judge_answer
from tools import LOCAL_SCRAPE_URL, build_tools

try:  # load API keys from the project .env if python-dotenv is available
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


def process_question(row: dict, client, fetch, args: argparse.Namespace,
                     cfg: SummarizerConfig) -> dict:
    """Build the canonical context for one question, then label it over k samples."""
    question, ground_truth = row["question"], row["ground_truth"]
    rec = {
        "id": row.get("id"),
        "question": question,
        "ground_truth": ground_truth,
        "wiki_links": row.get("wiki_links", []),
        "k": args.k,
    }
    try:
        ctx = build_and_cache_context(client, row, fetch, cfg, args.cache_dir)
        rec.update({"summarized": ctx.get("summarized"), "num_docs": ctx.get("num_docs"),
                    "doc_tokens": ctx.get("doc_tokens"), "verified": ctx.get("verified"),
                    "verify_attempts": ctx.get("verify_attempts"),
                    "from_cache": ctx.get("status") == "cached",
                    "context_status": ctx.get("cache_status") or ctx.get("status")})
        context = usable_context(ctx)
        if not context:
            # Either nothing fetched, or the answer did not survive compression.
            rec["status"] = ("skipped_verify_failed"
                             if rec["context_status"] == "verify_failed"
                             else "skipped_fetch_failed")
            rec["correct_count"] = 0
            return rec

        # k independent samples at temperature > 0: 'fails all k' is then a claim
        # that sampling cannot rescue the question, not an artifact of one decode.
        answers, correct = [], 0
        for _ in range(args.k):
            answer = answer_with_docs(client, args.model, question, context,
                                      max_tokens=args.answer_tokens,
                                      temperature=args.temperature)
            answers.append(answer)
            if judge_answer(question, ground_truth, answer):
                correct += 1
        rec["answers"] = answers
        rec["correct_count"] = correct
        rec["status"] = ("qwen_answerable" if correct == args.k
                         else "qwen_unanswerable" if correct == 0
                         else "qwen_borderline")
    except Exception as exc:  # keep the pool going; record what failed
        rec["status"] = "error"
        rec["error"] = str(exc)
    return rec


def _split_row(rec: dict) -> dict:
    """The fields downstream scripts need, plus the label provenance."""
    return {"id": rec["id"], "question": rec["question"], "ground_truth": rec["ground_truth"],
            "wiki_links": rec["wiki_links"], "correct_count": rec["correct_count"],
            "k": rec["k"], "summarized": rec.get("summarized"),
            "verified": rec.get("verified")}


def main(args: argparse.Namespace) -> None:
    rows = json.loads(Path(args.dataset).read_text())
    if args.limit is not None:
        rows = rows[: args.limit]

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    fetch = build_tools(["fetch_url"], scrape_url=args.scrape_url)[0]["fetch_url"]
    cfg = SummarizerConfig(model=args.model, summarizer_kind=args.summarizer,
                           summarizer_model=args.summarizer_model,
                           context_window=args.context_window,
                           summary_tokens=args.summary_tokens,
                           verify_compression=not args.no_verify)

    # A --limit run is a smoke test, not a split. Redirect its outputs so it can
    # never overwrite the canonical splits with a partial labelling (the doc_context
    # cache is still shared and still warmed, which is the point of a smoke test).
    if args.limit is not None and not args.write_partial:
        for attr in ("cache", "answerable_out", "unanswerable_out", "borderline_out"):
            path = getattr(args, attr)
            setattr(args, attr, path.with_name(f"{path.stem}_limit{args.limit}{path.suffix}"))
        print(f"--limit {args.limit}: writing to *_limit{args.limit}.* "
              f"(pass --write-partial to overwrite the real splits)")

    for path in (args.cache, args.answerable_out, args.unanswerable_out, args.borderline_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    print(f"summarizer: {args.summarizer} ({args.summarizer_model}), verify="
          f"{not args.no_verify}\ncache key signature: {cfg.signature()} "
          f"(must match the doc_context cache)")

    buckets: dict[str, list] = {"qwen_answerable": [], "qwen_unanswerable": [],
                                "qwen_borderline": []}
    counts: Counter = Counter()
    # Context-management diagnostics: how much of the corpus actually needs
    # compressing, and whether the answer survived it when it did.
    ctx_stats: Counter = Counter()
    doc_tokens: list[int] = []
    lock = Lock()
    with open(args.cache, "w") as cache, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_question, row, client, fetch, args, cfg) for row in rows]
        for future in tqdm(as_completed(futures), total=len(futures), desc="questions", unit="q"):
            rec = future.result()
            with lock:
                cache.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cache.flush()
            counts[rec["status"]] += 1
            if rec.get("doc_tokens"):
                doc_tokens.append(rec["doc_tokens"])
            if rec.get("summarized") is True:
                ctx_stats["compressed"] += 1
                if rec.get("verified") is True:
                    ctx_stats["verify_passed"] += 1
                elif rec.get("verified") is False:
                    ctx_stats["verify_failed"] += 1
                if (rec.get("verify_attempts") or 0) > 1:
                    ctx_stats["verify_retried"] += 1
            elif rec.get("summarized") is False:
                ctx_stats["verbatim"] += 1
            if rec.get("from_cache"):
                ctx_stats["served_from_cache"] += 1
            if rec["status"] in buckets:
                buckets[rec["status"]].append(_split_row(rec))

    args.answerable_out.write_text(json.dumps(buckets["qwen_answerable"], indent=2, ensure_ascii=False))
    args.unanswerable_out.write_text(json.dumps(buckets["qwen_unanswerable"], indent=2, ensure_ascii=False))
    args.borderline_out.write_text(json.dumps(buckets["qwen_borderline"], indent=2, ensure_ascii=False))
    # --- context management diagnostics ---
    budget = int(args.context_window * cfg.doc_budget_fraction)
    n_ctx = ctx_stats["compressed"] + ctx_stats["verbatim"]
    print(f"\n--- context management (doc budget {budget} tok "
          f"= {args.context_window} x {cfg.doc_budget_fraction}) ---")
    if n_ctx:
        pct = 100 * ctx_stats["compressed"] / n_ctx
        print(f"  needed compression : {ctx_stats['compressed']}/{n_ctx} ({pct:.0f}%)")
        print(f"  fit verbatim       : {ctx_stats['verbatim']}/{n_ctx}")
    if doc_tokens:
        doc_tokens.sort()
        m = len(doc_tokens)
        print(f"  raw doc tokens     : min={doc_tokens[0]}  median={doc_tokens[m // 2]}  "
              f"max={doc_tokens[-1]}  (over budget above {budget})")
    if ctx_stats["compressed"]:
        print(f"  verify passed      : {ctx_stats['verify_passed']}/{ctx_stats['compressed']}")
        print(f"  verify needed retry: {ctx_stats['verify_retried']}")
        print(f"  verify FAILED      : {ctx_stats['verify_failed']}  (excluded from all splits)")
    if ctx_stats["served_from_cache"]:
        print(f"  served from cache  : {ctx_stats['served_from_cache']} (no recompression)")

    print(f"\nlabels (k={args.k}, temp={args.temperature}, of {len(rows)}):", dict(counts))
    print(f"qwen_answerable   ({len(buckets['qwen_answerable'])}, {args.k}/{args.k} correct) -> {args.answerable_out}")
    print(f"qwen_unanswerable ({len(buckets['qwen_unanswerable'])}, 0/{args.k} correct) -> {args.unanswerable_out}")
    print(f"qwen_borderline   ({len(buckets['qwen_borderline'])}, excluded) -> {args.borderline_out}")
    print(f"per-question detail -> {args.cache}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path("datasets/frames_guaranteed_answerable.json"),
                    help="The guaranteed_answerable split from Step 1.")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--base-url", default="http://localhost:7472/v1", help="Local vLLM endpoint.")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--scrape-url", default=LOCAL_SCRAPE_URL,
                    help="fetch_url backend for cache misses (local Wikipedia cache by default).")
    ap.add_argument("--k", type=int, default=3, help="Qwen samples per question.")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="Sampling temperature (>0, so k samples actually differ).")
    ap.add_argument("--workers", type=int, default=5, help="Questions processed concurrently.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N questions (outputs get a _limitN suffix).")
    ap.add_argument("--write-partial", action="store_true",
                    help="Let a --limit run write to the canonical split files.")
    ap.add_argument("--context-window", type=int, default=32768,
                    help="Qwen context window; documents over a fraction of it are summarized.")
    ap.add_argument("--summary-tokens", type=int, default=2048, help="Per-document summary size.")
    ap.add_argument("--answer-tokens", type=int, default=2048, help="Max tokens for Qwen's answer.")
    ap.add_argument("--summarizer", choices=["gpt", "qwen"], default="gpt",
                    help="Who compresses over-budget documents (default GPT, for quality).")
    ap.add_argument("--summarizer-model", default="gpt-5.4",
                    help="Model used when --summarizer gpt.")
    ap.add_argument("--no-verify", action="store_true",
                    help="Skip the post-compression check that the answer survived.")
    ap.add_argument("--cache-dir", type=Path, default=Path("doc_context_cache"),
                    help="doc_context cache (must match cache_doc_summary's config).")
    ap.add_argument("--cache", type=Path, default=Path("outputs/agent_answerable_cache.jsonl"))
    ap.add_argument("--answerable-out", type=Path, default=Path("datasets/frames_qwen_answerable.json"))
    ap.add_argument("--unanswerable-out", type=Path, default=Path("datasets/frames_qwen_unanswerable.json"))
    ap.add_argument("--borderline-out", type=Path, default=Path("datasets/frames_qwen_borderline.json"))
    main(ap.parse_args())

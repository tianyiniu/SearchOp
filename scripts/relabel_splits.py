"""DEPRECATED — superseded by scripts/get_agent_answerable.py.

This script existed because the original get_agent_answerable.py labelled the
splits on on-the-fly, uncached context (live Serper, no split_links, one temp-0
pass) while the debate step evaluated on the cached context — so the label and the
evaluation disagreed, and ~20% of 'unanswerable' questions were in fact solved by
a single pass.

get_agent_answerable.py now does all of this itself, and more: it labels on the
canonical cached context over k samples, compresses over-budget documents with
GPT-5.4 instead of Qwen, and verifies that the answer survived compression before
a question is allowed into any split. There is no longer a reason to run this.

It is kept only for reference, and refuses to run without --force because it
writes to the SAME split files as get_agent_answerable.py and would silently
overwrite them with labels built from a weaker context pipeline.

    python3 scripts/get_agent_answerable.py     # <- use this instead
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

from doc_qa import SummarizerConfig, answer_with_docs, build_and_cache_context
from llm_judge import judge_answer
from tools import LOCAL_SCRAPE_URL, build_tools

try:  # load API keys from the project .env if python-dotenv is available
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


def process_question(row, client, model, fetch, cfg, cache_dir, k, temperature) -> dict:
    """Label one question by running single_pass k times on its canonical context."""
    qid, question, gt = row.get("id"), row["question"], row["ground_truth"]
    out = {"id": qid, "question": question, "ground_truth": gt,
           "wiki_links": row.get("wiki_links", []), "k": k}
    ctx = build_and_cache_context(client, row, fetch, cfg, cache_dir)
    context = ctx.get("doc_context", "")
    out["summarized"] = ctx.get("summarized")
    if not context:
        return {**out, "label": "no_context", "correct_count": 0}

    answers, correct = [], 0
    for _ in range(k):
        answer = answer_with_docs(client, model, question, context, temperature=temperature)
        answers.append(answer)
        if judge_answer(question, gt, answer):
            correct += 1
    label = "answerable" if correct == k else "unanswerable" if correct == 0 else "borderline"
    return {**out, "answers": answers, "correct_count": correct, "label": label}


def _split_row(rec: dict) -> dict:
    """The fields downstream scripts need, plus the label provenance."""
    return {"id": rec["id"], "question": rec["question"], "ground_truth": rec["ground_truth"],
            "wiki_links": rec["wiki_links"], "correct_count": rec["correct_count"], "k": rec["k"]}


def main(args: argparse.Namespace) -> None:
    if not args.force:
        sys.exit(
            "relabel_splits.py is DEPRECATED and would overwrite the split files that\n"
            "get_agent_answerable.py now produces (with labels built from a weaker\n"
            "context pipeline: Qwen compression, no post-compression verification).\n\n"
            "Use:  python3 scripts/get_agent_answerable.py\n"
            "Pass --force only if you specifically want the old behaviour."
        )
    rows = json.loads(Path(args.dataset).read_text())
    if args.limit is not None:
        rows = rows[: args.limit]

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    fetch = build_tools(["fetch_url"], scrape_url=args.scrape_url)[0]["fetch_url"]
    cfg = SummarizerConfig(model=args.model, context_window=args.context_window,
                           summary_tokens=args.summary_tokens)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    for path in (args.answerable_out, args.unanswerable_out, args.borderline_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    print(f"summarizer config signature: {cfg.signature()} (must match the doc_context cache)")

    buckets: dict[str, list] = {"answerable": [], "unanswerable": [], "borderline": []}
    counts: Counter = Counter()
    lock = Lock()
    with open(args.cache, "w") as cache, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_question, row, client, args.model, fetch, cfg,
                               args.cache_dir, args.k, args.temperature) for row in rows]
        for future in tqdm(as_completed(futures), total=len(futures), desc="questions", unit="q"):
            rec = future.result()
            with lock:
                cache.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cache.flush()
            counts[rec["label"]] += 1
            if rec["label"] in buckets:
                buckets[rec["label"]].append(_split_row(rec))

    args.answerable_out.write_text(json.dumps(buckets["answerable"], indent=2, ensure_ascii=False))
    args.unanswerable_out.write_text(json.dumps(buckets["unanswerable"], indent=2, ensure_ascii=False))
    args.borderline_out.write_text(json.dumps(buckets["borderline"], indent=2, ensure_ascii=False))
    print(f"\nlabels (k={args.k}, temp={args.temperature}):", dict(counts))
    print(f"answerable   ({len(buckets['answerable'])}) -> {args.answerable_out}")
    print(f"unanswerable ({len(buckets['unanswerable'])}) -> {args.unanswerable_out}")
    print(f"borderline   ({len(buckets['borderline'])}, excluded) -> {args.borderline_out}")
    print(f"per-question detail -> {args.cache}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="Run this deprecated script anyway, overwriting the split files.")
    ap.add_argument("--dataset", type=Path, default=Path("datasets/frames_guaranteed_answerable.json"))
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--base-url", default="http://localhost:7472/v1", help="Local vLLM endpoint.")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--scrape-url", default=LOCAL_SCRAPE_URL,
                    help="fetch_url backend for cache misses (local Wikipedia cache by default).")
    ap.add_argument("--k", type=int, default=3, help="single_pass samples per question.")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="single_pass sampling temperature (>0: 'can't get it even with sampling').")
    ap.add_argument("--context-window", type=int, default=32768)
    ap.add_argument("--summary-tokens", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=5, help="Questions processed concurrently.")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N questions.")
    ap.add_argument("--cache-dir", type=Path, default=Path("doc_context_cache"),
                    help="doc_context cache (must match cache_doc_summary's config).")
    ap.add_argument("--cache", type=Path, default=Path("outputs/relabel_cache.jsonl"))
    ap.add_argument("--answerable-out", type=Path, default=Path("datasets/frames_qwen_answerable.json"))
    ap.add_argument("--unanswerable-out", type=Path, default=Path("datasets/frames_qwen_unanswerable.json"))
    ap.add_argument("--borderline-out", type=Path, default=Path("datasets/frames_qwen_borderline.json"))
    main(ap.parse_args())

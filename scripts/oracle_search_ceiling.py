"""Control: what does the SEARCH-SIDE evidence path score with PERFECT retrieval?

eval_search_step.py measures search against an assumed ~100% ceiling, on the
grounds that every question in frames_qwen_answerable.json was verified answerable
by Qwen given its gold documents. But that verification used a DIFFERENT evidence
packaging than the search step does: the split was defined on GPT-5.4 query-aware
compressed, verification-checked contexts (median ~5.4k chars), while the search
step hands Qwen raw page text head-truncated by pack_truncate (up to ~98k chars).

So the 100% ceiling may not transfer. This script measures the ceiling that
actually applies, by giving the search-side evidence path the gold URLs — perfect
retrieval — and running the identical pack -> answer -> judge chain:

  raw_pack     gold docs -> pack_truncate -> answer_with_docs (max_tokens=2048)
               EXACTLY eval_search_step's path. This is the harness ceiling: no
               search method can beat it.
  raw_pack_fix same evidence at max_tokens=4096. The gap vs raw_pack is loss to
               reasoning-truncation alone, since both are judged both ways.
  head_small   head-truncated to the passage budget. The SIZE control for
               `passages`: same number of characters, chosen by position.
  passages     gold docs -> BM25 paragraph selection -> pack -> answer. Same size
               as head_small, chosen by query relevance, so the gap between them
               is selection QUALITY with size held fixed.
  passages_big BM25 selection at 3x the budget — separates "selection picks the
               wrong passages" from "the budget is simply too small".
  cached_ctx   the GPT-compressed context from doc_context_cache/ -> answer.
               The packaging that DEFINED the split; should be near-ceiling. A low
               score here means the pipeline, not the packaging, is at fault.

Every arm records TWO gradings of the same generation: `correct_raw` judges the
whole string (what eval_search_step does, so its numbers stay comparable) and
`correct_visible` judges only the post-<think> answer. The smoke run showed these
differ by ~25 points, because a long reasoning trace often states the right value
somewhere even when the committed answer is wrong — so the two must never be
compared across arms as though they measured the same thing.

Every arm is free apart from the judge: documents come from the local Wikipedia
server and the answerer is the local vLLM Qwen.

    python3 scripts/wiki_backend.py &                   # required (free fetches)
    # vLLM serving Qwen/Qwen3-14B on port 7472
    export OPENAI_API_KEY=...                           # judge only
    $PY scripts/oracle_search_ceiling.py --limit 40     # quick read
    $PY scripts/oracle_search_ceiling.py                # full 266
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from doc_qa import (SummarizerConfig, answer_with_docs, gold_urls,
                    load_cached_context, usable_context)
from eval_search_step import answer_in_text, is_refusal, pack_truncate
from llm_judge import judge_answer
from tools import LOCAL_SCRAPE_URL, FetchUrl

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

ARMS = ("raw_pack", "raw_pack_fix", "head_small", "passages", "passages_big", "cached_ctx")


# --- answer post-processing -------------------------------------------------

_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(answer: str) -> str:
    """The visible answer: reasoning traces removed.

    Qwen3 emits <think>...</think> before its answer. Judging (and the refusal
    heuristic) on the raw string reads the model's private deliberation as if it
    were the answer — a trace that says 'the documents do not mention...' scores
    as a refusal even when the final line is correct. An UNCLOSED <think> means
    the generation ran out of tokens mid-thought and there is no answer at all;
    that returns '' rather than the trace.
    """
    a = answer or ""
    if _THINK_OPEN.search(a) and "</think>" not in a.lower():
        return ""                                   # truncated mid-thought
    return _THINK_BLOCK.sub("", a).strip()


def truncated_mid_think(answer: str) -> bool:
    a = answer or ""
    return bool(_THINK_OPEN.search(a)) and "</think>" not in a.lower()


# --- BM25 passage selection -------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def chunk_page(text: str, target: int = 900) -> list[str]:
    """Split a page into ~`target`-char chunks on blank lines.

    Cached pages carry no section markup (cache_web_links stores extracted text),
    so paragraphs are the finest structure available. Short paragraphs are merged
    forward so a chunk is a coherent unit rather than a one-line caption.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        buf = f"{buf}\n\n{p}" if buf else p
        if len(buf) >= target:
            chunks.append(buf)
            buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def bm25_select(query: str, text: str, budget_chars: int,
                k1: float = 1.5, b: float = 0.75) -> str:
    """The `budget_chars` of `text` most relevant to `query`, by BM25 over chunks.

    Chunks are returned in DOCUMENT ORDER, not score order, so the extract still
    reads as a page. Written out rather than pulled from a library so the arm has
    no dependency the rest of the project lacks.
    """
    chunks = chunk_page(text)
    if not chunks:
        return text[:budget_chars]
    if sum(len(c) for c in chunks) <= budget_chars:
        return text[:budget_chars]

    docs = [_tokens(c) for c in chunks]
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n
    df: Counter = Counter()
    for d in docs:
        df.update(set(d))
    q_terms = set(_tokens(query))

    scores = []
    for i, d in enumerate(docs):
        tf = Counter(d)
        dl = len(d) or 1
        s = 0.0
        for t in q_terms:
            f = tf.get(t, 0)
            if not f:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        scores.append((s, i))

    scores.sort(reverse=True)
    keep, used = [], 0
    for s, i in scores:
        if used + len(chunks[i]) > budget_chars and keep:
            continue
        keep.append(i)
        used += len(chunks[i])
        if used >= budget_chars:
            break
    keep.sort()
    return "\n\n[...]\n\n".join(chunks[i] for i in keep)


# --- the four arms ----------------------------------------------------------

def build_evidence(arm: str, row: dict, docs: list[tuple[str, str]],
                   cfg: SummarizerConfig, args) -> str:
    """The doc_context each arm hands the answerer."""
    if arm in ("raw_pack", "raw_pack_fix"):
        return pack_truncate(docs, cfg)
    budget = args.passage_budget * (3 if arm == "passages_big" else 1)
    per_doc = max(1200, budget // max(1, len(docs)))
    if arm == "head_small":
        picked = [(url, text[:per_doc]) for url, text in docs]
    elif arm in ("passages", "passages_big"):
        picked = [(url, bm25_select(row["question"], text, per_doc)) for url, text in docs]
    else:
        raise ValueError(arm)
    return pack_truncate(picked, cfg)              # already small; pack just labels


def process_question(row: dict, arm: str, client, args, cfg: SummarizerConfig,
                     fetch) -> dict:
    qid, question, gt = row.get("id"), row["question"], row["ground_truth"]
    rec = {"id": qid, "question": question, "ground_truth": gt, "arm": arm}
    try:
        if arm == "cached_ctx":
            cached = load_cached_context(qid, cfg, args.cache_dir)
            doc_context = usable_context(cached) if cached else ""
            rec["n_docs"] = (cached or {}).get("num_docs", 0)
            if not doc_context:
                rec["status"] = "no_cached_context"
                return rec
        else:
            urls = gold_urls(row)
            docs = []
            for url in urls:
                try:
                    text = fetch(url)
                except Exception:
                    continue
                if text and not text.startswith(("[fetch_url", "No readable content")):
                    docs.append((url, text))
            rec["n_docs"] = len(docs)
            rec["n_gold_urls"] = len(urls)
            if not docs:
                rec["status"] = "fetch_failed"
                return rec
            doc_context = build_evidence(arm, row, docs, cfg, args)

        rec["context_chars"] = len(doc_context)
        rec["answer_in_context"] = answer_in_text(gt, doc_context)

        max_tokens = args.max_tokens if arm == "raw_pack" else args.max_tokens_fixed
        raw = answer_with_docs(client, args.model, question, doc_context,
                               max_tokens=max_tokens)
        visible = strip_think(raw)

        # Both gradings of the SAME generation, so the judging protocol is never
        # confounded with the arm: correct_raw is what eval_search_step reports,
        # correct_visible is what the model actually committed to.
        rec["answer"] = raw
        rec["truncated_mid_think"] = truncated_mid_think(raw)
        rec["correct_raw"] = bool(raw.strip()) and judge_answer(question, gt, raw)
        rec["correct_visible"] = bool(visible.strip()) and judge_answer(question, gt, visible)
        rec["downstream_correct"] = rec["correct_visible"]
        rec["refused"] = is_refusal(visible)
        rec["refused_raw"] = is_refusal(raw)
        rec["status"] = "ok"
    except Exception as exc:
        rec["status"] = "error"
        rec["error"] = str(exc)
    return rec


# --- driver -----------------------------------------------------------------

def aggregate(records: list[dict]) -> dict:
    ok = [r for r in records if r.get("status") == "ok"]
    n = len(ok)
    rate = lambda key: (sum(1 for r in ok if r.get(key)) / n) if n else 0.0
    mean = lambda key: (sum(r.get(key, 0) for r in ok) / n) if n else 0.0
    return {
        "n": n,
        "n_attempted": len(records),
        "n_dropped": len(records) - n,
        "accuracy_visible": rate("correct_visible"),
        "accuracy_raw_string": rate("correct_raw"),
        "downstream_accuracy": rate("downstream_correct"),
        "refusal_rate": rate("refused"),
        "refusal_rate_raw_string": rate("refused_raw"),
        "truncated_mid_think_rate": rate("truncated_mid_think"),
        "answer_in_context_rate": rate("answer_in_context"),
        "mean_context_chars": mean("context_chars"),
        "mean_docs": mean("n_docs"),
    }


def run_arm(arm: str, rows, client, args, cfg, fetch) -> list[dict]:
    records: list[dict] = []
    out_path = args.out_dir / f"oracle_{arm}_cache.jsonl"
    lock = Lock()
    with open(out_path, "w") as cache, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_question, row, arm, client, args, cfg, fetch)
                   for row in rows]
        for future in tqdm(as_completed(futures), total=len(futures), desc=arm, unit="q"):
            rec = future.result()
            with lock:
                cache.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cache.flush()
            records.append(rec)
    return records


def main(args: argparse.Namespace) -> None:
    rows = json.loads(Path(args.dataset).read_text())
    if args.limit is not None:
        # A seeded shuffle, so a partial run is a random sample rather than the
        # first N in dataset order (and is reproducible across arms and runs).
        rows = random.Random(args.seed).sample(rows, min(args.limit, len(rows)))
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    cfg = SummarizerConfig(model=args.model, context_window=args.context_window)
    fetch = FetchUrl(scrape_url=args.scrape_url)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for arm in arms:
        summary[arm] = aggregate(run_arm(arm, rows, client, args, cfg, fetch))

    args.summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    # acc_visible grades the committed answer; acc_raw grades the whole string
    # including the reasoning trace, which is what eval_search_step reports.
    print(f"\n{'arm':<16}{'n':>5}{'drop':>6}{'acc_visible':>13}{'acc_raw':>10}"
          f"{'refused':>9}{'trunc_think':>12}{'ctx_chars':>11}")
    for arm in arms:
        a = summary[arm]
        print(f"{arm:<16}{a['n']:>5}{a['n_dropped']:>6}{a['accuracy_visible']:>13.1%}"
              f"{a['accuracy_raw_string']:>10.1%}{a['refusal_rate']:>9.1%}"
              f"{a['truncated_mid_think_rate']:>12.1%}{a['mean_context_chars']:>11,.0f}")
    print(f"\nper-arm caches -> {args.out_dir}/oracle_<arm>_cache.jsonl")
    print(f"summary -> {args.summary_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path("datasets/frames_qwen_answerable.json"))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--base-url", default="http://localhost:7472/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--scrape-url", default=LOCAL_SCRAPE_URL,
                    help="fetch_url backend; defaults to the local Wikipedia server (free).")
    ap.add_argument("--context-window", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="Answerer budget for raw_pack (matches eval_search_step).")
    ap.add_argument("--max-tokens-fixed", type=int, default=4096,
                    help="Answerer budget for the other arms.")
    ap.add_argument("--passage-budget", type=int, default=8000,
                    help="Total chars of selected passages across documents.")
    ap.add_argument("--cache-dir", type=Path, default=Path("doc_context_cache"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="Evaluate a seeded random sample of N questions.")
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed for --limit.")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--summary-out", type=Path, default=Path("outputs/oracle_ceiling_summary.json"))
    main(ap.parse_args())

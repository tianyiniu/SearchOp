"""Step 1: build the FRAMES 'guaranteed_answerable' split.

A question is 'guaranteed_answerable' when a strong proprietary model (GPT-5.4)
*cannot* answer it from parametric knowledge alone, but *can* answer it once given
the ground-truth documents. That isolates questions where success is due to the
documents, not memorized knowledge — the foundation for separating retrieval from
aggregation in later steps.

Per-question procedure:
  1. Ask GPT-5.4 the question with NO documents. If it is already correct, skip
     (the model knows it parametrically — counted but discarded).
  2. Otherwise fetch the ground-truth documents (the question's wiki_links). If
     none can be fetched, skip and log.
  3. Ask GPT-5.4 again WITH those documents. If now correct -> guaranteed_answerable;
     otherwise the documents are insufficient, so skip and log.

Correctness for both passes is graded by judge_answer (gpt-5.4-mini).

    export OPENAI_API_KEY=...   # GPT-5.4 answerer + judge
    export SERPER_API_KEY=...   # document fetch
    python scripts/get_guaranteed_answerable.py --limit 50
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_judge import call_openai, judge_answer
from tools import build_tools

try:  # load API keys from the project .env if python-dotenv is available
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


NO_DOC_SYSTEM = (
    "Answer the question as accurately as you can from your own knowledge. Give "
    "your single best answer; do not refuse. End with one line: 'ANSWER: <answer>'."
)
WITH_DOC_SYSTEM = (
    "Answer the question using ONLY the documents provided; the answer is contained "
    "in them. End with one line: 'ANSWER: <answer>'."
)


def fetch_documents(urls: list[str], fetch, max_doc_chars: int) -> list[tuple[str, str]]:
    """Fetch each URL's full text. Returns (url, text) for the ones that worked;
    a failed or empty fetch is simply dropped."""
    docs: list[tuple[str, str]] = []
    for url in urls:
        try:
            text = fetch(url)
        except Exception:
            continue
        if not text or text.startswith(("[fetch_url", "No readable content")):
            continue
        docs.append((url, text[:max_doc_chars] if max_doc_chars else text))
    return docs


def build_doc_prompt(question: str, docs: list[tuple[str, str]]) -> str:
    blocks = [f"[Document {i}] {url}\n{text}" for i, (url, text) in enumerate(docs, 1)]
    return "\n\n".join(blocks) + f"\n\nQUESTION: {question}"


def process_question(row: dict, fetch, args: argparse.Namespace) -> dict:
    """Run the two-pass procedure for one question and return its decision record."""
    question, ground_truth = row["question"], row["ground_truth"]
    rec = {
        "id": row.get("id"),
        "question": question,
        "ground_truth": ground_truth,
        "wiki_links": row.get("wiki_links", []),
    }
    try:
        # Pass 1: no documents. Correct here => parametric knowledge => discard.
        answer_no_docs = call_openai(NO_DOC_SYSTEM, question, model=args.answer_model)
        rec["answer_no_docs"] = answer_no_docs
        rec["correct_no_docs"] = judge_answer(question, ground_truth, answer_no_docs)
        if rec["correct_no_docs"]:
            rec["status"] = "skipped_parametric"
            return rec

        # Fetch the ground-truth documents.
        docs = fetch_documents(rec["wiki_links"], fetch, args.max_doc_chars)
        rec["num_docs_total"] = len(rec["wiki_links"])
        rec["num_docs_fetched"] = len(docs)
        if not docs:
            rec["status"] = "skipped_fetch_failed"
            return rec

        # Pass 2: with documents. Correct here => guaranteed_answerable.
        answer_with_docs = call_openai(
            WITH_DOC_SYSTEM, build_doc_prompt(question, docs), model=args.answer_model)
        rec["answer_with_docs"] = answer_with_docs
        rec["correct_with_docs"] = judge_answer(question, ground_truth, answer_with_docs)
        rec["status"] = "guaranteed_answerable" if rec["correct_with_docs"] else "skipped_docs_insufficient"
    except Exception as exc:  # keep the pool going; record what failed
        rec["status"] = "error"
        rec["error"] = str(exc)
    return rec


def main(args: argparse.Namespace) -> None:
    rows = json.loads(Path(args.dataset).read_text())
    if args.limit is not None:
        rows = rows[: args.limit]

    fetch = build_tools(["fetch_url"])[0]["fetch_url"]  # full-text page fetch
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    guaranteed: list[dict] = []
    counts: Counter = Counter()
    lock = Lock()
    with open(args.cache, "w") as cache, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_question, row, fetch, args) for row in rows]
        for future in tqdm(as_completed(futures), total=len(futures), desc="questions", unit="q"):
            rec = future.result()
            with lock:
                cache.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cache.flush()
            counts[rec["status"]] += 1
            if rec["status"] == "guaranteed_answerable":
                guaranteed.append(rec)

    Path(args.out).write_text(json.dumps(guaranteed, indent=2, ensure_ascii=False))
    print(f"\nguaranteed_answerable: {len(guaranteed)}/{len(rows)} -> {args.out}")
    print("breakdown:", dict(counts))
    print(f"full decisions cached -> {args.cache}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path("datasets/frames_test_full.json"))
    ap.add_argument("--answer-model", default="gpt-5.4", help="Proprietary answerer (Responses API).")
    ap.add_argument("--workers", type=int, default=5, help="Questions processed concurrently.")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N questions.")
    ap.add_argument("--max-doc-chars", type=int, default=0,
                    help="Per-document char cap (0 = full text, the default).")
    ap.add_argument("--cache", type=Path, default=Path("outputs/guaranteed_answerable_cache.jsonl"),
                    help="JSONL of every question's decision.")
    ap.add_argument("--out", type=Path, default=Path("datasets/frames_guaranteed_answerable.json"),
                    help="The guaranteed_answerable subset.")
    main(ap.parse_args())

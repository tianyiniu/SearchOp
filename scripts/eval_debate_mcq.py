"""(SuperGPQA) Fixed-template debate evaluation on the qwen_unanswerable reasoning split.

Every question here is one Qwen fails k/k with chain-of-thought (prepare_supergpqa.py),
so single_pass and sample_vote are already established by construction and are NOT
re-run. This tests whether STRUCTURED aggregation recovers the failures:

  self_critique - solver -> critic -> revised solver (3 calls).
  fixed_debate  - k solvers answer, see each other and reconsider, a synthesizer
                  commits (2k + 1 calls).

No documents, thinking off, grading by exact letter match (no judge). Run it on the
answerable split too (--dataset ...answerable...) to check whether debate HURTS
already-solved questions (the model-conformity failure mode).

Parallelism: one task per (question, shape) pair, so both shapes across all
questions saturate the vLLM server at once. Raise --workers to fill the GPU.

    # vLLM serving Qwen/Qwen3-14B on port 7472
    python3 scripts/eval_debate_mcq.py --shapes self_critique,fixed_debate --workers 16
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import debate_mcq as D

try:  # load API keys from the project .env if python-dotenv is available
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# self_critique wants a low-but-nonzero temperature; debate wants diverse samples.
SHAPE_TEMP = {"self_critique": 0.3, "fixed_debate": 0.7}


def run_one(row: dict, shape: str, client, model: str, params: dict) -> dict:
    """Run one (question, shape) and grade by letter match."""
    rec = {"id": row["id"], "shape": shape, "field": row.get("field"),
           "difficulty": row.get("difficulty"), "answer_letter": row["answer_letter"]}
    try:
        schema = D.build_fixed_schema(shape, params["debate_k"])
        letter = D.execute_schema(client, model, row["question"], list(row["options"]),
                                  schema, SHAPE_TEMP.get(shape, 0.7), params["max_tokens"])
        rec.update(answer=letter, correct=(letter == row["answer_letter"]), status="ok")
    except Exception as exc:  # keep the pool going
        rec.update(status="error", error=str(exc))
    return rec


def summarize(records: list[dict], shapes: list[str]) -> dict:
    ok = [r for r in records if r.get("status") == "ok"]
    correct, total = Counter(), Counter()
    solved_by: dict[str, set] = defaultdict(set)
    for r in ok:
        total[r["shape"]] += 1
        if r["correct"]:
            correct[r["shape"]] += 1
            solved_by[r["id"]].add(r["shape"])
    ids = {r["id"] for r in ok}
    n = len(ids)

    def rate_over(subset_ok: list[dict], shape: str) -> float:
        t = sum(1 for r in subset_ok if r["shape"] == shape)
        c = sum(1 for r in subset_ok if r["shape"] == shape and r["correct"])
        return (c / t) if t else 0.0

    per_shape = {s: {"correct": correct[s], "total": total[s],
                     "solve_rate": (correct[s] / total[s]) if total[s] else 0.0}
                 for s in shapes}

    # heterogeneity: per-shape solve rate broken out by difficulty and by field.
    by_diff = {}
    for d in sorted({r["difficulty"] for r in ok}):
        sub = [r for r in ok if r["difficulty"] == d]
        by_diff[d] = {"n_questions": len({r["id"] for r in sub}),
                      **{s: rate_over(sub, s) for s in shapes}}
    by_field = {}
    for f in sorted({r["field"] for r in ok}):
        sub = [r for r in ok if r["field"] == f]
        nq = len({r["id"] for r in sub})
        if nq >= 10:  # only fields with enough questions to be meaningful
            by_field[f] = {"n_questions": nq, **{s: rate_over(sub, s) for s in shapes}}

    return {
        "n_questions": n,
        "per_shape": per_shape,
        "solved_by_any": len(solved_by),
        "solved_by_any_rate": (len(solved_by) / n) if n else 0.0,
        "by_difficulty": by_diff,
        "by_field": by_field,
        "errors": sum(1 for r in records if r.get("status") == "error"),
    }


def main(args: argparse.Namespace) -> None:
    rows = json.loads(Path(args.dataset).read_text())
    if args.limit is not None:
        rows = rows[: args.limit]
    shapes = [s.strip() for s in args.shapes.split(",") if s.strip()]
    params = {"debate_k": args.debate_k, "max_tokens": args.answer_tokens}

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    for path in (args.cache, args.summary_out):
        path.parent.mkdir(parents=True, exist_ok=True)

    # One task per (question, shape) -> maximal parallelism across the whole grid.
    tasks = [(row, shape) for row in rows for shape in shapes]
    print(f"{len(rows)} questions x {len(shapes)} shapes = {len(tasks)} tasks "
          f"| shapes: {shapes} | workers: {args.workers}")

    records: list[dict] = []
    lock = Lock()
    with open(args.cache, "w") as cache, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, row, shape, client, args.model, params)
                   for row, shape in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="debate", unit="task"):
            rec = future.result()
            with lock:
                cache.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cache.flush()
            records.append(rec)

    summary = summarize(records, shapes)
    args.summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\nper-shape solve rate:")
    for s in shapes:
        ps = summary["per_shape"][s]
        print(f"  {s:<16} {ps['correct']}/{ps['total']}  ({ps['solve_rate']:.1%})")
    print(f"solved by any shape: {summary['solved_by_any']}/{summary['n_questions']} "
          f"({summary['solved_by_any_rate']:.1%})")
    print("\nby difficulty:")
    for d, v in summary["by_difficulty"].items():
        print(f"  {d:<8} (n={v['n_questions']:>4})  " +
              "  ".join(f"{s}={v[s]:.1%}" for s in shapes))
    if summary["errors"]:
        print(f"\nerrors: {summary['errors']}")
    print(f"\ncache -> {args.cache}\nsummary -> {args.summary_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path("datasets/supergpqa_qwen_unanswerable.json"))
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--base-url", default="http://localhost:7472/v1", help="Local vLLM endpoint.")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--shapes", default="self_critique,fixed_debate",
                    help="Comma-separated fixed templates (single_pass/sample_vote intentionally omitted).")
    ap.add_argument("--debate-k", type=int, default=3, help="Solvers per round for fixed_debate.")
    ap.add_argument("--answer-tokens", type=int, default=3072, help="Max tokens per model call (thinking off).")
    ap.add_argument("--workers", type=int, default=16, help="(question,shape) tasks run concurrently.")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N questions.")
    ap.add_argument("--cache", type=Path, default=Path("outputs/debate_mcq_cache.jsonl"))
    ap.add_argument("--summary-out", type=Path, default=Path("outputs/debate_mcq_summary.json"))
    main(ap.parse_args())

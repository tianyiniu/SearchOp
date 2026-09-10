"""Best-of-N single_pass baseline (SuperGPQA MCQ) -- the resampling control.

The question this answers: how much of GT evolution's solve rate is just REPEATED
EXECUTION (best-of-N luck) rather than better schema structure? Evolution runs the
minimal schema, then up to 6 edits, each ONE execution, and with --confirm-runs 1 a
single lucky correct execution counts as solved -- i.e. up to 7 independent shots.

So the matched control is: run the SAME minimal single-solver schema N times and ask
"solved if ANY of the N is correct". We report the order-independent pass@k estimator
(Chen et al. 2021) for k=1..N:

    pass@k = 1 - C(N-c, k) / C(N, k)   averaged over questions   (c = #correct of N)

  pass@1  = expected single_pass accuracy.
  pass@N  = best-of-N (any correct) -- the resampling ceiling at N shots.

Overlay pass@k on evolution's cumulative solved_at_step: if best-of-7 ~= evolution's
rate, the evolution advantage is resampling, not structure. Also reports
majority_vote@N -- a DEPLOYABLE aggregation (no oracle), unlike best-of-N.

Thinking off, temp 0.7 (same executor as single_pass / evolution step-0).

PARALLELISM: tasks are flattened to (question x sample) -- N*|questions| INDEPENDENT
single-call tasks -- so every worker always has a call to make (no per-question
sequential dependency, no idle tail). Two levers to go faster:
  --workers   : concurrent in-flight calls. Raise toward the vLLM server's
                max_num_seqs (often ~256) to keep its continuous batch full.
  --base-urls : comma-separated vLLM endpoints; calls round-robin across them, so
                several GPU replicas (e.g. :7472,:7473) run in parallel.

    python scripts/bestofn_mcq.py --dataset datasets/supergpqa_filter_strict_full.json \
        --n 7 --workers 64 --base-urls http://localhost:7472/v1,http://localhost:7473/v1 \
        --evo-summary outputs/evolve_mcq_strict_summary.json \
        --cache outputs/bestofn_strict_cache.jsonl --summary-out outputs/bestofn_strict_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle
from math import comb
from pathlib import Path
from threading import Lock

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import debate_mcq as D

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


def run_sample(row: dict, client, model: str, temp: float, max_tokens: int) -> dict:
    """One single_pass execution (minimal schema, thinking off). Returns the letter."""
    try:
        letter = D.execute_schema(client, model, row["question"], list(row["options"]),
                                  D.MINIMAL_SCHEMA, temp, max_tokens)
        return {"id": row["id"], "letter": letter,
                "correct": (letter == row["answer_letter"]), "status": "ok"}
    except Exception as exc:
        return {"id": row["id"], "status": "error", "error": str(exc)}


def pass_at_k(n: int, c: int, k: int) -> float:
    """Prob. that >=1 of k draws (without replacement) is correct, given c correct of n."""
    if c <= 0:
        return 0.0
    if n - c < k:  # too few wrong answers to fill k draws -> guaranteed a correct one
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def summarize(records: list[dict], n: int) -> dict:
    ok = [r for r in records if r["status"] == "ok"]
    m = len(ok)
    passk = {}
    for k in range(1, n + 1):
        passk[k] = sum(pass_at_k(r["n"], r["n_correct"], k) for r in ok) / m if m else 0.0
    return {
        "n_questions": m,
        "n_samples": n,
        "pass_at_k": {k: round(v, 4) for k, v in passk.items()},
        "pass_at_1": round(passk.get(1, 0.0), 4),                 # = single_pass
        "best_of_n": round(passk.get(n, 0.0), 4),                 # any correct in N
        "majority_vote_at_n": round(sum(1 for r in ok if r["majority_correct"]) / m, 4) if m else 0.0,
        "incomplete": sum(1 for r in records if r.get("n", 0) < n),
        "errors": sum(r.get("n_errors", 0) for r in records),
    }


def overlay_evolution(evo_summary_path: Path, n: int) -> dict | None:
    """Evolution's cumulative solve rate by step k (each step = one more execution)."""
    if not evo_summary_path.exists():
        return None
    d = json.loads(evo_summary_path.read_text())
    sa = d.get("solved_at_step", {})
    nq = d.get("n_questions") or d.get("n_evaluated")
    if not nq:
        return None
    cum, out = 0, {}
    for step in range(0, n):            # step s -> after s+1 executions
        cum += sa.get(str(step), sa.get(step, 0))
        out[step + 1] = round(cum / nq, 4)
    return out


def main(args: argparse.Namespace) -> None:
    rows = json.loads(Path(args.dataset).read_text())
    if args.limit is not None:
        rows = rows[: args.limit]
    by_id = {r["id"]: r for r in rows}

    urls = [u.strip() for u in args.base_urls.split(",") if u.strip()]
    clients = cycle([OpenAI(base_url=u, api_key=args.api_key) for u in urls])
    for p in (args.cache, args.summary_out):
        p.parent.mkdir(parents=True, exist_ok=True)

    # flatten: N independent single-call tasks per question -> maximal concurrency
    tasks = [(row, clients_i) for row in rows for clients_i in range(args.n)]
    print(f"{len(rows)} questions x N={args.n} = {len(tasks)} sample-tasks | "
          f"endpoints={len(urls)} | workers={args.workers} | temp={args.temperature}")

    # per-question aggregation; a question's record is flushed once all N samples land
    agg = {r["id"]: {"letters": [], "n_correct": 0, "n_errors": 0} for r in rows}
    pending = {r["id"]: args.n for r in rows}
    records: list[dict] = []
    lock = Lock()

    def finalize(qid: str) -> dict:
        a = agg[qid]
        row = by_id[qid]
        tally = Counter(l for l in a["letters"] if l)
        majority = tally.most_common(1)[0][0] if tally else None
        return {"id": qid, "field": row.get("field"), "difficulty": row.get("difficulty"),
                "answer_letter": row["answer_letter"], "n": len(a["letters"]),
                "n_correct": a["n_correct"], "n_errors": a["n_errors"],
                "any_correct": a["n_correct"] > 0, "letters": a["letters"],
                "majority": majority, "majority_correct": (majority == row["answer_letter"]),
                "status": "ok"}

    with open(args.cache, "w") as cache, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_sample, row, next(clients), args.model,
                               args.temperature, args.answer_tokens) for row, _ in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="best-of-n", unit="call"):
            s = fut.result()
            qid = s["id"]
            with lock:
                a = agg[qid]
                if s["status"] == "ok":
                    a["letters"].append(s["letter"])
                    a["n_correct"] += int(s["correct"])
                else:
                    a["n_errors"] += 1
                pending[qid] -= 1
                if pending[qid] == 0:               # question complete -> flush its record
                    rec = finalize(qid)
                    cache.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    cache.flush()
                    records.append(rec)

    summary = summarize(records, args.n)
    evo = overlay_evolution(args.evo_summary, args.n)
    if evo:
        summary["evolution_cumulative_by_budget"] = evo
    args.summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\npass@1 (single_pass):    {summary['pass_at_1']:.1%}")
    print(f"best-of-{args.n} (any correct): {summary['best_of_n']:.1%}")
    print(f"majority-vote@{args.n}:        {summary['majority_vote_at_n']:.1%}  (deployable)")
    print("\npass@k  vs  evolution-cumulative (same budget = k executions):")
    for k in range(1, args.n + 1):
        e = f"{evo[k]:.1%}" if evo and k in evo else "  -  "
        print(f"  k={k}:  best-of-k {summary['pass_at_k'][k]:.1%}   |   evolution {e}")
    if summary["errors"] or summary["incomplete"]:
        print(f"\nerrors: {summary['errors']}  | incomplete questions: {summary['incomplete']}")
    print(f"\ncache -> {args.cache}\nsummary -> {args.summary_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("datasets/supergpqa_filter_strict_full.json"))
    ap.add_argument("--n", type=int, default=7, help="Samples per question (7 = evolution's max budget).")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--base-urls", default="http://localhost:7472/v1",
                    help="Comma-separated vLLM endpoints; calls round-robin across them (multi-GPU).")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--temperature", type=float, default=0.7, help="Match single_pass / evolution step-0.")
    ap.add_argument("--answer-tokens", type=int, default=3072)
    ap.add_argument("--workers", type=int, default=64,
                    help="Concurrent in-flight calls; raise toward the server's max_num_seqs.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--evo-summary", type=Path, default=Path("outputs/evolve_mcq_strict_summary.json"),
                    help="Evolution summary to overlay (cumulative solved_at_step).")
    ap.add_argument("--cache", type=Path, default=Path("outputs/bestofn_strict_cache.jsonl"))
    ap.add_argument("--summary-out", type=Path, default=Path("outputs/bestofn_strict_summary.json"))
    main(ap.parse_args())

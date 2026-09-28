"""(3) Debate-step evaluation: can different aggregation schemas solve qwen_unanswerable?

Every question in frames_qwen_unanswerable.json is known to be answerable from its
ground-truth documents (Step 1), yet a single Qwen pass over those documents fails
(Step 2). Here we give Qwen the SAME documents and try a small fixed library of
aggregation shapes, isolating synthesis from retrieval (no web search, tools off).

The document context comes from the doc_context cache (cache_doc_summary.py) so it
is byte-identical to what defined the split; on a cache miss we rebuild it from the
local Wikipedia cache. Correctness is graded by judge_answer (gpt-5.4-mini).

Shapes:
  single_pass        - one solver pass (control; ~the Step 2 baseline).
  self_refine        - solver -> critic -> revised solver (temp 0.3).
  sample_vote        - N independent samples, majority vote on the answer line.
  multi_agent_debate - K personas answer, see each other, then a synthesizer commits.

    export OPENAI_API_KEY=...                 # judge
    python3 scripts/wiki_backend.py &         # local doc cache (for cache misses)
    # vLLM serving Qwen/Qwen3-14B on port 7472
    python3 scripts/eval_debate_step.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from doc_qa import (SummarizerConfig, WITH_DOC_SYSTEM, answer_with_docs,
                    build_and_cache_context)
from llm_judge import judge_answer
from tools import LOCAL_SCRAPE_URL, build_tools

try:  # load API keys from the project .env if python-dotenv is available
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# Temperatures per shape. self_refine is intentionally low-but-nonzero.
SELF_REFINE_TEMP = 0.3
SAMPLE_TEMP = 0.7
DEBATE_TEMP = 0.7

CRITIC_SYSTEM = (
    "You are a critic. Given documents, a question, and a candidate answer, check "
    "the candidate strictly against the documents. Point out any factual error, "
    "unsupported claim, or missing piece, citing the relevant document. Do NOT "
    "give a final answer yourself — only critique."
)
SYNTH_SYSTEM = (
    "You are a synthesizer. Given documents, a question, and several candidate "
    "answers, resolve the disagreements using the documents and commit to one "
    "final answer. You MUST give a specific answer; never abstain. End with one "
    "line: 'ANSWER: <answer>'."
)
DEBATE_PERSONAS = [
    "You are a meticulous fact-checker who grounds every claim in the documents.",
    "You are a domain expert who reasons step by step about the question.",
    "You are a careful synthesizer who combines evidence across documents.",
]


# --- small helpers ---------------------------------------------------------

def chat(client, model, system, user, temperature, max_tokens=2048) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=temperature, max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def extract_answer(text: str) -> str:
    """Text after the last 'ANSWER:' marker, else the last non-empty line."""
    upper = text.upper()
    if "ANSWER:" in upper:
        return text[upper.rfind("ANSWER:") + 7:].strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


def normalize_answer(text: str) -> str:
    """Lowercased, punctuation/article-stripped answer line — the vote key."""
    a = extract_answer(text).lower()
    a = re.sub(r"[^\w\s]", " ", a)
    a = re.sub(r"\b(the|a|an)\b", " ", a)
    return re.sub(r"\s+", " ", a).strip()


def _base_user(question: str, context: str) -> str:
    return f"{context}\n\nQUESTION: {question}"


# --- debate shapes: (client, model, question, context, params) -> answer ----

def shape_single_pass(client, model, question, context, params):
    return answer_with_docs(client, model, question, context, temperature=0.0)


def shape_self_refine(client, model, question, context, params):
    t = SELF_REFINE_TEMP
    base = _base_user(question, context)
    candidate = answer_with_docs(client, model, question, context, temperature=t)
    critique = chat(client, model, CRITIC_SYSTEM,
                    f"{base}\n\nCANDIDATE ANSWER:\n{candidate}\n\nCritique it against the documents.", t)
    revise = (f"{base}\n\nA candidate answer was:\n{candidate}\n\nA critic noted:\n{critique}\n\n"
              "Using the documents, give the corrected final answer. End with 'ANSWER: <answer>'.")
    return chat(client, model, WITH_DOC_SYSTEM, revise, t)


def shape_sample_vote(client, model, question, context, params):
    n = params["sample_n"]
    samples = [answer_with_docs(client, model, question, context, temperature=SAMPLE_TEMP)
               for _ in range(n)]
    keys = [normalize_answer(s) for s in samples]
    tally = Counter(k for k in keys if k)
    if not tally:
        return samples[0]
    winner = tally.most_common(1)[0][0]
    # Return a full representative answer from the winning cluster.
    return next(s for s, k in zip(samples, keys) if k == winner)


def shape_multi_agent_debate(client, model, question, context, params):
    personas = DEBATE_PERSONAS[: params["debate_k"]]
    base = _base_user(question, context)
    # Round 1: each persona answers independently.
    round1 = [chat(client, model, f"{p} {WITH_DOC_SYSTEM}", base, DEBATE_TEMP) for p in personas]
    # Round 2: each persona sees the others and reconsiders.
    round2 = []
    for i, p in enumerate(personas):
        others = "\n\n".join(f"[Agent {j + 1}]: {extract_answer(a)}"
                             for j, a in enumerate(round1) if j != i)
        user = (f"{base}\n\nOther agents answered:\n{others}\n\n"
                "Reconsider using the documents and give your best answer. End with 'ANSWER: <answer>'.")
        round2.append(chat(client, model, f"{p} {WITH_DOC_SYSTEM}", user, DEBATE_TEMP))
    # Synthesize a single committed answer.
    candidates = "\n\n".join(f"[Agent {j + 1}]: {extract_answer(a)}" for j, a in enumerate(round2))
    user = f"{base}\n\nCandidate answers:\n{candidates}\n\nCommit to the single best final answer."
    return chat(client, model, SYNTH_SYSTEM, user, DEBATE_TEMP)


SHAPES = {
    "single_pass": shape_single_pass,
    "self_refine": shape_self_refine,
    "sample_vote": shape_sample_vote,
    "multi_agent_debate": shape_multi_agent_debate,
}


# --- per-question driver ---------------------------------------------------

def process_question(row, client, model, fetch, cfg, cache_dir, shapes, params) -> list[dict]:
    qid, question, gt = row.get("id"), row["question"], row["ground_truth"]
    ctx = build_and_cache_context(client, row, fetch, cfg, cache_dir)
    context = ctx.get("doc_context", "")
    base = {"id": qid, "question": question, "ground_truth": gt,
            "summarized": ctx.get("summarized"), "num_docs": ctx.get("num_docs")}
    if not context:
        return [{**base, "shape": None, "status": "skipped_no_context"}]

    records = []
    for name in shapes:
        try:
            answer = SHAPES[name](client, model, question, context, params)
            records.append({**base, "shape": name, "answer": answer,
                            "correct": judge_answer(question, gt, answer), "status": "ok"})
        except Exception as exc:  # keep the pool going
            records.append({**base, "shape": name, "status": "error", "error": str(exc)})
    return records


def summarize(records: list[dict], shapes: list[str]) -> dict:
    correct, total = Counter(), Counter()
    solved_by: dict[str, set] = {}
    for r in records:
        if r.get("status") != "ok":
            continue
        total[r["shape"]] += 1
        if r["correct"]:
            correct[r["shape"]] += 1
            solved_by.setdefault(r["id"], set()).add(r["shape"])
    n_questions = len({r["id"] for r in records})
    per_shape = {s: {"correct": correct[s], "total": total[s],
                     "solve_rate": (correct[s] / total[s]) if total[s] else 0.0}
                 for s in shapes}
    return {"n_questions": n_questions, "per_shape": per_shape,
            "solved_by_any": len(solved_by),
            "solved_by_any_rate": (len(solved_by) / n_questions) if n_questions else 0.0}


def main(args: argparse.Namespace) -> None:
    rows = json.loads(Path(args.dataset).read_text())
    if args.limit is not None:
        rows = rows[: args.limit]
    shapes = [s.strip() for s in args.shapes.split(",") if s.strip()]
    params = {"sample_n": args.sample_n, "debate_k": args.debate_k}

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    fetch = build_tools(["fetch_url"], scrape_url=args.scrape_url)[0]["fetch_url"]
    cfg = SummarizerConfig(model=args.model, context_window=args.context_window,
                           summary_tokens=args.summary_tokens)
    for path in (args.cache, args.summary_out):
        path.parent.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    lock = Lock()
    with open(args.cache, "w") as cache, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_question, row, client, args.model, fetch, cfg,
                               args.cache_dir, shapes, params) for row in rows]
        for future in tqdm(as_completed(futures), total=len(futures), desc="questions", unit="q"):
            recs = future.result()
            with lock:
                for rec in recs:
                    cache.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cache.flush()
            all_records.extend(recs)

    summary = summarize(all_records, shapes)
    args.summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nper-shape solve rate:")
    for s in shapes:
        ps = summary["per_shape"][s]
        print(f"  {s:<20} {ps['correct']}/{ps['total']}  ({ps['solve_rate']:.1%})")
    print(f"solved by any shape: {summary['solved_by_any']}/{summary['n_questions']} "
          f"({summary['solved_by_any_rate']:.1%})")
    print(f"cache -> {args.cache}\nsummary -> {args.summary_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path("datasets/frames_qwen_unanswerable.json"))
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--base-url", default="http://localhost:7472/v1", help="Local vLLM endpoint.")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--scrape-url", default=LOCAL_SCRAPE_URL,
                    help="fetch_url backend for cache misses (local Wikipedia cache by default).")
    ap.add_argument("--shapes", default=",".join(SHAPES),
                    help="Comma-separated subset of debate shapes to run.")
    ap.add_argument("--sample-n", type=int, default=5, help="Samples for sample_vote.")
    ap.add_argument("--debate-k", type=int, default=3, help="Personas for multi_agent_debate (max 3).")
    ap.add_argument("--context-window", type=int, default=32768)
    ap.add_argument("--summary-tokens", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=5, help="Questions processed concurrently.")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N questions.")
    ap.add_argument("--cache-dir", type=Path, default=Path("doc_context_cache"),
                    help="doc_context cache (must match cache_doc_summary's config).")
    ap.add_argument("--cache", type=Path, default=Path("outputs/debate_step_cache.jsonl"))
    ap.add_argument("--summary-out", type=Path, default=Path("outputs/debate_step_summary.json"))
    main(ap.parse_args())

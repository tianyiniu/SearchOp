"""(4) Test-time routing: can a zero-shot ranker pick the right debate shape?

The debate step (eval_debate_step.py) executes ALL four aggregation shapes on every
question — that is a train-time oracle you cannot run at test time. This script asks
the deployable question instead: given ONLY the question text (no execution, no ground
truth), can a ranker pick the single best shape to run?

We reuse the listwise-router machinery from gemini_router_example.py, but:
  - the candidate "schemas" are the 4 debate shapes (single_pass, self_refine,
    sample_vote, multi_agent_debate), each rendered as a short capability description;
  - the ranker is a LOCAL Qwen (the same vLLM used everywhere else) instead of Gemini,
    so routing is free.

Why no train/test split here: the ranker is zero-shot — it never sees any question's
correctness, so there is nothing fit on the data to leak. Every question in the pool is
a valid test point. (A split is only needed once we DISTILL a schema library from
observed outcomes; that is the next experiment, not this one.)

The pool is built straight from the debate cache: each (question, shape) record's
`correct` flag is the shape's answer_score. Metrics (from the router module):
  - router_top1 : accuracy if we execute only the ranker's #1 pick
  - random      : mean pool accuracy (pick a shape uniformly at random)
  - oracle      : best-shape-per-question (== solved_by_any)
  - recovery    : (router_top1 - random) / (oracle - random), the fraction of the
                  achievable routing gain the ranker captures

    # vLLM serving Qwen/Qwen3-14B on port 7472
    python3 scripts/route_debate.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import fmean

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Pure, dependency-free helpers from the router example (no google-genai needed).
from gemini_router_example import majority_top1, ndcg_at_k, parse_ranking, pass_at_k

try:  # load API keys from the project .env if python-dotenv is available
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


# --- the 4 debate shapes as routable candidates ----------------------------
# Order is fixed and used as the pool's schema index. Each description tells the
# ranker WHEN the shape tends to win, so it can match shape to question type.

SHAPES = ["single_pass", "self_refine", "sample_vote", "multi_agent_debate"]

SHAPE_TEXT = {
    "single_pass": (
        "single_pass — one direct answer from the documents, no aggregation "
        "(1 model call). Best when the answer is a single lookup the model will "
        "likely get right on the first try; wasteful reasoning only adds noise."),
    "self_refine": (
        "self_refine — draft an answer, a critic checks it strictly against the "
        "documents, then revise (3 calls). Best when first attempts make specific, "
        "checkable factual errors that a second look at the documents can catch."),
    "sample_vote": (
        "sample_vote — draw several independent answers and take the majority "
        "(5 calls, self-consistency). Best when the correct answer is within the "
        "model's range but unstable run-to-run; voting cancels the random slips."),
    "multi_agent_debate": (
        "multi_agent_debate — several personas answer, read each other, then a "
        "synthesizer commits (7 calls). Best for multi-hop questions whose answer "
        "must be assembled by combining evidence across several documents."),
}


def shape_block(order: list[int]) -> str:
    """Render the shapes numbered 1..K in the given presentation order."""
    return "\n\n".join(f"[Schema {label}]\n{SHAPE_TEXT[SHAPES[orig]]}"
                       for label, orig in enumerate(order, start=1))


ROUTER_SYSTEM = (
    "You route a question-answering agent to ONE aggregation strategy. The agent "
    "already has the relevant documents in context; your job is only to choose HOW "
    "it should reason over them. You are given a question and K candidate strategies "
    "and must RANK them from most- to least-likely to yield a CORRECT final answer.\n\n"
    "In `rationale`: (1) name 1-2 features of the question that should drive the "
    "choice (single-fact lookup vs multi-hop chaining vs numeric; is the answer likely "
    "unstable across runs?); (2) briefly weigh each strategy's fit; (3) justify the top "
    "pick. Be decisive even when strategies look similar — ties hurt the router.\n\n"
    "Respond with a SINGLE JSON object: `rationale` (string) and `ranking` (an array "
    "that is a permutation of 1..K, best first)."
)

ROUTER_USER = (
    "Question:\n{question}\n\n"
    "Candidate strategies (K = {k}):\n{block}\n\n"
    "Return JSON: {{\"rationale\": \"...\", \"ranking\": [best, ..., worst]}} "
    "where `ranking` is a permutation of 1..{k}."
)


# --- local Qwen ranker (drop-in for GeminiRanker.rank) ---------------------

class QwenRanker:
    """Listwise ranker backed by the local vLLM. Same `rank(...)->json_str` contract
    as GeminiRanker, so the routing loop is model-agnostic."""

    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def rank(self, system_prompt: str, user_prompt: str, k: int,
             temperature: float = 0.0, max_tokens: int = 1024) -> str:
        # Ask for a JSON object and turn OFF Qwen's <think> block so the response is
        # clean JSON. Both are best-effort: fall back if the server rejects them.
        for extra in ({"chat_template_kwargs": {"enable_thinking": False}}, {}):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": user_prompt}],
                    temperature=temperature, max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    extra_body=extra,
                )
                return resp.choices[0].message.content or ""
            except Exception:
                continue
        return ""


def rank_question(ranker: QwenRanker, question: str, seed: int = 42,
                  n_passes: int = 3, max_retries: int = 1) -> list[int]:
    """Rank the 4 shapes for one question over n_passes shuffled listwise calls.
    Returns pool indices best-first; index 0 is the routing pick."""
    k = len(SHAPES)
    per_pass: list[list[int] | None] = []
    for p in range(1, n_passes + 1):
        order = list(range(k))
        random.Random(seed ^ p).shuffle(order)  # deterministic per-pass shuffle
        user = ROUTER_USER.format(question=question, k=k, block=shape_block(order))
        ranking = None
        for attempt in range(max_retries + 1):
            raw = ranker.rank(ROUTER_SYSTEM, user, k,
                              temperature=0.0 if attempt == 0 else 0.4)
            ranking, _ = parse_ranking(raw, k)
            if ranking is not None:
                break
        # Map 1-indexed presentation ranking back to original pool indices.
        per_pass.append([order[label - 1] for label in ranking] if ranking else None)
    if all(r is None for r in per_pass):
        return list(range(k))  # total parse failure -> stable default order
    return majority_top1(per_pass, k)


# --- pool construction from the debate cache -------------------------------

def build_pool(cache_path: Path, full_dataset: Path) -> dict:
    """{id: {question, source, scores:[float x 4 aligned to SHAPES]}} — one entry per
    question that has an `ok` record for all 4 shapes."""
    by_q: dict[str, dict] = defaultdict(dict)
    questions: dict[str, str] = {}
    for line in cache_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") != "ok" or r.get("shape") not in SHAPES:
            continue
        by_q[r["id"]][r["shape"]] = 1.0 if r.get("correct") else 0.0
        questions[r["id"]] = r["question"]

    # Join reasoning types (the routing "source") from the full FRAMES set.
    reasoning: dict[str, str] = {}
    if full_dataset.exists():
        for row in json.loads(full_dataset.read_text()):
            rts = row.get("reasoning_types")
            if isinstance(rts, list):
                rts = "+".join(sorted(rts))
            reasoning[row.get("id")] = (rts or "unknown").strip() or "unknown"

    pool = {}
    for qid, shape_scores in by_q.items():
        if len(shape_scores) != len(SHAPES):  # need all shapes to route fairly
            continue
        pool[qid] = {"question": questions[qid],
                     "source": reasoning.get(qid, "unknown"),
                     "scores": [shape_scores[s] for s in SHAPES]}
    return pool


# --- evaluation ------------------------------------------------------------

def evaluate(pool: dict, ranker: QwenRanker, n_passes: int, seed: int, workers: int) -> dict:
    def _one(qid: str) -> dict:
        q = pool[qid]
        order = rank_question(ranker, q["question"], seed=seed ^ (hash(qid) & 0xFFFF),
                              n_passes=n_passes)
        scores = q["scores"]
        return {"id": qid, "source": q["source"], "picked": SHAPES[order[0]],
                "picked_correct": scores[order[0]],
                "pool_mean": fmean(scores), "pool_max": max(scores),
                "ndcg_at_2": ndcg_at_k([scores[i] for i in order], 2),
                "pass_at_2": pass_at_k([scores[i] for i in order], 2)}

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, qid): qid for qid in pool}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="route", unit="q"):
            rows.append(fut.result())

    def _safe_mean(vals: list[float]) -> float:
        vals = [v for v in vals if v == v]  # drop NaN (undefined NDCG on all-wrong Qs)
        return fmean(vals) if vals else float("nan")

    def _summ(rs: list[dict]) -> dict:
        rnd = fmean(r["pool_mean"] for r in rs)
        orc = fmean(r["pool_max"] for r in rs)
        top1 = fmean(r["picked_correct"] for r in rs)
        return {"n": len(rs), "random": rnd, "oracle": orc, "router_top1": top1,
                "recovery": (top1 - rnd) / (orc - rnd) if orc > rnd else float("nan"),
                "ndcg_at_2": _safe_mean([r["ndcg_at_2"] for r in rs]),
                "pass_at_2": _safe_mean([r["pass_at_2"] for r in rs])}

    # Best single FIXED shape (always pick shape X) — the baseline routing must beat.
    fixed = {s: fmean(pool[q]["scores"][i] for q in pool) for i, s in enumerate(SHAPES)}
    best_fixed = max(fixed, key=fixed.get)

    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    return {
        "overall": _summ(rows),
        "fixed_shape_accuracy": fixed,
        "best_fixed_shape": {"shape": best_fixed, "accuracy": fixed[best_fixed]},
        "router_pick_distribution": dict(Counter(r["picked"] for r in rows)),
        "by_source": {s: _summ(rs) for s, rs in sorted(by_source.items()) if len(rs) >= 3},
    }


def main(args: argparse.Namespace) -> None:
    pool = build_pool(args.cache, args.full_dataset)
    print(f"pool: {len(pool)} routable questions from {args.cache}")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    ranker = QwenRanker(client, args.model)
    summary = evaluate(pool, ranker, args.n_passes, args.seed, args.workers)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    o = summary["overall"]
    bf = summary["best_fixed_shape"]
    print(f"\n{'metric':<22}{'value':>10}")
    print(f"{'random (pick any)':<22}{o['random']:>10.1%}")
    print(f"{'best fixed shape':<22}{bf['accuracy']:>10.1%}   ({bf['shape']})")
    print(f"{'router top-1':<22}{o['router_top1']:>10.1%}")
    print(f"{'oracle (solved_by_any)':<22}{o['oracle']:>10.1%}")
    print(f"{'recovery':<22}{o['recovery']:>10.1%}")
    print(f"{'NDCG@2 / pass@2':<22}{o['ndcg_at_2']:>10.2f} / {o['pass_at_2']:.1%}")
    print(f"\nrouter picks: {summary['router_pick_distribution']}")
    print(f"summary -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=Path("outputs/debate_step_cache.jsonl"),
                    help="Debate-step per-(question,shape) cache -> the routing pool.")
    ap.add_argument("--full-dataset", type=Path, default=Path("datasets/frames_test_full.json"),
                    help="Source of reasoning_types for the by-source breakdown.")
    ap.add_argument("--model", default="Qwen/Qwen3-14B", help="Local ranker model (vLLM).")
    ap.add_argument("--base-url", default="http://localhost:7472/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--n-passes", type=int, default=3, help="Shuffled listwise passes per question.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("outputs/route_debate_summary.json"))
    main(ap.parse_args())

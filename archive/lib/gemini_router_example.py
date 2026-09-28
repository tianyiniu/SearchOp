"""Listwise LLM schema router (Gemini) — single-file edition.

Given a question and K candidate "schemas" (debate plans), pick the one most
likely to yield a correct answer WITHOUT executing any of them: ask a frontier
model (Gemini) to RANK the schemas for the question, and route to its top pick.

How it works, per question:
  1. Render each schema as compact text (rounds -> personas / tools / instruction
     + synthesis rule).
  2. Present the K schemas in a deterministic SHUFFLE (so the model can't just
     favour "schema 1" — position-bias mitigation).
  3. Ask Gemini for a JSON {rationale, ranking}, where `ranking` is a permutation
     of 1..K, best first. The shape is forced with Gemini's `response_schema`;
     the "valid permutation" check is done here in Python.
  4. With n_passes > 1, repeat 2-3 with different shuffles and pick the final
     top-1 by MAJORITY VOTE across the per-pass top-1s (Borda count only breaks
     ties / fills trailing positions used for NDCG / pass@k reporting).

A "schema" is a plain dict:
    {"rounds": [{"personas": [...], "tools": [...], "instruction": "..."}, ...],
     "final_synthesis": "..."}

Requires: pip install google-genai ; export GEMINI_API_KEY=...

    from gemini_router import GeminiRanker, route
    ranker = GeminiRanker(model="gemini-3.1-pro-preview")
    idx, schema = route(ranker, question, candidate_schemas, n_passes=3)
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from statistics import fmean


# ===========================================================================
# 1. Gemini client — forced-JSON ranking output
# ===========================================================================

class GeminiRanker:
    """Wraps the official google-genai SDK. `rank` uses Gemini's structured
    output (`response_schema`) to force a `{rationale, ranking}` JSON object,
    where `ranking` is a length-K integer array."""

    def __init__(self, model: str = "gemini-3.1-pro-preview", api_key: str | None = None) -> None:
        from google import genai
        self.model = model
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.client = genai.Client(api_key=key)

    def rank(self, system_prompt: str, user_prompt: str, k: int,
             temperature: float = 0.0, max_tokens: int = 4096) -> str:
        """Return raw JSON text: {"rationale": "...", "ranking": [int x K]}."""
        from google.genai import types
        response_schema = {
            "type": "object",
            "properties": {
                "rationale": {"type": "string"},
                "ranking": {"type": "array", "items": {"type": "integer"},
                            "minItems": k, "maxItems": k},
            },
            "required": ["rationale", "ranking"],
            "propertyOrdering": ["rationale", "ranking"],   # think before ranking
        }
        cfg = types.GenerateContentConfig(
            system_instruction=system_prompt, temperature=temperature,
            max_output_tokens=max_tokens, response_mime_type="application/json",
            response_schema=response_schema,
        )
        resp = self.client.models.generate_content(
            model=self.model, contents=user_prompt, config=cfg,
        )
        return resp.text or ""


# ===========================================================================
# 2. Render a schema dict as compact text for the model
# ===========================================================================

def schema_to_text(schema: dict) -> str:
    rounds = schema.get("rounds", []) or []
    lines = [f"{len(rounds)}-round debate, final synthesis: {schema.get('final_synthesis', '')}."]
    for i, r in enumerate(rounds, start=1):
        personas = ", ".join(r.get("personas") or [])
        tools = ", ".join(r.get("tools") or []) or "no tools"
        lines.append(f"Round {i}: personas=[{personas}] tools=[{tools}] "
                     f"instruction={r.get('instruction', '')}.")
    return "\n".join(lines)


# ===========================================================================
# 3. Listwise prompt + response parsing
# ===========================================================================

LISTWISE_SYSTEM = (
    "You are evaluating debate schemas for a question-answering agent. A schema "
    "is a high-level plan: how many rounds, which personas per round (analyst = "
    "explores hypotheses; critic = stress-tests them; synthesizer = "
    "consolidates), which tools per round (search_info = web search; fetch_url = "
    "download a URL; code_compute = run Python), the per-round instruction, and "
    "the final-synthesis rule.\n\n"
    "Given a question and K candidate schemas, RANK them from most- to "
    "least-likely to produce a CORRECT final answer when executed.\n\n"
    "In `rationale`: (1) note 1-2 features of the question that should drive "
    "selection (factual recall vs multi-hop chaining vs arithmetic; fresh vs "
    "stale knowledge); (2) briefly weigh each schema's tool fit, round depth, and "
    "persona mix for THIS question; (3) justify your top pick and your last. Be "
    "decisive even when schemas look similar — ties hurt the router.\n\n"
    "Respond with a SINGLE JSON object with fields `rationale` (string) and "
    "`ranking` (an array that is a permutation of 1..K, best first)."
)

LISTWISE_USER = (
    "Question:\n{question}\n\n"
    "Candidate schemas (K = {k}):\n{block}\n\n"
    "Return JSON: {{\"rationale\": \"...\", \"ranking\": [best, ..., worst]}} "
    "where `ranking` is a permutation of 1..{k}."
)


def render_block(schemas: list[dict], order: list[int]) -> str:
    """Render schemas numbered 1..K in the shuffled presentation `order`."""
    return "\n\n".join(
        f"[Schema {label}]\n{schema_to_text(schemas[orig])}"
        for label, orig in enumerate(order, start=1)
    )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_ranking(response: str, k: int) -> tuple[list[int] | None, str]:
    """Extract a length-K permutation of 1..K. Returns (ranking, rationale)."""
    if not response:
        return None, ""
    cleaned = re.sub(r"^```(?:json)?|```$", "", response.strip()).strip()
    try:
        obj = json.loads(cleaned)
    except Exception:
        m = _JSON_RE.search(cleaned)
        obj = json.loads(m.group(0)) if m else None
    if not isinstance(obj, dict):
        return None, response[:300]
    rationale = str(obj.get("rationale", "")).strip()
    try:
        ranking = [int(x) for x in obj.get("ranking", [])]
    except Exception:
        return None, rationale
    if len(ranking) != k or sorted(ranking) != list(range(1, k + 1)):
        return None, rationale
    return ranking, rationale


# ===========================================================================
# 4. Aggregation across passes (Borda + majority-vote-on-top-1)
# ===========================================================================

def borda_order(rankings: list[list[int] | None], k: int) -> list[int]:
    """Borda count over per-pass orderings (best-first lists of pool indices).
    Position p contributes K-1-p points. Returns indices best-first."""
    scores = [0] * k
    for r in rankings:
        if r:
            for pos, idx in enumerate(r):
                if 0 <= idx < k:
                    scores[idx] += k - 1 - pos
    return sorted(range(k), key=lambda i: (-scores[i], i))


def majority_top1(rankings: list[list[int] | None], k: int) -> list[int]:
    """Top-1 by majority vote across pass top-1s; the rest by Borda.

    Majority-vote-on-top-1 beats plain Borda as the routing decision because it
    keeps each pass's confident top pick instead of diluting it with noise from
    the middle of the ranking.
    """
    votes: dict[int, int] = {}
    for r in rankings:
        if r:
            votes[r[0]] = votes.get(r[0], 0) + 1
    order = borda_order(rankings, k)
    if not votes:
        return order
    borda_rank = {idx: pos for pos, idx in enumerate(order)}
    winner = min(votes, key=lambda i: (-votes[i], borda_rank[i], i))
    return [winner] + [i for i in order if i != winner]


# ===========================================================================
# 5. Rank / route
# ===========================================================================

@dataclass
class RankResult:
    order: list[int]                 # pool indices best-first (routing pick = order[0])
    rationales: list[str] = field(default_factory=list)
    per_pass_top1: list[int | None] = field(default_factory=list)
    n_parse_failed: int = 0


def rank_candidates(ranker: GeminiRanker, question: str, schemas: list[dict],
                    seed: int = 42, n_passes: int = 3, max_tokens: int = 4096,
                    max_retries: int = 2) -> RankResult:
    """Rank `schemas` for `question` over `n_passes` shuffled listwise calls."""
    k = len(schemas)
    if k <= 1:
        return RankResult(order=list(range(k)))

    per_pass_orig: list[list[int] | None] = []
    rationales: list[str] = []
    for p in range(1, n_passes + 1):
        order = list(range(k))
        random.Random(seed ^ p).shuffle(order)          # deterministic per-pass shuffle
        user = LISTWISE_USER.format(question=question, k=k, block=render_block(schemas, order))
        ranking, rationale = None, ""
        for attempt in range(max_retries + 1):
            try:
                raw = ranker.rank(LISTWISE_SYSTEM, user, k,
                                  temperature=0.0 if attempt == 0 else 0.3,
                                  max_tokens=max_tokens)
            except Exception:
                raw = ""
            ranking, rationale = parse_ranking(raw, k)
            if ranking is not None:
                break
        rationales.append(rationale)
        # Map the 1-indexed presentation ranking back to original pool indices.
        per_pass_orig.append([order[label - 1] for label in ranking] if ranking else None)

    n_failed = sum(1 for r in per_pass_orig if r is None)
    final = list(range(k)) if n_failed == n_passes else majority_top1(per_pass_orig, k)
    return RankResult(order=final, rationales=rationales,
                      per_pass_top1=[r[0] if r else None for r in per_pass_orig],
                      n_parse_failed=n_failed)


def route(ranker: GeminiRanker, question: str, schemas: list[dict],
          n_passes: int = 3, seed: int = 42) -> tuple[int, dict]:
    """Convenience: return (picked_index, picked_schema) for a question."""
    res = rank_candidates(ranker, question, schemas, seed=seed, n_passes=n_passes)
    return res.order[0], schemas[res.order[0]]


# ===========================================================================
# 6. Metrics + offline evaluation
# ===========================================================================
# To score the router you need a POOL: the candidate schemas per question, each
# with its known correctness (0/1) from having executed it:
#   {task_id: {"question": str, "source": str,
#              "schemas": [{"schema": <dict>, "answer_score": 0.0|1.0}, ...]}}

def _dcg(rels: list[float]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(scores_in_ranked_order: list[float], k: int) -> float:
    """NDCG@k with binary relevance (answer_score). NaN if no correct schema."""
    if not scores_in_ranked_order:
        return float("nan")
    dcg = _dcg(scores_in_ranked_order[:k])
    n_ideal = min(k, sum(1 for s in scores_in_ranked_order if s >= 1.0))
    if n_ideal == 0:
        return float("nan")
    return dcg / _dcg([1.0] * n_ideal)


def pass_at_k(scores_in_ranked_order: list[float], k: int) -> float:
    """1 if any of the top-k ranked schemas is correct, else 0."""
    if not scores_in_ranked_order:
        return float("nan")
    return float(max(scores_in_ranked_order[:k]) >= 1.0)


def evaluate_pool(ranker: GeminiRanker, pool: dict, n_passes: int = 3,
                  seed: int = 42, workers: int = 6) -> dict:
    """Route every question in `pool` and report top-1 accuracy + recovery,
    overall and per source. recovery = (top1 - random) / (oracle - random)."""
    def _one(tid: str) -> dict:
        q = pool[tid]
        schemas = [s["schema"] for s in q["schemas"]]
        scores = [float(s["answer_score"]) for s in q["schemas"]]
        res = rank_candidates(ranker, q["question"], schemas,
                              seed=seed ^ hash(tid), n_passes=n_passes)
        ranked_scores = [scores[i] for i in res.order]
        return {"source": q.get("source", ""), "picked_correct": scores[res.order[0]],
                "pool_mean": fmean(scores), "pool_max": max(scores),
                "ndcg_at_3": ndcg_at_k(ranked_scores, 3)}

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, tid): tid for tid in pool}
        for fut in as_completed(futs):
            rows.append(fut.result())

    def _summ(rs: list[dict]) -> dict:
        p1 = fmean(r["pool_mean"] for r in rs)
        oracle = fmean(r["pool_max"] for r in rs)
        top1 = fmean(r["picked_correct"] for r in rs)
        rec = (top1 - p1) / (oracle - p1) if oracle > p1 else float("nan")
        return {"n": len(rs), "random_pass_at_1": p1, "oracle_at_k": oracle,
                "router_top1": top1, "recovery": rec}

    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    return {"overall": _summ(rows),
            "by_source": {s: _summ(rs) for s, rs in sorted(by_source.items())}}


# ===========================================================================
# 7. Demo
# ===========================================================================

if __name__ == "__main__":
    # Three candidate plans for one question.
    SCHEMAS = [
        {"rounds": [{"personas": ["analyst"], "tools": ["search_info", "fetch_url"],
                     "instruction": "independently_research"}],
         "final_synthesis": "last_persona"},
        {"rounds": [
            {"personas": ["analyst"], "tools": ["search_info", "fetch_url"], "instruction": "gather_facts"},
            {"personas": ["critic"], "tools": ["search_info"], "instruction": "verify_and_check"},
            {"personas": ["synthesizer"], "tools": [], "instruction": "produce_final_answer"}],
         "final_synthesis": "synthesizer_persona"},
        {"rounds": [{"personas": ["analyst"], "tools": ["code_compute"],
                     "instruction": "think_and_plan"}],
         "final_synthesis": "last_persona"},
    ]
    QUESTION = ("Who composed the music for the film that Linus Roache starred in "
                "immediately after Priest (1994)?")

    ranker = GeminiRanker(model="gemini-3.1-pro-preview")
    idx, picked = route(ranker, QUESTION, SCHEMAS, n_passes=3)
    print(f"Routed to schema #{idx}: {picked['rounds']}")

    res = rank_candidates(ranker, QUESTION, SCHEMAS, n_passes=3)
    print("Full ranking (best->worst):", res.order)
    print("Per-pass top-1 votes:", res.per_pass_top1)

    pool = {"q1": {"question": QUESTION, "source": "demo",
                   "schemas": [{"schema": s, "answer_score": sc}
                               for s, sc in zip(SCHEMAS, [0.0, 1.0, 0.0])]}}
    print("Eval:", evaluate_pool(ranker, pool, n_passes=3)["overall"])

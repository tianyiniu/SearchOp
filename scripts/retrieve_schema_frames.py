"""Test-time schema selection by retrieval, OOD arm (FRAMES open-ended QA).

Same retrieval front-end as retrieve_schema_mcq.py -- embed the query, pull the
top-k most similar SOLVED train queries + their winning schemas, and have an
architect synthesize ONE tailored schema -- but a DIFFERENT back-end, because
FRAMES is open-ended multi-hop QA, not MCQ:

  executor : evolve_debate.execute_schema -- personas answer in FREE TEXT over the
             gold DOCUMENT CONTEXT (not from an option list). This is the exact
             executor the existing FRAMES debate/evolution numbers were produced
             with, so these OOD results are comparable to them.
  grading  : llm_judge.judge_answer(question, ground_truth, predicted_text) -- no
             letters, no exact match.
  context  : loaded from doc_context_cache/ (build_and_cache_context's cache). All
             11 FRAMES-OOD contexts are already cached & verified, so a run makes
             ZERO Serper/doc-pipeline calls -- only Qwen executor + GPT judge/synth.

This is a DOUBLE distribution shift from the corpus (MCQ->QA in format, and
parametric->document-grounded in reasoning): a deliberately strong OOD test of
whether schema structure discovered on SuperGPQA transfers at all. n=11, and ~2 of
those are known judge errors -- treat it as a qualitative smoke test, not a metric.

The index is the SAME one retrieve_schema_mcq.py builds (it already pre-embedded
the FRAMES queries), so build it there first, then run arms here. Per-arm cache
files + --summarize-only work identically, for tmux fan-out.

    # index already built by: retrieve_schema_mcq.py --build-index
    python3 scripts/retrieve_schema_frames.py --arms retrieve_synth --workers 8
    python3 scripts/retrieve_schema_frames.py --arms always_critic  --workers 8
    python3 scripts/retrieve_schema_frames.py --summarize-only
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from threading import Lock

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts

import debate_mcq as D
import evolve_debate as E                      # FRAMES doc-grounded executor
import retrieve_schema_mcq as RM               # shared retrieval front-end
from doc_qa import SummarizerConfig, load_cached_context
from llm_judge import call_openai, judge_answer

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


# --- FRAMES schema architect (no options; document-grounded framing) --------

FRAMES_SYNTH_SYSTEM = f"""You are a DEBATE-SCHEMA ARCHITECT. A blind executor answers a hard multi-hop factual question using PROVIDED DOCUMENTS by running a "schema": an ordered list of rounds. In each round the listed personas answer in turn, each seeing the documents, the question, and all prior responses; a final rule then commits one answer.

You are given a NEW question and, as guidance, a few examples of hard questions together with the debate schema that worked for each. Design ONE schema, tailored to the new question, that gives the executor the best chance to reach the correct answer ON ITS OWN.

DO NOT ANSWER THE QUESTION AND DO NOT LEAK FACTS. Choose STRUCTURE ONLY, from the fixed vocabulary below.

VOCABULARY
  personas: {list(D.PERSONAS)}  (solver=answer from the documents, critic=fault the prior answers, synthesizer=reconcile and commit one answer)
  final: {list(D.FINALS)}  ('synthesizer' requires a synthesizer in the LAST round; 'vote' takes the majority answer of the last round; 'last' takes the last persona's answer)
  limits: 1..{D.MAX_ROUNDS} rounds, 1..{D.MAX_PERSONAS} personas per round.

Design guidance: single-fact lookups often need only a solver, or a solver then a critic; multi-hop questions that chain several facts benefit from an added critic to check the chain, or a synthesizer to reconcile partial answers. Do NOT over-build -- extra rounds cost reliability.

Respond with STRICT JSON only, no prose:
{{"rounds":[{{"personas":["solver"]}}, {{"personas":["critic"]}}], "final":"last"}}"""


def frames_synth_user(question: str, neighbors: list[dict]) -> str:
    ex = []
    for i, nb in enumerate(neighbors, 1):
        ex.append(f"EXAMPLE {i} (similarity {nb.get('score', 0):.2f})\n"
                  f"  similar question: {nb['question'][:400]}\n"
                  f"  schema that solved it: {json.dumps(nb['schema'])}\n"
                  f"  ({RM.schema_gloss(nb['schema'])})")
    return (f"NEW QUESTION:\n{question}\n\n"
            f"SIMILAR SOLVED QUESTIONS AND THEIR WINNING SCHEMAS:\n" + "\n\n".join(ex) +
            "\n\nDesign ONE schema for the NEW question as the JSON object specified.")


def frames_synthesize(question, neighbors, model, retries=2) -> tuple[dict, bool]:
    for _ in range(retries):
        raw = call_openai(FRAMES_SYNTH_SYSTEM, frames_synth_user(question, neighbors), model=model)
        sch = RM.parse_schema(raw)
        if sch is not None:
            return sch, False
    return D.self_critique_schema(), True


# --- context loading -------------------------------------------------------

def load_context(qid: str, cfg: SummarizerConfig, cache_dir: str) -> str:
    d = load_cached_context(qid, cfg, cache_dir)
    if d and d.get("doc_context"):
        return d["doc_context"]
    files = sorted(glob.glob(str(Path(cache_dir) / f"{qid}__*.json")))  # any-config fallback
    if files:
        return json.loads(Path(files[-1]).read_text()).get("doc_context", "")
    return ""


# --- per-arm schema selection (same grammar; FRAMES synth/retrieval) --------

def select_schema(arm: str, row: dict, ctx: dict) -> tuple[dict, dict]:
    if arm == "single_pass":
        return deepcopy(D.MINIMAL_SCHEMA), {}
    if arm == "self_critique":
        return D.self_critique_schema(), {}
    if arm == "fixed_debate":
        return D.fixed_debate_schema(ctx["args"].debate_k), {}
    if arm == "always_critic":
        return deepcopy(RM.ALWAYS_CRITIC), {}

    index: RM.Index = ctx["index"]
    nbrs = index.topk(row["id"], row["question"], ctx["args"].top_k)
    if arm == "retrieve_copy":
        schema = deepcopy(nbrs[0]["schema"]) if nbrs else deepcopy(D.MINIMAL_SCHEMA)
        return schema, {"retrieved": [n["id"] for n in nbrs[:1]],
                        "retrieved_scores": [round(n["score"], 3) for n in nbrs[:1]]}
    if arm == "retrieve_synth":
        schema, fb = frames_synthesize(row["question"], nbrs, ctx["args"].synth_model)
        return schema, {"retrieved": [n["id"] for n in nbrs],
                        "retrieved_scores": [round(n["score"], 3) for n in nbrs],
                        "synth_fallback": fb}
    raise KeyError(arm)


def run_task(row: dict, arm: str, ctx: dict) -> dict:
    a = ctx["args"]
    rec = {"id": row["id"], "arm": arm, "question": row["question"],
           "ground_truth": row["ground_truth"]}
    try:
        context = ctx["contexts"].get(row["id"], "")
        if not context:
            rec.update(status="skipped_no_context")
            return rec
        schema, meta = select_schema(arm, row, ctx)
        answer = E.execute_schema(ctx["client"], a.model, row["question"], context,
                                  schema, a.exec_temp)
        rec.update(schema=schema, answer=answer.strip()[:500],
                   correct=judge_answer(row["question"], row["ground_truth"], answer),
                   status="ok", **meta)
    except Exception as exc:  # keep the pool alive
        rec.update(status="error", error=str(exc))
    return rec


# --- summary / report ------------------------------------------------------

def summarize_arm(records: list[dict], arm: str) -> dict:
    ok = [r for r in records if r["status"] == "ok"]
    n = len(ok)
    correct = sum(1 for r in ok if r["correct"])
    out = {"arm": arm, "n": n, "correct": correct,
           "solve_rate": (correct / n) if n else 0.0,
           "skipped_no_context": sum(1 for r in records if r["status"] == "skipped_no_context"),
           "errors": sum(1 for r in records if r["status"] == "error"),
           "solved_ids": [r["id"] for r in ok if r["correct"]]}
    if arm in ("retrieve_synth", "retrieve_copy"):
        out["schema_shapes"] = dict(Counter(RM.schema_gloss(r["schema"]) for r in ok).most_common(10))
    if arm == "retrieve_synth":
        out["synth_fallback_rate"] = (sum(1 for r in ok if r.get("synth_fallback")) / n) if n else 0.0
    return out


def report(by_arm: dict[str, list[dict]], arms: list[str], args, tag: str) -> None:
    print("\nsolve rate by arm:")
    summaries = {}
    for a in arms:
        s = summarize_arm(by_arm[a], a)
        summaries[a] = s
        extra = f"  (fallback {s['synth_fallback_rate']:.0%})" if a == "retrieve_synth" else ""
        print(f"  {a:<16} {s['correct']}/{s['n']}  ({s['solve_rate']:.1%}){extra}  solved={s['solved_ids']}")
        (args.outdir / f"retrieve_{a}_{tag}_summary.json").write_text(
            json.dumps(s, indent=2, ensure_ascii=False))

    pairs = [("retrieve_synth", "self_critique"), ("retrieve_synth", "always_critic"),
             ("retrieve_synth", "retrieve_copy"), ("retrieve_synth", "single_pass")]
    h2h = [hh for p in pairs if (hh := RM.head_to_head(by_arm, *p))]
    if h2h:
        print("\nhead-to-head (common questions):")
        for hh in h2h:
            onlys = [k for k in hh if k.startswith("only_")]
            print(f"  {hh['pair']:<34} both={hh['both']}  "
                  f"{onlys[0]}={hh[onlys[0]]}  {onlys[1]}={hh[onlys[1]]}  neither={hh['neither']}")
    if len(arms) > 1:
        combined = {"split": tag, "exec_temp": args.exec_temp,
                    "arms": {a: {"correct": summaries[a]["correct"], "n": summaries[a]["n"],
                                 "solve_rate": summaries[a]["solve_rate"]} for a in arms},
                    "head_to_head": h2h}
        out = args.outdir / f"retrieve_combined_{tag}_summary.json"
        out.write_text(json.dumps(combined, indent=2, ensure_ascii=False))
        print(f"\ncombined summary -> {out}")


# --- driver ----------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    tag = RM.split_tag(args.dataset)
    if args.summarize_only:
        by_arm = {a: RM.load_cache(args.outdir / f"retrieve_{a}_{tag}_cache.jsonl") for a in RM.ALL_ARMS}
        present = [a for a in RM.ALL_ARMS if by_arm[a]]
        if not present:
            raise SystemExit(f"no retrieve_*_{tag}_cache.jsonl in {args.outdir}")
        print(f"summarize-only: caches for {present} (split={tag})")
        report(defaultdict(list, {a: by_arm[a] for a in present}), present, args, tag)
        return

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in RM.ALL_ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {RM.ALL_ARMS}")

    rows = json.loads(Path(args.dataset).read_text())
    if args.limit is not None:
        rows = rows[: args.limit]

    cfg = SummarizerConfig()
    contexts = {r["id"]: load_context(r["id"], cfg, args.cache_dir) for r in rows}
    missing = [i for i, c in contexts.items() if not c]
    if missing:
        print(f"WARNING: {len(missing)} rows have no cached context (will be skipped): {missing}")

    ctx = {"args": args, "client": OpenAI(base_url=args.base_url, api_key=args.api_key),
           "contexts": contexts}
    if any(a in RM.NEEDS_INDEX for a in arms):
        if not args.index_out.exists():
            raise SystemExit(f"index {args.index_out} missing -- build it via "
                             f"retrieve_schema_mcq.py --build-index")
        ctx["index"] = RM.Index(args.index_out)
        print(f"index loaded: {len(ctx['index'].corpus)} exemplars, model={ctx['index'].embed_model}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    cache_paths = {a: args.outdir / f"retrieve_{a}_{tag}_cache.jsonl" for a in arms}
    caches = {a: open(cache_paths[a], "w") for a in arms}

    tasks = [(row, arm) for row in rows for arm in arms]
    print(f"{len(rows)} questions x {len(arms)} arms = {len(tasks)} tasks | arms={arms} | "
          f"split={tag} | exec_temp={args.exec_temp} | workers={args.workers}")

    by_arm: dict[str, list[dict]] = defaultdict(list)
    lock = Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_task, row, arm, ctx) for row, arm in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="frames", unit="task"):
            rec = fut.result()
            with lock:
                caches[rec["arm"]].write(json.dumps(rec, ensure_ascii=False) + "\n")
                caches[rec["arm"]].flush()
            by_arm[rec["arm"]].append(rec)
    for f in caches.values():
        f.close()

    report(by_arm, arms, args, tag)
    for a in arms:
        print(f"cache[{a}] -> {cache_paths[a]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="retrieve_synth",
                    help=f"Comma-separated arms. Choices: {RM.ALL_ARMS}")
    ap.add_argument("--summarize-only", action="store_true",
                    help="Rebuild summaries + head-to-head from existing per-arm caches.")
    ap.add_argument("--dataset", type=Path, default=Path("datasets/frames_ood_test.json"))
    ap.add_argument("--cache-dir", default="doc_context_cache",
                    help="Where build_and_cache_context stored the gold doc contexts.")
    ap.add_argument("--index-out", type=Path, default=Path("outputs/schema_retrieval_index.json"),
                    help="Shared index built by retrieve_schema_mcq.py --build-index.")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--synth-model", default="gpt-5.4-mini")
    ap.add_argument("--model", default="Qwen/Qwen3-14B", help="Executor (vLLM).")
    ap.add_argument("--base-url", default="http://localhost:7472/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--debate-k", type=int, default=3)
    ap.add_argument("--exec-temp", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--outdir", type=Path, default=Path("outputs"))
    main(ap.parse_args())

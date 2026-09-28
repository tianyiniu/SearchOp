"""Test-time schema selection by retrieval (SuperGPQA MCQ).

Turns the GT-aware evolution corpus into a TEST-TIME method. Offline we have, for
each solved TRAIN query, the schema that solved it (evolve_mcq_full_cache.jsonl).
At test time, for a held-out query with NO ground truth, we:

    1. embed the query, retrieve the top-k most similar SOLVED train queries,
    2. show a schema architect the query + those (similar question -> winning
       schema) exemplars, and ask it to design ONE tailored schema,
    3. execute that ONE schema once (thinking off) and grade by exact letter.

This is a genuinely 1-shot method (one schema, one execution) -- the same budget
as a fixed template and strictly fairer than evolution (up to 7 executions). The
architect is behind the information firewall: it sees only question TEXT and
exemplar SCHEMAS, never any answer, and may emit only grammar structure.

ARMS (each is one schema-selection policy; all executed + graded identically):
    single_pass    minimal single-solver schema            (the ~10-12% floor)
    self_critique  solver -> critic -> solver (fixed)       (the fixed bar to beat)
    fixed_debate   k solvers x2 -> synthesizer (fixed)
    always_critic  solver -> critic  (the MODAL winning schema; decisive control:
                   if this ties retrieval, similarity/indexing adds nothing)
    retrieve_copy  copy the nearest solved query's schema   (retrieval, no synthesis)
    retrieve_synth THE METHOD: retrieve top-k, architect synthesizes a schema

Execution temperature is UNIFORM across arms (--exec-temp, default 0.7, matching
how the corpus schemas were discovered), so the schema is the only thing that
varies -- self_critique here may read a hair off its 0.3 measurement, by design.

PARALLELISM / tmux fan-out: every arm writes its OWN cache+summary files
(outputs/retrieve_<arm>_<split>_{cache.jsonl,summary.json}), so you can launch one
arm per tmux with no collisions; within a process, tasks are the (question x arm)
grid over a thread pool. Retrieval arms need the index built once first.

    # 0) build the retrieval index ONCE (embeds train corpus + test + ood queries)
    python3 scripts/retrieve_schema_mcq.py --build-index

    # 1) then fan out, one arm per tmux, all on the 1000-question test split
    python3 scripts/retrieve_schema_mcq.py --arms retrieve_synth --workers 16
    python3 scripts/retrieve_schema_mcq.py --arms always_critic  --workers 16
    python3 scripts/retrieve_schema_mcq.py --arms self_critique  --workers 16
    ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from threading import Lock

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import debate_mcq as D
from llm_judge import call_openai

try:  # load API keys from the project .env if python-dotenv is available
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


# --- arm registry ----------------------------------------------------------

ALWAYS_CRITIC = {"rounds": [{"personas": ["solver"]}, {"personas": ["critic"]}], "final": "last"}
NEEDS_INDEX = {"retrieve_copy", "retrieve_synth"}
ALL_ARMS = ["single_pass", "self_critique", "fixed_debate",
            "always_critic", "retrieve_copy", "retrieve_synth"]


# --- schema architect (full-schema synthesis; firewall = grammar only) ------

SYNTH_SYSTEM = f"""You are a DEBATE-SCHEMA ARCHITECT. A blind executor answers a hard graduate multiple-choice question by running a "schema": an ordered list of rounds. In each round the listed personas answer in turn, each seeing the question and all prior responses; a final rule then commits exactly one option.

You are given a NEW question and, as guidance, a few SIMILAR questions together with the schema that worked for each. Design ONE schema, tailored to the new question, that gives the executor the best chance to reach the correct option ON ITS OWN.

DO NOT SOLVE THE QUESTION AND DO NOT LEAK AN ANSWER. Choose STRUCTURE ONLY, from the fixed vocabulary below. Never name an option or assert a fact.

VOCABULARY
  personas: {list(D.PERSONAS)}  (solver=answer from own knowledge, critic=fault the prior answers, synthesizer=reconcile and commit one option)
  final: {list(D.FINALS)}  ('synthesizer' requires a synthesizer persona in the LAST round; 'vote' takes the majority letter of the last round; 'last' takes the last persona's letter)
  limits: 1..{D.MAX_ROUNDS} rounds, 1..{D.MAX_PERSONAS} personas per round.

Design guidance: simple factual recall often needs only a solver, or a solver then a critic; multi-step or easily-miscalculated questions benefit from an added critic to check the deduction, or a synthesizer to reconcile diverging attempts. Do NOT over-build -- extra rounds cost reliability.

Respond with STRICT JSON only, no prose:
{{"rounds":[{{"personas":["solver"]}}, {{"personas":["critic"]}}], "final":"last"}}"""


def schema_gloss(schema: dict) -> str:
    rounds = "; ".join("+".join(r["personas"]) for r in schema["rounds"])
    return f"{len(schema['rounds'])} rounds [{rounds}] -> final={schema['final']}"


def synth_user_prompt(question: str, options: list[str], neighbors: list[dict]) -> str:
    ex = []
    for i, nb in enumerate(neighbors, 1):
        ex.append(f"EXAMPLE {i} (similarity {nb.get('score', 0):.2f})\n"
                  f"  similar question: {nb['question'][:400]}\n"
                  f"  schema that solved it: {json.dumps(nb['schema'])}\n"
                  f"  ({schema_gloss(nb['schema'])})")
    return (f"NEW QUESTION:\n{D.render_question(question, options)}\n\n"
            f"SIMILAR SOLVED QUESTIONS AND THEIR WINNING SCHEMAS:\n" + "\n\n".join(ex) +
            "\n\nDesign ONE schema for the NEW question as the JSON object specified.")


_SCHEMA_RE = D._JSON_RE  # reuse the greedy {...} matcher


def parse_schema(raw: str) -> dict | None:
    """Extract a grammar-valid {'rounds':..,'final':..} object, or None."""
    raw = D.re.sub(r"^```(?:json)?|```$", "", (raw or "").strip()).strip()
    obj = None
    try:
        obj = json.loads(raw)
    except Exception:
        m = _SCHEMA_RE.search(raw)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if isinstance(obj, dict):
        obj = {"rounds": obj.get("rounds"), "final": obj.get("final")}  # drop stray keys
        if D.validate(obj):
            return obj
    return None


def synthesize(question, options, neighbors, model, retries=2) -> tuple[dict, bool]:
    """Ask the architect for a schema; validate; fall back to self_critique.
    Returns (schema, used_fallback)."""
    for _ in range(retries):
        raw = call_openai(SYNTH_SYSTEM, synth_user_prompt(question, options, neighbors), model=model)
        sch = parse_schema(raw)
        if sch is not None:
            return sch, False
    return D.self_critique_schema(), True  # strongest fixed template as the fallback


# --- retrieval index -------------------------------------------------------

def _embed_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set (needed for embeddings)")
    return OpenAI(api_key=key)


def embed_texts(texts: list[str], model: str, batch: int = 128) -> list[list[float]]:
    client = _embed_client()
    out: list[list[float]] = []
    for i in tqdm(range(0, len(texts), batch), desc="embed", unit="batch"):
        chunk = [t.replace("\n", " ")[:8000] for t in texts[i:i + batch]]
        resp = client.embeddings.create(model=model, input=chunk)
        out.extend(d.embedding for d in resp.data)
    return out


def build_index(args) -> None:
    corpus_rows = json.loads(Path(args.corpus).read_text())
    evo = {}
    for line in open(args.evo_cache):
        r = json.loads(line)
        evo[r["id"]] = r

    exemplars = []
    for r in corpus_rows:
        e = evo.get(r["id"])
        if not e or not e.get("solved"):
            continue
        if args.nontrivial_only and (e.get("solved_at") or 0) == 0:
            continue
        exemplars.append({"id": r["id"], "question": r["question"],
                          "field": r.get("field"), "difficulty": r.get("difficulty"),
                          "schema": e["final_schema"], "solved_at": e.get("solved_at"),
                          "n_mods": e.get("n_mods")})
    print(f"corpus: {len(corpus_rows)} train rows -> {len(exemplars)} solved exemplars "
          f"({'non-trivial only' if args.nontrivial_only else 'all solved'})")

    embs = embed_texts([x["question"] for x in exemplars], args.embed_model)
    for x, v in zip(exemplars, embs):
        x["emb"] = v

    # pre-embed the eval queries so arm runs need no OpenAI embedding calls
    queries: dict[str, str] = {}
    for ds in args.embed_datasets:
        for r in json.loads(Path(ds).read_text()):
            queries.setdefault(r["id"], r["question"])
    qids = list(queries)
    qembs = embed_texts([queries[i] for i in qids], args.embed_model) if qids else []
    query_emb = {i: v for i, v in zip(qids, qembs)}

    args.index_out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.index_out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"embed_model": args.embed_model, "corpus": exemplars,
                               "queries": query_emb}, ensure_ascii=False))
    tmp.replace(args.index_out)  # atomic: safe even if arms are already fanning out
    print(f"index -> {args.index_out}  (corpus={len(exemplars)}, eval_queries={len(query_emb)})")


class Index:
    """In-memory retrieval index: cosine over normalized corpus embeddings."""

    def __init__(self, path: Path):
        if np is None:
            raise RuntimeError("numpy is required for retrieval arms")
        blob = json.loads(path.read_text())
        self.embed_model = blob["embed_model"]
        self.corpus = blob["corpus"]
        self.ids = [c["id"] for c in self.corpus]
        self.id_pos = {i: p for p, i in enumerate(self.ids)}
        M = np.asarray([c["emb"] for c in self.corpus], dtype="float32")
        self.M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
        self.queries = {i: np.asarray(v, dtype="float32") for i, v in blob["queries"].items()}
        self._lock = Lock()

    def _vec(self, qid: str, question: str) -> "np.ndarray":
        v = self.queries.get(qid)
        if v is None:  # not pre-embedded (unusual): embed once, cache in-process
            with self._lock:
                v = self.queries.get(qid)
                if v is None:
                    v = np.asarray(embed_texts([question], self.embed_model)[0], dtype="float32")
                    self.queries[qid] = v
        return v / (np.linalg.norm(v) + 1e-8)

    def topk(self, qid: str, question: str, k: int) -> list[dict]:
        scores = self.M @ self._vec(qid, question)
        if qid in self.id_pos:  # never retrieve the query itself
            scores[self.id_pos[qid]] = -1e9
        idx = np.argsort(-scores)[:k]
        out = []
        for j in idx:
            c = self.corpus[int(j)]
            out.append({"id": c["id"], "question": c["question"], "schema": c["schema"],
                        "solved_at": c["solved_at"], "score": float(scores[int(j)])})
        return out


# --- per-arm schema selection ----------------------------------------------

def select_schema(arm: str, row: dict, ctx: dict) -> tuple[dict, dict]:
    """Return (schema, meta) for one arm on one row. No answer is ever used here."""
    if arm == "single_pass":
        return deepcopy(D.MINIMAL_SCHEMA), {}
    if arm == "self_critique":
        return D.self_critique_schema(), {}
    if arm == "fixed_debate":
        return D.fixed_debate_schema(ctx["args"].debate_k), {}
    if arm == "always_critic":
        return deepcopy(ALWAYS_CRITIC), {}

    index: Index = ctx["index"]
    nbrs = index.topk(row["id"], row["question"], ctx["args"].top_k)
    if arm == "retrieve_copy":
        schema = deepcopy(nbrs[0]["schema"]) if nbrs else deepcopy(D.MINIMAL_SCHEMA)
        return schema, {"retrieved": [n["id"] for n in nbrs[:1]],
                        "retrieved_scores": [round(n["score"], 3) for n in nbrs[:1]]}
    if arm == "retrieve_synth":
        schema, fb = synthesize(row["question"], list(row["options"]), nbrs,
                                ctx["args"].synth_model)
        return schema, {"retrieved": [n["id"] for n in nbrs],
                        "retrieved_scores": [round(n["score"], 3) for n in nbrs],
                        "synth_fallback": fb}
    raise KeyError(arm)


def run_task(row: dict, arm: str, ctx: dict) -> dict:
    a = ctx["args"]
    rec = {"id": row["id"], "arm": arm, "field": row.get("field"),
           "difficulty": row.get("difficulty"), "answer_letter": row["answer_letter"]}
    try:
        schema, meta = select_schema(arm, row, ctx)
        letter = D.execute_schema(ctx["client"], a.model, row["question"],
                                  list(row["options"]), schema, a.exec_temp, a.answer_tokens)
        rec.update(schema=schema, answer=letter,
                   correct=(letter == row["answer_letter"]), status="ok", **meta)
    except Exception as exc:  # keep the pool alive
        rec.update(status="error", error=str(exc))
    return rec


# --- summary ---------------------------------------------------------------

def summarize_arm(records: list[dict], arm: str) -> dict:
    ok = [r for r in records if r["status"] == "ok"]
    n = len(ok)
    correct = sum(1 for r in ok if r["correct"])
    by_diff = {}
    for d in sorted({r["difficulty"] for r in ok if r["difficulty"]}):
        sub = [r for r in ok if r["difficulty"] == d]
        by_diff[d] = {"n": len(sub), "solve_rate": sum(r["correct"] for r in sub) / len(sub)}
    by_field = {}
    for f in sorted({r["field"] for r in ok if r["field"]}):
        sub = [r for r in ok if r["field"] == f]
        if len(sub) >= 10:
            by_field[f] = {"n": len(sub), "solve_rate": sum(r["correct"] for r in sub) / len(sub)}
    out = {"arm": arm, "n": n, "correct": correct,
           "solve_rate": (correct / n) if n else 0.0,
           "errors": sum(1 for r in records if r["status"] == "error"),
           "by_difficulty": by_diff, "by_field": by_field}
    if arm == "retrieve_synth":
        out["synth_fallback_rate"] = (sum(1 for r in ok if r.get("synth_fallback")) / n) if n else 0.0
        out["schema_shapes"] = dict(Counter(schema_gloss(r["schema"]) for r in ok).most_common(10))
    if arm == "retrieve_copy":
        out["schema_shapes"] = dict(Counter(schema_gloss(r["schema"]) for r in ok).most_common(10))
    return out


def head_to_head(by_arm: dict[str, list[dict]], a: str, b: str) -> dict | None:
    """Contingency of a vs b on their common solved-status question set."""
    if a not in by_arm or b not in by_arm:
        return None
    ca = {r["id"]: r["correct"] for r in by_arm[a] if r["status"] == "ok"}
    cb = {r["id"]: r["correct"] for r in by_arm[b] if r["status"] == "ok"}
    common = ca.keys() & cb.keys()
    both = only_a = only_b = neither = 0
    for i in common:
        if ca[i] and cb[i]:
            both += 1
        elif ca[i]:
            only_a += 1
        elif cb[i]:
            only_b += 1
        else:
            neither += 1
    return {"pair": f"{a}_vs_{b}", "n_common": len(common), "both": both,
            f"only_{a}": only_a, f"only_{b}": only_b, "neither": neither}


# --- driver ----------------------------------------------------------------

def split_tag(dataset: Path) -> str:
    stem = dataset.stem
    for pre in ("supergpqa_", "frames_"):
        stem = stem.replace(pre, "")
    return stem


def load_cache(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path)] if path.exists() else []


def report(by_arm: dict[str, list[dict]], arms: list[str], args, tag: str) -> None:
    """Per-arm summaries + head-to-heads + combined file. Shared by run and --summarize-only."""
    print("\nsolve rate by arm:")
    summaries = {}
    for a in arms:
        s = summarize_arm(by_arm[a], a)
        summaries[a] = s
        extra = f"  (fallback {s.get('synth_fallback_rate', 0):.1%})" if a == "retrieve_synth" else ""
        print(f"  {a:<16} {s['correct']}/{s['n']}  ({s['solve_rate']:.1%}){extra}")
        (args.outdir / f"retrieve_{a}_{tag}_summary.json").write_text(
            json.dumps(s, indent=2, ensure_ascii=False))

    pairs = [("retrieve_synth", "self_critique"), ("retrieve_synth", "always_critic"),
             ("retrieve_synth", "retrieve_copy"), ("retrieve_synth", "single_pass")]
    h2h = [hh for p in pairs if (hh := head_to_head(by_arm, *p))]
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


def main(args: argparse.Namespace) -> None:
    if args.build_index:
        build_index(args)
        return

    tag = split_tag(args.dataset)
    if args.summarize_only:  # rebuild the comparison from per-arm caches (tmux fan-out)
        by_arm = {a: load_cache(args.outdir / f"retrieve_{a}_{tag}_cache.jsonl") for a in ALL_ARMS}
        present = [a for a in ALL_ARMS if by_arm[a]]
        if not present:
            raise SystemExit(f"no retrieve_*_{tag}_cache.jsonl files in {args.outdir}")
        print(f"summarize-only: found caches for {present} (split={tag})")
        report(defaultdict(list, {a: by_arm[a] for a in present}), present, args, tag)
        return

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ALL_ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {ALL_ARMS}")

    rows = json.loads(Path(args.dataset).read_text())
    if args.limit is not None:
        rows = rows[: args.limit]

    ctx = {"args": args, "client": OpenAI(base_url=args.base_url, api_key=args.api_key)}
    if any(a in NEEDS_INDEX for a in arms):
        if not args.index_out.exists():
            raise SystemExit(f"index {args.index_out} missing -- run with --build-index first")
        ctx["index"] = Index(args.index_out)
        print(f"index loaded: {len(ctx['index'].corpus)} exemplars, "
              f"model={ctx['index'].embed_model}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    # one cache file handle per arm -> tmux-safe isolation
    cache_paths = {a: args.outdir / f"retrieve_{a}_{tag}_cache.jsonl" for a in arms}
    caches = {a: open(cache_paths[a], "w") for a in arms}

    tasks = [(row, arm) for row in rows for arm in arms]
    print(f"{len(rows)} questions x {len(arms)} arms = {len(tasks)} tasks | "
          f"arms={arms} | split={tag} | exec_temp={args.exec_temp} | workers={args.workers}")

    by_arm: dict[str, list[dict]] = defaultdict(list)
    lock = Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_task, row, arm, ctx) for row, arm in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="retrieve", unit="task"):
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
    # index build
    ap.add_argument("--build-index", action="store_true",
                    help="Build the retrieval index from the train corpus + eval queries, then exit.")
    ap.add_argument("--corpus", type=Path, default=Path("datasets/supergpqa_train.json"),
                    help="Train split whose SOLVED queries + winning schemas form the index.")
    ap.add_argument("--evo-cache", type=Path, default=Path("outputs/evolve_mcq_full_cache.jsonl"),
                    help="Per-query evolution cache providing final_schema/solved/solved_at.")
    ap.add_argument("--nontrivial-only", action="store_true",
                    help="Index only queries that needed an edit (solved_at>0); drop minimal-schema wins.")
    ap.add_argument("--embed-model", default="text-embedding-3-small")
    ap.add_argument("--embed-datasets", nargs="+", type=str,
                    default=["datasets/supergpqa_test.json", "datasets/frames_ood_test.json"],
                    help="Eval datasets whose queries are pre-embedded into the index.")
    ap.add_argument("--index-out", type=Path, default=Path("outputs/schema_retrieval_index.json"))
    # arm run
    ap.add_argument("--arms", default="retrieve_synth",
                    help=f"Comma-separated arms to run. Choices: {ALL_ARMS}")
    ap.add_argument("--summarize-only", action="store_true",
                    help="Skip execution; rebuild per-arm summaries + head-to-head + combined "
                         "from existing retrieve_*_<split>_cache.jsonl files (for tmux fan-out).")
    ap.add_argument("--dataset", type=Path, default=Path("datasets/supergpqa_test.json"),
                    help="Eval split (MCQ: options + answer_letter).")
    ap.add_argument("--top-k", type=int, default=3, help="Neighbors retrieved per query.")
    ap.add_argument("--synth-model", default="gpt-5.4-mini", help="Schema architect (OpenAI).")
    ap.add_argument("--model", default="Qwen/Qwen3-14B", help="Executor (vLLM).")
    ap.add_argument("--base-url", default="http://localhost:7472/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--debate-k", type=int, default=3, help="Solvers per round for fixed_debate.")
    ap.add_argument("--exec-temp", type=float, default=0.7,
                    help="Executor temperature, uniform across arms.")
    ap.add_argument("--answer-tokens", type=int, default=3072)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="Only the first N questions.")
    ap.add_argument("--outdir", type=Path, default=Path("outputs"))
    main(ap.parse_args())

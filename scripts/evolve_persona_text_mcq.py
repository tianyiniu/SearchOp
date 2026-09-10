"""E7 -- Reflective evolution of persona TEXT, with Pareto-front parent selection.

WHY THIS EXISTS
    E5 (evolve_cem_mcq.py) took single-execution accuracy from 8.1% to 10.9% by
    evolving TOPOLOGY, and the whole gain landed in departure precision -- 16.5% ->
    23.6%. That is, the improvement came from what a critic DOES when it fires, not
    from how often it fires. But `PERSONA_PROMPTS` is three frozen strings, and the
    winning schema (`solver; critic; critic`) helps only because it runs the same
    frozen instruction twice.

    So the axis with the most room is the one nothing has touched: the text. This is
    also what MASS (arXiv 2502.02533) predicts -- of its three optimization stages,
    topology-alone is the weakest.

GENOME
    (topology, prompts) where prompts maps persona -> system prompt text. Topology is
    held fixed per lineage (seeded from E5's winners) so the run measures the TEXT
    axis cleanly; --mutate-topology lets both drift if you want the joint search.

VARIATION -- reflection, not mutation (GEPA, arXiv 2507.19457)
    Sample executions of the parent that FAILED, replay them with transcripts, and
    ask a reflector LLM to (a) diagnose in natural language why the persona's
    intervention did not land, and (b) rewrite ONE persona's system prompt. Textual
    reflection makes moves large enough for a ~400-question batch to resolve, which
    token-level or random edits do not (see the power note in evol_debate_designs.md).

    INFORMATION FIREWALL. The reflector never sees the gold letter. It sees the
    question, the personas' outputs, and whether the run departed from the base
    model's answer. The artifact it writes is a GENERIC prompt applied to every
    question, so it structurally cannot encode a per-question answer. This is
    train-time optimization; the evolved prompt is applied blind at test time.

SELECTION -- Pareto front over per-question outcomes, not scalar mean
    With binary outcomes at an ~8% base rate, an individual joins the front when its
    solved set is not a subset of another's -- i.e. when it uniquely solves something.
    That keeps specialists a scalar mean would discard, which matters here because
    the arms correlate weakly. Parents are sampled from the front, not from the top-1.

    export OPENAI_API_KEY=...      # reflector (gpt-5.4-mini); not needed with --reflector-backend vllm
    python3 scripts/evolve_persona_text_mcq.py --generations 6 --population 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import debate_mcq as D
from llm_judge import call_openai
from schema_fitness import (BudgetExhausted, Evaluator, Pool, beta_lcb, canon, gloss,
                            load_base_letters, log, n_calls, summarize)

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


REFLECTOR_SYSTEM = """You are optimizing the SYSTEM PROMPT of one persona in a multi-agent debate that answers hard graduate-level multiple-choice questions.

A schema runs personas in order; each sees the question and (depending on the schema) the earlier responses. The executor is a small model that is WRONG on these questions by default, so the only way it can score is by MOVING OFF its first instinct for a good reason. Two quantities matter:
  departure rate      -- how often the final answer differs from the model's own first answer
  departure precision -- how often that departure is CORRECT

Departure precision is the binding constraint. A prompt that makes the model change its mind at random scores at chance (~12%); the current best scores ~24%.

You will see failed runs: the question, each persona's output, and whether the run departed. YOU ARE NOT TOLD THE CORRECT OPTION AND MUST NOT GUESS IT.

Your job:
1. Diagnose, in one or two sentences, why this persona's intervention failed to produce a correct departure. Was it too deferential? Did it re-derive the same reasoning? Did it object to something irrelevant? Did it change the answer with no real justification?
2. Rewrite that persona's system prompt to fix the diagnosed failure.

HARD CONSTRAINTS on the rewritten prompt:
- It is GENERIC. It is used verbatim for thousands of unrelated questions. It must contain NO facts, NO subject matter, NO option letters, and nothing specific to the questions you were shown.
- It must keep the persona's role coherent with its name.
- Unless the persona is an eliminator, it must still instruct the model to end with exactly one line 'ANSWER: <letter>' and to choose exactly one letter.
- Keep it under 130 words.

Respond with STRICT JSON only, no prose:
{"diagnosis":"...","persona":"<one of the personas shown>","prompt":"<the rewritten system prompt>"}"""


def reflect_prompt(schema: dict, prompts: dict, traces: list[dict], editable: list[str]) -> str:
    blocks = []
    for i, t in enumerate(traces, 1):
        rounds = "\n".join(
            f"  round {ri + 1} [{p}]: {(resp or '').strip()[:600]}"
            for ri, rnd in enumerate(t["rounds"]) for p, resp in rnd)
        blocks.append(f"FAILED RUN {i}\n  question: {t['question'][:600]}\n"
                      f"  base model's own answer: {t['base'] or '?'}\n"
                      f"  this run's answer: {t['letter'] or '?'}  "
                      f"({'DEPARTED' if t['departed'] else 'did NOT depart'})\n{rounds}")
    cur = "\n\n".join(f"--- current prompt for `{p}` ---\n{prompts.get(p, D.PERSONA_PROMPTS[p])}"
                      for p in editable)
    return (f"SCHEMA: {gloss(schema)}\n\nPERSONAS YOU MAY EDIT: {editable}\n\n{cur}\n\n"
            + "\n\n".join(blocks) +
            "\n\nDiagnose and rewrite ONE persona's system prompt as the JSON object specified.")


def parse_rewrite(raw: str, editable: list[str]) -> tuple[str, str, str] | None:
    raw = D.re.sub(r"^```(?:json)?|```$", "", (raw or "").strip()).strip()
    obj = None
    try:
        obj = json.loads(raw)
    except Exception:
        m = D._JSON_RE.search(raw)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if not isinstance(obj, dict):
        return None
    p, text = obj.get("persona"), obj.get("prompt")
    if p not in editable or not isinstance(text, str) or not (40 <= len(text) <= 1400):
        return None
    if p not in D.NON_ANSWERING and "ANSWER" not in text.upper():
        return None                       # must still instruct a commitment
    return p, text.strip(), str(obj.get("diagnosis", ""))[:300]


class Individual:
    _next = 0

    def __init__(self, schema: dict, prompts: dict, parent: str | None = None, gen: int = 0):
        Individual._next += 1
        self.id = f"i{Individual._next:03d}"
        self.schema, self.prompts = schema, dict(prompts)
        self.parent, self.gen = parent, gen
        self.diagnosis = ""
        self.edited = None
        self.stats: dict = {}
        self.solved: set[str] = set()

    @property
    def key(self) -> str:
        return canon(self.schema, self.prompts)

    def record(self) -> dict:
        return {"id": self.id, "gen": self.gen, "parent": self.parent,
                "schema": self.schema, "gloss": gloss(self.schema),
                "calls_per_exec": n_calls(self.schema), "edited_persona": self.edited,
                "diagnosis": self.diagnosis, "prompts": self.prompts, **self.stats}


def pareto_front(pop: list[Individual]) -> list[Individual]:
    """Non-dominated by per-question solved set: A dominates B iff solved(B) is a
    strict subset of solved(A). Individuals that uniquely solve something survive."""
    front = []
    for a in pop:
        if not any(b is not a and a.solved < b.solved for b in pop):
            front.append(a)
    return front or list(pop)


def collect_traces(ev: Evaluator, ind: Individual, qids: list[str], k: int,
                   rng: random.Random) -> list[dict]:
    """Replay a few FAILED questions with transcripts, for the reflector to read."""
    failed = [q for q in qids if q not in ind.solved]
    picks = rng.sample(failed, min(k, len(failed)))
    out = []
    for q in picks:
        row = ev.pool.rows[q]
        cost = n_calls(ind.schema)
        ev._charge(cost)
        try:
            letter, rounds = D.execute_schema(ev._client(), ev.model, row["question"],
                                              list(row["options"]), ind.schema,
                                              ev.temperature, ev.answer_tokens,
                                              prompts=ind.prompts, return_trace=True)
        except Exception:
            ev.tick(cost)
            continue
        ev.tick(cost)
        base = ev.pool.base.get(q)
        out.append({"question": row["question"], "letter": letter, "base": base,
                    "departed": letter is not None and letter != base, "rounds": rounds})
    return out


def main(args):
    rng = random.Random(args.seed)
    base = load_base_letters(args.bestofn_cache)
    pool = Pool(args.dataset, base, seed=args.seed, limit=args.pool_limit)
    ev = Evaluator(pool, args.base_urls, args.model, args.temperature, args.answer_tokens,
                   args.workers, args.exec_cache, max_calls=args.max_calls, api_key=args.api_key, progress=args.progress)

    if args.reflector_backend == "vllm":
        reflect = lambda s, u: D.chat(ev._client(), args.reflector_model or args.model,
                                      s, u, args.reflector_temperature, max_tokens=1400)
    else:
        # fail in seconds rather than after generation 0 has burned several thousand calls
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set. Export it, or pass "
                             "--reflector-backend vllm to run the reflector locally.")
        reflect = lambda s, u: call_openai(s, u, model=args.reflector_model or "gpt-5.4-mini")
    if args.elites >= args.population:
        raise SystemExit(f"--elites ({args.elites}) must be < --population ({args.population}); "
                         "otherwise no children are ever produced.")

    topologies = json.loads(Path(args.topologies).read_text()) if args.topologies else {
        "double_critic": {"rounds": [{"personas": ["solver"]}, {"personas": ["critic"]},
                                     {"personas": ["critic"]}], "final": "last"}}
    seeds = []
    for name, sch in topologies.items():
        if not D.validate(sch):
            raise SystemExit(f"seed topology {name!r} is not grammar-valid")
        seeds.append((name, sch))
    print(f"pool {len(pool)} | seed topologies: {[n for n, _ in seeds]} | "
          f"reflector {args.reflector_model or 'gpt-5.4-mini'} via {args.reflector_backend}")

    # one seed per topology -- extra copies would share a genome key and collapse
    pop = [Individual(sch, {}, gen=0) for _, sch in seeds][: args.population]
    batch = pool.batch(args.batch)
    history, everyone = [], {}

    try:
        for gen in range(args.generations + 1):
            for ei, ind in enumerate(pop, 1):                    # evaluate (cached if unchanged)
                ev.stage(f"gen {gen} eval {ei}/{len(pop)} {ind.id}")
                outcomes = ev.run(ind.schema, batch, prompts=ind.prompts or None)
                ind.stats = summarize(outcomes)
                ind.solved = {o["q"] for o in outcomes if o["correct"]}
                everyone[ind.key] = ind
            pop.sort(key=lambda i: -i.stats["lcb"])
            front = pareto_front(pop)
            best = pop[0]
            union = set().union(*(i.solved for i in pop)) if pop else set()
            log(f"\n=== generation {gen}/{args.generations} | calls {ev.calls} | "
                f"front {len(front)}/{len(pop)} | union coverage {len(union) / len(batch):.1%} ===")
            for i in pop:
                mark = "*" if i in front else " "
                log(f" {mark}{i.id} g{i.gen} acc {i.stats['accuracy']:>6.1%} "
                      f"lcb {i.stats['lcb']:>6.1%} D {i.stats['departure_rate']:>5.1%} "
                      f"P {i.stats['departure_precision']:>6.1%}  "
                      f"edited={i.edited or '-':<12} {gloss(i.schema)}")
            history.append({"generation": gen, "calls": ev.calls,
                            "front": [i.id for i in front],
                            "union_coverage": len(union) / len(batch),
                            "population": [i.record() for i in pop]})
            if gen == args.generations:
                break

            # BOUNDED. Every `continue` below (no failed runs to show, unparseable
            # reflector output, duplicate child) would otherwise spin forever -- a
            # misconfigured reflector would hang the run silently, producing nothing.
            want = max(1, args.population - args.elites)
            budget_attempts = want * 6 + 12
            children, attempts, n_bad = [], 0, 0
            while len(children) < want and attempts < budget_attempts:
                attempts += 1
                ev.stage(f"gen {gen} reflect {len(children) + 1}/{want} (try {attempts})")
                parent = rng.choice(front)
                editable = sorted({p for r in parent.schema["rounds"] for p in r["personas"]})
                traces = collect_traces(ev, parent, batch, args.reflect_traces, rng)
                if not traces:
                    n_bad += 1
                    continue
                got = None
                for _ in range(args.reflect_retries):
                    got = parse_rewrite(reflect(REFLECTOR_SYSTEM,
                                                reflect_prompt(parent.schema, parent.prompts,
                                                               traces, editable)), editable)
                    if got:
                        break
                if not got:
                    n_bad += 1
                    continue
                persona, text, diag = got
                child = Individual(parent.schema, {**parent.prompts, persona: text},
                                   parent=parent.id, gen=gen + 1)
                child.edited, child.diagnosis = persona, diag
                if child.key in everyone:                        # novelty rejection
                    continue
                children.append(child)
                log(f"   {parent.id} -> {child.id}: rewrote `{persona}` — {diag[:110]}")
            if n_bad:
                log(f"   ({n_bad} reflection attempts produced nothing usable)")
            if not children:
                log("   no viable children this generation — reflector is failing. Stopping.")
                break
            pop = pop[: args.elites] + children
    except BudgetExhausted as exc:
        log(f"\n! {exc} -- stopping cleanly and writing what we have")
    except KeyboardInterrupt:
        log("\n! interrupted -- writing what we have")
    ev.close()          # retire the progress bar before the report is printed

    ranked = sorted(everyone.values(), key=lambda i: -i.stats.get("lcb", 0))
    out = {"dataset": str(args.dataset), "batch": args.batch, "pool_size": len(pool),
           "generations_completed": len(history) - 1, "calls_spent": ev.calls,
           "executions": ev.executions, "errors": ev.errors,
           "best": ranked[0].record() if ranked else None,
           "leaderboard": [i.record() for i in ranked[: args.top]],
           "history": history}
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    ev.close()

    print("\n=== leaderboard by LCB ===")
    print(f"{'id':>6}{'gen':>5}{'acc':>8}{'lcb':>8}{'D':>7}{'P':>8}{'calls':>7}  edited      schema")
    for i in ranked[: args.top]:
        s = i.stats
        print(f"{i.id:>6}{i.gen:>5}{s.get('accuracy', 0):>8.1%}{s.get('lcb', 0):>8.1%}"
              f"{s.get('departure_rate', 0):>7.1%}{s.get('departure_precision', 0):>8.1%}"
              f"{n_calls(i.schema):>7}  {i.edited or '-':<11} {gloss(i.schema)}")
    if ranked and ranked[0].prompts:
        print(f"\n=== evolved prompts of {ranked[0].id} ===")
        for p, t in ranked[0].prompts.items():
            print(f"\n[{p}]\n{t}")
    print(f"\ncalls spent: {ev.calls}  executions: {ev.executions}  errors: {ev.errors}")
    print(f"execution cache -> {args.exec_cache}\nsummary -> {args.summary_out}")
    print("\nreminder: RECOVERY only. Re-score the winner on the retention pool -- a prompt that")
    print("wins by departing harder is exactly what §0.9.3 shows costs ~11pp on answerable questions.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("datasets/supergpqa_strict_train.json"))
    ap.add_argument("--bestofn-cache", type=Path, default=Path("outputs/bestofn_strict_cache.jsonl"))
    ap.add_argument("--topologies", type=Path, default=Path("candidate_schemas.json"),
                    help="JSON {name: schema} of seed topologies; prompts evolve on top of them.")
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--population", type=int, default=8)
    ap.add_argument("--elites", type=int, default=3, help="Individuals carried forward unchanged.")
    ap.add_argument("--batch", type=int, default=192, help="Questions per evaluation (paired prefix).")
    ap.add_argument("--reflect-traces", type=int, default=4, help="Failed runs shown to the reflector.")
    ap.add_argument("--reflect-retries", type=int, default=3)
    ap.add_argument("--reflector-backend", choices=("openai", "vllm"), default="openai")
    ap.add_argument("--reflector-model", default=None)
    ap.add_argument("--reflector-temperature", type=float, default=0.8)
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--base-urls", default="http://localhost:7472/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--answer-tokens", type=int, default=3072)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--pool-limit", type=int, default=None)
    ap.add_argument("--max-calls", type=int, default=90000)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-progress", dest="progress", action="store_false", default=True,
                    help="Disable the tqdm call-progress bar (use when redirecting to a log file).")
    ap.add_argument("--exec-cache", type=Path, default=Path("outputs/persona_text_exec_cache.jsonl"))
    ap.add_argument("--summary-out", type=Path, default=Path("outputs/evolve_persona_text_summary.json"))
    main(ap.parse_args())

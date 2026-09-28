"""E5 -- Cross-Entropy Method / EDA over a schema PCFG.

An estimation-of-distribution algorithm: instead of mutating individuals, learn
and refine a DISTRIBUTION over the grammar, sample from it, keep the elite
fraction, refit, repeat.

WHY THIS FAMILY, HERE
    The fitness is a Bernoulli at a ~4% base rate. Jin & Branke's survey names
    three ways to survive that: explicit averaging (re-evaluate), implicit
    averaging (use a big population so noise cancels across similar individuals),
    and selection modification (only prefer A over B under a statistical test).
    An EDA gets implicit averaging for free -- the model is refit from a whole
    elite SET, so no single noisy comparison can move it much. That is the
    cheapest noise robustness available, and it is why this is worth running
    before the more elaborate designs.

    It also has two properties the others do not:
      - it WARM-STARTS for free from the 845 solved schemas already sitting in
        outputs/evolve_mcq_strict_cache.jsonl -- the existing corpus becomes a
        prior instead of being thrown away;
      - its output is a SAMPLER, which is exactly what the portfolio design (E4)
        needs as a proposal distribution.

THE MODEL (factorized, ~40 free parameters, all Dirichlet-conjugate)
    P(n_rounds)                     over 1..6
    P(n_personas | bucket)          over 1..4
    P(persona    | bucket)          over (solver, critic, synthesizer)
    P(final)                        over (last, vote, synthesizer)
    bucket in (first, middle, last) -- position in the schema, so the model can
    learn "solver first, critic later" without a parameter per position.

WHAT TO WATCH
    Model entropy per iteration. If it collapses toward `solver; critic -> last`
    and the elite accuracy stops moving, the grammar is saturated and that is the
    answer -- the same conclusion the departure-precision ceiling points at, from
    an independent direction.

    export CUDA_VISIBLE_DEVICES=...   # vLLM serving Qwen/Qwen3-14B
    python3 scripts/evolve_cem_mcq.py --iters 6 --samples 32 --max-calls 80000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import debate_mcq as D
from schema_fitness import (BudgetExhausted, Evaluator, Pool, canon, gloss,
                            load_base_letters, log, n_calls, race, repair)

BUCKETS = ("first", "middle", "last")


def bucket_of(i: int, n: int) -> str:
    if i == 0:
        return "first"
    return "last" if i == n - 1 else "middle"


def _norm(v: list[float]) -> list[float]:
    s = sum(v)
    return [x / s for x in v] if s > 0 else [1.0 / len(v)] * len(v)


def _entropy(v: list[float]) -> float:
    return -sum(p * math.log(p + 1e-12) for p in v)


class SchemaPCFG:
    """Factorized distribution over the debate-schema grammar, including the E1
    genome extension: personas range over D.EXT_PERSONAS and each round carries a
    visibility gene drawn from D.SEES."""

    def __init__(self):
        self.n_rounds = _norm([1.0] * D.MAX_ROUNDS)
        self.n_personas = {b: _norm([1.0] * D.MAX_PERSONAS) for b in BUCKETS}
        self.persona = {b: _norm([1.0] * len(D.EXT_PERSONAS)) for b in BUCKETS}
        self.sees = {b: _norm([1.0] * len(D.SEES)) for b in BUCKETS}
        self.final = _norm([1.0] * len(D.FINALS))

    # -- sampling
    def sample(self, rng: random.Random) -> dict | None:
        nr = rng.choices(range(1, D.MAX_ROUNDS + 1), weights=self.n_rounds)[0]
        rounds = []
        for i in range(nr):
            b = bucket_of(i, nr)
            np_ = rng.choices(range(1, D.MAX_PERSONAS + 1), weights=self.n_personas[b])[0]
            rnd = {"personas": [rng.choices(list(D.EXT_PERSONAS), weights=self.persona[b])[0]
                                for _ in range(np_)]}
            sees = rng.choices(list(D.SEES), weights=self.sees[b])[0]
            if sees != D.DEFAULT_SEES:
                rnd["sees"] = sees
            rounds.append(rnd)
        return repair({"rounds": rounds,
                       "final": rng.choices(list(D.FINALS), weights=self.final)[0]})

    # -- fitting
    @staticmethod
    def counts_from(schemas: list[dict]) -> dict:
        c = {"n_rounds": [0.0] * D.MAX_ROUNDS,
             "n_personas": {b: [0.0] * D.MAX_PERSONAS for b in BUCKETS},
             "persona": {b: [0.0] * len(D.EXT_PERSONAS) for b in BUCKETS},
             "sees": {b: [0.0] * len(D.SEES) for b in BUCKETS},
             "final": [0.0] * len(D.FINALS)}
        for s in schemas:
            nr = len(s["rounds"])
            c["n_rounds"][nr - 1] += 1
            c["final"][D.FINALS.index(s["final"])] += 1
            for i, r in enumerate(s["rounds"]):
                b = bucket_of(i, nr)
                c["n_personas"][b][len(r["personas"]) - 1] += 1
                c["sees"][b][D.SEES.index(r.get("sees", D.DEFAULT_SEES))] += 1
                for p in r["personas"]:
                    c["persona"][b][D.EXT_PERSONAS.index(p)] += 1
        return c

    @classmethod
    def fit(cls, schemas: list[dict], alpha: float = 1.0) -> "SchemaPCFG":
        c = cls.counts_from(schemas)
        m = cls()
        m.n_rounds = _norm([x + alpha for x in c["n_rounds"]])
        m.final = _norm([x + alpha for x in c["final"]])
        for b in BUCKETS:
            m.n_personas[b] = _norm([x + alpha for x in c["n_personas"][b]])
            m.persona[b] = _norm([x + alpha for x in c["persona"][b]])
            m.sees[b] = _norm([x + alpha for x in c["sees"][b]])
        return m

    def blend(self, other: "SchemaPCFG", rho: float) -> "SchemaPCFG":
        """theta <- (1-rho)*self + rho*other. Smoothing against premature collapse."""
        m = SchemaPCFG()
        mix = lambda a, b: _norm([(1 - rho) * x + rho * y for x, y in zip(a, b)])
        m.n_rounds = mix(self.n_rounds, other.n_rounds)
        m.final = mix(self.final, other.final)
        for b in BUCKETS:
            m.n_personas[b] = mix(self.n_personas[b], other.n_personas[b])
            m.persona[b] = mix(self.persona[b], other.persona[b])
            m.sees[b] = mix(self.sees[b], other.sees[b])
        return m

    def entropy(self) -> float:
        e = _entropy(self.n_rounds) + _entropy(self.final)
        for b in BUCKETS:
            e += _entropy(self.n_personas[b]) + _entropy(self.persona[b]) + _entropy(self.sees[b])
        return e

    def to_dict(self) -> dict:
        return {"n_rounds": [round(x, 4) for x in self.n_rounds],
                "final": dict(zip(D.FINALS, [round(x, 4) for x in self.final])),
                "n_personas": {b: [round(x, 4) for x in v] for b, v in self.n_personas.items()},
                "persona": {b: dict(zip(D.EXT_PERSONAS, [round(x, 4) for x in v]))
                            for b, v in self.persona.items()},
                "sees": {b: dict(zip(D.SEES, [round(x, 4) for x in v]))
                         for b, v in self.sees.items()},
                "entropy": round(self.entropy(), 4)}


def warm_start(cache: Path, solved_only: bool = True, nontrivial: bool = True) -> list[dict]:
    """The existing GT-evolution corpus as a prior: the schemas that solved a
    question, optionally excluding those solved at step 0 (pure resampling luck)."""
    out = []
    if not cache.exists():
        return out
    for line in cache.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("status") != "ok":
            continue
        if solved_only and not r.get("solved"):
            continue
        if nontrivial and (r.get("solved_at") or 0) == 0:
            continue
        s = repair(r["final_schema"])
        if s:
            out.append(s)
    return out


def main(args):
    rng = random.Random(args.seed)
    base = load_base_letters(args.bestofn_cache)
    pool = Pool(args.dataset, base, seed=args.seed, limit=args.pool_limit)
    print(f"pool: {len(pool)} questions from {args.dataset} (base letters for {len(base)})")

    model = SchemaPCFG()
    if args.warm_start:
        seeds = warm_start(args.evo_cache, nontrivial=not args.include_trivial)
        if seeds:
            fitted = SchemaPCFG.fit(seeds, alpha=args.laplace)
            # The corpus predates the E1 genome extension, so nothing in it uses `sees`
            # or the new personas -- a pure refit would give them ~0 mass and the run
            # would never test them. Blend back toward uniform to keep them reachable.
            model = SchemaPCFG().blend(fitted, 1.0 - args.warm_explore)
            print(f"warm start: fitted to {len(seeds)} solved schemas from {args.evo_cache}")
            print(f"  entropy fitted {fitted.entropy():.3f} -> blended {model.entropy():.3f} "
                  f"(warm-explore {args.warm_explore:.2f}, keeps the new genes reachable)")
            print(f"  top shapes: {dict(Counter(gloss(s) for s in seeds).most_common(5))}")
        else:
            print(f"warm start requested but {args.evo_cache} yielded nothing -- uniform prior")

    ev = Evaluator(pool, args.base_urls, args.model, args.temperature, args.answer_tokens,
                   args.workers, args.exec_cache, max_calls=args.max_calls, api_key=args.api_key, progress=args.progress)
    stages = tuple(int(x) for x in args.stages.split(","))
    history, seen_best = [], []

    try:
        for it in range(1, args.iters + 1):
            # sample distinct candidates; carry the previous elites forward (elitism)
            cands, seen = [], set()
            for s in (history[-1]["elites"] if history else []):
                if canon(s) not in seen:
                    seen.add(canon(s))
                    cands.append(s)
            tries = 0
            while len(cands) < args.samples and tries < args.samples * 40:
                tries += 1
                s = model.sample(rng)
                if s and canon(s) not in seen:
                    seen.add(canon(s))
                    cands.append(s)
            log(f"\n=== iteration {it}/{args.iters} | {len(cands)} candidates "
                f"| model entropy {model.entropy():.3f} | calls {ev.calls} ===")

            ranked = race(cands, ev, stages=stages, keep_frac=args.keep_frac,
                          dep_band=(args.dep_min, args.dep_max) if args.dep_filter else None)
            if not ranked:
                log("  no survivors -- stopping")
                break
            n_elite = max(2, math.ceil(len(ranked) * args.elite_frac))
            elites = [s for s, _ in ranked[:n_elite]]
            model = model.blend(SchemaPCFG.fit(elites, alpha=args.laplace), args.rho)

            best_s, best_st = ranked[0]
            log(f"  best: {gloss(best_s)}")
            log(f"        acc {best_st['accuracy']:.1%} (lcb {best_st['lcb']:.1%}, "
                  f"n={best_st['n']}) | D {best_st['departure_rate']:.1%} "
                  f"x P {best_st['departure_precision']:.1%} | {n_calls(best_s)} calls/exec")
            log(f"  elites ({n_elite}): " + " | ".join(gloss(s) for s in elites[:4]))
            history.append({"iteration": it, "n_candidates": len(cands),
                            "calls_cumulative": ev.calls, "model": model.to_dict(),
                            "best": {"schema": best_s, "gloss": gloss(best_s),
                                     "calls_per_exec": n_calls(best_s), **best_st},
                            "elites": elites,
                            "ranked": [{"schema": s, "gloss": gloss(s),
                                        "calls_per_exec": n_calls(s), **st}
                                       for s, st in ranked]})
            seen_best.extend(history[-1]["ranked"])
    except BudgetExhausted as exc:
        log(f"\n! {exc} -- stopping cleanly and writing what we have")
    except KeyboardInterrupt:
        log("\n! interrupted -- writing what we have")
    ev.close()          # retire the progress bar before the report is printed

    # final: the single best by LCB over every candidate ever fully evaluated
    full = [r for r in seen_best if r["n"] >= stages[-1] * 0.9]
    ranking = sorted(full or seen_best, key=lambda r: -r["lcb"])
    out = {"dataset": str(args.dataset), "pool_size": len(pool), "stages": list(stages),
           "iters_completed": len(history), "calls_spent": ev.calls,
           "executions": ev.executions, "errors": ev.errors,
           "final_model": model.to_dict(),
           "best_overall": ranking[0] if ranking else None,
           "leaderboard": ranking[: args.top],
           "history": [{k: v for k, v in h.items() if k != "elites"} for h in history]}
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    ev.close()

    print(f"\n=== leaderboard (by LCB, fully-raced candidates) ===")
    print(f"{'acc':>7}{'lcb':>7}{'n':>6}{'D':>7}{'P':>7}{'calls':>7}  schema")
    for r in ranking[: args.top]:
        print(f"{r['accuracy']:>7.1%}{r['lcb']:>7.1%}{r['n']:>6}{r['departure_rate']:>7.1%}"
              f"{r['departure_precision']:>7.1%}{r['calls_per_exec']:>7}  {r['gloss']}")
    print(f"\ncalls spent: {ev.calls}  executions: {ev.executions}  errors: {ev.errors}")
    print(f"execution cache -> {args.exec_cache}\nsummary -> {args.summary_out}")
    print("\nreminder: fitness here is RECOVERY only (the strict-fail pool). Re-score the")
    print("winner on datasets/supergpqa_answerable_retention.json before claiming anything.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("datasets/supergpqa_strict_train.json"))
    ap.add_argument("--bestofn-cache", type=Path, default=Path("outputs/bestofn_strict_cache.jsonl"))
    ap.add_argument("--evo-cache", type=Path, default=Path("outputs/evolve_mcq_strict_cache.jsonl"),
                    help="Existing GT-evolution corpus used as the warm-start prior.")
    ap.add_argument("--warm-start", action="store_true", default=True)
    ap.add_argument("--no-warm-start", dest="warm_start", action="store_false",
                    help="Start from a uniform prior instead (the control for whether the "
                         "existing corpus carries any transferable structure).")
    ap.add_argument("--include-trivial", action="store_true",
                    help="Include schemas solved at step 0 (pure resampling luck) in the prior.")
    ap.add_argument("--warm-explore", type=float, default=0.35,
                    help="Mass blended back toward uniform after the warm-start refit. The corpus "
                         "predates the E1 genes (sees / contrarian / eliminator / independent), so "
                         "0 would make them unreachable.")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--samples", type=int, default=32, help="Distinct candidates per iteration.")
    ap.add_argument("--stages", default="64,192,512", help="Nested batch sizes for the race.")
    ap.add_argument("--keep-frac", type=float, default=0.5, help="Survivors per racing stage.")
    ap.add_argument("--elite-frac", type=float, default=0.25, help="Elite fraction refitting the model.")
    ap.add_argument("--rho", type=float, default=0.3, help="Model blend rate (lower = slower collapse).")
    ap.add_argument("--laplace", type=float, default=1.0, help="Dirichlet smoothing on the refit.")
    ap.add_argument("--dep-filter", action="store_true",
                    help="Enable the label-free stage-0 departure-rate band filter.")
    ap.add_argument("--dep-min", type=float, default=0.25)
    ap.add_argument("--dep-max", type=float, default=1.0)
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--base-urls", default="http://localhost:7472/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--answer-tokens", type=int, default=3072)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--pool-limit", type=int, default=None)
    ap.add_argument("--max-calls", type=int, default=80000, help="Hard LLM-call budget.")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-progress", dest="progress", action="store_false", default=True,
                    help="Disable the tqdm call-progress bar (use when redirecting to a log file).")
    ap.add_argument("--exec-cache", type=Path, default=Path("outputs/cem_exec_cache.jsonl"),
                    help="Shared execution cache; safe to point several runs at one file.")
    ap.add_argument("--summary-out", type=Path, default=Path("outputs/evolve_cem_summary.json"))
    main(ap.parse_args())

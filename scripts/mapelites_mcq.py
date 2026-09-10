"""E3-lite -- MAP-Elites over (departure rate x cost), with Beta-posterior deep-grid cells.

THE POINT
    Greedy objective-driven search collapses to one attractor -- empirically yours
    did: 99% of the architect's first edits were `add_round`, 89% byte-identical.
    MAP-Elites instead keeps the best solution PER BIN of a behavior space, so the
    run returns a frontier rather than a point, and diversity is maintained by
    construction rather than by hoping.

    The descriptors here are not arbitrary. Section 0.7 of evol_debate_designs.md
    establishes that on this corpus

        accuracy = D x P + (1-D) x P_stay,   P_stay ~ 1.4%

    where D = P(depart from the base model's letter) and P = P(correct | departed).
    So binning by D and by cost, and recording the best accuracy in each bin,
    directly traces the curve

        P*(D) = the best departure precision achievable at each departure rate

    which is the quantity the whole project is pressed against. Best P measured
    anywhere so far is 18.6%; the random-departure null is 12.0%. This run says
    whether 18.6% is a wall or a floor.

    Crucially, D is computed from the FINAL letter against cached base letters --
    no gold label and no new instrumentation. That is why this variant runs today
    while the full (D x churn x cost) grid waits on per-round logging.

NOISE HANDLING
    A 4%-base-rate Bernoulli fitness lets one lucky execution squat in a cell
    forever. Two defences, both from the noisy-QD literature (Flageat & Cully):
      - each cell holds a DEPTH of residents whose Beta posteriors accumulate, and
        the cell's elite is the highest LOWER confidence bound, not the highest
        point estimate;
      - a share of the iteration budget goes to DEEPENING -- re-evaluating an
        existing elite on a fresh replicate -- instead of to new candidates.

VARIATION IS DELIBERATELY LLM-FREE
    Mutation and crossover are structural and random (schema_fitness.mutate /
    .crossover). If random variation illuminates the space as well as the
    gpt-5.4-mini architect did, that is a result about how much the "guided" in
    "guided structural search" was ever worth.

    python3 scripts/mapelites_mcq.py --iters 220 --max-calls 80000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import debate_mcq as D
from schema_fitness import (BudgetExhausted, Evaluator, Pool, beta_lcb, canon, crossover,
                            gloss, load_base_letters, log, mutate, n_calls, repair)

COST_EDGES = (1, 2, 3, 4, 6, 9)          # -> bins: 1, 2, 3, 4-5, 6-8, 9+


def cost_bin(c: int) -> int:
    b = 0
    for i, e in enumerate(COST_EDGES):
        if c >= e:
            b = i
    return b


def cost_label(b: int) -> str:
    lo = COST_EDGES[b]
    hi = COST_EDGES[b + 1] - 1 if b + 1 < len(COST_EDGES) else None
    return f"{lo}" if hi == lo else (f"{lo}-{hi}" if hi else f"{lo}+")


class Resident:
    """One schema in one cell, with an accumulating Beta posterior."""

    def __init__(self, schema: dict):
        self.schema = schema
        self.canon = canon(schema)
        self.calls_per_exec = n_calls(schema)
        self.k = self.n = 0                 # correct / evaluated
        self.dep = self.dep_hit = 0
        self.reps = 0

    def absorb(self, st: dict) -> None:
        self.k += st["n_correct"]
        self.n += st["n"]
        self.dep += st["n_departed"]
        self.dep_hit += st["n_dep_correct"]
        self.reps += 1

    @property
    def accuracy(self) -> float:
        return self.k / self.n if self.n else 0.0

    @property
    def lcb(self) -> float:
        return beta_lcb(self.k, self.n)

    @property
    def departure_rate(self) -> float:
        return self.dep / self.n if self.n else 0.0

    @property
    def departure_precision(self) -> float:
        return self.dep_hit / self.dep if self.dep else 0.0

    def record(self) -> dict:
        return {"schema": self.schema, "gloss": gloss(self.schema),
                "calls_per_exec": self.calls_per_exec, "n": self.n, "n_correct": self.k,
                "accuracy": round(self.accuracy, 4), "lcb": round(self.lcb, 4),
                "departure_rate": round(self.departure_rate, 4),
                "departure_precision": round(self.departure_precision, 4), "reps": self.reps}


class Archive:
    def __init__(self, d_bins: int, depth: int):
        self.d_bins, self.depth = d_bins, depth
        self.cells: dict[tuple[int, int], list[Resident]] = {}
        self.curiosity: dict[tuple[int, int], float] = {}
        self.by_canon: dict[str, tuple[int, int]] = {}

    def d_bin(self, d: float) -> int:
        return min(self.d_bins - 1, int(d * self.d_bins))

    def place(self, res: Resident) -> tuple[tuple[int, int], bool]:
        key = (self.d_bin(res.departure_rate), cost_bin(res.calls_per_exec))
        cell = self.cells.setdefault(key, [])
        self.curiosity.setdefault(key, 1.0)
        prev = max((r.lcb for r in cell), default=-1.0)
        cell.append(res)
        self.by_canon[res.canon] = key
        cell.sort(key=lambda r: -r.lcb)
        del cell[self.depth:]                       # deep grid: bounded depth
        improved = res in cell and res.lcb > prev
        self.curiosity[key] = self.curiosity.get(key, 1.0) + (1.0 if improved else -0.2)
        self.curiosity[key] = max(0.1, self.curiosity[key])
        return key, improved

    def remove(self, res: Resident) -> None:
        key = self.by_canon.get(res.canon)
        if key and key in self.cells:
            self.cells[key] = [r for r in self.cells[key] if r is not res]

    def elite(self, key) -> Resident | None:
        cell = self.cells.get(key)
        return cell[0] if cell else None

    def elites(self) -> list[Resident]:
        return [c[0] for c in self.cells.values() if c]

    def all_residents(self) -> list[Resident]:
        return [r for c in self.cells.values() for r in c]

    def sample_cell(self, rng: random.Random):
        keys = [k for k, c in self.cells.items() if c]
        if not keys:
            return None
        w = [self.curiosity.get(k, 1.0) for k in keys]
        return rng.choices(keys, weights=w)[0]


def frontier(archive: Archive, d_bins: int) -> list[dict]:
    """P*(D): the best departure precision seen in each departure-rate bin."""
    out = []
    for b in range(d_bins):
        res = [r for r in archive.all_residents()
               if archive.d_bin(r.departure_rate) == b and r.n > 0]
        if not res:
            continue
        best = max(res, key=lambda r: r.departure_precision)
        best_acc = max(res, key=lambda r: r.lcb)
        out.append({"d_bin": b, "d_range": [round(b / d_bins, 2), round((b + 1) / d_bins, 2)],
                    "n_schemas": len(res),
                    "best_departure_precision": round(best.departure_precision, 4),
                    "at_schema": gloss(best.schema),
                    "best_lcb_accuracy": round(best_acc.lcb, 4),
                    "best_accuracy": round(best_acc.accuracy, 4),
                    "best_acc_schema": gloss(best_acc.schema)})
    return out


def main(args):
    rng = random.Random(args.seed)
    base = load_base_letters(args.bestofn_cache)
    pool = Pool(args.dataset, base, seed=args.seed, limit=args.pool_limit)
    print(f"pool: {len(pool)} questions | grid: {args.d_bins} departure bins x "
          f"{len(COST_EDGES)} cost bins | cell depth {args.depth}")

    ev = Evaluator(pool, args.base_urls, args.model, args.temperature, args.answer_tokens,
                   args.workers, args.exec_cache, max_calls=args.max_calls, api_key=args.api_key, progress=args.progress)
    archive = Archive(args.d_bins, args.depth)
    batch = pool.batch(args.batch)
    evaluated: dict[str, Resident] = {}

    seeds = [D.MINIMAL_SCHEMA,
             {"rounds": [{"personas": ["solver"]}, {"personas": ["critic"]}], "final": "last"},
             D.self_critique_schema(), D.fixed_debate_schema(3)]
    n_new = n_deepen = n_improved = 0

    try:
        for it in range(1, args.iters + 1):
            ev.stage(f"it {it}/{args.iters} cells={len(archive.cells)} "
                     f"tried={len(evaluated)}")
            deepen = (rng.random() < args.deepen_prob and len(archive.elites()) >= 4
                      and it > args.init_iters)
            if deepen:                                  # spend on sharpening a posterior
                key = archive.sample_cell(rng)
                res = archive.elite(key)
                st = ev.evaluate(res.schema, batch, rep=res.reps)
                # the descriptor is the ACCUMULATED departure rate, so a resident can
                # migrate cells as its posterior sharpens -- re-place rather than resort
                archive.remove(res)
                res.absorb(st)
                archive.place(res)
                n_deepen += 1
            else:
                if it <= len(seeds):
                    cand = seeds[it - 1]
                elif it <= args.init_iters or not archive.elites():
                    cand = repair({"rounds": [{"personas": [rng.choice(list(D.PERSONAS))
                                                            for _ in range(rng.randint(1, 2))]}
                                              for _ in range(rng.randint(1, 3))],
                                   "final": rng.choice(list(D.FINALS))})
                elif rng.random() < args.crossover_prob and len(archive.elites()) >= 2:
                    a, b = rng.sample(archive.elites(), 2)
                    cand = crossover(a.schema, b.schema, rng)
                else:
                    parent = archive.elite(archive.sample_cell(rng))
                    cand = mutate(parent.schema, rng)
                if cand is None:
                    continue
                c = canon(cand)
                if c in evaluated:                      # novelty rejection: already tried
                    continue
                st = ev.evaluate(cand, batch)
                res = Resident(cand)
                res.absorb(st)
                evaluated[c] = res
                _, improved = archive.place(res)
                n_new += 1
                n_improved += int(improved)

            if it % args.report_every == 0 or it == args.iters:
                el = archive.elites()
                best = max(el, key=lambda r: r.lcb) if el else None
                log(f"  it {it:>4} | cells {len(archive.cells):>3} | evaluated {len(evaluated):>4}"
                      f" | new {n_new} deepen {n_deepen} improved {n_improved}"
                      f" | calls {ev.calls}"
                      + (f" | best lcb {best.lcb:.1%} acc {best.accuracy:.1%} "
                         f"D {best.departure_rate:.0%} P {best.departure_precision:.1%} "
                         f"[{gloss(best.schema)}]" if best else ""))
    except BudgetExhausted as exc:
        log(f"\n! {exc} -- stopping cleanly and writing what we have")
    except KeyboardInterrupt:
        log("\n! interrupted -- writing what we have")
    ev.close()          # retire the progress bar before the report is printed

    # --- report ---------------------------------------------------------
    fr = frontier(archive, args.d_bins)
    print(f"\n=== P*(D) frontier: best departure precision per departure-rate bin ===")
    print(f"  null (uniform random departure) ~ 12.0% | best measured elsewhere = 18.6%\n")
    print(f"{'D range':>12}{'n':>5}{'best P':>9}{'best acc':>10}{'lcb':>8}  schema (best acc)")
    for f in fr:
        print(f"{str(f['d_range']):>12}{f['n_schemas']:>5}{f['best_departure_precision']:>9.1%}"
              f"{f['best_accuracy']:>10.1%}{f['best_lcb_accuracy']:>8.1%}  {f['best_acc_schema']}")

    print(f"\n=== archive grid: best LCB accuracy per (departure bin x cost bin) ===")
    header = "  D\\calls " + "".join(f"{cost_label(b):>9}" for b in range(len(COST_EDGES)))
    print(header)
    for db in range(args.d_bins):
        row = f"  {db / args.d_bins:.1f}-{(db + 1) / args.d_bins:.1f} "
        for cb in range(len(COST_EDGES)):
            e = archive.elite((db, cb))
            row += f"{e.lcb:>9.1%}" if e else f"{'.':>9}"
        print(row)

    ranked = sorted(archive.all_residents(), key=lambda r: -r.lcb)
    print(f"\n=== top {args.top} residents by LCB ===")
    print(f"{'acc':>7}{'lcb':>7}{'n':>6}{'D':>7}{'P':>7}{'calls':>7}{'reps':>6}  schema")
    for r in ranked[: args.top]:
        print(f"{r.accuracy:>7.1%}{r.lcb:>7.1%}{r.n:>6}{r.departure_rate:>7.1%}"
              f"{r.departure_precision:>7.1%}{r.calls_per_exec:>7}{r.reps:>6}  {gloss(r.schema)}")

    out = {"dataset": str(args.dataset), "pool_size": len(pool), "batch": args.batch,
           "d_bins": args.d_bins, "cost_edges": list(COST_EDGES), "depth": args.depth,
           "iterations": {"new": n_new, "deepen": n_deepen, "improved": n_improved},
           "calls_spent": ev.calls, "executions": ev.executions, "errors": ev.errors,
           "n_cells_filled": len(archive.cells), "n_schemas_evaluated": len(evaluated),
           "frontier_P_of_D": fr,
           "grid": {f"{db}_{cb}": (archive.elite((db, cb)).record()
                                   if archive.elite((db, cb)) else None)
                    for db in range(args.d_bins) for cb in range(len(COST_EDGES))},
           "leaderboard": [r.record() for r in ranked[: args.top]]}
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    ev.close()
    print(f"\ncalls spent: {ev.calls}  executions: {ev.executions}  errors: {ev.errors}")
    print(f"execution cache -> {args.exec_cache}\nsummary -> {args.summary_out}")
    print("\nreminder: fitness here is RECOVERY only (the strict-fail pool). A cell with a")
    print("high departure rate is exactly what will destroy retention -- re-score the")
    print("frontier on datasets/supergpqa_answerable_retention.json before believing it.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("datasets/supergpqa_strict_train.json"))
    ap.add_argument("--bestofn-cache", type=Path, default=Path("outputs/bestofn_strict_cache.jsonl"))
    ap.add_argument("--iters", type=int, default=220, help="Archive iterations (new + deepen).")
    ap.add_argument("--init-iters", type=int, default=12, help="Random-init iterations before mutation.")
    ap.add_argument("--batch", type=int, default=96, help="Questions per evaluation (paired prefix).")
    ap.add_argument("--d-bins", type=int, default=10, help="Departure-rate bins.")
    ap.add_argument("--depth", type=int, default=4, help="Residents kept per cell (deep grid).")
    ap.add_argument("--deepen-prob", type=float, default=0.25,
                    help="Share of iterations spent re-evaluating an elite instead of proposing.")
    ap.add_argument("--crossover-prob", type=float, default=0.3)
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--base-urls", default="http://localhost:7472/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--answer-tokens", type=int, default=3072)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--pool-limit", type=int, default=None)
    ap.add_argument("--max-calls", type=int, default=80000, help="Hard LLM-call budget.")
    ap.add_argument("--report-every", type=int, default=10)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-progress", dest="progress", action="store_false", default=True,
                    help="Disable the tqdm call-progress bar (use when redirecting to a log file).")
    ap.add_argument("--exec-cache", type=Path, default=Path("outputs/mapelites_exec_cache.jsonl"))
    ap.add_argument("--summary-out", type=Path, default=Path("outputs/mapelites_summary.json"))
    main(ap.parse_args())

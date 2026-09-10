"""Best-of-N for an ARBITRARY schema (not just the minimal one) + the departure
decomposition + deployable aggregators at every k.

WHY THIS EXISTS
    scripts/bestofn_mcq.py hardcodes MINIMAL_SCHEMA, so the only resampling curve
    in outputs/ is for single_pass. That leaves the two most load-bearing claims
    in the project uncontrolled:

    1. The portfolio sweep found a heterogeneous portfolio beats resampling by
       ~+2pp at matched calls -- but its only homogeneous control was
       single_pass@k (1 call/sample). The control it NEEDS is always_critic@k
       (2 calls/sample), because always_critic is the arm the portfolio is
       actually built around. If always_critic@4 (8 calls) reaches ~12%, the
       portfolio's 12.0% at 8 calls is not a win at all and E4 is dead.

    2. Nobody has measured whether repeated samples of ONE good schema DECORRELATE.
       That is the entire premise of ensembling. If 7 always_critic samples all
       depart to the same wrong letter, schema diversity is doing real work; if
       they scatter as widely as 6 different schemas do, the "portfolio" is just
       resampling in a costume.

WHAT IT REPORTS, per schema
    - pass@k, k=1..N (Chen et al. 2021), plotted per CALL (k x calls_per_sample)
    - deployable aggregators at each k -- plurality / anti_plurality / singleton /
      drop_base_first / drop_base_anti -- imported from portfolio_sweep.py so the
      rules are byte-identical to the sweep's
    - the departure decomposition: accuracy = D x P, where D = P(letter != base)
      and base is the modal letter of the cached 7 minimal-schema samples
    - DECORRELATION: P(two samples differ), mean distinct letters at k, mean
      distinct DEPARTURE letters at k -- the numbers that decide E4
    - per-sample sweep-compatible caches (--emit-sweep-arms), so the portfolio
      sweep can be re-run treating each repeat as its own arm

    python3 scripts/repeat_schema_mcq.py --schemas always_critic --n 7
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations, cycle
from math import comb
from pathlib import Path
from statistics import mean
from threading import Lock

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import debate_mcq as D
from portfolio_sweep import RULES          # one definition of the aggregators

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


ALWAYS_CRITIC = {"rounds": [{"personas": ["solver"]}, {"personas": ["critic"]}], "final": "last"}
SCHEMAS = {
    "single_pass": D.MINIMAL_SCHEMA,
    "always_critic": ALWAYS_CRITIC,
    "self_critique": D.self_critique_schema(),
    "fixed_debate": D.fixed_debate_schema(3),
}
AGG_RULES = ["plurality", "anti_plurality", "singleton", "drop_base_first", "drop_base_anti"]


def n_calls(schema: dict) -> int:
    return sum(len(r["personas"]) for r in schema["rounds"])


def pass_at_k(n: int, c: int, k: int) -> float:
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def run_sample(row, sname, schema, client, model, temp, max_tokens) -> dict:
    try:
        letter = D.execute_schema(client, model, row["question"], list(row["options"]),
                                  schema, temp, max_tokens)
        return {"id": row["id"], "schema_name": sname, "letter": letter,
                "correct": (letter == row["answer_letter"]), "status": "ok"}
    except Exception as exc:
        return {"id": row["id"], "schema_name": sname, "status": "error", "error": str(exc)}


# --- analysis --------------------------------------------------------------

def aggregate_at_k(recs, base, rules, kmax, repeats, rng):
    """{k: {rule: accuracy}} by subsampling k of the N letters per question."""
    out = {}
    for k in range(1, kmax + 1):
        per_rule = {r: [] for r in rules}
        for r in recs:
            ls, g = r["letters"], r["answer_letter"]
            if len(ls) < k:
                continue
            b = base.get(r["id"])
            idxs = (list(combinations(range(len(ls)), k))
                    if comb(len(ls), k) <= repeats else
                    [rng.sample(range(len(ls)), k) for _ in range(repeats)])
            hits = {ru: 0 for ru in rules}
            for idx in idxs:
                ballot = [(f"s{j}", ls[j]) for j in idx]
                ctx = {"weights": {f"s{j}": 1.0 for j in idx}, "base": b}
                for ru in rules:
                    hits[ru] += int(RULES[ru](ballot, ctx) == g)
            for ru in rules:
                per_rule[ru].append(hits[ru] / len(idxs))
        out[k] = {ru: (mean(v) if v else 0.0) for ru, v in per_rule.items()}
    return out


def decorrelation(recs, base, kmax):
    """How much do repeats of the SAME schema scatter? The E4 premise test."""
    pair_diff, dist_at_k, dep_dist_at_k = [], {k: [] for k in range(1, kmax + 1)}, {k: [] for k in range(1, kmax + 1)}
    for r in recs:
        ls = [l for l in r["letters"] if l]
        if len(ls) < 2:
            continue
        pairs = list(combinations(range(len(ls)), 2))
        pair_diff.append(sum(ls[a] != ls[b] for a, b in pairs) / len(pairs))
        b = base.get(r["id"])
        for k in range(1, min(kmax, len(ls)) + 1):
            combos = list(combinations(range(len(ls)), k))[:60]
            dist_at_k[k].append(mean(len({ls[j] for j in c}) for c in combos))
            dep_dist_at_k[k].append(mean(len({ls[j] for j in c if ls[j] != b}) for c in combos))
    return {"pairwise_disagreement": mean(pair_diff) if pair_diff else 0.0,
            "distinct_letters_at_k": {k: (mean(v) if v else 0.0) for k, v in dist_at_k.items()},
            "distinct_departures_at_k": {k: (mean(v) if v else 0.0) for k, v in dep_dist_at_k.items()}}


def summarize_schema(recs, schema, base, n, agg_repeats, rng):
    ok = [r for r in recs if r["n"] > 0]
    m = len(ok)
    calls = n_calls(schema)
    dep = dep_hit = execs = 0
    for r in ok:
        b = base.get(r["id"])
        if b is None:       # no cached base letter -> departure is undefined, not "always"
            continue
        for l, c in zip(r["letters"], r["correct_flags"]):
            execs += 1
            if l is not None and l != b:
                dep += 1
                dep_hit += int(c)
    passk = {k: (mean(pass_at_k(r["n"], r["n_correct"], k) for r in ok) if m else 0.0)
             for k in range(1, n + 1)}
    return {
        "schema": schema, "calls_per_sample": calls, "n_questions": m, "n_samples": n,
        "accuracy_at_1": round(passk.get(1, 0.0), 4),
        "pass_at_k": {k: round(v, 4) for k, v in passk.items()},
        "pass_at_k_by_calls": {k * calls: round(v, 4) for k, v in passk.items()},
        "departure_rate": round(dep / execs, 4) if execs else 0.0,
        "departure_precision": round(dep_hit / dep, 4) if dep else 0.0,
        "aggregators_at_k": {k: {ru: round(v, 4) for ru, v in d.items()}
                             for k, d in aggregate_at_k(ok, base, AGG_RULES, n, agg_repeats, rng).items()},
        "decorrelation": decorrelation(ok, base, n),
    }


# --- driver ----------------------------------------------------------------

def main(args):
    import random
    rng = random.Random(args.seed)

    rows = json.loads(Path(args.dataset).read_text())
    if args.limit is not None:
        rows = rows[: args.limit]
    by_id = {r["id"]: r for r in rows}

    names = [s.strip() for s in args.schemas.split(",") if s.strip()]
    schemas = dict(SCHEMAS)
    if args.schema_json:
        schemas.update(json.loads(Path(args.schema_json).read_text()))
    unknown = [s for s in names if s not in schemas]
    if unknown:
        raise SystemExit(f"unknown schemas {unknown}; known: {sorted(schemas)}")
    for s in names:
        if not D.validate(schemas[s]):
            raise SystemExit(f"schema {s!r} is not grammar-valid: {schemas[s]}")

    base = {}
    if args.bestofn_cache.exists():
        for line in args.bestofn_cache.open():
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("status") == "ok":
                    base[r["id"]] = r.get("majority")
        print(f"base letters from {args.bestofn_cache}: {len(base)}")
    else:
        print(f"  ! {args.bestofn_cache} missing -- departure and drop_base_* stats will be empty")

    urls = [u.strip() for u in args.base_urls.split(",") if u.strip()]
    clients = cycle([OpenAI(base_url=u, api_key=args.api_key) for u in urls])
    for p in (args.cache, args.summary_out):
        p.parent.mkdir(parents=True, exist_ok=True)

    tasks = [(row, s, i) for row in rows for s in names for i in range(args.n)]
    total_calls = sum(n_calls(schemas[s]) for s in names) * args.n * len(rows)
    print(f"{len(rows)} questions x {len(names)} schemas x N={args.n} = {len(tasks)} executions "
          f"= {total_calls} LLM calls | endpoints={len(urls)} | workers={args.workers}")
    for s in names:
        print(f"    {s:<16} {n_calls(schemas[s])} calls/sample  {json.dumps(schemas[s])}")

    agg = {(r["id"], s): {"letters": [], "correct_flags": [], "n_correct": 0, "n_errors": 0}
           for r in rows for s in names}
    pending = {key: args.n for key in agg}
    records, lock = [], Lock()

    with open(args.cache, "w") as cache, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_sample, row, s, schemas[s], next(clients), args.model,
                               args.temperature, args.answer_tokens)
                   for row, s, _ in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="repeat", unit="exec"):
            r = fut.result()
            key = (r["id"], r["schema_name"])
            with lock:
                a = agg[key]
                if r["status"] == "ok":
                    a["letters"].append(r["letter"])
                    a["correct_flags"].append(bool(r["correct"]))
                    a["n_correct"] += int(r["correct"])
                else:
                    a["n_errors"] += 1
                pending[key] -= 1
                if pending[key] == 0:
                    row = by_id[key[0]]
                    tally = Counter(l for l in a["letters"] if l)
                    rec = {"id": key[0], "schema_name": key[1], "field": row.get("field"),
                           "difficulty": row.get("difficulty"), "answer_letter": row["answer_letter"],
                           "n": len(a["letters"]), "n_correct": a["n_correct"],
                           "n_errors": a["n_errors"], "any_correct": a["n_correct"] > 0,
                           "letters": a["letters"], "correct_flags": a["correct_flags"],
                           "majority": tally.most_common(1)[0][0] if tally else None,
                           "status": "ok"}
                    cache.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    cache.flush()
                    records.append(rec)

    summary = {"dataset": str(args.dataset), "n_samples": args.n, "temperature": args.temperature,
               "per_schema": {}}
    for s in names:
        recs = [r for r in records if r["schema_name"] == s]
        summary["per_schema"][s] = summarize_schema(recs, schemas[s], base, args.n,
                                                    args.agg_repeats, rng)
    args.summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # --- report ---------------------------------------------------------
    for s in names:
        d = summary["per_schema"][s]
        c = d["calls_per_sample"]
        print(f"\n=== {s}  ({c} calls/sample, n={d['n_questions']}) ===")
        print(f"  accuracy@1 {d['accuracy_at_1']:.1%}   departure rate {d['departure_rate']:.1%}"
              f"   departure precision {d['departure_precision']:.1%}")
        print(f"  pairwise disagreement between two samples: {d['decorrelation']['pairwise_disagreement']:.1%}")
        print(f"\n  {'k':>3}{'calls':>7}{'pass@k':>9}" + "".join(f"{r[:13]:>14}" for r in AGG_RULES)
              + f"{'distinct':>10}{'dist.dep':>10}")
        for k in range(1, args.n + 1):
            a = d["aggregators_at_k"][k]
            print(f"  {k:>3}{k * c:>7}{d['pass_at_k'][k]:>9.1%}"
                  + "".join(f"{a[r]:>14.1%}" for r in AGG_RULES)
                  + f"{d['decorrelation']['distinct_letters_at_k'][k]:>10.2f}"
                  + f"{d['decorrelation']['distinct_departures_at_k'][k]:>10.2f}")

    if args.emit_sweep_arms:
        tag = args.dataset.stem.replace("supergpqa_", "").replace("frames_", "")
        for s in names:
            for i in range(args.n):
                out = args.outdir / f"retrieve_{s}_s{i + 1}_{tag}_cache.jsonl"
                with open(out, "w") as f:
                    for r in records:
                        if r["schema_name"] != s or i >= r["n"]:
                            continue
                        f.write(json.dumps({"id": r["id"], "arm": f"{s}_s{i + 1}",
                                            "field": r["field"], "difficulty": r["difficulty"],
                                            "answer_letter": r["answer_letter"],
                                            "answer": r["letters"][i],
                                            "correct": r["correct_flags"][i],
                                            "schema": schemas[s], "status": "ok"},
                                           ensure_ascii=False) + "\n")
        print(f"\nsweep-compatible per-sample arms -> {args.outdir}/retrieve_<schema>_s<i>_{tag}_cache.jsonl")
        print("  re-run portfolio_sweep.py with --arms always_critic_s1,...,always_critic_s7 to")
        print("  compare a HOMOGENEOUS portfolio against the heterogeneous one in one framework.")
    print(f"\ncache -> {args.cache}\nsummary -> {args.summary_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("datasets/supergpqa_strict_test.json"))
    ap.add_argument("--schemas", default="always_critic",
                    help=f"Comma-separated names from {sorted(SCHEMAS)} (or --schema-json).")
    ap.add_argument("--schema-json", type=Path, default=None,
                    help="JSON file of {name: schema} merged into the registry.")
    ap.add_argument("--n", type=int, default=7, help="Repeats per (question, schema).")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--base-urls", default="http://localhost:7472/v1",
                    help="Comma-separated vLLM endpoints; calls round-robin across them.")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--answer-tokens", type=int, default=3072)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bestofn-cache", type=Path, default=Path("outputs/bestofn_strict_cache.jsonl"),
                    help="Supplies the label-free base letter for the departure decomposition.")
    ap.add_argument("--agg-repeats", type=int, default=40, help="Subsamples per (question,k).")
    ap.add_argument("--emit-sweep-arms", action="store_true",
                    help="Also write one portfolio_sweep-compatible arm cache per repeat index.")
    ap.add_argument("--outdir", type=Path, default=Path("outputs"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", type=Path, default=Path("outputs/repeat_schema_strict_test_cache.jsonl"))
    ap.add_argument("--summary-out", type=Path,
                    default=Path("outputs/repeat_schema_strict_test_summary.json"))
    main(ap.parse_args())

"""Method 3: ONE master recipe, trimmed or stretched live, per question.

Top-down per-question adaptation. Every question starts the project's best
contrarian-free recipe -- the E7 winner (4 solvers -> reader solver -> 3 rewritten
critics, answer = last letter) -- but the recipe runs one round at a time, and
after every round a rule over the committed letters (plain code, no model call)
decides: end now, continue as planned, or append rounds beyond the plan.

  settled -- a speaker switched to a new letter and the NEXT speaker independently
             committed the same new letter. Confirmed switch: stop immediately,
             even mid-recipe. Later critics can wander off a correct fix; lock it.
  parked  -- the recipe finished and no post-round-1 commitment ever left the
             round-1 majority. Staring harder at the same debate won't help, so
             the extension brings in speakers who have NOT seen the stuck answer:
             eliminator (question only) -> expert + fresh solver (see only the
             eliminator's output; it commits no letter) -> verifier closes.
  churn   -- the recipe finished with the letter moving but never confirmed. The
             debate has candidates; the problem is judging them: verifier (stop if
             it backs any already-committed letter) -> synthesizer closes.

Every executed round is byte-identical to the fixed recipe's, so the built-in
fixed-recipe control (--no-baseline to skip) reuses the same round cache and only
pays for rounds the adaptive pass trimmed away.

    python3 scripts/adaptive_debate_mcq.py
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import debate_mcq as D
import schema_fitness as SF
from schema_fitness import BudgetExhausted, Pool, RoundRunner, load_base_letters, log

# The E7 winner topology (outputs/evolve_persona_text_summary.json, i019). Its
# rewritten critic prompt is SF.REWRITTEN_CRITIC.
MASTER_ROUNDS = [{"personas": ["solver", "solver", "solver", "solver"]},
                 {"personas": ["solver"]},
                 {"personas": ["critic"]},
                 {"personas": ["critic"]},
                 {"personas": ["critic"]}]
HARD_CAP = 20                            # speakers; structural max is 12


def question_prompts(row: dict) -> dict:
    return {"critic": SF.REWRITTEN_CRITIC,
            "expert": D.EXPERT_TMPL.format(
                field=row.get("field") or row.get("discipline") or "the relevant field")}


def committed_with_round(rounds: list, n: int) -> list[tuple[int, str]]:
    """(round index, letter) for every parseable commitment, in speaker order."""
    out = []
    for i, rnd in enumerate(rounds):
        for p, r in rnd:
            if p in D.NON_ANSWERING:
                continue
            if (l := D.extract_letter(r, n)):
                out.append((i, l))
    return out


def modal_letter(letters: list[str]) -> str | None:
    """Most common letter; ties go to the earliest-seen (sorted() is stable, so
    Counter.most_common already resolves ties by first insertion)."""
    return Counter(letters).most_common(1)[0][0] if letters else None


def is_confirmed_switch(commits: list[tuple[int, str]], baseline: str | None) -> bool:
    """The last two commitments are both post-round-1 speakers (who have seen the
    debate -- round 1's parallel solvers are opinions, not switches), agree with
    each other, and differ from the round-1 majority."""
    if len(commits) < 2:
        return False
    (ri, li), (rj, lj) = commits[-2], commits[-1]
    return ri >= 1 and rj >= 1 and li == lj and li != baseline


def run_question(runner: RoundRunner, row: dict, args) -> dict:
    qid, gold, n = row["id"], row["answer_letter"], len(row["options"])
    prompts = question_prompts(row)
    specs: list[dict] = []
    rounds: list = []

    def run(spec: dict) -> list:
        rounds.append(runner.run_round(qid, rounds, specs, spec, rep=args.rep,
                                       prompts=prompts))
        specs.append(spec)
        assert sum(len(s["personas"]) for s in specs) <= HARD_CAP
        return rounds[-1]

    track = closer = final = baseline = None
    for i, spec in enumerate(MASTER_ROUNDS):
        run(spec)
        commits = committed_with_round(rounds, n)
        if i == 0:
            baseline = modal_letter([l for _, l in commits])
            continue
        if is_confirmed_switch(commits, baseline):
            track = "settled_early" if i < len(MASTER_ROUNDS) - 1 else "settled_full"
            final = commits[-1][1]
            break

    if track is None:                                       # master finished unsettled
        post = [l for ri, l in committed_with_round(rounds, n) if ri >= 1]
        if not post or all(l == baseline for l in post):
            track = "parked"
            run({"personas": ["eliminator"], "sees": "none"})
            # a plain solver, NOT `independent`: independent is force-blinded by the
            # executor, and this voice must see the eliminator's survivor list
            pair = run({"personas": ["expert", "solver"], "sees": "last_round"})
            pl = [D.extract_letter(r, n) for _, r in pair]
            if len(pl) == 2 and pl[0] is not None and pl[0] == pl[1]:
                final, closer = pl[0], "newcomers_agree"    # two fresh voices agree
            else:
                v = run({"personas": ["verifier"], "sees": "all"})
                final, closer = (D.extract_letter(v[0][1], n) if v else None), "verifier"
        else:
            track = "churn"
            prior = set(SF.committed_letters(rounds, n))
            v = run({"personas": ["verifier"], "sees": "all"})
            vl = D.extract_letter(v[0][1], n) if v else None
            if vl is not None and vl in prior:              # some letter now backed twice
                final, closer = vl, "verifier_backed"
            else:
                s = run({"personas": ["synthesizer"], "sees": "all"})
                final, closer = (D.extract_letter(s[0][1], n) if s else None), "synthesizer"

    if final is None:                                       # closer unparseable
        letters = SF.committed_letters(rounds, n)
        final = letters[-1] if letters else None

    return {"qid": qid, "gold": gold, "track": track, "closer": closer,
            "baseline": baseline, "letter": final,
            "correct": final is not None and final == gold,
            "letters": [f"{ri}:{l}" for ri, l in committed_with_round(rounds, n)],
            "schema": SF.gloss({"rounds": specs, "final": "last"}),
            "n_calls": sum(len(s["personas"]) for s in specs)}


def run_fixed(runner: RoundRunner, row: dict, args) -> dict:
    """The un-adapted master recipe -- the control. Shares every prefix round with
    the adaptive pass through the cache, so only trimmed-away rounds cost calls."""
    qid, n = row["id"], len(row["options"])
    prompts = question_prompts(row)
    specs, rounds = [], []
    for spec in MASTER_ROUNDS:
        rounds.append(runner.run_round(qid, rounds, specs, spec, rep=args.rep,
                                       prompts=prompts))
        specs.append(spec)
    letter = D.final_letter("last", rounds, n)
    return {"qid": qid, "letter": letter,
            "correct": letter is not None and letter == row["answer_letter"]}


def run_over_pool(fn, qids, workers, runner, label):
    done, lock, stop = [0], threading.Lock(), [False]

    def work(qid):
        if stop[0]:
            return None
        try:
            r = fn(qid)
        except BudgetExhausted:
            stop[0] = True
            return None
        with lock:
            done[0] += 1
            runner.stage(f"{label} {done[0]}/{len(qids)}")
        return r

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = [r for r in pool.map(work, qids) if r is not None]
    if stop[0]:
        log("  call budget exhausted -- partial results below")
    return results


def main(args):
    base = load_base_letters(args.bestofn_cache)
    pool = Pool(args.dataset, base, seed=args.seed, limit=args.pool_limit)
    qids = pool.batch(args.batch) if args.batch else list(pool.order)
    runner = RoundRunner(pool.rows, args.base_urls, args.model, args.temperature,
                         args.answer_tokens, args.cache, max_calls=args.max_calls,
                         api_key=args.api_key, progress=args.progress)
    log(f"adaptive depth: {len(qids)} questions | master {SF.gloss({'rounds': MASTER_ROUNDS, 'final': 'last'})}")

    records = run_over_pool(lambda q: run_question(runner, pool.rows[q], args),
                            qids, args.workers, runner, "adaptive")
    fixed = []
    if args.baseline:
        fixed = run_over_pool(lambda q: run_fixed(runner, pool.rows[q], args),
                              [r["qid"] for r in records], args.workers, runner, "fixed")
    runner.close()

    with args.records_out.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    k = sum(r["correct"] for r in records)
    by_track = {t: {"n": sum(1 for r in records if r["track"] == t),
                    "correct": sum(r["correct"] for r in records if r["track"] == t)}
                for t in ("settled_early", "settled_full", "parked", "churn")}
    fk = sum(r["correct"] for r in fixed)
    fmap = {r["qid"]: r["correct"] for r in fixed}
    summary = {"dataset": str(args.dataset), "n": len(records), "n_correct": k,
               "accuracy": k / len(records) if records else 0.0,
               "by_track": by_track,
               "closers": dict(Counter(r["closer"] for r in records
                                       if r["closer"]).most_common()),
               "schema_histogram": dict(Counter(r["schema"]
                                                for r in records).most_common()),
               "avg_calls": (sum(r["n_calls"] for r in records)
                             / len(records)) if records else 0.0,
               "fixed_recipe": {"n": len(fixed), "n_correct": fk,
                                "accuracy": fk / len(fixed) if fixed else None,
                                "adaptive_right_fixed_wrong":
                                    sum(1 for r in records
                                        if r["correct"] and fmap.get(r["qid"]) is False),
                                "fixed_right_adaptive_wrong":
                                    sum(1 for r in records
                                        if not r["correct"] and fmap.get(r["qid"]))},
               "rep": args.rep, "calls_spent": runner.calls,
               "rounds_run": runner.rounds_run, "errors": runner.errors}
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\nadaptive accuracy: {summary['accuracy']:.1%} ({k}/{len(records)}) "
          f"at {summary['avg_calls']:.1f} calls/question")
    for t, s in by_track.items():
        if s["n"]:
            print(f"  {t:<14} n={s['n']:<5} acc={s['correct'] / s['n']:.1%}")
    if fixed:
        print(f"fixed recipe:      {summary['fixed_recipe']['accuracy']:.1%} "
              f"({fk}/{len(fixed)}) at 8.0 calls/question")
        print(f"  adaptive right / fixed wrong: "
              f"{summary['fixed_recipe']['adaptive_right_fixed_wrong']}   "
              f"fixed right / adaptive wrong: "
              f"{summary['fixed_recipe']['fixed_right_adaptive_wrong']}")
    print(f"calls spent: {runner.calls}  rounds: {runner.rounds_run}  errors: {runner.errors}")
    print(f"round cache -> {args.cache}\nrecords -> {args.records_out}\n"
          f"summary -> {args.summary_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("datasets/supergpqa_600_train.json"))
    ap.add_argument("--bestofn-cache", type=Path, default=Path("outputs/bestofn_strict_cache.jsonl"))
    ap.add_argument("--batch", type=int, default=None, help="First N of the seeded shuffle; default all.")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--no-baseline", dest="baseline", action="store_false", default=True,
                    help="Skip the fixed-recipe control pass.")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--base-urls", default="http://localhost:7472/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--answer-tokens", type=int, default=3072)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--pool-limit", type=int, default=None)
    ap.add_argument("--max-calls", type=int, default=2000000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-progress", dest="progress", action="store_false", default=True)
    ap.add_argument("--cache", type=Path, default=Path("outputs/adaptive_rounds_cache.jsonl"))
    ap.add_argument("--records-out", type=Path, default=Path("outputs/adaptive_debate_records.jsonl"))
    ap.add_argument("--summary-out", type=Path, default=Path("outputs/adaptive_debate_summary.json"))
    main(ap.parse_args())

"""pass@k and avg@k on the dev split for programs chosen on the train split.

Runs each program k times per dev question, at k different replicates, so the
runs are independent samples of the same program rather than one lucky draw:

  avg@k   mean accuracy over the k runs -- what one deployment gets, with the
          sampling noise averaged out. This is the number to quote.
  pass@k  at least one of the k runs was correct. Needs the answer key to pick
          the winner, so it is an ORACLE bound, not a deployable result. It is
          reported to show how much of the gap is sampling luck.
  vote@k  the most common committed letter across the k runs, ties going to the
          earliest. Deployable, and the honest way to spend k runs.

Programs come from an --evolved-out file (its top entries) and/or by name from
the built-in reference programs, so the evolved winner and the hand-written
comparators are measured the same way on the same questions and replicates.

Every round goes through the usual cache, so a replicate already recorded by an
earlier run is free. Cost for what is missing: k x questions x calls/question.

    # the three fresh reference programs plus the top evolved one, k=4, 300 q
    python scripts/eval_passk_programs.py --live --model Qwen/Qwen3.5-27B \\
        --no-eliminator --digest-head 300 --digest-tail 900 \\
        --evolved-out outputs/program_mcq_evolved_live_qwen27b_with_baselines.json \\
        --top 1 --refs program_b,program_fixed,solver_critic \\
        --k 4 --n-dev 300 \\
        --live-cache outputs/program_live_rounds_cache_qwen27b.jsonl \\
        --out outputs/passk_qwen27b.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import debate_mcq as D  # noqa: E402
import schema_fitness as SF  # noqa: E402
import evolve_program_mcq as M  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Confidence interval for a proportion; honest at small n, unlike +-sqrt(p(1-p)/n)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def sign_p(w: int, l: int) -> float:
    m = w + l
    if m == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evolved-out", type=Path, default=None,
                    help="Results file whose top programs should be measured.")
    ap.add_argument("--top", type=int, default=1, help="How many of its top programs.")
    ap.add_argument("--refs", default="program_b,program_fixed,solver_critic",
                    help="Comma-separated reference programs to measure alongside "
                         "(empty for none).")
    ap.add_argument("--k", type=int, default=4, help="Runs per question.")
    ap.add_argument("--reps", default=None,
                    help="Comma-separated replicate numbers to use (default 0..k-1). "
                         "Replicates already recorded are free.")
    ap.add_argument("--n-dev", type=int, default=None,
                    help="Measure a random subset of the dev half (same subset for "
                         "every program). Default: all of it.")
    ap.add_argument("--dataset", type=Path, default=ROOT / "datasets/supergpqa_strict_train.json")
    ap.add_argument("--seed", type=int, default=0,
                    help="Must match the run that produced the programs: it fixes the "
                         "train/dev split.")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/passk.json")
    # execution
    ap.add_argument("--live", action="store_true", help="Run missing rounds on the model.")
    ap.add_argument("--cache", type=Path, default=ROOT / "outputs/adaptive_rounds_cache.jsonl")
    ap.add_argument("--treegrow-cache", type=Path, default=ROOT / "outputs/treegrow_rounds_cache.jsonl")
    ap.add_argument("--live-cache", type=Path, default=ROOT / "outputs/program_live_rounds_cache.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8")
    ap.add_argument("--base-urls", default="http://localhost:7472/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--answer-tokens", type=int, default=None,
                    help="Output budget per reasoning call (default 3072; 6144 under --v2).")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-calls-per-question", type=int, default=16)
    ap.add_argument("--max-total-calls", type=int, default=400_000)
    ap.add_argument("--ignore-cache-lock", action="store_true")
    # must match the run being measured, or the programs mean something else
    ap.add_argument("--no-eliminator", action="store_true")
    ap.add_argument("--commit-followup", action="store_true",
                    help="Nudge a reply with no answer letter to commit (see "
                         "evolve_program_mcq.py). Changes the round cache key.")
    ap.add_argument("--digest-head", type=int, default=700)
    ap.add_argument("--digest-tail", type=int, default=0)
    ap.add_argument("--v2", action="store_true",
                    help="v2 pipeline: careful-reasoning prompts, 6144-token replies, and a "
                         "summary follow-up whose text is what later personas read. Changes "
                         "the round cache key (v=2). Needs a digest tail; replaces "
                         "--commit-followup.")
    args = ap.parse_args()

    if args.v2 and args.commit_followup:
        raise SystemExit("--v2 already commits through its summary call; drop --commit-followup")
    if args.v2 and args.digest_tail <= 0:
        raise SystemExit("--v2 needs --digest-tail > 0 (e.g. --digest-head 300 --digest-tail 900): "
                         "short replies are shown unsummarized and a head-only cut drops their answer")
    if args.answer_tokens is None:
        args.answer_tokens = 6144 if args.v2 else 3072
    if (args.digest_head, args.digest_tail) != (700, 0):
        D.set_digest(args.digest_head, args.digest_tail)
        print(f"digest window {args.digest_head}+{args.digest_tail} (non-default cache keys)")
    if args.no_eliminator:
        M.drop_eliminator()
    if args.commit_followup:
        D.set_commit_followup(True)
        print("commit follow-up ON (round cache keys carry c=1)")
    if args.v2:
        SF.set_v2(True)
        print(f"v2 pipeline ON: careful-reasoning prompts, {args.answer_tokens}-token replies, "
              f"summary follow-up (round cache keys carry v=2)")

    # the same split the search used
    rng = random.Random(args.seed)
    rows = {r["id"]: r for r in json.loads(args.dataset.read_text())}
    qids = sorted(rows)
    rng.shuffle(qids)
    dev = qids[len(qids) // 2:]
    if args.n_dev is not None and args.n_dev < len(dev):
        dev = random.Random(args.seed + 1).sample(dev, args.n_dev)
    reps = ([int(x) for x in args.reps.split(",")] if args.reps
            else list(range(args.k)))
    assert len(reps) == args.k, f"--reps must list {args.k} replicates"

    programs: dict[str, dict] = {}
    if args.evolved_out:
        got = json.loads(args.evolved_out.read_text())
        for i, t in enumerate(got["top"][: args.top]):
            programs[f"evolved_{i + 1}"] = t["program"]
    refs = {"program_b": M.PROGRAM_B_NOELIM if args.no_eliminator else M.PROGRAM_B,
            "program_b_vote": (M.PROGRAM_B_NOELIM_VOTE if args.no_eliminator
                               else M.PROGRAM_B_VOTE),
            "program_fixed": M.PROGRAM_FIXED, "program_fixed_vote": M.PROGRAM_FIXED_VOTE,
            **M.PROBES}
    refs.update({f"{k}_vote": M.with_vote_read(v) for k, v in M.PROBES.items()})
    for name in (x.strip() for x in args.refs.split(",") if x.strip()):
        if name not in refs:
            raise SystemExit(f"unknown reference {name!r}; choose from {sorted(refs)}")
        programs[name] = refs[name]
    for name, prog in programs.items():
        M.validate_program(prog)
    print(f"{len(programs)} programs x {len(dev)} dev questions x k={args.k} "
          f"(replicates {reps})")

    read_only = [args.cache, args.treegrow_cache]
    if args.live:
        runner = M.BudgetedRunner(rows, read_only, lock=not args.ignore_cache_lock,
                                  base_urls=args.base_urls, model=args.model,
                                  temperature=args.temperature,
                                  answer_tokens=args.answer_tokens,
                                  cache_path=args.live_cache,
                                  max_calls=args.max_total_calls,
                                  api_key=args.api_key, progress=True)
        runner.reset_budget(None)
    else:
        if args.live_cache.exists():
            read_only.append(args.live_cache)
        runner = M.CacheRunner(read_only)

    results: dict[str, dict] = {}
    try:
        for name, prog in programs.items():
            per_rep, letters, calls = [], [], []
            for rep in reps:
                ev = M.eval_program(prog, runner, rows, dev, rep,
                                    max_calls=args.max_calls_per_question,
                                    workers=args.workers if args.live else 1)
                per_rep.append(ev.marks)
                letters.append(ev.letters)
                calls.append(sum(ev.calls) / len(dev))
            n = len(dev)
            accs = [m.count("1") / n for m in per_rep]
            avg = sum(accs) / len(accs)
            passk = sum(1 for j in range(n) if any(m[j] == "1" for m in per_rep)) / n
            # vote@k: modal committed letter across the k runs ('?' does not vote)
            votes = 0
            for j in range(n):
                got = [L[j] for L in letters if L[j] != "?"]
                if got and Counter(got).most_common(1)[0][0] == rows[dev[j]]["answer_letter"]:
                    votes += 1
            vote = votes / n
            lo, hi = wilson(round(avg * n * len(reps)), n * len(reps))
            results[name] = {"per_run_acc": accs, "avg_at_k": avg, "pass_at_k": passk,
                             "vote_at_k": vote, "calls_per_q": sum(calls) / len(calls),
                             "total_calls_per_q": sum(calls), "n_dev": n, "reps": reps,
                             "avg_ci95": [lo, hi], "outcomes": per_rep,
                             "letters": ["".join(L) for L in letters]}
            print(f"  {name:16} avg@{args.k} {avg:6.2%}  pass@{args.k} {passk:6.2%}  "
                  f"vote@{args.k} {vote:6.2%}  runs {['%.1f%%' % (a*100) for a in accs]}")
    finally:
        if args.live:
            print(f"\nlive calls {runner.calls}, of which commit follow-ups {runner.followups}, "
                  f"summaries {runner.summaries}")
            if args.v2:
                print("v2 stats: " + ", ".join(f"{k} {v}" for k, v in D.V2_STATS.items()))
            runner.close()

    print(f"\n{'program':18}{'avg@%d' % args.k:>9}{'95% CI':>16}{'pass@%d' % args.k:>9}"
          f"{'vote@%d' % args.k:>9}{'calls/q':>9}{'k*calls':>9}")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["avg_at_k"]):
        lo, hi = r["avg_ci95"]
        print(f"{name:18}{r['avg_at_k']:>9.2%}{f'[{lo:.1%}, {hi:.1%}]':>16}"
              f"{r['pass_at_k']:>9.2%}{r['vote_at_k']:>9.2%}"
              f"{r['calls_per_q']:>9.1f}{r['total_calls_per_q']:>9.1f}")

    names = list(results)
    if len(names) > 1:
        top = max(names, key=lambda k: results[k]["avg_at_k"])
        print(f"\npaired against {top}, per replicate (wins/losses summed over the k runs):")
        for other in names:
            if other == top:
                continue
            w = sum(1 for a, b in zip(results[other]["outcomes"], results[top]["outcomes"])
                    for x, y in zip(a, b) if x == "1" and y != "1")
            l = sum(1 for a, b in zip(results[other]["outcomes"], results[top]["outcomes"])
                    for x, y in zip(a, b) if x != "1" and y == "1")
            print(f"  {other:18} +{w} / -{l}   sign test p = {sign_p(w, l):.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"dev_qids": dev, "k": args.k, "reps": reps, "model": args.model,
         "digest": [args.digest_head, args.digest_tail],
         "no_eliminator": args.no_eliminator, "commit_followup": args.commit_followup,
         "v2": args.v2, "answer_tokens": args.answer_tokens,
         "summaries": getattr(runner, "summaries", 0), "followups": getattr(runner, "followups", 0),
         "v2_stats": dict(D.V2_STATS) if args.v2 else None,
         "results": results}, indent=1))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()

"""A Self-Refine file with fewer rounds, made from one with more: no model calls.

A run of selfrefine.py with --max-iters 2 is the first 2 FEEDBACK -> REFINE rounds of a run with
--max-iters 4: the same prompts, the same seeds for every turn (they depend on the question, the run
index and the turn, not on --max-iters), and the loop's stop rule (the feedback says the answer is
correct) is checked turn by turn. So each run's turns are cut after round --iters, and the record
is rebuilt from them exactly as selfrefine.py builds it: the final answer (with --recover, the
most recent answer that names one if the last does not), its finish_reason, the tokens of the kept
turns, and the counts. "converted_from" in "selfrefine" names the source file.

Made on 2026-10-08 to turn the Qwen3.5-4B runs made with 4 rounds (selfrefine_it4_*) into the 2-round
runs run_baselines.py makes since 2026-10-07; those runs used the settings it uses now (28,672-token
answers, 24,576-token feedback, recovery). Runs with an error are not copied, so run_baselines.py
runs them again.

    python baselines/selfrefine_fewer_iters.py --data datasets/gpqa_diamond_test.json \\
        --src baselines/results/selfrefine_it4_qwen35_4b_think_gpqa_diamond_test_rec.jsonl \\
        --out baselines/results/selfrefine_it2_qwen35_4b_think_gpqa_diamond_test_rec.jsonl --iters 2
"""
import argparse
import json
import os

import tasks
from recover import answer_of


def cut(turns: list[dict], iters: int) -> list[dict]:
    """The turns a run with --max-iters iters would have made: INIT, then up to iters
    FEEDBACK -> REFINE rounds, stopping after a feedback that says the answer is correct."""
    kept, i = turns[:1], 1
    for _ in range(iters):
        if i >= len(turns) or turns[i]["role"] != "feedback":
            break
        kept.append(turns[i])
        i += 1
        if tasks.says_correct(kept[-1]["content"]):
            break
        if i >= len(turns) or turns[i]["role"] != "refine":
            break
        kept.append(turns[i])
        i += 1
    return kept


def convert(rec: dict, item: dict, iters: int, recover: bool, src: str) -> dict:
    """The record selfrefine.py --max-iters iters would have written for the same run."""
    turns = cut(rec["selfrefine"]["turns"], iters)
    whole = len(turns) == len(rec["selfrefine"]["turns"])
    ended = rec["selfrefine"].get("no_room") if whole else None     # a stop for want of room, if it is kept
    answers = [t for t in turns if t["role"] in ("init", "refine")]
    final = answers[-1] if answers else None
    fell_back = False
    if recover and final is not None and answer_of(final["content"], item) is None:
        named = [t for t in answers[:-1] if answer_of(t["content"], item) is not None]
        if named:
            final, fell_back = named[-1], True
    n_refine = sum(t["role"] == "refine" for t in turns)
    out = {k: v for k, v in rec.items() if k not in ("content", "finish_reason", "completion_tokens", "selfrefine")}
    out.update({
        "content": final["content"] if final else None,
        "finish_reason": answers[-1]["finish_reason"] if answers else None,
        "completion_tokens": sum(t["completion_tokens"] for t in turns),
        "selfrefine": {
            "n_refinements": n_refine,
            "stopped_early": n_refine < iters and ended is None,
            "init_answer": answers[0]["content"] if answers else None,
            "recovered_turns": sum(t.get("recovery", {}).get("calls", 0) > 0 for t in turns),
            "fell_back": fell_back,
            "no_room": ended,
            "turns": turns,
            "converted_from": src,
        },
    })
    return out


def main(args):
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists: not overwritten")
    items = {it["id"]: it for it in json.load(open(args.data))}
    runs = {}                                       # (id, run index) -> record; error-free, the last write wins
    with open(args.src) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error") is None and r.get("id") in items:
                runs[(r["id"], r["sample_idx"])] = r
    src = os.path.basename(args.src)
    with open(args.out + ".tmp", "w") as f:
        for (qid, _), r in sorted(runs.items()):
            f.write(json.dumps(convert(r, items[qid], args.iters, args.recover, src), ensure_ascii=False) + "\n")
    os.replace(args.out + ".tmp", args.out)
    print(f"{len(runs)} runs of {src} cut to {args.iters} rounds -> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--src", required=True, help="selfrefine.py output with more rounds")
    p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=2)
    p.add_argument("--no-recover", dest="recover", action="store_false",
                   help="the source was run without --recover (its final answer is never taken from an earlier turn)")
    main(p.parse_args())

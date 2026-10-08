"""Self-consistency (Wang et al., ICLR 2023) as an external baseline: sample several chain-of-thought
replies to the same prompt, read the answer of each, and keep the most common answer.

As in the paper:
  - the replies are independent samples of the direct prompt at the sampling temperature (here
    generate.py's samples, so the direct baseline and self-consistency share them);
  - the answer is a plain majority vote over the answers read off the replies (the paper's
    "unweighted sum", which its Table 1 finds as good as the weighted versions); a reply with no
    answer does not vote, and a tie goes to the earliest sample (tasks.vote; for math, answers
    that math-verify finds equal count as one).
The paper samples 40 replies per question; --n sets how many (default 5, the CoT-SC setting of
ADAS, AFlow and MaAS). Run r of --runs uses samples r*n .. r*n+n-1, so runs share no sample.

No model calls here: this groups the samples of --samples (generate.py output, or recover.py's
copy with the cut-off replies recovered) into runs and writes one line per (question, run) to
--out, with the run's n replies as "finals" (the scorer reads them and votes) and their summed
completion tokens. A run is written only when all n of its samples exist. --out is rewritten
from the samples on every call. run_baselines.py first fills the samples file
(generate.py --k runs*n, then recover.py).

    python baselines/self_consistency.py --data datasets/gpqa_diamond_test.json \\
        --samples baselines/results/direct_X_rec.jsonl --n 5 --runs 3 --out baselines/results/sc5_X_rec.jsonl
"""
import argparse
import json
import os


def load_samples(path: str, ids: set) -> dict:
    """(question id, sample index) -> record, error-free ones only; the last write wins."""
    samples = {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:              # partial last line from a killed run
                continue
            if r.get("error") is None and r.get("id") in ids:
                samples[(r["id"], r["sample_idx"])] = r
    return samples


def main(args):
    with open(args.data) as f:
        items = json.load(f)
    if args.limit:
        items = items[: args.limit]
    samples = load_samples(args.samples, {it["id"] for it in items})
    lines, short = [], 0
    for it in items:
        for run in range(args.runs):
            idx = list(range(run * args.n, (run + 1) * args.n))
            recs = [samples.get((it["id"], i)) for i in idx]
            if any(r is None for r in recs):
                short += 1
                continue
            lines.append({"id": it["id"], "sample_idx": run, "error": None,
                          "finals": [r.get("content") for r in recs],
                          "samples": idx,
                          "finish_reason": [r.get("finish_reason") for r in recs],
                          "completion_tokens": sum(r.get("completion_tokens") or 0 for r in recs),
                          "recovered": sum(r.get("recovery", {}).get("calls", 0) > 0 for r in recs)})
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        for rec in lines:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, args.out)
    print(f"self-consistency, {args.n} samples per run: {len(lines)} runs of {len(items)} questions x "
          f"{args.runs} written to {args.out}; {short} runs still miss samples")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--samples", required=True, help="generate.py output (or its recover.py copy)")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=5, help="samples voted over in one run")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--limit", type=int, default=0)
    main(p.parse_args())

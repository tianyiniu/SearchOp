"""Carve a stratified retention split out of the ANSWERABLE pool.

Every debate-side number in this project is measured on questions Qwen already
FAILS. That makes the corpus one-sided in a way that quietly licenses cheating:
the highest-scoring aggregators found by the portfolio sweep (`anti_plurality`,
`drop_base_*`) work precisely BECAUSE the base model's answer is wrong by
construction, and the departure-maximizing personas planned for E1 (`contrarian`,
`eliminator`) are optimizing the same one-sided objective.

On the answerable pool -- questions Qwen gets right 3/3 with CoT -- those methods
should invert. That is not a nuisance check, it is the co-objective:

    recovery  = accuracy on the strict-fail pool     (maximize)
    retention = accuracy on the answerable pool      (do not destroy)

Selection in every design in evol_debate_designs.md is over the PAIR, so this
split has to exist before any of them can be run honestly.

Sampling is stratified by (field, difficulty) with largest-remainder allocation,
so the retention set has the same composition as the fail set it is compared to.

    python3 scripts/make_retention_split.py --n 1000
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Proportional allocation over (field, difficulty), largest-remainder rounding."""
    rng = random.Random(seed)
    strata: dict[tuple, list] = defaultdict(list)
    for r in rows:
        strata[(r.get("field"), r.get("difficulty"))].append(r)
    total = len(rows)
    exact = {k: len(v) * n / total for k, v in strata.items()}
    alloc = {k: min(len(strata[k]), int(v)) for k, v in exact.items()}
    # hand out the remaining slots by largest fractional part
    short = n - sum(alloc.values())
    for k, _ in sorted(exact.items(), key=lambda kv: -(kv[1] - int(kv[1]))):
        if short <= 0:
            break
        if alloc[k] < len(strata[k]):
            alloc[k] += 1
            short -= 1
    out = []
    for k, take in alloc.items():
        if take:
            out.extend(rng.sample(strata[k], take))
    rng.shuffle(out)
    return out[:n]


def compare(sample: list[dict], reference: list[dict], key: str) -> None:
    s, r = Counter(x.get(key) for x in sample), Counter(x.get(key) for x in reference)
    ns, nr = sum(s.values()), sum(r.values())
    print(f"  {key}:")
    for k in sorted(s, key=lambda k: -s[k])[:8]:
        print(f"    {str(k):<34} retention {s[k] / ns:>6.1%}   reference {r.get(k, 0) / nr:>6.1%}")


def main(args):
    rows = json.loads(args.source.read_text())
    ref = json.loads(args.reference.read_text()) if args.reference.exists() else rows
    print(f"source: {args.source} ({len(rows)} rows)")
    print(f"reference composition: {args.reference} ({len(ref)} rows)\n")

    if args.n > len(rows):
        raise SystemExit(f"asked for {args.n} but the pool has {len(rows)}")
    sample = stratified_sample(rows, args.n, args.seed)

    ids = {r["id"] for r in sample}
    if len(ids) != len(sample):
        raise SystemExit("duplicate ids in the sample")
    overlap = ids & {r["id"] for r in ref}
    print(f"sampled {len(sample)} rows | id overlap with reference split: {len(overlap)}"
          f"{'  (expected 0 -- disjoint pools)' if not overlap else '  !! LEAKAGE'}\n")
    compare(sample, ref, "difficulty")
    compare(sample, ref, "field")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(sample, ensure_ascii=False, indent=1))
    manifest = {"created_from": str(args.source), "reference": str(args.reference),
                "n": len(sample), "seed": args.seed, "path": str(args.out),
                "purpose": "retention co-objective: accuracy on questions Qwen already answers",
                "difficulty": dict(Counter(r.get("difficulty") for r in sample)),
                "n_fields": len({r.get("field") for r in sample})}
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nretention split -> {args.out}\nmanifest -> {args.manifest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=Path("datasets/supergpqa_qwen_answerable.json"))
    ap.add_argument("--reference", type=Path, default=Path("datasets/supergpqa_strict_test.json"),
                    help="Split whose (field, difficulty) composition the sample should match.")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("datasets/supergpqa_answerable_retention.json"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("datasets/answerable_retention_manifest.json"))
    main(ap.parse_args())

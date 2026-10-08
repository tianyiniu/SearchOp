"""Draw dev_small: a 300-question subset of the dev split that keeps the same
mix of subject and difficulty as the full 1,515.

Strata are discipline x difficulty (34 cells). Each cell gets its proportional
share of the 300 (largest-remainder rounding, so the counts add up exactly).
Inside a cell, questions are taken round-robin across fields, so one large
field cannot crowd out the small ones.

    python scripts/make_dev_small.py
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def allocate(sizes: dict[tuple, int], n: int) -> dict[tuple, int]:
    """Largest-remainder apportionment of n over the strata."""
    total = sum(sizes.values())
    exact = {k: n * sz / total for k, sz in sizes.items()}
    alloc = {k: min(int(exact[k]), sizes[k]) for k in sizes}
    order = sorted(sizes, key=lambda k: (-(exact[k] - int(exact[k])), k))
    i = 0
    while sum(alloc.values()) < n:
        k = order[i % len(order)]
        if alloc[k] < sizes[k]:
            alloc[k] += 1
        i += 1
    return alloc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=ROOT / "datasets/supergpqa_program_search_dev.json")
    ap.add_argument("--out", type=Path, default=ROOT / "datasets/supergpqa_program_search_dev_small.json")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = ap.parse_args()
    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists; pass --force to redraw it")

    rows = json.loads(args.source.read_text())
    rng = random.Random(args.seed)
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["id"]):
        strata[(r["discipline"], r["difficulty"])].append(r)
    alloc = allocate({k: len(v) for k, v in strata.items()}, args.n)

    picked: list[dict] = []
    for k in sorted(strata):
        by_field: dict[str, list[dict]] = defaultdict(list)
        for r in strata[k]:
            by_field[r["field"]].append(r)
        fields = sorted(by_field)
        rng.shuffle(fields)
        for f in fields:
            rng.shuffle(by_field[f])
        take: list[dict] = []
        while len(take) < alloc[k]:
            for f in fields:
                if by_field[f] and len(take) < alloc[k]:
                    take.append(by_field[f].pop())
        picked.extend(take)

    keep = {r["id"] for r in picked}
    out_rows = [r for r in rows if r["id"] in keep]           # source order
    assert len(out_rows) == args.n == len(keep)
    args.out.write_text(json.dumps(out_rows, indent=1, ensure_ascii=False))

    print(f"{args.out}: {len(out_rows)} of {len(rows)} questions")
    for name in ("difficulty", "discipline"):
        full, small = Counter(r[name] for r in rows), Counter(r[name] for r in out_rows)
        print(f"\n{name:22s} {'dev':>6s} {'share':>7s} {'small':>6s} {'share':>7s}")
        for v, c in full.most_common():
            print(f"{v:22s} {c:6d} {c / len(rows):7.1%} {small[v]:6d} {small[v] / len(out_rows):7.1%}")
    print(f"\nfields covered: {len({r['field'] for r in out_rows})} of {len({r['field'] for r in rows})}")


if __name__ == "__main__":
    main()

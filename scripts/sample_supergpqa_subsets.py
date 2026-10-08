"""Draw two nested SuperGPQA subsets, each split evenly into train and test.

From the full SuperGPQA pool (datasets/supergpqa_raw.json, 26,529 questions):

    2k     2,000 questions  -> supergpqa_2k_train.json  (1,000), supergpqa_2k_test.json  (1,000)
    600      600 questions  -> supergpqa_600_train.json   (300), supergpqa_600_test.json    (300)

The 600 are a subset of the 2k, and the splits nest too: the 600's train half
is inside the 2k's train half and its test half inside the 2k's test half. So
anything labelled for the 600 is reused as is by the 2k, and no question is
train at one size and test at the other.

Sampling is systematic over a sorted list, which stratifies every level of the
sort at once. The pool is sorted by discipline, difficulty, field, subfield,
and a seeded random key inside a subfield; n questions are taken at evenly
spaced positions from a random start. Each discipline x difficulty cell, each
field inside it, and each subfield then gets its proportional share to within
one question, and the total is exact. No cell is too small to handle: a cell
whose share is under one question is included with that probability.

    2k    = systematic draw of 2,000 from the pool
    600   = systematic draw of 600 from the 2k (same sort)
    split = alternate train/test along the same sort: the 600 first, then the
            other 1,400, each from a random first side

Nothing is filtered: flawed items stay in, as they are in the benchmark.

    python scripts/sample_supergpqa_subsets.py
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEVELS = ("discipline", "difficulty", "field", "subfield")
FIELDS = ("id", "question", "options", "answer_letter", "discipline", "field", "subfield",
          "difficulty", "is_calculation")


def sort_key(rng: random.Random, rows: list[dict]) -> list[dict]:
    """The pool in stratification order, random inside a subfield."""
    tie = {r["id"]: rng.random() for r in rows}
    return sorted(rows, key=lambda r: tuple(r[k] for k in LEVELS) + (tie[r["id"]],))


def systematic(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """n rows at evenly spaced positions of the sorted list, from a random start."""
    if n > len(rows):
        raise SystemExit(f"asked for {n} of {len(rows)}")
    ordered = sort_key(rng, rows)
    step = len(ordered) / n
    start = rng.random() * step
    return [ordered[int(start + i * step)] for i in range(n)]


def alternate(rows: list[dict], rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Deal the sorted rows to train and test in turn, from a random first side."""
    if len(rows) % 2:
        raise SystemExit("an even split needs an even count")
    ordered = sort_key(rng, rows)
    first = rng.randrange(2)
    train = [r for i, r in enumerate(ordered) if i % 2 == first]
    test = [r for i, r in enumerate(ordered) if i % 2 != first]
    return train, test


def share(rows: list[dict], level: str) -> dict:
    c = Counter(r[level] for r in rows)
    return {k: v / len(rows) for k, v in c.items()}


def gap(rows: list[dict], pool: list[dict], level: str) -> tuple[float, float]:
    """(total variation distance, largest absolute share difference) to the pool."""
    a, b = share(rows, level), share(pool, level)
    diffs = [abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b)]
    return 0.5 * sum(diffs), max(diffs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=ROOT / "datasets/supergpqa_raw.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "datasets")
    ap.add_argument("--large", type=int, default=2000)
    ap.add_argument("--small", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="overwrite existing subset files")
    args = ap.parse_args()
    if args.small > args.large or args.small % 2 or args.large % 2:
        raise SystemExit("--small must be <= --large and both must be even")

    names = {(size, split): args.out_dir / f"supergpqa_{tag}_{split}.json"
             for size, tag in ((args.large, f"{args.large // 1000}k" if args.large % 1000 == 0 else args.large),
                               (args.small, args.small))
             for split in ("train", "test")}
    manifest_path = args.out_dir / "supergpqa_subsets_manifest.json"
    existing = [p for p in list(names.values()) + [manifest_path] if p.exists()]
    if existing and not args.force:
        raise SystemExit(f"{existing[0]} exists; the subsets are fixed once drawn (pass --force to redraw)")

    pool = []
    for r in json.loads(args.raw.read_text()):
        row = {"id": r["uuid"], **{k: r[k] for k in FIELDS if k != "id"}}
        if not 0 <= ord(row["answer_letter"]) - 65 < len(row["options"]):
            raise SystemExit(f"{row['id']}: answer letter outside its options")
        pool.append(row)
    rng = random.Random(args.seed)

    large = systematic(pool, args.large, rng)
    small = systematic(large, args.small, rng)
    small_ids = {r["id"] for r in small}
    small_train, small_test = alternate(small, rng)
    rest_train, rest_test = alternate([r for r in large if r["id"] not in small_ids], rng)
    splits = {(args.large, "train"): small_train + rest_train, (args.large, "test"): small_test + rest_test,
              (args.small, "train"): small_train, (args.small, "test"): small_test}

    # the nesting and the sizes, checked rather than assumed
    ids = {k: {r["id"] for r in v} for k, v in splits.items()}
    assert len(large) == len({r["id"] for r in large}) == args.large
    assert ids[(args.small, "train")] <= ids[(args.large, "train")]
    assert ids[(args.small, "test")] <= ids[(args.large, "test")]
    assert not ids[(args.large, "train")] & ids[(args.large, "test")]
    for (size, split), rows in splits.items():
        assert len(rows) == size // 2, (size, split, len(rows))

    print(f"pool {len(pool)}; distance to the pool's mix "
          "(TVD = total variation distance; max = largest single share difference)")
    print(f"{'set':>12} " + " ".join(f"{lv + ' TVD/max':>22}" for lv in ("discipline", "difficulty", "field")))
    report = {}
    for (size, split), rows in sorted(splits.items(), key=lambda kv: (-kv[0][0], kv[0][1] != "train")):
        stats = {lv: gap(rows, pool, lv) for lv in ("discipline", "difficulty", "field", "subfield")}
        report[f"{size}_{split}"] = {lv: {"tvd": round(t, 4), "max_abs": round(m, 4)} for lv, (t, m) in stats.items()}
        print(f"{f'{size} {split}':>12} " + " ".join(f"{stats[lv][0]:>13.3f} / {stats[lv][1]:.3f}"
                                                     for lv in ("discipline", "difficulty", "field")))
        names[(size, split)].write_text(json.dumps(rows, indent=1, ensure_ascii=False))

    manifest = {"source": str(args.raw.resolve()), "pool": len(pool), "seed": args.seed,
                "method": "systematic sampling over the pool sorted by " + ", ".join(LEVELS)
                          + "; 600 drawn from the 2k the same way; splits alternate along the same sort",
                "nesting": f"{args.small} train <= {args.large} train, {args.small} test <= {args.large} test",
                "files": {f"{size}_{split}": str(p.relative_to(ROOT)) for (size, split), p in names.items()},
                "ids": {f"{size}_{split}": sorted(i) for (size, split), i in ids.items()},
                "distance_to_pool": report}
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print("\ndifficulty share   pool  " + "  ".join(f"{s}_{p:>5}" for s, p in splits))
    for d in ("easy", "middle", "hard"):
        print(f"{d:>16}  {share(pool, 'difficulty')[d]:.3f}  "
              + "  ".join(f"{share(rows, 'difficulty').get(d, 0):>9.3f}" for rows in splits.values()))
    print("\nwrote " + ", ".join(str(p.relative_to(ROOT)) for p in names.values())
          + f", {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

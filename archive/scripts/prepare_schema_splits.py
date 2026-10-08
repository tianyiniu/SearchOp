"""Prepare the train/test splits for the TEST-TIME schema-selection experiment.

Goal of the larger project: pick an (near-)optimal debate schema for a *held-out*
query with no ground-truth answer, and evaluate only ONE candidate schema on it
(evaluating several would defeat the point). To study that we need a disjoint
corpus/test division:

  supergpqa_train.json  (~4027)  SCHEMA CORPUS  -- schemas are mined / a router is
                                                  fit here; ground truth may be used.
  supergpqa_test.json   ( 1000)  IN-DIST TEST   -- held out; each query gets ONE schema.
  frames_ood_test.json  (   11)  OOD TEST       -- FRAMES reasoning questions, a
                                                  distribution shift (multi-hop QA vs MCQ).

The 1000-question test split is drawn stratified by (field x difficulty) with
largest-remainder apportionment, so the corpus and the test set carry the SAME
difficulty/subject mix -- a router fit on the corpus sees the same distribution it
is scored on. Sampling is seeded and deterministic.

Nothing is overwritten in place: the source splits are copied to a timestamped
backup dir BEFORE anything is written, and the outputs are new filenames.

    python3 scripts/prepare_schema_splits.py                 # defaults: 1000 test, seed 0
    python3 scripts/prepare_schema_splits.py --n-test 1000 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def dump(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1))


def backup_datasets(datasets_dir: Path, backup_root: Path, stamp: str) -> Path:
    """Copy every split json (skip the multi-MB raw dump) to a timestamped dir
    BEFORE we write anything, so a bad run can always be undone."""
    dest = backup_root / f"datasets_{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in sorted(datasets_dir.glob("*.json")):
        if src.name.endswith("_raw.json"):  # 22MB canonical dump, unchanged by us
            continue
        shutil.copy2(src, dest / src.name)
        copied += 1
    print(f"backup: copied {copied} json files -> {dest}")
    return dest


def stratum_key(row: dict) -> tuple[str, str]:
    return (row.get("field", "?"), row.get("difficulty", "?"))


def allocate(strata_sizes: dict[tuple, int], n_test: int, total: int) -> dict[tuple, int]:
    """Largest-remainder apportionment: give each stratum floor(share), then hand
    the leftover seats to the largest fractional remainders. Sums to exactly n_test."""
    exact = {k: n_test * sz / total for k, sz in strata_sizes.items()}
    alloc = {k: int(v) for k, v in exact.items()}
    # never allocate more than the stratum holds
    alloc = {k: min(alloc[k], strata_sizes[k]) for k in alloc}
    remaining = n_test - sum(alloc.values())
    # rank strata by fractional remainder, break ties by stratum key for determinism
    order = sorted(exact, key=lambda k: (-(exact[k] - int(exact[k])), k))
    i = 0
    while remaining > 0 and i < len(order) * 4:  # a few passes in case of caps
        k = order[i % len(order)]
        if alloc[k] < strata_sizes[k]:
            alloc[k] += 1
            remaining -= 1
        i += 1
    return alloc


def split_supergpqa(rows: list[dict], n_test: int, seed: int) -> tuple[list[dict], list[dict], dict]:
    """Stratified split into (train, test). Deterministic given seed."""
    by_stratum: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_stratum[stratum_key(r)].append(r)
    sizes = {k: len(v) for k, v in by_stratum.items()}
    alloc = allocate(sizes, n_test, len(rows))

    rng = random.Random(seed)
    train, test = [], []
    for k in sorted(by_stratum):  # sorted keys => stable iteration
        bucket = by_stratum[k][:]
        rng.shuffle(bucket)
        take = alloc[k]
        test.extend(bucket[:take])
        train.extend(bucket[take:])
    # re-shuffle each split once so downstream order isn't stratum-blocked
    rng.shuffle(train)
    rng.shuffle(test)

    diag = {"n_strata": len(sizes), "allocated": sum(alloc.values())}
    return train, test, diag


def dist(rows: list[dict], attr: str) -> Counter:
    return Counter(r.get(attr) for r in rows)


def print_distribution_check(train: list[dict], test: list[dict]) -> None:
    nt, ne = len(train), len(test)
    print(f"\ndistribution check (train n={nt}  |  test n={ne}):")
    print(f"  {'difficulty':<10} {'train%':>8} {'test%':>8}")
    for d in ("easy", "middle", "hard"):
        tp = 100 * dist(train, "difficulty")[d] / nt if nt else 0
        ep = 100 * dist(test, "difficulty")[d] / ne if ne else 0
        print(f"  {d:<10} {tp:>7.1f}% {ep:>7.1f}%")
    # a few largest fields as a spot-check on subject balance
    top = [f for f, _ in dist(train, "field").most_common(6)]
    print(f"\n  {'field (top 6)':<40} {'train%':>8} {'test%':>8}")
    for f in top:
        tp = 100 * dist(train, "field")[f] / nt if nt else 0
        ep = 100 * dist(test, "field")[f] / ne if ne else 0
        print(f"  {f[:38]:<40} {tp:>7.1f}% {ep:>7.1f}%")


def main(args: argparse.Namespace) -> None:
    datasets = args.datasets_dir
    src = datasets / args.supergpqa
    frames_src = datasets / args.frames
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) BACK UP before touching anything.
    backup_datasets(datasets, args.backup_root, stamp)

    # 2) Stratified split of the SuperGPQA unanswerable pool.
    rows = load(src)
    if args.n_test >= len(rows):
        raise SystemExit(f"--n-test {args.n_test} >= pool size {len(rows)}")
    train, test, diag = split_supergpqa(rows, args.n_test, args.seed)

    # sanity: disjoint, complete, sized right
    train_ids, test_ids = {r["id"] for r in train}, {r["id"] for r in test}
    assert not (train_ids & test_ids), "train/test id overlap!"
    assert len(train) + len(test) == len(rows), "rows lost in split!"
    assert len(test) == args.n_test, f"test size {len(test)} != {args.n_test}"

    train_out = datasets / args.train_out
    test_out = datasets / args.test_out
    dump(train_out, train)
    dump(test_out, test)

    # 3) FRAMES -> canonical OOD test filename (straight copy, no sampling).
    ood_out = datasets / args.ood_out
    frames = load(frames_src)
    dump(ood_out, frames)

    print(f"\nsource pool: {len(rows)}  ({src.name})")
    print(f"  train (schema corpus) -> {train_out.name:<32} n={len(train)}")
    print(f"  test  (in-dist)       -> {test_out.name:<32} n={len(test)}")
    print(f"  ood   (FRAMES)        -> {ood_out.name:<32} n={len(frames)}")
    print(f"  strata (field x difficulty): {diag['n_strata']}")
    print_distribution_check(train, test)

    manifest = {
        "created": stamp, "seed": args.seed,
        "source": str(src), "n_source": len(rows),
        "train": {"path": str(train_out), "n": len(train)},
        "test": {"path": str(test_out), "n": len(test)},
        "ood": {"path": str(ood_out), "n": len(frames)},
        "n_strata": diag["n_strata"],
        "backup": f"{args.backup_root}/datasets_{stamp}",
    }
    (datasets / args.manifest_out).write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest -> {datasets / args.manifest_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets-dir", type=Path, default=Path("datasets"))
    ap.add_argument("--supergpqa", default="supergpqa_qwen_unanswerable.json",
                    help="SuperGPQA unanswerable pool to split.")
    ap.add_argument("--frames", default="frames_qwen_unanswerable.json",
                    help="FRAMES unanswerable questions used as the OOD test.")
    ap.add_argument("--n-test", type=int, default=1000, help="In-distribution test size.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-out", default="supergpqa_train.json")
    ap.add_argument("--test-out", default="supergpqa_test.json")
    ap.add_argument("--ood-out", default="frames_ood_test.json")
    ap.add_argument("--manifest-out", default="schema_splits_manifest.json")
    ap.add_argument("--backup-root", type=Path, default=Path("backups"))
    main(ap.parse_args())

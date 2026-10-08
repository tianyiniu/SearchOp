"""Split a train file's questions into a train split (the search's questions) and a dev
split (questions the search never sees; the global pipeline uses them only to choose its final
programs). The split is a fixed random draw: the same dataset, size and seed give the same
split. Each list keeps the dataset's order.

    python scripts/split_train_dev.py --dataset datasets/supergpqa_600_train.json \\
        --n-dev 100 --seed 0 --out outputs/pipeline_global_gptoss/run1/splits.json

A split file that already exists is kept if it is the same split, and refused if not (a
search's question set never changes under it).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def make_split(ids: list[str], n_dev: int, seed: int) -> dict:
    """{"train": [...], "dev": [...]}: `n_dev` ids drawn at random for dev, the rest train,
    each list in the order of `ids`."""
    if len(set(ids)) != len(ids):
        raise SystemExit("the dataset has repeated question ids")
    if not 0 < n_dev < len(ids):
        raise SystemExit(f"--n-dev must be between 1 and {len(ids) - 1}, not {n_dev}")
    dev = set(random.Random(seed).sample(sorted(ids), n_dev))
    return {"train": [q for q in ids if q not in dev], "dev": [q for q in ids if q in dev]}


def load_split(path: Path) -> dict:
    """A split file, checked: two disjoint lists of distinct ids, neither empty."""
    d = json.loads(Path(path).read_text())
    train, dev = d.get("train"), d.get("dev")
    if not train or not dev:
        raise SystemExit(f"{path}: needs non-empty 'train' and 'dev' lists")
    if len(set(train)) != len(train) or len(set(dev)) != len(dev) or set(train) & set(dev):
        raise SystemExit(f"{path}: the train and dev lists must hold distinct ids and share none")
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True, help="a JSON list of questions with 'id'")
    ap.add_argument("--n-dev", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ids = [r["id"] for r in json.loads(args.dataset.read_text())]
    split = make_split(ids, args.n_dev, args.seed)
    out = {"dataset": str(args.dataset), "n_dev": args.n_dev, "seed": args.seed, **split}
    if args.out.exists():
        old = load_split(args.out)
        if (old["train"], old["dev"]) != (split["train"], split["dev"]):
            raise SystemExit(f"{args.out} exists and holds a different split; move it to make a new one")
        print(f"{args.out} exists and holds this split: {len(split['train'])} train, {len(split['dev'])} dev")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"{len(split['train'])} train and {len(split['dev'])} dev questions -> {args.out}")


if __name__ == "__main__":
    main()

"""Draw the search questions of each group for the v3 program search.

Each group's search set is a fixed-size random sample of the group, drawn
without replacement and weighted towards the questions closest to the group's
centre. Closeness is the distance to the group's medoid in the same
representation the clustering used. The weight is linear in the closeness
rank: in a group of n questions the closest has weight n, the next n-1, and
the farthest 1. One seed fixes the draw; nothing else is tuned.

The rest of the group is its held-out set, from which champion picking later
draws a plain (unweighted) random sample, so held-out scores reflect the whole
group a router would send.

The groups themselves are not changed: the input clusters file is read, never
written, and the output is a new file with the same layout, so
program_space.load_groups reads either.

    python scripts/sample_cluster_subsets.py \\
        --clusters outputs/clusters_train_both.json --out outputs/clusters_train_both_v3.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cluster_questions as C  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def weighted_draw(order: list[int], n_pick: int, rng: np.random.Generator) -> list[int]:
    """`n_pick` positions of `order` (closest first), drawn one at a time
    without replacement with probability proportional to n, n-1, ..., 1."""
    n = len(order)
    if n <= n_pick:
        return list(range(n))
    w = np.arange(n, 0, -1, dtype=np.float64)
    return [int(i) for i in rng.choice(n, size=n_pick, replace=False, p=w / w.sum())]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clusters", type=Path, default=ROOT / "outputs/clusters_train_both.json")
    ap.add_argument("--vectors", type=Path, default=ROOT / "outputs/question_vectors_train.npz")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/clusters_train_both_v3.json")
    ap.add_argument("--per-cluster", type=int, default=50)
    ap.add_argument("--mh-max-mean", type=float, default=0.9,
                    help="as in cluster_questions.py, so the distances are the clustering's own")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = ap.parse_args()
    if args.out.resolve() == args.clusters.resolve():
        raise SystemExit("--out must not be the input clusters file")
    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists; the search questions are fixed once drawn "
                         f"(pass --force to redraw)")

    src = json.loads(args.clusters.read_text())
    vec = np.load(args.vectors)
    ids = vec["ids"].tolist()
    pos = {q: i for i, q in enumerate(ids)}
    x = C.unit(C.representation(vec, src["rep"], src["mh_weight"], args.mh_max_mean))
    rng = np.random.default_rng(args.seed)

    clusters = []
    print(f"{'group':>5} {'size':>5} {'search':>6} {'held':>5} {'medoid in':>9} "
          f"{'mean rank':>9} {'uniform':>8} {'old overlap':>11}")
    for c in src["clusters"]:
        members = list(c["members"])
        med = x[pos[c["medoid"]]]
        dist = {q: float(1.0 - x[pos[q]] @ med) for q in members}
        order = sorted(members, key=lambda q: (dist[q], q))          # closest first; id breaks ties
        picks = weighted_draw(list(range(len(order))), args.per_cluster, rng)
        subset = [order[i] for i in picks]
        chosen = set(subset)
        held = [q for q in members if q not in chosen]
        new = dict(c)
        new["subset"], new["held_out"] = subset, held
        new["subset_rank"] = {order[i]: i + 1 for i in picks}
        clusters.append(new)
        old = set(c.get("subset", [])[: args.per_cluster])
        print(f"{c['cluster']:>5} {len(members):>5} {len(subset):>6} {len(held):>5} "
              f"{str(c['medoid'] in chosen):>9} {np.mean([i + 1 for i in picks]):>9.1f} "
              f"{(len(order) + 1) / 2:>8.1f} {len(old & chosen):>11}")

    out = dict(src)
    out["clusters"] = clusters
    out["per_cluster"] = args.per_cluster
    out["subset_sampling"] = {"rule": "weighted without replacement, weight = n - closeness rank + 1",
                              "distance": "cosine to the group medoid in the clustering representation",
                              "seed": args.seed, "source": str(args.clusters.resolve())}
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

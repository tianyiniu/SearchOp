"""Draw a fixed subset of a dataset file with the same mix of subjects: systematic sampling over the
rows sorted by discipline, field, subfield, difficulty and id (the method of the SuperGPQA subsets,
datasets/supergpqa_subsets_manifest.json). Row i of the sorted list is taken for each of the n
evenly spaced points (i + 0.5) * len / n, so every subject keeps its share to within one question.
No randomness: the same file and size give the same subset.

    python scripts/subset_dataset.py --dataset datasets/math_l5_train.json --n 300 \\
        --out datasets/math_l5_300_train.json

Writes the subset (the rows as they are, in the source file's order) and, beside it,
<out stem>_manifest.json: the source, the method, the ids, and each subject's share in both files.
An existing output is kept if it is the same subset, and refused if not.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def draw(rows: list[dict], n: int) -> list[dict]:
    if not 0 < n < len(rows):
        raise SystemExit(f"--n must be between 1 and {len(rows) - 1}, not {n}")
    key = lambda r: tuple(str(r.get(f) or "") for f in ("discipline", "field", "subfield", "difficulty", "id"))
    ordered = sorted(rows, key=key)
    picked = {ordered[int((i + 0.5) * len(ordered) / n)]["id"] for i in range(n)}
    assert len(picked) == n
    return [r for r in rows if r["id"] in picked]


def shares(rows: list[dict]) -> dict[str, float]:
    c = Counter(r.get("discipline") for r in rows)
    return {k: round(v / len(rows), 4) for k, v in sorted(c.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = json.loads(args.dataset.read_text())
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit(f"{args.dataset} has repeated question ids")
    sub = draw(rows, args.n)
    if args.out.exists():
        old = json.loads(args.out.read_text())
        if [r["id"] for r in old] != [r["id"] for r in sub]:
            raise SystemExit(f"{args.out} exists and holds another subset; move it aside to draw again")
        print(f"{args.out} exists and is this subset; kept")
        return
    args.out.write_text(json.dumps(sub, indent=1, ensure_ascii=False))
    manifest = {"source": str(args.dataset), "n": args.n,
                "method": "systematic sampling over the rows sorted by discipline, field, subfield, difficulty, id",
                "share_by_discipline": {"source": shares(rows), "subset": shares(sub)},
                "ids": [r["id"] for r in sub]}
    man = args.out.with_name(args.out.stem + "_manifest.json")
    man.write_text(json.dumps(manifest, indent=1))
    print(f"{args.dataset.name} -> {args.out.name}: {len(sub)} of {len(rows)} questions; manifest {man.name}")
    for k, v in manifest["share_by_discipline"]["subset"].items():
        print(f"  {k:24s} {v:.3f} (source {manifest['share_by_discipline']['source'][k]:.3f})")


if __name__ == "__main__":
    main()

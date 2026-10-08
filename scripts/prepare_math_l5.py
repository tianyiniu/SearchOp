"""Write MATH level 5 (Hendrycks et al., 2021) in the layout the pipeline and the external baselines
read, split 50/50 into train and test with the same mix of subjects in both.

Source: EleutherAI/hendrycks_math on Hugging Face (pinned revision), the "test" split of its 7
subject configs, "Level 5" problems only: 1,324. The original train split is left out by default
(models are often trained on it); --source-splits train test adds its 2,304 level-5 problems.
Each row becomes:

    id                "math_" + the first 16 hex digits of the problem's SHA-256
    question          the problem
    options           [] (open answer)
    answer            the content of the solution's last \\boxed{...} (the benchmark's own rule:
                      baselines/tasks.py last_boxed); the solution itself is dropped, so no file
                      carries one
    discipline, field, subfield   the subject ("type": Algebra, Geometry, ...)
    difficulty        "Level 5"
    dataset           "math": the baselines' prompt and grading (math-verify; baselines/tasks.py)

Split, per subject: the rows in id order, shuffled with --seed; the first half goes to train. When
a subject has an odd number of rows, its extra row goes to train and test in turn (subjects in
alphabetical order), so the two files differ in size by at most one.

    python scripts/prepare_math_l5.py     # -> datasets/math_l5_train.json, math_l5_test.json, math_l5_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "baselines"))
from tasks import last_boxed  # noqa: E402

REPO, REVISION = "EleutherAI/hendrycks_math", "21a5633873b6a120296cce3e2df9d5550074f4a3"
SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus")
LEVEL = "Level 5"


def load(split: str) -> list[dict]:
    import pandas as pd
    from huggingface_hub import hf_hub_download
    rows = []
    for subject in SUBJECTS:
        path = hf_hub_download(REPO, f"{subject}/{split}-00000-of-00001.parquet", repo_type="dataset",
                               revision=REVISION)
        df = pd.read_parquet(path)
        for _, r in df[df["level"] == LEVEL].iterrows():
            answer = last_boxed(r["solution"])
            if not answer:
                raise SystemExit(f"no \\boxed answer in the solution of: {r['problem'][:120]!r}")
            rows.append({"id": "math_" + hashlib.sha256(r["problem"].encode()).hexdigest()[:16],
                         "question": r["problem"], "options": [], "answer": answer,
                         "discipline": r["type"], "field": r["type"], "subfield": r["type"],
                         "difficulty": LEVEL, "dataset": "math"})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-splits", nargs="+", default=["test"], choices=["test", "train"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="math_l5", help="output files: datasets/<name>_{train,test}.json")
    ap.add_argument("--force", action="store_true", help="overwrite existing output files")
    args = ap.parse_args()
    outs = {part: ROOT / f"datasets/{args.name}_{part}.json" for part in ("train", "test")}
    manifest_path = ROOT / f"datasets/{args.name}_manifest.json"
    if any(p.exists() for p in outs.values()) and not args.force:
        raise SystemExit(f"{outs['train']} or {outs['test']} exists (pass --force to rewrite them)")

    rows = [r for split in args.source_splits for r in load(split)]
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit("two problems have the same text")
    by_subject: dict[str, list[dict]] = {}
    for r in sorted(rows, key=lambda r: r["id"]):
        by_subject.setdefault(r["discipline"], []).append(r)

    parts: dict[str, list[dict]] = {"train": [], "test": []}
    odd = 0
    for subject in sorted(by_subject):
        group = by_subject[subject]
        random.Random(f"{args.seed}:{subject}").shuffle(group)
        half = len(group) // 2
        if len(group) % 2:                           # the extra row: train, then test, then train, ...
            half += odd % 2 == 0
            odd += 1
        parts["train"] += group[:half]
        parts["test"] += group[half:]

    for part, path in outs.items():
        path.write_text(json.dumps(parts[part], indent=1, ensure_ascii=False))
    manifest = {"source": {"repo": REPO, "revision": REVISION, "splits": args.source_splits, "level": LEVEL},
                "seed": args.seed,
                "questions": {part: len(rs) for part, rs in parts.items()},
                "by_subject": {s: {part: sum(r["discipline"] == s for r in parts[part]) for part in parts}
                               for s in sorted(by_subject)}}
    manifest_path.write_text(json.dumps(manifest, indent=1))
    for part, path in outs.items():
        print(f"{path.relative_to(ROOT)}: {len(parts[part])} problems; "
              f"{dict(sorted(Counter(r['discipline'] for r in parts[part]).items()))}")


if __name__ == "__main__":
    main()

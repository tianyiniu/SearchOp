"""Write AIME 2022-2025 in the layout the pipeline and the external baselines read, split 50/50 into
train and test with the same contests in both.

Source: allenai/aime-2022-2025 on Hugging Face (pinned revision): the 120 problems of AIME I and II,
2022 to 2025, 15 per contest. Each row becomes:

    id                "aime_<year>_<I|II>_<problem number, 2 digits>" (from the problem's AoPS url)
    question          the problem
    options           [] (open answer)
    answer            the integer answer without leading zeros ("033" -> "33"), as a model writes it;
                      the solution is dropped, so no file carries one
    discipline        "AIME"
    field, subfield   the contest ("2024 AIME II") and the problem ("Problem 7")
    difficulty        "AIME"
    dataset           "math": the baselines' prompt and grading (an answer in \\boxed{}, compared by
                      math-verify; baselines/tasks.py), as MATH level 5

Split, per contest: the rows in id order, shuffled with --seed; the first half goes to train. Each
contest has 15 problems, so its extra problem goes to train and test in turn (contests in order),
and both files hold 60 problems, 15 from each year.

    python scripts/prepare_aime.py     # -> datasets/aime_2022_2025_train.json, _test.json, _manifest.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO, REVISION = "allenai/aime-2022-2025", "73e1eba765ad5847cdb5d1e2e7aaf7b22b585798"
URL = re.compile(r"/(\d{4})_AIME_(I+)_Problems/Problem_(\d+)$")


def load() -> list[dict]:
    import pandas as pd
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO, "data/train-00000-of-00001.parquet", repo_type="dataset", revision=REVISION)
    rows = []
    for _, r in pd.read_parquet(path).iterrows():
        m = URL.search(r["url"])
        if not m:
            raise SystemExit(f"cannot read the contest from the url {r['url']!r}")
        year, contest, number = int(m.group(1)), m.group(2), int(m.group(3))
        if year != int(r["year"]):
            raise SystemExit(f"the url's year {year} differs from the row's {r['year']}: {r['url']}")
        answer = str(r["answer"]).strip()
        if not answer.isdigit() or not 0 <= int(answer) <= 999:
            raise SystemExit(f"not an AIME answer (an integer 0-999): {answer!r} ({r['url']})")
        rows.append({"id": f"aime_{year}_{contest}_{number:02d}", "question": r["problem"], "options": [],
                     "answer": str(int(answer)), "discipline": "AIME", "field": f"{year} AIME {contest}",
                     "subfield": f"Problem {number}", "difficulty": "AIME", "dataset": "math"})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="aime_2022_2025", help="output files: datasets/<name>_{train,test}.json")
    ap.add_argument("--force", action="store_true", help="overwrite existing output files")
    args = ap.parse_args()
    outs = {part: ROOT / f"datasets/{args.name}_{part}.json" for part in ("train", "test")}
    manifest_path = ROOT / f"datasets/{args.name}_manifest.json"
    if any(p.exists() for p in outs.values()) and not args.force:
        raise SystemExit(f"{outs['train']} or {outs['test']} exists (pass --force to rewrite them)")

    rows = load()
    if len({r["id"] for r in rows}) != len(rows) or len({r["question"] for r in rows}) != len(rows):
        raise SystemExit("two problems have the same id or the same text")
    by_contest: dict[str, list[dict]] = {}
    for r in sorted(rows, key=lambda r: r["id"]):
        by_contest.setdefault(r["field"], []).append(r)
    bad = {c: len(g) for c, g in by_contest.items() if len(g) != 15}
    if bad:
        raise SystemExit(f"contests without 15 problems: {bad}")

    parts: dict[str, list[dict]] = {"train": [], "test": []}
    odd = 0
    for contest in sorted(by_contest):
        group = by_contest[contest]
        random.Random(f"{args.seed}:{contest}").shuffle(group)
        half = len(group) // 2
        if len(group) % 2:                           # the extra row: train, then test, then train, ...
            half += odd % 2 == 0
            odd += 1
        parts["train"] += group[:half]
        parts["test"] += group[half:]

    for part, path in outs.items():
        path.write_text(json.dumps(parts[part], indent=1, ensure_ascii=False))
    manifest = {"source": {"repo": REPO, "revision": REVISION}, "seed": args.seed,
                "questions": {part: len(rs) for part, rs in parts.items()},
                "by_contest": {c: {part: sum(r["field"] == c for r in parts[part]) for part in parts}
                               for c in sorted(by_contest)}}
    manifest_path.write_text(json.dumps(manifest, indent=1))
    for part, path in outs.items():
        years = Counter(r["field"][:4] for r in parts[part])
        print(f"{path.relative_to(ROOT)}: {len(parts[part])} problems; per year {dict(sorted(years.items()))}")


if __name__ == "__main__":
    main()

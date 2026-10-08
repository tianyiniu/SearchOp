"""Convert the HLE text splits to the layout the labelling pipeline reads.

describe_hle_v4.py, embed_questions.py and cluster_questions.py read a JSON
list of rows with SuperGPQA's fields. The HLE files are JSONL with other
fields, so each row is rewritten:

    id, question      as they are. A multiple-choice question keeps its
                      "Answer Choices:" block inside the question text, which
                      is how the describer reads it.
    options           [] for every question: HLE has no separate option list.
    discipline        HLE's category (8 values)
    field, subfield   HLE's raw_subject (105 values in train)
    difficulty        "" (HLE has none)
    answer, answer_type   kept for later grading; the describer never reads them.

The rationale (a worked solution), author_name and canary are dropped, so no
file the pipeline reads carries a solution. The original .jsonl files are only
read.

    python scripts/prepare_hle.py        # both splits, datasets/hle_text_*.jsonl -> .json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("hle_text_train_800", "hle_text_test_200")


def convert(row: dict) -> dict:
    return {"id": row["id"], "question": row["question"], "options": [],
            "discipline": row["category"], "field": row["raw_subject"],
            "subfield": row["raw_subject"], "difficulty": "",
            "answer": row["answer"], "answer_type": row["answer_type"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--force", action="store_true", help="overwrite existing .json files")
    args = ap.parse_args()
    seen: dict[str, str] = {}
    for name in args.splits:
        src, dst = ROOT / f"datasets/{name}.jsonl", ROOT / f"datasets/{name}.json"
        if dst.exists() and not args.force:
            raise SystemExit(f"{dst} exists (pass --force to rewrite it)")
        rows = [convert(json.loads(line)) for line in src.read_text().splitlines() if line.strip()]
        for r in rows:
            if r["id"] in seen:
                raise SystemExit(f"question {r['id']} is in both {seen[r['id']]} and {name}")
            seen[r["id"]] = name
        dst.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
        mc = sum(r["answer_type"] == "multipleChoice" for r in rows)
        print(f"{src.name} -> {dst.name}: {len(rows)} questions ({mc} multiple choice, "
              f"{len(rows) - mc} open answer)")


if __name__ == "__main__":
    main()

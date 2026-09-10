"""Download SealQA's seal_hard test split to JSONL.

SealQA (https://huggingface.co/datasets/vtllms/sealqa) ships three configs —
seal_0 / seal_hard / longseal. This grabs seal_hard.

Writes one JSON object per line: {"id", "question", "ground_truth", "source"}.
    python download_sealqa.py            # -> datasets/sealqa_test_full.jsonl
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("datasets/sealqa_test_full.jsonl"))
    ap.add_argument("--config", default="seal_hard", help="seal_0 | seal_hard | longseal")
    args = ap.parse_args()

    ds = load_dataset("vtllms/sealqa", args.config, split="test")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            if not question or not answer:
                continue
            f.write(json.dumps({
                "id": f"sealqa_test_{i}",
                "question": question,
                "ground_truth": answer,
                "source": "sealqa",
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} questions to {args.out}")


if __name__ == "__main__":
    main()

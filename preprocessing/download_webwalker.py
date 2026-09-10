"""Download Google's FRAMES benchmark to JSONL.

WebWalkerQA (https://huggingface.co/datasets/callanwu/WebWalkerQA)

Writes one JSON object per line: {"id", "question", "ground_truth", "source"}.

    pip install datasets
    python download_frames.py            # -> datasets/webwalker_test_full.jsonl
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("datasets/webwalker_test_full.json"))
    args = ap.parse_args()

    ds = load_dataset("google/frames-benchmark", split="test")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            question = (row.get("Prompt") or "").strip()
            answer = (row.get("Answer") or "").strip()
            if not question or not answer:
                continue
            f.write(json.dumps({
                "id": f"frames_test_{i}",
                "question": question,
                "ground_truth": answer,
                "source": "frames",
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} questions to {args.out}")


if __name__ == "__main__":
    main()

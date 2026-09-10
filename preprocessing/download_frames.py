"""Download Google's FRAMES benchmark to JSON.

FRAMES (https://huggingface.co/datasets/google/frames-benchmark).

Each json object: {"id", "question", "answer", "reasoning_types", "wiki_links"}.

Command:
    python download_frames.py            # -> datasets/frames_test_full.jsonl
"""

import argparse
import json
from pathlib import Path
import sys
import ast

from datasets import load_dataset


def main(args):
    ds = load_dataset("google/frames-benchmark", split="test")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    out_objects = []
    for i, row in enumerate(ds):
        question = (row.get("Prompt") or "").strip()
        answer = (row.get("Answer") or "").strip()
        if not question or not answer:
            continue
        out_objects.append({
            "id": f"frames_{i}",
            "question": row["Prompt"].strip(),
            "ground_truth": row["Answer"].strip(),
            "reasoning_types": row["reasoning_types"].strip(),
            "wiki_links": ast.literal_eval(row["wiki_links"])
        })

    with open(args.out, "w") as f: 
	    json.dump(out_objects, f, indent=4, ensure_ascii=False)

    print(f"Wrote {len(out_objects)} questions to {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("datasets/frames_test_full.json"))
    args = ap.parse_args()
    main(args)

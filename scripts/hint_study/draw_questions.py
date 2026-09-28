"""Step 1: draw the study questions. 100 for train and 100 for test, with the
three kinds (remembering = recall, working-out = derive, mixed = both) equally
represented in each, drawn only from questions that already carry a knowledge
label from the describer. No new labelling calls.

The labelled pool is the described train split (outputs/question_templates_train.jsonl)
plus the described dev split (outputs/question_templates_dev_small.jsonl). The two
draws are disjoint. Each row keeps its SuperGPQA fields, its knowledge label and
where it came from (train or dev pool).

    python scripts/hint_study/draw_questions.py            # writes datasets/hint_study_{train,test}.json
    python scripts/hint_study/draw_questions.py --n 100 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATASETS, DEFAULT_OUT, KINDS, ROOT, write_json  # noqa: E402

POOLS = [  # (dataset rows, describer records, source tag)
    (DATASETS / "supergpqa_program_search_train.json", ROOT / "outputs/question_templates_train.jsonl", "train"),
    (DATASETS / "supergpqa_program_search_dev_small.json", ROOT / "outputs/question_templates_dev_small.jsonl", "dev"),
]


def labelled_pool() -> list[dict]:
    rows = []
    for data_path, label_path, source in POOLS:
        data = {r["id"]: r for r in json.loads(data_path.read_text())}
        with label_path.open() as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("error") or rec.get("knowledge") not in KINDS or rec["id"] not in data:
                    continue
                row = dict(data[rec["id"]])
                row["knowledge"] = rec["knowledge"]
                row["template"] = rec["template"]
                row["source"] = source
                rows.append(row)
    seen = set()
    return [r for r in rows if not (r["id"] in seen or seen.add(r["id"]))]


def per_kind_counts(n: int) -> dict[str, int]:
    """n split three ways as evenly as possible; the remainder goes to recall,
    then derive."""
    base, extra = divmod(n, 3)
    return {k: base + (1 if i < extra else 0) for i, k in enumerate(("recall", "derive", "both"))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100, help="questions per split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="where the manifest goes")
    args = ap.parse_args()

    pool = labelled_pool()
    rng = random.Random(args.seed)
    want = per_kind_counts(args.n)
    by_kind = {k: [r for r in pool if r["knowledge"] == k] for k in KINDS}
    print("labelled pool:", {k: len(v) for k, v in by_kind.items()})
    for k, need in want.items():
        if len(by_kind[k]) < 2 * need:
            raise SystemExit(f"{k}: need {2 * need} labelled questions for both splits, have {len(by_kind[k])}")
    train, test = [], []
    for k in KINDS:
        rows = sorted(by_kind[k], key=lambda r: r["id"])   # order independent of file order
        rng.shuffle(rows)
        test += rows[: want[k]]
        train += rows[want[k]: 2 * want[k]]
    rng.shuffle(train)
    rng.shuffle(test)

    def profile(rows):
        return {"n": len(rows), "kind": dict(Counter(r["knowledge"] for r in rows)),
                "source": dict(Counter(r["source"] for r in rows)),
                "difficulty_tag": dict(Counter(r["difficulty"] for r in rows)),
                "discipline": dict(Counter(r["discipline"] for r in rows).most_common())}

    write_json(DATASETS / "hint_study_train.json", train)
    write_json(DATASETS / "hint_study_test.json", test)
    manifest = {"seed": args.seed, "n_per_split": args.n, "per_kind": want,
                "pool": {k: len(v) for k, v in by_kind.items()},
                "train": profile(train), "test": profile(test)}
    write_json(args.out_dir / "draw_manifest.json", manifest)
    print(json.dumps(manifest, indent=1))
    print(f"wrote {DATASETS / 'hint_study_train.json'} and {DATASETS / 'hint_study_test.json'}")


if __name__ == "__main__":
    main()

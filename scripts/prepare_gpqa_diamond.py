"""Write GPQA-Diamond in the layout the pipeline and the external baselines read. Test set only:
198 questions are too few to split.

Source: fingertap/GPQA-Diamond on Hugging Face (pinned revision), 198 rows of {question, answer}.
Each question text ends with its four options as 'A. ...' to 'D. ...' lines, in an order fixed
by that copy; the answer is the right letter. Each row becomes:

    id                GPQA's Record ID
    question          the question text without the options block
    options           the four options, in fingertap's order (surrounding whitespace removed)
    answer_letter     fingertap's answer
    discipline        GPQA's high-level domain (Physics, Chemistry, Biology)
    field, subfield   GPQA's subdomain
    difficulty        "" (GPQA has none)
    dataset           "gpqa": the baselines' prompt and answer rule (baselines/tasks.py)

The record ID, domain and subdomain come from OpenAI's public copy of the same 198 questions
(simple-evals gpqa_diamond.csv), matched by question text (whitespace aside). Each match is
checked: exactly one row, with the same right answer. Nothing else is copied from it, so no
file carries a solution. Its SHA-256 goes into the manifest.

    python scripts/prepare_gpqa_diamond.py     # -> datasets/gpqa_diamond_test.json + _manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO, REVISION = "fingertap/GPQA-Diamond", "68be7564497676e07a77a042fdb587deb88c51c3"
PARQUET = "test/gpqa_diamond.parquet"
OPENAI_CSV = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
OUT = ROOT / "datasets/gpqa_diamond_test.json"
MANIFEST = ROOT / "datasets/gpqa_diamond_manifest.json"
N_QUESTIONS = 198

# the options block that ends every question: each label appears exactly once (checked below)
OPTIONS = re.compile(r"\n\nA\. (.*)\nB\. (.*)\nC\. (.*)\nD\. (.*)\Z", re.S)
LABELS = ("\n\nA. ", "\nB. ", "\nC. ", "\nD. ")


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def split_question(text: str) -> tuple[str, list[str]]:
    """(question without the options, the four options)."""
    counts = [text.count(label) for label in LABELS]
    if counts != [1, 1, 1, 1]:
        raise SystemExit(f"option labels not found exactly once ({counts}) in: {text[:120]!r}")
    m = OPTIONS.search(text)
    if m is None:
        raise SystemExit(f"no options block at the end of: {text[:120]!r}")
    return text[:m.start()].rstrip(), [o.strip() for o in m.groups()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="overwrite an existing output file")
    args = ap.parse_args()
    if OUT.exists() and not args.force:
        raise SystemExit(f"{OUT} exists (pass --force to rewrite it)")

    import pandas as pd
    from huggingface_hub import hf_hub_download

    source = pd.read_parquet(hf_hub_download(REPO, PARQUET, repo_type="dataset", revision=REVISION))
    raw = urllib.request.urlopen(OPENAI_CSV, timeout=60).read()
    meta = pd.read_csv(io.BytesIO(raw))
    if len(source) != N_QUESTIONS or len(meta) != N_QUESTIONS:
        raise SystemExit(f"expected {N_QUESTIONS} questions, got {len(source)} (fingertap) and {len(meta)} (OpenAI)")
    by_text: dict[str, list[dict]] = {}
    for _, r in meta.iterrows():
        by_text.setdefault(squash(r["Question"]), []).append(r.to_dict())

    rows = []
    for _, r in source.iterrows():
        question, options = split_question(r["question"])
        if r["answer"] not in "ABCD" or len(r["answer"]) != 1:
            raise SystemExit(f"answer {r['answer']!r} is not a letter A-D: {question[:120]!r}")
        found = by_text.get(squash(question), [])
        if len(found) != 1:
            raise SystemExit(f"{len(found)} rows of OpenAI's copy match: {question[:120]!r}")
        m = found[0]
        if squash(options["ABCD".index(r["answer"])]) != squash(m["Correct Answer"]):
            raise SystemExit(f"the right answers differ between the two copies: {question[:120]!r}")
        rows.append({"id": m["Record ID"], "question": question, "options": options,
                     "answer_letter": r["answer"], "discipline": m["High-level domain"],
                     "field": m["Subdomain"], "subfield": m["Subdomain"], "difficulty": "",
                     "dataset": "gpqa"})
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit("record IDs are not unique")

    OUT.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    manifest = {"source": {"repo": REPO, "revision": REVISION, "file": PARQUET},
                "metadata": {"url": OPENAI_CSV, "sha256": hashlib.sha256(raw).hexdigest(),
                             "columns_used": ["Record ID", "High-level domain", "Subdomain"]},
                "questions": len(rows),
                "by_discipline": dict(Counter(r["discipline"] for r in rows).most_common()),
                "by_answer_letter": dict(sorted(Counter(r["answer_letter"] for r in rows).items()))}
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    print(f"{OUT.relative_to(ROOT)}: {len(rows)} questions; {manifest['by_discipline']}; "
          f"answers {manifest['by_answer_letter']}")


if __name__ == "__main__":
    main()

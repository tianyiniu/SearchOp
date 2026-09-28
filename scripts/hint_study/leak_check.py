"""Step 3b: does the hint sheet give the answer away? The small model is shown
the sheet and the answer choices but not the question, K tries. If it still
picks the right letter LEAK_MIN times or more, the sheet leaks. A leaking
sheet is rewritten once by `hint_sheets.py --strict`; if that leaks too, the
question is dropped and counted.

    python scripts/hint_study/leak_check.py --split both
    python scripts/hint_study/leak_check.py --split train --k 1 --limit 1     # smoke test

Writes <out-dir>/leak.json keyed by "<question id>|<sheet version>".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def leak_prompt(row: dict, sheet: dict) -> str:
    return ("Below are notes written about a multiple-choice question that you have NOT been shown. "
            "Using only the notes, pick the choice that is most likely the correct answer to that question. "
            + C.answer_line(row) + "\n\nNotes:\n" + C.items_block(C.sheet_items(sheet))
            + "\n\nChoices:\n" + C.options_block(row) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="both", choices=["train", "test", "both"])
    ap.add_argument("--k", type=int, default=C.K)
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high"])
    ap.add_argument("--limit", type=int, default=0)
    C.add_student_args(ap)
    C.add_dir_args(ap)
    args = ap.parse_args()

    sheets = C.read_sheets(C.teacher_dir(args))
    leaks = C.read_json(args.out_dir / "leak.json", {})
    todo = []          # (row, sheet record) for every version not yet checked
    for split in C.splits(args.split):
        for r in C.load_split(split):
            for rec in sheets.get(r["id"], []):
                if f"{r['id']}|{rec['version']}" not in leaks:
                    todo.append((r, rec))
    todo = todo[: args.limit or None]
    print(f"{len(todo)} sheet versions to check")
    if not todo:
        return

    student = C.make_student(args)
    try:
        jobs = [C.Student.job(f"{r['id']}|leak|{rec['version']}|{args.effort}|{s}",
                              [{"role": "user", "content": leak_prompt(r, rec["sheet"])}], args.effort)
                for r, rec in todo for s in range(args.k)]
        recs = student.run(jobs, "leak check")
    finally:
        student.close()

    n_leak = 0
    for r, rec in todo:
        sc = C.score([recs[f"{r['id']}|leak|{rec['version']}|{args.effort}|{s}"] for s in range(args.k)], r)
        sc.update(id=r["id"], version=rec["version"], leak=sc["c"] >= C.LEAK_MIN, kind=r["knowledge"])
        leaks[f"{r['id']}|{rec['version']}"] = sc
        n_leak += sc["leak"]
    C.write_json(args.out_dir / "leak.json", leaks)
    print(f"{n_leak} of {len(todo)} sheets give the answer away (right {C.LEAK_MIN}+ of {args.k} without the question)")
    if n_leak:
        print("rewrite them with: python scripts/hint_study/hint_sheets.py --strict, then run this check again")


if __name__ == "__main__":
    main()

"""Step 2: can the small model do the question alone? K tries of the official
zero-shot prompt, full thinking on (reasoning effort high). A question "fails
alone" if it is right at most FAIL_MAX times out of K.

Replies that run out of room before giving a letter count as wrong, but are
tallied separately so "did not know" and "ran out of space" can be told apart.

    python scripts/hint_study/unaided.py --split both
    python scripts/hint_study/unaided.py --split train --limit 3 --k 1     # smoke test

Writes <out-dir>/unaided_<split>.json; the replies live in <out-dir>/student_cache.jsonl.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def summarize(per_q: dict[str, dict], rows: list[dict], k: int) -> dict:
    out = {}
    for kind, rs in [("all", rows)] + list(C.by_kind(rows).items()):
        qs = [per_q[r["id"]] for r in rs if r["id"] in per_q]
        if not qs:
            continue
        out[kind] = {"questions": len(qs),
                     f"avg@{k}": C.mean(q["c"] / q["n"] for q in qs),
                     f"pass@{k}": C.mean(q["c"] > 0 for q in qs),
                     "fails_alone": sum(q["fails_alone"] for q in qs),
                     "truncated_share": C.mean(q["truncated"] / q["n"] for q in qs),
                     "tokens_per_reply": C.mean(q["tokens"] / q["n"] for q in qs)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="both", choices=["train", "test", "both"])
    ap.add_argument("--k", type=int, default=C.K)
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high"],
                    help="gpt-oss reasoning effort; for a Qwen model 'low' is thinking off, the rest thinking on")
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (smoke test)")
    C.add_student_args(ap)
    C.add_dir_args(ap)
    args = ap.parse_args()

    student = C.make_student(args)
    try:
        for split in C.splits(args.split):
            rows = C.load_split(split)[: args.limit or None]
            jobs = [C.Student.job(f"{r['id']}|unaided|{args.effort}|{s}",
                                  [{"role": "user", "content": C.direct_prompt(r)}], args.effort)
                    for r in rows for s in range(args.k)]
            recs = student.run(jobs, f"unaided {split}")
            per_q = {}
            for r in rows:
                sc = C.score([recs[f"{r['id']}|unaided|{args.effort}|{s}"] for s in range(args.k)], r)
                sc.update(kind=r["knowledge"], answer=r["answer_letter"], fails_alone=sc["c"] <= C.FAIL_MAX)
                per_q[r["id"]] = sc
            summary = summarize(per_q, rows, args.k)
            C.write_json(args.out_dir / f"unaided_{split}.json",
                         {"split": split, "k": args.k, "effort": args.effort, "model": args.model, "fail_max": C.FAIL_MAX,
                          "summary": summary, "per_question": per_q})
            print(f"== {split}")
            for kind, s in summary.items():
                print(f"  {kind:6s} n={s['questions']:3d}  avg@{args.k} {100 * s[f'avg@{args.k}']:5.1f}  "
                      f"pass@{args.k} {100 * s[f'pass@{args.k}']:5.1f}  fails alone {s['fails_alone']:3d}  "
                      f"truncated {100 * s['truncated_share']:4.1f}%  tokens/reply {s['tokens_per_reply']:.0f}")
    finally:
        student.close()


if __name__ == "__main__":
    main()

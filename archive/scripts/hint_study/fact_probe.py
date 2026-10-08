"""Step 7: which of the facts on a sheet does the small model actually have?

For every fact on the sheet of a remembering or mixed middle question:
  - stating check: the fact's own short question, K tries, full thinking; the
    big model judges whether each reply states the fact
  - recognising check: the true fact next to its wrong version, "which one is
    right", both orders, two tries each

A fact is "can state it" if judged right at least REACH_MIN of K times;
otherwise "can only recognise it" if right at least 3 of the 4 recognising
tries; otherwise "does not have it". A question's bucket is the worst class
among its facts; questions with a fact the model does not have go to the
"needs a lookup" pile.

    python scripts/hint_study/fact_probe.py --split both
    python scripts/hint_study/fact_probe.py --split train --k 1 --limit 1     # smoke test

Writes <out-dir>/facts_<split>.json.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

CLASSES = ("can_state", "recognises_only", "does_not_have")


class Judgement(BaseModel):
    states_fact: bool = Field(description="true if the reply states the fact, or something equivalent to it, "
                                          "as its answer; false if it gives a different answer, hedges between "
                                          "several, or does not address it")
    note: str = Field(description="one short line saying why")


JUDGE_INSTRUCTIONS = ("You are grading whether a reply states a given fact. You are shown the fact, the question "
                      "that was asked, and the reply. Say whether the reply states the fact, or something "
                      "equivalent, as its answer. A reply that lists several possibilities without settling on the "
                      "right one does not count. Wording may differ; the content must match.")


def state_prompt(fact: dict) -> str:
    return ("From your own knowledge, answer the following question in a few sentences. Give your best single "
            "answer; do not list alternatives.\n\n" + fact["cue_question"] + "\n")


def recog_prompt(a: str, b: str) -> str:
    return ("Exactly one of the two statements below is correct. Say which one. The last line of your response "
            "should be 'Answer: 1' or 'Answer: 2'.\n\n1. " + a + "\n2. " + b + "\n")


def recog_pick(content: str | None) -> str | None:
    if not content:
        return None
    m = re.findall(r"Answer:\s*\**\s*([12])", content)
    return m[-1] if m else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="both", choices=["train", "test", "both"])
    ap.add_argument("--k", type=int, default=C.K)
    ap.add_argument("--all-with-sheet", action="store_true",
                    help="every remembering/mixed question with a sheet, not only the middle ones")
    ap.add_argument("--judge-effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--limit", type=int, default=0)
    C.add_student_args(ap)
    C.add_dir_args(ap)
    args = ap.parse_args()
    state_min = C.REACH_MIN if args.k == C.K else 1

    sheets = C.active_sheets(C.teacher_dir(args), args.out_dir)
    student = C.make_student(args)
    # the judgements are about this small model's replies, so they live with its files
    teacher = C.Teacher(args.out_dir / "judge_cache.jsonl", workers=6)
    try:
        for split in C.splits(args.split):
            mid = C.read_json(args.out_dir / f"middle_{split}.json", {"per_question": {}})
            rows = [r for r in C.load_split(split) if r["knowledge"] in ("recall", "both") and r["id"] in sheets
                    and (args.all_with_sheet or mid["per_question"].get(r["id"], {}).get("keep"))]
            rows = [r for r in rows if sheets[r["id"]]["sheet"]["facts"]][: args.limit or None]
            print(f"== {split}: {len(rows)} questions, "
                  f"{sum(len(sheets[r['id']]['sheet']['facts']) for r in rows)} facts")
            if not rows:
                continue

            def fkey(r, i, what, s):
                return f"{r['id']}|{what}|{sheets[r['id']]['version']}|{i}|{s}"

            jobs = []
            for r in rows:
                for i, fact in enumerate(sheets[r["id"]]["sheet"]["facts"]):
                    for s in range(args.k):
                        jobs.append(C.Student.job(fkey(r, i, "factstate", s),
                                                  [{"role": "user", "content": state_prompt(fact)}], "high", 8192))
                    for order in (0, 1):
                        a, b = (fact["text"], fact["false_version"]) if order == 0 else (fact["false_version"], fact["text"])
                        for s in range(2):
                            jobs.append(C.Student.job(fkey(r, i, f"factrec{order}", s),
                                                      [{"role": "user", "content": recog_prompt(a, b)}], "high", 4096))
            recs = student.run(jobs, f"fact checks {split}")

            jjobs = []
            for r in rows:
                for i, fact in enumerate(sheets[r["id"]]["sheet"]["facts"]):
                    for s in range(args.k):
                        reply = recs[fkey(r, i, "factstate", s)].get("content") or "(no reply)"
                        jjobs.append({"key": fkey(r, i, "factjudge", s), "instructions": JUDGE_INSTRUCTIONS,
                                      "text": f"Fact: {fact['text']}\n\nQuestion asked: {fact['cue_question']}\n\n"
                                              f"Reply:\n{reply[:6000]}",
                                      "schema": Judgement, "effort": args.judge_effort, "max_output_tokens": 2000})
            jrecs = teacher.run(jjobs, f"judging {split}")

            per_q = {}
            for r in rows:
                facts_out = []
                for i, fact in enumerate(sheets[r["id"]]["sheet"]["facts"]):
                    stated = sum(bool(jrecs[fkey(r, i, "factjudge", s)].get("out", {}).get("states_fact"))
                                 for s in range(args.k))
                    rec_right = 0
                    for order in (0, 1):
                        want = "1" if order == 0 else "2"
                        for s in range(2):
                            rec_right += recog_pick(recs[fkey(r, i, f"factrec{order}", s)].get("content")) == want
                    cls = ("can_state" if stated >= state_min else
                           "recognises_only" if rec_right >= 3 else "does_not_have")
                    facts_out.append({"text": fact["text"], "stated": stated, "of": args.k,
                                      "recognised": rec_right, "of4": 4, "class": cls})
                worst = max((CLASSES.index(f["class"]) for f in facts_out), default=0)
                per_q[r["id"]] = {"kind": r["knowledge"], "version": sheets[r["id"]]["version"],
                                  "facts": facts_out, "bucket": CLASSES[worst],
                                  "needs_lookup": CLASSES[worst] == "does_not_have"}
            summary = {}
            for kind, rs in [("all", rows)] + list(C.by_kind(rows).items()):
                qs = [per_q[r["id"]] for r in rs]
                if not qs:
                    continue
                facts = [f for q in qs for f in q["facts"]]
                summary[kind] = {"questions": len(qs), "facts": len(facts),
                                 "fact_classes": {c: sum(f["class"] == c for f in facts) for c in CLASSES},
                                 "question_buckets": {c: sum(q["bucket"] == c for q in qs) for c in CLASSES}}
            C.write_json(args.out_dir / f"facts_{split}.json",
                         {"split": split, "k": args.k, "model": args.model, "summary": summary, "per_question": per_q})
            for kind, s in summary.items():
                print(f"  {kind:6s} q={s['questions']:3d} facts={s['facts']:3d}  facts {s['fact_classes']}  "
                      f"questions {s['question_buckets']}")
    finally:
        student.close()
        teacher.close()


if __name__ == "__main__":
    main()

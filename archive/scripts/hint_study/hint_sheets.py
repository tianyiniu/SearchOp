"""Step 3a: the big model writes a hint sheet for every question the small
model failed alone.

One call per question returns: the teacher's own answer letter (if it does not
match the key the question is set aside as possibly mislabelled), the kind of
question (recall / derive / both), the facts (each with a short question that
asks for that fact on its own and a wrong version of it, for the fact checks of
step 7), the steps (each tagged with what kind of step it is) and, kept apart,
the final step that produces the answer. Facts and steps may never name or
quote an answer choice; a text check flags any that do, and step 3b tests it.

    python scripts/hint_study/hint_sheets.py --split both              # questions that failed alone
    python scripts/hint_study/hint_sheets.py --split train --all       # every question
    python scripts/hint_study/hint_sheets.py --split both --strict     # rewrite the sheets that leaked (after leak_check)
    python scripts/hint_study/hint_sheets.py --split train --limit 1   # smoke test, one call

Appends to <out-dir>/sheets.jsonl (one record per question and version); a rerun
skips versions already written.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

PROMPT_VERSION = "v1"

STEP_KINDS = ("set_up", "choose_law", "recall_constant", "algebra", "arithmetic",
              "case_split", "sanity_check", "match_option")
STEP_KIND_TEXT = """\
- set_up: name the quantities, unknowns, or objects the question is about and how they relate
- choose_law: pick the principle, formula, definition, or method that applies
- recall_constant: bring in a needed value, constant, or standard fact
- algebra: rearrange or combine expressions symbolically
- arithmetic: evaluate numbers
- case_split: split into cases or check each candidate in turn
- sanity_check: check units, limits, signs, or plausibility
- match_option: turn the result into the form the choices use"""


class Fact(BaseModel):
    text: str = Field(description="one plain sentence stating the fact so that it stands on its own; never names, "
                                  "quotes, or paraphrases an answer choice")
    cue_question: str = Field(description="a short question that asks for exactly this fact, with no reference to "
                                          "the original question or its choices")
    false_version: str = Field(description="the same sentence with one detail changed so that it is wrong, as "
                                           "plausible in wording as the true one")


class Step(BaseModel):
    text: str = Field(description="one checkable claim or one computation, in one or two sentences; never names, "
                                  "quotes, or paraphrases an answer choice, and never states the final result")
    kind: Literal[STEP_KINDS]


class HintSheet(BaseModel):
    answer: Literal[tuple(C.LETTERS)]
    kind: Literal[C.KINDS] = Field(description="recall: the answer is known or not, nothing to work out; "
                                               "derive: the answer must be worked out from things the question "
                                               "gives; both: some facts must be recalled and then something "
                                               "worked out")
    facts: list[Fact] = Field(description="the facts a solver must know; 2 to 6 for recall or both, empty for derive")
    steps: list[Step] = Field(description="the working-out steps in order, 3 to 8 for derive or both, empty for "
                                          "recall; the final step that produces the answer is NOT in this list")
    final_step: str = Field(description="the last step, the one that produces the answer or names the right "
                                        "choice; it is stored and never shown to the solver")


INSTRUCTIONS = f"""\
You are writing a hint sheet for a hard multiple-choice question. A weaker model will later be shown \
the question together with parts of your sheet, and we will measure how much of the sheet it needs. \
So the sheet must be correct, complete, and must never give the answer away by itself.

Do the following, in this order.
1. Solve the question yourself and give the letter of the correct choice.
2. Say what kind of question it is: recall, derive, or both.
3. Write the facts a solver has to know (for recall or both). Each fact is one plain sentence that \
stands on its own. For each fact also write a short question that asks for that fact without any of \
the original question's context, and a wrong version of the fact that differs in one detail.
4. Write the working-out steps (for derive or both), in order, each one thing a reader could check. \
Tag each step with its kind:
{STEP_KIND_TEXT}
5. Write the final step, the one that produces the answer or names the right choice, separately. It \
will never be shown.

Rules for facts and steps: never write a choice letter; never quote, name, or paraphrase the text of \
any answer choice; never state the final numeric result or the final conclusion. A fact may state a \
value that the solver must know (a constant, a definition, a rule) but not the value that is the \
answer. Keep every fact and step short. Use the language of the question."""

STRICT_ADDENDUM = """

The previous sheet for this question let a solver pick the right choice from the sheet alone, without \
seeing the question. Rewrite it so that this cannot happen: remove any number, name, or phrase that \
appears in a choice; make the last shown step stop one clear step earlier; prefer general rules to \
specific values wherever the specific value points at a choice."""


def build_input(row: dict) -> str:
    return (f"Question (the correct choice is one of {C.letters_of(row)}):\n\n" + C.question_block(row))


def overlap(sheet: dict, row: dict) -> list[str]:
    """Texts in the shown part of the sheet that contain an answer choice or a
    choice letter. A flag, not a verdict: the leak check is the real test."""
    shown = [f["text"] for f in sheet["facts"]] + [s["text"] for s in sheet["steps"]]
    hits = []
    for i, opt in enumerate(row["options"]):
        o = re.sub(r"\s+", " ", str(opt)).strip().lower()
        if len(o) < 3:
            continue
        for t in shown:
            tl = t.lower()
            if o in tl or re.search(rf"\b(option|choice)\s+{C.LETTERS[i]}\b", t):
                hits.append(f"{C.LETTERS[i]}: {t[:80]}")
                break
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="both", choices=["train", "test", "both"])
    ap.add_argument("--all", action="store_true", help="every question, not only the ones that failed alone")
    ap.add_argument("--strict", action="store_true",
                    help="rewrite, under stricter rules, the sheets whose newest version leaked (leak.json)")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    C.add_dir_args(ap)
    args = ap.parse_args()

    version = PROMPT_VERSION + ("-strict" if args.strict else "")
    instructions = INSTRUCTIONS + (STRICT_ADDENDUM if args.strict else "")
    tdir = C.teacher_dir(args)             # sheets and the teacher cache; shared between small models
    existing = C.read_sheets(tdir)
    leaks = C.read_json(args.out_dir / "leak.json", {})     # this small model's leak check

    rows = []
    for split in C.splits(args.split):
        srows = C.load_split(split)
        if args.strict:
            keep = []
            for r in srows:
                vs = existing.get(r["id"], [])
                if vs and leaks.get(f"{r['id']}|{vs[-1]['version']}", {}).get("leak"):
                    keep.append(r)
            srows = keep
        elif not args.all:
            un = C.read_json(args.out_dir / f"unaided_{split}.json")
            if un is None:
                raise SystemExit(f"no unaided_{split}.json yet; run unaided.py first or pass --all")
            srows = [r for r in srows if un["per_question"].get(r["id"], {}).get("fails_alone")]
        rows += [r for r in srows if not any(v["version"] == version for v in existing.get(r["id"], []))]
    rows = rows[: args.limit or None]
    print(f"{len(rows)} questions to write sheets for (version {version})")
    if not rows:
        return

    tdir.mkdir(parents=True, exist_ok=True)
    teacher = C.Teacher(tdir / "teacher_cache.jsonl", workers=args.workers)
    try:
        jobs = [{"key": f"{r['id']}|sheet|{version}", "instructions": instructions, "text": build_input(r),
                 "schema": HintSheet, "effort": args.effort, "max_output_tokens": 20_000} for r in rows]
        recs = teacher.run(jobs, f"hint sheets {version}")
    finally:
        teacher.close()

    n_ok = n_agree = n_flag = 0
    with (tdir / "sheets.jsonl").open("a") as f:
        for r in rows:
            rec = recs[f"{r['id']}|sheet|{version}"]
            if rec.get("error"):
                print(f"  {r['id']}: {rec['error'][:120]}")
                continue
            sheet = rec["out"]
            flags = overlap(sheet, r)
            out = {"id": r["id"], "version": version, "model": rec["model"], "effort": rec["effort"],
                   "kind_describer": r["knowledge"], "kind_teacher": sheet["kind"],
                   "teacher_agrees": sheet["answer"] == r["answer_letter"], "overlap": flags,
                   "sheet": sheet, "usage": rec["usage"]}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_ok += 1
            n_agree += out["teacher_agrees"]
            n_flag += bool(flags)
    print(f"wrote {n_ok} sheets: teacher agrees with the key on {n_agree}, "
          f"{n_flag} carry a choice-text flag (the leak check decides)")


if __name__ == "__main__":
    main()

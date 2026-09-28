"""Step 5: take the hints away from the end and see which way of working still
finishes the job.

For every middle question and every way of working we find the fewest hints
(the first k items of the trimmed sheet) from which that way of working is
right at least REACH_MIN times out of K. We do not test every k: first all
items (if that fails, the way of working cannot finish even with everything),
then halving: half the items, then a quarter or three quarters, and so on.
Three or four rounds pin down the smallest k that works. --full-curve also
tests every k for 'continue_high', which draws the plain curve.

Ways of working (each gets the question plus the first k items):
  continue_high   finish in one go, full thinking                 (the baseline; shares the gate's cache)
  continue_low    the same with thinking turned down (gpt-oss: effort low; Qwen: thinking off)
  vote            N separate continue_high attempts, majority letter (N = --vote-n, default 3)
  critic_first    one call says what could go wrong from here; a second call finishes having read that
  check_first     one call re-does the given items and says which hold; a second call finishes
  plan_first      one call writes the remaining steps as a plan; a second call carries it out
  restart_expert  solves from scratch in an expert voice, the items shown only as a colleague's notes

    python scripts/hint_study/hints_needed.py --split both --full-curve
    python scripts/hint_study/hints_needed.py --split train --modules continue_high --k 1 --limit 1   # smoke

Writes <out-dir>/hints_needed_<split>.json.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

FINISH = ("Continue from where the established items stop and finish the question. Reason step by step, "
          "then commit. ")


def notes_prompt(row: dict, items: list[dict], task: str) -> str:
    return (task + "\n\n" + C.question_block(row) + "\n\nEstablished so far:\n" + C.items_block(items) + "\n")


class Module:
    """A way of working. `first` returns the preparatory job or None; `answers`
    returns the jobs whose replies carry the letter (several for a vote)."""
    name = ""
    effort = "high"

    def first(self, row, items, s, key):
        return None

    def answers(self, row, items, s, key, first_text):
        raise NotImplementedError


class ContinueHigh(Module):
    name = "continue_high"

    def answers(self, row, items, s, key, first_text):
        # identical prompt and key to gate_with_sheet.py, so the gate's replies are reused
        return [C.Student.job(f"{row['id']}|sheet|{key['version']}|{C.item_key(items)}|high|{s}",
                              [{"role": "user", "content": C.with_hints_prompt(row, items)}], "high")]


class ContinueLow(Module):
    name, effort = "continue_low", "low"

    def answers(self, row, items, s, key, first_text):
        return [C.Student.job(f"{row['id']}|sheet|{key['version']}|{C.item_key(items)}|low|{s}",
                              [{"role": "user", "content": C.with_hints_prompt(row, items)}], "low")]


class Vote(Module):
    name = "vote"

    def __init__(self, n: int):
        self.n = n

    def answers(self, row, items, s, key, first_text):
        # its own sample indices (100 + n*s + j) so the votes are fresh draws, not the gate's
        return [C.Student.job(f"{row['id']}|sheet|{key['version']}|{C.item_key(items)}|high|{100 + self.n * s + j}",
                              [{"role": "user", "content": C.with_hints_prompt(row, items)}], "high")
                for j in range(self.n)]


class TwoStage(Module):
    """First call writes something that the second call reads before finishing."""
    first_task = ""
    second_intro = ""

    def first(self, row, items, s, key):
        return C.Student.job(f"{row['id']}|{self.name}:1|{key['version']}|{C.item_key(items)}|{s}",
                             [{"role": "user", "content": notes_prompt(row, items, self.first_task)}], "high")

    def answers(self, row, items, s, key, first_text):
        content = (C.with_hints_prompt(row, items) + "\n" + self.second_intro + "\n" + (first_text or "(none)") + "\n")
        return [C.Student.job(f"{row['id']}|{self.name}:2|{key['version']}|{C.item_key(items)}|{s}",
                              [{"role": "user", "content": content}], "high")]


class CriticFirst(TwoStage):
    name = "critic_first"
    first_task = ("You are a Critic. Do NOT answer the question. Read the question and the items established so far, "
                  "and write a short list of what could go wrong from here: a step that is easy to get wrong, a "
                  "constraint in the question that is easy to miss, a choice that looks right but is a trap. "
                  "Be specific. Never write an 'Answer:' line.")
    second_intro = "A critic has written the following warnings about what could go wrong from here:"


class CheckFirst(TwoStage):
    name = "check_first"
    first_task = ("You are a Checker. Do NOT answer the question. Re-do each of the established items in your own "
                  "words and say for each whether it holds, with the reason. If one does not hold, say what is "
                  "wrong with it. Never write an 'Answer:' line.")
    second_intro = "A checker has gone over the established items and reports:"


class PlanFirst(TwoStage):
    name = "plan_first"
    first_task = ("You are a Planner. Do NOT answer the question. Starting from the established items, write the "
                  "remaining steps needed to reach the answer as a short numbered plan, one action per line, "
                  "without carrying any of them out. Never write an 'Answer:' line.")
    second_intro = "A planner has written the remaining steps as a plan; carry it out:"


class RestartExpert(Module):
    name = "restart_expert"

    def answers(self, row, items, s, key, first_text):
        content = ("You are a leading expert in " + row.get("field", "this field") + ", answering a hard "
                   "graduate-level multiple-choice question inside your own specialty. Solve it from scratch, "
                   "reasoning step by step from your expert knowledge. A colleague's notes are attached; they may "
                   "or may not be useful. " + C.answer_line(row) + "\n\n" + C.question_block(row)
                   + "\n\nColleague's notes:\n" + C.items_block(items) + "\n")
        return [C.Student.job(f"{row['id']}|restart_expert|{key['version']}|{C.item_key(items)}|{s}",
                              [{"role": "user", "content": content}], "high")]


def make_modules(names: list[str], vote_n: int) -> list[Module]:
    all_mods = {m.name: m for m in [ContinueHigh(), ContinueLow(), Vote(vote_n), CriticFirst(), CheckFirst(),
                                    PlanFirst(), RestartExpert()]}
    bad = [n for n in names if n not in all_mods]
    if bad:
        raise SystemExit(f"unknown ways of working: {bad}; known: {list(all_mods)}")
    return [all_mods[n] for n in names]


def outcome(recs: dict, jobs: list[dict], row: dict) -> tuple[bool, list]:
    """One try's letter(s): a single reply's letter, or the majority of several."""
    preds = [C.extract_answer(recs[j["key"]].get("content"), row["options"])
             if recs[j["key"]].get("error") is None else None for j in jobs]
    if len(preds) == 1:
        return C.is_right(preds[0], row), preds
    votes = Counter(p for p in preds if p is not None)
    top = votes.most_common(1)[0][0] if votes else None
    return C.is_right(top, row), preds


def run_tests(student: C.Student, tests: list[tuple[dict, Module, int, dict]], k: int, label: str) -> dict:
    """tests: (row, module, n_hints, key) -> results keyed by (qid, module, n_hints):
    {n, c, preds}. Every test's first-stage jobs run together, then every
    answer job, so the server stays busy."""
    firsts, plan = [], {}
    for row, mod, kk, key in tests:
        items = key["items"][:kk]
        for s in range(k):
            j = mod.first(row, items, s, key)
            plan[(row["id"], mod.name, kk, s)] = j
            if j is not None:
                firsts.append(j)
    frecs = student.run(firsts, f"{label}: first calls") if firsts else {}
    answers, ajobs = [], {}
    for row, mod, kk, key in tests:
        items = key["items"][:kk]
        for s in range(k):
            fj = plan[(row["id"], mod.name, kk, s)]
            ftext = frecs[fj["key"]].get("content") if fj is not None else None
            js = mod.answers(row, items, s, key, ftext)
            ajobs[(row["id"], mod.name, kk, s)] = js
            answers += js
    arecs = student.run(answers, f"{label}: answers")
    out = {}
    for row, mod, kk, key in tests:
        rights, preds = [], []
        for s in range(k):
            ok, p = outcome(arecs, ajobs[(row["id"], mod.name, kk, s)], row)
            rights.append(ok)
            preds.append(p if len(p) > 1 else p[0])
        out[(row["id"], mod.name, kk)] = {"n": k, "c": sum(rights), "preds": preds}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="both", choices=["train", "test", "both"])
    ap.add_argument("--k", type=int, default=C.K)
    ap.add_argument("--modules", default="continue_high,continue_low,vote,critic_first,check_first,plan_first,restart_expert")
    ap.add_argument("--vote-n", type=int, default=3)
    ap.add_argument("--full-curve", action="store_true", help="also test every k for continue_high")
    ap.add_argument("--kinds", default="derive,both,recall", help="which question kinds to run")
    ap.add_argument("--limit", type=int, default=0)
    C.add_student_args(ap)
    C.add_dir_args(ap)
    args = ap.parse_args()
    reach_min = C.REACH_MIN if args.k == C.K else 1
    modules = make_modules([m for m in args.modules.split(",") if m], args.vote_n)
    kinds = set(args.kinds.split(","))

    student = C.make_student(args)
    try:
        for split in C.splits(args.split):
            mid = C.read_json(args.out_dir / f"middle_{split}.json")
            if mid is None:
                raise SystemExit(f"no middle_{split}.json; run gate_with_sheet.py first")
            rows = [r for r in C.load_split(split)
                    if mid["per_question"].get(r["id"], {}).get("keep") and r["knowledge"] in kinds][: args.limit or None]
            keys = {r["id"]: {"version": mid["per_question"][r["id"]]["version"],
                              "items": mid["per_question"][r["id"]]["min_items"]} for r in rows}
            print(f"== {split}: {len(rows)} middle questions, {len(modules)} ways of working")
            if not rows:
                continue
            results = {}       # (qid, module, k) -> {n, c, preds}

            # round 0: everything given. If that fails the way of working cannot finish at all.
            tests = [(r, m, len(keys[r["id"]]["items"]), keys[r["id"]]) for r in rows for m in modules]
            results.update(run_tests(student, tests, args.k, f"{split} all hints"))
            state = {}         # (qid, module) -> [lo, hi]: lo fails (or -1), hi passes
            for r in rows:
                n = len(keys[r["id"]]["items"])
                for m in modules:
                    state[(r["id"], m.name)] = [-1, n] if results[(r["id"], m.name, n)]["c"] >= reach_min else None
            rnd = 0
            while True:
                tests = []
                for (qid, mname), st in state.items():
                    if st is None or st[1] - st[0] <= 1:
                        continue
                    mid_k = (st[0] + st[1]) // 2
                    tests.append((next(r for r in rows if r["id"] == qid), next(m for m in modules if m.name == mname),
                                  mid_k, keys[qid]))
                if not tests:
                    break
                rnd += 1
                results.update(run_tests(student, tests, args.k, f"{split} halving round {rnd}"))
                for row, m, kk, _ in tests:
                    st = state[(row["id"], m.name)]
                    if results[(row["id"], m.name, kk)]["c"] >= reach_min:
                        st[1] = kk
                    else:
                        st[0] = kk
            if args.full_curve and any(m.name == "continue_high" for m in modules):
                m = next(m for m in modules if m.name == "continue_high")
                tests = [(r, m, kk, keys[r["id"]]) for r in rows for kk in range(len(keys[r["id"]]["items"]) + 1)
                         if (r["id"], m.name, kk) not in results]
                if tests:
                    results.update(run_tests(student, tests, args.k, f"{split} full curve"))

            per_q, summary = {}, {}
            for r in rows:
                n = len(keys[r["id"]]["items"])
                per_q[r["id"]] = {"kind": r["knowledge"], "n_items": n, "version": keys[r["id"]]["version"],
                                  "items": keys[r["id"]]["items"], "modules": {}}
                for m in modules:
                    st = state[(r["id"], m.name)]
                    tested = {str(kk): v for (q, mn, kk), v in results.items() if q == r["id"] and mn == m.name}
                    per_q[r["id"]]["modules"][m.name] = {"needed": None if st is None else st[1], "tested": tested}
            for m in modules:
                for kind, rs in [("all", rows)] + list(C.by_kind(rows).items()):
                    qs = [per_q[r["id"]] for r in rs]
                    if not qs:
                        continue
                    needed = [q["modules"][m.name]["needed"] for q in qs]
                    fin = [(nd, q["n_items"]) for nd, q in zip(needed, qs) if nd is not None]
                    summary.setdefault(m.name, {})[kind] = {
                        "questions": len(qs), "cannot_finish": sum(nd is None for nd in needed),
                        "needed_mean": C.mean(nd for nd, _ in fin),
                        "share_of_sheet_mean": C.mean(nd / n if n else 0.0 for nd, n in fin),
                        "finishes_with_no_hints": sum(nd == 0 for nd in needed)}
            curve = {}
            if args.full_curve:
                for r in rows:
                    for kk, v in per_q[r["id"]]["modules"].get("continue_high", {}).get("tested", {}).items():
                        curve.setdefault(kk, []).append(v["c"] / v["n"])
                curve = {kk: {"questions": len(v), "pass_rate": C.mean(v)} for kk, v in sorted(curve.items(), key=lambda x: int(x[0]))}
            C.write_json(args.out_dir / f"hints_needed_{split}.json",
                         {"split": split, "k": args.k, "reach_min": reach_min, "vote_n": args.vote_n, "model": args.model,
                          "summary": summary, "continue_high_curve": curve, "per_question": per_q})
            for mname, by in summary.items():
                s = by["all"]
                print(f"  {mname:15s} needed {s['needed_mean']:.2f} items ({100 * s['share_of_sheet_mean']:.0f}% of sheet)  "
                      f"no hints needed {s['finishes_with_no_hints']}  cannot finish {s['cannot_finish']}")
    finally:
        student.close()


if __name__ == "__main__":
    main()

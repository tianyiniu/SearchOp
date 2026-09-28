"""Step 4: keep the middle questions and trim each sheet to what is needed.

Gate: the question with its full sheet, K tries. Keep if right at least
PASS_MIN times. These are the middle questions.

Trim (--trim): leave each item out in turn, K tries each. Items whose removal
keeps the pass rate at PASS_MIN or more are candidates. They are removed one
at a time, most harmless first, re-testing after each removal, because two
items that each look unnecessary can be needed together. What is left is the
shortest sheet that still works.

    python scripts/hint_study/gate_with_sheet.py --split both --trim
    python scripts/hint_study/gate_with_sheet.py --split train --k 1 --limit 1    # smoke test

Writes <out-dir>/middle_<split>.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def key_of(qid: str, version: str, items: list[dict], effort: str, s: int) -> str:
    return f"{qid}|sheet|{version}|{C.item_key(items)}|{effort}|{s}"


def jobs_for(row: dict, version: str, items: list[dict], effort: str, k: int) -> list[dict]:
    return [C.Student.job(key_of(row["id"], version, items, effort, s),
                          [{"role": "user", "content": C.with_hints_prompt(row, items)}], effort)
            for s in range(k)]


def passes(student: C.Student, row: dict, version: str, items: list[dict], effort: str, k: int,
           recs: dict) -> dict:
    sc = C.score([recs[key_of(row["id"], version, items, effort, s)] for s in range(k)], row)
    sc["ok"] = sc["c"] >= C.PASS_MIN if k == C.K else sc["c"] > 0
    return sc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="both", choices=["train", "test", "both"])
    ap.add_argument("--k", type=int, default=C.K)
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high"])
    ap.add_argument("--trim", action="store_true", help="also find the shortest sheet that still works")
    ap.add_argument("--limit", type=int, default=0)
    C.add_student_args(ap)
    C.add_dir_args(ap)
    args = ap.parse_args()
    if args.k != C.K:
        print(f"note: k={args.k}; with fewer than {C.K} tries 'keep' means 'right at least once' (smoke test only)")

    sheets = C.active_sheets(C.teacher_dir(args), args.out_dir)
    student = C.make_student(args)
    try:
        for split in C.splits(args.split):
            rows = [r for r in C.load_split(split) if r["id"] in sheets][: args.limit or None]
            print(f"== {split}: {len(rows)} questions with a usable sheet")
            if not rows:
                continue
            vers = {r["id"]: sheets[r["id"]]["version"] for r in rows}
            full = {r["id"]: C.sheet_items(sheets[r["id"]]["sheet"]) for r in rows}

            # the gate
            jobs = [j for r in rows for j in jobs_for(r, vers[r["id"]], full[r["id"]], args.effort, args.k)]
            recs = student.run(jobs, f"gate {split}")
            per_q = {}
            for r in rows:
                sc = passes(student, r, vers[r["id"]], full[r["id"]], args.effort, args.k, recs)
                per_q[r["id"]] = {"version": vers[r["id"]], "kind": r["knowledge"], "n": sc["n"], "c": sc["c"],
                                  "preds": sc["preds"], "keep": sc["ok"], "leak_checked": sheets[r["id"]]["leak_checked"],
                                  "n_items": len(full[r["id"]]), "items": full[r["id"]],
                                  "min_items": full[r["id"]], "removed": [], "loo": {}}
            middle = [r for r in rows if per_q[r["id"]]["keep"]]
            print(f"   middle questions: {len(middle)} of {len(rows)}")

            if args.trim:
                # leave-one-out, all questions at once
                loo_jobs = []
                for r in middle:
                    its = full[r["id"]]
                    if len(its) < 2:
                        continue
                    for i in range(len(its)):
                        loo_jobs += jobs_for(r, vers[r["id"]], its[:i] + its[i + 1:], args.effort, args.k)
                recs = student.run(loo_jobs, f"leave-one-out {split}")
                cands = {}
                for r in middle:
                    its = full[r["id"]]
                    if len(its) < 2:
                        continue
                    cs = []
                    for i in range(len(its)):
                        sub = its[:i] + its[i + 1:]
                        sc = passes(student, r, vers[r["id"]], sub, args.effort, args.k, recs)
                        per_q[r["id"]]["loo"][C.item_key([its[i]])] = sc["c"]
                        if sc["ok"]:
                            cs.append((-sc["c"], i))
                    cands[r["id"]] = [i for _, i in sorted(cs)]      # most harmless first
                # greedy removal, one item per round across all questions
                current = {r["id"]: list(full[r["id"]]) for r in middle}
                pending = {qid: list(cs) for qid, cs in cands.items() if cs}
                while pending:
                    trial = {}
                    for qid, cs in pending.items():
                        i = cs[0]
                        trial[qid] = [it for it in current[qid] if not (it["kind"] == full[qid][i]["kind"]
                                                                        and it["idx"] == full[qid][i]["idx"])]
                    rows_by_id = {r["id"]: r for r in middle}
                    jobs = [j for qid, its in trial.items()
                            for j in jobs_for(rows_by_id[qid], vers[qid], its, args.effort, args.k)]
                    recs = student.run(jobs, f"trim round {split}")
                    for qid in list(pending):
                        i = pending[qid].pop(0)
                        sc = passes(student, rows_by_id[qid], vers[qid], trial[qid], args.effort, args.k, recs)
                        if sc["ok"] and len(trial[qid]) >= 1:
                            current[qid] = trial[qid]
                            per_q[qid]["removed"].append(C.item_key([full[qid][i]]))
                        if not pending[qid]:
                            del pending[qid]
                for qid, its in current.items():
                    per_q[qid]["min_items"] = its

            summary = {}
            for kind, rs in [("all", rows)] + list(C.by_kind(rows).items()):
                qs = [per_q[r["id"]] for r in rs]
                if qs:
                    kept = [q for q in qs if q["keep"]]
                    summary[kind] = {"with_sheet": len(qs), "middle": len(kept),
                                     "items_mean": C.mean(q["n_items"] for q in kept),
                                     "min_items_mean": C.mean(len(q["min_items"]) for q in kept),
                                     "unchecked_for_leak": sum(not q["leak_checked"] for q in qs)}
            C.write_json(args.out_dir / f"middle_{split}.json",
                         {"split": split, "k": args.k, "effort": args.effort, "model": args.model, "pass_min": C.PASS_MIN,
                          "trimmed": args.trim, "summary": summary, "per_question": per_q})
            for kind, s in summary.items():
                print(f"  {kind:6s} with sheet {s['with_sheet']:3d}  middle {s['middle']:3d}  "
                      f"items {s['items_mean']:.1f} -> {s['min_items_mean']:.1f}")
    finally:
        student.close()


if __name__ == "__main__":
    main()

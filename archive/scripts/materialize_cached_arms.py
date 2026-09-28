"""Materialize cached full-run results into the retrieve_<arm>_<split>_cache.jsonl
format that retrieve_schema_mcq.py --summarize-only reads.

Before the train/test split, the whole 5027-question pool was already evaluated:
  - self_critique, fixed_debate            -> outputs/debate_mcq_full_cache.jsonl
  - single_pass (= evolution step-0), evo  -> outputs/evolve_mcq_full_cache.jsonl

The 1000 test ids are a subset of that pool, so their statistics need no re-run --
just a filter + reshape. This writes, for the test split:
  outputs/retrieve_single_pass_test_cache.jsonl    (from evolution step-0, T=0.7)
  outputs/retrieve_self_critique_test_cache.jsonl  (cached; T=0.3 -- reused as-is)
  outputs/retrieve_fixed_debate_test_cache.jsonl   (cached; T=0.7)
and, as the oracle ceiling, outputs/evolve_mcq_test_summary.json (filtered evolution).

Only these three arms are cache-derivable; always_critic was never run, and the
retrieval arms are new -- run those fresh, then retrieve_schema_mcq.py
--summarize-only folds cached + fresh into one comparison. Re-running any of the
three arms fresh would overwrite its materialized file (harmless -- just newer data).

    python scripts/materialize_cached_arms.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def split_tag(dataset: Path) -> str:
    stem = dataset.stem
    for pre in ("supergpqa_", "frames_"):
        stem = stem.replace(pre, "")
    return stem


def write_cache(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(args: argparse.Namespace) -> None:
    rows = json.loads(args.split_dataset.read_text())
    meta = {r["id"]: {"field": r.get("field"), "difficulty": r.get("difficulty"),
                      "answer_letter": r.get("answer_letter")} for r in rows}
    test_ids = set(meta)
    tag = split_tag(args.split_dataset)
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"test split: {len(test_ids)} ids | tag={tag}\n")

    def base(qid: str, arm: str, correct: bool, answer, source: str) -> dict:
        return {"id": qid, "arm": arm, **meta[qid], "answer": answer,
                "correct": bool(correct), "status": "ok", "source": source}

    # --- single_pass: evolution step-0 (minimal schema, one execution @0.7) ---
    evo = {}
    for line in open(args.evo_cache):
        r = json.loads(line)
        if r["id"] in test_ids:
            evo[r["id"]] = r
    sp = []
    for qid in test_ids:
        s0 = evo[qid]["steps"][0]
        sp.append(base(qid, "single_pass", s0["correct"], s0.get("answer"),
                       "evolve_mcq_full_cache:step0"))
    write_cache(args.outdir / f"retrieve_single_pass_{tag}_cache.jsonl", sp)

    # --- self_critique + fixed_debate: from the fixed-template full cache -----
    dm = {"self_critique": {}, "fixed_debate": {}}
    for line in open(args.debate_cache):
        r = json.loads(line)
        if r.get("status") == "ok" and r["id"] in test_ids and r["shape"] in dm:
            dm[r["shape"]][r["id"]] = r
    for shape in ("self_critique", "fixed_debate"):
        recs = [base(qid, shape, dm[shape][qid]["correct"], dm[shape][qid].get("answer"),
                     "debate_mcq_full_cache")
                for qid in test_ids if qid in dm[shape]]
        write_cache(args.outdir / f"retrieve_{shape}_{tag}_cache.jsonl", recs)
        dm[shape + "_recs"] = recs

    # --- GT evolution: the oracle-ceiling summary for this split -------------
    solved = [qid for qid in test_ids if evo[qid].get("solved")]
    solved_at = Counter(evo[qid]["solved_at"] for qid in solved)
    evo_summary = {
        "split": tag, "n_questions": len(test_ids), "mode": "gt_aware",
        "solved": len(solved), "solve_rate": len(solved) / len(test_ids),
        "avg_mods_to_solve": (sum(evo[q]["solved_at"] for q in solved) / len(solved)) if solved else 0.0,
        "solved_at_step": dict(sorted(solved_at.items())),
        "source": "evolve_mcq_full_cache (filtered to test ids)",
    }
    (args.outdir / f"evolve_mcq_{tag}_summary.json").write_text(json.dumps(evo_summary, indent=2))

    # --- report -------------------------------------------------------------
    def rate(recs):
        n = len(recs)
        c = sum(1 for r in recs if r["correct"])
        return f"{c}/{n} ({c / n:.1%})" if n else "0/0"

    print("materialized -> retrieve_<arm>_%s_cache.jsonl:" % tag)
    print(f"  single_pass    {rate(sp)}")
    print(f"  self_critique  {rate(dm['self_critique_recs'])}   (T=0.3, reused as-is)")
    print(f"  fixed_debate   {rate(dm['fixed_debate_recs'])}")
    print(f"\noracle ceiling -> evolve_mcq_{tag}_summary.json:")
    print(f"  GT evolution   {evo_summary['solved']}/{evo_summary['n_questions']} "
          f"({evo_summary['solve_rate']:.1%})  solved_at={evo_summary['solved_at_step']}")
    print(f"\nstill to run fresh: always_critic, retrieve_copy, retrieve_synth")
    print(f"then: python scripts/retrieve_schema_mcq.py --dataset {args.split_dataset} --summarize-only")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dataset", type=Path, default=Path("datasets/supergpqa_test.json"))
    ap.add_argument("--debate-cache", type=Path, default=Path("outputs/debate_mcq_full_cache.jsonl"))
    ap.add_argument("--evo-cache", type=Path, default=Path("outputs/evolve_mcq_full_cache.jsonl"))
    ap.add_argument("--outdir", type=Path, default=Path("outputs"))
    main(ap.parse_args())

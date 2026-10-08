"""Seed stage of the cluster pipeline's per-group program search.

The seeds are chosen by content, not tied to the size of a generation:

    k   model-written   one per group, written in one batch by the guide model
                        (program_guide.GUIDE_MODEL), which is shown the literature
                        programs and the groups. They are cached beside the output
                        (<out>.llm.json) and reused; a cache written with more per
                        group gives the first program of each group.
    8   literature      program_space.seed_protocols() less every low-effort
                        program that has a high-effort version (name + "_high"):
                        mad, early_exit_agree, expert_first, direct_high,
                        self_refine_high, self_consistency_high,
                        verify_then_decide_high, fresh_on_disagree_high. Under the
                        v2 executor (no high-effort programs) all 8 protocols. The
                        archive only grows, so a group's A-slot holder never scores
                        below the best of them on that group's search questions,
                        nor the global holder on all of them.
    2   judge           with --judge-persona only (program_space.judge_seeds):
                        judge_on_disagree_high, pool_judge_high. Random edits rarely
                        insert a judge, so without these it might never be tried.
    0   random          only as a stand-in: if the guide model returns fewer
                        programs than asked (or none, with --no-guide), random
                        programs take the missing places. They are drawn one
                        after another and the first ones that pass a short sanity
                        run are kept (it drops programs that stop at once or run
                        to the turn cap).

With k = 3 groups that is 3 + 8 = 11 seeds (13 with the judge). Every seed's `source` is recorded,
and the search carries it along as the lineage of every descendant, so the
final programs can be traced back to a literature, model-written or random
ancestor.

    python scripts/program_seeds_cluster.py --out outputs/pipeline_cluster/seeds.json

Offline (no debate model, no guide model; random programs are not sanity-run):

    python scripts/program_seeds_cluster.py --no-live --no-guide --out /tmp/seeds.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import program_guide as G  # noqa: E402
import program_seeds as S  # noqa: E402
import program_space as P  # noqa: E402

ROOT = P.ROOT
LLM_PER_GROUP = 1                  # model-written seeds per question group


def literature_seeds() -> dict[str, dict]:
    """seed_protocols() less every low-effort program that has a high-effort
    version, so each protocol is seeded once (at high effort where it has one)."""
    lit = P.seed_protocols()
    return {name: prog for name, prog in lit.items() if f"{name}_high" not in lit}


def pick_written(written: list[dict], groups: list[int], per_group: int) -> list[dict]:
    """The first `per_group` model-written programs of each group, in group
    order (a cache written with more per group is cut down per group)."""
    return [w for g in groups for w in [x for x in written if x.get("group") == g][:per_group]]


def fixed_seeds() -> dict[str, tuple[str, dict]]:
    """The seeds that are not written by a model: name -> (source, program). The
    literature programs, then the judge programs when the judge is on."""
    out = {name: ("protocol", prog) for name, prog in literature_seeds().items()}
    out.update({name: ("judge", prog) for name, prog in P.judge_seeds().items()})
    return out


def seed_counts(k: int) -> tuple[int, int, int]:
    """(model-written, literature and judge, random) for k groups."""
    return LLM_PER_GROUP * k, len(fixed_seeds()), 0


def n_seeds(k: int) -> int:
    return sum(seed_counts(k))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clusters", type=Path, default=ROOT / "outputs/describe_v3/clusters_600_train.json")
    ap.add_argument("--dataset", type=Path, default=ROOT / "datasets/supergpqa_600_train.json")
    ap.add_argument("--per-group", type=int, default=50, help="as in the search; 0 = whole groups")
    ap.add_argument("--dev-split", type=Path, default=None,
                    help="a split file of split_train_dev.py: each group's dev questions are held out for "
                         "--pick-champions, the rest are search questions (with --per-group N > 0, the "
                         "first N of them)")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/pipeline_cluster/seeds.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sanity-per-group", type=int, default=2, help="sanity-run questions per group")
    ap.add_argument("--min-turns", type=float, default=2.0)
    ap.add_argument("--max-turns", type=float, default=12.0)
    ap.add_argument("--max-calls-per-question", type=int, default=None,
                    help="default: the turn cap (--turn-cap, 16 unless set)")
    ap.add_argument("--workers", type=int, default=64,
                    help="debates in flight at once; each may have up to 4 speakers in flight")
    ap.add_argument("--no-guide", action="store_true",
                    help="skip the model-written seeds (their places go to random programs)")
    ap.add_argument("--guide-effort", default=G.GUIDE_EFFORT)
    ap.add_argument("--seed-examples", type=int, default=10,
                    help="example questions per group shown to the guide model (its first search questions)")
    ap.add_argument("--seed-example-chars", type=int, default=2500,
                    help="an example question longer than this keeps its start and its end")
    ap.add_argument("--no-live", action="store_true",
                    help="no debate model: skip the sanity run")
    live = ap.add_argument_group("debate model")
    live.add_argument("--base-urls", default=P.DEFAULT_BASE_URLS)
    live.add_argument("--model", default=P.DEFAULT_MODEL)
    live.add_argument("--api-key", default="EMPTY")
    live.add_argument("--temperature", type=float, default=0.7)
    live.add_argument("--live-cache", type=Path, default=None,
                      help="round cache (default <out dir>/rounds_<model>.jsonl)")
    live.add_argument("--ignore-cache-lock", action="store_true")
    P.add_executor_args(ap)
    args = ap.parse_args()

    # offline, the window only labels the seed file; the search checks its own
    settings = P.configure_from_args(args, fallback_window=32768 if args.no_live else None)
    settings["model"] = args.model
    rng = random.Random(args.seed)
    groups = P.load_groups(args.clusters, args.per_group, args.dev_split)
    rows = {r["id"]: r for r in json.loads(args.dataset.read_text())}
    P.check_rows(rows.values())             # the dataset suits the answer mode
    if args.live_cache is None:
        args.live_cache = args.out.parent / f"rounds_{P.model_tag(args.model)}.jsonl"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(args.live_cache).parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; the seed set is fixed once written (move it to redo)")

    k = len(groups["groups"])
    n_llm, n_lit, n_random = seed_counts(k)
    n_judge = len(P.judge_seeds())
    print(f"k = {k} groups: {n_seeds(k)} seeds = {n_llm} model-written + {n_lit - n_judge} literature "
          + (f"+ {n_judge} judge " if n_judge else "") + f"+ {n_random} random")

    seeds: list[dict] = []
    for name, (source, prog) in fixed_seeds().items():
        P.validate_program(prog)
        seeds.append({"name": name, "source": source, "program": P.normalize_program(prog)})

    # model-written seeds, cached beside the output so a rerun never re-asks
    if not args.no_guide:
        llm_path = args.out.with_suffix(".llm.json")
        if llm_path.exists():
            written = json.loads(llm_path.read_text())
            print(f"reusing {len(written)} model-written programs from {llm_path}")
        else:
            client = G.make_client()
            # each group: its profile (template, typical steps, failure risks, knowledge) and
            # example questions; no difficulty labels and no scores
            texts = {g["group"]: "\n".join(filter(None, [
                         P.group_profile_text(g),
                         P.group_samples_text(g, rows, args.seed_examples, args.seed_example_chars)]))
                     for g in groups["groups"]}
            written = G.write_group_seeds(client, {s["name"]: s["program"] for s in seeds}, texts,
                                          effort=args.guide_effort, per_group=LLM_PER_GROUP,
                                          examples=args.seed_examples)
            llm_path.write_text(json.dumps(written, indent=1))
            print(f"{len(written)} model-written programs ({G.GUIDE_MODEL}) -> {llm_path}")
        used = {s["name"] for s in seeds}
        taken = {P.canon(s["program"]) for s in seeds}
        for w in pick_written(written, [g["group"] for g in groups["groups"]], LLM_PER_GROUP):
            P.validate_program(w["program"])
            if P.canon(w["program"]) in taken:
                continue
            name = w["name"] if w["name"] not in used else f"{w['name']}_{w.get('group', len(used))}"
            used.add(name)
            taken.add(P.canon(w["program"]))
            seeds.append({"name": name, "source": "llm", "group": w.get("group"),
                          "program": w["program"], "strategy": w.get("strategy", "")})
    n_random = n_seeds(k) - len(seeds)                # any shortfall above goes to random programs
    if n_random != seed_counts(k)[2]:
        print(f"note: {n_random} random programs stand in for model-written seeds the guide model "
              f"did not provide")

    # random seeds: the first n_random valid ones, in sampling order
    seen = {P.canon(s["program"]) for s in seeds}
    runner = None
    qids: list[str] = []
    if not args.no_live and n_random > 0:
        qids = S.sanity_questions(groups, args.sanity_per_group * k, rng)
        runner = P.make_runner(rows, args.live_cache, args.base_urls, args.model, args.temperature,
                               api_key=args.api_key, lock=not args.ignore_cache_lock)
        runner.reset_budget(None)
    picked: list[tuple[dict, dict | None]] = []
    n_drawn = 0
    try:
        while len(picked) < n_random:
            batch: list[dict] = []
            while len(batch) < 2 * (n_random - len(picked)):
                prog = P.random_program(rng)
                if P.canon(prog) not in seen and not P.reviewer_first(prog):
                    seen.add(P.canon(prog))
                    batch.append(prog)
            n_drawn += len(batch)
            if runner is None:
                picked += [(prog, None) for prog in batch][: n_random - len(picked)]
                break
            runner.stage(f"sanity run: {len(batch)} candidates x {len(qids)} questions")
            outs = P.run_pairs([(i, prog, q) for i, prog in enumerate(batch) for q in qids],
                               runner, rows, 0, args.max_calls_per_question, workers=args.workers,
                               desc="sanity")
            for i, prog in enumerate(batch):                 # sampling order decides
                done = [o for (j, q), o in outs.items() if j == i and o is not None]
                turns = sum(o["n_calls"] for o in done) / len(done) if done else 0.0
                acc = sum(o["correct"] for o in done) / len(done) if done else 0.0
                ok = bool(done) and args.min_turns <= turns <= args.max_turns
                if ok and len(picked) < n_random:
                    picked.append((prog, {"mean_turns": turns, "acc": acc, "n": len(done)}))
    finally:
        if runner is not None:
            runner.close()
    for j, (prog, sanity) in enumerate(picked):
        seeds.append({"name": f"random_{j}", "source": "random", "program": prog, "sanity": sanity})
    print(f"{len(picked)} random programs kept, the first valid of {n_drawn} drawn"
          + ("" if runner is not None else " (no sanity run)"))

    for s in seeds:
        s["summary"] = S.summarize(s["program"])
    out = {"settings": settings, "clusters": groups["source"], "per_group": args.per_group,
           "seed": args.seed, "k": k, "guide_model": None if args.no_guide else G.GUIDE_MODEL,
           "seeds": seeds}
    args.out.write_text(json.dumps(out, indent=1))
    report = [f"# Seed programs ({len(seeds)})", "",
              f"Executor: {settings}, model {args.model}", "",
              "| name | source | plan | rules | extra rounds | stop reads | sanity turns |",
              "|---|---|---|---|---|---|---|"]
    for s in seeds:
        sm, sa = s["summary"], s.get("sanity") or {}
        report.append(f"| {s['name']} | {s['source']} | {' > '.join(sm['plan'])} | {sm['n_rules']} | "
                      f"{', '.join(sm['moves']) or '-'} | {', '.join(sm['stop_reads'])} | "
                      + (f"{sa['mean_turns']:.1f} |" if sa else "- |"))
    report += ["", "## Programs", ""]
    for s in seeds:
        report.append(f"### {s['name']} ({s['source']}"
                      + (f", group {s['group']}" if s.get("group") is not None else "") + ")")
        if s.get("strategy"):
            report.append(s["strategy"])
        report.append("```json\n" + json.dumps(s["program"], indent=1) + "\n```")
        report.append("")
    args.out.with_suffix(".md").write_text("\n".join(report))
    print(f"wrote {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()

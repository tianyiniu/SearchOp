"""Seed stage of the v3 per-group program search.

The number of seeds is tied to the number of question groups k, so that
generation 0 is the same size as every later generation (4k programs):

    k   model-written   one per group, written in one batch by the guide model,
                        which is shown the literature programs and the groups
    8   literature      program_space.seed_protocols(), in their fixed order (fewer
                        only if 4k - k < 8, which needs k <= 2)
    rest random         random programs are drawn one after another and the
                        first ones that pass a short sanity run are kept (the
                        sanity run drops programs that stop at once or run to
                        the turn cap). No pool, no distance-based choice.

With k = 6 that is 6 + 8 + 10 = 24 seeds. Every seed's `source` is recorded,
and the search carries it along as the lineage of every descendant, so the
final programs can be traced back to a literature, model-written or random
ancestor.

    python scripts/program_seeds_v3.py --out outputs/cluster_search_v3/seeds.json

Offline (no debate model, no guide model; random programs are not sanity-run):

    python scripts/program_seeds_v3.py --no-live --no-guide --out /tmp/seeds.json
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
SEEDS_PER_GROUP = 4        # = children per generation per group (2 slots x 2 children)


def seed_counts(k: int) -> tuple[int, int, int]:
    """(model-written, literature, random) for k groups."""
    total = SEEDS_PER_GROUP * k
    n_llm = k
    n_lit = min(len(P.seed_protocols()), total - n_llm)
    return n_llm, n_lit, total - n_llm - n_lit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clusters", type=Path, default=ROOT / "outputs/clusters_train_both_v3.json")
    ap.add_argument("--dataset", type=Path, default=ROOT / "datasets/supergpqa_program_search_train.json")
    ap.add_argument("--per-group", type=int, default=50)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/cluster_search_v3/seeds.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sanity-per-group", type=int, default=2, help="sanity-run questions per group")
    ap.add_argument("--min-turns", type=float, default=2.0)
    ap.add_argument("--max-turns", type=float, default=12.0)
    ap.add_argument("--max-calls-per-question", type=int, default=16)
    ap.add_argument("--workers", type=int, default=64,
                    help="debates in flight at once; each may have up to 4 speakers in flight")
    ap.add_argument("--no-guide", action="store_true",
                    help="skip the model-written seeds (their places go to random programs)")
    ap.add_argument("--guide-effort", default=G.GUIDE_EFFORT)
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
    ap.add_argument("--digest-head", type=int, default=P.DIGEST_HEAD)
    ap.add_argument("--digest-tail", type=int, default=P.DIGEST_TAIL)
    ap.add_argument("--visible-reasoning", action="store_true",
                    help="for models that think in a hidden channel (gpt-oss): ask every speaker to "
                         "write its reasoning in the visible reply; changes prompts and cache keys")
    P.add_executor_args(ap)
    args = ap.parse_args()

    settings = P.configure_executor(args.digest_head, args.digest_tail, args.visible_reasoning,
                                    deep_think=args.deep_think, summary_words=args.summary_words)
    settings["model"] = args.model
    rng = random.Random(args.seed)
    groups = P.load_groups(args.clusters, args.per_group)
    rows = {r["id"]: r for r in json.loads(args.dataset.read_text())}
    if args.live_cache is None:
        args.live_cache = args.out.parent / f"rounds_{P.model_tag(args.model)}.jsonl"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(args.live_cache).parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; the seed set is fixed once written (move it to redo)")

    k = len(groups["groups"])
    n_llm, n_lit, n_random = seed_counts(k)
    print(f"k = {k} groups: {SEEDS_PER_GROUP * k} seeds = {n_llm} model-written + {n_lit} literature "
          f"+ {n_random} random")

    seeds: list[dict] = []
    for name, prog in list(P.seed_protocols().items())[:n_lit]:
        P.validate_program(prog)
        seeds.append({"name": name, "source": "protocol", "program": P.normalize_program(prog)})

    # model-written seeds, cached beside the output so a rerun never re-asks
    if not args.no_guide:
        llm_path = args.out.with_suffix(".llm.json")
        if llm_path.exists():
            written = json.loads(llm_path.read_text())
            print(f"reusing {len(written)} model-written programs from {llm_path}")
        else:
            client = G.make_client()
            texts = {g["group"]: P.group_profile_text(g, P.difficulty_mix(rows, g["search"]))
                     for g in groups["groups"]}
            written = G.write_group_seeds(client, {s["name"]: s["program"] for s in seeds}, texts,
                                          effort=args.guide_effort)
            llm_path.write_text(json.dumps(written, indent=1))
            print(f"{len(written)} model-written programs ({G.GUIDE_MODEL}) -> {llm_path}")
        used = {s["name"] for s in seeds}
        taken = {P.canon(s["program"]) for s in seeds}
        for w in written[:n_llm]:
            P.validate_program(w["program"])
            if P.canon(w["program"]) in taken:
                continue
            name = w["name"] if w["name"] not in used else f"{w['name']}_{w.get('group', len(used))}"
            used.add(name)
            taken.add(P.canon(w["program"]))
            seeds.append({"name": name, "source": "llm", "group": w.get("group"),
                          "program": w["program"], "strategy": w.get("strategy", "")})
    n_random = SEEDS_PER_GROUP * k - len(seeds)      # any shortfall above goes to random programs
    if n_random != seed_counts(k)[2]:
        print(f"note: {seed_counts(k)[2]} random seeds were planned, {n_random} are needed "
              f"(fewer model-written programs than groups)")

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
                if P.canon(prog) not in seen:
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
              f"Executor: v2, digest {settings['digest']}, model {args.model}", "",
              "| name | source | plan | rules | moves | stop reads | sanity turns |",
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

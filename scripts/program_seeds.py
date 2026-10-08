"""Seed stage of the per-group program search: the 20 programs generation 1
starts from.

    8 protocol programs   hand-written from the literature (program_space.PROTOCOLS)
    4 model-written       one batch from the guide model, shown the 8 and the
                          question groups, told to differ (program_guide.write_seeds)
    8 random              chosen from a pool of ~100 random programs: a short
                          sanity run drops the degenerate ones (stop at once, or
                          run to the cap), then a farthest-point pass over
                          structural distance from the 12 fixed seeds picks 8

Nothing from earlier experiments is used. The sanity run is the only use of
the debate model here; its recordings go to the search's cache file and are
reused by the warm-up.

    python scripts/program_seeds.py --out outputs/cluster_search/seeds.json

Offline (no debate model, no guide model; random programs are filtered by
structure only):

    python scripts/program_seeds.py --no-live --no-guide --out /tmp/seeds.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import program_guide as G  # noqa: E402
import program_space as P  # noqa: E402

ROOT = P.ROOT


def sanity_questions(groups: dict, n: int, rng: random.Random) -> list[str]:
    """`n` search questions spread over the groups, the same for every candidate."""
    pools = [list(g["search"]) for g in groups["groups"]]
    for p in pools:
        rng.shuffle(p)
    out: list[str] = []
    i = 0
    while len(out) < n and any(pools):
        p = pools[i % len(pools)]
        if p:
            out.append(p.pop())
        i += 1
    return out


def summarize(prog: dict) -> dict:
    moves = sorted({r["do"] for r in prog["rules"] if r["do"] != "continue" and not r["do"].startswith("stop:")})
    reads = sorted({r["do"][5:] for r in prog["rules"] if r["do"].startswith("stop:")} | {prog["default"][5:]})
    return {"plan": [P.plan_round_name(s) for s in prog["plan"]],
            "n_rules": len(prog["rules"]), "moves": moves, "stop_reads": reads}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clusters", type=Path, default=ROOT / "outputs/describe_v3/clusters_600_train.json")
    ap.add_argument("--dataset", type=Path, default=ROOT / "datasets/supergpqa_600_train.json")
    ap.add_argument("--per-group", type=int, default=50)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/cluster_search/seeds.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-llm", type=int, default=4)
    ap.add_argument("--n-random", type=int, default=8)
    ap.add_argument("--random-pool", type=int, default=100)
    ap.add_argument("--sanity-n", type=int, default=10, help="questions per candidate in the sanity run")
    ap.add_argument("--min-turns", type=float, default=2.0)
    ap.add_argument("--max-turns", type=float, default=12.0)
    ap.add_argument("--max-calls-per-question", type=int, default=16)
    ap.add_argument("--workers", type=int, default=64,
                    help="debates in flight at once; each may have up to 4 speakers in flight")
    ap.add_argument("--no-guide", action="store_true", help="skip the model-written seeds")
    ap.add_argument("--guide-effort", default=G.GUIDE_EFFORT)
    ap.add_argument("--no-live", action="store_true",
                    help="no debate model: skip the sanity run (structure-only pick)")
    live = ap.add_argument_group("debate model")
    live.add_argument("--base-urls", default=P.DEFAULT_BASE_URLS)
    live.add_argument("--model", default=P.DEFAULT_MODEL)
    live.add_argument("--api-key", default="EMPTY")
    live.add_argument("--temperature", type=float, default=0.7)
    live.add_argument("--live-cache", type=Path, default=None,
                      help="round cache (default outputs/cluster_search/rounds_<model>.jsonl)")
    live.add_argument("--ignore-cache-lock", action="store_true")
    ap.add_argument("--digest-head", type=int, default=P.DIGEST_HEAD)
    ap.add_argument("--digest-tail", type=int, default=P.DIGEST_TAIL)
    ap.add_argument("--visible-reasoning", action="store_true",
                    help="for models that think in a hidden channel (gpt-oss): ask every speaker to "
                         "write its reasoning in the visible reply; changes prompts and cache keys")
    args = ap.parse_args()

    settings = P.configure_executor(args.digest_head, args.digest_tail, args.visible_reasoning)
    settings["model"] = args.model
    rng = random.Random(args.seed)
    groups = P.load_groups(args.clusters, args.per_group)
    rows = {r["id"]: r for r in json.loads(args.dataset.read_text())}
    if args.live_cache is None:
        args.live_cache = args.out.parent / f"rounds_{P.model_tag(args.model)}.jsonl"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; the seed set is fixed once written (move it to redo)")

    seeds: list[dict] = []
    for name, prog in P.PROTOCOLS.items():
        P.validate_program(prog)
        seeds.append({"name": name, "source": "protocol", "program": P.normalize_program(prog)})
    print(f"{len(seeds)} protocol programs")

    # model-written seeds, cached beside the output so a rerun never re-asks
    if not args.no_guide and args.n_llm > 0:
        llm_path = args.out.with_suffix(".llm.json")
        if llm_path.exists():
            written = json.loads(llm_path.read_text())
            print(f"reusing {len(written)} model-written programs from {llm_path}")
        else:
            client = G.make_client()
            texts = [P.group_profile_text(g, P.difficulty_mix(rows, g["search"])) for g in groups["groups"]]
            written = G.write_seeds(client, {s["name"]: s["program"] for s in seeds}, texts,
                                    n_new=args.n_llm, effort=args.guide_effort)
            llm_path.write_text(json.dumps(written, indent=1))
            print(f"{len(written)} model-written programs -> {llm_path}")
        used = {s["name"] for s in seeds}
        for i, w in enumerate(written):
            P.validate_program(w["program"])
            name = w["name"] if w["name"] not in used else f"{w['name']}_{i}"   # names are lineages
            used.add(name)
            seeds.append({"name": name, "source": "llm", "program": w["program"],
                          "strategy": w.get("strategy", "")})

    # random pool
    pool: list[dict] = []
    seen = {P.canon(s["program"]) for s in seeds}
    while len(pool) < args.random_pool:
        prog = P.random_program(rng)
        key = P.canon(prog)
        if key in seen:
            continue
        seen.add(key)
        pool.append(prog)
    print(f"{len(pool)} random candidates")

    sanity: dict[int, dict] = {}
    if not args.no_live:
        qids = sanity_questions(groups, args.sanity_n, rng)
        runner = P.make_runner(rows, args.live_cache, args.base_urls, args.model, args.temperature,
                               api_key=args.api_key, lock=not args.ignore_cache_lock)
        runner.reset_budget(None)
        runner.stage(f"sanity run: {len(pool)} candidates x {len(qids)} questions")
        try:
            # every (candidate, question) debate in one pool, so the run is as
            # parallel as the server allows rather than one candidate at a time
            outs = P.run_pairs([(i, prog, q) for i, prog in enumerate(pool) for q in qids],
                               runner, rows, 0, args.max_calls_per_question, workers=args.workers,
                               desc="sanity")
        finally:
            runner.close()
        for i in range(len(pool)):
            done = [o for (j, q), o in outs.items() if j == i and o is not None]
            turns = sum(o["n_calls"] for o in done) / len(done) if done else 0.0
            acc = sum(o["correct"] for o in done) / len(done) if done else 0.0
            sanity[i] = {"mean_turns": turns, "acc": acc, "n": len(done)}
        keep = [i for i in range(len(pool))
                if sanity[i]["n"] and args.min_turns <= sanity[i]["mean_turns"] <= args.max_turns]
        print(f"sanity run on {len(qids)} questions: {len(keep)}/{len(pool)} candidates within "
              f"{args.min_turns}-{args.max_turns} mean turns")
    else:
        keep = list(range(len(pool)))
        print("no sanity run (--no-live): random programs filtered by structure only")

    fixed_tokens = [P.struct_tokens(s["program"]) for s in seeds]
    cand_tokens = [P.struct_tokens(pool[i]) for i in keep]
    picked = P.farthest_point(cand_tokens, args.n_random, P.struct_distance, fixed=fixed_tokens)
    for j, idx in enumerate(picked):
        i = keep[idx]
        seeds.append({"name": f"random_{j}", "source": "random", "program": pool[i],
                      "sanity": sanity.get(i)})
    print(f"{len(picked)} random programs picked by structural distance")

    for s in seeds:
        s["summary"] = summarize(s["program"])
    out = {"settings": settings, "clusters": groups["source"], "per_group": args.per_group,
           "seed": args.seed, "seeds": seeds}
    args.out.write_text(json.dumps(out, indent=1))
    report = [f"# Seed programs ({len(seeds)})", "",
              f"Executor: v2, digest {settings['digest']}, model {args.model}", "",
              "| name | source | plan | rules | moves | stop reads | sanity turns | sanity acc |",
              "|---|---|---|---|---|---|---|---|"]
    for s in seeds:
        sm = s["summary"]
        sa = s.get("sanity") or {}
        report.append(f"| {s['name']} | {s['source']} | {' > '.join(sm['plan'])} | {sm['n_rules']} | "
                      f"{', '.join(sm['moves']) or '-'} | {', '.join(sm['stop_reads'])} | "
                      f"{sa.get('mean_turns', float('nan')):.1f} | {sa.get('acc', float('nan')):.0%} |"
                      if sa else
                      f"| {s['name']} | {s['source']} | {' > '.join(sm['plan'])} | {sm['n_rules']} | "
                      f"{', '.join(sm['moves']) or '-'} | {', '.join(sm['stop_reads'])} | - | - |")
    report += ["", "## Programs", ""]
    for s in seeds:
        report.append(f"### {s['name']} ({s['source']})")
        if s.get("strategy"):
            report.append(s["strategy"])
        report.append("```json\n" + json.dumps(s["program"], indent=1) + "\n```")
        report.append("")
    args.out.with_suffix(".md").write_text("\n".join(report))
    print(f"wrote {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()

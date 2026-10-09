"""Step 7: the routed programs on the dev questions.

Two sets of per-group programs are compared, each routed the same way:

    held-out champions        the program picked per group on the held-out
                              training questions (champions.json)
    strongest grid programs   the slot-A holder of each group, picked on the
                              search questions alone (summary.json)

Both sets, the global champion and the literature baselines are run on EVERY
dev question, so any routing rule is scored from the saved marks without
another model call. The rows:

    routed, <set>             each question answered by the program of its group
                              (nearest medoid, from route_questions.py)
    ..., knn / question route the champions with the two comparison routes
    unsure -> global          the routed champion, except that the questions with
                              the smallest margin between their nearest two medoids
                              go to the global champion (lowest quarter, lowest half)
    random group, <set>       the mean over the set: what routing to a group drawn
                              at random would score
    global champion           one program for every question
    baselines                 direct, mad, self_refine (program_space.PROTOCOLS)
    best of <set> (bound)     per question, the best program of the set: the most
                              any router over it could reach (it also collects
                              luck, so read it as a loose upper bound)

With --routed-only (the pipeline's step 9 since 2026-10-09), each per-group program runs only on the
questions routed to its group, and the global program and the protocols on every question; a program
with several roles (the global program may also hold a group) runs each question once. The rows that
need a program's results on other groups' questions are then left out: the knn and question routes,
random group, best of (bound), and the other groups' column of the by-group table.

Replicates are run one after another: every program on every question at
replicate 1, then the tables (results_k1), then replicate 2 (results_k2), and
so on. Each table has avg@1 (first replicate only), avg@n, pass@n (any
replicate right) and speaker turns. Differences come with a paired standard
error over questions.

    python scripts/eval_routed_dev.py --run outputs/pipeline_cluster_gptoss/run3 \\
        --routes outputs/describe_v3/routes_600_test.json --model openai/gpt-oss-20b \\
        --base-urls http://localhost:7472/v1 --temperature 1.0 --visible-reasoning

The executor settings must equal those of the search (they are read from the
archive header and checked). Dev debates are recorded in their own round cache
in the output directory; running the script again costs nothing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evolve_pipeline_cluster as V3  # noqa: E402
import program_space as P  # noqa: E402
import schema_fitness as SF  # noqa: E402
from program_space import ProgRecord  # noqa: E402

ROOT = P.ROOT
log = V3.log


def mean_se(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    return m, (math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n) if n > 1 else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True,
                    help="search run directory (champions.json, summary.json and archive.jsonl are read; "
                         "only <run>/dev_eval is written)")
    ap.add_argument("--routes", type=Path, default=ROOT / "outputs/describe_v3/routes_600_test.json")
    ap.add_argument("--dataset", type=Path, default=ROOT / "datasets/supergpqa_600_test.json")
    ap.add_argument("--out", type=Path, default=None, help="default: <run>/dev_eval")
    ap.add_argument("--reps", type=int, default=3,
                    help="replicates, run one after another with the tables written after each")
    ap.add_argument("--baselines", default="direct,mad,self_refine")
    ap.add_argument("--max-calls-per-question", type=int, default=None,
                    help="default: the turn cap (--turn-cap, 16 unless set)")
    ap.add_argument("--no-live", action="store_true", help="cache only: score what is already recorded")
    ap.add_argument("--external", default="",
                    help="independently run baselines to show beside ours, as label=path pairs separated by "
                         "commas; each path is a score file written by baselines/score.py --save (its "
                         "per-question predictions are read). These are the baselines to report: they run "
                         "with the model's full thinking budget, outside this executor.")
    ap.add_argument("--no-champions", action="store_true",
                    help="champions.json is not read (the champion step was skipped): only the strongest grid "
                         "programs are routed, and the one-program-for-everyone row is the program with the best "
                         "score over all search questions (summary.json, top_overall)")
    ap.add_argument("--routed-only", action="store_true",
                    help="each per-group program runs only on the questions routed to its group (the global "
                         "program and the protocols on every question); the rows that need a program on other "
                         "groups' questions are left out. Off by default: every program on every question")
    live = ap.add_argument_group("debate model")
    live.add_argument("--base-urls", default=P.DEFAULT_BASE_URLS)
    live.add_argument("--model", default=P.DEFAULT_MODEL)
    live.add_argument("--api-key", default="EMPTY")
    live.add_argument("--temperature", type=float, default=0.7)
    live.add_argument("--workers", type=int, default=64)
    live.add_argument("--live-cache", type=Path, default=None,
                      help="round cache for the dev debates (default <out>/rounds_<model>.jsonl)")
    live.add_argument("--max-total-calls", type=int, default=200_000)
    live.add_argument("--ignore-cache-lock", action="store_true")
    P.add_executor_args(ap)
    args = ap.parse_args()

    with (args.run / "archive.jsonl").open() as f:
        header = json.loads(f.readline())
    # with no server (--no-live), the search's window stands in for the server's
    settings = P.configure_from_args(args, fallback_window=(header.get("settings") or {}).get("window"))
    settings["model"] = args.model
    if header.get("settings") != settings:
        raise SystemExit(f"the search ran with {header.get('settings')}, this run asks for {settings}")

    champs = None if args.no_champions else json.loads((args.run / "champions.json").read_text())
    routed = json.loads(args.routes.read_text())
    if Path(routed["clusters"]).resolve() != Path(header["clusters"]).resolve():
        raise SystemExit(f"the routes were made for {routed['clusters']}, the search used {header['clusters']}")
    routes = routed["routes"]
    rows = {r["id"]: r for r in json.loads(args.dataset.read_text())}
    P.check_rows(rows.values())             # the dataset suits the answer mode
    qids = [q for q in rows if q in routes]
    if len(qids) < len(rows):
        log(f"note: {len(rows) - len(qids)} dev questions have no route (no description) and are left out")
    if set(qids) & set(header["qids"]):
        raise SystemExit("some dev questions were search questions")

    # the programs, one record per distinct program however many roles it has
    records: dict[str, ProgRecord] = {}
    role: dict[str, str] = {}                      # role name -> program key

    def enrol(name: str, program: dict) -> None:
        prog = P.normalize_program(program)
        P.validate_program(prog)
        key = P.canon(prog)
        records.setdefault(key, ProgRecord(prog, name, name, 0))
        role[name] = key

    # the strongest grid program of each group: its slot-A holder, as the search left it
    summary = json.loads((args.run / "summary.json").read_text())
    slot_a = {int(x["group"]): x for x in summary["slots"] if x["slot"] == "A"}
    groups = sorted(slot_a) if champs is None else sorted(int(g) for g in champs["per_group"])
    if sorted(slot_a) != groups:
        raise SystemExit(f"{args.run / 'summary.json'} names slot-A holders for {sorted(slot_a)}, not {groups}")
    # the sets of per-group programs that are routed; the first is the main one
    sets: list[tuple[str, str]] = []
    if champs is not None:
        sets.append(("champion", "held-out champions"))
        for g in groups:
            d = champs["per_group"][str(g)]
            enrol(f"champion_{g}", next(p["program"] for p in d["programs"] if p["key"] == d["champion"]))
    sets.append(("grid", "strongest grid programs"))
    for g in groups:
        enrol(f"grid_{g}", json.loads(slot_a[g]["key"]))
    if champs is not None:
        global_label = "global champion"
        enrol("global", next(p["program"] for p in champs["global"]["programs"]
                             if p["key"] == champs["global"]["champion"]))
    else:
        # the global slot's holder (the strongest program over all search questions), if the
        # search had one; otherwise the best overall program
        g_slot = [x for x in summary["slots"] if x["slot"] == "G"]
        global_label = "global slot holder" if g_slot else "best overall program on the search questions"
        enrol("global", g_slot[0]["program"] if g_slot else summary["top_overall"][0]["program"])
    baselines = [b for b in args.baselines.split(",") if b]
    for b in baselines:
        enrol(b, P.PROTOCOLS[b])
    # independently run baselines: per question, the list of right/wrong marks of their runs,
    # and their completion tokens per run when the score file has them (score.py saves them now)
    external: dict[str, dict[str, list[int]]] = {}
    external_tokens: dict[str, dict[str, list[int]]] = {}
    for pair in [x for x in args.external.split(",") if x]:
        label, path = pair.split("=", 1)
        per_q = json.loads(Path(path).read_text())["per_question"]
        lacking = [q for q in qids if q not in per_q]
        if lacking:
            raise SystemExit(f"--external {label}: {len(lacking)} dev questions are not in {path}")
        # right/wrong per run: the scorer's marks (a judge's verdicts for open answers), or letter == key
        external[label] = {q: (list(per_q[q]["marks"]) if "marks" in per_q[q]
                               else [int(p == per_q[q]["answer"]) for p in per_q[q]["preds"]]) for q in qids}
        if all("tokens" in per_q[q] for q in qids):
            external_tokens[label] = {q: per_q[q]["tokens"] for q in qids}
    # the questions each role runs on: every question, or with --routed-only a per-group program's own
    # group's; a program with several roles runs the union of theirs, each question once
    set_prefixes = {prefix for prefix, _ in sets}

    def role_qids(name: str) -> list[str]:
        prefix, _, g = name.rpartition("_")
        if args.routed_only and prefix in set_prefixes:
            return [q for q in qids if routes[q]["group"] == int(g)]
        return qids
    role_q = {name: role_qids(name) for name in role}
    need: dict[str, set[str]] = {}
    for name, key in role.items():
        need.setdefault(key, set()).update(role_q[name])
    need_q = {key: [q for q in qids if q in s] for key, s in need.items()}
    log(f"{len(qids)} questions of {args.dataset.name}; {len(role)} roles, {len(records)} distinct programs; "
        f"{sum(len(v) for v in need_q.values())} debates per replicate"
        f"{' (routed only)' if args.routed_only else ''}; {args.reps} replicates; settings {settings}")

    out_dir = args.out or args.run / "dev_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.live_cache is None:
        args.live_cache = out_dir / f"rounds_{P.model_tag(args.model)}.jsonl"
    runner = P.make_runner(rows, args.live_cache, args.base_urls, args.model, args.temperature,
                           max_total_calls=args.max_total_calls, api_key=args.api_key,
                           lock=not args.ignore_cache_lock)
    shim = SimpleNamespace(runner=runner, rows=rows, args=args)     # what Search.run_jobs/replay read
    ctx = SimpleNamespace(args=args, settings=settings, records=records, role=role, role_q=role_q, routes=routes,
                          qids=qids, groups=groups, baselines=baselines, out_dir=out_dir,
                          sets=sets, global_label=global_label, external=external,
                          external_tokens=external_tokens)
    # one replicate at a time, every program on every question, and the tables
    # after each: the first-replicate numbers are out after a third of the work
    try:
        left = 0
        for rep in range(args.reps):
            for rec in records.values():
                V3.Search.replay(shim, rec, need_q[rec.key], rep)
            if not args.no_live:
                spent = V3.Search.run_jobs(shim, [(rec, q, rep) for rec in records.values() for q in need_q[rec.key]],
                                           f"dev, replicate {rep + 1} of {args.reps}")
                log(f"replicate {rep + 1}: {spent} speaker turns spent")
            left = write_report(ctx, list(range(rep + 1)))
        if left:                                  # a table with debates missing is not a result: stop nonzero
            raise SystemExit(f"{left} test debates are missing (server errors?); they are counted as wrong in "
                             f"the tables. Run the script again: it fills them from where it stopped.")
    except SF.BudgetExhausted as exc:              # the tables lack debates: not a result, stop nonzero
        log(f"stopped: {exc}")
        raise SystemExit(f"the call cap stopped the evaluation ({exc}); what was recorded is kept, run again "
                         f"(with a higher --max-total-calls) to finish")
    except KeyboardInterrupt:
        log("interrupted; what was recorded is kept, run again to finish")
        raise SystemExit(1)
    finally:
        if P.JUDGE is not None:
            log(f"judge {P.JUDGE.model}: {P.JUDGE.stats}")
        runner.close()


def write_report(ctx: SimpleNamespace, reps: list[int]) -> int:
    """The tables over the replicates in `reps` -> results_k<n>.json / .md. Returns the number of
    debates missing (they are counted as wrong, and the .md says so at its top)."""
    args, records, role, routes, qids, groups, baselines = (ctx.args, ctx.records, ctx.role, ctx.routes,
                                                            ctx.qids, ctx.groups, ctx.baselines)
    n = len(reps)
    routed_only, role_q = args.routed_only, ctx.role_q
    missing = {name: sum(len(records[k].gaps(role_q[name], rep)) for rep in reps) for name, k in role.items()}
    if any(missing.values()):
        log(f"debates missing (counted as wrong): { {a: m for a, m in missing.items() if m} }")

    def marks(name: str, q: str) -> list[int]:
        rec = records[role[name]]
        return [rec.reps.get(rep, {}).get(q, [0])[0] for rep in reps]

    def turns(name: str, q: str) -> float:
        rec = records[role[name]]
        vals = [rec.reps[rep][q][1] for rep in reps if q in rec.reps.get(rep, {})]
        return sum(vals) / len(vals) if vals else 0.0

    def tokens(name: str, q: str) -> float | None:
        """Mean completion tokens of the debate over the replicates (None if not logged)."""
        rec = records[role[name]]
        vals = [rec.reps[rep][q][5] for rep in reps
                if q in rec.reps.get(rep, {}) and len(rec.reps[rep][q]) > 5 and rec.reps[rep][q][5] is not None]
        return sum(vals) / len(vals) if vals else None

    # a row is, per question, a list of (weight, program) choices; one choice for
    # most rows, all group programs at equal weight for the random-group rows
    def routed_by(prefix: str, field: str, fallback: set[str] = frozenset()) -> list[list[tuple[float, str]]]:
        return [[(1.0, "global" if q in fallback else f"{prefix}_{routes[q][field]}")] for q in qids]

    def spread(prefix: str) -> list[list[tuple[float, str]]]:
        return [[(1.0 / len(groups), f"{prefix}_{g}") for g in groups] for _ in qids]

    sets, global_label = ctx.sets, ctx.global_label
    main_prefix, main_label = sets[0]                    # the knn / question / unsure rows use this set
    by_margin = sorted(qids, key=lambda q: routes[q]["margin"])
    rows_: list[tuple[str, list]] = [(f"routed, {label}", routed_by(prefix, "group")) for prefix, label in sets]
    if not routed_only:                                  # these use other groups' programs
        rows_ += [(f"routed, {main_label}, knn route", routed_by(main_prefix, "knn")),
                  (f"routed, {main_label}, question route", routed_by(main_prefix, "question"))]
    for frac, label in ((0.25, "quarter"), (0.5, "half")):
        rows_.append((f"unsure {label} -> global",
                      routed_by(main_prefix, "group", set(by_margin[: int(frac * len(qids))]))))
    if not routed_only:
        rows_ += [(f"random group, {label} (expected)", spread(prefix)) for prefix, label in sets]
    rows_ += [(global_label, [[(1.0, "global")] for _ in qids])]
    rows_ += [(f"in-executor {b}", [[(1.0, b)] for _ in qids]) for b in baselines]

    table, per_q = {}, {}
    for name, choice in rows_:
        first = [sum(w * marks(p, q)[0] for w, p in c) for c, q in zip(choice, qids)]
        avg = [sum(w * sum(marks(p, q)) / n for w, p in c) for c, q in zip(choice, qids)]
        anyc = [sum(w * max(marks(p, q)) for w, p in c) for c, q in zip(choice, qids)]
        t = [sum(w * turns(p, q) for w, p in c) for c, q in zip(choice, qids)]
        tok = [[tokens(p, q) for _, p in c] for c, q in zip(choice, qids)]
        tok_mean = (sum(sum(w * x for (w, _), x in zip(c, tq)) for c, tq in zip(choice, tok)) / len(qids)
                    if all(x is not None for tq in tok for x in tq) else None)
        per_q[name] = avg
        table[name] = {"avg@1": mean_se(first)[0], f"avg@{n}": mean_se(avg)[0], "se": mean_se(avg)[1],
                       f"pass@{n}": mean_se(anyc)[0], "turns": sum(t) / len(t), "tokens": tok_mean}
    # independently run baselines: their first n runs (fewer if they have fewer)
    for label, per in ctx.external.items():
        runs = [per[q][:n] for q in qids]
        avg = [sum(r) / len(r) for r in runs]
        per_q[f"external {label}"] = avg
        et = ctx.external_tokens.get(label)
        tok_mean = (sum(sum(et[q][:n]) / len(et[q][:n]) for q in qids) / len(qids)) if et else None
        table[f"external {label}"] = {"avg@1": mean_se([r[0] for r in runs])[0], f"avg@{n}": mean_se(avg)[0],
                                      "se": mean_se(avg)[1], f"pass@{n}": mean_se([max(r) for r in runs])[0],
                                      "turns": float("nan"), "tokens": tok_mean, "runs": min(len(r) for r in runs)}
    for prefix, label in ([] if routed_only else sets):  # needs every group's program on every question
        avg = [max(sum(marks(f"{prefix}_{g}", q)) / n for g in groups) for q in qids]
        per_q[f"best of {label} (bound)"] = avg
        table[f"best of {label} (bound)"] = {
            "avg@1": mean_se([max(marks(f"{prefix}_{g}", q)[0] for g in groups) for q in qids])[0],
            f"avg@{n}": mean_se(avg)[0], "se": mean_se(avg)[1],
            f"pass@{n}": mean_se([max(max(marks(f"{prefix}_{g}", q)) for g in groups) for q in qids])[0],
            "turns": sum(sum(turns(f"{prefix}_{g}", q) for g in groups) for q in qids) / len(qids),
            "tokens": None}

    report = [f"# Results on {args.dataset.name}: {args.model}, {len(qids)} questions, replicates 1..{n} of {args.reps}", "",
              f"avg@1 uses the first replicate only. avg@{n} is the mean over questions of each question's "
              f"mean mark over {n} replicate(s); ± is its standard error over questions. pass@{n} counts a "
              f"question as right if any replicate was. Turns are speaker turns per question (for a bound: "
              f"the cost of running every program of the set); tokens are completion tokens per question "
              + ("(every call of the debate, except a summary that no later speaker read; '-' where they were "
                 "not logged)." if ctx.settings.get("count_read_summaries") else
                 "(every call of the debate, summaries included; '-' where they were not logged)."), "",
              ("'Held-out champions' are the programs picked per group on the held-out training questions; "
               if len(sets) > 1 else "The champion step was skipped for this run, so no held-out champions. ")
              + "'Strongest grid programs' are the slot-A holders, picked on the search questions alone. "
              + f"'{global_label}' is one program used for every question. 'External' rows are the "
              "independently run baselines (full thinking budget, their own scorer): these are the baselines "
              "to report. Where given, a 'no recovery' row is the method exactly as published; the row without "
              "that label asks a reply cut off at its token limit, once, for the letter its reasoning supports "
              "(as this executor does). 'In-executor' rows are the same protocols run as debate programs under "
              "this executor's settings.", "",
              "| row | avg@1 | " + (f"avg@{n} | " if n > 1 else "") + f"± | pass@{n} | turns | tokens |",
              "|---|---|---|---|---|---|" + ("---|" if n > 1 else "")]
    for name, d in table.items():
        report.append(f"| {name} | {d['avg@1']:.1%} | " + (f"{d[f'avg@{n}']:.1%} | " if n > 1 else "")
                      + f"{d['se']:.1%} | {d[f'pass@{n}']:.1%} | "
                      + ("-" if d["turns"] != d["turns"] else f"{d['turns']:.1f}") + " | "
                      + ("-" if d.get("tokens") is None else f"{d['tokens']:,.0f}") + " |")

    # the baseline to beat: the best independently run one if any were given, else the best in-executor one
    pool_b = [f"external {x}" for x in ctx.external] or [f"in-executor {b}" for b in baselines]
    best_b = max(pool_b, key=lambda k: table[k][f"avg@{n}"]) if pool_b else None
    # and the best protocol run as a program in this same executor (the seeds include them)
    pool_in = [f"in-executor {b}" for b in baselines]
    best_in = max(pool_in, key=lambda k: table[k][f"avg@{n}"]) if pool_in else None
    beat = [(x, why) for x, why in
            ((best_b, "is the search worth anything (best baseline, picked on these questions)"),
             (best_in, "does the search beat the protocols in its own executor (best one, picked on these "
                       "questions)")) if x]
    beat = [(x, why) for i, (x, why) in enumerate(beat) if x not in [y for y, _ in beat[:i]]]
    diffs = {}
    report += ["", f"Differences in avg@{n}, with a paired standard error over questions:", "",
               "| row | minus | difference | paired ± | question |", "|---|---|---|---|---|"]
    for main, rand in [(f"routed, {label}", f"random group, {label} (expected)") for _, label in sets]:
        against = ([] if routed_only else [(rand, "is the routing itself worth anything")]) + \
                  [(global_label, "are per-group programs worth anything")] + beat
        for other, why in against:
            diff, se = V3.paired_se(per_q[main], per_q[other])
            diffs[f"{main} - {other}"] = {"diff": diff, "se": se}
            report.append(f"| {main} | {other} | {diff:+.1%} | {se:.1%} | {why} |")
    for other, why in beat:                              # the one program used for every question
        diff, se = V3.paired_se(per_q[global_label], per_q[other])
        diffs[f"{global_label} - {other}"] = {"diff": diff, "se": se}
        report.append(f"| {global_label} | {other} | {diff:+.1%} | {se:.1%} | {why} |")
    if len(sets) > 1:
        diff, se = V3.paired_se(per_q["routed, held-out champions"], per_q["routed, strongest grid programs"])
        diffs["held-out champions - strongest grid programs"] = {"diff": diff, "se": se}
        report.append(f"| routed, held-out champions | routed, strongest grid programs | {diff:+.1%} | {se:.1%} | "
                      f"does picking on held-out questions help |")

    # the per-question turn cap: a debate it stopped ran a shorter program than the one on record
    cut = {name: sum(P.cap_cuts(records[k], role_q[name], rep) for rep in reps) for name, k in role.items()}
    report += ["", f"Debates the turn cap stopped (the cap replaced a rule's round with the program's default "
                   f"stop), of " + ("the debates each program ran (its own group's questions, every question for "
                                   "the global program)" if routed_only else f"{len(qids) * n} per program") + ":", "",
               "| program | debates stopped by the cap |", "|---|---|"]
    report += [f"| {name} | {c} |" for name, c in cut.items()]

    report += ["", f"avg@{n} by the group the question was routed to (nearest medoid):", "",
               "| group | questions | " + " | ".join(f"its own, {label}" + ("" if routed_only else
                                                                           f" | other groups', {label} (mean)")
                                                      for _, label in sets)
               + " | global" + "".join(f" | {b}" for b in baselines) + " |",
               "|---|---|" + ("---|" if routed_only else "---|---|") * len(sets) + "---|" + "---|" * len(baselines)]
    per_group = {}

    def acc(name: str, qs: list[str]) -> float:
        return sum(sum(marks(name, q)) / n for q in qs) / len(qs)

    for g in groups:
        qs = [q for q in qids if routes[q]["group"] == g]
        if not qs:
            continue
        others = [h for h in groups if h != g]
        d = {"n": len(qs), "global": acc("global", qs), "baselines": {b: acc(b, qs) for b in baselines}}
        for prefix, _ in sets:
            d[prefix] = acc(f"{prefix}_{g}", qs)
            if not routed_only:                          # other groups' programs did not run on these
                d[f"other_{prefix}"] = sum(acc(f"{prefix}_{h}", qs) for h in others) / max(len(others), 1)
        per_group[str(g)] = d
        report.append(f"| {g} | {len(qs)} | " + " | ".join(f"{d[prefix]:.1%}" + ("" if routed_only else
                                                                                   f" | {d['other_' + prefix]:.1%}")
                                                          for prefix, _ in sets)
                      + f" | {d['global']:.1%}" + "".join(f" | {d['baselines'][b]:.1%}" for b in baselines) + " |")

    result = {"model": args.model, "settings": ctx.settings, "run": str(args.run), "routes": str(args.routes),
              "n_questions": len(qids), "reps": n, "reps_planned": args.reps, "table": table,
              "differences": diffs, "per_group": per_group, "missing": missing, "cut_by_cap": cut, "roles": role,
              "programs": {k: {"name": r.name, "program": r.program,
                               "reps": {str(rep): r.reps.get(rep, {}) for rep in reps}}
                           for k, r in records.items()}}
    if any(missing.values()):                     # shown in the table, not only in the log
        report[1:1] = ["", f"**INCOMPLETE: debates missing (counted as wrong above): "
                           f"{ {a: m for a, m in missing.items() if m} }. Run the script again to fill them.**"]
    (ctx.out_dir / f"results_k{n}.json").write_text(json.dumps(result, indent=1))
    (ctx.out_dir / f"results_k{n}.md").write_text("\n".join(report) + "\n")
    log("\n".join(report))
    log(f"\n-> {ctx.out_dir / f'results_k{n}.json'}, {ctx.out_dir / f'results_k{n}.md'}\n")
    return sum(missing.values())

if __name__ == "__main__":
    main()

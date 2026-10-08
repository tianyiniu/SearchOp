"""Test evaluation of the global pipeline: the final program of each cost level (final.json,
chosen on the dev split) and the baselines, on the test split.

The rows:

    v4 <level>              the level's final program (evolve_pipeline_global.py --select-dev)
    in-executor <protocol>  literature programs run as debate programs under the same
                            executor settings (default direct_high and self_refine_high)
    external <label>        independently run baselines (baselines/score.py files, read
                            with --external label=path,...), the ones to report

Replicates run one after another (every program on every question at replicate 1, then
the tables, then replicate 2, ...). Each table has avg@1, avg@n (the mean over questions
of each question's mean mark), its standard error, pass@n, turns and completion tokens
per question. Each global-pipeline program is compared with self-refine (paired by question). For each
global-pipeline program the report also lists the paths its debates took (the sequence of actions its
rules chose), with their share of the test debates and their accuracy: these are the
groups the search found. accuracy_vs_tokens.svg plots accuracy against tokens.

    python scripts/eval_pipeline_global.py --run outputs/pipeline_global_gptoss/run1 \\
        --dataset datasets/supergpqa_600_test.json --reps 3 --external self-refine=... [executor options]

The executor settings must equal the search's (read from the archive header, checked). The
test debates are recorded in their own round cache; running the script again costs nothing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evolve_pipeline_cluster as V3  # noqa: E402
import evolve_pipeline_global as V4  # noqa: E402
import program_space as P  # noqa: E402
import schema_fitness as SF  # noqa: E402
from program_space import ProgRecord  # noqa: E402

ROOT = P.ROOT
log = V3.log
SELF_REFINE_ROWS = ("external self-refine", "in-executor self_refine_high")   # the comparison rows, if present


def mean_se(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    return m, (math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n) if n > 1 else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="global-pipeline run directory (archive.jsonl, final.json)")
    ap.add_argument("--dataset", type=Path, default=ROOT / "datasets/supergpqa_600_test.json")
    ap.add_argument("--out", type=Path, default=None, help="default: <run>/test_eval")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--baselines", default="direct_high,self_refine_high",
                    help="literature programs run in this executor beside the global-pipeline programs")
    ap.add_argument("--max-calls-per-question", type=int, default=None,
                    help="default: the turn cap (--turn-cap, 16 unless set)")
    ap.add_argument("--no-live", action="store_true", help="cache only: score what is already recorded")
    ap.add_argument("--external", default="",
                    help="independently run baselines as label=path pairs separated by commas; each path is a "
                         "score file written by baselines/score.py --save")
    live = ap.add_argument_group("debate model")
    live.add_argument("--base-urls", default=P.DEFAULT_BASE_URLS)
    live.add_argument("--model", default=P.DEFAULT_MODEL)
    live.add_argument("--api-key", default="EMPTY")
    live.add_argument("--temperature", type=float, default=0.7)
    live.add_argument("--workers", type=int, default=64)
    live.add_argument("--live-cache", type=Path, default=None,
                      help="round cache for the test debates (default <out>/rounds_<model>.jsonl)")
    live.add_argument("--max-total-calls", type=int, default=200_000)
    live.add_argument("--ignore-cache-lock", action="store_true")
    P.add_executor_args(ap)
    args = ap.parse_args()

    with (args.run / "archive.jsonl").open() as f:
        header = json.loads(f.readline())
    if header.get("version") != 4:
        raise SystemExit(f"{args.run} is not a global search (archive version {header.get('version')})")
    # with no server (--no-live), the search's window stands in for the server's
    settings = P.configure_from_args(args, fallback_window=(header.get("settings") or {}).get("window"))
    settings["model"] = args.model
    if header.get("settings") != settings:
        raise SystemExit(f"the search ran with {header.get('settings')}, this run asks for {settings}")
    final = json.loads((args.run / "final.json").read_text())
    if final.get("settings") != settings:
        raise SystemExit(f"final.json was chosen under {final.get('settings')}, this run asks for {settings}")
    rows = {r["id"]: r for r in json.loads(args.dataset.read_text())}
    P.check_rows(rows.values())             # the dataset suits the answer mode
    qids = list(rows)
    if set(qids) & (set(header["qids"]) | set(header["dev"])):
        raise SystemExit("some test questions are train or dev questions")

    records: dict[str, ProgRecord] = {}
    role: dict[str, str] = {}                      # row name -> program key

    def enrol(name: str, program: dict) -> None:
        prog = P.normalize_program(program)
        P.validate_program(prog)
        key = P.canon(prog)
        records.setdefault(key, ProgRecord(prog, name, name, 0))
        role[name] = key

    finals, final_names = [], {}
    for level, _ in V4.LEVELS:
        d = final["levels"].get(level) or {}
        if d.get("chosen") is None:
            log(f"note: no final program for the {level} level")
            continue
        enrol(f"v4 {level}", next(c["program"] for c in d["candidates"] if c["key"] == d["chosen"]))
        finals.append(f"v4 {level}")
        final_names[f"v4 {level}"] = d["chosen_name"]
    if not finals:
        raise SystemExit(f"{args.run / 'final.json'} names no final program")
    baselines = [b for b in args.baselines.split(",") if b]
    for b in baselines:
        if b not in P.PROTOCOLS:
            raise SystemExit(f"--baselines: {b!r} is not one of {list(P.PROTOCOLS)}")
        enrol(f"in-executor {b}", P.PROTOCOLS[b])
    external: dict[str, dict[str, list[int]]] = {}
    external_tokens: dict[str, dict[str, list[int]]] = {}
    for pair in [x for x in args.external.split(",") if x]:
        if "=" not in pair:
            raise SystemExit(f"--external: {pair!r} is not label=path")
        label, path = pair.split("=", 1)
        per_q = json.loads(Path(path).read_text())["per_question"]
        lacking = [q for q in qids if q not in per_q]
        if lacking:
            raise SystemExit(f"--external {label}: {len(lacking)} test questions are not in {path}")
        external[label] = {q: (list(per_q[q]["marks"]) if "marks" in per_q[q]
                               else [int(p == per_q[q]["answer"]) for p in per_q[q]["preds"]]) for q in qids}
        if all("tokens" in per_q[q] for q in qids):
            external_tokens[label] = {q: per_q[q]["tokens"] for q in qids}
    log(f"{len(qids)} questions of {args.dataset.name}; {len(role)} rows, {len(records)} distinct programs; "
        f"{args.reps} replicates; settings {settings}")

    out_dir = args.out or args.run / "test_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.live_cache is None:
        args.live_cache = out_dir / f"rounds_{P.model_tag(args.model)}.jsonl"
    runner = P.make_runner(rows, args.live_cache, args.base_urls, args.model, args.temperature,
                           max_total_calls=args.max_total_calls, api_key=args.api_key,
                           lock=not args.ignore_cache_lock)
    # what Search.run_jobs/replay read; missing debates are flagged in the tables and end the run with an
    # error (below), as in the global search, so run_jobs does not wait for the server here
    shim = SimpleNamespace(runner=runner, rows=rows, args=args, WAIT_FOR_SERVER=False)
    ctx = SimpleNamespace(args=args, settings=settings, records=records, role=role, qids=qids, finals=finals,
                          baselines=baselines, out_dir=out_dir, external=external, external_tokens=external_tokens,
                          final_names=final_names)
    missing = None
    try:
        for rep in range(args.reps):
            for rec in records.values():
                V3.Search.replay(shim, rec, qids, rep)
            if not args.no_live:
                spent = V3.Search.run_jobs(shim, [(rec, q, rep) for rec in records.values() for q in qids],
                                           f"test, replicate {rep + 1} of {args.reps}")
                log(f"replicate {rep + 1}: {spent} new model calls")
            missing = write_report(ctx, list(range(rep + 1)))
    except SF.BudgetExhausted as exc:
        raise SystemExit(f"stopped: {exc}; what was recorded is kept, run again to finish")
    except KeyboardInterrupt:
        log("interrupted; what was recorded is kept, run again to finish")
        raise SystemExit(1)
    finally:
        runner.close()
    if missing:
        raise SystemExit(f"{sum(missing.values())} test debates are missing (server errors, counted as wrong in "
                         f"the tables); run again to finish them")


def write_report(ctx: SimpleNamespace, reps: list[int]) -> dict[str, int]:
    """The tables over the replicates in `reps` -> results_k<n>.json / .md and the graph.
    Returns the debates missing per row (counted as wrong)."""
    records, role, qids = ctx.records, ctx.role, ctx.qids
    n = len(reps)
    missing = {name: sum(len(records[k].gaps(qids, rep)) for rep in reps) for name, k in role.items()}
    if any(missing.values()):
        log(f"debates missing (counted as wrong): { {a: m for a, m in missing.items() if m} }")

    def results(name: str, q: str) -> list[list]:
        rec = records[role[name]]
        return [rec.reps[rep][q] for rep in reps if q in rec.reps.get(rep, {})]

    def marks(name: str, q: str) -> list[int]:
        rec = records[role[name]]
        return [rec.reps.get(rep, {}).get(q, [0])[0] for rep in reps]

    table: dict[str, dict] = {}
    per_q: dict[str, list[float]] = {}
    for name in role:
        avg = [sum(marks(name, q)) / n for q in qids]
        turns = [sum(v[1] for v in results(name, q)) / max(len(results(name, q)), 1) for q in qids]
        toks = [[v[5] for v in results(name, q) if len(v) > 5 and v[5] is not None] for q in qids]
        tok_mean = (sum(sum(t) / len(t) for t in toks) / len(qids)) if all(toks) else None
        per_q[name] = avg
        table[name] = {"avg@1": mean_se([marks(name, q)[0] for q in qids])[0], f"avg@{n}": mean_se(avg)[0],
                       "se": mean_se(avg)[1], f"pass@{n}": mean_se([max(marks(name, q)) for q in qids])[0],
                       "turns": sum(turns) / len(turns), "tokens": tok_mean}
    for label, per in ctx.external.items():
        runs = [per[q][:n] for q in qids]
        avg = [sum(r) / len(r) for r in runs]
        name = f"external {label}"
        per_q[name] = avg
        et = ctx.external_tokens.get(label)
        table[name] = {"avg@1": mean_se([r[0] for r in runs])[0], f"avg@{n}": mean_se(avg)[0],
                       "se": mean_se(avg)[1], f"pass@{n}": mean_se([max(r) for r in runs])[0],
                       "turns": None,
                       "tokens": (sum(sum(et[q][:n]) / len(et[q][:n]) for q in qids) / len(qids)) if et else None,
                       "runs": min(len(r) for r in runs)}

    report = [f"# Global pipeline results on {ctx.args.dataset.name}: {ctx.args.model}, {len(qids)} questions, "
              f"replicates 1..{n} of {ctx.args.reps}", "",
              f"avg@{n} is the mean over questions of each question's mean mark over {n} replicate(s); ± is its "
              f"standard error over questions. pass@{n} counts a question as right if any replicate was. Turns "
              f"are speaker turns per question; tokens are completion tokens per question (every call of the "
              f"debate, summaries included). 'v4 <level>' rows are the final programs, chosen on the dev split; "
              f"'external' rows are the independently run baselines (the ones to report); 'in-executor' rows are "
              f"literature programs run under this executor's settings.", ""]
    if any(missing.values()):
        report += [f"**Incomplete: {sum(missing.values())} debates are missing (server errors) and count as wrong: "
                   f"{ {a: m for a, m in missing.items() if m} }. Run the evaluation again to finish them.**", ""]
    for level, _ in V4.LEVELS:
        if f"v4 {level}" not in ctx.finals:
            report += [f"No program reached the {level} level, so it has no row.", ""]
    report += ["| row | avg@1 | " + (f"avg@{n} | " if n > 1 else "") + f"± | pass@{n} | turns | tokens |",
              "|---|---:|" + ("---:|" if n > 1 else "") + "---:|---:|---:|---:|"]
    for name, d in table.items():
        report.append(f"| {name} | {d['avg@1']:.1%} | " + (f"{d[f'avg@{n}']:.1%} | " if n > 1 else "")
                      + f"{d['se']:.1%} | {d[f'pass@{n}']:.1%} | "
                      + ("-" if d["turns"] is None else f"{d['turns']:.1f}") + " | "
                      + ("-" if d.get("tokens") is None else f"{d['tokens']:,.0f}") + " |")

    diffs = {}
    against = [x for x in SELF_REFINE_ROWS if x in table]
    if not against:
        log(f"note: no row named {' or '.join(SELF_REFINE_ROWS)}, so there is no table of differences")
    if against:
        report += ["", f"Differences in avg@{n} from self-refine, with a paired standard error over questions:", "",
                   "| row | minus | difference | paired ± |", "|---|---|---:|---:|"]
        for name in ctx.finals:
            for other in against:
                diff, se = V3.paired_se(per_q[name], per_q[other])
                diffs[f"{name} - {other}"] = {"diff": diff, "se": se}
                report.append(f"| {name} | {other} | {diff:+.1%} | {se:.1%} |")

    # the paths each final program's debates took: the groups its rules made
    paths: dict[str, list[dict]] = {}
    stops: dict[str, list[dict]] = {}
    report += ["", "## The paths of the global-pipeline programs", "",
               "A path is the sequence of rounds a program's rules chose on one test debate (question and "
               "replicate); the second table gives the rule that stopped the debate. Shares are of all the "
               "program's recorded test debates."]
    for name in ctx.finals:
        rec = records[role[name]]
        by: dict[str, list[list]] = defaultdict(list)
        for q in qids:
            for v in results(name, q):
                by[v[3]].append(v)
        total = sum(len(v) for v in by.values())
        rows = []
        for path, vs in sorted(by.items(), key=lambda kv: -len(kv[1])):
            toks = [v[5] for v in vs if len(v) > 5 and v[5] is not None]
            rows.append({"path": path, "debates": len(vs), "share": len(vs) / total if total else 0.0,
                         "acc": sum(v[0] for v in vs) / len(vs), "turns": sum(v[1] for v in vs) / len(vs),
                         "tokens": sum(toks) / len(toks) if toks else None})
        paths[name] = rows
        report += ["", f"### {name}: {ctx.final_names[name]}", "",
                   "```json", json.dumps(rec.program, indent=1), "```", "",
                   "| path | debates | share | accuracy | turns | tokens |", "|---|---:|---:|---:|---:|---:|"]
        for r in rows:
            report.append(f"| {r['path'] or '(none)'} | {r['debates']} | {r['share']:.1%} | {r['acc']:.1%} | "
                          f"{r['turns']:.1f} | " + ("-" if r["tokens"] is None else f"{r['tokens']:,.0f}") + " |")
        # the rule that stopped each debate (the last decision of its path)
        by_stop: dict[str, list[list]] = defaultdict(list)
        for q in qids:
            for v in results(name, q):
                by_stop[stop_label(rec.program, v[4] if len(v) > 4 else "")].append(v)
        stop_rows = []
        for label, vs in sorted(by_stop.items(), key=lambda kv: -len(kv[1])):
            stop_rows.append({"stop": label, "debates": len(vs), "share": len(vs) / total if total else 0.0,
                              "acc": sum(v[0] for v in vs) / len(vs), "turns": sum(v[1] for v in vs) / len(vs)})
        stops[name] = stop_rows
        report += ["", "| stopped by | debates | share | accuracy | turns |", "|---|---:|---:|---:|---:|"]
        for r in stop_rows:
            report.append(f"| {r['stop']} | {r['debates']} | {r['share']:.1%} | {r['acc']:.1%} | {r['turns']:.1f} |")

    # one dot per program: a final program that is also a baseline program shares its dot
    points: list[dict] = []
    by_key: dict[str, dict] = {}
    for name, d in table.items():
        if d.get("tokens") is None:
            continue
        key = role.get(name)
        if key is not None and key in by_key:
            by_key[key]["label"] += f" = {name}"
            continue
        pt = {"label": name, "x": d["tokens"], "y": 100 * d[f"avg@{n}"], "se": 100 * d["se"],
              "kind": "v4" if name in ctx.finals else "baseline"}
        points.append(pt)
        if key is not None:
            by_key[key] = pt
    if points:
        report += ["", "accuracy_vs_tokens.svg plots avg@n (± one standard error) against tokens per question."]

    result = {"model": ctx.args.model, "settings": ctx.settings, "run": str(ctx.args.run),
              "n_questions": len(qids), "reps": n, "reps_planned": ctx.args.reps, "table": table,
              "differences": diffs, "paths": paths, "stops": stops, "missing": missing, "roles": role,
              "final": dict(ctx.final_names),
              "programs": {k: {"name": r.name, "program": r.program,
                               "reps": {str(rep): r.reps.get(rep, {}) for rep in reps}}
                           for k, r in records.items()}}
    (ctx.out_dir / f"results_k{n}.json").write_text(json.dumps(result, indent=1))
    (ctx.out_dir / f"results_k{n}.md").write_text("\n".join(report) + "\n")
    if points:
        try:
            (ctx.out_dir / "accuracy_vs_tokens.svg").write_text(
                scatter_svg(points, f"Test accuracy against tokens ({len(qids)} questions, {n} replicate(s))"))
        except Exception as exc:                      # the tables are written; the graph is extra
            log(f"note: the graph could not be drawn ({type(exc).__name__}: {exc})")
    log("\n".join(report))
    log(f"\n-> {ctx.out_dir / f'results_k{n}.json'}, {ctx.out_dir / f'results_k{n}.md'}\n")
    return {a: m for a, m in missing.items() if m}


def stop_label(prog: dict, fired: str) -> str:
    """Which rule stopped a debate, from its recorded decisions ('0,0,2': the rule that decided
    each step, -1 = none held). A last decision whose rule does not stop means the plan was used
    up, the turn cap was reached or the step limit was reached, so the default stop read the answer."""
    idx = [int(i) for i in fired.split(",") if i.strip()] if fired else []
    if not idx:
        return "(no decision recorded)"
    k = idx[-1]
    if k < 0:
        return f"no rule held: {prog['default']}"
    rule = prog["rules"][k]
    when = " and ".join(rule["when"])
    if rule["do"].startswith("stop:"):
        return f"rule {k}: if {when}, {rule['do']}"
    return f"rule {k}: if {when}, {rule['do']} (plan used up, turn cap or step limit: {prog['default']})"


# --- the graph ------------------------------------------------------------------------------

def nice_step(span: float, target: int = 5) -> float:
    """A round tick step (1, 2 or 5 times a power of ten) giving about `target` ticks."""
    raw = max(span, 1e-9) / target
    mag = 10 ** math.floor(math.log10(raw))
    return next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw)


def scatter_svg(points: list[dict], title: str) -> str:
    """Accuracy (y, %) against tokens (x) as a standalone SVG: one dot per row with a
    +/- one-standard-error whisker, every dot labelled, global-pipeline programs (circles, series 1) and
    baselines (squares, series 2), a legend, light and dark themes, and a native tooltip
    on each dot. Colours are the reference palette's first two categorical slots."""
    W, H, L, R, T, B = 760, 460, 70, 190, 70, 56
    pw, ph = W - L - R, H - T - B
    xmax_raw = max(max(p["x"] for p in points) * 1.08, 1.0)
    xs = nice_step(xmax_raw)
    xmax = math.ceil(xmax_raw / xs) * xs
    ylo_raw = min(p["y"] - p["se"] for p in points)
    yhi_raw = max(p["y"] + p["se"] for p in points)
    ys = nice_step(max(yhi_raw - ylo_raw, 1.0))
    ylo, yhi = math.floor(ylo_raw / ys) * ys, math.ceil(yhi_raw / ys) * ys
    if yhi - ylo < ys:
        yhi = ylo + ys

    def px(x: float) -> float:
        return L + pw * x / xmax

    def py(y: float) -> float:
        return T + ph * (1 - (y - ylo) / (yhi - ylo))

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'class="viz-root" role="img" aria-label="{escape(title, {chr(34): "&quot;"})}">',
           "<style>",
           ".viz-root{--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;--grid:#e6e5e1;"
           "--series-1:#2a78d6;--series-2:#eb6834;font-family:system-ui,-apple-system,'Segoe UI',sans-serif}",
           "@media (prefers-color-scheme: dark){.viz-root{--surface-1:#1a1a19;--text-primary:#ffffff;"
           "--text-secondary:#c3c2b7;--grid:#383835;--series-1:#3987e5;--series-2:#d95926}}",
           ".bg{fill:var(--surface-1)}.grid{stroke:var(--grid);stroke-width:1}"
           ".axis{fill:var(--text-secondary);font-size:12px}.title{fill:var(--text-primary);font-size:15px;"
           "font-weight:600}.label{fill:var(--text-secondary);font-size:12px}"
           ".whisker{stroke:var(--text-secondary);stroke-width:1}.ring{stroke:var(--surface-1);stroke-width:2}"
           ".s1{fill:var(--series-1)}.s2{fill:var(--series-2)}.hit{fill:transparent}",
           "</style>",
           f'<rect class="bg" x="0" y="0" width="{W}" height="{H}"/>',
           f'<text class="title" x="{L}" y="26">{escape(title)}</text>']
    # legend, one row under the title
    lx = L
    for cls, shape, text in (("s1", "circle", "global-pipeline programs"), ("s2", "square", "baselines")):
        if shape == "circle":
            out.append(f'<circle class="{cls}" cx="{lx + 5}" cy="46" r="5"/>')
        else:
            out.append(f'<rect class="{cls}" x="{lx}" y="41" width="10" height="10" rx="2"/>')
        out.append(f'<text class="axis" x="{lx + 16}" y="50">{text}</text>')
        lx += 16 + 8 * len(text) + 24
    # grid and ticks
    k = 0
    while (y := ylo + k * ys) <= yhi + 1e-9:
        out.append(f'<line class="grid" x1="{L}" x2="{L + pw}" y1="{py(y):.1f}" y2="{py(y):.1f}"/>')
        out.append(f'<text class="axis" x="{L - 8}" y="{py(y) + 4:.1f}" text-anchor="end">{y:g}%</text>')
        k += 1
    k = 0
    while (x := k * xs) <= xmax + 1e-9:
        out.append(f'<line class="grid" x1="{px(x):.1f}" x2="{px(x):.1f}" y1="{T}" y2="{T + ph}"/>')
        out.append(f'<text class="axis" x="{px(x):.1f}" y="{T + ph + 18}" text-anchor="middle">'
                   f'{x / 1000:g}k</text>')
        k += 1
    out.append(f'<text class="axis" x="{L + pw / 2:.1f}" y="{H - 14}" text-anchor="middle">'
               f'completion tokens per question</text>')
    out.append(f'<text class="axis" transform="translate(18 {T + ph / 2:.1f}) rotate(-90)" '
               f'text-anchor="middle">test accuracy</text>')
    # labels: right of the dot, pushed apart vertically where they would overlap
    lab = sorted(({"p": p, "x": px(p["x"]) + 10, "y": py(p["y"]) + 4} for p in points), key=lambda d: d["y"])
    def width(d):
        return 7 * len(d["p"]["label"])                 # about 7 px a character at 12 px
    for i, a in enumerate(lab):
        for b in lab[:i]:
            if abs(a["y"] - b["y"]) < 15 and a["x"] < b["x"] + width(b) and b["x"] < a["x"] + width(a):
                a["y"] = b["y"] + 15
    for d in lab:
        p = d["p"]
        cx, cy = px(p["x"]), py(p["y"])
        cls = "s1" if p["kind"] == "v4" else "s2"
        tip = escape(f"{p['label']}: {p['y']:.1f}% ± {p['se']:.1f}, {p['x']:,.0f} tokens")
        out.append("<g>")
        out.append(f"<title>{tip}</title>")
        out.append(f'<line class="whisker" x1="{cx:.1f}" x2="{cx:.1f}" y1="{py(p["y"] - p["se"]):.1f}" '
                   f'y2="{py(p["y"] + p["se"]):.1f}"/>')
        if p["kind"] == "v4":
            out.append(f'<circle class="{cls} ring" cx="{cx:.1f}" cy="{cy:.1f}" r="5"/>')
        else:
            out.append(f'<rect class="{cls} ring" x="{cx - 5:.1f}" y="{cy - 5:.1f}" width="10" height="10" rx="2"/>')
        out.append(f'<circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="12"/>')
        out.append(f'<text class="label" x="{d["x"]:.1f}" y="{d["y"]:.1f}">{escape(p["label"])}</text>')
        out.append("</g>")
    out.append("</svg>")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()

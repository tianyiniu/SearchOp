"""Low-effort pairwise re-ranking of recorded test debates: which answer would a pairwise
judge pick from everything a debate proposed, and where in the debate the right answer was.

Every debate of a test evaluation (eval_routed_dev.py's results_k<n>.json; by default the
routed composite: each question debated by its group's program) is replayed from its round
cache, checked equal to the stored marks. Its committed answers (every answering speaker of
every round; open answers compared normalised, as the debate compares them) are ranked:

    1. by mentions (how many commitments name the answer), more first
    2. by the last round the answer appears in, later first
    3. by first commitment, earlier first

Each answer is argued by one speaker that committed it: a high-effort one if any, the latest
round, the first speaker of that round. Debates with two or more answers then get a knockout
on all answers: the top-ranked answer holds, the others challenge it in rank order. A match
is a two-response prompt to the model at --effort (low by default) ending 'CHOICE: 1' or
'CHOICE: 2'. Rules (--rules):

    both     both orders are asked; the challenger takes over only if it wins both
    first    one call, the holder shown first; the challenger takes over if it wins

Each rule runs its own knockout (calls are shared through the cache). Because a knockout
meets the challengers in rank order, "top 2" and "top 3" are the same knockout stopped after
one or two matches, so they are read off the same calls.

The report tallies, per role: proposed answers per debate, mentions per debate, the right
answer's mentions and its rank, and accuracy of the recorded final answer, the top-ranked
answer, each rule on the top 2 / top 3 / all answers, and the oracle (the right answer
proposed at all). Open answers (HLE) are graded by the run's judge (judge_answers.py, its
verdict cache); answers no debate ended on may need new verdicts (--no-grade: cache only).

    python scripts/pairwise_rerank.py --eval-dir outputs/pipeline_cluster_gptoss/run3/test_eval \\
        --name supergpqa_run3
    python scripts/pairwise_rerank.py --eval-dir outputs/pipeline_cluster_hle_gptoss/run1/test_eval \\
        --name hle_run1 --dataset datasets/hle_text_test_200.json

Writes outputs/pairwise_rerank/<name>/: calls.jsonl (every model call, resumed on a rerun),
debates.jsonl (one line per debate), report.md and report.json. --no-live reports from the
cache only (no model or judge calls): the tallies are complete, the knockouts as far as cached.
--combine <dir>... prints one table over finished runs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

import debate_mcq as D  # noqa: E402
import evolve_program_mcq as M  # noqa: E402
import program_space as P  # noqa: E402
from chooser_lab import Caller, argument, together  # noqa: E402
from gap_report import settings_argv  # noqa: E402

ROOT = P.ROOT
RULES = ("both", "first")
RULE_NAMES = {"both": "both orders", "first": "one order (holder first)"}
DEPTHS = (("top 2", 1), ("top 3", 2), ("all", None))      # matches played: 1, 2, all
PROMPT_CHARS = 3 * (32768 - 8000)       # past this a pair is shown as summaries (about 3 chars a token)

PAIR_SYSTEM = ("You are a Judge. Two responses to a hard question reach different final answers. Examine "
               "each response's reasoning against the exact wording of the question and find the first "
               "invalid step in each: a false fact, a wrong deduction, a miscalculation, a misread "
               "condition. How confident, long or well written a response is is not evidence that it is "
               "correct. Then decide which of the two final answers is correct; you must choose one of "
               "them. End your reply with exactly one line: 'CHOICE: 1' if the answer of Response 1 is "
               "correct, or 'CHOICE: 2' if the answer of Response 2 is correct.")
CHOICE_NUDGE = "Based on your reasoning above, reply with only the line 'CHOICE: 1' or 'CHOICE: 2'."
_CHOICE = re.compile(r"CHOICE\s*[:\-]?\s*\(?\s*(?:response\s*)?([12])\b", re.I)


# --- the executor of a recorded run ------------------------------------------------------

def configure(settings: dict, judge_cache: Path | None) -> int | None:
    """Configure the executor as the recorded run was (open answers and their judge too),
    check the result equals `settings`, and return the per-question turn cap."""
    s = {k: v for k, v in settings.items() if k not in ("answers", "judge")}
    argv = settings_argv(s)
    if settings.get("answers", "letters") == "open":
        argv += ["--answers", "open", "--judge-model", settings["judge"]["model"]]
        if judge_cache is not None:
            argv += ["--judge-cache", str(judge_cache)]
    ap = argparse.ArgumentParser()
    P.add_executor_args(ap)
    args = ap.parse_args(argv)
    args.base_urls, args.model, args.max_calls_per_question = "http://localhost:1/v1", settings["model"], None
    got = {**P.configure_from_args(args), "model": settings["model"]}
    if got != settings:
        raise SystemExit(f"could not reproduce the recorded settings:\n  recorded {settings}\n  got      {got}")
    return args.max_calls_per_question


class Recorder:
    """A runner wrapper that keeps the transcript, and the round specs, of its last debate."""

    def __init__(self, runner):
        self.runner, self.rounds, self.specs = runner, [], []

    def run_round(self, qid, all_rounds, executed_specs, round_spec, rep=0, prompts=None):
        out = self.runner.run_round(qid, all_rounds, executed_specs, round_spec, rep=rep, prompts=prompts)
        self.rounds = list(all_rounds) + [out]
        self.specs = list(executed_specs) + [round_spec]
        return out


# --- one debate ---------------------------------------------------------------------------

def mentions_of(rounds: list, specs: list, n: int) -> list[dict]:
    """Every commitment of the debate, in order: answer, round, position, persona, effort, text."""
    assert len(rounds) == len(specs), (len(rounds), len(specs))
    out = []
    for i, (rnd, spec) in enumerate(zip(rounds, specs)):
        effort = spec.get("effort", D.DEFAULT_EFFORT)
        for j, (persona, text) in enumerate(rnd):
            if persona in D.NON_ANSWERING:
                continue
            if (a := D.extract_letter(text, n)):
                out.append({"answer": a, "round": i, "pos": j, "persona": persona, "effort": effort, "text": text})
    return out


def rank_answers(mentions: list[dict]) -> list[dict]:
    """The distinct answers, ranked: mentions (more first), last round (later first), first
    commitment (earlier first). Each with its mentions, rounds and representative speaker."""
    by: dict[str, list[dict]] = {}
    for k, m in enumerate(mentions):
        by.setdefault(m["answer"], []).append({**m, "order": k})
    out = []
    for a, ms in by.items():
        rep = max(ms, key=lambda m: (m["effort"] == "high", m["round"], -m["pos"]))
        out.append({"answer": a, "mentions": len(ms), "last_round": max(m["round"] for m in ms),
                    "first_round": min(m["round"] for m in ms), "first": ms[0]["order"],
                    "efforts": sorted({m["effort"] for m in ms}), "rep": rep})
    out.sort(key=lambda x: (-x["mentions"], -x["last_round"], x["first"]))
    return out


def parse_choice(text: str | None) -> int | None:
    found = _CHOICE.findall((text or "").replace("*", ""))
    return int(found[-1]) if found else None


def parse_choice_followup(text: str | None) -> int | None:
    if (c := parse_choice(text)) is not None:
        return c
    m = re.fullmatch(r"\s*\(?\s*([12])\s*\)?\s*\.?\s*", (text or "").replace("*", ""))
    return int(m.group(1)) if m else None


def question_text(row: dict) -> str:
    return D.render_question(row["question"], list(row["options"])).strip()


def pair_prompt(row: dict, first: dict, second: dict, show: str) -> tuple[str, str]:
    """(user prompt, form used): `first` as Response 1. Full arguments fall back to the
    summaries when the pair would not fit the window."""
    def build(form):
        return (f"{question_text(row)}\n\nTwo responses to this question reach different final answers.\n\n"
                f"Response 1 (final answer: {first['answer']}):\n{argument(first['rep'], form)}\n\n"
                f"Response 2 (final answer: {second['answer']}):\n{argument(second['rep'], form)}\n\n"
                "Find the first invalid step in each response, then decide which final answer is correct. "
                "End with exactly one line: CHOICE: 1 or CHOICE: 2.")
    user = build(show)
    if show == "full" and len(PAIR_SYSTEM) + len(user) > PROMPT_CHARS:
        return build("summary"), "summary"
    return user, show


class Knockout:
    """The pairwise knockouts of one debate, one per rule, through the shared call cache."""

    def __init__(self, caller: Caller, effort: str, show: str):
        self.caller, self.effort, self.show = caller, effort, show

    def ask(self, deb: dict, row: dict, a: dict, b: dict) -> dict | None:
        """One call with `a` shown first: {"pick": answer or None, ...} or None if it failed."""
        user, form = pair_prompt(row, a, b, self.show)
        r = self.caller.ask(deb["q"], deb["rep"], f"pair_{self.effort}", PAIR_SYSTEM, user, self.effort,
                            parse_choice, CHOICE_NUDGE, parse_choice_followup)
        if r is None:
            return None
        pick = {1: a["answer"], 2: b["answer"]}.get(r["value"])
        return {"pick": pick, "completion": r["completion"], "calls": r["calls"], "form": form}

    def run(self, deb: dict, row: dict, rule: str) -> dict | None:
        """{"matches": [[holder, challenger, pick holder-first, pick challenger-first or None]],
        "completion", "calls"}; the winner after m matches is winner_after(matches, m)."""
        ranked = deb["ranked"]
        holder, matches, completion, calls = ranked[0], [], 0, 0
        for chal in ranked[1:]:
            if rule == "both":
                r1, r2 = together(lambda: self.ask(deb, row, holder, chal), lambda: self.ask(deb, row, chal, holder))
            else:
                r1, r2 = self.ask(deb, row, holder, chal), None
            if r1 is None or (rule == "both" and r2 is None):
                return None
            completion += r1["completion"] + (r2["completion"] if r2 else 0)
            calls += r1["calls"] + (r2["calls"] if r2 else 0)
            p1, p2 = r1["pick"], (r2["pick"] if r2 else None)
            matches.append([holder["answer"], chal["answer"], p1, p2])
            if p1 == chal["answer"] and (rule == "first" or p2 == chal["answer"]):
                holder = chal
        return {"matches": matches, "completion": completion, "calls": calls}


def winner_after(matches: list, depth: int | None, rule: str) -> str | None:
    """The holder after the first `depth` matches (None: all) of a recorded knockout."""
    if not matches:
        return None
    holder = matches[0][0]
    for h, chal, p1, p2 in matches[:depth]:
        assert h == holder, "matches out of order"
        if p1 == chal and (rule == "first" or p2 == chal):
            holder = chal
    return holder


# --- grading ------------------------------------------------------------------------------

def grader(live: bool):
    """(row, answer) -> True / False, or None when an open answer has no cached verdict and
    the judge may not be called."""
    if M.GRADER is None:
        return lambda row, a: a is not None and a == row.get("answer_letter")
    if live:
        return lambda row, a: bool(M.GRADER(row, a)) if a else False
    verdicts = P.JUDGE._verdicts
    return lambda row, a: (verdicts.get((row["id"], a)) if a else False)


# --- the report ---------------------------------------------------------------------------

def pct(k, n) -> str:
    return f"{100 * k / n:.1f}" if n else "-"


def paired(debs: list[dict], key, base="final_ok") -> tuple[float, float]:
    """Mean difference (points) of key - base over debates, and its standard error over
    questions (replicates averaged per question)."""
    per_q = defaultdict(list)
    for d in debs:
        per_q[d["q"]].append(float(bool(key(d))) - float(bool(d[base])))
    xs = [sum(v) / len(v) for v in per_q.values()]
    if len(xs) < 2:
        return (100 * xs[0] if xs else 0.0), 0.0
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return 100 * m, 100 * sd / math.sqrt(len(xs))


def bucket(x: int, top: int) -> str:
    return f"{top}+" if x >= top else str(x)


def role_report(debs: list[dict], rules: list[str]) -> tuple[list[str], list[str], dict]:
    """(notes and the accuracy table, the tallies, numbers) for one set of debates."""
    n = len(debs)
    split = [d for d in debs if d["n_answers"] >= 2]
    done = {r: [d for d in split if d["knockout"].get(r) is not None] for r in rules}
    oracle = sum(d["right_rank"] is not None for d in debs)
    final = sum(d["final_ok"] for d in debs)
    acc_lines, out = [], {"n": n, "split": len(split), "complete": {r: len(done[r]) for r in rules}}
    for r in rules:
        if len(done[r]) < len(split):
            acc_lines += [f"**{len(split) - len(done[r])} of {len(split)} split debates have no complete '{r}' "
                          f"knockout yet**: the '{r}' rows count them at their top-ranked answer.", ""]
    acc_lines += ["| answer | accuracy | vs recorded final (paired ±) | gap recovered | CF | WF | precision "
                  "| model calls per debate | completion tokens per debate |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def row(label, ok, calls=None, tokens=None):
        k = sum(bool(ok(d)) for d in debs)
        cf = sum(bool(ok(d)) and not d["final_ok"] for d in debs)
        wf = sum(d["final_ok"] and not ok(d) for d in debs)
        diff, se = paired(debs, ok)
        rec = (k - final) / (oracle - final) if oracle > final else None
        acc_lines.append(f"| {label} | {pct(k, n)} | {diff:+.1f} ± {se:.1f} | "
                         + ("-" if rec is None else f"{100 * rec:.0f}%") + f" | {cf} | {wf} | "
                         + (f"{100 * cf / (cf + wf):.0f}%" if cf + wf else "-")
                         + f" | {'-' if calls is None else f'{calls / n:.2f}'} | "
                         + f"{'-' if tokens is None else f'{tokens / n / 1000:.1f}k'} |")
        out.setdefault("accuracy", {})[label] = {"acc": 100 * k / n if n else None, "diff": diff, "se": se,
                                                 "recovered": rec, "cf": cf, "wf": wf,
                                                 "calls": None if calls is None else calls / n,
                                                 "tokens": None if tokens is None else tokens / n}

    row("recorded final", lambda d: d["final_ok"])
    row("top-ranked answer", lambda d: d["right_rank"] == 1)
    for r in rules:
        for label, depth in DEPTHS:
            def ok(d, r=r, depth=depth):
                ko = d["knockout"].get(r)
                if d["n_answers"] < 2 or ko is None:
                    return d["right_rank"] == 1
                return d["correct"].get(winner_after(ko["matches"], depth, r)) is True
            calls = sum(d["knockout"][r]["calls"] * share(d, r, depth) for d in done[r])
            tokens = sum(d["knockout"][r]["completion"] * share(d, r, depth) for d in done[r])
            row(f"pairwise, {RULE_NAMES[r]}, {label}", ok, calls, tokens)
    row("oracle (right answer proposed)", lambda d: d["right_rank"] is not None)

    tally = ["#### Proposed answers per debate, and where the right answer is", "",
             "| proposed answers | debates | right proposed | rank 1 | rank 2 | rank 3 | rank 4+ "
             "| recorded final right |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    by_k = defaultdict(list)
    for d in debs:
        by_k[bucket(d["n_answers"], 5)].append(d)
    for k in sorted(by_k, key=lambda s: int(s.rstrip("+"))):
        g = by_k[k]
        rk = Counter(bucket(d["right_rank"], 4) for d in g if d["right_rank"] is not None)
        tally.append(f"| {k} | {len(g)} ({pct(len(g), n)}%) | {pct(sum(d['right_rank'] is not None for d in g), len(g))}% "
                     f"| {pct(rk.get('1', 0), len(g))}% | {pct(rk.get('2', 0), len(g))}% | {pct(rk.get('3', 0), len(g))}% "
                     f"| {pct(rk.get('4+', 0), len(g))}% | {pct(sum(d['final_ok'] for d in g), len(g))}% |")
    out["by_proposed"] = {k: {"n": len(g), "rank": dict(Counter(bucket(d["right_rank"], 4) for d in g
                                                                if d["right_rank"] is not None))}
                          for k, g in by_k.items()}

    def hist(title, key, labels):
        c = Counter(key(d) for d in debs)
        tally.extend(["", f"#### {title}", "", "| " + " | ".join(labels) + " |",
                      "|" + "---:|" * len(labels),
                      "| " + " | ".join(f"{c.get(l, 0)} ({pct(c.get(l, 0), n)}%)" for l in labels) + " |"])
        out.setdefault("tallies", {})[title] = {l: c.get(l, 0) for l in labels}

    hist("Rank of the right answer (all debates)",
         lambda d: "absent" if d["right_rank"] is None else bucket(d["right_rank"], 4),
         ["1", "2", "3", "4+", "absent"])
    hist("Mentions per debate (commitments over all rounds)", lambda d: bucket(d["n_mentions"], 8),
         [str(i) for i in range(8)] + ["8+"])
    hist("Mentions of the right answer", lambda d: bucket(d["right_mentions"], 5), [str(i) for i in range(5)] + ["5+"])
    hist("Mentions of the top-ranked answer", lambda d: bucket(d["top_mentions"], 5),
         [str(i) for i in range(5)] + ["5+"])
    unknown = sum(d["ungraded"] for d in debs)
    if unknown:
        tally += ["", f"{unknown} proposed answers have no verdict yet (--no-grade or --no-live); they count as wrong."]
    out["ungraded"] = unknown
    return acc_lines, tally, out


def share(d: dict, rule: str, depth: int | None) -> float:
    """The share of a knockout's matches played when it stops after `depth` matches (its
    calls and tokens are spread evenly over its matches)."""
    m = len(d["knockout"][rule]["matches"])
    return 1.0 if depth is None or not m else min(depth, m) / m


def combine(dirs: list[Path]) -> None:
    lines = ["| run | role | debates | split | recorded final | top-ranked | both, top 2 | both, top 3 | both, all "
             "| one order, top 2 | one order, top 3 | one order, all | oracle | right at rank 4+ |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for d in dirs:
        rep = json.loads((d / "report.json").read_text())
        for role, r in rep["roles"].items():
            acc = r["accuracy"]
            cell = lambda k: f"{acc[k]['acc']:.1f}" if k in acc else "-"          # noqa: E731
            ranks = r["tallies"]["Rank of the right answer (all debates)"]
            lines.append(f"| {rep['name']} | {role} | {r['n']} | {r['split']} | {cell('recorded final')} | "
                         f"{cell('top-ranked answer')} | "
                         + " | ".join(cell(f"pairwise, {w}, {lab}") for w in RULE_NAMES.values()
                                      for lab, _ in DEPTHS)
                         + f" | {cell('oracle (right answer proposed)')} | {pct(ranks.get('4+', 0), r['n'])} |")
    print("\n".join(lines))


# --- main ---------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", type=Path, help="a test evaluation directory (eval_routed_dev.py)")
    ap.add_argument("--name", help="output name (default: the run directory's name)")
    ap.add_argument("--results", type=Path, default=None, help="default: the results_k<n>.json with the most replicates")
    ap.add_argument("--dataset", type=Path, default=ROOT / "datasets/supergpqa_600_test.json")
    ap.add_argument("--cache", type=Path, default=None, help="default: the rounds_*.jsonl in --eval-dir")
    ap.add_argument("--judge-cache", type=Path, default=None,
                    help="open answers: the run's verdict cache (default <run's output dir>/judge_<model>.jsonl)")
    ap.add_argument("--roles", default="routed", help="'routed' and/or role names of the results file, comma-separated")
    ap.add_argument("--rules", default="both,first", help=f"of {', '.join(RULES)}")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--show", default="full", choices=["full", "summary"],
                    help="each answer's argument: its speaker's reply and summary, or the summary only")
    ap.add_argument("--limit", type=int, default=None, help="the first N debates of each role (a sanity run)")
    ap.add_argument("--out", type=Path, default=None, help="default outputs/pairwise_rerank/<name>")
    ap.add_argument("--base-urls", default="http://localhost:7472/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--no-live", action="store_true", help="no model or judge calls: report what is cached")
    ap.add_argument("--no-grade", action="store_true", help="open answers: no new judge verdicts")
    ap.add_argument("--combine", type=Path, nargs="+", default=None, help="print one table over finished output dirs")
    args = ap.parse_args()
    if args.combine:
        combine(args.combine)
        return
    if args.eval_dir is None:
        raise SystemExit("--eval-dir is required")
    rules = [r for r in args.rules.split(",") if r]
    for r in rules:
        if r not in RULES:
            raise SystemExit(f"unknown rule {r!r}; choose from {RULES}")

    if args.results is None:
        found = sorted(args.eval_dir.glob("results_k*.json"), key=lambda p: int(p.stem.split("_k")[1]))
        if not found:
            raise SystemExit(f"no results_k*.json in {args.eval_dir}")
        args.results = found[-1]
    res = json.loads(args.results.read_text())
    settings = res["settings"]
    if settings.get("answers") == "open" and args.judge_cache is None:
        args.judge_cache = args.eval_dir.parent.parent / f"judge_{settings['judge']['model']}.jsonl"
        if not args.judge_cache.exists():
            raise SystemExit(f"no verdict cache at {args.judge_cache}; give --judge-cache")
    cap = configure(settings, args.judge_cache)
    rows = {r["id"]: r for r in json.loads(args.dataset.read_text())}
    routes = json.loads((ROOT / res["routes"]).read_text())["routes"]
    qids = [q for q in rows if q in routes]
    if not qids:
        raise SystemExit(f"no question of {args.dataset} is in the routes file {res['routes']}: wrong --dataset?")
    cache = args.cache or next(iter(sorted(args.eval_dir.glob("rounds_*.jsonl"))))
    name = args.name or args.eval_dir.parent.name
    out_dir = args.out or ROOT / "outputs/pairwise_rerank" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    live = not args.no_live
    grade = grader(live and not args.no_grade)

    # 1. replay the debates
    rec = Recorder(M.CacheRunner([cache]))
    roles = [r for r in args.roles.split(",") if r]
    groups = sorted({routes[q]["group"] for q in qids})
    debates: dict[str, list[dict]] = {}
    for role in roles:
        items, mismatches = [], 0
        for g in (groups if role == "routed" else [None]):
            key = res["roles"][f"grid_{g}" if role == "routed" else role]
            prog = P.normalize_program(json.loads(key))
            stored = res["programs"][key]["reps"]
            for rep in sorted(stored, key=int):
                for q in qids:
                    if (g is not None and routes[q]["group"] != g) or q not in stored[rep]:
                        continue
                    rec.rounds, rec.specs = [], []
                    row = rows[q]
                    o = M.run_program(prog, rec, row, rep=int(rep), max_calls=cap)
                    v = stored[rep][q]
                    if (o["letter"] or "?") != (v[2] or "?") or bool(o["correct"]) != bool(v[0]):
                        mismatches += 1
                    ms = mentions_of(rec.rounds, rec.specs, len(row["options"]))
                    items.append({"q": q, "rep": int(rep), "group": routes[q]["group"], "final": o["letter"],
                                  "final_ok": bool(v[0]), "turns": v[1], "mentions": ms,
                                  "ranked": rank_answers(ms)})
        if mismatches:
            raise SystemExit(f"{role}: {mismatches} replays differ from the stored marks: wrong cache or settings?")
        items.sort(key=lambda d: (d["q"], d["rep"]))
        debates[role] = items[:args.limit] if args.limit else items
    print(f"{name}: settings {settings}; " + ", ".join(f"{r} {len(v)} debates" for r, v in debates.items()))

    # 2. grade every proposed answer
    pairs = sorted({(d["q"], a["answer"]) for v in debates.values() for d in v for a in d["ranked"]})
    with ThreadPoolExecutor(max_workers=16) as pool:
        verdicts = dict(zip(pairs, tqdm(pool.map(lambda p: grade(rows[p[0]], p[1]), pairs), total=len(pairs),
                                        desc="grading", disable=M.GRADER is None)))
    for v in debates.values():
        for d in v:
            d["correct"] = {a["answer"]: verdicts[(d["q"], a["answer"])] for a in d["ranked"]}
            d["ungraded"] = sum(x is None for x in d["correct"].values())
            right = [i + 1 for i, a in enumerate(d["ranked"]) if d["correct"][a["answer"]] is True]
            d["right_rank"] = right[0] if right else None
            d["n_answers"], d["n_mentions"] = len(d["ranked"]), len(d["mentions"])
            d["right_mentions"] = sum(d["ranked"][i - 1]["mentions"] for i in right)
            d["top_mentions"] = d["ranked"][0]["mentions"] if d["ranked"] else 0
    if P.JUDGE is not None:
        print(f"judge: {P.JUDGE.stats}")

    # 3. the knockouts
    caller = Caller(out_dir / "calls.jsonl", args.base_urls, settings["model"], args.api_key, live=live)
    ko = Knockout(caller, args.effort, args.show)
    try:
        for role, v in debates.items():
            split = [d for d in v if d["n_answers"] >= 2]
            need = sum(len(d["ranked"]) - 1 for d in split)
            print(f"{role}: {len(split)} of {len(v)} debates proposed two or more answers; {need} matches per rule")
            for rule in rules:
                def job(d, rule=rule):
                    return ko.run(d, rows[d["q"]], rule)
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    got = list(tqdm(pool.map(job, split), total=len(split), desc=f"{role} {rule}"))
                for d in v:
                    d.setdefault("knockout", {})
                for d, r in zip(split, got):
                    d["knockout"][rule] = r
    finally:
        caller.close()
    print(f"model calls: {caller.new_calls} new, {caller.errors} failed")

    # 4. the report
    report = [f"# Pairwise re-ranking: {name}", "",
              f"Eval {args.eval_dir} ({args.results.name}); settings {settings}. Pairwise at effort "
              f"'{args.effort}', arguments '{args.show}', rules {', '.join(rules)}"
              + (f"; the first {args.limit} debates of each role only" if args.limit else "") + ".", "",
              "Answers are ranked by mentions (commitments over all rounds), then by the last round they appear "
              "in (later first). 'top 2' / 'top 3' are the knockout stopped after one / two matches. "
              "'gap recovered' = (accuracy - recorded final) / (oracle - recorded final). Debates with one "
              "proposed answer keep it. Model calls are the pairwise calls only (each counts one turn at low "
              "effort); the debate's own cost is unchanged.", ""]
    saved = {"name": name, "eval_dir": str(args.eval_dir), "settings": settings, "effort": args.effort,
             "show": args.show, "rules": rules, "limit": args.limit, "roles": {}}
    for role, v in debates.items():
        acc, tally, stats = role_report(v, rules)
        report += [f"## {role}: {len(v)} debates", "", "### Accuracy (percent of all debates)", ""] + acc
        report += ["", "### Tallies", ""] + tally + ["", "### Accuracy by group", ""]
        for g in groups:
            sub = [d for d in v if d["group"] == g]
            if sub:
                gacc, _, gs = role_report(sub, rules)
                stats.setdefault("by_group", {})[g] = gs
                report += [f"#### group {g}: {len(sub)} debates", ""] + gacc + [""]
        saved["roles"][role] = stats
        print(f"\n## {role}: {len(v)} debates\n" + "\n".join(acc))
    (out_dir / "report.md").write_text("\n".join(report) + "\n")
    (out_dir / "report.json").write_text(json.dumps(saved, indent=1))
    with (out_dir / "debates.jsonl").open("w") as fh:
        for role, v in debates.items():
            for d in v:
                fh.write(json.dumps({"role": role, "q": d["q"], "rep": d["rep"], "group": d["group"],
                                     "final": d["final"], "final_ok": d["final_ok"],
                                     "answers": [{k: a[k] for k in ("answer", "mentions", "last_round", "first_round",
                                                                    "efforts")} | {"correct": d["correct"][a["answer"]],
                                                                                   "argued_by": [a["rep"]["round"], a["rep"]["persona"], a["rep"]["effort"]]}
                                                 for a in d["ranked"]],
                                     "right_rank": d["right_rank"], "knockout": d.get("knockout", {})},
                                    ensure_ascii=False) + "\n")
    print(f"wrote {out_dir}/report.md, report.json, debates.jsonl")


if __name__ == "__main__":
    main()

"""Where the gap between "the right answer was in the debate" and "the debate's answer
was right" comes from, read off recorded test debates (no model calls).

Every program of a test evaluation (eval_routed_dev.py's results_k<n>.json: the
per-group programs, the routed composite, the global program and the baselines) is
replayed from its round cache, transcript and all. Per program:

    final            the program's answer is right (checked equal to the stored marks)
    covered          the right answer was committed by some speaker of the debate
                     (round 1, rounds 1-2, ..., all rounds: the oracle over the debate)
    plurality 1      the most common answer of round 1 is right (ties: the first
                     committed, as the executor's vote; "random" = expected over the tie)
    CF / WF          final right where the round-1 plurality was wrong (a correct flip)
                     / final wrong where it was right (a wrong flip); flip precision
                     CF / (CF + WF), net CF - WF

The present-but-wrong gap (covered - final) splits into three disjoint parts:

    late, not adopted       the right answer first appeared after round 1; final wrong
    minority, not adopted   it was in round 1 but not its plurality; final wrong
    lost (WF)               it was round 1's plurality; final wrong. Split into
                            abandoned (no speaker of the last answering round commits it:
                            the debate talked itself out of it) and outvoted (still
                            committed in the last answering round, not read)

and final = plurality 1 + CF - WF exactly. Everything is also broken down by the
pattern of round 1 (single speaker / unanimous / majority / plurality without a
majority / tie) and by group.

    python scripts/gap_report.py --eval-dir outputs/pipeline_cluster_gptoss/run3/test_eval

Writes <eval-dir>/gap_report.md and gap_report.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import debate_mcq as D  # noqa: E402
import evolve_program_mcq as M  # noqa: E402
import program_space as P  # noqa: E402
import schema_fitness as SF  # noqa: E402

ROOT = P.ROOT
PATTERNS = ("single", "unanimous", "majority", "plurality", "tie", "none")


# --- the executor of a recorded run ------------------------------------------------------

def settings_argv(settings: dict) -> list[str]:
    """add_executor_args options that reproduce a recorded settings dict (SuperGPQA runs)."""
    s = {k: v for k, v in settings.items() if k != "model"}
    if s.get("answers", "letters") != "letters" or s.get("eliminator") or s.get("deep_think"):
        raise SystemExit(f"settings {settings} are not a letters-only cluster-pipeline run; not supported here")
    argv = ["--executor", s["executor"]]
    if "digest" in s:
        argv += ["--digest-head", str(s["digest"][0]), "--digest-tail", str(s["digest"][1])]
    if "summary_words" in s:
        argv += ["--summary-words", str(s["summary_words"])]
    if "window" in s:
        argv += ["--context-window", str(s["window"])]
    if "high_cost" in s:
        argv += ["--high-cost", str(s["high_cost"])]
    if "turn_cap" in s:
        argv += ["--turn-cap", str(s["turn_cap"])]
    for key in ("total_cap",):
        if key in s:
            argv += ["--" + key.replace("_", "-"), str(s[key])]
    for key, flag in (("judge_persona", "--judge-persona"), ("any_round_width", "--any-round-width"),
                      ("visible_reasoning", "--visible-reasoning"), ("last_round_vote", "--last-round-vote"),
                      ("plain_instruction", "--plain-instruction"), ("count_read_summaries", "--count-read-summaries")):
        if s.get(key):
            argv.append(flag)
    return argv


def configure_like(settings: dict) -> int:
    """Configure this process's executor as the recorded run was, check that the result
    equals `settings`, and return the per-question turn cap."""
    ap = argparse.ArgumentParser()
    P.add_executor_args(ap)
    args = ap.parse_args(settings_argv(settings))
    args.base_urls, args.model, args.max_calls_per_question = "http://localhost:1/v1", settings["model"], None
    got = {**P.configure_from_args(args), "model": settings["model"]}
    if got != settings:
        raise SystemExit(f"could not reproduce the recorded settings:\n  recorded {settings}\n  got      {got}")
    return args.max_calls_per_question


# --- one debate ---------------------------------------------------------------------------

class Recorder:
    """A runner wrapper that keeps the transcript of the debate it last served."""

    def __init__(self, runner):
        self.runner, self.rounds = runner, []

    def run_round(self, qid, all_rounds, executed_specs, round_spec, rep=0, prompts=None):
        out = self.runner.run_round(qid, all_rounds, executed_specs, round_spec, rep=rep, prompts=prompts)
        self.rounds = list(all_rounds) + [out]
        return out


def round_letters(rnd: list, n: int) -> list[str]:
    return SF.committed_letters([rnd], n)


def plurality(letters: list[str]) -> tuple[str | None, list[str]]:
    """(the executor's vote: most common, ties to the first committed; the tied top answers)."""
    if not letters:
        return None, []
    cnt = Counter(letters)
    top = max(cnt.values())
    tied = [l for l in dict.fromkeys(letters) if cnt[l] == top]
    return tied[0], tied


def pattern(rnd: list, letters: list[str]) -> str:
    """The shape of round 1's answers."""
    answering = [p for p, _ in rnd if p not in D.NON_ANSWERING]
    if len(answering) <= 1:
        return "single"
    if not letters:
        return "none"
    cnt = Counter(letters)
    top = sorted(cnt.values(), reverse=True)
    if len(cnt) == 1:
        return "unanimous"
    if top[0] == top[1]:
        return "tie"
    return "majority" if top[0] * 2 > len(letters) else "plurality"


def anatomy(rounds: list, n: int, right: str, final: str | None) -> dict:
    """Everything the report counts about one debate, from its transcript and its answer."""
    per_round = [round_letters(r, n) for r in rounds]
    l1 = per_round[0] if per_round else []
    p1, tied = plurality(l1)
    cov = []                                   # right answer committed in rounds 1..i
    seen = False
    for ls in per_round:
        seen = seen or right in ls
        cov.append(seen)
    covered = bool(cov and cov[-1])
    cov1 = right in l1
    final_ok = final is not None and final == right
    p1_ok = p1 == right
    last = next((ls for ls in reversed(per_round) if ls), [])
    d = {"final": final_ok, "covered": covered, "cov1": cov1, "cov_by_round": cov,
         "p1": p1_ok, "p1_random": (right in tied) / len(tied) if tied else 0.0,
         "cf": final_ok and not p1_ok, "wf": p1_ok and not final_ok,
         "pattern": pattern(rounds[0], l1) if rounds else "none",
         "n_rounds": len(rounds), "distinct": len(set(l for ls in per_round for l in ls))}
    d["late_lost"] = covered and not cov1 and not final_ok
    d["minority_lost"] = cov1 and not p1_ok and not final_ok
    d["abandoned"] = d["wf"] and right not in last
    d["outvoted"] = d["wf"] and right in last
    return d


# --- aggregation ------------------------------------------------------------------------

def summarize(items: list[dict]) -> dict:
    """Percentages (of all debates) and counts for a list of anatomy() dicts."""
    n = len(items)
    if not n:
        return {"n": 0}
    s = lambda k: sum(bool(x[k]) for x in items)                       # noqa: E731
    cf, wf = s("cf"), s("wf")
    depth = max(len(x["cov_by_round"]) for x in items)
    by_round = [100 * sum(x["cov_by_round"][min(i, len(x["cov_by_round"]) - 1)] if x["cov_by_round"] else 0
                          for x in items) / n for i in range(depth)]
    out = {"n": n, "final": 100 * s("final") / n, "covered": 100 * s("covered") / n,
           "cov1": 100 * s("cov1") / n, "cov_by_round": by_round,
           "p1": 100 * s("p1") / n, "p1_random": 100 * sum(x["p1_random"] for x in items) / n,
           "gap": 100 * (s("covered") - s("final")) / n,
           "late_lost": 100 * s("late_lost") / n, "minority_lost": 100 * s("minority_lost") / n,
           "wf_pct": 100 * wf / n, "abandoned": 100 * s("abandoned") / n, "outvoted": 100 * s("outvoted") / n,
           "cf": cf, "wf": wf, "net": cf - wf, "precision": cf / (cf + wf) if cf + wf else None,
           "turns": sum(x["turns"] for x in items) / n,
           "tokens": sum(x["tokens"] or 0 for x in items) / n}
    # the identities the report rests on (exact, in counts)
    assert s("covered") - s("final") == s("late_lost") + s("minority_lost") + wf, "gap partition"
    assert s("final") == s("p1") + cf - wf, "flip identity"
    assert s("abandoned") + s("outvoted") == wf
    return out


def by(items: list[dict], key) -> dict:
    groups = defaultdict(list)
    for x in items:
        groups[key(x)].append(x)
    return {k: summarize(v) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


# --- the report ----------------------------------------------------------------------

def f1(x) -> str:
    return "-" if x is None else f"{x:.1f}"


def main_table(rows: list[tuple[str, dict]]) -> list[str]:
    out = ["| program | n | final | right present: round 1 | ... all rounds | gap | late, not adopted "
           "| minority, not adopted | lost (WF): abandoned | lost (WF): outvoted | round-1 plurality "
           "(random tie) | CF | WF | precision | net | turns | tokens |",
           "|---|" + "---:|" * 16]
    for name, s in rows:
        if not s.get("n"):
            continue
        out.append(f"| {name} | {s['n']} | {f1(s['final'])} | {f1(s['cov1'])} | {f1(s['covered'])} | "
                   f"{f1(s['gap'])} | {f1(s['late_lost'])} | {f1(s['minority_lost'])} | {f1(s['abandoned'])} | "
                   f"{f1(s['outvoted'])} | {f1(s['p1'])} ({f1(s['p1_random'])}) | {s['cf']} | {s['wf']} | "
                   + (f"{100 * s['precision']:.0f}%" if s["precision"] is not None else "-")
                   + f" | {s['net']:+d} | {s['turns']:.1f} | {s['tokens'] / 1000:.1f}k |")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", type=Path, required=True, help="a test evaluation directory (eval_routed_dev.py)")
    ap.add_argument("--results", default=None, help="default: the results_k<n>.json with the most replicates")
    ap.add_argument("--dataset", type=Path, default=ROOT / "datasets/supergpqa_600_test.json")
    ap.add_argument("--cache", type=Path, default=None, help="default: the rounds_*.jsonl in --eval-dir")
    args = ap.parse_args()

    if args.results is None:
        found = sorted(args.eval_dir.glob("results_k*.json"), key=lambda p: int(p.stem.split("_k")[1]))
        if not found:
            raise SystemExit(f"no results_k*.json in {args.eval_dir}")
        args.results = found[-1]
    res = json.loads(Path(args.results).read_text())
    cap = configure_like(res["settings"])
    rows = {r["id"]: r for r in json.loads(args.dataset.read_text())}
    routes = json.loads(Path(ROOT / res["routes"]).read_text())["routes"]
    cache = args.cache or next(iter(sorted(args.eval_dir.glob("rounds_*.jsonl"))))
    runner = M.CacheRunner([cache])
    rec = Recorder(runner)
    qids = [q for q in rows if q in routes]
    groups = sorted({routes[q]["group"] for q in qids})
    roles = res["roles"]

    per: dict[str, list[dict]] = {}
    mismatches = 0
    for role, key in roles.items():
        prog = P.normalize_program(json.loads(key))
        stored = res["programs"][key]["reps"]
        items = []
        for rep in sorted(stored, key=int):
            for q in qids:
                if role.startswith("grid_") and routes[q]["group"] != int(role.split("_")[1]):
                    continue
                if q not in stored[rep]:
                    continue
                rec.rounds = []
                o = M.run_program(prog, rec, rows[q], rep=int(rep), max_calls=cap)
                v = stored[rep][q]
                if (o["letter"] or "?") != (v[2] or "?") or bool(o["correct"]) != bool(v[0]):
                    mismatches += 1
                d = anatomy(rec.rounds, len(rows[q]["options"]), rows[q]["answer_letter"], o["letter"])
                d.update(q=q, rep=int(rep), group=routes[q]["group"], turns=o["n_calls"],
                         tokens=v[5] if len(v) > 5 else None)
                items.append(d)
        per[role] = items
    if mismatches:
        raise SystemExit(f"{mismatches} replays differ from the stored marks: wrong cache or settings?")
    # the routed composite: each question answered by its group's program
    per["routed"] = [x for g in groups for x in per.get(f"grid_{g}", [])]

    order = ["routed", "global"] + [f"grid_{g}" for g in groups] + \
            [r for r in roles if r not in ("global",) and not r.startswith("grid_")]
    tables = {name: summarize(per[name]) for name in order if name in per}
    patterns = {name: by(per[name], lambda x: x["pattern"]) for name in order if name in per}
    by_group = {name: by(per[name], lambda x: x["group"]) for name in order if name in per}

    lines = [f"# Gap report: {args.eval_dir}", "",
             f"{len(qids)} test questions, {len(res['programs'][next(iter(roles.values()))]['reps'])} replicates; "
             f"settings {res['settings']}. Percentages are of all debates of the row (question x replicate). "
             "gap = right present (all rounds) - final = late, not adopted + minority, not adopted + lost (WF). "
             "final = round-1 plurality + CF - WF.", "", "## All debates", ""]
    lines += main_table([(n, tables[n]) for n in order if n in tables])
    lines += ["", "## By the pattern of round 1", "",
              "single = one answering speaker; unanimous; majority = one answer from more than half; "
              "plurality = a unique most common answer without a majority; tie = several answers share the top "
              "count.", ""]
    for name in order:
        if name not in patterns:
            continue
        lines += [f"### {name}", ""]
        lines += main_table([(p, patterns[name][p]) for p in PATTERNS if p in patterns[name]])
        lines.append("")
    lines += ["## By group", ""]
    for name in order:
        if name not in by_group:
            continue
        lines += [f"### {name}", ""]
        lines += main_table([(f"group {g}", s) for g, s in by_group[name].items()])
        lines.append("")
    (args.eval_dir / "gap_report.md").write_text("\n".join(lines) + "\n")
    (args.eval_dir / "gap_report.json").write_text(json.dumps(
        {"eval_dir": str(args.eval_dir), "results": str(args.results), "settings": res["settings"],
         "programs": {n: json.loads(roles[n]) for n in roles},
         "all": tables, "by_pattern": patterns, "by_group": by_group}, indent=1))
    print("\n".join(main_table([(n, tables[n]) for n in order if n in tables])))
    print(f"\nwrote {args.eval_dir / 'gap_report.md'} and .json (all replays matched the stored marks)")


if __name__ == "__main__":
    main()

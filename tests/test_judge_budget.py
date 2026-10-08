"""Offline checks for the judge speaker and the turn budget (--judge-persona, --high-cost,
--turn-cap): that nothing changes without them; the settings, the vocabulary, the costs and the
cap with them; the judge's validity rules, prompt, candidate list and pick-only rule, with
letters and with open answers; the judge seeds; and a seeds -> search -> test-evaluation run
with them on. The fake debate model, guide and grading judge of the other v3 tests stand in
for every model: no vLLM, no API key.

    python tests/test_judge_budget.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))   # the code under test
sys.path.insert(0, str(Path(__file__).resolve().parent))                       # the other test files

import test_open_answers as TO  # noqa: E402  (installs the fake model, guide and grading judge)
import test_pipeline_cluster as T3  # noqa: E402
import test_routed_dev as TR  # noqa: E402
import debate_mcq as D  # noqa: E402
import evolve_program_mcq as M  # noqa: E402
import program_space as P  # noqa: E402
import program_guide as G  # noqa: E402
import evolve_pipeline_cluster as V  # noqa: E402
import program_seeds_cluster as S3  # noqa: E402
import eval_routed_dev as EV  # noqa: E402

T, check, run_cli, TMP = T3.T, T3.check, T3.run_cli, T3.TMP
_Resp, _Scripted = T3._Resp, T3._Scripted
GPT = "openai/gpt-oss-20b"
ON = ["--high-cost", "3", "--turn-cap", "20", "--judge-persona", "--any-round-width"]


def configure(*extra: str, answers: list[str] | None = None, cap=None) -> tuple[dict, argparse.Namespace]:
    ap = argparse.ArgumentParser()
    P.add_executor_args(ap)
    args = ap.parse_args(["--context-window", "65536", "--visible-reasoning"] + (answers or []) + list(extra))
    args.base_urls, args.model, args.max_calls_per_question = "http://localhost:1/v1", GPT, cap
    return P.configure_from_args(args), args


def fails(fn, *a, **kw) -> str | None:
    try:
        fn(*a, **kw)
    except (ValueError, SystemExit, AssertionError) as exc:
        return str(exc)
    return None


# --- 1. the budget -------------------------------------------------------------------------

def test_budget():
    print("budget")
    settings, args = configure()
    check("turn_cap" not in settings and "judge_persona" not in settings and settings["high_cost"] == 5
          and args.max_calls_per_question == 16 and P.MAX_TURNS == 16 and not D.JUDGE_ON,
          "defaults: 5 turns a high-effort speaker, a 16-turn cap, no judge; the settings as before", str(settings))
    base_rounds = set(P.PLAN_ROUNDS)
    check(len(base_rounds) == 28 and not any("judge" in r for r in base_rounds)
          and "solver_x4|high" not in base_rounds, "defaults: the 28 rounds of the v3 grammar")
    settings, args = configure(*ON)
    check(settings["high_cost"] == 3 and settings["turn_cap"] == 20 and settings["judge_persona"] is True
          and args.max_calls_per_question == 20 and D.HIGH_COST == 3 and P.MAX_TURNS == 20,
          "--high-cost 3 --turn-cap 20 --judge-persona: in the settings, and the call cap follows the turn cap",
          str(settings))
    new = set(P.PLAN_ROUNDS) - base_rounds
    check(new == {"judge", "judge|high", "solver_x4|high", "solver_x4|high|blind"} and base_rounds <= set(P.PLAN_ROUNDS),
          "the judge rounds, and four high-effort solvers (12 turns) now fit in a round", str(sorted(new)))
    check("judge|blind" not in P.PLAN_ROUNDS and "judge" in M.ACTIONS and "judge" in M.ROUND_KINDS,
          "the judge is a move and a round kind, never blind")
    check(D.spec_cost(P.PLAN_ROUNDS["solver_x3|high"]) == 9 and D.spec_cost(P.PLAN_ROUNDS["judge|high"]) == 3
          and D.spec_cost(P.PLAN_ROUNDS["solver_x2"]) == 2, "a high-effort speaker costs 3 turns")
    _, args = configure(*ON, cap=20)
    check(args.max_calls_per_question == 20, "an explicit cap equal to the turn cap is accepted")
    check("differs from the debate cap" in (fails(configure, *ON, cap=16) or ""),
          "an explicit cap that disagrees with the turn cap is refused")
    check("--high-cost, --turn-cap" in (fails(P.configure_executor, executor="v2", high_cost=3) or "")
          and "--high-cost, --turn-cap" in (fails(P.configure_executor, executor="v2", total_cap=21) or ""),
          "the budget options need the v3 executor")
    settings, args = configure()
    check("turn_cap" not in settings and P.MAX_TURNS == 16 and D.HIGH_COST == 5 and not D.JUDGE_ON
          and "judge" not in D.PERSONA_PROMPTS and len(P.PLAN_ROUNDS) == 28,
          "configuring again without the options restores every default")


class _Runner:
    """Answers each speaker from a script: persona -> list of letters, used in order."""

    def __init__(self, script: dict[str, list[str]]):
        self.script = {k: list(v) for k, v in script.items()}
        self.personas: list[str] = []

    def run_round(self, qid, rounds, specs, spec, rep=0, prompts=None):
        self.personas += spec["personas"]
        return [(p, f"Reasoning.\nANSWER: {self.script[p].pop(0)}") for p in spec["personas"]]


def test_cap():
    print("the cap in a debate")
    row = {"id": "q", "options": list("ABCDEFGHIJ"), "answer_letter": "C", "question": "?"}
    configure(*ON)
    seeds = P.judge_seeds()
    prog = seeds["judge_on_disagree_high"]
    r = _Runner({"solver": ["A", "B", "B"], "judge": ["A"]})
    out = M.run_program(prog, r, row, max_calls=P.MAX_TURNS)
    check(out["n_calls"] == 12 and out["letter"] == "A" and r.personas.count("judge") == 1,
          "three high-effort solvers disagree: the judge runs (12 turns) and its pick is the answer", str(out))
    r = _Runner({"solver": ["B", "B", "B"], "judge": ["A"]})
    out = M.run_program(prog, r, row, max_calls=P.MAX_TURNS)
    check(out["n_calls"] == 9 and out["letter"] == "B" and "judge" not in r.personas,
          "they agree: no judge, 9 turns")
    r = _Runner({"solver": ["A", "B", "C", "D"], "judge": ["C"]})
    out = M.run_program(seeds["pool_judge_high"], r, row, max_calls=P.MAX_TURNS)
    check(out["n_calls"] == 11 and out["letter"] == "C", "pool_judge_high: 6 + 2 + 3 turns, the judge decides")
    configure("--judge-persona")                        # judge on, but the old budget: 15 + 5 > 16
    r = _Runner({"solver": ["A", "B", "B"], "judge": ["A"]})
    out = M.run_program(P.judge_seeds()["judge_on_disagree_high"], r, row, max_calls=P.MAX_TURNS)
    check(out["n_calls"] == 15 and "judge" not in r.personas and out["letter"] == "B",
          "under the old budget the judge would pass the cap, so the program stops (last commit) instead")
    configure(*ON)
    big = {"plan": [dict(P.PLAN_ROUNDS["solver_x4|high"]), dict(P.PLAN_ROUNDS["solver_x3|high"])],
           "rules": [P.CONT()], "default": "stop:vote"}
    r = _Runner({"solver": list("ABCDABC")})
    out = M.run_program(big, r, row, max_calls=P.MAX_TURNS)
    check(out["n_calls"] == 12, "a round that would pass 20 turns is not run (12 + 9 > 20)")


# --- 1b. the width edit ------------------------------------------------------------------------------

def test_width():
    print("the width edit")
    configure()
    check(P.edit_kinds() == P.EDIT_KINDS and "set_width" not in P.edit_kinds(),
          "without --any-round-width the edit kinds are as before (plan_width, first round only)")
    settings, _ = configure(*ON)
    kinds = P.edit_kinds()
    check("set_width" in kinds and "plan_width" not in kinds and len(kinds) == len(P.EDIT_KINDS)
          and settings.get("any_round_width") is True, "with it, set_width takes plan_width's place", str(kinds))
    V_ = P.PLAN_ROUNDS
    prog = {"plan": [dict(V_["solver_x2"]), dict(V_["solver_x3|high|blind"]), dict(V_["critic"])],
            "rules": [P.CONT(), {"when": ["step==3", "n_distinct>=2"], "do": "solver_x2|high|blind"}],
            "default": "stop:vote"}
    P.validate_program(prog)
    rng = random.Random(2)
    seen, bad = Counter(), []
    for _ in range(600):
        child, op = P.mutate_width(prog, rng)
        changed = [(i, P.v3_name(a), P.v3_name(b)) for i, (a, b) in enumerate(zip(prog["plan"], child["plan"])) if a != b]
        changed += [("rule", prog["rules"][1]["do"], child["rules"][1]["do"])] if child["rules"] != prog["rules"] else []
        if len(changed) != 1 or fails(P.validate_program, child) is not None:
            bad.append(changed)
            continue
        where, old, new = changed[0]
        seen[where] += 1
        strip = lambda n: n.replace("solver_x2", "S").replace("solver_x3", "S").replace("solver_x4", "S").replace("solver", "S")
        if strip(old) != strip(new) or old == new:
            bad.append(changed)
    check(not bad and set(seen) == {0, 1, "rule"},
          "it changes one solver round at a time (first, later or extra), never its effort or visibility",
          f"{dict(seen)} bad {bad[:2]}")
    check("set_width" not in P.applicable_edits({"plan": [dict(V_["expert"])], "rules": [P.CONT()],
                                                 "default": "stop:vote"}),
          "a program without a solver round has no width edit")
    locked = {**prog, "rules": [P.CONT(), {"when": ["step==3"], "do": "solver_x2|high|blind"},
                                {"when": ["last:solver_x2|high|blind"], "do": "stop:vote"}]}
    kids = [P.mutate_uniform(locked, rng)[0] for _ in range(300)]
    check(all(fails(P.validate_program, k) is None for k in kids),
          "an edit that would orphan a condition (last:<old round>) is drawn again, never returned")
    check("solver_x4|high" in {P.v3_name(s) for _, _, s in P.width_edit_slots({"plan": [dict(V_["solver_x3|high"])],
                                                                             "rules": [P.CONT()], "default": "stop:vote"})},
          "under a 20-turn cap four high-effort solvers are a reachable width")
    configure("--any-round-width")                       # the old budget: 4 x 5 = 20 > 16
    check("solver_x4|high" not in {P.v3_name(s) for _, _, s in P.width_edit_slots(
          {"plan": [dict(P.PLAN_ROUNDS["solver_x3|high"])], "rules": [P.CONT()], "default": "stop:vote"})},
          "... and not under the old one (it is not a round of the vocabulary)")
    configure()


# --- 2. validity ---------------------------------------------------------------------------------

def test_validity():
    print("the judge's validity rules")
    configure(*ON)
    J = dict(P.PLAN_ROUNDS["judge|high"])
    S2 = dict(P.PLAN_ROUNDS["solver_x2"])
    first = {"plan": [J, S2], "rules": [P.CONT()], "default": "stop:vote"}
    check("cannot speak first" in (fails(P.validate_program, first) or ""), "a judge cannot open the plan")
    move = {"plan": [S2], "rules": [{"when": ["step==0"], "do": "judge"}, P.CONT()], "default": "stop:vote"}
    check("cannot speak first" in (fails(P.validate_program, move) or ""), "... nor be the move at step 0")
    later = {"plan": [S2, J], "rules": [P.CONT()], "default": "stop:last_commit"}
    cond = {"plan": [S2], "rules": [P.CONT(), {"when": ["step==1", "n_distinct>=2"], "do": "judge|high"},
                                    {"when": ["last_round:judge"], "do": "stop:last_commit"}], "default": "stop:vote"}
    check(fails(P.validate_program, later) is None and fails(P.validate_program, cond) is None,
          "a judge after an answering round is valid, and last_round:judge is a condition")
    for name, prog in P.judge_seeds().items():
        check(fails(P.validate_program, prog) is None and P.first_action(prog) == "continue", f"seed {name} is valid")
    rng = random.Random(4)
    progs = [P.random_program(rng) for _ in range(400)]
    check(not any(P.judge_first(p) for p in progs) and sum("judge" in P.canon(p) for p in progs) > 20,
          "random programs use the judge but never open with it", str(sum("judge" in P.canon(p) for p in progs)))
    parent = P.normalize_program(P.PROTOCOLS["mad"])
    kids = [P.mutate_uniform(parent, rng, donors=[P.judge_seeds()["pool_judge_high"]])[0] for _ in range(400)]
    check(all(fails(P.validate_program, k) is None for k in kids) and not any(P.judge_first(k) for k in kids),
          "every edit is valid and none opens with a judge")
    text = P.grammar_text()
    check("judge" in text and "cannot speak first" in text and "more than 20 turns" in text
          and "counts as\n    3 turns" in text, "the grammar text describes the judge and the budget")
    captured = {}

    def capture(client, instructions, user, fmt, effort, max_output_tokens=None):
        captured["i"] = instructions
        raise KeyboardInterrupt

    old = G._call
    G._call = capture
    try:
        G.write_group_seeds(None, {"direct": P.PROTOCOLS["direct"]}, {0: "group 0"})
    except KeyboardInterrupt:
        pass
    G._call = old
    check("synthesizer, judge" in captured.get("i", ""), "the seed writer is told the judge is a role")
    configure()
    check("judge" not in P.grammar_text(), "without --judge-persona the grammar has no judge")


# --- 3. the judge in the executor ------------------------------------------------------------------

def test_executor_letters():
    print("the judge speaker, letters")
    configure(*ON)
    opts = list("ABCDEFGHIJ")
    prior = [[("solver", "Because one.\nANSWER: B"), ("solver", "Because two.\nANSWER: C")]]
    spec = P.PLAN_ROUNDS["judge"]
    short = "A short argument. " * 30
    before = dict(D.JUDGE_STATS)

    def stat(k):
        return D.JUDGE_STATS[k] - before[k]

    c = _Scripted(_Resp(short + "\nANSWER: C"))
    (p, text), = D.execute_round(c, GPT, "Which?", opts, spec, prior, 1.0)
    user, system = c.requests[0]["messages"][1]["content"], c.requests[0]["messages"][0]["content"]
    check(p == "judge" and D.extract_letter(text, 10) == "C" and len(c.requests) == 1,
          "a pick among the candidates stands, with no extra call")
    check("[Round 1, solver 1] Answer: B" in user and "[Round 1, solver 2] Answer: C" in user
          and "Candidate answers (every distinct letter given so far): B, C." in user
          and user.index("Candidate answers") < user.index("Give your response"),
          "the judge sees the debate, then the candidate list")
    check(system.startswith("You are a Judge.") and "not evidence" in system and "do not propose a different option"
          in system, "the judge's prompt: choose among the candidates, a head count is not evidence")
    c = _Scripted(_Resp(short + "\nANSWER: E"), _Resp("ANSWER: B"))
    (_, text), = D.execute_round(c, GPT, "Which?", opts, spec, prior, 1.0)
    nudge = c.requests[1]
    check(D.extract_letter(text, 10) == "B" and D.JUDGE_MARK in text and stat("repicked") == 1,
          "an answer off the list gets one prompt to choose from it, and the choice is the commit")
    check(nudge["messages"][-1]["content"].startswith("Your answer must be one of the candidate letters: B, C.")
          and nudge["extra_body"] == {"reasoning_effort": "low"} and nudge["max_tokens"] == D.COMMIT_TOKENS_V3,
          "... a low-effort prompt with the commit prompt's limit", str(nudge["extra_body"]))
    check(D._digest([("judge", text)], 10).startswith("[judge] chose B"), "later speakers see the judge's pick")
    c = _Scripted(_Resp(short + "\nANSWER: E"), _Resp("I still think E."))
    (_, text), = D.execute_round(c, GPT, "Which?", opts, spec, prior, 1.0)
    check(D.extract_letter(text, 10) == "E" and stat("off_list") == 1,
          "if it still will not choose, its own answer stands (and is counted)")
    none = [[("solver", "Still thinking, no commitment.")]]
    c = _Scripted(_Resp(short + "\nANSWER: D"))
    (_, text), = D.execute_round(c, GPT, "Which?", opts, spec, none, 1.0)
    check("No answer has been given yet" in c.requests[0]["messages"][1]["content"]
          and D.extract_letter(text, 10) == "D" and len(c.requests) == 1 and stat("no_candidates") == 1,
          "with nothing committed it answers the question itself")
    c = _Scripted(_Resp("ANSWER: C", "hidden step " * 400), _Resp("The key check.\nANSWER: C"))
    (_, text), = D.execute_round(c, GPT, "Which?", opts, P.PLAN_ROUNDS["judge|high"], prior, 1.0)
    check(c.requests[0]["extra_body"] == {"reasoning_effort": "high"} and D.SUMMARY_MARK in text
          and D.extract_letter(text, 10) == "C", "judge|high: high effort, summarised like any speaker")
    check(stat("calls") == 5, "every judge call is counted", str(stat("calls")))


def test_executor_open():
    print("the judge speaker, open answers")
    configure(*ON, answers=["--answers", "open", "--judge-cache", str(TMP / "judge_open.jsonl")])
    check("judge" in D.PERSONA_PROMPTS and "do not propose a different answer" in D.PERSONA_PROMPTS["judge"],
          "the open-answer judge prompt")
    prior = [[("solver", "Work.\nANSWER: $x^2$"), ("solver", "Work.\nANSWER: 42"), ("solver", "Work.\nANSWER: **42**")]]
    spec = P.PLAN_ROUNDS["judge"]
    c = _Scripted(_Resp("A short argument. " * 30 + "\nANSWER: \\boxed{42}"))
    (_, text), = D.execute_round(c, GPT, "What?", [], spec, prior, 1.0)
    user = c.requests[0]["messages"][1]["content"]
    check("Candidate answers (every distinct answer given so far):\n- x^2\n- 42\n" in user
          and D.extract_letter(text, 0) == "42" and len(c.requests) == 1,
          "the candidates are the distinct normalised answers; a formatted pick of one of them stands")
    c = _Scripted(_Resp("A short argument. " * 30 + "\nANSWER: 41"), _Resp("ANSWER: $42$"))
    (_, text), = D.execute_round(c, GPT, "What?", [], spec, prior, 1.0)
    check(c.requests[1]["messages"][-1]["content"].startswith("Your answer must be one of the candidate answers:\n- x^2\n- 42")
          and D.extract_letter(text, 0) == "42", "an answer off the list is asked to choose, and the choice is read normalised")
    configure()


# --- 4. seeds, search and test evaluation with the options on ---------------------------------------

def test_pipeline(routes: Path):
    print("seeds, search and evaluation with the judge and the new budget")
    out = TMP / "judge_budget"
    common = ["--clusters", str(T3.CLUSTERS_V3), "--per-group", "3", "--live-cache", str(out / "rounds.jsonl"),
              "--workers", "8", "--visible-reasoning"] + T3.WINDOW + ON
    configure(*ON)
    check(S3.seed_counts(6) == (6, 10, 0), "seed counts: one model-written per group, 8 literature + 2 judge")
    run_cli(S3, common + ["--out", str(out / "seeds.json")])
    sd = json.loads((out / "seeds.json").read_text())
    src = Counter(s["source"] for s in sd["seeds"])
    check(src == {"protocol": 8, "judge": 2, "llm": 6} and sd["settings"]["turn_cap"] == 20
          and sd["settings"]["judge_persona"] is True and sd["settings"]["high_cost"] == 3,
          "16 seeds: 8 literature, 2 judge, 6 model-written, made under the new settings", str(dict(src)))
    run = out / "run"
    run_cli(V, common + ["--seeds", str(out / "seeds.json"), "--out", str(run), "--generations", "2"])
    lines = [json.loads(l) for l in (run / "archive.jsonl").open() if l.strip()]
    header, recs = lines[0], {}
    for d in lines[1:]:
        recs[d["key"]] = d
    check(header["settings"]["turn_cap"] == 20 and header["settings"]["judge_persona"] is True
          and header["settings"]["any_round_width"] is True, "the archive records the budget, the judge and the width edit")
    ops = Counter(d.get("op") for d in recs.values() if d["gen"] > 0)
    check("plan_width" not in ops, "no child was made by plan_width (set_width replaces it)", str(dict(ops)))
    turns = [v[1] for d in recs.values() for rep in d["reps"].values() for v in rep.values()]
    js = next(d for d in recs.values() if d["name"] == "judge_on_disagree_high")
    jt = Counter(v[1] for rep in js["reps"].values() for v in rep.values())
    check(max(turns) <= 20 and set(jt) <= {9, 12} and jt[12] > 0,
          "no debate passes 20 turns; the judge seed runs its judge on disagreement (12) and not otherwise (9)",
          str(dict(jt)))
    gens = [json.loads(l) for l in (run / "generations.jsonl").open() if l.strip()]
    check(all("judge" in g for g in gens if g["gen"] > 0) and gens[-1]["judge"]["calls"] > 0,
          "the generation log counts the judge calls", str(gens[-1].get("judge")))
    try:
        run_cli(V, common[:-len(ON)] + ["--seeds", str(out / "seeds.json"), "--out", str(run), "--generations", "2",
                                         "--resume"])
        check(False, "resuming without the options is refused")
    except SystemExit:
        check(True, "resuming without the options is refused")
    run_cli(V, common + ["--seeds", str(out / "seeds.json"), "--out", str(run), "--generations", "2", "--resume"])
    args = ["--run", str(run), "--routes", str(routes), "--workers", "8", "--no-champions", "--reps", "1",
            "--visible-reasoning"] + T3.WINDOW
    try:
        run_cli(EV, args + ["--out", str(out / "eval_bad")])
        check(False, "the evaluation refuses settings other than the search's")
    except SystemExit as exc:
        check("the search ran with" in str(exc), "the evaluation refuses settings other than the search's")
    run_cli(EV, args + ON + ["--out", str(out / "eval")])
    res = json.loads((out / "eval" / "results_k1.json").read_text())
    check(not any(res["missing"].values()) and res["settings"]["turn_cap"] == 20,
          "the test evaluation runs under the search's settings")
    configure()


def test_pipeline_open():
    print("open answers with the judge and the new budget")
    out = TMP / "judge_budget_open"
    common = ["--clusters", str(TO.CLUSTERS), "--dataset", str(TO.TRAIN), "--per-group", "2",
              "--live-cache", str(out / "rounds.jsonl"), "--workers", "8", "--visible-reasoning"] + TO.OPEN_ARGS + ON
    run_cli(S3, common + ["--out", str(out / "seeds.json")])
    src = Counter(s["source"] for s in json.loads((out / "seeds.json").read_text())["seeds"])
    check(src == {"protocol": 8, "judge": 2, "llm": 4}, "HLE: 14 seeds with the judge ones", str(dict(src)))
    run = out / "run"
    run_cli(V, common + ["--seeds", str(out / "seeds.json"), "--out", str(run), "--generations", "1"])
    lines = [json.loads(l) for l in (run / "archive.jsonl").open() if l.strip()]
    recs = {d["key"]: d for d in lines[1:]}
    js = next(d for d in recs.values() if d["name"] == "judge_on_disagree_high")
    answers = [v[2] for rep in js["reps"].values() for v in rep.values()]
    check(lines[0]["settings"].get("answers") == "open" and lines[0]["settings"]["judge_persona"] is True
          and all(a == "?" or D.normalize_answer(a) == a for a in answers),
          "the judge seed runs with open answers; its answers are normalised", str(Counter(answers).most_common(3)))
    configure()


if __name__ == "__main__":
    test_budget()
    test_cap()
    test_width()
    test_validity()
    test_executor_letters()
    test_executor_open()
    configure()
    T3.test_subsets()                  # writes the small test clusters (CLUSTERS_V3)
    routes = TR.test_routes()
    test_pipeline(routes)
    test_pipeline_open()
    print()
    if T.FAILURES:
        print(f"{len(T.FAILURES)} FAILED: " + "; ".join(T.FAILURES))
        sys.exit(1)
    print(f"all checks passed; judge stats {D.JUDGE_STATS}; temp dir {TMP}")

"""Offline checks for evolve_program_mcq's live mode. The model is replaced by
a stub that answers deterministically, so the budget, prescreen, cache
write-and-reuse, per-question call cap, fresh-replicate recheck and --save-all
paths all run end to end with no vLLM and no cost.

    python3 scripts/test_evolve_program_live.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import debate_mcq as D  # noqa: E402
import evolve_program_mcq as M  # noqa: E402

FAILURES: list[str] = []
CALLS = {"n": 0}


def check(okay: bool, label: str, extra: str = "") -> None:
    print(("  ok   " if okay else "  FAIL ") + label + (f"  [{extra}]" if extra else ""))
    if not okay:
        FAILURES.append(label)


def stub_execute_round(client, model, question, options, round_spec, all_rounds,
                       temperature, max_tokens=3072, prompts=None):
    """One canned response per persona: a letter chosen from a hash of the
    question and the position, so reruns agree and replicates can differ."""
    out = []
    for i, persona in enumerate(round_spec["personas"]):
        CALLS["n"] += 1
        seed = f"{question[:40]}|{len(all_rounds)}|{i}|{persona}"
        letter = "ABCD"[int(hashlib.sha1(seed.encode()).hexdigest(), 16) % 4]
        if persona in D.NON_ANSWERING:
            out.append((persona, "Ruling out options. SURVIVING: A B"))
        else:
            out.append((persona, f"Reasoning as {persona}.\nANSWER: {letter}"))
    return out


REAL_EXECUTE_ROUND = D.execute_round
D.execute_round = stub_execute_round

rows_all = json.loads(Path("datasets/supergpqa_strict_train.json").read_text())
ROWS = {r["id"]: r for r in rows_all[:24]}
TMP = Path(tempfile.mkdtemp(prefix="evolve_live_test_"))
EMPTY = TMP / "empty_cache.jsonl"
EMPTY.write_text("")


def make_runner(live_cache: Path) -> M.BudgetedRunner:
    return M.BudgetedRunner(ROWS, [EMPTY], base_urls="http://127.0.0.1:1/v1",
                            model="stub", temperature=0.7, answer_tokens=64,
                            cache_path=live_cache, max_calls=10_000,
                            api_key="EMPTY", progress=False)


def test_budget_and_cache():
    print("budget, cache write, cache reuse")
    runner = make_runner(TMP / "live1.jsonl")
    row = next(iter(ROWS.values()))
    prog = M.chain("critic")
    CALLS["n"] = 0
    runner.reset_budget(0)
    try:
        M.run_program(prog, runner, row)
        check(False, "budget 0 raises OffCache on a miss")
    except M.OffCache:
        check(CALLS["n"] == 0, "budget 0 makes no model calls")
    runner.reset_budget(10)
    out = M.run_program(prog, runner, row)
    check(out["n_calls"] == 2 and CALLS["n"] == 2, "budget 10 runs solver+critic live",
          str(CALLS["n"]))
    runner.reset_budget(0)
    out2 = M.run_program(prog, runner, row)
    check(out2["letter"] == out["letter"] and CALLS["n"] == 2,
          "same request again is a cache hit, no new calls")
    runner.reset_budget(1)
    try:
        M.run_program(M.chain("critic", "fresh"), runner, row)   # fresh costs 2
        check(False, "budget 1 cannot afford a 2-call round")
    except M.OffCache:
        check(CALLS["n"] == 2, "budget too small -> OffCache before any call, prefix reused")
    runner.close()
    lines = [json.loads(l) for l in (TMP / "live1.jsonl").open() if l.strip()]
    check(len(lines) == 2, "two new rounds written to the live cache file", str(len(lines)))
    runner2 = make_runner(TMP / "live1.jsonl")            # resume
    runner2.reset_budget(0)
    CALLS["n"] = 0
    M.run_program(prog, runner2, row)
    check(CALLS["n"] == 0, "a new runner reloads the live cache and replays it")
    check((TMP / "live1.lock").exists(), "lock file exists while the cache is open")
    try:
        make_runner(TMP / "live1.jsonl")
        check(False, "second runner on an open cache is refused")
    except SystemExit as exc:
        check("in use by another run" in str(exc), "second runner on an open cache is refused")
    runner2.close()
    check(not (TMP / "live1.lock").exists(), "lock file removed on close")
    import socket
    (TMP / "live1.lock").write_text(json.dumps({"host": socket.gethostname(), "pid": 999999999,
                                                "started": "earlier"}))
    runner3 = make_runner(TMP / "live1.jsonl")
    check(runner3._own_lock, "stale lock from a dead pid on this host is cleared")
    runner3.close()
    (TMP / "live1.lock").write_text(json.dumps({"host": "some-other-host", "pid": 1,
                                                "started": "earlier"}))
    try:
        make_runner(TMP / "live1.jsonl")
        check(False, "lock from another host is respected")
    except SystemExit:
        check(True, "lock from another host is respected")
    (TMP / "live1.lock").unlink()


def test_torn_records_are_skipped():
    print("torn cache lines that still parse as JSON are skipped")
    torn = TMP / "torn.jsonl"
    good = {"q": "x", "k": json.dumps({"rounds": [{"personas": ["solver"]}]}), "r": 0,
            "responses": [["solver", "ANSWER: A"]], "error": None}
    bad1 = dict(good, q="y", responses=[["solver"]])             # lost the text
    bad2 = dict(good, k='{"rounds":[{"persexpert"]')             # spliced key
    bad3 = dict(good, q="z", responses="ANSWER: A")              # not a list
    torn.write_text("\n".join(json.dumps(x) for x in (bad1, good, bad2, bad3)) + "\n")
    target = {}
    M.load_cache_into(target, torn)
    check(len(target) == 1 and ("x", good["k"], 0) in target, "only the well-formed record loaded",
          str(len(target)))
    runner = M.BudgetedRunner({}, [], base_urls="http://127.0.0.1:1/v1", model="stub",
                              temperature=0.7, answer_tokens=64, cache_path=torn,
                              max_calls=10, api_key="EMPTY", progress=False)
    check(len(runner._cache) == 1, "live runner drops malformed rounds from its own file",
          str(len(runner._cache)))
    runner.close()


def test_digest_window_and_cache_key():
    print("digest window: head+tail clipping and cache-key separation")
    import schema_fitness as SF
    text = "A" * 100 + "B" * 2000 + "C" * 100 + "\nANSWER: D"
    try:
        check(D.digest_signature() is None, "default window is not part of the key")
        d0 = D._clip(text)
        check(d0 == text.strip()[:700] and "ANSWER" not in d0,
              "default: head-only cut, conclusion lost", f"{len(d0)} chars")
        spec = [{"personas": ["critic"], "sees": "all"}]
        k_default = SF.path_key(spec)
        D.set_digest(300, 900)
        d1 = D._clip(text)
        check(d1.startswith("A" * 100) and d1.rstrip().endswith("ANSWER: D"),
              "head+tail: keeps the opening AND the conclusion", f"{len(d1)} chars")
        check("omitted" in d1, "elision marker names what was dropped")
        check(D.digest_signature() == (300, 900), "non-default window is reported")
        check(SF.path_key(spec) != k_default,
              "cache key changes with the window, so recordings cannot mix")
        short = "short answer\nANSWER: B"
        check(D._clip(short) == short, "a response inside the budget is untouched")
    finally:
        D.set_digest(700, 0)
    check(D.digest_signature() is None and SF.path_key([{"personas": ["critic"], "sees": "all"}]) == k_default,
          "restoring the default restores the original key")


def test_no_eliminator():
    print("--no-eliminator removes the moves and the seeds that need them")
    import importlib
    m = importlib.reload(M)
    check("eliminate" in m.ACTIONS and "elim_blind" in m.ACTIONS, "eliminator moves present by default")
    check(any(r["do"] == "elim_blind" for r in m.PROGRAM_B["rules"]),
          "experiment B's parked track uses the eliminator")
    m.drop_eliminator()
    check("eliminate" not in m.ACTIONS and "elim_blind" not in m.ACTIONS, "moves dropped")
    check("eliminator" not in m.ROUND_KINDS, "eliminator gone from the round-kind menu",
          str(m.ROUND_KINDS))
    try:
        m.validate_program(m.PROGRAM_B)
        check(False, "the old program_b should no longer validate")
    except AssertionError:
        check(True, "the old program_b no longer validates")
    m.validate_program(m.PROGRAM_B_NOELIM)
    check(True, "the eliminator-free seed validates")
    # the reference programs reported at the end must also be runnable
    for nm, p in (("program_fixed", m.PROGRAM_FIXED), ("solver_critic", m.PROBES["solver_critic"])):
        m.validate_program(p)
    check(True, "the other reference programs still validate without the eliminator")
    rng = random.Random(3)
    prog = m.PROGRAM_B_NOELIM
    for _ in range(300):
        prog = m.mutate_program(prog, rng)
        conds = [c for r in prog["rules"] for c in r["when"]]
        acts = [r["do"] for r in prog["rules"]]
        if any("elim" in c for c in conds) or any("elim" in a for a in acts):
            check(False, "a mutation reintroduced an eliminator move")
            break
    else:
        check(True, "300 mutations never reintroduce the eliminator")
    importlib.reload(M)          # leave the module as the other tests expect


def test_call_cap_and_round_kinds():
    print("per-question call cap and round-content conditions")
    runner = make_runner(TMP / "live2.jsonl")
    runner.reset_budget(None)
    row = next(iter(ROWS.values()))
    out = M.run_program(M.chain("critic", "critic", "critic"), runner, row, max_calls=3)
    check(out["n_calls"] == 3, "cap 3 stops after solver + 2 critics", str(out["n_calls"]))
    prog = {"plan": [M.TG_ROOT],
            "rules": [{"when": ["acts==0"], "do": "continue"},
                      {"when": ["last_round:solver"], "do": "critic"},
                      {"when": ["last_round:critic", "ran_round:solver"], "do": "verifier"}],
            "default": "stop:last_commit"}
    out = M.run_program(prog, runner, row)
    check(out["actions"] == ["continue", "critic", "verifier"],
          "last_round:/ran_round: see plan rounds by content", str(out["actions"]))
    check("solver" in M.ROUND_KINDS and "expert+solver" in M.ROUND_KINDS,
          "round kind menu built from the action specs")
    runner.close()
    for bad, label in (({"plan": [M.TG_ROOT], "rules": [{"when": ["stepp==1"], "do": "critic"}],
                         "default": "stop:last_commit"}, "unknown condition rejected"),
                       ({"plan": [M.TG_ROOT], "rules": [], "default": "critic"},
                        "non-stop default rejected"),
                       ({"plan": [M.TG_ROOT], "rules": [{"when": ["last_round:nobody"],
                                                          "do": "critic"}],
                         "default": "stop:last_commit"}, "unknown round kind rejected")):
        try:
            M.validate_program(bad)
            check(False, label)
        except (AssertionError, ValueError):
            check(True, label)
    for name, prog in {"program_b": M.PROGRAM_B, "program_fixed": M.PROGRAM_FIXED,
                       **M.PROBES}.items():
        M.validate_program(prog)
    check(True, "all seed programs still validate")


def test_eval_merge():
    print("eval_program: subset fill merges into an existing Eval")
    runner = make_runner(TMP / "live3.jsonl")
    qids = list(ROWS)[:6]
    prog = M.chain("critic")
    runner.reset_budget(0)
    ev = M.eval_program(prog, runner, ROWS, qids, 0)
    check(ev.covered == 0 and ev.marks.count("x") == 6, "all off-cache on an empty cache")
    runner.reset_budget(None)
    gaps = [i for i, m in enumerate(ev.marks) if m == "x"][:4]
    M.eval_program(prog, runner, ROWS, qids, 0, workers=2, only=gaps, into=ev)
    check(ev.covered == 4 and ev.marks[4] == "x", "only the requested gaps were filled",
          "".join(ev.marks))
    check(len(ev.paths) == 1 and ev.path_ids[0] == 0, "action paths recorded and indexed")
    check(ev.letters[0] != "?" and ev.calls[0] == 2, "letters and calls recorded")
    j = ev.to_json("train")
    check(set(j) >= {"train_outcomes", "train_letters", "train_calls", "train_path_ids",
                     "train_paths"}, "to_json carries the clustering fields")
    runner.close()


def test_evolve_live_end_to_end():
    print("evolve --live end to end on 24 questions with the stub model")
    ds = TMP / "mini.json"
    ds.write_text(json.dumps(list(ROWS.values())))
    args = argparse.Namespace(
        seed=0, dataset=ds, cache=EMPTY, treegrow_cache=EMPTY, rep=0, limit=None,
        max_calls_per_question=8, live=True, live_cache=TMP / "live4.jsonl",
        base_urls="http://127.0.0.1:1/v1", model="stub", temperature=0.7,
        answer_tokens=64, api_key="EMPTY", novelty_budget=40, prescreen_margin=1.0,
        min_coverage=0.0, workers=4, recheck_top=2, recheck_rep=1, recheck_n=8,
        recheck_baselines=True, ignore_cache_lock=False, no_eliminator=False,
        max_total_calls=100_000, generations=2, population=4, offspring=2,
        evolved_out=TMP / "evolved.json", save_all=TMP / "all.jsonl",
        bestofn_cache=Path("does/not/exist.jsonl"))
    CALLS["n"] = 0
    M.evolve(args)
    out = json.loads((TMP / "evolved.json").read_text())
    check(out["live"] is True and out["live_calls_total"] > 0, "live calls were made",
          str(out["live_calls_total"]))
    top = out["top"][0]
    check(top["dev_covered"] == 1.0, "dev gaps were filled live, not scored as wrong",
          str(top["dev_covered"]))
    check("dev_fresh_acc" in top and top["dev_fresh_rep"] == 1,
          "top programs got a fresh-replicate dev recheck")
    check(top["dev_fresh_n"] == 8 and len(out["recheck_qids"]) == 8
          and len(top["dev_fresh_outcomes"]) == 8, "recheck ran on the fixed 8-question subset")
    check(set(out["recheck_qids"]) <= set(json.loads((TMP / "all.jsonl").open().readline())["dev_qids"]),
          "recheck subset drawn from dev only")
    check("dev_paths" in top and "dev_fresh_letters" in top, "dev details saved")
    lines = [json.loads(l) for l in (TMP / "all.jsonl").open() if l.strip()]
    header, recs = lines[0], lines[1:]
    check("questions" in header and len(header["questions"]) == 24,
          "save-all header has per-question features")
    feat = next(iter(header["questions"].values()))
    check({"discipline", "difficulty", "n_options", "gold", "r1_majority"} <= set(feat),
          "feature fields present", str(sorted(feat)))
    check(all("train_paths" in r and "live_calls_spent" in r for r in recs),
          "every scored program carries paths and live spend")
    check(any(r["live_calls_spent"] > 0 for r in recs), "some candidates spent budget")
    check(all(r["live_calls_spent"] <= 40 + 8 for r in recs),
          "no candidate exceeded budget + one question's cap",
          str(max(r["live_calls_spent"] for r in recs)))
    check(set(out["baselines_fresh"]) == {"program_b", "program_b_vote", "program_fixed",
                                           "solver_critic"},
          "reference programs got the same fresh dev run")



def test_vote_read_and_commit_followup():
    print("vote read-off and commit follow-up")
    # vote: plurality over every committed letter, ties to the earliest
    import schema_fitness as SF
    from collections import Counter
    rounds = [[("solver", "ANSWER: A"), ("solver", "ANSWER: B"), ("solver", "ANSWER: B")],
              [("critic", "ANSWER: B")], [("critic", "no letter here")]]
    letters = SF.committed_letters(rounds, 4)
    check(Counter(letters).most_common(1)[0][0] == "B", "plurality picks the modal letter")
    tied = SF.committed_letters([[("solver", "ANSWER: C"), ("solver", "ANSWER: A")],
                                 [("critic", "ANSWER: A"), ("critic", "ANSWER: C")]], 4)
    check(Counter(tied).most_common(1)[0][0] == "C", "a tie goes to the letter committed first")
    prog = M.with_vote_read(M.PROGRAM_FIXED)
    M.validate_program(prog)
    check(prog["default"] == "stop:vote" and all(
        not r["do"].startswith("stop:") or r["do"] == "stop:vote" for r in prog["rules"]),
        "with_vote_read rewrites every stop")
    check("vote" in M.STOP_READS, "vote is a legal read")

    # commit follow-up: a truncated reply gets one continuation, appended
    class Msg:
        def __init__(self, c): self.content = c
    class Choice:
        def __init__(self, c): self.message = Msg(c); self.finish_reason = "stop"
    class Resp:
        def __init__(self, c): self.choices = [Choice(c)]
    class FakeCompletions:
        def __init__(self): self.calls = []
        def create(self, **kw):
            self.calls.append(kw)
            if len(kw["messages"]) == 4:           # the nudge: system, user, assistant, user
                return Resp("ANSWER: C")
            return Resp("Long derivation that never commits" * 3)
    class FakeClient:
        def __init__(self):
            self.chat = type("C", (), {})()
            self.chat.completions = FakeCompletions()
    seen = {"n": 0}
    D.on_followup = lambda: seen.__setitem__("n", seen["n"] + 1)
    D.set_commit_followup(True)
    try:
        cli = FakeClient()
        out = REAL_EXECUTE_ROUND(cli, "stub", "Q?", ["a", "b", "c", "d"],
                                 {"personas": ["solver"]}, [], 0.7, 64)
        text = out[0][1]
        check(D.extract_letter(text, 4) == "C", "follow-up commit is parseable", text[-40:])
        check(D.COMMIT_MARK in text, "appended text carries the marker")
        check(len(cli.chat.completions.calls) == 2, "exactly one extra call",
              len(cli.chat.completions.calls))
        check(cli.chat.completions.calls[1]["max_tokens"] == D.COMMIT_MAX_TOKENS,
              "follow-up uses the short budget")
        check(seen["n"] == 1, "runner hook counted the follow-up")
        # a prose commit reply is normalized to an ANSWER line; a derivation is not
        for txt, want in (("The answer is **D** because of the units.", "D"),
                          ("Based on the above, option (B) is correct.", "B"),
                          ("\\boxed{A}", "A"), ("C", "C"),
                          ("ANSWER: E", None),                 # E is out of range for 4 options
                          ("The Reynolds number is 2.3e5, so the flow is turbulent", None)):
            check(D.commit_letter_lenient(txt, 4) == want, f"lenient commit parse {txt[:30]!r}",
                  D.commit_letter_lenient(txt, 4))
        cli3 = FakeClient()
        cli3.chat.completions.create = (lambda **kw: Resp("So the answer is **B**, since x=2.")
                                        if len(kw["messages"]) == 4 else Resp("no commit here"))
        out3 = REAL_EXECUTE_ROUND(cli3, "stub", "Q?", ["a", "b", "c", "d"],
                                  {"personas": ["solver"]}, [], 0.7, 64)
        check(D.extract_letter(out3[0][1], 4) == "B" and out3[0][1].endswith("ANSWER: B"),
              "prose commit gets a normalized ANSWER line", out3[0][1][-60:])
        # a reply that already commits gets no follow-up
        cli2 = FakeClient()
        cli2.chat.completions.create = lambda **kw: Resp("Reasoning. ANSWER: B")
        out2 = REAL_EXECUTE_ROUND(cli2, "stub", "Q?", ["a", "b", "c", "d"],
                                  {"personas": ["solver"]}, [], 0.7, 64)
        check(D.COMMIT_MARK not in out2[0][1], "committed reply is left alone")
        # the cache key changes with the follow-up on, and only then
        k_on = SF.path_key([{"personas": ["solver"]}])
        D.set_commit_followup(False)
        k_off = SF.path_key([{"personas": ["solver"]}])
        check(k_on != k_off and '"c":"1"' in k_on and '"c"' not in k_off,
              "follow-up is part of the cache key", k_on)
    finally:
        D.set_commit_followup(False)
        D.on_followup = None


def test_v2_summary_mode():
    print("v2: careful-reasoning prompts, long replies, summary follow-up")
    import schema_fitness as SF

    class Msg:
        def __init__(self, c): self.content = c
    class Choice:
        def __init__(self, c): self.message = Msg(c); self.finish_reason = "stop"
    class Resp:
        def __init__(self, c): self.choices = [Choice(c)]

    class FakeClient:
        """Reasoning reply for a 2-message call, summary reply for the 4-message
        nudge. `summary` may be a string or an exception to raise."""
        def __init__(self, reasoning, summary):
            self.reasoning, self.summary, self.calls = reasoning, summary, []
            self.chat = type("C", (), {})()
            self.chat.completions = type("CC", (), {})()
            self.chat.completions.create = self.create
        def create(self, **kw):
            self.calls.append(kw)
            if len(kw["messages"]) == 4:
                if isinstance(self.summary, Exception):
                    raise self.summary
                return Resp(self.summary)
            return Resp(self.reasoning)

    opts = ["a", "b", "c", "d"]
    def run(cli):
        return REAL_EXECUTE_ROUND(cli, "stub", "Q?", opts, {"personas": ["solver"]}, [], 0.7, 64)[0][1]

    long_c = ("Careful derivation. SENTINEL_DERIVATION " * 40) + "\nANSWER: C"      # > 800 chars
    long_none = ("Careful derivation that never commits. " * 40)
    orig_prompts = dict(D.PERSONA_PROMPTS); orig_expert = D.EXPERT_TMPL
    orig_critic = SF.REWRITTEN_CRITIC; orig_instr = D.ANSWER_INSTR
    k_off = SF.path_key([{"personas": ["solver"]}])
    D._think_mode.clear()
    D.reset_v2_stats()
    seen = {"n": 0}
    try:
        # off by default
        cli = FakeClient(long_c, "ANSWER: C")
        t = run(cli)
        check(len(cli.calls) == 1 and D.SUMMARY_MARK not in t, "v2 off: one call, no marker")
        check('"v"' not in k_off, "v2 off: no v marker in the key")

        SF.set_v2(True)
        D.on_summary = lambda: seen.__setitem__("n", seen["n"] + 1)
        check(D.ANSWER_INSTR_V2 in D.PERSONA_PROMPTS["solver"]
              and D.ANSWER_INSTR_V2 in D.PERSONA_PROMPTS["verifier"]
              and D.ANSWER_INSTR_V2 in SF.REWRITTEN_CRITIC
              and D.ANSWER_INSTR_V2 in D.EXPERT_TMPL.format(field="X"),
              "v2 on: persona, critic and expert prompts carry the careful-reasoning text")
        k_on = SF.path_key([{"personas": ["solver"]}])
        check(k_on.endswith(',"v":"2"}') and k_on[:-len(',"v":"2"}')] == k_off[:-1],
              "v2 marker appended last; the v1 prefix is unchanged", k_on)
        D.set_commit_followup(True)
        check(D.commit_signature() is None and '"c"' not in SF.path_key([{"personas": ["solver"]}]),
              "commit follow-up marker suppressed under v2")
        D.set_commit_followup(False)

        # the normal case: long committed reply + agreeing summary
        cli = FakeClient(long_c, "Decisive step: X.\nANSWER: C")
        t = run(cli)
        check(len(cli.calls) == 2 and D.SUMMARY_MARK in t and D.extract_letter(t, 4) == "C",
              "two calls, marker present, letter kept", D.extract_letter(t, 4))
        c2 = cli.calls[1]
        check(c2["max_tokens"] == D.SUMMARY_MAX_TOKENS and len(c2["messages"]) == 4
              and c2["messages"][2]["content"] == long_c
              and c2["messages"][3]["content"] == D.SUMMARY_NUDGE,
              "summary call: short budget, 4 messages, full reply as the assistant turn")
        check(seen["n"] == 1 and D.V2_STATS["summaries"] == 1, "summary hook and counter fired")

        # disagreement: the full reply's letter wins and the summary's last line says so
        cli = FakeClient(long_c, "I now think B.\nANSWER: B")
        t = run(cli)
        full, summ = D._split_summary(t)
        check(D.extract_letter(t, 4) == "C" and D.extract_letter(summ, 4) == "C"
              and D.V2_STATS["disagreements"] == 1,
              "disagreement: full letter wins, summary's ANSWER line rewritten", summ[-30:])

        # recovery: cut-off reply, the summary commits (strict, then lenient)
        t = run(FakeClient(long_none, "Best supported: B.\nANSWER: B"))
        check(D.extract_letter(t, 4) == "B" and D.V2_STATS["recovered_commits"] == 1,
              "recovery: summary's ANSWER line is the commit")
        t = run(FakeClient(long_none, "So the answer is **B**, given the units."))
        check(D.extract_letter(t, 4) == "B" and t.endswith("ANSWER: B")
              and D.V2_STATS["recovered_commits"] == 2,
              "recovery: prose commit normalized to an ANSWER line", t[-40:])

        # failure: summary call raises, or returns nothing -> full text only
        cli = FakeClient(long_c, RuntimeError("server hiccup"))
        t = run(cli)
        check(t == long_c and D.V2_STATS["failed_summaries"] == 1,
              "summary exception: full reply stored, nothing raised")
        t = run(FakeClient(long_c, ""))
        check(t == long_c and D.V2_STATS["failed_summaries"] == 2, "empty summary: full reply stored")

        # short committed reply: no summary call
        cli = FakeClient("Short.\nANSWER: A", "ANSWER: A")
        t = run(cli)
        check(len(cli.calls) == 1 and D.SUMMARY_MARK not in t and D.V2_STATS["skipped_short"] == 1,
              "short committed reply is its own summary")

        # what later speakers see
        D.set_digest(300, 900)
        try:
            rounds = [[("solver", long_c + D.SUMMARY_MARK + "Key fact: Y.\nANSWER: C")]]
            shown = D._visible(rounds, "all", 4)
            check("Key fact: Y." in shown and "SENTINEL_DERIVATION" not in shown,
                  "digest shows the summary, not the derivation")
            nl = D._visible(rounds, "no_letters", 4)
            check("Key fact: Y." in nl and "ANSWER" not in nl.upper().replace("PRIOR REASONING", "")
                  and "SENTINEL_DERIVATION" not in nl,
                  "no_letters shows the summary without the commitment", nl)
            check(D._visible(rounds, "letters_only", 4) == "Prior committed answers: C x1",
                  "letters_only unchanged")
        finally:
            D.set_digest(700, 0)

        # through the runner: counts, charging, and a well-formed record on disk
        D.execute_round = REAL_EXECUTE_ROUND
        try:
            rows = {"q1": {"id": "q1", "question": "Q?", "options": opts, "answer_letter": "C"}}
            runner = SF.RoundRunner(rows, base_urls="http://127.0.0.1:1/v1", model="stub",
                                    temperature=0.7, answer_tokens=64,
                                    cache_path=TMP / "v2_cache.jsonl", progress=False)
            cli = FakeClient(long_c, "Because Z.\nANSWER: C")
            runner._client = lambda: cli
            out = runner.run_round("q1", [], [], {"personas": ["solver", "solver"]}, rep=0)
            check(runner.summaries == 2 and runner.calls == 4 and len(out) == 2,
                  "runner: 2 summaries, 4 calls charged", (runner.summaries, runner.calls))
            rec = json.loads((TMP / "v2_cache.jsonl").read_text().splitlines()[-1])
            check(M.well_formed(rec) and all(D.SUMMARY_MARK in t for _, t in rec["responses"])
                  and '"v":"2"' in rec["k"], "runner: record well-formed, marked, keyed v2")
            runner.close()
        finally:
            D.execute_round = stub_execute_round
    finally:
        SF.set_v2(False)
        D.set_commit_followup(False)
        D.on_summary = None
        D.reset_v2_stats()
    check(D.PERSONA_PROMPTS == orig_prompts and D.EXPERT_TMPL == orig_expert
          and SF.REWRITTEN_CRITIC == orig_critic and D.ANSWER_INSTR == orig_instr
          and SF.path_key([{"personas": ["solver"]}]) == k_off,
          "v2 off again: prompts, critic and key restored")


if __name__ == "__main__":
    test_budget_and_cache()
    test_torn_records_are_skipped()
    test_digest_window_and_cache_key()
    test_no_eliminator()
    test_vote_read_and_commit_followup()
    test_v2_summary_mode()
    test_call_cap_and_round_kinds()
    test_eval_merge()
    test_evolve_live_end_to_end()
    print("\n" + ("FAILED: " + ", ".join(FAILURES) if FAILURES else "all checks passed"))
    sys.exit(1 if FAILURES else 0)

"""Offline tests for the per-question methods (treegrow + adaptive depth).

No network, no model, no cost. Three jobs:
  * prove the execute_schema refactor (execute_round + final_letter) reproduces
    the legacy executor exactly, including the synthesizer-unparseable -> None
    early return that keeps old cached results comparable;
  * check the new shared plumbing (committed_letters, path_key, RoundRunner
    caching/budget/error handling) on its own;
  * script the two methods' decision logic end to end with a stubbed model.

    python3 scripts/test_treegrow_adaptive.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import debate_mcq as D
import schema_fitness as SF
import treegrow_mcq as T
import adaptive_debate_mcq as A

FAILURES: list[str] = []


def check(cond, label: str, detail: str = "") -> None:
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}  {detail}")
        FAILURES.append(label)


class FakeClient:
    """OpenAI stand-in; returns canned replies in order, records prompts."""

    def __init__(self, replies: list[str]) -> None:
        self.replies, self.prompts = list(replies), []
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.prompts.append(kw["messages"])
        text = self.replies.pop(0) if self.replies else ""

        class M:
            content = text

        class C:
            message = M()

        class R:
            choices = [C()]

        return R()


ROW = {"id": "q1", "question": "Which molecule?",
       "options": ["CD40", "CD86", "B7.1", "CD80", "B7"],
       "answer_letter": "E", "field": "Immunology"}


# --- 1. the execute_schema refactor must be invisible -----------------------

def test_executor_refactor_regression() -> None:
    print("\ntest_executor_refactor_regression")
    n5 = ROW["options"]

    def run(schema, replies):
        return D.execute_schema(FakeClient(replies), "m", "Q?", n5, schema, 0.7)

    check(run(D.MINIMAL_SCHEMA, ["ANSWER: B"]) == "B", "minimal schema -> solver letter")
    check(run(D.self_critique_schema(), ["ANSWER: A", "ANSWER: C", "ANSWER: B"]) == "B",
          "self_critique -> last letter")
    check(run(D.fixed_debate_schema(2),
              ["ANSWER: A", "ANSWER: B", "ANSWER: B", "ANSWER: A", "ANSWER: C"]) == "C",
          "fixed_debate -> synthesizer letter")
    check(run(D.fixed_debate_schema(2),
              ["ANSWER: A", "ANSWER: B", "ANSWER: B", "ANSWER: A", "no idea"]) is None,
          "unparseable synthesizer STILL returns None (legacy early return)")
    check(run({"rounds": [{"personas": ["solver"] * 3}], "final": "vote"},
              ["ANSWER: A", "ANSWER: B", "ANSWER: B"]) == "B", "vote -> majority")
    check(run({"rounds": [{"personas": ["solver"]}, {"personas": ["eliminator"]}],
               "final": "last"}, ["ANSWER: D", "SURVIVING: A"]) == "D",
          "non-answering last round falls back to the previous round")

    # within a round, personas must not see each other (the pre-refactor rule)
    fc = FakeClient(["FIRST REPLY ANSWER: A", "ANSWER: B"])
    D.execute_round(fc, "m", "Q?", n5, {"personas": ["solver", "solver"]}, [], 0.7)
    check("FIRST REPLY" not in fc.prompts[1][1]["content"],
          "second persona in a round does not see the first")

    # trace form agrees with letter form
    letter, trace = D.execute_schema(FakeClient(["ANSWER: C"]), "m", "Q?", n5,
                                     D.MINIMAL_SCHEMA, 0.7, return_trace=True)
    check(letter == "C" and trace == [[("solver", "ANSWER: C")]],
          "return_trace unchanged")
    check(D.final_letter("last", [[("eliminator", "SURVIVING: A")]], 5) is None,
          "no answering persona anywhere -> None")


def test_new_personas() -> None:
    print("\ntest_new_personas")
    check("verifier" in D.PERSONA_PROMPTS and "expert" in D.PERSONA_PROMPTS,
          "prompts registered")
    check("verifier" not in D.EXT_PERSONAS and "expert" not in D.EXT_PERSONAS,
          "EXT_PERSONAS unchanged (CEM / MAP-Elites gene pools intact)")
    check(D.validate({"rounds": [{"personas": ["verifier"]}], "final": "last"}),
          "validate accepts verifier")
    check(D.validate({"rounds": [{"personas": ["expert"]}], "final": "last"}),
          "validate accepts expert")
    check(not D.validate({"rounds": [{"personas": ["oracle"]}], "final": "last"}),
          "validate still rejects unknown personas")
    p = D.EXPERT_TMPL.format(field="Immunology")
    check("Immunology" in p and "ANSWER: <letter>" in p, "expert template injects field")


# --- 2. shared plumbing -----------------------------------------------------

def test_committed_letters_and_path_key() -> None:
    print("\ntest_committed_letters_and_path_key")
    rounds = [[("solver", "ANSWER: C")],
              [("eliminator", "SURVIVING: B, E"), ("critic", "ANSWER: E")],
              [("solver", "mumble")]]
    check(SF.committed_letters(rounds, 5) == ["C", "E"],
          "skips eliminator and unparseable", str(SF.committed_letters(rounds, 5)))

    r1 = [{"personas": ["solver"], "sees": "all"}]
    r2 = [{"personas": ["solver"]}]
    check(SF.path_key(r1) == SF.path_key(r2), "default sees omitted from the key")
    check(SF.path_key(r1, {"critic": "x"}) != SF.path_key(r1),
          "prompt overrides change the key")
    both = r2 + [{"personas": ["critic"], "sees": "none"}]
    check(SF.path_key(both) != SF.path_key(r2), "longer path -> different key")
    check(json.loads(SF.path_key(both))["rounds"][1] == {"personas": ["critic"], "sees": "none"},
          "non-default sees preserved")


def scripted(fn):
    """Monkeypatch D.execute_round with fn(round_spec, all_rounds); count calls."""
    calls = {"n": 0}

    def fake(client, model, question, options, round_spec, all_rounds,
             temperature, max_tokens=3072, prompts=None):
        calls["n"] += 1
        return fn(round_spec, all_rounds)

    return fake, calls


def by_persona(letters: dict):
    """Standard stub: each persona answers its scripted letter, eliminator strikes."""

    def fn(round_spec, all_rounds):
        out = []
        for p in round_spec["personas"]:
            if p in D.NON_ANSWERING:
                out.append((p, "SURVIVING: B, E"))
            else:
                l = letters.get(p)
                out.append((p, "" if l is None else f"reasoning\nANSWER: {l}"))
        return out

    return fn


def make_runner(tmp, max_calls=None):
    return SF.RoundRunner({ROW["id"]: ROW}, "http://localhost:1/v1", "m", 0.7, 64,
                          Path(tmp) / "cache.jsonl", max_calls=max_calls,
                          progress=False)


def test_round_runner_resume_and_errors() -> None:
    print("\ntest_round_runner_resume_and_errors")
    real = D.execute_round
    try:
        with tempfile.TemporaryDirectory() as tmp:
            fake, calls = scripted(by_persona({"solver": "C"}))
            D.execute_round = fake
            r = make_runner(tmp)
            spec = {"personas": ["solver", "solver"]}
            out = r.run_round("q1", [], [], spec)
            check(out == [("solver", "reasoning\nANSWER: C")] * 2, "round executed")
            check(r.calls == 2, "charged one call per persona", str(r.calls))
            out2 = r.run_round("q1", [], [], spec)
            check(out2 == out and r.calls == 2 and calls["n"] == 1,
                  "identical round is a cache hit -- no charge, no execution")
            r.run_round("q1", [out], [spec], {"personas": ["solver"]})
            check(r.calls == 3 and calls["n"] == 2, "longer path pays only the new round")
            r.close()

            def boom(*a, **k):
                raise RuntimeError("server down")
            D.execute_round = boom
            r2 = make_runner(tmp)
            check(r2.run_round("q1", [], [], spec) ==
                  [("solver", "reasoning\nANSWER: C")] * 2,
                  "cached round survives a resume (no re-execution)")
            bad = r2.run_round("q1", [], [], {"personas": ["critic"]})
            check(bad == [] and r2.errors == 1, "errored round returns [] and is counted")
            r2.close()

            r3 = make_runner(tmp)
            keys = set(r3._cache)
            check(all("critic" not in k[1] for k in keys),
                  "errored round NOT cached -- retried on the next run", str(keys))
            r3.close()

            r4 = make_runner(tmp, max_calls=1)
            try:
                # three solvers: NOT in the cache, so the charge must happen and trip
                r4.run_round("q1", [], [], {"personas": ["solver"] * 3})
                check(False, "budget enforced")
            except SF.BudgetExhausted:
                check(True, "budget enforced")
            r4.close()
    finally:
        D.execute_round = real


# --- 3. adaptive depth ------------------------------------------------------

def test_confirmed_switch() -> None:
    print("\ntest_confirmed_switch")
    f = A.is_confirmed_switch
    check(f([(0, "C"), (0, "C"), (1, "E"), (2, "E")], "C"), "post-round-1 pair, new letter")
    check(not f([(0, "C"), (0, "D"), (0, "D")], "C"), "round-1 pair alone never stops")
    check(not f([(0, "C"), (1, "E"), (2, "D")], "C"), "unconfirmed flip does not stop")
    check(not f([(0, "C"), (1, "C"), (2, "C")], "C"), "agreeing WITH baseline does not stop")
    check(not f([(0, "C"), (1, "E")], "C"), "single post-round-1 commit is not enough")
    check(not f([(0, "C"), (0, "D"), (1, "D")], "C"),
          "pair spanning round 1 does not stop (round-1 letters are opinions)")
    check(A.modal_letter(["C", "D", "C", "D"]) == "C", "modal tie -> earliest seen")


def run_adaptive(tmp, per_round_letters):
    """Drive run_question with a stub whose answers depend on how many rounds
    exist so far: per_round_letters[i] maps persona -> letter for round i."""

    def fn(round_spec, all_rounds):
        book = per_round_letters[len(all_rounds)]
        out = []
        for p in round_spec["personas"]:
            if p in D.NON_ANSWERING:
                out.append((p, "SURVIVING: B, E"))
            else:
                l = book.get(p) if isinstance(book, dict) else book
                out.append((p, "" if l is None else f"reasoning\nANSWER: {l}"))
        return out

    fake, calls = scripted(fn)
    real = D.execute_round
    D.execute_round = fake
    try:
        r = make_runner(tmp)
        args = types.SimpleNamespace(rep=0)
        rec = A.run_question(r, ROW, args)
        r.close()
        return rec
    finally:
        D.execute_round = real


def test_adaptive_tracks() -> None:
    print("\ntest_adaptive_tracks")
    with tempfile.TemporaryDirectory() as tmp:
        # settled early: solvers C, reader E, critic E -> stop after master round 3
        rec = run_adaptive(tmp + "/a", ["C", "E", "E", "X", "X"])
        check(rec["track"] == "settled_early" and rec["letter"] == "E" and rec["correct"],
              "confirmed switch trims the recipe", str(rec))
        check(rec["n_calls"] == 6, "settled early = 6 speakers, not 8", str(rec["n_calls"]))

        # parked -> extension -> expert+solver agree
        seq = ["C", "C", "C", "C", "C",
               None,                                    # eliminator (no letter anyway)
               {"expert": "E", "solver": "E"},          # newcomers agree
               "X"]
        rec = run_adaptive(tmp + "/b", seq)
        check(rec["track"] == "parked" and rec["closer"] == "newcomers_agree"
              and rec["letter"] == "E", "parked: agreeing newcomers close", str(rec))
        check("eliminator|blind" in rec["schema"] and "expert+solver|lastrnd" in rec["schema"],
              "parked extension appended with the right visibility", rec["schema"])

        # parked -> newcomers disagree -> verifier closes
        seq = ["C", "C", "C", "C", "C", None,
               {"expert": "E", "solver": "B"}, {"verifier": "E"}]
        rec = run_adaptive(tmp + "/c", seq)
        check(rec["closer"] == "verifier" and rec["letter"] == "E" and rec["n_calls"] == 12,
              "parked: verifier closes at 12 speakers", str(rec))

        # parked where even the newcomers keep the baseline -> baseline survives
        seq = ["C", "C", "C", "C", "C", None,
               {"expert": "C", "solver": "C"}, "X"]
        rec = run_adaptive(tmp + "/d", seq)
        check(rec["letter"] == "C" and rec["closer"] == "newcomers_agree",
              "parked: answer that survives fresh attackers is kept", str(rec))

        # churn -> verifier backs an already-committed letter
        rec = run_adaptive(tmp + "/e", ["C", "E", "D", "B", "C", {"verifier": "E"}, "X"])
        check(rec["track"] == "churn" and rec["closer"] == "verifier_backed"
              and rec["letter"] == "E", "churn: verifier backing stops it", str(rec))

        # churn -> verifier novel letter -> synthesizer closes
        rec = run_adaptive(tmp + "/f",
                           ["C", "E", "D", "C", "E",     # E never consecutive
                            {"verifier": "A"}, {"synthesizer": "E"}])
        check(rec["track"] == "churn" and rec["closer"] == "synthesizer"
              and rec["letter"] == "E", "churn: synthesizer closes", str(rec))

        # a failed extension round degrades to the verifier instead of crashing
        def flaky(round_spec, all_rounds):
            if "expert" in round_spec["personas"]:
                raise RuntimeError("server down")
            if round_spec["personas"] == ["eliminator"]:
                return [("eliminator", "SURVIVING: B, E")]
            letter = "C" if len(all_rounds) < 5 else "E"
            return [(p, f"ANSWER: {letter}") for p in round_spec["personas"]]

        fake, _ = scripted(flaky)
        real = D.execute_round
        D.execute_round = fake
        try:
            r = make_runner(tmp + "/g")
            rec = A.run_question(r, ROW, types.SimpleNamespace(rep=0))
            check(r.errors == 1, "failed round counted")
            r.close()
        finally:
            D.execute_round = real
        check(rec["track"] == "parked" and rec["closer"] == "verifier"
              and rec["letter"] == "E", "errored pair round degrades to verifier", str(rec))


def test_fixed_control_shares_cache() -> None:
    print("\ntest_fixed_control_shares_cache")
    real = D.execute_round

    def fn(round_spec, all_rounds):
        i = len(all_rounds)
        letter = ["C", "E", "E", "C", "C"][i] if i < 5 else "X"
        return [(p, f"ANSWER: {letter}") for p in round_spec["personas"]]

    fake, calls = scripted(fn)
    D.execute_round = fake
    try:
        with tempfile.TemporaryDirectory() as tmp:
            r = make_runner(tmp)
            args = types.SimpleNamespace(rep=0)
            rec = A.run_question(r, ROW, args)            # settles after 3 master rounds
            check(rec["track"] == "settled_early" and calls["n"] == 3, "adaptive ran 3 rounds")
            fx = A.run_fixed(r, ROW, args)
            check(calls["n"] == 5, "fixed control re-used the 3 cached prefix rounds",
                  str(calls["n"]))
            check(fx["letter"] == "C", "fixed recipe = last critic's letter")
            r.close()
    finally:
        D.execute_round = real


# --- 4. treegrow ------------------------------------------------------------

TREE_LETTERS = {"solver": "C", "critic": "C", "verifier": "E",
                "independent": "D", "expert": "B"}


def tree_args(**kw):
    base = dict(rep=0, depth=2, beam=2, plain_critic=False, inner_workers=1,
                seed=0, lessons=4, controller_temperature=0.2,
                verify_reps=3, verify_min=2)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_treegrow_search() -> None:
    print("\ntest_treegrow_search")
    real = D.execute_round
    fake, calls = scripted(by_persona(TREE_LETTERS))
    D.execute_round = fake
    try:
        with tempfile.TemporaryDirectory() as tmp:
            r = make_runner(tmp)
            rec, lessons = T.search_question(r, ROW, tree_args())
            check(rec["solved"] and rec["solve_depth"] == 1
                  and rec["winning_path"] == ["verifier"],
                  "verifier path wins at depth 1", str(rec))
            check(len(lessons) == 5 + 2 * 5, "5 depth-1 + 2 beam x 5 depth-2 lessons",
                  str(len(lessons)))
            d1 = {l["action"]: l for l in lessons if l["depth"] == 1}
            check(d1["verifier"]["child_solved"] and d1["verifier"]["subtree_solved"],
                  "winning move recorded as solved")
            check(d1["verifier"]["raw_hit"] and d1["verifier"]["rerun_hits"] == 3,
                  "hit verified by 3 fresh reruns", str(d1["verifier"]))
            check(rec["winning_rerun_hits"] == 3 and rec["n_flukes"] == 0,
                  "record carries verification counts", str(rec))
            check(all(l["rerun_hits"] == 0 for l in lessons if not l["raw_hit"]),
                  "misses are never re-run")
            check(not d1["critic"]["child_solved"] and d1["critic"]["subtree_solved"],
                  "dead move whose subtree later solves is backfilled")
            check(not d1["expert"]["subtree_solved"],
                  "unexpanded dead branch stays failed")
            check(all(l["pattern"].startswith("solver C") for l in d1.values()),
                  "lesson pattern is the parent state")
            r.close()

            # a solved root is a leaf: no expansions, no lessons
            fake2, _ = scripted(by_persona({"solver": "E"}))
            D.execute_round = fake2
            r2 = make_runner(tmp + "/root")
            rec2, lessons2 = T.search_question(r2, ROW, tree_args())
            check(rec2["solved"] and rec2["solve_depth"] == 0 and lessons2 == [],
                  "root already right -> stop immediately", str(rec2))
            r2.close()
    finally:
        D.execute_round = real


def test_treegrow_fluke_demotion() -> None:
    """A hit that fails verification must not count, must be recorded as a
    fluke, and must stay in the beam so the search keeps growing from it."""
    print("\ntest_treegrow_fluke_demotion")
    real = D.execute_round
    seen = {"verifier": 0}

    def fn(round_spec, all_rounds):
        out = []
        for p in round_spec["personas"]:
            if p == "verifier":
                seen["verifier"] += 1
                letter = "E" if seen["verifier"] == 1 else "C"   # lucky once, then wrong
            elif p == "eliminator":
                out.append((p, "SURVIVING: B, E")); continue
            else:
                letter = {"solver": "C", "critic": "C", "independent": "D",
                          "expert": "B"}[p]
            out.append((p, f"ANSWER: {letter}"))
        return out

    fake, _ = scripted(fn)
    D.execute_round = fake
    try:
        with tempfile.TemporaryDirectory() as tmp:
            r = make_runner(tmp)
            rec, lessons = T.search_question(r, ROW, tree_args(depth=1, beam=2))
            v = [l for l in lessons if l["action"] == "verifier"][0]
            check(v["raw_hit"] and v["rerun_hits"] == 0 and not v["child_solved"],
                  "lucky hit fails verification -> not solved", str(v))
            check(not rec["solved"] and rec["raw_solved"] and rec["n_flukes"] == 1,
                  "question recorded as raw-solved but not confirmed", str(rec))
            check(seen["verifier"] == 4, "path re-run exactly verify_reps=3 times",
                  str(seen))
            r.close()

            # with verification off, the same fluke counts (old behavior)
            seen["verifier"] = 0
            r2 = make_runner(tmp + "/off")
            rec2, _ = T.search_question(r2, ROW, tree_args(depth=1, beam=2, verify_reps=0))
            check(rec2["solved"] and rec2["winning_path"] == ["verifier"],
                  "--verify-reps 0 restores unverified behavior", str(rec2))
            r2.close()

            # demoted node stays eligible: with depth 2 it gets expanded further
            seen["verifier"] = 0
            r3 = make_runner(tmp + "/grow")
            # beam=2 keeps the first two distinct letters: critic (C) and the
            # demoted verifier node (E) -- it competes like any unsolved child
            rec3, lessons3 = T.search_question(r3, ROW, tree_args(depth=2, beam=2))
            grown = [l for l in lessons3 if l["depth"] == 2 and l["path_actions"] == ["verifier"]]
            check(len(grown) == 5, "demoted fluke node stays in the beam and is expanded",
                  str(len(grown)))
            check(not rec3["solved"], "still unsolved after growing from the fluke",
                  str(rec3))
            r3.close()
    finally:
        D.execute_round = real


def test_select_beam_and_state_type() -> None:
    print("\ntest_select_beam_and_state_type")

    def node(letter):
        resp = [] if letter is None else [[("solver", f"ANSWER: {letter}")]]
        return T.Node(specs=[], rounds=resp, actions=[])

    picked = T.select_beam([node("C"), node("C"), node("D"), node(None)], 2, 5)
    letters = [T.standing(c.rounds, 5) for c in picked]
    check(letters == ["C", "D"], "distinct letters preferred", str(letters))
    picked = T.select_beam([node(None), node("C")], 2, 5)
    check(T.standing(picked[0].rounds, 5) == "C", "parseable before None")

    check(T.state_type([[("solver", "ANSWER: C")]] * 3, 5) == "parked", "parked")
    check(T.state_type([[("solver", "ANSWER: C")], [("critic", "ANSWER: E")]], 5)
          == "moved", "moved")
    check(T.state_type([[("solver", "ANSWER: C")], [("critic", "ANSWER: E")],
                        [("solver", "ANSWER: A")]], 5) == "scattered", "scattered")


def test_controller() -> None:
    print("\ntest_controller")
    check(T.parse_choice("I would run the Verifier next.") == "verifier", "word in prose")
    check(T.parse_choice("eliminate") == "eliminate", "exact word")
    check(T.parse_choice("STOP") == "stop", "case-insensitive")
    check(T.parse_choice("try the eliminator") == "eliminate", "eliminator matches eliminate")
    check(T.parse_choice("hmm") is None and T.parse_choice(None) is None, "garbage -> None")

    lesson = {"pattern": "solver C; critic C", "action": "verifier",
              "child_letter": "E", "subtree_solved": True}
    check("SUCCEEDED" in T.render_lesson(lesson) and "verifier" in T.render_lesson(lesson),
          "lesson renders")

    bank = {(1, "parked"): [dict(lesson, subtree_solved=s) for s in (True, False) * 4]}
    import random as _r
    sel = T.pick_lessons(bank, 1, "parked", 4, _r.Random(0))
    check(len(sel) == 4 and sum(l["subtree_solved"] for l in sel) == 2,
          "balanced worked/failed", str(len(sel)))
    check(len(T.pick_lessons(bank, 3, "parked", 4, _r.Random(0))) == 4,
          "missing depth falls back to same state type")

    real = D.execute_round
    fake, _ = scripted(by_persona(TREE_LETTERS))
    D.execute_round = fake
    try:
        with tempfile.TemporaryDirectory() as tmp:
            r = make_runner(tmp)
            ctrl = FakeClient(["verifier", "stop"])
            rec = T.run_controller_question(r, ctrl, "m", ROW, bank, tree_args(depth=4))
            check(rec["picks"] == ["verifier", "stop"] and rec["correct"]
                  and rec["letter"] == "E", "controller drives to the right answer",
                  str(rec))
            check(rec["persona_calls"] == 2 and rec["controller_calls"] == 2,
                  "call accounting", str(rec))

            ctrl2 = FakeClient(["nonsense", "still nonsense", "stop"])
            rec2 = T.run_controller_question(r, ctrl2, "m", ROW, bank, tree_args(depth=2))
            check(rec2["picks"][0] == "critic", "double-garbage defaults to critic",
                  str(rec2["picks"]))
            r.close()
    finally:
        D.execute_round = real


if __name__ == "__main__":
    test_executor_refactor_regression()
    test_new_personas()
    test_committed_letters_and_path_key()
    test_round_runner_resume_and_errors()
    test_confirmed_switch()
    test_adaptive_tracks()
    test_fixed_control_shares_cache()
    test_treegrow_search()
    test_treegrow_fluke_demotion()
    test_select_beam_and_state_type()
    test_controller()
    print("\n" + ("FAILED: " + ", ".join(FAILURES) if FAILURES else "all checks passed"))
    sys.exit(1 if FAILURES else 0)

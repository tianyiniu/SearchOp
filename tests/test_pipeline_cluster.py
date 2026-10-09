"""Offline checks for the cluster pipeline's per-group search (sample_cluster_subsets,
program_seeds_cluster, evolve_pipeline_cluster) under the v3 executor (effort
and visibility per round, no reply cap, answer-locked summaries, blind
speakers cached one by one). Reuses the fake guide model of
test_program_clusters.py and gives its fake debate model the v3 protocol, so
the executor, the round cache and its keys run exactly as they would live. No
vLLM, no API key.

    python tests/test_pipeline_cluster.py
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import types
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))   # the code under test
sys.path.insert(0, str(Path(__file__).resolve().parent))                       # the other test files

import test_program_clusters as T  # noqa: E402  (installs the fake clients on import)
import debate_mcq as D  # noqa: E402
import evolve_program_mcq as M  # noqa: E402
import schema_fitness as SF  # noqa: E402
import program_guide as G  # noqa: E402
import program_space as P  # noqa: E402
import evolve_pipeline_cluster as V  # noqa: E402
import program_seeds_cluster as S3  # noqa: E402
import sample_cluster_subsets as SUB  # noqa: E402

check, run_cli, TMP = T.check, T.run_cli, T.TMP
CLUSTERS_V3 = TMP / "clusters_v3.json"
CACHE = TMP / "rounds_v3.jsonl"
WINDOW = ["--context-window", "65536"]


# --- the fake debate model, v3 protocol -------------------------------------------------
# Replies come in four shapes, fixed by a hash of the conversation: a short committed
# reply (shown as it is), a long committed one (summarised), one that never commits
# (letter-first summary), and a bare letter with hidden reasoning (summarised from the
# reasoning). High effort adds hidden reasoning to every shape.

class _Msg:
    def __init__(self, content, reasoning=None):
        self.content, self.reasoning_content = content, reasoning


class _Resp:
    def __init__(self, content, reasoning=None):
        self.choices = [types.SimpleNamespace(message=_Msg(content, reasoning), finish_reason="stop")]
        self.usage = types.SimpleNamespace(prompt_tokens=100,
                                           completion_tokens=len(((content or "") + (reasoning or "")).split()))


V3_CALLS: Counter = Counter()
REQUESTS: list[dict] = []
_old_create = T.FakeCompletions.create


def _create(self, model, messages, temperature=None, max_tokens=None, extra_body=None, **kw):
    if not D.V3:
        return _old_create(self, model, messages, temperature, max_tokens, extra_body, **kw)
    REQUESTS.append({"max_tokens": max_tokens, "temperature": temperature, "extra_body": extra_body, **kw})
    last = messages[-1]["content"]
    if last.startswith("Now write a summary"):
        V3_CALLS["summary"] += 1
        letter = re.search(r"Your answer is final: ([A-J])", last).group(1)
        return _Resp(f"The decisive step points one way.\nANSWER: {letter}")
    if last.startswith("Your reply ended before"):
        V3_CALLS["letter_first"] += 1
        return _Resp("ANSWER: B\nThe reasoning so far favoured B.")
    if last == D.COMMIT_NUDGE:
        V3_CALLS["commit"] += 1
        return _Resp("ANSWER: C")
    T.MODEL_CALLS["chat"] += 1
    T.FakeCompletions.salt += 1
    high = (extra_body or {}).get("chat_template_kwargs", {}).get("enable_thinking", False)
    V3_CALLS["high" if high else "low"] += 1
    h = int(hashlib.sha1(f"{messages[0]['content'][:30]}|{last}|{T.FakeCompletions.salt}".encode()).hexdigest(), 16)
    letter, shape = "ABCD"[h % 4], (h // 4) % 4
    thinking = "Private reasoning step. " * 60 if high else None
    if shape == 0:
        return _Resp("A short argument. " * 30 + f"\nANSWER: {letter}", thinking)
    if shape == 1:
        return _Resp("One long derivation step. " * 150 + f"\nANSWER: {letter}", thinking)
    if shape == 2:
        return _Resp("Still working through the options without finishing. " * 20, thinking)
    return _Resp(f"ANSWER: {letter}", "Hidden reasoning. " * 40)


T.FakeCompletions.create = _create

# the fake guide learns the one-program-per-group request
_old_parse = T.FakeResponses.parse


GUIDE_REQUESTS: list[dict] = []          # what the seed writer was sent


def _parse(self, model, instructions, input, reasoning, text_format, max_output_tokens):
    if text_format is G.GroupSeedBatch:
        self.calls += 1
        GUIDE_REQUESTS.append({"instructions": instructions, "input": input})
        todo = json.loads(input.split("for each of these groups: ", 1)[1].split(".", 1)[0])
        n = int(re.search(r"Write (\d+) new program", input).group(1))
        progs = []
        for g in todo:
            for i in range(n):
                child, _ = P.mutate_uniform(P.PROTOCOLS["fresh_on_disagree"], self.rng)
                progs.append(G.GroupSeedProgram(group=g, name=f"fake_{g}_{i}", strategy="a fake variant",
                                                program=G.Program(**child)))
        return T.FakeParsed(G.GroupSeedBatch(programs=progs))
    return _old_parse(self, model, instructions, input, reasoning, text_format, max_output_tokens)


T.FakeResponses.parse = _parse


def test_statistics():
    print("statistics")
    mean, se = V.paired_se([1, 1, 0, 0], [1, 0, 0, 1])
    check(abs(mean) < 1e-12 and abs(se - (2 / 3) ** 0.5 / 2) < 1e-9, "paired standard error", f"{mean}, {se:.4f}")
    pct = V.percentile_ranks({"a": 0.1, "b": 0.3, "c": 0.3, "d": 0.5})
    check(pct["a"] < pct["b"] == pct["c"] < pct["d"] and abs(pct["b"] - 0.5) < 1e-9,
          "percentile ranks share ties", str(pct))


def test_edits():
    print("uniform edits")
    rng = random.Random(1)
    direct = P.normalize_program(P.PROTOCOLS["direct"])
    kinds = P.applicable_edits(direct)
    check("drop_rule" not in kinds and "swap_rules" not in kinds and "plan_drop" not in kinds
          and "plan_add" in kinds and "set_effort" in kinds and "set_visibility" not in kinds
          and "crossover" not in kinds, "inapplicable kinds are left out", str(kinds))
    check("crossover" in P.applicable_edits(direct, [P.PROTOCOLS["mad"]]), "crossover needs a donor")
    seen = Counter()
    prog = P.normalize_program(P.PROTOCOLS["early_exit_agree"])
    donors = [P.PROTOCOLS["mad"], P.PROTOCOLS["fresh_on_disagree_high"]]
    for _ in range(1300):
        child, op = P.mutate_uniform(prog, rng, donors=donors)
        P.validate_program(child)
        seen[op] += 1
    n_kinds = len(P.applicable_edits(prog, donors))
    check(set(seen) == set(P.applicable_edits(prog, donors)), "every applicable kind is drawn", str(dict(seen)))
    check(min(seen.values()) > 0.6 * 1300 / n_kinds, "kinds are drawn about equally", str(dict(seen)))
    old = random.Random(3)
    new = random.Random(3)
    same = all(P.mutate_plan(prog, old) == P.mutate_plan(prog, new, None) for _ in range(20))
    check(same, "the v2 plan edit is unchanged when no kind is named")


# --- the v3 executor ----------------------------------------------------------------------

class _Scripted:
    """A client that returns the given responses in order (an Exception is
    raised instead) and keeps every request."""

    def __init__(self, *resps):
        self.resps, self.requests = list(resps), []
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, model, messages, max_tokens=None, **kw):
        self.requests.append({"messages": messages, "max_tokens": max_tokens, **kw})
        r = self.resps.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_executor_v3():
    print("v3 executor")
    GPT, QWEN = "openai/gpt-oss-20b", "Qwen/Qwen3.5-35B-A3B-FP8"
    short = "A short argument. " * 30 + "\nANSWER: D"
    long = "One long derivation step. " * 150 + "\nANSWER: D"
    unfinished = "Still weighing the options. " * 150

    def stats():
        return dict(D.V2_STATS)

    c = _Scripted(_Resp(short))
    out = D.chat_v3(c, GPT, "sys", "question", 10, "low")
    req = c.requests[0]
    est = D._est_tokens(req["messages"])
    check(out == short.strip() and len(c.requests) == 1, "a committed reply within 500 words is shown as it is")
    check(req["extra_body"] == {"reasoning_effort": "low"} and req["temperature"] == 1.0 and req["top_p"] == 1.0,
          "gpt-oss low effort: reasoning_effort low, temperature 1, top_p 1", str(req))
    check(req["max_tokens"] == D.WINDOW - est - D.SUMMARY_RESERVE, "a reply may use the window less the reserve",
          f"{req['max_tokens']} vs {D.WINDOW} - {est} - {D.SUMMARY_RESERVE}")

    c = _Scripted(_Resp(long), _Resp("The step that decides it.\nANSWER: D"))
    out = D.chat_v3(c, QWEN, "sys", "question", 10, "low")
    nudge = c.requests[1]["messages"][-1]["content"]
    check(D.SUMMARY_MARK in out and D.extract_letter(out, 10) == "D" and "Your answer is final: D" in nudge
          and "must not reconsider, change or argue against" in nudge, "a long reply is summarised, answer locked")
    q = c.requests[0]
    check(q["extra_body"] == {"top_k": 40, "min_p": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
          and q["presence_penalty"] == 2.0 and q["top_p"] == 1.0, "Qwen low effort: thinking off, card sampling",
          str(q))
    check(c.requests[1]["max_tokens"] == D.SUMMARY_TOKENS_V3 and c.requests[1]["extra_body"]["chat_template_kwargs"]
          == {"enable_thinking": False}, "the summary call is low effort with its own limit")

    before = stats()
    c = _Scripted(_Resp(long), _Resp("On reflection B is better.\nANSWER: B"))
    out = D.chat_v3(c, QWEN, "sys", "question", 10, "low")
    check(out == long.strip() and D.extract_letter(out, 10) == "D"
          and stats()["rejected_summaries"] == before["rejected_summaries"] + 1,
          "a summary naming another letter is not used; the reply's letter stands")
    c = _Scripted(_Resp(long), _Resp("On reflection B is better.\n**Answer:** B"))
    out = D.chat_v3(c, QWEN, "sys", "question", 10, "low")
    check(out == long.strip(), "... also when it names it in markdown")
    c = _Scripted(_Resp(long), _Resp("The key step.\n**ANSWER: D**"))
    out = D.chat_v3(c, QWEN, "sys", "question", 10, "low")
    check(out.endswith("The key step.\nANSWER: D") and D.extract_letter(out, 10) == "D",
          "an agreeing summary keeps one plain ANSWER line", repr(out[-40:]))

    c = _Scripted(_Resp("A short argument. " * 30 + "\n\n**Answer:** H"))
    out = D.chat_v3(c, GPT, "sys", "question", 10, "low")
    check(len(c.requests) == 1 and D.extract_letter(out, 10) == "H", "a commitment in markdown counts as committed")

    c = _Scripted(_Resp(unfinished), _Resp("ANSWER: E\nThe reasoning so far pointed to E."))
    out = D.chat_v3(c, QWEN, "sys", "question", 10, "low")
    check(D.extract_letter(out, 10) == "E" and c.requests[1]["messages"][-1]["content"].startswith(
          "Your reply ended before"), "an uncommitted reply commits letter-first in its summary")

    before = stats()
    c = _Scripted(_Resp(unfinished), _Resp("The reasoning compared the options."), _Resp("I pick option G."))
    out = D.chat_v3(c, QWEN, "sys", "question", 10, "low")
    check(D.extract_letter(out, 10) == "G" and c.requests[2]["messages"][-1]["content"] == D.COMMIT_NUDGE
          and stats()["nudged_commits"] == before["nudged_commits"] + 1,
          "no letter in the summary: the commit prompt, read leniently")
    c = _Scripted(_Resp("The derivation says the answer is B, probably. " * 60 + "\nANSWER: nothing"),
                  _Resp("Summary without a letter; option F was ruled out."), _Resp(""))
    out = D.chat_v3(c, QWEN, "sys", "question", 10, "low")
    check(D.extract_letter(out, 10) is None, "a letter is never read from a summary's prose")

    c = _Scripted(_Resp("ANSWER: A", "hidden step " * 3000), _Resp("It follows from step two.\nANSWER: A"))
    out = D.chat_v3(c, GPT, "sys", "question", 10, "high")
    shown = c.requests[1]["messages"][2]["content"]
    check(c.requests[0]["extra_body"] == {"reasoning_effort": "high"} and D.SUMMARY_MARK in out
          and shown.startswith("[the end of my private reasoning]") and len(shown) < D.THINKING_TAIL + 200,
          "high effort: always summarised, from the end of its hidden reasoning")
    c = _Scripted(_Resp("ANSWER: A", "hidden step " * 50), _Resp("It follows from step two.\nANSWER: A"))
    D.chat_v3(c, GPT, "sys", "question", 10, "low")
    check(len(c.requests) == 2 and "[the end of my private reasoning]" in c.requests[1]["messages"][2]["content"],
          "a bare-letter reply is summarised from its hidden reasoning")
    c = _Scripted(_Resp(long), _Resp(None, "The summary came back in the reasoning field.\nANSWER: D"))
    before = stats()
    out = D.chat_v3(c, QWEN, "sys", "question", 10, "high")
    check(out == long.strip() and stats()["failed_summaries"] == before["failed_summaries"] + 1,
          "hidden reasoning returned in place of a summary is a failed summary")
    q = c.requests[0]
    check(q["extra_body"] == {"top_k": 20, "min_p": 0.0, "chat_template_kwargs": {"enable_thinking": True}}
          and q["presence_penalty"] == 1.5 and q["top_p"] == 0.95, "Qwen high effort: thinking on, card sampling")

    err = Exception("This model's maximum context length is 65536 tokens. However, you requested 70000 tokens "
                    "(9000 in the messages, 61000 in the completion).")
    c = _Scripted(err, _Resp(short))
    D.chat_v3(c, GPT, "sys", "question", 10, "low")
    check(c.requests[1]["max_tokens"] == D.WINDOW - 9000 - D.SUMMARY_RESERVE,
          "a request over the window is sent again with the server's prompt size", str(c.requests[1]["max_tokens"]))
    err = Exception("This model's maximum context length is 65536 tokens. However, you requested 61000 output "
                    "tokens and your prompt contains 9000 input tokens, for a total of 70000 tokens. Please "
                    "reduce the length of the input prompt or the number of requested output tokens.")
    c = _Scripted(err, _Resp(short))                # vLLM 0.18's wording (the servers' version)
    D.chat_v3(c, GPT, "sys", "question", 10, "low")
    check(len(c.requests) > 1 and c.requests[1]["max_tokens"] == D.WINDOW - 9000 - D.SUMMARY_RESERVE,
          "vLLM 0.18's refusal is read too, and the request sent again", str([r["max_tokens"] for r in c.requests]))
    check(D.spec_cost({"personas": ["solver"] * 3, "effort": "high"}) == 15
          and D.spec_cost({"personas": ["critic", "verifier"]}) == 2, "a high-effort speaker costs 5 turns")


def test_cache_v3():
    print("v3 round cache")
    rows = {r["id"]: r for r in json.loads(T.DATASET.read_text())}
    q = sorted(rows)[0]
    cache = TMP / "cache_v3_unit.jsonl"
    runner = P.make_runner(rows, cache)
    runner.reset_budget(None)
    k1 = SF.path_key([P.PLAN_ROUNDS["solver_x2|high"], P.PLAN_ROUNDS["critic|high"]])
    check('"effort":"high"' in k1 and '"v":"3"' in k1 and '"w":65536' in k1, "path keys carry effort, v3, window", k1)
    check(SF.path_key([dict(P.PLAN_ROUNDS["solver_x2"], sees="none"), P.PLAN_ROUNDS["critic"]])
          == SF.path_key([P.PLAN_ROUNDS["solver_x2"], P.PLAN_ROUNDS["critic"]]),
          "the first round's visibility is not part of the key")
    n0 = T.MODEL_CALLS["chat"]
    direct = M.run_program(P.PROTOCOLS["direct"], runner, rows[q], rep=0, max_calls=16)
    n1 = T.MODEL_CALLS["chat"]
    sc = M.run_program(P.PROTOCOLS["self_consistency"], runner, rows[q], rep=0, max_calls=16)
    n2 = T.MODEL_CALLS["chat"]
    fresh_prog = {"plan": [P.PLAN_ROUNDS["solver_x2"]], "rules": [P.CONT(), {"when": ["step==1"], "do": "solver_x2|blind"}],
                  "default": "stop:vote"}
    fresh = M.run_program(fresh_prog, runner, rows[q], rep=0, max_calls=16)
    n3 = T.MODEL_CALLS["chat"]
    check((n1 - n0, n2 - n1, n3 - n2) == (1, 3, 0),
          "blind speakers are shared: direct is SC's first solver, a later blind pair is its last two",
          f"{(n1 - n0, n2 - n1, n3 - n2)}")
    check(fresh["letter"] == sc["letter"] and direct["tokens"] > 0 and sc["tokens"] >= direct["tokens"],
          "the same four samples give the same vote; tokens are counted", f"{fresh['letter']} {sc['letter']}")
    M.run_program(P.PROTOCOLS["direct"], runner, rows[q], rep=1, max_calls=16)
    check(T.MODEL_CALLS["chat"] == n3 + 1, "another replicate draws a fresh sample")
    M.run_program(P.PROTOCOLS["direct_high"], runner, rows[q], rep=0, max_calls=16)
    check(T.MODEL_CALLS["chat"] == n3 + 2, "effort is part of a blind speaker's key")
    runner.close()
    replay = M.CacheRunner([cache])
    again = M.run_program(P.PROTOCOLS["self_consistency"], replay, rows[q], rep=0, max_calls=16)
    check(again["letter"] == sc["letter"] and again["actions"] == sc["actions"], "a replay reads the same debate")
    lines = [json.loads(l) for l in cache.open()]
    check(all(len(d["responses"]) == 1 for d in lines if '"blind"' in d["k"]),
          "blind speakers are recorded one by one")


def test_grammar_v3():
    print("v3 grammar")
    vocab = P.v3_vocabulary()
    check("solver_x4|high" not in vocab and "critic|blind" not in vocab and "solver_x3|high|blind" in vocab
          and all(D.spec_cost(s) <= 16 for s in vocab.values()), "no round over the cap, blind only for answerers",
          str(len(vocab)))
    bad = [({"plan": [P.PLAN_ROUNDS["solver_x2"]], "rules": [{"when": ["r1_majority>=3"], "do": "stop:vote"}],
             "default": "stop:vote"}, "narrow opening"),
           ({"plan": [P.PLAN_ROUNDS["solver"]], "rules": [{"when": ["last:verifier"], "do": "stop:vote"}],
             "default": "stop:vote"}, "a move no rule makes"),
           ({"plan": [P.PLAN_ROUNDS["solver"]], "rules": [{"when": ["acts==1"], "do": "stop:vote"}],
             "default": "stop:vote"}, "acts"),
           ({"plan": [P.PLAN_ROUNDS["solver"]], "rules": [{"when": ["ran_round:critic"], "do": "stop:vote"}],
             "default": "stop:vote"}, "a round kind the program lacks"),
           ({"plan": [dict(P.PLAN_ROUNDS["solver_x2|blind"])], "rules": [P.CONT()], "default": "stop:vote"},
            "a blind first round")]
    for prog, why in bad:
        try:
            P.validate_program(prog)
            check(False, f"rejected: {why}")
        except (AssertionError, ValueError):
            check(True, f"rejected: {why}")
    # the rules are checked before the first round too: a program must start its debate there.
    # This is llm_g0_1 of outputs/pipeline_cluster_gptoss (2026-09-28): it assumed the plan runs by
    # itself, stopped at step 0 on every question and scored 0%.
    dead = {"plan": [{"personas": ["solver", "solver"]}],
            "rules": [{"when": ["step==1", "r1_majority==2"], "do": "verifier|high"},
                      {"when": ["step==1"], "do": "expert|high|blind"},
                      {"when": ["step==2", "last_round:expert"], "do": "synthesizer"},
                      {"when": ["step==2", "last_round:verifier", "n_distinct>=2"], "do": "synthesizer"}],
            "default": "stop:last_commit"}
    try:
        P.validate_program(P.normalize_program(dead))
        check(False, "rejected: a program that stops before its first round")
    except ValueError as exc:
        check("stops before its first round" in str(exc), "rejected: a program that stops before its first round",
              str(exc))
    check(P.first_action(dead) == "stop:last_commit", "its first action is the default stop")
    starts = [dict(dead, rules=[P.CONT()] + dead["rules"]),
              dict(dead, rules=[{"when": ["step==0"], "do": "continue"}] + dead["rules"]),
              dict(dead, rules=[{"when": ["step<1"], "do": "solver_x2|high"}] + dead["rules"])]
    for prog in starts:
        P.validate_program(P.normalize_program(prog))
    check([P.first_action(p) for p in starts] == ["continue", "continue", "solver_x2|high"],
          "a program that starts with continue or a round is valid")
    check(all(P.first_action(p) == "continue" for p in P.PROTOCOLS.values()), "every literature program starts")
    check("before the first round" in P.grammar_text() and "plan does not start by itself" in P.grammar_text()
          and "before the first round" in G.GROUP_SEED_INSTRUCTIONS, "the model is told the rules run at step 0")
    # replay: a program that passes the check really runs a round on every question
    fake_rows = {"q": {"id": "q", "question": "?", "options": ["a", "b", "c", "d"], "answer_letter": "A"}}

    class Runner:
        def run_round(self, qid, rounds, specs, spec, rep=0, prompts=None):
            return [(p, "ANSWER: A") for p in spec["personas"]]
    for prog in starts:
        out = M.run_program(P.normalize_program(prog), Runner(), fake_rows["q"])
        check(out["n_calls"] >= 1 and out["letter"] == "A", "it runs a round and answers", str(out))
    out = M.run_program(P.normalize_program(dead), Runner(), fake_rows["q"])
    check(out["n_calls"] == 0 and out["letter"] is None, "the dead program runs nothing (as in the failed run)")
    rng = random.Random(4)
    progs = [P.random_program(rng) for _ in range(300)]
    check(not any(c.startswith("acts") for p in progs for r in p["rules"] for c in r["when"]),
          "random programs never use acts")
    check(all(not P.first_action(p).startswith("stop:") for p in progs), "random programs all start their debate")
    kids = [P.mutate_uniform(p, rng, donors=list(P.PROTOCOLS.values()))[0] for p in progs[:100]
            for _ in range(5)]
    kids += [P.mutate_uniform(p, rng, donors=progs[:20])[0] for p in P.PROTOCOLS.values() for _ in range(40)]
    check(all(not P.first_action(k).startswith("stop:") for k in kids),
          "edited children all start their debate", str(len(kids)))
    prog = {"plan": [P.PLAN_ROUNDS["solver_x2"]],
            "rules": [P.CONT(), {"when": ["step==5"], "do": "critic"}, {"when": ["step==1"], "do": "verifier|high"}],
            "default": "stop:last_commit"}
    pruned, w = P.prune(prog, {0: 10, 2: 10})
    check(pruned["rules"] == [prog["rules"][0], prog["rules"][2]] and w == [11, 11], "a rule that never fired is pruned")
    whole, w = P.prune(prog, {})
    check(whole == prog and w == [1, 1, 1], "nothing fired: the program is kept whole")
    counts = Counter(P._pick(3, rng, [1, 1, 98]) for _ in range(2000))
    check(counts[2] > 1800, "rule edits pick rules in proportion to their weight", str(counts))
    donor = P.PROTOCOLS["mad"]
    kids = {P.canon(P.crossover(prog, donor, rng)) for _ in range(40)}
    want = {P.canon(P.normalize_program({"plan": prog["plan"], "rules": donor["rules"], "default": donor["default"]})),
            P.canon(P.normalize_program({"plan": donor["plan"], "rules": prog["rules"], "default": prog["default"]}))}
    check(kids == want, "crossover joins one program's plan to the other's rules")
    slots_v = P.round_edit_slots(prog, "visibility")
    slots_e = P.round_edit_slots(prog, "effort")
    check([(w_, i) for w_, i, _ in slots_v] == [] and len(slots_e) == 3,
          "visibility never switches the first round or a critic; effort switches any round that stays in bounds",
          f"{[(a, b) for a, b, _ in slots_v]} {[(a, b) for a, b, _ in slots_e]}")
    four = {"plan": [P.PLAN_ROUNDS["solver_x4"]], "rules": [P.CONT()], "default": "stop:vote"}
    check(P.round_edit_slots(four, "effort") == [], "four solvers cannot go high effort (20 turns)")
    txt = P.grammar_text()
    check("|high" in txt and "blind" in txt and "deep_think" not in txt and "acts==" not in txt,
          "the grammar text describes effort and visibility")


def test_subsets():
    print("search questions")
    run_cli(SUB, ["--out", str(CLUSTERS_V3), "--per-cluster", "50"])
    src = json.loads((P.ROOT / "tests/data/clusters_train_both.json").read_text())
    d = json.loads(CLUSTERS_V3.read_text())
    check([c["members"] for c in d["clusters"]] == [c["members"] for c in src["clusters"]],
          "the groups themselves are unchanged")
    ok = all(len(c["subset"]) == 50 and len(set(c["subset"])) == 50
             and not set(c["subset"]) & set(c["held_out"])
             and set(c["subset"]) | set(c["held_out"]) == set(c["members"]) for c in d["clusters"])
    check(ok, "50 per group, disjoint from held-out, together the whole group")
    closer = [sum(c["subset_rank"].values()) / 50 < (c["size"] + 1) / 2 for c in d["clusters"]]
    check(sum(closer) >= 5, "the draw leans towards the centre", str(closer))
    whole = P.load_groups(CLUSTERS_V3, 0)
    check(all(sorted(g["search"]) == sorted(c["members"]) and not g["held_out"]
              for g, c in zip(whole["groups"], d["clusters"])), "--per-group 0: whole groups, nothing held out")
    new_fmt = P.ROOT / "outputs/describe_v3/clusters_600_train.json"
    if new_fmt.exists():                            # the describer-v3 format: 'challenges', not 'risks'
        g3 = P.load_groups(new_fmt, 0)
        check(sum(len(g["search"]) for g in g3["groups"]) == 300 and all(g["risks"] for g in g3["groups"]),
              "the describer-v3 clusters load whole, with their challenge profile")
    first = [c["subset"] for c in d["clusters"]]
    run_cli(SUB, ["--out", str(CLUSTERS_V3), "--per-cluster", "50", "--force"])
    check(first == [c["subset"] for c in json.loads(CLUSTERS_V3.read_text())["clusters"]],
          "the draw is fixed by its seed")


def test_dev_split_and_plain():
    print("dev split and plain instruction")
    import split_train_dev as SPLIT
    d = json.loads(CLUSTERS_V3.read_text())
    members = [q for c in d["clusters"] for q in c["members"]]
    split_path = TMP / "v3_split.json"
    sp = SPLIT.make_split(members, len(members) // 3, 0)
    split_path.write_text(json.dumps(sp))
    whole = P.load_groups(CLUSTERS_V3, 0)
    g = P.load_groups(CLUSTERS_V3, 0, split_path)
    dev = set(sp["dev"])
    check(all(s["search"] == [q for q in w["search"] if q not in dev] and s["held_out"] == [q for q in w["search"] if q in dev]
              for s, w in zip(g["groups"], whole["groups"])),
          "a dev split: each group's dev questions are held out, the rest are search questions, in the file's order")
    check(g["dev_split"] == str(split_path.resolve()) and "dev_split" not in whole, "the split is named only when used")
    capped = P.load_groups(CLUSTERS_V3, 3, split_path)    # HLE (2026-10-07): a dev split with a cap
    check(all(c["search"] == s["search"][:3] and c["held_out"] == s["held_out"]
              for c, s in zip(capped["groups"], g["groups"])),
          "a dev split with a cap: the first 3 non-dev questions of each group are search questions, the dev "
          "questions held out")
    short = TMP / "v3_split_short.json"
    short.write_text(json.dumps({"train": sp["train"][1:], "dev": sp["dev"]}))
    try:
        P.load_groups(CLUSTERS_V3, 0, short)
        check(False, "a split that misses a group question is refused")
    except SystemExit:
        check(True, "a split that misses a group question is refused")

    # the plain instruction: no request to think longer, its own cache keys, off again unchanged
    before = (dict(D.PERSONA_PROMPTS), SF.REWRITTEN_CRITIC, D.EXPERT_TMPL, SF.blind_key("solver", "high", 0))
    on = P.configure_executor(executor="v3", window=65536, plain_instruction=True)
    texts = list(D.PERSONA_PROMPTS.values()) + [SF.REWRITTEN_CRITIC, D.EXPERT_TMPL]
    check(on.get("plain_instruction") is True and all("Take the space" not in t for t in texts)
          and all(D.ANSWER_INSTR_PLAIN in t for t in [D.PERSONA_PROMPTS["verifier"], SF.REWRITTEN_CRITIC, D.EXPERT_TMPL]),
          "--plain-instruction: every prompt asks only for the ANSWER line, and the settings name it "
          "(the solver's system prompt is a general one and carries no answer line)")
    key_on = json.loads(SF.blind_key("solver", "high", 0))
    off = P.configure_executor(executor="v3", window=65536)
    check(key_on.get("i") == "1" and "i" not in json.loads(SF.blind_key("solver", "high", 0)),
          "the plain instruction has its own cache keys")
    check((dict(D.PERSONA_PROMPTS), SF.REWRITTEN_CRITIC, D.EXPERT_TMPL, SF.blind_key("solver", "high", 0)) == before
          and "plain_instruction" not in off, "switched off, every prompt, key and setting is as before")


def test_seeds() -> Path:
    print("seed stage")
    check(S3.seed_counts(6) == (6, 8, 0) and S3.seed_counts(3) == (3, 8, 0)
          and S3.seed_counts(10) == (10, 8, 0), "seeds: one model-written per group and 8 literature programs",
          f"{S3.seed_counts(6)} {S3.seed_counts(3)} {S3.seed_counts(10)}")
    out = TMP / "v3" / "seeds.json"
    n0 = len(GUIDE_REQUESTS)
    run_cli(S3, ["--clusters", str(CLUSTERS_V3), "--per-group", "3", "--out", str(out),
                 "--live-cache", str(CACHE), "--sanity-per-group", "1", "--min-turns", "1",
                 "--workers", "8"] + WINDOW)
    seeds = json.loads(out.read_text())["seeds"]
    later = GUIDE_REQUESTS[n0 + 1:]                  # asked again for the groups a rejected program left short
    check(not any(P.reviewer_first(P.normalize_program(s["program"])) for s in seeds)
          and all("speaks in the first round" in r["input"] for r in later)
          and "must have no critic, verifier or synthesizer" in GUIDE_REQUESTS[n0]["instructions"],
          "the seed writer is told: no reviewer in the first round; a program with one is asked for again",
          f"{len(later)} requests after the first")
    src = Counter(s["source"] for s in seeds)
    check(len(seeds) == 14 and src == {"protocol": 8, "llm": 6}, "14 seeds = 8 literature + 6 model-written",
          str(dict(src)))
    check(sorted(s["group"] for s in seeds if s["source"] == "llm") == list(range(6)),
          "one model-written seed per group")
    check([s["name"] for s in seeds if s["source"] == "protocol"] == ["mad", "early_exit_agree", "expert_first", "direct_high", "self_refine_high", "self_consistency_high", "verify_then_decide_high", "fresh_on_disagree_high"],
          "the literature seeds: the high-effort programs and the three without a high-effort version")
    check(len({P.canon(s["program"]) for s in seeds}) == 14, "all seeds differ")
    # a cache of model-written programs holding three per group (as run2's did) gives the first of each
    cache = [{"name": f"llm_g{g}_{i}_x", "group": g, "program": P.PROTOCOLS["direct"]} for g in range(3) for i in (1, 2, 3)]
    check([w["name"] for w in S3.pick_written(cache, [0, 1, 2], 1)] == ["llm_g0_1_x", "llm_g1_1_x", "llm_g2_1_x"],
          "a cache with three per group gives the first of each group")
    check(all(not P.first_action(s["program"]).startswith("stop:") for s in seeds),
          "every seed starts its debate")
    check(json.loads(out.read_text())["settings"]["executor"] == "v3", "the seeds were made under v3")
    req = GUIDE_REQUESTS[n0]                          # the first request names every group
    rows = {r["id"]: r for r in json.loads(T.DATASET.read_text())}
    blocks = req["input"].split("GROUPS OF QUESTION\n", 1)[1].split("\n\n")
    groups = P.load_groups(CLUSTERS_V3, 3)["groups"]
    check(len([b for b in blocks if b.startswith("group ")]) == 6
          and all("  typical steps: " in b and "example questions (3 of the group's" in b and b.count("--- example") == 3
                  for b in blocks if b.startswith("group ")),
          "each group: its profile (typical steps) and its example questions (all 3 search questions here)")
    check("typical moves" not in req["input"] and "difficulty" not in req["input"],
          "no 'moves' and no difficulty labels in what the seed writer sees")
    shown = P.group_samples_text(groups[0], rows, 10)
    want = D.render_question(rows[groups[0]["search"][0]]["question"], rows[groups[0]["search"][0]]["options"]).strip()
    check("\n".join("    " + l for l in want.splitlines()) in shown and "up to 10 example questions" in req["instructions"],
          "an example is the question and options as the debaters see them, and no more")
    g = {"search": ["q1", "q2", "q3", "q4"], "medoid": "q3", "size": 4}
    toy = {q: {"question": f"Question {q}. " * (400 if q == "q2" else 1), "options": ["x", "y"]} for q in g["search"]}
    t = P.group_samples_text(g, toy, 4, max_chars=600)
    check(t.split("--- example 1 ---\n")[1].startswith("    Question q3.") and t.count("--- example") == 4,
          "a group listed in the dataset's order shows its medoid first")
    check(P.group_samples_text({**g, "search": ["q3", "q2", "q1"]}, toy, 2).count("Question q2.") > 1
          and "characters omitted" in t and all(e.rstrip().endswith("B) y") for e in t.split("--- example")[1:]),
          "a representative order is kept; a long question keeps its start and its options")
    return out


def test_search(seeds_path: Path):
    print("search")
    run = TMP / "v3" / "run"
    common = ["--seeds", str(seeds_path), "--out", str(run), "--clusters", str(CLUSTERS_V3),
              "--per-group", "3", "--live-cache", str(CACHE), "--workers", "8"] + WINDOW
    run_cli(V, common + ["--generations", "2"])
    lines = [json.loads(l) for l in (run / "archive.jsonl").open() if l.strip()]
    header, recs = lines[0], {}
    for d in lines[1:]:
        recs[d["key"]] = d
    qids = header["qids"]
    check(header["version"] == 3 and len(qids) == 18, "header names format version 3 and 18 questions")
    seeds = [d for d in recs.values() if d["gen"] == 0]
    check(len(seeds) == 14 and all(len(d["reps"]["0"]) == 18 and len(d["reps"]["1"]) == 18 for d in seeds),
          "every seed has both replicates on every question")
    check(header["settings"]["executor"] == "v3" and header["settings"]["window"] == 65536,
          "the archive header names the v3 executor and the window")
    gens = [json.loads(l) for l in (run / "generations.jsonl").open() if l.strip()]
    check([g["gen"] for g in gens] == [0, 1, 2], "generations 0, 1, 2 logged")
    check(all(g["children"] <= 26 and g["parents"] == 13 for g in gens[1:]), "13 parents, at most 26 children",
          str([(g["parents"], g["children"]) for g in gens[1:]]))
    kids = [d for d in recs.values() if d["gen"] > 0]
    check(all((d["meta"]["slot"] in "AB" and d["meta"]["target"] in range(6))
              or (d["meta"]["slot"] == "G" and d["meta"]["target"] == "all") for d in kids),
          "every child records its slot and target group")
    check(any(d["meta"]["slot"] == "G" for d in kids), "the global slot breeds")
    check(all(len(v) == 6 for d in recs.values() for rec in d["reps"].values() for v in rec.values()),
          "every result records the rules fired and the tokens")
    check(all(d["reps"]["0"][q][5] is not None for d in seeds for q in d["reps"]["0"]),
          "tokens are known for every seed debate")
    check(all(d["op"] in P.EDIT_KINDS for d in kids), "every child is one named edit")
    check(all(P.plan_cost(d["program"]) <= P.MAX_TURNS for d in recs.values()),
          "no program's plan rounds cost more than the turn cap")
    check(all(recs[d["parent"]]["lineage"] == d["lineage"] for d in kids), "children inherit their seed lineage")
    check(all(len(d["reps"]["0"]) == 18 and "screen" not in d["meta"] for d in kids),
          "every child is scored once on every search question, with no screen")
    slot_sets = [set(q) for q in header["groups"].values()]

    def whole_slots(qs):                 # a union of whole slot question sets (a group's, or all)
        rest = set(qs)
        if rest == set(qids):
            return True
        for g in slot_sets:
            if g <= rest:
                rest -= g
        return not rest
    check(all(whole_slots(d["reps"]["1"]) for d in kids if d["reps"].get("1")),
          "a child's second run (if it ever held a slot) covers whole slots' questions, nothing else")
    check(all("turns_children" in g and "turns_stage1" not in g for g in gens[1:]), "the log has one child stage")
    # slot rules, recomputed from the archive on file
    s = V.Search(types.SimpleNamespace(out=str(TMP / "v3" / "check"), seed=0, max_calls_per_question=16),
                 {}, P.load_groups(CLUSTERS_V3, 3), None, {})
    for d in recs.values():
        r = P.ProgRecord.from_json(d)
        r.dup_of = None
        s.archive[r.key] = r
    for r in s.archive.values():
        s.mark_duplicate(r)
    slots = s.compute_slots()
    logged = {(x["group"], x["slot"]): x["key"] for x in gens[-1]["slots"]}
    check(slots == logged, "the logged slots follow from the archive")
    pool = s.eligible()
    check(all(abs(s.score(s.archive[slots[(g, "A")]], g) - max(s.score(r, g) for r in pool)) < 1e-9
              for g in s.group_ids), "slot A is the group's best score")
    check(all(slots[(g, "A")] != slots[(g, "B")] for g in s.group_ids), "slot B is never slot A's holder")
    check(abs(s.archive[slots[(V.GLOBAL, "G")]].score(s.qids) - max(r.score(s.qids) for r in pool)) < 1e-9,
          "the global slot holds the best score over all search questions")
    check(all(abs(s.gaps_b[(g, slots[(g, "B")])] - max(s.gaps_b[(g, r.key)] for r in pool
                                                     if r.key != slots[(g, "A")])) < 1e-9
              for g in s.group_ids), "slot B is the largest rank gap")
    check(all(not s.archive[k].gaps(s.gq[g], 1) for (g, _), k in slots.items()),
          "every slot holder has both replicates on its group")
    check(all(s.archive[k].dup_of is None and not s.archive[k].gaps(s.qids, 0) for k in slots.values()),
          "slot holders are fully scored and not duplicates")
    summary = json.loads((run / "summary.json").read_text())
    check(sum(summary["holders_by_source"].values()) == 13
          and set(summary["holders_by_source"]) <= {"protocol", "llm", "random"},
          "the summary traces every slot holder to a seed source", str(summary["holders_by_source"]))
    check(all("program_pruned" in x for x in summary["slots"]), "the summary shows the pruned holders")

    # resume
    before = T.MODEL_CALLS["chat"]
    run_cli(V, common + ["--generations", "3", "--resume"])
    gens = [json.loads(l) for l in (run / "generations.jsonl").open() if l.strip()]
    check(gens[-1]["gen"] == 3 and T.MODEL_CALLS["chat"] > before, "resume runs generation 3")
    before = T.MODEL_CALLS["chat"]
    run_cli(V, common + ["--generations", "3", "--resume"])
    check(T.MODEL_CALLS["chat"] == before, "a finished search resumes without spending")
    try:
        run_cli(V, [a if a != "3" else "4" for a in common] + ["--generations", "3", "--resume"])
        check(False, "a different question set is refused on resume")
    except SystemExit:
        check(True, "a different question set is refused on resume")

    # champions
    run_cli(V, common + ["--pick-champions", "--heldout-cap", "4"])
    champs = json.loads((run / "champions.json").read_text())
    check(champs["baselines"] == ["direct", "mad", "self_refine"], "baselines are named")
    ok = True
    # a baseline is a row of its own, or a finalist with the same text (reported once, marked as a baseline)
    base_keys = {P.canon(P.normalize_program(P.PROTOCOLS[b])) for b in ("direct", "mad", "self_refine")}
    for g, res in list(champs["per_group"].items()) + [("all", champs["global"])]:
        fin = [p for p in res["programs"] if p["finalist"]]
        ok &= base_keys == {p["key"] for p in res["programs"] if p["baseline"]} and 1 <= len(fin) <= 5
        ok &= res["champion"] in {p["key"] for p in fin}
        champ = next(p for p in res["programs"] if p["key"] == res["champion"])
        cheap = next(p for p in res["programs"] if p["key"] == res["cheapest_within_one_se"])
        ok &= cheap["vs_champion"]["within_one_se"] and (cheap["turns"] or 0) <= (champ["turns"] or 0)
        ok &= all(p["n_missing"] == 0 for p in res["programs"])
    check(ok, "each group: baselines scored, champion a finalist, cheapest within one SE no dearer")
    # the groups' debates run in one pool, then the global set's: a program of a group and of the global
    # set ran once on the group's questions, and the global set read that debate from the cache
    glob = {p["key"]: p for p in champs["global"]["programs"]}
    shared = [(p, glob[p["key"]]) for res in champs["per_group"].values() for p in res["programs"] if p["key"] in glob]
    check(shared and all(p["marks"] == {q: g["marks"][q] for q in p["marks"]} for p, g in shared),
          "a program of a group and of the global set has the same results on the group's questions in both",
          f"{len(shared)} shared")
    held = {q for c in json.loads(CLUSTERS_V3.read_text())["clusters"] for q in c["held_out"] + c["subset"][3:]}
    used = {q for res in champs["per_group"].values() for p in res["programs"] for q in p["marks"]}
    check(used <= held and not used & set(qids), "held-out questions are not search questions")
    after = [json.loads(l) for l in (run / "archive.jsonl").open() if l.strip()]
    check(len(after) == len([json.loads(l) for l in (run / "archive.jsonl").open() if l.strip()])
          and all(set(d["reps"]["0"]) <= set(qids) for d in after[1:]),
          "held-out results never enter the search archive")


def test_interruptions(seeds_path: Path):
    """A run stopped part-way (here by the call cap, which takes the same exit
    as Ctrl-C) must resume into a complete, consistent archive."""
    print("interrupted runs")
    run, cache = TMP / "v3b" / "run", TMP / "v3b" / "rounds.jsonl"
    common = ["--seeds", str(seeds_path), "--out", str(run), "--clusters", str(CLUSTERS_V3),
              "--per-group", "3", "--live-cache", str(cache), "--workers", "8"] + WINDOW

    def state():
        lines = [json.loads(l) for l in (run / "archive.jsonl").open() if l.strip()]
        recs = {d["key"]: d for d in lines if not d.get("header")}
        gens = ([json.loads(l) for l in (run / "generations.jsonl").open() if l.strip()]
                if (run / "generations.jsonl").exists() else [])
        return recs, gens

    # stopped in the middle of generation 0
    run_cli(V, common + ["--generations", "1", "--max-total-calls", "400"])
    recs, gens = state()
    check(len(recs) == 0 and not gens, "a stop inside generation 0 stores no half-scored seed")
    run_cli(V, common + ["--generations", "0", "--resume"])
    recs, gens = state()
    seeds = [d for d in recs.values() if d["gen"] == 0]
    check(len(seeds) == 14 and all(len(d["reps"]["0"]) == 18 and len(d["reps"]["1"]) == 18 for d in seeds)
          and [g["gen"] for g in gens] == [0], "resume completes the seeds")

    # stopped in the middle of generation 1
    run_cli(V, common + ["--generations", "2", "--resume", "--max-total-calls", "120"])
    recs, gens = state()
    check([g["gen"] for g in gens] == [0], "a stop inside generation 1 does not log it as done",
          str([g["gen"] for g in gens]))
    n_partial_gen = sum(1 for d in recs.values() if d["gen"] == 1)
    run_cli(V, common + ["--generations", "2", "--resume"])
    recs, gens = state()
    check([g["gen"] for g in gens] == [0, 1, 2], "resume redoes generation 1 and goes on to 2",
          str([g["gen"] for g in gens]))
    kids1 = [d for d in recs.values() if d["gen"] == 1]
    check(len(kids1) >= n_partial_gen and len({d["name"] for d in recs.values()}) == len(recs),
          "children of the interrupted generation are kept and every name is unique",
          f"{n_partial_gen} before, {len(kids1)} after")
    holders = [(x["key"], x["group"]) for x in gens[-1]["slots"]]
    groups = P.load_groups(CLUSTERS_V3, 3)
    gq = {g["group"]: g["search"] for g in groups["groups"]}
    gq[V.GLOBAL] = [q for g in groups["groups"] for q in g["search"]]
    check(all(set(gq[g]) <= set(recs[k]["reps"].get("1", {})) for k, g in holders),
          "after the resume every slot holder is confirmed")
    summary = json.loads((run / "summary.json").read_text())
    check(summary["generations"] == 2 and len(summary["slots"]) == 13, "summary written after the resume")


def test_over_cap():
    """A child whose plan rounds cost more than the turn cap is drawn again, never kept, and such a
    seed is refused, in the cluster and the global search alike: the cap would stop every question before
    its last plan round (the Qwen 9B cluster-pipeline run's global holder, 2 high solvers > high critic > high
    solver = 20 turns, ran as its first two rounds on every question)."""
    print("plans over the turn cap")
    import evolve_pipeline_global as V4
    sr = P.normalize_program(P.PROTOCOLS["self_refine_high"])
    wide = json.loads(json.dumps(sr))
    wide["plan"][0]["personas"] = ["solver", "solver"]
    fit = json.loads(json.dumps(sr))
    fit["plan"][1].pop("effort")
    P.validate_program(wide)
    P.validate_program(fit)
    check(P.MAX_TURNS == 16 and P.plan_cost(sr) == 15 and P.plan_cost(wide) == 20 and P.plan_cost(fit) == 11,
          "plan costs: self_refine_high 15, its 2-solver widening 20, its low-effort critic 11")
    real = P.mutate_uniform, P.mutate_by_family          # the global search's draw, and the cluster search's
    try:
        for name, search in (("cluster", V.Search), ("global", V4.Search)):
            s = V.Search(types.SimpleNamespace(out=str(TMP / "v3" / f"overcap_{name}"), seed=0,
                                               max_calls_per_question=16),
                         {}, P.load_groups(CLUSTERS_V3, 3), None, {})
            s.replay = lambda rec, qids: None               # nothing recorded: no behaviour to compare
            parent = P.ProgRecord(sr, "self_refine_high", "self_refine_high", 0)
            drawn = iter([(wide, "plan"), (fit, "effort")])
            P.mutate_uniform = P.mutate_by_family = lambda *a, **k: next(drawn)
            stats = Counter()
            child = search.draw_child(s, parent, s.group_ids[0], "A", "c", [], set(), random.Random(0), stats)
            check(child is not None and child.key == P.canon(fit) and stats["redraw_over_cap"] == 1,
                  f"{name}: the over-cap child is drawn again and the next one kept", str(stats))
            try:
                search.seed(s, [{"name": "wide", "program": wide, "source": "protocol"}])
                check(False, f"{name}: a seed over the cap is refused")
            except SystemExit as e:
                check("over the 16-turn cap" in str(e) and not s.archive, f"{name}: a seed over the cap is refused")
    finally:
        P.mutate_uniform, P.mutate_by_family = real


if __name__ == "__main__":
    P.configure_executor(executor="v3", window=65536)      # the entry scripts do the same
    test_statistics()
    test_executor_v3()
    test_cache_v3()
    test_grammar_v3()
    test_edits()
    test_subsets()
    test_dev_split_and_plain()
    seeds_path = test_seeds()
    test_search(seeds_path)
    test_interruptions(seeds_path)
    test_over_cap()
    print()
    if T.FAILURES:
        print(f"{len(T.FAILURES)} FAILED: " + "; ".join(T.FAILURES))
        sys.exit(1)
    print(f"all checks passed (model calls: {T.MODEL_CALLS}); temp dir {TMP}")

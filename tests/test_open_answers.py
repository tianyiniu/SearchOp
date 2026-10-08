"""Offline checks for open answers (HLE): the executor's answer format and
normalisation, the open-answer prompts, the cache key, the judge (cache,
concurrency, failures), grading inside a program, a whole seeds -> search ->
test-evaluation run on the real HLE splits and groups, and the external
baselines' HLE format, recovery and judge scoring. A fake debate model, a fake
guide, a fake judge and a fake HTTP server stand in for every model: no vLLM,
no API key.

    python tests/test_open_answers.py
"""

from __future__ import annotations

import hashlib
import json
import re
import socket
import subprocess
import sys
import threading
import time
import types
from argparse import Namespace
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))   # the code under test
sys.path.insert(0, str(Path(__file__).resolve().parent))                       # the other test files

import test_pipeline_cluster as T3  # noqa: E402  (installs the fake clients and the v3 fake model)
import debate_mcq as D  # noqa: E402
import schema_fitness as SF  # noqa: E402
import evolve_program_mcq as M  # noqa: E402
import program_space as P  # noqa: E402
import program_guide as G  # noqa: E402
import judge_answers as J  # noqa: E402
import evolve_pipeline_cluster as V  # noqa: E402
import program_seeds_cluster as S3  # noqa: E402
import eval_routed_dev as E  # noqa: E402
import train_baselines as TB  # noqa: E402

T = T3.T
check, run_cli, TMP = T.check, T.run_cli, T.TMP
ROOT = P.ROOT
PY = sys.executable
TRAIN = ROOT / "datasets/hle_text_train_800.json"
TEST = ROOT / "datasets/hle_text_test_200.json"
CLUSTERS = ROOT / "outputs/describe_hle_v4/clusters_train.json"
ROUTES = ROOT / "outputs/describe_hle_v4/routes_test.json"
ROWS = {r["id"]: r for p in (TRAIN, TEST) for r in json.loads(p.read_text())}
GOLD_BY_QUESTION = {r["question"][:300]: r["answer"] for r in ROWS.values()}
JUDGE_CACHE = TMP / "judge.jsonl"
OPEN_ARGS = ["--context-window", "65536", "--answers", "open", "--judge-cache", str(JUDGE_CACHE)]


# --- the fake judge: right when the normalised answers are equal ----------------------------
JUDGE_CALLS = Counter()


def _fake_ask(self, question, gold, answer):
    JUDGE_CALLS["calls"] += 1
    same = D.normalize_answer(answer) == D.normalize_answer(gold)
    return J.Verdict(extracted_final_answer=answer, reasoning="fake", correct="yes" if same else "no"), \
        {"input": 10, "output": 5}


_REAL_ASK = J.Judge._ask            # the real API call, for the failure check
J.Judge._ask = _fake_ask


# --- the fake debate model, open answers ----------------------------------------------------
# The v3 fake of test_pipeline_cluster in the open format: answers are the key or a wrong
# value, written in varied formatting ($...$, **...**, \boxed{}), so normalisation matters.

_v3_create = T.FakeCompletions.create
OPEN_CALLS: Counter = Counter()


def _gold(user: str) -> str:
    for q, g in GOLD_BY_QUESTION.items():
        if user.startswith(q):
            return g
    return "0"


def _formatted(ans: str, h: int) -> str:
    return (ans, f"${ans}$", f"**{ans}**", f"\\boxed{{{ans}}}")[h % 4] if len(ans) > 1 else (ans, f"({ans})", f"{ans}.")[h % 3]


def _create_open(self, model, messages, temperature=None, max_tokens=None, extra_body=None, **kw):
    if not (D.V3 and D.OPEN):
        return _v3_create(self, model, messages, temperature, max_tokens, extra_body, **kw)
    last = messages[-1]["content"]
    if last.startswith("Now write a summary"):
        OPEN_CALLS["summary"] += 1
        locked = re.search(r"Your answer is final: (.*)\. This summary reports", last, re.S).group(1)
        if int(hashlib.sha1(messages[-2]["content"].encode()).hexdigest(), 16) % 5 == 0:
            OPEN_CALLS["argued"] += 1
            return T3._Resp("On reflection another value fits.\nANSWER: 999")      # must be rejected
        return T3._Resp(f"The decisive step points one way.\nANSWER: {locked}")
    if last.startswith("Your reply ended before"):
        OPEN_CALLS["answer_first"] += 1
        return T3._Resp(f"ANSWER: {_gold(messages[1]['content'])}\nThe reasoning so far favoured it.")
    if last == D.COMMIT_NUDGE:
        OPEN_CALLS["commit"] += 1
        return T3._Resp("ANSWER: 5")
    T.MODEL_CALLS["chat"] += 1
    T.FakeCompletions.salt += 1
    high = (extra_body or {}).get("chat_template_kwargs", {}).get("enable_thinking", False)
    user = messages[1]["content"]
    h = int(hashlib.sha1(f"{messages[0]['content'][:30]}|{last}|{T.FakeCompletions.salt}".encode()).hexdigest(), 16)
    ans = _formatted(_gold(user) if h % 3 else "17", h // 3)
    shape = (h // 12) % 4
    thinking = "Private reasoning step. " * 60 if high else None
    if shape == 0:
        return T3._Resp("A short argument. " * 30 + f"\nANSWER: {ans}", thinking)
    if shape == 1:
        return T3._Resp("One long derivation step. " * 150 + f"\nANSWER: {ans}", thinking)
    if shape == 2:
        return T3._Resp("Still working through the problem without finishing. " * 20, thinking)
    return T3._Resp(f"ANSWER: {ans}", "Hidden reasoning. " * 40)


T.FakeCompletions.create = _create_open


def configure_open():            # as the gpt-oss pipeline runs: visible reasoning on
    return P.configure_executor(executor="v3", window=65536, visible_reasoning=True, answers="open",
                                judge_cache=JUDGE_CACHE)


def configure_letters():
    return P.configure_executor(executor="v3", window=65536, visible_reasoning=True)


# --- 1. formats, prompts, keys -----------------------------------------------------------------

def test_formats_and_prompts():
    print("open answers: format, prompts, keys")
    configure_letters()
    letter_prompts = dict(D.PERSONA_PROMPTS)
    letter_key = SF.path_key([P.PLAN_ROUNDS["solver"]])
    letter_critic = SF.REWRITTEN_CRITIC
    settings = configure_open()
    check(settings.get("answers") == "open" and settings.get("judge") == {"model": "gpt-6-luna", "effort": "medium",
                                                                           "prompt": J.PROMPT_VERSION},
          "the settings name the answer mode and the judge", str(settings))
    check(M.GRADER is not None and P.JUDGE is not None and P.JUDGE.cache_path == JUDGE_CACHE, "a judge grades answers")
    used = ("solver", "critic", "verifier", "synthesizer", "expert")
    texts = [D.PERSONA_PROMPTS[p] for p in used] + [SF.REWRITTEN_CRITIC, D.EXPERT_TMPL.format(field="topology"),
                                                     D.COMMIT_NUDGE, D.summary_nudge_v3("42"), D.summary_nudge_v3(None)]
    check(not any(re.search(r"option|multiple-choice|exactly one letter", t, re.I) for t in texts),
          "no open-answer prompt speaks of options or letters to choose")
    # the solver's system prompt is a general one (2026-10-06): the answer format reaches it in the user
    # message, which ends with ANSWER_INSTR as every open-answer speaker's does
    check(all(D.ANSWER_FORMAT in D.PERSONA_PROMPTS[p] for p in used if p != "solver")
          and D.ANSWER_FORMAT in SF.REWRITTEN_CRITIC and D.ANSWER_FORMAT in D.ANSWER_INSTR
          and D.ANSWER_INSTR in D._user_message("<q>", "", "", D.ANSWER_INSTR),
          "every answering persona is given the answer format (the solver in its user message)")
    check(D.VISIBLE_SENTENCE_OPEN in D.ANSWER_INSTR and D.VISIBLE_SENTENCE_OPEN in SF.REWRITTEN_CRITIC,
          "visible reasoning (gpt-oss) is asked for in the open wording")
    check(D.COMMIT_NUDGE == D.COMMIT_NUDGE_OPEN and "ANSWER: <final answer>" in D.COMMIT_NUDGE, "the commit prompt asks for the answer")
    check("Your answer is final: x^2+1." in D.summary_nudge_v3("x^2+1") and "ANSWER: x^2+1" in D.summary_nudge_v3("x^2+1"),
          "the summary prompt locks the answer")
    key = SF.path_key([P.PLAN_ROUNDS["solver"]])
    k, lk = json.loads(key), json.loads(letter_key)
    check(k.pop("o") == "1" and k.pop("t") != lk.pop("t") and k == lk,
          "open recordings get their own key (o=1, and their own prompts' signature t), the rest unchanged", key)
    g = P.grammar_text()
    check("letter" not in g and "committed answer" in g and "agree on an answer" in g,
          "the grammar text speaks of answers")
    check("most with a short exact answer" in G.GROUP_SEED_INSTRUCTIONS.format(
        task="most with a short exact answer", roles="", grammar="", n_existing=0, n_groups=0, n_per_group=1,
        plural="", spread="", axes="", examples=""), "the seed writer's task line is filled in")
    configure_letters()
    check(D.PERSONA_PROMPTS == letter_prompts and SF.path_key([P.PLAN_ROUNDS["solver"]]) == letter_key
          and SF.REWRITTEN_CRITIC == letter_critic and M.GRADER is None and not D.OPEN,
          "switching back restores the multiple-choice prompts, key, critic and grading exactly")
    D.OPEN = True
    check(D.extract_letter("x\n**ANSWER:** $\\frac{7}{2}$\nmore\nANSWER: **12**", 0) == "12"
          and D.extract_letter("no answer line here", 0) is None
          and D.extract_letter("Final ANSWER: 3", 0) is None, "the last ANSWER line is the commitment")
    check(D._answer_letters("ANSWER: 1\nANSWER: $2$", 0) == ["1", "2"]
          and D._strip_answer_lines("keep\nANSWER: 1\nkeep too") == "keep\n\nkeep too",
          "summary locking reads and strips ANSWER lines")
    check(D.normalize_answer("\\displaystyle \\frac{454514}{3516975}") == "\\frac{454514}{3516975}"
          and D.normalize_answer("$\\displaystyle x$") == "x" and D.normalize_answer("\\displaystyles") == "\\displaystyles",
          "LaTeX size commands are formatting")
    check(D.commit_letter_lenient("the answer is 7", 0) is None and D.commit_letter_lenient("ANSWER: 7", 0) == "7",
          "no answer is read from prose")
    D.OPEN = False
    rows = [ROWS[q] for q in list(ROWS)[:3]]
    try:
        P.check_rows(rows)
        check(False, "HLE rows are refused in letter mode")
    except SystemExit as exc:
        check("--answers open" in str(exc), "HLE rows are refused in letter mode", str(exc)[:80])
    configure_open()
    P.check_rows(rows)
    check(True, "HLE rows are accepted with open answers")


# --- 2. the executor ---------------------------------------------------------------------------

def test_executor_open():
    print("open answers: executor")
    configure_open()
    GPT = "openai/gpt-oss-20b"
    R = T3._Resp
    short = "Work. " * 20 + "\nANSWER: $\\frac{2}{3}$"
    c = T3._Scripted(R(short))
    out = D.chat_v3(c, GPT, "sys", "question", 0, "low")
    check(out == short and D.extract_letter(out, 0) == "\\frac{2}{3}", "a short committed reply is kept as it is")
    c = T3._Scripted(R("Short.\nANSWER: 42", "hidden " * 300), R("Summary of it.\nANSWER: 42"))
    out = D.chat_v3(c, GPT, "sys", "question", 0, "high")
    check(D.SUMMARY_MARK in out and D.extract_letter(out, 0) == "42"
          and "Your answer is final: 42." in c.requests[1]["messages"][-1]["content"],
          "a high-effort reply is summarised with its answer locked")
    c = T3._Scripted(R("Short.\nANSWER: 42", "hidden " * 300), R("Actually it is 41.\nANSWER: 41"))
    out = D.chat_v3(c, GPT, "sys", "question", 0, "high")
    check(D.SUMMARY_MARK not in out and D.extract_letter(out, 0) == "42", "a summary that changes the answer is not used")
    c = T3._Scripted(R("Still working. " * 200), R("ANSWER: x^2+1\nThat is where it led."))
    out = D.chat_v3(c, GPT, "sys", "question", 0, "low")
    check(D.extract_letter(out, 0) == "x^2+1" and c.requests[1]["messages"][-1]["content"].startswith("Your reply ended"),
          "an uncommitted reply commits through the answer-first summary")
    c = T3._Scripted(R("Still working. " * 200), R("A summary with no answer line."), R("ANSWER: 7\nbest guess"))
    out = D.chat_v3(c, GPT, "sys", "question", 0, "low")
    check(D.extract_letter(out, 0) == "7" and c.requests[2]["messages"][-1]["content"] == D.COMMIT_NUDGE_OPEN,
          "failing that, the commit prompt gets the answer")
    c = T3._Scripted(R("Reasoning. " * 20 + "\n**ANSWER:** **12**"))
    out = D.chat_v3(c, GPT, "sys", "question", 0, "low")
    check(D.extract_letter(out, 0) == "12", "a markdown ANSWER line is read")
    # the digest later speakers see names the answer
    check(D._digest([("solver", "ANSWER: $5$")], 0).startswith("[solver] chose 5:"), "the digest names the answer")


# --- 3. programs: votes over answers, graded by the judge ----------------------------------------

class _Runner:
    """Rounds from a script: each call gives the next round's replies."""

    def __init__(self, rounds):
        self.rounds = list(rounds)

    def run_round(self, qid, rounds, specs, spec, rep=0, prompts=None):
        replies = self.rounds.pop(0)
        return [(p, r) for p, r in zip(spec["personas"], replies)]


def test_programs_and_judge():
    print("open answers: programs and the judge")
    configure_open()
    row = {"id": "q1", "question": "What is 2/3 of 1?", "options": [], "answer": "2/3", "answer_type": "exactMatch"}
    before = JUDGE_CALLS["calls"]
    vote = P.normalize_program(P.PROTOCOLS["self_consistency_high"])     # three solvers, plurality
    out = M.run_program(vote, _Runner([["ANSWER: $2/3$", "ANSWER: 2/3.", "ANSWER: 0.6667"]]), row)
    check(out["letter"] == "2/3" and out["correct"], "a vote counts equal answers in different formatting together", str(out))
    out = M.run_program(vote, _Runner([["ANSWER: 2/3", "ANSWER: 0.6667", "ANSWER: 0.6667"]]), row)
    check(out["letter"] == "0.6667" and not out["correct"], "the plurality answer is graded")
    out = M.run_program(vote, _Runner([["no line", "none", "nothing"]]), row)
    check(out["letter"] is None and not out["correct"], "no answer is wrong")
    check(JUDGE_CALLS["calls"] - before == 2, "one judge call per distinct (question, answer)",
          str(JUDGE_CALLS["calls"] - before))
    out = M.run_program(vote, _Runner([["ANSWER: 2/3", "ANSWER: 2/3", "ANSWER: 1"]]), row)
    check(out["correct"] and JUDGE_CALLS["calls"] - before == 2, "a verdict is read back from the cache")
    # rules compare normalised answers: fresh_on_disagree_high adds a blind solver only on disagreement
    fod = P.normalize_program(P.PROTOCOLS["fresh_on_disagree_high"])
    out = M.run_program(fod, _Runner([["ANSWER: $\\frac{1}{2}$", "ANSWER: \\frac{1}{2}"]]), row)
    check(out["actions"] == ["continue"], "two formats of one answer agree (no tie-break round)", str(out["actions"]))
    out = M.run_program(fod, _Runner([["ANSWER: 1/2", "ANSWER: 0.5"], ["ANSWER: 2/3"]]), row)
    check(out["actions"] == ["continue", "solver|high|blind"] and out["letter"] == "2/3" and out["correct"],
          "different answers disagree and the tie-break decides")
    # the judge on its own: the cache survives a restart, concurrent grades of one pair make one call
    judge = J.Judge(TMP / "judge2.jsonl")
    n0 = JUDGE_CALLS["calls"]
    check(judge.grade("q", "Q?", "B", "B") and not judge.grade("q", "Q?", "B", "C")
          and judge.grade("q", "Q?", "B", None) is False, "right, wrong, and no answer (no call)")
    again = J.Judge(TMP / "judge2.jsonl")
    check(again.grade("q", "Q?", "B", "B") and again.stats["calls"] == 0 and JUDGE_CALLS["calls"] - n0 == 2,
          "verdicts are reloaded from the cache file")
    slow_calls = Counter()

    def slow_ask(self, question, gold, answer):
        slow_calls["n"] += 1
        time.sleep(0.2)
        return J.Verdict(extracted_final_answer=answer, reasoning="", correct="yes"), {}
    J.Judge._ask = slow_ask
    j3 = J.Judge(TMP / "judge3.jsonl")
    threads = [threading.Thread(target=j3.grade, args=("q", "Q?", "7", "7")) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    check(slow_calls["n"] == 1, "eight concurrent grades of one pair make one call", str(slow_calls["n"]))
    J.Judge._ask = _fake_ask

    class _Down:
        def __init__(self):
            self.responses = self

        def parse(self, **kw):
            raise RuntimeError("API down")
    j4 = J.Judge(TMP / "judge4.jsonl", retries=2)
    j4._client = _Down()
    orig_sleep, J.time.sleep = J.time.sleep, lambda s: None
    try:
        J.Judge._ask = _REAL_ASK
        j4.grade("q", "Q?", "7", "7")
        check(False, "a judge that keeps failing stops the run")
    except RuntimeError:
        check(not (TMP / "judge4.jsonl").exists() or not (TMP / "judge4.jsonl").read_text().strip(),
              "a judge that keeps failing stops the run, and nothing is cached")
    finally:
        J.time.sleep = orig_sleep
        J.Judge._ask = _fake_ask



# --- 4. the pipeline on the real HLE splits and groups -----------------------------------------

def test_pipeline():
    print("open answers: seeds, search and test evaluation on the HLE splits")
    out = TMP / "hle"
    common = ["--clusters", str(CLUSTERS), "--dataset", str(TRAIN), "--per-group", "2",
              "--live-cache", str(out / "rounds.jsonl"), "--workers", "8", "--visible-reasoning"] + OPEN_ARGS
    try:
        run_cli(S3, common[:-len(OPEN_ARGS)] + ["--context-window", "65536", "--out", str(TMP / "bad.json")])
        check(False, "the seed stage refuses HLE in letter mode")
    except SystemExit as exc:
        check("--answers open" in str(exc), "the seed stage refuses HLE in letter mode")
    run_cli(S3, common + ["--out", str(out / "seeds.json")])
    seeds = json.loads((out / "seeds.json").read_text())
    src = Counter(s["source"] for s in seeds["seeds"])
    check(src == {"protocol": 8, "llm": 4} and seeds["settings"].get("answers") == "open",
          "12 seeds: 8 literature + one per group (k = 4), made with open answers", str(dict(src)))
    run = out / "run"
    run_cli(V, common + ["--seeds", str(out / "seeds.json"), "--out", str(run), "--generations", "2"])
    lines = [json.loads(l) for l in (run / "archive.jsonl").open() if l.strip()]
    header, recs = lines[0], {}
    for d in lines[1:]:
        recs[d["key"]] = d
    qids = header["qids"]
    check(header["settings"].get("answers") == "open" and len(qids) == 8, "the search runs with open answers, 8 questions")
    letters = [v[2] for d in recs.values() for rep in d["reps"].values() for v in rep.values()]
    wrapped = re.compile(r"\$[^$]+\$|\*\*.*\*\*|\\boxed\{.*\}|\([A-Z]\)")
    check(any(len(l) > 1 for l in letters) and all(D.normalize_answer(l) == l for l in letters if l != "?")
          and not any(wrapped.fullmatch(l) for l in letters),
          "archived answers are the normalised open answers (a gold answer of two $-spans keeps them)",
          str(Counter(letters).most_common(4)))
    marks = [v[0] for d in recs.values() for rep in d["reps"].values() for v in rep.values()]
    check(0 < sum(marks) < len(marks), "some answers are graded right and some wrong", f"{sum(marks)}/{len(marks)}")
    verdicts = [json.loads(l) for l in JUDGE_CACHE.open()]
    check(len({(v["qid"], v["answer"]) for v in verdicts}) == len(verdicts), "each (question, answer) is judged once",
          f"{len(verdicts)} verdicts")
    n_before = JUDGE_CALLS["calls"]
    run_cli(V, common + ["--seeds", str(out / "seeds.json"), "--out", str(run), "--generations", "2", "--resume"])
    check(JUDGE_CALLS["calls"] == n_before, "a resumed search replays its grading from the cache")
    # train comparison and the test evaluation read the scorer's marks for external baselines
    ext_train = out / "ext_train.json"
    per_q = {q: {"preds": ["x", "y"], "marks": [1, 0], "answer": ROWS[q]["answer"], "tokens": [10, 20]} for q in qids}
    ext_train.write_text(json.dumps({"report": {}, "per_question": per_q}))
    run_cli(TB, ["--compare", "--run", str(run), "--external", f"self-refine={ext_train}"])
    tb = json.loads((run / "train_baselines.json").read_text())
    check(abs(tb["table"]["external self-refine"]["acc"] - 0.5) < 1e-9, "the train comparison reads marks (a judge's verdicts)")
    test_ids = [r["id"] for r in json.loads(TEST.read_text())]
    ext_test = out / "ext_test.json"
    ext_test.write_text(json.dumps({"report": {}, "per_question": {
        q: {"preds": ["x"], "marks": [1 if i % 4 == 0 else 0], "answer": ROWS[q]["answer"], "tokens": [5]}
        for i, q in enumerate(test_ids)}}))
    run_cli(E, ["--run", str(run), "--routes", str(ROUTES), "--dataset", str(TEST), "--no-champions",
                "--out", str(out / "test_eval"), "--reps", "1", "--workers", "8", "--visible-reasoning",
                "--baselines", "direct,self_refine,mad,direct_high,self_refine_high",
                "--external", f"self-refine={ext_test}"] + OPEN_ARGS)
    res = json.loads((out / "test_eval/results_k1.json").read_text())
    check(res["n_questions"] == 200 and abs(res["table"]["external self-refine"]["avg@1"] - 0.25) < 1e-9,
          "the test evaluation runs on the 200 HLE test questions and reads the external marks",
          f"{res['n_questions']}, {res['table']['external self-refine']['avg@1']}")
    check(res["settings"].get("answers") == "open" and 0 < res["table"]["global slot holder"]["avg@1"] < 1,
          "the searched programs are graded on the test split", str(res["table"]["global slot holder"]["avg@1"]))
    check(OPEN_CALLS["argued"] > 0 and OPEN_CALLS["answer_first"] > 0 and OPEN_CALLS["summary"] > 0,
          "the fake exercised summaries, rejected summaries and answer-first commits", str(dict(OPEN_CALLS)))


def test_pipeline_run2():
    """HLE under the SuperGPQA run2 design (2026-10-07; pipeline_cluster_setup.sh, hle): 100 of the 800
    train questions held out as a dev split, each group's search questions capped (50 in the run), the
    run2 flags, seeds, search, the champion step on the dev questions, the comparison with the external
    baselines (graded by the judge) and the test evaluation of the champions."""
    print("open answers: the run2 design on HLE (a dev split with a per-group cap)")
    import compare_external_baselines as CE
    import split_train_dev as SPL
    out = TMP / "hle_run2"
    out.mkdir(parents=True, exist_ok=True)
    split = SPL.make_split([r["id"] for r in json.loads(TRAIN.read_text())], 100, 0)
    splits = out / "splits.json"
    splits.write_text(json.dumps({"dataset": str(TRAIN), "n_dev": 100, "seed": 0, **split}))
    dev = set(split["dev"])
    raw = json.loads(CLUSTERS.read_text())["clusters"]
    capped, old, whole = P.load_groups(CLUSTERS, 50, splits), P.load_groups(CLUSTERS, 50), P.load_groups(CLUSTERS, 0, splits)
    ok = True
    for g, w, c in zip(capped["groups"], whole["groups"], raw):
        pool = c["subset"] + [q for q in c["held_out"] if q not in c["subset"]]
        nondev = [q for q in pool if q not in dev]
        ok &= g["search"] == nondev[:50] and g["held_out"] == [q for q in pool if q in dev]
        ok &= set(g["search"]) <= set(c["subset"])           # inside the ordered (representative) subset
        ok &= w["search"] == nondev and w["held_out"] == g["held_out"]   # per-group 0: as before
    check(ok, "with a dev split and a cap of 50, a group's search questions are its first 50 non-dev questions "
              "in the clusters file's order, and its held-out questions are its dev questions")
    search_all = [q for g in capped["groups"] for q in g["search"]]
    check(len(search_all) == 200 and not set(search_all) & dev
          and sorted(q for g in capped["groups"] for q in g["held_out"]) == sorted(dev),
          "200 search questions, none of them dev; the 100 dev questions are exactly the held-out ones")
    same = len(set(search_all) & {q for g in old["groups"] for q in g["search"]})
    check(150 <= same < 200, "most search questions are HLE run1's (gpt-oss reuses its baselines on them)",
          f"{same} of 200")
    try:
        P.load_groups(CLUSTERS, 95, splits)
        check(False, "a cap beyond the ordered subset is refused")
    except SystemExit as exc:
        check("ordered subset" in str(exc), "a cap beyond the ordered subset is refused", str(exc)[:80])

    # the pipeline at 2 search questions per group, with the run's flags
    flags = ["--context-window", "32768", "--plain-instruction", "--last-round-vote", "--count-read-summaries",
             "--answers", "open", "--judge-model", "gpt-6-luna", "--judge-cache", str(JUDGE_CACHE),
             "--high-cost", "3", "--turn-cap", "15", "--total-cap", "21"]
    common = ["--clusters", str(CLUSTERS), "--dataset", str(TRAIN), "--per-group", "2", "--dev-split", str(splits),
              "--live-cache", str(out / "rounds.jsonl"), "--workers", "8"] + flags
    run_cli(S3, common + ["--out", str(out / "seeds.json")])
    seeds = json.loads((out / "seeds.json").read_text())
    st = seeds["settings"]
    check(st.get("answers") == "open" and st.get("plain_instruction") and st.get("count_read_summaries")
          and st.get("turn_cap") == 15 and st.get("total_cap") == 21 and st.get("high_cost") == 3,
          "the seeds are made under open answers with the run2 flags", str(st))
    check(not any(P.reviewer_first(s["program"]) for s in seeds["seeds"]),
          "no seed opens with a critic, verifier or synthesizer")
    run = out / "run"
    search = common + ["--seeds", str(out / "seeds.json"), "--out", str(run), "--tie-questions", "1"]
    run_cli(V, search + ["--generations", "2"])
    header = json.loads((run / "archive.jsonl").open().readline())
    want_q = [q for g in P.load_groups(CLUSTERS, 2, splits)["groups"] for q in g["search"]]
    check(sorted(header["qids"]) == sorted(want_q) and len(want_q) == 8 and not set(want_q) & dev,
          "the search runs on the first 2 non-dev questions of each group")
    run_cli(V, search + ["--pick-champions", "--reps", "2", "--baselines", ""])
    champs = json.loads((run / "champions.json").read_text())
    held_ok = all(set(r["programs"][0]["marks"]) == set(g["held_out"])
                  for g in capped["groups"] for r in [champs["per_group"][str(g["group"])]])
    check(all(r["champion"] for r in champs["per_group"].values()) and champs["global"]["champion"] and held_ok
          and champs["global"]["held_out_n"] == 100,
          "the champion step picks per group on that group's dev questions, and overall on all 100")
    search_q = out / "search_questions.json"
    run_cli(TB, ["--export", "--clusters", str(CLUSTERS), "--per-group", "2", "--dev-split", str(splits),
                 "--dataset", str(TRAIN), "--out", str(search_q)])
    sq = [r["id"] for r in json.loads(search_q.read_text())]
    check(sq == want_q, "the exported search questions are the search's")

    # the comparison with the external baselines: ours graded by the judge, as the external marks are
    def score_file(qids, marks_of, name):
        p = out / name
        p.write_text(json.dumps({"report": {}, "per_question": {
            q: {"preds": ["1", "2", "3"], "marks": marks_of(i), "answer": "x", "tokens": [900, 1000, 1100]}
            for i, q in enumerate(qids)}}))
        return p
    ext_d, ext_sr = score_file(sq, lambda i: [1, 0, 0], "ext_d.json"), score_file(sq, lambda i: [0, 1, i % 2], "ext_sr.json")
    rawf = out / "ext_direct_raw.jsonl"
    rawf.write_text("".join(json.dumps({"id": q, "sample_idx": k, "error": None,
                                        "finish_reason": "length" if (i + k) % 2 else "stop"}) + "\n"
                            for i, q in enumerate(sq) for k in range(3)))
    judged = JUDGE_CALLS["calls"]
    try:
        run_cli(CE, ["--questions", str(search_q), "--out", str(out / "external_baselines"),
                     "--external-direct", str(ext_d), "--external-direct-raw", str(rawf),
                     "--external-selfrefine", str(ext_sr), "--model", P.DEFAULT_MODEL,
                     "--base-urls", "http://localhost:1/v1", "--live-cache", str(out / "rounds.jsonl"),
                     "--workers", "8", "--ignore-cache-lock"] + flags)
    except SystemExit as exc:                               # 0 on a pass, 1 on a fail
        check(exc.code in (0, 1), "the comparison ends with a verdict (0 pass, 1 fail)", str(exc.code))
    ce = json.loads((out / "external_baselines.json").read_text())
    names = [c["name"] for c in ce["checks"]]
    marks = [m for p in ("direct_high", "self_refine_high") for v in ce["per_question"][p].values() for m in v["right"]]
    check(ce["settings"].get("answers") == "open" and ce["settings"].get("judge", {}).get("model") == "gpt-6-luna"
          and any(n.endswith("no answer") for n in names) and not any("letter" in n for n in names),
          "the comparison runs on HLE, under the judge, and names its check 'no answer'", str(names))
    check(len(marks) == 2 * 3 * len(sq) and 0 < sum(marks) < len(marks) and JUDGE_CALLS["calls"] >= judged,
          "our programs' answers are graded by the judge in the comparison", f"{sum(marks)}/{len(marks)}")
    # what run_pipeline_cluster.sh checks before the search: the comparison's settings are the run's
    ap = __import__("argparse").ArgumentParser()
    P.add_executor_args(ap)
    gate = {**P.configure_from_args(ap.parse_args(flags)), "model": P.DEFAULT_MODEL}
    check(ce["settings"] == gate, "the comparison's settings equal the run's (the start check would accept it)",
          str({k for k in set(gate) | set(ce["settings"]) if gate.get(k) != ce["settings"].get(k)}))
    test_ids = [r["id"] for r in json.loads(TEST.read_text())]
    run_cli(E, ["--run", str(run), "--routes", str(ROUTES), "--dataset", str(TEST), "--out", str(out / "test_eval"),
                "--live-cache", str(out / "test_rounds.jsonl"), "--reps", "1", "--workers", "8",
                "--baselines", "direct_high,self_refine_high",
                "--external", f"self-refine={score_file(test_ids, lambda i: [i % 2], 'ext_test.json')}"] + flags)
    res = json.loads((out / "test_eval/results_k1.json").read_text())
    check(res["n_questions"] == 200 and res["settings"].get("answers") == "open"
          and 0 < res["table"]["routed, held-out champions"]["avg@1"] < 1
          and abs(res["table"]["external self-refine"]["avg@1"] - 0.5) < 1e-9,
          "the test evaluation routes the champions over the 200 HLE test questions and grades them",
          str({k: round(v["avg@1"], 3) for k, v in res["table"].items()}))


# --- 5. the external baselines on HLE ------------------------------------------------------------

FAKE_SERVER = r'''
import json, sys, hashlib
from aiohttp import web
log = open(sys.argv[2], "a")
def reply(msgs):
    sys_msg = msgs[0]["content"] if msgs[0]["role"] == "system" else ""
    mc = "Answer: {your chosen answer}" in sys_msg
    label = "Answer" if mc else "Exact Answer"
    last = msgs[-1]["content"]
    if "ran out of space" in last or last.startswith("STOP."):
        return f"{label}: {'B' if mc else '7'}", "stop", ""
    if "There may be an error" in last:
        return "it is correct", "stop", ""
    if last.startswith("Using the feedback above"):
        return f"Explanation: redone.\n{label}: {'C' if mc else '8'}\nConfidence: 50%", "stop", ""
    h = int(hashlib.sha1(last.encode()).hexdigest(), 16)
    if h % 2 == 0:                                      # cut off in the reasoning
        return None, "length", "a long derivation that ran out of room"
    return f"Explanation: worked it out.\n**{label}:** {'A' if mc else '$42$'}\nConfidence: 80%", "stop", ""
async def handle(request):
    body = await request.json()
    content, finish, reasoning = reply(body["messages"])
    log.write(json.dumps({"messages": body["messages"], "effort": body.get("reasoning_effort")}) + "\n"); log.flush()
    return web.json_response({"choices": [{"message": {"content": content, "reasoning": reasoning},
                                           "finish_reason": finish}], "usage": {"completion_tokens": 10}})
app = web.Application(); app.router.add_post("/v1/chat/completions", handle)
web.run_app(app, port=int(sys.argv[1]), print=None)
'''


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_baselines():
    print("open answers: the external baselines on HLE")
    sys.path.insert(0, str(ROOT / "baselines"))
    import score  # noqa: E402
    import hle_format  # noqa: E402
    shapes = {  # response endings seen from gpt-oss, and HLE's own format
        "Explanation: x\nExact Answer: 42\nConfidence: 90%": "42",
        "**Exact Answer:**  \nReplace residues 47-50 with alanines.\n\n**Confidence:** 95 %": "Replace residues 47-50 with alanines",
        "Explanation: x\nAnswer: C  \n\nConfidence: 95 %": "C",
        "### Exact Answer\n$\\frac{1}{2}$\n\n### Confidence\n80%": "\\frac{1}{2}",
        "**Exact Answer:** **12**": "12",
        "Explanation: the answer: is not here\nExact Answer: 7": "7",
        "Answer choices are listed.\nno answer line": None,
        "Exact Answer:\n\nConfidence: 50%": None,
        "Exact Answer: 3\nExplanation: revised\nExact Answer: 4\nConfidence: 1%": "4",
    }
    got = {t: hle_format.extract(t) for t in shapes}
    check(got == shapes, "HLE answers are read on the label's line, below it, or under a heading",
          str({k[:25]: v for k, v in got.items() if v != shapes[k]}))
    out = TMP / "baselines"
    out.mkdir(exist_ok=True)
    test_rows = json.loads(TEST.read_text())
    mc = [r for r in test_rows if r["answer_type"] == "multipleChoice"][:3]
    ex = [r for r in test_rows if r["answer_type"] != "multipleChoice"][:3]
    data = out / "data.json"
    data.write_text(json.dumps(mc + ex))
    (out / "server.py").write_text(FAKE_SERVER)
    port = _free_port()
    srv = subprocess.Popen([PY, str(out / "server.py"), str(port), str(out / "requests.jsonl")])
    try:
        time.sleep(2.5)
        ep = f"http://127.0.0.1:{port}"
        common = ["--data", str(data), "--endpoints", ep, "--k", "2", "--max-retries", "1"]
        subprocess.run([PY, str(ROOT / "baselines/generate.py"), "--out", str(out / "direct.jsonl")] + common,
                       check=True, capture_output=True)
        subprocess.run([PY, str(ROOT / "baselines/recover.py"), "--data", str(data), "--results", str(out / "direct.jsonl"),
                        "--out", str(out / "direct_rec.jsonl"), "--endpoints", ep, "--reasoning-effort", "high"],
                       check=True, capture_output=True)
        subprocess.run([PY, str(ROOT / "baselines/selfrefine.py"), "--out", str(out / "sr_rec.jsonl"), "--recover",
                        "--max-tokens", "24576", "--feedback-max-tokens", "16384"] + common, check=True, capture_output=True)
    finally:
        srv.terminate()
    reqs = [json.loads(l) for l in (out / "requests.jsonl").open()]
    firsts = [r for r in reqs if len(r["messages"]) == 2]
    check(firsts and all(r["messages"][0]["role"] == "system" and r["messages"][0]["content"] in
                         (hle_format.SYSTEM_EXACT, hle_format.SYSTEM_MC) for r in firsts),
          "every first request uses HLE's own format as the system prompt")
    check(all((r["messages"][0]["content"] == hle_format.SYSTEM_MC) == (r["messages"][1]["content"] in
                                                                       [x["question"] for x in mc]) for r in firsts),
          "multiple-choice questions get HLE's multiple-choice format")
    recs = [r for r in reqs if "ran out of space" in r["messages"][-1]["content"]]
    check(recs and all(r["effort"] == "low" and "[the end of my reasoning]" in r["messages"][-2]["content"]
                       for r in recs), "a cut-off reply is asked for its answer, at low effort, from its reasoning")
    fb = [r for r in reqs if "There may be an error" in r["messages"][-1]["content"]]
    check(fb and all("option" not in r["messages"][-1]["content"] for r in fb), "Self-Refine's HLE critique has no options")
    direct_rec = [json.loads(l) for l in (out / "direct_rec.jsonl").open()]
    check(len(direct_rec) == 12 and all(hle_format.extract(r["content"]) is not None for r in direct_rec),
          "every direct sample has an answer after recovery", str(len(direct_rec)))
    sr = [json.loads(l) for l in (out / "sr_rec.jsonl").open()]
    check(len(sr) == 12 and all(r["error"] is None and hle_format.extract(r["content"]) for r in sr),
          "every Self-Refine run ends with an answer", str(len(sr)))
    # scoring: answers read off HLE's answer line, graded by the judge, saved as marks
    before = JUDGE_CALLS["calls"]
    args = Namespace(data=str(data), results=str(out / "direct_rec.jsonl"), k=2, any_k=False, all_samples=False,
                     save=str(out / "direct_rec_k2.json"), judge_model=None, judge_cache=str(out / "judge.jsonl"),
                     judge_workers=4)
    score.main(args)
    sc = json.loads((out / "direct_rec_k2.json").read_text())["per_question"]
    gold = {r["id"]: r["answer"] for r in mc + ex}
    check(len(sc) == 6 and all(v["marks"] == [int(p is not None and D.normalize_answer(p) == D.normalize_answer(gold[q]))
                                              for p in v["preds"]] for q, v in sc.items()),
          "the scorer's marks are the judge's verdicts on the extracted answers")
    check(all(p is None or not p.startswith("$") for v in sc.values() for p in v["preds"]),
          "extracted answers are normalised ('$42$' -> '42')", str([v["preds"] for v in sc.values()][:3]))
    check(JUDGE_CALLS["calls"] - before <= len({(q, p) for q, v in sc.items() for p in v["preds"] if p}),
          "one judge call per distinct answer")


if __name__ == "__main__":
    test_formats_and_prompts()
    test_executor_open()
    test_programs_and_judge()
    test_pipeline()
    test_pipeline_run2()
    test_baselines()
    configure_letters()
    print()
    if T.FAILURES:
        print(f"{len(T.FAILURES)} FAILED: " + "; ".join(T.FAILURES))
        sys.exit(1)
    print(f"all checks passed (judge calls {dict(JUDGE_CALLS)}, open-answer calls {dict(OPEN_CALLS)}); temp dir {TMP}")

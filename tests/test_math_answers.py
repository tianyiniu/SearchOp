"""Offline checks for MATH answers (--answers math, scripts/math_answers.py): answer keys against
math-verify on recorded replies and on every MATH Level 5 key, the worker processes under threads,
reading answers from \\boxed{} and ANSWER lines, the prompts and the cache key, the executor's summary
and recovery paths, programs graded by math-verify, and that switching back leaves the
multiple-choice and HLE executors exactly as they were. A scripted client stands in for the model.

    python tests/test_math_answers.py
"""

from __future__ import annotations

import json
import re
import sys
import threading
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))   # the code under test
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baselines"))
sys.path.insert(0, str(Path(__file__).resolve().parent))                       # the other test files

import test_pipeline_cluster as T3  # noqa: E402  (installs the fake clients)
import debate_mcq as D  # noqa: E402
import schema_fitness as SF  # noqa: E402
import evolve_program_mcq as M  # noqa: E402
import program_space as P  # noqa: E402
import adaptive_debate_mcq as B  # noqa: E402
import math_answers as MA  # noqa: E402
import tasks  # noqa: E402  (baselines/tasks.py: the external baselines' MATH prompt and grading)

T = T3.T
check, TMP = T.check, T.TMP
ROOT = P.ROOT
R = T3._Resp
QWEN = "Qwen/Qwen3.5-9B"
TRAIN = json.loads((ROOT / "datasets/math_l5_train.json").read_text())
TEST = json.loads((ROOT / "datasets/math_l5_test.json").read_text())


def configure_math(**kw):
    return P.configure_executor(executor="v3", window=32768, answers="math", plain_instruction=True,
                                last_round_vote=True, count_read_summaries=True, high_cost=3, turn_cap=15,
                                total_cap=21, **kw)


def configure_letters():
    return P.configure_executor(executor="v3", window=32768, plain_instruction=True, last_round_vote=True,
                                count_read_summaries=True, high_cost=3, turn_cap=15, total_cap=21)


def recorded_answers() -> dict[str, set[str]]:
    """Every distinct last-\\boxed answer of the recorded Qwen3.5-4B baseline replies, per question."""
    out: dict[str, set[str]] = {}
    for f in ("sc5_qwen35_4b_think_math_l5_test_rec.jsonl", "selfrefine_it4_qwen35_4b_think_math_l5_test_rec.jsonl",
              "mad_a3r2_qwen35_4b_think_math_l5_test_rec.jsonl", "direct_qwen35_4b_think_math_l5_test_rec.jsonl"):
        path = ROOT / "baselines/results" / f
        if not path.exists():
            continue
        for line in path.open():
            r = json.loads(line)
            texts = r.get("finals") or [r.get("response") or r.get("final") or ""]
            for t in texts if isinstance(texts, list) else [texts]:
                if isinstance(t, str) and (a := tasks.last_boxed(t)) is not None:
                    out.setdefault(r["id"], set()).add(a)
    return out


# --- 1. keys and grading against math-verify ----------------------------------------------------

def read(a: str) -> str | None:
    """What the executor reads from a reply whose answer is \\boxed{a}: the answer's key."""
    return D.extract_letter("so \\boxed{" + a + "}", 0)


def test_keys():
    print("math answers: keys and grading against math-verify")
    configure_math()
    same = [("\\frac{3}{4}", "0.75"), ("\\dfrac{3}{4}", "\\frac{3}{4}"), ("\\sqrt{8}", "2\\sqrt{2}"),
            ("4(1+\\sqrt{2})", "4+4\\sqrt{2}"), ("\\left(\\frac{3}{5},\\frac{8}{3}\\right]", "(\\frac{3}{5}, \\frac{8}{3}]"),
            ("\\sqrt{0.4}", "\\frac{\\sqrt{10}}{5}"), ("3.2", "\\frac{16}{5}"), ("4(3 - x)(3 + x)", "4(3-x)(3+x)")]
    for a, b in same:
        check(MA.key(a) == MA.key(b), f"one key for {a!r} and {b!r}", f"{MA.key(a)!r} / {MA.key(b)!r}")
    differ = [("\\text{C,E}", "C,E"), ("\\frac{1}{3}", "0.333"), ("4(3-x)(3+x)", "36-4x^2"), ("2", "-2"), ("(1,2)", "[1,2]")]
    for a, b in differ:
        check(MA.key(a) != MA.key(b), f"different keys for {a!r} and {b!r}", f"{MA.key(a)!r} / {MA.key(b)!r}")
    check(MA.key("\\text{Monday}") == "Monday" and D.extract_letter("\\boxed{\\text{C,E}}", 0) == "\\text{C,E}",
          "a \\text{} wrapper is dropped only where math-verify reads the answer the same")
    check(MA.key("4(3-x)(3+x)") == "4(3-x)(3+x)", "an answer with variables keeps its own algebra (factored stays factored)")
    # every key of the MATH Level 5 test and train splits: stable, and graded right against itself
    golds = sorted({r["answer"] for r in TRAIN + TEST})
    unstable = [g for g in golds if read(read(g)) != read(g) or D.extract_letter("ANSWER: " + read(g), 0) != read(g)]
    check(not unstable, f"the key of each of the {len(golds)} MATH keys is stable (read again, boxed or on an ANSWER line)",
          str(unstable[:5]))
    wrong = [g for g in golds if not MA.grade(g, read(g))]
    check(not wrong, f"each of the {len(golds)} MATH keys, read as an answer, is graded right", str(wrong[:5]))
    # recorded answers: grading the key gives math-verify's verdict on the answer itself; no false merges
    answers = recorded_answers()
    gold = {r["id"]: r["answer"] for r in TEST}
    pairs = merged_wrongly = verdicts = verdict_diff = 0
    for q, v in answers.items():
        keys = {a: read(a) for a in v}
        for a in v:
            verdicts += 1
            verdict_diff += MA.grade(gold[q], keys[a]) != tasks.correct({"dataset": "math", "answer": gold[q]}, a)
        for a, b in combinations(sorted(v), 2):
            if keys[a] == keys[b]:
                pairs += 1
                merged_wrongly += not tasks.same({"dataset": "math"}, a, b)
    check(verdicts > 500 and verdict_diff == 0,
          f"on {verdicts} recorded answers, grading the key gives math-verify's verdict on the answer", str(verdict_diff))
    check(merged_wrongly == 0, f"no two recorded answers share a key unless math-verify calls them equal ({pairs} pairs)")
    check(MA.STATS["worker_failures"] == 0, "no worker call failed", str(MA.STATS))


def test_threads():
    print("math answers: worker processes under threads")
    probe = [f"\\frac{{{i}}}{{{i + 7}}}" for i in range(1, 60)] + [f"{i}.5" for i in range(40)]
    expect = {a: MA.key(a) for a in probe[:5]}
    MA._keys.clear()                       # make the threads do the work themselves
    out, errors = {}, []

    def work(chunk):
        try:
            for a in chunk:
                out[a] = MA.key(a)
                MA.grade("\\frac{1}{2}", a)
        except Exception as exc:           # noqa: BLE001
            errors.append(repr(exc))

    threads = [threading.Thread(target=work, args=(probe[i::16],)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check(not errors and len(out) == len(probe), "16 threads key and grade answers with no error", str(errors[:2]))
    check(all(out[a] == k for a, k in expect.items()) and out["0.5"] == "\\frac{1}{2}" and MA.grade("\\frac{1}{2}", "0.5"),
          "the results in threads are those of a single caller")
    check(MA.STATS["worker_failures"] == 0, "no worker call failed", str(MA.STATS))
    # a script read from stdin ('python -', as the pipeline's shell steps run their checks): its main
    # file name '<stdin>' names no file, and the workers must still start (they failed before 2026-10-07)
    import subprocess
    script = ("import sys\nsys.path.insert(0, %r)\nimport math_answers as MA\n"
              "print(MA.key('0.75'), MA.grade('\\\\frac{3}{4}', '0.75'), MA.STATS['worker_failures'])\n"
              % str(ROOT / "scripts"))
    p = subprocess.run([sys.executable, "-"], input=script, capture_output=True, text=True, cwd=str(TMP))
    check(p.stdout.split() == ["\\frac{3}{4}", "True", "0"] and "Traceback" not in p.stderr,
          "the workers start under a script read from stdin", (p.stdout + p.stderr[-300:]).strip())


# --- 2. reading answers ----------------------------------------------------------------------------

def test_reading():
    print("math answers: reading answers")
    configure_math()
    cases = [("so the answer is $\\boxed{\\frac{1}{2}}$.", "\\frac{1}{2}"),
             ("first \\boxed{3}, then \\boxed{4}", "4"),
             ("\\boxed{3}\nANSWER: 5", "5"),
             ("ANSWER: 5\nthen \\boxed{3}", "3"),
             ("ANSWER: \\boxed{0.5}", "\\frac{1}{2}"),
             ("an earlier \\boxed{3} then a box cut off: \\boxed{\\frac{1}{", None),
             ("\\boxed 7 is it", MA.key("7 is it")),
             ("\\fbox{12}", "12"),
             ("**ANSWER:** $x^2+1$", "x^2+1"),
             ("no answer here", None),
             ("ANSWER:", None)]
    for text, want in cases:
        got = D.extract_letter(text, 0)
        check(got == want, f"read {text[:40]!r} as {want!r}", repr(got))
    check(D._answer_letters("\\boxed{0.75} and then\nANSWER: \\frac{3}{4}", 0) == ["\\frac{3}{4}", "\\frac{3}{4}"]
          and D._answer_letters("\\boxed{\\frac{1}{", 0) == [],
          "summary checking reads every answer as its key, skipping an open box")
    check(D.commit_letter_lenient("ANSWER: 0.5\nbest guess", 0) == "\\frac{1}{2}"
          and D.commit_letter_lenient("the answer is 7", 0) is None, "the letter follow-up reads only an answer, not prose")
    # the same box rule as the external baselines (tasks.last_boxed) on every recorded reply
    texts = []
    for line in (ROOT / "baselines/results/sc5_qwen35_4b_think_math_l5_test_rec.jsonl").open():
        texts += [t for t in json.loads(line).get("finals") or [] if isinstance(t, str) and "ANSWER" not in t.upper()]
    texts = texts[:600]
    same_box = sum((tasks.last_boxed(t) is None) == (D.extract_letter(t, 0) is None)
                   and (tasks.last_boxed(t) is None or MA.key(D.normalize_answer(tasks.last_boxed(t))) == D.extract_letter(t, 0))
                   for t in texts)
    check(same_box == len(texts), f"on {len(texts)} recorded replies the answer read is the baselines' last box (keyed)",
          f"{same_box}/{len(texts)}")


# --- 3. prompts, settings, cache key, switching back -----------------------------------------------

def test_prompts():
    print("math answers: prompts, settings and keys")
    letters = configure_letters()
    letter_state = (dict(D.PERSONA_PROMPTS), D.EXPERT_TMPL, D.ANSWER_INSTR, D.COMMIT_NUDGE, SF.REWRITTEN_CRITIC,
                    SF.path_key([P.PLAN_ROUNDS["solver"]]), D.prompt_signature())
    hle = P.configure_executor(executor="v3", window=32768, answers="open", plain_instruction=True,
                               judge_cache=TMP / "judge_math_test.jsonl")
    hle_sig = D.prompt_signature()
    s = configure_math()
    check(s.get("answers") == "math" and "judge" not in s and P.JUDGE is None and M.GRADER is MA.grade_row,
          "the settings name the answer mode, and math-verify grades (no judge)", str(s))
    check(D.OPEN and D.MATH and s["prompts"] not in (letters["prompts"], hle_sig),
          "MATH answers have their own prompt signature", s["prompts"])
    k = json.loads(SF.path_key([P.PLAN_ROUNDS["solver"]]))
    check(k.get("o") == "m", "MATH recordings get their own cache key ('o': 'm')", str(k))
    used = ("critic", "verifier", "synthesizer", "expert")
    texts = [D.PERSONA_PROMPTS[p] for p in used] + [SF.REWRITTEN_CRITIC, D.EXPERT_TMPL.format(field="Algebra")]
    check(all("Put your final answer within \\boxed{}." in t for t in texts),
          "every answering speaker is asked for its answer in \\boxed{}")
    check(not any(re.search(r"\boption|multiple-choice|letter", t, re.I) for t in texts + [D.COMMIT_NUDGE]),
          "no MATH prompt speaks of options or letters")
    row = TEST[0]
    prompts = B.question_prompts(row)
    check("\\boxed{}" in prompts["expert"] and row["field"] in prompts["expert"],
          "the per-question expert prompt fills in the field and keeps \\boxed{}", prompts["expert"][:120])
    # the solver: the external direct baseline's request, then the discussion
    c = T3._Scripted(R("Work. " * 30 + "\nSo $\\boxed{5}$."))
    out = D.execute_round(c, QWEN, row["question"], row["options"], {"personas": ["solver"]}, [], 1.0)
    sent = c.requests[0]["messages"]
    check(sent[-1]["content"] == tasks.messages(row)[0]["content"]
          and sent[0]["content"] == D.solver_system(D.SOLVER_SYSTEM, "low"),
          "a blind solver is sent the external baseline's MATH request, with the solver's system prompt")
    check(D.extract_letter(out[0][1], 0) == "5", "and its boxed answer is read")
    prior = [[("solver", "Work.\n\\boxed{5}"), ("solver", "Other work.\nANSWER: 0.5")]]
    c = T3._Scripted(R("Checked.\n\\boxed{5}"))
    D.execute_round(c, QWEN, row["question"], row["options"], {"personas": ["solver"]}, prior, 1.0,
                    prior_specs=[{"personas": ["solver", "solver"]}])
    user = c.requests[0]["messages"][-1]["content"]
    check(user.startswith(tasks.messages(row)[0]["content"]) and "[Round 1, solver 2] Answer: \\frac{1}{2}" in user
          and "You speak in round 2." in user, "a solver that sees the debate gets the request, then the discussion")
    c = T3._Scripted(R("Critique.\n\\boxed{5}"))
    D.execute_round(c, QWEN, row["question"], row["options"], {"personas": ["critic"]}, prior, 1.0,
                    prompts=prompts, prior_specs=[{"personas": ["solver", "solver"]}])
    check(c.requests[0]["messages"][0]["content"] == SF.REWRITTEN_CRITIC
          and c.requests[0]["messages"][-1]["content"].endswith("Put your final answer within \\boxed{}."),
          "the critic gets the MATH critic prompt and the boxed instruction")
    # switching back: the multiple-choice executor exactly as before
    configure_letters()
    state = (dict(D.PERSONA_PROMPTS), D.EXPERT_TMPL, D.ANSWER_INSTR, D.COMMIT_NUDGE, SF.REWRITTEN_CRITIC,
             SF.path_key([P.PLAN_ROUNDS["solver"]]), D.prompt_signature())
    check(state == letter_state and not D.OPEN and not D.MATH and M.GRADER is None,
          "switching back restores the multiple-choice prompts, key, critic, signature and grading exactly")
    P.configure_executor(executor="v3", window=32768, answers="open", plain_instruction=True,
                         judge_cache=TMP / "judge_math_test.jsonl")
    check(D.prompt_signature() == hle_sig and not D.MATH, "and the HLE executor too")
    configure_math()
    try:
        P.check_rows(TEST[:3])
        ok = True
    except SystemExit:
        ok = False
    check(ok, "MATH rows are accepted")
    configure_letters()
    try:
        P.check_rows(TEST[:3])
        check(False, "MATH rows are refused in letter mode")
    except SystemExit:
        check(True, "MATH rows are refused in letter mode")


# --- 4. the executor: summaries and recovery ---------------------------------------------------------

def test_executor():
    print("math answers: summaries and recovery")
    configure_math()
    c = T3._Scripted(R("Short.\n\\boxed{0.75}", "hidden " * 300), R("Summary.\n\\boxed{\\frac{3}{4}}\nANSWER: \\frac{3}{4}"))
    out = D.chat_v3(c, QWEN, "sys", "question", 0, "high")
    check(D.SUMMARY_MARK in out and D.extract_letter(out, 0) == "\\frac{3}{4}"
          and "Your answer is final: \\frac{3}{4}." in c.requests[1]["messages"][-1]["content"],
          "a high-effort reply is summarised with its answer locked; an equal form in the summary is accepted")
    c = T3._Scripted(R("Short.\n\\boxed{3}", "hidden " * 300), R("Actually 4.\n\\boxed{4}"))
    out = D.chat_v3(c, QWEN, "sys", "question", 0, "high")
    check(D.SUMMARY_MARK not in out and D.extract_letter(out, 0) == "3", "a summary that gives another answer is not used")
    c = T3._Scripted(R("Still working. " * 600 + "so \\boxed{\\frac{1}{"), R("ANSWER: 7\nwhere it led"))
    out = D.chat_v3(c, QWEN, "sys", "question", 0, "low")
    check(D.extract_letter(out, 0) == "7" and c.requests[1]["messages"][-1]["content"].startswith("Your reply ended"),
          "a reply cut off inside its box gets its answer from the answer-first summary")
    c = T3._Scripted(R("Still working. " * 600), R("No answer line."), R("ANSWER: 2\\sqrt{2}"))
    out = D.chat_v3(c, QWEN, "sys", "question", 0, "low")
    check(D.extract_letter(out, 0) == "2 \\sqrt{2}" and c.requests[2]["messages"][-1]["content"] == D.COMMIT_NUDGE_MATH,
          "failing that, the follow-up asking for the answer gets it")


# --- 5. programs: votes over keys, graded by math-verify ----------------------------------------------

class _Runner:
    def __init__(self, rounds):
        self.rounds = list(rounds)

    def run_round(self, qid, rounds, specs, spec, rep=0, prompts=None):
        return [(p, r) for p, r in zip(spec["personas"], self.rounds.pop(0))]


def test_programs():
    print("math answers: programs")
    configure_math()
    row = {"id": "m1", "question": "Find x.", "options": [], "answer": "\\frac{3}{4}", "dataset": "math"}
    vote = P.normalize_program(P.PROTOCOLS["self_consistency_high"])          # three solvers, plurality
    out = M.run_program(vote, _Runner([["\\boxed{1}", "\\boxed{0.75}", "so \\boxed{\\dfrac{3}{4}}"]]), row)
    check(out["letter"] == "\\frac{3}{4}" and out["correct"], "a vote counts equal answers in different forms together",
          str(out))
    out = M.run_program(vote, _Runner([["\\boxed{1}", "\\boxed{1.0}", "\\boxed{\\frac{3}{4}}"]]), row)
    check(out["letter"] == "1" and not out["correct"], "the plurality answer is graded by math-verify")
    out = M.run_program(vote, _Runner([["nothing", "\\boxed{\\frac{3}{", "no"]]), row)
    check(out["letter"] is None and not out["correct"], "no answer is graded wrong")
    prog = {"plan": [{"personas": ["solver", "solver"]}, {"personas": ["critic"]}],
            "rules": [{"when": ["step==1", "n_distinct>=2"], "do": "continue"},
                      {"when": ["plan_left", "step==0"], "do": "continue"}], "default": "stop:last_commit"}
    P.validate_program(prog)
    out = M.run_program(prog, _Runner([["\\boxed{0.75}", "\\boxed{\\frac{3}{4}}"], ["\\boxed{2}"]]), row)
    check(out["actions"] == ["continue"] and out["correct"], "equal forms are one answer to the rules (n_distinct 1)",
          str(out))
    out = M.run_program(prog, _Runner([["\\boxed{0.75}", "\\boxed{2}"], ["ANSWER: 0.75"]]), row)
    check(out["actions"] == ["continue", "continue"] and out["letter"] == "\\frac{3}{4}" and out["correct"],
          "different answers continue the debate; the last round's answer is read", str(out))
    check(MA.STATS["worker_failures"] == 0, "no worker call failed", str(MA.STATS))


# --- 6. the pipeline: seeds, search, champions, comparisons, test evaluation -------------------------
# Small synthetic groups (one per subject) stand in for the MATH clusters; the fake model answers in
# \boxed{} (the key, an equal form of it, or a wrong value) and follows every follow-up prompt.


import hashlib as _hl  # noqa: E402,F811

import compare_external_baselines as CE  # noqa: E402
import eval_routed_dev as E  # noqa: E402
import evolve_pipeline_cluster as V  # noqa: E402
import program_seeds_cluster as S3  # noqa: E402
import train_baselines as TB  # noqa: E402
from collections import Counter  # noqa: E402

run_cli = T.run_cli
SUBJECTS = ("Algebra", "Number Theory", "Geometry")
MATH_CALLS: Counter = Counter()
_prev_create = T.FakeCompletions.create


def build_tiny(out: Path) -> dict:
    """24 train questions (8 per subject; 2 of each held out as dev), 9 test questions, their groups
    and routes."""
    out.mkdir(parents=True, exist_ok=True)
    train = [r for s in SUBJECTS for r in [x for x in json.loads((ROOT / "datasets/math_l5_300_train.json")
                                                                  .read_text()) if x["discipline"] == s][:8]]
    test = [r for s in SUBJECTS for r in [x for x in TEST if x["discipline"] == s][:3]]
    prof = {"steps": {"calculate_value": 0.8, "formalize_conditions_and_model": 0.5},
            "challenges": {"calculation_or_convention_slip": 0.6}, "knowledge": {"derive": 1.0}}
    clusters = {"rep": "description", "mh_weight": 0.5, "k": 3, "seed": 0, "per_cluster": 100,
                "labels": {r["id"]: i for i, s in enumerate(SUBJECTS) for r in train if r["discipline"] == s},
                "clusters": [{"cluster": i, "size": 8, "medoid": ids[0], "medoid_template": f"A {s.lower()} problem.",
                              "subset": ids, "held_out": [], "members": ids, **prof}
                             for i, s in enumerate(SUBJECTS)
                             for ids in [[r["id"] for r in train if r["discipline"] == s]]]}
    dev = [ids[j] for c in clusters["clusters"] for ids in [c["subset"]] for j in (1, 5)]
    paths = {"train": out / "train.json", "test": out / "test.json", "clusters": out / "clusters.json",
             "routes": out / "routes.json", "splits": out / "splits.json"}
    paths["train"].write_text(json.dumps(train))
    paths["test"].write_text(json.dumps(test))
    paths["clusters"].write_text(json.dumps(clusters))
    paths["splits"].write_text(json.dumps({"train": [r["id"] for r in train if r["id"] not in dev], "dev": dev}))
    routes = {r["id"]: {"group": SUBJECTS.index(r["discipline"]), "second": (SUBJECTS.index(r["discipline"]) + 1) % 3,
                        "margin": 0.1 + 0.01 * i, "distances": {"0": 0.5, "1": 0.6, "2": 0.7},
                        "knn": SUBJECTS.index(r["discipline"]), "question": SUBJECTS.index(r["discipline"])}
              for i, r in enumerate(test)}
    paths["routes"].write_text(json.dumps({"clusters": str(paths["clusters"]), "routes": routes}))
    return paths


def _gold(text: str) -> str:
    for r in TRAIN + TEST:
        if text.startswith(r["question"][:200]):
            return r["answer"]
    return "0"


def _create_math(self, model, messages, temperature=None, max_tokens=None, extra_body=None, **kw):
    if not (D.V3 and D.MATH):
        return _prev_create(self, model, messages, temperature, max_tokens, extra_body, **kw)
    last = messages[-1]["content"]
    if last.startswith("Now write a summary"):
        MATH_CALLS["summary"] += 1
        locked = re.search(r"Your answer is final: (.*)\. This summary reports", last, re.S).group(1)
        if int(_hl.sha1(messages[-2]["content"].encode()).hexdigest(), 16) % 6 == 0:
            MATH_CALLS["argued"] += 1
            return T3._Resp("On reflection the value is another one.\n\\boxed{999}")     # must be rejected
        return T3._Resp(f"The decisive step points one way.\nANSWER: {locked}")
    if last.startswith("Your reply ended before"):
        MATH_CALLS["answer_first"] += 1
        return T3._Resp(f"ANSWER: {_gold(messages[1]['content'] if len(messages) > 1 else '')}\nIt led there.")
    if last == D.COMMIT_NUDGE:
        MATH_CALLS["commit"] += 1
        return T3._Resp("ANSWER: 5")
    T.MODEL_CALLS["chat"] += 1
    T.FakeCompletions.salt += 1
    high = (extra_body or {}).get("chat_template_kwargs", {}).get("enable_thinking", False)
    user = next(m["content"] for m in messages if m["role"] == "user")
    gold = _gold(user)
    h = int(_hl.sha1(f"{messages[0]['content'][:30]}|{last}|{T.FakeCompletions.salt}".encode()).hexdigest(), 16)
    ans = gold if h % 3 else "17"
    if ans == gold and re.fullmatch(r"-?\d+", gold) and h % 7 == 0:
        ans = gold + ".0"                                  # an equal form: one answer to the vote
        MATH_CALLS["equal_form"] += 1
    form = (f"\\boxed{{{ans}}}", f"So the answer is $\\boxed{{{ans}}}$.", f"ANSWER: {ans}")[(h // 3) % 3]
    shape = (h // 9) % 4
    thinking = "Private reasoning step. " * 60 if high else None
    if shape == 0:
        return T3._Resp("A short argument. " * 30 + "\n" + form, thinking)
    if shape == 1:
        return T3._Resp("One long derivation step. " * 150 + "\n" + form, thinking)
    if shape == 2:
        MATH_CALLS["unfinished"] += 1
        return T3._Resp("Still working through the problem without finishing. " * 20 + " \\boxed{\\frac{1}{", thinking)
    return T3._Resp(form, "Hidden reasoning. " * 40)


T.FakeCompletions.create = _create_math


def test_pipeline():
    print("math answers: seeds, search, champions, comparisons and test evaluation")
    out = TMP / "math_pipeline"
    paths = build_tiny(out / "data")
    flags = ["--context-window", "32768", "--answers", "math", "--plain-instruction", "--last-round-vote",
             "--count-read-summaries", "--high-cost", "3", "--turn-cap", "15", "--total-cap", "21"]
    common = ["--clusters", str(paths["clusters"]), "--dataset", str(paths["train"]), "--per-group", "0",
              "--dev-split", str(paths["splits"]), "--live-cache", str(out / "rounds.jsonl"), "--workers", "8"] + flags
    before_failures = MA.STATS["worker_failures"]
    run_cli(S3, common + ["--out", str(out / "seeds.json")])
    seeds = json.loads((out / "seeds.json").read_text())
    src = Counter(s["source"] for s in seeds["seeds"])
    check(seeds["settings"].get("answers") == "math" and src.get("llm", 0) >= 3 and src.get("protocol", 0) >= 8,
          "the seed stage writes literature and model-written seeds under MATH answers", str(dict(src)))
    run = out / "run"
    search = common + ["--seeds", str(out / "seeds.json"), "--out", str(run), "--tie-questions", "1"]
    run_cli(V, search + ["--generations", "2"])
    lines = [json.loads(l) for l in (run / "archive.jsonl").open() if l.strip()]
    header, recs = lines[0], {}
    for d in lines[1:]:
        recs[d["key"]] = d
    check(header["settings"].get("answers") == "math" and len(header["qids"]) == 18,
          "the search runs under MATH answers on the 18 non-dev questions", str(len(header["qids"])))
    answers = [v[2] for d in recs.values() for rep in d["reps"].values() for v in rep.values()]
    check(not any(re.search(r"\\boxed|\$|ANSWER", a) for a in answers) and any(a == "17" for a in answers),
          "archived answers are keys (no box, no dollar signs, no ANSWER label)", str(Counter(answers).most_common(4)))
    marks = [v[0] for d in recs.values() for rep in d["reps"].values() for v in rep.values()]
    check(0 < sum(marks) < len(marks), "some answers are graded right and some wrong", f"{sum(marks)}/{len(marks)}")
    gold_of = {r["id"]: r["answer"] for r in TRAIN}
    bad = [(q, v[2], v[0]) for d in recs.values() for rep in d["reps"].values() for q, v in rep.items()
           if v[2] != "?" and bool(v[0]) != tasks.correct({"dataset": "math", "answer": gold_of[q]}, v[2])]
    check(not bad, "every archived mark is math-verify's verdict on the archived answer (as the baselines grade)",
          str(bad[:3]))
    n_calls = T.MODEL_CALLS["chat"]
    run_cli(V, search + ["--generations", "2", "--resume"])
    check(T.MODEL_CALLS["chat"] == n_calls, "a resumed search replays every debate from the cache")
    run_cli(V, search + ["--pick-champions", "--reps", "2", "--baselines", ""])
    champs = json.loads((run / "champions.json").read_text())
    check(all(r["champion"] for r in champs["per_group"].values()) and champs["global"]["champion"],
          "the champion step picks a champion per group and overall on the dev questions")
    # the external baselines' score files: marks and tokens per run (baselines/score.py --save)
    def score_file(qids, marks_of, name):
        p = out / name
        p.write_text(json.dumps({"report": {}, "per_question": {
            q: {"preds": ["1", "2", "3"], "marks": marks_of(i), "answer": "x", "tokens": [900, 1000, 1100]}
            for i, q in enumerate(qids)}}))
        return p
    search_q = out / "search_questions.json"
    run_cli(TB, ["--export", "--clusters", str(paths["clusters"]), "--per-group", "0", "--dev-split",
                 str(paths["splits"]), "--dataset", str(paths["train"]), "--out", str(search_q)])
    sq = [r["id"] for r in json.loads(search_q.read_text())]
    check(len(sq) == 18 and set(sq) == set(header["qids"]), "the exported search questions are the search's")
    ext_sr = score_file(sq, lambda i: [1, 0, i % 2], "ext_sr_k3.json")
    ext_d = score_file(sq, lambda i: [1, 1, 0], "ext_direct_k3.json")
    run_cli(TB, ["--compare", "--run", str(run), "--external", f"self-refine={ext_sr},direct={ext_d}"])
    tb = json.loads((run / "train_baselines.json").read_text())
    check(abs(tb["table"]["external direct"]["acc"] - 2 / 3) < 1e-9, "the train comparison reads the external marks")
    raw = out / "ext_direct_raw.jsonl"
    raw.write_text("".join(json.dumps({"id": q, "sample_idx": k, "error": None,
                                       "finish_reason": "length" if (i + k) % 9 == 0 else "stop"}) + "\n"
                           for i, q in enumerate(sq) for k in range(3)))
    try:
        run_cli(CE, ["--questions", str(search_q), "--out", str(out / "external_baselines"),
                     "--external-direct", str(ext_d), "--external-direct-raw", str(raw),
                     "--external-selfrefine", str(ext_sr), "--model", QWEN, "--base-urls", "http://localhost:1/v1",
                     "--live-cache", str(out / "rounds.jsonl"), "--workers", "8", "--ignore-cache-lock"]
                + flags)
    except SystemExit as exc:                              # it exits 0 on a pass, 1 on a fail
        check(exc.code in (0, 1), "the comparison ends with a verdict (0 pass, 1 fail)", str(exc.code))
    ce = json.loads((out / "external_baselines.json").read_text())
    names = [c["name"] for c in ce["checks"]]
    check(ce["settings"].get("answers") == "math" and any(n.endswith("no answer") for n in names)
          and not any("letter" in n for n in names),
          "the comparison runs on MATH and names its check 'no answer'", str(names))
    run_cli(E, ["--run", str(run), "--routes", str(paths["routes"]), "--dataset", str(paths["test"]),
                "--out", str(out / "test_eval"), "--live-cache", str(out / "test_rounds.jsonl"), "--reps", "1",
                "--workers", "8", "--baselines", "direct_high,self_refine_high",
                "--external", f"self-refine={score_file([r['id'] for r in json.loads(paths['test'].read_text())], lambda i: [i % 2], 'ext_test_k1.json')}"]
            + flags)
    res = json.loads((out / "test_eval/results_k1.json").read_text())
    check(res["n_questions"] == 9 and res["settings"].get("answers") == "math"
          and 0 < res["table"]["routed, held-out champions"]["avg@1"] <= 1,
          "the test evaluation routes the champions over the MATH test questions and grades them",
          str({k: round(v["avg@1"], 3) for k, v in res["table"].items()}))
    check(MATH_CALLS["argued"] > 0 and MATH_CALLS["answer_first"] > 0 and MATH_CALLS["equal_form"] > 0
          and MATH_CALLS["summary"] > 0, "the fake exercised locked summaries, rejected ones, answer-first "
          "recovery and equal answer forms", str(dict(MATH_CALLS)))
    check(MA.STATS["worker_failures"] == before_failures, "no math-verify worker call failed", str(MA.STATS))


if __name__ == "__main__":
    test_keys()
    test_threads()
    test_reading()
    test_prompts()
    test_executor()
    test_programs()
    test_pipeline()
    configure_letters()
    print(f"\n{len(T.FAILURES)} failures" if T.FAILURES else "\nall checks passed")
    sys.exit(1 if T.FAILURES else 0)

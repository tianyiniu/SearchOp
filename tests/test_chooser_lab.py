"""Offline checks for gap_report.py and chooser_lab.py: the gap accounting on toy
transcripts (its identities, the round-1 patterns), the executor settings of the recorded
runs, pools read from a toy recorded cache, every chooser against a scripted model (picks,
off-list re-picks, the tournament rule, the verify fallback), the call cache (resume, no
repeated calls), the judge replayed from the recording without a call, the metrics, and
that the recorded caches are never written. No vLLM.

    python tests/test_chooser_lab.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import types
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))   # the code under test
sys.path.insert(0, str(Path(__file__).resolve().parent))                       # the other test files

import adaptive_debate_mcq as B  # noqa: E402
import debate_mcq as D  # noqa: E402
import evolve_program_mcq as M  # noqa: E402
import program_space as P  # noqa: E402
import schema_fitness as SF  # noqa: E402
import gap_report as GR  # noqa: E402
import chooser_lab as CL  # noqa: E402

FAILURES: list[str] = []
TMP = Path(tempfile.mkdtemp(prefix="chooser_lab_test_"))
GPT = "openai/gpt-oss-20b"
RUN2 = {"executor": "v3", "digest": [2000, 2000], "summary_words": 500, "window": 32768, "high_cost": 5,
        "eliminator": False, "visible_reasoning": True, "model": GPT}
RUN3 = {"executor": "v3", "digest": [2000, 2000], "summary_words": 500, "window": 32768, "high_cost": 3,
        "eliminator": False, "turn_cap": 20, "judge_persona": True, "any_round_width": True,
        "visible_reasoning": True, "model": GPT}


def check(cond, name: str, detail: str = "") -> None:
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else f"  [{detail}]"))
    if not cond:
        FAILURES.append(name)


def ans(letter: str, words: str = "Reasoning") -> str:
    return f"{words} about the question.\nANSWER: {letter}"


# --- 1. gap accounting -------------------------------------------------------------------

def test_gap():
    print("gap accounting")
    s = lambda *ls: [("solver", ans(l)) for l in ls]                       # noqa: E731
    cases = {
        "minority, adopted": ([s("B", "C", "B"), [("critic", ans("C"))]], "C", "C"),
        "minority, not adopted": ([s("B", "C", "B")], "C", "B"),
        "plurality, abandoned": ([s("C", "C", "B"), s("B", "B", "B")], "C", "B"),
        "plurality, outvoted": ([s("C", "C", "B"), s("B", "C", "B")], "C", "B"),
        "late, not adopted": ([s("A", "A"), [("critic", ans("C"))], s("A")], "C", "A"),
        "never present": ([s("A", "B")], "C", "A"),
        "single, right": ([s("C")], "C", "C"),
    }
    got = {k: GR.anatomy(r, 4, right, final) for k, (r, right, final) in cases.items()}
    check(got["minority, adopted"]["cf"] and not got["minority, adopted"]["p1"] and got["minority, adopted"]["cov1"],
          "a minority answer adopted later is a correct flip")
    check(got["minority, not adopted"]["minority_lost"] and not got["minority, not adopted"]["wf"],
          "a round-1 minority never adopted is 'minority, not adopted'")
    check(got["plurality, abandoned"]["abandoned"] and not got["plurality, abandoned"]["outvoted"],
          "the round-1 plurality, gone from the last round: abandoned")
    check(got["plurality, outvoted"]["outvoted"] and got["plurality, outvoted"]["wf"],
          "the round-1 plurality, still committed in the last round but not read: outvoted")
    check(got["late, not adopted"]["late_lost"] and got["late, not adopted"]["cov_by_round"] == [False, True, True],
          "an answer first seen in round 2 and not read: late; coverage by round", str(got["late, not adopted"]))
    check(not got["never present"]["covered"] and got["single, right"]["pattern"] == "single", "never present; single")
    items = [dict(v, turns=1, tokens=10) for v in got.values()]
    summ = GR.summarize(items)                      # asserts the identities itself
    check(summ["cf"] == 1 and summ["wf"] == 2 and summ["net"] == -1 and abs(summ["precision"] - 1 / 3) < 1e-9,
          "flip counts and precision", str(summ))
    pats = {k: GR.pattern(r, SF.committed_letters([r], 4)) for k, r in {
        "unanimous": s("A", "A", "A"), "majority": s("A", "A", "A", "B"), "plurality": s("A", "A", "B", "C"),
        "tie": s("A", "A", "B", "B"), "all different": s("A", "B", "C"), "single": s("A")}.items()}
    check(pats == {"unanimous": "unanimous", "majority": "majority", "plurality": "plurality", "tie": "tie",
                   "all different": "tie", "single": "single"}, "round-1 patterns", str(pats))
    check(GR.plurality(["B", "C", "C", "B"]) == ("B", ["B", "C"]), "plurality: ties to the first committed")


# --- 2. the recorded executor ----------------------------------------------------------------

def test_settings():
    print("executor settings of the recorded runs")
    check(GR.configure_like(RUN2) == 16 and D.HIGH_COST == 5 and not D.JUDGE_ON, "run2: cap 16, no judge")
    check(GR.configure_like(RUN3) == 20 and D.HIGH_COST == 3 and D.JUDGE_ON and "judge|high" in P.PLAN_ROUNDS,
          "run3: cap 20, high costs 3, the judge")
    try:
        GR.configure_like({**RUN3, "summary_words": 400, "digest": [1, 2]})
        bad = True
    except SystemExit:
        bad = False
    check(bad, "non-default values are passed through, not dropped")


# --- 3. pools and choosers --------------------------------------------------------------------

ROWS = {f"q{i}": {"id": f"q{i}", "question": f"Toy question {i}?", "options": ["alpha", "beta", "gamma", "delta"],
                  "answer_letter": right, "field": "toys"}
        for i, right in enumerate(["C", "D", "A", "B"], 1)}
HIGH = {"q1": "BBCB", "q2": "ACDA", "q3": "AAAA", "q4": "BCBB"}
LOW = {"q1": "CC", "q2": "BB", "q3": "AA", "q4": "CC"}


def write_cache(path: Path) -> None:
    lines = []
    for q, row in ROWS.items():
        prompts = B.question_prompts(row)
        for eff, letters in (("high", HIGH[q]), ("low", LOW[q])):
            for i, l in enumerate(letters):
                text = ans(l, f"{eff} solver {i} of {q} says {l}") + (
                    D.SUMMARY_MARK + f"Summary by {eff} {i}: {l} because.\nANSWER: {l}" if eff == "high" else "")
                lines.append({"q": q, "k": SF.blind_key("solver", eff, i, prompts), "r": 0,
                              "responses": [["solver", text]], "error": None,
                              "usage": [{"persona": "solver", "calls": 1, "prompt": 10, "completion": 100}]})
    # the judge of q1's three-solver pool, as judge_on_disagree_high recorded it
    specs = [P.PLAN_ROUNDS["solver_x3|high"], P.PLAN_ROUNDS["judge|high"]]
    lines.append({"q": "q1", "k": SF.path_key(specs, B.question_prompts(ROWS["q1"])), "r": 0,
                  "responses": [["judge", ans("C", "the judge")]], "error": None,
                  "usage": [{"persona": "judge", "calls": 1, "prompt": 10, "completion": 777}]})
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))


class _Msg:
    def __init__(self, content, reasoning):
        self.content, self.reasoning_content = content, reasoning


class _Resp:
    def __init__(self, content, reasoning="thinking " * 5):
        self.choices = [types.SimpleNamespace(message=_Msg(content, reasoning), finish_reason="stop")]
        self.usage = types.SimpleNamespace(prompt_tokens=50, completion_tokens=20)


class Fake:
    """A scripted chooser model: `policy(system, user, followup_nudge or None) -> reply`."""

    def __init__(self, policy):
        self.policy, self.requests = policy, []
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, model, messages, max_tokens=None, **kw):
        self.requests.append(messages)
        nudge = messages[3]["content"] if len(messages) == 4 else None
        return _Resp(self.policy(messages[0]["content"], messages[1]["content"], nudge))


def shown_pair(user: str) -> tuple[str, str]:
    a = re.search(r"Response 1 \(answer ([A-J])\)", user).group(1)
    b = re.search(r"Response 2 \(answer ([A-J])\)", user).group(1)
    return a, b


def right_of(user: str) -> str:
    q = re.search(r"Toy question (\d)", user).group(1)
    return ROWS[f"q{q}"]["answer_letter"]


def make_lab(policy, cache_file: Path, live_runner: bool = True, show: str = "summary"):
    runner = M.BudgetedRunner(ROWS, [], lock=False, base_urls="http://localhost:1/v1", model=GPT,
                              temperature=1.0, answer_tokens=P.ANSWER_TOKENS,
                              cache_path=TMP / f"lab_rounds_{len(list(TMP.iterdir()))}.jsonl", max_calls=None,
                              progress=False)
    CL.load_recorded(runner, [cache_file])
    runner.reset_budget(None if live_runner else 0)
    caller = CL.Caller(TMP / "calls.jsonl", "", GPT, live=False)
    fake = Fake(policy)
    caller.live, caller.clients = True, iter(lambda: fake, None)
    return CL.Lab(ROWS, caller, runner, show, True), fake, runner, caller


def digest(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def test_lab():
    print("pools and choosers")
    GR.configure_like(RUN3)
    cache = TMP / "recorded.jsonl"
    write_cache(cache)
    before = digest(cache)

    def right_policy(system, user, nudge):
        if nudge is not None:
            return "ANSWER: C"
        if system.startswith("You are a Verifier"):
            l = re.search(r"The participant answered ([A-J])", user).group(1)
            return "Checked.\nVERDICT: " + ("CORRECT" if l == right_of(user) else "INCORRECT")
        return ans(right_of(user), "picked")

    lab, fake, runner, caller = make_lab(right_policy, cache)
    pools = {p["q"]: p for p in CL.load_pools(runner, ROWS, "3H")}
    check(sorted(pools) == ["q1", "q2", "q3", "q4"] and [CL.is_split(pools[q]) for q in sorted(pools)] == [True, True, False, True],
          "3H: four pools read from the recording; q3 unanimous")
    check(pools["q1"]["plural"] == "B" and pools["q1"]["tied"] == ["B"] and pools["q2"]["tied"] == ["A", "C", "D"]
          and pools["q2"]["plural"] == "A" and CL.ranked(pools["q2"]) == ["A", "C", "D"],
          "votes, plurality and ranking (ties to the first committed)")
    mixed = {p["q"]: p for p in CL.load_pools(runner, ROWS, "2H2L")}
    check(mixed["q1"]["votes"] == Counter({"B": 2, "C": 2}) and mixed["q1"]["tied"] == ["B", "C"]
          and [s["effort"] for s in mixed["q1"]["speakers"]] == ["high", "high", "low", "low"]
          and not CL.load_pools(runner, ROWS, "4H")[0]["speakers"][3]["text"].startswith("low"),
          "2H2L: two high then two low; 4H: the fourth high solver")
    rep = CL.representative(pools["q1"], "C")
    check(CL.argument(rep, "summary") == "Summary by high 2: C because.\nANSWER: C"
          and CL.argument(rep, "full").startswith("high solver 2 of q1") and "[summary]" in CL.argument(rep, "full"),
          "an argument is the summary, or reply and summary")

    j = lab.run(pools["q1"], "judge")
    check(j["chosen"] == "C" and j["completion"] == 777 and runner.novel_calls == 0 and not fake.requests,
          "the judge is replayed from the recording: no model call", str(j))
    lab0, _, runner0, _ = make_lab(right_policy, cache, live_runner=False)
    check(lab0.run(pools["q2"], "judge") is None, "a judge round not recorded is missing without a live server")
    runner0.close()

    n0 = len(fake.requests)
    t = lab.run(pools["q1"], "pairwise")
    check(t["chosen"] == "C" and len(fake.requests) - n0 == 2 and t["matches"] == [["B", "C", "C", "C"]],
          "pairwise: the challenger that wins both orders takes over", str(t))
    t2 = lab.run(pools["q2"], "pairwise")
    check(t2["chosen"] == "D" and len(t2["matches"]) == 2, "pairwise over three answers: two matches", str(t2))
    req = next(m for m in fake.requests if "Response 1 (answer B)" in m[1]["content"])
    check("[solver]" not in req[1]["content"] and "Summary by high 0: B" in req[1]["content"]
          and "Summary by high 2: C" in req[1]["content"], "the pairwise prompt shows the two summaries, no names")
    n1 = len(fake.requests)
    lab.run(pools["q1"], "pairwise")
    pr = lab.probe(pools["q1"])
    check(len(fake.requests) == n1 and pr["right_first"] == "C" and pr["right_second"] == "C" and pr["wrong"] == "B",
          "the probe reuses the tournament's calls (same prompts): no new call", str(pr))

    r = lab.run(pools["q4"], "resolve")
    user = fake.requests[-1][1]["content"]
    check(r["chosen"] == "B" and "B) beta" in user and "C) gamma" in user and "A) alpha" not in user
          and "D) delta" not in user, "resolve: only the candidate options are shown", user[:200])
    jb = lab.run(pools["q2"], "judge_blind")
    user = fake.requests[-1][1]["content"]
    check(jb["chosen"] == "D" and "[Response 1]" in user and "[solver]" not in user and "B, D" not in user
          and "A, C, D" in user, "judge_blind: unlabelled responses, sorted candidate list", user[-400:])
    v = lab.run(pools["q2"], "verify")
    check(v["chosen"] == "D" and v["verdicts"] == {"A": "INCORRECT", "C": "INCORRECT", "D": "CORRECT"},
          "verify: the one answer verified correct", str(v))

    # the off-list pick, the missing pick, and the verify fallbacks
    def stubborn(system, user, nudge):
        if nudge is not None:
            return "ANSWER: C" if "B, C" in nudge else "VERDICT: CORRECT"
        if system.startswith("You are a Verifier"):
            return "I cannot decide."
        return "ANSWER: A"                       # never a candidate here

    lab2, fake2, runner2, _ = make_lab(stubborn, cache, show="full")
    p = lab2.pair(pools["q1"], "B", "C")
    check(p["value"] == "C" and p["calls"] == 2 and "ANSWER: A" in p["text"] and len(fake2.requests[-1]) == 4
          and "[the end of my private reasoning]" in fake2.requests[-1][2]["content"],
          "an off-list pick gets one low-effort pick prompt with the reasoning tail", str(p))
    v = lab2.run(pools["q2"], "verify")
    check(v["chosen"] == "A" and set(v["verdicts"].values()) == {"CORRECT"},
          "verify: every answer CORRECT -> the plurality", str(v))

    def no_pick(system, user, nudge):
        return "still nothing" if nudge else "no answer line"

    lab3, _, runner3, _ = make_lab(no_pick, cache, show="full")
    r = lab3.run(pools["q1"], "resolve")
    check(r["chosen"] == "B" and r["no_pick"], "no pick at all -> the plurality, counted", str(r))

    # resume: a new caller over the same file makes no call
    n_lines = len((TMP / "calls.jsonl").read_text().splitlines())
    lab4, fake4, runner4, _ = make_lab(right_policy, cache)
    lab4.run(pools["q1"], "pairwise")
    lab4.run(pools["q2"], "verify")
    check(not fake4.requests and len((TMP / "calls.jsonl").read_text().splitlines()) == n_lines,
          "a rerun reads every call from the lab cache")
    for x in (runner, runner2, runner3, runner4):
        x.close()
    check(digest(cache) == before, "the recorded cache is never written")

    # the metrics
    stats = CL.chooser_stats([(pools["q1"], {"chosen": "C", "calls": 2, "completion": 40}),
                              (pools["q2"], {"chosen": "D", "calls": 4, "completion": 80}),
                              (pools["q4"], {"chosen": "C", "calls": 2, "completion": 40})])
    check(stats["n"] == 3 and stats["cf"] == 1 and stats["wf"] == 1 and stats["overrides"] == 2 and stats["clear"] == 2
          and stats["ties"] == 1 and stats["tie_acc"] == 100.0 and abs(stats["tie_random"] - 100 / 3) < 1e-9
          and abs(stats["acc"] - 200 / 3) < 1e-9 and abs(stats["plurality"] - 100 / 3) < 1e-9
          and stats["precision"] == 0.5 and stats["calls"] == 8 / 3, "chooser metrics", str(stats))
    ps = CL.probe_stats([(pools["q1"], {"wrong": "B", "right_first": "C", "right_second": "B", "calls": 2,
                                        "completion": 40})])
    check(ps["acc_right_first"] == 100 and ps["acc_right_second"] == 0 and ps["split"] == 100
          and ps["picks_response_1"] == 100, "probe metrics: a model that always picks response 1", str(ps))


if __name__ == "__main__":
    test_gap()
    test_settings()
    test_lab()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: " + "; ".join(FAILURES))
        sys.exit(1)
    print(f"all checks passed; temp dir {TMP}")

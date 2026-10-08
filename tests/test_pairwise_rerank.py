"""Offline checks of pairwise_rerank.py: ranking, the arguing speaker, the knockout rules and
their top-2 / top-3 stops, the choice parser, and the call cache (a scripted fake model).

    python tests/test_pairwise_rerank.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))   # the code under test
sys.path.insert(0, str(Path(__file__).resolve().parent))                       # the other test files

import chooser_lab as L  # noqa: E402
import debate_mcq as D  # noqa: E402
import pairwise_rerank as R  # noqa: E402

FAILS = 0


def check(cond: bool, what: str) -> None:
    global FAILS
    print(("ok   " if cond else "FAIL ") + what)
    FAILS += not cond


def say(letter: str, body: str = "") -> str:
    return f"{body}reasoning\n\nANSWER: {letter}"


# --- ranking and the arguing speaker ---
rounds = [[("solver", say("A", "s0 ")), ("solver", say("B", "s1 ")), ("solver", say("C", "s2 "))],
          [("critic", say("B", "c0 "))],
          [("solver", say("C", "s3 ")), ("eliminator", say("A"))]]
specs = [{"personas": ["solver"] * 3, "effort": "high"}, {"personas": ["critic"]}, {"personas": ["solver", "eliminator"], "effort": "high"}]
ms = R.mentions_of(rounds, specs, 4)
check([m["answer"] for m in ms] == ["A", "B", "C", "B", "C"], "mentions skip non-answering personas")
ranked = R.rank_answers(ms)
check([a["answer"] for a in ranked] == ["C", "B", "A"], "rank: mentions, then the later last round (C round 2 > B round 1)")
check(ranked[0]["rep"]["text"].startswith("s3") and ranked[1]["rep"]["text"].startswith("s1"),
      "argued by the latest high-effort speaker (B: the round-0 solver, not the low-effort critic)")
tie = R.rank_answers(R.mentions_of([[("solver", say("D")), ("solver", say("A"))]], [{"personas": ["solver"] * 2}], 4))
check([a["answer"] for a in tie] == ["D", "A"], "full tie: first committed first")

# --- the choice parser ---
check(R.parse_choice("blah\nCHOICE: 2") == 2 and R.parse_choice("**CHOICE:** 1") == 1
      and R.parse_choice("CHOICE: 1 ... CHOICE: 2") == 2 and R.parse_choice("no pick") is None, "CHOICE lines read")
check(R.parse_choice_followup("2") == 2 and R.parse_choice_followup("(1).") == 1
      and R.parse_choice_followup("maybe 3") is None, "follow-up reads a bare 1 or 2")

# --- winner_after ---
m = [["A", "B", "B", "B"], ["B", "C", "B", "C"], ["B", "D", "D", "B"]]
check(R.winner_after(m, 1, "both") == "B" and R.winner_after(m, 2, "both") == "B" and R.winner_after(m, None, "both") == "B",
      "both orders: a challenger must win twice")
m1 = [["A", "B", "B", None], ["B", "C", "C", None], ["C", "D", "C", None]]
check(R.winner_after(m1, 1, "first") == "B" and R.winner_after(m1, 2, "first") == "C"
      and R.winner_after(m1, None, "first") == "C", "one order: stops after 1, 2, all matches")


# --- the knockout through a scripted model and the call cache ---
class FakeCompletions:
    def __init__(self, prefer):
        self.prefer, self.calls = prefer, 0

    def create(self, model, messages, **kw):
        self.calls += 1
        user = messages[-1]["content"]
        r1 = user.split("Response 1 (final answer: ")[1].split(")")[0]
        r2 = user.split("Response 2 (final answer: ")[1].split(")")[0]
        pick = 1 if self.prefer.index(r1) < self.prefer.index(r2) else 2
        msg = SimpleNamespace(content=f"checked both\n\nCHOICE: {pick}", reasoning_content="think")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                               usage=SimpleNamespace(completion_tokens=10, prompt_tokens=100))


tmp = Path(tempfile.mkdtemp())
fake = FakeCompletions(prefer=["B", "C", "A"])          # the model always prefers B, then C, then A
client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
caller = L.Caller(tmp / "calls.jsonl", "http://unused", "openai/gpt-oss-20b", live=False)
caller.live, caller.clients = True, iter(lambda: client, None)
orig_pt = D._prompt_tokens
D._prompt_tokens = lambda c, m, msgs: 100
try:
    row = {"id": "q1", "question": "Q?", "options": ["a", "b", "c", "d"]}
    deb = {"q": "q1", "rep": 0, "ranked": ranked}            # C, B, A
    ko = R.Knockout(caller, "low", "full")
    both = ko.run(deb, row, "both")
    check(both["matches"] == [["C", "B", "B", "B"], ["B", "A", "B", "B"]] and fake.calls == 4,
          "both orders: B beats C twice, then holds against A (4 calls)")
    first = ko.run(deb, row, "first")
    check(fake.calls == 4 and R.winner_after(first["matches"], None, "first") == "B",
          "one order reuses the cached holder-first calls (no new call)")
    caller2 = L.Caller(tmp / "calls.jsonl", "http://unused", "openai/gpt-oss-20b", live=False)
    again = R.Knockout(caller2, "low", "full").run(deb, row, "both")
    check(again is not None and again["matches"] == both["matches"], "a rerun is served from the cache file")
    low = R.pair_prompt(row, ranked[0], ranked[1], "full")
    check(low[1] == "full" and "Response 2 (final answer: B)" in low[0], "pair prompt: full arguments, answers labelled")
    big = {**ranked[0], "rep": {**ranked[0]["rep"], "text": "x" * R.PROMPT_CHARS + "\n\nANSWER: C"}}
    check(R.pair_prompt(row, big, ranked[1], "full")[1] == "summary", "an oversized pair falls back to summaries")
finally:
    D._prompt_tokens = orig_pt

print(f"\n{'all checks passed' if not FAILS else f'{FAILS} FAILED'}")
sys.exit(1 if FAILS else 0)

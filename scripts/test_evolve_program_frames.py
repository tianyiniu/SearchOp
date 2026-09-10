"""Offline checks for evolve_program_frames.py: stubbed search, fetch, model
and judge, so the whole program interpreter runs end to end with no services,
no keys and no cost.

    python3 scripts/test_evolve_program_frames.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_search_step as ES  # noqa: E402
import evolve_program_frames as F  # noqa: E402
from doc_qa import SummarizerConfig  # noqa: E402

FAILURES: list[str] = []


def check(okay: bool, label: str, extra: str = "") -> None:
    print(("  ok   " if okay else "  FAIL ") + label + (f"  [{extra}]" if extra else ""))
    if not okay:
        FAILURES.append(label)


# --- stubs -------------------------------------------------------------------

GOLD = "Roudnice nad Labem"


class StubClient:
    """Answers by recognizing which prompt the pipeline sent."""

    def __init__(self):
        self.chat = type("Chat", (), {"completions": self})()
        self.n_checks = 0

    def create(self, **kw):
        system = kw["messages"][0]["content"]
        user = kw["messages"][1]["content"]
        if system.startswith("Break the question"):
            content = "who edited the winning film\nwhere was the editor born"
        elif system.startswith("Write "):
            content = "editor of Bored in Brno\nBrozek film editor"
        elif system.startswith("You are choosing") or system.startswith("You are following"):
            content = "d1\nd2"
        elif system.startswith("Answer the sub-question"):
            content = "Jiri Brozek" if "edited" in user else GOLD
        elif system.startswith("Rewrite the sub-question"):
            content = "Jiri Brozek birthplace"
        elif system.startswith("You are auditing"):
            self.n_checks += 1
            content = "MISSING: Jiri Brozek birth city" if self.n_checks == 1 else "SUPPORTED"
        else:  # the answerer (doc_qa.WITH_DOC_SYSTEM)
            content = f"<think>weighing the documents</think>\nThe editor was born in {GOLD}."
        msg = type("M", (), {"content": content})()
        choice = type("C", (), {"message": msg})()
        return type("R", (), {"choices": [choice]})()


def stub_search(query: str) -> str:
    tag = abs(hash(query)) % 3 + 1
    return (f"[1] Page {tag}\n    URL: https://en.wikipedia.org/wiki/Page_{tag}\n"
            f"    a snippet about {query[:40]}\n"
            f"[2] Editor page\n    URL: https://en.wikipedia.org/wiki/Jiri_Brozek\n"
            f"    Jiri Brozek, film editor, born in {GOLD}")


class StubFetch:
    def fetch_raw(self, url: str) -> dict:
        return {"url": url, "text": f"Full text of {url}. Born in {GOLD} in 1947.",
                "markup": "", "error": None}


class DummyCache:
    hits = misses = 0

    def get_or(self, kind, payload, fn):
        return fn()


class StubRuntime(F.Runtime):
    def __init__(self):
        self.cache = DummyCache()
        self.qwen = StubClient()
        self.model = "stub"
        self.search = stub_search
        self.fetch = StubFetch()
        self.links = lambda doc, limit=40: [("linked page",
                                             "https://en.wikipedia.org/wiki/Linked")]
        self.cfg = SummarizerConfig(model="stub")
        self.judge_calls = 0

    def judged(self, question, gt, answer):
        self.judge_calls += 1
        return GOLD.lower() in (answer or "").lower()


ROW = {"id": "t1",
       "question": "Where was the editor of the 2003 winner born?",
       "ground_truth": GOLD,
       "wiki_links": ["https://en.wikipedia.org/wiki/Jiri_Brozek"]}


# --- tests -------------------------------------------------------------------

def test_v0_end_to_end():
    print("v0 runs end to end within its budgets")
    rt = StubRuntime()
    F.validate_program(F.PROGRAM_V0)
    st = F.run_program(F.PROGRAM_V0, rt, ROW)
    real = [a for a in st.actions if not a.startswith("noop")]
    check(real[0] == "decompose", "decompose first", str(real[:3]))
    check("search" in real, "searched")
    check(st.n_searches <= F.MAX_SEARCHES, "search cap respected", str(st.n_searches))
    check(st.n_fetches <= F.MAX_FETCHES, "fetch cap respected", str(st.n_fetches))
    check(len(st.store.docs) > 0, "store holds documents", str(len(st.store.docs)))
    check(st.n_checks >= 1, "checker ran", str(st.n_checks))
    check(st.verdict == "supported", "second check satisfied", str(st.verdict))
    check(len(st.actions) <= F.MAX_ACTS, "action cap respected", str(len(st.actions)))


def test_budget_and_noop():
    print("degenerate programs terminate instead of looping")
    rt = StubRuntime()
    greedy = {"rules": [{"when": [], "do": "search"}], "default": "stop"}
    st = F.run_program(greedy, rt, ROW)
    check(st.n_searches <= F.MAX_SEARCHES, "greedy search hits the cap and stops",
          str(st.n_searches))
    hopeless = {"rules": [{"when": [], "do": "read"}], "default": "stop"}
    st2 = F.run_program(hopeless, rt, ROW)
    check(all(a.startswith("noop") for a in st2.actions), "read with no docs is a noop")
    check(len(st2.actions) <= 3, "three noops end the run", str(len(st2.actions)))


def test_hop_rewrites_query():
    print("hop turns a found fact into the next standalone query")
    rt = StubRuntime()
    prog = {"rules": [{"when": ["acts==0"], "do": "decompose"},
                      {"when": ["acts==1"], "do": "search"},
                      {"when": ["acts==2"], "do": "hop"},
                      {"when": ["acts==3"], "do": "hop"}],
            "default": "stop"}
    st = F.run_program(prog, rt, ROW)
    check(len(st.hop_answers) >= 1, "a sub-question got answered", str(st.hop_answers))
    check(any("birthplace" in q for q in st.queue),
          "rewritten query joined the queue", str(st.queue[:3]))


def test_determinism():
    print("same program, same question, same path")
    a = F.run_program(F.PROGRAM_V0, StubRuntime(), ROW)
    b = F.run_program(F.PROGRAM_V0, StubRuntime(), ROW)
    check(a.actions == b.actions, "action sequences identical")
    check(sorted(a.store.docs) == sorted(b.store.docs), "same documents")


def test_mutations_stay_runnable():
    print("random mutants validate and run without crashing")
    rng = random.Random(7)
    prog = F.PROGRAM_V0
    ran = 0
    for i in range(200):
        prog = F.mutate_program(prog, rng, rand_cond=F.frames_condition,
                                rand_act=F.frames_action, defaults=("stop",))
        try:
            F.validate_program(prog)
        except AssertionError:
            check(False, f"mutant {i} failed validation")
            return
        if i % 10 == 0:
            F.run_program(prog, StubRuntime(), ROW)
            ran += 1
    check(True, f"200 mutants valid, {ran} executed end to end")


def test_light_and_full_scoring():
    print("scoring paths")
    rt = StubRuntime()
    check(F.light_score(F.PROGRAM_V0, rt, ROW) is True, "light score judges the answer")
    ES.judge_answer = lambda q, gt, a: GOLD.lower() in (a or "").lower()
    recs = F.full_record(F.PROGRAM_V0, rt, ROW, "program_test")
    r = recs[0]
    check(r.get("status") == "ok", "full record ok", str(r.get("error", "")))
    check(r.get("downstream_correct") is True, "full record graded correct")
    check(r.get("recall_surfaced") == 1.0, "gold URL counted as surfaced",
          str(r.get("recall_surfaced")))
    check("answer_in_evidence" in r and "actions" in r, "record carries the new fields")


def test_split_is_stable():
    print("train/test split is a function of the seed only")
    rows = [{"id": f"q{i}"} for i in range(20)]
    a = F.split_qids(rows, 0)
    b = F.split_qids(list(reversed(rows)), 0)
    check(a == b, "row order does not change the split")
    check(not set(a[0]) & set(a[1]), "halves are disjoint")


if __name__ == "__main__":
    test_v0_end_to_end()
    test_budget_and_noop()
    test_hop_rewrites_query()
    test_determinism()
    test_mutations_stay_runnable()
    test_light_and_full_scoring()
    test_split_is_stable()
    print("\n" + ("FAILED: " + ", ".join(FAILURES) if FAILURES else "all checks passed"))
    sys.exit(1 if FAILURES else 0)

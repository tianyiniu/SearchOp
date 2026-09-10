"""Tests for the eval_search_step changes. No network, no model, no cost.

Two jobs:
  * prove the pieces the EXISTING configs run through still behave exactly as
    they did (select_evidence, pack_truncate, the legacy decompose parsing);
  * check the new pieces (pack_mixed, query variants, link ranking, document
    selection) on their own.

    python3 scripts/test_eval_plumbing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import eval_search_step as E
from doc_qa import SummarizerConfig
from evidence import Doc

FAILURES: list[str] = []


def check(cond, label: str, detail: str = "") -> None:
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}  {detail}")
        FAILURES.append(label)


class FakeClient:
    """Stands in for the OpenAI client; returns canned replies in order."""

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


# --- 1. what the existing configs run through ------------------------------

def old_select_evidence(policy, snippets, pages):
    """Verbatim copy of select_evidence before it was split into two lists."""
    if policy == "union":
        return snippets + pages
    if policy == "snippets_only":
        return snippets
    return pages or snippets


def test_select_evidence_unchanged() -> None:
    print("\ntest_select_evidence_unchanged")
    cases = [
        ([("search: a", "A"), ("search: b", "B")], [("u1", "P1"), ("u2", "P2")]),
        ([("search: a", "A")], []),
        ([], [("u1", "P1")]),
        ([], []),
    ]
    ok = True
    for snippets, pages in cases:
        for policy in ("union", "snippets_only", "pages_else_snippets"):
            snips, pgs = E.select_evidence(policy, snippets, pages)
            if snips + pgs != old_select_evidence(policy, snippets, pages):
                ok = False
                print(f"    mismatch policy={policy} snippets={snippets} pages={pages}")
    check(ok, "same evidence list for every policy and input")


def test_pack_truncate_unchanged() -> None:
    print("\ntest_pack_truncate_unchanged")
    cfg = SummarizerConfig()
    docs = [("search: a", "A" * 20000), ("u1", "P" * 60000)]
    out = E.pack_truncate(docs, cfg)
    budget = int(cfg.context_window * cfg.doc_budget_fraction) * E.CHARS_PER_TOKEN
    per = max(500, budget // 2)
    check(out.startswith("[Document 1] search: a\n"), "block format unchanged")
    check("[Document 2] u1\n" in out, "second block present")
    check(out.count("A") == min(20000, per), "even split applied to doc 1")
    check(E.pack_truncate([], cfg) == "", "empty -> empty")


def test_legacy_decompose_parsing_unchanged() -> None:
    """The <think> bug is real and stays put unless the flag is set, because
    every number recorded so far was produced with it."""
    print("\ntest_legacy_decompose_parsing_unchanged")
    reply = ("<think>\nOkay, let's tackle this.\nFirst, I need to identify the artist.\n"
             "</think>\nWho released Father of Asahd?\nWhich high school did he attend?\n"
             "How many Olympic teams did the diver join?")
    sconf = E.PRESETS["decompose_react_union"]
    subs = E._decompose(sconf, FakeClient([reply]), "m", "Q")
    check(subs[0] == "<think>", "legacy still takes '<think>' as sub-question 1", subs[0])
    check(len(subs) == sconf.max_subqueries, "legacy still keeps max_subqueries lines", str(len(subs)))

    fixed = E._decompose(E.PRESETS["decompose_react_fixq"], FakeClient([reply]), "m", "Q")
    check(fixed == ["Who released Father of Asahd?",
                    "Which high school did he attend?",
                    "How many Olympic teams did the diver join?"],
          "fixed version returns the real sub-questions", str(fixed))

    # an unclosed <think> yields nothing -> fall back to the question itself
    only = E._decompose(E.PRESETS["decompose_react_fixq"],
                        FakeClient(["<think>\nran out of tokens"]), "m", "THE QUESTION")
    check(only == ["THE QUESTION"], "unclosed think falls back to the question", str(only))


# --- 2. the new pieces -----------------------------------------------------

def test_pack_mixed() -> None:
    print("\ntest_pack_mixed")
    cfg = SummarizerConfig()
    budget = int(cfg.context_window * cfg.doc_budget_fraction) * E.CHARS_PER_TOKEN

    # the case pack_truncate gets wrong: many snippets, a few pages
    snippets = [(f"search: q{i}", "S" * 1200) for i in range(24)]
    pages = [(f"u{i}", "P" * 60000) for i in range(4)]
    mixed = E.pack_mixed(snippets, pages, cfg)
    even = E.pack_truncate(snippets + pages, cfg)
    per_even = max(500, budget // 28)
    check(mixed.count("S") == 24 * 1200, "every snippet kept whole", str(mixed.count("S")))
    check(mixed.count("P") > even.count("P"),
          "pages get more room than under the even split",
          f"mixed={mixed.count('P')} even={even.count('P')}")
    check(even.count("P") == 4 * per_even, "even split really was the smaller share",
          f"{even.count('P')} vs {4 * per_even}")
    check(len(mixed) <= budget * 1.1, "total stays near the budget", str(len(mixed)))

    # worst case: 24 maximum-size snippet blocks must not crowd pages out
    fat = [(f"search: q{i}", "S" * 4000) for i in range(24)]
    out = E.pack_mixed(fat, pages, cfg)
    check(out.count("P") >= 4 * 500, "pages still get their floor", str(out.count("P")))
    check(len(out) <= budget * 1.1, "worst case stays near the budget", str(len(out)))

    check(E.pack_mixed([], [], cfg) == "", "empty -> empty")
    check(E.pack_mixed([], pages, cfg).startswith("[Document 1] u0"), "pages-only numbering")
    nums = [ln for ln in E.pack_mixed(snippets, pages, cfg).splitlines() if ln.startswith("[Document")]
    check([n.split("]")[0] for n in nums] == [f"[Document {i}" for i in range(1, 29)],
          "document numbering is continuous across snippets and pages")


def test_query_variants_and_dedup() -> None:
    print("\ntest_query_variants_and_dedup")
    sconf = E.PRESETS["breadth"]
    reply = "<think>\nplanning\n</think>\nMinardi M194 driver\nWho drove the Minardi M194 in 1994?\n- Minardi M194"
    got = E._query_variants(sconf, FakeClient([reply]), "m", "which car in 1994")
    check(got[0] == "which car in 1994", "sub-question itself always kept", got[0])
    check(len(got) == sconf.queries_per_subq, "capped at queries_per_subq", str(len(got)))
    check("<think>" not in " ".join(got), "reasoning trace not used as a query", str(got))

    check(E._query_variants(E.PRESETS["decompose"], FakeClient([]), "m", "x") == ["x"],
          "queries_per_subq<=1 means no extra model call")

    class Boom(FakeClient):
        def create(self, **kw):
            raise RuntimeError("model down")

    check(E._query_variants(sconf, Boom([]), "m", "x") == ["x"],
          "a model failure degrades to the sub-question alone")

    check(E._dedup(["A b", "a  B", "c", "C", "", "  "]) == ["A b", "c"],
          "dedup is case- and whitespace-insensitive", str(E._dedup(["A b", "a  B", "c", "C"])))


def test_shortlist_links() -> None:
    """Regression test for the ranking bug.

    The shortlist used to be ordered by overlap between the anchor text and the
    question's words, which is backwards for multi-hop: the page you need is the
    one the question does NOT name. Reconstructed here from the real case --
    frames_41 asks which Formula One car the nephew drove, the answer is Minardi
    M194 at document position 50, and its anchor shares no word with the
    question while every "Formula One" / "Ferrari" / "racing driver" link does.
    Under the old scoring it was dropped before the model saw it.
    """
    print("\ntest_shortlist_links")
    # anchors that DO share words with the question, filling the early positions
    links = [(f"Formula One racing driver {i}", f"https://x.org/u{i}") for i in range(50)]
    # ...and the one that shares nothing, sitting at position 50
    links.append(("Minardi M194", "https://en.wikipedia.org/wiki/Minardi_M194"))
    links += [(f"Ferrari Italy {i}", f"https://x.org/v{i}") for i in range(200)]

    top = E._shortlist_links(links, 60)
    check(len(top) == 60, "shortlist size respected", str(len(top)))
    check([u for _, u in top] == [u for _, u in links[:60]],
          "document order preserved exactly")
    check(any(u.endswith("Minardi_M194") for _, u in top),
          "a link sharing NO words with the question survives the shortlist",
          str([a for a, _ in top[:3]]))

    check(E._shortlist_links(links, 10)[0] == links[0], "takes from the front")
    check(len(E._shortlist_links(links, 10)) == 10, "smaller limit respected")
    check(E._shortlist_links(links, 10000) == links, "limit above length returns all")
    check(E._shortlist_links([], 10) == [], "no links -> empty")


def test_select_docs() -> None:
    print("\ntest_select_docs")
    docs = [Doc(doc_id=f"d{i}", url=f"https://x.org/{i}", key=f"x.org/{i}", title=f"T{i}",
                snippets=[("q", f"snippet {i}")]) for i in range(1, 11)]
    sconf = E.PRESETS["breadth_read"]

    picked = E._select_docs(sconf, FakeClient(["<think>\nhmm\n</think>\nd3\nd7\nd3"]),
                            "m", "Q", docs, 3)
    check([d.doc_id for d in picked] == ["d3", "d7"], "ids parsed, duplicates dropped",
          str([d.doc_id for d in picked]))

    few = docs[:2]
    check(E._select_docs(sconf, FakeClient([]), "m", "Q", few, 6) == few,
          "fewer candidates than k -> no model call, take them all")
    check(E._select_docs(sconf, FakeClient([]), "m", "Q", [], 6) == [],
          "no candidates -> nothing")

    prose = E._select_docs(sconf, FakeClient(["I think the first few look best."]),
                           "m", "Q", docs, 3)
    check(len(prose) == 3, "prose reply falls back to the front of the list", str(len(prose)))
    check(E._select_docs(sconf, FakeClient(["d99\nd404"]), "m", "Q", docs, 3)[0].doc_id == "d1",
          "unknown ids fall back rather than selecting nothing")


def test_decompose_prompts() -> None:
    print("\ntest_decompose_prompts")
    reply = ("<think>\nOkay, let's tackle this.\n</think>\n"
             "Who released the album Father of Asahd?\nWhich high school did he attend?")

    plain = FakeClient([reply])
    E._decompose(E.PRESETS["decompose_snip_read"], plain, "m", "Q")
    check("stand alone" not in plain.prompts[0][0]["content"],
          "plain preset uses the original decompose prompt")

    sc = FakeClient([reply])
    E._decompose(E.PRESETS["decompose_snip_read_sc"], sc, "m", "Q")
    check("stand alone" in sc.prompts[0][0]["content"],
          "self-contained preset uses the self-contained prompt")
    check("he" in sc.prompts[0][0]["content"] and "that city" in sc.prompts[0][0]["content"],
          "self-contained prompt names the pronoun failure it is preventing")

    # the accidental context-preserving behaviour must be what the first two get
    for n in ("decompose_snip", "decompose_snip_read"):
        subs = E._decompose(E.PRESETS[n], FakeClient([reply]), "m", "Q")
        check(subs[0] == "<think>", f"{n}: keeps the trace-line decomposition", subs[0])
    subs = E._decompose(E.PRESETS["decompose_snip_read_sc"], FakeClient([reply]), "m", "Q")
    check(subs[0] == "Who released the album Father of Asahd?",
          "sc: reads real sub-questions", subs[0])


def test_snippets_only_stage() -> None:
    """The whole point of the pipeline is that stage 2 has no fetch tool. If
    fetch_k=0 did not actually remove it, the plan would silently be the old one."""
    print("\ntest_snippets_only_stage")
    for n in ("decompose_snip", "decompose_snip_read", "decompose_snip_read_sc"):
        check(E.PRESETS[n].fetch_k == 0, f"{n}: fetch_k is 0 (snippets-only stage)",
              str(E.PRESETS[n].fetch_k))
        check(E.PRESETS[n].query_plan == "decompose_react_read", f"{n}: uses the new plan")
    check(E.PRESETS["decompose_snip"].select_k == 0,
          "decompose_snip stops before reading anything")
    check(E.PRESETS["decompose_snip_read"].select_k > 0,
          "decompose_snip_read goes on to read")
    # and the fetch_k guard added to run_decompose_react must not touch the
    # presets that were already measured on it
    for n in ("decompose_react", "decompose_react_union", "decompose_react_fixq"):
        check(E.PRESETS[n].fetch_k > 0, f"{n}: still gets the fetch tool",
              str(E.PRESETS[n].fetch_k))


def test_parallel_preserves_order() -> None:
    """The point of _parallel is that results come back in INPUT order. If they
    came back in completion order, the evidence blocks would be ordered by which
    network call returned first and a rerun would not reproduce."""
    print("\ntest_parallel_preserves_order")
    import random as _r
    import time as _t

    def slow(x):
        # deliberately inverted: later items finish first
        _t.sleep((20 - x) * 0.003)
        return x * 10

    items = list(range(20))
    expected = [x * 10 for x in items]
    for w in (1, 2, 8, 64):
        check(E._parallel(slow, items, w) == expected, f"order preserved at workers={w}")
    check(E._parallel(slow, [], 8) == [], "empty input")
    check(E._parallel(slow, [3], 8) == [30], "single item")

    calls = {"n": 0}

    def counted(x):
        calls["n"] += 1
        return x

    E._parallel(counted, list(range(7)), 4)
    check(calls["n"] == 7, "every item processed exactly once", str(calls["n"]))

    def boom(x):
        if x == 3:
            raise ValueError("boom")
        return x

    try:
        E._parallel(boom, list(range(6)), 4)
        check(False, "exceptions propagate")
    except ValueError:
        check(True, "exceptions propagate")


def test_inner_workers_wiring() -> None:
    print("\ntest_inner_workers_wiring")
    for name, sconf in E.PRESETS.items():
        check(sconf.inner_workers >= 1, f"{name}: inner_workers sane", str(sconf.inner_workers))
    from dataclasses import replace as _replace
    s = _replace(E.PRESETS["decompose_snip"], inner_workers=1)
    check(s.inner_workers == 1 and E.PRESETS["decompose_snip"].inner_workers == 4,
          "--inner-workers override does not mutate the preset")


def test_presets_consistent() -> None:
    print("\ntest_presets_consistent")
    for name, sconf in E.PRESETS.items():
        check(sconf.name == name, f"{name}: name matches key", sconf.name)
        check(sconf.query_plan in E.PLANS, f"{name}: plan exists", sconf.query_plan)
        check(sconf.evidence in ("union", "snippets_only", "pages_else_snippets"),
              f"{name}: evidence policy valid", sconf.evidence)
        check(sconf.packing in ("even", "mixed"), f"{name}: packing valid", sconf.packing)
        check(sconf.link_source in E.LINK_SOURCES, f"{name}: link source valid", sconf.link_source)
    old = ["react_baseline", "single_shot", "decompose", "iterative_snippets",
           "decompose_react", "react_union", "decompose_react_union"]
    check(all(E.PRESETS[n].packing == "even" for n in old),
          "every pre-existing preset still packs the old way")
    check(all(not E.PRESETS[n].strip_think_subqueries for n in old),
          "every pre-existing preset keeps the old decompose parsing")
    check(all(not E.PRESETS[n].follow_links for n in old),
          "no pre-existing preset follows links")


if __name__ == "__main__":
    test_select_evidence_unchanged()
    test_pack_truncate_unchanged()
    test_legacy_decompose_parsing_unchanged()
    test_pack_mixed()
    test_query_variants_and_dedup()
    test_shortlist_links()
    test_select_docs()
    test_decompose_prompts()
    test_snippets_only_stage()
    test_parallel_preserves_order()
    test_inner_workers_wiring()
    test_presets_consistent()
    print("\n" + ("FAILED: " + ", ".join(FAILURES) if FAILURES else "all checks passed"))
    sys.exit(1 if FAILURES else 0)

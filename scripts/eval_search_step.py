"""(2) Search-step evaluation: which search configs gather the most useful evidence?

Every question in frames_qwen_answerable.json can be answered by Qwen IF it has the
right documents (Steps 1-2). So search quality can be measured cleanly here, two ways:

  - retrieval overlap : do the URLs the agent surfaces / fetches include the gold
                        wiki_links? (recall/precision after URL normalization)
  - downstream answer : feed the agent's RETRIEVED documents into the same simple
                        Qwen call used to build the splits; since these questions are
                        qwen_answerable, a correct answer means search found enough.
                        This survives gold redundancy (equivalent non-gold pages).

Search is configured, not debated (a SearchConfig: query plan + budgets). fetch_url
hits LIVE Serper (open-web URLs miss the local cache), so this costs money — debug
with --limit first. Correctness is graded by judge_answer (gpt-5.4-mini).

    export OPENAI_API_KEY=...  SERPER_API_KEY=...
    # vLLM serving Qwen/Qwen3-14B on port 7472
    python3 scripts/eval_search_step.py --limit 10        # debug
    python3 scripts/eval_search_step.py                   # full
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from doc_qa import SummarizerConfig, answer_with_docs, gold_urls, normalize_url
from evidence import EvidenceStore, extract_links
from iterative_summarization import CHARS_PER_TOKEN
from llm_judge import judge_answer
from store_tools import DocumentLinks, StoreFetch, StoreLinks, StoreSearch, WikiApiLinks
from tools import SERPER_SCRAPE_URL, FetchUrl, SearchInfo
from tool_calling import run_with_tools

try:  # load API keys from the project .env if python-dotenv is available
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

SEARCH_SYSTEM = (
    "You are a research assistant gathering evidence to answer a question. Use "
    "search_info to find relevant sources and fetch_url to read the most promising "
    "ones. Search thoroughly and from multiple angles; gather the facts needed to "
    "answer. You do not need to produce a final answer."
)
DECOMPOSE_SYSTEM = (
    "Break the question into up to {k} focused web-search queries that together "
    "gather everything needed to answer it. Output one query per line, no numbering "
    "or extra text."
)

# Why this exists: reading sub-questions off the reply WITHOUT removing the
# reasoning trace turned out to score 6.4 points higher than reading them with it
# removed, and to find 5 points more gold. The trace lines are not good queries,
# but they restate the whole question with its context ("the artist who released
# Father of Asahd and went to the same high school as an Olympic diver"), whereas
# a clean sub-question says "Which high school did he attend?" and is run as a
# standalone search where "he" refers to nothing. The accident was preserving
# context. This prompt asks for that on purpose.
DECOMPOSE_SELFCONTAINED_SYSTEM = (
    "Break the question into up to {k} web-search queries that together gather "
    "everything needed to answer it. Each query is run as a SEPARATE search that "
    "cannot see the original question or the other queries, so each one must "
    "stand alone: name the people, places, works and dates explicitly instead of "
    "writing 'he', 'that city' or 'the album'. Where a later step depends on a "
    "fact you do not know yet, describe what identifies it rather than referring "
    "back. Output one query per line, no numbering or extra text."
)

# The agent in the snippets-only stage has no fetch tool, so telling it to read
# pages (as SEARCH_SYSTEM does) would just waste turns on a tool that is not
# there. The second sentence is the point of the stage: keep searching for what
# the last set of snippets pointed at.
SEARCH_SNIPPETS_SYSTEM = (
    "You are gathering evidence to answer a question using web search only. You "
    "cannot open pages — work from the result snippets. When a snippet tells you "
    "something you did not know, search again for the next thing it points to; "
    "that is how you get from the question to the facts it depends on. Search "
    "thoroughly and from several angles. You do not need to produce a final answer."
)


@dataclass
class SearchConfig:
    name: str
    query_plan: str          # "iterative" | "single" | "decompose" | "decompose_react"
    top_n: int = 5           # search_info results per query
    fetch_k: int = 3         # URLs fetched per query (0 = snippets only)
    budget: int = 20         # max tool calls (iterative / per-subquery decompose_react)
    max_subqueries: int = 4  # decompose / decompose_react plans
    # Strip Qwen's <think> block before reading sub-questions off the reply.
    # Without it the "sub-questions" are the first lines of the reasoning trace
    # — literally '<think>', "Okay, let's tackle this question step by step...",
    # "First, I need to identify..." — so the decompose plans have never actually
    # decomposed anything. Defaults to False purely so every number recorded
    # before this was found stays reproducible; every new preset sets it True.
    strip_think_subqueries: bool = False
    # Which decomposition prompt to use: "plain" (the original) or
    # "self_contained" (each sub-question names its own entities). Only has any
    # effect when strip_think_subqueries is True — with the trace left in, the
    # sub-questions are trace lines and the prompt never gets read off.
    decompose_prompt: str = "plain"
    # How the retrieved material becomes the answerer's evidence:
    #   "pages_else_snippets" - fetched pages if there are any, else snippets
    #   "union"               - snippets AND fetched pages together
    # The first is the original behaviour and is kept so the published numbers
    # stay reproducible; it silently discards every snippet as soon as one page is
    # fetched, which is most runs.
    evidence: str = "pages_else_snippets"
    # Cap on a page handed back to an agent mid-loop (None = uncapped, original
    # behaviour). Uncapped fetches are what overflow Qwen's 32k context and abort
    # the run — 18/266 decompose_react and 8/266 react_baseline questions.
    fetch_chars: int | None = None
    answer_max_tokens: int = 2048
    # How the answerer's character budget is divided.
    #   "even"  - budget/n_items each (pack_truncate, the original)
    #   "mixed" - small items whole, pages share what is left (pack_mixed)
    # Only matters once there are many items; see pack_mixed.
    packing: str = "even"

    # --- forced breadth (the "breadth" plans) -----------------------------
    # The measured agent spends 5.47 of its 24 allowed tool calls and fills
    # 13.6% of the answerer's budget, so neither limit is what stops it — it
    # decides it is done. These plans take the decision away and issue a fixed
    # number of searches.
    queries_per_subq: int = 3      # phrasings generated per sub-question
    max_searches: int = 24         # hard cap on searches per question
    # --- selective reading ------------------------------------------------
    select_k: int = 6              # documents the model picks to actually read
    digest: bool = False           # summarize a page for the agent, store it whole
    digest_words: int = 180
    # --- link following ---------------------------------------------------
    follow_links: bool = False
    link_rounds: int = 1           # how many times to expand the frontier
    links_per_page: int = 40       # shortlist size shown to the model per page
    link_select_k: int = 4         # links actually followed per round
    # Which backend answers "what does this document point at". "document" is
    # the corpus-agnostic one (links come out of the markup of the page we
    # fetched); "wiki_api" is a free stand-in for measuring on this dataset
    # without paying to re-scrape. --link-source overrides it for a whole run.
    link_source: str = "document"
    # Execution knob, not part of the method: how many things a SINGLE question
    # may do at once (its sub-question agents, its page fetches, its per-policy
    # scoring). --workers parallelises across questions; this parallelises inside
    # one, which is what shortens a single question's wall clock. Results do not
    # depend on it -- everything parallel here is independent, and the writes are
    # replayed in a fixed order. Overridden for a whole run by --inner-workers.
    inner_workers: int = 4


PRESETS = {
    "react_baseline": SearchConfig("react_baseline", "iterative", top_n=5, budget=20),
    "single_shot": SearchConfig("single_shot", "single", top_n=5, fetch_k=3),
    "decompose": SearchConfig("decompose", "decompose", top_n=5, fetch_k=2, max_subqueries=4),
    "iterative_snippets": SearchConfig("iterative_snippets", "iterative", top_n=5, fetch_k=0, budget=10),
    # Decompose into <=3 sub-questions, run an independent ReAct agent on each, then
    # let the downstream answerer aggregate the pooled evidence.
    "decompose_react": SearchConfig("decompose_react", "decompose_react",
                                    top_n=5, budget=8, max_subqueries=3),
    # Same plans, with the harness losses removed: snippets are kept alongside
    # fetched pages, a fetched page cannot blow the context, and the answerer has
    # room to finish its reasoning. Paired with the originals above, the delta is
    # the cost of the harness rather than of the search plan.
    "react_union": SearchConfig("react_union", "iterative", top_n=5, budget=20,
                                evidence="union", fetch_chars=8000,
                                answer_max_tokens=4096),
    "decompose_react_union": SearchConfig("decompose_react_union", "decompose_react",
                                          top_n=5, budget=8, max_subqueries=3,
                                          evidence="union", fetch_chars=8000,
                                          answer_max_tokens=4096),

    # Identical to decompose_react_union except that the sub-questions are read
    # from the reply with the reasoning trace removed. Paired against it, the
    # difference is the cost of having been decomposing '<think>' and two lines
    # of deliberation instead of the question, and nothing else.
    "decompose_react_fixq": SearchConfig("decompose_react_fixq", "decompose_react",
                                         top_n=5, budget=8, max_subqueries=3,
                                         evidence="union", fetch_chars=8000,
                                         answer_max_tokens=4096,
                                         strip_think_subqueries=True),

    # Decompose -> a snippets-only agent per sub-question -> choose what to read
    # -> read it. Same budget and sub-question count as decompose_react_union, so
    # against it the difference is the pipeline shape and nothing else.
    #
    #   _snip    stops after the snippets stage; isolates "does removing the
    #            fetch tool make the agent search more and find more?"
    #   (plain)  adds the choose-and-read stage
    #   _sc      swaps in the self-contained decomposition prompt
    #
    # strip_think_subqueries stays False on the first two on purpose: leaving the
    # reasoning trace in scored 6.4 points HIGHER than removing it, because the
    # trace restates the question's context. _sc is the principled version of
    # that accident and is the only one that turns it on.
    "decompose_snip": SearchConfig("decompose_snip", "decompose_react_read",
                                   top_n=5, budget=8, max_subqueries=3, fetch_k=0,
                                   select_k=0, evidence="snippets_only",
                                   packing="mixed", answer_max_tokens=4096),
    "decompose_snip_read": SearchConfig("decompose_snip_read", "decompose_react_read",
                                        top_n=5, budget=8, max_subqueries=3, fetch_k=0,
                                        select_k=6, fetch_chars=8000,
                                        evidence="union", packing="mixed",
                                        answer_max_tokens=4096),
    "decompose_snip_read_sc": SearchConfig("decompose_snip_read_sc", "decompose_react_read",
                                           top_n=5, budget=8, max_subqueries=3, fetch_k=0,
                                           select_k=6, fetch_chars=8000,
                                           evidence="union", packing="mixed",
                                           answer_max_tokens=4096,
                                           strip_think_subqueries=True,
                                           decompose_prompt="self_contained"),

    # --- the new plans ----------------------------------------------------
    # Deliberately nested, so the three ideas can be told apart. Each preset
    # adds exactly one thing to the one above it; a difference between two
    # adjacent rows is attributable, which is the whole point of running them
    # together off the same dataset.
    #
    #   breadth              search a lot more, snippets only         (idea 2)
    #   breadth_read         + choose pages and read them             (+ idea 3)
    #   breadth_link_raw     + follow the links out of what was read  (+ idea 1)
    #   breadth_link         + choose those links from summaries      (+ idea 4)
    #
    # The digest (idea 4) is deliberately only switched on in the last one. It
    # exists to let a reader work from a page without carrying the whole page,
    # and the only place this pipeline needs that is deciding which link fills a
    # gap: the six pages already read run to ~150k characters and do not fit in
    # a prompt, while their summaries do. Turning it on in breadth_read would
    # buy a summarizer call per fetch that nothing ever reads.
    "breadth": SearchConfig("breadth", "breadth", top_n=5, fetch_k=0,
                            max_subqueries=4, queries_per_subq=3, max_searches=24,
                            evidence="snippets_only", packing="mixed",
                            answer_max_tokens=4096, strip_think_subqueries=True),
    "breadth_read": SearchConfig("breadth_read", "breadth_read", top_n=5,
                                 max_subqueries=4, queries_per_subq=3, max_searches=24,
                                 select_k=6, digest=False, fetch_chars=8000,
                                 evidence="union", packing="mixed",
                                 answer_max_tokens=4096, strip_think_subqueries=True),
    "breadth_link_raw": SearchConfig("breadth_link_raw", "breadth_read", top_n=5,
                                     max_subqueries=4, queries_per_subq=3, max_searches=24,
                                     select_k=6, digest=False, fetch_chars=8000,
                                     follow_links=True, link_rounds=1, links_per_page=40,
                                     link_select_k=4,
                                     evidence="union", packing="mixed",
                                     answer_max_tokens=4096, strip_think_subqueries=True),
    "breadth_link": SearchConfig("breadth_link", "breadth_read", top_n=5,
                                 max_subqueries=4, queries_per_subq=3, max_searches=24,
                                 select_k=6, digest=True, fetch_chars=8000,
                                 follow_links=True, link_rounds=1, links_per_page=40,
                                 link_select_k=4,
                                 evidence="union", packing="mixed",
                                 answer_max_tokens=4096, strip_think_subqueries=True),
}

_URL_RE = re.compile(r"URL:\s*(\S+)")

# Which backend answers "what does this document point at". The method itself
# does not care which is plugged in; see store_tools for why the wiki one exists.
LINK_SOURCES = {"document": DocumentLinks, "wiki_api": WikiApiLinks}


# --- small helpers ---------------------------------------------------------

def chat(client, model, system, user, temperature=0.3, max_tokens=1024) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=temperature, max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def parse_search_urls(results_text: str) -> list[str]:
    """URLs from a search_info result block, in order."""
    return _URL_RE.findall(results_text)


def _is_good_page(text: str) -> bool:
    """A real fetched page — not a tool-error string ('[fetch_url ...]',
    '[tool fetch_url failed: ...]') or an empty/no-content result. Real page text
    never starts with '['."""
    t = (text or "").strip()
    return bool(t) and not t.startswith("[") and not t.startswith("No readable content")


# --- de-confounding the downstream metric ----------------------------------

def _norm_text(s: str) -> str:
    """Lowercase, punctuation -> space, collapse whitespace (for substring match)."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (s or "").lower())).strip()


def answer_in_text(gold: str, text: str) -> bool:
    """Backend-free retrieval signal: does the (normalized) gold answer appear in
    the (normalized) text? Coarse — misses paraphrase (e.g. '1969' vs spelled out),
    but tells retrieval-miss apart from backend/refusal failure with no LLM."""
    g = _norm_text(gold)
    return bool(g) and g in _norm_text(text)


_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(answer: str) -> str:
    """The answer the model actually committed to, with its reasoning removed.

    Qwen3 emits <think>...</think> before answering, and every cached answer in
    this project contains one. Grading (or refusal-checking) the raw string reads
    private deliberation as if it were the answer: a trace that muses 'the
    documents do not mention...' scores as a refusal even when the final line is
    right, and a trace that happens to state the right value scores as correct
    even when the committed answer is wrong. An UNCLOSED <think> means generation
    ran out of tokens mid-thought, so there is no answer at all -> ''.
    """
    a = answer or ""
    if _THINK_OPEN.search(a) and "</think>" not in a.lower():
        return ""
    return _THINK_BLOCK.sub("", a).strip()


def truncated_mid_think(answer: str) -> bool:
    """Generation died inside the reasoning trace, so no answer was ever emitted."""
    a = answer or ""
    return bool(_THINK_OPEN.search(a)) and "</think>" not in a.lower()


_REFUSAL_MARKERS = (
    "do not mention", "does not mention", "doesn't mention", "not mentioned",
    "do not contain", "does not contain", "not contain", "not provide",
    "does not provide", "cannot determine", "can't determine", "cannot be determined",
    "no information", "not enough information", "insufficient", "unable to",
    "i don't know", "i do not know", "cannot find", "no relevant",
)


def is_refusal(answer: str) -> bool:
    """Heuristic: the answer abstains rather than commits to a value."""
    a = (answer or "").strip().lower()
    return (not a) or any(m in a for m in _REFUSAL_MARKERS)


def pack_truncate(docs: list[tuple[str, str]], cfg: SummarizerConfig) -> str:
    """Pack docs into [Document i] blocks, truncating to a fixed char budget split
    evenly across docs. Deterministic and near-lossless — unlike LLM summarization,
    it doesn't drop the fact before the answerer sees it (the search downstream
    metric should measure retrieval, not a lossy compressor)."""
    if not docs:
        return ""
    budget_chars = int(cfg.context_window * cfg.doc_budget_fraction) * CHARS_PER_TOKEN
    per_doc = max(500, budget_chars // len(docs))
    blocks = [f"[Document {i}] {src}\n{text[:per_doc]}" for i, (src, text) in enumerate(docs, 1)]
    return "\n\n".join(blocks)


def pack_mixed(snippets: list[tuple[str, str]], pages: list[tuple[str, str]],
               cfg: SummarizerConfig, small_chars: int = 4000,
               snippet_fraction: float = 0.6) -> str:
    """Pack snippets and pages without splitting the budget evenly between them.

    pack_truncate gives every item budget/n characters, which is fine at the 5.5
    items a current run produces. Forced breadth produces far more: at 24
    searches the even split hands each item ~4,000 characters, and by ~100 items
    a fetched page is clipped to 910 — so searching more would quietly destroy
    reading, and the two changes would be impossible to tell apart in the
    results. Here the small items (search blocks are capped at 4,000 characters
    by search_info, and are usually far shorter) go in whole, and the pages
    divide whatever is left. With few items this behaves like pack_truncate;
    it only diverges once breadth makes the even split wrong.
    """
    if not snippets and not pages:
        return ""
    budget = int(cfg.context_window * cfg.doc_budget_fraction) * CHARS_PER_TOKEN
    blocks: list[str] = []
    used = 0
    if snippets:
        # Snippets go in whole, but they are not allowed to eat the page budget:
        # search_info caps a result block at 4,000 characters, and 24 of those
        # would be 96,000 of a 98,304 budget. In practice a block is ~1,200, so
        # this ceiling almost never binds — it just stops the worst case from
        # crowding pages out entirely.
        per_snip = min(small_chars, max(300, int(budget * snippet_fraction) // len(snippets)))
        for src, text in snippets:
            chunk = text[:per_snip]
            blocks.append(f"[Document {len(blocks) + 1}] {src}\n{chunk}")
            used += len(chunk)
    if pages:
        per_page = max(500, (budget - used) // len(pages))
        for src, text in pages:
            blocks.append(f"[Document {len(blocks) + 1}] {src}\n{text[:per_page]}")
    return "\n\n".join(blocks)


# --- the search plans: -> EvidenceStore ------------------------------------
# Every plan returns the store it filled. The four original plans below record
# EXACTLY what they used to record — the accumulation lines are the same lines,
# writing to store.add_search / store.add_fetch instead of to two locals — so
# store.legacy_view() gives back the (searches, fetched) pair they used to
# return, unchanged. test_evidence.py checks both accumulation styles.

def _make_tools(sconf: SearchConfig, scrape_url: str, want_markup: bool = False):
    """SearchInfo/FetchUrl honoring the config's top_n and page cap (build_tools
    hardcodes both). ``want_markup`` asks the backend for markdown too, which is
    where link-following gets its pointers."""
    return (SearchInfo(top_n=sconf.top_n),
            FetchUrl(scrape_url=scrape_url, max_chars=sconf.fetch_chars,
                     want_markup=want_markup))


def _parallel(fn, items, workers: int) -> list:
    """Map ``fn`` over ``items`` concurrently, returning results IN INPUT ORDER.

    Order is the whole point. Everything parallelised in this file is already
    independent, so running it concurrently cannot change any single result --
    but the order results are written back in decides the order of the evidence
    blocks, and a rerun whose evidence depends on which network call returned
    first is not reproducible. Exceptions propagate, as they did when this code
    was a plain for-loop.
    """
    if len(items) <= 1 or workers <= 1:
        return [fn(x) for x in items]
    out: list = [None] * len(items)
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        futures = {pool.submit(fn, x): i for i, x in enumerate(items)}
        for fut in as_completed(futures):
            out[futures[fut]] = fut.result()
    return out


def _run_subagents(sconf, client, model, subqueries, system, schemas, registry):
    """One agent per sub-question, run concurrently.

    They were always independent -- each gets a fresh conversation containing
    only its own sub-question and cannot see what the others found -- so this
    changes nothing about what any of them does. Their tool calls are replayed
    into the store in sub-question order afterwards, so the evidence is what
    running them one after another would have produced.
    """
    return _parallel(
        lambda sq: run_with_tools(client, model, system, sq, schemas, registry,
                                  max_tool_calls=sconf.budget),
        list(subqueries), sconf.inner_workers)


def _record_agent_calls(store, results) -> None:
    """Replay agents' tool calls into the store, in order."""
    for result in results:
        for tc in result.tool_calls:
            if tc.name == "search_info":
                store.add_search(tc.arguments.get("query", ""), tc.result)
            elif tc.name == "fetch_url":
                url = tc.arguments.get("query") or tc.arguments.get("url") or ""
                if url and _is_good_page(tc.result):
                    store.add_fetch(url, tc.result)               # setdefault semantics


def _read_docs(sconf, reader, docs) -> None:
    """Fetch and summarize the chosen documents concurrently, write them in order.

    This is the slowest stage of the reading plans by a wide margin -- a Serper
    scrape can sit for the full 30-second read timeout, and six of them in a row
    is three minutes of a question's wall clock spent waiting on the network.
    """
    todo = [d for d in docs if not d.fetched]
    if not todo:
        return
    prepared = _parallel(lambda d: reader.fetch_only(d.doc_id), todo, sconf.inner_workers)
    for item in prepared:
        reader.record(item)


def _decompose(sconf: SearchConfig, client, model, question: str) -> list[str]:
    """Break the question into sub-questions.

    The parsing is the original: take non-blank lines, strip list punctuation,
    keep the first max_subqueries. What changed is that the reasoning trace can
    now be removed first — see SearchConfig.strip_think_subqueries for why that
    matters and why it is off by default.
    """
    system = (DECOMPOSE_SELFCONTAINED_SYSTEM if sconf.decompose_prompt == "self_contained"
              else DECOMPOSE_SYSTEM)
    raw = chat(client, model, system.format(k=sconf.max_subqueries), question)
    if sconf.strip_think_subqueries:
        raw = strip_think(raw)
    subqueries = [ln.strip("-*0123456789. ").strip() for ln in raw.splitlines() if ln.strip()]
    return subqueries[: sconf.max_subqueries] or [question]


VARIANTS_SYSTEM = (
    "Write {n} DIFFERENT web-search queries that would each help answer the given "
    "sub-question. Vary them: one using just the proper nouns, one phrased as a "
    "natural question, one using likely article or page titles. Output one query "
    "per line, no numbering, no commentary."
)


def _query_variants(sconf, client, model, subquestion: str) -> list[str]:
    """Several phrasings of one sub-question.

    A search engine is sensitive to phrasing and the agent only ever tries one,
    so a sub-question whose answer exists gets one shot at a lucky wording. The
    sub-question itself is always kept, so this can only add coverage.
    """
    out = [subquestion]
    if sconf.queries_per_subq <= 1:
        return out
    try:
        raw = strip_think(chat(client, model,
                               VARIANTS_SYSTEM.format(n=sconf.queries_per_subq),
                               subquestion, max_tokens=1024))
    except Exception:
        return out
    for line in raw.splitlines():
        q = line.strip("-*0123456789. ").strip()
        if q and len(q) < 300:
            out.append(q)
    return out[: sconf.queries_per_subq]


def _dedup(queries: list[str]) -> list[str]:
    """Drop repeated queries, case- and whitespace-insensitively, keeping order.
    Two sub-questions often generate the same phrasing, and a duplicate query
    returns duplicate results for the price of a real search."""
    seen, out = set(), []
    for q in queries:
        key = " ".join(q.lower().split())
        if key and key not in seen:
            seen.add(key)
            out.append(q)
    return out


DIGEST_SYSTEM = (
    "Summarize the page below for a researcher who is deciding what to look up "
    "next. Keep every concrete fact that bears on their question: names, dates, "
    "numbers, titles. Above all keep the names of OTHER people, places, works or "
    "organizations the question depends on — those are what they must look up "
    "next, and losing one ends their search. Drop navigation and unrelated "
    "sections. Report only what the page says; do not answer the question and do "
    "not speculate. At most {n} words."
)


def make_digester(client, model, sconf: SearchConfig, max_input_chars: int = 24000):
    """A function (focus, page_text) -> short summary for the AGENT to read.

    The answerer never sees this; it reads the full page out of the store. So a
    digest that drops a detail costs one bad next-query, not the answer. What it
    must not drop is the name of the next entity in the chain, which is what the
    prompt above is mostly about.
    """
    def digest(focus: str, text: str) -> str:
        raw = chat(client, model, DIGEST_SYSTEM.format(n=sconf.digest_words),
                   f"Question: {focus}\n\nPage:\n{text[:max_input_chars]}",
                   max_tokens=1536)
        # An unclosed <think> means it ran out of tokens mid-thought and there is
        # no summary; fall back to the head of the page rather than nothing.
        return strip_think(raw).strip() or text[:1500]
    return digest


def run_iterative(sconf, client, model, question, scrape_url):
    search, fetch = _make_tools(sconf, scrape_url)
    tools = [search] if sconf.fetch_k == 0 else [search, fetch]   # snippets-only -> no fetch tool
    registry = {t.name: t for t in tools}
    schemas = [t.to_openai_schema() for t in tools]
    result = run_with_tools(client, model, SEARCH_SYSTEM, question, schemas, registry,
                            max_tool_calls=sconf.budget)
    store = EvidenceStore(question)
    for tc in result.tool_calls:
        if tc.name == "search_info":
            store.add_search(tc.arguments.get("query", ""), tc.result)
        elif tc.name == "fetch_url":
            url = tc.arguments.get("query") or tc.arguments.get("url") or ""
            if url and _is_good_page(tc.result):
                store.add_fetch(url, tc.result)                   # setdefault semantics
    return store


def run_single(sconf, client, model, question, scrape_url):
    search, fetch = _make_tools(sconf, scrape_url)
    results_text = search(question)
    store = EvidenceStore(question)
    store.add_search(question, results_text)
    if sconf.fetch_k > 0:
        for url in parse_search_urls(results_text)[: sconf.fetch_k]:
            try:
                text = fetch(url)
            except Exception:  # transient Serper scrape failure -> skip this URL
                continue
            if _is_good_page(text):
                store.add_fetch(url, text, overwrite=True)        # `fetched[url] = text`
    return store


def run_decompose(sconf, client, model, question, scrape_url):
    subqueries = _decompose(sconf, client, model, question)
    search, fetch = _make_tools(sconf, scrape_url)
    store = EvidenceStore(question)
    store.subqueries = list(subqueries)
    for sq in subqueries:
        results_text = search(sq)
        store.add_search(sq, results_text)
        if sconf.fetch_k > 0:
            for url in parse_search_urls(results_text)[: sconf.fetch_k]:
                if not store.has_fetched_raw(url):
                    try:
                        text = fetch(url)
                    except Exception:  # transient Serper scrape failure -> skip this URL
                        continue
                    if _is_good_page(text):
                        store.add_fetch(url, text, overwrite=True)
    return store


def run_decompose_react(sconf, client, model, question, scrape_url):
    """Decompose the question into up to max_subqueries sub-questions, run an
    INDEPENDENT ReAct agent on each (its own search+fetch loop), and pool all the
    evidence. The downstream answerer then aggregates the pool into one answer, so
    this uses the standard evidence path and the identical metric suite."""
    subqueries = _decompose(sconf, client, model, question)

    search, fetch = _make_tools(sconf, scrape_url)
    # fetch_k == 0 means snippets-only, the way it already does in run_iterative.
    # No existing preset on this plan sets it (they all leave fetch_k at 3), so
    # this changes nothing that has been measured.
    tools = [search] if sconf.fetch_k == 0 else [search, fetch]
    registry = {t.name: t for t in tools}
    schemas = [t.to_openai_schema() for t in tools]

    store = EvidenceStore(question)
    store.subqueries = list(subqueries)
    _record_agent_calls(store, _run_subagents(sconf, client, model, subqueries,
                                              SEARCH_SYSTEM, schemas, registry))
    return store


# --- the new plans ---------------------------------------------------------

def _breadth_search(sconf, client, model, question, search, store) -> None:
    """Issue a fixed number of searches instead of letting the agent stop.

    Idea 2. The measured agent spends 5.47 of 24 allowed tool calls and 13.6% of
    the answerer's budget, so it is not limited by either — it just decides it is
    finished after about one search per sub-question. Here the sub-questions are
    generated, each is expanded into several phrasings, and every one is run.
    Snippets only, so 24 searches cost the answerer far less than one page would.
    """
    subqs = _decompose(sconf, client, model, question)
    store.subqueries = list(subqs)
    variants = _parallel(lambda sq: _query_variants(sconf, client, model, sq),
                         list(subqs), sconf.inner_workers)
    queries = _dedup([question] + [q for lst in variants for q in lst])[: sconf.max_searches]

    def run_one(q):
        try:
            return (q, search(q))
        except Exception:  # one bad query must not end the sweep
            return None

    for got in _parallel(run_one, queries, sconf.inner_workers):
        if got is not None:
            store.add_search(*got)


SELECT_SYSTEM = (
    "You are choosing which pages to read in full. Below is the question and a "
    "numbered list of candidate pages with short descriptions. Pick the {k} most "
    "likely to contain facts the question needs — prefer pages covering parts of "
    "the question nothing else covers, over several pages about the same thing. "
    "Output ONLY the ids, one per line, like:\nd3\nd7"
)

LINK_SELECT_SYSTEM = (
    "You are following links to find a fact you are still missing. Below is the "
    "question, summaries of the pages you have already read, and a numbered list "
    "of pages those link to. Pick the {k} links most likely to supply something "
    "the question needs that you do NOT already have. Prefer a link that fills a "
    "gap over one that elaborates what you already know. "
    "Output ONLY the ids, one per line, like:\nd3\nd7"
)


def _select_docs(sconf, client, model, preamble, candidates, k, instruction=None) -> list:
    """Ask the model which candidate documents are worth reading.

    Deliberately model-based rather than scored: on this codebase's own numbers,
    query-aware model selection kept 98.1% of answers in a compressed context
    where BM25 kept 75.2%, and BM25 passage selection lost outright to simply
    taking the head of the page. So a scorer is used only to shorten the list
    before the model sees it, never to make the choice.
    """
    if not candidates:
        return []
    if len(candidates) <= k:
        return list(candidates)
    listing = "\n".join(
        f"[{d.doc_id}] {d.title or d.url}\n    {d.best_snippet()[:300]}" for d in candidates)
    try:
        raw = strip_think(chat(client, model,
                               (instruction or SELECT_SYSTEM).format(k=k),
                               f"{preamble}\n\nCandidates:\n{listing}",
                               max_tokens=1024))
    except Exception:
        return list(candidates[:k])
    by_id = {d.doc_id: d for d in candidates}
    picked, seen = [], set()
    for token in re.findall(r"\bd\d+\b", raw):
        if token in by_id and token not in seen:
            seen.add(token)
            picked.append(by_id[token])
    # A model that answers with prose instead of ids should not silently mean
    # "read nothing"; fall back to the front of the (already ranked) list.
    return (picked or list(candidates))[:k]


def _shortlist_links(links, limit: int):
    """Shortlist one page's links for the model to choose from, in DOCUMENT ORDER.

    An earlier version ranked by overlap between the anchor text and the
    question's words. That is exactly backwards for multi-hop. Measured on
    frames_41 — "what Formula One car was driven in 1994 by the nephew of a
    driver who drove a Ferrari 312T..." — the answer is Minardi M194, and the
    question contains neither "Minardi" nor "M194", because not naming it is the
    whole question. Its anchor scored zero while hundreds of anchors like
    "Formula One", "racing driver", "Italy" and "Ferrari" scored high and filled
    every slot, so the one link worth following was dropped before the model ever
    saw it. Ranking by name-match defeats the only reason to follow links at all.

    Document order has no such bias: it is structural, not semantic — body
    before navigation boxes, lead before tail. On the same page the three links
    that mattered sit at positions 9, 19 and 50 of 771, and 100 anchors cost
    about 5,000 characters, so there is little to gain by filtering harder.
    """
    return list(links[:limit])


def run_decompose_react_read(sconf, client, model, question, scrape_url):
    """Decompose, search each part with an agent that can ONLY see snippets, then
    pick what is worth reading and read it.

    The difference from run_breadth is the second stage, and the measurements say
    it is the stage that matters. run_breadth replaced the agent with a fixed list
    of pre-written query phrasings: 2.5x the searches (4.1 -> 10.4) and LESS gold
    found (47.2% -> 42.1%), because what got thrown away was the agent choosing
    its next query from what the last one returned. Here the agent keeps choosing,
    it just cannot spend its 32k context on a page — which is what stopped it
    searching in the first place, since one uncapped fetch costs about fifty
    searches' worth of room.

    Reading then happens once, at the end, out of the pooled snippets. By then
    the choice is made with everything that was found in view, rather than mid-run
    with only the current sub-question's results.
    """
    subqueries = _decompose(sconf, client, model, question)
    search, fetch = _make_tools(sconf, scrape_url)
    store = EvidenceStore(question)
    store.subqueries = list(subqueries)

    # Stage 2: one agent per sub-question, search tool only, run concurrently.
    registry = {search.name: search}
    schemas = [search.to_openai_schema()]
    _record_agent_calls(store, _run_subagents(sconf, client, model, subqueries,
                                              SEARCH_SNIPPETS_SYSTEM, schemas, registry))

    # Stages 3-4: choose from the pooled, deduplicated documents, then read them.
    if sconf.select_k <= 0:
        return store
    digest_fn = make_digester(client, model, sconf) if sconf.digest else None
    reader = StoreFetch(fetch, store, digest_fn=digest_fn,
                        agent_chars=sconf.fetch_chars, focus=question)
    candidates = [d for d in store.docs.values() if not d.fetched]
    _read_docs(sconf, reader,
               _select_docs(sconf, client, model, f"Question: {question}",
                            candidates, sconf.select_k))
    store.n_fetch_errors = reader.n_fetch_errors
    return store


def run_breadth(sconf, client, model, question, scrape_url):
    """Idea 2 alone: many searches, snippets only, nothing fetched."""
    search, _ = _make_tools(sconf, scrape_url)
    store = EvidenceStore(question)
    _breadth_search(sconf, client, model, question, search, store)
    return store


def run_breadth_read(sconf, client, model, question, scrape_url):
    """Breadth, then choose what to read, then optionally follow its links.

    The pipeline, in order:
      1. decompose and run every phrasing of every sub-question (idea 2)
      2. pool and deduplicate the results — one entry per page (idea 3)
      3. the model picks which pages are worth reading
      4. read them; the store keeps the full page, the agent gets a summary (idea 4)
      5. for each page read, shortlist its outgoing links and let the model pick
         the ones to follow, then read those too (idea 1)

    Step 5 is the one that reaches pages no query can. A question like "what car
    did the nephew of X drive" names neither the nephew nor the car, so nothing
    surfaces them — but X's page links to both.
    """
    search, fetch = _make_tools(sconf, scrape_url, want_markup=sconf.follow_links)
    store = EvidenceStore(question)
    _breadth_search(sconf, client, model, question, search, store)

    digest_fn = make_digester(client, model, sconf) if sconf.digest else None
    reader = StoreFetch(fetch, store, digest_fn=digest_fn,
                        agent_chars=sconf.fetch_chars, focus=question)

    def read(docs) -> None:
        _read_docs(sconf, reader, docs)

    candidates = [d for d in store.docs.values() if not d.fetched]
    read(_select_docs(sconf, client, model, f"Question: {question}",
                      candidates, sconf.select_k))

    if not sconf.follow_links:
        store.n_fetch_errors = reader.n_fetch_errors
        return store

    link_source = LINK_SOURCES[sconf.link_source]()
    expanded: set[str] = set()
    for _ in range(sconf.link_rounds):
        frontier = []
        for doc in list(store.fetched_docs()):
            if doc.doc_id in expanded:       # a later round must not re-list the
                continue                     # same page's links as if they were new
            expanded.add(doc.doc_id)
            shortlist = _shortlist_links(link_source(doc, limit=sconf.links_per_page),
                                         sconf.links_per_page)
            if shortlist:
                frontier.extend(store.add_links(doc.doc_id, shortlist))
        fresh = [d for d in frontier if not d.fetched]
        if not fresh:
            break
        # This is where the digests earn their keep. The choice is "which link
        # fills a gap", which cannot be made without knowing what the pages
        # already read actually said — and their full text (6 pages, ~150k
        # characters) does not fit in the prompt. The digests do.
        read_so_far = "\n".join(
            f"[{d.doc_id}] {d.title or d.url}: {(d.digest or d.best_snippet())[:600]}"
            for d in store.fetched_docs())
        read(_select_docs(sconf, client, model,
                          f"Question: {question}\n\nPages already read:\n{read_so_far}",
                          fresh, sconf.link_select_k, instruction=LINK_SELECT_SYSTEM))
    store.n_fetch_errors = reader.n_fetch_errors
    return store


PLANS = {"iterative": run_iterative, "single": run_single, "decompose": run_decompose,
         "decompose_react": run_decompose_react, "breadth": run_breadth,
         "breadth_read": run_breadth_read,
         "decompose_react_read": run_decompose_react_read}


# --- per-question driver ---------------------------------------------------

def _recall(gold: set, found: set) -> float | None:
    return len(gold & found) / len(gold) if gold else None


def select_evidence(policy: str, snippets: list, pages: list) -> tuple[list, list]:
    """The evidence the answerer reads under one policy, kept split into
    (snippets, pages) because they cost very different amounts to pack.

    Snippets are a query-conditioned extract of a page; a fetched page is raw
    prose. Under "pages_else_snippets" — the original behaviour — a single fetch
    discards every snippet the agent gathered, which is most runs.
    """
    if policy == "union":
        return snippets, pages
    if policy == "snippets_only":
        return snippets, []
    return ([], pages) if pages else (snippets, [])       # "pages_else_snippets"


def score_evidence(rec: dict, policy: str, snippets: list, pages: list, question: str,
                   gt: str, client, model, cfg, sconf) -> dict:
    """Answer + grade one evidence policy. Retrieval fields are copied in from
    `rec`, so every policy record describes the same retrieval."""
    evidence = snippets + pages
    out = dict(rec, evidence_policy=policy, n_evidence=len(evidence))

    # Backend-free retrieval signals: is the gold answer present in the raw
    # retrieved evidence (retrieval), and does it survive packing (packing loss)?
    raw_evidence = "\n\n".join(text for _, text in evidence)
    doc_context = (pack_mixed(snippets, pages, cfg) if sconf.packing == "mixed"
                   else pack_truncate(evidence, cfg))            # deterministic, no LLM
    out["answer_in_evidence"] = answer_in_text(gt, raw_evidence)
    out["answer_in_context"] = answer_in_text(gt, doc_context)
    out["context_chars"] = len(doc_context)

    # downstream: the same simple Qwen call used to build the splits.
    answer = (answer_with_docs(client, model, question, doc_context,
                               max_tokens=sconf.answer_max_tokens) if evidence else "")
    visible = strip_think(answer)
    # Graded both ways on the same generation: correct_raw is the original metric
    # (whole string, reasoning trace included), correct_visible grades only what
    # the model committed to. They differ, so a comparison must pick one and keep it.
    out["answer"] = answer
    out["truncated_mid_think"] = truncated_mid_think(answer)
    out["correct_raw"] = bool(answer) and judge_answer(question, gt, answer)
    out["correct_visible"] = bool(visible) and judge_answer(question, gt, visible)
    correct = out["correct_raw"]                                 # unchanged headline metric
    out["downstream_correct"] = correct
    out["refused"] = is_refusal(answer)
    out["refused_visible"] = is_refusal(visible)
    out["outcome"] = "correct" if correct else ("refused" if out["refused"] else "wrong")
    out["status"] = "ok"
    return out


def process_question(row, sconf, client, model, cfg, scrape_url,
                     policies: list[str] | None = None) -> list[dict]:
    """Run the search plan ONCE, then score it under each evidence policy.

    The evidence policy is a post-hoc choice about what to hand the answerer — it
    does not change what was searched or fetched. Scoring several policies off a
    single retrieval pass makes the comparison exactly paired (same queries, same
    pages, same question) and costs no extra search API calls, only local answer
    calls plus judging.
    """
    qid, question, gt = row.get("id"), row["question"], row["ground_truth"]
    policies = policies or [sconf.evidence]
    base = {"id": qid, "question": question, "ground_truth": gt, "config": sconf.name}
    try:
        store = PLANS[sconf.query_plan](sconf, client, model, question, scrape_url)
        searches, fetched = store.legacy_view()

        # retrieval overlap vs gold wiki_links (normalized both sides)
        gold = {normalize_url(u) for u in gold_urls(row)}
        surfaced = store.search_keys()          # what a QUERY returned
        found = store.found_keys()              # queries plus links actually shown
        fetched_norm = store.fetched_keys()
        base.update(
            n_search=len(searches), n_fetch=len(fetched),
            num_surfaced=len(surfaced), num_fetched=len(fetched_norm),
            recall_surfaced=_recall(gold, surfaced),
            recall_fetched=_recall(gold, fetched_norm),
            precision_fetched=(len(gold & fetched_norm) / len(fetched_norm)) if fetched_norm else None,
            found_all=bool(gold) and gold <= surfaced,
            # Link-following reaches pages no query returned, so it needs its own
            # pair of numbers. recall_surfaced/found_all stay query-only, which
            # is how every run recorded so far was measured; recall_found and
            # found_all_any count links too. A link is only counted once it has
            # actually been put in front of the agent — never merely because it
            # was one of four hundred links on a page that was read.
            recall_found=_recall(gold, found),
            found_all_any=bool(gold) and gold <= found,
            n_docs=len(store.docs), n_links=store.n_links_followed(),
            # A failed scrape no longer aborts the question, so the count has to
            # be visible instead: a plan that fetched little because the backend
            # was down must not read as a plan that chose to fetch little.
            n_fetch_errors=store.n_fetch_errors,
            # The sub-questions and the queries actually issued. Without these
            # the last comparison could not be explained after the fact — the
            # sub-questions turned out to be the whole story.
            subqueries=list(store.subqueries),
            queries=[q for q, _ in searches],
            # Keep the raw URL sets so a failure can be diagnosed without paying
            # for the search again.
            surfaced_urls=sorted(surfaced), fetched_urls=sorted(fetched_norm),
            gold_urls=sorted(gold),
        )

        snippets = store.snippet_blocks()       # one entry per SEARCH (legacy shape)
        pages = store.pages()                   # (url, page text), in fetch order
        # Each policy is an independent answer + two judge calls off the SAME
        # retrieval, so they can run together; the results are still returned in
        # the order the policies were listed.
        return _parallel(
            lambda p: score_evidence(base, p, *select_evidence(p, snippets, pages),
                                     question, gt, client, model, cfg, sconf),
            policies, sconf.inner_workers)
    except Exception as exc:  # keep the pool going
        # One error record per policy, so a failed question is missing from every
        # policy's subset rather than only from some.
        return [dict(base, evidence_policy=p, status="error", error=str(exc))
                for p in policies]


def aggregate(records: list[dict]) -> dict:
    ok = [r for r in records if r.get("status") == "ok"]

    def mean(key):
        vals = [r[key] for r in ok if r.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else 0.0

    n = len(ok)
    rate = lambda key: (sum(1 for r in ok if r.get(key)) / n) if n else 0.0
    have_ans = [r for r in ok if r.get("answer_in_evidence")]
    return {
        "n": n,
        # retrieval (URL overlap vs gold)
        "recall_surfaced": mean("recall_surfaced"),
        "recall_fetched": mean("recall_fetched"),
        "precision_fetched": mean("precision_fetched"),
        "found_all_rate": rate("found_all"),
        # queries plus links; identical to the two above when nothing follows links
        "recall_found": mean("recall_found"),
        "found_all_any_rate": rate("found_all_any"),
        # retrieval (backend-free: answer text present in evidence) + packing loss
        "answer_in_evidence_rate": rate("answer_in_evidence"),
        "answer_in_context_rate": rate("answer_in_context"),
        # answer backend
        "downstream_accuracy": rate("downstream_correct"),
        "accuracy_visible": rate("correct_visible"),
        "accuracy_raw_string": rate("correct_raw"),
        "refusal_rate": rate("refused"),
        "refusal_rate_visible": rate("refused_visible"),
        "truncated_mid_think_rate": rate("truncated_mid_think"),
        # backend quality given retrieval succeeded: correct when the fact was present
        "accuracy_given_evidence": (sum(1 for r in have_ans if r.get("downstream_correct")) / len(have_ans))
                                   if have_ans else 0.0,
        "mean_searches": mean("n_search"),
        "mean_fetches": mean("n_fetch"),
        "mean_docs": mean("n_docs"),
        "mean_links": mean("n_links"),
        "mean_fetch_errors": mean("n_fetch_errors"),
    }


def run_config(sconf, rows, client, args, cfg, policies) -> list[dict]:
    """All records for one config: len(policies) per question, one retrieval each."""
    records: list[dict] = []
    out_path = args.out_dir / f"search_{sconf.name}_cache.jsonl"
    lock = Lock()
    with open(out_path, "w") as cache, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_question, row, sconf, client, args.model, cfg,
                               args.scrape_url, policies)
                   for row in rows]
        for future in tqdm(as_completed(futures), total=len(futures), desc=sconf.name, unit="q"):
            for rec in future.result():
                with lock:
                    cache.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    cache.flush()
                records.append(rec)
    return records


def main(args: argparse.Namespace) -> None:
    rows = json.loads(Path(args.dataset).read_text())
    if args.limit is not None:
        rows = rows[: args.limit]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    cfg = SummarizerConfig(model=args.model, context_window=args.context_window,
                           summary_tokens=args.summary_tokens)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)

    policies = [p.strip() for p in args.evidence_policies.split(",") if p.strip()]

    summary, all_records = {}, {}
    for name in configs:
        sconf = PRESETS[name]
        if args.link_source:
            # A backend choice, not a method choice: it only changes where the
            # outgoing links of an already-fetched page come from.
            sconf = replace(sconf, link_source=args.link_source)
        if args.inner_workers is not None:
            sconf = replace(sconf, inner_workers=args.inner_workers)
        records = run_config(sconf, rows, client, args, cfg, policies or None)
        for policy in {r["evidence_policy"] for r in records}:
            rows_p = [r for r in records if r["evidence_policy"] == policy]
            # Keyed by config alone when there is only one policy, so single-policy
            # runs keep the summary shape the earlier results were written in.
            key = name if len(policies) <= 1 else f"{name}|{policy}"
            summary[key] = aggregate(rows_p)
            all_records[key] = rows_p

    # Errors are not uniform across configs (uncapped fetches abort ReAct runs but
    # never snippet-only ones), so per-config n differs and the headline numbers
    # are computed over different question sets. Report the intersection too.
    keys = list(all_records)
    if len(keys) > 1:
        ok_ids = [{r["id"] for r in all_records[k] if r.get("status") == "ok"} for k in keys]
        common = set.intersection(*ok_ids)
        summary["_common_subset"] = {
            "n": len(common),
            "configs": {k: aggregate([r for r in all_records[k] if r["id"] in common])
                        for k in keys},
        }

    args.summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    configs = keys
    # retrieval = ans_in_evid (fact present in evidence, backend-free);
    # ans_in_ctx = survives packing; downstream = correct; acc|evid = correct when fact present.
    w = max(20, max(len(k) for k in configs) + 2)
    print(f"\n{'config':<{w}}{'ans_in_evid':>12}{'ans_in_ctx':>11}{'downstream':>11}"
          f"{'refused':>9}{'acc|evid':>9}{'recall_surf':>12}")
    for name in configs:
        a = summary[name]
        print(f"{name:<{w}}{a['answer_in_evidence_rate']:>12.1%}{a['answer_in_context_rate']:>11.1%}"
              f"{a['downstream_accuracy']:>11.1%}{a['refusal_rate']:>9.1%}"
              f"{a['accuracy_given_evidence']:>9.1%}{a['recall_surfaced']:>12.1%}")
    if "_common_subset" in summary:
        sub = summary["_common_subset"]
        print(f"\ncommon subset (n={sub['n']}): the only fair cross-config comparison")
        print(f"{'config':<{w}}{'downstream':>11}{'acc_visible':>13}{'trunc_think':>12}{'recall_surf':>12}")
        for name in configs:
            a = sub["configs"][name]
            print(f"{name:<{w}}{a['downstream_accuracy']:>11.1%}{a['accuracy_visible']:>13.1%}"
                  f"{a['truncated_mid_think_rate']:>12.1%}{a['recall_surfaced']:>12.1%}")
    print(f"\nper-config caches -> {args.out_dir}/search_<config>_cache.jsonl")
    print(f"summary -> {args.summary_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path("datasets/frames_qwen_answerable.json"))
    ap.add_argument("--configs", default=",".join(PRESETS),
                    help="Comma-separated subset of search presets to evaluate.")
    ap.add_argument("--evidence-policies", default="",
                    help="Comma-separated evidence policies to score from EACH retrieval "
                         "pass (pages_else_snippets | union | snippets_only). The policy "
                         "only decides what the answerer reads, so several can be scored "
                         "off one pass — exactly paired, with no extra search API calls. "
                         "Default: just the config's own policy.")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--base-url", default="http://localhost:7472/v1", help="Local vLLM endpoint.")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--scrape-url", default=SERPER_SCRAPE_URL,
                    help="fetch_url backend (live Serper by default; open-web URLs miss the local cache).")
    ap.add_argument("--link-source", default="", choices=["", *LINK_SOURCES],
                    help="Where the outgoing links of a fetched page come from. "
                         "'document' (the default in every preset) reads them out of "
                         "the markup the scrape backend returned, which is corpus-"
                         "agnostic but needs a backend that returns markdown. "
                         "'wiki_api' gets them free from MediaWiki, for measuring "
                         "link-following on this dataset without paying to re-scrape.")
    ap.add_argument("--context-window", type=int, default=32768)
    ap.add_argument("--summary-tokens", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=5, help="Questions processed concurrently.")
    ap.add_argument("--inner-workers", type=int, default=None,
                    help="How much ONE question may do at once: its sub-question agents, "
                         "its page fetches, its per-policy scoring (default 4). Total "
                         "in-flight work is roughly --workers x this, so raise --workers "
                         "for throughput and this to shorten a single question. Serper "
                         "calls are capped separately by SERPER_MAX_CONCURRENCY (16); "
                         "raise that too if searches start queueing.")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N questions.")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--summary-out", type=Path, default=Path("outputs/search_step_summary.json"))
    main(ap.parse_args())

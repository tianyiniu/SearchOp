"""Evolve search-control programs on FRAMES (experiment 2).

A program is an ordered list of if-then rules, the same shape as
evolve_program_mcq.py. After every action it is consulted with the observable
state of the search (queries left, documents found, whether the checker was
satisfied) and answers with the next action. One program serves every question;
each question takes its own path through the rules. The answer key is never
visible to a rule.

ACTIONS (each one bounded, so budgets are enforceable):
    decompose   split the question into sub-questions; they join the query queue
    variants    add extra phrasings of each sub-question to the queue
    search      run ONE web search with the next queued query
    read        pick up to 3 unread documents and fetch their full text
    links       list what the read pages link to, follow the 2 most promising
    hop         answer one sub-question from the evidence, then rewrite the next
                sub-question into a standalone query using that fact
    check       draft an answer, ask whether the evidence supports it; if not,
                queue a search for the missing fact
    stop        finish; the draft (or a fresh answer) is graded

DETERMINISM AND COST. Every external effect — web search, page fetch, every
model call, the judge — goes through a persistent on-disk cache keyed by exact
input. First use pays; every replay is free. Two programs that share a step pay
for it once, which makes comparisons paired. The cache file grows large
(fetched pages are stored whole); it lives next to the other round caches.

SETUP (the script checks all of this at startup and says what is missing):
    1. SERPER_API_KEY   — web search AND page fetches, both live and paid.
       Both are capped per question and cached on disk by exact input, so an
       identical call is never paid for twice, across runs included.
    2. OPENAI_API_KEY   — the judge (gpt-5.4-mini).
    3. vLLM serving Qwen/Qwen3-14B on :7472.

    Fetches default to live Serper. The local wiki server
    (scripts/wiki_backend.py, --scrape-url http://127.0.0.1:5000/) remains an
    option for free runs, but it holds only the 1,052 gold pages: every other
    URL 404s, which both changes behavior and leaks gold membership into
    selection. Live-plus-cache is the honest default.

MODES
    --v0                run the hand-written program on the dataset (full metrics,
                        cache file compatible with frames_headroom.py)
    --evolve            random rule edits, raced on the train half only
    --test PROG.json    score a saved program on the held-out test half
    --limit N           first N questions (debug; use before any paid run)

    python3 scripts/evolve_program_frames.py --v0 --limit 5      # smoke test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from doc_qa import SummarizerConfig, answer_with_docs, normalize_url  # noqa: E402
from evidence import EvidenceStore  # noqa: E402
from eval_search_step import (  # noqa: E402
    DECOMPOSE_SYSTEM, SearchConfig, _query_variants, _read_docs, _select_docs,
    aggregate, chat, is_refusal, pack_mixed, score_evidence, strip_think)
from evolve_program_mcq import canon_prog, mutate_program  # noqa: E402
from llm_judge import judge_answer  # noqa: E402
from store_tools import StoreFetch, StoreSearch, WikiApiLinks  # noqa: E402
from tools import LOCAL_SCRAPE_URL, SERPER_SCRAPE_URL, FetchUrl, SearchInfo  # noqa: E402

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# One shared SearchConfig for the imported helpers. Values follow the measured
# best practices: capped pages (uncapped fetches overflowed Qwen's 32k context
# and aborted 18/266 decompose_react questions), union evidence, mixed packing,
# room for the answerer to finish reasoning.
SCONF = SearchConfig("program", "program", top_n=5, fetch_chars=8000,
                     evidence="union", packing="mixed", answer_max_tokens=4096,
                     select_k=3, links_per_page=40, link_select_k=2,
                     queries_per_subq=3, inner_workers=4)

# Hard per-question budgets, identical for every program — a program cannot win
# by spending more, only by spending better.
MAX_SEARCHES = 12
MAX_FETCHES = 8
MAX_ACTS = 24
MAX_QUEUE = 30


# --- the persistent call cache ----------------------------------------------

class CallCache:
    """Append-only jsonl cache: sha1(kind + payload) -> value. Everything the
    pipeline does to the outside world goes through here, so a repeated step is
    free and a rerun is deterministic."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, object] = {}
        self._lock = threading.Lock()
        if self.path.exists():
            for line in tqdm(self.path.open(errors="replace"),
                             desc=f"load {self.path.name}", unit="line",
                             miniters=10_000, leave=False):
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        self._mem[r["k"]] = r["v"]
                    except json.JSONDecodeError:
                        continue
        self._fh = self.path.open("a")
        self.hits = self.misses = 0

    @staticmethod
    def key(kind: str, payload: str) -> str:
        return hashlib.sha1(f"{kind}\x1f{payload}".encode()).hexdigest()

    def get_or(self, kind: str, payload: str, fn):
        k = self.key(kind, payload)
        with self._lock:
            if k in self._mem:
                self.hits += 1
                return self._mem[k]
        v = fn()
        with self._lock:
            self.misses += 1
            self._mem[k] = v
            self._fh.write(json.dumps({"k": k, "kind": kind, "v": v},
                                      ensure_ascii=False) + "\n")
            self._fh.flush()
        return v


class CachingClient:
    """Wraps an OpenAI-compatible client so chat.completions.create is cached by
    exact request. Makes every model call in the pipeline deterministic across
    runs and shared across programs."""

    class _Msg:
        def __init__(self, content):
            self.message = type("M", (), {"content": content})()

    class _Resp:
        def __init__(self, content):
            self.choices = [CachingClient._Msg(content)]

    def __init__(self, client, cache: CallCache):
        self._client, self._cache = client, cache
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kw):
        payload = json.dumps({k: kw.get(k) for k in
                              ("model", "messages", "temperature", "max_tokens")},
                             sort_keys=True, ensure_ascii=False)
        content = self._cache.get_or("chat", payload, lambda: (
            self._client.chat.completions.create(**kw).choices[0].message.content or ""))
        return CachingClient._Resp(content)


class CachedSearch:
    """search_info through the cache. Payload is the query alone, so a query
    repeated by any program in any run costs one Serper call ever."""

    name = "search_info"

    def __init__(self, inner: SearchInfo, cache: CallCache):
        self._inner, self._cache = inner, cache

    def __call__(self, query: str) -> str:
        return self._cache.get_or("search", f"{self._inner.top_n}\x1f{query}",
                                  lambda: self._inner(query))

    def to_openai_schema(self):
        return self._inner.to_openai_schema()


class CachedFetch:
    """fetch_url through the cache (fetch_raw interface, as StoreFetch expects)."""

    name = "fetch_url"

    def __init__(self, inner: FetchUrl, cache: CallCache):
        self._inner, self._cache = inner, cache

    def fetch_raw(self, url: str) -> dict:
        return self._cache.get_or("fetch", url, lambda: self._inner.fetch_raw(url))

    def to_openai_schema(self):
        return self._inner.to_openai_schema()


# --- per-question runtime state ---------------------------------------------

class QState:
    def __init__(self, row: dict):
        self.row = row
        self.store = EvidenceStore(row["question"])
        self.queue: list[str] = [row["question"]]
        self.subqs: list[str] = []
        self.hop_answers: dict[int, str] = {}     # subq index -> answer text
        self.actions: list[str] = []
        self.n_searches = self.n_fetches = self.n_checks = 0
        self.last_new_docs: int | None = None
        self.verdict: str | None = None           # None | "supported" | "unsupported"
        self.draft: str = ""
        self.links_done = False


NUMERIC = {"acts": lambda s: len([a for a in s.actions if not a.startswith("noop")]),
           "searches": lambda s: s.n_searches,
           "fetched": lambda s: s.n_fetches,
           "checks": lambda s: s.n_checks,
           "docs": lambda s: len(s.store.docs),
           "unread": lambda s: sum(1 for d in s.store.docs.values() if not d.fetched),
           "facts": lambda s: len(s.hop_answers),
           "queue": lambda s: len(s.queue)}

_NUM_RE = re.compile(r"(%s)(==|>=|<=|<|>)(\d+)" % "|".join(NUMERIC))


def cond(c: str, st: QState) -> bool:
    if m := _NUM_RE.fullmatch(c):
        v = NUMERIC[m.group(1)](st)
        op, k = m.group(2), int(m.group(3))
        return {"==": v == k, ">=": v >= k, "<=": v <= k, "<": v < k, ">": v > k}[op]
    if c == "queue_empty":
        return not st.queue
    if c == "has_subqs":
        return bool(st.subqs)
    if c == "checked":
        return st.verdict is not None
    if c == "supported":
        return st.verdict == "supported"
    if c == "unsupported":
        return st.verdict == "unsupported"
    if c == "draft_refused":
        return bool(st.draft) and is_refusal(strip_think(st.draft))
    if c == "no_new_docs":
        return st.last_new_docs == 0
    if c.startswith("last:"):
        return bool(st.actions) and st.actions[-1] == c[5:]
    if c.startswith("ran:"):
        return c[4:] in st.actions
    raise ValueError(f"unknown condition {c!r}")


# --- the actions -------------------------------------------------------------

HOP_ANSWER_SYSTEM = (
    "Answer the sub-question from the documents below, in at most a sentence. "
    "If the documents do not answer it, reply exactly: UNKNOWN.")
HOP_REWRITE_SYSTEM = (
    "Rewrite the sub-question as ONE standalone web-search query, substituting "
    "the known fact for any pronoun or description that refers to it. Output "
    "only the query.")
CHECK_SYSTEM = (
    "You are auditing a draft answer against the evidence summaries below. If "
    "the evidence states the facts the draft relies on, reply exactly "
    "'SUPPORTED'. Otherwise reply 'MISSING: <one standalone web-search query "
    "for the single most important missing fact>'.")


class Runtime:
    """Shared clients, tools and budgets for one run. Thread-safe: each question
    gets its own QState; the caches lock internally."""

    def __init__(self, args, cache: CallCache):
        from openai import OpenAI

        self.cache = cache
        self.qwen = CachingClient(OpenAI(base_url=args.base_url, api_key=args.api_key),
                                  cache)
        self.model = args.model
        self.search = CachedSearch(SearchInfo(top_n=SCONF.top_n), cache)
        self.fetch = CachedFetch(FetchUrl(scrape_url=args.scrape_url,
                                          max_chars=SCONF.fetch_chars), cache)
        self.links = WikiApiLinks()
        self.cfg = SummarizerConfig(model=args.model)
        self.judge_calls = 0

    def aux(self, system: str, user: str, max_tokens: int = 1024) -> str:
        return strip_think(chat(self.qwen, self.model, system, user,
                                max_tokens=max_tokens))

    def judged(self, question: str, gt: str, answer: str) -> bool:
        if not answer:
            return False
        self.judge_calls += 1
        return bool(self.cache.get_or(
            "judge", f"{question}\x1f{gt}\x1f{answer}",
            lambda: judge_answer(question, gt, answer)))


def _snippet_listing(st: QState, limit: int = 30) -> str:
    docs = list(st.store.docs.values())[:limit]
    return "\n".join(f"[{d.doc_id}] {d.title or d.url}: {d.best_snippet()[:200]}"
                     for d in docs)


def _draft(rt: Runtime, st: QState) -> str:
    snippets = st.store.snippet_blocks()
    pages = st.store.pages()
    ctx = pack_mixed(snippets, pages, rt.cfg)
    if not ctx:
        return ""
    return answer_with_docs(rt.qwen, rt.model, st.row["question"], ctx,
                            max_tokens=SCONF.answer_max_tokens)


def do_decompose(rt: Runtime, st: QState) -> bool:
    if st.subqs:
        return False
    # decompose_snip behaviour on purpose: the raw reply lines, reasoning trace
    # included, are the queries. Measured 6.4 points ABOVE stripping the trace,
    # because trace lines restate the question's context (see eval_search_step's
    # note on strip_think_subqueries). The cleaned lines double as sub-questions
    # for the hop/check actions.
    raw = chat(rt.qwen, rt.model,
               DECOMPOSE_SYSTEM.format(k=SCONF.max_subqueries), st.row["question"])
    lines = [ln.strip("-*0123456789. ").strip() for ln in raw.splitlines() if ln.strip()]
    clean = [ln.strip("-*0123456789. ").strip()
             for ln in strip_think(raw).splitlines() if ln.strip()]
    st.subqs = (clean or lines)[: SCONF.max_subqueries]
    st.store.subqueries = list(st.subqs)
    st.queue.extend(lines[:8])
    del st.queue[MAX_QUEUE:]
    return True


def do_variants(rt: Runtime, st: QState) -> bool:
    if not st.subqs or "variants" in st.actions:
        return False
    for sq in st.subqs:
        st.queue.extend(_query_variants(SCONF, rt.qwen, rt.model, sq)[1:])
    seen: set[str] = set()
    st.queue = [q for q in st.queue
                if (k := " ".join(q.lower().split())) and k not in seen
                and not seen.add(k)][:MAX_QUEUE]
    return True


def do_search(rt: Runtime, st: QState) -> bool:
    if not st.queue or st.n_searches >= MAX_SEARCHES:
        return False
    query = st.queue.pop(0)
    before = len(st.store.docs)
    try:
        StoreSearch(rt.search, st.store, annotate=False)(query)
    except Exception:
        st.last_new_docs = 0
        return True                                   # a bad query still spent a turn
    st.n_searches += 1
    st.last_new_docs = len(st.store.docs) - before
    return True


def do_read(rt: Runtime, st: QState) -> bool:
    candidates = [d for d in st.store.docs.values() if not d.fetched]
    room = MAX_FETCHES - st.n_fetches
    if not candidates or room <= 0:
        return False
    reader = StoreFetch(rt.fetch, st.store, agent_chars=SCONF.fetch_chars,
                        focus=st.row["question"])
    chosen = _select_docs(SCONF, rt.qwen, rt.model,
                          f"Question: {st.row['question']}", candidates,
                          min(SCONF.select_k, room))
    _read_docs(SCONF, reader, chosen)
    st.n_fetches += sum(1 for d in chosen if st.store.get(d.doc_id).fetched)
    st.store.n_fetch_errors += reader.n_fetch_errors
    return True


def do_links(rt: Runtime, st: QState) -> bool:
    read_docs = st.store.fetched_docs()
    room = MAX_FETCHES - st.n_fetches
    if not read_docs or st.links_done or room <= 0:
        return False
    st.links_done = True
    frontier = []
    for doc in read_docs:
        shortlist = rt.cache.get_or("links", doc.key,
                                    lambda d=doc: rt.links(d, limit=SCONF.links_per_page))
        if shortlist:
            frontier.extend(st.store.add_links(doc.doc_id, [tuple(x) for x in shortlist]))
    fresh = [d for d in frontier if not d.fetched]
    if not fresh:
        return True
    reader = StoreFetch(rt.fetch, st.store, agent_chars=SCONF.fetch_chars,
                        focus=st.row["question"])
    chosen = _select_docs(SCONF, rt.qwen, rt.model,
                          f"Question: {st.row['question']}\n\nAlready read:\n"
                          + _snippet_listing(st), fresh,
                          min(SCONF.link_select_k, room))
    _read_docs(SCONF, reader, chosen)
    st.n_fetches += sum(1 for d in chosen if st.store.get(d.doc_id).fetched)
    st.store.n_fetch_errors += reader.n_fetch_errors
    return True


def do_hop(rt: Runtime, st: QState) -> bool:
    todo = [i for i in range(len(st.subqs)) if i not in st.hop_answers]
    if not todo or not st.store.docs:
        return False
    i = todo[0]
    ctx = _snippet_listing(st) or "(no documents)"
    ans = rt.aux(HOP_ANSWER_SYSTEM, f"Sub-question: {st.subqs[i]}\n\nDocuments:\n{ctx}")
    if not ans or "unknown" in ans.lower()[:20]:
        st.hop_answers[i] = ""                        # tried; do not retry forever
        return True
    st.hop_answers[i] = ans
    if i + 1 < len(st.subqs) and len(st.queue) < MAX_QUEUE:
        query = rt.aux(HOP_REWRITE_SYSTEM,
                       f"Known fact: {st.subqs[i]} -> {ans}\n"
                       f"Sub-question: {st.subqs[i + 1]}", max_tokens=256)
        if query:
            st.queue.insert(0, query.splitlines()[0][:200])
    return True


def do_check(rt: Runtime, st: QState) -> bool:
    if not st.store.docs or st.n_checks >= 2:
        return False
    st.n_checks += 1
    st.draft = _draft(rt, st)
    visible = strip_think(st.draft)
    verdict = rt.aux(CHECK_SYSTEM,
                     f"Question: {st.row['question']}\n\nDraft answer: {visible[:800]}"
                     f"\n\nEvidence:\n{_snippet_listing(st)}")
    if verdict.upper().startswith("SUPPORTED"):
        st.verdict = "supported"
    else:
        st.verdict = "unsupported"
        m = re.search(r"MISSING\s*:\s*(.+)", verdict, re.I | re.S)
        if m and len(st.queue) < MAX_QUEUE:
            st.queue.insert(0, m.group(1).strip().splitlines()[0][:200])
    return True


DO = {"decompose": do_decompose, "variants": do_variants, "search": do_search,
      "read": do_read, "links": do_links, "hop": do_hop, "check": do_check}


def frames_condition(rng: random.Random) -> str:
    kind = rng.randrange(4)
    if kind == 0:
        name = rng.choice(list(NUMERIC))
        return f"{name}{rng.choice(['==', '>=', '<'])}{rng.randrange(13)}"
    if kind == 1:
        return rng.choice(["queue_empty", "has_subqs", "checked", "supported",
                           "unsupported", "draft_refused", "no_new_docs"])
    if kind == 2:
        return f"last:{rng.choice(list(DO))}"
    return f"ran:{rng.choice(list(DO))}"


def frames_action(rng: random.Random) -> str:
    return rng.choice(list(DO) + ["stop"])


def validate_program(prog: dict) -> None:
    for rule in prog["rules"] + [{"when": [], "do": prog["default"]}]:
        assert rule["do"] in DO or rule["do"] == "stop", rule["do"]


def run_program(prog: dict, rt: Runtime, row: dict) -> QState:
    st = QState(row)
    noops = 0
    for _ in range(MAX_ACTS):
        act = prog["default"]
        for rule in prog["rules"]:
            if all(cond(c, st) for c in rule["when"]):
                act = rule["do"]
                break
        if act == "stop":
            break
        if DO[act](rt, st):
            st.actions.append(act)
            noops = 0
        else:                     # infeasible in this state (empty queue, caps...)
            st.actions.append(f"noop:{act}")
            noops += 1
            if noops >= 3:
                break
    return st


# --- programs ----------------------------------------------------------------

PROGRAM_V0 = {
    "rules": [
        {"when": ["acts==0"], "do": "decompose"},
        {"when": ["acts==1"], "do": "variants"},
        {"when": ["queue>=1", "searches<8"], "do": "search"},   # drain the queue
        {"when": ["fetched==0", "docs>=1"], "do": "read"},
        {"when": ["checks==0", "docs>=1"], "do": "check"},
        {"when": ["unsupported", "queue>=1", "searches<11"], "do": "search"},
        {"when": ["unsupported", "unread>=1", "fetched<6"], "do": "read"},
        {"when": ["unsupported", "queue_empty", "checks<2"], "do": "check"},
    ],
    "default": "stop",
}


# --- scoring -----------------------------------------------------------------

def full_record(prog: dict, rt: Runtime, row: dict, tag: str) -> list[dict]:
    """One record in the exact shape of the search caches, so frames_headroom.py
    and the existing summaries can read program runs with no changes."""
    from eval_search_step import process_question  # reused only for its shape

    base = {"id": row.get("id"), "question": row["question"],
            "ground_truth": row["ground_truth"], "config": tag}
    try:
        st = run_program(prog, rt, row)
        store = st.store
        from doc_qa import gold_urls

        gold = {normalize_url(u) for u in gold_urls(row)}
        surfaced, found = store.search_keys(), store.found_keys()
        fetched_norm = store.fetched_keys()
        base.update(
            n_search=st.n_searches, n_fetch=len(store.pages()),
            num_surfaced=len(surfaced), num_fetched=len(fetched_norm),
            recall_surfaced=(len(gold & surfaced) / len(gold)) if gold else None,
            recall_fetched=(len(gold & fetched_norm) / len(gold)) if gold else None,
            precision_fetched=(len(gold & fetched_norm) / len(fetched_norm))
                              if fetched_norm else None,
            found_all=bool(gold) and gold <= surfaced,
            recall_found=(len(gold & found) / len(gold)) if gold else None,
            found_all_any=bool(gold) and gold <= found,
            n_docs=len(store.docs), n_links=store.n_links_followed(),
            n_fetch_errors=store.n_fetch_errors,
            subqueries=list(store.subqueries),
            queries=[q for q, _ in store.searches],
            actions=st.actions, program_verdict=st.verdict,
            surfaced_urls=sorted(surfaced), fetched_urls=sorted(fetched_norm),
            gold_urls=sorted(gold))
        return [score_evidence(base, "union", store.snippet_blocks(), store.pages(),
                               row["question"], row["ground_truth"],
                               rt.qwen, rt.model, rt.cfg, SCONF)]
    except Exception as exc:
        return [dict(base, evidence_policy="union", status="error", error=str(exc))]


def light_score(prog: dict, rt: Runtime, row: dict) -> bool:
    """Evolution fitness: one answer, one judge call, on the committed answer
    only (the reasoning trace is not graded)."""
    try:
        st = run_program(prog, rt, row)
        answer = strip_think(st.draft or _draft(rt, st))
        return rt.judged(row["question"], row["ground_truth"], answer)
    except Exception:
        return False


def _pmap(fn, items, workers, desc: str | None = None):
    def bar(gen):
        if desc and len(items) > 1:
            return tqdm(gen, total=len(items), desc=desc, unit="q", leave=False)
        return gen

    if workers <= 1 or len(items) <= 1:
        return list(bar(map(fn, items)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(bar(pool.map(fn, items)))


# --- preflight ---------------------------------------------------------------

def preflight(args) -> list[str]:
    problems = []
    try:
        requests.post(args.scrape_url, json={"url": "https://example.org/preflight"},
                      timeout=5)
    except Exception:
        if args.scrape_url == LOCAL_SCRAPE_URL:
            problems.append("local page server not running. Start it first:\n"
                            "      python3 scripts/wiki_backend.py &")
        else:
            problems.append(f"page backend unreachable: {args.scrape_url}")
    if args.scrape_url == SERPER_SCRAPE_URL:
        print("NOTE: page fetches go to LIVE Serper (paid; each URL cached, "
              "paid once ever).")
    if not os.getenv("SERPER_API_KEY"):
        problems.append("SERPER_API_KEY not set (web search). export it or put it in .env")
    if not os.getenv("OPENAI_API_KEY"):
        problems.append("OPENAI_API_KEY not set (judge). export it or put it in .env")
    try:
        requests.get(args.base_url.rstrip("/") + "/models", timeout=5)
    except Exception:
        problems.append(f"vLLM not reachable at {args.base_url} "
                        "(needs Qwen/Qwen3-14B serving)")
    return problems


# --- modes -------------------------------------------------------------------

def split_qids(rows: list[dict], seed: int) -> tuple[list[str], list[str]]:
    qids = sorted(r["id"] for r in rows)
    random.Random(seed).shuffle(qids)
    half = len(qids) // 2
    return qids[:half], qids[half:]


def mode_v0(args, rt: Runtime, rows: list[dict]) -> None:
    validate_program(PROGRAM_V0)
    tag = "program_v0"
    out_path = args.out_dir / f"search_{tag}_cache.jsonl"
    records: list[dict] = []
    lock = threading.Lock()
    with out_path.open("w") as fh:
        def work(row):
            recs = full_record(PROGRAM_V0, rt, row, tag)
            with lock:
                for r in recs:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
            return recs

        for recs in _pmap(work, rows, args.workers, desc=f"v0 ({len(rows)} q)"):
            records.extend(recs)
    summary = aggregate(records)
    print(json.dumps(summary, indent=2))
    print(f"\nbars: decompose_snip 46.3% downstream on the common subset")
    print(f"records -> {out_path}")
    print(f"cache: {rt.cache.hits} hits / {rt.cache.misses} misses; "
          f"judge calls {rt.judge_calls}")


def mode_evolve(args, rt: Runtime, rows: list[dict]) -> None:
    rng = random.Random(args.seed)
    by_id = {r["id"]: r for r in rows}
    train, _test = split_qids(rows, args.split_seed)
    order = list(train)
    rng.shuffle(order)
    stage1, stage2 = order[:args.stage1], order[:args.stage2]

    scores: dict[str, dict[int, bool]] = {}     # canon -> {qid-index outcomes}

    def acc(prog: dict, qids: list[str]) -> float:
        key = canon_prog(prog)
        got = scores.setdefault(key, {})
        todo = [q for q in qids if q not in got]
        outcomes = _pmap(lambda q: (q, light_score(prog, rt, by_id[q])),
                         todo, args.workers,
                         desc=f"scoring {len(todo)} q" if todo else None)
        got.update(dict(outcomes))
        return sum(got[q] for q in qids) / len(qids)

    population = [("v0", PROGRAM_V0)]
    print(f"train {len(train)} (race {len(stage1)}->{len(stage2)}), "
          f"{args.generations} generations, population {args.population}")
    for gen in range(1, args.generations + 1):
        children = []
        for name, parent in population[:args.population]:
            for _ in range(args.offspring):
                child = mutate_program(parent, rng, rand_cond=frames_condition,
                                       rand_act=frames_action, defaults=("stop",))
                try:
                    validate_program(child)
                except AssertionError:
                    continue
                if canon_prog(child) not in scores:
                    children.append((name, child))
        pool = population + children
        s1 = sorted(pool, key=lambda t: -acc(t[1], stage1))
        survivors = s1[:max(args.population, len(s1) // 2)]
        population = sorted(survivors, key=lambda t: -acc(t[1], stage2))[:args.population]
        best_name, best = population[0]
        print(f"gen {gen:>3}: best {acc(best, stage2):.1%} on {len(stage2)} train q "
              f"(lineage {best_name}); cache {rt.cache.hits}h/{rt.cache.misses}m; "
              f"judge calls {rt.judge_calls}")

    args.evolved_out.parent.mkdir(parents=True, exist_ok=True)
    args.evolved_out.write_text(json.dumps(
        [{"lineage": n, "train_acc": acc(p, stage2), "program": p}
         for n, p in population[:5]], indent=2))
    print(f"top programs -> {args.evolved_out}\n"
          f"score the winner on the untouched test half with:\n"
          f"  --test {args.evolved_out}")


def mode_test(args, rt: Runtime, rows: list[dict]) -> None:
    progs = json.loads(Path(args.test).read_text())
    prog = progs[0]["program"] if isinstance(progs, list) else progs
    validate_program(prog)
    _train, test = split_qids(rows, args.split_seed)
    by_id = {r["id"]: r for r in rows}
    test_rows = [by_id[q] for q in test]
    tag = "program_evolved"
    records = []
    for recs in _pmap(lambda row: full_record(prog, rt, row, tag), test_rows,
                      args.workers, desc=f"test half ({len(test_rows)} q)"):
        records.extend(recs)
    out_path = args.out_dir / f"search_{tag}_cache.jsonl"
    with out_path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(aggregate(records), indent=2))
    print(f"records -> {out_path} (test half, n={len(test_rows)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v0", action="store_true")
    ap.add_argument("--evolve", action="store_true")
    ap.add_argument("--test", type=Path, default=None,
                    help="Saved program json to score on the held-out test half.")
    ap.add_argument("--dataset", type=Path,
                    default=Path("datasets/frames_qwen_answerable.json"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--base-url", default="http://localhost:7472/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--scrape-url", default=SERPER_SCRAPE_URL,
                    help="Page backend. Default: live Serper (paid, cached by URL "
                         "so nothing is fetched twice). The local wiki server "
                         f"({LOCAL_SCRAPE_URL}) is free but holds only the gold "
                         "pages — see the docstring caveat.")
    ap.add_argument("--call-cache", type=Path,
                    default=Path("outputs/program_frames_callcache.jsonl"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--population", type=int, default=8)
    ap.add_argument("--offspring", type=int, default=2)
    ap.add_argument("--stage1", type=int, default=24,
                    help="Questions every candidate is scored on first.")
    ap.add_argument("--stage2", type=int, default=60,
                    help="Questions survivors are scored on.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0,
                    help="Train/test split seed. Never change between evolve and test.")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--evolved-out", type=Path,
                    default=Path("outputs/program_frames_evolved.json"))
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    if not (args.v0 or args.evolve or args.test):
        ap.print_help()
        return
    if not args.skip_preflight:
        problems = preflight(args)
        if problems:
            print("PREFLIGHT FAILED:")
            for p in problems:
                print(f"  - {p}")
            print("Fix the above (or --skip-preflight to override).")
            sys.exit(1)

    rows = json.loads(args.dataset.read_text())
    if args.limit is not None:
        rows = rows[: args.limit]
    rt = Runtime(args, CallCache(args.call_cache))
    if args.v0:
        mode_v0(args, rt, rows)
    if args.evolve:
        mode_evolve(args, rt, rows)
    if args.test:
        mode_test(args, rt, rows)


if __name__ == "__main__":
    main()

"""Tool adapters that write into an EvidenceStore, plus the link sources.

Two audiences read the same retrieved page and they want opposite things:

  the agent      is deciding what to look up next, inside a 32k conversation
                 where every tool result is appended verbatim. It needs enough
                 of the page to name the next thing to search for, and it needs
                 that to cost ~800 characters, not ~8,000.
  the answerer   is producing the final answer from a 98k-character evidence
                 budget that current runs fill to 13.6%. It should get the whole
                 page, uncompressed, because pack_truncate is deliberately
                 near-lossless so the metric measures retrieval and not a
                 lossy summarizer.

So the wrapper keeps the full page in the store and hands the agent a digest.
A digest that loses a detail costs the agent one bad next-query; it does not
cost the answerer the fact, because the answerer never reads the digest.

These wrappers are for the NEW plans. The existing plans deliberately keep their
own recording logic (see eval_search_step) so that what they produce is provably
unchanged; see test_evidence.test_legacy_view_identical.
"""

from __future__ import annotations

import re
import threading

import requests

from evidence import EvidenceStore, extract_links
from tools import cap_text

_DOC_ID = re.compile(r"^d\d+$")


# --- link sources ----------------------------------------------------------
# A link source answers one question: "what does this document point at?".
# Nothing about the method depends on which one is plugged in.

class DocumentLinks:
    """The default, and the only corpus-agnostic one: links come out of the
    markup of the page we already fetched. Requires a backend that returns
    markdown or HTML (FetchUrl(want_markup=True)); with a text-only backend it
    simply yields nothing, which is the honest answer for a corpus whose
    documents carry no pointers."""

    name = "document"

    def __call__(self, doc, limit: int = 400) -> list[tuple[str, str]]:
        return extract_links(doc.markup or "", doc.url, max_links=limit)


class WikiApiLinks:
    """Free link source for wikipedia.org URLs, so link-following can be measured
    without paying to re-scrape 266 questions' worth of pages.

    This is an evaluation backend, not part of the method. It asks MediaWiki for
    the page's rendered HTML and then runs the SAME generic extractor over it, so
    what the agent sees is what a markdown-capable scrape backend would have
    given it -- just without the bill. Non-Wikipedia URLs fall through to the
    document's own markup.
    """

    name = "wiki_api"
    API = "https://{host}/w/api.php"

    def __init__(self, timeout: int = 20, user_agent: str = "SearchOp-research/1.0") -> None:
        self.timeout = timeout
        self.headers = {"User-Agent": user_agent}
        self._cache: dict[str, list[tuple[str, str]]] = {}
        self._lock = threading.Lock()
        self._fallback = DocumentLinks()

    def __call__(self, doc, limit: int = 400) -> list[tuple[str, str]]:
        m = re.match(r"https?://([a-z0-9.-]*wikipedia\.org)/wiki/(.+)$", doc.url, re.I)
        if not m:
            return self._fallback(doc, limit)
        host, title = m.group(1).replace("en.m.", "en."), m.group(2)
        with self._lock:
            hit = self._cache.get(doc.key)
        if hit is None:
            try:
                resp = requests.get(
                    self.API.format(host=host),
                    params={"action": "parse", "page": title, "prop": "text",
                            "format": "json", "redirects": "1", "formatversion": "2"},
                    headers=self.headers, timeout=self.timeout)
                resp.raise_for_status()
                html = (resp.json().get("parse", {}) or {}).get("text", "") or ""
            except Exception:
                return self._fallback(doc, limit)
            hit = extract_links(html, f"https://{host}/wiki/{title}", max_links=1000)
            # Drop the maintenance namespaces; they are never evidence and they
            # would otherwise crowd out the article links.
            hit = [(a, u) for a, u in hit
                   if not re.search(r"/wiki/(File|Category|Help|Template|Special|Portal|"
                                    r"Wikipedia|Talk|User|Module):", u, re.I)]
            with self._lock:
                self._cache[doc.key] = hit
        return hit[:limit]


# --- tool adapters ---------------------------------------------------------

class StoreSearch:
    """search_info that records every hit into the store."""

    name = "search_info"

    def __init__(self, search, store: EvidenceStore, annotate: bool = True) -> None:
        self.search = search
        self.store = store
        # Re-render each hit with its document id so the agent can say "fetch d7"
        # instead of pasting a URL, and so a page it has already seen is visibly
        # the same page rather than a fresh-looking result.
        self.annotate = annotate

    def __call__(self, query: str) -> str:
        raw = self.search(query)
        docs = self.store.add_search(query, raw)
        if not self.annotate or not docs:
            return raw
        lines = []
        for doc in docs:
            mark = " [already read]" if doc.fetched else ""
            lines.append(f"[{doc.doc_id}] {doc.title}{mark}\n    URL: {doc.url}\n"
                         f"    {doc.best_snippet()}")
        return "\n".join(lines)

    def to_openai_schema(self) -> dict:
        return self.search.to_openai_schema()


class StoreFetch:
    """fetch_url that keeps the whole page and returns a short digest.

    Accepts a document id ('d7') or a URL. ``focus`` is the sub-question the
    digest should be written against -- compressing against the sub-question
    rather than the original question is what keeps the entity the next hop
    depends on from being summarized away.
    """

    name = "fetch_url"

    def __init__(self, fetch, store: EvidenceStore, *, digest_fn=None,
                 agent_chars: int | None = 8000, focus: str = "") -> None:
        self.fetch = fetch
        self.store = store
        self.digest_fn = digest_fn
        self.agent_chars = agent_chars
        self.focus = focus
        # Counted rather than raised: a run where half the fetches failed and a
        # run where they all worked should not look the same in the summary.
        self.n_fetch_errors = 0

    # The work is split in two so a batch of pages can be fetched concurrently
    # without the store's contents depending on which network call happened to
    # finish first. ``fetch_only`` does the slow part (network, then the
    # summarizer) and touches nothing shared; ``record`` does the writes and is
    # called one page at a time, in whatever order the caller chose. Sequential
    # callers just use __call__ and never see the difference.

    def fetch_only(self, ref: str) -> dict:
        """Network + digest for one reference. No writes -- safe to run in parallel."""
        ref = (ref or "").strip()
        known = self.store.get(ref) if _DOC_ID.match(ref) else None
        if _DOC_ID.match(ref) and known is None:
            return {"ref": ref, "url": "", "reply":
                    f"[no document called {ref!r}. Use a document id from a search result, or a URL.]"}
        url = known.url if known else ref

        # Already read once: hand back what we produced then rather than paying
        # for the page and the digest a second time.
        if known is not None and known.fetched:
            return {"ref": ref, "url": url,
                    "reply": known.digest or cap_text(known.raw_text or "", self.agent_chars)}

        # A scrape backend fails transiently all the time — read timeouts and 5xx
        # from scrape.serper.dev accounted for 46% of questions in the first run
        # of the reading plans, because the exception escaped and killed the whole
        # question. The old plans always guarded this (`try: fetch(url) except:
        # continue`, and run_with_tools catches tool exceptions); this one has to
        # as well. One retry first, since most of those failures were timeouts.
        for attempt in range(2):
            try:
                got = self.fetch.fetch_raw(url)
                break
            except Exception as exc:
                if attempt:
                    return {"ref": ref, "url": url, "failed": True,
                            "reply": f"[fetch_url failed for {url}: {exc}]"}
        if got["error"]:
            return {"ref": ref, "url": url, "failed": True, "reply": got["error"]}

        out = {"ref": ref, "url": got["url"], "text": got["text"], "markup": got["markup"]}
        if self.digest_fn is not None:
            try:
                out["digest"] = self.digest_fn(self.focus or self.store.question, got["text"])
            except Exception as exc:  # a summarizer hiccup must not lose the page
                out["digest_error"] = str(exc)
        return out

    def record(self, prepared: dict) -> str:
        """Write one fetch_only result into the store. Call from ONE thread."""
        if "text" not in prepared:
            if prepared.get("failed"):
                self.store.mark_fetch_failed(prepared["url"])
                self.n_fetch_errors += 1
            return prepared["reply"]
        doc = self.store.add_fetch(prepared["url"], prepared["text"],
                                   markup=prepared.get("markup"))
        if self.digest_fn is None:
            return cap_text(prepared["text"], self.agent_chars)
        if "digest" not in prepared:
            return (cap_text(prepared["text"], self.agent_chars)
                    + f"\n[digest unavailable: {prepared.get('digest_error')}]")
        doc.digest = prepared["digest"]
        return (f"[{doc.doc_id}] {doc.title or doc.url}\n{doc.digest}\n"
                f"[full page kept for the final answer; "
                f"list_links {doc.doc_id} to see what it points at]")

    def __call__(self, ref: str) -> str:
        return self.record(self.fetch_only(ref))

    def to_openai_schema(self) -> dict:
        schema = self.fetch.to_openai_schema()
        schema["function"]["description"] = (
            "Read a page. Input: a document id from a search result (like 'd7') "
            "or a full URL. Output: a short summary of what the page says about "
            "your question; the full text is kept automatically for the final "
            "answer, so you do not need to quote it."
        )
        return schema


class StoreLinks:
    """list_links: the pages an already-fetched document points at.

    This is the tool that makes multi-hop reachable. A question like 'what car
    did the nephew of X drive' names neither the nephew nor the car, so no query
    surfaces them -- but X's page links to both. Only the links actually handed
    to the agent are registered in the store, so a page is never counted as
    'found' because it happened to be one of four hundred links on something we
    read.
    """

    name = "list_links"

    def __init__(self, store: EvidenceStore, link_source=None, top_k: int = 40) -> None:
        self.store = store
        self.link_source = link_source or DocumentLinks()
        self.top_k = top_k

    def __call__(self, ref: str) -> str:
        doc = self.store.get((ref or "").strip())
        if doc is None:
            return f"[no document called {ref!r}. Fetch a page first, then list its links.]"
        if not doc.fetched:
            return f"[{doc.doc_id} has not been read yet. fetch_url {doc.doc_id} first.]"
        links = self.link_source(doc, limit=self.top_k)
        if not links:
            return (f"[{doc.doc_id} exposes no links. This backend returns plain text, "
                    f"so there are no pointers to follow.]")
        added = self.store.add_links(doc.doc_id, links)
        lines = [f"[{d.doc_id}] {d.title or d.url}" + (" [already read]" if d.fetched else "")
                 for d in added]
        return (f"{len(added)} pages linked from {doc.doc_id}:\n" + "\n".join(lines)
                + "\n[fetch_url any of these ids to read it]")

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "List the pages an already-read document links to. Input: a "
                    "document id like 'd3'. Use this when a page you read mentions "
                    "something you now need to look up -- following its link is more "
                    "reliable than guessing a search query for a name you do not know."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string",
                                  "description": "Document id of a page you already read."}
                    },
                    "required": ["query"],
                },
            },
        }

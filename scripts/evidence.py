"""One place to put everything a search plan finds, indexed by document.

Today each plan function accumulates two loose locals -- ``searches`` (a list of
``(query, raw result block)``) and ``fetched`` (a ``{url: page text}`` dict) --
and throws them away when it returns. That is fine for "search a bit, fetch a
page", but it cannot express the things the newer plans need:

  * the same page found by three queries is three copies buried inside three
    4,000-character result blobs, and the retrieval metrics count it three times;
  * there is nowhere to hang a short summary of a fetched page, so the agent's
    conversation gets the raw page or nothing;
  * there is nowhere to record that a document was reached by following a link
    out of another document rather than by a query, which is what link-following
    needs in order to know how far it has walked.

``EvidenceStore`` is those two locals merged into one object keyed by document,
with slots for the rest. It is deliberately two layers:

  event logs   ``searches`` and ``_fetches`` record what the tools actually did,
               in order, with first-write-wins per raw URL string. These ARE the
               old ``searches``/``fetched`` values -- ``legacy_view()`` hands
               them back in exactly the old shape, so an existing plan keeps
               producing byte-identical evidence.
  document     ``docs`` indexes those events by normalized URL and adds the new
  index        fields. Deriving it rather than storing it separately means there
               is still only one source of truth.

A store belongs to one question and is written from one thread (the tool loop is
sequential within a question), so nothing here locks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import unescape
from urllib.parse import urljoin, urlparse

from doc_qa import normalize_url

# The same expression eval_search_step.parse_search_urls uses. Hit boundaries in
# a result block are defined by THIS regex and nothing else, so the URLs
# parse_hits returns are exactly parse_search_urls' list, in the same order --
# the retrieval metrics depend on that and test_evidence.py asserts it.
_URL_RE = re.compile(r"URL:\s*(\S+)")
_HIT_MARKER = re.compile(r"^[ \t]*\[\d+\]", re.MULTILINE)


@dataclass
class Hit:
    """One result parsed out of a search_info block."""
    title: str
    url: str
    snippet: str


def _line_bounds(text: str, pos: int) -> tuple[int, int]:
    """(start, end) offsets of the line containing ``pos``; end excludes '\\n'."""
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    return start, (len(text) if end == -1 else end)


def parse_hits(results_text: str) -> list[Hit]:
    """Parse a search_info result block into hits.

    search_info renders each hit as three lines::

        [1] Some Title
            URL: https://example.org/page
            a snippet of text

    but the block can be truncated mid-hit, snippets can wrap onto extra lines,
    and a snippet can itself contain something that looks like a '[2]' marker.
    So hits are cut at ``URL:`` occurrences -- never at '[n]' markers -- which
    guarantees one hit per URL and keeps this parser's URL list identical to
    ``parse_search_urls``. The '[n]' markers are used only to decide where a
    snippet stops, where a mistake costs a few cosmetic characters and nothing
    that is measured.
    """
    text = results_text or ""
    matches = list(_URL_RE.finditer(text))
    hits: list[Hit] = []
    for i, m in enumerate(matches):
        url_start, url_end = _line_bounds(text, m.start())

        # Title: the last non-blank line above this URL line, after the previous
        # hit's URL line, with its '[n] ' prefix removed.
        prev_end = _line_bounds(text, matches[i - 1].start())[1] if i else 0
        above = [ln for ln in text[prev_end:url_start].splitlines() if ln.strip()]
        title = re.sub(r"^[ \t]*\[\d+\][ \t]*", "", above[-1]).strip() if above else ""

        # Snippet: from just after this URL line to whichever comes first -- the
        # next '[n]' marker, or the title line of the next hit, or the end.
        seg_start = min(url_end + 1, len(text))
        seg_end = len(text)
        marker = _HIT_MARKER.search(text, seg_start)
        if i + 1 < len(matches):
            nxt_line_start = _line_bounds(text, matches[i + 1].start())[0]
            # back up one line to leave the next hit's title out of this snippet
            seg_end = max(seg_start, text.rfind("\n", 0, max(nxt_line_start - 1, 0)) + 1)
            if marker and marker.start() < seg_end:
                seg_end = marker.start()
        elif marker:
            seg_end = marker.start()
        snippet = " ".join(text[seg_start:seg_end].split())
        hits.append(Hit(title=title, url=m.group(1), snippet=snippet))
    return hits


# --- link extraction -------------------------------------------------------
# Corpus-agnostic on purpose: links come out of whatever markup the fetch
# backend hands back for a document we already retrieved. Nothing here knows
# what site it is looking at.

_MD_LINK = re.compile(r"\[([^\]\n]{1,300})\]\(\s*(<?)([^\s<>)]+)\2\s*(?:\"[^\"]*\")?\)")
_HTML_LINK = re.compile(r"<a\b[^>]*?href\s*=\s*[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
                        re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_SKIP_SCHEMES = ("javascript:", "mailto:", "tel:", "data:", "#")


def extract_links(markup: str, base_url: str, max_links: int = 400) -> list[tuple[str, str]]:
    """(anchor text, absolute URL) pairs from a markdown or HTML document.

    Deduplicated by absolute URL, first anchor wins, source order preserved.
    Self-links and same-page fragments are dropped -- following them wastes a
    slot in the frontier and can loop. Capped at ``max_links`` because a single
    encyclopedia page can carry a thousand of them.
    """
    markup = markup or ""
    pairs: list[tuple[str, str]] = []
    for m in _MD_LINK.finditer(markup):
        pairs.append((m.group(1), m.group(3)))
    for m in _HTML_LINK.finditer(markup):
        pairs.append((_TAG.sub(" ", m.group(2)), m.group(1)))

    base_key = normalize_url(base_url) if base_url else ""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for anchor, href in pairs:
        # HTML-escaped hrefs are the norm, not the exception: a query string is
        # written href="...?a=1&amp;b=2", and leaving it escaped produces a URL
        # that 404s and a dedup key that never matches the same page found any
        # other way.
        href = unescape(href).strip()
        anchor = unescape(anchor)
        if not href or href.lower().startswith(_SKIP_SCHEMES):
            continue
        href = href.split("#", 1)[0]
        if not href:
            continue
        absolute = urljoin(base_url, href) if base_url else href
        if not urlparse(absolute).scheme.startswith("http"):
            continue
        key = normalize_url(absolute)
        if not key or key == base_key or key in seen:
            continue
        seen.add(key)
        out.append((" ".join(anchor.split())[:200], absolute))
        if len(out) >= max_links:
            break
    return out


# --- the store -------------------------------------------------------------

@dataclass
class Doc:
    """One document the plan has seen, however it was reached."""
    doc_id: str                                        # "d3"
    url: str                                           # as retrieved; fetchable
    key: str                                           # normalize_url(url); identity
    title: str = ""
    snippets: list[tuple[str, str]] = field(default_factory=list)   # (query, snippet)
    raw_text: str | None = None                        # full page, once fetched
    markup: str | None = None                          # markdown/HTML, if the backend gave it
    digest: str | None = None                          # short summary shown to the agent
    found_via: str = ""                                # "search:<query>" | "link:<doc_id>"
    hop: int = 0                                       # 0 = from a query, 1 = linked from a hop-0 page
    status: str = "surfaced"                           # surfaced | fetched | fetch_failed
    # Every raw URL string that was fetched and resolved to this document. Two
    # strings can normalize to one document (http/https, trailing slash), and
    # the legacy fetched-dict counted them separately, so they are kept in order
    # to reproduce that view exactly.
    fetch_urls: list[str] = field(default_factory=list)

    @property
    def fetched(self) -> bool:
        return self.raw_text is not None

    def best_snippet(self) -> str:
        """The longest snippet seen for this document (they vary by query)."""
        return max((s for _, s in self.snippets), key=len, default="")


class EvidenceStore:
    def __init__(self, question: str = "") -> None:
        self.question = question
        # What the plan decided to break the question into. Recorded because the
        # first comparison of the decompose plans could not be explained without
        # it: the sub-questions turned out to be the cause and there was no way
        # to look at them after the fact.
        self.subqueries: list[str] = []
        # Fetches that failed after retrying. A transient scrape outage and a
        # genuinely bad plan produce the same low fetch count, so this is what
        # tells them apart when reading a summary afterwards.
        self.n_fetch_errors = 0
        self.docs: dict[str, Doc] = {}          # doc_id -> Doc
        self._by_key: dict[str, str] = {}       # normalized url -> doc_id
        # --- event logs: what the tools actually did, in order ---
        self.searches: list[tuple[str, str]] = []     # (query, raw result block)
        self._fetches: list[tuple[str, str]] = []     # (raw url string, page text)
        self._fetch_pos: dict[str, int] = {}          # raw url -> index into _fetches
        self._seq = 0

    # -- writes ------------------------------------------------------------

    def _touch(self, url: str, *, title: str = "", found_via: str = "",
               hop: int | None = None) -> Doc:
        """Get or create the document for ``url``. Never creates a duplicate.

        ``hop`` is how the document was DISCOVERED, so only a discovery passes
        it: add_search passes 0, add_links passes parent+1, and add_fetch passes
        nothing at all. Reading a page is not a way of finding it. When add_fetch
        defaulted this to 0 it looked like a discovery at depth 0 and overwrote
        the provenance of any page reached by a link, so every link actually
        followed stopped counting as a link and n_links_followed() ended up
        reporting only the links that were ignored.
        """
        key = normalize_url(url)
        doc_id = self._by_key.get(key)
        if doc_id is not None:
            doc = self.docs[doc_id]
            if title and not doc.title:
                doc.title = title
            # A page first reached by a link and later returned by a query is a
            # search hit: the shallower, more direct provenance wins.
            if hop is not None and hop < doc.hop:
                doc.hop, doc.found_via = hop, found_via
            return doc
        self._seq += 1
        doc = Doc(doc_id=f"d{self._seq}", url=url, key=key, title=title,
                  found_via=found_via, hop=0 if hop is None else hop)
        self.docs[doc.doc_id] = doc
        self._by_key[key] = doc.doc_id
        return doc

    def add_search(self, query: str, results_text: str) -> list[Doc]:
        """Record a search and index its hits. Returns the documents it touched."""
        self.searches.append((query, results_text))
        out = []
        for hit in parse_hits(results_text):
            doc = self._touch(hit.url, title=hit.title, found_via=f"search:{query}", hop=0)
            if hit.snippet:
                doc.snippets.append((query, hit.snippet))
            out.append(doc)
        return out

    def add_fetch(self, url: str, text: str, markup: str | None = None,
                  overwrite: bool = False) -> Doc:
        """Record a successful fetch.

        ``overwrite`` picks which of the two accumulation styles the existing
        plans use, so each keeps producing exactly what it produced before:
        False is ``fetched.setdefault(url, text)`` (run_iterative,
        run_decompose, run_decompose_react) and True is ``fetched[url] = text``
        (run_single). They differ only when one raw URL string is fetched twice
        in a run and the two fetches return different text, but "only differs
        rarely" is not the same as "is identical".
        """
        doc = self._touch(url, found_via=f"fetch:{url}")
        pos = self._fetch_pos.get(url)
        if pos is None:
            self._fetch_pos[url] = len(self._fetches)
            self._fetches.append((url, text))
            doc.fetch_urls.append(url)
        elif overwrite:
            # dict[k] = v on an existing key keeps its position and replaces the
            # value; replacing in place reproduces that through legacy_view().
            self._fetches[pos] = (url, text)
        if doc.raw_text is None or overwrite:
            doc.raw_text = text
        if markup and (doc.markup is None or overwrite):
            doc.markup = markup
        doc.status = "fetched"
        return doc

    def has_fetched_raw(self, url: str) -> bool:
        """Was this exact URL string already fetched? (run_decompose skips those.)"""
        return url in self._fetch_pos

    def mark_fetch_failed(self, url: str) -> Doc:
        doc = self._touch(url, found_via=f"fetch:{url}")
        if not doc.fetched:
            doc.status = "fetch_failed"
        return doc

    def add_links(self, from_doc_id: str, links: list[tuple[str, str]]) -> list[Doc]:
        """Index the outgoing links of an already-fetched document.

        The anchor text is kept as a snippet: it is the only description of the
        target we have before fetching it, and it is what the agent picks from.
        """
        parent = self.docs.get(from_doc_id)
        hop = (parent.hop + 1) if parent else 1
        out = []
        for anchor, url in links:
            doc = self._touch(url, title=anchor, found_via=f"link:{from_doc_id}", hop=hop)
            if anchor and not doc.snippets:
                doc.snippets.append((f"link from {from_doc_id}", anchor))
            out.append(doc)
        return out

    # -- reads -------------------------------------------------------------

    def get(self, ref: str) -> Doc | None:
        """Look a document up by doc_id ('d3') or by URL, whichever was given."""
        ref = (ref or "").strip()
        if ref in self.docs:
            return self.docs[ref]
        doc_id = self._by_key.get(normalize_url(ref))
        return self.docs.get(doc_id) if doc_id else None

    def legacy_view(self) -> tuple[list[tuple[str, str]], dict[str, str]]:
        """The ``(searches, fetched)`` pair the plan functions used to return.

        Reconstructed from the event logs, so it is identical to what the old
        code built -- same order, same duplicate raw URL strings, same
        first-write-wins. This is what keeps the existing configs reproducible.
        """
        return self.searches, dict(self._fetches)

    def snippet_blocks(self) -> list[tuple[str, str]]:
        """Legacy snippet evidence: one entry per SEARCH, not per document."""
        return [(f"search: {q}", text) for q, text in self.searches]

    def pages(self) -> list[tuple[str, str]]:
        """Legacy page evidence: one entry per fetched raw URL string, in order."""
        return list(self._fetches)

    def search_keys(self) -> set[str]:
        """Normalized URLs that a QUERY surfaced. Excludes link discoveries, so
        recall_surfaced stays comparable with every run recorded so far."""
        return {normalize_url(u) for _, text in self.searches
                for u in _URL_RE.findall(text)}

    def found_keys(self) -> set[str]:
        """Normalized URLs reached by any means -- queries or links."""
        return set(self._by_key)

    def fetched_keys(self) -> set[str]:
        return {normalize_url(u) for u, _ in self._fetches}

    def fetched_docs(self) -> list[Doc]:
        return [d for d in self.docs.values() if d.fetched]

    def n_links_followed(self) -> int:
        return sum(1 for d in self.docs.values() if d.found_via.startswith("link:"))

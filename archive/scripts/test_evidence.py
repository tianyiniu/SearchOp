"""Tests for the evidence store. No network, no model, no cost.

The important one is test_urls_match_legacy: the retrieval metrics are defined by
``eval_search_step.parse_search_urls`` (a bare ``URL:\\s*(\\S+)`` findall) and every
number recorded so far was computed with it. If the store's parser disagreed even
once, recall_surfaced would silently shift and old and new runs would stop being
comparable. So it is checked against adversarial blocks built by the same code
path search_info uses.

    python3 scripts/test_evidence.py
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evidence import EvidenceStore, extract_links, parse_hits
from eval_search_step import parse_search_urls

FAILURES: list[str] = []


def check(cond, label: str, detail: str = "") -> None:
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}  {detail}")
        FAILURES.append(label)


def render_block(hits, max_chars: int | None = None) -> str:
    """Build a result block exactly the way tools.SearchInfo.__call__ does."""
    lines = [f"[{i}] {title}\n    URL: {url}\n    {snippet}"
             for i, (title, url, snippet) in enumerate(hits, start=1)]
    text = "\n".join(lines)
    if max_chars is not None and len(text) > max_chars:
        dropped = len(text) - max_chars
        text = (text[:max_chars]
                + f"\n\n[search_info truncated: {dropped} more characters omitted. "
                  f"Use fetch_url on a result URL above to read a full page.]")
    return text


# --- 1. the parser agrees with the legacy URL extractor --------------------

def test_urls_match_legacy() -> None:
    print("\ntest_urls_match_legacy")
    rnd = random.Random(20260804)
    # Nasty pieces that a real Serper title or snippet can genuinely contain.
    nasty = [
        "Normal Title", "Title with [2] in it", "URL: not-really-a-url",
        "multi\nline snippet", "trailing spaces   ", "", "[3] leading marker",
        "see URL: http://decoy.example/x for more", "unicode — em dash and ()",
        "brackets [a](b) markdown-ish",
    ]
    cases = 0
    for trial in range(300):
        n = rnd.randint(1, 6)
        hits = []
        for i in range(n):
            hits.append((rnd.choice(nasty),
                         f"https://en.wikipedia.org/wiki/Page_{trial}_{i}",
                         rnd.choice(nasty)))
        cap = rnd.choice([None, None, 60, 150, 400, 4000])
        block = render_block(hits, cap)
        mine = [h.url for h in parse_hits(block)]
        legacy = parse_search_urls(block)
        cases += 1
        if mine != legacy:
            check(False, f"trial {trial}", f"\n    mine={mine}\n    legacy={legacy}\n    block={block!r}")
            return
    check(True, f"URL lists identical on {cases} adversarial blocks")


def test_parser_fields() -> None:
    print("\ntest_parser_fields")
    block = render_block([
        ("Father of Asahd", "https://en.wikipedia.org/wiki/Father_of_Asahd",
         "the eleventh studio album by DJ Khaled"),
        ("DJ Khaled", "https://en.wikipedia.org/wiki/DJ_Khaled",
         "attended Dr. Phillips High School"),
    ])
    hits = parse_hits(block)
    check(len(hits) == 2, "two hits", str(len(hits)))
    check(hits[0].title == "Father of Asahd", "first title", hits[0].title)
    check(hits[0].url == "https://en.wikipedia.org/wiki/Father_of_Asahd", "first url", hits[0].url)
    check(hits[0].snippet == "the eleventh studio album by DJ Khaled", "first snippet", repr(hits[0].snippet))
    check(hits[1].title == "DJ Khaled", "second title", hits[1].title)
    check(hits[1].snippet == "attended Dr. Phillips High School", "second snippet", repr(hits[1].snippet))
    check(parse_hits("") == [], "empty block -> no hits")
    check(parse_hits("No results found for: foo") == [], "no-results block -> no hits")


# --- 2. the store reproduces the old (searches, fetched) exactly -----------

def test_legacy_view_identical() -> None:
    """Replay a plan's tool calls through the store and compare against the
    accumulation the old plan functions did inline."""
    print("\ntest_legacy_view_identical")
    calls = [
        ("search", "khaled album", render_block([
            ("Father of Asahd", "https://en.wikipedia.org/wiki/Father_of_Asahd", "album"),
            ("DJ Khaled", "https://en.wikipedia.org/wiki/DJ_Khaled", "producer")])),
        ("fetch", "https://en.wikipedia.org/wiki/DJ_Khaled", "PAGE-A"),
        ("search", "khaled high school", render_block([
            ("DJ Khaled", "https://en.wikipedia.org/wiki/DJ_Khaled", "Dr. Phillips"),
            ("Mark Ruiz", "https://en.wikipedia.org/wiki/Mark_Ruiz", "diver")])),
        # same page again, and a second string that normalizes to the same page
        ("fetch", "https://en.wikipedia.org/wiki/DJ_Khaled", "PAGE-A-SECOND-TRY"),
        ("fetch", "https://en.wikipedia.org/wiki/DJ_Khaled/", "PAGE-A-TRAILING-SLASH"),
        ("fetch", "https://en.wikipedia.org/wiki/Mark_Ruiz", "PAGE-B"),
    ]

    # --- old behaviour, copied from run_iterative/run_decompose_react ---
    old_searches: list[tuple[str, str]] = []
    old_fetched: dict[str, str] = {}
    for kind, a, b in calls:
        if kind == "search":
            old_searches.append((a, b))
        else:
            old_fetched.setdefault(a, b)

    # --- through the store ---
    store = EvidenceStore("q")
    for kind, a, b in calls:
        if kind == "search":
            store.add_search(a, b)
        else:
            store.add_fetch(a, b)
    new_searches, new_fetched = store.legacy_view()

    check(new_searches == old_searches, "searches identical")
    check(new_fetched == old_fetched, "fetched dict identical",
          f"\n    old={list(old_fetched)}\n    new={list(new_fetched)}")
    check(list(new_fetched.items()) == list(old_fetched.items()), "fetched ORDER identical")
    check(store.pages() == list(old_fetched.items()), "pages() == legacy page evidence")
    check(store.snippet_blocks() == [(f"search: {q}", t) for q, t in old_searches],
          "snippet_blocks() == legacy snippet evidence")

    # legacy metric definitions, recomputed both ways
    from doc_qa import normalize_url
    old_surfaced = {normalize_url(u) for s in old_searches for u in parse_search_urls(s[1])}
    old_fetched_norm = {normalize_url(u) for u in old_fetched}
    check(store.search_keys() == old_surfaced, "search_keys() == legacy surfaced set")
    check(store.fetched_keys() == old_fetched_norm, "fetched_keys() == legacy fetched set")

    # --- and the new, deduplicated view ---
    check(len(store.docs) == 3, "three distinct documents", str(len(store.docs)))
    khaled = store.get("https://en.wikipedia.org/wiki/DJ_Khaled")
    check(khaled is not None and len(khaled.snippets) == 2,
          "one doc carries both of its snippets",
          str(khaled and len(khaled.snippets)))
    check(khaled is not None and khaled.raw_text == "PAGE-A",
          "first fetched text wins", str(khaled and khaled.raw_text))
    check(khaled is not None and khaled.fetch_urls ==
          ["https://en.wikipedia.org/wiki/DJ_Khaled", "https://en.wikipedia.org/wiki/DJ_Khaled/"],
          "both raw URL strings recorded on the one doc", str(khaled and khaled.fetch_urls))
    check(store.get("d1") is store.get("https://en.wikipedia.org/wiki/Father_of_Asahd"),
          "lookup by doc_id and by url agree")


def test_both_accumulation_styles() -> None:
    """run_single writes ``fetched[url] = text`` (last wins) while the other
    three write ``fetched.setdefault(url, text)`` (first wins). Both have to come
    back out of the store unchanged, including the value AND its position."""
    print("\ntest_both_accumulation_styles")
    fetches = [("https://a.org/1", "FIRST"), ("https://b.org/2", "B"),
               ("https://a.org/1", "SECOND"), ("https://c.org/3", "C"),
               ("https://a.org/1", "THIRD")]
    for overwrite, label in ((False, "setdefault"), (True, "assignment")):
        old: dict[str, str] = {}
        for url, text in fetches:
            if overwrite:
                old[url] = text
            else:
                old.setdefault(url, text)
        store = EvidenceStore("q")
        for url, text in fetches:
            store.add_fetch(url, text, overwrite=overwrite)
        _, new = store.legacy_view()
        check(list(new.items()) == list(old.items()),
              f"{label}: dict and order identical",
              f"\n    old={list(old.items())}\n    new={list(new.items())}")
        doc = store.get("https://a.org/1")
        check(doc.raw_text == old["https://a.org/1"],
              f"{label}: doc text matches the dict", str(doc.raw_text))
        check(store.has_fetched_raw("https://a.org/1"), f"{label}: has_fetched_raw")
        check(not store.has_fetched_raw("https://zzz.org/9"), f"{label}: has_fetched_raw negative")


# --- 3. link extraction ----------------------------------------------------

def test_extract_links() -> None:
    print("\ntest_extract_links")
    base = "https://en.wikipedia.org/wiki/Pierluigi_Martini"
    md = (
        "Pierluigi Martini drove for [Minardi](/wiki/Minardi) and the "
        "[Minardi M194](https://en.wikipedia.org/wiki/Minardi_M194) in 1994. "
        "His uncle [Giancarlo Martini](/wiki/Giancarlo_Martini) raced too. "
        "See [himself](/wiki/Pierluigi_Martini) and [top](#section) and "
        "[mail](mailto:x@y.z) and [js](javascript:void(0))."
    )
    links = extract_links(md, base)
    urls = [u for _, u in links]
    check("https://en.wikipedia.org/wiki/Minardi_M194" in urls, "absolute link kept")
    check("https://en.wikipedia.org/wiki/Minardi" in urls, "relative link made absolute")
    check("https://en.wikipedia.org/wiki/Giancarlo_Martini" in urls, "second relative link")
    check(not any("Pierluigi_Martini" in u for u in urls), "self-link dropped", str(urls))
    check(not any(u.startswith(("mailto:", "javascript:")) for u in urls), "non-http dropped")
    check(len(urls) == len(set(urls)), "no duplicates")
    anchors = dict((u, a) for a, u in links)
    check(anchors["https://en.wikipedia.org/wiki/Minardi_M194"] == "Minardi M194", "anchor text kept")

    html = ('<p><a href="/wiki/Ferrari_312T">Ferrari 312T</a> and '
            '<a href="https://example.org/x?a=1#frag"><b>Ex</b> ternal</a></p>')
    hlinks = extract_links(html, base)
    hurls = [u for _, u in hlinks]
    check("https://en.wikipedia.org/wiki/Ferrari_312T" in hurls, "html relative link")
    check("https://example.org/x?a=1" in hurls, "html absolute link, fragment stripped", str(hurls))
    check(dict((u, a) for a, u in hlinks)["https://example.org/x?a=1"] == "Ex ternal",
          "html anchor tags stripped")

    # Real pages HTML-escape their query strings; leaving them escaped yields a
    # URL that 404s and a dedup key that never matches the same page found any
    # other way. (Observed on live MediaWiki output.)
    esc = ('<a href="https://www.google.com/search?as_eq=wikipedia&amp;q=%22X%22">a &amp; b</a>'
           '<a href="/wiki/Y?p=1&amp;q=2">Y</a>')
    elinks = extract_links(esc, base)
    eurls = [u for _, u in elinks]
    check("&amp;" not in " ".join(eurls), "html entities decoded in urls", str(eurls))
    check("https://www.google.com/search?as_eq=wikipedia&q=%22X%22" in eurls,
          "escaped query string resolves correctly", str(eurls))
    check(dict((u, a) for a, u in elinks)["https://www.google.com/search?as_eq=wikipedia&q=%22X%22"]
          == "a & b", "html entities decoded in anchor text")

    check(extract_links("", base) == [], "empty markup -> no links")
    check(extract_links("plain text with no links at all", base) == [], "plain text -> no links")
    capped = extract_links(" ".join(f"[a{i}](/wiki/P{i})" for i in range(50)), base, max_links=10)
    check(len(capped) == 10, "max_links respected", str(len(capped)))


def test_links_into_store() -> None:
    print("\ntest_links_into_store")
    store = EvidenceStore("q")
    store.add_search("pierluigi martini", render_block([
        ("Pierluigi Martini", "https://en.wikipedia.org/wiki/Pierluigi_Martini", "F1 driver")]))
    d = store.get("https://en.wikipedia.org/wiki/Pierluigi_Martini")
    store.add_fetch(d.url, "page text")
    store.add_links(d.doc_id, [("Minardi M194", "https://en.wikipedia.org/wiki/Minardi_M194")])
    m = store.get("https://en.wikipedia.org/wiki/Minardi_M194")
    check(m is not None, "linked doc added")
    check(m.hop == 1, "linked doc is hop 1", str(m and m.hop))
    check(m.found_via == f"link:{d.doc_id}", "provenance records the parent", str(m and m.found_via))
    check(m.key not in store.search_keys(), "link discovery does NOT count as surfaced")
    check(m.key in store.found_keys(), "link discovery does count as found")
    check(store.n_links_followed() == 1, "link count")

    # READING a linked page must not erase the fact that a link is how we got
    # there. add_fetch used to default hop=0, which looked like a depth-0
    # discovery and overwrote the provenance, so every link actually followed
    # stopped being counted and n_links_followed() reported only the ignored ones.
    store.add_fetch(m.url, "the minardi m194 page")
    check(m.hop == 1, "reading a linked page keeps its hop", str(m.hop))
    check(m.found_via == f"link:{d.doc_id}",
          "reading a linked page keeps its link provenance", m.found_via)
    check(store.n_links_followed() == 1,
          "a link that was followed still counts as a link", str(store.n_links_followed()))
    store.add_links(m.doc_id, [("Minardi", "https://en.wikipedia.org/wiki/Minardi")])
    check(store.get("https://en.wikipedia.org/wiki/Minardi").hop == 2,
          "a second round is depth 2, not depth 1",
          str(store.get("https://en.wikipedia.org/wiki/Minardi").hop))

    # a page first seen via a link, later returned by a query, becomes a search hit
    store.add_search("minardi m194", render_block([
        ("Minardi M194", "https://en.wikipedia.org/wiki/Minardi_M194", "1994 car")]))
    check(m.hop == 0, "query provenance overrides link provenance", str(m.hop))
    check(m.key in store.search_keys(), "now counts as surfaced")
    check(len(store.docs) == 3, "still no duplicate doc", str(len(store.docs)))


if __name__ == "__main__":
    test_urls_match_legacy()
    test_parser_fields()
    test_legacy_view_identical()
    test_both_accumulation_styles()
    test_extract_links()
    test_links_into_store()
    print("\n" + ("FAILED: " + ", ".join(FAILURES) if FAILURES else "all checks passed"))
    sys.exit(1 if FAILURES else 0)

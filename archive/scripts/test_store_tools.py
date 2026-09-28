"""Tests for the store-aware tool wrappers. No network, no model, no cost.

The first block is a regression test, not a feature test: FetchUrl.__call__ was
refactored into fetch_raw + cap_text so a caller can get the full page and the
markup, and every existing config runs through that method. So the new
implementation is compared against a verbatim copy of the old one over a matrix
of backend responses, and the outgoing request body is checked to be unchanged.

    python3 scripts/test_store_tools.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("SERPER_API_KEY", "test-key-not-used")

import tools
from tools import LOCAL_SCRAPE_URL, SERPER_SCRAPE_URL, FetchUrl, _html_to_text, _looks_like_url
from evidence import EvidenceStore
from store_tools import StoreFetch, StoreLinks, StoreSearch

FAILURES: list[str] = []
REQUESTS: list[dict] = []


def check(cond, label: str, detail: str = "") -> None:
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}  {detail}")
        FAILURES.append(label)


class FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload, self.status_code = payload, status

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def install_fake_backend(payload: dict, status: int = 200):
    """Replace tools.requests.post and record what was sent."""
    def fake_post(url, json=None, headers=None, timeout=None):
        REQUESTS.append({"url": url, "json": json, "headers": headers})
        return FakeResponse(payload, status)
    tools.requests.post = fake_post


# --- 1. FetchUrl.__call__ is byte-identical to the pre-refactor version ----

def old_fetch_call(self, url: str) -> str:
    """Verbatim copy of FetchUrl.__call__ as it was before fetch_raw existed."""
    url = url.strip()
    if not _looks_like_url(url):
        return (f"[fetch_url expects a single URL, got {url!r}. "
                f"Use search_info to find a source first.]")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    resp = tools.requests.post(
        self.scrape_url,
        json={"url": url},
        headers=None if self._is_local else tools._serper_headers(),
        timeout=self.timeout,
    )
    if not (self._is_local and resp.status_code == 404):
        resp.raise_for_status()
    data = resp.json()
    text = (data.get("text") or data.get("markdown") or "").strip()
    if not text and data.get("html"):
        text = _html_to_text(str(data["html"]))
    if not text:
        return f"No readable content at: {url}"
    if self.max_chars is not None and len(text) > self.max_chars:
        dropped = len(text) - self.max_chars
        text = (text[: self.max_chars]
                + f"\n\n[fetch_url truncated: {dropped} more characters omitted.]")
    return text


def test_fetch_unchanged() -> None:
    print("\ntest_fetch_unchanged")
    payloads = [
        {"text": "hello world " * 500},
        {"text": "   spaced out   "},
        {"markdown": "# md only\n[a](/b)"},
        {"html": "<p>html <b>only</b></p><script>x</script>"},
        {"text": "", "markdown": "md fallback"},
        {"text": "", "markdown": "", "html": "<p>html fallback</p>"},
        {"text": ""},
        {},
    ]
    urls = ["https://en.wikipedia.org/wiki/X", "en.wikipedia.org/wiki/X",
            "  https://example.org/a  ", "not a url at all", "http://x.io/p"]
    caps = [None, 10, 50, 100000]
    statuses = [(200, LOCAL_SCRAPE_URL), (404, LOCAL_SCRAPE_URL), (200, SERPER_SCRAPE_URL)]

    mismatches = 0
    total = 0
    for payload in payloads:
        for status, scrape in statuses:
            for cap in caps:
                for url in urls:
                    install_fake_backend(payload, status)
                    f = FetchUrl(scrape_url=scrape, max_chars=cap)
                    try:
                        new = f(url)
                    except Exception as exc:
                        new = f"RAISED {type(exc).__name__}"
                    try:
                        old = old_fetch_call(f, url)
                    except Exception as exc:
                        old = f"RAISED {type(exc).__name__}"
                    total += 1
                    if new != old:
                        mismatches += 1
                        if mismatches == 1:
                            print(f"    first mismatch: payload={payload} status={status} "
                                  f"cap={cap} url={url!r}\n      new={new!r}\n      old={old!r}")
    check(mismatches == 0, f"identical output on {total} combinations", f"{mismatches} mismatches")


def test_request_body_unchanged() -> None:
    print("\ntest_request_body_unchanged")
    REQUESTS.clear()
    install_fake_backend({"text": "page"})
    FetchUrl(scrape_url=LOCAL_SCRAPE_URL)("https://a.org/b")
    check(REQUESTS[-1]["json"] == {"url": "https://a.org/b"},
          "default request body unchanged", str(REQUESTS[-1]["json"]))
    FetchUrl(scrape_url=LOCAL_SCRAPE_URL, want_markup=True)("https://a.org/b")
    check(REQUESTS[-1]["json"] == {"url": "https://a.org/b", "includeMarkdown": True},
          "markup request adds exactly one field", str(REQUESTS[-1]["json"]))


def test_fetch_raw_gives_full_text_and_markup() -> None:
    print("\ntest_fetch_raw_gives_full_text_and_markup")
    install_fake_backend({"text": "x" * 5000, "markdown": "[Link](/wiki/Target)"})
    f = FetchUrl(scrape_url=LOCAL_SCRAPE_URL, max_chars=100, want_markup=True)
    got = f.fetch_raw("https://a.org/b")
    check(len(got["text"]) == 5000, "fetch_raw text is uncapped", str(len(got["text"])))
    check(got["markup"] == "[Link](/wiki/Target)", "markup returned", got["markup"])
    check(len(f("https://a.org/b")) < 200, "__call__ still caps")


# --- 2. the wrappers -------------------------------------------------------

def block(*hits: tuple[str, str, str]) -> str:
    return "\n".join(f"[{i}] {t}\n    URL: {u}\n    {s}"
                     for i, (t, u, s) in enumerate(hits, start=1))


class FakeSearch:
    name = "search_info"

    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping, self.calls = mapping, []

    def __call__(self, query: str) -> str:
        self.calls.append(query)
        return self.mapping.get(query, "No results found for: " + query)

    def to_openai_schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": "d",
                "parameters": {"type": "object", "properties": {}, "required": []}}}


def test_store_search() -> None:
    print("\ntest_store_search")
    raw = block(("Pierluigi Martini", "https://en.wikipedia.org/wiki/Pierluigi_Martini", "F1 driver"),
                ("Enzo Ferrari", "https://en.wikipedia.org/wiki/Enzo_Ferrari", "founder"))
    store = EvidenceStore("q")
    plain = StoreSearch(FakeSearch({"martini": raw}), store, annotate=False)
    check(plain("martini") == raw, "annotate=False returns the block untouched")
    check(len(store.docs) == 2, "hits indexed", str(len(store.docs)))

    store2 = EvidenceStore("q")
    ann = StoreSearch(FakeSearch({"martini": raw}), store2, annotate=True)
    out = ann("martini")
    check("[d1]" in out and "[d2]" in out, "annotated output carries doc ids", out)
    check("URL: https://en.wikipedia.org/wiki/Pierluigi_Martini" in out, "URL still present")
    check("F1 driver" in out, "snippet still present")
    check(ann("nothing") == "No results found for: nothing", "empty result passes through")
    check(ann.to_openai_schema()["function"]["name"] == "search_info", "schema delegated")


def test_store_fetch() -> None:
    print("\ntest_store_fetch")
    raw = block(("Pierluigi Martini", "https://en.wikipedia.org/wiki/Pierluigi_Martini", "F1 driver"))
    store = EvidenceStore("what car did the nephew drive")
    search = StoreSearch(FakeSearch({"m": raw}), store)
    search("m")

    # note the trailing space: fetch_raw strips the payload, exactly as the
    # pre-refactor code did, so the stored length is 14999 and not 15000
    page = "PAGE " * 3000
    full_len = len(page.strip())
    install_fake_backend({"text": page,
                          "markdown": "drove the [Minardi M194](/wiki/Minardi_M194)"})
    fetcher = FetchUrl(scrape_url=LOCAL_SCRAPE_URL, want_markup=True)

    # no digest -> capped raw text, full page still stored
    sf = StoreFetch(fetcher, store, digest_fn=None, agent_chars=200)
    out = sf("d1")
    check(len(out) < 400, "agent sees a capped page", str(len(out)))
    doc = store.get("d1")
    check(doc.raw_text is not None and len(doc.raw_text) == full_len,
          "store keeps the FULL page", str(doc.raw_text and len(doc.raw_text)))
    check(doc.status == "fetched", "status updated")

    # digest mode
    store2 = EvidenceStore("what car did the nephew drive")
    StoreSearch(FakeSearch({"m": raw}), store2)("m")
    seen = {}

    def digest_fn(focus, text):
        seen["focus"], seen["len"] = focus, len(text)
        return "He drove the Minardi M194 in 1994."

    sf2 = StoreFetch(fetcher, store2, digest_fn=digest_fn, focus="which car in 1994")
    out2 = sf2("d1")
    check("Minardi M194" in out2, "digest returned to the agent")
    check(seen["focus"] == "which car in 1994", "digest written against the sub-question", seen["focus"])
    check(seen["len"] == full_len, "digest sees the full page", str(seen["len"]))
    check(store2.get("d1").raw_text is not None and len(store2.get("d1").raw_text) == full_len,
          "full page still stored in digest mode")

    # second fetch of the same doc is served from the store, no backend call
    before = len(REQUESTS)
    out3 = sf2("d1")
    check(len(REQUESTS) == before, "re-fetch does not hit the backend")
    check("Minardi M194" in out3, "re-fetch returns the stored digest")

    # unknown doc id, and a fetch error
    check(sf2("d99").startswith("[no document called"), "unknown doc id reported")
    install_fake_backend({"text": ""})
    err = sf2("https://en.wikipedia.org/wiki/Missing")
    check(err.startswith("No readable content"), "fetch error passed through", err)
    check(store2.get("https://en.wikipedia.org/wiki/Missing").status == "fetch_failed",
          "failed fetch recorded as failed")

    # a digest_fn that blows up must not lose the page
    install_fake_backend({"text": "REAL PAGE TEXT"})
    boom = StoreFetch(fetcher, store2, digest_fn=lambda f, t: 1 / 0)
    out4 = boom("https://en.wikipedia.org/wiki/Boom")
    check("REAL PAGE TEXT" in out4, "digest failure falls back to the page", out4[:80])
    check(store2.get("https://en.wikipedia.org/wiki/Boom").raw_text == "REAL PAGE TEXT",
          "page stored despite digest failure")


def test_fetch_failure_does_not_escape() -> None:
    """A transient scrape failure killed 46-67% of questions in the first run of
    the reading plans, because the exception escaped StoreFetch and aborted the
    whole question. It must be caught, retried once, and counted."""
    print("\ntest_fetch_failure_does_not_escape")
    raw = block(("Pierluigi Martini", "https://en.wikipedia.org/wiki/Pierluigi_Martini", "F1"))
    store = EvidenceStore("q")
    StoreSearch(FakeSearch({"m": raw}), store)("m")

    calls = {"n": 0}

    def always_times_out(url, json=None, headers=None, timeout=None):
        calls["n"] += 1
        raise TimeoutError("Read timed out. (read timeout=30)")

    tools.requests.post = always_times_out
    sf = StoreFetch(FetchUrl(scrape_url=LOCAL_SCRAPE_URL), store)
    out = sf("d1")
    check(isinstance(out, str) and "failed" in out, "returns a message instead of raising", out[:60])
    check(calls["n"] == 2, "retried exactly once before giving up", str(calls["n"]))
    check(sf.n_fetch_errors == 1, "failure counted", str(sf.n_fetch_errors))
    check(store.get("d1").status == "fetch_failed", "recorded as a failed fetch")

    # a failure on the first attempt that succeeds on the retry must be kept
    state = {"n": 0}

    def fails_once(url, json=None, headers=None, timeout=None):
        state["n"] += 1
        if state["n"] == 1:
            raise TimeoutError("transient")
        REQUESTS.append({"url": url, "json": json, "headers": headers})
        return FakeResponse({"text": "RECOVERED PAGE"})

    tools.requests.post = fails_once
    store2 = EvidenceStore("q")
    StoreSearch(FakeSearch({"m": raw}), store2)("m")
    sf2 = StoreFetch(FetchUrl(scrape_url=LOCAL_SCRAPE_URL), store2)
    got = sf2("d1")
    check("RECOVERED PAGE" in got, "retry succeeds and the page comes back", got[:60])
    check(sf2.n_fetch_errors == 0, "a recovered fetch is not counted as an error")
    check(store2.get("d1").raw_text == "RECOVERED PAGE", "page stored after retry")


def test_fetch_only_and_record_split() -> None:
    """fetch_only does the slow work with no writes so a batch can run in
    parallel; record does the writes in whatever order the caller picked. The
    two together must equal what __call__ did in one step."""
    print("\ntest_fetch_only_and_record_split")
    raw = block(("A", "https://en.wikipedia.org/wiki/A", "sa"),
                ("B", "https://en.wikipedia.org/wiki/B", "sb"))
    store = EvidenceStore("q")
    StoreSearch(FakeSearch({"m": raw}), store)("m")
    install_fake_backend({"text": "PAGE BODY"})
    sf = StoreFetch(FetchUrl(scrape_url=LOCAL_SCRAPE_URL), store, agent_chars=None)

    prep = sf.fetch_only("d1")
    check(store.get("d1").raw_text is None, "fetch_only writes nothing to the store")
    check(prep["text"] == "PAGE BODY", "fetch_only returns the page", str(prep)[:70])
    out = sf.record(prep)
    check(store.get("d1").raw_text == "PAGE BODY", "record writes it")
    check("PAGE BODY" in out, "record returns the reply")

    # writing in a chosen order, not completion order
    store2 = EvidenceStore("q")
    StoreSearch(FakeSearch({"m": raw}), store2)("m")
    sf2 = StoreFetch(FetchUrl(scrape_url=LOCAL_SCRAPE_URL), store2, agent_chars=None)
    preps = [sf2.fetch_only("d2"), sf2.fetch_only("d1")]
    for p in preps:
        sf2.record(p)
    check([u for u, _ in store2.pages()] ==
          ["https://en.wikipedia.org/wiki/B", "https://en.wikipedia.org/wiki/A"],
          "pages land in the order they were recorded", str([u for u, _ in store2.pages()]))

    # a failure survives the split, and is still counted
    def boom(url, json=None, headers=None, timeout=None):
        raise TimeoutError("down")
    tools.requests.post = boom
    store3 = EvidenceStore("q")
    StoreSearch(FakeSearch({"m": raw}), store3)("m")
    sf3 = StoreFetch(FetchUrl(scrape_url=LOCAL_SCRAPE_URL), store3)
    r = sf3.record(sf3.fetch_only("d1"))
    check("failed" in r, "failure reply preserved", r[:50])
    check(sf3.n_fetch_errors == 1, "failure counted at record time", str(sf3.n_fetch_errors))
    check(store3.get("d1").status == "fetch_failed", "failure recorded on the doc")


def test_store_links() -> None:
    print("\ntest_store_links")
    raw = block(("Pierluigi Martini", "https://en.wikipedia.org/wiki/Pierluigi_Martini", "F1 driver"))
    store = EvidenceStore("q")
    StoreSearch(FakeSearch({"m": raw}), store)("m")
    install_fake_backend({"text": "page text",
                          "markdown": "drove the [Minardi M194](/wiki/Minardi_M194) and "
                                      "his uncle [Giancarlo Martini](/wiki/Giancarlo_Martini)"})
    fetcher = FetchUrl(scrape_url=LOCAL_SCRAPE_URL, want_markup=True)
    links = StoreLinks(store)

    check(links("d1").startswith("[d1 has not been read yet"), "links before fetch refused")
    StoreFetch(fetcher, store, agent_chars=None)("d1")
    out = links("d1")
    check("Minardi M194" in out and "Giancarlo Martini" in out, "links listed", out)
    m = store.get("https://en.wikipedia.org/wiki/Minardi_M194")
    check(m is not None and m.hop == 1, "linked doc registered at hop 1")
    check(m.key in store.found_keys(), "linked doc counts as found")
    check(m.key not in store.search_keys(), "linked doc does NOT count as surfaced")
    check(links("d99").startswith("[no document called"), "unknown ref reported")

    # a text-only backend has no links to give, and says so
    store3 = EvidenceStore("q")
    StoreSearch(FakeSearch({"m": raw}), store3)("m")
    install_fake_backend({"text": "plain text only, no markup"})
    StoreFetch(FetchUrl(scrape_url=LOCAL_SCRAPE_URL), store3, agent_chars=None)("d1")
    check("exposes no links" in StoreLinks(store3)("d1"), "text-only backend reports no links")
    check(len(store3.docs) == 1, "no phantom documents added", str(len(store3.docs)))


if __name__ == "__main__":
    test_fetch_unchanged()
    test_request_body_unchanged()
    test_fetch_raw_gives_full_text_and_markup()
    test_store_search()
    test_store_fetch()
    test_fetch_failure_does_not_escape()
    test_fetch_only_and_record_split()
    test_store_links()
    print("\n" + ("FAILED: " + ", ".join(FAILURES) if FAILURES else "all checks passed"))
    sys.exit(1 if FAILURES else 0)

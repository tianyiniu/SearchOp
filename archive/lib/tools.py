"""Three tools for an LLM agent: web search, page fetch, and code execution.

- search_info  : Google-style web search via the Serper API -> top hits.
- fetch_url    : download + extract the readable text of one URL (Serper scrape).
- code_compute : run a Python snippet in a subprocess and return its stdout.

Each tool is a plain callable ``tool(query: str) -> str``. ``to_openai_schema()``
returns the JSON the OpenAI / vLLM tool-calling API expects, and ``build_tools()``
returns a ``{name: tool}`` registry plus the schema list — ready to hand to the
loop in ``tool_calling.py``.

search_info and fetch_url need a Serper API key (https://serper.dev):
    export SERPER_API_KEY=...
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import threading
from pathlib import Path

import requests

SERPER_SEARCH_URL = "https://google.serper.dev/search"
SERPER_SCRAPE_URL = "https://scrape.serper.dev"

# Local cache server (scripts/wiki_backend.py) that mocks the scrape API from the
# on-disk Wikipedia cache. Pass this as fetch_url's scrape_url to read pages from
# disk for free instead of calling Serper. See build_tools(scrape_url=...).
LOCAL_SCRAPE_URL = "http://127.0.0.1:5000/"


# Ceiling on how many Serper requests are in flight at once, across every thread
# in the process. Parallelism is now nested -- questions run concurrently, and
# inside a question the sub-question agents and the page fetches run concurrently
# too -- so without a shared ceiling the load on Serper is workers x inner
# workers, and read timeouts from scrape.serper.dev already aborted 46-67% of
# questions at the old, lower concurrency. Raise it with SERPER_MAX_CONCURRENCY
# if the backend turns out to tolerate more.
_SERPER_GATE = threading.BoundedSemaphore(int(os.getenv("SERPER_MAX_CONCURRENCY", "16")))


def _serper_headers() -> dict[str, str]:
    key = os.environ.get("SERPER_API_KEY")
    if not key:
        raise RuntimeError("SERPER_API_KEY is not set (needed by search_info / fetch_url)")
    return {"X-API-KEY": key, "Content-Type": "application/json"}


# ---------------------------------------------------------------------------
# search_info
# ---------------------------------------------------------------------------

class SearchInfo:
    name = "search_info"

    def __init__(self, top_n: int = 5, max_chars: int = 4000, timeout: int = 30) -> None:
        self.top_n = top_n
        self.max_chars = max_chars
        self.timeout = timeout

    def __call__(self, query: str) -> str:
        query = " ".join(query.split()).strip()
        if not query:
            return "search_info received an empty query."
        with _SERPER_GATE:
            resp = requests.post(
                SERPER_SEARCH_URL,
                json={"q": query, "num": self.top_n},
                headers=_serper_headers(),
                timeout=self.timeout,
            )
        resp.raise_for_status()
        hits = resp.json().get("organic", [])[: self.top_n]
        if not hits:
            return f"No results found for: {query}"
        lines = []
        for i, hit in enumerate(hits, start=1):
            title = (hit.get("title") or "").strip()
            url = (hit.get("link") or "").strip()
            snippet = (hit.get("snippet") or "").strip()
            lines.append(f"[{i}] {title}\n    URL: {url}\n    {snippet}")
        text = "\n".join(lines)
        if len(text) > self.max_chars:
            dropped = len(text) - self.max_chars
            text = (text[: self.max_chars]
                    + f"\n\n[search_info truncated: {dropped} more characters omitted. "
                      f"Use fetch_url on a result URL above to read a full page.]")
        return text

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Run a web search. Input: free-text keywords (NOT a URL). "
                    "Output: a numbered list of the top results, each with a "
                    "title, URL, and short snippet. Follow up with fetch_url on "
                    "a result URL to read its full text."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search keywords."}
                    },
                    "required": ["query"],
                },
            },
        }


# ---------------------------------------------------------------------------
# fetch_url
# ---------------------------------------------------------------------------

def _looks_like_url(s: str) -> bool:
    if s.startswith(("http://", "https://")):
        return True
    host = s.split("/", 1)[0]
    return "." in host and " " not in s


def _html_to_text(html: str) -> str:
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<[^>]+>", " ", html)
    html = (html.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            .replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " "))
    return re.sub(r"\s+", " ", html).strip()


def cap_text(text: str, max_chars: int | None) -> str:
    """Truncate to ``max_chars`` with the note the agent is used to seeing.

    One implementation, because two callers cap the same page for two different
    audiences: FetchUrl caps what goes straight back into a conversation, and the
    store-aware wrapper caps what it shows the agent while keeping the full page.
    A drift between them would silently change what old configs produce.
    """
    if max_chars is None or len(text) <= max_chars:
        return text
    dropped = len(text) - max_chars
    return text[:max_chars] + f"\n\n[fetch_url truncated: {dropped} more characters omitted.]"


class FetchUrl:
    name = "fetch_url"

    def __init__(self, timeout: int = 30, scrape_url: str = SERPER_SCRAPE_URL,
                 max_chars: int | None = None, want_markup: bool = False) -> None:
        self.timeout = timeout
        # Which scrape backend to hit: SERPER_SCRAPE_URL (real, paid) or
        # LOCAL_SCRAPE_URL (the local on-disk cache server). Both speak the same
        # POST {"url": ...} -> {"text": ...} protocol.
        self.scrape_url = scrape_url
        # Cap on the returned text. None (the default) returns the whole page,
        # which is what the document-context builders want. An agent loop wants a
        # cap: a fetched page goes back into the conversation verbatim, and cached
        # Wikipedia pages run to a median of ~25k chars and a p90 of ~117k, so a
        # single uncapped fetch can overflow a 32k-token context on its own.
        self.max_chars = max_chars
        # Ask the backend for markdown as well as text. Only link-following needs
        # it (a page's outgoing links are the pointers to the pages you have not
        # found yet) and asking changes the request body, so it is off by default
        # and every existing caller keeps sending the exact same request.
        self.want_markup = want_markup

    @property
    def _is_local(self) -> bool:
        return self.scrape_url != SERPER_SCRAPE_URL

    def fetch_raw(self, url: str) -> dict:
        """The backend response, normalized to {url, text, markup, error}.

        ``text`` is the page's plain text and is UNCAPPED -- ``max_chars`` is
        applied by ``__call__``, which is what goes back into a conversation,
        while a caller that stores the page wants all of it. ``markup`` is the
        markdown or HTML the backend supplied, or "" if it supplied none; link
        extraction needs it, and a text-only backend (the local cache server)
        simply has no links to give. ``error`` carries the exact string
        ``__call__`` has always returned when there is nothing readable.
        """
        url = url.strip()
        if not _looks_like_url(url):
            return {"url": url, "text": "", "markup": "",
                    "error": (f"[fetch_url expects a single URL, got {url!r}. "
                              f"Use search_info to find a source first.]")}
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        body: dict = {"url": url}
        if self.want_markup:
            body["includeMarkdown"] = True
        # The local cache server is an in-memory dict lookup and wants no
        # throttling; only real Serper goes through the gate.
        if self._is_local:
            resp = requests.post(self.scrape_url, json=body, headers=None,
                                 timeout=self.timeout)
        else:
            with _SERPER_GATE:
                resp = requests.post(self.scrape_url, json=body,
                                     headers=_serper_headers(), timeout=self.timeout)
        # A cache miss on the local server is a 404 with an empty body; treat it
        # as "no content" (like an empty Serper result) rather than an error.
        if not (self._is_local and resp.status_code == 404):
            resp.raise_for_status()
        data = resp.json()
        markdown = (data.get("markdown") or "").strip()
        html = str(data.get("html") or "")
        text = (data.get("text") or markdown or "").strip()
        if not text and html:
            text = _html_to_text(html)
        return {"url": url, "text": text, "markup": markdown or html,
                "error": "" if text else f"No readable content at: {url}"}

    def __call__(self, url: str) -> str:
        got = self.fetch_raw(url)
        return got["error"] or cap_text(got["text"], self.max_chars)

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Download the full text of a single, explicit URL (starting "
                    "with http:// or https://). Use it on a URL returned by "
                    "search_info, not on a search query."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "One full URL."}
                    },
                    "required": ["query"],
                },
            },
        }


# ---------------------------------------------------------------------------
# code_compute
# ---------------------------------------------------------------------------

def _is_expression(code: str) -> bool:
    import ast
    try:
        ast.parse(code, mode="eval")
        return True
    except SyntaxError:
        return False


class CodeCompute:
    name = "code_compute"

    def __init__(self, timeout: int = 30, max_chars: int = 5000) -> None:
        self.timeout = timeout
        self.max_chars = max_chars

    def __call__(self, query: str) -> str:
        code = query.strip()
        if not code:
            return "code_compute received empty input."
        # A bare expression ("2**10") is auto-printed so the model sees a result;
        # code that already prints (or is a multi-statement script) runs as-is.
        if _is_expression(code) and not code.lstrip().startswith("print"):
            code = f"print({code})"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
            path = Path(f.name)
        try:
            proc = subprocess.run(
                ["python3", str(path)],
                capture_output=True, text=True, timeout=self.timeout,
            )
            out = proc.stdout.strip()
            if proc.returncode != 0 and proc.stderr.strip():
                out = (out + "\n" if out else "") + "Error: " + proc.stderr.strip()
            return (out or "(no output)")[: self.max_chars]
        except subprocess.TimeoutExpired:
            return f"code_compute timed out after {self.timeout}s"
        finally:
            path.unlink(missing_ok=True)

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Execute Python in a subprocess and return its stdout. Pass "
                    "a bare expression for quick math (e.g. '2**10') or a full "
                    "script using print(). Standard library only."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string",
                                  "description": "A Python expression or script."}
                    },
                    "required": ["query"],
                },
            },
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def build_tools(enabled: list[str] | None = None,
                scrape_url: str = SERPER_SCRAPE_URL) -> tuple[dict, list]:
    """Return ({name: tool}, [openai_schema, ...]) for the enabled tools.

    ``enabled`` defaults to all three. Pass a subset (e.g. ["code_compute"]) to
    restrict what the model may call. ``scrape_url`` chooses fetch_url's backend:
    SERPER_SCRAPE_URL (default, real/paid) or LOCAL_SCRAPE_URL (local cache).
    """
    all_tools = [SearchInfo(), FetchUrl(scrape_url=scrape_url), CodeCompute()]
    if enabled is not None:
        all_tools = [t for t in all_tools if t.name in enabled]
    registry = {t.name: t for t in all_tools}
    schemas = [t.to_openai_schema() for t in all_tools]
    return registry, schemas

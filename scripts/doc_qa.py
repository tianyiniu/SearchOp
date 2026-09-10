"""Shared document-QA helpers + a cache for summarized document context.

The expensive, NON-deterministic step in these experiments is compressing a
question's long ground-truth documents into a context that fits Qwen. If every
consumer re-summarizes, they each get a slightly different context and results
stop being comparable. So we summarize ONCE and cache the packed context, keyed
by the question id + the summarizer configuration; everyone else reads it back.

This module owns:
  - SummarizerConfig : the single source of truth for summarization parameters
                       (its .signature() is part of the cache key, so changing a
                       parameter automatically invalidates stale entries).
  - gold_urls        : a question's ground-truth URLs (splitting comma-packed
                       wiki_links entries, mirroring cache_web_links.split_links).
  - fetch_urls       : fetch each URL's text, dropping failures.
  - pack_documents   : pack (url, text) docs into one structured string,
                       summarizing each when the combined text exceeds the budget.
  - the context cache: context_cache_path / load_cached_context /
                       build_and_cache_context.

Consumers: cache_doc_summary.py (writes the cache), and later eval_debate_step.py
(reads it) and eval_search_step.py (reuses the packing procedure on its own docs).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from iterative_summarization import estimate_tokens, iterative_summarize
from llm_judge import call_openai, judge_answer

# Bump when iterative_summarization's prompts change, so cached contexts built
# with the old prompt are no longer reused (the signature changes).
# v2: compression moved to GPT by default + post-compression verification.
PROMPT_VERSION = "v2"


# --- summarizer configuration (single source of truth) ---------------------

@dataclass(frozen=True)
class SummarizerConfig:
    model: str = "Qwen/Qwen3-14B"          # the consumer model (owns the context window)
    summarizer_kind: str = "gpt"           # "gpt" (Responses API) or "qwen" (vLLM chat)
    summarizer_model: str = "gpt-5.4"      # model used when summarizer_kind == "gpt"
    context_window: int = 32768
    summary_tokens: int = 2048
    doc_budget_fraction: float = 0.75      # summarize once combined docs exceed this fraction
    temperature: float = 0.3
    # After compressing, ask the summarizer model to answer the question from the
    # compressed context alone. If it cannot, the compression dropped the facts
    # that matter, so retry once at summary_tokens * verify_retry_multiplier.
    verify_compression: bool = True
    verify_retry_multiplier: int = 2
    prompt_version: str = PROMPT_VERSION

    def signature(self) -> str:
        """Short stable hash of the config — part of the cache key. Uses hashlib
        (not builtin hash(), which is salted per process) so it agrees across runs."""
        blob = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def make_summarize_call(cfg: SummarizerConfig, summary_tokens: int | None = None):
    """The backend `iterative_summarize` should use for this config.

    Returns a ``summarize_call(system, user) -> str`` for the "gpt" kind, or None
    for "qwen" (which drives the vLLM client directly). Deriving this from the
    config — rather than each caller wiring its own lambda — is what makes GPT
    compression the uniform default across every consumer of the cache.
    """
    if cfg.summarizer_kind != "gpt":
        return None
    budget = summary_tokens or cfg.summary_tokens
    return lambda system, user: call_openai(
        system, user, model=cfg.summarizer_model, max_output_tokens=budget * 2)


# --- ground-truth URL extraction -------------------------------------------

def split_links(entry: str) -> list[str]:
    """A wiki_links element is usually one URL, but a few pack several comma-joined
    URLs. Split only on a comma followed by http(s):// so genuine path commas
    ('Tulsa,_Oklahoma') stay intact. Mirrors cache_web_links.split_links."""
    parts = re.split(r"\s*,\s*(?=https?://)", entry.strip())
    return [p.strip().rstrip(",").strip() for p in parts if p.strip().rstrip(",").strip()]


def gold_urls(row: dict) -> list[str]:
    """A question's ground-truth URLs: every wiki_links entry, comma-split and
    de-duplicated in first-seen order."""
    seen: dict[str, None] = {}
    for entry in row.get("wiki_links", []):
        for url in split_links(entry):
            seen.setdefault(url, None)
    return list(seen)


def normalize_url(url: str) -> str:
    """Canonical form for comparing URLs (retrieval-overlap metric): scheme- and
    fragment-insensitive, mobile Wikipedia folded to desktop, no trailing slash.
    Page-title case is preserved (Wikipedia titles are case-sensitive)."""
    url = url.strip().split("#", 1)[0]
    url = re.sub(r"^https?://", "", url)
    url = url.replace("en.m.wikipedia.org", "en.wikipedia.org")
    return url.rstrip("/")


# --- fetch + pack ----------------------------------------------------------

def fetch_urls(urls: list[str], fetch) -> list[tuple[str, str]]:
    """Fetch each URL's full text. Returns (url, text) for the ones that worked;
    a failed or empty fetch is dropped."""
    docs: list[tuple[str, str]] = []
    for url in urls:
        try:
            text = fetch(url)
        except Exception:
            continue
        if not text or text.startswith(("[fetch_url", "No readable content")):
            continue
        docs.append((url, text))
    return docs


def pack_documents(client, question: str, docs: list[tuple[str, str]],
                   cfg: SummarizerConfig, summarize_call=None,
                   summary_tokens: int | None = None) -> tuple[str, dict]:
    """Pack (url, text) docs into one structured string. If the combined text
    exceeds the document budget, each document is compressed with query-aware
    iterative summarization. Returns (doc_context, meta).

    ``summary_tokens`` overrides the config's per-document budget (used by the
    verification retry). ``summarize_call`` overrides the backend; when omitted it
    is derived from the config, so "gpt" configs compress with GPT automatically.
    """
    per_doc_tokens = summary_tokens or cfg.summary_tokens
    if summarize_call is None:
        summarize_call = make_summarize_call(cfg, per_doc_tokens)
    total_tokens = sum(estimate_tokens(text) for _, text in docs)
    doc_budget = int(cfg.context_window * cfg.doc_budget_fraction)
    summarized = total_tokens > doc_budget
    if summarized:
        docs = [(url, iterative_summarize(
                    client, cfg.model, question, text,
                    context_window=cfg.context_window, summary_tokens=per_doc_tokens,
                    temperature=cfg.temperature, summarize_call=summarize_call))
                for url, text in docs]
    blocks = [f"[Document {i}] {url}\n{text}" for i, (url, text) in enumerate(docs, 1)]
    meta = {"num_docs": len(docs), "doc_tokens": total_tokens, "summarized": summarized,
            "summary_tokens": per_doc_tokens if summarized else None}
    return "\n\n".join(blocks), meta


# --- answer a question from a document context -----------------------------

WITH_DOC_SYSTEM = (
    "Answer the question using ONLY the documents provided; the answer is contained "
    "in them. End with one line: 'ANSWER: <answer>'."
)


VERIFY_SYSTEM = (
    "Answer the question using ONLY the documents provided. If the documents do not "
    "contain enough information to answer, reply exactly: INSUFFICIENT. Otherwise end "
    "with one line: 'ANSWER: <answer>'."
)


def verify_context(question: str, ground_truth: str, doc_context: str,
                   cfg: SummarizerConfig) -> tuple[bool, str]:
    """Check that compression preserved the facts needed to answer.

    Asks the summarizer model to answer the question from the COMPRESSED context
    alone and grades it against the ground truth. A failure means the summariser
    dropped the load-bearing facts — the context is unusable for measuring whether
    a weaker model can aggregate, because there is nothing left to aggregate.

    Returns (passed, answer).
    """
    answer = call_openai(
        VERIFY_SYSTEM, f"{doc_context}\n\nQUESTION: {question}", model=cfg.summarizer_model)
    return judge_answer(question, ground_truth, answer), answer


def compress_and_verify(client, question: str, ground_truth: str,
                        docs: list[tuple[str, str]], cfg: SummarizerConfig,
                        summarize_call=None) -> tuple[str, dict]:
    """Pack the documents, then verify the compression did not lose the answer.

    Verification only applies when compression actually happened — an uncompressed
    context is the documents verbatim, so there is nothing to have lost. On a failed
    check the documents are re-compressed once at a larger per-document budget and
    re-verified; the retry budget is capped so N docs still fit the document budget.

    Returns (doc_context, meta) where meta['verified'] is True/False/None
    (None = not applicable, i.e. no compression occurred).
    """
    doc_context, meta = pack_documents(client, question, docs, cfg, summarize_call)
    if not (cfg.verify_compression and meta["summarized"] and ground_truth):
        return doc_context, {**meta, "verified": None, "verify_attempts": 0}

    passed, answer = verify_context(question, ground_truth, doc_context, cfg)
    meta = {**meta, "verified": passed, "verify_answer": answer, "verify_attempts": 1}
    if passed:
        return doc_context, meta

    # Retry at a larger per-document budget, capped so the packed context still
    # fits: N docs * per_doc_tokens must stay inside the document budget.
    doc_budget = int(cfg.context_window * cfg.doc_budget_fraction)
    cap = doc_budget // max(1, len(docs))
    retry_tokens = min(cfg.summary_tokens * cfg.verify_retry_multiplier, cap)
    if retry_tokens <= cfg.summary_tokens:
        return doc_context, {**meta, "verify_note": "no headroom to retry"}

    doc_context, retry_meta = pack_documents(
        client, question, docs, cfg, summarize_call, summary_tokens=retry_tokens)
    passed, answer = verify_context(question, ground_truth, doc_context, cfg)
    return doc_context, {**retry_meta, "verified": passed, "verify_answer": answer,
                         "verify_attempts": 2}


def answer_with_docs(client, model: str, question: str, doc_context: str,
                     max_tokens: int = 2048, temperature: float = 0.0) -> str:
    """The single 'simple Qwen call given documents' — shared by the dataset
    filter (Step 2), the debate step's single_pass shape, and the search step's
    downstream metric, so all three answer identically."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": WITH_DOC_SYSTEM},
                  {"role": "user", "content": f"{doc_context}\n\nQUESTION: {question}"}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


# --- context cache ---------------------------------------------------------

def context_cache_path(question_id: str, cfg: SummarizerConfig, cache_dir) -> Path:
    """Cache file for one question's packed context under a given summarizer config.
    The config signature in the name lets different configs coexist and makes a
    config change a guaranteed miss."""
    return Path(cache_dir) / f"{question_id}__{cfg.signature()}.json"


def load_cached_context(question_id: str, cfg: SummarizerConfig, cache_dir) -> dict | None:
    """Return the cached context record for (question, config), or None on a miss."""
    path = context_cache_path(question_id, cfg, cache_dir)
    return json.loads(path.read_text()) if path.exists() else None


def _atomic_write(path: Path, record: dict) -> None:
    """Write JSON via a temp file + rename, so concurrent/simultaneous runs never
    observe a half-written cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(record, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def build_and_cache_context(client, row: dict, fetch, cfg: SummarizerConfig,
                            cache_dir, summarize_call=None, overwrite: bool = False) -> dict:
    """Fetch a question's gold documents, pack+summarize them, and cache the result.

    Returns the cache record. Status is one of:
      'cached'        - served from disk (may itself be a 'verify_failed' record)
      'ok'            - context built, and verified if it was compressed
      'fetch_failed'  - no document could be fetched
      'verify_failed' - compressed, but the answer did not survive compression even
                        after the retry; callers should exclude the question
    """
    qid = row.get("id")
    path = context_cache_path(qid, cfg, cache_dir)
    if path.exists() and not overwrite:
        rec = json.loads(path.read_text())
        rec["cache_status"] = rec.get("status")
        rec["status"] = "cached"
        return rec

    urls = gold_urls(row)
    docs = fetch_urls(urls, fetch)
    base = {"id": qid, "question": row.get("question"), "urls": urls,
            "summarizer": asdict(cfg)}
    if not docs:
        rec = {**base, "doc_context": "", "num_docs": 0, "summarized": False,
               "status": "fetch_failed"}
    else:
        doc_context, meta = compress_and_verify(
            client, row["question"], row.get("ground_truth", ""), docs, cfg, summarize_call)
        status = "verify_failed" if meta.get("verified") is False else "ok"
        rec = {**base, "doc_context": doc_context, **meta, "status": status}
    _atomic_write(path, rec)
    return rec


def usable_context(rec: dict) -> str:
    """The context from a cache record, or '' if the record is not usable.

    Collapses the 'cached' indirection: a record read back from disk carries its
    original status in cache_status, so a cached verify_failed stays excluded.
    """
    status = rec.get("cache_status") if rec.get("status") == "cached" else rec.get("status")
    return rec.get("doc_context", "") if status == "ok" else ""

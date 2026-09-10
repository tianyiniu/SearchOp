"""Query-aware iterative (refine) summarization for long documents.

A long web page can blow past a small model's context window (Qwen3-14B holds
~32k tokens, a long Wikipedia article is far more). Instead of truncating, this
walks the document in section-aligned chunks — each up to about half the context
window — and at every step folds the next chunk into a running summary capped at
``summary_tokens``. The model is always shown the query the summary is meant to
serve, so it keeps query-relevant facts (names, dates, numbers) and drops the
rest. After the whole document is consumed (typically 3-4 passes for a long
article) we are left with a short, query-focused digest that fits in context.

    from iterative_summarization import iterative_summarize
    digest = iterative_summarize(client, "Qwen/Qwen3-14B", question, long_page_text)
"""

from __future__ import annotations

import re
from typing import Any

# Rough token estimate (good enough for budgeting; avoids a tokenizer dependency).
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


SUMMARIZE_SYSTEM = """You are condensing a long document so another system can later answer a specific question. You are given the QUESTION, the SUMMARY SO FAR of earlier parts of the document, and the NEXT PART of the document.

Produce an updated summary that:
- keeps every fact that could help answer the question (names, dates, numbers, relationships, definitions),
- folds in the NEXT PART without dropping relevant facts already in the summary so far,
- drops material irrelevant to the question,
- is dense notes, not flowing prose,
- stays under about {summary_tokens} tokens.

Output only the updated summary, nothing else."""

SUMMARIZE_USER = """QUESTION: {query}

SUMMARY SO FAR:
{prior}

NEXT PART OF THE DOCUMENT:
{chunk}

Write the updated summary now."""


def _chunk(text: str, budget_chars: int) -> list[str]:
    """Split ``text`` into chunks of at most ``budget_chars``, breaking at natural
    paragraph boundaries; hard-split any single paragraph that is itself too big."""
    pieces: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= budget_chars:
            pieces.append(para)
        else:  # a single huge paragraph -> slice it
            pieces.extend(para[i: i + budget_chars] for i in range(0, len(para), budget_chars))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > budget_chars:
            chunks.append(current)
            current = piece
        else:
            current = piece if not current else f"{current}\n\n{piece}"
    if current:
        chunks.append(current)
    return chunks


def iterative_summarize(
    client: Any,
    model: str,
    query: str,
    text: str,
    context_window: int = 32768,
    summary_tokens: int = 2048,
    temperature: float = 0.3,
    summarize_call=None,
) -> str:
    """Condense ``text`` into a <=``summary_tokens`` summary focused on ``query``.

    Feeds the document in chunks of about half the context window, refining one
    running summary across the passes. By default each chunk is summarized via the
    OpenAI-compatible chat endpoint (the same vLLM client the agent uses); pass
    ``summarize_call(system_prompt, user_prompt) -> str`` to use a different
    backend (e.g. GPT-5.4's Responses API), in which case ``client``/``model`` are
    unused."""
    budget_chars = (context_window // 2) * CHARS_PER_TOKEN
    chunks = _chunk(text, budget_chars)
    system = SUMMARIZE_SYSTEM.format(summary_tokens=summary_tokens)

    summary = ""
    for chunk in chunks:
        user = SUMMARIZE_USER.format(query=query, prior=summary or "(nothing yet)", chunk=chunk)
        if summarize_call is not None:
            summary = summarize_call(system, user).strip()
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=temperature,
                max_tokens=summary_tokens,
            )
            summary = (response.choices[0].message.content or "").strip()
    return summary

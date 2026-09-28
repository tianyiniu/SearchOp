"""LLM judge for answer correctness, using OpenAI's API.

Given a question, the gold answer, and a model's predicted answer, ask the judge
model to decide CORRECT or INCORRECT. The prompt is a trimmed version of OpenAI's
SimpleQA grader — correctness ONLY (no not-attempted category, no trace/quality
scoring).

    pip install openai
    export OPENAI_API_KEY=...

    from llm_judge import judge_answer
    judge_answer("What is the capital of Australia?", "Canberra", "It's Canberra.")  # -> True
"""

from __future__ import annotations

import os
import re

from openai import OpenAI

# ---------------------------------------------------------------------------
# Judge prompt (correctness only: CORRECT vs INCORRECT)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are a strict, impartial grader. Decide whether a predicted answer matches the gold target answer for a question.

Grade exactly one of CORRECT or INCORRECT:

- CORRECT: The predicted answer contains all the important information in the gold target and contradicts none of it. Only semantic meaning matters — capitalization, punctuation, and phrasing do not. Hedging ("I believe...", "it appears that...") is fine as long as the gold target is fully covered and nothing stated is wrong. Numeric answers must match the gold target's precision. Minor typos in proper names are tolerated.

- INCORRECT: The predicted answer contradicts the gold target, omits its important information, refuses ("I don't know"), is empty, or is off-topic. Hedging does not rescue a wrong fact.

End your response with exactly one line, and nothing after it:
GRADE: CORRECT
or
GRADE: INCORRECT"""

JUDGE_USER_TEMPLATE = """Question: {question}

Gold target answer: {ground_truth}

Predicted answer: {predicted_answer}

Decide CORRECT or INCORRECT and end with the required 'GRADE:' line."""


# ---------------------------------------------------------------------------
# Barebones OpenAI call
# ---------------------------------------------------------------------------

_client = None


def _openai_client() -> "OpenAI":
    """Lazily create one shared OpenAI client (needs OPENAI_API_KEY)."""
    global _client
    if _client is None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        _client = OpenAI(api_key=key)
    return _client


def call_openai(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-5.4-mini-2026-03-17",
    max_output_tokens: int = 2048 * 2,
) -> str:
    """One OpenAI Responses-API call. Returns the response text ('' if empty).

    Uses the Responses API (`responses.create`): the system prompt goes in
    `instructions`, the user prompt in `input`, and the text is read off the
    `output_text` helper. Newer models reject Chat-Completions params like
    `max_tokens`/`temperature`, so we pass `max_output_tokens` and no temperature.
    """
    response = _openai_client().responses.create(
        model=model,
        instructions=system_prompt,
        input=user_prompt,
        max_output_tokens=max_output_tokens,
    )
    return response.output_text or ""


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

# Last-match: if the model echoes the rubric's example line before its verdict,
# we want the trailing line it actually wrote.
_GRADE_RE = re.compile(r"GRADE\s*:\s*(CORRECT|INCORRECT)", re.IGNORECASE)


def judge_answer(
    question: str,
    ground_truth: str,
    predicted_answer: str,
    model: str = "gpt-5.4-mini-2026-03-17",
) -> bool:
    """Return True iff the judge grades the predicted answer CORRECT.

    Falls back to a lenient scan if the 'GRADE:' line is missing; defaults to
    False (incorrect) when the verdict can't be parsed at all.
    """
    user_prompt = JUDGE_USER_TEMPLATE.format(
        question=question, ground_truth=ground_truth, predicted_answer=predicted_answer,
    )
    response = call_openai(JUDGE_SYSTEM_PROMPT, user_prompt, model=model)
    matches = _GRADE_RE.findall(response)
    if matches:
        return matches[-1].upper() == "CORRECT"
    # No explicit GRADE line — fall back to the last CORRECT/INCORRECT word.
    words = re.findall(r"\b(CORRECT|INCORRECT)\b", response, re.IGNORECASE)
    return bool(words) and words[-1].upper() == "CORRECT"


if __name__ == "__main__":
    q = "In what year did the first human land on the Moon?"
    gt = "1969"
    for pred in ["The Apollo 11 landing was in 1969.", "It happened in 1972.", "I'm not sure."]:
        print(f"{judge_answer(q, gt, pred)!s:<5}  <-  {pred!r}")

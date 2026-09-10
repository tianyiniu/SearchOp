"""LLM judge for answer correctness, using Google's official Gemini API.

Given a question, the gold answer, and a model's predicted answer, ask Gemini to
decide CORRECT or INCORRECT. The prompt is a trimmed version of OpenAI's SimpleQA
grader — correctness ONLY (no not-attempted category, no trace/quality scoring).

    pip install google-genai
    export GEMINI_API_KEY=...

    from judge_gemini import judge_answer
    judge_answer("What is the capital of Australia?", "Canberra", "It's Canberra.")  # -> True
"""

from __future__ import annotations

import os
import re

from google import genai
from google.genai import types

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
# Barebones Gemini call (official google-genai SDK)
# ---------------------------------------------------------------------------

_client = None


def _gemini_client() -> "genai.Client":
    """Lazily create one shared Gemini client (needs GEMINI_API_KEY)."""
    global _client
    if _client is None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        _client = genai.Client(api_key=key)
    return _client


def call_gemini(
    system_prompt: str,
    user_prompt: str,
    model: str = "gemini-3.5-flash",   # swap for whatever Gemini model you have
    temperature: float = 0.0,
    max_output_tokens: int = 2048,
) -> str:
    """One Gemini text generation. Returns the response text ('' if empty)."""
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    response = _gemini_client().models.generate_content(
        model=model, contents=user_prompt, config=config,
    )
    return response.text or ""


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
    model: str = "gemini-3.5-flash",
) -> bool:
    """Return True iff Gemini grades the predicted answer CORRECT.

    Falls back to a lenient scan if the 'GRADE:' line is missing; defaults to
    False (incorrect) when the verdict can't be parsed at all.
    """
    user_prompt = JUDGE_USER_TEMPLATE.format(
        question=question, ground_truth=ground_truth, predicted_answer=predicted_answer,
    )
    response = call_gemini(JUDGE_SYSTEM_PROMPT, user_prompt, model=model)
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

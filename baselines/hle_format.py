"""HLE for the external baselines: the benchmark's own response format, reading the answer
off a response, and the prompts that ask a cut-off response for its answer.

A dataset row is HLE's when it has an "answer_type" (scripts/prepare_hle.py writes it;
SuperGPQA rows have none). The system prompts are HLE's (hle_eval/run_model_predictions.py):
an explanation, the answer on its own line ("Exact Answer:" for an open answer, "Answer:"
for a multiple-choice one) and a confidence. The answer read off a response is the last
"Exact Answer:" / "Final Answer:" / "Answer:" line, normalised as the debate executor
normalises its answers (debate_mcq.normalize_answer), and it is graded by the same judge
(scripts/judge_answers.py) with the same verdict cache.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from debate_mcq import normalize_answer  # noqa: E402

SYSTEM_EXACT = ("Your response should be in the following format:\n"
                "Explanation: {your explanation for your final answer}\n"
                "Exact Answer: {your succinct, final answer}\n"
                "Confidence: {your confidence score between 0% and 100% for your answer}")
SYSTEM_MC = ("Your response should be in the following format:\n"
             "Explanation: {your explanation for your answer choice}\n"
             "Answer: {your chosen answer}\n"
             "Confidence: {your confidence score between 0% and 100% for your answer}")

# an answer label at the start of a line (markdown around it allowed), with its colon, or alone on its
# line as a heading; and the labels that end an answer written below its label
_ANSWER_LABEL = re.compile(r"^[ \t>*#`-]*(?:exact[ \t]+answer|final[ \t]+answer|answer)[ \t*]*(?::|$)",
                           re.I | re.M)
_NEXT_LABEL = re.compile(r"^[ \t>*#`-]*(?:confidence|explanation)[ \t*]*(?::|$)", re.I | re.M)


def is_hle(item: dict) -> bool:
    return "answer_type" in item


def is_mc(item: dict) -> bool:
    return item.get("answer_type") == "multipleChoice"


def messages(item: dict) -> list[dict]:
    """HLE's request: its format as the system prompt, the question as the user turn."""
    return [{"role": "system", "content": SYSTEM_MC if is_mc(item) else SYSTEM_EXACT},
            {"role": "user", "content": item["question"]}]


def extract(text: str | None) -> str | None:
    """The normalised answer under the response's last answer label, or None: the rest of the
    label's line, or, when that is empty ('**Exact Answer:**' on a line of its own), the lines
    below it up to a blank line or the next label (Confidence, Explanation)."""
    labels = list(_ANSWER_LABEL.finditer(text or ""))
    if not labels:
        return None
    rest = text[labels[-1].end():]
    if (nxt := _NEXT_LABEL.search(rest)) is not None:
        rest = rest[:nxt.start()]
    line, _, below = rest.partition("\n")
    if normalize_answer(line) is not None:
        return normalize_answer(line)
    return normalize_answer(below.strip().split("\n\n")[0])


def _label(item: dict) -> str:
    return "Answer: {your chosen answer}" if is_mc(item) else "Exact Answer: {your succinct, final answer}"


def recover_prompt(item: dict) -> str:
    return ("Your response above ran out of space before you gave an answer. Based only on the reasoning "
            "above, give the final answer it supports; do not start the derivation again. Reply in the "
            f"required format; the line '{_label(item)}' is the one that must be there.")


def commit_prompt(item: dict) -> str:
    return ("STOP. You ran out of space. Do not continue the derivation. Reply with only the line "
            f"'{_label(item)}' for the answer the reasoning above supports.")


# Self-Refine's FEEDBACK and REFINE prompts for HLE: the SuperGPQA ones with the options
# taken out (FEEDBACK) and the answer format pointed at the system prompt's (REFINE).
FEEDBACK_PROMPT = (
    "There may be an error in the answer above because of a misreading of the question, a wrong "
    "fact, a flawed derivation, or an alternative that was dismissed too quickly. What is the error? "
    "To find it, go through the reasoning one step at a time and check whether each step follows "
    "and whether it is consistent with the question.\n\n"
    "Do not give a revised answer here, only the critique. If, after checking every step, the "
    "reasoning and the answer are sound, reply with exactly: it is correct"
)
REFINE_PROMPT = ("Using the feedback above, work the question out again and give an improved answer, in the "
                 "required format.")

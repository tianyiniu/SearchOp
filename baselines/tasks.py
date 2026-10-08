"""What changes between the datasets, for the external baselines.

A row's kind follows from its fields:

    supergpqa   no "dataset" and no "answer_type"     up to 10 options; SuperGPQA's zero-shot prompt
    hle         "answer_type" (prepare_hle.py)        HLE's own system prompt; graded by a judge model
    gpqa        "dataset": "gpqa" (prepare_gpqa_diamond.py)   4 options; OpenAI simple-evals' GPQA prompt
    math        "dataset": "math" (prepare_math_l5.py)        open answer in \\boxed{}; graded by math-verify

For each kind this gives the first request (messages), the answer read off a reply (answer_of: a
letter, HLE's normalised answer line, or the content of the last \\boxed{}), whether two answers
are the same (same, for votes), grading (correct; HLE is graded by the judge in the scorers), and
the prompts of the later turns: Self-Refine's feedback and refine, the debate turn, and the recovery
of a reply cut off before its answer. For SuperGPQA and HLE every prompt and every answer read is
exactly what generate.py, selfrefine.py, recover.py and score.py used before this module.

The math prompt is the one the Qwen and DeepSeek-R1 model cards give for math ("Please reason step
by step, and put your final answer within \\boxed{}."). math-verify runs its time limits with
signals, so `same` and `correct` for math work in the main thread only (the scorers call them
there; the generation scripts only call answer_of).
"""
from __future__ import annotations

import re
from functools import lru_cache

import hle_format
from score import LETTERS, extract_answer

KINDS = ("supergpqa", "hle", "gpqa", "math")


def kind(item: dict) -> str:
    if hle_format.is_hle(item):
        return "hle"
    k = item.get("dataset", "supergpqa")
    if k not in KINDS:
        raise ValueError(f"row {item.get('id')}: unknown dataset {k!r}")
    return k


# --- the first request ---------------------------------------------------------------------------
# config/prompt/zero-shot.yaml in the SuperGPQA repo (YAML literal block, keeps the trailing newline)
PROMPT_TEMPLATE = (
    "Answer the following multiple choice question. There is only one correct answer. "
    "The last line of your response should be in the format 'Answer: $LETTER' (without quotes), "
    "where LETTER is one of A, B, C, D, E, F, G, H, I, or J.\n\n{}\n"
)
# QUERY_TEMPLATE_MULTICHOICE in openai/simple-evals (common.py), as its GPQA eval sends it
GPQA_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your response should be of the "
    "following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by "
    "step before answering.\n\n{question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
)
MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
# --explain (off by default; run_baselines.py turns it on for gpt-oss): one sentence after the first
# prompt asking for the reasoning in the reply itself. gpt-oss keeps its reasoning in a hidden
# channel, and its visible reply to SuperGPQA's prompt is a median 9 characters ('Answer: C'), so
# Self-Refine's feedback and the other MAD agents, which see only the visible reply, had nothing to
# examine; Du et al.'s own first prompts ask for this ("Explain your reasoning."). Not for HLE,
# whose own format already asks for an explanation.
EXPLAIN_SENTENCE = "Explain your reasoning in your response before you give the final answer."
EXPLAIN = False


def set_explain(on: bool) -> None:
    global EXPLAIN
    EXPLAIN = bool(on)


def build_prompt(item: dict) -> str:
    """SuperGPQA's zero-shot prompt."""
    body = item["question"] + "\n" + "\n".join(
        f"{chr(65 + i)}) {opt}" for i, opt in enumerate(item["options"])
    )
    return PROMPT_TEMPLATE.format(body)


def messages(item: dict) -> list[dict]:
    """The first request of every method: the dataset's own prompt."""
    k = kind(item)
    if k == "hle":
        return hle_format.messages(item)
    if k == "gpqa":
        a, b, c, d = item["options"]
        text = GPQA_TEMPLATE.format(question=item["question"], A=a, B=b, C=c, D=d)
    elif k == "math":
        text = f"{item['question']}\n{MATH_INSTRUCTION}"
    else:
        text = build_prompt(item)
    if EXPLAIN:
        text = f"{text.rstrip()}\n\n{EXPLAIN_SENTENCE}"
    return [{"role": "user", "content": text}]


# --- reading and comparing answers ---------------------------------------------------------------

def last_boxed(text: str | None) -> str | None:
    """The content of the last \\boxed{...} (or \\fbox{...}) in `text`, braces balanced, stripped;
    '\\boxed 5' (no braces) gives the rest of its line up to a '$'. None if there is none. This
    is MATH's own rule (last_boxed_only_string + remove_boxed in hendrycks/math), for the key and
    for replies alike."""
    text = text or ""
    start = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if start < 0:
        return None
    rest = text[start + (6 if text.startswith("\\boxed", start) else 5):]
    if rest.startswith(" "):
        return rest.split("\n")[0].split("$")[0].strip() or None
    if not rest.startswith("{"):
        return None
    depth = 0
    for i, ch in enumerate(rest):
        depth += (ch == "{") - (ch == "}")
        if depth == 0:
            return rest[1:i].strip() or None
    return None                                        # unbalanced: the reply was cut off inside it


def answer_of(text: str | None, item: dict) -> str | None:
    """The answer a reply gives, or None: a letter (SuperGPQA's extraction rule; for GPQA only
    A-D count), HLE's normalised answer line, or the content of the last \\boxed{} (math)."""
    k = kind(item)
    if k == "hle":
        return hle_format.extract(text)
    if k == "math":
        return last_boxed(text)
    if k == "gpqa":
        return extract_answer(text, item["options"], LETTERS[:len(item["options"])])
    return extract_answer(text, item["options"])


@lru_cache(maxsize=None)
def _math(answer: str):
    from math_verify import parse
    return parse("\\boxed{" + answer + "}")


def _math_equal(a: str, b: str) -> bool:
    from math_verify import verify
    return bool(verify(_math(a), _math(b)))


def same(item: dict, a: str, b: str) -> bool:
    """Whether two answers are the same answer: equal letters or strings, or (math) equal by
    math-verify, so '\\frac{1}{2}' and '0.5' are one answer in a vote."""
    if a == b:
        return True
    return kind(item) == "math" and _math_equal(a, b)


def correct(item: dict, answer: str | None) -> bool:
    """Whether an answer is right. Letters: equal to the key's letter; math: equal to the key by
    math-verify. HLE needs the judge (judge_answers.py), which the scorers call in one batch."""
    if answer is None:
        return False
    k = kind(item)
    if k == "hle":
        raise ValueError("HLE answers are graded by the judge model (judge_answers.Judge)")
    if k == "math":
        return _math_equal(item["answer"], answer)
    return answer == item["answer_letter"]


def vote(answers: list[str | None], item: dict) -> str | None:
    """The most common answer, as self-consistency's majority vote and Du et al.'s most_frequent
    take it: answers that cannot be read (None) do not vote, and a tie goes to the answer given
    first. Equal answers (same) count together, under the spelling given first."""
    groups: list[list] = []                            # [answer, count], in order of first appearance
    for a in answers:
        if a is None:
            continue
        for g in groups:
            if same(item, g[0], a):
                g[1] += 1
                break
        else:
            groups.append([a, 1])
    return max(groups, key=lambda g: g[1])[0] if groups else None    # max keeps the first of a tie


# --- the prompts of later turns ------------------------------------------------------------------
# The answer format, said again in a later turn (refine, debate, recovery).
ANSWER_FORMAT = {
    "supergpqa": ("The last line of your response should be in the format 'Answer: $LETTER' (without "
                  "quotes), where LETTER is one of A, B, C, D, E, F, G, H, I, or J."),
    "gpqa": ("The last line of your response should be of the following format: 'Answer: $LETTER' "
             "(without quotes) where LETTER is one of ABCD."),
    "math": MATH_INSTRUCTION,
    "hle": "Give your answer in the required format.",
}

# Self-Refine (selfrefine.py). FEEDBACK: the GSM prompt asserts an error exists; here it may also
# report that the answer is sound, which is what stops the loop (run.py checks for "it is correct").
FEEDBACK_PROMPT = (
    "There may be an error in the answer above because of a misreading of the question, a wrong "
    "fact, a flawed derivation, or an option that was dismissed too quickly. What is the error? "
    "To find it, go through the reasoning one step at a time and check whether each step follows "
    "and whether it is consistent with the question and the options.\n\n"
    "Do not give a revised answer here, only the critique. If, after checking every step, the "
    "reasoning and the chosen option are sound, reply with exactly: it is correct"
)
REFINE_LEAD = "Using the feedback above, work the question out again and give an improved answer. "
REFINE_PROMPT = REFINE_LEAD + ANSWER_FORMAT["supergpqa"]
STOP_PHRASE = "it is correct"
_MARKUP_CMDS = re.compile(r"\\(?:text|textbf|textit|mathrm|mathbf|boxed|fbox)\b")
_MARKUP = re.compile(r"\\[ ,;:!]|~|[{}$*`_]")


def says_correct(feedback: str | None) -> bool:
    """Self-Refine's stop rule: the feedback contains 'it is correct' (run.py's check), read
    through LaTeX and markdown. Under the math prompt the model boxes its verdict as
    '\\boxed{it \\ is \\ correct}', which the plain check misses. On the 13,719 saved
    SuperGPQA and HLE feedback replies this rule and the plain check agree on every one."""
    text = _MARKUP.sub(" ", _MARKUP_CMDS.sub(" ", (feedback or "").lower()))
    return STOP_PHRASE in re.sub(r"\s+", " ", text)


def feedback_prompt(item: dict) -> str:
    """Multiple choice: FEEDBACK_PROMPT; open answers (HLE, math): HLE's, which does not mention options."""
    return hle_format.FEEDBACK_PROMPT if kind(item) in ("hle", "math") else FEEDBACK_PROMPT


def refine_prompt(item: dict) -> str:
    k = kind(item)
    return hle_format.REFINE_PROMPT if k == "hle" else REFINE_LEAD + ANSWER_FORMAT[k]


# Recovery of a reply cut off before its answer (recover.py)
RECOVER_LEAD = ("Your response above ran out of space before you gave an answer. Based only on the reasoning "
                "above, choose the option it supports; do not start the derivation again. ")
RECOVER_PROMPT = RECOVER_LEAD + ANSWER_FORMAT["supergpqa"]
COMMIT_PROMPT = (
    "STOP. You ran out of space. Do not continue the derivation. Reply with only the line "
    "'Answer: $LETTER' (without quotes) for the single option the reasoning above supports."
)
RECOVER_MATH = ("Your response above ran out of space before you gave an answer. Based only on the reasoning "
                "above, give the final answer it supports; do not start the derivation again. Put your "
                "final answer within \\boxed{}.")
COMMIT_MATH = ("STOP. You ran out of space. Do not continue the derivation. Reply with only the final answer "
               "the reasoning above supports, within \\boxed{}.")


# A Self-Refine feedback turn cut off before any visible text (selfrefine.py --recover-feedback)
FEEDBACK_RECOVER = ("Your response above ran out of space before you wrote the critique. Based only on the "
                    "reasoning above, write the critique it supports; do not start the check again. If the "
                    "reasoning found the answer sound, reply with exactly: it is correct")
FEEDBACK_COMMIT = ("STOP. You ran out of space. In at most three sentences, state the main error the reasoning "
                   "above found; if it found none, reply with exactly: it is correct")


def feedback_recover_prompts(item: dict) -> tuple[str, str]:
    """(recovery prompt, shorter commit prompt) for a cut-off feedback turn; the same for every dataset."""
    return FEEDBACK_RECOVER, FEEDBACK_COMMIT


def recover_prompts(item: dict) -> tuple[str, str]:
    """(recovery prompt, shorter commit prompt) for the row's dataset."""
    k = kind(item)
    if k == "hle":
        return hle_format.recover_prompt(item), hle_format.commit_prompt(item)
    if k == "math":
        return RECOVER_MATH, COMMIT_MATH
    return RECOVER_LEAD + ANSWER_FORMAT[k], COMMIT_PROMPT


# The debate turn (mad.py): Du et al.'s construct_message (composable-models/llm_multiagent_debate).
# Multiple choice and HLE take the wording of their MMLU script, math the wording of their GSM
# script (which says the problem again). Their closing format request ('(X)', or a single number
# in \boxed{}) is replaced by the dataset's own, so every method is read by the same rule.
DEBATE_PREFIX = "These are the solutions to the problem from other agents: "
DEBATE_SOLUTION = "\n\n One agent solution: ```{}```"
DEBATE_MC = ("\n\n Using the reasoning from other agents as additional advice, can you give an updated "
             "answer? Examine your solution and that other agents step by step. ")
DEBATE_MATH = ("\n\n Using the solutions from other agents as additional information, can you provide your "
               "answer to the math problem? \n The original math problem is {}. Your final answer should be "
               "in the form \\boxed{{answer}}, at the end of your response.")
DEBATE_ALONE = "Can you double check that your answer is correct. "


def debate_message(item: dict, others: list[str]) -> str:
    """The user turn of a debate round: the other agents' replies of the previous round, then the
    request for an updated answer (with no other agents, Du et al.'s request to check the answer)."""
    k = kind(item)
    if not others:
        return DEBATE_ALONE + ANSWER_FORMAT[k]
    text = DEBATE_PREFIX + "".join(DEBATE_SOLUTION.format(o) for o in others)
    if k == "math":
        return text + DEBATE_MATH.format(item["question"])
    return text + DEBATE_MC + ANSWER_FORMAT[k]

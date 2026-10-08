"""Shared machinery for the SuperGPQA debate experiments (MCQ, no documents).

Ports the schema grammar + executor from eval_debate_step.py / evolve_debate.py to
the reasoning-MCQ setting, so the fixed-template eval and the schema-evolution loop
run the EXACT same executor:

  - No documents. Personas reason from parametric knowledge, not from a context.
  - Discrete answers. Grading and voting are on the option LETTER — exact match, no
    LLM judge, no answer-text normalization (the noise sources we hit on FRAMES).
  - Thinking OFF by default. Qwen's <think> trace is disabled so multi-round debate
    stays inside the context window; the collection step already used CoT to certify
    these questions are hard, and the schemas add breadth, not a longer hidden trace.
    Personas still reason step by step in their VISIBLE output.

A schema is the same tiny structure as the FRAMES version:

    schema = { rounds: [ {personas: [...]}, ... ], final: "last" | "vote" | "synthesizer" }
    personas in (solver, critic, synthesizer)
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import urllib.request
from collections import Counter
from copy import deepcopy

LETTERS = "ABCDEFGHIJ"


# --- prompting + letter grading --------------------------------------------

ANSWER_INSTR_V1 = "Reason step by step, then end with exactly one line: 'ANSWER: <letter>'."
# v2: the same commitment format, but the persona is asked to reason carefully
# and given the room to do it (a 6144-token reply budget, then a summary call --
# see set_v2). No braces: this text passes through EXPERT_TMPL.format().
ANSWER_INSTR_V2 = ("Think this through carefully before you answer: work out what the "
                   "question is really asking, evaluate each option on its merits, check "
                   "the facts or calculations your choice depends on, and consider what "
                   "would change your mind. Take the space you need. Then end with exactly "
                   "one line: 'ANSWER: <letter>'.")
# The plain instruction (off by default; set_plain_instruction): only the commitment line,
# with no request to think longer. How long a speaker thinks is then set by the effort
# sent with the request alone. Measured on Qwen3.5-9B (global-pipeline run1, test): the V2 sentence
# made a high-effort solver think 1.5x longer than SuperGPQA's plain prompt (median),
# and run out of the window on 3.6% of calls, at no gain in accuracy.
ANSWER_INSTR_PLAIN = "End with exactly one line: 'ANSWER: <letter>'."
PLAIN_INSTR = False
ANSWER_INSTR = ANSWER_INSTR_V1

# Visible reasoning (off by default; nothing changes for any model unless it is
# switched on). Models that think in a hidden channel (gpt-oss) put only
# 'ANSWER: X' in the visible reply: measured median 9 characters, so a later
# critic or verifier was shown bare letters and had nothing to examine. With
# this on, one sentence is appended to the commitment instruction asking for
# the reasoning in the reply itself, a committed reply too short to carry any
# reasoning still gets the summary call, and the round cache key carries g=1.
# No braces in the sentence: it passes through EXPERT_TMPL.format().
VISIBLE_REASONING = False
VISIBLE_SENTENCE_BRIEF = (" Your reply is read by future participants who cannot see your private "
                          "thinking, so write your reasoning out in the reply itself (the key facts "
                          "or steps, and why the other options are wrong) before the ANSWER line.")
# The fuller request, for the reply-instruction probe (archive/scripts/probe_executor_v3.py). Content only, no
# target length. It becomes the default only if the probe shows longer replies at no accuracy cost.
VISIBLE_SENTENCE_DETAILED = (" Your reply is read by future participants who cannot see your private "
                             "thinking, so write your reasoning out in the reply itself before the ANSWER "
                             "line: the decisive facts or steps in the order you used them, the numbers of "
                             "any calculation, which options you ruled out and why, and the point you are "
                             "least sure of.")
VISIBLE_SENTENCE = VISIBLE_SENTENCE_BRIEF
VISIBLE_MIN_CHARS = 80            # under visible reasoning, a shorter reply is not its own summary


def render_question(question: str, options: list[str]) -> str:
    body = "\n".join(f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options))
    return f"{question}\n\n{body}"


_ANS = re.compile(r"ANSWER\s*:\s*\(?\s*([A-J])\b", re.I)


def extract_letter(text: str, n_options: int) -> str | None:
    """The letter from the last 'ANSWER:' line, or None if there is no valid one.
    With open answers on (set_open_answers), the normalised answer of the last
    ANSWER line instead, whatever `n_options` is: every reader of a commitment
    (votes, agreement, the rules, the final read) goes through here."""
    if OPEN:
        found = _math_found(text) if MATH else open_answers(text)
        return found[-1] if found else None
    for m in reversed(_ANS.findall(text or "")):
        letter = m.upper()
        if 0 <= LETTERS.index(letter) < n_options:
            return letter
    return None


# --- open answers (HLE) ---------------------------------------------------------------
# Off by default: nothing changes for the multiple-choice datasets. With it on, a speaker's
# commitment is the whole of its last line 'ANSWER: <answer>', in the fixed format that
# ANSWER_FORMAT asks for, and the commitment's identity (for votes, agreement and the rules)
# is that answer after normalize_answer. Grading is not done here: a judge model compares the
# final answer with the key (judge_answers.py, set up by program_space.configure_from_args).
# It changes every prompt, so it is part of the round cache key ("o").

OPEN = False
# No braces anywhere below: these texts pass through EXPERT_TMPL.format().
ANSWER_FORMAT = ("Put the final answer alone on that line, with nothing else: for a question with answer "
                 "choices, only the letter of your choice, as in ANSWER: C; otherwise the answer in its "
                 "simplest exact form, such as an integer, a fraction a/b, a decimal, an expression in plain "
                 "LaTeX without dollar signs, or a word or short phrase, with no explanation and no "
                 "trailing period.")
ANSWER_INSTR_OPEN = ("Think this through carefully before you answer: work out what the question is really "
                     "asking, check the facts, derivations or calculations your answer depends on, and "
                     "consider what would change your mind. Take the space you need. Then end with exactly "
                     "one line: 'ANSWER: <final answer>'. " + ANSWER_FORMAT)
ANSWER_INSTR_OPEN_PLAIN = "End with exactly one line: 'ANSWER: <final answer>'. " + ANSWER_FORMAT
VISIBLE_SENTENCE_OPEN = (" Your reply is read by future participants who cannot see your private thinking, "
                         "so write your reasoning out in the reply itself (the key facts or steps, and why "
                         "the alternatives fail) before the ANSWER line.")
# an ANSWER line: the label at the start of a line (markdown around it allowed), then the answer
_ANS_OPEN = re.compile(r"^[ \t>*#`-]*ANSWER[ \t*]*:(.*)$", re.I | re.M)
_WRAPPERS = (("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]"), ("`", "`"))
_WRAPPER_CMDS = re.compile(r"\\(?:boxed|text|textbf|mathrm)\{(.*)\}", re.S)
# MATH answers keep a \\text{...}: math-verify reads '\\text{C,E}' and 'C,E' as different answers, and
# grading must read what the speaker wrote (math_answers.key drops the wrapper when math-verify agrees)
_WRAPPER_CMDS_MATH = re.compile(r"\\boxed\{(.*)\}", re.S)


def _balanced(s: str) -> bool:
    depth = 0
    for ch in s:
        depth += (ch == "{") - (ch == "}")
        if depth < 0:
            return False
    return depth == 0


def normalize_answer(text: str | None, _cmds: re.Pattern = _WRAPPER_CMDS) -> str | None:
    """An answer's identity: the text with only its formatting removed -- runs of
    two or more asterisks at the ends (markdown bold, whole or left over from a bold
    label; a single '*' is kept, it can be notation), a wrapping `...`, $...$,
    \\(...\\) or \\[...\\], a wrapping \\boxed{...} or \\text{...}, a single-letter
    choice's '(C)' / 'C)' / 'C.', one trailing period, \\displaystyle / \\textstyle,
    repeated whitespace. Case is
    kept (in SMILES it matters). Idempotent: normalizing a normalized answer returns
    it unchanged. None if nothing is left. (MATH answers pass _WRAPPER_CMDS_MATH: only a
    wrapping \\boxed{...} is removed of the commands.)"""
    a, prev = (text or "").strip(), None
    while a != prev:
        prev = a
        a = re.sub(r"^\*{2,}|\*{2,}$", "", a.strip()).strip()
        for left, right in _WRAPPERS:
            inner = a[len(left):len(a) - len(right)]
            if a.startswith(left) and a.endswith(right) and inner.strip() and left not in inner:
                a = inner.strip()                  # '$x$ and $y$' is not one wrapped answer
        if (m := _cmds.fullmatch(a)) and _balanced(m.group(1)):
            a = m.group(1).strip()
        if (m := re.fullmatch(r"\(?([A-Z])[).]?", a)) and a != m.group(1):
            a = m.group(1)
        if a.endswith(".") and not a.endswith(".."):
            a = a[:-1].rstrip()
        a = re.sub(r"\\(?:displaystyle|textstyle)\b\s*", "", a).strip()    # LaTeX size, not content
    a = re.sub(r"\s+", " ", a).strip()
    return a or None


def open_answers(text: str | None) -> list[str]:
    """The normalised answers of every ANSWER line in `text`, in order (lines
    whose answer is empty are skipped)."""
    return [a for m in _ANS_OPEN.findall(text or "") if (a := normalize_answer(m)) is not None]


def set_open_answers(on: bool) -> None:
    """Use program_space.configure_executor (--answers open) from entry scripts,
    which also switches the critic override and the grader."""
    global OPEN
    OPEN = bool(on)
    _rebuild_prompts()


def open_signature() -> str | None:
    """'1' with open answers on ('m' with MATH answers), else None. Part of the cache key."""
    return "m" if MATH else "1" if OPEN else None


# --- MATH answers (--answers math; off by default, 2026-10-07) -------------------------------------
# Open answers (OPEN is on too) to MATH questions. A speaker gives its answer in \boxed{...} (MATH's own
# format) or on an ANSWER line (what the summary and letter follow-ups ask for); the last one in its
# text counts. An answer is read as its key (math_answers.key): two answers are the same answer when
# math-verify says so, and the key is what votes, rules and later speakers see. A final answer is
# graded by math-verify (math_answers.grade_row), as the external baselines grade it. The solver's user
# message is the external direct baseline's request (baselines/tasks.py): the question, then
# MATH_INSTRUCTION; the other speakers are asked for the same \boxed{} answer. It changes every
# prompt, so it is part of the round cache key ("o": "m") and of the prompt signature.
MATH = False
MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
ANSWER_INSTR_MATH = ("Think this through carefully before you answer: work out what the question is really "
                     "asking, check the derivations or calculations your answer depends on, and consider "
                     "what would change your mind. Take the space you need. Then put your final answer "
                     "within \\boxed{}.")
ANSWER_INSTR_MATH_PLAIN = "Put your final answer within \\boxed{}."
VISIBLE_SENTENCE_MATH = (" Your reply is read by future participants who cannot see your private thinking, "
                         "so write your reasoning out in the reply itself (the key steps, and why the "
                         "alternatives fail) before your final answer.")
COMMIT_NUDGE_MATH = ("STOP. You ran out of space. Do not continue or summarize the derivation. Your reply "
                     "must BEGIN with the line 'ANSWER: <final answer>', giving the final answer the "
                     "reasoning so far supports, in its simplest exact form. You may add one short "
                     "sentence after that line, nothing more.")


def set_math_answers(on: bool) -> None:
    """Use program_space.configure_executor (--answers math) from entry scripts, which also turns on
    open answers, switches the critic override and sets the grader."""
    global MATH
    MATH = bool(on)
    _rebuild_prompts()


def math_question(question: str) -> str:
    """The external direct baseline's request for a MATH question (baselines/tasks.messages)."""
    return f"{question}\n{MATH_INSTRUCTION}"


def _boxed_answers(text: str) -> list[tuple[int, str | None]]:
    """(position, content) of every \\boxed and \\fbox in `text`, by MATH's own rule
    (baselines/tasks.last_boxed, applied to each one): the content of its balanced braces; '\\boxed 5'
    gives the rest of its line up to a '$'. None for one with neither, or left open (a reply cut off
    inside it): as there, such a last box gives no answer."""
    out: list[tuple[int, str | None]] = []
    for m in re.finditer(r"\\(boxed|fbox)", text or ""):
        rest, a = text[m.end():], None
        if rest.startswith(" "):
            a = rest.split("\n")[0].split("$")[0].strip() or None
        elif rest.startswith("{"):
            depth = 0
            for i, ch in enumerate(rest):
                depth += (ch == "{") - (ch == "}")
                if depth == 0:
                    a = rest[1:i].strip() or None
                    break
        out.append((m.start(), a))
    return out


def _math_found(text: str | None) -> list[str | None]:
    """Every answer in `text` in the order it appears (each \\boxed and each ANSWER line), as its key
    (math_answers.key, of the answer as normalize_answer leaves it, a \\text{} kept); None for a box
    that gives none."""
    import math_answers as MA
    found = [(m.start(), m.group(1)) for m in _ANS_OPEN.finditer(text or "")] + _boxed_answers(text or "")
    found.sort(key=lambda t: t[0])
    keys = []
    for _, raw in found:
        a = normalize_answer(raw, _WRAPPER_CMDS_MATH) if raw is not None else None
        if a is not None or raw is None:          # an ANSWER line with nothing on it is skipped
            keys.append(MA.key(a) if a is not None else None)
    return keys


def math_answers(text: str | None) -> list[str]:
    """The keys of every answer in `text`, in order (_math_found without the boxes that give none)."""
    return [k for k in _math_found(text) if k is not None]


# --- token accounting -------------------------------------------------------------
# Every model call adds its prompt and completion tokens to a per-thread tally.
# execute_round gathers the tallies of its speakers, and the round runner writes
# them into the round's recording ("usage"), so cost can be reported in tokens
# and not only in turns. Nothing reads them back: a recording with or without
# them replays the same.

_tokens = threading.local()


def _count_tokens(resp) -> None:
    tally, usage = getattr(_tokens, "tally", None), getattr(resp, "usage", None)
    if tally is None or usage is None:
        return
    tally["calls"] += 1
    tally["prompt"] += getattr(usage, "prompt_tokens", 0) or 0
    tally["completion"] += getattr(usage, "completion_tokens", 0) or 0


def last_round_usage() -> list[dict] | None:
    """Per speaker, in speaker order, the tokens of the round this thread last
    executed; None if it has been read already or no round has run."""
    used, _tokens.last_round = getattr(_tokens, "last_round", None), None
    return used


def _message_text(msg) -> str:
    """The model's visible output. A server running --reasoning-parser puts the
    thinking in `reasoning_content` and only what follows in `content`; if the
    model never closes its thinking block, `content` comes back empty and the
    answer, if there is one, is in the reasoning field. Prefer content, fall
    back to reasoning, so both server configurations behave the same here."""
    content = (getattr(msg, "content", None) or "").strip()
    if content:
        return content
    return (getattr(msg, "reasoning_content", None)
            or getattr(msg, "reasoning", None) or "")


# How to ask a Qwen model not to emit a thinking block. Tried in order; the
# first one the server accepts is remembered per (client, model) so the cost is
# one probe per run, not per call. Order matters for reproducibility: the
# original mechanism is first, so Qwen3-14B behaves exactly as it always has.
_THINK_OFF = (
    {"chat_template_kwargs": {"enable_thinking": False}},   # Qwen3 / Qwen3.5 chat template
    # ({"reasoning_effort": "none"} was here; a live probe on Qwen3.5-27B showed
    #  it leaves thinking ON, so it could only ever lock in the wrong mode.)
    {},                                                     # kwarg rejected: server default
)
_think_mode: dict[tuple[int, str], dict] = {}
_think_lock = threading.Lock()


def chat(client, model: str, system: str, user: str, temperature: float,
         max_tokens: int = 3072, thinking: bool = False) -> str:
    """One chat call. With thinking=False, ask the server to skip the thinking
    block, trying each known mechanism until one is accepted AND actually
    yields visible output. An unusable reply (empty, or cut off before the
    model committed) makes the next mechanism worth trying."""
    if thinking:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature, max_tokens=max_tokens)
        _count_tokens(resp)
        return _message_text(resp.choices[0].message)

    key = (id(client), model)
    with _think_lock:
        known = _think_mode.get(key)
    for extra in ((known,) if known is not None else _THINK_OFF):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=temperature, max_tokens=max_tokens, extra_body=extra)
            _count_tokens(resp)
        except Exception:
            continue                       # server rejected the kwarg: try the next
        text = _message_text(resp.choices[0].message)
        # Truncated mid-thought is the signature of thinking that was not
        # disabled; do not lock in a mechanism that produces it. (getattr:
        # test stubs and some clients do not expose finish_reason.)
        truncated = getattr(resp.choices[0], "finish_reason", None) == "length"
        if text and not truncated:
            with _think_lock:
                _think_mode.setdefault(key, extra)
            return text
        if text:
            # A long reply that hit the cap is still a reply. Falling through to
            # the next mechanism would spend a second full-length call and, on a
            # server without a thinking-off default, could lock thinking ON for
            # the whole run. Report what came back and leave the mode undecided.
            return text
    return ""


# --- commit follow-up ---------------------------------------------------------
# On Qwen3.5-27B about one solver reply in five ran out of output tokens before
# its ANSWER line (median 9.2k chars, half of them at the 3072-token cap), so the
# call counted as silence. With the follow-up on, a reply that commits no letter
# gets ONE short continuation: the same conversation plus a nudge to commit now,
# with a tiny output budget. The nudge text is appended to the stored reply, so
# extract_letter and the digest later personas read both see the commitment.
# It changes what a recording contains, so it is part of the round cache key
# (schema_fitness.path_key) and old recordings never replay as if they had it.

COMMIT_FOLLOWUP = False
# First version of the nudge asked for "exactly one line"; 68% of replies began
# a prose summary instead and were cut off at a 48-token budget. This one puts
# the letter FIRST, allows a short justification after it, and gives room.
COMMIT_NUDGE_LETTER = ("STOP. You ran out of space. Do not continue or summarize the derivation. "
                       "Your reply must BEGIN with the line 'ANSWER: X', where X is the single "
                       "option letter you choose based on the reasoning so far. You may add one "
                       "short sentence after that line, nothing more.")
COMMIT_NUDGE_OPEN = ("STOP. You ran out of space. Do not continue or summarize the derivation. Your reply "
                     "must BEGIN with the line 'ANSWER: <final answer>', giving the final answer the "
                     "reasoning so far supports, in the required format (for a question with answer "
                     "choices, only the letter). You may add one short sentence after that line, nothing "
                     "more.")
COMMIT_NUDGE = COMMIT_NUDGE_LETTER           # _rebuild_prompts picks by the answer mode
COMMIT_MARK = "\n\n[cut off; asked for its answer]\n"
COMMIT_MAX_TOKENS = 160

# A commit reply is a direct answer to "which letter", so reading a letter out of
# its prose is safe in a way it is not for a mid-derivation mention. Tried in
# order on the follow-up text only; the last match wins.
_COMMIT_PATS = (
    re.compile(r"ANSWER\s*[:=\-]?\s*\**\(?\s*([A-J])\b", re.I),
    re.compile(r"\\boxed\{\s*\(?([A-J])\)?\s*\}"),
    re.compile(r"(?:answer|option|choice)\s*(?:is|:|would be|should be)?\s*\**\(?([A-J])\)?\**(?![a-z])", re.I),
    re.compile(r"^\W*([A-J])\W*$"),
)


def commit_letter_lenient(text: str, n_options: int) -> str | None:
    """The letter a commit reply names, or None. With open answers, only an
    ANSWER line counts (there is no letter to spot in prose)."""
    if OPEN:
        return extract_letter(text, n_options)
    for pat in _COMMIT_PATS:
        for m in reversed(pat.findall(text or "")):
            letter = m.upper()
            if LETTERS.index(letter) < n_options:
                return letter
    return None
on_followup = None          # callable(); a runner sets it to account for the extra call


def set_commit_followup(on: bool) -> None:
    global COMMIT_FOLLOWUP
    COMMIT_FOLLOWUP = bool(on)


def commit_signature() -> str | None:
    """'1' when the follow-up is on, else None. Part of the cache key. Under v2
    the summary call does the committing, so the follow-up never runs and the
    key must not claim it did."""
    return "1" if (COMMIT_FOLLOWUP and not V2) else None


def commit_followup(client, model: str, system: str, user: str, partial: str,
                    temperature: float, thinking: bool = False,
                    n_options: int | None = None) -> str:
    """One continuation of a reply that committed no letter. Returns the text to
    append to the reply (marker + the model's commit line), or '' on failure.
    With `n_options`, a reply that names its letter in prose rather than on an
    ANSWER line gets a normalized 'ANSWER: X' line appended, so extract_letter
    and every later persona read it the same way."""
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": partial},
                {"role": "user", "content": COMMIT_NUDGE}]
    key = (id(client), model)
    with _think_lock:
        known = _think_mode.get(key)
    if thinking:
        extras = [None]
    elif known is not None:
        extras = [known]                   # the mechanism chat() settled on
    else:
        extras = list(_THINK_OFF)
    for extra in extras:
        try:
            kw = {} if extra is None else {"extra_body": extra}
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
                max_tokens=COMMIT_MAX_TOKENS, **kw)
            _count_tokens(resp)
        except Exception:
            continue
        text = _message_text(resp.choices[0].message)
        if text:
            if on_followup is not None:
                on_followup()
            text = text.strip()
            if n_options is not None and extract_letter(text, n_options) is None:
                if (l := commit_letter_lenient(text, n_options)) is not None:
                    text += f"\nANSWER: {l}"
            return COMMIT_MARK + text
    return ""


def chat_commit(client, model: str, system: str, user: str, temperature: float,
                max_tokens: int, n_options: int, thinking: bool = False) -> str:
    """chat(), plus the commit follow-up when it is on and the reply has no letter."""
    text = chat(client, model, system, user, temperature, max_tokens, thinking)
    if COMMIT_FOLLOWUP and text and extract_letter(text, n_options) is None:
        text = text + commit_followup(client, model, system, user, text, temperature,
                                      thinking, n_options=n_options)
    return text


# --- v2: careful reasoning, room to finish, then a summary for the others -------
# Under v2 every answering persona makes TWO calls. The first is the reasoning
# reply, with a budget large enough that it finishes (6144 tokens by default;
# on the 27B recordings the 99th percentile sat at the old 3072 cap and accuracy
# was flat with length, so the point is to stop losing commits, not to reason
# longer). The second asks the same persona, in the same conversation, for a
# short summary ending in its ANSWER line. The stored reply is
#     full reply + SUMMARY_MARK + summary
# still one (persona, text) pair, so every cache and consumer is unchanged:
# extract_letter reads the LAST ANSWER line (the summary's, reconciled to the
# full reply's letter), and the digest later personas read shows only the part
# after the marker. v2 is part of the round cache key ("v":"2").

V2 = False
SUMMARY_MARK = "\n\n[v2 summary for later speakers]\n"
SUMMARY_NUDGE = ("Now write a brief summary of your reasoning for future participants, "
                 "in at most 120 words: the decisive facts or steps, the options you ruled "
                 "out and why, and the point you are least sure of. Do not add new analysis. "
                 "End with exactly one line 'ANSWER: X' -- the same letter you chose above. "
                 "If you gave no answer above, give one now.")
SUMMARY_MAX_TOKENS = 320          # <=120 words + the ANSWER line; conclusions ran 60-120 tokens
V2_SHORT_CHARS = 800              # a reply this short that already commits is its own summary
on_summary = None                 # callable(); a runner sets it to account for the extra call

V2_STATS: dict[str, int] = {"summaries": 0, "skipped_short": 0, "disagreements": 0,
                            "recovered_commits": 0, "failed_summaries": 0,
                            # v3 only: a summary that named another letter (not used), a letter
                            # obtained by the commit prompt, a reply left with no letter at all
                            "rejected_summaries": 0, "nudged_commits": 0, "uncommitted": 0}
_stats_lock = threading.Lock()


def _bump(name: str) -> None:
    with _stats_lock:
        V2_STATS[name] += 1


def reset_v2_stats() -> None:
    with _stats_lock:
        for k in V2_STATS:
            V2_STATS[k] = 0


# --- summary length -----------------------------------------------------------
# The summary is all a later speaker sees of a reply (through the digest window).
# 120 words is the original; a longer one carries more of the derivation. The
# length is part of the round cache key when it is not 120, so recordings made
# under different lengths never replay as one another.

SUMMARY_WORDS = 120
_SUMMARY_NUDGE_120, _SUMMARY_TOKENS_120 = SUMMARY_NUDGE, SUMMARY_MAX_TOKENS


def set_summary_words(n: int) -> None:
    global SUMMARY_WORDS, SUMMARY_NUDGE, SUMMARY_MAX_TOKENS
    if n < 40:
        raise ValueError("a summary needs at least 40 words")
    SUMMARY_WORDS = int(n)
    if n == 120:
        SUMMARY_NUDGE, SUMMARY_MAX_TOKENS = _SUMMARY_NUDGE_120, _SUMMARY_TOKENS_120
        return
    SUMMARY_NUDGE = (_SUMMARY_NUDGE_120
                     .replace("a brief summary", "a comprehensive summary")
                     .replace("in at most 120 words: the decisive facts or steps,",
                              f"in at most {n} words. Future participants see ONLY this summary, never "
                              f"your full reasoning, so make it complete enough that a reader could follow "
                              f"and check your whole argument from it; use the space you need. Cover: the "
                              f"decisive facts or steps in the order you used them, every calculation with "
                              f"its numbers and intermediate results, the assumptions you made,"))
    # Room to finish: models overrun the word limit (the 9B's 90th percentile was 1.23x it) and
    # formula-heavy text runs past 2 tokens a word, so allow 3 a word. At the original cap
    # (2.7 a word) 5% of the 9B's summaries were cut off; the cap only bounds a runaway reply.
    SUMMARY_MAX_TOKENS = 3 * int(n) + 60


def summary_signature() -> int | None:
    """The word limit when it is not the original 120, else None. Part of the cache key."""
    return None if SUMMARY_WORDS == 120 else SUMMARY_WORDS


# Two settings kept from the deep-think speaker (a v2-only option, removed 2026-10-06), with their
# values unchanged: how much of a reply's hidden reasoning the summary call is shown, and the turns
# one high-effort speaker counts unless --high-cost says otherwise.
THINKING_TAIL = 16000             # characters of thinking shown to the summary call (its end)
DEFAULT_HIGH_COST = 5             # the reasoning-high direct baseline averaged 9.9k tokens a call,
                                  # an ordinary reply ~2k


# --- the judge speaker (v3 only) ------------------------------------------------------------
# A speaker that chooses among the answers already committed. It sees the debate as any seeing
# speaker does, plus the list of candidate answers (every distinct answer committed so far), and
# is told that how many chose an answer is not evidence. It may not propose an answer of its own:
# a pick outside the list gets one low-effort prompt to choose from the list (judge_pick_nudge);
# if that fails too, its own answer stands (counted in JUDGE_STATS["off_list"]). With no answer
# committed yet it answers the question itself. Off unless set_judge_persona(True): the prompt,
# the plan rounds and the moves exist only then, so nothing changes for a run without it. Its
# rounds have keys of their own (the persona is in the key), so no other recording changes.

JUDGE_ON = False
JUDGE_PERSONA = "judge"
JUDGE_PROMPT = ("You are a Judge. Earlier participants have answered this question; their answers and "
                "the reasoning behind them are shown, followed by the list of candidate answers. Decide "
                "which candidate is correct. Judge each one by the strength of its reasoning against the "
                "exact wording of the question: check its decisive facts, steps and calculations, and "
                "find the specific error in every candidate you reject. How many participants chose an "
                "answer is not evidence that it is correct. You must choose one of the listed "
                "candidates; do not propose a different {what}. ")
JUDGE_MARK = "\n\n[judge: asked to choose one of the candidates]\n"
JUDGE_STATS: dict[str, int] = {"calls": 0, "no_candidates": 0, "repicked": 0, "off_list": 0}


def set_judge_persona(on: bool) -> None:
    """Use program_space.configure_executor (--judge-persona) from entry scripts, which also
    adds the judge's plan rounds and moves."""
    global JUDGE_ON
    JUDGE_ON = bool(on)
    _rebuild_prompts()


def judge_candidates(all_rounds: list[list[tuple[str, str]]], n: int) -> list[str]:
    """The distinct answers committed so far, in the order they first appeared."""
    return list(dict.fromkeys(l for rnd in all_rounds for p, r in rnd
                              if p not in NON_ANSWERING and (l := extract_letter(r, n))))


def judge_block(cands: list[str]) -> str:
    """The candidate list shown to a judge after the debate."""
    if not cands:
        return ("No answer has been given yet, so there is nothing to choose between: answer the "
                "question yourself.")
    if OPEN:
        return ("Candidate answers (every distinct answer given so far):\n"
                + "\n".join(f"- {c}" for c in cands)
                + "\nChoose exactly one of them, and write it on the ANSWER line exactly as it appears here.")
    return (f"Candidate answers (every distinct letter given so far): {', '.join(cands)}. "
            f"Choose exactly one of them.")


def judge_pick_nudge(cands: list[str]) -> str:
    """The follow-up for a judge whose answer is not one of the candidates."""
    if OPEN:
        return ("Your answer must be one of the candidate answers:\n" + "\n".join(f"- {c}" for c in cands)
                + "\nBased on your reasoning above, reply with only the line 'ANSWER: <candidate>' for the "
                  "candidate it supports, written exactly as listed.")
    return (f"Your answer must be one of the candidate letters: {', '.join(cands)}. Based on your reasoning "
            f"above, reply with only the line 'ANSWER: X' for the candidate it supports.")


def judge_restrict(client, model: str, system: str, user: str, text: str, cands: list[str],
                   n_options: int) -> str:
    """A judge's stored reply with its answer kept to the candidates: unchanged when it
    already names one (or there are none); else one low-effort pick prompt, whose answer is
    appended (JUDGE_MARK + an ANSWER line, which readers take as the last one) if it names a
    candidate. Otherwise the reply stands as it is. Never raises."""
    with _stats_lock:
        JUDGE_STATS["calls"] += 1
        if not cands:
            JUDGE_STATS["no_candidates"] += 1
    if not cands or extract_letter(text, n_options) in cands:
        return text
    pick = _followup_v3(client, model, system, user, text or "(no reply)", judge_pick_nudge(cands),
                        COMMIT_TOKENS_V3)
    if pick and on_followup is not None:
        on_followup()
    got = extract_letter(pick, n_options) or commit_letter_lenient(pick, n_options)
    with _stats_lock:
        JUDGE_STATS["repicked" if got in cands else "off_list"] += 1
    return text + JUDGE_MARK + f"ANSWER: {got}" if got in cands else text


def turn_cost(personas) -> int:
    """Speaker turns a round of low-effort speakers counts as: one per speaker."""
    return len(list(personas))


# The solver's system prompt: a general one (2026-10-06), with SOLVER_STEP added on a high-effort
# call (thinking on) and SOLVER_EXPLAIN on every call. Under v3 with multiple choice its user message is SuperGPQA's own prompt
# (official_question), so the solver gets the request of the external direct baseline plus this
# system prompt. Until 2026-10-06 the solver was "You are a Solver answering a hard graduate-level
# multiple-choice question. Reason step by step from your own knowledge, then ..." with the answer
# instruction.
SOLVER_SYSTEM = ("You are a helpful assistant with broad expert knowledge. You answer hard questions from "
                 "every field of study accurately, using your own knowledge. Read each question carefully "
                 "before you answer.")
SOLVER_STEP = " Think step by step."
# The reply must carry the reasoning: on a low-effort call (thinking off) a short reply is shown to later
# speakers as it is, and on a high-effort call the summary later speakers read sees only the end of
# the thinking (the last 16,000 characters; Qwen3.5-9B's thinking ran longer in 63% of its Direct CoT
# replies) besides the reply. This is the sentence the external baselines use to ask for it
# (baselines/tasks.py, EXPLAIN_SENTENCE).
SOLVER_EXPLAIN = " Explain your reasoning in your response before you give the final answer."


# The expert reasons step by step only on a high-effort call (user, 2026-10-06); on a low-effort call
# that sentence is EXPERT_LOW instead (speaker_system).
EXPERT_STEP = "Reason step by step from your expert knowledge of the field."
EXPERT_LOW = "Answer from your expert knowledge of the field."


def solver_system(system: str, effort: str) -> str:
    """A solver's system prompt for one call: `system` (SOLVER_SYSTEM, or a per-question override),
    with SOLVER_STEP added on a high-effort call, then SOLVER_EXPLAIN on every call."""
    return system + (SOLVER_STEP if effort == "high" else "") + SOLVER_EXPLAIN


def speaker_system(persona: str, system: str, effort: str) -> str:
    """The system prompt of one call: the solver's by solver_system; the expert's with EXPERT_STEP
    replaced by EXPERT_LOW on a low-effort call; every other persona's as it is."""
    if persona == OFFICIAL_PERSONA:
        return solver_system(system, effort)
    if persona == "expert" and effort != "high":
        return system.replace(EXPERT_STEP, EXPERT_LOW)
    return system
# The critic reads the whole discussion, not only the leading answer (2026-10-06; it replaced both
# the original critic and the E7 critic of schema_fitness, which challenged the leading answer).
CRITIC_BODY = (
    "You are a Critic. Read the whole discussion so far: the reasoning and the answer of every speaker "
    "in every round. Check each line of reasoning for errors: wrong facts, misapplied rules or formulas, "
    "calculation mistakes, unstated assumptions, overlooked constraints, and options that were rejected "
    "without a good reason. Do not prefer an answer because more speakers gave it. Then give the answer "
    "that the reasoning best supports, and explain why. ")
CRITIC_BODY_OPEN = CRITIC_BODY.replace("and options that were rejected", "and answers that were rejected")


def _build_prompts(instr: str) -> tuple[str, dict[str, str]]:
    """The expert template and the persona system prompts, built around one
    answer instruction. Called at import (v1) and by set_v2. No prompt uses the
    word 'commit' (2026-10-06)."""
    if OPEN:
        return _build_prompts_open(instr)
    expert = ("You are a leading expert in {field}, answering a hard graduate-level "
              "multiple-choice question inside your own specialty. " + EXPERT_STEP + " Explain your "
              "reasoning in your response before you give the final answer. " + instr +
              " Always choose exactly one letter; never abstain.")
    prompts = {
        # a general system prompt: under v3 the solver's user message is the dataset's own prompt
        # (SUPERGPQA_PROMPT), the request of the external direct baseline
        "solver": SOLVER_SYSTEM,
        "critic": CRITIC_BODY + instr + " Always choose exactly one letter; never abstain.",
        "synthesizer": ("You are a Synthesizer. Weigh the prior answers and their "
                        "reasoning, resolve the disagreements, and choose the single "
                        "best option. Explain your choice in your response. You MUST choose "
                        "exactly one letter. " + instr),
        "contrarian": ("You are a Contrarian. The standing answer is probably wrong. Build the "
                       "strongest positive case for a DIFFERENT option: find the reading of the "
                       "question, or the piece of knowledge, under which another option is the "
                       "correct one. You may not endorse the standing answer. " + instr),
        "eliminator": ("You are an Eliminator. Do NOT answer the question. Work through the "
                       "options and rule out the ones that cannot be correct, giving a specific "
                       "reason for each — a violated constraint, a wrong magnitude, a definition "
                       "that does not match. End with a line listing the options that survive: "
                       "'SURVIVING: <letters>'. Never write an 'ANSWER:' line."),
        "independent": ("You are an Independent Solver working alone on a hard graduate-level "
                        "multiple-choice question. Reason step by step from your own knowledge "
                        "and give your answer. " + instr + " Always choose exactly one letter."),
        "verifier": ("You are a Verifier. Do not re-derive the solution from scratch. Take each "
                     "answer given so far and test it directly against the exact "
                     "wording of the question: does that option satisfy every requirement the "
                     "question states? Test the most likely other option the same way. Keep the "
                     "answer if it satisfies every requirement; otherwise change to the option that "
                     "does. Show each check in your response. " + instr
                     + " Always choose exactly one letter; never abstain."),
        "expert": expert.format(field="the question's field"),
    }
    if JUDGE_ON:
        prompts[JUDGE_PERSONA] = (JUDGE_PROMPT.format(what="option") + instr
                                  + " Always choose exactly one letter; never abstain.")
    return expert, prompts


def _build_prompts_open(instr: str) -> tuple[str, dict[str, str]]:
    """The personas of _build_prompts for open-answer questions (set_open_answers):
    the same roles, with 'answer' where they said 'option' or 'letter'."""
    one = " Always give exactly one final answer; never abstain."
    expert = ("You are a leading expert in {field}, answering a hard expert-level question inside "
              "your own specialty. " + EXPERT_STEP + " Explain your reasoning in your response before you "
              "give the final answer. " + instr.replace("{", "{{").replace("}", "}}") + one)
    # (the template is filled with .format(field=...): MATH's '\\boxed{}' must not read as a field)
    prompts = {
        # a general system prompt; the user message carries the question and the answer format
        "solver": SOLVER_SYSTEM,
        "critic": CRITIC_BODY_OPEN + instr + one,
        "synthesizer": ("You are a Synthesizer. Weigh the prior answers and their reasoning, resolve the "
                        "disagreements, and give the single best answer. Explain your choice in your "
                        "response. You MUST give exactly one final answer. " + instr),
        "contrarian": ("You are a Contrarian. The standing answer is probably wrong. Build the strongest "
                       "positive case for a DIFFERENT answer: find the reading of the question, or the "
                       "piece of knowledge, under which another answer is the correct one. You may not "
                       "endorse the standing answer. " + instr),
        "eliminator": ("You are an Eliminator. Do NOT answer the question. Work through the candidate "
                       "answers and rule out the ones that cannot be correct, giving a specific reason for "
                       "each. End with a line listing the candidates that survive: 'SURVIVING: <list>'. "
                       "Never write an 'ANSWER:' line."),
        "independent": ("You are an Independent Solver working alone on a hard expert-level question. "
                        "Reason step by step from your own knowledge and give your answer. " + instr + one),
        "verifier": ("You are a Verifier. Do not re-derive the solution from scratch. Take each answer "
                     "given so far and test it directly against the exact wording of the question: "
                     "does it satisfy every requirement the question states? Test the most likely other "
                     "answer the same way. Keep the answer if it satisfies every requirement; otherwise "
                     "change to the answer that does. Show each check in your response. " + instr + one),
        "expert": expert.format(field="the question's field"),
    }
    if JUDGE_ON:
        prompts[JUDGE_PERSONA] = JUDGE_PROMPT.format(what="answer") + instr + one
    return expert, prompts


def set_v2(on: bool) -> None:
    """Switch the executor between v1 (original prompts, one call per persona)
    and v2. Rebuilds the persona prompts in place so every consumer that reads
    the module globals at call time sees the change. The critic override that
    adaptive_debate_mcq passes per question lives in schema_fitness; use
    schema_fitness.set_v2 from entry scripts so both are switched together."""
    global V2
    V2 = bool(on)
    _rebuild_prompts()


def _rebuild_prompts() -> None:
    """The commitment instruction and every persona prompt, from the current
    V2 and VISIBLE_REASONING switches. With VISIBLE_REASONING off this is
    exactly what set_v2 always built."""
    global ANSWER_INSTR, EXPERT_TMPL, PERSONA_PROMPTS, COMMIT_NUDGE
    if OPEN and MATH:
        ANSWER_INSTR = ((ANSWER_INSTR_MATH_PLAIN if PLAIN_INSTR else ANSWER_INSTR_MATH)
                        + (VISIBLE_SENTENCE_MATH if VISIBLE_REASONING else ""))
        COMMIT_NUDGE = COMMIT_NUDGE_MATH
    elif OPEN:
        ANSWER_INSTR = ((ANSWER_INSTR_OPEN_PLAIN if PLAIN_INSTR else ANSWER_INSTR_OPEN)
                        + (VISIBLE_SENTENCE_OPEN if VISIBLE_REASONING else ""))
        COMMIT_NUDGE = COMMIT_NUDGE_OPEN
    else:
        ANSWER_INSTR = ANSWER_INSTR_PLAIN if PLAIN_INSTR else ANSWER_INSTR_V2 if V2 else ANSWER_INSTR_V1
        if VISIBLE_REASONING:
            ANSWER_INSTR = ANSWER_INSTR + VISIBLE_SENTENCE
        COMMIT_NUDGE = COMMIT_NUDGE_LETTER
    EXPERT_TMPL, PERSONA_PROMPTS = _build_prompts(ANSWER_INSTR)


def set_visible_reasoning(on: bool, detailed: bool = False) -> None:
    """Use schema_fitness.set_visible_reasoning from entry scripts, so the
    per-question critic override changes with the persona prompts. `detailed`
    picks VISIBLE_SENTENCE_DETAILED over the original sentence."""
    global VISIBLE_REASONING, VISIBLE_SENTENCE
    VISIBLE_REASONING = bool(on)
    VISIBLE_SENTENCE = VISIBLE_SENTENCE_DETAILED if detailed else VISIBLE_SENTENCE_BRIEF
    _rebuild_prompts()


def set_plain_instruction(on: bool) -> None:
    """The plain commitment instruction (ANSWER_INSTR_PLAIN) in place of the v1/v2 one,
    for every persona. Use program_space.configure_executor (--plain-instruction) from
    entry scripts, which also switches the critic override."""
    global PLAIN_INSTR
    PLAIN_INSTR = bool(on)
    _rebuild_prompts()


def plain_signature() -> str | None:
    """'1' with the plain instruction on, else None. Part of the cache key."""
    return "1" if PLAIN_INSTR else None


# --- the solver's user message: the dataset's own prompt -------------------------------------
# Under v3 with multiple-choice questions, a solver speaker's user message is SuperGPQA's own
# zero-shot prompt with the question (baselines/tasks.py, PROMPT_TEMPLATE and build_prompt): the
# request of the external direct baseline, with the general system prompt SOLVER_SYSTEM before it.
# A solver that sees the discussion gets the same prompt, then the discussion. Measured on
# Qwen3.5-9B (cluster-pipeline run1, test), a high-effort solver under the earlier Solver prompt used about 1.6
# times the external direct baseline's tokens (its summary included) and scored about 2 points
# lower (55.4% against 57.4%). Since 2026-10-06; open answers keep their own user message.
SUPERGPQA_PROMPT = (
    "Answer the following multiple choice question. There is only one correct answer. "
    "The last line of your response should be in the format 'Answer: $LETTER' (without quotes), "
    "where LETTER is one of A, B, C, D, E, F, G, H, I, or J.\n\n{}\n"
)
OFFICIAL_PERSONA = "solver"


def official_question(question: str, options: list[str]) -> str:
    """SuperGPQA's zero-shot prompt for one question, as baselines/tasks.build_prompt writes it."""
    body = question + "\n" + "\n".join(f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options))
    return SUPERGPQA_PROMPT.format(body)


# --- summary tokens kept apart (off by default; set_summary_split) ---------------------------
# With it on, a speaker's token tally also records, as "summary", the completion tokens of its
# answer-locked summary call (the summary of a reply that committed a letter). The round runner
# keeps them apart, so evolve_program_mcq (--count-read-summaries) can leave out the summary of a
# speaker that no later speaker read: a summary is only for later speakers, and it never changes the
# answer. A summary that gives an uncommitted reply its letter is not split off, because the speaker
# needs it to answer at all. Nothing that a speaker is sent or says changes.
SUMMARY_SPLIT = False


def set_summary_split(on: bool) -> None:
    """Use program_space.configure_executor (--count-read-summaries) from entry scripts."""
    global SUMMARY_SPLIT
    SUMMARY_SPLIT = bool(on)


def visible_signature() -> str | None:
    """'1' when visible reasoning is on ('d' with the detailed sentence), else
    None. Part of the cache key."""
    if not VISIBLE_REASONING:
        return None
    return "d" if VISIBLE_SENTENCE == VISIBLE_SENTENCE_DETAILED else "1"


def v2_signature() -> str | None:
    """'2' when v2 is on, '3' under v3, else None. Part of the cache key."""
    return "3" if V3 else "2" if V2 else None


def _split_summary(text: str) -> tuple[str, str | None]:
    """(full reply, summary or None). rsplit so a model that happens to emit
    the marker text inside its derivation does not win."""
    if SUMMARY_MARK not in (text or ""):
        return text or "", None
    full, summary = text.rsplit(SUMMARY_MARK, 1)
    return full, summary


def summarize_reply(client, model: str, system: str, user: str, full: str,
                    temperature: float) -> str:
    """The summary call: same conversation plus the nudge. Returns the model's
    text, or '' on any failure (never raises: the round runner would retry the
    whole round and throw away every persona's full reply)."""
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": full},
                {"role": "user", "content": SUMMARY_NUDGE}]
    key = (id(client), model)
    with _think_lock:
        known = _think_mode.get(key)
    extras = [known] if known is not None else list(_THINK_OFF)
    for extra in extras:
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
                max_tokens=SUMMARY_MAX_TOKENS, extra_body=extra)
            _count_tokens(resp)
            text = _message_text(resp.choices[0].message)
        except Exception:
            continue
        if text:
            if on_summary is not None:
                on_summary()
            return text.strip()
    return ""


def _reconcile(full: str, summary: str, n_options: int) -> str:
    """The text to store: full + marker + summary, with the summary's last
    ANSWER line guaranteed to name the committed letter. The full reply's
    letter is authoritative when it exists; otherwise the summary's letter is
    the commitment (that is how a cut-off reply gets recovered)."""
    l_full = extract_letter(full, n_options)
    l_sum = extract_letter(summary, n_options)
    if l_full is not None:
        if l_sum != l_full:
            if l_sum is not None:
                _bump("disagreements")
            summary = _strip_commitment(summary) + f"\nANSWER: {l_full}"
    elif l_sum is None:
        lenient = commit_letter_lenient(summary, n_options)
        if lenient is not None:
            summary += f"\nANSWER: {lenient}"
            _bump("recovered_commits")
    else:
        _bump("recovered_commits")
    return full + SUMMARY_MARK + summary


def _is_own_summary(full: str) -> bool:
    """A reply short enough to be shown to later speakers as it is, with no
    summary call. Under the original 120-word summary that is the original
    character test. Under a longer limit it is any reply within the limit: a
    summary of it could only repeat it or, as seen live, pad it with analysis
    the speaker never did (a 180-word reply got a 625-word 'summary'). Later
    speakers then read the reply itself, so the digest window must hold a reply
    of that length. Deep-think is not routed through here: its reasoning is in
    the hidden thinking however short its visible reply, so it is always summarised."""
    if SUMMARY_WORDS == 120:
        return len(full) < V2_SHORT_CHARS
    return len(full.split()) <= SUMMARY_WORDS


def chat_v2(client, model: str, system: str, user: str, temperature: float,
            max_tokens: int, n_options: int) -> str:
    """Reasoning call, then the summary call, then reconciliation."""
    full = chat(client, model, system, user, temperature, max_tokens)
    if not full:
        return full
    if (_is_own_summary(full) and extract_letter(full, n_options) is not None
            and not (VISIBLE_REASONING and len(full) < VISIBLE_MIN_CHARS)):
        _bump("skipped_short")
        return full
    summary = summarize_reply(client, model, system, user, full, temperature)
    if not summary:
        _bump("failed_summaries")
        return full
    _bump("summaries")
    return _reconcile(full, summary, n_options)


# --- v3: effort per round, no reply cap, model-card sampling, answer-locked summaries ----
# The v2 prompts, with five changes to how each speaker is called:
#
#   effort    every round is low or high effort, sent explicitly on every request (gpt-oss:
#             reasoning_effort; Qwen: thinking off / on), so nothing depends on a server
#             default. A high-effort speaker counts HIGH_COST turns. It replaced the v2 deep-think speaker.
#   no cap    a reply may use the whole context window, less its prompt and a reserve kept
#             for the summary call that follows it (SUMMARY_RESERVE). The summaries keep the
#             context of later speakers small, so replies do not need a cap of their own.
#   sampling  each model card's settings for the effort level (SAMPLING). The run's
#             --temperature is not used.
#   summary   a committed reply of at most SUMMARY_WORDS words is shown to later speakers as
#             it is. Otherwise the summary prompt names the reply's letter and forbids changing
#             it; a summary whose ANSWER line names another letter is not used (later speakers
#             see the reply). A high-effort reply, or a reply too short to carry any reasoning,
#             is summarised from the end of its hidden reasoning plus its visible reply.
#   commit    no reply is left without a letter if it can be helped: an uncommitted reply's
#             summary prompt asks for the letter FIRST; failing that, COMMIT_NUDGE asks for the
#             letter alone, and only that answer is read leniently. A letter is never taken
#             from a summary's prose.
# v3 is part of the round cache key ("v":"3", and the window as "w").

V3 = False
EFFORTS = ("low", "high")
DEFAULT_EFFORT = "low"
HIGH_COST = DEFAULT_HIGH_COST     # turns one high-effort speaker counts as;
                                  # program_space.configure_v3 sets it (--high-cost)
WINDOW = 32768                    # the server's context window (max_model_len); set_v3 sets it
SUMMARY_RESERVE = 2048            # tokens a reply leaves free: the summary call's prompt adds the
                                  # nudge (~250 tokens) to the reply, and its output is below
SUMMARY_TOKENS_V3 = 1792          # the summary call's limit (on gpt-oss it includes its own thinking)
COMMIT_TOKENS_V3 = 512            # the commit prompt's limit (gpt-oss thinks first even at low effort)
# A reply's limit needs its prompt's size. The server's own count is used (vLLM's /tokenize with the
# chat messages: it equalled the billed prompt tokens exactly on gpt-oss), plus a small margin for
# template differences (Qwen's thinking tags). Only if that cannot be had is the size estimated
# from characters, below the densest text measured (debate replies and questions: median 4.3
# characters a token, 5% under 2.45, least 1.77). This matters: an undercount would let a reply eat
# the room its summary needs. (vLLM 0.18, the version the servers run, refuses a request whose
# prompt + max_tokens is over its window; _complete then asks once more with the room it reports.)
CHARS_PER_TOKEN = 1.75
PROMPT_MARGIN = 64
REPLY_TIMEOUT = 3600.0            # a whole-window reply on a busy server takes many minutes
MIN_REPLY_TOKENS = 512

# Model-card sampling per effort. top_k and min_p are vLLM extensions and go in extra_body.
SAMPLING: dict[str, dict[str, dict]] = {
    # OpenAI: temperature 1.0, top_p 1.0, whatever the reasoning effort
    "gpt-oss": {"low": {"temperature": 1.0, "top_p": 1.0},
                "high": {"temperature": 1.0, "top_p": 1.0}},
    # Qwen3.5 card: thinking off (reasoning tasks) and thinking on (general)
    "qwen3.5": {"low": {"temperature": 1.0, "top_p": 1.0, "top_k": 40, "min_p": 0.0, "presence_penalty": 2.0},
                "high": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}},
}


def model_family(model: str) -> str:
    m = model.lower()
    if "gpt-oss" in m:
        return "gpt-oss"
    if "qwen3.5" in m:
        return "qwen3.5"
    raise ValueError(f"no v3 sampling settings for {model!r}; add its family to debate_mcq.SAMPLING")


def request_params(model: str, effort: str) -> dict:
    """The sampling and effort fields of one v3 request, as create() keyword arguments."""
    if effort not in EFFORTS:
        raise ValueError(f"effort must be one of {EFFORTS}, got {effort!r}")
    fam = model_family(model)
    params = dict(SAMPLING[fam][effort])
    extra = {k: params.pop(k) for k in ("top_k", "min_p") if k in params}
    if fam == "gpt-oss":
        extra["reasoning_effort"] = effort
    else:
        extra["chat_template_kwargs"] = {"enable_thinking": effort == "high"}
    return {**params, "extra_body": extra}


def set_v3(on: bool, window: int | None = None) -> None:
    """Switch the v3 executor on (it keeps the v2 prompts) or off. Use
    program_space.configure_executor from entry scripts."""
    global V3, V2, WINDOW
    V3 = bool(on)
    if on:
        V2 = True
        if window:
            WINDOW = int(window)
    _rebuild_prompts()


def set_high_cost(n: int) -> None:
    """The turns one high-effort speaker counts as. Use program_space.configure_executor
    (--high-cost) from entry scripts. Cost decides which actions fit under the per-question
    cap, never what a speaker says, so it is not part of the cache key."""
    global HIGH_COST
    if int(n) < 1:
        raise ValueError(f"a high-effort speaker must count as at least 1 turn, not {n}")
    HIGH_COST = int(n)


def window_signature() -> int | None:
    return WINDOW if V3 else None


def spec_cost(spec: dict) -> int:
    """Speaker turns one round counts as: one per speaker, HIGH_COST for a
    high-effort speaker."""
    per = HIGH_COST if spec.get("effort", DEFAULT_EFFORT) == "high" else 1
    return per * len(spec["personas"])


def _est_tokens(messages: list[dict]) -> int:
    """Conservative size of a prompt from its characters (the fallback of _prompt_tokens)."""
    return int(sum(len(m["content"]) for m in messages) / CHARS_PER_TOKEN) + 16 * len(messages) + 64


def _prompt_tokens(client, model: str, messages: list[dict]) -> int:
    """The prompt's size in tokens: the server's count (vLLM /tokenize) plus
    PROMPT_MARGIN, or _est_tokens if the server cannot be asked (a transient
    failure falls back for that one call only)."""
    base = str(getattr(client, "base_url", "") or "").rstrip("/")
    if base:
        root = base[:-3] if base.endswith("/v1") else base
        try:
            req = urllib.request.Request(
                root + "/tokenize", data=json.dumps({"model": model, "messages": messages}).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {getattr(client, 'api_key', '') or 'EMPTY'}"})
            with urllib.request.urlopen(req, timeout=60) as fh:
                return int(json.loads(fh.read())["count"]) + PROMPT_MARGIN
        except Exception:
            pass
    return _est_tokens(messages)


# how vLLM reports the prompt size when it refuses a request that does not fit its window
_PROMPT_SIZE = (re.compile(r"request has (\d+) input tokens"), re.compile(r"\((\d+) in the messages"),
                re.compile(r"Input length \((\d+)\)"),
                re.compile(r"prompt contains (?:at least )?(\d+) input tokens"))   # vLLM 0.18


def _complete(client, model: str, messages: list[dict], effort: str, max_tokens: int):
    """One v3 request. A server that refuses prompt + max_tokens over its
    window (vLLM 0.18: "...your prompt contains N input tokens...") is asked once more with the
    limit worked out from the prompt size it reported; a prompt that does not
    fit on its own is an error."""
    params = request_params(model, effort)
    try:
        resp = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens,
                                              timeout=REPLY_TIMEOUT, **params)
    except Exception as exc:
        text = str(exc)
        if "context length" not in text and "context window" not in text:
            raise
        n = next((int(m.group(1)) for pat in _PROMPT_SIZE if (m := pat.search(text))), None)
        if n is None or n + 64 > WINDOW:
            raise
        reserve = SUMMARY_RESERVE if max_tokens > SUMMARY_TOKENS_V3 else 0     # only a reply keeps one
        resp = client.chat.completions.create(model=model, messages=messages,
                                              max_tokens=max(64, min(WINDOW - n - reserve, max_tokens)),
                                              timeout=REPLY_TIMEOUT, **params)
    _count_tokens(resp)
    return resp


def _parts(resp) -> tuple[str, str]:
    """(hidden reasoning, visible reply) of a response; either may be ''."""
    msg = resp.choices[0].message
    thinking = (getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or "").strip()
    return thinking, (getattr(msg, "content", None) or "").strip()


_SUMMARY_BODY_V3 = ("Future participants see ONLY this summary, never your full reasoning, so make it "
                    "complete enough that a reader could follow and check your whole argument from it. "
                    "Cover: the decisive facts or steps in the order you used them, every calculation with "
                    "its numbers and intermediate results, the assumptions you made, the options you ruled "
                    "out and why, and the point you are least sure of.")


_SUMMARY_BODY_OPEN = _SUMMARY_BODY_V3.replace("the options you ruled out", "the alternatives you ruled out")


def summary_nudge_v3(letter: str | None) -> str:
    """The summary prompt: answer-locked for a committed reply, letter-first
    for an uncommitted one (with open answers, the answer in place of the letter)."""
    body = _SUMMARY_BODY_OPEN if OPEN else _SUMMARY_BODY_V3
    if OPEN and letter is None:
        return ("Your reply ended before it gave an answer. Based only on the reasoning above, "
                "begin with the line 'ANSWER: <final answer>', giving the final answer that reasoning "
                "supports in the required format (for a question with answer choices, only the letter). "
                f"Then summarise that reasoning for future participants, in at most {SUMMARY_WORDS} "
                f"words. {body} Do not add new analysis.")
    if letter is not None:
        return (f"Now write a summary of your reasoning above for future participants, in at most "
                f"{SUMMARY_WORDS} words. {body} Your answer is final: {letter}. This summary "
                f"reports the reasoning in your reply above. It must not reconsider, change or argue "
                f"against that answer, and it must not add new analysis. End with exactly one line "
                f"'ANSWER: {letter}'.")
    return ("Your reply ended before it gave an answer. Based only on the reasoning above, "
            "begin with the line 'ANSWER: X', where X is the single option letter that reasoning "
            f"supports. Then summarise that reasoning for future participants, in at most "
            f"{SUMMARY_WORDS} words. {_SUMMARY_BODY_V3} Do not add new analysis.")


def _answer_letters(text: str, n_options: int) -> list[str]:
    """The letters of every valid 'ANSWER: X' in `text`, in order, markdown
    emphasis ignored ('**Answer:** B' counts). With open answers, every ANSWER
    line's normalised answer (MATH answers: every answer's key, math_answers)."""
    if OPEN:
        return math_answers(text) if MATH else open_answers(text)
    return [m.upper() for m in _ANS.findall((text or "").replace("*", ""))
            if LETTERS.index(m.upper()) < n_options]


def _strip_answer_lines(text: str) -> str:
    """`text` without its ANSWER lines, markdown emphasis ignored when finding them."""
    if OPEN:
        return _ANS_OPEN.sub("", text or "").strip()
    return "\n".join(l for l in (text or "").splitlines() if not _ANS.search(l.replace("*", ""))).strip()


def _opening(system: str | None, user: str) -> list[dict]:
    """The first messages of a v3 request: the system prompt (none for the official solver), then
    the user message."""
    return ([] if system is None else [{"role": "system", "content": system}]) + [{"role": "user", "content": user}]


def _followup_v3(client, model: str, system: str | None, user: str, shown: str, nudge: str,
                 max_tokens: int) -> str:
    """A summary or commit call at low effort: the same conversation, the reply
    as `shown`, then the nudge. Only the visible text counts: hidden reasoning
    returned in its place is a failed call. A request the server refuses ('' is
    returned) is part of the reply's record; a server that cannot be reached, times
    out or fails (since 2026-10-06) raises, so the round is not recorded and is run
    again, instead of being kept for good without its summary or letter."""
    messages = _opening(system, user) + [{"role": "assistant", "content": shown}, {"role": "user", "content": nudge}]
    try:
        _, text = _parts(_complete(client, model, messages, "low", max_tokens))
    except Exception as exc:
        if _server_trouble(exc):
            raise
        return ""
    return text


def _server_trouble(exc: Exception) -> bool:
    """A connection failure, a timeout, a rate limit or a server-side (5xx) error: not an answer
    about this request, so the round must be run again."""
    import openai
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError,
                        openai.InternalServerError)):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and status >= 500


def chat_v3(client, model: str, system: str | None, user: str, n_options: int,
            effort: str = DEFAULT_EFFORT, info: dict | None = None) -> str:
    """One speaker under v3. Returns the text to store: the visible reply, or
    reply + SUMMARY_MARK + what later speakers read. Its last ANSWER line is the
    speaker's letter (none if it never committed). `info`, if given, is filled
    with what happened: the reply's size, its finish reason, and the path taken
    (a V2_STATS name: skipped_short, summaries, rejected_summaries, ...).
    `system` None sends no system message (the official solver prompt)."""
    def end(stat: str, text: str) -> str:
        _bump(stat)
        if info is not None:
            info["path"] = stat
        return text

    messages = _opening(system, user)
    room = max(MIN_REPLY_TOKENS, WINDOW - _prompt_tokens(client, model, messages) - SUMMARY_RESERVE)
    resp = _complete(client, model, messages, effort, room)
    thinking, visible = _parts(resp)
    letter = extract_letter(visible, n_options)
    if (letter is None and not OPEN
            and (bold := extract_letter(visible.replace("*", ""), n_options)) is not None):
        # 'Answer: **H**' / '**Answer:** H' (gpt-oss: 9 of 200 replies in the probe, 3 low effort and
        # 6 high): a commitment in markdown; stored with a plain ANSWER line so every reader finds it
        letter, visible = bold, visible + f"\nANSWER: {bold}"
    if info is not None:
        info.update(visible_words=len(visible.split()), thinking_chars=len(thinking), max_tokens=room,
                    finish_reason=getattr(resp.choices[0], "finish_reason", None), committed=letter is not None)
    if not (thinking or visible):
        return end("uncommitted", "")
    # the reasoning is in the hidden channel: high effort, or a reply too short to carry it
    hidden = bool(thinking) and (effort == "high" or len(visible) < VISIBLE_MIN_CHARS)
    if letter is not None and not hidden and len(visible.split()) <= SUMMARY_WORDS:
        return end("skipped_short", visible)
    tail = thinking[-THINKING_TAIL:] if hidden else ""
    shown = f"[the end of my private reasoning]\n{tail}\n\n[my reply]\n{visible}" if tail else visible
    full = visible or "(no reply: the reasoning ran out of room)"
    tally = getattr(_tokens, "tally", None)
    before = tally["completion"] if tally is not None else 0
    summary = _followup_v3(client, model, system, user, shown, summary_nudge_v3(letter), SUMMARY_TOKENS_V3)
    if SUMMARY_SPLIT and letter is not None and tally is not None:     # a summary for later speakers only
        tally["summary"] = tally.get("summary", 0) + tally["completion"] - before
    if summary and on_summary is not None:
        on_summary()
    said = _answer_letters(summary, n_options)
    if letter is not None:
        if not summary:
            return end("failed_summaries", full)
        if any(l != letter for l in said):
            return end("rejected_summaries", full)       # it argued for another letter: show the reply
        return end("summaries", full + SUMMARY_MARK + _strip_answer_lines(summary) + f"\nANSWER: {letter}")
    if said:                                             # letter-first: the first ANSWER line commits
        return end("recovered_commits", full + SUMMARY_MARK + _strip_answer_lines(summary) + f"\nANSWER: {said[0]}")
    commit = _followup_v3(client, model, system, user, shown, COMMIT_NUDGE, COMMIT_TOKENS_V3)
    if commit and on_followup is not None:
        on_followup()
    got = extract_letter(commit, n_options) or commit_letter_lenient(commit, n_options)
    shown_part = SUMMARY_MARK + summary if summary else ""
    if got is None:
        return end("uncommitted", full + shown_part)
    return end("nudged_commits", full + (shown_part or SUMMARY_MARK) + COMMIT_MARK + f"ANSWER: {got}")


# --- schema grammar --------------------------------------------------------

PERSONAS = ("solver", "critic", "synthesizer")          # the ORIGINAL grammar; unchanged so that
FINALS = ("last", "vote", "synthesizer")                # evolve_debate_mcq.py stays reproducible
MAX_ROUNDS = 6
MAX_PERSONAS = 4
EDIT_ACTIONS = ("add_round", "modify_round", "remove_round", "set_final", "give_up")

# --- genome extension (E1) -------------------------------------------------
# Two new genes, both motivated by the departure decomposition: accuracy = D x P.
#
#   sees  -- per-round visibility. Until now every persona saw every prior persona's
#            COMMITTED LETTER plus its reasoning, so each added round was another
#            chance to anchor on a wrong commitment. This gene makes anchoring an
#            evolvable choice rather than a fixed property of the executor.
#   personas -- three that shape departure directly: `contrarian` may not re-select
#            the standing letter, `eliminator` rules options out without answering,
#            `independent` answers with no visibility at all.
#
# Old schemas carry no `sees` key and only the original three personas, so they
# validate and execute exactly as before.

SEES = ("all", "none", "letters_only", "no_letters", "last_round")
DEFAULT_SEES = "all"
EXT_PERSONAS = PERSONAS + ("contrarian", "eliminator", "independent")
NON_ANSWERING = ("eliminator",)                  # contributes reasoning, commits no letter
FORCED_BLIND = ("independent",)                  # ignores the round's `sees` and sees nothing

# --- per-question extension (treegrow / adaptive depth) ---------------------
# Two personas for the per-question methods. Deliberately NOT in EXT_PERSONAS:
# adding them there would silently widen the CEM / MAP-Elites gene pools and
# invalidate those searches' learned distributions. validate() accepts them.
#
#   verifier -- audits the ANSWER, not the reasoning: tests each committed letter
#            against the literal wording of the question, then keeps or switches.
#   expert   -- field-named authority solver. The static prompt is a fallback;
#            callers inject the question's real field per question via
#            prompts={"expert": EXPERT_TMPL.format(field=row["field"])}.
NEW_PERSONAS = ("verifier", "expert")

# The persona prompts, built around the v1 instruction at import. set_v2 rebuilds
# them; the texts themselves live in _build_prompts so v1 and v2 differ only in
# the commitment instruction.
EXPERT_TMPL, PERSONA_PROMPTS = _build_prompts(ANSWER_INSTR)

MINIMAL_SCHEMA = {"rounds": [{"personas": ["solver"]}], "final": "last"}


# --- fixed templates (as schemas, so eval and evolve share the executor) ---

def self_critique_schema() -> dict:
    """Solver -> critic -> revised solver."""
    return {"rounds": [{"personas": ["solver"]}, {"personas": ["critic"]},
                       {"personas": ["solver"]}], "final": "last"}


def fixed_debate_schema(k: int = 3) -> dict:
    """k solvers answer independently, see each other and reconsider, a synthesizer commits."""
    k = max(1, min(k, MAX_PERSONAS))
    return {"rounds": [{"personas": ["solver"] * k}, {"personas": ["solver"] * k},
                       {"personas": ["synthesizer"]}], "final": "synthesizer"}


def build_fixed_schema(name: str, debate_k: int = 3) -> dict:
    if name == "self_critique":
        return self_critique_schema()
    if name == "fixed_debate":
        return fixed_debate_schema(debate_k)
    raise KeyError(f"unknown fixed template {name!r}")


# --- executor: run a schema, return the committed LETTER --------------------

# How much of a prior persona's response a later one is shown. The default
# (700 head, no tail) is the original behaviour and is what every Qwen3-14B
# recording was produced under -- do not change it for those runs.
#
# Why a tail is worth having: measured over 60k recorded responses, 78% run
# past 700 characters and the committed conclusion sits at 98-99% of the text,
# so a head-only window keeps the setup and discards the payoff. The
# eliminator was the extreme case: its SURVIVING line reached the next
# persona 2% of the time.
DIGEST_HEAD = 700
DIGEST_TAIL = 0


def set_digest(head: int, tail: int = 0) -> None:
    """Set the visible window for this process. Also changes the round cache
    key (see schema_fitness.path_key), so recordings made under different
    windows can never be replayed as if they were the same."""
    global DIGEST_HEAD, DIGEST_TAIL
    DIGEST_HEAD, DIGEST_TAIL = head, tail


def digest_signature() -> tuple[int, int] | None:
    """(head, tail) when non-default, else None. Part of the cache key."""
    return None if (DIGEST_HEAD, DIGEST_TAIL) == (700, 0) else (DIGEST_HEAD, DIGEST_TAIL)


def _clip(text: str) -> str:
    """The visible part of one response: the opening, then the closing, with a
    marker where the middle was dropped. With tail=0 this is a plain head cut,
    byte-identical to what the original code produced."""
    t = (text or "").strip()
    head, tail = DIGEST_HEAD, DIGEST_TAIL
    if tail <= 0:
        return t[:head]
    if len(t) <= head + tail:
        return t
    return f"{t[:head]}\n[... {len(t) - head - tail} characters omitted ...]\n{t[-tail:]}"


def _shown(text: str) -> str:
    """What a later persona is shown of one prior reply: the v2 summary when the
    reply carries one (whole, no clipping), otherwise the head/tail clip."""
    full, summary = _split_summary(text)
    return summary.strip() if summary is not None else _clip(full)


def _digest(prior: list[tuple[str, str]], n: int) -> str:
    """Show each prior persona's chosen letter and its (bounded) reasoning so later
    personas can actually debate the reasoning, not just the letters."""
    return "\n\n".join(f"[{p}] chose {extract_letter(r, n) or '?'}:\n{_shown(r)}"
                       for p, r in prior)


def _strip_commitment(text: str) -> str:
    """Drop the 'ANSWER: X' lines so reasoning can be shown without the commitment."""
    return re.sub(r"^.*ANSWER\s*:\s*\(?\s*[A-J]\b.*$", "", text or "", flags=re.I | re.M).strip()


# --- the discussion as later speakers see it (2026-10-06) ---------------------------------------
# Round by round: each round names how many speakers it had and what they saw (only the question,
# or the rounds before it), so a reader knows that speakers of one round answered at the same time
# and never saw each other. Each speaker shows its answer, then its summary (or its reply, if short)
# without its answer line. A speaker that sees the discussion is also told which round it speaks in
# and how many others answer in that round at the same time. Until 2026-10-06 the earlier replies
# were one flat list under "Prior responses:" ("[solver] chose B: ..."), with no rounds.
DISCUSSION_HEAD = ("The discussion so far. It ran in rounds. Speakers in the same round answered at the same "
                   "time, so none of them saw the other replies of that round.")


def _round_saw(i: int, spec: dict | None) -> str | None:
    """What the speakers of round `i` (0-based) saw, from the round's spec; None if it is not known."""
    if spec is None:
        return None
    mode = spec.get("sees", DEFAULT_SEES)
    if i == 0 or mode == "none" or all(p in FORCED_BLIND for p in spec["personas"]):
        return "only the question"
    earlier = "round 1" if i == 1 else f"rounds 1 to {i}"
    if mode == "last_round":
        return f"round {i}"
    if mode == "letters_only":
        return f"only the answers of {earlier}"
    if mode == "no_letters":
        return f"the reasoning of {earlier}, without the answers"
    return earlier


def _drop_final_answer(text: str) -> str:
    """`text` without the answer line(s) at its end: the round's listing names the answer."""
    lines = (text or "").rstrip().splitlines()
    while lines and (_ANS_OPEN.match(lines[-1]) if OPEN else _ANS.search(lines[-1].replace("*", ""))):
        lines.pop()
    return "\n".join(lines).strip()


def _discussion(all_rounds: list[list[tuple[str, str]]], n: int, prior_specs: list[dict] | None = None,
                last_only: bool = False) -> str:
    """The discussion, round by round (only the last round with `last_only`). `prior_specs`, the
    specs of the rounds in `all_rounds`, tells what each round's speakers saw; without them that
    is left out."""
    blocks = []
    for i in range(len(all_rounds) - 1 if last_only else 0, len(all_rounds)):
        rnd = all_rounds[i]
        if not rnd:
            continue
        spec = prior_specs[i] if prior_specs is not None and i < len(prior_specs) else None
        k = len(rnd)
        head = f"Round {i + 1} ({k} speaker{'' if k == 1 else 's'}"
        if (saw := _round_saw(i, spec)) is not None:
            head += f"; {'it' if k == 1 else 'they'} saw {saw}"
        parts = [head + ")"]
        count, seen = Counter(p for p, _ in rnd), Counter()
        for p, r in rnd:
            seen[p] += 1
            tag = f"[Round {i + 1}, {p} {seen[p]}]" if count[p] > 1 else f"[Round {i + 1}, {p}]"
            if p not in NON_ANSWERING:
                tag += f" Answer: {extract_letter(r, n) or 'none'}"
            body = _drop_final_answer(_shown(r))
            parts.append(f"{tag}\n{body}" if body else tag)
        blocks.append("\n\n".join(parts))
    return DISCUSSION_HEAD + "\n\n" + "\n\n".join(blocks) if blocks else ""


def _where(round_no: int, k: int) -> str:
    """Told to a speaker that sees the discussion: the round it speaks in, and the other speakers
    of that round (`k` speakers in all), who answer at the same time."""
    out = f"You speak in round {round_no}."
    if k == 2:
        out += " 1 other speaker answers in this round at the same time; you do not see its reply."
    elif k > 2:
        out += f" {k - 1} other speakers answer in this round at the same time; you do not see their replies."
    return out


# A critic, verifier or synthesizer reviews the answers given so far. They never speak blind, so
# only as the first speaker of a debate is there nothing to review; the search does put them there
# (12-43% of the programs of past runs opened with one, and gpt-oss run2's best programs opened with a
# high-effort verifier). Such a speaker is told so, and works on the question itself (2026-10-06).
REVIEWERS = ("critic", "verifier", "synthesizer")
NOTHING_TO_REVIEW = ("No one has answered this question yet, so there is no discussion to read. Work on the "
                     "question and its options yourself, in your role, and give your answer.")


def _user_message(base: str, ctx: str, where: str, instr: str) -> str:
    """A speaker's user message (every persona but the v3 multiple-choice solver)."""
    if not ctx:
        return f"{base}\n\n{instr}"
    return f"{base}\n\n{ctx}\n\n" + " ".join(x for x in (where, "Give your response.", instr) if x)


def _solver_message(head: str, ctx: str, where: str) -> str:
    """The v3 multiple-choice solver's user message: the dataset's own prompt, then the discussion."""
    if not ctx:
        return head
    return f"{head}\n{ctx}\n\n" + " ".join(x for x in (where, "Give your response.") if x)


def _visible(all_rounds: list[list[tuple[str, str]]], mode: str, n: int,
             prior_specs: list[dict] | None = None) -> str:
    """The discussion a persona is shown, under one visibility mode."""
    if mode == "none" or not all_rounds:
        return ""
    flat = [pr for rnd in all_rounds for pr in rnd]
    if mode == "last_round":
        flat = all_rounds[-1]
    if not flat:
        return ""
    if mode == "letters_only":
        tally = Counter(l for p, r in flat if p not in NON_ANSWERING
                        and (l := extract_letter(r, n)))
        if not tally:
            return ""
        return "Answers so far: " + ", ".join(f"{l} x{c}" for l, c in tally.most_common())
    if mode == "no_letters":     # the reasoning, with the answers removed
        parts = [f"[{p}]:\n{_strip_commitment(_shown(r))}" for p, r in flat]
        return "Prior reasoning (conclusions withheld):\n" + "\n\n".join(parts)
    return _discussion(all_rounds, n, prior_specs, last_only=mode == "last_round")      # "all", "last_round"


_SIGNATURE: tuple | None = None


def prompt_signature() -> str:
    """A short hash of every text a speaker is sent: the persona prompts, the solver's question
    prompt, the question layout, the discussion format, the user messages and the follow-up
    prompts, under the current switches. It is part of every round cache key and of the settings
    (since 2026-10-06), so a change to any of these texts gives new recordings, and an archive made
    with other texts is refused instead of being continued with these."""
    global _SIGNATURE
    state = (tuple(sorted(PERSONA_PROMPTS.items())), EXPERT_TMPL, ANSWER_INSTR, SUMMARY_WORDS, OPEN, V3,
             COMMIT_NUDGE, MATH)                 # the texts themselves, never an object's address
    if _SIGNATURE is not None and _SIGNATURE[0] == state:
        return _SIGNATURE[1]
    rounds = [[("solver", "Step one.\nANSWER: A"), ("solver", "Step two.\nANSWER: B")], [("critic", "A flaw.\nANSWER: B")]]
    ctx = _discussion(rounds, 4, [{"personas": ["solver", "solver"]}, {"personas": ["critic"]}])
    texts = {"personas": PERSONA_PROMPTS, "expert": EXPERT_TMPL, "instr": ANSWER_INSTR,
                       "solver_high": solver_system(PERSONA_PROMPTS["solver"], "high"),
                       "solver_low": solver_system(PERSONA_PROMPTS["solver"], "low"),
                       "expert_low": speaker_system("expert", EXPERT_TMPL, "low"),
                       "nothing_to_review": NOTHING_TO_REVIEW,
                       "question": official_question("Q?", ["x", "y"]), "layout": render_question("Q?", ["x", "y"]),
                       "user": _user_message("<q>", ctx, _where(3, 3), ANSWER_INSTR),
                       "solver": _solver_message("<q>", ctx, _where(3, 2)),
                       "summary": [summary_nudge_v3("A"), summary_nudge_v3(None)], "commit": COMMIT_NUDGE,
                       "marks": [SUMMARY_MARK, COMMIT_MARK], "v2_summary": SUMMARY_NUDGE}
    if MATH:                                     # named only with MATH answers: every other blob is as before
        texts["math"] = {"solver": _solver_message(math_question("<q>"), ctx, _where(3, 2))}
    blob = json.dumps(texts, sort_keys=True)
    _SIGNATURE = (state, hashlib.sha1(blob.encode()).hexdigest()[:10])
    return _SIGNATURE[1]


def _standing_letter(all_rounds, n: int) -> str | None:
    """The most recent committed letter, for the contrarian's exclusion."""
    for rnd in reversed(all_rounds):
        for persona, resp in reversed(rnd):
            if persona in NON_ANSWERING:
                continue
            if (l := extract_letter(resp, n)):
                return l
    return None


def _answering(round_responses):
    return [(p, r) for p, r in round_responses if p not in NON_ANSWERING]


# Speakers within one round never see each other, so their calls are
# independent and can be in flight at once. Off by default so every earlier
# script behaves exactly as before; the per-group search turns it on (a
# four-solver round of 6144-token replies is four long calls, and in series
# they were the throughput bottleneck).
ROUND_PARALLEL = False


def set_round_parallel(on: bool) -> None:
    global ROUND_PARALLEL
    ROUND_PARALLEL = bool(on)


def execute_round(client, model, question, options, round_spec, all_rounds, temperature,
                  max_tokens: int = 3072, prompts: dict | None = None,
                  prior_specs: list[dict] | None = None) -> list[tuple[str, str]]:
    """Run ONE round on top of an existing transcript and return its (persona,
    response) pairs. `all_rounds` is not modified; personas within the round do
    not see each other. Extracted verbatim from execute_schema so the
    per-question methods (treegrow, adaptive depth) run rounds incrementally
    through the exact code path the batch executor uses. `prior_specs`, the
    specs of the rounds in `all_rounds` (the round runner passes them), lets the
    discussion say what each earlier round's speakers saw."""
    book = {**PERSONA_PROMPTS, **(prompts or {})}
    base = render_question(question, options)
    n = len(options)
    mode = round_spec.get("sees", DEFAULT_SEES)
    effort = round_spec.get("effort", DEFAULT_EFFORT)
    if effort != DEFAULT_EFFORT and not V3:
        raise ValueError("a round's effort can only be set under the v3 executor")

    def one(slot: int) -> tuple[str, str]:
        """Speaker number `slot`, with its own token tally (speakers of one round
        may run in separate threads)."""
        _tokens.tally = used[slot] = {"persona": personas[slot], "calls": 0, "prompt": 0, "completion": 0}
        try:
            return _one(personas[slot])
        finally:
            _tokens.tally = None

    def _one(persona: str) -> tuple[str, str]:
        ctx = _visible(all_rounds, "none" if persona in FORCED_BLIND else mode, n, prior_specs)
        where = _where(len(all_rounds) + 1, len(round_spec["personas"])) if ctx else ""
        system = speaker_system(persona, book[persona], effort)
        if V3 and (not OPEN or MATH) and persona == OFFICIAL_PERSONA:
            # the dataset's own prompt (the external direct baseline's request), then the discussion
            head = math_question(question) if MATH else official_question(question, options)
            return persona, chat_v3(client, model, system, _solver_message(head, ctx, where), n, effort)
        instr = ("Rule out options; do NOT give an ANSWER line."
                 if persona in NON_ANSWERING else ANSWER_INSTR)
        if persona == "contrarian" and (sl := _standing_letter(all_rounds, n)):
            instr = f"You may NOT choose option {sl}. " + instr
        judging = persona == JUDGE_PERSONA
        if judging:                                # the debate, then the answers to choose from
            if not V3:
                raise ValueError("the judge speaker needs the v3 executor")
            cands = judge_candidates(all_rounds, n)
            ctx = (ctx + "\n\n" if ctx else "") + judge_block(cands)
        if persona in REVIEWERS and not ctx:       # first in the debate: nothing to review yet
            instr = f"{NOTHING_TO_REVIEW} {instr}"
        user = _user_message(base, ctx, where, instr)
        if persona in NON_ANSWERING:
            text = chat(client, model, book[persona], user, temperature, max_tokens)
        elif V3:                                   # sampling and reply room are v3's own
            text = chat_v3(client, model, system, user, n, effort)
            if judging:
                text = judge_restrict(client, model, book[persona], user, text, cands, n)
        elif V2:
            text = chat_v2(client, model, book[persona], user, temperature, max_tokens, n)
        else:
            text = chat_commit(client, model, book[persona], user, temperature,
                               max_tokens, n)
        return persona, text

    personas = list(round_spec["personas"])
    used: list[dict | None] = [None] * len(personas)
    if ROUND_PARALLEL and len(personas) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(personas)) as pool:
            out = list(pool.map(one, range(len(personas))))          # order preserved
    else:
        out = [one(i) for i in range(len(personas))]
    _tokens.last_round = used             # read by the round runner, in this same thread
    return out


def final_letter(final: str, all_rounds: list[list[tuple[str, str]]], n: int) -> str | None:
    """The committed letter under a schema's `final` rule. Tail of execute_schema,
    extracted verbatim -- including the synthesizer path that returns None when the
    synthesizer's output has no parseable letter (legacy results reproduce)."""
    last = all_rounds[-1]
    letter = None
    if final == "synthesizer":
        for persona, resp in reversed(last):
            if persona == "synthesizer":
                # return here even when the letter is unparseable -- matches the
                # pre-extension executor exactly, so legacy results reproduce
                return extract_letter(resp, n)
    if final == "vote":
        tally = Counter(l for _, r in _answering(last) if (l := extract_letter(r, n)))
        if tally:
            letter = tally.most_common(1)[0][0]
    if letter is None:                                   # "last", and fallbacks
        ans = _answering(last)
        if ans:
            letter = extract_letter(ans[-1][1], n)
        else:                                            # last round commits nothing
            for rnd in reversed(all_rounds[:-1]):
                if (a := _answering(rnd)):
                    letter = extract_letter(a[-1][1], n)
                    break
    return letter


def execute_schema(client, model, question, options, schema, temperature,
                   max_tokens: int = 3072, prompts: dict | None = None,
                   return_trace: bool = False):
    """Run one schema over the question and return the final committed letter
    (or None if no parseable letter was produced).

    `prompts` overrides PERSONA_PROMPTS per persona (E7 evolves these).
    `return_trace` additionally returns the per-round (persona, response) lists.
    """
    n = len(options)
    all_rounds: list[list[tuple[str, str]]] = []
    for rnd in schema["rounds"]:
        all_rounds.append(execute_round(client, model, question, options, rnd,
                                        all_rounds, temperature, max_tokens, prompts))
    letter = final_letter(schema["final"], all_rounds, n)
    return (letter, all_rounds) if return_trace else letter


# --- schema validation + edit application (grammar-only; verbatim port) -----

def validate(schema: dict) -> bool:
    rounds = schema.get("rounds")
    if not rounds or not (1 <= len(rounds) <= MAX_ROUNDS):
        return False
    ok_personas = EXT_PERSONAS + NEW_PERSONAS
    for rnd in rounds:
        ps = rnd.get("personas")
        if not ps or not (1 <= len(ps) <= MAX_PERSONAS) or any(p not in ok_personas for p in ps):
            return False
        if rnd.get("sees", DEFAULT_SEES) not in SEES:
            return False
    if schema.get("final") not in FINALS:
        return False
    if schema["final"] == "synthesizer" and "synthesizer" not in rounds[-1]["personas"]:
        return False
    # the last round must be able to commit an answer
    if all(p in NON_ANSWERING for p in rounds[-1]["personas"]):
        return False
    return True


def apply_edit(schema: dict, op: dict) -> tuple[dict | None, str]:
    new = deepcopy(schema)
    rounds = new["rounds"]
    n = len(rounds)
    action = op.get("action")
    if action in ("add_round", "modify_round"):
        personas = [p for p in (op.get("personas") or []) if p in PERSONAS]
        if not personas:
            return None, f"{action} needs >=1 valid persona"
        rnd = {"personas": personas[:MAX_PERSONAS]}
        pos = op.get("position")
        if action == "add_round":
            idx = (pos - 1) if isinstance(pos, int) and 1 <= pos <= n + 1 else n
            rounds.insert(idx, rnd)
        else:
            if not (isinstance(pos, int) and 1 <= pos <= n):
                return None, f"modify_round position {pos!r} out of range"
            rounds[pos - 1] = rnd
    elif action == "remove_round":
        pos = op.get("position")
        if not (isinstance(pos, int) and 1 <= pos <= n):
            return None, f"remove_round position {pos!r} out of range"
        if n <= 1:
            return None, "cannot remove the only round"
        rounds.pop(pos - 1)
    elif action == "set_final":
        if op.get("final") not in FINALS:
            return None, f"final must be one of {FINALS}"
        new["final"] = op["final"]
    else:
        return None, f"unknown action {action!r}"
    return new, ""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_edit(raw: str) -> dict | None:
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip()).strip()
    try:
        obj = json.loads(raw)
    except Exception:
        m = _JSON_RE.search(raw)
        try:
            obj = json.loads(m.group(0)) if m else None
        except Exception:
            obj = None
    return obj if isinstance(obj, dict) and obj.get("action") in EDIT_ACTIONS else None

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

import json
import re
import threading
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
VISIBLE_SENTENCE = (" Your reply is read by other participants who cannot see your private "
                    "thinking, so write your reasoning out in the reply itself (the key facts "
                    "or steps, and why the rival options fail) before the ANSWER line.")
VISIBLE_MIN_CHARS = 80            # under visible reasoning, a shorter reply is not its own summary


def render_question(question: str, options: list[str]) -> str:
    body = "\n".join(f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options))
    return f"{question}\n\n{body}"


_ANS = re.compile(r"ANSWER\s*:\s*\(?\s*([A-J])\b", re.I)


def extract_letter(text: str, n_options: int) -> str | None:
    """The letter from the last 'ANSWER:' line, or None if there is no valid one."""
    for m in reversed(_ANS.findall(text or "")):
        letter = m.upper()
        if 0 <= LETTERS.index(letter) < n_options:
            return letter
    return None


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
COMMIT_NUDGE = ("STOP. You ran out of space. Do not continue or summarize the derivation. "
                "Your reply must BEGIN with the line 'ANSWER: X', where X is the single "
                "option letter you choose based on the reasoning so far. You may add one "
                "short sentence after that line, nothing more.")
COMMIT_MARK = "\n\n[cut off; asked to commit]\n"
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
    """The letter a commit reply names, or None."""
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
SUMMARY_NUDGE = ("Now write a brief summary of your reasoning for the other participants, "
                 "in at most 120 words: the decisive facts or steps, the options you ruled "
                 "out and why, and the point you are least sure of. Do not add new analysis. "
                 "End with exactly one line 'ANSWER: X' -- the same letter you chose above. "
                 "If you did not commit above, commit now.")
SUMMARY_MAX_TOKENS = 320          # <=120 words + the ANSWER line; conclusions ran 60-120 tokens
V2_SHORT_CHARS = 800              # a reply this short that already commits is its own summary
on_summary = None                 # callable(); a runner sets it to account for the extra call

V2_STATS: dict[str, int] = {"summaries": 0, "skipped_short": 0, "disagreements": 0,
                            "recovered_commits": 0, "failed_summaries": 0}
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
                              f"in at most {n} words. The other participants see ONLY this summary, never "
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


# --- the deep-think speaker -----------------------------------------------------
# One speaker that is given room to think: the model's highest reasoning
# setting and no reply cap, so the server lets it use whatever is left of the
# context window after the prompt. Every other speaker keeps the short budget.
# Off unless set_deep_think(True): the persona prompt, the move and the plan
# round only exist when it is on, so nothing changes for a run without it.
#
# Its thinking arrives in the reply's reasoning field and is NOT stored (it runs
# to ~10k tokens a call). What is stored is the visible reply plus the summary,
# and the summary call is shown the end of the thinking so it can report it.

DEEP_THINK = False
DEEP_PERSONA = "deep_think"
DEEP_PROMPT = ("You are a Deep Thinker answering a hard graduate-level multiple-choice question. "
               "You have as much room as you need. Think long and hard before you answer: work the "
               "problem from first principles, test every option against the exact wording of the "
               "question, re-derive any result you are not sure of by a second route, and look for "
               "the mistake you are most likely to have made. If earlier answers are shown, treat "
               "them as claims to check, not as evidence. Only then commit. ")
# both families' switches in one request: gpt-oss reads reasoning_effort, the
# Qwen chat template reads enable_thinking, and each ignores the other's
DEEP_EXTRA = {"reasoning_effort": "high", "chat_template_kwargs": {"enable_thinking": True}}
# The runner's clients give up on a request after 900 s and then send it again from the start.
# A deep-think turn that uses the whole window (30k tokens) took 810 s on a busy gpt-oss server
# (37 tokens/s per request), so it gets its own, longer limit.
DEEP_TIMEOUT = 3600.0
DEEP_THINKING_TAIL = 16000        # characters of thinking shown to the summary call (its end)
DEEP_COST = 5                     # speaker turns one deep-think turn counts as: the reasoning-high
                                  # direct baseline averaged 9.9k tokens a call, an ordinary reply ~2k
DEEP_STATS: dict[str, int] = {"calls": 0, "no_visible_reply": 0}


def set_deep_think(on: bool) -> None:
    """Use program_space.configure_executor from entry scripts, which also adds
    the move and the plan round."""
    global DEEP_THINK
    DEEP_THINK = bool(on)
    _rebuild_prompts()


def turn_cost(personas) -> int:
    """Speaker turns a round counts as (a deep-think turn counts DEEP_COST)."""
    return sum(DEEP_COST if p == DEEP_PERSONA else 1 for p in personas)


def chat_deep(client, model: str, system: str, user: str, temperature: float) -> tuple[str, str]:
    """(thinking, visible reply) of one deep-think call. No max_tokens: the
    server then allows everything left in its window after the prompt."""
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "system", "content": system},
                               {"role": "user", "content": user}],
        temperature=temperature, extra_body=DEEP_EXTRA, timeout=DEEP_TIMEOUT)
    _count_tokens(resp)
    msg = resp.choices[0].message
    thinking = (getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or "").strip()
    return thinking, (getattr(msg, "content", None) or "").strip()


def chat_v2_deep(client, model: str, system: str, user: str, temperature: float,
                 n_options: int) -> str:
    """The deep-think turn under v2: the long call, then the usual summary call
    with the end of the thinking put in front of the visible reply, then the
    usual reconciliation. A call that ran out of room before it answered has no
    visible reply; the summary call is then its chance to commit, as it is for
    any cut-off reply."""
    thinking, visible = chat_deep(client, model, system, user, temperature)
    with _stats_lock:
        DEEP_STATS["calls"] += 1
        DEEP_STATS["no_visible_reply"] += not visible
    if not (thinking or visible):
        return ""
    tail = thinking[-DEEP_THINKING_TAIL:]
    shown = (f"[the end of my private reasoning]\n{tail}\n\n[my reply]\n{visible}" if tail else visible)
    full = visible or "(no reply: the reasoning ran out of room)\n" + thinking[-2000:]
    summary = summarize_reply(client, model, system, user, shown, temperature)
    if not summary:
        _bump("failed_summaries")
        return full
    _bump("summaries")
    return _reconcile(full, summary, n_options)


def _build_prompts(instr: str) -> tuple[str, dict[str, str]]:
    """The expert template and the persona system prompts, built around one
    commitment instruction. Called at import (v1) and by set_v2."""
    expert = ("You are a leading expert in {field}, answering a hard graduate-level "
              "multiple-choice question inside your own specialty. Reason step by step "
              "from your expert knowledge of the field, then commit. " + instr +
              " Always choose exactly one letter; never abstain.")
    prompts = {
        "solver": ("You are a Solver answering a hard graduate-level multiple-choice "
                   "question. Reason step by step from your own knowledge, then commit. "
                   + instr + " Always choose exactly one letter; never abstain."),
        "critic": ("You are a Critic. Examine the prior answers and their reasoning for a "
                   "flaw — a wrong deduction, a miscalculation, a misread option, an "
                   "overlooked constraint. Give the corrected choice with your reasoning. "
                   + instr),
        "synthesizer": ("You are a Synthesizer. Weigh the prior answers and their "
                        "reasoning, resolve the disagreements, and commit to the single "
                        "best option. You MUST choose exactly one letter. " + instr),
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
                        "and commit. " + instr + " Always choose exactly one letter."),
        "verifier": ("You are a Verifier. Do not re-derive the solution from scratch. Take each "
                     "answer letter committed so far and test it directly against the exact "
                     "wording of the question: does that option satisfy every requirement the "
                     "question states? Test the strongest rival option the same way. Keep the "
                     "answer that survives these checks; switch if it does not. " + instr +
                     " Always choose exactly one letter; never abstain."),
        "expert": expert.format(field="the question's field"),
    }
    if DEEP_THINK:
        prompts[DEEP_PERSONA] = DEEP_PROMPT + instr + " Always choose exactly one letter; never abstain."
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
    global ANSWER_INSTR, EXPERT_TMPL, PERSONA_PROMPTS
    ANSWER_INSTR = ANSWER_INSTR_V2 if V2 else ANSWER_INSTR_V1
    if VISIBLE_REASONING:
        ANSWER_INSTR = ANSWER_INSTR + VISIBLE_SENTENCE
    EXPERT_TMPL, PERSONA_PROMPTS = _build_prompts(ANSWER_INSTR)


def set_visible_reasoning(on: bool) -> None:
    """Use schema_fitness.set_visible_reasoning from entry scripts, so the
    per-question critic override changes with the persona prompts."""
    global VISIBLE_REASONING
    VISIBLE_REASONING = bool(on)
    _rebuild_prompts()


def visible_signature() -> str | None:
    """'1' when visible reasoning is on, else None. Part of the cache key."""
    return "1" if VISIBLE_REASONING else None


def v2_signature() -> str | None:
    """'2' when v2 is on, else None. Part of the cache key."""
    return "2" if V2 else None


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


def _visible(all_rounds: list[list[tuple[str, str]]], mode: str, n: int) -> str:
    """The prior-response context a persona is shown, under one visibility mode."""
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
        return "Prior committed answers: " + ", ".join(f"{l} x{c}" for l, c in tally.most_common())
    if mode == "no_letters":     # the reasoning, with the commitments removed
        parts = [f"[{p}]:\n{_strip_commitment(_shown(r))}" for p, r in flat]
        return "Prior reasoning (conclusions withheld):\n" + "\n\n".join(parts)
    return "Prior responses:\n" + _digest(flat, n)      # "all"


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
                  max_tokens: int = 3072, prompts: dict | None = None) -> list[tuple[str, str]]:
    """Run ONE round on top of an existing transcript and return its (persona,
    response) pairs. `all_rounds` is not modified; personas within the round do
    not see each other. Extracted verbatim from execute_schema so the
    per-question methods (treegrow, adaptive depth) run rounds incrementally
    through the exact code path the batch executor uses."""
    book = {**PERSONA_PROMPTS, **(prompts or {})}
    base = render_question(question, options)
    n = len(options)
    mode = round_spec.get("sees", DEFAULT_SEES)

    def one(slot: int) -> tuple[str, str]:
        """Speaker number `slot`, with its own token tally (speakers of one round
        may run in separate threads)."""
        _tokens.tally = used[slot] = {"persona": personas[slot], "calls": 0, "prompt": 0, "completion": 0}
        try:
            return _one(personas[slot])
        finally:
            _tokens.tally = None

    def _one(persona: str) -> tuple[str, str]:
        ctx = _visible(all_rounds, "none" if persona in FORCED_BLIND else mode, n)
        instr = ("Rule out options; do NOT give an ANSWER line."
                 if persona in NON_ANSWERING else ANSWER_INSTR)
        if persona == "contrarian" and (sl := _standing_letter(all_rounds, n)):
            instr = f"You may NOT choose option {sl}. " + instr
        user = f"{base}\n\n{ctx}\n\nGive your response. {instr}" if ctx else f"{base}\n\n{instr}"
        if persona in NON_ANSWERING:
            text = chat(client, model, book[persona], user, temperature, max_tokens)
        elif persona == DEEP_PERSONA:
            text = chat_v2_deep(client, model, book[persona], user, temperature, n)
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

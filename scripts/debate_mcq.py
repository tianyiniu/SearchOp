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

ANSWER_INSTR = "Reason step by step, then end with exactly one line: 'ANSWER: <letter>'."


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
    {"chat_template_kwargs": {"enable_thinking": False}},   # Qwen3 chat template
    {"reasoning_effort": "none"},                           # Qwen3.5 / newer vLLM
    {},                                                     # nothing worked: let it think
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
        if known is not None:
            return text                    # mechanism already chosen; report what came back
    return ""


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

EXPERT_TMPL = ("You are a leading expert in {field}, answering a hard graduate-level "
               "multiple-choice question inside your own specialty. Reason step by step "
               "from your expert knowledge of the field, then commit. " + ANSWER_INSTR +
               " Always choose exactly one letter; never abstain.")

PERSONA_PROMPTS = {
    "solver": ("You are a Solver answering a hard graduate-level multiple-choice "
               "question. Reason step by step from your own knowledge, then commit. "
               + ANSWER_INSTR + " Always choose exactly one letter; never abstain."),
    "critic": ("You are a Critic. Examine the prior answers and their reasoning for a "
               "flaw — a wrong deduction, a miscalculation, a misread option, an "
               "overlooked constraint. Give the corrected choice with your reasoning. "
               + ANSWER_INSTR),
    "synthesizer": ("You are a Synthesizer. Weigh the prior answers and their "
                    "reasoning, resolve the disagreements, and commit to the single "
                    "best option. You MUST choose exactly one letter. " + ANSWER_INSTR),
    "contrarian": ("You are a Contrarian. The standing answer is probably wrong. Build the "
                   "strongest positive case for a DIFFERENT option: find the reading of the "
                   "question, or the piece of knowledge, under which another option is the "
                   "correct one. You may not endorse the standing answer. " + ANSWER_INSTR),
    "eliminator": ("You are an Eliminator. Do NOT answer the question. Work through the "
                   "options and rule out the ones that cannot be correct, giving a specific "
                   "reason for each — a violated constraint, a wrong magnitude, a definition "
                   "that does not match. End with a line listing the options that survive: "
                   "'SURVIVING: <letters>'. Never write an 'ANSWER:' line."),
    "independent": ("You are an Independent Solver working alone on a hard graduate-level "
                    "multiple-choice question. Reason step by step from your own knowledge "
                    "and commit. " + ANSWER_INSTR + " Always choose exactly one letter."),
    "verifier": ("You are a Verifier. Do not re-derive the solution from scratch. Take each "
                 "answer letter committed so far and test it directly against the exact "
                 "wording of the question: does that option satisfy every requirement the "
                 "question states? Test the strongest rival option the same way. Keep the "
                 "answer that survives these checks; switch if it does not. " + ANSWER_INSTR +
                 " Always choose exactly one letter; never abstain."),
    "expert": EXPERT_TMPL.format(field="the question's field"),
}

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


def _digest(prior: list[tuple[str, str]], n: int) -> str:
    """Show each prior persona's chosen letter and its (bounded) reasoning so later
    personas can actually debate the reasoning, not just the letters."""
    return "\n\n".join(f"[{p}] chose {extract_letter(r, n) or '?'}:\n{_clip(r)}"
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
        parts = [f"[{p}]:\n{_clip(_strip_commitment(r))}" for p, r in flat]
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
    responses = []
    for persona in round_spec["personas"]:
        ctx = _visible(all_rounds, "none" if persona in FORCED_BLIND else mode, n)
        instr = ("Rule out options; do NOT give an ANSWER line."
                 if persona in NON_ANSWERING else ANSWER_INSTR)
        if persona == "contrarian" and (sl := _standing_letter(all_rounds, n)):
            instr = f"You may NOT choose option {sl}. " + instr
        user = f"{base}\n\n{ctx}\n\nGive your response. {instr}" if ctx else f"{base}\n\n{instr}"
        responses.append((persona, chat(client, model, book[persona],
                                        user, temperature, max_tokens)))
    return responses


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

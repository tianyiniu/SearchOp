"""The model-guided parts of the program search: seed programs written in one
batch, and guided edits of an existing program from digests of where it
failed. Both go through the OpenAI Responses API with a structured output, so
what comes back is a program object, not prose to parse.

The guiding model is gpt-5.6-terra at medium reasoning by default. It never
sees the debate model's prompts or the test questions; at search time it sees
training questions' transcripts in digest form (letters per round, gold, and
the last speaker's short summary).

Operator-first edits: the search draws the KIND of edit (drop a rule, change
an action, change the opening rounds, ...) and asks the model to make the best
edit of that kind. A model asked "what would you change" mostly adds rounds;
asked "which rule should go", it has to weigh what each rule is worth. The
returned program is checked against the grammar and against the drawn kind
(a drop must remove exactly one rule, and so on); anything else is rejected
and the caller falls back to a random edit.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

import debate_mcq as D  # noqa: E402
import evolve_program_mcq as M  # noqa: E402
import program_space as P  # noqa: E402

GUIDE_MODEL = "gpt-5.6-terra"          # the model that labelled the questions (describe_questions.py)
GUIDE_EFFORT = "medium"


def make_client():
    """An OpenAI client for the guide model; the key comes from .env."""
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(P.ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (expected in .env)")
    return OpenAI()


# --- structured output -----------------------------------------------------------

class PlanRound(BaseModel):
    personas: list[str]
    sees: str | None = Field(default=None, description="omit or null for the default 'all'")


class Rule(BaseModel):
    when: list[str]
    do: str


class Program(BaseModel):
    plan: list[PlanRound]
    rules: list[Rule]
    default: str


class SeedProgram(BaseModel):
    name: str = Field(description="short snake_case name")
    strategy: str = Field(description="one sentence: what this program does differently and why")
    program: Program


class SeedBatch(BaseModel):
    programs: list[SeedProgram]


class Edit(BaseModel):
    rationale: str = Field(description="one or two sentences: which rule or round and why")
    program: Program


def to_program(p: Program) -> dict:
    prog = {"plan": [{"personas": r.personas, **({"sees": r.sees} if r.sees else {})} for r in p.plan],
            "rules": [{"when": r.when, "do": r.do} for r in p.rules],
            "default": p.default}
    prog = P.normalize_program(prog)
    P.validate_program(prog)
    return prog


def _call(client, instructions: str, user: str, text_format, effort: str = GUIDE_EFFORT,
          max_output_tokens: int = 6000, retries: int = 3):
    """One structured call with backoff. Returns (parsed, usage) or raises."""
    delay = 2.0
    for attempt in range(retries):
        try:
            resp = client.responses.parse(model=GUIDE_MODEL, instructions=instructions, input=user,
                                          reasoning={"effort": effort}, text_format=text_format,
                                          max_output_tokens=max_output_tokens)
            if resp.output_parsed is None:
                raise ValueError(f"no parsed output (status {getattr(resp, 'status', '?')})")
            u = getattr(resp, "usage", None)
            usage = {"input": getattr(u, "input_tokens", None), "output": getattr(u, "output_tokens", None)}
            return resp.output_parsed, usage
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2


# --- seed programs -------------------------------------------------------------------

SEED_INSTRUCTIONS = """You design control programs for a multi-agent debate that answers hard
graduate-level multiple-choice questions. Copies of one language model play
roles (solver, critic, verifier, expert, independent solver, synthesizer). A
program decides, after every round, what to run next or when to stop and how
to read the answer off the transcript.

{grammar}

You will be shown {n_existing} existing programs and a description of the
kinds of question the debate faces. Write {n_new} NEW programs. Requirements:
- Each must be valid under the grammar above, using only the listed plan
  specs, conditions and actions, exactly as spelled.
- Each must differ from every existing program and from each other in at
  least two of: opening width, stop read, the set of moves used, the
  conditions it branches on.
- Aim each at a different strategy or a different kind of question; say
  which in the strategy line.
- Keep programs short (2 to 6 rules). Prefer programs whose typical cost is
  between 2 and 10 speaker turns.
- Do not simply add rounds to an existing program.
Return exactly {n_new} programs."""


def write_seeds(client, existing: dict[str, dict], group_texts: list[str], n_new: int = 4,
                effort: str = GUIDE_EFFORT) -> list[dict]:
    """`n_new` model-written programs, each validated. Retries once with the
    rejects named if the first batch has too few valid programs."""
    instructions = SEED_INSTRUCTIONS.format(grammar=P.grammar_text(), n_existing=len(existing),
                                            n_new=n_new)
    shown = "\n".join(f"{name}: {json.dumps(prog, separators=(',', ':'))}"
                      for name, prog in existing.items())
    user = ("EXISTING PROGRAMS\n" + shown + "\n\nKINDS OF QUESTION\n" + "\n".join(group_texts)
            + f"\n\nWrite {n_new} new programs.")
    out: list[dict] = []
    seen = {P.canon(p) for p in existing.values()}
    feedback = ""
    for _ in range(3):
        batch, usage = _call(client, instructions, user + feedback, SeedBatch, effort)
        rejects = []
        for sp in batch.programs:
            try:
                prog = to_program(sp.program)
            except Exception as exc:                        # grammar miss
                rejects.append(f"{sp.name}: {type(exc).__name__}: {exc}"[:200])
                continue
            key = P.canon(prog)
            if key in seen:
                rejects.append(f"{sp.name}: identical to an existing program")
                continue
            seen.add(key)
            out.append({"name": "llm_" + "".join(ch if ch.isalnum() else "_" for ch in sp.name)[:32],
                        "strategy": sp.strategy, "program": prog, "usage": usage})
            if len(out) == n_new:
                return out
        feedback = ("\n\nYour previous batch had these problems; write "
                    f"{n_new - len(out)} more valid, different programs:\n" + "\n".join(rejects))
    return out


# --- one seed program per question group (the v3 seed stage) ---------------------------

class GroupSeedProgram(BaseModel):
    group: int = Field(description="the number of the question group this program is written for")
    name: str = Field(description="short snake_case name")
    strategy: str = Field(description="one sentence: why this program suits that group")
    program: Program


class GroupSeedBatch(BaseModel):
    programs: list[GroupSeedProgram]


GROUP_SEED_INSTRUCTIONS = """You design control programs for a multi-agent debate that answers hard
graduate-level multiple-choice questions. Copies of one language model play
roles (solver, critic, verifier, expert, independent solver, synthesizer). A
program decides, after every round, what to run next or when to stop and how
to read the answer off the transcript.

{grammar}

You will be shown {n_existing} existing programs and a description of
{n_groups} groups of question, each with the reasoning it typically needs and
the way it typically goes wrong. Write exactly ONE new program for EACH group,
aimed at that group's reasoning and its main failure risk. Requirements:
- Each must be valid under the grammar above, using only the listed plan
  specs, conditions and actions, exactly as spelled.
- Each must differ from every existing program and from each other in at
  least two of: opening width, stop read, the set of moves used, the
  conditions it branches on.
- Say in the strategy line why the program suits its group.
- Keep programs short (2 to 6 rules). Prefer programs whose typical cost is
  between 2 and 10 speaker turns.
- Do not simply add rounds to an existing program.
Return one program per group, each labelled with its group number."""


def write_group_seeds(client, existing: dict[str, dict], group_texts: dict[int, str],
                      effort: str = GUIDE_EFFORT) -> list[dict]:
    """One model-written program per question group, each validated, in group
    order. A group whose program was invalid or a duplicate is asked for again
    (with the problem named), a few times at most."""
    instructions = GROUP_SEED_INSTRUCTIONS.format(grammar=P.grammar_text(), n_existing=len(existing),
                                                  n_groups=len(group_texts))
    shown = "\n".join(f"{name}: {json.dumps(prog, separators=(',', ':'))}"
                      for name, prog in existing.items())
    base = "EXISTING PROGRAMS\n" + shown + "\n\nGROUPS OF QUESTION\n"
    got: dict[int, dict] = {}
    seen = {P.canon(p) for p in existing.values()}
    feedback = ""
    for _ in range(4):
        todo = [g for g in group_texts if g not in got]
        if not todo:
            break
        user = (base + "\n".join(group_texts[g] for g in todo)
                + f"\n\nWrite one new program for each of these groups: {todo}." + feedback)
        if got:
            user += ("\n\nPrograms already written for the other groups (differ from these too):\n"
                     + "\n".join(f"{w['name']}: {json.dumps(w['program'], separators=(',', ':'))}"
                                 for w in got.values()))
        batch, usage = _call(client, instructions, user, GroupSeedBatch, effort)
        rejects = []
        for sp in batch.programs:
            if sp.group not in todo or sp.group in got:
                continue
            try:
                prog = to_program(sp.program)
            except Exception as exc:                        # grammar miss
                rejects.append(f"group {sp.group} ({sp.name}): {type(exc).__name__}: {exc}"[:200])
                continue
            key = P.canon(prog)
            if key in seen:
                rejects.append(f"group {sp.group} ({sp.name}): identical to an existing program")
                continue
            seen.add(key)
            clean = "".join(ch if ch.isalnum() else "_" for ch in sp.name)[:32]
            got[sp.group] = {"name": f"llm_g{sp.group}_{clean}", "group": sp.group,
                             "strategy": sp.strategy, "program": prog, "usage": usage}
        feedback = ("\n\nYour previous batch had these problems:\n" + "\n".join(rejects)) if rejects else ""
    return [got[g] for g in sorted(got)]


# --- guided edits --------------------------------------------------------------------

OP_TEXT = {
    "change_action": "change the action of exactly ONE existing rule (its conditions stay)",
    "add_rule": "insert exactly ONE new rule (1 or 2 conditions) at the position where it helps",
    "change_cond": "change or add exactly ONE condition of ONE existing rule",
    "change_default": "change the default stop read only",
    "drop_rule": "remove exactly ONE rule; everything else stays",
    "swap_rules": "reorder the rules by swapping exactly TWO of them",
    "plan": "change the opening rounds only: the width of the first round, or replace, add or drop one plan round",
}

EDIT_INSTRUCTIONS = """You improve a control program for a multi-agent debate that answers hard
graduate-level multiple-choice questions. Copies of one language model play
roles (solver, critic, verifier, expert, independent solver, synthesizer). The
program decides, after every round, what to run next or when to stop and how
to read the answer off the transcript.

{grammar}

You will be shown the current program, how it scores on each group of
questions, the group it is being improved for, and digests of questions in
that group where it failed and where it succeeded. A digest lists, round by
round, which speakers ran and which letter each committed, the letter the
program finally read, and the correct letter. Some failures were RIGHT after
the opening round and lost the answer later: a shorter or more careful
program would have kept them. Weigh those as seriously as the outright misses.

Make exactly one edit of the requested kind and return the whole edited
program. Do not make any other change. When the kind allows a choice of move
or condition, prefer one the program does not already use. Keep the program
valid under the grammar, with every condition and action spelled exactly as
listed."""


def edit_user_text(parent: dict, op: str, scores: list[str], group_text: str,
                   digests: list[str]) -> str:
    return ("CURRENT PROGRAM\n" + json.dumps(parent, indent=1)
            + "\n\nSCORES BY GROUP (accuracy, mean speaker turns)\n" + "\n".join(scores)
            + "\n\nGROUP TO IMPROVE\n" + group_text
            + "\n\nDIGESTS\n" + "\n\n".join(digests)
            + f"\n\nREQUESTED EDIT: {OP_TEXT[op]}.")


def guided_edit(client, parent: dict, op: str, scores: list[str], group_text: str,
                digests: list[str], effort: str = GUIDE_EFFORT) -> tuple[dict | None, str]:
    """The model's edit of `parent` of kind `op`, or (None, reason) if it did
    not return a valid program that differs from the parent and matches the
    kind. `op` is a rule operator name or "plan"."""
    instructions = EDIT_INSTRUCTIONS.format(grammar=P.grammar_text())
    try:
        edit, _ = _call(client, instructions, edit_user_text(parent, op, scores, group_text, digests),
                        Edit, effort)
        child = to_program(edit.program)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"[:200]
    if P.canon(child) == P.canon(parent):
        return None, "unchanged"
    want = P.expected_rule_delta(op)
    got = len(child["rules"]) - len(parent["rules"])
    if want is not None and got != want:
        return None, f"rule count changed by {got}, expected {want} for {op}"
    if op == "plan" and child["rules"] != parent["rules"]:
        return None, "plan edit touched the rules"
    if op != "plan" and child["plan"] != parent["plan"]:
        return None, "rule edit touched the plan"
    return child, edit.rationale


# --- digests of recorded questions -------------------------------------------------

class Tracing:
    """A runner proxy that remembers the rounds it returned, so a replay of a
    program on a cached question yields its transcript without a second
    interpreter. Never spends: the inner runner decides that."""

    def __init__(self, inner):
        self.inner = inner
        self.rounds: list[tuple[dict, list]] = []

    def run_round(self, qid, all_rounds, executed_specs, round_spec, rep=0, prompts=None):
        out = self.inner.run_round(qid, all_rounds, executed_specs, round_spec, rep=rep,
                                   prompts=prompts)
        self.rounds.append((round_spec, out))
        return out


def digest(prog: dict, runner, row: dict, group: int, rep: int = 0,
           summary_chars: int = 400, max_calls: int = 16) -> tuple[str, bool] | None:
    """One question's digest for the guide, as (text, lost_later), or None if
    the transcript is not fully on the cache (nothing is ever spent to build a
    digest). `lost_later` marks a failure whose opening round was right."""
    tr = Tracing(runner)
    try:
        out = M.run_program(prog, tr, row, rep=rep, max_calls=max_calls)
    except M.OffCache:
        return None
    n, gold = len(row["options"]), row.get("answer_letter")
    lines = []
    opening_right = False
    for i, (spec, resp) in enumerate(tr.rounds):
        letters = [D.extract_letter(r, n) or "?" for p, r in resp if p not in D.NON_ANSWERING]
        who = "+".join(dict.fromkeys(spec["personas"]))
        width = len(spec["personas"])
        lines.append(f"  round {i + 1} {who}" + (f" x{width}" if width > 1 else "") + ": "
                     + ", ".join(letters))
        if i == 0 and letters:
            maj = Counter(l for l in letters if l != "?").most_common(1)
            opening_right = bool(maj) and maj[0][0] == gold
    lost_later = (not out["correct"]) and opening_right
    verdict = "CORRECT" if out["correct"] else "WRONG"
    note = " (the opening round's majority was right; the answer was lost later)" if lost_later else ""
    head = (f"[question {row['id'][:8]}, group {group}] correct letter {gold}; program read "
            f"{out['letter'] or '?'} -> {verdict}{note}; {out['n_calls']} turns; "
            f"actions: {', '.join(out['actions']) or 'none'}")
    body = "\n".join(lines)
    tail = ""
    if tr.rounds and tr.rounds[-1][1]:
        last_p, last_r = tr.rounds[-1][1][-1]
        _, summ = D._split_summary(last_r)
        shown = (summ if summ is not None else last_r).strip().replace("\n", " ")
        tail = f"\n  last speaker ({last_p}) said: \"{shown[:summary_chars]}\""
    return head + "\n" + body + tail, lost_later


def build_digests(prog: dict, rec, runner, rows: dict, qids: list[str], group: int,
                  rng: random.Random, n_fail: int = 5, n_ok: int = 2,
                  max_calls: int = 16) -> list[str]:
    """Digests of up to n_fail failures and n_ok successes of `rec` (a
    ProgRecord) on `qids` at replicate 0. Failures whose opening round was
    right are taken first, so the guide always sees the case for stopping
    earlier when there is one."""
    r0 = rec.reps.get(0, {})
    fails = [q for q in qids if q in r0 and r0[q][0] == 0]
    oks = [q for q in qids if q in r0 and r0[q][0] == 1]
    rng.shuffle(fails)
    rng.shuffle(oks)
    fail_digests: list[tuple[str, bool]] = []
    for q in fails[: 3 * n_fail]:
        d = digest(prog, runner, rows[q], group, max_calls=max_calls)
        if d is not None:
            fail_digests.append(d)
    fail_digests.sort(key=lambda d: not d[1])                 # lost-later first, stable
    out = [d[0] for d in fail_digests[:n_fail]]
    n_added = 0
    for q in oks:
        if n_added >= n_ok:
            break
        d = digest(prog, runner, rows[q], group, max_calls=max_calls)
        if d is not None:
            out.append(d[0])
            n_added += 1
    return out

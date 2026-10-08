"""The question describer for SuperGPQA, prompt v3 (outputs/describe_v3: the 600 subset's groups
and routes). One describer file per dataset since 2026-10-07: each holds its own prompt and
vocabulary, so a new dataset never overwrites another's (this file was rebuilt that day from
archive/scripts/describe_questions copy.py by undoing the two recorded v3 -> v4 wording edits).

Step 1 of the per-type search: describe every question's reasoning shape.

A stronger model reads each question WITH its options and returns a small
structured record: the ordered reasoning steps a solver needs and the one to
three reasoning challenges most likely to make a competent solver fail (both
from fixed vocabularies), whether the answer is recalled or derived, and a
scrubbed template paragraph of the solution path. The record is used for
grouping and routing only. It is never shown to the debate agents, so nothing
the stronger model knows leaks into the debate.

Vocabulary (since v2). Every step and challenge says what in the question makes it
apply, so the describer can check it rather than guess, and none is on for
nearly every question (v1's "compare the choices precisely" was on for 94%).
The challenges say where an error comes from (a fact that cannot be reasoned
out, a condition read past, a slip among close options, a reversed direction),
because that is what decides which debate program can recover from it. The
dataset's answer key is taken as correct: questions are not screened for flaws.

Uses the OpenAI Responses API (not the chat-completions API that the vLLM
servers speak) with a JSON schema so every record parses. Output is a JSONL
file keyed by question id; rerunning skips ids already done, so an interrupted
run resumes for free. Records carry a prompt-version tag; bump PROMPT_VERSION
after editing the prompt and the old records are ignored.

Order of operations:

    # 0. let the model propose the vocabulary from batches of questions seen
    #    side by side; paste the printed STEPS/CHALLENGES over the ones below,
    #    edit, bump PROMPT_VERSION (v3's lists came from this pass on the 2k
    #    train split, with the question-defect challenge removed and one step
    #    definition cleared of it by hand)
    python3 scripts/describe_supergpqa_v3.py --discover --dataset datasets/supergpqa_2k_train.json
    # 1. draft ~10 anchor records, correct them by hand in the JSON
    python3 scripts/describe_supergpqa_v3.py --draft-anchors 12 --anchors outputs/describe_v3/anchors.json
    # 2. label the train split with the anchors in every call
    python3 scripts/describe_supergpqa_v3.py --anchors outputs/describe_v3/anchors.json \\
        --dataset datasets/supergpqa_2k_train.json \\
        --out outputs/describe_v3/templates_train.jsonl --workers 8
    # (run_describe_v3.sh runs every step, for both splits and both subset sizes)
    # smoke test of any step: add --limit 5

Why anchors instead of batch labelling: a batch lets the model compare
questions, but then each label depends on what else was in the batch. Fixed
anchors give every call the same comparison set.

The API key is read from OPENAI_API_KEY (the repo's .env is loaded).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import debate_mcq as D  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

PROMPT_VERSION = "v3"
MODEL = "gpt-6-sol"
# (input, output) dollars per million tokens, for the cost line; None prints tokens only
PRICE_PER_M: tuple[float, float] | None = (2.0, 10.0)   # gpt-6-sol list price; ignores the cached-input discount

# --- the vocabulary -----------------------------------------------------------
# Reasoning steps. The describer lists the ones a solver needs, in order. Each
# definition names what in the question makes the step apply. A step that every
# question needs (reading the options, comparing them) is not a step here.
STEPS = {
    'retrieve_specific_fact':                       'Recall a particular established fact, value, name, or association. This applies when the requested item cannot be derived from the supplied information; it does not cover retrieving a reusable rule to apply to a case.',
    'select_governing_rule':                        'Identify a reusable principle, equation, definition, or procedure that governs a result. This applies when the necessary rule is not fully supplied; it does not cover choosing a particular fact or merely carrying out an already specified procedure.',
    'classify_or_infer_from_case':                  'Use defining features or evidence to identify a category, explanation, consequence, or appropriate action for a described case. This applies when case details determine the choice; it does not cover recalling an isolated association without using those details.',
    'apply_qualifiers_and_boundaries':              'Determine how a stated condition, exception, direction, reference point, or scope restriction changes the applicable case. This applies when overlooking that qualification would change the result; it does not cover building the main equation from the givens.',
    'formalize_inputs_and_structure':               'Translate words, notation, or an arrangement into usable variables, constraints, components, or a model. This applies when interpreting the supplied setup must precede derivation; it does not cover choosing an unstated governing rule or executing the resulting calculation.',
    'derive_symbolic_relationship':                 'Transform expressions or constraints to obtain an equivalent form or an intermediate relationship. This applies when algebraic, logical, or calculus work is needed before the result can be evaluated; it does not cover chiefly numerical substitution.',
    'calculate_or_enumerate_result':                'Evaluate values, count possibilities, or generate a requested numerical result from an established setup. This applies when execution of arithmetic or finite enumeration determines the choice; it does not cover a separate unit conversion or a chiefly symbolic derivation.',
    'reconcile_units_and_scales':                   'Convert or check units, proportions, reference bases, precision, or numerical scales. This applies when supplied and requested quantities cannot be compared or combined as written; it does not cover ordinary arithmetic on already compatible quantities.',
    'track_stages_and_contributions':               'Carry intermediate states, repeated operations, or separate contributions through to a final result. This applies when the order of stages or the accounting across parts matters; it does not cover checking the independent claims bundled in an answer choice.',
    'evaluate_candidates_and_answer_completeness':  'Compare the eligible candidates against a stated optimum, or check every component of a compound answer choice. This applies when the question asks for an extreme (the largest, the best, the first) or when options bundle several claims; it does not replace classifying one described case.',
}

# Reasoning challenges. The describer lists one to three, most likely first.
# Each says where the error comes from, which is what decides whether another
# look (a verifier, a critic, a fresh solver, a vote) can recover from it.
CHALLENGES = {
    'missing_specific_knowledge':               'A required particular fact or convention is not known. This applies when the decisive information is neither supplied nor derivable from the question; it does not cover selecting the wrong general rule.',
    'wrong_rule_or_setup':                      'An inapplicable governing relationship is chosen, or the givens are represented by the wrong model. This applies before correct execution can begin; it does not cover a slip while calculating from a sound setup.',
    'confused_neighboring_concepts':            'Related categories, terms, roles, or interpretations are treated as interchangeable. This applies when their defining meanings separate the choices; it does not cover options that differ only in a digit or an unchecked second component.',
    'overlooked_condition_or_scope':            'A stated restriction, exception, decisive case clue, or limit on a claim is missed or applied to the wrong case. This applies when honoring the wording changes validity; it does not cover reversing a sign or losing track of an ordered stage.',
    'calculation_or_scale_slip':                'Execution of a sound method fails through arithmetic, symbolic manipulation, conversion, rounding, or numerical precision. This applies after the intended relationship and inputs are established; it does not cover choosing that relationship incorrectly.',
    'direction_or_stage_mixup':                 'The solver reverses a direction or relationship, uses the wrong reference point, or loses the position or state of an ordered process. This applies when orientation or stage identity is decisive; it does not cover a conversion-factor or ordinary arithmetic slip.',
    'incomplete_or_imprecise_option_check':     'An answer is accepted after checking only part of it, or a small difference between otherwise similar choices is missed. This applies to compound and near-duplicate options; it does not cover confusion about the underlying definitions of two categories.',
}

KNOWLEDGE = ("recall", "derive", "both")

Step = Literal[tuple(STEPS)]                # type: ignore[valid-type]
Challenge = Literal[tuple(CHALLENGES)]      # type: ignore[valid-type]
Knowledge = Literal[KNOWLEDGE]              # type: ignore[valid-type]


class Description(BaseModel):
    steps: list[Step] = Field(description="The reasoning steps this question requires, in the order a careful solver takes them, 1 to 5 entries, no repeats. List a step only if this question needs it.")
    challenges: list[Challenge] = Field(description="The 1 to 3 reasoning challenges most likely to make a competent solver choose a wrong option, most likely first, no repeats.")
    knowledge: Knowledge = Field(description="Whether the answer is mainly recalled, mainly derived, or both.")
    template: str = Field(description="A paragraph of 60 to 100 words describing how the question is solved, with every subject-specific noun, name, number and option text replaced by a generic placeholder. See the instructions for what it must cover and in what order.")


# The dataset's questions, options and keys are taken as correct: no prompt may
# invite labels about question defects (see the v2 run, where one crept back in).
WELL_POSED = (
    "Take the question, its options and its answer key as correct and well-posed. Never "
    "suggest that a question is ambiguous, underspecified, damaged or without a unique "
    "answer, and never propose or apply a label for that."
)

TEMPLATE_RULES = (
    "The template is a paragraph of less than 300 words that covers, in this order:\n"
    "  1. what the question supplies (a scenario, measurements, a quoted definition, a list of "
    "statements, ...) and what it asks for;\n"
    "  2. the solution path, step by step, at the level of operations (recall the governing "
    "relation, express the unknown in terms of the givens, ...);\n"
    "  3. the kind of distinction the decision turns on (an exact year, the scope of a definition, "
    "a sign convention, which condition governs, ...), phrased so that it would read the same "
    "whichever option were correct: never say which option, or which kind of option, is right;\n"
    "  4. how the options are built (ten close numeric values, near-synonymous labels, "
    "combinations of numbered statements, distinct alternatives, ...).\n"
    "It must be domain-free: no subject-specific nouns, names, numbers, units or quoted option "
    "text. Use placeholders such as 'the governing relation', 'the named work', 'the target "
    "quantity', 'the listed statements'. Write it so that a question from a different field with "
    "the same solution shape would get nearly the same paragraph. Do not comment on whether "
    "the question or its options are well-posed."
)


def build_instructions(anchors: list[dict] | None = None) -> str:
    steps = "\n".join(f"  {k}: {v}" for k, v in STEPS.items())
    challenges = "\n".join(f"  {k}: {v}" for k, v in CHALLENGES.items())
    text = (
        "You describe HOW a multiple-choice question is solved and where solvers go wrong. "
        "You may work the question out privately, but nothing you write may name, hint at, or "
        "narrow down the correct option. " + WELL_POSED + "\n\n"
        "Reasoning steps (use these keys only; list only the steps this question needs, in order):\n"
        + steps + "\n\n"
        "Reasoning challenges (use these keys only; the one to three most likely reasons a "
        "competent solver picks a wrong option, most likely first):\n" + challenges + "\n\n"
        + TEMPLATE_RULES
    )
    if anchors:
        # The same reference set in every call, so each label is calibrated
        # against fixed examples rather than against whatever else was in a batch.
        shown = []
        for a in anchors:
            shown.append("Question:\n" + D.render_question(a["question"], a["options"])
                         + "\nRecord:\n" + json.dumps({"steps": a["steps"], "challenges": a["challenges"],
                                                       "knowledge": a["knowledge"],
                                                       "template": a["template"]}))
        text += ("\n\nReference examples. Label the new question consistently with these:\n\n"
                 + "\n\n".join(shown))
    return text


def cost_text(tok_in: int, tok_out: int) -> str:
    if PRICE_PER_M is None:
        return ""
    return f" (~${tok_in / 1e6 * PRICE_PER_M[0] + tok_out / 1e6 * PRICE_PER_M[1]:.2f})"


def build_input(row: dict) -> str:
    return "Question:\n" + D.render_question(row["question"], row["options"])


# --- scrub check ----------------------------------------------------------------
# The template is meant to carry the shape of the solution, not its content. A
# cheap check: content words the template shares with the question text.
_STOP = set("""a an the of to in on for and or is are was were be been being by with as at from
that this these those it its which what who whom whose when where why how not no nor into
than then there their they them we you your our he she his her him if but so such can could
would should may might must do does did done has have had having will shall each any all some
one two more most other another only also very much many few between among over under after
before during about against without within along through following given using use used based
per via each same different both either neither whether across""".split())


def scrub_overlap(template: str, row: dict) -> list[str]:
    """Content words (>= 4 letters) the template shares with the question or options."""
    words = lambda s: {w for w in re.findall(r"[a-z][a-z\-]{3,}", s.lower()) if w not in _STOP}
    src = words(row["question"]) | words(" ".join(row["options"]))
    return sorted(words(template) & src)


# --- API ------------------------------------------------------------------------

def describe_one(client: OpenAI, row: dict, effort: str, max_output_tokens: int,
                 instructions: str, retries: int = 4) -> dict:
    delay = 2.0
    for attempt in range(retries):
        try:
            resp = client.responses.parse(
                model=MODEL,
                instructions=instructions,
                input=build_input(row),
                reasoning={"effort": effort},
                text_format=Description,
                max_output_tokens=max_output_tokens,
            )
            desc: Description = resp.output_parsed
            if desc is None:
                raise ValueError(f"no parsed output (status {resp.status})")
            usage = resp.usage
            return {
                "id": row["id"], "prompt_version": PROMPT_VERSION, "model": MODEL,
                "effort": effort,
                "steps": list(dict.fromkeys(desc.steps)),     # drop repeats, keep order
                "challenges": list(dict.fromkeys(desc.challenges)), "knowledge": desc.knowledge,
                "template": desc.template.strip(),
                "scrub_overlap": scrub_overlap(desc.template, row),
                "usage": {"input": usage.input_tokens, "output": usage.output_tokens,
                          "reasoning": getattr(getattr(usage, "output_tokens_details", None),
                                               "reasoning_tokens", None)},
            }
        except Exception as exc:                       # rate limits, transient 5xx, parse misses
            if attempt == retries - 1:
                return {"id": row["id"], "prompt_version": PROMPT_VERSION, "model": MODEL,
                        "error": f"{type(exc).__name__}: {exc}"[:500]}
            time.sleep(delay)
            delay *= 2


# --- vocabulary discovery ---------------------------------------------------------
# This pass lets the model propose the vocabulary while looking at many
# questions side by side: each call sees a random batch from across the fields
# and says what separates how they are solved and where solvers go wrong; a
# final call merges every batch's proposals into one list. The merged list is
# printed as Python literals to paste over STEPS and CHALLENGES (edit by hand
# first; then bump PROMPT_VERSION and run the labelling pass). Run it on a
# train split only, so no test question shapes the vocabulary.

DISCOVER_VERSION = "d2"        # v2 principles: steps and challenges, each with when it applies


class Proposal(BaseModel):
    key: str = Field(description="short snake_case name")
    definition: str = Field(description="one line, domain-free: what a solver does (for a step) or where the error comes from (for a challenge)")
    applies_when: str = Field(description="one line, domain-free: what in a question makes this item apply, concrete enough that two readers would agree for any question")
    members: list[int] = Field(description="1-based numbers of the questions in this batch the item applies to")


class Discovery(BaseModel):
    steps: list[Proposal] = Field(description="reasoning steps: distinct operations a careful solver performs; a question needs one to four; 5 to 12 of them")
    challenges: list[Proposal] = Field(description="reasoning challenges: why a competent solver picks a wrong option, named by where the error comes from; a question has one to three; 4 to 10 of them")
    notes: str = Field(description="at most 60 words: what most strongly separates how these questions are solved and where they go wrong")


class MergedItem(BaseModel):
    key: str
    definition: str = Field(description="one or two sentences, domain-free: what the item is, then what in a question makes it apply, and when it does not where a neighbouring item is close")
    merged_from: list[str] = Field(description="keys of the raw proposals folded into this item")
    coverage: int = Field(description="how many raw proposals (weighted by their member counts) this item absorbs")


class MergedVocabulary(BaseModel):
    steps: list[MergedItem]
    challenges: list[MergedItem]
    notes: str = Field(description="at most 100 words: what was dropped or folded together, and why")


DISCOVER_INSTRUCTIONS = (
    "You are given a batch of multiple-choice questions from many academic fields. "
    "Describe how they differ in HOW they are solved and WHERE solvers go wrong, "
    "ignoring topic. You may work the questions out privately; do not report answers.\n\n"
    "Propose two lists.\n"
    "1. Reasoning STEPS: the distinct operations a careful solver has to perform (recall "
    "something, work something out, apply something to a case, rank items, ...). A "
    "question usually needs one to four. Leave out anything nearly every question needs, "
    "such as reading the options or comparing them with the question.\n"
    "2. Reasoning CHALLENGES: the reasons a competent solver picks a wrong option, each "
    "named by where the error comes from (a missing piece of knowledge, a misreading, a "
    "slip, a confusion between similar things, ...). A question has one to three.\n\n"
    "For every item give a snake_case key, a one-line definition, an 'applies when' line "
    "naming what in a question makes the item apply (concrete enough that two readers "
    "would agree for any question), and the numbers of the questions in this batch it "
    "applies to. Every question must get at least one step and at least one challenge.\n\n"
    "Generality rules. Every item must apply to at least three questions in the batch "
    "and to no more than about three quarters of them. Keys, definitions and 'applies "
    "when' lines must make sense to a reader from a different field: no subject terms "
    "(no 'Reynolds', 'lesion', 'chord'). If an item can only be phrased with subject "
    "terms, it is too specific; find the shape it shares with other questions and name "
    "that instead.\n\n" + WELL_POSED + " Items describe the solver's reasoning and the solver's "
    "errors only, never defects in the question."
)

MERGE_INSTRUCTIONS = (
    "You are given proposals for a vocabulary of reasoning steps and reasoning "
    "challenges, collected from many independent batches of multiple-choice questions. "
    "Merge them into ONE vocabulary: {n_steps} steps and {n_challenges} challenges. Fold "
    "near-duplicates together, keep items that separate questions from each other, and "
    "drop items that apply to almost every question or almost none. Items in the same "
    "list must not overlap: for any question, a reader must be able to tell which of two "
    "similar items applies. Each merged definition is one or two domain-free sentences: "
    "what the item is, then what in a question makes it apply, and, where a neighbouring "
    "item is close, when it does not. Keys are snake_case. Report which raw keys went "
    "into each merged item. " + WELL_POSED + " Drop any proposal about defects in the question "
    "itself (ambiguity, missing or damaged information, overlapping or conflicting options)."
)

KINDS = ("steps", "challenges")


def discover_batches(rows: list[dict], n_batches: int, batch_size: int, rng: random.Random) -> list[list[dict]]:
    """Random batches without replacement while the pool lasts, so every batch
    mixes fields and difficulties."""
    order = rows[:]
    rng.shuffle(order)
    batches, pos = [], 0
    for _ in range(n_batches):
        if pos + batch_size > len(order):
            rng.shuffle(order)
            pos = 0
        batches.append(order[pos: pos + batch_size])
        pos += batch_size
    return batches


def discover_one(client: OpenAI, batch: list[dict], effort: str, max_output_tokens: int) -> dict:
    body = "\n\n".join(f"### Question {i + 1} [{r['discipline']}]\n"
                       + D.render_question(r["question"], r["options"])
                       for i, r in enumerate(batch))
    ids = [r["id"] for r in batch]
    try:
        resp = client.responses.parse(
            model=MODEL, instructions=DISCOVER_INSTRUCTIONS, input=body,
            reasoning={"effort": effort}, text_format=Discovery,
            max_output_tokens=max_output_tokens)
        out: Discovery = resp.output_parsed
        if out is None:
            raise ValueError(f"no parsed output (status {resp.status})")
        to_ids = lambda nums: [ids[n - 1] for n in nums if 1 <= n <= len(ids)]
        rec = {"ids": ids, "version": DISCOVER_VERSION, "model": MODEL, "notes": out.notes,
               "usage": {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}}
        for kind in KINDS:
            rec[kind] = [{"key": p.key, "definition": p.definition, "applies_when": p.applies_when,
                          "members": to_ids(p.members)} for p in getattr(out, kind)]
        return rec
    except Exception as exc:
        return {"ids": ids, "version": DISCOVER_VERSION, "error": f"{type(exc).__name__}: {exc}"[:500]}


def merge_proposals(client: OpenAI, raw: list[dict], n_steps: int, n_challenges: int,
                    effort: str) -> MergedVocabulary:
    def table(kind: str) -> str:
        lines = []
        for b, rec in enumerate(raw):
            for p in rec[kind]:
                lines.append(f"- [b{b}] {p['key']}: {p['definition']} Applies when: {p['applies_when']} "
                             f"(applied to {len(p['members'])} of {len(rec['ids'])})")
        return "\n".join(lines)
    body = ("## Proposed reasoning steps\n" + table("steps")
            + "\n\n## Proposed reasoning challenges\n" + table("challenges")
            + "\n\n## Batch notes\n" + "\n".join(f"- {r['notes']}" for r in raw))
    resp = client.responses.parse(
        model=MODEL, instructions=MERGE_INSTRUCTIONS.format(n_steps=n_steps, n_challenges=n_challenges),
        input=body, reasoning={"effort": effort}, text_format=MergedVocabulary,
        max_output_tokens=32000)
    if resp.output_parsed is None:
        raise ValueError(f"merge call returned no parsed output (status {resp.status})")
    return resp.output_parsed


def as_literal(name: str, items: list[MergedItem]) -> str:
    width = max(len(i.key) for i in items) + 4
    body = "\n".join(f"    {(repr(i.key) + ':'):{width}} {i.definition!r}," for i in items)
    return f"{name} = {{\n{body}\n}}"


def discover(args) -> None:
    rng = random.Random(args.seed)
    rows = json.loads(args.dataset.read_text())
    batches = discover_batches(rows, args.n_batches, args.batch_size, rng)
    print(f"discovery: {len(batches)} batches x {args.batch_size} questions from {args.dataset.name} "
          f"({MODEL}, effort {args.effort}, {DISCOVER_VERSION})")
    client = OpenAI()
    raw_path = args.out.with_suffix(".raw.jsonl")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Batch proposals are cached: a rerun with the same raw file only redoes
    # the merge (so --n-steps/--n-challenges can be changed for free). Delete
    # the raw file, or point --out elsewhere, to sample new batches.
    raw = []
    if raw_path.exists():
        recs = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
        raw = [r for r in recs if "error" not in r and r.get("version") == DISCOVER_VERSION
               and r.get("model") == MODEL]
        if not raw:
            sys.exit(f"{raw_path} holds no {DISCOVER_VERSION} batches from {MODEL}; "
                     "move it aside or pass another --out")
        print(f"reusing {len(raw)} batches from {raw_path}; only the merge is rerun")
    else:
        with raw_path.open("w") as fh, ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(discover_one, client, b, args.effort, args.max_output_tokens)
                    for b in batches]
            for fut in tqdm(as_completed(futs), total=len(futs), unit="batch"):
                rec = fut.result()
                fh.write(json.dumps(rec) + "\n")
                if "error" in rec:
                    tqdm.write(f"ERROR: {rec['error']}")
                else:
                    raw.append(rec)
        tok_in = sum(r["usage"]["input"] for r in raw)
        tok_out = sum(r["usage"]["output"] for r in raw)
        print(f"{len(raw)} batches ok; tokens in {tok_in}, out {tok_out} "
              + cost_text(tok_in, tok_out))
    if not raw:
        sys.exit("every batch failed")
    print("\nraw proposal keys, most frequent first:")
    for kind in KINDS:
        c = Counter(p["key"] for r in raw for p in r[kind])
        print(f"  {kind}: " + ", ".join(f"{k}({n})" for k, n in c.most_common(25)))

    merged = merge_proposals(client, raw, args.n_steps, args.n_challenges, args.effort)
    args.out.write_text(json.dumps(merged.model_dump(), indent=1))
    print(f"\nwrote {raw_path} and {args.out}\n")
    print("# --- paste over STEPS / CHALLENGES in describe_supergpqa_v3.py after editing ---")
    print(as_literal("STEPS", merged.steps))
    print()
    print(as_literal("CHALLENGES", merged.challenges))
    print(f"\n# merge notes: {merged.notes}")
    for kind in KINDS:
        print(f"\n{kind}:")
        for i in getattr(merged, kind):
            print(f"  {i.key:34s} coverage {i.coverage:>3}  <- {', '.join(i.merged_from)[:120]}")


def load_done(path: Path) -> dict[str, dict]:
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("prompt_version") == PROMPT_VERSION and rec.get("model") == MODEL \
                    and "error" not in rec:
                done[rec["id"]] = rec
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=ROOT / "datasets/supergpqa_2k_train.json")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/describe_v3/templates_train.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="first N questions only (smoke test)")
    ap.add_argument("--effort", default="medium", choices=["none", "low", "medium", "high"])
    ap.add_argument("--max-output-tokens", type=int, default=4000,
                    help="cap per call; reasoning tokens count against it")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--show", action="store_true", help="print each record as it arrives")
    anc = ap.add_argument_group("anchors: fixed reference examples shown in every labelling call")
    anc.add_argument("--anchors", type=Path, default=None,
                     help="JSON list of hand-checked records (question, options, steps, challenges, knowledge, template)")
    anc.add_argument("--draft-anchors", type=int, default=None, metavar="N",
                     help="label N random questions without anchors, write them to --anchors for hand correction, stop")
    disc = ap.add_argument_group("vocabulary discovery (--discover)")
    disc.add_argument("--discover", action="store_true",
                      help="propose the step/challenge vocabulary from batches of questions instead of labelling")
    disc.add_argument("--n-batches", type=int, default=25)
    disc.add_argument("--batch-size", type=int, default=40)
    disc.add_argument("--n-steps", type=int, default=10, help="size of the merged step list")
    disc.add_argument("--n-challenges", type=int, default=8, help="size of the merged challenge list")
    disc.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.discover:
        if args.out == ap.get_default("out"):
            args.out = ROOT / f"outputs/vocab_discovery_{args.dataset.stem.removeprefix('supergpqa_')}.json"
        if args.max_output_tokens == ap.get_default("max_output_tokens"):
            args.max_output_tokens = 32000       # a whole batch's lists, plus reasoning
        discover(args)
        return

    rows = json.loads(args.dataset.read_text())
    if args.draft_anchors:
        # label a spread of questions without anchors, write them out for hand
        # correction, and stop; the corrected file is then passed as --anchors
        if args.anchors.exists():
            sys.exit(f"{args.anchors} already exists; it may hold hand edits. "
                     "Move it aside or pass a different --anchors path.")
        rng = random.Random(args.seed)
        rows = rng.sample(rows, args.draft_anchors)
        args.out = args.anchors
        args.anchors = None
        print(f"drafting {len(rows)} anchor records into {args.out}; correct them by hand")
    anchors = None
    if args.anchors:
        anchors = json.loads(args.anchors.read_text())
        if any(a.get("prompt_version") != PROMPT_VERSION for a in anchors):
            sys.exit(f"anchors in {args.anchors} were labelled under another prompt version")
        stale = sorted({k for a in anchors for k in a["steps"] if k not in STEPS}
                       | {k for a in anchors for k in a["challenges"] if k not in CHALLENGES})
        if stale:
            sys.exit(f"anchors in {args.anchors} use keys outside the current vocabulary: {stale}")
        print(f"{len(anchors)} anchor examples in every call")
    instructions = build_instructions(anchors)
    if args.limit is not None:
        rows = rows[: args.limit]
    done = {} if args.draft_anchors else load_done(args.out)
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} questions, {len(done)} already described, {len(todo)} to do "
          f"({MODEL}, effort {args.effort}, prompt {PROMPT_VERSION})")
    if not todo:
        return

    client = OpenAI()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_err = 0
    tok_in = tok_out = 0
    drafted = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(describe_one, client, r, args.effort, args.max_output_tokens,
                            instructions): r for r in todo}
        if args.draft_anchors:
            for fut in tqdm(as_completed(futs), total=len(futs), unit="q"):
                rec, row = fut.result(), futs[fut]
                if "error" in rec:
                    tqdm.write(f"ERROR {rec['id']}: {rec['error']}")
                    continue
                drafted.append({"id": row["id"], "discipline": row["discipline"],
                                "question": row["question"], "options": row["options"],
                                "steps": rec["steps"], "challenges": rec["challenges"],
                                "knowledge": rec["knowledge"], "template": rec["template"],
                                "prompt_version": PROMPT_VERSION})
            drafted.sort(key=lambda a: a["discipline"])
            args.out.write_text(json.dumps(drafted, indent=1, ensure_ascii=False))
            print(f"wrote {len(drafted)} draft anchors to {args.out}")
            return
        fh = args.out.open("a")
        for fut in tqdm(as_completed(futs), total=len(futs), unit="q"):
            rec = fut.result()
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if "error" in rec:
                n_err += 1
                tqdm.write(f"ERROR {rec['id']}: {rec['error']}")
                continue
            tok_in += rec["usage"]["input"]
            tok_out += rec["usage"]["output"]
            if args.show:
                row = futs[fut]
                tqdm.write(f"\n[{row['discipline']} / {row['field']}] {row['question'][:160]}")
                tqdm.write(f"  steps={rec['steps']} challenges={rec['challenges']} knowledge={rec['knowledge']}")
                tqdm.write(f"  template: {rec['template']}")
                if rec["scrub_overlap"]:
                    tqdm.write(f"  overlap with question: {rec['scrub_overlap']}")
        fh.close()
    n_ok = len(todo) - n_err
    print(f"done: {n_ok} ok, {n_err} errors; tokens in {tok_in}, out {tok_out} "
          + cost_text(tok_in, tok_out))
    if n_err:
        print("errors are not cached; rerun the same command to retry them")


if __name__ == "__main__":
    main()

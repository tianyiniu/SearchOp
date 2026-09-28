"""Step 1 of the per-type search: describe every question's reasoning shape.

A stronger model reads each question WITH its options and returns a small
structured record: the ordered list of reasoning moves a solver would need
(from a fixed vocabulary), the most likely way to get it wrong, whether the
answer is recalled or derived, and a one-line scrubbed template. The record is
used for grouping and routing only. It is never shown to the debate agents, so
nothing the stronger model knows leaks into the debate.

Uses the OpenAI Responses API (not the chat-completions API that the vLLM
servers speak) with a JSON schema so every record parses. Output is a JSONL
file keyed by question id; rerunning skips ids already done, so an interrupted
run resumes for free. Records carry a prompt-version tag; bump PROMPT_VERSION
after editing the prompt and the old records are ignored.

Order of operations:

    # 0. let the model propose the vocabulary from batches of questions seen
    #    side by side; paste the printed STEPS/RISKS over the ones below, edit,
    #    bump PROMPT_VERSION
    python3 scripts/describe_questions.py --discover --n-batches 25 --batch-size 40
    # 1. draft ~10 anchor records, correct them by hand in the JSON
    python3 scripts/describe_questions.py --draft-anchors 10 --anchors outputs/anchors.json
    # 2. label the train split with the anchors in every call
    python3 scripts/describe_questions.py --anchors outputs/anchors.json \\
        --dataset datasets/supergpqa_program_search_train.json \\
        --out outputs/question_templates_train.jsonl --workers 8
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

PROMPT_VERSION = "v1"
MODEL = "gpt-5.6-terra"

# --- the vocabulary -----------------------------------------------------------
# Reasoning moves. The describer lists the ones a solver needs, in order. Keep
# this short: the multi-hot vector over it is one of the two things we cluster.
STEPS = {
    'retrieve_canonical_knowledge':         'Recall a specific established fact, association, date, name, value, or conventional label needed to answer the prompt.',
    'classify_by_defining_features':        'Match a description, function, or feature set to the category, term, or member defined by it.',
    'infer_from_joint_clues':               'Combine the decisive observations in a case to infer the explanation, state, mechanism, or response best supported by all of them.',
    'apply_rules_and_formal_constraints':   'Identify the controlling rule and determine which candidate satisfies every required condition, exception, or logical constraint.',
    'formulate_quantitative_model':         'Translate stated quantities, dependencies, constraints, and initial conditions into a relation or model that determines the answer.',
    'execute_quantitative_derivation':      'Substitute into a valid model and carry out the arithmetic, algebraic, or staged derivation to obtain the requested result.',
    'normalize_units_and_representations':  'Put quantities, units, signs, reference levels, and equivalent representations into a consistent form before comparing results.',
    'reason_about_relations_and_order':     'Determine an answer from relative position, direction, sequence, dependence, extremity, or structural relations among elements.',
    'audit_compound_claims':                'Evaluate each component of a bundled option independently and select only the combination with exactly the supported components.',
    'discriminate_option_precision':        "Compare choices against the prompt's exact qualifiers, scope, wording, completeness, and requested level of specificity.",
}

RISKS = {
    'confusable_fact_recall':                'A nearby fact, date, name, value, or association is recalled instead of the exact requested one.',
    'near_label_or_category_confusion':      "A related term or category is chosen despite failing the target's exact defining boundary or property.",
    'qualifier_or_scope_oversight':          'A limiting condition, exception, quantifier, requested purpose, or completeness requirement is overlooked.',
    'wrong_model_or_rule_application':       'An inapplicable governing relation, rule, constraint, or convention is used to model the situation.',
    'calculation_scale_or_conversion_slip':  'A sound approach yields a wrong result through arithmetic, unit conversion, scale, sign, factor, or reference handling.',
    'direction_or_mechanism_reversal':       'The causal direction, ordering, comparison basis, baseline, or mechanism is reversed or misconstrued.',
    'compound_option_bookkeeping_error':     'A multi-part answer is accepted without independently checking every included and omitted claim.',
}
KNOWLEDGE = ("recall", "derive", "both")

Step = Literal[tuple(STEPS)]          # type: ignore[valid-type]
Risk = Literal[tuple(RISKS)]          # type: ignore[valid-type]
Knowledge = Literal[KNOWLEDGE]        # type: ignore[valid-type]


class Description(BaseModel):
    steps: list[Step] = Field(description="Reasoning moves in the order a careful solver would take them, 1 to 5 entries, no repeats.")
    risk: Risk = Field(description="The single most likely way a competent solver gets this question wrong.")
    knowledge: Knowledge = Field(description="Whether the answer is mainly recalled, mainly derived, or both.")
    template: str = Field(description="One sentence, at most 25 words, describing the solution path with every domain-specific noun, number and option replaced by a placeholder such as 'the quantity' or 'the rule'.")


def build_instructions(anchors: list[dict] | None = None) -> str:
    steps = "\n".join(f"  {k}: {v}" for k, v in STEPS.items())
    risks = "\n".join(f"  {k}: {v}" for k, v in RISKS.items())
    text = (
        "You describe HOW a multiple-choice question is solved, not what the answer is. "
        "Do not solve it and do not name or hint at the correct option.\n\n"
        "Reasoning moves (use these keys only):\n" + steps + "\n\n"
        "Failure risks (use these keys only):\n" + risks + "\n\n"
        "The template must be domain-free: no subject-specific nouns, no numbers, no "
        "quoted option text. Write it so that a question from a different field with "
        "the same solution shape would get the same template."
    )
    if anchors:
        # The same reference set in every call, so each label is calibrated
        # against fixed examples rather than against whatever else was in a batch.
        shown = []
        for a in anchors:
            shown.append("Question:\n" + D.render_question(a["question"], a["options"])
                         + "\nRecord:\n" + json.dumps({"steps": a["steps"], "risk": a["risk"],
                                                       "knowledge": a["knowledge"],
                                                       "template": a["template"]}))
        text += ("\n\nReference examples. Label the new question consistently with these:\n\n"
                 + "\n\n".join(shown))
    return text


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
                "risk": desc.risk, "knowledge": desc.knowledge,
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
# The hand-written STEPS and RISKS above are guesses. This pass lets the model
# propose the vocabulary while looking at many questions side by side: each
# call sees a random batch from across the fields and says what separates how
# they are solved; a final call merges every batch's proposals into one list.
# The merged list is printed as Python literals to paste over STEPS and RISKS
# (edit by hand first; then bump PROMPT_VERSION and run the labelling pass).

class Proposal(BaseModel):
    key: str = Field(description="short snake_case name")
    definition: str = Field(description="one line, domain-free, saying what a solver does (for a move) or how they go wrong (for a risk)")
    members: list[int] = Field(description="1-based numbers of the questions in this batch the item applies to")


class Discovery(BaseModel):
    moves: list[Proposal] = Field(description="reasoning moves: reusable atomic steps; a question may need several; 5 to 12 of them")
    risks: list[Proposal] = Field(description="ways a competent solver most often gets these questions wrong; 4 to 8 of them")
    notes: str = Field(description="at most 60 words: what most strongly separates how these questions are solved")


class MergedItem(BaseModel):
    key: str
    definition: str
    merged_from: list[str] = Field(description="keys of the raw proposals folded into this item")
    coverage: int = Field(description="how many raw proposals (weighted by their member counts) this item absorbs")


class MergedVocabulary(BaseModel):
    moves: list[MergedItem]
    risks: list[MergedItem]
    notes: str = Field(description="at most 100 words: what was dropped and why")


DISCOVER_INSTRUCTIONS = (
    "You are given a batch of multiple-choice questions from many academic fields. "
    "Your job is to describe how they differ in HOW they are solved, ignoring topic. "
    "Do not solve them.\n\n"
    "Propose two lists.\n"
    "1. Reasoning MOVES: atomic, reusable steps a careful solver performs (retrieve a "
    "fact, apply a formula, set up a model of the situation, rank items, rule out "
    "options, ...). A question usually needs several moves. Prefer moves that split "
    "this batch into different groups over moves that apply to everything. Do not "
    "name moves after subjects.\n"
    "2. Failure RISKS: the single most likely way a competent solver gets each "
    "question wrong (a slip, a misreading, a missing fact, two defensible options, ...).\n\n"
    "For every item list which questions in the batch it applies to, by number. Every "
    "question must get at least one move and exactly one risk.\n\n"
    "Generality rules. Every item must apply to at least three questions in the batch. "
    "Every key and definition must make sense to a reader from a different field: no "
    "subject terms (no 'Reynolds', 'lesion', 'chord'). If an item can only be phrased "
    "with subject terms, it is too specific; find the shape it shares with other "
    "questions and name that instead."
)

MERGE_INSTRUCTIONS = (
    "You are given proposals for a vocabulary of reasoning moves and failure risks, "
    "collected from many independent batches of multiple-choice questions. Merge them "
    "into ONE vocabulary: {n_moves} moves and {n_risks} risks. Fold near-duplicates "
    "together, keep items that separate questions from each other, and drop items "
    "that apply to almost every question or almost none. Definitions must be "
    "domain-free and one line each. Keys are snake_case. Report which raw keys went "
    "into each merged item."
)


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
    try:
        resp = client.responses.parse(
            model=MODEL, instructions=DISCOVER_INSTRUCTIONS, input=body,
            reasoning={"effort": effort}, text_format=Discovery,
            max_output_tokens=max_output_tokens)
        out: Discovery = resp.output_parsed
        if out is None:
            raise ValueError(f"no parsed output (status {resp.status})")
        ids = [r["id"] for r in batch]
        to_ids = lambda nums: [ids[n - 1] for n in nums if 1 <= n <= len(ids)]
        return {"ids": ids,
                "moves": [{"key": p.key, "definition": p.definition, "members": to_ids(p.members)}
                          for p in out.moves],
                "risks": [{"key": p.key, "definition": p.definition, "members": to_ids(p.members)}
                          for p in out.risks],
                "notes": out.notes,
                "usage": {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}}
    except Exception as exc:
        return {"ids": [r["id"] for r in batch], "error": f"{type(exc).__name__}: {exc}"[:500]}


def merge_proposals(client: OpenAI, raw: list[dict], n_moves: int, n_risks: int,
                    effort: str) -> MergedVocabulary:
    def table(kind: str) -> str:
        lines = []
        for b, rec in enumerate(raw):
            for p in rec[kind]:
                lines.append(f"- [{kind[:-1]} b{b}] {p['key']}: {p['definition']} "
                             f"(applied to {len(p['members'])} of {len(rec['ids'])})")
        return "\n".join(lines)
    body = ("## Proposed moves\n" + table("moves") + "\n\n## Proposed risks\n" + table("risks")
            + "\n\n## Batch notes\n" + "\n".join(f"- {r['notes']}" for r in raw))
    resp = client.responses.parse(
        model=MODEL, instructions=MERGE_INSTRUCTIONS.format(n_moves=n_moves, n_risks=n_risks),
        input=body, reasoning={"effort": effort}, text_format=MergedVocabulary,
        max_output_tokens=16000)
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
    print(f"discovery: {len(batches)} batches x {args.batch_size} questions "
          f"({MODEL}, effort {args.effort})")
    client = OpenAI()
    raw_path = args.out.with_suffix(".raw.jsonl")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Batch proposals are cached: a rerun with the same raw file only redoes
    # the merge (so --n-moves/--n-risks can be changed for free). Delete the
    # raw file, or point --out elsewhere, to sample new batches.
    raw = []
    if raw_path.exists():
        raw = [r for r in map(json.loads, raw_path.read_text().splitlines()) if "error" not in r]
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
              f"(~${tok_in / 1e6 * 2 + tok_out / 1e6 * 12:.2f})")
    if not raw:
        sys.exit("every batch failed")
    print("\nraw proposal keys, most frequent first:")
    for kind in ("moves", "risks"):
        c = Counter(p["key"] for r in raw for p in r[kind])
        print(f"  {kind}: " + ", ".join(f"{k}({n})" for k, n in c.most_common(25)))

    merged = merge_proposals(client, raw, args.n_moves, args.n_risks, args.effort)
    args.out.write_text(json.dumps(merged.model_dump(), indent=1))
    print(f"\nwrote {raw_path} and {args.out}\n")
    print("# --- paste over STEPS / RISKS in describe_questions.py after editing ---")
    print(as_literal("STEPS", merged.moves))
    print()
    print(as_literal("RISKS", merged.risks))
    print(f"\n# merge notes: {merged.notes}")
    for kind, items in (("moves", merged.moves), ("risks", merged.risks)):
        print(f"\n{kind}:")
        for i in items:
            print(f"  {i.key:24s} coverage {i.coverage:>3}  <- {', '.join(i.merged_from)[:120]}")


def load_done(path: Path) -> dict[str, dict]:
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("prompt_version") == PROMPT_VERSION and "error" not in rec:
                done[rec["id"]] = rec
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=ROOT / "datasets/supergpqa_program_search_train.json")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/question_templates_train.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="first N questions only (smoke test)")
    ap.add_argument("--effort", default="medium", choices=["none", "low", "medium", "high"])
    ap.add_argument("--max-output-tokens", type=int, default=4000,
                    help="cap per call; reasoning tokens count against it")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--show", action="store_true", help="print each record as it arrives")
    anc = ap.add_argument_group("anchors: fixed reference examples shown in every labelling call")
    anc.add_argument("--anchors", type=Path, default=None,
                     help="JSON list of hand-checked records (question, options, steps, risk, knowledge, template)")
    anc.add_argument("--draft-anchors", type=int, default=None, metavar="N",
                     help="label N random questions without anchors, write them to --anchors for hand correction, stop")
    disc = ap.add_argument_group("vocabulary discovery (--discover)")
    disc.add_argument("--discover", action="store_true",
                      help="propose the move/risk vocabulary from batches of questions instead of labelling")
    disc.add_argument("--n-batches", type=int, default=25)
    disc.add_argument("--batch-size", type=int, default=40)
    disc.add_argument("--n-moves", type=int, default=10, help="size of the merged move list")
    disc.add_argument("--n-risks", type=int, default=7, help="size of the merged risk list")
    disc.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.discover:
        if args.out == ap.get_default("out"):
            args.out = ROOT / "outputs/vocab_discovery.json"
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
                                "steps": rec["steps"], "risk": rec["risk"],
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
                tqdm.write(f"  steps={rec['steps']} risk={rec['risk']} knowledge={rec['knowledge']}")
                tqdm.write(f"  template: {rec['template']}")
                if rec["scrub_overlap"]:
                    tqdm.write(f"  overlap with question: {rec['scrub_overlap']}")
        fh.close()
    n_ok = len(todo) - n_err
    print(f"done: {n_ok} ok, {n_err} errors; tokens in {tok_in}, out {tok_out} "
          f"(~${tok_in / 1e6 * 2 + tok_out / 1e6 * 12:.2f} at $2/$12 per M)")
    if n_err:
        print("errors are not cached; rerun the same command to retry them")


if __name__ == "__main__":
    main()

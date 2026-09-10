"""(SuperGPQA) Evolve a debate schema per qwen_unanswerable reasoning question.

Same guided structural search as evolve_debate.py, moved to the MCQ setting:
  - executor : Qwen (thinking off) runs a schema and commits a LETTER.
  - grading  : exact letter match against the gold option — no judge.
  - modifier : a schema architect that proposes ONE structural edit from a fixed
               vocabulary (personas / final). It may see the answer (GT-aware) or
               not (--blind), but it NEVER writes facts — the information firewall.

Modes:
  (default)  GT-aware. The modifier knows the correct letter and what the executor
             produced. This is the train-time ORACLE upper bound on routing.
  --blind    Test-time. The modifier sees no correct answer; it edits on the
             executor's INSTABILITY across repeated runs, and stopping is by
             self-consistency (letter agreement), not correctness. judge/grade is
             logged only, never fed back. Gap vs GT-aware = value of the oracle.

Parallelism is across questions (each question's search is inherently sequential —
every edit depends on the previous execution). Raise --workers to fill the GPU.

    export OPENAI_API_KEY=...    # GT-aware modifier (gpt-5.4-mini); not needed with --modifier-backend vllm
    # vLLM serving Qwen/Qwen3-14B on port 7472
    python3 scripts/evolve_debate_mcq.py --limit 1000 --workers 16
    python3 scripts/evolve_debate_mcq.py --limit 1000 --blind --modifier-backend vllm --workers 16
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from threading import Lock

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import debate_mcq as D
from llm_judge import call_openai

try:  # load API keys from the project .env if python-dotenv is available
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


# --- modifier system prompts (MCQ; the firewall is the fixed vocabulary) ----

MODIFIER_SYSTEM = f"""You are a DEBATE-SCHEMA ARCHITECT running a guided search. A blind executor answers a hard multiple-choice question by running a "schema": an ordered list of rounds, where each round's personas answer in turn, seeing the question and all prior responses. The current schema gave a WRONG answer. You know the correct option and can see what the executor produced. Propose ONE small structural change that should help the executor reach the correct answer ON ITS OWN.

DO NOT LEAK THE ANSWER. You may ONLY choose structure from the fixed vocabularies below — never write facts or name the correct option.

Make the SMALLEST change that fixes the failure. Prefer: add a critic round when a deduction went unchecked; add a synthesizer when partial answers were not reconciled; repeat the solver for a second independent opinion; switch the final synthesis.

VOCABULARY
  personas: {list(D.PERSONAS)}  (solver=answer the question, critic=fault prior answers, synthesizer=commit one option)
  final: {list(D.FINALS)}  ('synthesizer' requires a synthesizer in the last round)

ACTIONS (choose exactly one):
  {{"action":"add_round","position":<1-indexed insert point>,"personas":[...]}}
  {{"action":"modify_round","position":<int>,"personas":[...]}}
  {{"action":"remove_round","position":<int>}}
  {{"action":"set_final","final":"..."}}
  {{"action":"give_up"}}

Respond with STRICT JSON only, no prose:
{{"diagnosis":"why it failed (1 sentence)","action":"...","position":<int if needed>,"personas":[...if needed],"final":"...if needed","rationale":"why this helps (1 sentence)"}}"""

BLIND_MODIFIER_SYSTEM = f"""You are a DEBATE-SCHEMA ARCHITECT improving how a blind executor reasons through a hard multiple-choice question. A schema is an ordered list of rounds; each round's personas answer in turn, seeing the question and all prior responses. You do NOT know the correct option. The executor's answers were UNSTABLE across repeated runs (low agreement), which signals the current structure is not reasoning reliably. Propose ONE small structural change that should make the reasoning more thorough and self-consistent.

You may ONLY choose structure from the fixed vocabularies below.

Prefer: add a critic round to verify an unchecked deduction; add a synthesizer to reconcile partial answers; repeat the solver for a second independent opinion; switch the final synthesis.

VOCABULARY
  personas: {list(D.PERSONAS)}  (solver=answer the question, critic=fault prior answers, synthesizer=commit one option)
  final: {list(D.FINALS)}  ('synthesizer' requires a synthesizer in the last round)

ACTIONS (choose exactly one):
  {{"action":"add_round","position":<1-indexed insert point>,"personas":[...]}}
  {{"action":"modify_round","position":<int>,"personas":[...]}}
  {{"action":"remove_round","position":<int>}}
  {{"action":"set_final","final":"..."}}
  {{"action":"give_up"}}

Respond with STRICT JSON only, no prose:
{{"diagnosis":"why the reasoning is unstable (1 sentence)","action":"...","position":<int if needed>,"personas":[...if needed],"final":"...if needed","rationale":"why this helps (1 sentence)"}}"""


# --- GT-aware evolution ----------------------------------------------------

def _modifier_prompt(question, gold, schema, answer, steps) -> str:
    history = "\n".join(f"  step {s['step']}: {s['action']} -> {'CORRECT' if s['correct'] else 'wrong'}"
                        for s in steps[-6:])
    return (f"QUESTION:\n  {question}\n\n"
            f"CORRECT OPTION (you know this; the executor does NOT):\n  {gold}\n\n"
            f"CURRENT SCHEMA:\n{json.dumps(schema)}\n\n"
            f"EXECUTOR'S WRONG ANSWER (letter):\n  {answer!r}\n\n"
            f"HISTORY:\n{history or '  (none)'}\n\n"
            "Propose ONE structural change as the JSON object specified.")


def evolve(client, model, modifier_fn, question, options, gold, args) -> dict:
    schema = deepcopy(D.MINIMAL_SCHEMA)
    steps: list[dict] = []

    def run(sch):
        return D.execute_schema(client, model, question, options, sch,
                                args.temperature, args.answer_tokens)

    def solved(sch, first_correct):
        if not first_correct:
            return False
        for _ in range(args.confirm_runs - 1):  # guard lucky rerolls
            if run(sch) != gold:
                return False
        return True

    answer = run(schema)
    correct = (answer == gold)
    steps.append({"step": 0, "action": "minimal", "schema": deepcopy(schema),
                  "answer": answer, "correct": correct})
    is_solved = solved(schema, correct)
    solved_at = 0 if is_solved else None

    mods = 0
    while not is_solved and mods < args.max_mods:
        op = None
        for _ in range(3):  # recovery tries on bad JSON / invalid edits
            raw = modifier_fn(MODIFIER_SYSTEM, _modifier_prompt(question, gold, schema, answer, steps))
            parsed = D.parse_edit(raw)
            if parsed is None:
                continue
            if parsed["action"] == "give_up":
                op = parsed
                break
            edited, _ = D.apply_edit(schema, parsed)
            if edited is not None and D.validate(edited):
                parsed["_edited"] = edited
                op = parsed
                break
        if op is None or op["action"] == "give_up":
            break

        mods += 1
        schema = op["_edited"]
        answer = run(schema)
        correct = (answer == gold)
        steps.append({"step": mods, "action": op["action"], "schema": deepcopy(schema),
                      "answer": answer, "correct": correct,
                      "diagnosis": str(op.get("diagnosis", ""))[:200],
                      "rationale": str(op.get("rationale", ""))[:200]})
        if solved(schema, correct):
            is_solved, solved_at = True, mods

    return {"solved": is_solved, "solved_at": solved_at, "n_mods": mods,
            "final_schema": schema, "mode": "gt_aware", "steps": steps}


# --- blind (test-time) evolution -------------------------------------------

def _consistency(client, model, question, options, schema, args) -> tuple[str | None, float]:
    """Run the schema n times; return (majority letter, agreement fraction)."""
    letters = [D.execute_schema(client, model, question, options, schema,
                                args.temperature, args.answer_tokens)
               for _ in range(max(1, args.consistency_samples))]
    tally = Counter(l for l in letters if l)
    if not tally:
        return None, 0.0
    letter, cnt = tally.most_common(1)[0]
    return letter, cnt / len(letters)


def _blind_modifier_prompt(question, schema, answer, agreement, steps) -> str:
    history = "\n".join(f"  step {s['step']}: {s['action']} -> agreement {s.get('agreement', 0):.0%}"
                        for s in steps[-6:])
    return (f"QUESTION:\n  {question}\n\n"
            f"CURRENT SCHEMA:\n{json.dumps(schema)}\n\n"
            f"EXECUTOR'S ANSWER (majority letter over repeated runs):\n  {answer!r}\n"
            f"AGREEMENT ACROSS RUNS: {agreement:.0%}  (low = unstable reasoning)\n\n"
            f"HISTORY:\n{history or '  (none)'}\n\n"
            "Propose ONE structural change as the JSON object specified.")


def evolve_blind(client, model, modifier_fn, question, options, gold, args) -> dict:
    schema = deepcopy(D.MINIMAL_SCHEMA)
    steps: list[dict] = []

    answer, agreement = _consistency(client, model, question, options, schema, args)
    steps.append({"step": 0, "action": "minimal", "schema": deepcopy(schema),
                  "answer": answer, "agreement": agreement, "correct": (answer == gold)})
    confident = agreement >= args.confidence_threshold
    stop_at = 0 if confident else None

    mods = 0
    while not confident and mods < args.max_mods:
        op = None
        for _ in range(3):
            raw = modifier_fn(BLIND_MODIFIER_SYSTEM,
                              _blind_modifier_prompt(question, schema, answer, agreement, steps))
            parsed = D.parse_edit(raw)
            if parsed is None:
                continue
            if parsed["action"] == "give_up":
                op = parsed
                break
            edited, _ = D.apply_edit(schema, parsed)
            if edited is not None and D.validate(edited):
                parsed["_edited"] = edited
                op = parsed
                break
        if op is None or op["action"] == "give_up":
            break

        mods += 1
        schema = op["_edited"]
        answer, agreement = _consistency(client, model, question, options, schema, args)
        steps.append({"step": mods, "action": op["action"], "schema": deepcopy(schema),
                      "answer": answer, "agreement": agreement, "correct": (answer == gold),
                      "diagnosis": str(op.get("diagnosis", ""))[:200],
                      "rationale": str(op.get("rationale", ""))[:200]})
        if agreement >= args.confidence_threshold:
            confident, stop_at = True, mods

    return {"solved": (answer == gold), "solved_at": stop_at if stop_at is not None else mods,
            "n_mods": mods, "final_agreement": agreement, "confident": confident,
            "final_schema": schema, "mode": "blind", "steps": steps}


# --- driver ----------------------------------------------------------------

def process_question(row, client, model, modifier_fn, args) -> dict:
    qid, question, gold = row["id"], row["question"], row["answer_letter"]
    options = list(row["options"])
    base = {"id": qid, "question": question, "answer_letter": gold,
            "field": row.get("field"), "difficulty": row.get("difficulty")}
    try:
        traj = (evolve_blind if args.blind else evolve)(
            client, model, modifier_fn, question, options, gold, args)
        return {**base, **traj, "status": "ok"}
    except Exception as exc:  # keep the pool going
        return {**base, "solved": False, "status": "error", "error": str(exc)}


def summarize(records: list[dict]) -> dict:
    ok = [r for r in records if r.get("status") == "ok"]
    solved = [r for r in ok if r.get("solved")]
    solved_at = Counter(r["solved_at"] for r in solved)
    finals = Counter(r["final_schema"]["final"] for r in solved)
    n_rounds = Counter(len(r["final_schema"]["rounds"]) for r in solved)
    return {
        "n_questions": len(records),
        "n_evaluated": len(ok),
        "solved": len(solved),
        "solve_rate": (len(solved) / len(ok)) if ok else 0.0,
        "avg_mods_to_solve": (sum(r["solved_at"] for r in solved) / len(solved)) if solved else 0.0,
        "solved_at_step": dict(sorted(solved_at.items())),
        "final_synthesis_of_solved": dict(finals),
        "n_rounds_of_solved": dict(sorted(n_rounds.items())),
        "errors": sum(1 for r in records if r.get("status") == "error"),
    }


def main(args: argparse.Namespace) -> None:
    rows = json.loads(Path(args.dataset).read_text())
    if args.limit is not None:
        rows = rows[: args.limit]
    # Convenience: a vllm modifier defaults to the served executor model.
    if args.modifier_backend == "vllm" and args.modifier_model == "gpt-5.4-mini":
        args.modifier_model = args.model

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    for path in (args.cache, args.summary_out):
        path.parent.mkdir(parents=True, exist_ok=True)

    if args.modifier_backend == "vllm":
        modifier_fn = lambda s, u: D.chat(client, args.modifier_model, s, u,
                                          args.modifier_temperature, max_tokens=1024)
    else:
        modifier_fn = lambda s, u: call_openai(s, u, model=args.modifier_model)
    print(f"mode={'BLIND (test-time)' if args.blind else 'gt-aware'}  "
          f"modifier={args.modifier_model} via {args.modifier_backend}  "
          f"| {len(rows)} questions, workers={args.workers}")

    records: list[dict] = []
    lock = Lock()
    with open(args.cache, "w") as cache, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_question, row, client, args.model, modifier_fn, args)
                   for row in rows]
        for future in tqdm(as_completed(futures), total=len(futures), desc="evolve", unit="q"):
            rec = future.result()
            with lock:
                cache.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cache.flush()
            records.append(rec)

    summary = summarize(records)
    args.summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nsolved by evolution: {summary['solved']}/{summary['n_evaluated']} "
          f"({summary['solve_rate']:.1%})")
    print(f"avg edits to solve: {summary['avg_mods_to_solve']:.2f}")
    print(f"solved at step: {summary['solved_at_step']}")
    print(f"final synthesis of solved: {summary['final_synthesis_of_solved']}")
    if summary["errors"]:
        print(f"errors: {summary['errors']}")
    print(f"trajectories -> {args.cache}\nsummary -> {args.summary_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path("datasets/supergpqa_qwen_unanswerable.json"))
    ap.add_argument("--model", default="Qwen/Qwen3-14B", help="Blind executor (vLLM).")
    ap.add_argument("--modifier-model", default="gpt-5.4-mini",
                    help="Schema architect. With --modifier-backend vllm, defaults to the served model.")
    ap.add_argument("--modifier-backend", choices=("openai", "vllm"), default="openai",
                    help="Where the modifier runs: 'openai' (frontier) or 'vllm' (local Qwen).")
    ap.add_argument("--modifier-temperature", type=float, default=0.4,
                    help="Modifier temperature (vllm backend only).")
    ap.add_argument("--blind", action="store_true",
                    help="Test-time mode: modifier sees no answer; stop on self-consistency.")
    ap.add_argument("--consistency-samples", type=int, default=3,
                    help="[--blind] executions per schema to measure letter agreement.")
    ap.add_argument("--confidence-threshold", type=float, default=0.67,
                    help="[--blind] agreement fraction that counts as 'confident' and stops editing.")
    ap.add_argument("--base-url", default="http://localhost:7472/v1", help="Local vLLM endpoint.")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--max-mods", type=int, default=6, help="Max structural edits per question.")
    ap.add_argument("--temperature", type=float, default=0.7, help="Executor temperature.")
    ap.add_argument("--answer-tokens", type=int, default=3072, help="Max tokens per model call (thinking off).")
    ap.add_argument("--confirm-runs", type=int, default=1,
                    help="Re-runs that must all be correct to count as solved (guards lucky rerolls).")
    ap.add_argument("--workers", type=int, default=16, help="Questions processed concurrently.")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N questions.")
    ap.add_argument("--cache", type=Path, default=Path("outputs/evolve_mcq_cache.jsonl"))
    ap.add_argument("--summary-out", type=Path, default=Path("outputs/evolve_mcq_summary.json"))
    main(ap.parse_args())

"""Evolve a debate schema per qwen_unanswerable question (a simplified evolve.py).

The fixed-library debate shapes help, so this searches for a *per-question* debate
structure instead of guessing one. A schema here is deliberately tiny — no tools,
no instruction axis, no nesting (the debate step has the documents already):

    schema = { rounds: [ {personas: [...]}, ... ], final: "last"|"vote"|"synthesizer" }

  personas: solver (answer from docs), critic (fault prior answers vs docs),
            synthesizer (commit one final answer). Each round's personas answer in
            turn, seeing the documents and all prior responses.

Three models:
  - executor (Qwen, ground-truth-BLIND) runs the schema on the cached doc context.
  - judge    (gpt-5.4-mini) decides CORRECT/INCORRECT.
  - modifier (gpt-5.4-mini, ground-truth-AWARE) proposes ONE structural edit.

Information firewall: the modifier sees the answer but may only choose STRUCTURE
from fixed vocabularies (personas / final) — it never writes facts, so it guides
the search without leaking the answer into what the blind executor runs.

Two ablations added on top of the original GT-aware loop:
  --modifier-backend vllm : run the schema architect on the LOCAL Qwen instead of
      gpt-5.4-mini. Tests whether the same model that executes can also evolve — if
      so the whole loop is local and free (#3a).
  --blind : the deployable, test-time loop. The modifier sees NO ground truth and
      NO correctness; it edits on the executor's INSTABILITY across repeated runs,
      and stopping is by self-consistency (answer agreement), not by the judge.
      judge_answer is used only to log final correctness. The gap between this and
      the GT-aware solve rate is the value of the oracle (#3b).

    export OPENAI_API_KEY=...                 # judge (+ modifier unless --modifier-backend vllm)
    python3 scripts/wiki_backend.py &         # local doc cache (cache misses)
    # vLLM serving Qwen/Qwen3-14B on port 7472
    python3 scripts/evolve_debate.py --limit 20                                   # GT-aware, gpt modifier
    python3 scripts/evolve_debate.py --limit 20 --modifier-backend vllm           # GT-aware, Qwen modifier
    python3 scripts/evolve_debate.py --limit 20 --blind --modifier-backend vllm   # test-time, Qwen modifier
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from threading import Lock

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from doc_qa import SummarizerConfig, build_and_cache_context
from llm_judge import call_openai, judge_answer
from tools import LOCAL_SCRAPE_URL, build_tools

try:  # load API keys from the project .env if python-dotenv is available
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# --- grammar ---------------------------------------------------------------

PERSONAS = ("solver", "critic", "synthesizer")
FINALS = ("last", "vote", "synthesizer")
MAX_ROUNDS = 6
MAX_PERSONAS = 4
EDIT_ACTIONS = ("add_round", "modify_round", "remove_round", "set_final", "give_up")

PERSONA_PROMPTS = {
    "solver": ("You are a Solver. Using ONLY the documents, answer the question. "
               "Reason step by step, then end with 'ANSWER: <answer>'. Always commit "
               "to a best guess; never abstain."),
    "critic": ("You are a Critic. Examine the prior answers against the documents. "
               "Identify the specific error or missing fact and state the corrected "
               "answer with evidence. End with 'ANSWER: <answer>'."),
    "synthesizer": ("You are a Synthesizer. Integrate the documents and the prior "
                    "answers into one final answer, resolving disagreements using the "
                    "documents. You MUST commit. End with 'ANSWER: <answer>'."),
}

MINIMAL_SCHEMA = {"rounds": [{"personas": ["solver"]}], "final": "last"}

MODIFIER_SYSTEM = f"""You are a DEBATE-SCHEMA ARCHITECT running a guided search. A blind executor answers a hard question by running a "schema": an ordered list of rounds, where each round's personas answer in turn, seeing the documents and all prior responses. The current schema gave a WRONG answer. You know the correct answer and can see what the executor produced. Propose ONE small structural change that should help the executor reach the correct answer ON ITS OWN.

DO NOT LEAK THE ANSWER. You may ONLY choose structure from the fixed vocabularies below — never write facts.

Make the SMALLEST change that fixes the failure. Prefer: add a critic round when a claim went unchecked; add a synthesizer when partial answers were not combined; repeat a perspective for a second opinion; switch the final synthesis.

VOCABULARY
  personas: {list(PERSONAS)}  (solver=answer from docs, critic=fault prior answers, synthesizer=commit one answer)
  final: {list(FINALS)}  ('synthesizer' requires a synthesizer in the last round)

ACTIONS (choose exactly one):
  {{"action":"add_round","position":<1-indexed insert point>,"personas":[...]}}
  {{"action":"modify_round","position":<int>,"personas":[...]}}
  {{"action":"remove_round","position":<int>}}
  {{"action":"set_final","final":"..."}}
  {{"action":"give_up"}}

Respond with STRICT JSON only, no prose:
{{"diagnosis":"why it failed (1 sentence)","action":"...","position":<int if needed>,"personas":[...if needed],"final":"...if needed","rationale":"why this helps (1 sentence)"}}"""

# Blind (test-time) modifier: it does NOT know the answer and gets NO correctness
# signal. It edits on the executor's INSTABILITY across repeated runs (low agreement
# = unreliable reasoning), so the whole loop is deployable without ground truth.
BLIND_MODIFIER_SYSTEM = f"""You are a DEBATE-SCHEMA ARCHITECT improving how a blind executor reasons over documents to answer a hard question. A schema is an ordered list of rounds; each round's personas answer in turn, seeing the documents and all prior responses. You do NOT know the correct answer. The executor's answers were UNSTABLE across repeated runs (low agreement), which signals the current structure is not reasoning reliably. Propose ONE small structural change that should make the reasoning more thorough and self-consistent.

You may ONLY choose structure from the fixed vocabularies below.

Prefer: add a critic round to verify unchecked claims; add a synthesizer to combine partial answers; repeat a perspective for a second opinion; switch the final synthesis.

VOCABULARY
  personas: {list(PERSONAS)}  (solver=answer from docs, critic=fault prior answers, synthesizer=commit one answer)
  final: {list(FINALS)}  ('synthesizer' requires a synthesizer in the last round)

ACTIONS (choose exactly one):
  {{"action":"add_round","position":<1-indexed insert point>,"personas":[...]}}
  {{"action":"modify_round","position":<int>,"personas":[...]}}
  {{"action":"remove_round","position":<int>}}
  {{"action":"set_final","final":"..."}}
  {{"action":"give_up"}}

Respond with STRICT JSON only, no prose:
{{"diagnosis":"why the reasoning is unstable (1 sentence)","action":"...","position":<int if needed>,"personas":[...if needed],"final":"...if needed","rationale":"why this helps (1 sentence)"}}"""


# --- schema validation + edit application ----------------------------------

def validate(schema: dict) -> bool:
    rounds = schema.get("rounds")
    if not rounds or not (1 <= len(rounds) <= MAX_ROUNDS):
        return False
    for rnd in rounds:
        ps = rnd.get("personas")
        if not ps or not (1 <= len(ps) <= MAX_PERSONAS) or any(p not in PERSONAS for p in ps):
            return False
    if schema.get("final") not in FINALS:
        return False
    if schema["final"] == "synthesizer" and "synthesizer" not in rounds[-1]["personas"]:
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


# --- executor (the blind Qwen that runs a schema) --------------------------

def chat(client, model, system, user, temperature, max_tokens=2048) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature, max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def extract_answer(text: str) -> str:
    upper = text.upper()
    if "ANSWER:" in upper:
        return text[upper.rfind("ANSWER:") + 7:].strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


def _normalize(text: str) -> str:
    a = extract_answer(text).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", a)).strip()


def execute_schema(client, model, question, context, schema, temperature) -> str:
    base = f"{context}\n\nQUESTION: {question}"
    prior: list[tuple[str, str]] = []
    all_rounds: list[list[tuple[str, str]]] = []
    for rnd in schema["rounds"]:
        responses = []
        for persona in rnd["personas"]:
            if prior:
                digest = "\n\n".join(f"[{p}]: {extract_answer(r)}" for p, r in prior)
                user = f"{base}\n\nPrior responses:\n{digest}\n\nProvide your response."
            else:
                user = f"{base}\n\nProvide your response."
            responses.append((persona, chat(client, model, PERSONA_PROMPTS[persona], user, temperature)))
        prior = prior + responses
        all_rounds.append(responses)

    last = all_rounds[-1]
    if schema["final"] == "synthesizer":
        for persona, resp in reversed(last):
            if persona == "synthesizer":
                return extract_answer(resp)
    if schema["final"] == "vote":
        keys = [_normalize(r) for _, r in last]
        tally = Counter(k for k in keys if k)
        if tally:
            winner = tally.most_common(1)[0][0]
            return next(extract_answer(r) for (_, r), k in zip(last, keys) if k == winner)
    return extract_answer(last[-1][1])  # "last" (and fallbacks)


# --- evolution loop --------------------------------------------------------

def _modifier_prompt(question, gt, schema, answer, steps) -> str:
    history = "\n".join(f"  step {s['step']}: {s['action']} -> {'CORRECT' if s['correct'] else 'wrong'}"
                        for s in steps[-6:])
    return (f"QUESTION:\n  {question}\n\n"
            f"CORRECT ANSWER (you know this; the executor does NOT):\n  {gt}\n\n"
            f"CURRENT SCHEMA:\n{json.dumps(schema)}\n\n"
            f"EXECUTOR'S WRONG ANSWER:\n  {answer.strip()[:400]!r}\n\n"
            f"HISTORY:\n{history or '  (none)'}\n\n"
            "Propose ONE structural change as the JSON object specified.")


def evolve(client, model, modifier_fn, question, gt, context,
           max_mods, temperature, confirm_runs) -> dict:
    schema = deepcopy(MINIMAL_SCHEMA)
    steps: list[dict] = []

    def solved(sch, first_answer, first_correct):
        """first run already done; require all remaining confirm runs correct too."""
        if not first_correct:
            return False
        for _ in range(confirm_runs - 1):
            if not judge_answer(question, gt, execute_schema(client, model, question, context, sch, temperature)):
                return False
        return True

    answer = execute_schema(client, model, question, context, schema, temperature)
    correct = judge_answer(question, gt, answer)
    steps.append({"step": 0, "action": "minimal", "schema": deepcopy(schema),
                  "answer": answer, "correct": correct})
    is_solved = solved(schema, answer, correct)
    solved_at = 0 if is_solved else None

    mods = 0
    while not is_solved and mods < max_mods:
        op = None
        for attempt in range(3):  # a couple of recovery tries on bad JSON / invalid edits
            raw = modifier_fn(MODIFIER_SYSTEM, _modifier_prompt(question, gt, schema, answer, steps))
            parsed = parse_edit(raw)
            if parsed is None:
                continue
            if parsed["action"] == "give_up":
                op = parsed
                break
            edited, _ = apply_edit(schema, parsed)
            if edited is not None and validate(edited):
                parsed["_edited"] = edited
                op = parsed
                break
        if op is None or op["action"] == "give_up":
            break

        mods += 1
        schema = op["_edited"]
        answer = execute_schema(client, model, question, context, schema, temperature)
        correct = judge_answer(question, gt, answer)
        steps.append({"step": mods, "action": op["action"], "schema": deepcopy(schema),
                      "answer": answer, "correct": correct,
                      "diagnosis": str(op.get("diagnosis", ""))[:200],
                      "rationale": str(op.get("rationale", ""))[:200]})
        if solved(schema, answer, correct):
            is_solved, solved_at = True, mods

    return {"id": None, "question": question, "ground_truth": gt, "solved": is_solved,
            "solved_at": solved_at, "n_mods": mods, "final_schema": schema,
            "mode": "gt_aware", "steps": steps}


# --- blind (test-time) evolution: no ground truth drives the search --------

def _consistency(client, model, question, context, schema, temperature, n) -> tuple[str, float]:
    """Run the schema n times; return (majority answer, agreement fraction). Agreement
    is the top answer-cluster's share — the self-consistency proxy for confidence that
    replaces the (unavailable at test time) correctness signal."""
    answers = [execute_schema(client, model, question, context, schema, temperature)
               for _ in range(max(1, n))]
    keys = [_normalize(a) for a in answers]
    tally = Counter(k for k in keys if k)
    if not tally:
        return answers[0], 0.0
    winner, cnt = tally.most_common(1)[0]
    rep = next(a for a, k in zip(answers, keys) if k == winner)
    return rep, cnt / len(answers)


def _blind_modifier_prompt(question, schema, answer, agreement, steps) -> str:
    history = "\n".join(f"  step {s['step']}: {s['action']} -> agreement {s.get('agreement', 0):.0%}"
                        for s in steps[-6:])
    return (f"QUESTION:\n  {question}\n\n"
            f"CURRENT SCHEMA:\n{json.dumps(schema)}\n\n"
            f"EXECUTOR'S ANSWER (majority over repeated runs):\n  {answer.strip()[:400]!r}\n"
            f"AGREEMENT ACROSS RUNS: {agreement:.0%}  (low = unstable reasoning)\n\n"
            f"HISTORY:\n{history or '  (none)'}\n\n"
            "Propose ONE structural change as the JSON object specified.")


def evolve_blind(client, model, modifier_fn, question, gt, context,
                 max_mods, temperature, samples, threshold) -> dict:
    """Deployable variant: the modifier never sees the answer, and STOPPING is driven
    by self-consistency (answer agreement >= threshold), not by a correctness judge.
    judge_answer is called ONLY to log final correctness for evaluation — never fed
    back into the loop."""
    schema = deepcopy(MINIMAL_SCHEMA)
    steps: list[dict] = []

    answer, agreement = _consistency(client, model, question, context, schema, temperature, samples)
    steps.append({"step": 0, "action": "minimal", "schema": deepcopy(schema),
                  "answer": answer, "agreement": agreement,
                  "correct": judge_answer(question, gt, answer)})  # log only
    confident = agreement >= threshold
    stop_at = 0 if confident else None

    mods = 0
    while not confident and mods < max_mods:
        op = None
        for _ in range(3):
            raw = modifier_fn(BLIND_MODIFIER_SYSTEM,
                              _blind_modifier_prompt(question, schema, answer, agreement, steps))
            parsed = parse_edit(raw)
            if parsed is None:
                continue
            if parsed["action"] == "give_up":
                op = parsed
                break
            edited, _ = apply_edit(schema, parsed)
            if edited is not None and validate(edited):
                parsed["_edited"] = edited
                op = parsed
                break
        if op is None or op["action"] == "give_up":
            break

        mods += 1
        schema = op["_edited"]
        answer, agreement = _consistency(client, model, question, context, schema, temperature, samples)
        steps.append({"step": mods, "action": op["action"], "schema": deepcopy(schema),
                      "answer": answer, "agreement": agreement,
                      "correct": judge_answer(question, gt, answer),  # log only
                      "diagnosis": str(op.get("diagnosis", ""))[:200],
                      "rationale": str(op.get("rationale", ""))[:200]})
        if agreement >= threshold:
            confident, stop_at = True, mods

    solved = judge_answer(question, gt, answer)  # final correctness, for reporting only
    return {"id": None, "question": question, "ground_truth": gt, "solved": solved,
            "solved_at": stop_at if stop_at is not None else mods, "n_mods": mods,
            "final_agreement": agreement, "confident": confident,
            "final_schema": schema, "mode": "blind", "steps": steps}


def process_question(row, client, model, modifier_fn, fetch, cfg, cache_dir, args) -> dict:  # noqa: C901
    qid, question, gt = row.get("id"), row["question"], row["ground_truth"]
    ctx = build_and_cache_context(client, row, fetch, cfg, cache_dir)
    context = ctx.get("doc_context", "")
    if not context:
        return {"id": qid, "question": question, "ground_truth": gt,
                "solved": False, "status": "skipped_no_context"}
    try:
        if args.blind:
            traj = evolve_blind(client, model, modifier_fn, question, gt, context,
                                args.max_mods, args.temperature,
                                args.consistency_samples, args.confidence_threshold)
        else:
            traj = evolve(client, model, modifier_fn, question, gt, context,
                          args.max_mods, args.temperature, args.confirm_runs)
        traj["id"], traj["status"] = qid, "ok"
        return traj
    except Exception as exc:  # keep the pool going
        return {"id": qid, "question": question, "ground_truth": gt,
                "solved": False, "status": "error", "error": str(exc)}


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
    }


def main(args: argparse.Namespace) -> None:
    rows = json.loads(Path(args.dataset).read_text())
    if args.limit is not None:
        rows = rows[: args.limit]
    # Convenience: a vllm modifier defaults to the served executor model.
    if args.modifier_backend == "vllm" and args.modifier_model == "gpt-5.4-mini":
        args.modifier_model = args.model

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    fetch = build_tools(["fetch_url"], scrape_url=args.scrape_url)[0]["fetch_url"]
    cfg = SummarizerConfig(model=args.model, context_window=args.context_window,
                           summary_tokens=args.summary_tokens)
    for path in (args.cache, args.summary_out):
        path.parent.mkdir(parents=True, exist_ok=True)

    # The modifier (schema architect) is either the frontier OpenAI model or the
    # local Qwen served by vLLM — the ablation for "can Qwen evolve schemas too?".
    if args.modifier_backend == "vllm":
        modifier_fn = lambda s, u: chat(client, args.modifier_model, s, u, args.modifier_temperature)
    else:
        modifier_fn = lambda s, u: call_openai(s, u, model=args.modifier_model)
    print(f"mode={'BLIND (test-time)' if args.blind else 'gt-aware'}  "
          f"modifier={args.modifier_model} via {args.modifier_backend}")

    records: list[dict] = []
    lock = Lock()
    with open(args.cache, "w") as cache, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_question, row, client, args.model, modifier_fn,
                               fetch, cfg, args.cache_dir, args) for row in rows]
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
    print(f"trajectories -> {args.cache}\nsummary -> {args.summary_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path("datasets/frames_qwen_unanswerable.json"))
    ap.add_argument("--model", default="Qwen/Qwen3-14B", help="Blind executor (vLLM).")
    ap.add_argument("--modifier-model", default="gpt-5.4-mini",
                    help="Schema architect model. With --modifier-backend vllm, set this to the "
                         "served model, e.g. Qwen/Qwen3-14B.")
    ap.add_argument("--modifier-backend", choices=("openai", "vllm"), default="openai",
                    help="Where the modifier runs: 'openai' (frontier) or 'vllm' (local Qwen).")
    ap.add_argument("--modifier-temperature", type=float, default=0.4,
                    help="Modifier temperature (vllm backend only).")
    ap.add_argument("--blind", action="store_true",
                    help="Test-time mode: modifier sees no ground truth; stop on self-consistency, "
                         "not on a correctness judge. The deployable variant.")
    ap.add_argument("--consistency-samples", type=int, default=3,
                    help="[--blind] executions per schema to measure answer agreement.")
    ap.add_argument("--confidence-threshold", type=float, default=0.67,
                    help="[--blind] agreement fraction that counts as 'confident' and stops editing.")
    ap.add_argument("--base-url", default="http://localhost:7472/v1", help="Local vLLM endpoint.")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--scrape-url", default=LOCAL_SCRAPE_URL,
                    help="fetch_url backend for cache misses (local Wikipedia cache by default).")
    ap.add_argument("--max-mods", type=int, default=6, help="Max structural edits per question.")
    ap.add_argument("--temperature", type=float, default=0.7, help="Executor temperature.")
    ap.add_argument("--confirm-runs", type=int, default=1,
                    help="Re-runs that must all be correct to count as solved (guards lucky rerolls).")
    ap.add_argument("--context-window", type=int, default=32768)
    ap.add_argument("--summary-tokens", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=5, help="Questions processed concurrently.")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N questions.")
    ap.add_argument("--cache-dir", type=Path, default=Path("doc_context_cache"),
                    help="doc_context cache (must match cache_doc_summary's config).")
    ap.add_argument("--cache", type=Path, default=Path("outputs/evolve_debate_cache.jsonl"))
    ap.add_argument("--summary-out", type=Path, default=Path("outputs/evolve_debate_summary.json"))
    main(ap.parse_args())

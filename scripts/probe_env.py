"""Is this environment ready for a live program-evolution run on a given model?

Exercises the exact code paths the run uses, in order, and stops at the first
failure with a plain explanation:

  1. the venv has the imports the scripts need
  2. the dataset and the two original recordings are present and readable
  3. vLLM answers at --base-url and serves exactly --model
  4. one debate round through debate_mcq.execute_round (thinking off) returns
     a response with a parseable ANSWER letter, for a solver AND for a persona
     that reads prior context (critic)
  5. the eliminator persona (commits no letter) still returns text
  6. BudgetedRunner runs a round live, writes it to a scratch cache, and
     replays it on a second call without touching the model
  7. a 32-call burst through the runner (throughput estimate, error count)
  8. --live-cache / output paths are writable and hold no other run's lock

    python scripts/probe_env.py --model Qwen/Qwen3.5-27B
    python scripts/probe_env.py --model Qwen/Qwen3.5-27B --live-cache outputs/program_live_rounds_cache_qwen27b.jsonl

Prints "ALL TESTS PASS" on success, exit code 0; otherwise the failing step
and exit code 1. Costs about 40 model calls.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

FAILED = False


def step(label: str):
    print(f"[ {label} ]")


def ok(msg: str) -> None:
    print(f"   ok   {msg}")


def fail(msg: str, hint: str = "") -> None:
    global FAILED
    FAILED = True
    print(f"   FAIL {msg}")
    if hint:
        print(f"        -> {hint}")
    print("\nPROBE FAILED")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--base-url", default="http://localhost:7472/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--dataset", type=Path, default=ROOT / "datasets/supergpqa_strict_train.json")
    ap.add_argument("--cache", type=Path, default=ROOT / "outputs/adaptive_rounds_cache.jsonl")
    ap.add_argument("--treegrow-cache", type=Path, default=ROOT / "outputs/treegrow_rounds_cache.jsonl")
    ap.add_argument("--live-cache", type=Path, default=ROOT / "outputs/program_live_rounds_cache.jsonl")
    ap.add_argument("--burst", type=int, default=32, help="Concurrent calls for the throughput test.")
    ap.add_argument("--skip-recordings", action="store_true",
                    help="Do not require the two 14B recordings (a fresh-model run uses empty ones).")
    ap.add_argument("--v2", action="store_true",
                    help="Probe the v2 pipeline: careful-reasoning prompts, 6144-token replies and a "
                         "summary call per persona, so the throughput number matches the overnight run.")
    ap.add_argument("--answer-tokens", type=int, default=None,
                    help="max_tokens per reasoning call (default 6144 with --v2, else 3072).")
    args = ap.parse_args()
    if args.answer_tokens is None:
        args.answer_tokens = 6144 if args.v2 else 3072

    # 1 --------------------------------------------------------------------
    step("1. imports")
    try:
        import openai, scipy, tqdm  # noqa: F401
        import debate_mcq as D
        import schema_fitness as SF
        import adaptive_debate_mcq as B
        if args.v2:
            D.set_digest(300, 900)
            SF.set_v2(True)
        import evolve_program_mcq as M
    except Exception as exc:
        fail(f"import error: {exc!r}",
             "activate the venv: /nas-ssd2/tianyin4/cache/venvs/vllm-host/bin/python3")
    ok(f"python {sys.version.split()[0]}, openai {openai.__version__}, all project modules import")

    # 2 --------------------------------------------------------------------
    step("2. data files")
    if not args.dataset.exists():
        fail(f"dataset missing: {args.dataset}")
    rows = json.loads(args.dataset.read_text())
    if not rows or "options" not in rows[0] or "answer_letter" not in rows[0]:
        fail("dataset has an unexpected shape")
    ok(f"dataset: {len(rows)} questions")
    if not args.skip_recordings:
        for p in (args.cache, args.treegrow_cache):
            if not p.exists():
                fail(f"recording missing: {p}", "pass --skip-recordings for a fresh-model run with empty caches")
            with p.open(errors="replace") as fh:
                first = fh.readline()
            try:
                r = json.loads(first)
                assert {"q", "k", "r", "responses"} <= set(r)
            except Exception:
                fail(f"recording unreadable or wrong shape: {p}")
            ok(f"recording readable: {p.name} ({p.stat().st_size // 1_000_000} MB)")

    # 3 --------------------------------------------------------------------
    step("3. vLLM server")
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=60, max_retries=0)
    try:
        served = [m.id for m in client.models.list().data]
    except Exception as exc:
        fail(f"cannot reach {args.base_url}: {exc!r}", "is vLLM up on this port in this environment?")
    if args.model not in served:
        fail(f"server serves {served}, not {args.model!r}",
             "start the right model or pass the served name with --model")
    ok(f"{args.base_url} serves {args.model}")

    # 4 --------------------------------------------------------------------
    step("4. one debate round per persona kind (thinking off)")
    row = rows[0]
    n = len(row["options"])
    prompts = B.question_prompts(row)
    t0 = time.perf_counter()
    try:
        r1 = D.execute_round(client, args.model, row["question"], list(row["options"]),
                             {"personas": ["solver"]}, [], 0.7, args.answer_tokens, prompts)
    except Exception as exc:
        fail(f"solver call failed: {exc!r}")
    dt = time.perf_counter() - t0
    persona, text = r1[0]
    letter = D.extract_letter(text, n)
    if not text.strip():
        fail("solver returned empty content",
             "with --reasoning-parser the answer may be in reasoning_content; the executor reads content only")
    if "<think>" in text.lower():
        fail("solver output contains a <think> block: thinking was not disabled",
             "check the chat template / enable_thinking handling for this model")
    if letter is None:
        fail(f"no parseable ANSWER letter in solver output (first 300 chars): {text[:300]!r}")
    ok(f"solver: {len(text)} chars, ANSWER {letter}, {dt:.1f}s")
    try:
        r2 = D.execute_round(client, args.model, row["question"], list(row["options"]),
                             {"personas": ["critic"], "sees": "all"}, [r1], 0.7, args.answer_tokens, prompts)
    except Exception as exc:
        fail(f"critic call failed: {exc!r}")
    if D.extract_letter(r2[0][1], n) is None:
        fail(f"critic (rewritten prompt, sees prior round) gave no ANSWER letter: {r2[0][1][:300]!r}")
    ok(f"critic reading the solver: ANSWER {D.extract_letter(r2[0][1], n)}")

    # 5 --------------------------------------------------------------------
    step("5. non-answering persona")
    try:
        r3 = D.execute_round(client, args.model, row["question"], list(row["options"]),
                             {"personas": ["eliminator"], "sees": "none"}, [], 0.7, args.answer_tokens, prompts)
    except Exception as exc:
        fail(f"eliminator call failed: {exc!r}")
    if not r3[0][1].strip():
        fail("eliminator returned empty content")
    ok(f"eliminator: {len(r3[0][1])} chars"
       + ("" if "SURVIVING" in r3[0][1].upper() else "  (no SURVIVING line; tolerated, it commits no letter)"))

    # 6 --------------------------------------------------------------------
    step("6. BudgetedRunner: live round, cache write, replay")
    scratch = Path(tempfile.mkdtemp(prefix="probe_env_"))
    rows_by_id = {r["id"]: r for r in rows[:max(64, 2 + args.burst)]}
    runner = M.BudgetedRunner(rows_by_id, [], base_urls=args.base_url, model=args.model,
                              temperature=0.7, answer_tokens=args.answer_tokens,
                              cache_path=scratch / "probe_cache.jsonl", max_calls=500,
                              api_key=args.api_key, progress=False)
    runner.reset_budget(None)
    q = rows[1]["id"]
    try:
        out = M.run_program(M.chain("critic"), runner, rows_by_id[q])
    except Exception as exc:
        fail(f"run_program live failed: {exc!r}")
    calls_after = runner.calls
    expect = 2 + runner.summaries if args.v2 else 2
    if calls_after != expect:
        fail(f"solver->critic should cost {expect} calls, runner charged {calls_after}")
    runner.reset_budget(0)                           # replay only
    out2 = M.run_program(M.chain("critic"), runner, rows_by_id[q])
    if runner.calls != calls_after or out2["letter"] != out["letter"]:
        fail("second run of the same program was not served from the cache")
    written = [json.loads(l) for l in (scratch / "probe_cache.jsonl").open() if l.strip()]
    if len(written) != 2 or not all(M.well_formed(w) for w in written):
        fail("cache file does not hold 2 well-formed rounds")
    if runner.errors:
        fail(f"runner recorded {runner.errors} errored rounds")
    ok(f"live solver->critic = {out['letter']}, cached, replayed, 2 well-formed rounds on disk")

    # 7 --------------------------------------------------------------------
    step(f"7. throughput: {args.burst} concurrent solver calls")
    qids = [r["id"] for r in rows[2:2 + args.burst]]
    runner.reset_budget(None)
    before = runner.calls
    t0 = time.perf_counter()
    ev = M.eval_program(M.chain(), runner, rows_by_id, qids, rep=0, workers=args.burst)
    dt = time.perf_counter() - t0
    made = runner.calls - before
    if ev.covered != len(qids) or runner.errors:
        fail(f"burst: covered {ev.covered}/{len(qids)}, errors {runner.errors}")
    rate = made / dt
    ok(f"{made} calls in {dt:.0f}s -> {rate:.2f} calls/s (~{rate * 3600:,.0f}/hour); "
       + (f"{runner.summaries} summary calls; " if args.v2 else "")
       + f"letters parsed on all {ev.covered}")
    if rate < 0.3:
        print("        note: slow; a full run at this rate takes many days. "
              "Check --max-num-seqs / GPU sharing on the server.")
    runner.close()

    # 8 --------------------------------------------------------------------
    step("8. output paths")
    lock = args.live_cache.with_suffix(".lock")
    if lock.exists():
        holder = lock.read_text().strip()
        fail(f"live cache {args.live_cache} is locked by another run: {holder}",
             "wait for it, use a different --live-cache, or delete the lock if that run is dead")
    args.live_cache.parent.mkdir(parents=True, exist_ok=True)
    probe_file = args.live_cache.parent / ".probe_write_test"
    try:
        probe_file.write_text("x")
        probe_file.unlink()
    except OSError as exc:
        fail(f"cannot write to {args.live_cache.parent}: {exc!r}")
    if args.live_cache.exists():
        n_lines = sum(1 for _ in args.live_cache.open(errors="replace"))
        ok(f"live cache exists with {n_lines} lines (will be reused); no lock")
        print("        note: make sure those rounds came from THIS model; recordings are not keyed by model")
    else:
        ok(f"live cache will be created at {args.live_cache}; no lock")

    print("\nALL TESTS PASS")


if __name__ == "__main__":
    main()

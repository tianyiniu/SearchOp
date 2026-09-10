"""Re-certify the qwen_unanswerable set under the EVAL mode (thinking OFF).

The existing datasets/supergpqa_qwen_unanswerable.json was filtered with k=3 CoT
(thinking ON). But every downstream schema experiment runs thinking OFF, and a 3-
sample "all wrong" test only certifies LOW solve-probability, not zero -- so ~12% of
that pool are still solved by a single thinking-off pass (mode mismatch + weak k).

This tightens the pool to what's genuinely hard under the mode we evaluate in: run the
SAME executor single_pass uses (MINIMAL_SCHEMA, thinking OFF, temp 0.7) up to k times
per question, short-circuiting on the first correct answer, and KEEP only questions
answered incorrectly on all k attempts (0/k). The result is the intersection
(thinking-ON 0/3) AND (thinking-OFF 0/k) -- questions single_pass fails k times.

An unparseable answer (no 'ANSWER: X' line) counts as a MISS, exactly as the eval
grader treats it; the per-question count of unparseable attempts is logged so
purely-formatting retentions can be spotted.

No train/test split here. Writes:
  datasets/supergpqa_filter_strict_full.json     kept: thinking-off 0/k
  datasets/supergpqa_thinkoff_recovered.json     dropped: thinking-off solved >=1 (label noise)

    # vLLM serving Qwen/Qwen3-14B on port 7472 (thinking off -> short, FREE/local)
    python scripts/filter_strict_thinkoff.py --k 3 --workers 16
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import debate_mcq as D  # same executor + letter grading the eval uses


def process_question(row: dict, client, model: str, args) -> dict:
    """Run single_pass (thinking off) up to k times; short-circuit on first correct."""
    gold = row["answer_letter"]
    options = list(row["options"])
    out = dict(row)  # preserve all original fields (incl. thinking-ON provenance)
    try:
        n_correct = n_unparse = attempts = 0
        answered = False
        for _ in range(args.k):
            attempts += 1
            letter = D.execute_schema(client, model, row["question"], options,
                                      D.MINIMAL_SCHEMA, args.temperature, args.answer_tokens)
            if letter is None:
                n_unparse += 1
            elif letter == gold:
                n_correct += 1
                answered = True
                break  # solvable thinking-off -> recovered, stop early
        out.update(thinkoff_k=attempts, thinkoff_correct=n_correct,
                   thinkoff_unparsed=n_unparse,
                   kept=(not answered), status="ok")
    except Exception as exc:  # keep the pool going; don't silently drop
        out.update(kept=False, status="error", error=str(exc))
    return out


def breakdown(rows: list[dict]) -> str:
    diff = Counter(r.get("difficulty") for r in rows)
    return "  ".join(f"{d}={diff.get(d, 0)}" for d in ("easy", "middle", "hard"))


def main(args: argparse.Namespace) -> None:
    rows = json.loads(Path(args.dataset).read_text())
    if args.limit is not None:
        rows = rows[: args.limit]

    # guard: a --limit smoke test must NOT clobber the canonical full outputs
    strict_out, recovered_out, cache = args.strict_out, args.recovered_out, args.cache
    if args.limit is not None and not args.write_partial:
        strict_out = strict_out.with_name(f"{strict_out.stem}_limit{args.limit}{strict_out.suffix}")
        recovered_out = recovered_out.with_name(f"{recovered_out.stem}_limit{args.limit}{recovered_out.suffix}")
        cache = cache.with_name(f"{cache.stem}_limit{args.limit}{cache.suffix}")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    cache.parent.mkdir(parents=True, exist_ok=True)
    print(f"re-certifying {len(rows)} questions thinking-OFF (k={args.k}, temp={args.temperature}, "
          f"workers={args.workers})")

    records: list[dict] = []
    lock = Lock()
    with open(cache, "w") as ch, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_question, row, client, args.model, args) for row in rows]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="strict-filter", unit="q"):
            rec = fut.result()
            with lock:
                ch.write(json.dumps(rec, ensure_ascii=False) + "\n")
                ch.flush()
            records.append(rec)

    ok = [r for r in records if r["status"] == "ok"]
    kept = [r for r in ok if r["kept"]]
    recovered = [r for r in ok if not r["kept"]]
    errors = [r for r in records if r["status"] == "error"]
    # retained purely because every attempt was unparseable (suspicious: formatting, not reasoning)
    kept_all_unparsed = [r for r in kept if r["thinkoff_unparsed"] == r["thinkoff_k"]]

    Path(strict_out).write_text(json.dumps(kept, ensure_ascii=False, indent=1))
    Path(recovered_out).write_text(json.dumps(recovered, ensure_ascii=False, indent=1))

    n = len(ok)
    print(f"\ninput (thinking-ON unanswerable): {len(rows)}")
    print(f"  KEPT   (thinking-off 0/{args.k}): {len(kept)}  ({len(kept)/n:.1%} of evaluated)  -> {strict_out.name}")
    print(f"  recovered (solved >=1x off):     {len(recovered)}  ({len(recovered)/n:.1%})  -> {recovered_out.name}")
    print(f"    == thinking-off single_pass solve rate on the pool: {len(recovered)/n:.1%}")
    print(f"  kept difficulty:      {breakdown(kept)}")
    print(f"  recovered difficulty: {breakdown(recovered)}")
    if kept_all_unparsed:
        print(f"  NOTE: {len(kept_all_unparsed)} kept had ALL attempts unparseable (formatting, not reasoning)")
    if errors:
        print(f"  errors (excluded from both): {len(errors)}")
    print(f"\ncache -> {cache}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("datasets/supergpqa_qwen_unanswerable.json"),
                    help="The thinking-ON unanswerable pool to re-certify.")
    ap.add_argument("--k", type=int, default=3, help="Attempts; keep questions wrong on all k.")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--base-url", default="http://localhost:7472/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help=">0 so the k attempts are independent draws (matches collection/eval).")
    ap.add_argument("--answer-tokens", type=int, default=3072, help="Max tokens per call (thinking off).")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="Smoke test on first N (auto-suffixes outputs).")
    ap.add_argument("--write-partial", action="store_true",
                    help="Allow --limit to write the canonical (unsuffixed) output filenames.")
    ap.add_argument("--strict-out", type=Path, default=Path("datasets/supergpqa_filter_strict_full.json"))
    ap.add_argument("--recovered-out", type=Path, default=Path("datasets/supergpqa_thinkoff_recovered.json"))
    ap.add_argument("--cache", type=Path, default=Path("outputs/filter_strict_cache.jsonl"))
    main(ap.parse_args())

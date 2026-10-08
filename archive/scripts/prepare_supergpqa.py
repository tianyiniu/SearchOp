"""Step 0 (SuperGPQA): build the qwen_unanswerable *reasoning* split.

The debate-side failures on FRAMES were reasoning failures, not retrieval or
comprehension failures (the gold facts were in context; the model mis-computed,
mis-counted, or mis-selected). So this moves the debate experiments onto a pure
reasoning benchmark — SuperGPQA, graduate-level multiple choice across 72 fields —
where there are NO documents and grading is an exact letter match (no LLM judge,
so none of the judge noise we saw on FRAMES).

Procedure per question:
  1. Round-robin over (field x difficulty) cells so the candidate order is balanced
     across subjects and difficulties (not front-loaded by whatever order the
     dataset ships in).
  2. Ask Qwen WITH chain-of-thought (thinking on), grading each attempt by letter
     match, and STOP as soon as the label is decided:
       correct on any attempt  -> qwen_answerable  (stop early; the common case)
       wrong on all k attempts -> qwen_unanswerable  (the debate target set)
     A correct first attempt costs 1 call instead of k -- that is the compute
     saving, since most of the pool is answerable. An unanswerable verdict still
     costs the full k. There is no borderline bucket: the first correct answer
     settles it, so the mixed case folds into answerable.
  3. Stop once --target unanswerable are collected (or the pool is exhausted).

Truncation guard (a lesson from the FRAMES analysis): Qwen's <think> trace is
unbounded and can hit the token cap before it ever commits an answer. A truncated
or unparseable attempt is an INVALID sample, not a wrong one — it is retried at a
larger token budget, and a question that still cannot produce a valid answer is
marked 'invalid' and excluded, so truncation never masquerades as unanswerable.

Because collection already establishes the single-pass and self-consistency (k
samples) outcome, downstream schema evaluation can SKIP the single_pass and
sample_vote baselines and go straight to the aggregation shapes.

    export OPENAI_API_KEY=...    # not needed here (letter match, no judge) but harmless
    # vLLM serving Qwen/Qwen3-14B on port 7472
    python3 scripts/prepare_supergpqa.py --target 5000
    python3 scripts/prepare_supergpqa.py --limit 40      # smoke test
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from pathlib import Path
from threading import Lock

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:  # load API keys from the project .env if python-dotenv is available
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


# Only these canonical fields are ever read. In particular the RLOrchestrator
# copy's `step_back`/`keywords` augmentations are NOT part of the canonical
# dataset and would leak reasoning if included — pulling straight from HF avoids
# them entirely.
RAW_FIELDS = ("uuid", "question", "options", "answer_letter",
              "discipline", "field", "subfield", "difficulty", "is_calculation")

LETTERS = "ABCDEFGHIJ"


def load_pool(raw_path: Path, hf_name: str, split: str) -> list[dict]:
    """Load the canonical SuperGPQA pool, caching a trimmed copy to disk so the
    HF download happens once."""
    if raw_path.exists():
        return json.loads(raw_path.read_text())
    from datasets import load_dataset  # imported lazily so --help needs no network

    ds = load_dataset(hf_name, split=split)
    rows = [{k: r.get(k) for k in RAW_FIELDS} for r in ds]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(rows, ensure_ascii=False))
    print(f"downloaded {len(rows)} rows from {hf_name}[{split}] -> {raw_path}")
    return rows


# --- prompting + grading ---------------------------------------------------

SYSTEM = (
    "You are answering a hard graduate-level multiple-choice question. Think step "
    "by step, then end with exactly one line: 'ANSWER: <letter>' giving the letter "
    "of the single best option."
)


def render_prompt(question: str, options: list[str]) -> str:
    body = "\n".join(f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options))
    return f"{question}\n\n{body}\n\nEnd with 'ANSWER: <letter>'."


_ANS = re.compile(r"ANSWER\s*:\s*\(?\s*([A-J])\b", re.I)


def extract_letter(text: str, n_options: int) -> str | None:
    """The letter from the last 'ANSWER:' line, or None if there is no valid one."""
    for m in reversed(_ANS.findall(text or "")):
        letter = m.upper()
        if 0 <= LETTERS.index(letter) < n_options:
            return letter
    return None


def is_truncated(text: str) -> bool:
    """A <think> block that was never closed -> the model hit the token cap mid-
    reasoning and never reached its answer."""
    return "<think>" in (text or "") and "</think>" not in text


def chat(client: OpenAI, model: str, user: str, max_tokens: int,
         temperature: float, thinking: bool) -> str:
    # enable_thinking is best-effort: fall back to a plain call if the server
    # rejects the kwarg (mirrors route_debate.QwenRanker).
    for extra in ({"chat_template_kwargs": {"enable_thinking": thinking}}, {}):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": user}],
                temperature=temperature, max_tokens=max_tokens, extra_body=extra)
            return resp.choices[0].message.content or ""
        except Exception:
            continue
    return ""


def one_attempt(client, model, user, gold, n_options, args) -> tuple[str | None, bool]:
    """One CoT attempt, retrying at a larger token budget while the output is
    truncated/unparseable. Returns (letter or None, correct). None => invalid
    even after retries (do NOT count as a wrong answer)."""
    budget = args.answer_tokens
    for _ in range(args.max_retries + 1):
        raw = chat(client, model, user, budget, args.temperature, thinking=True)
        letter = extract_letter(raw, n_options)
        if letter is not None:
            return letter, (letter == gold)
        budget = budget * 2 if is_truncated(raw) else int(budget * 1.5)
    return None, False


def process_question(row: dict, client, args) -> dict:
    """Label one question, short-circuiting as soon as the outcome is decided.

    A question is unanswerable only if Qwen fails ALL k attempts, so the first
    correct attempt settles it as answerable and we stop — the common case, and
    where the compute is saved (1 call instead of k). An unanswerable verdict still
    costs the full k. A truncated/unparseable attempt means the label can't be
    trusted, so it stops and the question is excluded."""
    opts = list(row["options"])
    gold = row["answer_letter"]
    rec = {"id": row["uuid"], "question": row["question"], "options": opts,
           "answer_letter": gold, "discipline": row.get("discipline"),
           "field": row.get("field"), "subfield": row.get("subfield"),
           "difficulty": row.get("difficulty"), "k": args.k}
    try:
        user = render_prompt(row["question"], opts)
        letters = []
        status = "qwen_unanswerable"  # holds only if every attempt is a valid miss
        for _ in range(args.k):
            letter, ok = one_attempt(client, args.model, user, gold, len(opts), args)
            letters.append(letter)
            if letter is None:          # truncated/unparseable -> can't trust the label
                status = "invalid_unparseable"
                break
            if ok:                      # correct once -> answerable; skip the rest
                status = "qwen_answerable"
                break
            # a valid miss -> keep checking
        rec.update(answers=letters, n_attempts=len(letters),
                   correct_count=(1 if status == "qwen_answerable" else 0))
        rec["status"] = status
    except Exception as exc:  # keep the pool going; record what failed
        rec["status"] = "error"
        rec["error"] = str(exc)
    return rec


# --- balanced candidate ordering -------------------------------------------

def round_robin_order(pool: list[dict], subject_key: str, seed: int) -> list[int]:
    """Indices into `pool`, ordered round-robin across (subject x difficulty)
    cells so labeling stays balanced across subjects and difficulties even if we
    stop early. Cells and within-cell order are shuffled deterministically."""
    cells: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(pool):
        cells[(r.get(subject_key), r.get("difficulty"))].append(i)
    rng = random.Random(seed)
    queues = [deque(idxs) for idxs in cells.values()]
    for q in queues:
        rng.shuffle(q)
    rng.shuffle(queues)  # randomize cell visitation order

    order: list[int] = []
    while any(queues):
        for q in queues:
            if q:
                order.append(q.popleft())
    return order


def _split_row(rec: dict) -> dict:
    """Fields downstream schema scripts need, plus label provenance."""
    keys = ("id", "question", "options", "answer_letter", "discipline", "field",
            "subfield", "difficulty", "correct_count", "k")
    return {k: rec[k] for k in keys}


def main(args: argparse.Namespace) -> None:
    pool = load_pool(args.raw, args.hf_name, args.split)
    print(f"pool: {len(pool)} questions | subject axis: {args.subject_key} "
          f"| target unanswerable: {args.target}")

    order = round_robin_order(pool, args.subject_key, args.seed)
    if args.limit is not None:
        order = order[: args.limit]

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    for path in (args.cache, args.answerable_out, args.unanswerable_out):
        path.parent.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list] = {"qwen_answerable": [], "qwen_unanswerable": []}
    counts: Counter = Counter()
    n_unans = labeled = 0
    lock = Lock()
    cursor = iter(order)
    pbar = tqdm(total=min(len(order), args.max_label), desc="labeling", unit="q")
    with open(args.cache, "w") as cache, ThreadPoolExecutor(max_workers=args.workers) as ex:
        # Chunked so we follow the round-robin order AND can stop near the target;
        # overshoot is bounded by one chunk.
        while n_unans < args.target and labeled < args.max_label:
            chunk = list(islice(cursor, args.chunk))
            if not chunk:
                break
            futures = [ex.submit(process_question, pool[i], client, args) for i in chunk]
            for future in as_completed(futures):
                rec = future.result()
                with lock:
                    cache.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    cache.flush()
                counts[rec["status"]] += 1
                labeled += 1
                pbar.update(1)
                if rec["status"] in buckets:
                    buckets[rec["status"]].append(_split_row(rec))
                if rec["status"] == "qwen_unanswerable":
                    n_unans += 1
    pbar.close()

    args.answerable_out.write_text(json.dumps(buckets["qwen_answerable"], indent=2, ensure_ascii=False))
    args.unanswerable_out.write_text(json.dumps(buckets["qwen_unanswerable"], indent=2, ensure_ascii=False))

    print(f"\nlabeled {labeled} questions (k={args.k}, temp={args.temperature}):", dict(counts))
    print(f"qwen_unanswerable ({len(buckets['qwen_unanswerable'])}, 0/{args.k}) -> {args.unanswerable_out}")
    print(f"qwen_answerable   ({len(buckets['qwen_answerable'])}, correct within {args.k}) -> {args.answerable_out}")
    if counts.get("invalid_unparseable"):
        print(f"invalid/unparseable (excluded, truncation guard): {counts['invalid_unparseable']}")
    if counts.get("error"):
        print(f"errors: {counts['error']}")

    # balance report: unanswerable distribution across the subject axis x difficulty
    cell = Counter((r["field"], r["difficulty"]) for r in buckets["qwen_unanswerable"])
    by_diff = Counter(r["difficulty"] for r in buckets["qwen_unanswerable"])
    print(f"\nunanswerable by difficulty: {dict(by_diff)}")
    print(f"unanswerable spans {len({f for f, _ in cell})} fields; "
          f"top cells: {cell.most_common(5)}")
    print(f"per-question detail -> {args.cache}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-name", default="m-a-p/SuperGPQA", help="HuggingFace dataset id.")
    ap.add_argument("--split", default="train", help="HF split (SuperGPQA ships one split).")
    ap.add_argument("--raw", type=Path, default=Path("datasets/supergpqa_raw.json"),
                    help="Cached trimmed copy of the pool (downloaded once).")
    ap.add_argument("--subject-key", default="field", choices=["field", "discipline", "subfield"],
                    help="Subject axis for round-robin (crossed with difficulty).")
    ap.add_argument("--target", type=int, default=5000, help="Stop after this many unanswerable.")
    ap.add_argument("--max-label", type=int, default=30000,
                    help="Safety cap on total questions labeled.")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--base-url", default="http://localhost:7472/v1", help="Local vLLM endpoint.")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--k", type=int, default=3, help="CoT attempts per question.")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="Sampling temperature (>0 so the k attempts differ).")
    ap.add_argument("--answer-tokens", type=int, default=8192,
                    help="Token budget per CoT attempt (thinking traces are long).")
    ap.add_argument("--max-retries", type=int, default=1,
                    help="Retries for a truncated/unparseable attempt (budget grows each time).")
    ap.add_argument("--workers", type=int, default=8, help="Questions labeled concurrently.")
    ap.add_argument("--chunk", type=int, default=None,
                    help="Candidates per scheduling chunk (default: workers x 10).")
    ap.add_argument("--limit", type=int, default=None, help="Only consider the first N candidates.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", type=Path, default=Path("outputs/supergpqa_prepare_cache.jsonl"))
    ap.add_argument("--unanswerable-out", type=Path, default=Path("datasets/supergpqa_qwen_unanswerable.json"))
    ap.add_argument("--answerable-out", type=Path, default=Path("datasets/supergpqa_qwen_answerable.json"))
    args = ap.parse_args()
    if args.chunk is None:
        args.chunk = args.workers * 10
    main(args)

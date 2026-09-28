"""
Given a dataset and LLM, run a simple React loop to filter for questions that the model does not answer correctly.

For each question in the dataset, the agent (Qwen3-14B served by a local vLLM)
answers using the three tools (search_info, fetch_url, code_compute) under a
budget of 20 tool calls. A Gemini judge then grades the answer against the gold
target. Questions graded INCORRECT are collected until we have ``--num-examples``
of them; every result (right or wrong) is cached to a JSONL file for later use.

    # vLLM serving Qwen/Qwen3-14B on port 7472, plus:
    export SERPER_API_KEY=...   # web tools
    export GEMINI_API_KEY=...   # judge
    python scripts/react_data_filter.py --num-examples 50
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

# Make the project root importable when run as `python scripts/react_data_filter.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iterative_summarization import estimate_tokens, iterative_summarize
from llm_judge import judge_answer
from tool_calling import run_with_tools
from tools import build_tools

try:  # load API keys from the project .env if python-dotenv is available
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


SYSTEM_PROMPT = (
    "You are a careful research assistant answering a question with tools. Work "
    "in a ReAct style: reason about what you still need, call a tool to get it, "
    "read the result, and repeat until you can answer. The tools are search_info "
    "(web search), fetch_url (read one page), and code_compute (run Python). When "
    "confident, stop calling tools and give a short final answer on one line "
    "prefixed with 'ANSWER: '."
)


def final_answer(text: str) -> str:
    """Pull the text after the last 'ANSWER:' marker, or return the whole reply."""
    marker = "ANSWER:"
    idx = text.rfind(marker)
    return (text[idx + len(marker):] if idx != -1 else text).strip()


def summarizing_fetch(base_fetch, client, args, question):
    """Wrap fetch_url so any page over the summary budget is replaced by a
    query-aware iterative summary (using the agent model) before the agent sees
    it — keeping long pages from overflowing the model's context window."""
    def fetch(url: str) -> str:
        text = base_fetch(url)
        if estimate_tokens(text) <= args.summary_tokens:
            return text
        summary = iterative_summarize(
            client, args.model, question, text,
            context_window=args.context_window, summary_tokens=args.summary_tokens,
        )
        return f"[Long page summarized for the question; original ~{estimate_tokens(text)} tokens]\n\n{summary}"
    return fetch


def main(args: argparse.Namespace) -> None:
    questions = json.loads(Path(args.dataset).read_text())
    if args.limit is not None:
        questions = questions[: args.limit]

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    base_registry, schemas = build_tools()  # all three tools

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    incorrect: list[dict] = []
    bar = tqdm(questions, desc="evaluated", unit="q")
    with open(args.cache, "w") as cache:
        for i, row in enumerate(bar):
            question, ground_truth = row["question"], row["ground_truth"]
            # Per-question registry: fetch_url summarizes long pages against this question.
            registry = {**base_registry,
                        "fetch_url": summarizing_fetch(base_registry["fetch_url"], client, args, question)}
            try:
                result = run_with_tools(
                    client, args.model, SYSTEM_PROMPT, question, schemas, registry,
                    max_tool_calls=args.max_tool_calls,
                )
                answer = final_answer(result.answer)
                correct = judge_answer(question, ground_truth, answer)
                error = None
            except Exception as exc:  # keep the run going; record what failed
                result, answer, correct, error = None, "", False, str(exc)

            record = {
                "id": row.get("id", f"row_{i}"),
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "correct": correct,
                "error": error,
                "num_model_calls": result.num_model_calls if result else 0,
                "num_tool_calls": len(result.tool_calls) if result else 0,
                "tool_calls": [dataclasses.asdict(tc) for tc in result.tool_calls] if result else [],
            }
            cache.write(json.dumps(record, ensure_ascii=False) + "\n")
            cache.flush()  # incremental, so a crash keeps the work so far

            if error is None and not correct:
                incorrect.append(record)
            bar.set_postfix(collected=f"{len(incorrect)}/{args.num_examples}")

            if len(incorrect) >= args.num_examples:
                break
    bar.close()

    Path(args.out).write_text(json.dumps(incorrect, indent=2, ensure_ascii=False))
    print(f"\nCollected {len(incorrect)} incorrect questions -> {args.out}")
    print(f"All {i + 1} results cached -> {args.cache}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path("datasets/frames_test_full.json"))
    ap.add_argument("--num-examples", type=int, default=10,
                    help="Stop once this many INCORRECT questions are collected.")
    ap.add_argument("--max-tool-calls", type=int, default=20,
                    help="Tool-call budget per question.")
    ap.add_argument("--context-window", type=int, default=32768,
                    help="Agent model context window; long fetched pages are summarized to fit it.")
    ap.add_argument("--summary-tokens", type=int, default=2048,
                    help="Summary size, and the page-size threshold that triggers summarization.")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--base-url", default="http://localhost:7472/v1",
                    help="Local vLLM OpenAI-compatible endpoint.")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only scan the first N questions of the dataset.")
    ap.add_argument("--cache", type=Path, default=Path("outputs/react_cache.jsonl"),
                    help="JSONL of every result (correct and incorrect).")
    ap.add_argument("--out", type=Path, default=Path("outputs/react_incorrect.json"),
                    help="JSON of the collected incorrect questions.")
    main(ap.parse_args())

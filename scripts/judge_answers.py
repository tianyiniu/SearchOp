"""Grading open answers (HLE) with a judge model.

A final answer is right when the judge says it matches the key. The prompt is
HLE's own judge prompt (the benchmark's official grading), given the question,
the answer being graded and the correct answer; the judge's confidence field is
left out (nothing here reads it). Every verdict is kept in one JSONL cache:

    {"qid", "answer", "correct", "extracted", "reasoning", "model", "effort", "v", "usage"}

keyed by (question id, answer) for this judge model, effort and prompt version,
so a (question, answer) pair is graded once for the search, the test evaluation
and the external baselines alike, and a replay never calls the model. Answers
are graded as given: callers pass the normalised answer (debate_mcq.
normalize_answer), which is also what the debate compares.

The client is made on the first verdict that is not in the cache, so a replay of
recorded debates needs no API key. A failed call is retried with backoff for
several minutes; after that the error is raised, and the run stops rather than
count an ungraded answer as wrong.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
JUDGE_MODEL = "gpt-6-luna"
JUDGE_EFFORT = "medium"
PROMPT_VERSION = "hle1"
DEFAULT_CACHE_DIR = ROOT / "outputs/judge_cache"

# HLE's judge prompt (hle_eval/run_judge_results.py), without the confidence field
JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect."""


class Verdict(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]


def default_cache(model: str = JUDGE_MODEL, effort: str = JUDGE_EFFORT) -> Path:
    return DEFAULT_CACHE_DIR / f"{model}_{effort}_{PROMPT_VERSION}.jsonl"


class Judge:
    """Thread-safe, cached grading of (question, answer) pairs."""

    def __init__(self, cache_path: Path | None = None, model: str = JUDGE_MODEL,
                 effort: str = JUDGE_EFFORT, retries: int = 8):
        self.model, self.effort, self.retries = model, effort, retries
        self.cache_path = Path(cache_path) if cache_path else default_cache(model, effort)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._verdicts: dict[tuple[str, str], bool] = {}
        if self.cache_path.exists():
            for line in self.cache_path.open():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:              # a line cut short by a kill: graded again
                    continue
                if (r.get("model"), r.get("effort"), r.get("v")) == (model, effort, PROMPT_VERSION):
                    self._verdicts[(r["qid"], r["answer"])] = bool(r["correct"])
        self._lock = threading.Lock()
        self._inflight: dict[tuple[str, str], threading.Event] = {}
        self._client = None
        self.stats = {"cached": 0, "calls": 0, "input_tokens": 0, "output_tokens": 0, "retries": 0}

    def settings(self) -> dict:
        return {"model": self.model, "effort": self.effort, "prompt": PROMPT_VERSION}

    def _get_client(self):
        with self._lock:
            if self._client is None:
                from dotenv import load_dotenv
                from openai import OpenAI
                load_dotenv(ROOT / ".env")
                if not os.environ.get("OPENAI_API_KEY"):
                    raise SystemExit("OPENAI_API_KEY is not set (expected in .env): the judge needs it")
                self._client = OpenAI()
            return self._client

    def _ask(self, question: str, gold: str, answer: str) -> tuple[Verdict, dict]:
        client = self._get_client()
        prompt = JUDGE_PROMPT.format(question=question, response=answer, correct_answer=gold)
        delay = 2.0
        for attempt in range(self.retries):
            try:
                resp = client.responses.parse(model=self.model, input=prompt, text_format=Verdict,
                                              reasoning={"effort": self.effort}, max_output_tokens=4000)
                if resp.output_parsed is None:
                    raise ValueError(f"no parsed verdict (status {getattr(resp, 'status', '?')})")
                u = getattr(resp, "usage", None)
                return resp.output_parsed, {"input": getattr(u, "input_tokens", None),
                                            "output": getattr(u, "output_tokens", None)}
            except Exception:
                if attempt == self.retries - 1:
                    raise
                with self._lock:
                    self.stats["retries"] += 1
                time.sleep(delay)
                delay = min(delay * 2, 120.0)
        raise RuntimeError("unreachable")

    def grade(self, qid: str, question: str, gold: str, answer: str | None) -> bool:
        """Whether `answer` (normalised) matches the key of question `qid`. No
        answer is wrong without a call. Concurrent calls for the same pair wait
        for one verdict."""
        if answer is None or not str(answer).strip():
            return False
        key = (qid, answer)
        while True:
            with self._lock:
                if key in self._verdicts:
                    self.stats["cached"] += 1
                    return self._verdicts[key]
                event = self._inflight.get(key)
                if event is None:
                    event = self._inflight[key] = threading.Event()
                    break
            event.wait()                    # another thread is grading this pair
        try:
            verdict, usage = self._ask(question, gold, answer)
            correct = verdict.correct == "yes"
            record = {"qid": qid, "answer": answer, "correct": correct,
                      "extracted": verdict.extracted_final_answer, "reasoning": verdict.reasoning,
                      "model": self.model, "effort": self.effort, "v": PROMPT_VERSION, "usage": usage}
            with self._lock:
                with self.cache_path.open("a") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                self._verdicts[key] = correct
                self.stats["calls"] += 1
                self.stats["input_tokens"] += usage.get("input") or 0
                self.stats["output_tokens"] += usage.get("output") or 0
            return correct
        finally:
            with self._lock:
                self._inflight.pop(key, None)
            event.set()

    def grade_row(self, row: dict, answer: str | None) -> bool:
        """grade() for a dataset row (id, question, answer)."""
        return self.grade(row["id"], row["question"], row["answer"], answer)

    def grade_many(self, items: list[tuple[dict, str | None]], workers: int = 16) -> list[bool]:
        """grade_row over many (row, answer) pairs, in order, `workers` at a time."""
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda it: self.grade_row(*it), items))

"""Shared pieces of the hint-sheet study (see backward_scaffold_experiment.md).

- fixed choices: K = 5 tries per test and the three thresholds that go with it
- the question pool (only questions that already carry a knowledge label)
- Student: gpt-oss-20b behind a vLLM server, with a replay cache so that any
  run can be repeated for free and an interrupted run resumes where it stopped
- Teacher: gpt-5.6-terra through the Responses API with a JSON schema, also cached
- prompt building blocks and the answer-letter extraction (the official
  SuperGPQA rule, from baselines/score.py)

Every student call is keyed by a string that says exactly what was asked
(question id, condition, sample index). Two runs that ask the same thing get
the same recorded reply.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "baselines"))
from score import extract_answer  # noqa: E402  (official SuperGPQA extraction; main thread only)

DATASETS = ROOT / "datasets"
DEFAULT_OUT = ROOT / "outputs" / "hint_study"

# --- fixed choices --------------------------------------------------------------
K = 5            # tries per test
FAIL_MAX = 1     # right at most this often out of K with no hints  -> "fails alone"
PASS_MIN = 4     # right at least this often out of K with the full sheet -> "succeeds with hints"
REACH_MIN = 3    # a way of working "finishes" from k hints if right at least this often
LEAK_MIN = 3     # sheet + options with no question right this often -> the sheet gives it away

STUDENT_MODEL = "openai/gpt-oss-20b"
DEFAULT_ENDPOINTS = "http://localhost:7472"
TEACHER_MODEL = "gpt-5.6-terra"
WINDOW = 32_768            # the server's context window (both families are served with 32,768)
MAX_TOKENS = 28_672        # same cap as the external baselines

# Two small-model families. gpt-oss takes a reasoning effort (low / medium / high).
# Qwen 3.5 takes a thinking switch: "low" means thinking off, anything else thinking
# on. Sampling follows the external baselines of each family (baselines/run_baselines_*.sh).
SAMPLING = {"gptoss": {"temperature": 1.0, "top_p": 1.0},
            "qwen": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}}


def family_of(model: str) -> str:
    return "qwen" if "qwen" in model.lower() else "gptoss"


def model_tag(model: str) -> str:
    return model.rsplit("/", 1)[-1].replace(".", "").replace("-", "_").lower()


def add_student_args(ap) -> None:
    g = ap.add_argument_group("small model")
    g.add_argument("--model", default=STUDENT_MODEL,
                   help="openai/gpt-oss-20b (reasoning effort) or a Qwen 3.5 model such as Qwen/Qwen3.5-9B "
                        "(thinking on; 'low' effort means thinking off)")
    g.add_argument("--endpoints", default=DEFAULT_ENDPOINTS, help="comma-separated vLLM servers")
    g.add_argument("--workers", type=int, default=32)


def add_dir_args(ap) -> None:
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                    help="the study files for this small model (one folder per small model)")
    ap.add_argument("--teacher-dir", type=Path, default=None,
                    help="where the hint sheets and the teacher cache live (default: --out-dir); point every "
                         "small model at one folder so each sheet is written once")


def teacher_dir(args) -> Path:
    return args.teacher_dir or args.out_dir


def make_student(args) -> "Student":
    """The default model keeps the original cache file name, so a run started
    before --model existed carries on with its cache; any other model gets its
    own file."""
    name = "student_cache.jsonl" if args.model == STUDENT_MODEL else f"student_cache_{model_tag(args.model)}.jsonl"
    return Student(args.out_dir / name, args.endpoints, args.model, workers=args.workers)

LETTERS = "ABCDEFGHIJ"
KINDS = ("recall", "derive", "both")

ANSWER_LINE = ("The last line of your response should be in the format 'Answer: $LETTER' "
               "(without quotes), where LETTER is one of {letters}.")


# --- questions ----------------------------------------------------------------------
def load_split(name: str) -> list[dict]:
    """The study questions: datasets/hint_study_<name>.json (train or test)."""
    path = DATASETS / f"hint_study_{name}.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing; run scripts/hint_study/draw_questions.py first")
    return json.loads(path.read_text())


def splits(arg: str) -> list[str]:
    return ["train", "test"] if arg == "both" else [arg]


def letters_of(row: dict) -> str:
    return ", ".join(LETTERS[: len(row["options"])])


def options_block(row: dict) -> str:
    return "\n".join(f"{LETTERS[i]}) {opt}" for i, opt in enumerate(row["options"]))


def question_block(row: dict) -> str:
    return row["question"].strip() + "\n" + options_block(row)


def answer_line(row: dict) -> str:
    return ANSWER_LINE.format(letters=letters_of(row))


def direct_prompt(row: dict) -> str:
    """The official SuperGPQA zero-shot prompt, as baselines/generate.py sends it."""
    return ("Answer the following multiple choice question. There is only one correct answer. "
            + answer_line(row) + "\n\n" + question_block(row) + "\n")


def is_right(pred: str | None, row: dict) -> bool:
    return pred is not None and pred == row["answer_letter"]


def score(recs: list[dict], row: dict) -> dict:
    """Extract a letter from every reply and count. Main thread only (the
    extraction uses signal.alarm against slow regexes)."""
    preds = [extract_answer(r.get("content"), row["options"]) if r.get("error") is None else None
             for r in recs]
    return {"n": len(recs), "c": sum(is_right(p, row) for p in preds), "preds": preds,
            "truncated": sum(r.get("finish_reason") == "length" for r in recs),
            "errors": sum(r.get("error") is not None for r in recs),
            "tokens": sum(r.get("completion_tokens") or 0 for r in recs)}


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def by_kind(rows: list[dict]) -> dict[str, list[dict]]:
    out = {k: [] for k in KINDS}
    for r in rows:
        out[r["knowledge"]].append(r)
    return out


# --- the hint sheet as an ordered list of items ------------------------------------
def sheet_items(sheet: dict) -> list[dict]:
    """Facts first, then the steps, never the held-back final step. Each item
    knows where it came from so a trimmed sheet can be rebuilt."""
    items = [{"kind": "fact", "idx": i, "text": f["text"]} for i, f in enumerate(sheet.get("facts", []))]
    items += [{"kind": "step", "idx": i, "text": s["text"], "step_kind": s.get("kind")}
              for i, s in enumerate(sheet.get("steps", []))]
    return items


def items_block(items: list[dict]) -> str:
    if not items:
        return "(nothing has been established yet)"
    return "\n".join(f"{i + 1}. {it['text']}" for i, it in enumerate(items))


def item_key(items: list[dict]) -> str:
    """A short stable name for a subset of items, for cache keys."""
    return ",".join(f"{it['kind'][0]}{it['idx']}" for it in items) or "none"


def with_hints_prompt(row: dict, items: list[dict]) -> str:
    """The question with the hints. Used by the gate (all items), the trimming
    (subsets) and the plain 'just continue' way of working (the first k items),
    so those runs share one cache."""
    return ("Answer the following multiple choice question. There is only one correct answer. "
            "A colleague has already established the facts and steps listed under 'Established so far'; "
            "you may rely on them and continue from where they stop. " + answer_line(row)
            + "\n\n" + question_block(row) + "\n\nEstablished so far:\n" + items_block(items) + "\n")


# --- the replay cache ---------------------------------------------------------------
class Cache:
    """One JSONL file, one record per line, keyed by rec['key']. Append-only;
    a killed run leaves at most one broken last line, which is skipped."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.recs: dict[str, dict] = {}
        self.lock = threading.Lock()
        if self.path.exists():
            with self.path.open() as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("error") is None:
                        self.recs[r["key"]] = r
        self.f = self.path.open("a")

    def get(self, key: str) -> dict | None:
        return self.recs.get(key)

    def put(self, rec: dict) -> None:
        with self.lock:
            if rec.get("error") is None:
                self.recs[rec["key"]] = rec
            self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.f.flush()

    def close(self) -> None:
        self.f.close()


def seed_of(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % (2**31)


# --- the student ---------------------------------------------------------------------
class Student:
    """gpt-oss-20b behind vLLM's chat-completions endpoint. `run` takes a list of
    jobs, answers the ones not in the cache in parallel, and returns every
    record keyed by job key. A job is a dict with key, messages, effort and an
    optional max_tokens."""

    def __init__(self, cache_path: Path, endpoints: str = DEFAULT_ENDPOINTS, model: str = STUDENT_MODEL,
                 workers: int = 32, max_retries: int = 5, budget: int | None = None):
        self.cache = Cache(cache_path)
        self.urls = [e.rstrip("/") + "/v1/chat/completions" for e in endpoints.split(",")]
        self.model, self.workers, self.max_retries = model, workers, max_retries
        self.family = family_of(model)
        self.budget = budget          # calls this run may make; None = no limit
        self.spent = 0
        self.session = requests.Session()

    def _switch(self, effort: str) -> dict:
        if self.family == "qwen":
            return {"chat_template_kwargs": {"enable_thinking": effort != "low"}}
        return {"reasoning_effort": effort}

    @staticmethod
    def job(key: str, messages: list[dict], effort: str = "high", max_tokens: int | None = None) -> dict:
        return {"key": key, "messages": messages, "effort": effort, "max_tokens": max_tokens}

    def _max_tokens(self, job: dict) -> int:
        # leave room for the prompt inside the window; 4 chars per token is a safe over-estimate
        prompt_chars = sum(len(m["content"]) for m in job["messages"])
        room = WINDOW - prompt_chars // 3 - 256
        return max(1024, min(job.get("max_tokens") or MAX_TOKENS, room))

    def _one(self, i: int, job: dict) -> dict:
        payload = {"model": self.model, "messages": job["messages"], "max_tokens": self._max_tokens(job),
                   "seed": seed_of(job["key"]), **SAMPLING[self.family], **self._switch(job["effort"])}
        url = self.urls[i % len(self.urls)]
        err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.post(url, json=payload, timeout=(30, None))
                data = resp.json()
                if resp.status_code == 200:
                    choice = data["choices"][0]
                    msg = choice["message"]
                    return {"key": job["key"], "effort": job["effort"], "model": self.model,
                            "content": msg.get("content"),
                            "reasoning": msg.get("reasoning") or msg.get("reasoning_content"),
                            "finish_reason": choice.get("finish_reason"),
                            "completion_tokens": data["usage"]["completion_tokens"], "error": None}
                err = f"HTTP {resp.status_code}: {str(data)[:300]}"
            except (requests.RequestException, ValueError, KeyError) as exc:
                err = f"{type(exc).__name__}: {exc}"[:300]
            time.sleep(min(60, 2 ** attempt) + random.random())
        return {"key": job["key"], "effort": job["effort"], "error": err}

    def run(self, jobs: list[dict], label: str = "") -> dict[str, dict]:
        out, todo = {}, []
        seen = set()
        for j in jobs:
            rec = self.cache.get(j["key"])
            if rec is not None:
                out[j["key"]] = rec
            elif j["key"] not in seen:
                seen.add(j["key"])
                todo.append(j)
        if self.budget is not None and self.spent + len(todo) > self.budget:
            raise SystemExit(f"{label}: {len(todo)} calls needed, {self.budget - self.spent} left in the budget")
        if todo:
            random.Random(0).shuffle(todo)           # a partial run is a fair sample
            stats = {"tokens": 0, "trunc": 0, "err": 0, "t0": time.time()}
            with ThreadPoolExecutor(self.workers) as pool, tqdm(total=len(todo), desc=label or "student",
                                                                 dynamic_ncols=True) as bar:
                futs = [pool.submit(self._one, i, j) for i, j in enumerate(todo)]
                for fut in as_completed(futs):
                    rec = fut.result()
                    self.cache.put(rec)
                    out[rec["key"]] = rec
                    stats["tokens"] += rec.get("completion_tokens") or 0
                    stats["trunc"] += rec.get("finish_reason") == "length"
                    stats["err"] += rec.get("error") is not None
                    el = time.time() - stats["t0"]
                    bar.set_postfix(tok_s=f"{stats['tokens'] / el:.0f}", trunc=stats["trunc"], err=stats["err"])
                    bar.update(1)
            self.spent += len(todo)
        return out

    def close(self) -> None:
        self.cache.close()


# --- the teacher ---------------------------------------------------------------------
class Teacher:
    """gpt-5.6-terra through the Responses API with a pydantic schema, cached the
    same way. `parse` returns the parsed record as a dict (plus usage), or a
    record with an 'error' field."""

    def __init__(self, cache_path: Path, model: str = TEACHER_MODEL, workers: int = 4, retries: int = 4):
        from openai import OpenAI
        self.client = OpenAI()
        self.cache = Cache(cache_path)
        self.model, self.workers, self.retries = model, workers, retries

    def _one(self, key: str, instructions: str, text: str, schema, effort: str, max_output_tokens: int) -> dict:
        delay = 2.0
        for attempt in range(self.retries):
            try:
                resp = self.client.responses.parse(model=self.model, instructions=instructions, input=text,
                                                   reasoning={"effort": effort}, text_format=schema,
                                                   max_output_tokens=max_output_tokens)
                parsed = resp.output_parsed
                if parsed is None:
                    raise ValueError(f"no parsed output (status {resp.status})")
                u = resp.usage
                return {"key": key, "model": self.model, "effort": effort, "error": None,
                        "out": parsed.model_dump(),
                        "usage": {"input": u.input_tokens, "output": u.output_tokens}}
            except Exception as exc:                      # rate limits, 5xx, parse misses
                if attempt == self.retries - 1:
                    return {"key": key, "model": self.model, "effort": effort,
                            "error": f"{type(exc).__name__}: {exc}"[:500]}
                time.sleep(delay)
                delay *= 2

    def run(self, jobs: list[dict], label: str = "") -> dict[str, dict]:
        """jobs: dicts with key, instructions, text, schema, effort, max_output_tokens."""
        out, todo = {}, []
        for j in jobs:
            rec = self.cache.get(j["key"])
            if rec is not None:
                out[j["key"]] = rec
            else:
                todo.append(j)
        if todo:
            with ThreadPoolExecutor(self.workers) as pool, tqdm(total=len(todo), desc=label or "teacher",
                                                                 dynamic_ncols=True) as bar:
                futs = [pool.submit(self._one, j["key"], j["instructions"], j["text"], j["schema"],
                                    j.get("effort", "high"), j.get("max_output_tokens", 16_000)) for j in todo]
                for fut in as_completed(futs):
                    rec = fut.result()
                    self.cache.put(rec)
                    out[rec["key"]] = rec
                    bar.update(1)
        return out

    def close(self) -> None:
        self.cache.close()


# --- reading the study files ------------------------------------------------------------
def read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False))


def read_sheets(sheets_dir: Path) -> dict[str, list[dict]]:
    """All hint-sheet records by question id, oldest version first."""
    sheets: dict[str, list[dict]] = {}
    path = sheets_dir / "sheets.jsonl"
    if path.exists():
        with path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("error") is None:
                    sheets.setdefault(r["id"], []).append(r)
    return sheets


def active_sheets(sheets_dir: Path, out_dir: Path | None = None) -> dict[str, dict]:
    """The sheet to use per question: the newest version whose teacher answer
    matches the key and which this small model's leak check did not flag (a
    version not yet checked counts as usable, so gate_with_sheet can run before
    leak_check in a smoke test, but the report says which versions were
    unchecked). Sheets are shared between small models; the leak check is not."""
    leaks = read_json((out_dir or sheets_dir) / "leak.json", {})
    chosen = {}
    for qid, versions in read_sheets(sheets_dir).items():
        for rec in reversed(versions):
            if not rec["teacher_agrees"]:
                continue
            lk = leaks.get(f"{qid}|{rec['version']}")
            if lk is not None and lk["leak"]:
                continue
            rec = dict(rec)
            rec["leak_checked"] = lk is not None
            chosen[qid] = rec
            break
    return chosen

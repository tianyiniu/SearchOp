"""Recovery for a reply cut off at its token limit, for the external baselines.

A reply that reaches max_tokens before it gives an answer (finish_reason "length" and no letter by
score.py's SuperGPQA extraction rule) is scored as wrong. The debate executor instead asks the model,
once, for the letter its reasoning supports (scripts/debate_mcq.py, chat_v3). This gives the
external baselines the same treatment, so a comparison measures the methods rather than how each
handles a cut-off:

    the same conversation, then the end of the cut-off reply (the last 16,000 characters of its
    reasoning, as the executor shows, and any visible text) as the assistant's turn, then
    RECOVER_PROMPT, at low effort (gpt-oss) or with thinking off (Qwen), up to 2,048 tokens. If
    that names no letter, one shorter nudge (COMMIT_PROMPT, 512 tokens), as the executor does.

selfrefine.py --recover does this inside its loop. For generate.py's direct samples it is done
afterwards, which gives exactly what a run with it built in would give (a direct sample is one
reply, and generate.py keeps its reasoning):

    python baselines/recover.py --data <dataset.json> --results direct_X.jsonl --out direct_X_rec.jsonl

Every sample of --results is copied to --out. A recovered one keeps its original reply under
"recovery" and gets the recovery reply as "content" (what score.py reads), with the recovery's
tokens added to "completion_tokens"; its finish_reason stays "length", so score.py's truncation
rate still counts the cut-offs. Resumable: samples already in --out are skipped, and a failed
request writes nothing, so running again retries it.

HLE rows (hle_format.py) are asked in HLE's own format ('Exact Answer:' / 'Answer:' line), and
their answer is read as score.py reads it (hle_format.extract); GPQA-Diamond and MATH rows in
theirs (tasks.recover_prompts: the letter A-D, or the answer in \\boxed{}).
"""
import argparse
import asyncio
import json
import os
import time

import aiohttp
from tqdm import tqdm

import tasks
from generate import question_messages, request_one, sample_seed

TAIL_CHARS = 16000             # debate_mcq.THINKING_TAIL
RECOVER_TOKENS = 2048
COMMIT_TOKENS = 512
RECOVER_PROMPT, COMMIT_PROMPT = tasks.RECOVER_PROMPT, tasks.COMMIT_PROMPT     # SuperGPQA's
SEED_OFFSET = 500000           # recovery seeds never collide with the samples' own


def answer_of(text: str | None, item: dict) -> str | None:
    """The answer score.py reads from a response: the SuperGPQA or GPQA letter, HLE's answer
    line, or math's last \\boxed{} (tasks.answer_of)."""
    return tasks.answer_of(text, item)


def prompts_for(item: dict) -> tuple[str, str]:
    """(recovery prompt, commit prompt) for the row's dataset."""
    return tasks.recover_prompts(item)


def needs_recovery(reply: dict, item: dict) -> bool:
    """Cut off at the token limit and no answer by score.py's rule."""
    return reply.get("finish_reason") == "length" and answer_of(reply.get("content"), item) is None


def shown_text(reply: dict) -> str | None:
    """What the recovery call sees as the model's own turn: the end of the cut-off reply."""
    tail = (reply.get("reasoning") or "")[-TAIL_CHARS:]
    visible = reply.get("content") or ""
    parts = ([f"[the end of my reasoning]\n{tail}"] if tail.strip() else []) + \
            ([f"[my reply so far]\n{visible}"] if visible.strip() else [])
    return "\n\n".join(parts) or None


def low_effort(args) -> dict:
    """The request fields of a recovery call: the family's cheapest thinking setting."""
    if args.family == "qwen":
        return {"top_k": args.top_k, "min_p": args.min_p, "presence_penalty": args.presence_penalty,
                "chat_template_kwargs": {"enable_thinking": False}}
    return {"reasoning_effort": "low"}


async def recover(session, url, args, messages: list[dict], reply: dict, item: dict, seed: int,
                  prompts: tuple[str, str] | None = None, accept=None):
    """Ask for the answer a cut-off reply's reasoning supports. `messages` is the conversation that
    produced `reply` (ending with the user turn it answered). Returns (result, None) or (None, error);
    result is {"content", "completion_tokens", "calls"} ("content" None if there was nothing to show).
    `prompts` and `accept` (default: the dataset's answer prompts, and a reply with a readable
    answer) let other turns use it: Self-Refine's feedback (tasks.feedback_recover_prompts)."""
    text = shown_text(reply)
    if text is None:
        return {"content": None, "completion_tokens": 0, "calls": 0}, None
    convo = messages + [{"role": "assistant", "content": text}]
    tokens = calls = 0
    content = None
    recover_prompt, commit_prompt = prompts or prompts_for(item)
    accept = accept or (lambda text: answer_of(text, item) is not None)
    for prompt, limit in ((recover_prompt, RECOVER_TOKENS), (commit_prompt, COMMIT_TOKENS)):
        payload = {"model": args.model, "messages": convo + [{"role": "user", "content": prompt}],
                   "max_tokens": limit, "temperature": args.temperature, "top_p": args.top_p,
                   "seed": (seed + calls) % (2**31), **low_effort(args)}
        data, err = await request_one(session, url, payload, args.max_retries)
        if data is None:
            return None, err
        calls += 1
        tokens += data["usage"]["completion_tokens"]
        content = data["choices"][0]["message"].get("content") or ""
        if accept(content):
            break
    return {"content": content, "completion_tokens": tokens, "calls": calls}, None


def with_recovery(reply: dict, got: dict) -> dict:
    """`reply` with the recovery reply as its content; the original content is kept."""
    out = dict(reply)
    out["recovery"] = {"original_content": reply.get("content"), "content": got["content"],
                       "completion_tokens": got["completion_tokens"], "calls": got["calls"]}
    if got["content"] is not None:
        out["content"] = got["content"]
    out["completion_tokens"] = (reply.get("completion_tokens") or 0) + got["completion_tokens"]
    return out


def add_request_args(p: argparse.ArgumentParser) -> None:
    """The request settings, as generate.py and selfrefine.py spell them."""
    p.add_argument("--endpoints", default="http://localhost:7472")
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--family", default="gptoss", choices=["gptoss", "qwen"])
    p.add_argument("--top-k", type=int, default=20, help="qwen only")
    p.add_argument("--min-p", type=float, default=0.0, help="qwen only")
    p.add_argument("--presence-penalty", type=float, default=1.5, help="qwen only")
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--explain", action="store_true",
                   help="ask for the reasoning in the reply itself (tasks.EXPLAIN_SENTENCE); off by default")
    p.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high"],
                   help="the samples' own effort (accepted so the same arguments as generate.py can be "
                        "passed); the recovery call always uses low")


async def main(args):
    tasks.set_explain(args.explain)                  # the first prompt (see tasks.py)
    with open(args.data) as f:
        items = {it["id"]: it for it in json.load(f)}
    samples = {}                                     # (id, sample index) -> record, last write wins
    with open(args.results) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error") is None and r.get("id") in items:
                samples[(r["id"], r["sample_idx"])] = r
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("error") is None:
                    done.add((r["id"], r["sample_idx"]))
    todo = [(key, r) for key, r in samples.items() if key not in done]
    need = [(key, r) for key, r in todo if needs_recovery(r, items[key[0]])]
    print(f"{len(samples)} samples in {args.results}: {len(done)} already in {args.out}, "
          f"{len(todo) - len(need)} to copy, {len(need)} cut off without an answer to recover")

    out_f = open(args.out, "a")
    for key, r in todo:
        if not needs_recovery(r, items[key[0]]):
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out_f.flush()
    if not need:
        out_f.close()
        return

    url = args.endpoints.split(",")[0].rstrip("/") + "/v1/chat/completions"
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    pbar = tqdm(total=len(need), dynamic_ncols=True)
    stats = {"answered": 0, "errors": 0, "t0": time.time()}
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:

        async def run(key, r):
            item = items[key[0]]
            async with sem:
                got, err = await recover(session, url, args, question_messages(item),
                                         r, item, sample_seed(key[0], key[1]) + SEED_OFFSET)
            async with lock:
                if err is not None:
                    stats["errors"] += 1
                else:
                    rec = with_recovery(r, got)
                    stats["answered"] += answer_of(rec["content"], item) is not None
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                pbar.update(1)
                pbar.set_postfix(answered=stats["answered"], err=stats["errors"])

        await asyncio.gather(*(run(key, r) for key, r in need))
    pbar.close()
    out_f.close()
    print(f"recovered {len(need)} cut-off samples in {time.time() - stats['t0']:.0f}s: {stats['answered']} now "
          f"give an answer, {stats['errors']} failed requests (run again to retry them)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--results", required=True, help="generate.py output (its samples keep their reasoning)")
    p.add_argument("--out", required=True)
    p.add_argument("--concurrency", type=int, default=64)
    add_request_args(p)
    asyncio.run(main(p.parse_args()))

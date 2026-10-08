"""Direct chain-of-thought baseline: sample k responses per SuperGPQA question from a vLLM
OpenAI-compatible server.

Copied from /nas-ssd2/hwang/tianyi/supergpqa_eval/generate.py and tailored to openai/gpt-oss-20b on
the 300-question dev split (supergpqa_program_search_dev_small.json): the Qwen thinking switch is
replaced by gpt-oss's reasoning effort (high by default), the Qwen-specific sampling settings are
dropped, and max_tokens leaves room for the prompt inside the server's 32,768-token window.

--family qwen switches back to the original's Qwen request (thinking switch, top_k, min_p,
presence penalty) for the Qwen servers; the default, gptoss, is unchanged.

Prompt follows the official SuperGPQA zero-shot template; an HLE row (one with an
"answer_type", scripts/prepare_hle.py) gets HLE's own request instead (hle_format.py), and a
GPQA-Diamond or MATH row its dataset's prompt (tasks.py).
Every finished sample is appended to --out immediately, so an interrupted run resumes
where it stopped.
"""
import argparse
import asyncio
import hashlib
import json
import os
import random
import time
from collections import defaultdict

import aiohttp
from tqdm import tqdm

import tasks
from tasks import PROMPT_TEMPLATE, build_prompt  # noqa: F401  (SuperGPQA's prompt, kept importable here)


def question_messages(item):
    """The request's messages: the dataset's own prompt (tasks.messages): SuperGPQA's zero-shot
    prompt, HLE's format, GPQA's simple-evals prompt or the math instruction."""
    import tasks
    return tasks.messages(item)


def sample_seed(qid, idx):
    return (int(hashlib.sha256(qid.encode()).hexdigest()[:8], 16) + idx) % (2**31)


def load_done(path):
    done = defaultdict(set)
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:  # partial last line from a killed run
                    continue
                if r.get("error") is None:
                    done[r["id"]].add(r["sample_idx"])
    return done


def model_settings(args) -> dict:
    """The request fields that depend on the model family. gpt-oss takes a reasoning effort;
    Qwen takes the thinking switch and the sampling settings of its model card (as in the
    collaborator's original script)."""
    if args.family == "qwen":
        return {"top_k": args.top_k, "min_p": args.min_p, "presence_penalty": args.presence_penalty,
                "chat_template_kwargs": {"enable_thinking": args.thinking}}
    return {"reasoning_effort": args.reasoning_effort}      # gpt-oss: low / medium / high


CONTEXT_MARGIN = 64        # tokens kept free when max_tokens is cut to fit the window
MIN_ROOM = 1024            # below this, a request that does not fit is an error, not a short reply


async def fit_to_window(session, url, payload):
    """For a request the server rejected as too long for its window: max_tokens cut to the room the
    prompt leaves (the prompt counted exactly by the server's /tokenize, with the same chat
    template settings), or None if that room is under MIN_ROOM tokens or cannot be found. vLLM's
    rejection gives only a lower bound on the prompt's length, hence the count."""
    body = {"model": payload["model"], "messages": payload["messages"]}
    if "chat_template_kwargs" in payload:
        body["chat_template_kwargs"] = payload["chat_template_kwargs"]
    try:
        async with session.post(url.replace("/v1/chat/completions", "/tokenize"), json=body) as resp:
            got = await resp.json(content_type=None)
            if resp.status != 200:
                return None
    except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError):
        return None
    room = got["max_model_len"] - got["count"] - CONTEXT_MARGIN
    return room if MIN_ROOM <= room < payload["max_tokens"] else None


async def request_one(session, url, payload, max_retries):
    """POST with retries. A request the server rejects as too long for its window (HTTP 400,
    'maximum context length') is sent again at once with max_tokens cut to fit (fit_to_window);
    the reply then carries "max_tokens_cut": the limit it was sent with. Requests that fit are
    sent exactly as given."""
    cut = None
    for attempt in range(max_retries):
        try:
            async with session.post(url, json=payload) as resp:
                data = await resp.json(content_type=None)
                if resp.status == 200:
                    if cut is not None:
                        data["max_tokens_cut"] = cut
                    return data, None
                err = f"HTTP {resp.status}: {str(data)[:300]}"
                if resp.status == 400 and cut is None and "maximum context length" in str(data):
                    if (room := await fit_to_window(session, url, payload)) is not None:
                        payload, cut = {**payload, "max_tokens": room}, room
                        continue
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
            err = f"{type(e).__name__}: {e}"
        await asyncio.sleep(min(60, 2**attempt) + random.random())
    return None, err


async def main(args):
    tasks.set_explain(args.explain)                  # the first prompt (see tasks.py)
    with open(args.data) as f:
        items = json.load(f)
    if args.limit:
        items = items[: args.limit]

    done = load_done(args.out)
    # Missing samples among indices 0..k-1. Samples a question already has at index >= k (from a
    # run with larger k) can bring it to k in total, so the missing indices that still get it there
    # run first; the rest complete the fixed indices 0..k-1, which score.py uses by default.
    first, rest = [], []
    for it in items:
        have = done[it["id"]]
        missing = [s for s in range(args.k) if s not in have]
        need = max(0, args.k - len(have))
        first += [(it, s) for s in missing[:need]]
        rest += [(it, s) for s in missing[need:]]
    # Shuffle within each phase so a partial run is an unbiased sample across difficulty/discipline.
    random.Random(0).shuffle(first)
    random.Random(1).shuffle(rest)
    jobs = first + rest
    n_done = sum(len({s for s in v if s < args.k}) for v in done.values())
    print(f"{len(items)} questions x k={args.k}: {n_done} done, {len(jobs)} to run "
          f"({len(first)} reach k samples per question, then {len(rest)} complete indices 0..k-1)")
    if not jobs:
        return

    endpoints = [e.rstrip("/") + "/v1/chat/completions" for e in args.endpoints.split(",")]
    extra = model_settings(args)

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    pbar = tqdm(total=len(jobs), smoothing=0.05, dynamic_ncols=True)
    stats = {"tokens": 0, "errors": 0, "truncated": 0, "t0": time.time()}
    out_f = open(args.out, "a")

    # Non-streaming: nothing arrives until the whole response is generated, so a read timeout
    # would silently drop the longest generations. Only bound the connection attempt by default.
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=args.request_timeout or None)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

        async def run(job_idx, item, s):
            async with sem:
                payload = {
                    "model": args.model,
                    "messages": question_messages(item),
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "seed": sample_seed(item["id"], s),
                    **extra,
                }
                url = endpoints[job_idx % len(endpoints)]
                data, err = await request_one(session, url, payload, args.max_retries)

            rec = {"id": item["id"], "sample_idx": s, "error": err}
            if data is not None:
                choice = data["choices"][0]
                msg = choice["message"]
                rec.update(
                    content=msg.get("content"),
                    reasoning=msg.get("reasoning") or msg.get("reasoning_content"),
                    finish_reason=choice.get("finish_reason"),
                    completion_tokens=data["usage"]["completion_tokens"],
                )
                if "max_tokens_cut" in data:           # the prompt left less room than --max-tokens
                    rec["max_tokens_cut"] = data["max_tokens_cut"]
                stats["tokens"] += rec["completion_tokens"]
                stats["truncated"] += rec["finish_reason"] == "length"
            else:
                stats["errors"] += 1

            async with lock:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                pbar.update(1)
                el = time.time() - stats["t0"]
                pbar.set_postfix(tok_s=f"{stats['tokens'] / el:.0f}",
                                 trunc=stats["truncated"], err=stats["errors"])

        await asyncio.gather(*(run(i, it, s) for i, (it, s) in enumerate(jobs)))

    pbar.close()
    out_f.close()
    el = time.time() - stats["t0"]
    print(f"finished {len(jobs)} samples in {el / 3600:.2f}h, {stats['tokens'] / el:.0f} tok/s, "
          f"truncated={stats['truncated']}, errors={stats['errors']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--endpoints", default="http://localhost:7472", help="comma-separated; requests are round-robined")
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=64)
    # The server's window is 32,768 tokens and vLLM rejects a request whose prompt + max_tokens
    # exceeds it, so max_tokens stays below that. Sampling follows OpenAI's gpt-oss guidance
    # (temperature 1.0, top_p 1.0), the same temperature the program search used.
    p.add_argument("--max-tokens", type=int, default=28672)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high"])
    p.add_argument("--family", default="gptoss", choices=["gptoss", "qwen"],
                   help="gptoss: send --reasoning-effort. qwen: send the thinking switch, top_k, min_p "
                        "and presence_penalty instead (set --temperature/--top-p for Qwen too)")
    p.add_argument("--top-k", type=int, default=20, help="qwen only")
    p.add_argument("--min-p", type=float, default=0.0, help="qwen only")
    p.add_argument("--presence-penalty", type=float, default=1.5, help="qwen only")
    p.add_argument("--no-thinking", dest="thinking", action="store_false", help="qwen only")
    p.add_argument("--explain", action="store_true",
                   help="ask for the reasoning in the reply itself (tasks.EXPLAIN_SENTENCE); off by default")
    p.add_argument("--request-timeout", type=float, default=0, help="seconds to wait for a response; 0 = no limit")
    p.add_argument("--max-retries", type=int, default=5)
    asyncio.run(main(p.parse_args()))
